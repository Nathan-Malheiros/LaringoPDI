"""Fase B: o PDI isolado e na segmentação clássica, técnica por técnica.

Responde, com as regras de docs/DECISIONS.md (2026-11-06):
- qual técnica melhora a imagem, e por quê (contraste, entropia, nitidez, saturação; PSNR e SSIM com ruído);
- qual técnica segmenta melhor a glote (Dice, IoU, HD95 e falha, com parâmetros ajustados só no treino);
- a que custo (milissegundos por frame, medidos em um único núcleo da CPU).

Saída em results/tables/pdi/; a tabela principal é `tabela_unica.csv`.
Este estudo não altera `segmentation_best.json`: o método usado pela MLP já foi fixado antes dele.
"""
from __future__ import annotations

import math
import sys
import time

import cv2
import numpy as np
import pandas as pd

from laringopdi import config as C
from laringopdi import data as D
from laringopdi.custo import latest_stages, medir
from laringopdi.etapa1 import _plt, _wilcoxon, cmd_denoise_study
from laringopdi.evaluate import _holm, seg_per_fold

CHUNK = 60
ENHANCE = ("stretch", "he", "clahe", "spec", "homo")     # avaliadas pela regra visual
FILTER_KEY = {"mean": "mean3", "median": "median5", "gauss_lpf": "lpf40"}   # parâmetros padrão do registry
QUALITY = ("entropy", "rms_contrast", "lap_var", "saturated_frac", "spec_frac")
SAT_TOLERANCE = 0.01
NAMES = {"raw": "sem PDI", "stretch": "alargamento", "he": "equalização", "clahe": "CLAHE", "mean": "média", "median": "mediana",
         "gauss_lpf": "passa-baixa", "spec": "remoção de reflexo", "homo": "homomórfica"}
METHODS = {"otsu": "Otsu", "adaptive": "adaptativa", "otsu_morph": "Otsu + morf.", "adaptive_morph": "adaptativa + morf.",
           "region_grow": "cresc. de região", "watershed": "watershed"}


def _apply(g: np.ndarray, tech: str) -> np.ndarray:
    from laringopdi.pdi.registry import DEFAULTS, TECHNIQUES
    return g if tech == "raw" else TECHNIQUES[tech](g, {}, **DEFAULTS[tech])


def _frame_rows(path, seg, techs, grid, work_size) -> tuple[list[dict], list[dict]]:
    """Qualidade da imagem e segmentação de um frame, em todas as técnicas × métodos × parâmetros."""
    from laringopdi.pdi.quality import image_quality
    from laringopdi.pdi.registry import to_work_size
    from laringopdi.pdi.segmentation import seg_metrics, segment

    cv2.setNumThreads(1)
    g = to_work_size(D.imread(path, gray=True), work_size)
    gt = cv2.resize((D.imread(seg, gray=True) > 127).astype(np.uint8), g.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
    qual, rows = [], []
    for t in techs:
        gp = _apply(g, t)
        q = image_quality(gp)
        qual.append({"technique": t, **{k: q[k] for k in QUALITY}})
        for method, plist in grid.items():
            for k, prm in enumerate(plist):
                rows.append({"preprocess": t, "method": method, "param_id": k, **seg_metrics(segment(gp, method, **prm), gt)})
    return qual, rows


def _run_frames(cfg: dict, n_jobs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Processa em blocos gravados em disco: se for interrompido, retoma do último bloco concluído."""
    from joblib import Parallel, delayed
    from tqdm import tqdm

    from laringopdi.pdi.segmentation import PARAM_GRID

    sc, raw, ws = cfg["segmentation"], C.path(cfg, "raw"), cfg["data"]["work_size"]
    grid = {m: PARAM_GRID[m] for m in sc["methods"]}
    fr = D.load_frames(cfg)
    out = C.path(cfg, "cache").parent / "pdi_study"
    out.mkdir(parents=True, exist_ok=True)
    n = math.ceil(len(fr) / CHUNK)
    with Parallel(n_jobs=n_jobs) as par:
        for i in tqdm(range(n), desc="Técnicas e segmentação (blocos de 60 frames)", unit="bloco", file=sys.stdout):
            q_p, s_p = out / f"q_{i:04d}.csv.gz", out / f"s_{i:04d}.csv.gz"
            if q_p.exists():
                continue
            part = fr.iloc[i * CHUNK:(i + 1) * CHUNK]
            res = par(delayed(_frame_rows)(raw / a, raw / b, sc["preprocess"], grid, ws) for a, b in zip(part["path"], part["seg_path"]))
            pd.concat([pd.DataFrame(r[1]).assign(path=p) for r, p in zip(res, part["path"])]).round(5).to_csv(s_p, index=False)
            pd.concat([pd.DataFrame(r[0]).assign(path=p) for r, p in zip(res, part["path"])]).round(6).to_csv(q_p, index=False)   # por último: marca o bloco como pronto
    qual = pd.concat([pd.read_csv(out / f"q_{i:04d}.csv.gz") for i in range(n)], ignore_index=True)
    seg = pd.concat([pd.read_csv(out / f"s_{i:04d}.csv.gz") for i in range(n)], ignore_index=True)
    return qual, seg


# ----------------------------------------------------------------------------- PDI isolado
def visual_verdicts(qv: pd.DataFrame, enhance=ENHANCE, tol: float = SAT_TOLERANCE, alpha: float = 0.05) -> pd.DataFrame:
    """`qv`: uma linha por técnica × vídeo. Melhora = contraste RMS maior, entropia não menor e
    saturação até `tol` acima da referência, com Wilcoxon pareado por vídeo e Holm por métrica."""
    wide = qv.pivot(index="video_id", columns="technique")
    rows = []
    for t in enhance:
        r = {"technique": t}
        for met in ("rms_contrast", "entropy"):
            a, b = wide[met]["raw"], wide[met][t]
            r[f"delta_{met}"], r[f"p_{met}"] = float((b - a).mean()), _wilcoxon(a, b)
        r["delta_saturated_frac"] = float((wide["saturated_frac"][t] - wide["saturated_frac"]["raw"]).mean())
        rows.append(r)
    df = pd.DataFrame(rows)
    for met in ("rms_contrast", "entropy"):
        df[f"p_holm_{met}"] = _holm(df[f"p_{met}"].to_numpy())
    rms_up = (df["delta_rms_contrast"] > 0) & (df["p_holm_rms_contrast"] < alpha)
    ent_ok = ~((df["delta_entropy"] < 0) & (df["p_holm_entropy"] < alpha))
    sat_ok = df["delta_saturated_frac"] <= tol
    df["verdict"] = np.where(rms_up & ent_ok & sat_ok, "melhora", "não melhora")
    df["motivo"] = [("atende aos três critérios" if u and e and s else "; ".join(
        x for x, bad in (("contraste não aumenta", not u), ("entropia cai", not e), ("satura além do limite", not s)) if bad))
        for u, e, s in zip(rms_up, ent_ok, sat_ok)]
    return df


def denoise_verdicts(frames: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Filtro × sem filtro, por tipo de ruído: reduz o ruído se PSNR e SSIM aumentam (Wilcoxon por frame; Holm entre filtros)."""
    rows = []
    for noise, g in frames.groupby("noise"):
        w = g.pivot(index="frame_id", columns="filter")
        for f in [c for c in w["psnr"].columns if c != "none"]:
            r = {"noise": noise, "filter": f, "ms": g[g["filter"] == f]["ms"].median()}
            for met in ("psnr", "ssim"):
                a, b = w[met]["none"], w[met][f]
                r.update({f"{met}_none": a.mean(), met: b.mean(), f"delta_{met}": (b - a).mean(), f"p_{met}": _wilcoxon(a, b)})
            rows.append(r)
    df = pd.DataFrame(rows)
    for met in ("psnr", "ssim"):
        df[f"p_holm_{met}"] = df.groupby("noise")[f"p_{met}"].transform(lambda p: _holm(p.to_numpy()))
    up = (df["delta_psnr"] > 0) & (df["p_holm_psnr"] < alpha) & (df["delta_ssim"] > 0) & (df["p_holm_ssim"] < alpha)
    df["verdict"] = np.where(up, "reduz o ruído", "não reduz")
    return df


# ----------------------------------------------------------------------------- segmentação
def seg_verdicts(video_dice: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """`video_dice`: índice (method, video_id), colunas = técnicas. Técnica × raw no mesmo método; Holm dentro do método."""
    rows = []
    for meth, g in video_dice.groupby(level="method"):
        cur = []
        for t in [c for c in g.columns if c != "raw"]:
            x = g[["raw", t]].dropna()
            cur.append({"method": meth, "technique": t, "n_videos": len(x), "dice_raw": x["raw"].mean(), "dice_tech": x[t].mean(),
                        "delta_dice": (x[t] - x["raw"]).mean(), "p": _wilcoxon(x["raw"], x[t])})
        if cur:
            d = pd.DataFrame(cur)
            d["p_holm"] = _holm(d["p"].to_numpy())
            rows.append(d)
    df = pd.concat(rows, ignore_index=True)
    df["verdict"] = np.where(df["p_holm"] < alpha, np.where(df["delta_dice"] > 0, "melhora", "piora"), "neutra")
    return df


def _video_dice(cfg: dict, seg: pd.DataFrame, per_fold: pd.DataFrame, fr: pd.DataFrame) -> pd.DataFrame:
    """Dice por vídeo, com os parâmetros escolhidos no treino do fold em que o vídeo é teste."""
    x = seg.merge(fr[["path", "video_id"]], on="path")
    test_of = {v: f["fold"] for f in D.load_splits(cfg) for v in f["test"]}
    x["fold"] = x["video_id"].map(test_of)
    sel = x.merge(per_fold[["fold", "preprocess", "method", "param_id"]], on=["fold", "preprocess", "method", "param_id"])
    return sel.groupby(["method", "preprocess", "video_id"])["dice"].mean().unstack("preprocess")


def timing(cfg: dict, per_fold: pd.DataFrame, n: int = 100) -> pd.DataFrame:
    """Milissegundos por frame em um único núcleo, com a imagem em tamanho de trabalho: técnica e segmentação."""
    from tqdm import tqdm

    from laringopdi.pdi.registry import to_work_size
    from laringopdi.pdi.segmentation import PARAM_GRID, segment

    cv2.setNumThreads(1)
    fr = D.load_frames(cfg)
    s = fr.sample(min(n, len(fr)), random_state=0)
    imgs = [to_work_size(D.imread(C.path(cfg, "raw") / p, gray=True), cfg["data"]["work_size"]) for p in s["path"]]
    rows = []
    for t in tqdm(cfg["segmentation"]["preprocess"], desc="Medindo o tempo de cada técnica", unit="técnica", file=sys.stdout):
        ms, processed = [], []
        for g in imgs:
            t0 = time.perf_counter()
            processed.append(_apply(g, t))
            ms.append((time.perf_counter() - t0) * 1000)
        ms_tech = 0.0 if t == "raw" else float(np.median(ms))
        for meth in cfg["segmentation"]["methods"]:
            pid = per_fold[(per_fold["preprocess"] == t) & (per_fold["method"] == meth)]["param_id"].mode().iloc[0]
            prm = PARAM_GRID[meth][int(pid)]
            ms2 = []
            for g in processed:
                t0 = time.perf_counter()
                segment(g, meth, **prm)
                ms2.append((time.perf_counter() - t0) * 1000)
            rows.append({"technique": t, "method": meth, "ms_technique": ms_tech, "ms_segmentation": float(np.median(ms2)),
                         "ms_total": ms_tech + float(np.median(ms2))})
    cv2.setNumThreads(-1)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- tabela única
def single_table(vis: pd.DataFrame, visv: pd.DataFrame, den: pd.DataFrame, summ: pd.DataFrame, segv: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por técnica. O melhor método é escolhido pelo Dice no **treino**; os números são do teste."""
    rows = []
    for t in vis["technique"]:
        q = vis[vis["technique"] == t].iloc[0]
        s = summ[summ["preprocess"] == t]
        b = s.sort_values("train_dice", ascending=False).iloc[0]
        r = {"tecnica": t, "nome": NAMES.get(t, t), "entropia": q["entropy"], "contraste_rms": q["rms_contrast"],
             "laplaciano": q["lap_var"], "saturacao": q["saturated_frac"], "fracao_especular": q["spec_frac"]}
        v = visv[visv["technique"] == t]
        r["veredito_visual"] = v["verdict"].iloc[0] if len(v) else ("referência" if t == "raw" else "ver ruído")
        r["motivo_visual"] = v["motivo"].iloc[0] if len(v) else ""
        d = den[den["filter"] == FILTER_KEY[t]] if t in FILTER_KEY else den.iloc[0:0]
        for noise, lab in (("sp_5", "sal_pimenta5"), ("gauss_s10", "gauss10")):
            x = d[d["noise"] == noise]
            r[f"psnr_{lab}"] = x["psnr"].iloc[0] if len(x) else np.nan
            r[f"ssim_{lab}"] = x["ssim"].iloc[0] if len(x) else np.nan
            r[f"ruido_{lab}"] = x["verdict"].iloc[0] if len(x) else ""
        sv = segv[(segv["technique"] == t) & (segv["method"] == b["method"])]
        ms_raw = summ[(summ["preprocess"] == "raw") & (summ["method"] == b["method"])]["ms_total"].iloc[0]
        r.update({"melhor_metodo": b["method"], "dice": b["dice_mean"], "iou": b["iou_mean"], "hd95_mediana": b["hd95_median"],
                  "taxa_falha": b["fail_rate"], "delta_dice_vs_raw": sv["delta_dice"].iloc[0] if len(sv) else 0.0,
                  "p_holm": sv["p_holm"].iloc[0] if len(sv) else np.nan, "veredito_seg": sv["verdict"].iloc[0] if len(sv) else "referência",
                  "ms_tecnica": b["ms_technique"], "ms_segmentacao": b["ms_segmentation"], "ms_total": b["ms_total"],
                  "ms_extra_vs_raw": b["ms_total"] - ms_raw})
        for m in ("otsu", "adaptive_morph", "otsu_morph", "region_grow"):
            x = s[s["method"] == m]
            r[f"dice_{m}"] = x["dice_mean"].iloc[0] if len(x) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def _figure(cfg: dict, summ: pd.DataFrame, table: pd.DataFrame) -> None:
    """Heatmap de um só matiz (magnitude do Dice) e dispersão Dice × custo com rótulos diretos."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    plt = _plt()
    techs, methods = list(table["tecnica"]), [m for m in METHODS if m in set(summ["method"])]
    M = summ.pivot(index="method", columns="preprocess", values="dice_mean").loc[methods, techs]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1.45, 1]})
    a.imshow(M.to_numpy(), cmap="Blues", vmin=0, vmax=max(0.8, float(M.max().max())), aspect="auto")
    a.set_xticks(range(len(techs)), [NAMES.get(t, t) for t in techs], rotation=35, ha="right")
    a.set_yticks(range(len(methods)), [METHODS[m] for m in methods])
    a.tick_params(length=0)
    for sp in a.spines.values():
        sp.set_visible(False)
    top = float(M.max().max())
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.iat[i, j]
            a.text(j, i, f"{v:.2f}".replace(".", ","), ha="center", va="center", fontsize=8, color="white" if v > 0.55 * max(0.8, top) else "#1B2A4A")
    a.set_title("Dice no teste: método de segmentação × técnica de PDI")
    b.scatter(table["ms_total"], table["dice"], s=46, color="#2a78d6", edgecolor="white", linewidth=1, zorder=3)
    for _, r in table.iterrows():
        b.annotate(r["nome"], (r["ms_total"], r["dice"]), xytext=(6, 4), textcoords="offset points", fontsize=8, color="#1B2A4A")
    b.set_xscale("log")
    b.xaxis.set_major_locator(FixedLocator([3, 5, 10, 15, 20]))
    b.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    b.xaxis.set_minor_locator(NullLocator())
    b.set_xlabel("Custo: técnica + segmentação (ms por frame, 1 núcleo da CPU, escala log)")
    b.set_ylabel("Dice no teste (melhor método de cada técnica)")
    b.grid(color="#C9CFDD", lw=0.5, alpha=0.6)
    b.set_axisbelow(True)
    b.set_title("O ganho na segmentação compensa o custo?")
    fig.tight_layout()
    fig.savefig(C.path(cfg, "figures") / "pdi_segmentacao_custo.png")
    plt.close(fig)


def _stage_summary(cfg: dict) -> list[dict]:
    cols = ["etapa", "duracao_s", "cpu_nucleos_medio", "ram_processo_pico_gb", "gpu_usada_pelo_processo"]
    s = latest_stages(cfg)[cols].query("etapa.str.startswith('pdi/')").astype(object)
    return s.where(s.notna(), None).to_dict("records")


def cmd_pdi_study(cfg: dict, args) -> dict:
    t0 = time.perf_counter()
    out = C.path(cfg, "tables") / "pdi"
    out.mkdir(parents=True, exist_ok=True)
    print("[1/4] Redução de ruído: PSNR e SSIM", flush=True)
    denoise_args = type("A", (), {"force": True, "jobs": args.jobs})()
    with medir(cfg, "pdi/1_ruido", "Ruído sintético, filtros, PSNR e SSIM") as e:
        r_ = cmd_denoise_study(cfg, denoise_args)
        e.n_itens = r_["n_frames"]
    den = denoise_verdicts(pd.read_csv(C.path(cfg, "tables") / "denoise_frames.csv.gz"))
    den.round(5).to_csv(out / "ruido.csv", index=False)

    print("[2/4] Qualidade da imagem e segmentação em todos os frames", flush=True)
    fr = D.load_frames(cfg)
    with medir(cfg, "pdi/2_tecnicas_e_segmentacao", "Técnicas, qualidade e segmentação em todos os frames", n_itens=len(fr)):
        qual, seg = _run_frames(cfg, args.jobs)

    print("[3/4] Parâmetros pelo treino e testes estatísticos", flush=True)
    from laringopdi.pdi.segmentation import PARAM_GRID
    grid = {m: PARAM_GRID[m] for m in cfg["segmentation"]["methods"]}
    with medir(cfg, "pdi/3_ajuste_e_estatistica", "Parâmetros pelo treino e testes estatísticos"):
        per_fold, chosen = seg_per_fold(seg, grid, D.load_splits(cfg), cfg["segmentation"]["tune_fold"], fr)
    per_fold.to_csv(out / "segmentacao_por_fold.csv", index=False)
    qv = qual.merge(fr[["path", "video_id"]], on="path").groupby(["technique", "video_id"])[list(QUALITY)].mean().reset_index()
    order = [t for t in cfg["segmentation"]["preprocess"]]
    vis = qual.groupby("technique")[list(QUALITY)].mean().loc[order].reset_index()
    enhance = tuple(t for t in ENHANCE if t in order)
    visv = visual_verdicts(qv, enhance)
    vis.merge(visv, on="technique", how="left").round(5).to_csv(out / "visual.csv", index=False)
    vd = _video_dice(cfg, seg, per_fold, fr)
    segv = seg_verdicts(vd)

    print("[4/4] Custo por técnica e tabela única", flush=True)
    with medir(cfg, "pdi/4_tempo_por_tecnica", "Tempo em 1 núcleo: 100 frames por técnica e método", n_itens=100):
        tm = timing(cfg, per_fold)
    summ = per_fold.groupby(["method", "preprocess"]).agg(
        train_dice=("train_dice", "mean"), dice_mean=("dice", "mean"), dice_sd=("dice", "std"), iou_mean=("iou", "mean"),
        hd95_median=("hd95_median", "median"), fail_rate=("fail_rate", "mean")).reset_index()
    summ = summ.merge(tm, left_on=["method", "preprocess"], right_on=["method", "technique"]).drop(columns="technique")
    summ.merge(segv.rename(columns={"technique": "preprocess"})[["method", "preprocess", "delta_dice", "p_holm", "verdict"]],
               on=["method", "preprocess"], how="left").round(5).to_csv(out / "segmentacao.csv", index=False)
    table = single_table(vis, visv, den, summ, segv)
    table.round(5).to_csv(out / "tabela_unica.csv", index=False)
    _figure(cfg, summ, table)

    resumo = {"frames": len(fr), "videos": int(fr["video_id"].nunique()), "tecnicas": len(order), "metodos": len(grid),
              "melhor_no_treino_fold0": chosen, "minutos": round((time.perf_counter() - t0) / 60, 1),
              "hardware": C.hardware(), "git_sha": C.git_sha()}
    old = C.path(cfg, "tables") / "segmentation_frames.csv.gz"
    if old.exists():              # conferência: o código novo reproduz os números antigos nas técnicas em comum?
        o = pd.read_csv(old, usecols=["path", "preprocess", "method", "param_id", "dice"])
        m = seg.merge(o, on=["path", "preprocess", "method", "param_id"], suffixes=("", "_old"))
        resumo["conferencia_antigo"] = {"linhas": len(m), "dif_max_dice": float((m["dice"] - m["dice_old"]).abs().max()) if len(m) else None}
    resumo["etapas"] = _stage_summary(cfg)
    C.write_json(out / "resumo.json", resumo)
    cols = ["tecnica", "veredito_visual", "melhor_metodo", "dice", "veredito_seg", "ms_total"]
    return {"minutos": resumo["minutos"], "tabela": table[cols].round(3).to_dict("records")}

"""Estudos que fecham o prometido na Etapa 1 (docs/ETAPA2_PLANO.md): ruído e PSNR/SSIM (P2), realce e
saturação (P1), MLP-18 (P4, P6), atenção normalizada (P5), tempo de processamento e relatório P1–P6.

Nada aqui altera `results/runs` dos experimentos existentes, os caches E00–E14, os splits nem o
`segmentation_best.json`. Regras de veredito: `docs/DECISIONS.md` (entrada de 2026-11-04).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from laringopdi import config as C
from laringopdi import data as D
from laringopdi.evaluate import _holm

INK, DIM, GRID = "#1B2A4A", "#5B6785", "#C9CFDD"
SERIES = {"ref": "#8A94AD", "mean": "#2a78d6", "median": "#eb6834", "gauss_lpf": "#1baf7a", "bilat": "#eda100"}
ENH_LABEL = {"raw": "raw", "stretch": "alargamento", "he": "HE", "clahe": "CLAHE", "homo": "homomórfica", "bilat": "bilateral"}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK,
                         "xtick.color": DIM, "ytick.color": DIM, "axes.titleweight": "bold", "axes.titlesize": 9.5,
                         "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 200, "savefig.bbox": "tight"})
    return plt


def _fig_dir(cfg: dict):
    p = C.path(cfg, "figures") / "relatorio"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _wilcoxon(a, b) -> float:
    from scipy.stats import wilcoxon
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(wilcoxon(b, a).pvalue) if (a != b).any() else 1.0


# ----------------------------------------------------------------------------- Bloco 2: ruído (P2)
NOISES = {"gauss_s10": ("gauss", 10), "gauss_s20": ("gauss", 20), "sp_5": ("sp", 0.05), "sp_10": ("sp", 0.10)}
NOISE_LABEL = {"gauss_s10": "gaussiano σ=10", "gauss_s20": "gaussiano σ=20", "sp_5": "sal e pimenta 5%", "sp_10": "sal e pimenta 10%"}
FILTERS = {"none": ("none", {}), "mean3": ("mean", {"ksize": 3}), "mean7": ("mean", {"ksize": 7}),
           "mean15": ("mean", {"ksize": 15}), "median3": ("median", {"ksize": 3}), "median5": ("median", {"ksize": 5}),
           "lpf15": ("gauss_lpf", {"sigma": 15}), "lpf40": ("gauss_lpf", {"sigma": 40}), "bilat": ("bilat", None)}
FILTER_LABEL = {"none": "sem filtro", "mean3": "média 3", "mean7": "média 7", "mean15": "média 15", "median3": "mediana 3",
                "median5": "mediana 5", "lpf15": "passa-baixa σ15", "lpf40": "passa-baixa σ40", "bilat": "bilateral"}


def add_noise(g: np.ndarray, noise: str, rng: np.random.Generator) -> np.ndarray:
    kind, v = NOISES[noise]
    if kind == "gauss":
        return np.clip(np.round(g + rng.normal(0, v, g.shape)), 0, 255).astype(np.uint8)
    r, out = rng.random(g.shape), g.copy()
    out[r < v / 2] = 0
    out[r > 1 - v / 2] = 255
    return out


def apply_filter(img: np.ndarray, name: str) -> np.ndarray:
    from laringopdi.pdi.registry import DEFAULTS, TECHNIQUES
    tech, prm = FILTERS[name]
    if tech == "none":
        return img
    return TECHNIQUES[tech](img, {}, **(DEFAULTS[tech] if prm is None else prm))


def canny_limits(g: np.ndarray, sigma: float = 0.33) -> tuple[int, int]:
    v = np.median(g[g > 0]) if (g > 0).any() else np.median(g)
    return int(max(0, (1 - sigma) * v)), int(min(255, (1 + sigma) * v))


def edge_f1(ref: np.ndarray, out: np.ndarray) -> float:
    """F1 entre dois mapas de bordas (preservação de borda, pixel a pixel)."""
    tp = int((ref & out).sum())
    n = int(ref.sum()) + int(out.sum())
    return 1.0 if n == 0 else 2.0 * tp / n


def _denoise_frame(path, fid: int, work_size: int) -> list[dict]:
    import time

    import cv2
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    from laringopdi.pdi.registry import to_work_size
    g = to_work_size(D.imread(path, gray=True), work_size)
    lo, hi = canny_limits(g)
    e0 = cv2.Canny(g, lo, hi) > 0
    rows = []
    for ni, noise in enumerate(NOISES):
        noisy = add_noise(g, noise, np.random.default_rng([0, fid, ni]))
        for fname in FILTERS:
            t0 = time.perf_counter()
            out = apply_filter(noisy, fname)
            ms = (time.perf_counter() - t0) * 1000
            rows.append({"frame_id": fid, "noise": noise, "filter": fname,
                         "psnr": peak_signal_noise_ratio(g, out, data_range=255),
                         "ssim": structural_similarity(g, out, data_range=255),
                         "edge_f1": edge_f1(e0, cv2.Canny(out, lo, hi) > 0), "ms": ms})
    return rows


def denoise_sample(cfg: dict, n: int = 300, seed: int = 0) -> pd.DataFrame:
    """No máximo 1 frame por vídeo, com as famílias na proporção dos vídeos (maiores restos)."""
    p = C.path(cfg, "tables") / "denoise_sample.json"
    fr = D.load_frames(cfg)
    if p.exists():
        keep = json.loads(p.read_text(encoding="utf-8"))["paths"]
        return fr.set_index("path").loc[keep].reset_index()
    one = fr.sample(frac=1, random_state=seed).drop_duplicates("video_id")
    cnt = one["family"].value_counts()
    quota = cnt * n / cnt.sum()
    take = np.floor(quota).astype(int)
    for fam in (quota - take).sort_values(ascending=False).index[: n - int(take.sum())]:
        take[fam] += 1
    s = pd.concat([one[one["family"] == f].head(int(k)) for f, k in take.items()]).sort_values("path")
    C.write_json(p, {"n": len(s), "seed": seed, "per_family": {k: int(v) for k, v in take.items()}, "paths": s["path"].tolist()})
    return s.reset_index(drop=True)


def _p2_tests(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Mediana 3×3 × média 3×3 em sal e pimenta (família de 4 testes do P2) e, como contraste, no
    ruído gaussiano, onde também entra o passa-baixa σ40 (família separada, não entra no veredito)."""
    rows = []
    pairs = [("sp_5", "mean3", "median3", "P2"), ("sp_10", "mean3", "median3", "P2"),
             ("gauss_s10", "mean3", "median3", "contraste"), ("gauss_s20", "mean3", "median3", "contraste"),
             ("gauss_s10", "lpf40", "median3", "contraste"), ("gauss_s20", "lpf40", "median3", "contraste")]
    for noise, a, b, fam in pairs:
        w = df[df["noise"] == noise].pivot(index="frame_id", columns="filter")
        for met in ("psnr", "ssim"):
            x, y = w[met][a], w[met][b]
            rows.append({"family": fam, "noise": noise, "metric": met, "a": a, "b": b, "n": len(x), "mean_a": x.mean(),
                         "mean_b": y.mean(), "delta_b_minus_a": (y - x).mean(), "p": _wilcoxon(x, y)})
    out = pd.DataFrame(rows)
    out["p_holm"] = out.groupby("family")["p"].transform(lambda p: _holm(p.to_numpy()))
    return out


def _denoise_figure(cfg: dict, summ: pd.DataFrame) -> None:
    plt = _plt()
    order = list(FILTERS)
    color = {f: SERIES["ref" if FILTERS[f][0] == "none" else FILTERS[f][0]] for f in order}
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.6), sharey=True)
    for c, noise in enumerate(NOISES):
        s = summ[summ["noise"] == noise].set_index("filter").loc[order]
        for r, (met, lab) in enumerate((("psnr", "PSNR (dB)"), ("ssim", "SSIM"))):
            ax = axes[r, c]
            y = np.arange(len(order))[::-1]
            ax.barh(y, s[f"{met}_mean"], height=0.72, color=[color[f] for f in order])
            ax.set_yticks(y, [FILTER_LABEL[f] for f in order])
            ax.set_xlabel(lab)
            if met == "ssim":
                ax.set_xlim(0, 1)
            ax.tick_params(length=0)
            ax.grid(axis="x", color=GRID, lw=0.5, alpha=0.6)
            ax.set_axisbelow(True)
            for yi, v in zip(y, s[f"{met}_mean"]):
                ax.text(v, yi, f" {v:.1f}" if met == "psnr" else f" {v:.2f}", va="center", fontsize=7, color=DIM)
            if r == 0:
                ax.set_title(NOISE_LABEL[noise])
    h = [plt.Rectangle((0, 0), 1, 1, color=SERIES[k]) for k in SERIES]
    fig.legend(h, ["sem filtro", "média", "mediana", "passa-baixa gaussiano", "bilateral"], loc="lower center", ncol=5,
               frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(_fig_dir(cfg) / "denoise.png")
    plt.close(fig)


def _denoise_example(cfg: dict, sample: pd.DataFrame) -> None:
    from laringopdi.pdi.registry import to_work_size
    plt = _plt()
    r = sample.iloc[0]
    g = to_work_size(D.imread(C.path(cfg, "raw") / r["path"], gray=True), cfg["data"]["work_size"])
    noisy = add_noise(g, "sp_5", np.random.default_rng([0, 0, 2]))
    cols = [(g, "original"), (noisy, "sal e pimenta 5%")] + [(apply_filter(noisy, f), FILTER_LABEL[f]) for f in ("mean3", "median3", "lpf15")]
    fig, axes = plt.subplots(1, 5, figsize=(11, 2.8))
    for a, (im, t) in zip(axes, cols):
        a.imshow(im, cmap="gray", vmin=0, vmax=255); a.set_title(t); a.axis("off")
    fig.savefig(C.path(cfg, "figures") / "denoise_exemplo.png")
    plt.close(fig)


def cmd_denoise_study(cfg: dict, args) -> dict:
    from joblib import Parallel, delayed

    from laringopdi.report import _save_table
    t = C.path(cfg, "tables")
    sample = denoise_sample(cfg)
    fp = t / "denoise_frames.csv.gz"
    if args.force or not fp.exists():
        raw, ws = C.path(cfg, "raw"), cfg["data"]["work_size"]
        res = Parallel(n_jobs=args.jobs, batch_size=4)(delayed(_denoise_frame)(raw / p, i, ws) for i, p in enumerate(sample["path"]))
        pd.DataFrame([x for r in res for x in r]).to_csv(fp, index=False)
    df = pd.read_csv(fp)
    g = df.groupby(["noise", "filter"])
    summ = g[["psnr", "ssim", "edge_f1"]].agg(["mean", "std"])
    summ.columns = [f"{m}_{'mean' if s == 'mean' else 'sd'}" for m, s in summ.columns]
    summ["ms_mean"] = g["ms"].mean()
    summ = summ.reset_index()
    _save_table(cfg, summ, "denoise_summary", "Redução de ruído sintético: PSNR, SSIM e preservação de borda (média e dp por frame)")
    p2 = _p2_tests(df, cfg)
    p2.round(6).to_csv(t / "denoise_p2.csv", index=False)
    _denoise_figure(cfg, summ)
    _denoise_example(cfg, sample)
    return {"n_frames": len(sample), "p2": p2[p2["family"] == "P2"][["noise", "metric", "mean_a", "mean_b", "p_holm"]].round(4).to_dict("records")}


# ----------------------------------------------------------------------------- Bloco 3: realce (P1)
ENHANCE_METRICS = ("entropy", "rms_contrast", "lap_var", "saturated_frac")
ENH_QUALITY = ("raw", "stretch", "he", "clahe", "homo")
SEG_NEW = ("stretch", "he")
SEG_ALL = ("raw", "stretch", "he", "clahe", "homo", "bilat")


def _enh_frame(path, work_size: int) -> list[dict]:
    from laringopdi.pdi.quality import image_quality
    from laringopdi.pdi.registry import DEFAULTS, TECHNIQUES, to_work_size
    g = to_work_size(D.imread(path, gray=True), work_size)
    rows = []
    for tech in ENH_QUALITY:
        out = g if tech == "raw" else TECHNIQUES[tech](g, {}, **DEFAULTS[tech])
        q = image_quality(out)
        rows.append({"technique": tech, **{k: q[k] for k in ENHANCE_METRICS}, "sat_low": float((out == 0).mean()),
                     "sat_high": float((out == 255).mean())})
    return rows


def _enhance_quality(cfg: dict, n_jobs: int, force: bool) -> pd.DataFrame:
    from joblib import Parallel, delayed
    p = C.path(cfg, "tables") / "enhance_quality.csv"
    if p.exists() and not force:
        return pd.read_csv(p)
    fr, raw, ws = D.load_frames(cfg), C.path(cfg, "raw"), cfg["data"]["work_size"]
    res = Parallel(n_jobs=n_jobs, batch_size=16)(delayed(_enh_frame)(raw / x, ws) for x in fr["path"])
    df = pd.DataFrame([x for r in res for x in r])
    s = df.groupby("technique").agg(["mean", "std"])
    s.columns = [f"{m}_{'mean' if k == 'mean' else 'sd'}" for m, k in s.columns]
    s = s.loc[list(ENH_QUALITY)].reset_index()
    s.insert(1, "n_frames", len(fr))
    s.round(5).to_csv(p, index=False)
    return s


def _seg_long(cfg: dict, n_jobs: int, force: bool) -> pd.DataFrame:
    """Linhas por frame dos 6 realces: reaproveita raw/clahe/homo/bilat do estudo original e calcula só
    stretch e he. Antes, confere em 60 frames que o código ainda reproduz os números salvos."""
    from joblib import Parallel, delayed

    from laringopdi.evaluate import _seg_frame
    from laringopdi.pdi.segmentation import PARAM_GRID
    t = C.path(cfg, "tables")
    p = t / "segmentation_etapa1_frames.csv.gz"
    if p.exists() and not force:
        return pd.read_csv(p)
    sc, ws, raw = cfg["segmentation"], cfg["data"]["work_size"], C.path(cfg, "raw")
    grid = {m: PARAM_GRID[m] for m in sc["methods"]}
    fr = D.load_frames(cfg)
    old = pd.read_csv(t / "segmentation_frames.csv.gz")
    chk = fr.sample(60, random_state=0)
    res = Parallel(n_jobs=n_jobs)(delayed(_seg_frame)(raw / a, raw / b, ["raw", "clahe"], grid, ws) for a, b in zip(chk["path"], chk["seg_path"]))
    now = pd.concat([pd.DataFrame(r).assign(path=x) for r, x in zip(res, chk["path"])], ignore_index=True)
    key = ["preprocess", "method", "param_id", "path"]
    m = now.merge(old, on=key, suffixes=("", "_old"))
    C.write_json(t / "segmentation_etapa1_check.json", {"frames": 60, "rows": len(m), "max_abs_diff_dice": float((m["dice"] - m["dice_old"]).abs().max())})
    res = Parallel(n_jobs=n_jobs, batch_size=8)(
        delayed(_seg_frame)(raw / a, raw / b, list(SEG_NEW), grid, ws) for a, b in zip(fr["path"], fr["seg_path"]))
    new = pd.concat([pd.DataFrame(r).assign(path=x) for r, x in zip(res, fr["path"])], ignore_index=True)
    long = pd.concat([old, new[old.columns]], ignore_index=True)
    long.to_csv(p, index=False)
    return long


def _video_dice(cfg: dict, long: pd.DataFrame, per_fold: pd.DataFrame) -> pd.DataFrame:
    """Dice por vídeo nos parâmetros escolhidos no treino de cada fold: índice (method, video_id), colunas = realces."""
    fr = D.load_frames(cfg)[["path", "video_id"]]
    x = long.merge(fr, on="path")
    test_of = {v: f["fold"] for f in D.load_splits(cfg) for v in f["test"]}
    x["fold"] = x["video_id"].map(test_of)
    sel = x.merge(per_fold[["fold", "preprocess", "method", "param_id"]], on=["fold", "preprocess", "method", "param_id"])
    return sel.groupby(["method", "preprocess", "video_id"])["dice"].mean().unstack("preprocess")


def seg_pairs(vid: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """Wilcoxon pareado por vídeo, b × a, em cada método, com Holm entre os métodos."""
    rows = []
    for meth, g in vid.groupby(level="method"):
        x = g[[a, b]].dropna()
        rows.append({"comparison": f"{b}_vs_{a}", "method": meth, "n_videos": len(x), "dice_a": x[a].mean(), "dice_b": x[b].mean(),
                     "delta_b_minus_a": (x[b] - x[a]).mean(), "p": _wilcoxon(x[a], x[b])})
    df = pd.DataFrame(rows)
    df["p_holm"] = _holm(df["p"].to_numpy())
    return df


def _classif_context(cfg: dict) -> pd.DataFrame:
    """E02 (HE) × E03 (CLAHE) na classificação do Estágio 1: só contexto, não entra no veredito de P1."""
    from laringopdi.report import paired_compare
    rows = []
    for model in cfg["stage1"]["models"]:
        for task in cfg["stage1"]["tasks"]:
            r = paired_compare(cfg, ("E02", model), ("E03", model), task)
            if r:
                rows.append({"model": model, "task": task, **r})
    return pd.DataFrame(rows)


def _realce_figure(cfg: dict, q: pd.DataFrame, summ: pd.DataFrame) -> None:
    plt = _plt()
    fig = plt.figure(figsize=(11, 3.7))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.75], wspace=0.62)
    tech = list(ENH_QUALITY)
    ys = np.arange(len(tech))[::-1]
    for i, (col, lab, fmt) in enumerate((("entropy", "Entropia (bits)", "{:.2f}"), ("rms_contrast", "Contraste RMS", "{:.3f}"),
                                         ("saturated_frac", "Fração saturada", "{:.1%}"))):
        ax = fig.add_subplot(gs[0, i])
        v = q.set_index("technique").loc[tech, f"{col}_mean"]
        ax.barh(ys, v, height=0.68, color=SERIES["mean"])
        ax.set_yticks(ys, [ENH_LABEL[t] for t in tech] if i == 0 else [""] * len(tech))
        ax.tick_params(length=0)
        ax.set_title(lab)
        ax.grid(axis="x", color=GRID, lw=0.5, alpha=0.6)
        ax.set_axisbelow(True)
        for y, x in zip(ys, v):
            ax.text(x, y, " " + fmt.format(x), va="center", fontsize=7, color=DIM)
        ax.margins(x=0.28)
    ax = fig.add_subplot(gs[0, 3])
    meth = ["region_grow", "otsu_morph", "watershed", "adaptive", "otsu"]
    w = summ.pivot(index="method", columns="preprocess", values="dice_mean").loc[meth, list(SEG_ALL)]
    im = ax.imshow(w.to_numpy(), cmap="Blues", vmin=0, vmax=0.7, aspect="auto")
    ax.set_xticks(range(len(SEG_ALL)), [ENH_LABEL[p] for p in SEG_ALL], rotation=30, ha="right")
    ax.set_yticks(range(len(meth)), meth)
    ax.tick_params(length=0)
    ax.set_title("Dice médio (método × realce)")
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            v = w.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if v > 0.42 else INK)
    fig.savefig(_fig_dir(cfg) / "realce.png")
    plt.close(fig)


def _realce_example(cfg: dict) -> None:
    """1 frame escuro da amostra do ruído: as 4 versões e seus histogramas (figura local, tem frame do BAGLS)."""
    from laringopdi.pdi.registry import DEFAULTS, TECHNIQUES, to_work_size
    plt = _plt()
    raw, ws = C.path(cfg, "raw"), cfg["data"]["work_size"]
    s = denoise_sample(cfg)
    imgs = [to_work_size(D.imread(raw / p, gray=True), ws) for p in s["path"]]
    g = imgs[int(np.argsort([i.mean() for i in imgs])[len(imgs) // 10])]
    fig, axes = plt.subplots(2, 4, figsize=(10, 4.2))
    for c, t in enumerate(("raw", "stretch", "he", "clahe")):
        out = g if t == "raw" else TECHNIQUES[t](g, {}, **DEFAULTS[t])
        axes[0, c].imshow(out, cmap="gray", vmin=0, vmax=255); axes[0, c].set_title(ENH_LABEL[t]); axes[0, c].axis("off")
        axes[1, c].hist(out.ravel(), bins=64, range=(0, 255), color=SERIES["mean"])
        axes[1, c].set_yticks([]); axes[1, c].set_xlabel("intensidade")
    fig.tight_layout()
    fig.savefig(C.path(cfg, "figures") / "realce_exemplo.png")
    plt.close(fig)


def cmd_enhance_study(cfg: dict, args) -> dict:
    from laringopdi.evaluate import seg_per_fold
    from laringopdi.pdi.segmentation import PARAM_GRID
    t = C.path(cfg, "tables")
    q = _enhance_quality(cfg, args.jobs, args.force)
    long = _seg_long(cfg, args.jobs, args.force)
    grid = {m: PARAM_GRID[m] for m in cfg["segmentation"]["methods"]}
    per_fold, _ = seg_per_fold(long, grid, D.load_splits(cfg), cfg["segmentation"]["tune_fold"], D.load_frames(cfg))
    per_fold.to_csv(t / "segmentation_etapa1.csv", index=False)
    summ = per_fold.groupby(["method", "preprocess"]).agg(
        dice_mean=("dice", "mean"), dice_sd=("dice", "std"), iou_mean=("iou", "mean"),
        hd95_median=("hd95_median", "median"), fail_rate=("fail_rate", "mean")).reset_index().sort_values("dice_mean", ascending=False)
    summ.round(4).to_csv(t / "segmentation_etapa1_summary.csv", index=False)
    ref = pd.read_csv(t / "segmentation_summary.csv")
    cmp_ = summ.round(4).merge(ref, on=["method", "preprocess"], suffixes=("", "_ref"))
    agree = {"rows": len(cmp_), "max_abs_diff_dice": float((cmp_["dice_mean"] - cmp_["dice_mean_ref"]).abs().max()),
             "max_abs_diff_iou": float((cmp_["iou_mean"] - cmp_["iou_mean_ref"]).abs().max())}
    vid = _video_dice(cfg, long, per_fold)
    pairs = pd.concat([seg_pairs(vid, "he", "clahe"), seg_pairs(vid, "he", "stretch"), seg_pairs(vid, "raw", "stretch")], ignore_index=True)
    pairs.round(6).to_csv(t / "seg_p1.csv", index=False)
    _classif_context(cfg).round(4).to_csv(t / "realce_classif_context.csv", index=False)
    _realce_figure(cfg, q, summ)
    _realce_example(cfg)
    chk = json.loads((t / "segmentation_etapa1_check.json").read_text(encoding="utf-8"))
    return {"agree_with_summary": agree, "recheck_60_frames": chk, "rows": len(long),
            "sat": q.set_index("technique")["saturated_frac_mean"].round(4).to_dict(),
            "clahe_vs_he": pairs[pairs["comparison"] == "clahe_vs_he"][["method", "dice_a", "dice_b", "p_holm"]].round(4).to_dict("records")}


# ----------------------------------------------------------------------------- Bloco 4: MLP-18 (P4, P6)
F18_LABEL = {"f18shape": "forma", "f18int": "intensidade", "f18tex": "textura"}


def _run_metrics(cfg: dict, exp: str, task: str, model: str) -> list[dict]:
    root = C.path(cfg, "runs") / exp / task / model
    return [json.loads((root / f"fold{k}_seed0" / "metrics.json").read_text(encoding="utf-8")) for k in cfg["stage1"]["folds"]]


def _importance(cfg: dict, exp: str, model: str) -> pd.DataFrame:
    rows = []
    for t in ("A", "B"):
        ms = _run_metrics(cfg, exp, t, model)
        for fam in ms[0]["perm_importance"]:
            v = [m["perm_importance"][fam]["mean"] for m in ms]
            rows.append({"exp": exp, "task": t, "family": fam, "drop_mean": float(np.mean(v)), "drop_sd_folds": float(np.std(v, ddof=1)), "n_folds": len(v)})
    return pd.DataFrame(rows)


def _importance_figure(cfg: dict, imp: pd.DataFrame) -> None:
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.5), sharex=True, sharey=True)
    fams = list(F18_LABEL)
    y = np.arange(len(fams))[::-1]
    for ax, t in zip(axes, ("A", "B")):
        s = imp[imp["task"] == t].set_index("family").loc[fams]
        ax.barh(y, s["drop_mean"], height=0.62, color=SERIES["mean"], xerr=s["drop_sd_folds"], error_kw={"ecolor": DIM, "lw": 0.8})
        ax.set_yticks(y, [F18_LABEL[f] for f in fams])
        ax.axvline(0, color=INK, lw=0.7)
        ax.tick_params(length=0)
        ax.grid(axis="x", color=GRID, lw=0.5, alpha=0.6)
        ax.set_axisbelow(True)
        ax.set_title(f"Tarefa {t}")
        ax.set_xlabel("queda da acurácia balanceada ao permutar a família")
        for yi, f in zip(y, fams):
            ax.annotate(f"{s.loc[f, 'drop_mean']:+.3f}", (0, yi), xytext=(4, 9), textcoords="offset points", fontsize=7.5, color=DIM)
    fig.tight_layout()
    fig.savefig(_fig_dir(cfg) / "importancia_c6.png")
    plt.close(fig)


def cmd_mlp18_study(cfg: dict, args) -> dict:
    """Roda o C6 (se faltar), a importância por família e a comparação com a ResNet-18 do E00 (P4)."""
    from laringopdi.report import paired_compare
    from laringopdi.train import cmd_classic
    t = C.path(cfg, "tables")
    args.exp = "C6"
    cmd_classic(cfg, args)
    imp = _importance(cfg, "C6", "mlp18")
    imp.round(4).to_csv(t / "importance_c6.csv", index=False)
    pd.concat([_importance(cfg, e, s["model"]) for e, s in cfg["classic_experiments"].items() if e != "C6"], ignore_index=True) \
        .round(4).to_csv(t / "importance_classic.csv", index=False)
    rows = []
    for sys_, model in (("C6", "mlp18"), ("C1", "mlp")):
        for task in ("A", "B"):
            r = paired_compare(cfg, ("E00", "resnet18"), (sys_, model), task)
            rows.append({"system": sys_, "task": task, "f1_resnet18": r["f1_a"], "f1_classic": r["f1_b"], "delta": r["delta"], "ci_low": r["ci_low"],
                         "ci_high": r["ci_high"], "ci_width": r["ci_high"] - r["ci_low"], "mcnemar_p": r["mcnemar_p"], "n_seeds": r["n_seeds"], "n_videos": r["n_videos"]})
    p4 = pd.DataFrame(rows)
    p4.round(4).to_csv(t / "mlp18_p4.csv", index=False)
    f = pd.read_csv(C.path(cfg, "cache").parent / "features" / "f18.csv", usecols=["video_id", "f18shape_area_rel"], dtype={"video_id": str})
    empty = f["f18shape_area_rel"] == 0
    summ = []
    for task in ("A", "B"):
        ms = _run_metrics(cfg, "C6", task, "mlp18")
        v = pd.DataFrame([m["video"] for m in ms]).drop(columns=["confusion", "n"])
        summ.append({"task": task, **{f"{c}_mean": v[c].mean() for c in v}, "f1_sd": v["f1_macro"].std(), "n_features": ms[0]["n_features"]})
    C.write_json(t / "mlp18_summary.json", {"per_task": summ, "empty_mask_frame_rate": float(empty.mean()),
                                            "videos_all_empty": int(empty.groupby(f["video_id"]).all().sum()), "n_frames": len(f)})
    _importance_figure(cfg, imp)
    return {"f1": {s["task"]: round(s["f1_macro_mean"], 3) for s in summ}, "empty_rate": round(float(empty.mean()), 4),
            "p4": p4[["system", "task", "f1_resnet18", "f1_classic", "ci_low", "ci_high"]].round(3).to_dict("records"),
            "imp": imp[["task", "family", "drop_mean"]].round(3).to_dict("records")}


# ----------------------------------------------------------------------------- Bloco 5: atenção (P5)
P5_NOCROP = ["E01", "E04", "E05", "E10", "E11", "E14"]
P5_CROP = ["E07", "E08"]


def attention_video_ratio(cfg: dict) -> pd.DataFrame:
    """Razão de atenção por vídeo (energia na glote ÷ área da glote; 1 = acaso). Só vídeos com rótulo na
    tarefa; frames sem glote são descartados porque a razão fica indefinida."""
    from laringopdi.train import frame_labels
    a = pd.read_csv(C.path(cfg, "tables") / "attention_frames.csv.gz", dtype={"video_id": str},
                    usecols=["exp", "model", "task", "video_id", "att_glottis", "area_glottis"])
    lab = frame_labels(cfg).drop_duplicates("video_id").set_index("video_id")
    keep = {t: set(lab.index[lab[D.task_col(t)] >= 0]) for t in ("A", "B")}
    a = a[[v in keep[t] for v, t in zip(a["video_id"], a["task"])]]
    a = a[(a["area_glottis"] > 0) & a["att_glottis"].notna()].copy()
    a["ratio"] = a["att_glottis"] / a["area_glottis"]
    return a.groupby(["exp", "model", "task", "video_id"])["ratio"].mean().reset_index()


def p5_tests(v: pd.DataFrame, exps: list[str], family: str) -> pd.DataFrame:
    rows = []
    for (model, task), g in v.groupby(["model", "task"]):
        w = g.pivot(index="video_id", columns="exp", values="ratio")
        sub = []
        for e in exps:
            x = w[["E00", e]].dropna()
            sub.append({"family": family, "model": model, "task": task, "exp": e, "n_videos": len(x), "ratio_E00": x["E00"].mean(), "ratio_exp": x[e].mean(),
                        "median_E00": x["E00"].median(), "median_exp": x[e].median(), "delta": (x[e] - x["E00"]).mean(), "p": _wilcoxon(x["E00"], x[e])})
        d = pd.DataFrame(sub)
        d["p_holm"] = _holm(d["p"].to_numpy())
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def _p5_validity(cfg: dict, exps: list[str]) -> pd.DataFrame:
    """Fração dos pixels da glote com X = 0 em cada experimento: eles saem da energia do Grad-CAM (o
    `gradcam` usou valid = X > 0), mas continuam na área, o que reduz a razão artificialmente."""
    from laringopdi.features import _unpack
    size, rows = cfg["data"]["image_size"], []
    for e in exps:
        X, _, _ = D.load_cache(cfg, e)
        gt = _unpack(np.load(D.cache_paths(cfg, e)["masks"])["gt_mask_lb"], size)
        zero = total = 0
        for i in range(len(gt)):
            px = np.asarray(X[i, ..., 0])[gt[i]]
            zero += int((px == 0).sum())
            total += px.size
        rows.append({"exp": e, "pipeline": cfg["experiments"][e]["pipeline"], "gt_zero_frac": zero / max(total, 1)})
    return pd.DataFrame(rows)


def _p5_figure(cfg: dict, df: pd.DataFrame, cfg_alpha: float) -> None:
    plt = _plt()
    fig, axes = plt.subplots(2, 2, figsize=(9, 5), sharex=True, sharey=True)
    names = {e: cfg["experiments"][e]["pipeline"] for e in P5_NOCROP}
    y = np.arange(len(P5_NOCROP))[::-1]
    for r, model in enumerate(("resnet18", "simplecnn")):
        for c, task in enumerate(("A", "B")):
            ax = axes[r, c]
            s = df[(df["model"] == model) & (df["task"] == task)].set_index("exp").loc[P5_NOCROP]
            base = s["ratio_E00"].mean()
            ax.axvline(base, color=SERIES["ref"], lw=1.6, label="E00 (sem PDI)")
            ax.axvline(1, color=INK, lw=0.7, ls=":", label="acaso (razão = 1)")
            sig = s["p_holm"] < cfg_alpha
            ax.scatter(s["ratio_exp"], y, s=46, color=[SERIES["median"] if b else SERIES["mean"] for b in sig], zorder=3)
            for yi, e in zip(y, P5_NOCROP):
                ax.annotate(f"{s.loc[e, 'ratio_exp']:.2f}" + (" *" if s.loc[e, "p_holm"] < cfg_alpha else ""), (s.loc[e, "ratio_exp"], yi),
                            xytext=(6, 3), textcoords="offset points", fontsize=7.5, color=DIM)
            ax.set_yticks(y, [names[e] for e in P5_NOCROP])
            ax.grid(axis="x", color=GRID, lw=0.5, alpha=0.6)
            ax.set_axisbelow(True)
            ax.tick_params(length=0)
            ax.set_title(f"{'ResNet-18' if model == 'resnet18' else 'SimpleCNN'}, Tarefa {task}")
            if r == 1:
                ax.set_xlabel("razão de atenção na glote (média por vídeo)")
    h, l = axes[0, 0].get_legend_handles_labels()
    dot = [plt.Line2D([], [], ls="", marker="o", color=SERIES["mean"]), plt.Line2D([], [], ls="", marker="o", color=SERIES["median"])]
    fig.legend(h + dot, l + ["técnica, sem diferença", "técnica, p_Holm < 0,05 (*)"], loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(_fig_dir(cfg) / "atencao_p5.png")
    plt.close(fig)


def cmd_attention_p5(cfg: dict, args) -> dict:
    t = C.path(cfg, "tables")
    v = attention_video_ratio(cfg)
    df = pd.concat([p5_tests(v, P5_NOCROP, "sem_recorte"), p5_tests(v, P5_CROP, "recorte")], ignore_index=True)
    df.round(6).to_csv(t / "attention_p5.csv", index=False)
    val = _p5_validity(cfg, ["E00"] + P5_NOCROP)
    val.round(5).to_csv(t / "attention_p5_validity.csv", index=False)
    a = cfg["stats"]["alpha"]
    _p5_figure(cfg, df[df["family"] == "sem_recorte"], a)
    n = df[df["family"] == "sem_recorte"]
    return {"combos": len(n), "up": int(((n["p_holm"] < a) & (n["delta"] > 0)).sum()), "down": int(((n["p_holm"] < a) & (n["delta"] < 0)).sum()),
            "gt_zero": val.set_index("exp")["gt_zero_frac"].round(4).to_dict()}


# ----------------------------------------------------------------------------- Bloco 6: tempo
def cmd_timing(cfg: dict, args) -> dict:
    """Mediana em ms/frame (CPU, um processo, escala de trabalho), com 5 aquecimentos descartados."""
    import platform
    import time

    import cv2
    import torch

    from laringopdi.evaluate import load_model
    from laringopdi.features import _unpack, features18
    from laringopdi.pdi.registry import TECHNIQUES, resolve_params, run_pipeline, to_work_size
    from laringopdi.pdi.segmentation import best_config, segment
    from laringopdi.train import _prep, run_dir

    raw, ws, size = C.path(cfg, "raw"), cfg["data"]["work_size"], cfg["data"]["image_size"]
    s = denoise_sample(cfg).head(100)
    imgs, gts = [], []
    for p, q in zip(s["path"], s["seg_path"]):
        im = D.imread(raw / p, gray=True)
        imgs.append(im)
        gts.append(D.imread(raw / q, gray=True) > 127)
    work = [to_work_size(i, ws) for i in imgs]
    wgt = [cv2.resize(m.astype(np.uint8), w.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool) for m, w in zip(gts, work)]
    rows = []

    def measure(kind: str, name: str, fn) -> None:
        for _ in range(5):
            fn(0)
        ts = []
        for i in range(len(s)):
            t0 = time.perf_counter()
            fn(i)
            ts.append((time.perf_counter() - t0) * 1000)
        rows.append({"kind": kind, "name": name, "median_ms": float(np.median(ts)), "p90_ms": float(np.percentile(ts, 90)), "n": len(ts)})

    for name in TECHNIQUES:
        prm = resolve_params(name)[name]
        measure("técnica", name, lambda i, n=name, q=prm: TECHNIQUES[n](work[i], {"orig_shape": work[i].shape, "work_shape": work[i].shape, "gt_mask": wgt[i]}, **q))
    for e, v in cfg["experiments"].items():
        measure("pipeline", f"{e} ({v['pipeline']})", lambda i, pl=v["pipeline"], q=v.get("params"): run_pipeline(imgs[i], pl, q, size, ws, gts[i]))
    b = best_config()
    pre = b.get("preprocess", "raw")
    measure("segmentação", f"{b['method']} + {pre}", lambda i: segment(work[i] if pre == "raw" else TECHNIQUES[pre](work[i], {}, **resolve_params(pre)[pre]), b["method"], **b["params"]))
    X, paths, _ = D.load_cache(cfg, "E00")
    pos = {p: i for i, p in enumerate(paths)}
    idx = [pos[p] for p in s["path"]]
    masks = _unpack(np.load(D.cache_paths(cfg, "E13")["masks"])["glottis_mask_lb"], size)
    g224 = [np.ascontiguousarray(X[i, ..., 0]) for i in idx]
    measure("características", "18 características (C6)", lambda i: features18(g224[i], masks[idx[i]]))
    for model_name in ("resnet18", "simplecnn"):
        m = load_model(cfg, "E00", "A", model_name, 0)
        xs = [torch.from_numpy(np.array(X[i])[None]) for i in idx]
        dev = torch.device("cpu")

        def infer(i, net=m):
            with torch.no_grad():
                net(_prep(xs[i], dev))
        measure("inferência", f"{model_name} (CPU, batch 1)", infer)
    import joblib
    bundle = joblib.load(run_dir(cfg, "C6", "A", "mlp18", 0, 0) / "model.joblib")
    vec = pd.DataFrame([features18(g224[i], masks[idx[i]]) for i in range(len(idx))])[bundle["columns"]].to_numpy()
    measure("inferência", "MLP C6 (1 vetor)", lambda i: bundle["model"].predict_proba(bundle["scaler"].transform(vec[i:i + 1])))
    df = pd.DataFrame(rows)
    df["cpu"] = platform.processor()
    df["torch_threads"] = torch.get_num_threads()
    df.round(3).to_csv(C.path(cfg, "tables") / "timing.csv", index=False)
    return {"rows": len(df), "cpu": platform.processor(), "slow": df.sort_values("median_ms").tail(4)[["name", "median_ms"]].round(1).to_dict("records")}


# ----------------------------------------------------------------------------- Bloco 7: relatório P1–P6
P_TEXT = {
    "P1": ("CLAHE > equalização global", "a iluminação é desigual e a equalização global \"estoura\" o tecido",
           "Confirmada: CLAHE > HE no Dice do `region_grow` (p_Holm < 0,05), em pelo menos 3 dos 5 métodos, e fração saturada de HE > CLAHE. Parcial: vale só parte disso. Refutada: HE > CLAHE com significância no `region_grow`."),
    "P2": ("mediana > média em sal e pimenta", "a mediana ignora valores extremos e preserva a borda",
           "Confirmada: mediana 3×3 > média 3×3 em PSNR e SSIM nas duas densidades (p_Holm < 0,05). Parcial: vale só em parte. Refutada: a média vence com significância e a mediana em nenhum teste."),
    "P3": ("limiar adaptativo > global (Otsu)", "o reflexo desloca o limiar único",
           "Confirmada: Dice por vídeo de `adaptive` > `otsu` sobre `raw`, com p < 0,05 e Δ ≥ 0,05. Parcial: p < 0,05 com Δ < 0,05. Refutada: Otsu > adaptativo com p < 0,05."),
    "P4": ("MLP ≈ CNN em saudável × lesão estrutural (Tarefa B)", "a lesão deforma a glote",
           "Confirmada (\"competitiva\"): o IC 95% do ΔF1 por vídeo (C6 − ResNet-18 E00) contém 0 ou está acima de 0. Refutada: está todo abaixo de 0."),
    "P5": ("PDI na entrada aumenta a atenção da CNN na glote", "menos reflexo, mais contraste",
           "Das 24 combinações técnica × modelo × tarefa sem recorte: confirmada se ≥ 50% sobem (p_Holm < 0,05) e nenhuma cai; parcial se 1 ou mais sobem; refutada se nenhuma sobe e alguma cai; inconclusiva se nenhuma é significativa."),
    "P6": ("a forma da glote é o grupo de características mais importante", "nódulo e pólipo mudam o contorno",
           "No C6, a queda média por permutação da família forma é a maior das 3 (e positiva): confirmada nas duas tarefas; parcial em uma; refutada em nenhuma."),
}
P_H_MAP = [("P1", "H2"), ("P2", "sem par"), ("P3", "H4, em parte"), ("P4", "H6, em parte"), ("P5", "H1 (atenção)"), ("P6", "sem par")]


def _fp(p: float) -> str:
    return "< 0,001" if p < 0.001 else f"{p:.3f}".replace(".", ",")


def _fn(v: float, d: int = 3) -> str:
    return f"{v:.{d}f}".replace(".", ",")


def _verdict_p1(cfg: dict, a: float) -> dict:
    t = C.path(cfg, "tables")
    s = pd.read_csv(t / "seg_p1.csv")
    q = pd.read_csv(t / "enhance_quality.csv").set_index("technique")
    c = s[s["comparison"] == "clahe_vs_he"].set_index("method")
    rg = c.loc["region_grow"]
    va = bool(rg["delta_b_minus_a"] > 0 and rg["p_holm"] < a)
    refuted = bool(rg["delta_b_minus_a"] < 0 and rg["p_holm"] < a)
    n = int(((c["delta_b_minus_a"] > 0) & (c["p_holm"] < a)).sum())
    n_he = int(((c["delta_b_minus_a"] < 0) & (c["p_holm"] < a)).sum())
    sh, sc = q.loc["he", "saturated_frac_mean"], q.loc["clahe", "saturated_frac_mean"]
    vb = bool(sh > sc)
    v = "refutada" if refuted else "confirmada" if (va and vb and n >= 3) else "parcialmente confirmada" if (va or vb) else "inconclusiva"
    ctx = pd.read_csv(t / "realce_classif_context.csv")
    txt = (f"`region_grow`: Dice CLAHE {_fn(rg['dice_b'])} × HE {_fn(rg['dice_a'])}, p_Holm {_fp(rg['p_holm'])}. "
           f"CLAHE vence com significância em {n} de 5 métodos; HE vence em {n_he}. Fração saturada: HE {sh:.2%} × CLAHE {sc:.2%}. "
           f"Ressalva: a imagem sem realce já tem {q.loc['raw', 'saturated_frac_mean']:.2%} de pixels em 0 ou 255, quase todos pretos (fundo do frame); "
           f"o HE quase não acrescenta nada a isso, e o CLAHE fica baixo porque tira esses pixels do zero ({q.loc['raw', 'sat_low_mean']:.2%} → {q.loc['clahe', 'sat_low_mean']:.2%}). "
           "Logo, a condição (b) é atendida, mas não demonstra que o HE \"estoura\" o tecido. "
           f"Contexto, fora do veredito: na classificação do Estágio 1, ΔF1 (CLAHE − HE) varia de {_fn(ctx['delta'].min())} a {_fn(ctx['delta'].max())}, "
           f"com McNemar mínimo {_fp(ctx['mcnemar_p'].min())}.")
    return {"verdict": v, "evidence": txt, "numbers": {"a_region_grow": va, "b_saturation": vb, "methods_clahe_wins": n, "methods_he_wins": n_he,
                                                       "dice_clahe": float(rg["dice_b"]), "dice_he": float(rg["dice_a"]), "p_holm": float(rg["p_holm"]),
                                                       "sat_he": float(sh), "sat_clahe": float(sc)}}


def _verdict_p2(cfg: dict, a: float) -> dict:
    d = pd.read_csv(C.path(cfg, "tables") / "denoise_p2.csv")
    p = d[d["family"] == "P2"]
    med = int(((p["delta_b_minus_a"] > 0) & (p["p_holm"] < a)).sum())
    mea = int(((p["delta_b_minus_a"] < 0) & (p["p_holm"] < a)).sum())
    v = "confirmada" if med == 4 else "refutada" if (mea and not med) else "parcialmente confirmada" if med else "inconclusiva"
    x = p.set_index(["noise", "metric"])
    txt = "; ".join(f"{NOISE_LABEL[n]}: PSNR mediana {_fn(x.loc[(n, 'psnr'), 'mean_b'], 1)} × média {_fn(x.loc[(n, 'psnr'), 'mean_a'], 1)} dB, "
                    f"SSIM {_fn(x.loc[(n, 'ssim'), 'mean_b'], 2)} × {_fn(x.loc[(n, 'ssim'), 'mean_a'], 2)}" for n in ("sp_5", "sp_10"))
    txt += f". Mediana vence com significância em {med} de 4 testes (maior p_Holm {_fp(p['p_holm'].max())}; n = {int(p['n'].iloc[0])} frames). A referência é o próprio frame, que já tem ruído."
    return {"verdict": v, "evidence": txt, "numbers": {"median_wins": med, "mean_wins": mea, "max_p_holm": float(p["p_holm"].max())}}


def _verdict_p3(cfg: dict, a: float) -> dict:
    t = C.path(cfg, "tables")
    vid = _video_dice(cfg, pd.read_csv(t / "segmentation_frames.csv.gz"), pd.read_csv(t / "segmentation.csv"))["raw"].unstack("method")
    x = vid[["otsu", "adaptive"]].dropna()
    delta = float((x["adaptive"] - x["otsu"]).mean())
    p = _wilcoxon(x["otsu"], x["adaptive"])
    v = ("confirmada" if delta >= 0.05 else "parcialmente confirmada") if (p < a and delta > 0) else "refutada" if (p < a and delta < 0) else "inconclusiva"
    m = vid.mean()
    txt = (f"Sobre `raw`, por vídeo (n = {len(x)}): `adaptive` {_fn(x['adaptive'].mean())} × `otsu` {_fn(x['otsu'].mean())}, Δ = {_fn(delta)}, p {_fp(p)}. "
           f"Nuance, fora do veredito: `otsu_morph` {_fn(m['otsu_morph'])} e `region_grow` {_fn(m['region_grow'])}.")
    return {"verdict": v, "evidence": txt, "numbers": {"dice_adaptive": float(x["adaptive"].mean()), "dice_otsu": float(x["otsu"].mean()), "delta": delta, "p": p,
                                                       "dice_otsu_morph": float(m["otsu_morph"]), "dice_region_grow": float(m["region_grow"])}}


def _verdict_p4(cfg: dict, a: float) -> dict:
    d = pd.read_csv(C.path(cfg, "tables") / "mlp18_p4.csv").set_index(["system", "task"])
    r = d.loc[("C6", "B")]
    v = "refutada" if r["ci_high"] < 0 else "confirmada"

    def f(x):
        return (f"F1 {_fn(x['f1_classic'])} × ResNet-18 {_fn(x['f1_resnet18'])}, ΔF1 {_fn(x['delta'])} [IC 95% {_fn(x['ci_low'])}; {_fn(x['ci_high'])}], McNemar p {_fp(x['mcnemar_p'])}")
    txt = (f"C6, Tarefa B ({int(r['n_videos'])} vídeos): {f(r)}. Largura do IC: {_fn(r['ci_width'])}; \"competitiva\" não significa equivalente. "
           f"Contexto: C6, Tarefa A: {f(d.loc[('C6', 'A')])}. Extensão C1 (187 características), Tarefa B: {f(d.loc[('C1', 'B')])}. "
           "Comparação contra a semente 0 da ResNet-18, porque o C6 tem uma semente só.")
    return {"verdict": v, "evidence": txt, "numbers": {k: float(r[k]) for k in ("f1_classic", "f1_resnet18", "delta", "ci_low", "ci_high", "ci_width", "mcnemar_p")}}


def _verdict_p5(cfg: dict, a: float) -> dict:
    t = C.path(cfg, "tables")
    d = pd.read_csv(t / "attention_p5.csv")
    n = d[d["family"] == "sem_recorte"]
    up = n[(n["p_holm"] < a) & (n["delta"] > 0)]
    down = n[(n["p_holm"] < a) & (n["delta"] < 0)]
    v = "confirmada" if (len(up) >= len(n) / 2 and down.empty) else "parcialmente confirmada" if len(up) else "refutada" if len(down) else "inconclusiva"
    val = pd.read_csv(t / "attention_p5_validity.csv").set_index("exp")["gt_zero_frac"]
    crop = d[d["family"] == "recorte"].groupby("exp")[["ratio_E00", "ratio_exp"]].mean()
    txt = (f"Sobem com significância: {len(up)} de {len(n)} combinações"
           + (f" ({', '.join(sorted(set(up['exp'])))})" if len(up) else "") + f"; caem: {len(down)}"
           + (f" ({', '.join(sorted(set(down['exp'])))})" if len(down) else "")
           + f". Razão média no E00: {_fn(n['ratio_E00'].mean(), 2)} (1 = acaso). "
           f"Pixels da glote com X = 0, que a medida exclui da energia: E00 {val['E00']:.2%}, " + ", ".join(f"{e} {val[e]:.2%}" for e in val.index if e != "E00")
           + ". Com recorte, à parte e com a ressalva da área: " + "; ".join(f"{e} {_fn(r['ratio_exp'], 2)} (E00 {_fn(r['ratio_E00'], 2)})" for e, r in crop.iterrows()) + ".")
    return {"verdict": v, "evidence": txt, "numbers": {"n": len(n), "up": len(up), "down": len(down), "up_exps": sorted(set(up["exp"])), "down_exps": sorted(set(down["exp"]))}}


def _verdict_p6(cfg: dict, a: float) -> dict:
    t = C.path(cfg, "tables")
    d = pd.read_csv(t / "importance_c6.csv")
    won = {}
    for task, g in d.groupby("task"):
        s = g.set_index("family")["drop_mean"]
        won[task] = bool(s.idxmax() == "f18shape" and s["f18shape"] > 0)
    k = sum(won.values())
    v = "confirmada" if k == 2 else "parcialmente confirmada" if k == 1 else "refutada"
    fmt = lambda task: ", ".join(f"{F18_LABEL[r['family']]} {_fn(r['drop_mean'])}" for _, r in d[d["task"] == task].iterrows())
    c = pd.read_csv(t / "importance_classic.csv")
    top = c.groupby(["task", "family"])["drop_mean"].mean().reset_index()
    tops = "; ".join(f"Tarefa {task}: " + ", ".join(f"{r['family']} {_fn(r['drop_mean'])}" for _, r in g.sort_values("drop_mean", ascending=False).iterrows()) for task, g in top.groupby("task"))
    txt = f"C6, queda média por permutação. Tarefa A: {fmt('A')}. Tarefa B: {fmt('B')}. Complementar, média de C1–C5 (187 características): {tops}."
    return {"verdict": v, "evidence": txt, "numbers": {"shape_top": won}}


def cmd_etapa1_report(cfg: dict, args) -> dict:
    """docs/hipoteses_etapa1.md + results/tables/hypotheses_etapa1.json, a partir das tabelas e das regras pré-registradas."""
    a = cfg["stats"]["alpha"]
    fn = {"P1": _verdict_p1, "P2": _verdict_p2, "P3": _verdict_p3, "P4": _verdict_p4, "P5": _verdict_p5, "P6": _verdict_p6}
    res = {}
    for k, f in fn.items():
        st, why, rule = P_TEXT[k]
        res[k] = {"statement": st, "rationale": why, "rule": rule, **f(cfg, a)}
    C.write_json(C.path(cfg, "tables") / "hypotheses_etapa1.json", res)
    md = ["# Hipóteses da apresentação da Etapa 1 (P1–P6)", "",
          f"> Gerado por `python -m laringopdi etapa1-report` (git {C.git_sha()}). **Não edite à mão.**",
          "> As regras foram fixadas em `docs/DECISIONS.md` (2026-11-04) antes de gerar os números novos. P1, P3, P4 e P6 já tinham evidência parcial vista, como a entrada registra.", "",
          "## Resumo", "", "| ID | Hipótese | Veredito |", "|---|---|---|",
          *[f"| {k} | {r['statement']} | **{r['verdict']}** |" for k, r in res.items()], "", "## Detalhe", ""]
    for k, r in res.items():
        md += [f"### {k}: {r['statement']}", "", f"- **Por que esperávamos isso:** {r['rationale']}.", f"- **Regra pré-registrada:** {r['rule']}",
               f"- **Evidência:** {r['evidence']}", f"- **Veredito: {r['verdict']}.**", ""]
    md += ["## Correspondência com as hipóteses internas (`docs/hipoteses.md`)", "", "| Apresentada | Interna |", "|---|---|", *[f"| {p} | {h} |" for p, h in P_H_MAP], ""]
    (C.path(cfg, "docs") / "hipoteses_etapa1.md").write_text("\n".join(md), encoding="utf-8")
    return {k: r["verdict"] for k, r in res.items()}

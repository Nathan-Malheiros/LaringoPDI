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
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.75], wspace=0.42)
    tech = list(ENH_QUALITY)
    ys = np.arange(len(tech))[::-1]
    for i, (col, lab, fmt) in enumerate((("entropy", "Entropia (bits)", "{:.2f}"), ("rms_contrast", "Contraste RMS", "{:.3f}"),
                                         ("saturated_frac", "Fração saturada (0 ou 255)", "{:.2%}"))):
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

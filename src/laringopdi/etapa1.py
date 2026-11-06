"""Estudo de redução de ruído prometido na Etapa 1: ruído sintético, PSNR, SSIM e preservação de borda."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from laringopdi import config as C
from laringopdi import data as D
from laringopdi.evaluate import _holm

INK, DIM, GRID = "#1B2A4A", "#5B6785", "#C9CFDD"
SERIES = {"ref": "#8A94AD", "mean": "#2a78d6", "median": "#eb6834", "gauss_lpf": "#1baf7a"}


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
           "lpf15": ("gauss_lpf", {"sigma": 15}), "lpf40": ("gauss_lpf", {"sigma": 40})}
FILTER_LABEL = {"none": "sem filtro", "mean3": "média 3", "mean7": "média 7", "mean15": "média 15", "median3": "mediana 3",
                "median5": "mediana 5", "lpf15": "passa-baixa σ15", "lpf40": "passa-baixa σ40"}


def add_noise(g: np.ndarray, noise: str, rng: np.random.Generator) -> np.ndarray:
    return noise_image(g, *NOISES[noise], rng)


def noise_image(g: np.ndarray, kind: str, v: float, rng: np.random.Generator) -> np.ndarray:
    """kind = "gauss" (v = σ na escala 0–255) ou "sp" (v = densidade total de sal e pimenta)."""
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
    fig.legend(h, ["sem filtro", "média", "mediana", "passa-baixa gaussiano"], loc="lower center", ncol=4,
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

"""Características clássicas por frame (forma glótica, textura, HOG, bordas) e agregação por vídeo.

Famílias (prefixos): `shape_` (máscara da glote), `tex_` (GLCM, LBP, Gabor), `hog_`, `edge_`.
Entrada: cinza 224×224 dos caches (canal 0) + máscara glótica (clássica ou ground truth).
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern
from skimage.measure import label, regionprops

from laringopdi import config as C
from laringopdi import data as D

FAMILIES = ("shape_", "tex_", "hog_", "edge_")
VARIANTS = {   # variante → (experimento da imagem, experimento da máscara, chave da máscara)
    "full_classic": ("E00", "E13", "glottis_mask_lb"),
    "roi_classic": ("E08", "E08", "glottis_mask_lb"),
    "full_gt": ("E00", "E00", "gt_mask_lb"),
    "f18": ("E00", "E13", "glottis_mask_lb"),     # as 18 características da Etapa 1 (C6), com a máscara do C1
}
FAMILIES_18 = ("f18shape_", "f18int_", "f18tex_")


def shape_features(m: np.ndarray) -> dict:
    """Forma do maior componente da máscara: área relativa, perímetro, excentricidade, solidez,
    razão de aspecto, compacidade e simetria esquerda/direita em torno do eixo principal."""
    keys = ["area_rel", "perimeter_rel", "eccentricity", "solidity", "aspect", "compactness", "symmetry", "n_comp", "empty"]
    lab = label(m)
    if lab.max() == 0:
        return {f"shape_{k}": (1.0 if k == "empty" else 0.0) for k in keys}
    props = regionprops(lab)
    r = max(props, key=lambda p: p.area)
    a, per = float(r.area), float(r.perimeter) or 1.0
    comp = (lab == r.label).astype(np.uint8)
    cy, cx = r.centroid
    ang = -np.degrees(r.orientation)               # alinha o eixo maior com a vertical
    M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
    rot = cv2.warpAffine(comp, M, comp.shape[::-1], flags=cv2.INTER_NEAREST)
    M2 = np.float32([[-1, 0, 2 * cx], [0, 1, 0]])    # espelha em torno da vertical que passa pelo centróide
    mir = cv2.warpAffine(rot, M2, rot.shape[::-1], flags=cv2.INTER_NEAREST)
    inter, union = (rot & mir).sum(), (rot | mir).sum()
    return {"shape_area_rel": a / m.size, "shape_perimeter_rel": float(per / np.sqrt(a)), "shape_eccentricity": float(r.eccentricity),
            "shape_solidity": float(r.solidity), "shape_aspect": float(r.axis_major_length / max(r.axis_minor_length, 1e-6)),
            "shape_compactness": float(4 * np.pi * a / per ** 2), "shape_symmetry": float(inter / max(union, 1)),
            "shape_n_comp": float(lab.max()), "shape_empty": 0.0}


def texture_features(g: np.ndarray) -> dict:
    out = {}
    q = (g // 4).astype(np.uint8)                   # 64 níveis
    glcm = graycomatrix(q, distances=[1, 3], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=64,
                        symmetric=True, normed=True)
    for p in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"):
        v = graycoprops(glcm, p)                    # (distâncias, ângulos)
        for i, d in enumerate((1, 3)):
            out[f"tex_glcm_{p}_d{d}"] = float(v[i].mean())
    lbp = local_binary_pattern(g, 8, 1, method="uniform")
    h = np.bincount(lbp[g > 0].astype(int), minlength=10)[:10]
    out.update({f"tex_lbp_{i}": float(x) for i, x in enumerate(h / max(h.sum(), 1))})
    small = cv2.resize(g, (112, 112), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    for f in (0.1, 0.25):                           # frequência em ciclos/pixel (λ = 1/f), σ = 0,56·λ
        lam = 1 / f
        sig = 0.56 * lam
        ks = int(6 * sig) | 1
        for t in range(4):
            th = t * np.pi / 4
            re = cv2.filter2D(small, cv2.CV_32F, cv2.getGaborKernel((ks, ks), sig, th, lam, 1.0, 0))
            im = cv2.filter2D(small, cv2.CV_32F, cv2.getGaborKernel((ks, ks), sig, th, lam, 1.0, np.pi / 2))
            mag = np.hypot(re, im)
            out[f"tex_gabor_f{f}_t{t}_mean"] = float(mag.mean())
            out[f"tex_gabor_f{f}_t{t}_std"] = float(mag.std())
    return out


def edge_hog_features(g: np.ndarray) -> dict:
    from laringopdi.pdi.texture import auto_canny, sobel_mag

    b = cv2.GaussianBlur(g, (5, 5), 0)
    can = auto_canny(b) > 0
    h, w = g.shape
    c = can[h // 4:3 * h // 4, w // 4:3 * w // 4]
    out = {"edge_canny_density": float(can.mean()), "edge_canny_center": float(c.mean()),
           "edge_sobel_mean": float(sobel_mag(b).mean())}
    hv = hog(cv2.resize(g, (64, 64), interpolation=cv2.INTER_AREA), orientations=9, pixels_per_cell=(16, 16),
             cells_per_block=(2, 2), feature_vector=True)
    out.update({f"hog_{i}": float(x) for i, x in enumerate(hv)})
    return out


def frame_features(g: np.ndarray, m: np.ndarray) -> dict:
    return {**shape_features(m), **texture_features(g), **edge_hog_features(g)}


def _letterbox_region(g: np.ndarray) -> np.ndarray:
    """Área da imagem fora do pad: o menor retângulo com pixels > 0 (o pad são linhas ou colunas zeradas)."""
    out = np.zeros(g.shape, bool)
    rows, cols = np.nonzero((g > 0).any(1))[0], np.nonzero((g > 0).any(0))[0]
    if rows.size:
        out[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1] = True
    return out


def _hist_entropy(values: np.ndarray, bins: int, max_value: int) -> float:
    """Entropia de Shannon (bits) do histograma."""
    h = np.bincount(values.astype(int).ravel(), minlength=max_value + 1)[:bins].astype(float)
    p = h[h > 0] / max(h.sum(), 1.0)
    return float(-(p * np.log2(p)).sum())


def features18(g: np.ndarray, m: np.ndarray) -> dict:
    """As 18 características prometidas na Etapa 1: forma (6), intensidade do tecido (6) e textura (6).
    Com máscara vazia, a forma vai a 0 e o contraste glote × tecido também; não há indicador no
    vetor, para manter exatamente 18 entradas."""
    from scipy.stats import kurtosis, skew

    from laringopdi.pdi.texture import auto_canny, sobel_mag

    s = shape_features(m)
    out = {f"f18shape_{k}": s[f"shape_{k}"] for k in ("area_rel", "perimeter_rel", "eccentricity", "solidity", "aspect", "symmetry")}
    region = _letterbox_region(g)
    tissue, glottis = g[region & ~m], g[region & m]
    if tissue.size:
        v = tissue.astype(np.float64)
        flat = v.min() == v.max()
        out |= {"f18int_mean": float(v.mean()), "f18int_std": float(v.std()), "f18int_entropy": _hist_entropy(tissue, 256, 255),
                "f18int_skew": 0.0 if flat else float(skew(v)), "f18int_kurtosis": 0.0 if flat else float(kurtosis(v))}
    else:
        out |= {f"f18int_{k}": 0.0 for k in ("mean", "std", "entropy", "skew", "kurtosis")}
    out["f18int_contrast"] = float((tissue.mean() - glottis.mean()) / 255.0) if tissue.size and glottis.size else 0.0
    b = cv2.GaussianBlur(g, (5, 5), 0)
    out |= {"f18tex_canny_density": float((auto_canny(b) > 0).mean()), "f18tex_sobel_mean": float(sobel_mag(b).mean())}
    glcm = graycomatrix((g // 4).astype(np.uint8), distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=64,
                        symmetric=True, normed=True)
    for p in ("contrast", "homogeneity", "energy"):
        out[f"f18tex_glcm_{p}"] = float(graycoprops(glcm, p).mean())
    out["f18tex_lbp_entropy"] = _hist_entropy(local_binary_pattern(g, 8, 1, method="uniform")[g > 0], 10, 9)
    return out


def _unpack(packed: np.ndarray, size: int) -> np.ndarray:
    return np.unpackbits(packed, axis=-1)[..., :size * size].reshape(-1, size, size).astype(bool)


def build_features(cfg: dict, variant: str, n_jobs: int = -1) -> pd.DataFrame:
    """Características por frame de uma variante (em cache em data/processed/features/)."""
    from joblib import Parallel, delayed

    out = C.path(cfg, "cache").parent / "features" / f"{variant}.csv"
    if out.exists():
        return pd.read_csv(out, dtype={"video_id": str})
    img_exp, mask_exp, mkey = VARIANTS[variant]
    X, paths, meta = D.load_cache(cfg, img_exp)
    masks = _unpack(np.load(D.cache_paths(cfg, mask_exp)["masks"])[mkey], cfg["data"]["image_size"])
    fn = features18 if variant == "f18" else frame_features
    feats = Parallel(n_jobs=n_jobs, batch_size=32)(delayed(fn)(np.ascontiguousarray(X[i, ..., 0]), masks[i])
                                                   for i in range(len(paths)))
    df = pd.DataFrame(feats)
    df.insert(0, "video_id", meta["video_ids"])
    df.insert(0, "path", paths)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def video_features(df: pd.DataFrame) -> pd.DataFrame:
    """Um vetor por vídeo: média e desvio-padrão das características dos frames amostrados."""
    cols = [c for c in df.columns if c.startswith(FAMILIES)]
    agg = df.groupby("video_id")[cols].agg(["mean", "std"])
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    return agg.fillna(0.0)


def video_features18(df: pd.DataFrame) -> pd.DataFrame:
    """Um vetor por vídeo com exatamente 18 entradas: só a média dos frames (C6)."""
    cols = [c for c in df.columns if c.startswith(FAMILIES_18)]
    return df.groupby("video_id")[cols].mean().fillna(0.0)

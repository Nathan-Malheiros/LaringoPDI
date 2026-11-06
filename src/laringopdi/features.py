"""As 18 características prometidas na Etapa 1 (forma, intensidade e textura) e a agregação por vídeo.

Entrada: cinza 224×224 (canal 0 do cache) + máscara da glote obtida pela segmentação clássica.
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import label, regionprops

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


def video_features18(df: pd.DataFrame) -> pd.DataFrame:
    """Um vetor por vídeo com exatamente 18 entradas: só a média dos frames."""
    cols = [c for c in df.columns if c.startswith(FAMILIES_18)]
    return df.groupby("video_id")[cols].mean().fillna(0.0)

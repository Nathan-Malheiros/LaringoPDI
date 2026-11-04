"""Métricas da própria imagem (sem rótulo): entropia, contraste RMS, fração especular,
variância do Laplaciano (nitidez), fração saturada (pixels em 0 ou 255, o "estouro" do realce)
e flags de fallback do pipeline."""
from __future__ import annotations

import cv2
import numpy as np
from skimage.measure import shannon_entropy

from laringopdi.pdi.enhance import specular_mask


def image_quality(img: np.ndarray, ctx: dict | None = None) -> dict:
    """Calcula as métricas sobre o 1º canal (cinza processado) dentro da área válida
    (fora do pad do letterbox)."""
    ctx = ctx or {}
    lb = ctx.get("letterbox")
    if lb:
        img = img[lb["top"]:lb["top"] + lb["h"], lb["left"]:lb["left"] + lb["w"]]
    g = np.ascontiguousarray(img[..., 0] if img.ndim == 3 else img)
    return {
        "entropy": float(shannon_entropy(g)),
        "rms_contrast": float(g.std() / 255.0),
        "spec_frac": float(specular_mask(g).mean()),
        "lap_var": float(cv2.Laplacian(g, cv2.CV_64F).var()),
        "saturated_frac": float(((g == 0) | (g == 255)).mean()),
        "fov_fallback": ctx.get("fov_fallback"),
        "roi_fallback": ctx.get("roi_fallback"),
    }

"""Métricas da própria imagem (sem rótulo): entropia, contraste RMS, fração especular,
variância do Laplaciano (nitidez) e flags de fallback do pipeline."""
from __future__ import annotations

import cv2
import numpy as np
from skimage.measure import shannon_entropy

from laringopdi.pdi.enhance import specular_mask


def image_quality(img: np.ndarray, ctx: dict | None = None) -> dict:
    """Calcula as métricas sobre os 3 primeiros canais convertidos para cinza, apenas na
    área válida (fora do pad do letterbox)."""
    ctx = ctx or {}
    lb = ctx.get("letterbox")
    if lb:
        img = img[lb["top"]:lb["top"] + lb["h"], lb["left"]:lb["left"] + lb["w"]]
    rgb = img[..., :3] if img.shape[2] >= 3 else np.repeat(img, 3, axis=2)
    g = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY)
    valid = g > 0
    gv = g[valid] if valid.any() else g.ravel()
    return {
        "entropy": float(shannon_entropy(gv)),
        "rms_contrast": float(gv.std() / 255.0),
        "spec_frac": float(specular_mask(np.ascontiguousarray(rgb))[valid].mean()) if valid.any() else 0.0,
        "lap_var": float(cv2.Laplacian(g, cv2.CV_64F)[valid].var()) if valid.any() else 0.0,
        "fov_fallback": ctx.get("fov_fallback"),
        "roi_fallback": ctx.get("roi_fallback"),
    }

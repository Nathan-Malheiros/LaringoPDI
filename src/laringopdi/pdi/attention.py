"""Métricas de atenção: quanto da energia do Grad-CAM cai dentro da glote (máscara de ground truth)
e sobre os reflexos especulares (máscaras do PDI)."""
from __future__ import annotations

import numpy as np


def energy_fraction(cam: np.ndarray, mask: np.ndarray | None, valid: np.ndarray | None = None) -> float:
    """Fração da energia do mapa (soma dos valores ≥ 0) dentro da máscara, na área válida."""
    if mask is None:
        return float("nan")
    cam = np.clip(cam, 0, None)
    if valid is not None:
        cam = cam * valid
    tot = cam.sum()
    return float((cam * mask).sum() / tot) if tot > 0 else float("nan")


def attention_metrics(cam: np.ndarray, gt: np.ndarray | None, spec: np.ndarray | None,
                      valid: np.ndarray | None = None) -> dict:
    """Energia na glote, energia nos reflexos e as respectivas áreas (para comparar com o acaso:
    um mapa uniforme teria fração de energia = fração de área)."""
    area = (valid.sum() if valid is not None else cam.size) or 1
    return {"att_glottis": energy_fraction(cam, gt, valid), "area_glottis": float(gt.sum() / area) if gt is not None else np.nan,
            "att_spec": energy_fraction(cam, spec, valid), "area_spec": float(spec.sum() / area) if spec is not None else np.nan}

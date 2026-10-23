"""Técnicas geométricas: campo de visão (fov) e recortes em torno da glote (roi_gt, roi)."""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi

from laringopdi.pdi.registry import crop_ctx, register
from laringopdi.pdi.segmentation import apply_best


def _largest_component(mask: np.ndarray) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(mask, bool)
    return lab == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))


@register("fov", blur=5, max_thresh=40, min_frac=0.2, max_frac=0.97)
def fov(img: np.ndarray, ctx: dict, blur: int = 5, max_thresh: int = 40, min_frac: float = 0.2,
        max_frac: float = 0.97) -> np.ndarray:
    """O que faz: segmenta o campo de visão (limiar = min(Otsu, 40) → maior componente →
    preenchimento de buracos), zera o fundo e recorta a caixa envolvente. Sem borda escura
    (máscara fora de 20–97% da imagem) vira identidade.
    Por que importa: vinheta e moldura preta não são anatomia e podem virar atalho para a rede.
    Parâmetros padrão: blur 5 (mediana), teto do limiar 40.
    Referência: Otsu, 1979; Bernal et al., 2012 (pré-processamento endoscópico)."""
    g = cv2.medianBlur(img, blur)
    t, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = ndi.binary_fill_holes(_largest_component(g > min(t, max_thresh)))
    if not (min_frac <= mask.mean() <= max_frac):
        ctx["fov_fallback"] = True
        return img
    ys, xs = np.nonzero(mask)
    x0, y0, w, h = int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
    ctx["fov_fallback"], ctx["fov_bbox"] = False, (x0, y0, w, h)
    shape = img.shape[:2]
    ctx["fov_mask"] = mask
    crop_ctx(ctx, shape, x0, y0, w, h)
    out = img[y0:y0 + h, x0:x0 + w].copy()
    out[~ctx["fov_mask"]] = 0
    return out


def crop_around(img: np.ndarray, ctx: dict, mask: np.ndarray | None, margin: float, min_side: float,
                fallback_frac: float) -> np.ndarray:
    """Recorta a caixa da máscara expandida por `margin` (lado mínimo `min_side` da imagem);
    máscara vazia → recorte central de `fallback_frac`."""
    h, w = img.shape[:2]
    if mask is None or not mask.any():
        bw, bh = round(w * fallback_frac), round(h * fallback_frac)
        x0, y0 = (w - bw) // 2, (h - bh) // 2
        ctx["roi_fallback"] = True
    else:
        ys, xs = np.nonzero(mask)
        cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
        bw = min(w, max((xs.max() - xs.min() + 1) * (1 + 2 * margin), w * min_side))
        bh = min(h, max((ys.max() - ys.min() + 1) * (1 + 2 * margin), h * min_side))
        x0, y0 = int(np.clip(cx - bw / 2, 0, w - bw)), int(np.clip(cy - bh / 2, 0, h - bh))
        bw, bh = int(bw), int(bh)
        ctx["roi_fallback"] = False
    ctx["roi_bbox"] = (int(x0), int(y0), int(bw), int(bh))
    ctx["roi_mask"] = mask if mask is not None else np.zeros((h, w), bool)
    crop_ctx(ctx, (h, w), x0, y0, bw, bh)
    return img[y0:y0 + bh, x0:x0 + bw].copy()


@register("roi_gt", margin=0.5, min_side=0.3, fallback_frac=0.6)
def roi_gt(img: np.ndarray, ctx: dict, margin: float = 0.5, min_side: float = 0.3,
           fallback_frac: float = 0.6) -> np.ndarray:
    """O que faz: recorta em torno da máscara de glote do especialista (ground truth),
    com margem de 50% e lado mínimo de 30% da imagem.
    Por que importa: é o **limite superior** do recorte — mostra quanto um ROI perfeito
    ajudaria; não é utilizável na prática, pois exige anotação.
    Parâmetros padrão: margin 0,5; min_side 0,3; fallback central 60% (máscara vazia).
    Referência: Gómez et al., 2020 (BAGLS)."""
    return crop_around(img, ctx, ctx.get("gt_mask"), margin, min_side, fallback_frac)


@register("roi", method="best", margin=0.5, min_side=0.3, fallback_frac=0.6)
def roi(img: np.ndarray, ctx: dict, method: str = "best", margin: float = 0.5, min_side: float = 0.3,
        fallback_frac: float = 0.6, seg_params: dict | None = None, seg_pre: str = "raw") -> np.ndarray:
    """O que faz: segmenta a glote com o melhor método clássico da Fase 3 (resolvido em
    `results/tables/segmentation_best.json`) e recorta em torno dela com margem de 50%.
    Sem glote detectada: recorte central de 60%.
    Por que importa: as lesões estruturais (nódulos, pólipos, cistos, edema) ficam na borda livre
    das pregas, ao redor da fenda glótica; o recorte descarta o contexto irrelevante.
    Parâmetros padrão: método = melhor da Fase 3; margin 0,5; min_side 0,3.
    Referência: Otsu, 1979; Gonzalez & Woods (morfologia); Gómez et al., 2020."""
    m = apply_best(img, method, seg_params, seg_pre)
    ctx["glottis_mask"] = m
    return crop_around(img, ctx, m, margin, min_side, fallback_frac)

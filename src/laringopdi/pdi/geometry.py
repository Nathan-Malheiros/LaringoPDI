"""Técnicas geométricas/de segmentação: campo de visão (fov) e região glótica (roi)."""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi

from laringopdi.pdi.registry import register


def _largest_component(mask: np.ndarray) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(mask, bool)
    return lab == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))


@register("fov", blur=5, max_thresh=60, min_frac=0.2)
def fov(img: np.ndarray, ctx: dict, blur: int = 5, max_thresh: int = 60, min_frac: float = 0.2) -> np.ndarray:
    """O que faz: segmenta o campo de visão circular do endoscópio (Otsu no cinza → maior
    componente conexo → preenchimento de buracos), zera o fundo e recorta a caixa envolvente.
    Por que importa: a moldura preta e o texto/OSD fora do círculo não são anatomia e
    podem virar atalho para a rede; também padroniza a escala entre aparelhos.
    Parâmetros padrão: blur=5 (mediana), limiar = min(Otsu, 60) para que o Otsu não
    caia dentro do tecido, fallback se a máscara cobre < 20% da imagem.
    Referência: Otsu, 1979; Bernal et al., 2012 (pré-processamento de imagens endoscópicas)."""
    g = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), blur)
    t, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = ndi.binary_fill_holes(_largest_component(g > min(t, max_thresh)))
    if mask.mean() < min_frac:
        ctx["fov_fallback"] = True
        ctx["fov_bbox"] = (0, 0, img.shape[1], img.shape[0])
        ctx["fov_mask"] = np.ones(img.shape[:2], bool)
        return img
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    ctx["fov_fallback"] = False
    ctx["fov_bbox"] = (int(x0), int(y0), int(x1 - x0), int(y1 - y0))
    ctx["fov_mask"] = mask[y0:y1, x0:x1]
    out = img[y0:y1, x0:x1].copy()
    out[~ctx["fov_mask"]] = 0
    return out


def roi_candidate(img: np.ndarray, ctx: dict, method: str = "otsu", block: int = 51, C: int = 5,
                  open_k: int = 5, close_k: int = 15, min_frac: float = 0.005,
                  max_frac: float = 0.4) -> np.ndarray | None:
    """Máscara do componente escuro (fenda glótica/via aérea) mais central, ou None."""
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (5, 5), 0)
    valid = ctx.get("fov_mask")
    valid = valid if valid is not None and valid.shape == g.shape else np.ones(g.shape, bool)
    if method == "otsu":
        t, _ = cv2.threshold(g[valid].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dark = g <= t
    elif method == "adaptive":
        dark = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                     block | 1, C) > 0
    else:
        raise ValueError(f"método de ROI desconhecido: {method}")
    dark &= cv2.erode(valid.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    m = cv2.morphologyEx(cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_OPEN, ko), cv2.MORPH_CLOSE, kc)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
    area = valid.sum()
    h, w = g.shape
    best, best_d = None, np.inf
    for i in range(1, n):
        frac = stats[i, cv2.CC_STAT_AREA] / area
        if not (min_frac <= frac <= max_frac):
            continue
        d = np.hypot((cent[i][0] - w / 2) / w, (cent[i][1] - h / 2) / h)
        if d < best_d:
            best, best_d = i, d
    return None if best is None or best_d > 0.35 else ndi.binary_fill_holes(lab == best)


@register("roi", method="otsu", block=51, C=5, open_k=5, close_k=15, margin=0.6,
          min_side=0.35, fallback_frac=0.6)
def roi(img: np.ndarray, ctx: dict, method: str = "otsu", block: int = 51, C: int = 5,
        open_k: int = 5, close_k: int = 15, margin: float = 0.6, min_side: float = 0.35,
        fallback_frac: float = 0.6) -> np.ndarray:
    """O que faz: localiza a fenda glótica (região escura central) por limiarização — Otsu
    global ou adaptativa gaussiana — com abertura/fechamento morfológicos, escolhe o
    componente mais central e recorta uma caixa expandida (margem de 60%, lado mínimo de
    35% da imagem) que inclui as pregas vocais. Sem candidato plausível: recorte central de 60%.
    Por que importa: as lesões (nódulos, pólipos, cistos, leucoplasia) ficam na borda livre
    das pregas, ao redor da fenda; recortar reduz o contexto irrelevante (epiglote, aritenoides, tubo).
    Parâmetros padrão: block=51, C=5 (adaptativa); kernels 5 (abertura) e 15 (fechamento);
    componente com 0,5–40% do FOV e centróide a ≤ 0,35 do centro.
    Referência: Otsu, 1979; Gonzalez & Woods, Digital Image Processing (morfologia);
    Laves et al., 2019 (segmentação de imagens laríngeas)."""
    m = roi_candidate(img, ctx, method, block, C, open_k, close_k)
    h, w = img.shape[:2]
    if m is None:
        bw, bh = round(w * fallback_frac), round(h * fallback_frac)
        x0, y0 = (w - bw) // 2, (h - bh) // 2
        ctx["roi_fallback"] = True
        ctx["roi_mask"] = np.zeros((h, w), bool)
    else:
        ys, xs = np.nonzero(m)
        cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
        bw = max((xs.max() - xs.min()) * (1 + 2 * margin), w * min_side)
        bh = max((ys.max() - ys.min()) * (1 + 2 * margin), h * min_side)
        x0, y0 = int(max(0, cx - bw / 2)), int(max(0, cy - bh / 2))
        bw, bh = int(min(w - x0, bw)), int(min(h - y0, bh))
        ctx["roi_fallback"] = False
        ctx["roi_mask"] = m
    ctx["roi_bbox"] = (int(x0), int(y0), int(bw), int(bh))
    for k in ("fov_mask", "spec_mask"):          # mantém máscaras anteriores alinhadas ao recorte
        if k in ctx and ctx[k].shape == (h, w):
            ctx[k] = ctx[k][y0:y0 + bh, x0:x0 + bw]
    ctx["roi_mask"] = ctx["roi_mask"][y0:y0 + bh, x0:x0 + bw]
    return img[y0:y0 + bh, x0:x0 + bw].copy()

"""Segmentação clássica da glote (área escura entre as pregas vocais) e métricas contra o
ground truth do BAGLS. Todos os métodos recebem cinza uint8 e devolvem máscara booleana."""
from __future__ import annotations

import json

import cv2
import numpy as np
from scipy import ndimage as ndi

from laringopdi import config as C


def _blur(g: np.ndarray, k: int = 5) -> np.ndarray:
    return cv2.GaussianBlur(g, (k | 1, k | 1), 0)


def _otsu_dark(g: np.ndarray) -> np.ndarray:
    _, b = cv2.threshold(g, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return b.astype(bool)


def pick_central(mask: np.ndarray, min_frac: float = 0.0005, max_frac: float = 0.3,
                 max_dist: float = 0.4) -> np.ndarray:
    """Componente conexo de área plausível mais próximo do centro (ou máscara vazia)."""
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    h, w = mask.shape
    best, best_d = 0, np.inf
    for i in range(1, n):
        if not (min_frac <= stats[i, cv2.CC_STAT_AREA] / mask.size <= max_frac):
            continue
        d = np.hypot((cent[i][0] - w / 2) / w, (cent[i][1] - h / 2) / h)
        if d < best_d:
            best, best_d = i, d
    if best == 0 or best_d > max_dist:
        return np.zeros_like(mask, bool)
    return ndi.binary_fill_holes(lab == best)


def seg_otsu(g: np.ndarray, blur: int = 5) -> np.ndarray:
    """O que faz: limiar global de Otsu sobre a imagem suavizada; pixels escuros = glote.
    Por que importa: a glote aberta é a região mais escura do frame (via aérea sem retorno de
    luz); é o método mais simples, sem nenhuma restrição espacial — linha de base da segmentação.
    Parâmetros padrão: blur gaussiano 5.
    Referência: Otsu, 1979."""
    return _otsu_dark(_blur(g, blur))


def seg_adaptive(g: np.ndarray, block: int = 51, C: int = 5, blur: int = 5) -> np.ndarray:
    """O que faz: limiar adaptativo gaussiano (pixel escuro em relação à vizinhança block×block).
    Por que importa: tolera o decaimento da iluminação do endoscópio do centro para a borda,
    que faz o Otsu global incluir cantos escuros.
    Parâmetros padrão: bloco 51, C=5, blur 5.
    Referência: Gonzalez & Woods, Digital Image Processing, seção 10.3."""
    b = cv2.adaptiveThreshold(_blur(g, blur), 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                              block | 1, C)
    return b.astype(bool)


def seg_otsu_morph(g: np.ndarray, blur: int = 5, open_k: int = 3, close_k: int = 7) -> np.ndarray:
    """O que faz: Otsu + abertura (remove ruído) + fechamento (une a fenda) + componente
    conexo central de área plausível + preenchimento de buracos.
    Por que importa: incorpora o conhecimento anatômico de que a glote é uma região única,
    alongada e central no frame da videoendoscopia.
    Parâmetros padrão: blur 5, abertura 3, fechamento 7 (elípticos).
    Referência: Otsu, 1979; Serra, 1982 (morfologia matemática)."""
    m = _otsu_dark(_blur(g, blur)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k)))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k)))
    return pick_central(m.astype(bool))


def seg_region_grow(g: np.ndarray, tol: int = 10, blur: int = 5, center_frac: float = 0.5) -> np.ndarray:
    """O que faz: crescimento de região (flood fill de faixa fixa ±tol) a partir do pixel mais
    escuro da janela central.
    Por que importa: a glote é um "poço" escuro e conexo; crescer a partir da semente evita
    capturar regiões escuras desconexas (sombras, bordas).
    Parâmetros padrão: tol=10 níveis de cinza, blur 5, janela central de 50%.
    Referência: Adams & Bischof, 1994 (seeded region growing)."""
    b = _blur(g, blur)
    h, w = b.shape
    y0, x0 = int(h * (1 - center_frac) / 2), int(w * (1 - center_frac) / 2)
    win = b[y0:h - y0, x0:w - x0]
    sy, sx = np.unravel_index(int(np.argmin(win)), win.shape)
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(b.copy(), mask, (int(sx + x0), int(sy + y0)), 255, tol, tol,
                  8 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (255 << 8))
    m = mask[1:-1, 1:-1] > 0
    return ndi.binary_fill_holes(m) if m.mean() <= 0.3 else np.zeros_like(m)


def seg_watershed(g: np.ndarray, blur: int = 5, fg_frac: float = 0.5) -> np.ndarray:
    """O que faz: watershed com marcadores — Otsu escuro → transformada de distância → núcleos
    (dist > fg_frac·máx) como marcadores de objeto, dilatação como fundo certo — e escolhe a
    bacia central.
    Por que importa: separa a glote de regiões escuras vizinhas que tocam nela (ventrículos,
    sombras das aritenoides), algo que o limiar sozinho não consegue.
    Parâmetros padrão: blur 5, fg_frac 0,5.
    Referência: Vincent & Soille, 1991; Meyer, 1994 (watershed com marcadores)."""
    b = _blur(g, blur)
    dark = cv2.morphologyEx(_otsu_dark(b).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    if not dark.any():
        return np.zeros(g.shape, bool)
    bg = cv2.dilate(dark, np.ones((3, 3), np.uint8), iterations=3)
    dist = cv2.distanceTransform(dark, cv2.DIST_L2, 5)
    # núcleo por componente (dist > fg_frac × máximo DO PRÓPRIO componente): cada região escura,
    # inclusive a glote, ganha um marcador — com o máximo global só a maior região era semeada.
    n, comp = cv2.connectedComponents(dark)
    cmax = np.asarray(ndi.maximum(dist, comp, index=np.arange(n)))
    fg = ((comp > 0) & (dist > fg_frac * cmax[comp])).astype(np.uint8)
    _, markers = cv2.connectedComponents(fg)
    markers = markers + 1
    markers[(bg > 0) & (fg == 0)] = 0
    markers = cv2.watershed(cv2.cvtColor(b, cv2.COLOR_GRAY2BGR), markers.astype(np.int32))
    return pick_central(markers > 1)


def adaptive_morph_steps(g: np.ndarray, block: int = 51, C: int = 5, blur: int = 5, kernel: int = 5) -> dict:
    """Máscaras de cada etapa: limiar adaptativo → abertura → fechamento → componente central."""
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    limiar = seg_adaptive(g, block, C, blur).astype(np.uint8)
    abertura = cv2.morphologyEx(limiar, cv2.MORPH_OPEN, se)
    fechamento = cv2.morphologyEx(abertura, cv2.MORPH_CLOSE, se)
    return {"limiar": limiar.astype(bool), "abertura": abertura.astype(bool), "fechamento": fechamento.astype(bool),
            "componente": pick_central(fechamento.astype(bool))}


def seg_adaptive_morph(g: np.ndarray, block: int = 51, C: int = 5, blur: int = 5, kernel: int = 5) -> np.ndarray:
    """O que faz: limiar adaptativo + abertura (tira pontinhos) + fechamento (fecha buracos) + componente
    conexo de área plausível mais próximo do centro.
    Por que importa: é o método apresentado na Etapa 1. O limiar por região tolera a iluminação
    desigual, e a morfologia transforma o resultado numa região única, que é o que a glote é.
    Parâmetros padrão: janela 51, C=5, blur 5, elipse 5×5.
    Referência: Gonzalez & Woods, Digital Image Processing, seções 9 e 10.3."""
    return adaptive_morph_steps(g, block, C, blur, kernel)["componente"]


SEGMENTERS = {"otsu": seg_otsu, "adaptive": seg_adaptive, "otsu_morph": seg_otsu_morph,
              "region_grow": seg_region_grow, "watershed": seg_watershed, "adaptive_morph": seg_adaptive_morph}
PARAM_GRID = {
    "adaptive_morph": [{"block": b, "C": c} for b in (31, 51, 101) for c in (5, 10)],
    "otsu": [{"blur": b} for b in (3, 5, 9)],
    "adaptive": [{"block": b, "C": c} for b in (31, 51, 101) for c in (5, 10)],
    "otsu_morph": [{"open_k": o, "close_k": c} for o in (3, 5) for c in (7, 15)],
    "region_grow": [{"tol": t} for t in (5, 10, 20)],
    "watershed": [{"fg_frac": f} for f in (0.3, 0.5, 0.7)],
}
DEFAULT_BEST = {"method": "otsu_morph", "preprocess": "raw", "params": {}}


def segment(g: np.ndarray, method: str, **params) -> np.ndarray:
    return SEGMENTERS[method](g, **params)


def best_config() -> dict:
    """Método/pré-processamento/parâmetros fixados pela Fase 3 (padrão até lá: Otsu+morfologia)."""
    p = C.ROOT / "results" / "tables" / "segmentation_best.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else DEFAULT_BEST


def resolve_best(p: dict) -> dict:
    b = best_config()
    return {**p, "method": b["method"], "seg_params": b["params"], "seg_pre": b.get("preprocess", "raw")}


def apply_best(g: np.ndarray, method: str, seg_params: dict | None = None, seg_pre: str = "raw") -> np.ndarray:
    from laringopdi.pdi.registry import DEFAULTS, TECHNIQUES
    if seg_pre != "raw":
        g = TECHNIQUES[seg_pre](g, {}, **DEFAULTS[seg_pre])
    return segment(g, method, **(seg_params or {}))


# ----------------------------------------------------------------------------- métricas
def dice(a: np.ndarray, b: np.ndarray) -> float:
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else float(2 * (a & b).sum() / s)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = (a | b).sum()
    return 1.0 if u == 0 else float((a & b).sum() / u)


def hd95(a: np.ndarray, b: np.ndarray) -> float:
    """Distância de Hausdorff (percentil 95) entre os contornos, em pixels."""
    if not a.any() or not b.any():
        return float("nan")
    sa, sb = a ^ ndi.binary_erosion(a), b ^ ndi.binary_erosion(b)
    d = np.concatenate([ndi.distance_transform_edt(~sb)[sa], ndi.distance_transform_edt(~sa)[sb]])
    return float(np.percentile(d, 95))


def seg_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    return {"dice": dice(pred, gt), "iou": iou(pred, gt), "hd95": hd95(pred, gt),
            "fail": float(gt.any() and not pred.any())}

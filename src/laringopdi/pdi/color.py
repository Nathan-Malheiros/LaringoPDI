"""Espaços de cor e realce vascular (pseudo-NBI)."""
from __future__ import annotations

import cv2
import numpy as np

from laringopdi.pdi.registry import register


@register("lab")
def lab(img: np.ndarray, ctx: dict) -> np.ndarray:
    """O que faz: converte a entrada para CIELAB (L, a, b em uint8 do OpenCV).
    Por que importa: separa luminância de cromaticidade; a vermelhidão (a*) e o
    amarelado/branco (b*, L) são sinais de inflamação e leucoplasia. Para a ResNet pré-treinada
    em RGB a troca pode prejudicar (H4).
    Parâmetros padrão: nenhum.
    Referência: CIE 1976; Gonzalez & Woods, cap. 6."""
    return cv2.cvtColor(img, cv2.COLOR_RGB2LAB)


@register("hsv")
def hsv(img: np.ndarray, ctx: dict) -> np.ndarray:
    """O que faz: converte a entrada para HSV (H em 0–255, escala completa).
    Por que importa: matiz e saturação isolam a cor da mucosa da intensidade da luz;
    reflexos têm S baixo e V alto. Teste de H4 junto com `lab`.
    Parâmetros padrão: nenhum.
    Referência: Smith, 1978; Gonzalez & Woods, cap. 6."""
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV_FULL)


def vessel_map(img: np.ndarray, clip: float = 2.0, tiles: int = 8, kernel: int = 15) -> tuple[np.ndarray, np.ndarray]:
    """(G após CLAHE, black-hat normalizado 0–255) — vasos aparecem claros no black-hat."""
    op = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles))
    g = op.apply(img[..., 1])
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, k)
    return g, cv2.normalize(bh, None, 0, 255, cv2.NORM_MINMAX)


@register("vessel", clip=2.0, tiles=8, kernel=15, alpha=1.0)
def vessel(img: np.ndarray, ctx: dict, clip: float = 2.0, tiles: int = 8, kernel: int = 15,
           alpha: float = 1.0) -> np.ndarray:
    """O que faz: realce vascular "pseudo-NBI" — canal G + CLAHE; black-hat morfológico
    (fechamento − imagem) extrai estruturas escuras finas (vasos); subtrai-se α·black-hat de
    G e de B e monta-se a imagem como (G', B', B'), o mapeamento de cores do NBI.
    Por que importa: hemoglobina absorve verde/azul, então os vasos têm mais contraste em G;
    padrões vasculares atípicos acompanham leucoplasia e câncer glótico.
    Parâmetros padrão: CLAHE clip 2,0 / 8×8; elemento estruturante elíptico 15 px; α=1.
    Referência: Gono et al., 2004 (NBI); Ni et al., 2011 (padrões vasculares laríngeos)."""
    op = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles))
    g, bh = vessel_map(img, clip, tiles, kernel)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    b = op.apply(img[..., 2])
    bh_b = cv2.morphologyEx(b, cv2.MORPH_BLACKHAT, k)
    g2 = np.clip(g.astype(np.int16) - alpha * cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, k), 0, 255).astype(np.uint8)
    b2 = np.clip(b.astype(np.int16) - alpha * bh_b, 0, 255).astype(np.uint8)
    ctx["vessel_map"] = bh
    return np.dstack([g2, b2, b2])

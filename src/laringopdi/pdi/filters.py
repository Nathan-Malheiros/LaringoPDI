"""Filtros espaciais de redução de ruído."""
from __future__ import annotations

import cv2
import numpy as np

from laringopdi.pdi.registry import register


@register("bilat", d=9, sigma_color=75, sigma_space=75)
def bilat(img: np.ndarray, ctx: dict, d: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
    """O que faz: filtro bilateral — média ponderada por proximidade espacial e de intensidade.
    Por que importa: suaviza ruído de sensor e compressão JPEG da mucosa preservando as
    bordas das pregas e das lesões (que o filtro gaussiano borraria).
    Parâmetros padrão: d=9, σintensidade=75, σespaço=75.
    Referência: Tomasi & Manduchi, 1998."""
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


@register("mean", ksize=3)
def mean(img: np.ndarray, ctx: dict, ksize: int = 3) -> np.ndarray:
    """O que faz: filtro da média — cada pixel vira a média da janela ksize×ksize.
    Por que importa: é o suavizador espacial mais simples. Reduz o ruído gaussiano do sensor, mas
    espalha os valores extremos em vez de removê-los e borra a borda da glote cada vez mais à
    medida que a janela cresce; serve de referência para a mediana e o bilateral.
    Parâmetros padrão: ksize=3 (o estudo de ruído usa 3, 7 e 15).
    Referência: Gonzalez & Woods, Digital Image Processing, seção 3.5."""
    return cv2.blur(img, (ksize, ksize))


@register("median", ksize=5)
def median(img: np.ndarray, ctx: dict, ksize: int = 5) -> np.ndarray:
    """O que faz: filtro da mediana em janela ksize×ksize por canal.
    Por que importa: remove ruído impulsivo e pequenos pontos brilhantes, mas pode apagar
    detalhes finos (vasos, pequenos nódulos) — o experimento mede esse custo.
    Parâmetros padrão: ksize=5.
    Referência: Gonzalez & Woods, Digital Image Processing, seção 3.5."""
    return cv2.medianBlur(img, ksize)

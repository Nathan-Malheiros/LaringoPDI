"""Bordas, textura (LBP) e fusões multicanal M1/M2."""
from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

from laringopdi.pdi.color import vessel_map
from laringopdi.pdi.registry import register


def sobel_mag(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.normalize(cv2.magnitude(gx, gy), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def auto_canny(gray: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    v = np.median(gray[gray > 0]) if (gray > 0).any() else np.median(gray)
    return cv2.Canny(gray, int(max(0, (1 - sigma) * v)), int(min(255, (1 + sigma) * v)))


def lbp_map(gray: np.ndarray, P: int = 8, R: float = 1) -> np.ndarray:
    m = local_binary_pattern(gray, P, R, method="uniform")      # valores 0..P+1
    return (m * (255 / (P + 1))).astype(np.uint8)


def _gray(img: np.ndarray, blur: int = 5) -> np.ndarray:
    return cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (blur, blur), 0)


@register("edges", blur=5, sigma=0.33)
def edges(img: np.ndarray, ctx: dict, blur: int = 5, sigma: float = 0.33) -> np.ndarray:
    """O que faz: empilha [cinza, magnitude de Sobel, Canny automático] em 3 canais
    (limiares do Canny = (1∓σ)·mediana).
    Por que importa: a forma da borda livre das pregas vocais (irregular no câncer,
    abaulada no pólipo/nódulo) é um dos critérios do laringologista; testa se explicitar
    bordas ajuda a rede.
    Parâmetros padrão: blur gaussiano 5, σ=0,33.
    Referência: Sobel, 1968; Canny, 1986; regra da mediana (Rosebrock, 2015)."""
    g = _gray(img, blur)
    s, c = sobel_mag(g), auto_canny(g, sigma)
    ctx["edges_sobel"], ctx["edges_canny"] = s, c
    return np.dstack([g, s, c])


@register("lbp", P=8, R=1)
def lbp(img: np.ndarray, ctx: dict, P: int = 8, R: float = 1) -> np.ndarray:
    """O que faz: mapa LBP uniforme (P=8 vizinhos, raio 1) replicado em 3 canais.
    Por que importa: codifica micro-textura da mucosa (lisa × rugosa/queratinizada),
    invariante a mudanças monotônicas de brilho da iluminação.
    Parâmetros padrão: P=8, R=1, "uniform".
    Referência: Ojala, Pietikäinen & Mäenpää, 2002."""
    m = lbp_map(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), P, R)
    ctx["lbp"] = m
    return np.dstack([m, m, m])


@register("m1", clip=2.0, tiles=8, P=8, R=1)
def m1(img: np.ndarray, ctx: dict, clip: float = 2.0, tiles: int = 8, P: int = 8, R: float = 1) -> np.ndarray:
    """O que faz: fusão multicanal M1 = [L com CLAHE, magnitude de Sobel, LBP uniforme].
    Por que importa: entrega à rede, num formato de 3 canais compatível com a ResNet,
    luminância realçada + forma + textura — descritores clássicos do PDI.
    Parâmetros padrão: CLAHE 2,0/8×8; Sobel 3×3; LBP P=8, R=1.
    Referência: combinação das referências de `clahe`, `edges` e `lbp`."""
    L = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(
        cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[..., 0])
    return np.dstack([L, sobel_mag(cv2.GaussianBlur(L, (5, 5), 0)), lbp_map(L, P, R)])


@register("m2", clip=2.0, tiles=8, kernel=15)
def m2(img: np.ndarray, ctx: dict, clip: float = 2.0, tiles: int = 8, kernel: int = 15) -> np.ndarray:
    """O que faz: fusão multicanal M2 = RGB + magnitude de Sobel + mapa vascular (black-hat), 5 canais.
    Por que importa: mantém a cor original e acrescenta bordas e vasos como canais extras;
    exige adaptar a 1ª convolução (pesos extras inicializados pela média dos canais RGB).
    Parâmetros padrão: CLAHE 2,0/8×8; black-hat 15 px.
    Referência: combinação das referências de `edges` e `vessel`."""
    _, bh = vessel_map(img, clip, tiles, kernel)
    return np.dstack([img, sobel_mag(_gray(img)), bh])

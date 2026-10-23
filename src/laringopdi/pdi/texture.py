"""Bordas, textura (LBP) e fusões multicanal M1/M2 (entrada em cinza)."""
from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

from laringopdi.pdi.registry import register
from laringopdi.pdi.segmentation import apply_best, resolve_best


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


@register("edges", blur=5, sigma=0.33)
def edges(img: np.ndarray, ctx: dict, blur: int = 5, sigma: float = 0.33) -> np.ndarray:
    """O que faz: empilha [cinza, magnitude de Sobel, Canny automático] em 3 canais
    (limiares do Canny = (1∓σ)·mediana, sobre a imagem suavizada).
    Por que importa: o contorno da glote e a borda livre das pregas (irregular no carcinoma,
    abaulada no pólipo e no nódulo) são critérios do laringologista; testa se explicitar
    bordas ajuda a rede.
    Parâmetros padrão: blur gaussiano 5, σ=0,33.
    Referência: Sobel, 1968; Canny, 1986; regra da mediana (Rosebrock, 2015)."""
    g = cv2.GaussianBlur(img, (blur, blur), 0)
    return np.dstack([img, sobel_mag(g), auto_canny(g, sigma)])


@register("lbp", P=8, R=1)
def lbp(img: np.ndarray, ctx: dict, P: int = 8, R: float = 1) -> np.ndarray:
    """O que faz: mapa LBP uniforme (P=8 vizinhos, raio 1).
    Por que importa: codifica a micro-textura da mucosa (lisa × rugosa/queratinizada) e é
    invariante a mudanças monotônicas de brilho da iluminação.
    Parâmetros padrão: P=8, R=1, "uniform".
    Referência: Ojala, Pietikäinen & Mäenpää, 2002."""
    return lbp_map(img, P, R)


@register("m1", clip=2.0, tiles=8, P=8, R=1)
def m1(img: np.ndarray, ctx: dict, clip: float = 2.0, tiles: int = 8, P: int = 8, R: float = 1) -> np.ndarray:
    """O que faz: fusão multicanal M1 = [cinza com CLAHE, magnitude de Sobel, LBP uniforme].
    Por que importa: entrega à rede, em 3 canais compatíveis com a ResNet, luminância
    realçada + forma + textura — descritores clássicos do PDI.
    Parâmetros padrão: CLAHE 2,0/8×8; Sobel 3×3; LBP P=8, R=1.
    Referência: combinação das referências de `clahe`, `edges` e `lbp`."""
    L = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(img)
    return np.dstack([L, sobel_mag(cv2.GaussianBlur(L, (5, 5), 0)), lbp_map(L, P, R)])


@register("m2", method="best")
def m2(img: np.ndarray, ctx: dict, method: str = "best", seg_params: dict | None = None,
       seg_pre: str = "raw") -> np.ndarray:
    """O que faz: fusão multicanal M2 = [cinza, máscara glótica da segmentação clássica,
    magnitude de Sobel].
    Por que importa: dá à rede, explicitamente, a forma da glote (que reflete o fechamento
    glótico e a presença de massas na borda das pregas) junto da imagem e das bordas.
    Parâmetros padrão: segmentação = melhor método da Fase 3; Sobel 3×3 após blur 5.
    Referência: combinação das referências de `roi` e `edges`."""
    if method == "best":
        b = resolve_best({})
        method, seg_params, seg_pre = b["method"], b["seg_params"], b["seg_pre"]
    m = apply_best(img, method, seg_params, seg_pre)
    ctx["glottis_mask"] = m
    return np.dstack([img, m.astype(np.uint8) * 255, sobel_mag(cv2.GaussianBlur(img, (5, 5), 0))])

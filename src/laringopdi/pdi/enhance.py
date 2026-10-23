"""Realce de intensidade: reflexo especular, HE, CLAHE e filtragem homomórfica."""
from __future__ import annotations

import cv2
import numpy as np

from laringopdi.pdi.registry import register


def _on_L(img: np.ndarray, fn) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[..., 0] = fn(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def specular_mask(img: np.ndarray, v_min: int = 220, s_max: int = 40) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    return (hsv[..., 2] >= v_min) & (hsv[..., 1] <= s_max)


@register("spec", v_min=220, s_max=40, dilate=3, radius=5)
def spec(img: np.ndarray, ctx: dict, v_min: int = 220, s_max: int = 40, dilate: int = 3,
         radius: int = 5) -> np.ndarray:
    """O que faz: detecta reflexos especulares (V alto e S baixo no HSV), dilata a máscara
    e preenche por inpainting de Telea (marcha rápida).
    Por que importa: a mucosa úmida da laringe reflete a luz branca; os brilhos saturados
    escondem a textura e lembram leucoplasia (placa branca), podendo confundir a rede.
    Parâmetros padrão: V ≥ 220, S ≤ 40 (escala 0–255), dilatação 3 px, raio de inpaint 5.
    Referência: Arnold et al., 2010 (reflexos especulares em endoscopia); Telea, 2004."""
    m = specular_mask(img, v_min, s_max).astype(np.uint8)
    if dilate:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1,) * 2))
    ctx["spec_mask"] = m.astype(bool)
    ctx["spec_frac"] = float(m.mean())
    return cv2.inpaint(img, m, radius, cv2.INPAINT_TELEA) if m.any() else img


@register("he")
def he(img: np.ndarray, ctx: dict) -> np.ndarray:
    """O que faz: equalização global do histograma no canal L do LAB (cor preservada).
    Por que importa: corrige subexposição global, mas amplifica ruído e satura regiões
    já claras (reflexos); serve de contraste para H2 (CLAHE/homomórfica > HE).
    Parâmetros padrão: nenhum.
    Referência: Gonzalez & Woods, Digital Image Processing, cap. 3."""
    return _on_L(img, cv2.equalizeHist)


@register("clahe", clip=2.0, tiles=8)
def clahe(img: np.ndarray, ctx: dict, clip: float = 2.0, tiles: int = 8) -> np.ndarray:
    """O que faz: equalização adaptativa por blocos com limite de contraste (CLAHE) no canal L.
    Por que importa: a iluminação do laringoscópio é central e decai para as bordas;
    o CLAHE realça contraste local (borda livre das pregas, vasos) sem estourar o centro.
    Parâmetros padrão: clip 2.0, grade 8×8.
    Referência: Zuiderveld, 1994; Pizer et al., 1987."""
    op = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles))
    return _on_L(img, op.apply)


def homo_transfer(shape: tuple[int, int], gl: float = 0.5, gh: float = 1.75, d0: float = 30,
                  c: float = 1.0) -> np.ndarray:
    """H(u,v) gaussiano de alta ênfase, centrado (fftshift)."""
    h, w = shape
    u = np.arange(h) - h / 2
    v = np.arange(w) - w / 2
    D2 = u[:, None] ** 2 + v[None, :] ** 2
    return (gh - gl) * (1 - np.exp(-c * D2 / d0 ** 2)) + gl


def homomorphic(channel: np.ndarray, gl: float = 0.5, gh: float = 1.75, d0: float = 30,
                c: float = 1.0) -> np.ndarray:
    z = np.log1p(channel.astype(np.float64))
    Z = np.fft.fftshift(np.fft.fft2(z))
    s = np.real(np.fft.ifft2(np.fft.ifftshift(Z * homo_transfer(z.shape, gl, gh, d0, c))))
    out = np.expm1(s)
    lo, hi = np.percentile(out, [0.5, 99.5])
    return np.clip((out - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8)


@register("homo", gl=0.5, gh=1.75, d0=30, c=1.0)
def homo(img: np.ndarray, ctx: dict, gl: float = 0.5, gh: float = 1.75, d0: float = 30,
         c: float = 1.0) -> np.ndarray:
    """O que faz: filtragem homomórfica no canal L — log → FFT → filtro passa-altas gaussiano
    de alta ênfase H(u,v) = (γH−γL)(1−e^{−cD²/D0²})+γL → IFFT → exp → reescala por percentis.
    Por que importa: separa iluminação (baixa frequência, γL atenua) de refletância
    (alta frequência, γH realça); compensa o vinhetamento da fonte de luz pontual.
    Parâmetros padrão: γL=0,5; γH=1,75; D0=30 (em imagens com lado maior 512); c=1.
    Referência: Gonzalez & Woods, Digital Image Processing, seção 4.9."""
    fov_mask = ctx.get("fov_mask")
    out = _on_L(img, lambda L: homomorphic(L, gl, gh, d0, c))
    if fov_mask is not None and fov_mask.shape == img.shape[:2]:
        out[~fov_mask] = 0
    return out

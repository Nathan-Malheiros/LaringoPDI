"""Realce e frequência em cinza: reflexo especular, alargamento, HE, CLAHE, filtragem homomórfica
e passa-baixa gaussiano."""
from __future__ import annotations

import cv2
import numpy as np

from laringopdi.pdi.registry import register


def specular_mask(g: np.ndarray, pct: float = 90, v_min: int = 200, max_frac: float = 0.05) -> np.ndarray:
    """Pixels muito claros: limiar = max(Otsu sobre a cauda acima do percentil `pct`, `v_min`)."""
    if g.ndim == 3:
        g = cv2.cvtColor(g[..., :3], cv2.COLOR_RGB2GRAY)
    tail = g[g >= np.percentile(g, pct)]
    if tail.size < 16 or tail.min() == tail.max():
        t = max(int(tail.max()) if tail.size else 255, v_min)
    else:
        t, _ = cv2.threshold(tail.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        t = max(int(t), v_min)
    m = g > t if (g >= t).mean() > max_frac else g >= t
    return m


@register("spec", pct=90, v_min=200, max_frac=0.05, dilate=2, radius=5)
def spec(img: np.ndarray, ctx: dict, pct: float = 90, v_min: int = 200, max_frac: float = 0.05,
         dilate: int = 2, radius: int = 5) -> np.ndarray:
    """O que faz: detecta reflexos especulares por limiar de intensidade (Otsu sobre a cauda do
    histograma acima do percentil 90, com piso 200), dilata a máscara e preenche por inpainting
    de Telea.
    Por que importa: a mucosa úmida das pregas reflete a luz do endoscópio; os brilhos saturados
    escondem a borda livre e podem ser confundidos com lesões esbranquiçadas.
    Parâmetros padrão: percentil 90, piso 200, dilatação 2 px, raio de inpaint 5.
    Referência: Arnold et al., 2010 (reflexos em endoscopia); Telea, 2004."""
    m = specular_mask(img, pct, v_min, max_frac).astype(np.uint8)
    if dilate:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1,) * 2))
    ctx["spec_mask"] = m.astype(bool)
    ctx["spec_frac"] = float(m.mean())
    return cv2.inpaint(img, m, radius, cv2.INPAINT_TELEA) if m.any() else img


@register("he")
def he(img: np.ndarray, ctx: dict) -> np.ndarray:
    """O que faz: equalização global do histograma.
    Por que importa: corrige subexposição global, mas amplifica ruído e satura os reflexos;
    serve de contraste para H2 (CLAHE/homomórfica > HE).
    Parâmetros padrão: nenhum.
    Referência: Gonzalez & Woods, Digital Image Processing, cap. 3."""
    return cv2.equalizeHist(img)


@register("clahe", clip=2.0, tiles=8)
def clahe(img: np.ndarray, ctx: dict, clip: float = 2.0, tiles: int = 8) -> np.ndarray:
    """O que faz: equalização adaptativa por blocos com limite de contraste (CLAHE).
    Por que importa: a iluminação do endoscópio é central e decai para as bordas; o CLAHE
    realça o contraste local (borda livre das pregas, contorno da glote) sem estourar o centro.
    Parâmetros padrão: clip 2,0, grade 8×8.
    Referência: Zuiderveld, 1994; Pizer et al., 1987."""
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(img)


@register("stretch", p_low=1, p_high=99)
def stretch(img: np.ndarray, ctx: dict, p_low: float = 1, p_high: float = 99) -> np.ndarray:
    """O que faz: alargamento linear de contraste s = a·r + b, levando os percentis p_low e p_high
    a 0 e 255, com recorte do que fica fora desse intervalo.
    Por que importa: é a correção mais simples para um frame escuro ou lavado; como a transformação
    é global, não compensa a iluminação desigual do endoscópio, e por isso serve de base para
    comparar com a equalização global e o CLAHE.
    Parâmetros padrão: p_low=1, p_high=99 (percentis, que toleram os reflexos melhor que min/max).
    Referência: Gonzalez & Woods, Digital Image Processing, seção 3.2."""
    lo, hi = np.percentile(img, [p_low, p_high])
    if hi <= lo:
        return img.copy()
    return np.clip(np.round((img.astype(np.float64) - lo) * 255.0 / (hi - lo)), 0, 255).astype(np.uint8)


def _freq_d2(shape: tuple[int, int]) -> np.ndarray:
    """D²(u,v): distância ao quadrado até o centro do espectro (fftshift)."""
    h, w = shape
    u = np.arange(h) - h / 2
    v = np.arange(w) - w / 2
    return u[:, None] ** 2 + v[None, :] ** 2


def homo_transfer(shape: tuple[int, int], gl: float = 0.5, gh: float = 1.75, d0: float = 30,
                  c: float = 1.0) -> np.ndarray:
    """H(u,v) gaussiano de alta ênfase, centrado (fftshift)."""
    return (gh - gl) * (1 - np.exp(-c * _freq_d2(shape) / d0 ** 2)) + gl


def gauss_lpf_transfer(shape: tuple[int, int], sigma: float = 40) -> np.ndarray:
    """H(u,v) = exp(−D²/2σ²) do passa-baixa gaussiano, centrado (fftshift)."""
    return np.exp(-_freq_d2(shape) / (2.0 * sigma ** 2))


@register("gauss_lpf", sigma=40)
def gauss_lpf(img: np.ndarray, ctx: dict, sigma: float = 40) -> np.ndarray:
    """O que faz: filtragem no domínio da frequência com passa-baixa gaussiano — FFT → multiplica
    por H(u,v) = exp(−D²/2σ²) → IFFT → recorte em 0–255.
    Por que importa: mostra na prática que o ruído e as bordas finas moram nas altas frequências;
    atenuá-las reduz o ruído, mas também borra a borda livre das pregas, que é o que importa para
    segmentar a glote. O σ controla esse compromisso, e o filtro gaussiano não produz o anelamento
    do passa-baixa ideal.
    Parâmetros padrão: σ=40 (o estudo de ruído usa 15 e 40), em pixels de frequência, na escala
    de trabalho (lado maior ≤ 512).
    Referência: Gonzalez & Woods, Digital Image Processing, seção 4.8."""
    Z = np.fft.fftshift(np.fft.fft2(img.astype(np.float64)))
    s = np.real(np.fft.ifft2(np.fft.ifftshift(Z * gauss_lpf_transfer(img.shape, sigma))))
    return np.clip(np.round(s), 0, 255).astype(np.uint8)


@register("homo", gl=0.5, gh=1.75, d0=30, c=1.0)
def homo(img: np.ndarray, ctx: dict, gl: float = 0.5, gh: float = 1.75, d0: float = 30,
         c: float = 1.0) -> np.ndarray:
    """O que faz: filtragem homomórfica — log → FFT → filtro gaussiano de alta ênfase
    H(u,v) = (γH−γL)(1−e^{−cD²/D0²})+γL → IFFT → exp → reescala pelos percentis 0,5–99,5.
    Por que importa: separa a iluminação (baixa frequência, atenuada por γL) da refletância
    (alta frequência, realçada por γH); compensa o vinhetamento da fonte de luz pontual.
    Parâmetros padrão: γL=0,5; γH=1,75; D0=30 (lado maior ≤ 512); c=1.
    Referência: Gonzalez & Woods, Digital Image Processing, seção 4.9."""
    z = np.log1p(img.astype(np.float64))
    Z = np.fft.fftshift(np.fft.fft2(z))
    s = np.real(np.fft.ifft2(np.fft.ifftshift(Z * homo_transfer(z.shape, gl, gh, d0, c))))
    out = np.expm1(s)
    lo, hi = np.percentile(out, [0.5, 99.5])
    return np.clip((out - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8)

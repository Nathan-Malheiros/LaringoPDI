"""Registry de técnicas de PDI e execução de pipelines encadeados ("clahe>roi").

Toda técnica tem a assinatura `f(img, ctx, **params) -> img` e trabalha em **escala de cinza**
(uint8, 2-D); fusões multicanal devolvem (H, W, C). A conversão para cinza é sempre o 1º passo
e o `raw` (resize preservando a proporção + pad + replicação para 3 canais) é sempre o último.
Máscaras `*_mask` no `ctx` (inclusive `gt_mask`) acompanham recortes e o letterbox.
"""
from __future__ import annotations

import cv2
import numpy as np

from laringopdi.config import stable_hash

TECHNIQUES: dict = {}
DEFAULTS: dict = {}


def register(name: str, **defaults):
    def deco(f):
        TECHNIQUES[name] = f
        DEFAULTS[name] = defaults
        return f
    return deco


def parse(pipeline: str) -> list[str]:
    steps = [s.strip() for s in str(pipeline).split(">") if s.strip()]
    for s in steps:
        if s not in TECHNIQUES:
            raise KeyError(f"técnica desconhecida: {s!r} (disponíveis: {sorted(TECHNIQUES)})")
    return steps


def resolve_params(pipeline: str, params: dict | None = None) -> dict:
    """Parâmetros efetivos por passo; `method: best` é trocado pelo método fixado na Fase 3."""
    from laringopdi.pdi.segmentation import resolve_best

    params = params or {}
    out = {}
    for s in parse(pipeline):
        p = {**DEFAULTS[s], **params.get(s, {})}
        out[s] = resolve_best(p) if p.get("method") == "best" else p
    return out


def pipeline_hash(pipeline: str, params: dict | None = None, size: int = 224, work_size: int = 512) -> str:
    return stable_hash({"p": pipeline, "params": resolve_params(pipeline, params), "size": size,
                        "work": work_size, "gray": True})


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img[..., :3], cv2.COLOR_RGB2GRAY) if img.shape[2] >= 3 else img[..., 0]
    return img


def to_work_size(img: np.ndarray, work_size: int | None, interp=cv2.INTER_AREA) -> np.ndarray:
    """Reduz o lado maior para `work_size`; nunca amplia."""
    h, w = img.shape[:2]
    if not work_size or max(h, w) <= work_size:
        return img
    s = work_size / max(h, w)
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=interp)


def letterbox(img: np.ndarray, size: int = 224, interp=cv2.INTER_AREA) -> tuple[np.ndarray, dict]:
    """Redimensiona preservando a proporção e completa com zeros até `size`×`size`."""
    h, w = img.shape[:2]
    s = size / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    if img.ndim == 3 and img.shape[2] > 4:          # cv2.resize aceita no máximo 4 canais
        r = np.dstack([cv2.resize(img[..., c], (nw, nh), interpolation=interp) for c in range(img.shape[2])])
    else:
        r = cv2.resize(img, (nw, nh), interpolation=interp)
    if r.ndim == 2 and img.ndim == 3:
        r = r[..., None]
    top, left = (size - nh) // 2, (size - nw) // 2
    pad = [(top, size - nh - top), (left, size - nw - left)] + [(0, 0)] * (r.ndim - 2)
    return np.pad(r, pad), {"scale": s, "top": top, "left": left, "h": nh, "w": nw}


def crop_ctx(ctx: dict, shape: tuple[int, int], x0: int, y0: int, w: int, h: int) -> None:
    """Recorta todas as máscaras do ctx alinhadas à imagem atual."""
    for k in [k for k in ctx if k.endswith("_mask")]:
        if isinstance(ctx[k], np.ndarray) and ctx[k].shape[:2] == tuple(shape):
            ctx[k] = ctx[k][y0:y0 + h, x0:x0 + w]


@register("raw", size=224)
def raw(img: np.ndarray, ctx: dict, size: int = 224) -> np.ndarray:
    """O que faz: converte para cinza, redimensiona preservando a proporção, completa com preto
    até 224×224 e replica o canal 3× para a CNN.
    Por que importa: baseline sem PDI; não distorce a geometria da glote e das pregas vocais.
    Parâmetros padrão: size=224 (entrada da ResNet-18).
    Referência: He et al., 2016 (ResNet); prática padrão de letterboxing."""
    out, ctx["letterbox"] = letterbox(to_gray(img), size)
    return np.repeat(out[..., None], 3, axis=2) if out.ndim == 2 else out


@register("crop_square")
def crop_square(img: np.ndarray, ctx: dict) -> np.ndarray:
    """O que faz: recorta o maior quadrado central do frame; depois do resize, a imagem não tem bordas pretas.
    Por que importa: é um controle, não uma técnica do estudo. Elimina a proporção do vídeo, que muda com o
    hospital e o equipamento e sozinha já separa as classes (docs/DECISIONS.md, 2026-11-07).
    Parâmetros padrão: nenhum.
    Referência: verificação de fator de confusão; a glote fica no centro do frame do BAGLS."""
    h, w = img.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    crop_ctx(ctx, (h, w), x0, y0, s, s)
    return img[y0:y0 + s, x0:x0 + s]


def run_pipeline(img: np.ndarray, pipeline: str, params: dict | None = None, size: int = 224,
                 work_size: int | None = 512, gt_mask: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Executa o pipeline e devolve (imagem size×size×C uint8 com C ≥ 3, ctx)."""
    ctx: dict = {"orig_shape": img.shape[:2]}
    img = to_work_size(to_gray(img), work_size)
    if gt_mask is not None:
        gm = cv2.resize(gt_mask.astype(np.uint8), img.shape[::-1], interpolation=cv2.INTER_NEAREST)
        ctx["gt_mask"] = gm.astype(bool)
    ctx["work_shape"] = img.shape[:2]
    for step, p in resolve_params(pipeline, params).items():
        if step != "raw":
            img = TECHNIQUES[step](img, ctx, **p)
    out, ctx["letterbox"] = letterbox(img, size)
    if out.ndim == 2:
        out = np.repeat(out[..., None], 3, axis=2)
    elif out.shape[2] < 3:
        out = np.concatenate([out] + [out[..., -1:]] * (3 - out.shape[2]), axis=2)
    for k in [k for k in ctx if k.endswith("_mask")]:
        if isinstance(ctx[k], np.ndarray) and ctx[k].shape[:2] == img.shape[:2]:
            ctx[k + "_lb"] = letterbox(ctx[k].astype(np.uint8), size, cv2.INTER_NEAREST)[0].astype(bool)
    return out, ctx

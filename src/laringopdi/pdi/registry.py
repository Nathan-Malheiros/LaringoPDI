"""Registry de técnicas de PDI e execução de pipelines encadeados ("fov>spec>clahe").

Toda técnica tem a assinatura `f(img, ctx, **params) -> img`, recebe RGB uint8 e devolve
uint8 (H, W, C). Máscaras e artefatos vão para `ctx`. O `raw` (resize preservando a
proporção + pad) é sempre aplicado ao final; máscaras `*_mask` são transformadas junto.
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
    params = params or {}
    return {s: {**DEFAULTS[s], **params.get(s, {})} for s in parse(pipeline)}


def pipeline_hash(pipeline: str, params: dict | None = None, size: int = 224, work_size: int = 512) -> str:
    return stable_hash({"p": pipeline, "params": resolve_params(pipeline, params), "size": size, "work": work_size})


def to_work_size(img: np.ndarray, work_size: int | None) -> np.ndarray:
    """Reduz o lado maior para `work_size` (INTER_AREA); nunca amplia."""
    h, w = img.shape[:2]
    if not work_size or max(h, w) <= work_size:
        return img
    s = work_size / max(h, w)
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


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


@register("raw", size=224)
def raw(img: np.ndarray, ctx: dict, size: int = 224) -> np.ndarray:
    """O que faz: redimensiona preservando a proporção e completa com preto até 224×224.
    Por que importa: baseline sem PDI; evita distorcer a geometria das pregas vocais.
    Parâmetros padrão: size=224 (entrada da ResNet-18).
    Referência: He et al., 2016 (ResNet); prática padrão de letterboxing."""
    out, ctx["letterbox"] = letterbox(img, size)
    return out


def run_pipeline(img: np.ndarray, pipeline: str, params: dict | None = None,
                 size: int = 224, work_size: int | None = 512) -> tuple[np.ndarray, dict]:
    """Executa o pipeline e devolve (imagem size×size×C uint8, ctx)."""
    ctx: dict = {"orig_shape": img.shape[:2]}
    img = to_work_size(img, work_size)
    ctx["work_shape"] = img.shape[:2]
    for step, p in resolve_params(pipeline, params).items():
        if step == "raw":
            continue
        img = TECHNIQUES[step](img, ctx, **p)
        if img.ndim == 2:
            img = img[..., None]
    out, ctx["letterbox"] = letterbox(img, size)
    if out.ndim == 2:
        out = out[..., None]
    for k in [k for k in ctx if k.endswith("_mask")]:
        m = ctx[k].astype(np.uint8)
        if m.shape[:2] == img.shape[:2]:
            ctx[k + "_lb"] = letterbox(m, size, cv2.INTER_NEAREST)[0].astype(bool)
    return out, ctx

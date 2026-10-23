"""Telemetria opcional do treino para o painel LaringoMonitor (só observação).

Liga apenas se existir `results/live/enabled` (ou a variável LARINGO_LIVE_DIR). Não consome números
aleatórios nem altera o treino: lê o lote já aumentado, as saídas da rede, as ativações (hooks de
leitura) e a variação dos pesos depois de cada passo do otimizador. Arquivos (escrita atômica):
  status.json   estado, curvas, camadas (ativação e ‖ΔW‖/‖W‖), histograma de pesos, passos do PDI
  batch.jpg     lote atual (com augmentation) com rótulo real e palpite da rede
  val.jpg       Grad-CAM em 6 frames fixos da validação + contorno e caixa da glote (ground truth)
  filters0.jpg / filters.jpg   filtros da 1ª camada no início e na época atual
  fmaps.jpg     mapas de ativação da 1ª camada para um frame
  pdi_<k>.jpg   o pipeline de PDI do experimento aplicado passo a passo a um frame real
"""
from __future__ import annotations

import inspect
import json
import os
import re
import time
from pathlib import Path

import cv2
import numpy as np

from laringopdi import config as C
from laringopdi import data as D

NAMES_PT = {"healthy": "saudável", "disorder": "com distúrbio", "structural": "estrutural", "functional": "funcional"}
SAMPLE_EVERY = 10          # passos entre medições de ativação e de ajuste dos pesos
MAX_LAYERS = 12            # camadas mostradas no diagrama


def _dir() -> Path | None:
    env = os.environ.get("LARINGO_LIVE_DIR")
    if env:
        return Path(env)
    p = C.ROOT / "results" / "live"
    return p if (p / "enabled").exists() else None


def _atomic(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _jpg(img: np.ndarray, q: int = 88) -> bytes:
    if img.ndim == 2:
        img = np.dstack([img] * 3)
    return cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes()


def _denorm(x) -> np.ndarray:
    """Tensor normalizado (N,3,H,W) → uint8 (N,H,W,3)."""
    from laringopdi.train import IMAGENET_MEAN, IMAGENET_STD
    a = x.detach().float().cpu().numpy().transpose(0, 2, 3, 1)
    a = a * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    return np.clip(a * 255, 0, 255).astype(np.uint8)


def _ascii(s: str) -> str:
    for a, b in (("á", "a"), ("ú", "u"), ("ã", "a"), ("ç", "c"), ("é", "e"), ("í", "i"), ("ó", "o")):
        s = s.replace(a, b)
    return s


def _caption(img: np.ndarray, top: str, bottom: str, ok: bool | None) -> np.ndarray:
    img = cv2.resize(img, (200, 200), interpolation=cv2.INTER_AREA).copy()
    col = (124, 255, 178) if ok else (255, 92, 122) if ok is False else (220, 230, 255)
    cv2.rectangle(img, (0, 0), (199, 22), (6, 10, 31), -1)
    cv2.rectangle(img, (0, 178), (199, 199), (6, 10, 31), -1)
    cv2.putText(img, top, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 230, 255), 1, cv2.LINE_AA)
    cv2.putText(img, bottom, (6, 194), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (199, 199), col, 2)
    return img


def _diverging(k: np.ndarray) -> np.ndarray:
    """Kernel 2-D → RGB: negativo em azul/ciano, positivo em laranja, zero escuro."""
    m = np.abs(k).max() or 1.0
    v = k / m
    out = np.zeros(k.shape + (3,), np.float32)
    pos, neg = np.clip(v, 0, 1), np.clip(-v, 0, 1)
    out[..., 0] = 255 * pos + 92 * neg * .3
    out[..., 1] = 157 * pos + 242 * neg
    out[..., 2] = 77 * pos + 255 * neg
    return np.clip(out + 12, 0, 255).astype(np.uint8)


def _doc_part(f, key: str) -> str:
    d = inspect.getdoc(f) or ""
    m = re.search(rf"{key}:\s*(.*?)(?=\n?(?:O que faz|Por que importa|Parâmetros padrão|Referência):|\Z)", d, re.S)
    return " ".join(m.group(1).split()) if m else ""


class LiveTap:
    def __init__(self, out: Path, cfg, exp, task, model_name, fold, seed, d, epochs, iva, yva):
        self.out, self.cfg = out, cfg
        out.mkdir(parents=True, exist_ok=True)
        for f in out.glob("pdi_*.jpg"):
            f.unlink(missing_ok=True)
        names = D.task_names(cfg, task)
        self.names = [_ascii(NAMES_PT.get(n, n)) for n in names]
        self.st = {"exp": exp, "pipeline": cfg["experiments"].get(exp, {}).get("pipeline", exp), "task": task,
                   "task_name": cfg["tasks"][task]["name"], "model": model_name, "fold": fold, "seed": seed,
                   "classes": [NAMES_PT.get(n, n) for n in names], "epochs_max": epochs, "epoch": 0, "step": 0,
                   "steps": 0, "loss": None, "loss_hist": [], "val_hist": [], "best": None, "state": "training",
                   "n_train": int(len(d["train"][0])), "n_val": int(len(iva)), "started": time.time(),
                   "updated": time.time(), "layers": [], "pdi": [], "files": {}}
        self._last_batch = 0.0
        self._sampling = False
        self._hooks, self._acts, self._prev = [], {}, {}
        per = 6 // max(1, len(names)) + 1
        pick = np.concatenate([np.nonzero(yva == c)[0][:per] for c in range(len(names))])[:6]
        self.val_idx, self.val_y, self.X = iva[pick], yva[pick], d["X"]
        try:
            from laringopdi.features import _unpack
            m = np.load(D.cache_paths(cfg, exp)["masks"])
            order = np.argsort(np.argsort(self.val_idx))
            self.gt = _unpack(m["gt_mask_lb"][np.sort(self.val_idx)], cfg["data"]["image_size"])[order]
        except Exception:
            self.gt = None
        try:
            self._pdi_steps(d)
        except Exception:
            pass
        self._write()

    @classmethod
    def maybe(cls, *a, **k):
        out = _dir()
        if out is None:
            return None
        try:
            return cls(out, *a, **k)
        except Exception:
            return None

    # ------------------------------------------------------------------ PDI passo a passo
    def _pdi_steps(self, d) -> None:
        from laringopdi.pdi import TECHNIQUES, parse, run_pipeline
        i = int(self.val_idx[0])
        rel = d["paths"][i]
        raw = C.path(self.cfg, "raw")
        g = D.imread(raw / rel, gray=True)
        lab = d["meta"]
        seg = raw / rel.replace(".png", "_seg.png")
        gt = D.imread(seg, gray=True) > 127 if seg.exists() else None
        steps = [s for s in parse(self.st["pipeline"]) if s != "raw"]
        params = self.cfg["experiments"].get(self.st["exp"], {}).get("params")
        size, ws = self.cfg["data"]["image_size"], self.cfg["data"]["work_size"]
        out0, _ = run_pipeline(g, "raw", None, size, ws, gt_mask=gt)
        items = [("original", out0, "Frame em cinza, redimensionado para 224×224 sem distorcer.", "")]
        for k in range(len(steps)):
            o, _ = run_pipeline(g, ">".join(steps[:k + 1]), params, size, ws, gt_mask=gt)
            f = TECHNIQUES[steps[k]]
            items.append((steps[k], o, _doc_part(f, "O que faz"), _doc_part(f, "Por que importa")))
        self.st["pdi"] = []
        for k, (name, img, what, why) in enumerate(items):
            show = img[..., :3] if img.shape[2] >= 3 else img[..., 0]
            _atomic(self.out / f"pdi_{k}.jpg", _jpg(cv2.resize(np.ascontiguousarray(show), (300, 300), interpolation=cv2.INTER_AREA)))
            self.st["pdi"].append({"step": name, "what": what, "why": why, "file": f"pdi_{k}.jpg"})
        _ = lab

    # ------------------------------------------------------------------ rede: camadas, ativações, pesos
    def attach(self, model) -> None:
        """Registra hooks de leitura nas camadas convolucionais/lineares e fotografa os filtros iniciais."""
        try:
            import torch
            layers = [(n, m) for n, m in model.named_modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))]
            if len(layers) > MAX_LAYERS:
                keep = np.unique(np.linspace(0, len(layers) - 1, MAX_LAYERS).round().astype(int))
                layers = [layers[i] for i in keep]
            self.layers = layers
            self.st["layers"] = [{"name": n.replace("features.", "bloco ").replace(".", " "),
                                  "kind": "conv" if isinstance(m, torch.nn.Conv2d) else "linear",
                                  "units": int(m.out_channels if isinstance(m, torch.nn.Conv2d) else m.out_features),
                                  "kernel": list(m.kernel_size) if isinstance(m, torch.nn.Conv2d) else None,
                                  "params": int(sum(p.numel() for p in m.parameters())),
                                  "act": [], "upd": None} for n, m in layers]
            for i, (_, m) in enumerate(layers):
                self._hooks.append(m.register_forward_hook(self._make_hook(i)))
            self.first = next(m for _, m in model.named_modules() if isinstance(m, torch.nn.Conv2d))
            self.last = [m for _, m in model.named_modules() if isinstance(m, torch.nn.Linear)][-1]
            self._snapshot()
            self._filters("filters0.jpg")
            self._filters("filters.jpg")
            self._write()
        except Exception:
            self.layers = []

    def _make_hook(self, i: int):
        def hook(_m, _inp, out):
            if self._sampling:
                o = out.detach().float()
                a = o.abs().mean(dim=(0, 2, 3)) if o.ndim == 4 else o.abs().mean(dim=0)
                self._acts[i] = a
        return hook

    def _snapshot(self) -> None:
        self._prev = {i: m.weight.detach().float().clone() for i, (_, m) in enumerate(self.layers)}

    def _filters(self, fname: str) -> None:
        w = self.first.weight.detach().float().cpu().numpy().mean(axis=1)          # (out, k, k)
        tile = 36 if w.shape[-1] <= 3 else 40
        tiles = [cv2.resize(_diverging(k), (tile, tile), interpolation=cv2.INTER_NEAREST) for k in w[:64]]
        cols = 8
        while len(tiles) % cols:
            tiles.append(np.full_like(tiles[0], 6))
        pad = lambda t: cv2.copyMakeBorder(t, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(6, 10, 31))
        rows = [np.hstack([pad(t) for t in tiles[r:r + cols]]) for r in range(0, len(tiles), cols)]
        _atomic(self.out / fname, _jpg(np.vstack(rows), 92))

    def pre(self, step: int) -> None:
        self._sampling = step % SAMPLE_EVERY == 0

    def batch(self, xin, yb, logits, epoch: int, step: int, steps: int, loss: float) -> None:
        try:
            self.st.update(epoch=epoch + 1, step=step + 1, steps=steps, loss=loss)
            if self._sampling and self.layers:
                for i, (_, m) in enumerate(self.layers):
                    w = m.weight.detach().float()
                    rel = float((w - self._prev[i]).norm() / (self._prev[i].norm() + 1e-12))
                    a = self._acts.get(i)
                    L = self.st["layers"][i]
                    L["upd"] = rel
                    if a is not None:
                        k = a[:8].cpu().numpy()
                        L["act"] = [float(v) for v in k]
                        L["act_mean"] = float(a.mean())
                self._snapshot()
                self.st["loss_hist"] = (self.st["loss_hist"] + [[epoch + (step + 1) / steps, loss]])[-400:]
                self._sampling = False
            now = time.time()
            if now - self._last_batch < 2.0:
                return
            self._last_batch = now
            imgs = _denorm(xin[:8])
            prob = logits[:8].detach().float().softmax(1).cpu().numpy()
            y = yb[:8].cpu().numpy()
            tiles = [_caption(im, f"real: {self.names[t]}", f"rede: {self.names[p.argmax()]} {p.max():.0%}", p.argmax() == t)
                     for im, t, p in zip(imgs, y, prob)]
            while len(tiles) < 8:
                tiles.append(np.zeros_like(tiles[0]))
            _atomic(self.out / "batch.jpg", _jpg(np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:8])])))
            self._write()
        except Exception:
            pass

    def epoch(self, model, ep: int, val_f1: float, device) -> None:
        try:
            import torch
            from pytorch_grad_cam import GradCAM

            from laringopdi.models import gradcam_layer
            from laringopdi.train import _prep
            self.st["val_hist"].append([ep + 1, val_f1])
            self.st["best"] = max(val_f1, self.st["best"] or 0)
            model.eval()
            order = torch.from_numpy(np.argsort(np.argsort(self.val_idx)))
            x = _prep(torch.from_numpy(np.array(self.X[np.sort(self.val_idx)])), device)[order.to(device)]
            grab = {}
            h = self.first.register_forward_hook(lambda _m, _i, o: grab.setdefault("f", o.detach().float()))
            with torch.no_grad():
                prob = torch.softmax(model(x).float(), 1).cpu().numpy()
            h.remove()
            with GradCAM(model=model, target_layers=[gradcam_layer(model)]) as cam:
                maps = cam(input_tensor=x, targets=None)
            imgs = _denorm(x)
            tiles = []
            for i, (im, mp, t, p) in enumerate(zip(imgs, maps, self.val_y, prob)):
                heat = cv2.applyColorMap((mp * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
                v = (0.58 * im + 0.42 * heat).astype(np.uint8)
                att = float("nan")
                if self.gt is not None and self.gt[i].any():
                    g = self.gt[i].astype(np.uint8)
                    cs, _ = cv2.findContours(g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(v, cs, -1, (124, 255, 178), 2)
                    ys, xs = np.nonzero(g)
                    cv2.rectangle(v, (int(xs.min()) - 6, int(ys.min()) - 6), (int(xs.max()) + 6, int(ys.max()) + 6), (92, 242, 255), 2)
                    cv2.putText(v, "glote", (int(xs.min()) - 4, max(12, int(ys.min()) - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (92, 242, 255), 1, cv2.LINE_AA)
                    pm = np.clip(mp, 0, None)
                    att = float((pm * self.gt[i]).sum() / max(pm.sum(), 1e-9))
                tiles.append(_caption(v, f"real: {self.names[t]}  glote {att:.0%}",
                                      f"rede: {self.names[p.argmax()]} {p.max():.0%}", p.argmax() == t))
            grid = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:6])]) if len(tiles) == 6 else np.hstack(tiles)
            _atomic(self.out / "val.jpg", _jpg(grid))
            if "f" in grab:                                                   # mapas de ativação da 1ª camada
                fm = grab["f"][0].cpu().numpy()[:16]
                tiles = []
                for a in fm:
                    a = (a - a.min()) / (np.ptp(a) + 1e-9)
                    c = cv2.applyColorMap((a * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)[..., ::-1]
                    tiles.append(cv2.copyMakeBorder(cv2.resize(c, (96, 96)), 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(6, 10, 31)))
                while len(tiles) % 8:
                    tiles.append(np.full_like(tiles[0], 6))
                _atomic(self.out / "fmaps.jpg", _jpg(np.vstack([np.hstack(tiles[r:r + 8]) for r in range(0, len(tiles), 8)])))
            self._filters("filters.jpg")
            w = self.last.weight.detach().float().cpu().numpy().ravel()
            hcount, edges = np.histogram(w, bins=30)
            self.st["weights_hist"] = {"counts": hcount.tolist(), "lo": float(edges[0]), "hi": float(edges[-1])}
            self.st["val_epoch"] = ep + 1
            self._write()
        except Exception:
            pass

    def finish(self, test_video_f1: float) -> None:
        try:
            for h in self._hooks:
                h.remove()
            self.st.update(state="done", test_video_f1=test_video_f1)
            self._write()
        except Exception:
            pass

    def _write(self):
        self.st["updated"] = time.time()
        _atomic(self.out / "status.json", json.dumps(self.st, ensure_ascii=False).encode())

import json

import cv2
import numpy as np
import pytest

from laringopdi import config as C


def synth_glottis(seed: int = 0, h: int = 256, w: int = 256, video_seed: int | None = None):
    """Frame sintético em cinza: fundo com textura de baixa frequência (única por vídeo), glote
    escura central (elipse) e reflexos brilhantes. Devolve (img, gt_mask)."""
    vs = np.random.default_rng(seed if video_seed is None else video_seed)
    rng = np.random.default_rng(seed)
    bg = cv2.resize(vs.uniform(110, 190, (6, 6)), (w, h), interpolation=cv2.INTER_CUBIC)
    img = np.clip(bg + rng.normal(0, 4, (h, w)), 0, 255).astype(np.uint8)
    gt = np.zeros((h, w), np.uint8)
    ax = (int(vs.integers(12, 22)), int(vs.integers(45, 75)))
    cv2.ellipse(gt, (w // 2 + int(vs.integers(-10, 10)), h // 2), ax, float(vs.integers(-10, 10)), 0, 360, 1, -1)
    img[gt > 0] = rng.integers(15, 35)
    for _ in range(3):
        cv2.circle(img, (int(rng.integers(w * .25, w * .75)), int(rng.integers(h * .25, h * .75))), 3, 255, -1)
    return img, gt.astype(bool)


@pytest.fixture
def glottis():
    return synth_glottis()


def write_fake_bagls(cfg, videos: dict, frames_per_video: int = 4, dup: tuple | None = None):
    """Cria `training/<n>.png`, `<n>_seg.png` e `<n>.meta` no formato do BAGLS.
    `dup=(origem, cópia)`: o vídeo `cópia` reusa a aparência do vídeo `origem`."""
    from laringopdi import data as D
    raw = C.path(cfg, "raw") / "training"
    names = list(videos)
    n = 0
    for vid, status in videos.items():
        src = dup[0] if dup and vid == dup[1] else vid
        vseed = names.index(src) + 100
        for k in range(frames_per_video):
            img, gt = synth_glottis(seed=vseed * 10 + k, video_seed=vseed)
            D.imwrite(raw / f"{n}.png", img)
            D.imwrite(raw / f"{n}_seg.png", gt.astype(np.uint8) * 255)
            (raw / f"{n}.meta").write_text(json.dumps({"Video Id": vid, "Subject disorder status": status,
                                                       "Color": False, "Subject sex": "f"}))
            n += 1


@pytest.fixture
def tmp_cfg(tmp_path):
    cfg = C.load()
    for k in cfg["paths"]:
        cfg["paths"][k] = str(tmp_path / cfg["paths"][k])
    return cfg

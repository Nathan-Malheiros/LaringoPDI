import cv2
import numpy as np
import pytest

from laringopdi import config as C


def synth_larynx(seed: int = 0, h: int = 480, w: int = 640) -> np.ndarray:
    """Imagem sintética: fundo preto, campo circular rosado, fenda glótica escura, reflexos."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.uint8)
    cv2.circle(img, (w // 2, h // 2), int(h * 0.45), (190, 110, 110), -1)
    img = np.clip(img.astype(int) + rng.integers(-15, 15, img.shape) * (img > 0), 0, 255).astype(np.uint8)
    cv2.ellipse(img, (w // 2, h // 2), (25, 80), 0, 0, 360, (40, 20, 20), -1)
    for _ in range(4):
        cv2.circle(img, (int(rng.integers(w * 0.35, w * 0.65)), int(rng.integers(h * 0.3, h * 0.7))), 6, (255, 255, 255), -1)
    return img


@pytest.fixture
def larynx():
    return synth_larynx()


@pytest.fixture
def tmp_cfg(tmp_path):
    cfg = C.load()
    for k in cfg["paths"]:
        cfg["paths"][k] = str(tmp_path / cfg["paths"][k])
    return cfg

import numpy as np
import pandas as pd
import pytest

from laringopdi.etapa1 import FILTERS, add_noise, apply_filter, edge_f1
from laringopdi.features import FAMILIES_18, _letterbox_region, features18


def test_add_noise_salt_pepper_density_and_determinism():
    g = np.full((400, 400), 128, np.uint8)
    a = add_noise(g, "sp_10", np.random.default_rng(1))
    assert np.array_equal(a, add_noise(g, "sp_10", np.random.default_rng(1)))
    assert (a != 128).mean() == pytest.approx(0.10, abs=0.01)
    assert set(np.unique(a)) == {0, 128, 255}


@pytest.mark.parametrize("name", list(FILTERS))
def test_apply_filter_keeps_shape(name):
    g = np.random.default_rng(0).integers(0, 256, (64, 80)).astype(np.uint8)
    out = apply_filter(g, name)
    assert out.shape == g.shape and out.dtype == np.uint8


def test_median_beats_mean_on_salt_pepper():
    from skimage.metrics import peak_signal_noise_ratio as psnr
    g = np.full((200, 200), 120, np.uint8)
    n = add_noise(g, "sp_5", np.random.default_rng(0))
    assert psnr(g, apply_filter(n, "median3"), data_range=255) > psnr(g, apply_filter(n, "mean3"), data_range=255)


def test_edge_f1():
    a = np.zeros((4, 4), bool)
    assert edge_f1(a, a) == 1.0
    b = a.copy()
    b[0, 0] = True
    assert edge_f1(b, b) == 1.0 and edge_f1(b, a) == 0.0


def _frame():
    g = np.zeros((224, 224), np.uint8)
    g[40:184, :] = np.random.default_rng(0).integers(90, 200, (144, 224))     # pad em cima e embaixo
    m = np.zeros((224, 224), bool)
    m[90:130, 100:120] = True
    g[m] = 20
    return g, m


def test_features18_has_exactly_18_by_family():
    g, m = _frame()
    f = features18(g, m)
    assert len(f) == 18 and all(np.isfinite(v) for v in f.values())
    for fam in FAMILIES_18:
        assert sum(k.startswith(fam) for k in f) == 6


def test_features18_contrast_and_pad():
    g, m = _frame()
    f = features18(g, m)
    assert f["f18int_contrast"] > 0.2                  # a glote é mais escura que o tecido
    assert f["f18int_mean"] > 90                       # o pad não entra na média do tecido
    assert _letterbox_region(g).sum() == 144 * 224


def test_features18_empty_mask():
    g, _ = _frame()
    f = features18(g, np.zeros_like(g, dtype=bool))
    assert len(f) == 18 and f["f18int_contrast"] == 0.0 and f["f18shape_area_rel"] == 0.0
    assert all(np.isfinite(v) for v in f.values())


def test_adaptive_morph_steps_and_result():
    from laringopdi.pdi.segmentation import adaptive_morph_steps, seg_adaptive_morph
    g = np.full((200, 200), 150, np.uint8)
    g[70:130, 95:105] = 20
    g[10, 10] = 20                                     # ponto isolado: a abertura precisa apagar
    s = adaptive_morph_steps(g)
    assert list(s) == ["limiar", "abertura", "fechamento", "componente"]
    assert s["limiar"][10, 10] and not s["abertura"][10, 10]
    assert s["componente"][100, 100] and not s["componente"][10, 10]
    assert np.array_equal(seg_adaptive_morph(g), s["componente"])

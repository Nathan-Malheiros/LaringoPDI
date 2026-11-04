import numpy as np
import pandas as pd
import pytest

from laringopdi.etapa1 import FILTERS, add_noise, apply_filter, edge_f1, seg_pairs
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


def test_seg_pairs_holm_and_direction():
    idx = pd.MultiIndex.from_product([["m1", "m2"], [str(i) for i in range(30)]], names=["method", "video_id"])
    rng = np.random.default_rng(0)
    a = rng.random(60)
    vid = pd.DataFrame({"x": a, "y": a + 0.1 + rng.random(60) * 0.01}, index=idx)
    r = seg_pairs(vid, "x", "y")
    assert len(r) == 2 and (r["delta_b_minus_a"] > 0).all() and (r["p_holm"] >= r["p"]).all() and (r["p_holm"] < 0.05).all()


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


def test_p5_tests_holm_within_model_task():
    from laringopdi.etapa1 import p5_tests
    rng = np.random.default_rng(0)
    rows = []
    for v in range(40):
        base = rng.random() + 1
        rows += [("E00", "m", "A", str(v), base), ("E01", "m", "A", str(v), base + 0.5 + rng.random() * 0.01), ("E04", "m", "A", str(v), base + rng.normal(0, 0.1))]
    v = pd.DataFrame(rows, columns=["exp", "model", "task", "video_id", "ratio"])
    r = p5_tests(v, ["E01", "E04"], "sem_recorte").set_index("exp")
    assert r.loc["E01", "delta"] > 0.4 and r.loc["E01", "p_holm"] < 0.05 and r.loc["E01", "n_videos"] == 40
    assert (r["p_holm"] >= r["p"]).all()


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


def test_bootstrap_delta_sens_sign():
    from laringopdi.etapa1 import bootstrap_delta_sens
    y = np.array([1] * 40 + [0] * 40)
    pa = np.array([0] * 40 + [0] * 40)
    pb = np.array([1] * 40 + [0] * 40)
    lo, hi = bootstrap_delta_sens(y, pa, pb, 200)
    assert lo == 1.0 and hi == 1.0


def test_old_verdict_family_does_not_change_with_class_jobs():
    """Com os jobs novos em disco, a família padrão de vereditos da ResNet-18 continua a mesma."""
    import pandas as pd

    from laringopdi import config as C
    from laringopdi.evaluate import collect_runs, verdicts
    cfg = C.load()
    t = C.path(cfg, "tables") / "verdicts_resnet18.csv"
    if not t.exists() or collect_runs(cfg).empty:
        pytest.skip("sem resultados locais")
    v = verdicts(cfg, "resnet18")
    assert not set(v["exp"]) & set(cfg["stage_aula"]["experiments"])
    old = pd.read_csv(t).set_index(["task", "exp"])
    new = v.set_index(["task", "exp"])
    assert set(new.index) == set(old.index)
    assert (new["verdict"] == old.loc[new.index, "verdict"]).all()

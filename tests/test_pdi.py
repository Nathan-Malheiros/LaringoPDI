import numpy as np
import pytest

from laringopdi import config as C
from laringopdi.pdi import TECHNIQUES, letterbox, parse, run_pipeline
from laringopdi.pdi.quality import image_quality

CFG = C.load()
PIPES = [(e, v["pipeline"], v.get("params")) for e, v in CFG["experiments"].items() if v.get("pipeline")]


def test_all_spec_techniques_registered():
    for t in ["raw", "fov", "spec", "he", "clahe", "homo", "bilat", "median", "lab", "hsv",
              "vessel", "roi", "edges", "lbp", "m1", "m2"]:
        assert t in TECHNIQUES
        assert TECHNIQUES[t].__doc__ and "Por que importa" in TECHNIQUES[t].__doc__


def test_parse_rejects_unknown():
    with pytest.raises(KeyError):
        parse("fov>nope")


@pytest.mark.parametrize("h,w", [(480, 640), (640, 480), (100, 100)])
def test_letterbox_shape(h, w):
    out, lb = letterbox(np.full((h, w, 3), 128, np.uint8), 224)
    assert out.shape == (224, 224, 3)
    assert max(lb["h"], lb["w"]) == 224


@pytest.mark.parametrize("exp,pipe,params", PIPES)
def test_pipelines_run(larynx, exp, pipe, params):
    out, ctx = run_pipeline(larynx, pipe, params)
    assert out.dtype == np.uint8 and out.shape[:2] == (224, 224)
    assert out.shape[2] == (5 if "m2" in pipe else 3)
    q = image_quality(out, ctx)
    assert np.isfinite(q["entropy"]) and q["entropy"] > 0


def test_fov_crops_black_border(larynx):
    _, ctx = run_pipeline(larynx, "fov")
    x, y, w, h = ctx["fov_bbox"]
    assert not ctx["fov_fallback"] and w < 512 * 0.8 and h <= 384


def test_spec_removes_highlights(larynx):
    from laringopdi.pdi.enhance import specular_mask
    out, ctx = run_pipeline(larynx, "fov>spec")
    assert ctx["spec_frac"] > 0
    assert specular_mask(out).mean() < 0.2 * ctx["spec_frac"]
    assert ctx["spec_mask_lb"].shape == (224, 224)


@pytest.mark.parametrize("method", ["otsu", "adaptive"])
def test_roi_finds_glottis(larynx, method):
    _, ctx = run_pipeline(larynx, "fov>roi", {"roi": {"method": method}})
    assert "roi_bbox" in ctx and isinstance(ctx["roi_fallback"], bool)
    if method == "otsu":
        assert not ctx["roi_fallback"]


def test_roi_fallback_on_flat_image():
    _, ctx = run_pipeline(np.full((300, 300, 3), 150, np.uint8), "roi")
    assert ctx["roi_fallback"]


def test_determinism(larynx):
    a, _ = run_pipeline(larynx, "fov>spec>clahe>m1")
    b, _ = run_pipeline(larynx, "fov>spec>clahe>m1")
    assert np.array_equal(a, b)

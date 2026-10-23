import numpy as np
import pytest

from laringopdi import config as C
from laringopdi.pdi import TECHNIQUES, letterbox, parse, run_pipeline
from laringopdi.pdi.quality import image_quality
from laringopdi.pdi.segmentation import PARAM_GRID, SEGMENTERS, dice, hd95, seg_metrics, segment

CFG = C.load()
PIPES = [(e, v["pipeline"], v.get("params")) for e, v in CFG["experiments"].items() if v.get("pipeline")]


def test_all_spec_techniques_registered():
    for t in ["raw", "fov", "spec", "he", "clahe", "homo", "bilat", "median", "roi_gt", "roi",
              "edges", "lbp", "m1", "m2"]:
        assert t in TECHNIQUES and "Por que importa" in TECHNIQUES[t].__doc__
    for t in ("lab", "hsv", "vessel"):
        assert t not in TECHNIQUES
    for f in SEGMENTERS.values():
        assert "Por que importa" in f.__doc__


def test_parse_rejects_unknown():
    with pytest.raises(KeyError):
        parse("clahe>nope")


@pytest.mark.parametrize("h,w", [(256, 512), (512, 256), (100, 100)])
def test_letterbox_shape(h, w):
    out, lb = letterbox(np.full((h, w), 128, np.uint8), 224)
    assert out.shape == (224, 224) and max(lb["h"], lb["w"]) == 224


@pytest.mark.parametrize("exp,pipe,params", PIPES)
def test_pipelines_run(glottis, exp, pipe, params):
    img, gt = glottis
    out, ctx = run_pipeline(img, pipe, params, gt_mask=gt)
    assert out.dtype == np.uint8 and out.shape == (224, 224, 3)
    assert ctx["gt_mask_lb"].shape == (224, 224)
    assert np.isfinite(image_quality(out, ctx)["entropy"])


def test_rgb_input_becomes_gray(glottis):
    img, _ = glottis
    a, _ = run_pipeline(np.dstack([img, img, img]), "raw")
    assert np.array_equal(a[..., 0], a[..., 1])


def test_spec_removes_highlights(glottis):
    img, _ = glottis
    out, ctx = run_pipeline(img, "spec", work_size=None, size=256)
    assert ctx["spec_frac"] > 0
    assert (out[..., 0] >= 250).mean() < 0.1 * (img >= 250).mean()


def test_roi_gt_contains_glottis(glottis):
    img, gt = glottis
    _, ctx = run_pipeline(img, "roi_gt", gt_mask=gt)
    assert not ctx["roi_fallback"] and ctx["gt_mask"].sum() == gt.sum()


def test_roi_classic_finds_glottis(glottis):
    img, gt = glottis
    _, ctx = run_pipeline(img, "roi", gt_mask=gt)
    assert not ctx["roi_fallback"]
    assert ctx["gt_mask"].sum() >= 0.9 * gt.sum()      # o recorte mantém a glote


def test_roi_fallback_on_flat_image():
    _, ctx = run_pipeline(np.full((200, 200), 150, np.uint8), "roi")
    assert ctx["roi_fallback"]


@pytest.mark.parametrize("method", list(SEGMENTERS))
def test_segmenters_run_all_params(glottis, method):
    img, gt = glottis
    for p in PARAM_GRID[method]:
        m = segment(img, method, **p)
        assert m.dtype == bool and m.shape == gt.shape


@pytest.mark.parametrize("method", ["otsu_morph", "region_grow", "watershed"])
def test_central_segmenters_are_accurate(glottis, method):
    img, gt = glottis
    assert dice(segment(img, method), gt) > 0.7


def test_seg_metrics_edge_cases():
    e, f = np.zeros((10, 10), bool), np.ones((10, 10), bool)
    assert dice(e, e) == 1.0 and np.isnan(hd95(e, f))
    m = seg_metrics(e, f)
    assert m["fail"] == 1.0 and m["dice"] == 0.0


def test_determinism(glottis):
    img, gt = glottis
    a, _ = run_pipeline(img, "spec>clahe>m1", gt_mask=gt)
    b, _ = run_pipeline(img, "spec>clahe>m1", gt_mask=gt)
    assert np.array_equal(a, b)

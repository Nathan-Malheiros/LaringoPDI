import cv2
import numpy as np
import pandas as pd
import torch

from laringopdi import config as C
from laringopdi.evaluate import compute_metrics, metrics_both_levels
from laringopdi.features import FAMILIES, frame_features, shape_features
from laringopdi.models import SimpleCNN, gradcam_layer
from laringopdi.train import _balance, make_shards, stage_jobs
from tests.conftest import synth_glottis

CFG = C.load()


def test_config_keeps_classic_hyperparams():
    assert "mlp" in CFG["classic"] and "C1" in CFG["classic_experiments"]


def test_compute_metrics_binary_and_multiclass():
    y = np.array([0, 0, 1, 1])
    m = compute_metrics(y, np.array([[.9, .1], [.6, .4], [.3, .7], [.8, .2]]))
    assert m["bal_acc"] == 0.75 and m["sensitivity"] == 0.5 and m["specificity"] == 1.0 and m["auc"] == 0.75
    m3 = compute_metrics(np.array([0, 1, 2]), np.eye(3))
    assert m3["f1_macro"] == 1.0 and m3["confusion"] == np.eye(3, dtype=int).tolist()


def test_video_level_averages_frames():
    df = pd.DataFrame({"path": list("abcd"), "video_id": ["v1", "v1", "v2", "v2"], "y": [1, 1, 0, 0],
                       "p0": [.6, .2, .9, .7], "p1": [.4, .8, .1, .3]})
    m = metrics_both_levels(df)
    assert m["video"]["n"] == 2 and m["video"]["bal_acc"] == 1.0 and m["frame"]["n"] == 4


def test_simplecnn_forward_and_gradcam_layer():
    net = SimpleCNN(3)
    assert net(torch.zeros(2, 3, 224, 224)).shape == (2, 3)
    assert isinstance(gradcam_layer(net), torch.nn.Conv2d)


def test_features_families_and_shape():
    img, gt = synth_glottis()
    g = cv2.resize(img, (224, 224))
    m = cv2.resize(gt.astype(np.uint8), (224, 224), interpolation=cv2.INTER_NEAREST).astype(bool)
    f = frame_features(g, m)
    assert all(k.startswith(FAMILIES) for k in f) and all(np.isfinite(v) for v in f.values())
    s = shape_features(m)
    assert s["shape_symmetry"] > 0.8 and s["shape_aspect"] > 2 and s["shape_empty"] == 0
    assert shape_features(np.zeros((50, 50), bool))["shape_empty"] == 1


def test_balance_oversamples_minority():
    y = np.array([0] * 8 + [1] * 2)
    b = _balance(np.arange(10), y, 0)
    assert np.bincount(y[b]).tolist() == [8, 8]


def test_shards_cover_all_jobs_once(tmp_cfg):
    out = make_shards(tmp_cfg, "stage1", 5)
    jobs = [tuple(j.values()) for v in out["shards"].values() for j in v]
    assert len(jobs) == len(set(jobs)) == len(stage_jobs(tmp_cfg, "stage1")) == 140

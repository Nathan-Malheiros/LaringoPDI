"""Fumaça ponta a ponta sobre um BAGLS sintético: index → splits → eda → gallery → seg-study → cache."""
import json

import numpy as np

from laringopdi import config as C
from laringopdi import data as D
from laringopdi import evaluate as E
from laringopdi import report as R
from laringopdi.pdi import segmentation as S
from tests.test_splits import _videos
from tests.conftest import write_fake_bagls


def test_smoke_pipeline(tmp_cfg, monkeypatch):
    tmp_cfg["data"]["frames_per_video"] = 3
    tmp_cfg["segmentation"]["preprocess"] = ["raw", "clahe"]
    write_fake_bagls(tmp_cfg, _videos(), frames_per_video=4)
    D.build_index(tmp_cfg)
    D.sample_frames(tmp_cfg)
    D.build_groups(tmp_cfg)
    D.make_splits(tmp_cfg)
    assert R.eda(tmp_cfg)["videos"] == 29
    assert (C.path(tmp_cfg, "docs") / "dataset_report.md").exists()

    out = E.seg_study(tmp_cfg, n_jobs=1)
    tables = C.path(tmp_cfg, "tables")
    best = json.loads((tables / "segmentation_best.json").read_text(encoding="utf-8"))
    assert best["method"] in S.SEGMENTERS and out["top3"][0]["dice_mean"] > 0.5
    assert (C.path(tmp_cfg, "figures") / "segmentation_examples.png").exists()

    monkeypatch.setattr(S, "best_config", lambda: best)
    R.gallery(tmp_cfg)
    for e in ("E00", "E07", "E08", "E13"):
        r = D.build_cache(tmp_cfg, e, n_jobs=1)
        assert r["shape"] == [81, 224, 224, 3]
    X, paths, meta = D.load_cache(tmp_cfg, "E07")
    assert X.dtype == np.uint8 and "gt_mask_lb" in meta["mask_keys"]
    assert D.build_cache(tmp_cfg, "E07", n_jobs=1)["skipped"]
    assert len(D.quality_summary(tmp_cfg)) == 4

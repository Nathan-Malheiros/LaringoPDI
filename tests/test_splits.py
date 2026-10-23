import shutil

import numpy as np

from laringopdi import config as C
from laringopdi import data as D


def _fake_dataset(cfg, per_class=10):
    raw = C.path(cfg, "raw")
    rng = np.random.default_rng(0)
    for i, c in enumerate(cfg["data"]["classes"]):
        for k in range(per_class):
            D.imwrite(raw / f"{i + 1}" / f"img_{c}_{k}.png", rng.integers(0, 255, (64, 80, 3), dtype=np.uint8))
    # quase-duplicata (mesma imagem) em classe 1: precisa cair no mesmo grupo
    shutil.copy(raw / "1" / "img_reinke_edema_0.png", raw / "1" / "img_reinke_edema_dup.png")
    return raw


def test_index_groups_splits_no_leak(tmp_cfg):
    _fake_dataset(tmp_cfg)
    s = D.build_index(tmp_cfg)
    assert s["n_images"] == 81 and s["per_class"]["normal"] == 10 and s["exact_duplicates"] == 1
    g = D.build_groups(tmp_cfg)
    assert g["confirmed_pairs"] >= 1 and g["n_groups"] == 80
    D.make_splits(tmp_cfg)
    folds = D.load_splits(tmp_cfg)
    import pandas as pd
    grp = pd.read_csv(C.path(tmp_cfg, "groups")).set_index("path")["group"].to_dict()
    D.check_leakage(folds, grp)
    a, b = "1/img_reinke_edema_0.png", "1/img_reinke_edema_dup.png"
    for f in folds:
        for part in ("train", "val", "test"):
            assert (a in f[part]) == (b in f[part])


def test_splits_refuse_overwrite(tmp_cfg):
    import pytest
    _fake_dataset(tmp_cfg)
    D.build_index(tmp_cfg)
    D.make_splits(tmp_cfg)
    with pytest.raises(SystemExit):
        D.make_splits(tmp_cfg)


def test_real_splits_no_leak():
    """Roda sobre os splits reais quando existirem."""
    import pytest
    cfg = C.load()
    if not (C.path(cfg, "splits") / "folds.json").exists():
        pytest.skip("splits reais ainda não gerados")
    import pandas as pd
    grp = pd.read_csv(C.path(cfg, "groups")).set_index("path")["group"].to_dict()
    D.check_leakage(D.load_splits(cfg), grp)

import pandas as pd
import pytest

from laringopdi import config as C
from laringopdi import data as D
from tests.conftest import write_fake_bagls

CFG = C.load()


@pytest.mark.parametrize("raw,fam", [
    ("healthy", "healthy"), ("Muscle tension dysphonia", "functional"), ("Vocal fold nodules", "structural"),
    ("Muscle tension dysphonia with nodules", "structural"), ("", "unknown"), ("other", "other"),
    ("Posterior insufficient glottic closure", "functional"), ("Bilateral Reinke's edema", "structural"),
    ("leucoplacia", "structural"), ("Vocal insufficiency and contact granuloma", "structural"),
    ("None (higher phonation)", "healthy"),
    ("None (half year post-phonomicrosurgery for polipoid mid-membranous lesions)", "unknown"),
    ("None (one year after presumption of a pseudocyst/sulcus in left vocal fold)", "unknown"),
    ("scar", "structural"), ("Bilateral vergeture with bowed insufficient glottic closure", "structural"),
    ("Bowed/oval insufficient glottic closure (left more than right); stifness right vocal fold", "structural"),
    ("spasmodic dysphonia", "functional"), ("Laryngitis", "functional")])
def test_disorder_family(raw, fam):
    assert D.disorder_family(raw, CFG) == fam


def _videos():
    v = {f"h{i}": "healthy" for i in range(10)}
    v.update({f"f{i}": "Muscle tension dysphonia" for i in range(8)})
    v.update({f"s{i}": "Nodules" for i in range(8)})
    v.update({"u0": "Unknown", "o0": "other", "h_dup": "healthy"})
    return v


def test_bagls_index_sampling_groups_splits(tmp_cfg):
    tmp_cfg["data"]["frames_per_video"] = 3
    write_fake_bagls(tmp_cfg, _videos(), frames_per_video=4, dup=("h0", "h_dup"))
    s = D.build_index(tmp_cfg)
    assert s["n_videos"] == 29 and s["n_frames"] == 116
    assert s["task_videos"]["B"] == {-1: 10, 0: 11, 1: 8}
    meta = D.sample_frames(tmp_cfg)
    assert meta["n_videos"] == 27 and meta["n_frames"] == 81
    g = D.build_groups(tmp_cfg)
    assert g["confirmed_pairs"] >= 1
    grp = pd.read_csv(C.path(tmp_cfg, "groups"), dtype={"video_id": str}).set_index("video_id")["group"]
    assert grp["h0"] == grp["h_dup"]
    D.make_splits(tmp_cfg)
    folds = D.load_splits(tmp_cfg)
    D.check_leakage(folds, grp.to_dict())
    for f in folds:
        assert ("h0" in f["test"]) == ("h_dup" in f["test"])
    with pytest.raises(SystemExit):
        D.make_splits(tmp_cfg)                       # nunca sobrescreve sem --force
    assert len(D.fold_frames(tmp_cfg, 0, "test", "B")) > 0


def test_real_splits_no_leak():
    cfg = C.load()
    if not (C.path(cfg, "splits") / "folds.json").exists():
        pytest.skip("splits reais ainda não gerados")
    grp = pd.read_csv(C.path(cfg, "groups"), dtype={"video_id": str}).set_index("video_id")["group"].to_dict()
    D.check_leakage(D.load_splits(cfg), grp)

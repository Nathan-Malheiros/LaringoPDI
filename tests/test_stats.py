"""OOF por vídeo com várias sementes (ensemble) e sementes comuns entre experimentos."""
import pandas as pd

from laringopdi import config as C
from laringopdi import evaluate as E


def _preds(cfg, exp, k, s, p1):
    d = C.path(cfg, "runs") / exp / "A" / "resnet18" / f"fold{k}_seed{s}"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"path": [f"{k}a", f"{k}b"], "video_id": [f"v{k}"] * 2, "y": [1, 1],
                  "p0": [1 - p1] * 2, "p1": [p1] * 2}).to_csv(d / "preds.csv", index=False)


def test_oof_ensemble_and_common_seeds(tmp_cfg):
    folds = tmp_cfg["study"]["folds"]
    for k in folds:
        _preds(tmp_cfg, "E00", k, 0, 0.8)   # semente 0 acerta
        _preds(tmp_cfg, "E00", k, 1, 0.1)   # semente 1 erra com mais confiança → média 0,45 → erra
        _preds(tmp_cfg, "E01", k, 0, 0.6)
    _preds(tmp_cfg, "E00", folds[0], 2, 0.9)  # semente 2 incompleta: não conta
    assert E.run_seeds(tmp_cfg, "E00", "A", "resnet18") == {0, 1}
    assert E.run_seeds(tmp_cfg, "E01", "A", "resnet18") == {0}
    assert (E.oof_video_preds(tmp_cfg, "E00", "A", "resnet18", [0])["pred"] == 1).all()
    o = E.oof_video_preds(tmp_cfg, "E00", "A", "resnet18", [0, 1])
    assert len(o) == len(folds) and (o["pred"] == 0).all()
    assert E.oof_video_preds(tmp_cfg, "E01", "A", "resnet18", [0, 1]).empty

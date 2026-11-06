"""Ponta a ponta em CPU sobre um BAGLS sintético (~30 frames): índice → splits → segmentação →
cache → treino (1 época) → vereditos → Grad-CAM. Alvo: < 5 min."""
import json
from types import SimpleNamespace

from laringopdi import data as D
from laringopdi import evaluate as E
from laringopdi import train as T
from tests.conftest import write_fake_bagls


def test_end_to_end(tmp_cfg):
    vids = {f"h{i}": "healthy" for i in range(5)}
    vids.update({f"f{i}": "Muscle tension dysphonia" for i in range(5)})
    vids.update({f"s{i}": "Vocal fold nodules" for i in range(5)})
    tmp_cfg["data"]["frames_per_video"] = 2
    tmp_cfg["data"]["inner_val_frac"] = 0.34          # poucos vídeos: 3 sub-folds internos
    tmp_cfg["segmentation"]["preprocess"] = ["raw", "clahe"]
    tmp_cfg["study"].update({"experiments": ["E00", "E03"], "tasks": ["A"], "seeds": [0]})
    write_fake_bagls(tmp_cfg, vids, frames_per_video=2)
    D.build_index(tmp_cfg); D.sample_frames(tmp_cfg); D.build_groups(tmp_cfg); D.make_splits(tmp_cfg)
    E.seg_study(tmp_cfg, n_jobs=1)
    for e in ("E00", "E03"):
        D.build_cache(tmp_cfg, e, n_jobs=1)
    for e in ("E00", "E03"):
        for k in range(5):
            r = T.train_job(tmp_cfg, e, "A", "resnet18", k, 0, epochs=1)
            assert 0 <= r["test_video_f1"] <= 1
    m = json.loads((T.run_dir(tmp_cfg, "E03", "A", "resnet18", 0, 0) / "metrics.json").read_text(encoding="utf-8"))
    assert {"seconds_train", "seconds_to_best", "best_epoch", "seconds_per_epoch"} <= set(m["cost"])
    assert m["hardware"]["cpu_threads"] > 0 and m["uso"]["duracao_s"] > 0
    assert {"train_loss", "train_video_f1", "val_loss", "val_video_f1", "sec_train"} <= set(m["history"][0])
    v = E.verdicts(tmp_cfg, "resnet18", tasks=("A",), exps=["E03"])
    assert len(v) == 1 and v["verdict"].iloc[0] in {"melhora", "neutra", "piora"}
    g = E.cmd_gradcam(tmp_cfg, SimpleNamespace(exp=None))
    assert g["frames"] == 2 * 30                       # 2 experimentos × todos os frames de teste

"""Ponta a ponta em CPU sobre um BAGLS sintético (~30 frames): índice → splits → segmentação →
cache → treino (1 época) → clássico → resumo/vereditos → Grad-CAM. Alvo: < 5 min."""
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
    tmp_cfg["classic"]["mlp"]["early_stopping"] = False  # 10% de ~10 vídeos não estratifica
    for st in ("stage1", "stage2"):
        tmp_cfg[st].update({"experiments": ["E00", "E03"], "tasks": ["A"], "seeds": [0]})
    tmp_cfg["stage2"]["models"] = ["simplecnn"]
    write_fake_bagls(tmp_cfg, vids, frames_per_video=2)
    D.build_index(tmp_cfg); D.sample_frames(tmp_cfg); D.build_groups(tmp_cfg); D.make_splits(tmp_cfg)
    E.seg_study(tmp_cfg, n_jobs=1)
    for e in ("E00", "E03", "E08", "E13"):
        D.build_cache(tmp_cfg, e, n_jobs=1)
    for e in ("E00", "E03"):
        for k in range(5):
            r = T.train_job(tmp_cfg, e, "A", "simplecnn", k, 0, epochs=1)
            assert 0 <= r["test_video_f1"] <= 1
    args = SimpleNamespace(exp="C1,C4", jobs=1, force=False)
    out = T.cmd_classic(tmp_cfg, args)
    assert set(out["video_f1_mean"]) == {"C1/A", "C1/B", "C4/A", "C4/B"}
    v = E.verdicts(tmp_cfg, "simplecnn", tasks=("A",), exps=["E03"])
    assert len(v) == 1 and v["verdict"].iloc[0] in {"melhora", "neutra", "piora"}
    g = E.cmd_gradcam(tmp_cfg, SimpleNamespace(exp=None))
    assert g["frames"] == 2 * 30                       # 2 experimentos × todos os frames de teste

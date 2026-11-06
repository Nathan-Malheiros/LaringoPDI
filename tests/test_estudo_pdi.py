import numpy as np
import pandas as pd
from types import SimpleNamespace

from laringopdi import config as C
from laringopdi import data as D
from laringopdi.estudo_pdi import cmd_pdi_study, denoise_verdicts, seg_verdicts, visual_verdicts
from tests.conftest import write_fake_bagls


def _qv():
    rng = np.random.default_rng(0)
    rows = []
    for v in range(30):
        base = rng.normal(0.2, 0.01)
        ent = rng.normal(6, 0.1)
        sat = rng.uniform(0.1, 0.2)
        rows.append({"technique": "raw", "video_id": v, "rms_contrast": base, "entropy": ent, "saturated_frac": sat})
        rows.append({"technique": "bom", "video_id": v, "rms_contrast": base + rng.uniform(.02, .04), "entropy": ent + rng.uniform(.01, .1), "saturated_frac": sat})
        rows.append({"technique": "estoura", "video_id": v, "rms_contrast": base + rng.uniform(.02, .04), "entropy": ent, "saturated_frac": sat + 0.05})
        rows.append({"technique": "apaga", "video_id": v, "rms_contrast": base - rng.uniform(.02, .04), "entropy": ent - rng.uniform(.1, .2), "saturated_frac": sat})
    return pd.DataFrame(rows)


def test_visual_verdict_applies_the_three_criteria():
    v = visual_verdicts(_qv(), ("bom", "estoura", "apaga")).set_index("technique")
    assert v.loc["bom", "verdict"] == "melhora"
    assert v.loc["estoura", "verdict"] == "não melhora" and "satura" in v.loc["estoura", "motivo"]
    assert "contraste" in v.loc["apaga", "motivo"] and "entropia" in v.loc["apaga", "motivo"]


def test_seg_verdict_direction_and_neutral():
    rng = np.random.default_rng(0)
    raw = rng.uniform(0.3, 0.6, 40)
    idx = pd.MultiIndex.from_product([["otsu"], range(40)], names=["method", "video_id"])
    vd = pd.DataFrame({"raw": raw, "sobe": raw + rng.uniform(.05, .1, 40), "desce": raw - rng.uniform(.05, .1, 40),
                       "igual": raw + rng.normal(0, 1e-4, 40)}, index=idx)
    v = seg_verdicts(vd).set_index("technique")
    assert v.loc["sobe", "verdict"] == "melhora" and v.loc["desce", "verdict"] == "piora" and v.loc["igual", "verdict"] == "neutra"
    assert (v["p_holm"] >= v["p"]).all()


def test_denoise_verdict_requires_psnr_and_ssim():
    rows = []
    for i in range(30):
        rows += [{"noise": "sp_5", "frame_id": i, "filter": "none", "psnr": 20 + i * .01, "ssim": .5 + i * .001, "ms": 0},
                 {"noise": "sp_5", "frame_id": i, "filter": "bom", "psnr": 25 + i * .02, "ssim": .7 + i * .002, "ms": 1},
                 {"noise": "sp_5", "frame_id": i, "filter": "ruim", "psnr": 25 + i * .02, "ssim": .4 - i * .001, "ms": 1}]
    v = denoise_verdicts(pd.DataFrame(rows)).set_index("filter")
    assert v.loc["bom", "verdict"] == "reduz o ruído" and v.loc["ruim", "verdict"] == "não reduz"


def test_pdi_study_end_to_end_and_resumes(tmp_cfg):
    vids = {f"h{i}": "healthy" for i in range(5)}
    vids.update({f"s{i}": "Vocal fold nodules" for i in range(5)})
    tmp_cfg["data"]["frames_per_video"] = 2
    tmp_cfg["data"]["inner_val_frac"] = 0.34
    tmp_cfg["segmentation"].update({"preprocess": ["raw", "clahe", "median"], "methods": ["otsu_morph", "region_grow"]})
    write_fake_bagls(tmp_cfg, vids, frames_per_video=2)
    D.build_index(tmp_cfg); D.sample_frames(tmp_cfg); D.build_groups(tmp_cfg); D.make_splits(tmp_cfg)
    args = SimpleNamespace(jobs=1)
    cmd_pdi_study(tmp_cfg, args)
    out = C.path(tmp_cfg, "tables") / "pdi"
    t = pd.read_csv(out / "tabela_unica.csv")
    assert list(t["tecnica"]) == ["raw", "clahe", "median"]
    assert t.loc[0, "veredito_seg"] == "referência" and t.loc[0, "ms_tecnica"] == 0
    assert t.loc[2, "veredito_visual"] == "ver ruído" and t["psnr_sal_pimenta5"].notna()[2]
    assert t["dice"].between(0, 1).all() and (t["ms_total"] > 0).all()
    assert (C.path(tmp_cfg, "figures") / "pdi_segmentacao_custo.png").exists()
    r = C.path(tmp_cfg, "cache").parent / "pdi_study" / "q_0000.csv.gz"
    stamp = r.stat().st_mtime_ns
    cmd_pdi_study(tmp_cfg, args)
    assert r.stat().st_mtime_ns == stamp                  # blocos prontos não são refeitos

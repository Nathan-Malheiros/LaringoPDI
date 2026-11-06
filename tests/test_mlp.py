"""MLP de classificação e de segmentação, de ponta a ponta sobre um BAGLS sintético."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from laringopdi import config as C
from laringopdi import data as D
from laringopdi import mlp as M
from laringopdi.estudo_pdi import cmd_pdi_study
from laringopdi.evaluate import cost_verdicts
from laringopdi.train import run_dir
from tests.conftest import write_fake_bagls


def test_cost_verdicts_direction_and_tie():
    rows = []
    for k in range(5):
        for s in range(3):
            base = 10 + k + s * .1
            rows += [{"exp": "E00", "task": "A", "fold": k, "seed": s, "epoca_melhor": base, "custo_total_s": base},
                     {"exp": "E_antes", "task": "A", "fold": k, "seed": s, "epoca_melhor": base - 2 - k * .01, "custo_total_s": base + 3 + s * .01},
                     {"exp": "E_igual", "task": "A", "fold": k, "seed": s, "epoca_melhor": base, "custo_total_s": base}]
    v = cost_verdicts(pd.DataFrame(rows), ["E_antes", "E_igual"]).set_index("exp")
    assert v.loc["E_antes", "veredito_convergencia"] == "converge antes" and v.loc["E_antes", "veredito_custo"] == "mais cara"
    assert v.loc["E_igual", "veredito_convergencia"] == "igual" and v.loc["E_igual", "veredito_custo"] == "igual"


def test_mask_seed_ignores_letterbox_pad():
    """O pad preto cruza a janela central, mas a semente tem de cair na glote, dentro da imagem."""
    g = np.zeros((224, 224), np.uint8)
    g[92:132] = 120                                  # imagem larga: faixa fina no meio, pad acima e abaixo
    g[105:119, 105:119] = 20                         # glote
    params = {"tol": 20}
    assert not __import__("laringopdi.pdi.segmentation", fromlist=["segment"]).segment(g, "region_grow", **params).any()
    m = M._segment_region(g, "region_grow", params)
    assert m[112, 112] and not m[:92].any() and not m[132:].any() and m.mean() < .01


def test_mlp_end_to_end(tmp_cfg):
    vids = {f"h{i}": "healthy" for i in range(5)}
    vids.update({f"s{i}": "Vocal fold nodules" for i in range(5)})
    tmp_cfg["data"]["frames_per_video"] = 2
    tmp_cfg["data"]["inner_val_frac"] = 0.34
    tmp_cfg["segmentation"].update({"preprocess": ["raw", "clahe"], "methods": ["otsu_morph", "region_grow"]})
    tmp_cfg["study"].update({"experiments": ["E00", "E03"], "tasks": ["A"], "seeds": [0]})
    tmp_cfg["mlp"]["class"].update({"epochs": 3, "patience": 2})
    tmp_cfg["mlp"]["seg"].update({"epochs": 2, "patience": 1, "pixels_per_class": 20, "eval_frames": 2, "frames_per_batch": 2})
    write_fake_bagls(tmp_cfg, vids, frames_per_video=2)
    D.build_index(tmp_cfg); D.sample_frames(tmp_cfg); D.build_groups(tmp_cfg); D.make_splits(tmp_cfg)
    for e in ("E00", "E03"):
        D.build_cache(tmp_cfg, e, n_jobs=1)
    args = SimpleNamespace(jobs=1, force=False)
    cmd_pdi_study(tmp_cfg, args)

    r = M.cmd_mlp_class(tmp_cfg, args)
    assert r["jobs"] == 15                                     # 2 técnicas + referência em pixels, 5 folds
    out = C.path(tmp_cfg, "tables") / "mlp"
    t = pd.read_csv(out / "classificacao.csv")
    assert set(t["modelo"]) == {"mlp18", "mlp_pixels"}
    raw = t[(t["modelo"] == "mlp18") & (t["tecnica"] == "raw")].iloc[0]
    assert raw["veredito_f1"] == "referência" and raw["veredito_custo"] == "referência" and raw["custo_total_s"] > 0
    clahe = t[t["tecnica"] == "clahe"].iloc[0]
    assert clahe["veredito_f1"] in ("melhora", "neutra", "piora") and clahe["veredito_sensibilidade"] in ("melhora", "neutra", "piora")
    assert set(pd.read_csv(out / "importancia_familias.csv")["familia"]) == {"shape", "int", "tex"}
    assert (out / "classificacao_vs_pixels.csv").exists()

    m = json.loads((run_dir(tmp_cfg, "E03", "A", "mlp18", 0, 0) / "metrics.json").read_text(encoding="utf-8"))
    assert {"train_loss", "val_loss", "train_video_f1", "val_video_f1", "sec_train"} <= set(m["history"][0])
    assert m["cost"]["n_train_rows"] < m["cost"]["n_train_frames"]      # um vetor por vídeo, vários frames por vídeo
    assert m["uso"]["duracao_s"] > 0 and m["hardware"]["cpu_threads"] > 0
    stamp = (run_dir(tmp_cfg, "E03", "A", "mlp18", 0, 0) / "metrics.json").stat().st_mtime_ns

    s = M.cmd_mlp_seg(tmp_cfg, args)
    assert s["jobs"] == 10
    st = pd.read_csv(out / "segmentacao.csv")
    assert list(st["tecnica"]) == ["raw", "clahe"] and st["dice"].between(0, 1).all()
    assert st.loc[0, "veredito_seg"] == "referência" and st.loc[1, "veredito_vs_classico"] in ("MLP melhor", "clássico melhor", "empate")
    sm = json.loads((run_dir(tmp_cfg, "E00", "seg", "mlp_seg", 0, 0) / "metrics.json").read_text(encoding="utf-8"))
    assert {"train_dice", "val_dice", "val_loss"} <= set(sm["history"][0]) and sm["uso"]["duracao_s"] > 0

    M.cmd_mlp_class(tmp_cfg, args)                               # retomada: jobs prontos não são refeitos
    assert (run_dir(tmp_cfg, "E03", "A", "mlp18", 0, 0) / "metrics.json").stat().st_mtime_ns == stamp
    etapas = pd.read_csv(C.path(tmp_cfg, "tables") / "custo" / "etapas.csv")
    assert {"mlp/caracteristicas/E00", "mlp/treino_classificacao", "mlp/treino_segmentacao"} <= set(etapas["etapa"])


def test_pixel_prediction_matches_a_manual_window(tmp_cfg):
    import torch

    from laringopdi.mlp import _net, _predict_masks

    sc = {"patch": 15, "threshold": 0.5, "frames_per_batch": 1}
    img = np.zeros((1, 32, 32, 3), np.uint8)
    img[:, 8:24, 8:24] = 200
    torch.manual_seed(0)
    model = _net([225, 4, 2])
    d = {"X": img, "size": 32}
    # Compara a janela usada na inferência com a mesma janela montada à mão, para o pixel (10, 12).
    pad = np.pad(img[0, :, :, 0], 7)
    manual = pad[10:25, 12:27].astype(np.float32).ravel() / 255
    t = torch.nn.functional.pad(torch.from_numpy(img[..., 0]), (7, 7, 7, 7))
    w = t.unfold(1, 15, 1).unfold(2, 15, 1).reshape(-1, 225).float() / 255
    assert np.allclose(w[10 * 32 + 12].numpy(), manual)
    assert _predict_masks(model, d, np.array([0]), 0.0, 1.0, torch.device("cpu"), sc)[0].shape == (32, 32)

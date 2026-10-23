"""Métricas, estatística, Grad-CAM e estudo de segmentação clássica da glote (Fase 3)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from laringopdi import config as C
from laringopdi import data as D


# ----------------------------------------------------------------------------- segmentação
def _seg_frame(p, seg, pres, grid, work_size):
    import cv2

    from laringopdi.pdi.registry import DEFAULTS, TECHNIQUES, to_work_size
    from laringopdi.pdi.segmentation import seg_metrics, segment

    g = to_work_size(D.imread(p, gray=True), work_size)
    gt = cv2.resize((D.imread(seg, gray=True) > 127).astype(np.uint8), g.shape[::-1],
                    interpolation=cv2.INTER_NEAREST).astype(bool)
    rows = []
    for pre in pres:
        gp = g if pre == "raw" else TECHNIQUES[pre](g, {}, **DEFAULTS[pre])
        for method, plist in grid.items():
            for k, prm in enumerate(plist):
                rows.append({"preprocess": pre, "method": method, "param_id": k,
                             **seg_metrics(segment(gp, method, **prm), gt)})
    return rows


def seg_study(cfg: dict, n_jobs: int = -1, force: bool = False) -> dict:
    """Roda todos os métodos × pré-processamentos × grade de parâmetros em cada frame amostrado
    (uma vez) e, por fold, escolhe os parâmetros só pelo Dice médio no treino e reporta no teste."""
    from joblib import Parallel, delayed

    from laringopdi.pdi.segmentation import PARAM_GRID

    tables = C.path(cfg, "tables")
    long_p = tables / "segmentation_frames.csv.gz"
    sc = cfg["segmentation"]
    grid = {m: PARAM_GRID[m] for m in sc["methods"]}
    fr = D.load_frames(cfg)
    raw = C.path(cfg, "raw")
    if force or not long_p.exists():
        res = Parallel(n_jobs=n_jobs, batch_size=8)(
            delayed(_seg_frame)(raw / p, raw / s, sc["preprocess"], grid, cfg["data"]["work_size"])
            for p, s in zip(fr["path"], fr["seg_path"]))
        long = pd.concat([pd.DataFrame(r).assign(path=p) for r, p in zip(res, fr["path"])], ignore_index=True)
        long.to_csv(long_p, index=False)
    long = pd.read_csv(long_p).merge(fr[["path", "video_id"]], on="path")
    rows, chosen = [], {}
    for f in D.load_splits(cfg):
        tr_v = set(f["train"]) | set(f["val"])
        te_v = set(f["test"])
        tr = long[long["video_id"].isin(tr_v)]
        te = long[long["video_id"].isin(te_v)]
        best = tr.groupby(["preprocess", "method", "param_id"])["dice"].mean().reset_index() \
                 .sort_values("dice", ascending=False).drop_duplicates(["preprocess", "method"])
        for _, b in best.iterrows():
            t = te[(te["preprocess"] == b["preprocess"]) & (te["method"] == b["method"]) & (te["param_id"] == b["param_id"])]
            rows.append({"fold": f["fold"], "preprocess": b["preprocess"], "method": b["method"],
                         "param_id": int(b["param_id"]), "params": json.dumps(grid[b["method"]][int(b["param_id"])]),
                         "train_dice": b["dice"], "dice": t["dice"].mean(), "iou": t["iou"].mean(),
                         "hd95_median": t["hd95"].median(), "fail_rate": t["fail"].mean(), "n_test_frames": len(t)})
        if f["fold"] == sc["tune_fold"]:
            top = best.iloc[0]
            chosen = {"method": top["method"], "preprocess": top["preprocess"],
                      "params": grid[top["method"]][int(top["param_id"])], "train_dice": float(top["dice"]),
                      "selected_on": f"treino+val do fold {f['fold']} (só máscaras de glote)", "git_sha": C.git_sha()}
    per_fold = pd.DataFrame(rows)
    per_fold.to_csv(tables / "segmentation.csv", index=False)
    summ = per_fold.groupby(["method", "preprocess"]).agg(
        dice_mean=("dice", "mean"), dice_sd=("dice", "std"), iou_mean=("iou", "mean"),
        hd95_median=("hd95_median", "median"), fail_rate=("fail_rate", "mean")).reset_index() \
        .sort_values("dice_mean", ascending=False)
    summ.round(4).to_csv(tables / "segmentation_summary.csv", index=False)
    C.write_json(tables / "segmentation_best.json", chosen)
    _seg_figure(cfg, summ, fr, grid, per_fold)
    return {"best": chosen, "top3": summ.head(3)[["method", "preprocess", "dice_mean", "fail_rate"]]
            .round(3).to_dict("records")}


def _seg_figure(cfg, summ, fr, grid, per_fold, n=4):
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from laringopdi.pdi.registry import DEFAULTS, TECHNIQUES, to_work_size
    from laringopdi.pdi.segmentation import dice, segment

    raw = C.path(cfg, "raw")
    test0 = fr[fr["video_id"].isin(set(D.load_splits(cfg)[0]["test"]))]
    picks = test0.sample(frac=1, random_state=0).groupby("family").head(2).head(n)
    top = summ.head(3).to_dict("records")
    fig, axes = plt.subplots(len(picks), 2 + len(top), figsize=(2.6 * (2 + len(top)), 2.4 * len(picks)), squeeze=False)
    for i, (_, r) in enumerate(picks.iterrows()):
        g = to_work_size(D.imread(raw / r["path"], gray=True), cfg["data"]["work_size"])
        gt = cv2.resize((D.imread(raw / r["seg_path"], gray=True) > 127).astype(np.uint8), g.shape[::-1],
                        interpolation=cv2.INTER_NEAREST).astype(bool)
        cols = [(g, f"original ({r['family']})"), (gt, "ground truth")]
        for t in top:
            pf = per_fold[(per_fold["fold"] == 0) & (per_fold["method"] == t["method"]) &
                          (per_fold["preprocess"] == t["preprocess"])].iloc[0]
            gp = g if t["preprocess"] == "raw" else TECHNIQUES[t["preprocess"]](g, {}, **DEFAULTS[t["preprocess"]])
            m = segment(gp, t["method"], **json.loads(pf["params"]))
            vis = np.dstack([g, g, g]).copy()
            vis[m] = (0.5 * vis[m] + [0, 127, 0]).astype(np.uint8)
            cols.append((vis, f"{t['method']}/{t['preprocess']} Dice={dice(m, gt):.2f}"))
        for a, (im, title) in zip(axes[i], cols):
            a.imshow(im, cmap="gray"); a.set_title(title, fontsize=7); a.axis("off")
    fig.tight_layout()
    fig.savefig(C.path(cfg, "figures") / "segmentation_examples.png", dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------- métricas de classificação
def compute_metrics(y: np.ndarray, prob: np.ndarray) -> dict:
    """Acurácia balanceada, F1-macro, sensibilidade, especificidade, AUC e matriz de confusão.
    Binário: sensibilidade/especificidade da classe 1. Multiclasse: médias macro (one-vs-rest)."""
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

    y = np.asarray(y).astype(int)
    prob = np.asarray(prob, dtype=float)
    k = prob.shape[1]
    pred = prob.argmax(1)
    cm = confusion_matrix(y, pred, labels=list(range(k)))
    tp = np.diag(cm).astype(float)
    sens = tp / np.maximum(cm.sum(1), 1)
    spec = np.array([(cm.sum() - cm[i].sum() - cm[:, i].sum() + cm[i, i]) / max(cm.sum() - cm[i].sum(), 1)
                     for i in range(k)])
    try:
        auc = roc_auc_score(y, prob[:, 1]) if k == 2 else roc_auc_score(y, prob, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")
    return {"n": int(len(y)), "bal_acc": float(balanced_accuracy_score(y, pred)),
            "f1_macro": float(f1_score(y, pred, average="macro", labels=list(range(k)), zero_division=0)),
            "sensitivity": float(sens[1] if k == 2 else sens.mean()),
            "specificity": float(spec[1] if k == 2 else spec.mean()),
            "auc": float(auc), "confusion": cm.tolist()}


def video_level(df: pd.DataFrame, prob_cols: list[str]) -> pd.DataFrame:
    """Agrega por vídeo: média das probabilidades dos frames; rótulo do vídeo."""
    return df.groupby("video_id").agg({**{c: "mean" for c in prob_cols}, "y": "first"}).reset_index()


def metrics_both_levels(preds: pd.DataFrame) -> dict:
    pc = sorted((c for c in preds.columns if c[0] == "p" and c[1:].isdigit()), key=lambda c: int(c[1:]))
    v = video_level(preds, pc)
    return {"frame": compute_metrics(preds["y"], preds[pc].to_numpy()),
            "video": compute_metrics(v["y"], v[pc].to_numpy())}


# ----------------------------------------------------------------------------- resumo dos runs
def collect_runs(cfg: dict) -> pd.DataFrame:
    """Lê todos os results/runs/<exp>/<task>/<model>/fold{k}_seed{s}/metrics.json."""
    import json as _json
    rows = []
    for p in C.path(cfg, "runs").glob("*/*/*/fold*_seed*/metrics.json"):
        m = _json.loads(p.read_text(encoding="utf-8"))
        exp, task, model, run = p.parts[-5:-1]
        fold, seed = run.replace("fold", "").split("_seed")
        base = {"exp": exp, "task": task, "model": model, "fold": int(fold), "seed": int(seed),
                "seconds": m.get("seconds"), "epochs_run": m.get("epochs_run")}
        for lvl in ("frame", "video", "val_frame", "val_video"):
            for k, v in m.get(lvl, {}).items():
                if k not in ("confusion", "n"):
                    base[f"{lvl}_{k}"] = v
        rows.append(base)
    return pd.DataFrame(rows)


def cmd_summary(cfg: dict, args) -> dict:
    """results/summary.csv (todos os runs) + results/tables/stage1_ranking.csv."""
    df = collect_runs(cfg)
    if df.empty:
        return {"runs": 0}
    df.sort_values(["exp", "task", "model", "fold", "seed"]).to_csv(C.path(cfg, "results") / "summary.csv", index=False)
    s1 = df[df["exp"].isin(cfg["stage1"]["experiments"]) & df["model"].isin(cfg["stage1"]["models"])]
    out = {"runs": len(df)}
    if not s1.empty:
        # A ESCOLHA usa só a validação interna (val_video_f1); o teste aparece só para registro.
        rk = s1.groupby(["exp", "task", "model"]).agg(val_f1=("val_video_f1_macro", "mean"), val_f1_sd=("val_video_f1_macro", "std"),
                                                       test_f1=("video_f1_macro", "mean"), n=("fold", "count")).reset_index()
        base = rk[rk["exp"] == "E00"].set_index(["task", "model"])["val_f1"]
        rk["delta_vs_E00"] = rk["val_f1"] - rk.set_index(["task", "model"]).index.map(base).to_numpy()
        # ranking: média do Δ de validação vs E00 entre tarefas e modelos (E00 e o limite superior E07 fora da escolha)
        score = rk.groupby("exp")["delta_vs_E00"].mean().rename("mean_delta").reset_index()
        rk = rk.merge(score, on="exp").sort_values(["mean_delta", "exp", "task", "model"], ascending=[False, True, True, True])
        rk.round(4).to_csv(C.path(cfg, "tables") / "stage1_ranking.csv", index=False)
        top = [e for e in score.sort_values("mean_delta", ascending=False)["exp"] if e not in ("E00", "E07")]
        out["top4"] = top[:cfg["stage1"]["top_k"]]
    return out

"""MLP na classificação e na segmentação, com e sem cada técnica de PDI (Etapas 3 e 4 do estudo).

Classificação (`mlp-class`, CPU, 1 núcleo):
- `mlp18`: técnica → crescimento de região → 18 características → média por vídeo → 18–32–16–2;
- `mlp_pixels`: referência sem nenhum PDI: imagem reduzida a 32×32 → 1024–256–64–2, por frame.

Segmentação (`mlp-seg`, GPU): cada pixel vira um vetor com a vizinhança 15×15 (225 valores) → 225–64–32–2;
depois do limiar, fica o componente central, como nos métodos clássicos. A GPU é necessária porque a
inferência passa por todos os pixels de todos os frames.

Cada job grava o mesmo que a CNN: métricas no teste, curvas por época, custo, uso da máquina e hardware.
Regras em docs/DECISIONS.md (2026-11-06).
"""
from __future__ import annotations

import json
import sys
import time

import cv2
import numpy as np
import pandas as pd

from laringopdi import config as C
from laringopdi import data as D
from laringopdi.custo import Medidor, medir
from laringopdi.evaluate import _holm, compute_metrics, cost_verdicts, metrics_both_levels, run_costs, verdicts
from laringopdi.pausa import esperar_se_pausado
from laringopdi.features import FAMILIES_18, _letterbox_region, _unpack, features18, video_features18
from laringopdi.train import _conf, _preds_df, _save, _weighted_loss, frame_labels, run_dir, set_seed

_memo: dict = {}


def _net(sizes: list[int]):
    from torch import nn

    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    return nn.Sequential(*layers[:-1])


def _study(cfg: dict) -> list[str]:
    return cfg["study"]["experiments"]


def _tech(cfg: dict, exp: str) -> str:
    return cfg["experiments"][exp]["pipeline"]


def _pipeline_meta(cfg: dict, exp: str) -> dict:
    return {"pipeline": _tech(cfg, exp), "hash": D.cache_paths(cfg, exp)["hash"]}


# ----------------------------------------------------------------------------- 18 características por técnica
def _feature_paths(cfg: dict, exp: str) -> tuple:
    d = C.path(cfg, "cache").parent / "features18"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{exp}.csv.gz", d / f"{exp}.tempo.json"


def _segment_region(g: np.ndarray, method: str, params: dict) -> np.ndarray:
    """Segmenta só a área da imagem, como no estudo de segmentação. O pad preto não pode receber a semente."""
    from laringopdi.pdi.segmentation import segment

    m = np.zeros(g.shape, bool)
    ys, xs = np.nonzero(_letterbox_region(g))
    if ys.size:
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        m[y0:y1, x0:x1] = segment(g[y0:y1, x0:x1], method, **params).astype(bool)
    return m


def _feature_frame(g: np.ndarray, method: str, params: dict) -> dict:
    cv2.setNumThreads(1)
    m = _segment_region(g, method, params)
    return {**features18(g, m), "mask_empty": float(not m.any())}


def build_features(cfg: dict, exp: str, n_jobs: int, force: bool = False) -> pd.DataFrame:
    """Máscara e 18 características de todos os frames, sobre a imagem com a técnica. Só o método é fixo."""
    from joblib import Parallel, delayed
    from tqdm import tqdm

    from laringopdi.pdi.segmentation import best_config

    out, tempo = _feature_paths(cfg, exp)
    if out.exists() and tempo.exists() and not force:
        return pd.read_csv(out, dtype={"video_id": str})
    best = best_config()
    X, paths, meta = D.load_cache(cfg, exp)
    n = len(paths)
    rows = []
    with medir(cfg, f"mlp/caracteristicas/{exp}", f"Máscara e 18 características com {_tech(cfg, exp)}", n_itens=n):
        for s in tqdm(range(0, n, 256), desc=f"Características {_tech(cfg, exp)}", unit="bloco", file=sys.stdout, leave=False):
            frames = [np.array(X[i, :, :, 0]) for i in range(s, min(s + 256, n))]
            rows += Parallel(n_jobs=n_jobs, batch_size=8)(delayed(_feature_frame)(g, best["method"], best["params"]) for g in frames)
    df = pd.DataFrame(rows)
    df.insert(0, "video_id", [str(v) for v in meta["video_ids"]])
    df.insert(0, "path", paths)
    df.to_csv(out, index=False)
    measure_feature_time(cfg, exp)
    return df


def measure_feature_time(cfg: dict, exp: str) -> float:
    """Tempo em 1 núcleo, nas mesmas condições das técnicas. O aquecimento evita contar a carga das bibliotecas."""
    from laringopdi.pdi.segmentation import best_config

    best = best_config()
    X, paths, _ = D.load_cache(cfg, exp)
    idx = np.random.default_rng(0).choice(len(paths), min(100, len(paths)), replace=False)
    frames = [np.array(X[i, :, :, 0]) for i in idx]
    for g in frames[:5]:
        _feature_frame(g, best["method"], best["params"])
    t0 = time.perf_counter()
    for g in frames:
        _feature_frame(g, best["method"], best["params"])
    ms = (time.perf_counter() - t0) * 1000 / len(frames)
    C.write_json(_feature_paths(cfg, exp)[1], {"ms_por_frame": ms, "n": len(frames), "hardware": C.hardware()})
    return ms


def _feature_ms(cfg: dict, exp: str) -> float:
    return json.loads(_feature_paths(cfg, exp)[1].read_text(encoding="utf-8"))["ms_por_frame"]


def _pdi_ms(cfg: dict) -> dict:
    """Milissegundos por frame de cada técnica, medidos no estudo do PDI (1 núcleo)."""
    t = pd.read_csv(C.path(cfg, "tables") / "pdi" / "tabela_unica.csv").set_index("tecnica")["ms_tecnica"]
    return {e: float(t[_tech(cfg, e)]) for e in _study(cfg)}


# ----------------------------------------------------------------------------- classificação
def _class_rows(cfg: dict, exp: str, model: str) -> dict:
    """Matriz de entrada: um vetor por vídeo (`mlp18`) ou por frame (`mlp_pixels`)."""
    key = (str(C.path(cfg, "cache")), exp, model)
    if key in _memo:
        return _memo[key]
    if model == "mlp18":
        v = video_features18(pd.read_csv(_feature_paths(cfg, exp)[0], dtype={"video_id": str}))
        vid = v.index.to_numpy(dtype=str)
        out = {"X": v.to_numpy(np.float32), "video": vid, "paths": vid, "columns": list(v.columns)}
    else:
        X, paths, meta = D.load_cache(cfg, exp)
        small = np.stack([cv2.resize(np.array(X[i, :, :, 0]), (32, 32), interpolation=cv2.INTER_AREA).ravel() for i in range(len(paths))])
        out = {"X": small.astype(np.float32) / 255, "video": np.array([str(v) for v in meta["video_ids"]]), "paths": np.array(paths), "columns": None}
    _memo[key] = out
    return out


def _permutation_importance(prob_fn, X: np.ndarray, y: np.ndarray, columns: list[str], repeats: int, seed: int) -> dict:
    """Queda da acurácia balanceada ao embaralhar, entre os vídeos, as colunas de uma família inteira."""
    rng = np.random.default_rng(seed)
    base = compute_metrics(y, prob_fn(X))["bal_acc"]
    out = {}
    for fam in FAMILIES_18:
        cols = [i for i, c in enumerate(columns) if c.startswith(fam)]
        drops = []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[:, cols] = X[rng.permutation(len(X))][:, cols]
            drops.append(base - compute_metrics(y, prob_fn(Xp))["bal_acc"])
        out[fam.replace("f18", "").rstrip("_")] = {"mean": float(np.mean(drops)), "sd": float(np.std(drops))}
    return out


def class_job(cfg: dict, exp: str, task: str, model_name: str, fold: int, seed: int, force: bool = False) -> dict:
    """Treina uma MLP de classificação na CPU, com o early stopping na validação interna do fold."""
    import torch

    rd = run_dir(cfg, exp, task, model_name, fold, seed)
    if (rd / "metrics.json").exists() and not force:
        return {"skipped": True}
    mc = cfg["mlp"]["class"]
    t0 = time.perf_counter()
    uso = Medidor().start()
    set_seed(seed)
    torch.set_num_threads(mc["threads"])
    d = _class_rows(cfg, exp, model_name)
    lab = frame_labels(cfg)
    labels = lab.assign(video_id=lab["video_id"].astype(str)).groupby("video_id")[D.task_col(task)].first()
    y = labels.loc[d["video"]].to_numpy()
    sp = D.load_splits(cfg)[fold]
    part = {p: np.nonzero(np.isin(d["video"], [str(v) for v in sp[p]]) & (y >= 0))[0] for p in ("train", "val", "test")}
    itr, iva, ite = part["train"], part["val"], part["test"]
    mean, std = d["X"][itr].mean(0), d["X"][itr].std(0)              # z-score só com o treino
    std[std == 0] = 1.0
    Z = torch.from_numpy((d["X"] - mean) / std)
    n_classes = len(D.task_names(cfg, task))
    hidden = mc["hidden18"] if model_name == "mlp18" else mc["hidden_pixels"]
    sizes = [Z.shape[1], *hidden, n_classes]
    model = _net(sizes)
    w = len(itr) / (n_classes * np.maximum(np.bincount(y[itr], minlength=n_classes), 1))
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32))
    opt = torch.optim.AdamW(model.parameters(), lr=mc["lr"], weight_decay=mc["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=mc["epochs"])
    ytr = torch.from_numpy(y[itr].astype(np.int64))
    Ztr = Z[itr]
    gen = torch.Generator().manual_seed(seed)

    def prob_of(tensor):
        model.eval()
        with torch.no_grad():
            return torch.softmax(model(tensor), 1).numpy()

    def video_metrics(idx, prob):
        return metrics_both_levels(_preds_df(d["paths"][idx], d["video"][idx], y[idx], prob))["video"]

    best, best_ep, best_state, wait, hist, train_sec, sec_to_best = -1.0, -1, None, 0, [], 0.0, 0.0
    for ep in range(mc["epochs"]):
        te = time.perf_counter()
        model.train()
        losses = []
        for b in torch.randperm(len(itr), generator=gen).split(mc["batch_size"]):
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(Ztr[b]), ytr[b])
            loss.backward()
            opt.step()
            losses.append(loss.item())
        cur_lr = opt.param_groups[0]["lr"]
        sched.step()
        sec_train = time.perf_counter() - te
        train_sec += sec_train
        tv = time.perf_counter()
        pv = prob_of(Z[iva])
        vm = video_metrics(iva, pv)
        sec_val = time.perf_counter() - tv
        pt_ = prob_of(Ztr)
        hist.append({"epoch": ep, "lr": cur_lr, "train_loss_aug": float(np.mean(losses)), "train_loss": _weighted_loss(pt_, y[itr], w),
                     "train_video_f1": video_metrics(itr, pt_)["f1_macro"], "val_loss": _weighted_loss(pv, y[iva], w),
                     "val_video_f1": vm["f1_macro"], "val_video_sensitivity": vm["sensitivity"], "sec_train": sec_train, "sec_val": sec_val})
        if vm["f1_macro"] > best:
            best, best_ep, wait, sec_to_best = vm["f1_macro"], ep, 0, train_sec
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= mc["patience"]:
                break
    model.load_state_dict(best_state)
    ti = time.perf_counter()
    pt = prob_of(Z[ite])
    infer_ms = (time.perf_counter() - ti) * 1000 / max(len(ite), 1)
    preds = _preds_df(d["paths"][ite], d["video"][ite], y[ite], pt)
    vm = metrics_both_levels(_preds_df(d["paths"][iva], d["video"][iva], y[iva], prob_of(Z[iva])))
    n_frames = int(lab["video_id"].astype(str).isin(set(d["video"][itr])).sum())
    cost = {"seconds_total": time.perf_counter() - t0, "seconds_train": train_sec, "seconds_to_best": sec_to_best, "epochs_run": len(hist),
            "best_epoch": best_ep, "seconds_per_epoch": train_sec / len(hist), "infer_ms_per_frame": infer_ms,
            "n_train_frames": n_frames, "n_train_rows": int(len(itr)), "n_params": int(sum(p.numel() for p in model.parameters())), "gpu_peak_mb": None}
    m = {**metrics_both_levels(preds), "val_frame": vm["frame"], "val_video": vm["video"], "epochs_run": len(hist), "best_val_video_f1": best,
         "seconds": cost["seconds_total"], "device": "cpu", "cost": cost, "history": hist}
    if model_name == "mlp18":
        m["perm_importance"] = _permutation_importance(lambda a: prob_of(torch.from_numpy(a)), Z[ite].numpy(), y[ite], d["columns"],
                                                       cfg["classic"]["permutation_repeats"], seed)
    m |= {"uso": uso.stop(), "hardware": C.hardware()}
    if seed == 0:
        rd.mkdir(parents=True, exist_ok=True)
        torch.save({"state": best_state, "sizes": sizes, "mean": mean, "std": std, "columns": d["columns"]}, rd / "model.pt")
    _save(rd, m, preds, _conf(cfg, exp, task, model_name, fold, seed, _pipeline_meta(cfg, exp), {"train": mc, "sizes": sizes}))
    return {"test_video_f1": m["video"]["f1_macro"]}


def class_jobs(cfg: dict) -> list[dict]:
    """A semente é o laço externo. A referência em pixels é uma só, pois não depende de técnica."""
    st = cfg["study"]
    jobs = []
    for s in st["seeds"]:
        for e in st["experiments"]:
            jobs += [{"exp": e, "task": t, "model": "mlp18", "fold": k, "seed": s} for t in st["tasks"] for k in st["folds"]]
        jobs += [{"exp": "E00", "task": t, "model": "mlp_pixels", "fold": k, "seed": s} for t in st["tasks"] for k in st["folds"]]
    return jobs


def _mean_metrics(cfg: dict, model: str, exp: str, task: str) -> dict:
    ms = []
    for p in (C.path(cfg, "runs") / exp / task / model).glob("fold*_seed*/metrics.json"):
        ms.append(json.loads(p.read_text(encoding="utf-8"))["video"])
    f = pd.DataFrame(ms)
    return {"f1": f["f1_macro"].mean(), "f1_dp": f["f1_macro"].std(), "sensibilidade": f["sensitivity"].mean(), "especificidade": f["specificity"].mean(),
            "auc": f["auc"].mean(), "acuracia_balanceada": f["bal_acc"].mean(), "n_jobs": len(f)}


def _class_tables(cfg: dict, out) -> pd.DataFrame:
    tasks, exps = cfg["study"]["tasks"], _study(cfg)
    others = [e for e in exps if e != "E00"]
    ms_pdi = _pdi_ms(cfg)
    ms_total = {e: ms_pdi[e] + _feature_ms(cfg, e) for e in exps}
    costs = run_costs(cfg, "mlp18", ms_total)
    conv = cost_verdicts(costs, others)
    vf, vs = verdicts(cfg, "mlp18", tasks, exps=others), verdicts(cfg, "mlp18", tasks, exps=others, metric="sensitivity")
    vs_pixels = pd.concat([verdicts(cfg, "mlp18", tasks, exps=exps, metric=m, baseline_model="mlp_pixels").assign(metrica=m) for m in ("f1", "sensitivity")])
    vs_pixels.round(5).to_csv(out / "classificacao_vs_pixels.csv", index=False)
    feats = {e: pd.read_csv(_feature_paths(cfg, e)[0], usecols=["mask_empty"])["mask_empty"].mean() for e in exps}
    rows = []
    for t in tasks:
        for e in exps:
            r = {"modelo": "mlp18", "tarefa": t, "tecnica": _tech(cfg, e), **_mean_metrics(cfg, "mlp18", e, t),
                 "mascara_vazia": feats[e], "ms_tecnica": ms_pdi[e], "ms_mascara_e_caracteristicas": _feature_ms(cfg, e)}
            c = costs[(costs["exp"] == e) & (costs["task"] == t)]
            r |= {k: c[k].mean() for k in ("epoca_melhor", "epocas", "s_ate_melhor", "s_por_epoca", "sobreajuste", "custo_total_s")}
            for name, v in (("f1", vf), ("sensibilidade", vs)):
                key = "f1" if name == "f1" else "sensitivity"
                s = v[(v["exp"] == e) & (v["task"] == t)]
                if len(s):
                    s = s.iloc[0]
                    r |= {f"delta_{name}": s[f"delta_{key}"], f"ic_{name}_inf": s["ci_low"], f"ic_{name}_sup": s["ci_high"],
                          f"mcnemar_holm_{name}": s["mcnemar_p_holm"], f"wilcoxon_holm_{name}": s["wilcoxon_p_holm"], f"veredito_{name}": s["verdict"]}
                else:
                    r[f"veredito_{name}"] = "referência"
            s = conv[(conv["exp"] == e) & (conv["task"] == t)]
            r |= {k: (s.iloc[0][k] if len(s) else "referência") for k in ("veredito_convergencia", "veredito_custo")}
            rows.append(r)
        c = run_costs(cfg, "mlp_pixels", {"E00": 0.0})
        c = c[c["task"] == t]
        rows.append({"modelo": "mlp_pixels", "tarefa": t, "tecnica": "sem PDI (32×32)", **_mean_metrics(cfg, "mlp_pixels", "E00", t),
                     **{k: c[k].mean() for k in ("epoca_melhor", "epocas", "s_ate_melhor", "s_por_epoca", "sobreajuste", "custo_total_s")}, "ms_tecnica": 0.0})
    table = pd.DataFrame(rows)
    table.round(5).to_csv(out / "classificacao.csv", index=False)
    imp = []
    for t in tasks:
        vals = {}
        for k in cfg["study"]["folds"]:
            m = json.loads((run_dir(cfg, "E00", t, "mlp18", k, 0) / "metrics.json").read_text(encoding="utf-8"))
            for fam, v in m["perm_importance"].items():
                vals.setdefault(fam, []).append(v["mean"])
        imp += [{"tarefa": t, "familia": f, "queda_media": float(np.mean(v)), "dp_folds": float(np.std(v))} for f, v in vals.items()]
    pd.DataFrame(imp).round(5).to_csv(out / "importancia_familias.csv", index=False)
    return table


def _history(cfg: dict, exp: str, task: str, model: str, col: str) -> pd.Series:
    rows = []
    for p in (C.path(cfg, "runs") / exp / task / model).glob("fold*_seed*/metrics.json"):
        rows += [{"epoch": h["epoch"], "v": h[col], "run": p.parent.name} for h in json.loads(p.read_text(encoding="utf-8"))["history"]]
    d = pd.DataFrame(rows)
    n = d["run"].nunique()
    g = d.groupby("epoch")["v"].agg(["mean", "count"])
    return g[g["count"] >= n / 2]["mean"]            # só épocas que ao menos metade dos treinos alcançou


def _class_figure(cfg: dict, table: pd.DataFrame) -> None:
    from laringopdi.etapa1 import _plt

    plt = _plt()
    tasks = cfg["study"]["tasks"]
    fig, axes = plt.subplots(1, len(tasks), figsize=(11, 4.2), sharex=True, squeeze=False)
    for ax, t in zip(axes[0], tasks):
        s = table[(table["tarefa"] == t)].iloc[::-1]
        y = np.arange(len(s))
        ax.barh(y + .19, s["f1"], height=.34, color="#2a78d6", label="F1-macro")
        ax.barh(y - .19, s["sensibilidade"], height=.34, color="#eb6834", label="Sensibilidade")
        ax.set_yticks(y, s["tecnica"])
        ax.grid(axis="x", color="#C9CFDD", lw=.5, alpha=.6)
        ax.set_axisbelow(True)
        ax.set_xlim(0, 1)
        ax.set_title(f"Tarefa {t}: MLP por técnica (teste, por vídeo)")
        ax.tick_params(length=0)
    axes[0][0].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(C.path(cfg, "figures") / "mlp_classificacao.png")
    plt.close(fig)
    if {"E00", "E03"} <= set(_study(cfg)):
        fig, axes = plt.subplots(1, len(tasks), figsize=(11, 3.6), squeeze=False)
        for ax, t in zip(axes[0], tasks):
            for e, color in (("E00", "#7D8BB8"), ("E03", "#2a78d6")):
                name = "sem realce" if e == "E00" else "CLAHE"
                ax.plot(_history(cfg, e, t, "mlp18", "train_video_f1"), color=color, lw=2, ls="--", label=f"{name}, treino")
                ax.plot(_history(cfg, e, t, "mlp18", "val_video_f1"), color=color, lw=2, label=f"{name}, validação")
            ax.set_xlabel("Época")
            ax.set_ylabel("F1-macro por vídeo")
            ax.set_title(f"Tarefa {t}: curva de treino da MLP")
            ax.grid(color="#C9CFDD", lw=.5, alpha=.6)
            ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(C.path(cfg, "figures") / "mlp_curvas.png")
        plt.close(fig)


def cmd_mlp_class(cfg: dict, args) -> dict:
    from tqdm import tqdm

    out = C.path(cfg, "tables") / "mlp"
    out.mkdir(parents=True, exist_ok=True)
    print("[1/3] Máscara e 18 características por técnica", flush=True)
    for e in tqdm(_study(cfg), desc="Técnicas", unit="técnica", file=sys.stdout):
        build_features(cfg, e, args.jobs, args.force)
    jobs = class_jobs(cfg)
    if (out / "classificacao.csv").exists() and not args.force and all((run_dir(cfg, j["exp"], j["task"], j["model"], j["fold"], j["seed"]) / "metrics.json").exists() for j in jobs):
        return {"jobs": len(jobs), "situacao": "já concluído"}
    print(f"[2/3] Treino: {len(jobs)} jobs na CPU", flush=True)
    with medir(cfg, "mlp/treino_classificacao", "Todas as MLPs de classificação", n_itens=len(jobs)):
        bar = tqdm(jobs, desc="Treino das MLPs", unit="job", file=sys.stdout)
        for j in bar:
            esperar_se_pausado(cfg)
            bar.set_postfix_str(f"{j['model']} {j['exp']} tarefa {j['task']} fold {j['fold']} semente {j['seed']}")
            class_job(cfg, j["exp"], j["task"], j["model"], j["fold"], j["seed"], args.force)
    print("[3/3] Vereditos e tabelas", flush=True)
    with medir(cfg, "mlp/estatistica_classificacao", "Vereditos, custo e importância"):
        table = _class_tables(cfg, out)
        _class_figure(cfg, table)
    cols = ["modelo", "tarefa", "tecnica", "f1", "sensibilidade", "veredito_f1", "veredito_sensibilidade", "veredito_custo"]
    return {"jobs": len(jobs), "tabela": table[cols].round(3).to_dict("records")}


# ----------------------------------------------------------------------------- segmentação pixel a pixel
def _load_seg(cfg: dict, exp: str) -> dict:
    X, paths, meta = D.load_cache(cfg, exp)
    packed = np.load(D.cache_paths(cfg, exp)["masks"])["gt_mask_lb"]
    return {"X": X, "paths": np.array(paths), "video": np.array([str(v) for v in meta["video_ids"]]), "packed": packed, "size": X.shape[1]}


def _sample_patches(d: dict, idx: np.ndarray, patch: int, per_class: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """Vizinhanças balanceadas: até `per_class` pixels de glote e `per_class` de fora dela, dentro da imagem (sem o pad)."""
    r = patch // 2
    xs, ys = [], []
    for i in idx:
        g = np.array(d["X"][i, :, :, 0])
        m = _unpack(d["packed"][[i]], d["size"])[0]
        region = _letterbox_region(g)
        windows = np.lib.stride_tricks.sliding_window_view(cv2.copyMakeBorder(g, r, r, r, r, cv2.BORDER_CONSTANT, value=0), (patch, patch))
        for cls, coords in ((1, np.argwhere(m & region)), (0, np.argwhere(~m & region))):
            if len(coords):
                pick = coords[rng.choice(len(coords), min(per_class, len(coords)), replace=False)]
                xs.append(windows[pick[:, 0], pick[:, 1]].reshape(len(pick), -1))
                ys.append(np.full(len(pick), cls, np.int64))
    return np.concatenate(xs), np.concatenate(ys)


def _predict_masks(model, d: dict, idx: np.ndarray, mean: float, std: float, device, sc: dict) -> list[np.ndarray]:
    """Passa todos os pixels pela MLP. Cada lote vira (pixels, 225) na GPU; depois vêm o limiar e o componente central."""
    import torch
    import torch.nn.functional as F

    from laringopdi.pdi.segmentation import pick_central

    r = sc["patch"] // 2
    out = []
    model.eval()
    for s in range(0, len(idx), sc["frames_per_batch"]):
        frames = np.stack([np.array(d["X"][i, :, :, 0]) for i in idx[s:s + sc["frames_per_batch"]]])
        t = F.pad(torch.from_numpy(frames).to(device), (r, r, r, r))
        w = t.unfold(1, sc["patch"], 1).unfold(2, sc["patch"], 1).reshape(-1, sc["patch"] ** 2).float().div_(255).sub_(mean).div_(std)
        with torch.no_grad(), torch.autocast(device.type, enabled=device.type == "cuda"):
            prob = torch.softmax(model(w), 1)[:, 1].float().reshape(frames.shape).cpu().numpy()
        for g, p in zip(frames, prob):
            out.append(pick_central((p > sc["threshold"]) & _letterbox_region(g)))
    return out


def _mean_dice(pred: list[np.ndarray], d: dict, idx: np.ndarray) -> tuple[float, float]:
    from laringopdi.pdi.segmentation import dice, iou

    gt = _unpack(d["packed"][idx], d["size"])
    return float(np.mean([dice(p, g) for p, g in zip(pred, gt)])), float(np.mean([iou(p, g) for p, g in zip(pred, gt)]))


def seg_job(cfg: dict, exp: str, fold: int, seed: int, force: bool = False) -> dict:
    """Treina a MLP de pixels de uma técnica e segmenta todos os frames de teste do fold."""
    import torch

    from laringopdi.pdi.segmentation import seg_metrics

    rd = run_dir(cfg, exp, "seg", "mlp_seg", fold, seed)
    if (rd / "metrics.json").exists() and not force:
        return {"skipped": True}
    sc = cfg["mlp"]["seg"]
    t0 = time.perf_counter()
    uso = Medidor().start()
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = _load_seg(cfg, exp)
    sp = D.load_splits(cfg)[fold]
    part = {p: np.nonzero(np.isin(d["video"], [str(v) for v in sp[p]]))[0] for p in ("train", "val", "test")}
    itr, iva, ite = part["train"], part["val"], part["test"]
    rng = np.random.default_rng(seed)
    n_eval = sc["eval_frames"]
    eval_tr = np.sort(rng.choice(itr, min(n_eval, len(itr)), replace=False))
    eval_va = np.sort(rng.choice(iva, min(n_eval, len(iva)), replace=False))
    Ptr, Ytr = _sample_patches(d, itr, sc["patch"], sc["pixels_per_class"], rng)
    Pva, Yva = _sample_patches(d, eval_va, sc["patch"], sc["pixels_per_class"], rng)
    mean, std = float(Ptr.mean() / 255), max(float(Ptr.std() / 255), 1e-6)     # normalização só com o treino
    Ptr_t, Ytr_t = torch.from_numpy(Ptr).to(device), torch.from_numpy(Ytr).to(device)
    Pva_t, Yva_t = torch.from_numpy(Pva).to(device), torch.from_numpy(Yva).to(device)
    prep = lambda a: a.float().div_(255).sub_(mean).div_(std)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    sizes = [sc["patch"] ** 2, *sc["hidden"], 2]
    model = _net(sizes).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=sc["lr"], weight_decay=sc["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=sc["epochs"])
    gen = torch.Generator(device=device).manual_seed(seed)
    best, best_ep, best_state, wait, hist, train_sec, sec_to_best = -1.0, -1, None, 0, [], 0.0, 0.0
    for ep in range(sc["epochs"]):
        te = time.perf_counter()
        model.train()
        losses = []
        for b in torch.randperm(len(Ptr_t), generator=gen, device=device).split(sc["batch_size"]):
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(prep(Ptr_t[b])), Ytr_t[b])
            loss.backward()
            opt.step()
            losses.append(loss.detach())
        cur_lr = opt.param_groups[0]["lr"]
        sched.step()
        sec_train = time.perf_counter() - te
        train_sec += sec_train
        tv = time.perf_counter()
        val_dice, val_iou = _mean_dice(_predict_masks(model, d, eval_va, mean, std, device, sc), d, eval_va)
        model.eval()
        with torch.no_grad():
            val_loss = float(sum(loss_fn(model(prep(Pva_t[b])), Yva_t[b]).item() * len(b) for b in torch.arange(len(Pva_t), device=device).split(65536)) / len(Pva_t))
        sec_val = time.perf_counter() - tv
        train_dice, _ = _mean_dice(_predict_masks(model, d, eval_tr, mean, std, device, sc), d, eval_tr)
        hist.append({"epoch": ep, "lr": cur_lr, "train_loss": float(torch.stack(losses).mean()), "val_loss": val_loss, "train_dice": train_dice,
                     "val_dice": val_dice, "val_iou": val_iou, "sec_train": sec_train, "sec_val": sec_val})
        if val_dice > best:
            best, best_ep, wait, sec_to_best = val_dice, ep, 0, train_sec
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= sc["patience"]:
                break
    model.load_state_dict(best_state)
    ti = time.perf_counter()
    pred = _predict_masks(model, d, ite, mean, std, device, sc)
    infer_ms = (time.perf_counter() - ti) * 1000 / max(len(ite), 1)
    gt = _unpack(d["packed"][ite], d["size"])
    rows = [{"path": d["paths"][i], "video_id": d["video"][i], **seg_metrics(p, g)} for i, p, g in zip(ite, pred, gt)]
    frames = pd.DataFrame(rows)
    cost = {"seconds_total": time.perf_counter() - t0, "seconds_train": train_sec, "seconds_to_best": sec_to_best, "epochs_run": len(hist),
            "best_epoch": best_ep, "seconds_per_epoch": train_sec / len(hist), "infer_ms_per_frame": infer_ms, "n_train_frames": int(len(itr)),
            "n_train_rows": int(len(Ptr)), "n_params": int(sum(p.numel() for p in model.parameters())),
            "gpu_peak_mb": float(torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else None}
    m = {"seg": {"dice": float(frames["dice"].mean()), "iou": float(frames["iou"].mean()), "hd95_median": float(frames["hd95"].median()),
                 "fail_rate": float(frames["fail"].mean())}, "epochs_run": len(hist), "best_val_dice": best, "seconds": cost["seconds_total"],
         "device": str(device), "cost": cost, "uso": uso.stop(), "hardware": C.hardware(), "history": hist}
    rd.mkdir(parents=True, exist_ok=True)
    torch.save({"state": best_state, "sizes": sizes, "mean": mean, "std": std}, rd / "model.pt")
    _save(rd, m, frames, _conf(cfg, exp, "seg", "mlp_seg", fold, seed, _pipeline_meta(cfg, exp), {"train": sc, "sizes": sizes}))
    return {"test_dice": m["seg"]["dice"]}


def _seg_tables(cfg: dict, out) -> pd.DataFrame:
    from laringopdi.etapa1 import _wilcoxon
    from laringopdi.estudo_pdi import _run_frames, _video_dice, seg_verdicts

    exps = _study(cfg)
    seed = cfg["mlp"]["seg"]["seed"]
    fr_all = []
    for e in exps:
        for k in cfg["study"]["folds"]:
            f = pd.read_csv(run_dir(cfg, e, "seg", "mlp_seg", k, seed) / "preds.csv", dtype={"video_id": str})
            fr_all.append(f.assign(tecnica=_tech(cfg, e), fold=k))
    fr = pd.concat(fr_all)
    vd = fr.groupby(["tecnica", "video_id"])["dice"].mean().unstack("tecnica")
    vd.index = pd.MultiIndex.from_product([["mlp"], vd.index], names=["method", "video_id"])
    sv = seg_verdicts(vd).set_index("technique")
    # MLP × o melhor método clássico da mesma técnica, com os mesmos vídeos.
    pdi = pd.read_csv(C.path(cfg, "tables") / "pdi" / "tabela_unica.csv").set_index("tecnica")
    frames = D.load_frames(cfg)
    _, seg = _run_frames(cfg, 1)
    classic = _video_dice(cfg, seg, pd.read_csv(C.path(cfg, "tables") / "pdi" / "segmentacao_por_fold.csv"), frames)
    classic.index = classic.index.set_levels(classic.index.levels[1].astype(str), level=1)
    vs = []
    for e in exps:
        t = _tech(cfg, e)
        pair = pd.concat([classic.loc[pdi.loc[t, "melhor_metodo"]][t].rename("c"), vd.loc["mlp"][t].rename("m")], axis=1).dropna()
        vs.append({"tecnica": t, "metodo_classico": pdi.loc[t, "melhor_metodo"], "dice_classico": pair["c"].mean(), "delta_vs_classico": (pair["m"] - pair["c"]).mean(),
                   "p_vs_classico": _wilcoxon(pair["c"], pair["m"])})
    vs = pd.DataFrame(vs)
    vs["p_holm_vs_classico"] = _holm(vs["p_vs_classico"].to_numpy())
    alpha = cfg["stats"]["alpha"]
    vs["veredito_vs_classico"] = np.where(vs["p_holm_vs_classico"] < alpha, np.where(vs["delta_vs_classico"] > 0, "MLP melhor", "clássico melhor"), "empate")
    ms = _pdi_ms(cfg)
    costs = run_costs(cfg, "mlp_seg", ms)
    conv = cost_verdicts(costs, [e for e in exps if e != "E00"])
    rows = []
    for e in exps:
        t = _tech(cfg, e)
        f, c = fr[fr["tecnica"] == t], costs[costs["exp"] == e]
        ms_infer = [json.loads((run_dir(cfg, e, "seg", "mlp_seg", k, seed) / "metrics.json").read_text(encoding="utf-8"))["cost"]["infer_ms_per_frame"] for k in cfg["study"]["folds"]]
        s = conv[conv["exp"] == e]
        rows.append({"tecnica": t, "dice": vd.loc["mlp"][t].mean(), "dice_por_frame": f["dice"].mean(), "iou": f["iou"].mean(), "hd95_mediana": f["hd95"].median(), "taxa_falha": f["fail"].mean(),
                     "delta_dice_vs_raw": 0.0 if t == "raw" else sv.loc[t, "delta_dice"], "p_holm": None if t == "raw" else sv.loc[t, "p_holm"],
                     "veredito_seg": "referência" if t == "raw" else sv.loc[t, "verdict"], **{k: c[k].mean() for k in ("epoca_melhor", "epocas", "s_ate_melhor", "s_por_epoca", "sobreajuste", "custo_total_s")},
                     "ms_tecnica": ms[e], "ms_inferencia_gpu": float(np.mean(ms_infer)), "veredito_convergencia": s.iloc[0]["veredito_convergencia"] if len(s) else "referência",
                     "veredito_custo": s.iloc[0]["veredito_custo"] if len(s) else "referência"})
    table = pd.DataFrame(rows).merge(vs.drop(columns="p_vs_classico"), on="tecnica")
    table.round(5).to_csv(out / "segmentacao.csv", index=False)
    return table


def _seg_figure(cfg: dict, table: pd.DataFrame) -> None:
    from laringopdi.estudo_pdi import NAMES
    from laringopdi.etapa1 import _plt

    plt = _plt()
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})
    y = np.arange(len(table))[::-1]
    a.barh(y + .19, table["dice"], height=.34, color="#2a78d6", label="MLP por pixel")
    a.barh(y - .19, table["dice_classico"], height=.34, color="#7D8BB8", label="Melhor método clássico")
    a.set_yticks(y, [NAMES.get(t, t) for t in table["tecnica"]])
    a.set_xlabel("Dice no teste (por vídeo)")
    a.set_title("MLP × segmentação clássica, por técnica")
    a.grid(axis="x", color="#C9CFDD", lw=.5, alpha=.6)
    a.set_axisbelow(True)
    a.tick_params(length=0)
    a.legend(frameon=False)
    if {"E00", "E03"} <= set(_study(cfg)):
        for e, color, name in (("E00", "#7D8BB8", "sem PDI"), ("E03", "#2a78d6", "CLAHE")):
            b.plot(_history(cfg, e, "seg", "mlp_seg", "train_dice"), color=color, lw=2, ls="--", label=f"{name}, treino")
            b.plot(_history(cfg, e, "seg", "mlp_seg", "val_dice"), color=color, lw=2, label=f"{name}, validação")
        b.set_xlabel("Época")
        b.set_ylabel("Dice")
        b.set_title("Curva de treino da MLP por pixel")
        b.grid(color="#C9CFDD", lw=.5, alpha=.6)
        b.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(C.path(cfg, "figures") / "mlp_segmentacao.png")
    plt.close(fig)


def cmd_mlp_seg(cfg: dict, args) -> dict:
    import torch
    from tqdm import tqdm

    out = C.path(cfg, "tables") / "mlp"
    out.mkdir(parents=True, exist_ok=True)
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "não encontrada; rodando na CPU", flush=True)
    seed = cfg["mlp"]["seg"]["seed"]
    jobs = [(e, k) for e in _study(cfg) for k in cfg["study"]["folds"]]
    if (out / "segmentacao.csv").exists() and not args.force and all((run_dir(cfg, e, "seg", "mlp_seg", k, seed) / "metrics.json").exists() for e, k in jobs):
        return {"jobs": len(jobs), "situacao": "já concluído"}
    with medir(cfg, "mlp/treino_segmentacao", "Todas as MLPs de segmentação, com inferência nos testes", n_itens=len(jobs)):
        bar = tqdm(jobs, desc="MLP por pixel", unit="job", file=sys.stdout)
        for e, k in bar:
            esperar_se_pausado(cfg)
            bar.set_postfix_str(f"{_tech(cfg, e)} fold {k}")
            seg_job(cfg, e, k, seed, args.force)
    with medir(cfg, "mlp/estatistica_segmentacao", "Vereditos e comparação com os métodos clássicos"):
        table = _seg_tables(cfg, out)
        _seg_figure(cfg, table)
    cols = ["tecnica", "dice", "veredito_seg", "dice_classico", "veredito_vs_classico", "veredito_custo"]
    return {"jobs": len(jobs), "tabela": table[cols].round(3).to_dict("records")}

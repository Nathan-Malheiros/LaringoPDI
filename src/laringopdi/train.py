"""Treino genérico por job: CNN (ResNet-18 / SimpleCNN), linear probe e shards para o Colab.

Um job = (experimento, tarefa, modelo, fold, semente) → results/runs/<exp>/<task>/<model>/fold{k}_seed{s}/
com metrics.json (teste por frame e por vídeo + validação interna), preds.csv e config.yaml.
Jobs concluídos são pulados.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import yaml

from laringopdi import config as C
from laringopdi import data as D
from laringopdi.evaluate import compute_metrics, metrics_both_levels

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ----------------------------------------------------------------------------- rótulos e dados
def frame_labels(cfg: dict) -> pd.DataFrame:
    """Rótulos dos frames amostrados (path, video_id, family, task_*). Gravado junto dos caches para
    o Colab não precisar do index.csv (que não vai para o Git)."""
    p = C.path(cfg, "cache") / "frame_labels.csv"
    if p.exists():
        return pd.read_csv(p, dtype={"video_id": str})
    fr = D.load_frames(cfg)[["path", "video_id", "family", "task_a_label", "task_b_label", "task_c_label"]]
    fr.to_csv(p, index=False)
    return fr


def job_data(cfg: dict, exp: str, task: str, fold: int) -> dict:
    """Índices (na ordem do cache) e rótulos de treino/val/teste do fold, só com rótulo na tarefa."""
    X, paths, meta = D.load_cache(cfg, exp)
    lab = frame_labels(cfg).set_index("path").loc[paths].reset_index()
    y = lab[D.task_col(task)].to_numpy()
    sp = D.load_splits(cfg)[fold]
    out = {"X": X, "paths": np.array(paths), "video": lab["video_id"].to_numpy(), "meta": meta}
    for part in ("train", "val", "test"):
        idx = np.nonzero(lab["video_id"].isin(set(sp[part])).to_numpy() & (y >= 0))[0]
        out[part] = (idx, y[idx])
    out["n_classes"] = len(D.task_names(cfg, task))
    return out


def run_dir(cfg: dict, exp: str, task: str, model: str, fold: int, seed: int):
    return C.path(cfg, "runs") / exp / task / model / f"fold{fold}_seed{seed}"


def _preds_df(paths, video, y, prob) -> pd.DataFrame:
    df = pd.DataFrame({"path": paths, "video_id": video, "y": y})
    for k in range(prob.shape[1]):
        df[f"p{k}"] = prob[:, k]
    return df


def _save(rd, metrics: dict, preds: pd.DataFrame, conf: dict) -> None:
    rd.mkdir(parents=True, exist_ok=True)
    preds.to_csv(rd / "preds.csv", index=False)
    (rd / "config.yaml").write_text(yaml.safe_dump(conf, allow_unicode=True, sort_keys=False), encoding="utf-8")
    C.write_json(rd / "metrics.json", metrics)          # por último: sua existência marca o job como concluído


def _conf(cfg, exp, task, model, fold, seed, meta, extra=None) -> dict:
    return {"exp": exp, "task": task, "model": model, "fold": fold, "seed": seed, "git_sha": C.git_sha(),
            "pipeline": meta["pipeline"], "pipeline_hash": meta["hash"], "train": cfg["train"], **(extra or {})}


# ----------------------------------------------------------------------------- CNN
def set_seed(seed: int) -> None:
    import random

    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def _loader(X, idx, y, batch, shuffle, seed):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    xt = torch.from_numpy(np.ascontiguousarray(X[np.sort(idx)]))      # pré-carregado em memória (uint8)
    order = np.argsort(np.argsort(idx))
    ds = TensorDataset(xt[torch.from_numpy(order)], torch.from_numpy(y.astype(np.int64)))
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=0, generator=g)


def _prep(xb, device, aug=None):
    import torch
    x = xb.to(device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255)
    if aug is not None:                                      # parâmetros sorteados por imagem, não por lote
        with torch.autocast(device.type, enabled=False):     # sob AMP o RandomAffine satura os pixels (bug de 2026-10-23)
            x = torch.stack([aug(xi.float()) for xi in x])
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return (x - mean) / std


def _predict(model, loader, device):
    import torch
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast(device.type, enabled=device.type == "cuda"):
        for xb, _ in loader:
            out.append(torch.softmax(model(_prep(xb, device)).float(), 1).cpu().numpy())
    return np.concatenate(out)


def train_job(cfg: dict, exp: str, task: str, model_name: str, fold: int, seed: int,
              epochs: int | None = None, force: bool = False) -> dict:
    """Treina uma CNN com o protocolo fixo e avalia no teste do fold (frame e vídeo)."""
    import torch
    from torch import nn
    from torchvision.transforms import v2

    from laringopdi.models import build

    rd = run_dir(cfg, exp, task, model_name, fold, seed)
    if (rd / "metrics.json").exists() and not force:
        return {"skipped": True, "run": str(rd)}
    t0 = time.time()
    set_seed(seed)
    tc = cfg["train"]
    epochs = epochs or (tc["epochs_fast"] if model_name.endswith("_fast") else tc["epochs"])
    d = job_data(cfg, exp, task, fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (itr, ytr), (iva, yva), (ite, yte) = d["train"], d["val"], d["test"]
    dl_tr = _loader(d["X"], itr, ytr, tc["batch_size"], True, seed)
    dl_va = _loader(d["X"], iva, yva, 128, False, seed)
    dl_te = _loader(d["X"], ite, yte, 128, False, seed)
    model = build(model_name, d["n_classes"], cfg).to(device)
    w = len(ytr) / (d["n_classes"] * np.maximum(np.bincount(ytr, minlength=d["n_classes"]), 1))
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))
    lr = tc["lr"]["resnet18" if model_name == "resnet18" else "simplecnn"]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler(enabled=tc["amp"] and device.type == "cuda")
    a = tc["augment"]                                                   # só geométrico
    aug = v2.Compose([v2.RandomHorizontalFlip(a["hflip"]),
                      v2.RandomAffine(degrees=a["rotate_deg"], scale=tuple(a["scale"]))])
    from laringopdi.live import LiveTap
    tap = LiveTap.maybe(cfg, exp, task, model_name, fold, seed, d, epochs, iva, yva)   # só observação
    if tap:
        tap.attach(model)
    best, best_state, wait, hist = -1.0, None, 0, []
    for ep in range(epochs):
        te = time.time()
        model.train()
        for step, (xb, yb) in enumerate(dl_tr):
            opt.zero_grad(set_to_none=True)
            if tap:
                tap.pre(step)
            with torch.autocast(device.type, enabled=scaler.is_enabled()):
                xin = _prep(xb, device, aug)
                out = model(xin)
                loss = loss_fn(out, yb.to(device))
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            if tap:
                tap.batch(xin, yb, out, ep, step, len(dl_tr), loss.item(), opt.param_groups[0]["lr"])
        sched.step()
        pv = _predict(model, dl_va, device)
        vm = metrics_both_levels(_preds_df(d["paths"][iva], d["video"][iva], yva, pv))
        f1 = vm["video"][tc["monitor"]]                                 # early stopping no F1-macro por vídeo
        hist.append({"epoch": ep, "val_video_f1": f1, "loss": loss.item(), "sec": time.time() - te})
        if tap:
            tap.epoch(model, ep, f1, device, pv, yva, vm, iva)
        if f1 > best:
            best, wait = f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= tc["patience"]:
                break
    model.load_state_dict(best_state)
    pv = _predict(model, dl_va, device)
    pt = _predict(model, dl_te, device)
    preds = _preds_df(d["paths"][ite], d["video"][ite], yte, pt)
    vm = metrics_both_levels(_preds_df(d["paths"][iva], d["video"][iva], yva, pv))
    m = {**metrics_both_levels(preds), "val_frame": vm["frame"], "val_video": vm["video"],
         "epochs_run": len(hist), "best_val_video_f1": best, "seconds": time.time() - t0,
         "device": str(device), "history": hist}
    if tap:
        tap.finish(m["video"]["f1_macro"])
    if seed == 0:                                                       # pesos só da semente 0 (Grad-CAM e app)
        rd.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, rd / "model.pt")
    _save(rd, m, preds, _conf(cfg, exp, task, model_name, fold, seed, d["meta"], {"epochs_max": epochs, "lr": lr}))
    return {"run": str(rd), "test_video_f1": m["video"]["f1_macro"], "seconds": round(m["seconds"], 1)}


# ----------------------------------------------------------------------------- linear probe
def probe_features(cfg: dict, exp: str, batch: int = 64):
    """Características de 512-d da ResNet-18 ImageNet congelada (GAP), em cache por experimento."""
    import torch

    from laringopdi.models import resnet18

    cp = D.cache_paths(cfg, exp)
    out = cp["x"].with_name(cp["x"].stem + ".probe.npy")
    if out.exists():
        return np.load(out)
    X, _, _ = D.load_cache(cfg, exp)
    m = resnet18(2)
    m.fc = torch.nn.Identity()
    m.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m.to(device)
    feats = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(np.array(X[i:i + batch]))
            feats.append(m(_prep(xb, device)).cpu().numpy())
    F = np.concatenate(feats).astype(np.float32)
    np.save(out, F)
    return F


def probe_job(cfg: dict, exp: str, task: str, fold: int, F: np.ndarray, force: bool = False) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rd = run_dir(cfg, exp, task, "probe", fold, 0)
    if (rd / "metrics.json").exists() and not force:
        return {"skipped": True}
    t0 = time.time()
    d = job_data(cfg, exp, task, fold)
    (itr, ytr), (iva, yva), (ite, yte) = d["train"], d["val"], d["test"]
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=cfg["probe"]["C"], max_iter=cfg["probe"]["max_iter"],
                                                            class_weight="balanced"))
    clf.fit(F[itr], ytr)                                                # escala e pesos só do treino
    preds = _preds_df(d["paths"][ite], d["video"][ite], yte, clf.predict_proba(F[ite]))
    vm = metrics_both_levels(_preds_df(d["paths"][iva], d["video"][iva], yva, clf.predict_proba(F[iva])))
    m = {**metrics_both_levels(preds), "val_frame": vm["frame"], "val_video": vm["video"], "seconds": time.time() - t0}
    _save(rd, m, preds, _conf(cfg, exp, task, "probe", fold, 0, d["meta"], {"probe": cfg["probe"]}))
    return {"video_f1": m["video"]["f1_macro"]}


def cmd_probe(cfg: dict, args) -> dict:
    if args.exp == "E14":                                   # define o E14 a partir do ranking do Estágio 1
        from laringopdi.evaluate import cmd_summary
        return greedy_e14(cfg, cmd_summary(cfg, args)["top4"])
    exps = cfg["stage1"]["experiments"] if not args.exp or args.exp == "stage1" else args.exp.split(",")
    done = 0
    for e in exps:
        F = probe_features(cfg, e)
        for t in cfg["stage1"]["tasks"]:
            for k in cfg["stage1"]["folds"]:
                probe_job(cfg, e, t, k, F, args.force)
                done += 1
    from laringopdi.evaluate import cmd_summary
    return {"probe_jobs": done, **cmd_summary(cfg, args)}


# ----------------------------------------------------------------------------- shards
def stage_jobs(cfg: dict, stage: str) -> list[dict]:
    st = cfg[stage]
    models = [m for m in st["models"] if m != "probe"]
    return [{"exp": e, "task": t, "model": m, "fold": k, "seed": s}
            for e in st["experiments"] for m in models for t in st["tasks"] for k in st["folds"] for s in st["seeds"]]


def make_shards(cfg: dict, stage: str, n_shards: int) -> dict:
    """Distribui os jobs por experimento (cada pessoa precisa de poucos caches) e equilibra a contagem."""
    jobs = stage_jobs(cfg, stage)
    exps = sorted({j["exp"] for j in jobs})
    shards = {i: [] for i in range(n_shards)}
    for e in sorted(exps, key=lambda e: -sum(j["exp"] == e for j in jobs)):
        i = min(shards, key=lambda i: len(shards[i]))
        shards[i] += [j for j in jobs if j["exp"] == e]
    out = {"stage": stage, "n_shards": n_shards, "n_jobs": len(jobs),
           "shards": {str(i): v for i, v in shards.items()},
           "caches": {str(i): sorted({j["exp"] for j in v}) for i, v in shards.items()}}
    C.write_json(C.path(cfg, "results") / "shards" / f"{stage}.json", out)
    return out


def cmd_shards(cfg: dict, args) -> dict:
    out = make_shards(cfg, args.stage, args.n_shards)
    return {"stage": out["stage"], "n_jobs": out["n_jobs"], "caches_per_shard": out["caches"]}


def cmd_train(cfg: dict, args) -> dict:
    """Um job (`--exp --task --model --fold --seed`) ou um shard inteiro (`--stage --shard`)."""
    if args.shard is not None:
        p = C.path(cfg, "results") / "shards" / f"{args.stage}.json"
        jobs = json.loads(p.read_text(encoding="utf-8"))["shards"][str(args.shard)]
        res = [train_job(cfg, j["exp"], j["task"], j["model"], j["fold"], j["seed"], args.epochs, args.force) for j in jobs]
        return {"shard": args.shard, "jobs": len(res), "skipped": sum(bool(r.get("skipped")) for r in res)}
    return train_job(cfg, args.exp, args.task, args.model, args.fold, args.seed, args.epochs, args.force)


# ----------------------------------------------------------------------------- braço clássico
CLASSIC_VARIANT = {("full", "classic"): "full_classic", ("roi", "classic"): "roi_classic", ("full", "gt"): "full_gt"}


def _classic_model(name: str, cfg: dict, seed: int):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC

    cc = cfg["classic"]
    if name == "mlp":
        return MLPClassifier(hidden_layer_sizes=tuple(cc["mlp"]["hidden_layer_sizes"]), alpha=cc["mlp"]["alpha"],
                             max_iter=cc["mlp"]["max_iter"], early_stopping=cc["mlp"]["early_stopping"], random_state=seed)
    if name == "svm":
        return SVC(C=cc["svm"]["C"], gamma=cc["svm"]["gamma"], probability=True, class_weight="balanced", random_state=seed)
    if name == "rf":
        return RandomForestClassifier(n_estimators=cc["rf"]["n_estimators"], class_weight="balanced", random_state=seed, n_jobs=-1)
    raise ValueError(name)


def _balance(idx: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """Sobreamostragem aleatória até equilibrar as classes (o MLP do sklearn não aceita pesos de classe)."""
    rng = np.random.default_rng(seed)
    n = np.bincount(y).max()
    return np.concatenate([rng.choice(idx[y == c], n, replace=True) for c in np.unique(y)])


def family_importance(clf, Xte: np.ndarray, yte: np.ndarray, cols: list[str], repeats: int, seed: int) -> dict:
    """Queda da acurácia balanceada ao permutar, em conjunto, todas as colunas de uma família."""
    from sklearn.metrics import balanced_accuracy_score

    from laringopdi.features import FAMILIES
    rng = np.random.default_rng(seed)
    base = balanced_accuracy_score(yte, clf.predict(Xte))
    out = {}
    for fam in FAMILIES:
        j = [i for i, c in enumerate(cols) if c.startswith(fam)]
        drops = []
        for _ in range(repeats):
            Xp = Xte.copy()
            Xp[:, j] = Xp[rng.permutation(len(Xp))][:, j]
            drops.append(base - balanced_accuracy_score(yte, clf.predict(Xp)))
        out[fam.rstrip("_")] = {"mean": float(np.mean(drops)), "sd": float(np.std(drops))}
    return out


def classic_job(cfg: dict, cid: str, task: str, fold: int, V: pd.DataFrame, labels: pd.Series,
                seed: int = 0, force: bool = False) -> dict:
    from sklearn.preprocessing import StandardScaler

    spec = cfg["classic_experiments"][cid]
    rd = run_dir(cfg, cid, task, spec["model"], fold, seed)
    if (rd / "metrics.json").exists() and not force:
        return {"skipped": True}
    t0 = time.time()
    sp = D.load_splits(cfg)[fold]
    vids = V.index.to_numpy()
    y = labels.loc[vids].to_numpy()
    tr = np.nonzero(np.isin(vids, sp["train"] + sp["val"]) & (y >= 0))[0]
    te = np.nonzero(np.isin(vids, sp["test"]) & (y >= 0))[0]
    sc = StandardScaler().fit(V.iloc[tr].to_numpy())                    # escala só do treino
    Xtr, Xte = sc.transform(V.iloc[tr].to_numpy()), sc.transform(V.iloc[te].to_numpy())
    clf = _classic_model(spec["model"], cfg, seed)
    if spec["model"] == "mlp":
        b = _balance(np.arange(len(tr)), y[tr], seed)
        clf.fit(Xtr[b], y[tr][b])
    else:
        clf.fit(Xtr, y[tr])
    prob = clf.predict_proba(Xte)
    preds = pd.DataFrame({"path": vids[te], "video_id": vids[te], "y": y[te]})
    for k in range(prob.shape[1]):
        preds[f"p{k}"] = prob[:, k]
    vm = compute_metrics(y[te], prob)
    m = {"frame": vm, "video": vm, "unit": "video", "n_features": V.shape[1],
         "perm_importance": family_importance(clf, Xte, y[te], list(V.columns), cfg["classic"]["permutation_repeats"], seed),
         "seconds": time.time() - t0}
    conf = {"exp": cid, "task": task, "model": spec["model"], "fold": fold, "seed": seed, "git_sha": C.git_sha(),
            "pipeline": f"classic:{spec['region']}/{spec['mask']}", "pipeline_hash": "-", "classic": cfg["classic"]}
    _save(rd, m, preds, conf)
    return {"video_f1": vm["f1_macro"]}


def cmd_classic(cfg: dict, args) -> dict:
    from laringopdi.features import build_features, video_features

    lab = frame_labels(cfg).drop_duplicates("video_id").set_index("video_id")
    ids = list(cfg["classic_experiments"]) if not args.exp else args.exp.split(",")
    cache, res = {}, {}
    for cid in ids:
        spec = cfg["classic_experiments"][cid]
        var = CLASSIC_VARIANT[(spec["region"], spec["mask"])]
        if var not in cache:
            cache[var] = video_features(build_features(cfg, var, args.jobs))
        for t in ("A", "B"):
            f1 = [classic_job(cfg, cid, t, k, cache[var], lab[D.task_col(t)], 0, args.force).get("video_f1")
                  for k in cfg["stage1"]["folds"]]
            res[f"{cid}/{t}"] = round(float(np.nanmean([x for x in f1 if x is not None])), 3) if any(x is not None for x in f1) else "skipped"
    from laringopdi.evaluate import cmd_summary
    cmd_summary(cfg, args)
    return {"video_f1_mean": res}


# ----------------------------------------------------------------------------- E14 (combinação gulosa)
STEP_ORDER = ["spec", "bilat", "median", "he", "clahe", "homo", "roi_gt", "roi", "edges", "lbp", "m1", "m2"]


def _compose(steps: list[str]) -> str:
    return ">".join(sorted(steps, key=STEP_ORDER.index))


def _probe_val_f1(cfg: dict, pipeline: str) -> float:
    """F1-macro por vídeo na VALIDAÇÃO interna (média em tarefas × folds) do probe para um pipeline."""
    eid = "G_" + pipeline.replace(">", "+")
    cfg["experiments"][eid] = {"pipeline": pipeline, "desc": "candidato E14"}
    D.build_cache(cfg, eid)
    F = probe_features(cfg, eid)
    f1 = []
    for t in cfg["stage1"]["tasks"]:
        for k in cfg["stage1"]["folds"]:
            probe_job(cfg, eid, t, k, F)
            m = json.loads((run_dir(cfg, eid, t, "probe", k, 0) / "metrics.json").read_text(encoding="utf-8"))
            f1.append(m["val_video"]["f1_macro"])
    return float(np.mean(f1))


def greedy_e14(cfg: dict, top: list[str]) -> dict:
    """Seleção gulosa entre as técnicas do top-k do Estágio 1 (só validação; `roi_gt` excluído)."""
    pool = [cfg["experiments"][e]["pipeline"] for e in top]
    pool = [p for p in pool if ">" not in p and p not in ("roi_gt", "raw")]
    trail = []
    best_steps = [pool[0]]
    best = _probe_val_f1(cfg, _compose(best_steps))
    trail.append({"pipeline": _compose(best_steps), "val_f1": best, "accepted": True})
    rep = {"edges", "lbp", "m1", "m2"}                     # saídas multicanal: no máximo uma, sempre no fim
    for cand in pool[1:]:
        if cand in rep and rep & set(best_steps):
            trail.append({"pipeline": _compose(best_steps + [cand]), "val_f1": None, "accepted": False,
                          "note": "incompatível: duas representações multicanal"})
            continue
        steps = best_steps + [cand]
        f = _probe_val_f1(cfg, _compose(steps))
        ok = f > best
        trail.append({"pipeline": _compose(steps), "val_f1": f, "accepted": ok})
        if ok:
            best, best_steps = f, steps
    out = {"E14": _compose(best_steps), "val_f1": best, "trail": trail, "top": top, "git_sha": C.git_sha()}
    C.write_json(C.path(cfg, "tables") / "e14_greedy.json", out)
    return out

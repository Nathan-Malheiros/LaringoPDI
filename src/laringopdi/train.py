"""Treino genérico por job da ResNet-18.

Um job = (experimento, tarefa, modelo, fold, semente) → results/runs/<exp>/<task>/<model>/fold{k}_seed{s}/
com metrics.json, preds.csv e config.yaml. Jobs concluídos são pulados.

O metrics.json guarda, além do teste por frame e por vídeo:
- `history`: por época, perda e F1 no treino e na validação, taxa de aprendizado e tempo;
- `cost`: tempo total, tempo até a melhor época, épocas até convergir e pico de memória;
- `uso`: CPU, RAM, GPU e energia medidas durante o job (laringopdi.custo);
- `hardware`: CPU, RAM e GPU em que o job rodou.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import yaml

from laringopdi import config as C
from laringopdi import data as D
from laringopdi.custo import Medidor
from laringopdi.evaluate import metrics_both_levels

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ----------------------------------------------------------------------------- rótulos e dados
def frame_labels(cfg: dict) -> pd.DataFrame:
    """Rótulos dos frames amostrados (path, video_id, family, task_*), gravados junto dos caches."""
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


def _weighted_loss(prob: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    """Entropia cruzada ponderada (a mesma do treino), calculada a partir das probabilidades."""
    w = weights[y]
    return float(-(w * np.log(np.clip(prob[np.arange(len(y)), y], 1e-7, 1.0))).sum() / w.sum())


def train_job(cfg: dict, exp: str, task: str, model_name: str, fold: int, seed: int,
              epochs: int | None = None, force: bool = False) -> dict:
    """Treina a CNN com o protocolo fixo, registra treino e custo e avalia no teste do fold."""
    import torch
    from torch import nn
    from torchvision.transforms import v2

    from laringopdi.models import build

    rd = run_dir(cfg, exp, task, model_name, fold, seed)
    if (rd / "metrics.json").exists() and not force:
        return {"skipped": True, "run": str(rd)}
    t0 = time.perf_counter()
    uso = Medidor().start()                                             # CPU, RAM, GPU e energia durante o job
    set_seed(seed)
    tc = cfg["train"]
    epochs = epochs or tc["epochs"]
    d = job_data(cfg, exp, task, fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (itr, ytr), (iva, yva), (ite, yte) = d["train"], d["val"], d["test"]
    dl_tr = _loader(d["X"], itr, ytr, tc["batch_size"], True, seed)
    dl_tr_eval = _loader(d["X"], itr, ytr, 128, False, seed)       # treino sem augmentation, para medir o ajuste
    dl_va = _loader(d["X"], iva, yva, 128, False, seed)
    dl_te = _loader(d["X"], ite, yte, 128, False, seed)
    model = build(model_name, d["n_classes"], cfg).to(device)
    w = len(ytr) / (d["n_classes"] * np.maximum(np.bincount(ytr, minlength=d["n_classes"]), 1))
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))
    lr = tc["lr"][model_name]
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
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    best, best_ep, best_state, wait, hist = -1.0, -1, None, 0, []
    train_sec = 0.0
    sec_to_best = 0.0
    for ep in range(epochs):
        te = time.perf_counter()
        model.train()
        losses = []
        for step, (xb, yb) in enumerate(dl_tr):
            opt.zero_grad(set_to_none=True)
            if tap:
                tap.pre(step)
            with torch.autocast(device.type, enabled=scaler.is_enabled()):
                xin = _prep(xb, device, aug)
                out = model(xin)
                loss = loss_fn(out, yb.to(device))
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            losses.append(loss.item())
            if tap:
                tap.batch(xin, yb, out, ep, step, len(dl_tr), loss.item(), opt.param_groups[0]["lr"])
        cur_lr = opt.param_groups[0]["lr"]
        sched.step()
        sec_train = time.perf_counter() - te
        train_sec += sec_train                                          # só o tempo de ajuste dos pesos
        tv = time.perf_counter()
        pv = _predict(model, dl_va, device)
        vm = metrics_both_levels(_preds_df(d["paths"][iva], d["video"][iva], yva, pv))
        f1 = vm["video"][tc["monitor"]]                                 # early stopping no F1-macro por vídeo
        sec_val = time.perf_counter() - tv
        pt_ = _predict(model, dl_tr_eval, device)                       # medição; não entra no tempo de treino
        tm = metrics_both_levels(_preds_df(d["paths"][itr], d["video"][itr], ytr, pt_))
        hist.append({"epoch": ep, "lr": cur_lr, "train_loss_aug": float(np.mean(losses)),
                     "train_loss": _weighted_loss(pt_, ytr, w), "train_video_f1": tm["video"]["f1_macro"],
                     "val_loss": _weighted_loss(pv, yva, w), "val_video_f1": f1,
                     "val_video_sensitivity": vm["video"]["sensitivity"], "sec_train": sec_train, "sec_val": sec_val})
        if tap:
            tap.epoch(model, ep, f1, device, pv, yva, vm, iva)
        if f1 > best:
            best, best_ep, wait, sec_to_best = f1, ep, 0, train_sec
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= tc["patience"]:
                break
    model.load_state_dict(best_state)
    pv = _predict(model, dl_va, device)
    ti = time.perf_counter()
    pt = _predict(model, dl_te, device)
    ms_frame = (time.perf_counter() - ti) * 1000 / max(len(ite), 1)
    preds = _preds_df(d["paths"][ite], d["video"][ite], yte, pt)
    vm = metrics_both_levels(_preds_df(d["paths"][iva], d["video"][iva], yva, pv))
    cost = {"seconds_total": time.perf_counter() - t0, "seconds_train": train_sec, "seconds_to_best": sec_to_best,
            "epochs_run": len(hist), "best_epoch": best_ep, "seconds_per_epoch": train_sec / max(len(hist), 1),
            "infer_ms_per_frame": ms_frame, "n_train_frames": int(len(itr)),
            "gpu_peak_mb": float(torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else None}
    m = {**metrics_both_levels(preds), "val_frame": vm["frame"], "val_video": vm["video"],
         "epochs_run": len(hist), "best_val_video_f1": best, "seconds": cost["seconds_total"],
         "device": str(device), "cost": cost, "uso": uso.stop(), "hardware": C.hardware(), "history": hist}
    if tap:
        tap.finish(m["video"]["f1_macro"])
    if seed == 0:                                                       # pesos só da semente 0 (Grad-CAM e app)
        rd.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, rd / "model.pt")
    _save(rd, m, preds, _conf(cfg, exp, task, model_name, fold, seed, d["meta"], {"epochs_max": epochs, "lr": lr}))
    return {"run": str(rd), "test_video_f1": m["video"]["f1_macro"], "seconds": round(cost["seconds_total"], 1)}


def study_jobs(cfg: dict) -> list[dict]:
    """Todos os jobs do estudo. A semente é o laço externo: a semente 0 termina inteira primeiro."""
    st = cfg["study"]
    return [{"exp": e, "task": t, "model": m, "fold": k, "seed": s}
            for s in st["seeds"] for e in st["experiments"] for m in st["models"] for t in st["tasks"] for k in st["folds"]]


def cmd_train(cfg: dict, args) -> dict:
    """Um job (`--exp --task --model --fold --seed`) ou o estudo inteiro (`--all`)."""
    if not args.all:
        return train_job(cfg, args.exp, args.task, args.model, args.fold, args.seed, args.epochs, args.force)
    from tqdm import tqdm

    jobs = study_jobs(cfg)
    if args.seed_only is not None:
        jobs = [j for j in jobs if j["seed"] == args.seed_only]
    done = skipped = 0
    bar = tqdm(jobs, desc="Treino da ResNet-18", unit="job")
    for j in bar:
        bar.set_postfix_str(f"{j['exp']} tarefa {j['task']} fold {j['fold']} semente {j['seed']}")
        r = train_job(cfg, j["exp"], j["task"], j["model"], j["fold"], j["seed"], args.epochs, args.force)
        skipped += bool(r.get("skipped"))
        done += not r.get("skipped")
    return {"jobs": len(jobs), "treinados": done, "pulados": skipped}


# ----------------------------------------------------------------------------- utilitários da MLP
def balance(idx: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """Sobreamostragem aleatória até equilibrar as classes (o MLP do sklearn não aceita pesos de classe)."""
    rng = np.random.default_rng(seed)
    n = np.bincount(y).max()
    return np.concatenate([rng.choice(idx[y == c], n, replace=True) for c in np.unique(y)])

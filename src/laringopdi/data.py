"""Índice do dataset, deduplicação (pHash + SSIM), splits agrupados e checagem de vazamento."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from laringopdi import config as C

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
# Palavras-chave (inglês e chinês) para inferir a classe pelo nome da pasta/arquivo.
CLASS_KEYWORDS = [
    ("reinke", "reinke_edema"), ("任克", "reinke_edema"), ("水肿", "reinke_edema"),
    ("cancer", "glottic_cancer"), ("carcinoma", "glottic_cancer"), ("癌", "glottic_cancer"),
    ("granuloma", "granuloma"), ("肉芽", "granuloma"),
    ("leukoplakia", "leukoplakia"), ("白斑", "leukoplakia"),
    ("cyst", "cyst"), ("囊肿", "cyst"),
    ("nodule", "nodules"), ("小结", "nodules"),
    ("polyp", "polyps"), ("息肉", "polyps"),
    ("normal", "normal"), ("正常", "normal"),
]


def imread(p: Path | str) -> np.ndarray:
    """Lê imagem como RGB uint8 (funciona com caminhos Unicode no Windows)."""
    buf = np.fromfile(str(p), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"imagem ilegível: {p}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def imwrite(p: Path | str, img: np.ndarray) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.ndim == 3 and img.shape[2] == 3 else img
    ok, enc = cv2.imencode(p.suffix or ".png", out)
    enc.tofile(str(p))


def infer_class(rel: Path, classes: list[str]) -> str | None:
    """Classe a partir das pastas (mais interna primeiro) e, por fim, do nome do arquivo."""
    for part in list(reversed(rel.parts[:-1])) + [rel.stem]:
        p = part.lower().strip()
        num = re.match(r"^(\d+)\b", p)
        if num and part != rel.stem and 1 <= int(num.group(1)) <= len(classes) and len(p) <= 3:
            return classes[int(num.group(1)) - 1]
        for kw, c in CLASS_KEYWORDS:
            if kw in p:
                return c
    return None


def task_labels(df: pd.DataFrame, cfg: dict, task: str) -> np.ndarray:
    """Rótulos inteiros da tarefa A/B/C a partir da coluna `class`."""
    if task == "B":
        return df["class"].map({c: i for i, c in enumerate(cfg["data"]["classes"])}).to_numpy()
    pos = set(cfg["tasks"][task]["positive"])
    return df["class"].isin(pos).astype(int).to_numpy()


def task_names(cfg: dict, task: str) -> list[str]:
    return cfg["data"]["classes"] if task == "B" else cfg["tasks"][task]["labels"]


# ----------------------------------------------------------------------------- índice
def build_index(cfg: dict) -> dict:
    raw = C.path(cfg, "raw")
    if not raw.exists() or not any(raw.iterdir()):
        raise SystemExit(f"Dataset ausente em {raw}. Faça o download manual (ver README).")
    classes = cfg["data"]["classes"]
    rx = re.compile(cfg["data"]["patient_id_regex"]) if cfg["data"].get("patient_id_regex") else None
    rows, unknown, unreadable, others = [], [], [], []
    for p in sorted(raw.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(raw)
        if p.suffix.lower() not in IMG_EXT:
            others.append(rel.as_posix())
            continue
        cls = infer_class(rel, classes)
        if cls is None:
            unknown.append(rel.as_posix())
            continue
        try:
            img = imread(p)
        except ValueError:
            unreadable.append(rel.as_posix())
            continue
        m = rx.search(p.name) if rx else None
        rows.append({"path": rel.as_posix(), "class": cls, "class_idx": classes.index(cls),
                     "height": img.shape[0], "width": img.shape[1],
                     "sha1": hashlib.sha1(p.read_bytes()).hexdigest(),
                     "patient_id": m.group(1) if m else ""})
    df = pd.DataFrame(rows)
    df.to_csv(C.path(cfg, "index"), index=False)
    counts = Counter(df["class"]) if len(df) else Counter()
    summary = {
        "n_images": len(df), "expected": 3057, "per_class": {c: counts.get(c, 0) for c in classes},
        "has_patient_id": bool(len(df) and (df["patient_id"] != "").all()),
        "n_patients": int(df.loc[df["patient_id"] != "", "patient_id"].nunique()) if len(df) else 0,
        "exact_duplicates": int(len(df) - df["sha1"].nunique()) if len(df) else 0,
        "unknown_class": unknown[:50], "n_unknown_class": len(unknown),
        "unreadable": unreadable, "non_image_files": others[:50],
        "small_classes": [c for c in classes if counts.get(c, 0) < cfg["data"]["min_class_count"]],
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    C.write_json(C.path(cfg, "logs") / "index_summary.json", summary)
    return summary


def load_index(cfg: dict) -> pd.DataFrame:
    p = C.path(cfg, "index")
    if not p.exists():
        raise SystemExit("index.csv ausente: rode `python -m laringopdi index`.")
    return pd.read_csv(p, keep_default_na=False, dtype={"patient_id": str})


# ----------------------------------------------------------------------------- dedup
def _popcount64(x: np.ndarray) -> np.ndarray:
    return np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(1)


def build_groups(cfg: dict) -> dict:
    """Agrupa imagens por paciente (se houver ID) e por quase-duplicata (pHash + SSIM)."""
    import imagehash
    from PIL import Image
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from skimage.metrics import structural_similarity as ssim

    df = load_index(cfg)
    dd = cfg["data"]["dedup"]
    raw = C.path(cfg, "raw")
    n = len(df)
    hashes = np.zeros(n, np.uint64)
    thumbs = np.zeros((n, dd["thumb"], dd["thumb"]), np.uint8)
    for i, rel in enumerate(df["path"]):
        img = imread(raw / rel)
        hashes[i] = np.uint64(int(str(imagehash.phash(Image.fromarray(img))), 16))
        thumbs[i] = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (dd["thumb"],) * 2,
                               interpolation=cv2.INTER_AREA)
    pairs = []
    for i in range(n - 1):
        d = _popcount64(hashes[i] ^ hashes[i + 1:])
        for j in np.nonzero(d <= dd["phash_max_dist"])[0] + i + 1:
            s = float(ssim(thumbs[i], thumbs[j], data_range=255))
            pairs.append((i, int(j), int(d[j - i - 1]), s, s >= dd["ssim_min"]))
    pairs_df = pd.DataFrame(pairs, columns=["i", "j", "phash_dist", "ssim", "dup"])
    pairs_df["path_i"] = df["path"].to_numpy()[pairs_df["i"]] if len(pairs_df) else []
    pairs_df["path_j"] = df["path"].to_numpy()[pairs_df["j"]] if len(pairs_df) else []
    pairs_df.to_csv(C.path(cfg, "logs") / "dedup_pairs.csv", index=False)

    edges = [(i, j) for i, j, _, _, dup in pairs if dup]
    has_pid = (df["patient_id"] != "").all()
    if has_pid:
        for _, idx in df.groupby("patient_id").indices.items():
            edges += [(idx[0], k) for k in idx[1:]]
    e = np.array(edges, dtype=int).reshape(-1, 2)
    graph = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    out = pd.DataFrame({"path": df["path"], "group": labels,
                        "phash": [f"{int(h):016x}" for h in hashes]})
    out.to_csv(C.path(cfg, "groups"), index=False)
    sizes = Counter(labels)
    cross = pairs_df[pairs_df["dup"]] if len(pairs_df) else pairs_df
    n_cross = int((df["class"].to_numpy()[cross["i"]] != df["class"].to_numpy()[cross["j"]]).sum()) if len(cross) else 0
    summary = {"n_images": n, "group_source": "patient_id+near_dup" if has_pid else "near_dup",
               "candidate_pairs": len(pairs), "confirmed_pairs": int(sum(p[4] for p in pairs)),
               "dup_pairs_cross_class": n_cross, "n_groups": len(sizes),
               "n_groups_multi": sum(1 for v in sizes.values() if v > 1),
               "largest_group": max(sizes.values()), "params": dd}
    C.write_json(C.path(cfg, "logs") / "dedup_summary.json", summary)
    return summary


# ----------------------------------------------------------------------------- splits
def check_leakage(folds: list[dict], group_of: dict[str, int]) -> None:
    """Falha se algum grupo aparece em mais de uma partição do mesmo fold, ou se o
    conjunto de teste dos folds não for uma partição do dataset."""
    all_test = []
    for f in folds:
        parts = {k: set(f[k]) for k in ("train", "val", "test")}
        for a, b in (("train", "test"), ("val", "test"), ("train", "val")):
            assert not parts[a] & parts[b], f"fold {f['fold']}: imagem em {a} e {b}"
            ga = {group_of[p] for p in parts[a]}
            gb = {group_of[p] for p in parts[b]}
            assert not ga & gb, f"fold {f['fold']}: grupo compartilhado entre {a} e {b}"
        all_test += f["test"]
    assert len(all_test) == len(set(all_test)) == len(group_of), "teste não particiona o dataset"


def make_splits(cfg: dict, force: bool = False) -> dict:
    from sklearn.model_selection import StratifiedGroupKFold

    out = C.path(cfg, "splits") / "folds.json"
    if out.exists() and not force:
        raise SystemExit(f"{out} já existe; splits são gerados uma única vez (use --force só com aprovação).")
    df = load_index(cfg)
    gp = C.path(cfg, "groups")
    if not gp.exists():
        build_groups(cfg)
    groups = pd.read_csv(gp).set_index("path").loc[df["path"], "group"].to_numpy()
    y = df["class_idx"].to_numpy()        # estratifica pelas 8 classes (cobre A, B e C)
    paths = df["path"].to_numpy()
    seed = cfg["data"]["split_seed"]
    n_inner = max(2, round(1 / cfg["data"]["inner_val_frac"]))
    outer = StratifiedGroupKFold(n_splits=cfg["data"]["n_folds"], shuffle=True, random_state=seed)
    folds = []
    for k, (tr, te) in enumerate(outer.split(paths, y, groups)):
        inner = StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=seed + 1 + k)
        itr, iva = next(inner.split(tr, y[tr], groups[tr]))
        folds.append({"fold": k, "train": paths[tr[itr]].tolist(), "val": paths[tr[iva]].tolist(),
                      "test": paths[te].tolist()})
    check_leakage(folds, dict(zip(paths, groups.tolist())))
    payload = {"meta": {"seed": seed, "n_folds": len(folds), "n_images": len(paths),
                        "inner_val_frac": cfg["data"]["inner_val_frac"], "git_sha": C.git_sha(),
                        "created": datetime.now().isoformat(timespec="seconds"),
                        "group_source": json.loads((C.path(cfg, "logs") / "dedup_summary.json")
                                                   .read_text(encoding="utf-8"))["group_source"]},
               "folds": folds}
    C.write_json(out, payload)
    cls = dict(zip(paths, df["class"]))
    rows = [{"fold": f["fold"], "part": part, "class": c, "n": n}
            for f in folds for part in ("train", "val", "test")
            for c, n in Counter(cls[p] for p in f[part]).items()]
    pd.DataFrame(rows).to_csv(C.path(cfg, "splits") / "split_counts.csv", index=False)
    return {"folds": len(folds), "sizes": [{k: len(f[k]) for k in ("train", "val", "test")} for f in folds]}


def load_splits(cfg: dict) -> list[dict]:
    p = C.path(cfg, "splits") / "folds.json"
    if not p.exists():
        raise SystemExit("splits ausentes: rode `python -m laringopdi splits`.")
    return json.loads(p.read_text(encoding="utf-8"))["folds"]


# ----------------------------------------------------------------------------- cache do PDI
def _process_one(p, pipeline, params, size, work_size):
    from laringopdi.pdi import run_pipeline
    from laringopdi.pdi.quality import image_quality

    out, ctx = run_pipeline(imread(p), pipeline, params, size, work_size)
    masks = {k: np.packbits(ctx[k]) for k in ("fov_mask_lb", "spec_mask_lb", "roi_mask_lb") if k in ctx}
    extra = {k: ctx[k] for k in ("roi_bbox", "fov_bbox", "spec_frac") if k in ctx}
    return out, image_quality(out, ctx), masks, extra


def cache_paths(cfg: dict, exp: str) -> dict:
    from laringopdi.pdi import pipeline_hash

    e = cfg["experiments"][exp]
    h = pipeline_hash(e["pipeline"], e.get("params"), cfg["data"]["image_size"], cfg["data"]["work_size"])
    base = C.path(cfg, "cache") / f"{exp}_{h}"
    return {"hash": h, "x": base.with_suffix(".npy"), "masks": base.with_suffix(".masks.npz"),
            "meta": base.with_suffix(".json")}


def build_cache(cfg: dict, exp: str, n_jobs: int = -1, force: bool = False) -> dict:
    """Aplica o pipeline do experimento a todas as imagens e grava (N, 224, 224, C) uint8 +
    máscaras empacotadas + metadados + métricas de qualidade. Pula se já existir."""
    from joblib import Parallel, delayed

    from laringopdi.pdi.registry import resolve_params

    e = cfg["experiments"][exp]
    if not e.get("pipeline"):
        raise SystemExit(f"{exp} sem pipeline definido.")
    cp = cache_paths(cfg, exp)
    qpath = C.path(cfg, "results") / "pdi_quality" / f"{exp}.csv"
    if cp["x"].exists() and cp["meta"].exists() and qpath.exists() and not force:
        return {"exp": exp, "skipped": True}
    df = load_index(cfg)
    raw, size, ws = C.path(cfg, "raw"), cfg["data"]["image_size"], cfg["data"]["work_size"]
    res = Parallel(n_jobs=n_jobs, batch_size=16)(
        delayed(_process_one)(raw / p, e["pipeline"], e.get("params"), size, ws) for p in df["path"])
    X = np.stack([r[0] for r in res])
    np.save(cp["x"], X)
    mkeys = sorted({k for r in res for k in r[2]})
    np.savez_compressed(cp["masks"], **{k: np.stack([r[2].get(k, np.packbits(np.zeros((size, size), bool)))
                                                    for r in res]) for k in mkeys})
    q = pd.DataFrame([r[1] | {"path": p, "class": c} for r, p, c in zip(res, df["path"], df["class"])])
    q.insert(0, "exp", exp)
    qpath.parent.mkdir(parents=True, exist_ok=True)
    q.to_csv(qpath, index=False)
    C.write_json(cp["meta"], {"exp": exp, "pipeline": e["pipeline"],
                              "params": resolve_params(e["pipeline"], e.get("params")),
                              "hash": cp["hash"], "shape": list(X.shape), "paths": df["path"].tolist(),
                              "mask_keys": mkeys, "extra": [r[3] for r in res], "git_sha": C.git_sha()})
    return {"exp": exp, "shape": list(X.shape), "skipped": False}


def quality_summary(cfg: dict) -> pd.DataFrame:
    """Agrega results/pdi_quality/*.csv em results/pdi_quality.csv (média por experimento)."""
    files = sorted((C.path(cfg, "results") / "pdi_quality").glob("E*.csv"))
    if not files:
        raise SystemExit("sem métricas de qualidade: rode `cache` antes.")
    q = pd.concat([pd.read_csv(f) for f in files])
    for k in ("fov_fallback", "roi_fallback"):
        q[k] = q[k].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
    agg = q.groupby("exp")[["entropy", "rms_contrast", "spec_frac", "lap_var", "fov_fallback", "roi_fallback"]].mean()
    agg.round(4).to_csv(C.path(cfg, "results") / "pdi_quality.csv")
    return agg


def load_cache(cfg: dict, exp: str) -> tuple[np.ndarray, list[str], dict]:
    cp = cache_paths(cfg, exp)
    if not cp["x"].exists():
        raise SystemExit(f"cache de {exp} ausente: rode `python -m laringopdi cache --exp {exp}`.")
    meta = json.loads(cp["meta"].read_text(encoding="utf-8"))
    return np.load(cp["x"], mmap_mode="r"), meta["paths"], meta

"""Adaptadores de dataset, índice, amostragem de frames, grupos por vídeo, splits e cache do PDI."""
from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from laringopdi import config as C

INDEX_COLUMNS = ["path", "video_id", "frame", "disorder_raw", "family", "task_a_label", "task_b_label",
                 "task_c_label", "is_rgb", "has_seg", "seg_path", "seg_area", "official_split",
                 "height", "width"]


def imread(p: Path | str, gray: bool = False) -> np.ndarray:
    """Lê imagem como RGB uint8 (ou cinza); funciona com caminhos Unicode no Windows."""
    buf = np.fromfile(str(p), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"imagem ilegível: {p}")
    return img if gray else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def imwrite(p: Path | str, img: np.ndarray) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.ndim == 3 and img.shape[2] == 3 else img
    cv2.imencode(p.suffix or ".png", out)[1].tofile(str(p))


# ----------------------------------------------------------------------------- rótulos
def disorder_family(raw: str, cfg: dict) -> str:
    """Família do status em texto livre pelas regras ordenadas de configs/label_map.yaml."""
    s = str(raw).strip().lower()
    lm = cfg["label_map"]
    for r in lm["rules"]:
        if s in [str(x).lower() for x in r.get("exact", [])] or any(str(k).lower() in s for k in r.get("any", [])):
            return r["family"]
    return lm.get("default", "other")


FAMILY_LABELS = {"healthy": (0, 0, 0), "functional": (1, -1, 1), "structural": (1, 1, 2),
                 "other": (-1, -1, -1), "unknown": (-1, -1, -1)}


def task_col(task: str) -> str:
    return f"task_{task.lower()}_label"


def task_names(cfg: dict, task: str) -> list[str]:
    return cfg["tasks"][task]["labels"]


# ----------------------------------------------------------------------------- adaptadores
class DatasetAdapter:
    """Interface: `build_index()` devolve DataFrame com INDEX_COLUMNS (+ colunas `meta_*`)."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.root = C.path(cfg, "raw")

    def build_index(self) -> pd.DataFrame:
        raise NotImplementedError


class Laryngoscope8Adapter(DatasetAdapter):
    """Stub: o Laryngoscope8 está inacessível (DECISIONS.md, 2026-10-22). A versão funcional do
    índice por pastas de classe está no commit c96d1f7, caso o dataset reapareça."""

    def build_index(self) -> pd.DataFrame:
        raise SystemExit("Laryngoscope8Adapter é um stub (dataset inacessível); ver DECISIONS.md.")


def _norm_key(k: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(k).lower()).strip("_")


def _find_key(meta: dict, *needles: str) -> str | None:
    for k in meta:
        if all(n in k for n in needles):
            return k
    return None


def _read_meta(p: Path) -> dict:
    for ext in (".meta", ".json"):
        m = p.with_suffix(ext)
        if m.exists():
            return {_norm_key(k): v for k, v in json.loads(m.read_text(encoding="utf-8")).items()}
    return {}


def _bagls_row(p: Path, root: Path) -> dict:
    from PIL import Image

    meta = _read_meta(p)
    seg = p.with_name(p.stem + "_seg.png")
    with Image.open(p) as im:
        w, h = im.size
        mode = im.mode
    seg_area = float((imread(seg, gray=True) > 127).mean()) if seg.exists() else 0.0
    vk = _find_key(meta, "video", "id") or _find_key(meta, "video")
    dk = _find_key(meta, "disorder")
    ck = _find_key(meta, "color") or _find_key(meta, "colour")
    dis = meta.get(dk, "") if dk else ""
    dis = "; ".join(map(str, dis)) if isinstance(dis, list) else str(dis)
    color = meta.get(ck) if ck else None
    rel = p.relative_to(root)
    row = {"path": rel.as_posix(), "video_id": str(meta.get(vk, "")) if vk else "",
           "frame": int(p.stem) if p.stem.isdigit() else p.stem, "disorder_raw": dis,
           "is_rgb": bool(color) if isinstance(color, bool) else (mode not in ("L", "I;16", "I")),
           "has_seg": seg.exists(), "seg_path": seg.relative_to(root).as_posix() if seg.exists() else "",
           "seg_area": seg_area, "official_split": rel.parts[0] if len(rel.parts) > 1 else "",
           "height": h, "width": w}
    row.update({f"meta_{k}": (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in meta.items()})
    return row


class BaglsAdapter(DatasetAdapter):
    """BAGLS: `<id>.png` + `<id>_seg.png` + metadados por frame (`<id>.meta`/`.json`)."""

    def build_index(self) -> pd.DataFrame:
        from joblib import Parallel, delayed

        frames = [p for p in sorted(self.root.rglob("*.png")) if not p.stem.endswith("_seg")]
        if not frames:
            raise SystemExit(f"Nenhum frame .png em {self.root}.")
        rows = Parallel(n_jobs=-1, batch_size=256)(delayed(_bagls_row)(p, self.root) for p in frames)
        df = pd.DataFrame(rows)
        if (df["video_id"] == "").all():
            raise SystemExit("Metadados sem ID de vídeo: inspecione os arquivos e ajuste o BaglsAdapter.")
        # status do vídeo = união dos status dos seus frames
        vid_dis = df.groupby("video_id")["disorder_raw"].agg(
            lambda s: "; ".join(sorted({x for x in s if x}))).rename("disorder_video")
        df = df.join(vid_dis, on="video_id")
        df["disorder_raw"] = df["disorder_video"]
        df["family"] = df["disorder_raw"].map(lambda s: disorder_family(s, self.cfg))
        labs = np.array([FAMILY_LABELS[f] for f in df["family"]])
        df["task_a_label"], df["task_b_label"], df["task_c_label"] = labs[:, 0], labs[:, 1], labs[:, 2]
        meta_cols = sorted(c for c in df.columns if c.startswith("meta_"))
        return df[INDEX_COLUMNS + meta_cols]


ADAPTERS = {"bagls": BaglsAdapter, "laryngoscope8": Laryngoscope8Adapter}


# ----------------------------------------------------------------------------- índice
def build_index(cfg: dict) -> dict:
    raw = C.path(cfg, "raw")
    if not raw.exists() or not any(raw.iterdir()):
        raise SystemExit(f"Dataset ausente em {raw}. Faça o download manual (ver README).")
    df = ADAPTERS[cfg["dataset"]](cfg).build_index()
    df.to_csv(C.path(cfg, "index"), index=False)
    vids = df.drop_duplicates("video_id")
    fam_v = vids["family"].value_counts().to_dict()
    status_map = (vids.groupby(["disorder_raw", "family"]).size().reset_index(name="videos")
                  .sort_values(["family", "videos"], ascending=[True, False]))
    status_map.to_csv(C.path(cfg, "tables") / "status_map.csv", index=False)
    summary = {
        "n_frames": len(df), "n_videos": len(vids), "expected": {"frames": 59250, "videos": 640},
        "videos_per_family": fam_v,
        "frames_per_family": df["family"].value_counts().to_dict(),
        "n_distinct_status": int(len(status_map)),
        "videos_rgb": int(vids["is_rgb"].sum()), "frames_with_seg": int(df["has_seg"].sum()),
        "frames_seg_nonempty": int((df["seg_area"] > 0).sum()),
        "official_split_videos": vids["official_split"].value_counts().to_dict(),
        "task_videos": {t: vids[task_col(t)].value_counts().sort_index().to_dict() for t in "ABC"},
        "subject_id_column": [c for c in df.columns if "subject" in c and "id" in c] or None,
        "meta_columns": [c for c in df.columns if c.startswith("meta_")],
        "small_classes_videos": {f: n for f, n in fam_v.items()
                                 if f in ("healthy", "functional", "structural") and n < cfg["data"]["min_class_count"]},
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    C.write_json(C.path(cfg, "logs") / "index_summary.json", summary)
    return summary


def load_index(cfg: dict) -> pd.DataFrame:
    p = C.path(cfg, "index")
    if not p.exists():
        raise SystemExit("index.csv ausente: rode `python -m laringopdi index`.")
    return pd.read_csv(p, keep_default_na=False, dtype={"video_id": str, "disorder_raw": str})


# ----------------------------------------------------------------------------- amostragem
def sample_frames(cfg: dict, force: bool = False) -> dict:
    """Até `frames_per_video` frames com máscara não vazia, uniformemente espaçados por vídeo,
    só de vídeos com rótulo na Tarefa A. Determinístico; salvo uma única vez."""
    out = C.path(cfg, "splits") / "frames.json"
    if out.exists() and not force:
        return json.loads(out.read_text(encoding="utf-8"))["meta"]
    df = load_index(cfg)
    df = df[(df["task_a_label"] >= 0) & df["has_seg"] & (df["seg_area"] > 0)]
    k = cfg["data"]["frames_per_video"]
    picks = []
    for _, g in df.groupby("video_id", sort=True):
        g = g.sort_values("frame", key=lambda s: pd.to_numeric(s, errors="coerce").fillna(0))
        idx = np.unique(np.linspace(0, len(g) - 1, min(k, len(g))).round().astype(int))
        picks += g["path"].iloc[idx].tolist()
    meta = {"frames_per_video": k, "n_frames": len(picks), "n_videos": int(df["video_id"].nunique()),
            "criterion": "has_seg & seg_area>0 & task_a_label>=0; linspace por ordem de frame",
            "created": datetime.now().isoformat(timespec="seconds")}
    C.write_json(out, {"meta": meta, "frames": picks})
    return meta


def load_frames(cfg: dict) -> pd.DataFrame:
    """Índice restrito aos frames amostrados (na ordem de frames.json)."""
    p = C.path(cfg, "splits") / "frames.json"
    if not p.exists():
        raise SystemExit("frames.json ausente: rode `python -m laringopdi splits`.")
    paths = json.loads(p.read_text(encoding="utf-8"))["frames"]
    return load_index(cfg).set_index("path").loc[paths].reset_index()


# ----------------------------------------------------------------------------- grupos
def _popcount64(x: np.ndarray) -> np.ndarray:
    return np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(1)


def build_groups(cfg: dict) -> dict:
    """Funde vídeos quase-duplicados (possível mesmo sujeito): pHash do frame central amostrado
    de cada vídeo (Hamming ≤ limiar) confirmado por SSIM."""
    import imagehash
    from PIL import Image
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from skimage.metrics import structural_similarity as ssim

    fr = load_frames(cfg)
    dd, raw = cfg["data"]["dedup"], C.path(cfg, "raw")
    central = fr.groupby("video_id", sort=True)["path"].agg(lambda s: s.iloc[len(s) // 2])
    vids = central.index.to_numpy()
    n = len(vids)
    hashes = np.zeros(n, np.uint64)
    thumbs = np.zeros((n, dd["thumb"], dd["thumb"]), np.uint8)
    for i, rel in enumerate(central):
        g = imread(raw / rel, gray=True)
        hashes[i] = np.uint64(int(str(imagehash.phash(Image.fromarray(g))), 16))
        thumbs[i] = cv2.resize(g, (dd["thumb"],) * 2, interpolation=cv2.INTER_AREA)
    pairs = []
    for i in range(n - 1):
        d = _popcount64(hashes[i] ^ hashes[i + 1:])
        for j in np.nonzero(d <= dd["phash_max_dist"])[0] + i + 1:
            s = float(ssim(thumbs[i], thumbs[j], data_range=255))
            pairs.append((vids[i], vids[j], int(d[j - i - 1]), s, s >= dd["ssim_min"]))
    pd.DataFrame(pairs, columns=["video_i", "video_j", "phash_dist", "ssim", "dup"]).to_csv(
        C.path(cfg, "logs") / "dedup_video_pairs.csv", index=False)
    pos = {v: i for i, v in enumerate(vids)}
    e = np.array([(pos[a], pos[b]) for a, b, _, _, dup in pairs if dup], dtype=int).reshape(-1, 2)
    _, labels = connected_components(coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)),
                                     directed=False)
    pd.DataFrame({"video_id": vids, "group": labels, "phash": [f"{int(h):016x}" for h in hashes]}).to_csv(
        C.path(cfg, "groups"), index=False)
    sizes = Counter(labels)
    summary = {"n_videos": n, "candidate_pairs": len(pairs), "confirmed_pairs": int(sum(p[4] for p in pairs)),
               "n_groups": len(sizes), "n_groups_multi": sum(1 for v in sizes.values() if v > 1),
               "largest_group": max(sizes.values()) if sizes else 0, "params": dd}
    C.write_json(C.path(cfg, "logs") / "dedup_summary.json", summary)
    return summary


# ----------------------------------------------------------------------------- splits
def check_leakage(folds: list[dict], group_of: dict[str, int]) -> None:
    """Falha se um vídeo ou grupo aparece em duas partições do mesmo fold, ou se os testes
    dos folds não particionam o conjunto de vídeos."""
    all_test = []
    for f in folds:
        parts = {k: set(f[k]) for k in ("train", "val", "test")}
        for a, b in (("train", "test"), ("val", "test"), ("train", "val")):
            assert not parts[a] & parts[b], f"fold {f['fold']}: vídeo em {a} e {b}"
            assert not {group_of[v] for v in parts[a]} & {group_of[v] for v in parts[b]}, \
                f"fold {f['fold']}: grupo compartilhado entre {a} e {b}"
        all_test += f["test"]
    assert len(all_test) == len(set(all_test)) == len(group_of), "teste não particiona os vídeos"


def make_splits(cfg: dict, force: bool = False) -> dict:
    from sklearn.model_selection import StratifiedGroupKFold

    out = C.path(cfg, "splits") / "folds.json"
    if out.exists() and not force:
        raise SystemExit(f"{out} já existe; splits são gerados uma única vez (--force só com aprovação).")
    fr = load_frames(cfg)
    if not C.path(cfg, "groups").exists():
        build_groups(cfg)
    vids = fr.drop_duplicates("video_id").set_index("video_id")
    grp = pd.read_csv(C.path(cfg, "groups"), dtype={"video_id": str}).set_index("video_id")["group"]
    v = vids.index.to_numpy()
    y = vids["task_c_label"].to_numpy()            # estratifica por saudável/funcional/estrutural
    g = grp.loc[v].to_numpy()
    seed = cfg["data"]["split_seed"]
    n_inner = max(2, round(1 / cfg["data"]["inner_val_frac"]))
    folds = []
    for k, (tr, te) in enumerate(StratifiedGroupKFold(cfg["data"]["n_folds"], shuffle=True,
                                                      random_state=seed).split(v, y, g)):
        itr, iva = next(StratifiedGroupKFold(n_inner, shuffle=True, random_state=seed + 1 + k)
                        .split(tr, y[tr], g[tr]))
        folds.append({"fold": k, "train": v[tr[itr]].tolist(), "val": v[tr[iva]].tolist(), "test": v[te].tolist()})
    check_leakage(folds, dict(zip(v, g.tolist())))
    C.write_json(out, {"meta": {"seed": seed, "unit": "video_id", "stratify": "task_c_label",
                                "n_folds": len(folds), "n_videos": len(v), "git_sha": C.git_sha(),
                                "created": datetime.now().isoformat(timespec="seconds")},
                       "folds": folds})
    fam = vids["family"].to_dict()
    nfr = fr.groupby("video_id").size().to_dict()
    rows = [{"fold": f["fold"], "part": part, "family": fm, "videos": sum(1 for x in f[part] if fam[x] == fm),
             "frames": sum(nfr[x] for x in f[part] if fam[x] == fm)}
            for f in folds for part in ("train", "val", "test") for fm in ("healthy", "functional", "structural")]
    pd.DataFrame(rows).to_csv(C.path(cfg, "splits") / "split_counts.csv", index=False)
    return {"folds": len(folds), "sizes": [{k: len(f[k]) for k in ("train", "val", "test")} for f in folds]}


def load_splits(cfg: dict) -> list[dict]:
    p = C.path(cfg, "splits") / "folds.json"
    if not p.exists():
        raise SystemExit("splits ausentes: rode `python -m laringopdi splits`.")
    return json.loads(p.read_text(encoding="utf-8"))["folds"]


def fold_frames(cfg: dict, fold: int, part: str, task: str | None = None) -> pd.DataFrame:
    """Frames amostrados de uma partição do fold (opcionalmente só os com rótulo na tarefa)."""
    fr = load_frames(cfg)
    fr = fr[fr["video_id"].isin(set(load_splits(cfg)[fold][part]))]
    return fr[fr[task_col(task)] >= 0] if task else fr


# ----------------------------------------------------------------------------- cache do PDI
def _process_one(p, seg, pipeline, params, size, work_size):
    from laringopdi.pdi import run_pipeline
    from laringopdi.pdi.quality import image_quality

    gt = imread(seg, gray=True) > 127 if seg else None
    out, ctx = run_pipeline(imread(p, gray=True), pipeline, params, size, work_size, gt_mask=gt)
    masks = {k: np.packbits(ctx[k]) for k in ("gt_mask_lb", "spec_mask_lb", "roi_mask_lb", "glottis_mask_lb")
             if k in ctx}
    extra = {k: ctx[k] for k in ("roi_bbox", "spec_frac") if k in ctx}
    return out, image_quality(out, ctx), masks, extra


def cache_paths(cfg: dict, exp: str) -> dict:
    from laringopdi.pdi import pipeline_hash

    e = cfg["experiments"][exp]
    h = pipeline_hash(e["pipeline"], e.get("params"), cfg["data"]["image_size"], cfg["data"]["work_size"])
    base = C.path(cfg, "cache") / f"{exp}_{h}"
    return {"hash": h, "x": base.with_suffix(".npy"), "masks": base.with_suffix(".masks.npz"),
            "meta": base.with_suffix(".json")}


def build_cache(cfg: dict, exp: str, n_jobs: int = -1, force: bool = False) -> dict:
    """Aplica o pipeline do experimento aos frames amostrados e grava (N, 224, 224, C) uint8 +
    máscaras empacotadas + metadados + métricas de qualidade. Pula se já existir."""
    from joblib import Parallel, delayed

    from laringopdi.pdi import resolve_params

    e = cfg["experiments"][exp]
    if not e.get("pipeline"):
        raise SystemExit(f"{exp} sem pipeline definido.")
    cp = cache_paths(cfg, exp)
    qpath = C.path(cfg, "results") / "pdi_quality" / f"{exp}.csv"
    if cp["x"].exists() and cp["meta"].exists() and qpath.exists() and not force:
        return {"exp": exp, "skipped": True}
    fr = load_frames(cfg)
    raw, size, ws = C.path(cfg, "raw"), cfg["data"]["image_size"], cfg["data"]["work_size"]
    res = Parallel(n_jobs=n_jobs, batch_size=16)(
        delayed(_process_one)(raw / p, raw / s if s else None, e["pipeline"], e.get("params"), size, ws)
        for p, s in zip(fr["path"], fr["seg_path"]))
    X = np.stack([r[0] for r in res])
    np.save(cp["x"], X)
    mkeys = sorted({k for r in res for k in r[2]})
    empty = np.packbits(np.zeros((size, size), bool))
    np.savez_compressed(cp["masks"], **{k: np.stack([r[2].get(k, empty) for r in res]) for k in mkeys})
    q = pd.DataFrame([r[1] | {"path": p, "video_id": v, "family": f}
                      for r, p, v, f in zip(res, fr["path"], fr["video_id"], fr["family"])])
    q.insert(0, "exp", exp)
    qpath.parent.mkdir(parents=True, exist_ok=True)
    q.to_csv(qpath, index=False)
    C.write_json(cp["meta"], {"exp": exp, "pipeline": e["pipeline"],
                              "params": resolve_params(e["pipeline"], e.get("params")), "hash": cp["hash"],
                              "shape": list(X.shape), "paths": fr["path"].tolist(),
                              "video_ids": fr["video_id"].tolist(), "mask_keys": mkeys,
                              "extra": [r[3] for r in res], "git_sha": C.git_sha()})
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

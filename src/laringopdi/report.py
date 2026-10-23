"""Relatórios e figuras: EDA do dataset, galeria do PDI, tabelas finais (CSV + LaTeX)."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from laringopdi import config as C
from laringopdi import data as D


def _read_json(p):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _sample_per_class(cfg: dict, n: int = 1, seed: int = 0) -> pd.DataFrame:
    """Amostra fixa por classe, tirada do treino do fold 0 quando os splits existem."""
    df = D.load_index(cfg)
    sp = C.path(cfg, "splits") / "folds.json"
    if sp.exists():
        df = df[df["path"].isin(set(D.load_splits(cfg)[0]["train"]))]
    return (df.groupby("class", group_keys=False)
              .apply(lambda g: g.sample(min(n, len(g)), random_state=seed))
              .set_index("class").reindex([c for c in cfg["data"]["classes"] if c in set(df["class"])]))


def _show(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else (img[..., :3] if img.shape[2] >= 3 else img[..., 0])


# ----------------------------------------------------------------------------- EDA
def eda(cfg: dict) -> dict:
    df = D.load_index(cfg)
    figs, docs, logs = C.path(cfg, "figures"), C.path(cfg, "docs"), C.path(cfg, "logs")
    classes = cfg["data"]["classes"]
    counts = df["class"].value_counts().reindex(classes, fill_value=0)
    minc = cfg["data"]["min_class_count"]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(counts.index, counts.values, color=["#c0392b" if v < minc else "#2c6e91" for v in counts.values])
    ax.axhline(minc, ls="--", c="gray", lw=1)
    ax.set_ylabel("imagens"); ax.set_title("Imagens por classe (vermelho: < %d)" % minc)
    plt.xticks(rotation=30, ha="right"); fig.tight_layout(); fig.savefig(figs / "eda_class_counts.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(df["width"], df["height"], s=6, alpha=0.4)
    ax.set_xlabel("largura (px)"); ax.set_ylabel("altura (px)"); ax.set_title("Resolução das imagens")
    fig.tight_layout(); fig.savefig(figs / "eda_sizes.png", dpi=120); plt.close(fig)

    samp = _sample_per_class(cfg, 4)
    fig, axes = plt.subplots(4, len(samp.index.unique()), figsize=(2 * len(samp.index.unique()), 8))
    for j, c in enumerate(samp.index.unique()):
        rows = samp.loc[[c]]
        for i in range(4):
            a = axes[i, j]; a.axis("off")
            if i < len(rows):
                a.imshow(D.imread(C.path(cfg, "raw") / rows.iloc[i]["path"]))
            if i == 0:
                a.set_title(c, fontsize=8)
    fig.tight_layout(); fig.savefig(figs / "eda_samples.png", dpi=100); plt.close(fig)

    idx = _read_json(logs / "index_summary.json") or {}
    dd = _read_json(logs / "dedup_summary.json")
    split_counts = C.path(cfg, "splits") / "split_counts.csv"
    sizes = df.groupby(["width", "height"]).size().sort_values(ascending=False).head(5)
    task_rows = []
    for t in ("A", "C"):
        y = D.task_labels(df, cfg, t)
        task_rows.append(f"| {t} ({cfg['tasks'][t]['name']}) | " + " / ".join(
            f"{n}: {(y == i).sum()}" for i, n in enumerate(D.task_names(cfg, t))) + " |")
    small = [c for c in classes if counts[c] < minc]
    md = [
        "# Relatório do dataset — Laryngoscope8", "",
        f"Gerado por `python -m laringopdi eda` (git {C.git_sha()}). Não edite à mão.", "",
        "## Índice", "",
        f"- Imagens indexadas: **{len(df)}** (esperado: 3057; divergência: {len(df) - 3057:+d})",
        f"- ID de paciente: **{'sim' if idx.get('has_patient_id') else 'não'}**"
        + (f" ({idx.get('n_patients')} pacientes)" if idx.get("has_patient_id") else
           " — grupos anti-vazamento vêm de quase-duplicatas (pHash + SSIM)"),
        f"- Duplicatas exatas (SHA-1): {idx.get('exact_duplicates', '?')}; arquivos sem classe: "
        f"{idx.get('n_unknown_class', '?')}; ilegíveis: {len(idx.get('unreadable', []))}", "",
        "| Classe | Imagens | % |", "|---|---:|---:|",
        *[f"| {c}{' ⚠️' if counts[c] < minc else ''} | {counts[c]} | {100 * counts[c] / max(len(df), 1):.1f} |" for c in classes],
        "", "![classes](figures/eda_class_counts.png)", "",
        "## Tarefas binárias", "", "| Tarefa | Distribuição |", "|---|---|", *task_rows, "",
        "## Resolução", "", "Resoluções mais frequentes (L×A: n): "
        + ", ".join(f"{w}×{h}: {n}" for (w, h), n in sizes.items()), "",
        "![tamanhos](figures/eda_sizes.png)", "", "![amostras](figures/eda_samples.png)", "",
    ]
    if dd:
        md += ["## Deduplicação (pHash + SSIM)", "",
               f"- Parâmetros: Hamming ≤ {dd['params']['phash_max_dist']}, SSIM ≥ {dd['params']['ssim_min']} "
               f"(cinza {dd['params']['thumb']}×{dd['params']['thumb']})",
               f"- Pares candidatos: {dd['candidate_pairs']}; confirmados: {dd['confirmed_pairs']} "
               f"(entre classes diferentes: {dd['dup_pairs_cross_class']})",
               f"- Grupos: {dd['n_groups']} ({dd['n_groups_multi']} com mais de 1 imagem; maior: {dd['largest_group']})",
               "- Pares em `logs/dedup_pairs.csv` para conferência visual.", ""]
    if split_counts.exists():
        sc = pd.read_csv(split_counts).pivot_table(index=["fold", "part"], columns="class", values="n",
                                                   fill_value=0).reindex(columns=classes, fill_value=0)
        md += ["## Splits (StratifiedGroupKFold, 5 folds, agrupado)", "",
               "| fold | parte | " + " | ".join(classes) + " | total |", "|---|---|" + "---:|" * (len(classes) + 1),
               *[f"| {f} | {p} | " + " | ".join(str(int(v)) for v in r.values) + f" | {int(r.sum())} |"
                 for (f, p), r in sc.iterrows()], ""]
    md += ["## Classes com menos de %d imagens" % minc, "",
           ("Sinalizadas: " + ", ".join(small) + ". **Decisão de método pendente com o usuário.**") if small
           else "Nenhuma.", ""]
    (docs / "dataset_report.md").write_text("\n".join(md), encoding="utf-8")
    return {"n": len(df), "small_classes": small}


# ----------------------------------------------------------------------------- galeria
def gallery(cfg: dict) -> dict:
    from laringopdi.pdi import enhance, geometry, run_pipeline
    from laringopdi.pdi.registry import to_work_size

    figs, raw = C.path(cfg, "figures"), C.path(cfg, "raw")
    ws, size = cfg["data"]["work_size"], cfg["data"]["image_size"]
    samp = _sample_per_class(cfg, 1, seed=1)
    imgs = {c: D.imread(raw / r["path"]) for c, r in samp.iterrows()}
    exps = [e for e in cfg["stage1"]["experiments"]]

    fig, axes = plt.subplots(len(exps), len(imgs), figsize=(1.6 * len(imgs), 1.6 * len(exps)))
    for i, e in enumerate(exps):
        ex = cfg["experiments"][e]
        for j, (c, img) in enumerate(imgs.items()):
            out, _ = run_pipeline(img, ex["pipeline"], ex.get("params"), size, ws)
            a = axes[i, j]; a.imshow(_show(out), cmap="gray"); a.set_xticks([]); a.set_yticks([])
            if i == 0: a.set_title(c, fontsize=7)
            if j == 0: a.set_ylabel(f"{e}\n{ex['pipeline']}", fontsize=6)
    fig.tight_layout(); fig.savefig(figs / "gallery_grid.png", dpi=110); plt.close(fig)

    picks = list(imgs.values())[:3]
    # spec passo a passo
    fig, axes = plt.subplots(len(picks), 4, figsize=(10, 2.6 * len(picks)))
    for i, img in enumerate(picks):
        ctx: dict = {}
        x = geometry.fov(to_work_size(img, ws), ctx)
        m0 = enhance.specular_mask(x)
        y = enhance.spec(x, ctx)
        over = x.copy(); over[ctx["spec_mask"]] = (0, 255, 0)
        for a, im, t in zip(axes[i], [x, m0, over, y], ["FOV", "máscara HSV", "dilatada", "inpaint TELEA"]):
            a.imshow(im, cmap="gray"); a.set_title(t, fontsize=8); a.axis("off")
    fig.tight_layout(); fig.savefig(figs / "gallery_spec_steps.png", dpi=110); plt.close(fig)

    # homomórfica: espectro e H(u,v)
    p = cfg["experiments"]["E05"].get("params", {}).get("homo", {})
    hp = {**{"gl": 0.5, "gh": 1.75, "d0": 30, "c": 1.0}, **p}
    ctx = {}
    x = geometry.fov(to_work_size(picks[0], ws), ctx)
    L = cv2.cvtColor(x, cv2.COLOR_RGB2LAB)[..., 0]
    Z = np.fft.fftshift(np.fft.fft2(np.log1p(L.astype(float))))
    H = enhance.homo_transfer(L.shape, **hp)
    y = enhance.homo(x, ctx, **hp)
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.4))
    for a, im, t in zip(axes, [L, np.log1p(np.abs(Z)), H, np.log1p(np.abs(Z * H)),
                               cv2.cvtColor(y, cv2.COLOR_RGB2LAB)[..., 0]],
                        ["L (FOV)", "log|F{ln L}|", f"H(u,v) γL={hp['gl']} γH={hp['gh']} D0={hp['d0']}",
                         "log|H·F|", "L filtrado"]):
        im_ = a.imshow(im, cmap="gray" if t[0] == "L" else "magma"); a.set_title(t, fontsize=8); a.axis("off")
    fig.tight_layout(); fig.savefig(figs / "gallery_homo.png", dpi=110); plt.close(fig)

    # ROI: Otsu × adaptativa
    fig, axes = plt.subplots(len(picks), 5, figsize=(13, 2.6 * len(picks)))
    for i, img in enumerate(picks):
        ctx = {}
        x = geometry.fov(to_work_size(img, ws), ctx)
        cols = [(x, "FOV")]
        for method in ("otsu", "adaptive"):
            c2 = dict(ctx)
            m = geometry.roi_candidate(x, c2, method)
            crop = geometry.roi(x, c2, method=method)
            vis = x.copy()
            bx, by, bw, bh = c2["roi_bbox"]
            cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 255, 0), 3)
            cols += [(m if m is not None else np.zeros(x.shape[:2]), f"{method}: componente"),
                     (vis, f"{method}: caixa{' (fallback)' if c2['roi_fallback'] else ''}")]
        for a, (im, t) in zip(axes[i], cols):
            a.imshow(im, cmap="gray"); a.set_title(t, fontsize=8); a.axis("off")
    fig.tight_layout(); fig.savefig(figs / "gallery_roi.png", dpi=110); plt.close(fig)
    return {"figures": ["gallery_grid.png", "gallery_spec_steps.png", "gallery_homo.png", "gallery_roi.png"]}

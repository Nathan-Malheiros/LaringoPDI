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

FAMILIES = ["healthy", "functional", "structural"]


def _read_json(p):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _frames_or_index(cfg: dict) -> pd.DataFrame:
    sp = C.path(cfg, "splits") / "frames.json"
    if sp.exists():
        return D.load_frames(cfg)
    df = D.load_index(cfg)
    return df[(df["task_a_label"] >= 0) & (df["seg_area"] > 0)]


def _sample(cfg: dict, n: int = 2, seed: int = 0) -> pd.DataFrame:
    """Amostra fixa por família, do treino do fold 0 quando os splits existem."""
    df = _frames_or_index(cfg)
    if (C.path(cfg, "splits") / "folds.json").exists():
        df = df[df["video_id"].isin(set(D.load_splits(cfg)[0]["train"]))]
    df = df.drop_duplicates("video_id")
    return pd.concat([df[df["family"] == f].sample(min(n, (df["family"] == f).sum()), random_state=seed)
                      for f in FAMILIES if (df["family"] == f).any()])


def _md_table(df: pd.DataFrame) -> list[str]:
    cols = list(df.columns)
    return ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols),
            *["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]]


# ----------------------------------------------------------------------------- EDA
def eda(cfg: dict) -> dict:
    from laringopdi.pdi.enhance import specular_mask

    df = D.load_index(cfg)
    figs, docs, logs = C.path(cfg, "figures"), C.path(cfg, "docs"), C.path(cfg, "logs")
    raw = C.path(cfg, "raw")
    vids = df.drop_duplicates("video_id")
    idx = _read_json(logs / "index_summary.json") or {}
    minc = cfg["data"]["min_class_count"]

    fam = pd.DataFrame({"videos": vids["family"].value_counts(), "frames": df["family"].value_counts()}) \
        .fillna(0).astype(int).reindex(FAMILIES + ["other", "unknown"], fill_value=0)
    smap_p = C.path(cfg, "tables") / "status_map.csv"
    smap = pd.read_csv(smap_p, keep_default_na=False) if smap_p.exists() else pd.DataFrame()

    # amostra de frames para estatísticas de intensidade (máx. 3 por vídeo)
    fr = _frames_or_index(cfg).groupby("video_id", group_keys=False).head(3)
    hist = {f: np.zeros(256) for f in FAMILIES}
    stats = []
    for _, r in fr.iterrows():
        g = D.imread(raw / r["path"], gray=True)
        if r["family"] in hist:
            hist[r["family"]] += np.bincount(g.ravel(), minlength=256) / g.size
        stats.append({"family": r["family"], "spec_frac": float(specular_mask(g).mean()),
                      "mean": float(g.mean()), "seg_area": r["seg_area"]})
    st = pd.DataFrame(stats)

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    axes[0].bar(fam.index, fam["videos"], color=["#c0392b" if v < minc else "#2c6e91" for v in fam["videos"]])
    axes[0].axhline(minc, ls="--", c="gray", lw=1); axes[0].set_title(f"Vídeos por família (vermelho: < {minc})")
    axes[0].tick_params(axis="x", rotation=30)
    for f in FAMILIES:
        if hist[f].sum():
            axes[1].plot(hist[f] / hist[f].sum(), label=f)
    axes[1].set_title("Histograma de cinza médio"); axes[1].legend(fontsize=7)
    for f in FAMILIES:
        s = st.loc[st["family"] == f, "spec_frac"]
        if len(s):
            axes[2].hist(s, bins=30, alpha=0.5, label=f)
    axes[2].set_title("Fração especular por frame"); axes[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(figs / "eda_overview.png", dpi=110); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    res = df.groupby(["width", "height"]).size().reset_index(name="n")
    ax.scatter(res["width"], res["height"], s=10 + 200 * res["n"] / res["n"].max(), alpha=0.6)
    ax.set_xlabel("largura (px)"); ax.set_ylabel("altura (px)"); ax.set_title("Resoluções (área ∝ nº de frames)")
    fig.tight_layout(); fig.savefig(figs / "eda_sizes.png", dpi=110); plt.close(fig)

    samp = _sample(cfg, 3)
    fig, axes = plt.subplots(len(FAMILIES), 3, figsize=(9, 7), squeeze=False)
    for i, f in enumerate(FAMILIES):
        rows = samp[samp["family"] == f]
        for j in range(3):
            a = axes[i, j]; a.axis("off")
            if j < len(rows):
                a.imshow(D.imread(raw / rows.iloc[j]["path"], gray=True), cmap="gray")
                a.set_title(f"{f}: {rows.iloc[j]['disorder_raw'][:28]}", fontsize=7)
    fig.tight_layout(); fig.savefig(figs / "eda_samples.png", dpi=100); plt.close(fig)

    tasks = pd.DataFrame([{"tarefa": t, **{n: int((vids[D.task_col(t)] == i).sum())
                                            for i, n in enumerate(D.task_names(cfg, t))}} for t in "ABC"]).fillna(0)
    site_col = next((c for c in df.columns if c.startswith("meta_") and any(k in c for k in ("hospital", "institution", "site", "camera"))), None)
    small = [f for f in FAMILIES if fam.loc[f, "videos"] < minc]
    md = ["# Relatório do dataset — BAGLS", "",
          f"Gerado por `python -m laringopdi eda` (git {C.git_sha()}). Não edite à mão.", "",
          "## Contagens", "",
          f"- Frames: **{len(df)}** (esperado ~59.250); vídeos: **{len(vids)}** (esperado 640)",
          f"- Vídeos RGB: {int(vids['is_rgb'].sum())} (todos são convertidos para cinza)",
          f"- Frames com máscara: {int(df['has_seg'].sum())}; com glote visível (máscara não vazia): {int((df['seg_area'] > 0).sum())}",
          f"- ID de sujeito nos metadados: {idx.get('subject_id_column') or 'não (só ID de vídeo)'}",
          f"- Split oficial (vídeos): {vids['official_split'].value_counts().to_dict()} — ignorado nos folds", "",
          "### Por família", "", *_md_table(fam.reset_index().rename(columns={"index": "família"})), "",
          "### Texto do status → família (revisão humana; regras em `configs/label_map.yaml`)", "",
          *_md_table(smap.assign(disorder_raw=smap.get("disorder_raw", pd.Series(dtype=str))
                                 .map(lambda x: f"`{x}`" if x else "*(vazio)*"))), "",
          "### Tarefas (vídeos)", "", *_md_table(tasks), "",
          "![visão geral](figures/eda_overview.png)", "", "![resoluções](figures/eda_sizes.png)", "",
          "![amostras](figures/eda_samples.png)", ""]
    if len(st):
        md += ["### Intensidade (até 3 frames por vídeo)", "",
               *_md_table(st.groupby("family")[["mean", "spec_frac", "seg_area"]].mean().round(4).reset_index()), ""]
    if site_col:
        md += [f"### Por `{site_col}` (vídeos)", "",
               *_md_table(vids.groupby([site_col, "family"]).size().unstack(fill_value=0).reset_index()), ""]
    fj = _read_json(C.path(cfg, "splits") / "frames.json")
    if fj:
        md += ["## Amostragem de frames", "", f"- {fj['meta']['n_frames']} frames de {fj['meta']['n_videos']} vídeos "
               f"(até {fj['meta']['frames_per_video']} por vídeo; critério: {fj['meta']['criterion']})", ""]
    dd = _read_json(logs / "dedup_summary.json")
    if dd:
        md += ["## Vídeos quase-duplicados (pHash do frame central + SSIM)", "",
               f"- Hamming ≤ {dd['params']['phash_max_dist']}, SSIM ≥ {dd['params']['ssim_min']}: "
               f"{dd['candidate_pairs']} pares candidatos, {dd['confirmed_pairs']} confirmados; "
               f"{dd['n_groups']} grupos ({dd['n_groups_multi']} com mais de 1 vídeo; maior: {dd['largest_group']})", ""]
    scp = C.path(cfg, "splits") / "split_counts.csv"
    if scp.exists():
        sc = pd.read_csv(scp).pivot_table(index=["fold", "part"], columns="family", values="videos", fill_value=0)
        md += ["## Splits (StratifiedGroupKFold por vídeo, 5 folds) — vídeos", "",
               *_md_table(sc.reset_index()), ""]
    md += [f"## Famílias com menos de {minc} vídeos", "",
           ("Sinalizadas: " + ", ".join(small) + ". **Decisão de método pendente com o usuário.**") if small else "Nenhuma.", ""]
    (docs / "dataset_report.md").write_text("\n".join(md), encoding="utf-8")
    return {"frames": len(df), "videos": len(vids), "small_families": small}


# ----------------------------------------------------------------------------- galeria
def gallery(cfg: dict) -> dict:
    from laringopdi.pdi import enhance, run_pipeline
    from laringopdi.pdi.registry import to_work_size
    from laringopdi.pdi.segmentation import best_config, segment

    figs, raw = C.path(cfg, "figures"), C.path(cfg, "raw")
    ws, size = cfg["data"]["work_size"], cfg["data"]["image_size"]
    samp = _sample(cfg, 2, seed=1)
    items = [(r["family"], D.imread(raw / r["path"], gray=True), D.imread(raw / r["seg_path"], gray=True) > 127)
             for _, r in samp.iterrows()]
    exps = cfg["stage1"]["experiments"]

    fig, axes = plt.subplots(len(exps), len(items), figsize=(1.7 * len(items), 1.7 * len(exps)), squeeze=False)
    for i, e in enumerate(exps):
        ex = cfg["experiments"][e]
        for j, (fam, g, gt) in enumerate(items):
            out, _ = run_pipeline(g, ex["pipeline"], ex.get("params"), size, ws, gt_mask=gt)
            a = axes[i, j]; a.imshow(out[..., :3] if out.shape[2] >= 3 else out[..., 0], cmap="gray")
            a.set_xticks([]); a.set_yticks([])
            if i == 0: a.set_title(fam, fontsize=7)
            if j == 0: a.set_ylabel(f"{e}\n{ex['pipeline']}", fontsize=6)
    fig.tight_layout(); fig.savefig(figs / "gallery_grid.png", dpi=110); plt.close(fig)

    picks = items[:3]
    fig, axes = plt.subplots(len(picks), 4, figsize=(10, 2.6 * len(picks)), squeeze=False)
    for i, (_, g, _) in enumerate(picks):
        x = to_work_size(g, ws)
        ctx: dict = {}
        m0 = enhance.specular_mask(x)
        y = enhance.spec(x, ctx)
        over = np.dstack([x, x, x]).copy(); over[ctx["spec_mask"]] = (0, 255, 0)
        for a, im, t in zip(axes[i], [x, m0, over, y], ["original", "limiar (cauda)", "dilatada", "inpaint TELEA"]):
            a.imshow(im, cmap="gray"); a.set_title(t, fontsize=8); a.axis("off")
    fig.tight_layout(); fig.savefig(figs / "gallery_spec_steps.png", dpi=110); plt.close(fig)

    hp = {"gl": 0.5, "gh": 1.75, "d0": 30, "c": 1.0, **cfg["experiments"]["E04"].get("params", {}).get("homo", {})}
    x = to_work_size(picks[0][1], ws)
    Z = np.fft.fftshift(np.fft.fft2(np.log1p(x.astype(float))))
    H = enhance.homo_transfer(x.shape, **hp)
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.4))
    for a, im, t, cm in zip(axes, [x, np.log1p(np.abs(Z)), H, np.log1p(np.abs(Z * H)), enhance.homo(x, {}, **hp)],
                            ["original", "log|F{ln f}|", f"H(u,v) γL={hp['gl']} γH={hp['gh']} D0={hp['d0']}",
                             "log|H·F|", "filtrada"], ["gray", "magma", "magma", "magma", "gray"]):
        a.imshow(im, cmap=cm); a.set_title(t, fontsize=8); a.axis("off")
    fig.tight_layout(); fig.savefig(figs / "gallery_homo.png", dpi=110); plt.close(fig)

    b = best_config()
    fig, axes = plt.subplots(len(picks), 4, figsize=(11, 2.6 * len(picks)), squeeze=False)
    for i, (_, g, gt) in enumerate(picks):
        x = to_work_size(g, ws)
        gtw = cv2.resize(gt.astype(np.uint8), x.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
        m = segment(x, b["method"], **b["params"])
        c1, c2 = {"gt_mask": gtw}, {}
        from laringopdi.pdi.geometry import roi, roi_gt
        roi_gt(x, c1); roi(x, c2, method=b["method"], seg_params=b["params"], seg_pre=b.get("preprocess", "raw"))
        vis = np.dstack([x, x, x]).copy()
        for (bx, by, bw, bh), col in ((c1["roi_bbox"], (0, 255, 0)), (c2["roi_bbox"], (255, 0, 0))):
            cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), col, 2)
        for a, im, t in zip(axes[i], [x, gtw, m, vis], ["original", "ground truth", f"clássica: {b['method']}",
                                                         "caixas: verde=roi_gt, vermelho=roi"]):
            a.imshow(im, cmap="gray"); a.set_title(t, fontsize=8); a.axis("off")
    fig.tight_layout(); fig.savefig(figs / "gallery_roi.png", dpi=110); plt.close(fig)
    return {"figures": ["gallery_grid.png", "gallery_spec_steps.png", "gallery_homo.png", "gallery_roi.png"]}


# ----------------------------------------------------------------------------- doc das técnicas
def pdi_doc(cfg: dict) -> str:
    """Gera docs/pdi_tecnicas.md a partir das docstrings (fonte única) e da matriz de experimentos."""
    import inspect
    import re

    from laringopdi.pdi import TECHNIQUES, resolve_params
    from laringopdi.pdi.segmentation import PARAM_GRID, SEGMENTERS, best_config

    def parts(f):
        d = inspect.getdoc(f) or ""
        keys = ["O que faz", "Por que importa", "Parâmetros padrão", "Referência"]
        out = {}
        for i, k in enumerate(keys):
            nxt = "|".join(re.escape(x) for x in keys[i + 1:]) or "$^"
            m = re.search(rf"{re.escape(k)}:\s*(.*?)(?=\n?(?:{nxt}):|\Z)", d, re.S)
            out[k] = " ".join(m.group(1).split()) if m else ""
        return out

    md = ["# Técnicas de PDI — LaringoPDI (BAGLS, escala de cinza)", "",
          f"Gerado por `python -m laringopdi gallery` a partir das docstrings (git {C.git_sha()}). Não edite à mão.", "",
          "Toda imagem é convertida para cinza e reduzida a lado maior ≤ "
          f"{cfg['data']['work_size']} px antes do PDI; ao final, letterbox {cfg['data']['image_size']}×"
          f"{cfg['data']['image_size']} e replicação para 3 canais. Parâmetros vêm da literatura, nunca do classificador.", "",
          "## Matriz de experimentos", "", "| ID | Pipeline | Descrição |", "|---|---|---|",
          *[f"| {e} | `{v['pipeline']}` | {v['desc']} |" for e, v in cfg["experiments"].items()], "",
          "## Técnicas", ""]
    for name, f in TECHNIQUES.items():
        p = parts(f)
        md += [f"### `{name}`", "", f"- **O que faz:** {p['O que faz']}", f"- **Por que importa na laringoscopia:** {p['Por que importa']}",
               f"- **Parâmetros padrão:** {p['Parâmetros padrão']}", f"- **Referência:** {p['Referência']}", ""]
    b = best_config()
    md += ["## Segmentação clássica da glote (Fase 3)", "",
           f"Melhor configuração atual (usada por `roi` e `m2`): **{b['method']}** sobre **{b.get('preprocess', 'raw')}**, "
           f"parâmetros `{json.dumps(b['params'])}`" + (f" — {b['selected_on']}." if b.get("selected_on") else " (padrão, antes da Fase 3)."), ""]
    for name, f in SEGMENTERS.items():
        p = parts(f)
        md += [f"### `{name}`", "", f"- **O que faz:** {p['O que faz']}", f"- **Por que importa na laringoscopia:** {p['Por que importa']}",
               f"- **Parâmetros padrão:** {p['Parâmetros padrão']} · grade de ajuste: `{json.dumps(PARAM_GRID[name])}`",
               f"- **Referência:** {p['Referência']}", ""]
    md += ["## Figuras para revisão (Marcos Vinicius; segmentação: Marcos Mendes e Tales)", "",
           "Geradas localmente em `docs/figures/` e **não versionadas** (contêm frames do BAGLS): "
           "`gallery_grid.png`, `gallery_spec_steps.png`, `gallery_homo.png`, `gallery_roi.png`, `segmentation_examples.png`.", ""]
    out = C.path(cfg, "docs") / "pdi_tecnicas.md"
    out.write_text("\n".join(md), encoding="utf-8")
    return str(out)

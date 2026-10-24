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


# ----------------------------------------------------------------------------- tabelas finais
def to_latex(df: pd.DataFrame, path, caption: str) -> None:
    """Tabela LaTeX simples (booktabs), sem dependências extras."""
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v).replace("_", r"\_").replace("%", r"\%")
    lines = [r"\begin{table}[ht]", r"\centering", rf"\caption{{{caption}}}", r"\begin{tabular}{" + "l" * df.shape[1] + "}",
             r"\toprule", " & ".join(fmt(c) for c in df.columns) + r" \\", r"\midrule",
             *[" & ".join(fmt(v) for v in r) + r" \\" for r in df.itertuples(index=False)],
             r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines), encoding="utf-8")


def _save_table(cfg, df: pd.DataFrame, name: str, caption: str) -> None:
    t = C.path(cfg, "tables")
    df.round(4).to_csv(t / f"{name}.csv", index=False)
    to_latex(df.round(3), t / f"{name}.tex", caption)


def cmd_report(cfg: dict, args) -> dict:
    """Gera vereditos, H4 da segmentação, clássico × CNN e docs/hipoteses.md com o que já existe."""
    from laringopdi.evaluate import cmd_summary, collect_runs, seg_h4_test, verdicts

    cmd_summary(cfg, args)
    runs = collect_runs(cfg)
    out, made = {}, []
    for model in sorted(set(runs["model"]) & {"probe", "simplecnn_fast", "resnet18", "simplecnn"}):
        v = verdicts(cfg, model)
        if not v.empty:
            _save_table(cfg, v, f"verdicts_{model}", f"Vereditos do PDI vs E00 ({model}, F1-macro por vídeo)")
            out[model] = v.groupby(["task", "verdict"]).size().unstack(fill_value=0).to_dict("index")
            made.append(model)
    h4 = seg_h4_test(cfg)
    _save_table(cfg, h4, "seg_h4", "H4: Dice por vídeo com realce vs raw (Wilcoxon, Holm)")
    cls = runs[runs["exp"].str.startswith("C")]
    cnn = runs[runs["model"].isin(["probe", "resnet18", "simplecnn"]) & (runs["exp"] == "E00")]
    cmp_ = pd.concat([cls, cnn]).groupby(["exp", "model", "task"]).agg(
        f1=("video_f1_macro", "mean"), f1_sd=("video_f1_macro", "std"), auc=("video_auc", "mean"),
        bal_acc=("video_bal_acc", "mean"), n=("fold", "count")).reset_index()
    if not cmp_.empty:
        _save_table(cfg, cmp_, "classic_vs_cnn", "Braço clássico × CNN (E00), por vídeo")
    _hypotheses_doc(cfg, runs, h4, cmp_, made)
    return {"verdicts": out, "h4_melhora": int((h4["verdict"] == "melhora").sum()), "h4_total": len(h4)}


def paired_compare(cfg: dict, a: tuple, b: tuple, task: str) -> dict:
    """Compara dois sistemas (exp, modelo) por vídeo, OOF, nas sementes comuns: ΔF1 = b − a,
    IC 95% bootstrap e McNemar. Usado em H5 (roi × roi_gt) e H6 (clássico × CNN)."""
    from sklearn.metrics import f1_score
    from laringopdi.evaluate import bootstrap_delta_f1, mcnemar_p, oof_video_preds, run_seeds

    seeds = sorted(run_seeds(cfg, a[0], task, a[1]) & run_seeds(cfg, b[0], task, b[1]))
    if not seeds:
        return {}
    m = oof_video_preds(cfg, a[0], task, a[1], seeds).merge(
        oof_video_preds(cfg, b[0], task, b[1], seeds), on=["video_id", "y"], suffixes=("_a", "_b"))
    y, pa, pb = m["y"].to_numpy(), m["pred_a"].to_numpy(), m["pred_b"].to_numpy()
    lo, hi = bootstrap_delta_f1(y, pa, pb, cfg["stats"]["n_boot"])
    fa, fb = f1_score(y, pa, average="macro"), f1_score(y, pb, average="macro")
    return {"f1_a": fa, "f1_b": fb, "delta": fb - fa, "ci_low": lo, "ci_high": hi,
            "mcnemar_p": mcnemar_p(pa == y, pb == y), "n_seeds": len(seeds), "n_videos": len(y)}


def _fmt_cmp(r: dict) -> str:
    return "—" if not r else (f"F1 {r['f1_a']:.3f} → {r['f1_b']:.3f}, Δ={r['delta']:+.3f} "
                              f"[IC {r['ci_low']:+.3f}; {r['ci_high']:+.3f}], McNemar p={r['mcnemar_p']:.2g}")


def _hypotheses_doc(cfg, runs, h4, cmp_, made) -> None:
    """docs/hipoteses.md: evidência e conclusão explícita de H1–H6, com as regras fixadas em DECISIONS.md."""
    from scipy.stats import wilcoxon
    from laringopdi.evaluate import verdicts

    t = C.path(cfg, "tables")
    V = {m: pd.read_csv(t / f"verdicts_{m}.csv") for m in made if (t / f"verdicts_{m}.csv").exists()}
    labels = {"probe": "probe (ResNet congelada)", "simplecnn_fast": "SimpleCNN rápida", "resnet18": "ResNet-18 fine-tuning",
              "simplecnn": "SimpleCNN completa"}
    main = next((m for m in ("resnet18", "probe") if m in V), None)

    def row(m, e, task):
        df = V.get(m, pd.DataFrame())
        r = df[(df["exp"] == e) & (df["task"] == task)] if not df.empty else df
        return None if r.empty else r.iloc[0]

    def fmt(r):
        return "—" if r is None else (f"ΔF1={r['oof_delta_f1']:+.3f} [IC {r['ci_low']:+.3f}; {r['ci_high']:+.3f}], "
                                      f"{int(r['n_pairs'])} pares, **{r['verdict']}**")

    def ev(e):
        return [f"  - {labels[m]}: A {fmt(row(m, e, 'A'))}; B {fmt(row(m, e, 'B'))}." for m in V if row(m, e, "A") is not None]

    L = ["# Hipóteses — estado da evidência", "",
         f"Gerado por `python -m laringopdi report` (git {C.git_sha()}). Não edite à mão. Modelos com resultados: "
         f"{', '.join(labels.get(m, m) for m in made) or 'nenhum'}. Hipóteses pré-registradas e regras de veredito em DECISIONS.md.", "",
         "Veredito de cada técnica contra o E00 (F1-macro por vídeo): **melhora/piora** só se o IC 95% bootstrap excluir 0 "
         "e o McNemar OOF (ou, com ≥ 10 pares fold×semente, o Wilcoxon) corrigido por Holm der p < 0,05; senão **neutra**.", "",
         "## Resumo dos vereditos", "", "| modelo | tarefa | melhora | neutra | piora |", "|---|---|---:|---:|---:|"]
    for m, df in V.items():
        for task, g in df.groupby("task"):
            c = g["verdict"].value_counts()
            L.append(f"| {labels[m]} | {task} | {c.get('melhora', 0)} | {c.get('neutra', 0)} | {c.get('piora', 0)} |")
    concl = {}

    # H1 ---------------------------------------------------------------------------------------
    L += ["", "## H1 — `spec` mantém ou melhora o F1 e reduz a atenção sobre os reflexos", "- F1 (E01 × E00):", *ev("E01")]
    f1_ok = main is not None and all(row(main, "E01", k) is None or row(main, "E01", k)["verdict"] != "piora" for k in "AB")
    att_p = t / "attention_frames.csv.gz"
    att_ok = None
    if att_p.exists():
        af = pd.read_csv(att_p)
        for m in sorted(set(af["model"])):
            for task in ("A", "B"):
                g = af[(af["model"] == m) & (af["task"] == task)]
                pr = g[g["exp"] == "E00"].merge(g[g["exp"] == "E01"], on=["fold", "path"], suffixes=("_0", "_1"))
                pr = pr.dropna(subset=["att_spec_0", "att_spec_1"])
                if len(pr) < 10:
                    continue
                d = pr["att_spec_1"] - pr["att_spec_0"]
                p = float(wilcoxon(pr["att_spec_1"], pr["att_spec_0"]).pvalue) if (d != 0).any() else 1.0
                L.append(f"- Atenção nos reflexos, {labels.get(m, m)} {task}: E00 {pr['att_spec_0'].mean():.3f} → E01 "
                         f"{pr['att_spec_1'].mean():.3f} (área dos reflexos {pr['area_spec_0'].mean():.3f} = nível do acaso; "
                         f"mediana Δ {d.median():+.4f}; Wilcoxon p={p:.2g}; {len(pr)} frames).")
                ok = d.median() < 0 and p < 0.05
                att_ok = ok if att_ok is None else (att_ok and ok)
    if att_ok is None:
        L.append("- Atenção (Grad-CAM): sem pares E00 × E01 (`python -m laringopdi gradcam` com pesos do E01).")
    concl["H1"] = ("não testada (atenção)" if att_ok is None else
                   "suportada" if (att_ok and f1_ok) else "parcialmente suportada" if (att_ok or f1_ok) else "não suportada")

    # H2 ---------------------------------------------------------------------------------------
    L += ["", "## H2 — CLAHE/homomórfica > HE global (classificação e segmentação)", "- Contra o E00:",
          *ev("E02"), *ev("E03"), *ev("E04"), "- Confronto direto contra o HE (E02 como baseline):"]
    h2_cls = []
    for m in [m for m in ("probe", "simplecnn_fast") if m in V]:
        vh = verdicts(cfg, m, baseline="E02", exps=["E03", "E04"])
        for r in vh.itertuples():
            L.append(f"  - {labels[m]} {r.task}, {r.pipeline} − he: ΔF1={r.oof_delta_f1:+.3f} "
                     f"[IC {r.ci_low:+.3f}; {r.ci_high:+.3f}], **{r.verdict}**.")
            h2_cls.append(r.verdict)
    s4 = h4.set_index(["method", "preprocess"])
    L.append(f"- Segmentação (crescimento de região, método escolhido): Δ Dice vs raw — clahe "
             f"{s4.loc[('region_grow', 'clahe'), 'delta']:+.3f}, homo {s4.loc[('region_grow', 'homo'), 'delta']:+.3f}. "
             "O HE não entrou no estudo de segmentação (escopo da Seção 5).")
    concl["H2"] = ("suportada" if h2_cls and all(v == "melhora" for v in h2_cls) else
                   "parcialmente suportada" if "melhora" in h2_cls else
                   "não suportada (sem diferença detectável na classificação)" if h2_cls else "não testada")

    # H3 ---------------------------------------------------------------------------------------
    L += ["", "## H3 — o PDI ajuda mais a SimpleCNN do que a ResNet",
          "Diferença de ΔF1 (técnica − E00) entre a SimpleCNN e a ResNet nas mesmas técnicas × tarefas (Wilcoxon pareado)."]
    h3 = []
    for sm, rm in (("simplecnn", "resnet18"), ("simplecnn_fast", "probe")):
        if sm in V and rm in V:
            mm = V[sm].merge(V[rm], on=["task", "exp"], suffixes=("_s", "_r"))
            if len(mm) < 3:
                continue
            d = mm["oof_delta_f1_s"] - mm["oof_delta_f1_r"]
            p = float(wilcoxon(mm["oof_delta_f1_s"], mm["oof_delta_f1_r"]).pvalue) if (d != 0).any() else 1.0
            L.append(f"- {labels[sm]} × {labels[rm]}: ΔF1 médio {mm['oof_delta_f1_s'].mean():+.3f} × "
                     f"{mm['oof_delta_f1_r'].mean():+.3f}; mediana da diferença {d.median():+.3f}; p={p:.2g}; "
                     f"{len(mm)} pares técnica×tarefa.")
            h3.append(bool(d.median() > 0 and p < 0.05))
    concl["H3"] = "não testada" if not h3 else "suportada" if all(h3) else "parcialmente suportada" if any(h3) else "não suportada"

    # H4 ---------------------------------------------------------------------------------------
    L += ["", "## H4 — o realce melhora o Dice da segmentação clássica", "",
          "| método | pré | Dice raw | Dice pré | Δ | p (Holm) | veredito |", "|---|---|---:|---:|---:|---:|---|",
          *[f"| {r.method} | {r.preprocess} | {r.dice_raw:.3f} | {r.dice_pre:.3f} | {r.delta:+.3f} | {r.p_holm:.2g} | {r.verdict} |"
            for r in h4.itertuples()], "",
          "- Com ~587 vídeos pareados, p pequenos acompanham efeitos desprezíveis (Otsu, |Δ| ≤ 0,002); leia o Δ."]
    nm, npi = int((h4["verdict"] == "melhora").sum()), int((h4["verdict"] == "piora").sum())
    best = h4.loc[h4["dice_pre"].idxmax()]
    concl["H4"] = (f"parcialmente suportada (depende do par método × realce: {nm} melhoram, {npi} pioram; "
                   f"melhor = {best['method']} + {best['preprocess']}, Δ {best['delta']:+.3f})")

    # H5 ---------------------------------------------------------------------------------------
    L += ["", "## H5 — a ROI ajuda mais na Tarefa B do que na A, e `roi` se aproxima de `roi_gt`",
          "- E08 (`roi`) × E00:", *ev("E08"), "- E07 (`roi_gt`) × E00:", *ev("E07"), "- `roi` × `roi_gt` (E07 como referência):"]
    h5a, h5b = [], []
    for m in V:
        rA, rB = row(m, "E08", "A"), row(m, "E08", "B")
        if rA is not None and rB is not None:
            L.append(f"  - {labels[m]}: ΔF1(B) − ΔF1(A) do `roi` = {rB['oof_delta_f1'] - rA['oof_delta_f1']:+.3f}.")
            h5a.append(bool(rB["oof_delta_f1"] > rA["oof_delta_f1"]))
        for task in ("A", "B"):
            r = paired_compare(cfg, ("E07", m), ("E08", m), task)
            if r:
                L.append(f"  - {labels[m]} {task}: roi_gt → roi {_fmt_cmp(r)}.")
                h5b.append(bool(r["ci_low"] <= 0 <= r["ci_high"] or r["delta"] > 0))
    concl["H5"] = ("não testada" if not h5a else "suportada" if all(h5a) and all(h5b) else
                   "parcialmente suportada" if any(h5a) or any(h5b) else "não suportada")
    L.append("- Ressalva: \"ajuda mais em B\" compara efeitos que, isoladamente, podem ser neutros; \"se aproxima\" é a leitura "
             "de um IC que cruza 0 (ausência de diferença detectável, não equivalência demonstrada).")

    # H6 ---------------------------------------------------------------------------------------
    cnn = "resnet18" if "resnet18" in V else "probe"
    L += ["", "## H6 — forma glótica + MLP é competitiva com a CNN na Tarefa A e inferior na B",
          f"Referência CNN: E00 com {labels[cnn]}. Competitiva = IC do ΔF1 (clássico − CNN) contém 0 ou é > 0; inferior = IC < 0."]
    if not cmp_.empty:
        L += ["", "| exp | modelo | tarefa | F1 vídeo | dp | AUC |", "|---|---|---|---:|---:|---:|",
              *[f"| {r.exp} | {r.model} | {r.task} | {r.f1:.3f} | {r.f1_sd:.3f} | {r.auc:.3f} |" for r in cmp_.itertuples()], ""]
    res6 = {}
    for task in ("A", "B"):
        r = paired_compare(cfg, ("E00", cnn), ("C1", "mlp"), task)
        if r:
            L.append(f"- C1 (forma + textura + MLP) × E00 {labels[cnn]}, {task}: {_fmt_cmp(r)}.")
            res6[task] = "inferior" if r["ci_high"] < 0 else "competitiva"
    concl["H6"] = ("não testada" if len(res6) < 2 else
                   "suportada" if res6 == {"A": "competitiva", "B": "inferior"} else
                   f"parcialmente suportada (A: {res6['A']}; B: {res6['B']})" if res6["A"] == "competitiva" else
                   f"não suportada (A: {res6['A']}; B: {res6['B']})")
    if cnn != "resnet18":
        L.append("- Provisório: a comparação final usa a ResNet-18 do Estágio 2.")

    L += ["", "## Conclusões", "", "| hipótese | conclusão |", "|---|---|", *[f"| {k} | {v} |" for k, v in concl.items()], "",
          "> Resultado neutro é resultado: com ~118 vídeos de teste por fold e efeitos de ±0,03 no F1, o estudo só tem poder para "
          "efeitos grandes. As conclusões descrevem a evidência deste desenho, não um efeito nulo provado."]
    (C.path(cfg, "docs") / "hipoteses.md").write_text("\n".join(L), encoding="utf-8")
    C.write_json(t / "hypotheses.json", concl)

"""Análise da CNN (Fase D) e síntese do estudo (Fase E).

`cnn`: vereditos em F1 e sensibilidade, convergência, custo, atenção na glote e figuras.
`sintese`: tabela por técnica, hipóteses H1–H6 com as regras de docs/DECISIONS.md (2026-11-04) e custo por fase.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from laringopdi import config as C
from laringopdi import data as D
from laringopdi.custo import latest_stages
from laringopdi.evaluate import _holm, cost_verdicts, run_costs, verdicts
from laringopdi.mlp import _history, _mean_metrics, _pdi_ms, _study, _tech

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#7D8BB8"
NAMES = {"raw": "sem PDI", "stretch": "alargamento", "he": "equalização", "clahe": "CLAHE", "mean": "média", "median": "mediana",
         "gauss_lpf": "passa-baixa", "spec": "remoção de reflexo", "homo": "homomórfica"}


# ----------------------------------------------------------------------------- CNN
def attention_table(cfg: dict) -> pd.DataFrame:
    """Atenção na glote por vídeo, contra `raw`: Wilcoxon pareado com Holm dentro de cada tarefa."""
    from scipy.stats import wilcoxon

    p = C.path(cfg, "tables") / "attention_frames.csv.gz"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, dtype={"video_id": str})
    df = df[(df["model"] == "resnet18") & df["exp"].isin(_study(cfg))]
    rows = []
    for t, g in df.groupby("task"):
        v = g.groupby(["exp", "video_id"])["att_glottis"].mean().unstack("exp")
        area = g.groupby("exp")["area_glottis"].mean()
        vazio = g.groupby("exp")["att_glottis"].apply(lambda s: float(s.isna().mean()))
        part = []
        for e in [e for e in _study(cfg) if e in v.columns]:
            x = v[["E00", e]].dropna()
            d = x[e] - x["E00"]
            part.append({"task": t, "exp": e, "tecnica": _tech(cfg, e), "att_glote": v[e].mean(), "area_glote": area[e], "razao_att_area": v[e].mean() / area[e],
                         "frames_sem_energia": vazio[e], "delta_att": d.mean(), "p": 1.0 if e == "E00" or not (d != 0).any() else float(wilcoxon(d).pvalue)})
        a = pd.DataFrame(part)
        o = a["exp"] != "E00"
        a.loc[o, "p_holm"] = _holm(a.loc[o, "p"].to_numpy())
        a["veredito_atencao"] = "referência"
        a.loc[o, "veredito_atencao"] = np.where(a.loc[o, "p_holm"] < cfg["stats"]["alpha"], np.where(a.loc[o, "delta_att"] > 0, "aumenta", "diminui"), "neutra")
        rows.append(a)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _cnn_table(cfg: dict) -> pd.DataFrame:
    tasks, exps = cfg["study"]["tasks"], _study(cfg)
    others = [e for e in exps if e != "E00"]
    ms = _pdi_ms(cfg)
    costs = run_costs(cfg, "resnet18", ms)
    conv = cost_verdicts(costs, others)
    vf, vs = verdicts(cfg, "resnet18", tasks, exps=others), verdicts(cfg, "resnet18", tasks, exps=others, metric="sensitivity")
    extras = []
    for p in C.path(cfg, "runs").glob("*/*/resnet18/fold*_seed*/metrics.json"):
        m = json.loads(p.read_text(encoding="utf-8"))
        extras.append({"exp": p.parts[-5], "task": p.parts[-4], "ms_inferencia_frame": m["cost"]["infer_ms_per_frame"], "gpu_pico_mb": m["cost"]["gpu_peak_mb"], "s_total": m["cost"]["seconds_total"]})
    extras = pd.DataFrame(extras)
    rows = []
    for t in tasks:
        for e in exps:
            c = costs[(costs["exp"] == e) & (costs["task"] == t)]
            if c.empty:
                continue
            x = extras[(extras["exp"] == e) & (extras["task"] == t)]
            r = {"modelo": "resnet18", "tarefa": t, "tecnica": _tech(cfg, e), **_mean_metrics(cfg, "resnet18", e, t), "ms_tecnica": ms[e],
                 **{k: c[k].mean() for k in ("epoca_melhor", "epocas", "s_ate_melhor", "s_por_epoca", "sobreajuste", "custo_pdi_s", "custo_total_s")},
                 "s_job_medio": x["s_total"].mean(), "ms_inferencia_frame": x["ms_inferencia_frame"].mean(), "gpu_pico_mb": x["gpu_pico_mb"].max()}
            for name, v, key in (("f1", vf, "f1"), ("sensibilidade", vs, "sensitivity")):
                s = v[(v["exp"] == e) & (v["task"] == t)]
                if len(s):
                    s = s.iloc[0]
                    r |= {f"delta_{name}": s[f"delta_{key}"], f"ic_{name}_inf": s["ci_low"], f"ic_{name}_sup": s["ci_high"], f"mcnemar_holm_{name}": s["mcnemar_p_holm"],
                          f"wilcoxon_holm_{name}": s["wilcoxon_p_holm"], f"veredito_{name}": s["verdict"]}
                else:
                    r[f"veredito_{name}"] = "referência"
            s = conv[(conv["exp"] == e) & (conv["task"] == t)]
            r |= {k: (s.iloc[0][k] if len(s) else "referência") for k in ("veredito_convergencia", "veredito_custo")}
            rows.append(r)
    return pd.DataFrame(rows)


def _cnn_curves(cfg: dict) -> None:
    from laringopdi.etapa1 import _plt

    plt = _plt()
    tasks = cfg["study"]["tasks"]
    fig, axes = plt.subplots(1, len(tasks), figsize=(11, 3.6), squeeze=False)
    for ax, t in zip(axes[0], tasks):
        for e, color, name in (("E00", GRAY, "sem PDI"), ("E03", BLUE, "CLAHE")):
            ax.plot(_history(cfg, e, t, "resnet18", "train_video_f1"), color=color, lw=2, ls="--", label=f"{name}, treino")
            ax.plot(_history(cfg, e, t, "resnet18", "val_video_f1"), color=color, lw=2, label=f"{name}, validação")
        ax.set(xlabel="Época", ylabel="F1-macro por vídeo", title=f"Tarefa {t}: curva de treino da ResNet-18")
        ax.grid(color="#C9CFDD", lw=.5, alpha=.6)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(C.path(cfg, "figures") / "cnn_curvas.png")
    plt.close(fig)


def cmd_cnn(cfg: dict, args) -> dict:
    out = C.path(cfg, "tables") / "cnn"
    out.mkdir(parents=True, exist_ok=True)
    table = _cnn_table(cfg)
    table.round(5).to_csv(out / "classificacao.csv", index=False)
    att = attention_table(cfg)
    if not att.empty:
        att.round(5).to_csv(out / "atencao.csv", index=False)
    _cnn_curves(cfg)
    return {"linhas": len(table), "atencao": len(att)}


# ----------------------------------------------------------------------------- hipóteses
def video_dice(cfg: dict, keep: set[tuple[str, str]]) -> pd.DataFrame:
    """Dice por vídeo (preprocess, method) com o parâmetro escolhido no treino de cada fold."""
    tables = C.path(cfg, "tables") / "pdi"
    chosen = pd.read_csv(tables / "segmentacao_por_fold.csv")[["fold", "preprocess", "method", "param_id"]]
    fr = D.load_frames(cfg)[["path", "video_id"]].assign(video_id=lambda d: d["video_id"].astype(str))
    test_of = {str(v): f["fold"] for f in [{"fold": k, "v": s["test"]} for k, s in enumerate(D.load_splits(cfg))] for v in f["v"]}
    parts = []
    for p in sorted((C.path(cfg, "cache").parent / "pdi_study").glob("s_*.csv.gz")):
        s = pd.read_csv(p, usecols=["preprocess", "method", "param_id", "dice", "path"])
        parts.append(s[[(a, b) in keep for a, b in zip(s["preprocess"], s["method"])]])
    s = pd.concat(parts, ignore_index=True).merge(fr, on="path")
    s["fold"] = s["video_id"].map(test_of)
    s = s.merge(chosen, on=["fold", "preprocess", "method", "param_id"])
    return s.groupby(["preprocess", "method", "video_id"])["dice"].mean().reset_index()


def _h1(cfg: dict) -> dict:
    from scipy.stats import wilcoxon

    methods = cfg["segmentation"]["methods"]
    v = video_dice(cfg, {(p, m) for p in ("he", "clahe") for m in methods})
    rows = []
    for m in methods:
        x = v[v["method"] == m].pivot(index="video_id", columns="preprocess", values="dice").dropna()
        d = x["clahe"] - x["he"]
        rows.append({"metodo": m, "dice_clahe": x["clahe"].mean(), "dice_he": x["he"].mean(), "delta": d.mean(), "p": float(wilcoxon(d).pvalue) if (d != 0).any() else 1.0})
    r = pd.DataFrame(rows)
    r["p_holm"] = _holm(r["p"].to_numpy())
    r["clahe_melhor"] = (r["p_holm"] < .05) & (r["delta"] > 0)
    rg = r[r["metodo"] == "region_grow"].iloc[0]
    vis = pd.read_csv(C.path(cfg, "tables") / "pdi" / "tabela_unica.csv").set_index("tecnica")
    sat = bool(vis.loc["he", "saturacao"] > vis.loc["clahe", "saturacao"])
    half = int(r["clahe_melhor"].sum()) * 2 >= len(r)
    v_ = "confirmada" if rg["clahe_melhor"] and half and sat else ("refutada" if rg["p_holm"] < .05 and rg["delta"] < 0 else "parcialmente confirmada")
    return {"veredito": v_, "dice_region_grow_clahe": rg["dice_clahe"], "dice_region_grow_he": rg["dice_he"], "p_holm_region_grow": rg["p_holm"],
            "metodos_clahe_melhor": int(r["clahe_melhor"].sum()), "metodos": len(r), "metodos_he_melhor": int(((r["p_holm"] < .05) & (r["delta"] < 0)).sum()),
            "saturacao_he": vis.loc["he", "saturacao"], "saturacao_clahe": vis.loc["clahe", "saturacao"]}


def _h2(cfg: dict) -> dict:
    d = pd.read_csv(C.path(cfg, "tables") / "pdi" / "ruido.csv")
    sp = sorted(n for n in d["noise"].unique() if n.startswith("sp"))
    rows = []
    for n in sp:
        a, b = d[(d["noise"] == n) & (d["filter"] == "median3")].iloc[0], d[(d["noise"] == n) & (d["filter"] == "mean3")].iloc[0]
        rows.append({"ruido": n, "psnr_mediana": a["psnr"], "psnr_media": b["psnr"], "ssim_mediana": a["ssim"], "ssim_media": b["ssim"]})
    ok = all(r["psnr_mediana"] > r["psnr_media"] and r["ssim_mediana"] > r["ssim_media"] for r in rows)
    return {"veredito": "confirmada" if ok else "parcialmente confirmada", "por_densidade": rows,
            "nota": "Comparação das médias da tabela de ruído; cada filtro passou pelo teste pareado contra a imagem ruidosa."}


def _h3(cfg: dict) -> dict:
    from scipy.stats import wilcoxon

    v = video_dice(cfg, {("raw", "otsu"), ("raw", "adaptive")}).pivot(index="video_id", columns="method", values="dice").dropna()
    d = v["adaptive"] - v["otsu"]
    p = float(wilcoxon(d).pvalue)
    verdict = "confirmada" if p < .05 and d.mean() >= .05 else ("parcialmente confirmada" if p < .05 and d.mean() > 0 else ("refutada" if p < .05 else "inconclusiva"))
    return {"veredito": verdict, "dice_adaptativo": v["adaptive"].mean(), "dice_otsu": v["otsu"].mean(), "delta": d.mean(), "p": p, "n_videos": len(v)}


def _h4(cfg: dict) -> dict:
    r = verdicts(cfg, "mlp18", ["B"], exps=["E00"], baseline_model="resnet18")
    s = verdicts(cfg, "mlp18", ["B"], exps=["E00"], metric="sensitivity", baseline_model="resnet18")
    if r.empty:
        return {"veredito": "sem dados"}
    a, b = r.iloc[0], s.iloc[0]
    return {"veredito": "refutada" if a["ci_high"] < 0 else "confirmada", "f1_cnn": a["f1_base"], "f1_mlp": a["f1_exp"], "delta_f1": a["delta_f1"], "ic_f1": [a["ci_low"], a["ci_high"]],
            "sens_cnn": b["sensitivity_base"], "sens_mlp": b["sensitivity_exp"], "ic_sens": [b["ci_low"], b["ci_high"]]}


def _h5(cfg: dict) -> dict:
    a = attention_table(cfg)
    if a.empty:
        return {"veredito": "sem dados"}
    o = a[a["exp"] != "E00"]
    up, down = int((o["veredito_atencao"] == "aumenta").sum()), int((o["veredito_atencao"] == "diminui").sum())
    v = "confirmada" if up * 2 >= len(o) and not down else ("parcialmente confirmada" if up else ("refutada" if down else "inconclusiva"))
    return {"veredito": v, "aumentam": up, "diminuem": down, "combinacoes": len(o), "quais_aumentam": (o[o["veredito_atencao"] == "aumenta"]["tecnica"] + " " + o[o["veredito_atencao"] == "aumenta"]["task"]).tolist(),
            "quais_diminuem": (o[o["veredito_atencao"] == "diminui"]["tecnica"] + " " + o[o["veredito_atencao"] == "diminui"]["task"]).tolist()}


def _h6(cfg: dict) -> dict:
    d = pd.read_csv(C.path(cfg, "tables") / "mlp" / "importancia_familias.csv")
    top = {t: g.sort_values("queda_media").iloc[-1] for t, g in d.groupby("tarefa")}
    n = sum(x["familia"] == "shape" and x["queda_media"] > 0 for x in top.values())
    return {"veredito": "confirmada" if n == len(top) else ("parcialmente confirmada" if n else "refutada"), "familia_mais_importante": {t: x["familia"] for t, x in top.items()},
            "queda": d.round(4).to_dict("records")}


# ----------------------------------------------------------------------------- tabelas e figuras da síntese
def _technique_table(cfg: dict) -> pd.DataFrame:
    tables = C.path(cfg, "tables")
    pdi = pd.read_csv(tables / "pdi" / "tabela_unica.csv").set_index("tecnica")
    seg = pd.read_csv(tables / "mlp" / "segmentacao.csv").set_index("tecnica")
    mc = pd.read_csv(tables / "mlp" / "classificacao.csv")
    cnn = pd.read_csv(tables / "cnn" / "classificacao.csv")
    att_p = tables / "cnn" / "atencao.csv"
    att = pd.read_csv(att_p) if att_p.exists() else pd.DataFrame()
    rows = []
    for e in _study(cfg):
        t = _tech(cfg, e)
        r = {"exp": e, "tecnica": t, "nome": NAMES[t], "visual": pdi.loc[t, "veredito_visual"], "ms_tecnica": pdi.loc[t, "ms_tecnica"], "dice_classico": pdi.loc[t, "dice"], "seg_classica": pdi.loc[t, "veredito_seg"],
             "dice_mlp": seg.loc[t, "dice"], "seg_mlp": seg.loc[t, "veredito_seg"], "mlp_vs_classico": seg.loc[t, "veredito_vs_classico"]}
        for task in cfg["study"]["tasks"]:
            for prefix, tb in (("mlp", mc[mc["modelo"] == "mlp18"]), ("cnn", cnn)):
                s = tb[(tb["tarefa"] == task) & (tb["tecnica"] == t)]
                if s.empty:
                    continue
                s = s.iloc[0]
                r |= {f"{prefix}_{task}_f1": s["f1"], f"{prefix}_{task}_sens": s["sensibilidade"], f"{prefix}_{task}_esp": s["especificidade"], f"{prefix}_{task}_auc": s["auc"], f"{prefix}_{task}_v_f1": s["veredito_f1"],
                      f"{prefix}_{task}_v_sens": s["veredito_sensibilidade"], f"{prefix}_{task}_conv": s["veredito_convergencia"], f"{prefix}_{task}_custo": s["veredito_custo"], f"{prefix}_{task}_custo_s": s["custo_total_s"]}
            if not att.empty:
                a = att[(att["task"] == task) & (att["tecnica"] == t)]
                if len(a):
                    r[f"atencao_{task}"] = a.iloc[0]["veredito_atencao"]
        rows.append(r)
    return pd.DataFrame(rows)


def _phase_costs(cfg: dict) -> pd.DataFrame:
    s = latest_stages(cfg)
    if s.empty:
        return s
    groups = [("PDI e segmentação clássica", lambda e: e == "pdi-study"), ("Caches das técnicas", lambda e: e.startswith("cache/")), ("MLP de classificação", lambda e: e == "mlp-class"),
              ("MLP de segmentação", lambda e: e == "mlp-seg"), ("CNN: 270 treinos", lambda e: e == "train"), ("Grad-CAM", lambda e: e == "gradcam")]
    rows = []
    for name, f in groups:
        g = s[s["etapa"].map(f)]
        if g.empty:
            continue
        rows.append({"fase": name, "etapas_medidas": len(g), "duracao_h": g["duracao_s"].sum() / 3600, "cpu_nucleos_medio": g["cpu_nucleos_medio"].mean(), "ram_processo_pico_gb": g["ram_processo_pico_gb"].max(),
                     "gpu_usada": "sim" if (g["gpu_usada_pelo_processo"] == "sim").any() else "não", "gpu_mem_pico_mb": g["gpu_sistema_mem_pico_mb"].max() if (g["gpu_usada_pelo_processo"] == "sim").any() else np.nan,
                     "gpu_potencia_media_w": g["gpu_sistema_potencia_w_media"].mean() if (g["gpu_usada_pelo_processo"] == "sim").any() else np.nan,
                     "gpu_energia_wh": g["gpu_sistema_energia_wh"].sum() if (g["gpu_usada_pelo_processo"] == "sim").any() else np.nan, "disco_delta_gb": g["disco_delta_gb"].sum()})
    return pd.DataFrame(rows)


def _delta_figure(cfg: dict) -> None:
    from laringopdi.etapa1 import _plt

    plt = _plt()
    tables = C.path(cfg, "tables")
    mc = pd.read_csv(tables / "mlp" / "classificacao.csv")
    cnn = pd.read_csv(tables / "cnn" / "classificacao.csv")
    tasks, metrics = cfg["study"]["tasks"], (("f1", "F1-macro"), ("sensibilidade", "Sensibilidade"))
    fig, axes = plt.subplots(len(tasks), 2, figsize=(11, 7), sharey=True, squeeze=False)
    order = [_tech(cfg, e) for e in _study(cfg) if e != "E00"]
    y = np.arange(len(order))[::-1]
    for i, t in enumerate(tasks):
        for j, (m, label) in enumerate(metrics):
            ax = axes[i][j]
            ax.axvline(0, color="#1b2a4a", lw=1)
            for tb, color, off, name in ((cnn, BLUE, .15, "ResNet-18"), (mc[mc["modelo"] == "mlp18"], ORANGE, -.15, "MLP")):
                s = tb[tb["tarefa"] == t].set_index("tecnica").reindex(order)
                ax.errorbar(s[f"delta_{m}"], y + off, xerr=[(s[f"delta_{m}"] - s[f"ic_{m}_inf"]).clip(lower=0), (s[f"ic_{m}_sup"] - s[f"delta_{m}"]).clip(lower=0)],
                            fmt="o", ms=5, color=color, ecolor=color, elinewidth=1.5, capsize=0, label=name)
            ax.set_yticks(y, [NAMES[o] for o in order])
            ax.grid(axis="x", color="#C9CFDD", lw=.5, alpha=.6)
            ax.set_title(f"Tarefa {t}: diferença de {label} contra a imagem sem PDI")
            ax.set_xlabel("Diferença (positivo = técnica melhor), com IC 95%")
            ax.tick_params(length=0)
    axes[0][0].legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(C.path(cfg, "figures") / "sintese_deltas.png")
    plt.close(fig)


def _tradeoff_figure(cfg: dict) -> None:
    from laringopdi.etapa1 import _plt

    plt = _plt()
    cnn = pd.read_csv(C.path(cfg, "tables") / "cnn" / "classificacao.csv")
    tasks = cfg["study"]["tasks"]
    fig, axes = plt.subplots(1, len(tasks), figsize=(11, 4), squeeze=False)
    for ax, t in zip(axes[0], tasks):
        s = cnn[cnn["tarefa"] == t]
        ax.scatter(s["custo_total_s"], s["sensibilidade"], color=BLUE, s=36, zorder=3)
        for _, r in s.iterrows():
            ax.annotate(NAMES[r["tecnica"]], (r["custo_total_s"], r["sensibilidade"]), xytext=(5, 4), textcoords="offset points", fontsize=8, color="#1b2a4a")
        ax.set(xlabel="Custo por treino: PDI + treino até a melhor validação (s)", ylabel="Sensibilidade no teste, por vídeo", title=f"Tarefa {t}: o ganho compensa o custo? (ResNet-18)")
        ax.grid(color="#C9CFDD", lw=.5, alpha=.6)
    fig.tight_layout()
    fig.savefig(C.path(cfg, "figures") / "sintese_custo_beneficio.png")
    plt.close(fig)


def _phase_figure(cfg: dict, phases: pd.DataFrame) -> None:
    from laringopdi.etapa1 import _plt

    plt = _plt()
    s = phases.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.barh(s["fase"], s["duracao_h"], color=BLUE, height=.55)
    for i, v in enumerate(s["duracao_h"]):
        ax.annotate(f"{v:.2f} h".replace(".", ","), (v, i), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color="#1b2a4a")
    ax.set_xlabel("Tempo medido (horas)")
    ax.set_title("Tempo de cada fase do estudo")
    ax.grid(axis="x", color="#C9CFDD", lw=.5, alpha=.6)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(C.path(cfg, "figures") / "custo_fases.png")
    plt.close(fig)


def cmd_sintese(cfg: dict, args) -> dict:
    out = C.path(cfg, "tables") / "sintese"
    out.mkdir(parents=True, exist_ok=True)
    table = _technique_table(cfg)
    table.round(5).to_csv(out / "por_tecnica.csv", index=False)
    phases = _phase_costs(cfg)
    phases.round(3).to_csv(C.path(cfg, "tables") / "custo" / "por_fase.csv", index=False)
    hyp = {"H1": _h1(cfg), "H2": _h2(cfg), "H3": _h3(cfg), "H4": _h4(cfg), "H5": _h5(cfg), "H6": _h6(cfg), "hardware": C.hardware(), "git_sha": C.git_sha()}
    C.write_json(out / "hipoteses.json", hyp)
    _delta_figure(cfg)
    _tradeoff_figure(cfg)
    if not phases.empty:
        _phase_figure(cfg, phases)
    return {"tecnicas": len(table), "hipoteses": {k: v["veredito"] for k, v in hyp.items() if isinstance(v, dict) and "veredito" in v}}

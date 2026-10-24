"""Gera o relatório técnico do LaringoPDI em PDF.

Uso:
  python tools/relatorio_pdf.py --out docs/relatorio_tecnico.pdf --publico     # sem frames do BAGLS (vai para o Git)
  python tools/relatorio_pdf.py --out "<Área de Trabalho>/LaringoPDI_relatorio.pdf"   # completo (local)

Todos os números saem das tabelas em results/ e logs/ (nada é digitado à mão).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from laringopdi import config as C  # noqa: E402

FIG = ROOT / "docs" / "figures" / "relatorio"
T = ROOT / "results" / "tables"
INK, DIM, CY, VI, OR, GR, RD = "#1B2A4A", "#5B6785", "#0E9FB5", "#7B5BD6", "#E07B2E", "#2E9E6A", "#D6455D"
EXP_COLORS = {"raw": "#8A94AD", "spec": "#5C9DFF", "he": "#B07CF0", "clahe": "#0E9FB5", "homo": "#E07B2E", "bilat": "#2E9E6A",
              "median": "#6CBF5A", "roi_gt": "#D6455D", "roi": "#F08A9C", "clahe>roi": "#C2547F", "edges": "#2F5BD3",
              "lbp": "#8E44AD", "m1": "#16A085", "m2": "#D35400"}
plt.rcParams.update({"font.family": "Arial", "font.size": 9.5, "axes.edgecolor": "#C9CFDD", "axes.labelcolor": INK,
                     "xtick.color": DIM, "ytick.color": DIM, "axes.titleweight": "bold", "axes.titlesize": 10.5,
                     "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200,
                     "savefig.bbox": "tight"})


def save(fig, name: str) -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    p = FIG / f"{name}.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def rj(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


# ============================================================================ figuras de teoria (sintéticas)
def synth():
    from tests.conftest import synth_glottis
    img, gt = synth_glottis(seed=3, h=256, w=256)
    # vinheta para ilustrar iluminação não uniforme
    yy, xx = np.mgrid[:256, :256]
    vig = 1 - 0.55 * (((xx - 128) ** 2 + (yy - 128) ** 2) / 128 ** 2)
    return np.clip(img * np.clip(vig, .25, 1), 0, 255).astype(np.uint8), gt


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(10, 2.3))
    ax.axis("off")
    boxes = [("Frame BAGLS\n(cinza)", "#E8ECF6"), ("PDI\n(variável)", "#DDF4F7"), ("Cache 224×224\n(3 canais)", "#E8ECF6"),
             ("Classificador\n(fixo)", "#EEE6FB"), ("Avaliação\nframe e vídeo", "#E8ECF6"), ("Estatística\nvereditos", "#FCEBDD")]
    for i, (t, c) in enumerate(boxes):
        x = i * 1.7
        ax.add_patch(plt.Rectangle((x, 0), 1.4, 1, fc=c, ec="#9AA6C2", lw=1.2, zorder=2))
        ax.text(x + .7, .5, t, ha="center", va="center", fontsize=9.5, color=INK, zorder=3)
        if i < len(boxes) - 1:
            ax.annotate("", (x + 1.68, .5), (x + 1.42, .5), arrowprops=dict(arrowstyle="->", color=DIM, lw=1.4))
    ax.text(1.7 + .7, -.32, "spec · he · clahe · homo · bilat · median\nroi · edges · lbp · M1 · M2", ha="center", fontsize=7.5, color=DIM)
    ax.text(3 * 1.7 + .7, -.32, "ResNet-18 · SimpleCNN\nprobe · MLP/SVM/RF", ha="center", fontsize=7.5, color=DIM)
    ax.set_xlim(-.1, 10); ax.set_ylim(-.6, 1.1)
    return save(fig, "pipeline")


def fig_enhance():
    from laringopdi.pdi import enhance
    g, _ = synth()
    outs = [("Original (vinheta)", g), ("Equalização global (HE)", enhance.he(g, {})),
            ("CLAHE (clip 2, 8×8)", enhance.clahe(g, {})), ("Homomórfica", enhance.homo(g, {}))]
    fig, axes = plt.subplots(2, 4, figsize=(11, 4.6), gridspec_kw={"height_ratios": [3, 1.3]})
    for j, (t, im) in enumerate(outs):
        axes[0, j].imshow(im, cmap="gray", vmin=0, vmax=255); axes[0, j].set_title(t, fontsize=9.5); axes[0, j].axis("off")
        axes[1, j].hist(im.ravel(), 64, color=CY if j else DIM); axes[1, j].set_yticks([]); axes[1, j].set_xlim(0, 255)
    fig.suptitle("Realce de contraste num frame sintético: imagem e histograma", fontsize=10.5, color=INK, fontweight="bold")
    return save(fig, "teoria_realce")


def fig_homo_theory():
    from laringopdi.pdi.enhance import homo_transfer
    H = homo_transfer((201, 201), .5, 1.75, 30)
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.3))
    ax[0].imshow(H, cmap="magma"); ax[0].set_title("H(u,v) no plano de frequências"); ax[0].axis("off")
    d = np.arange(0, 100)
    ax[1].plot(d, (1.75 - .5) * (1 - np.exp(-d ** 2 / 30 ** 2)) + .5, color=OR, lw=2.2)
    ax[1].axhline(.5, ls=":", color=DIM); ax[1].axhline(1.75, ls=":", color=DIM)
    ax[1].text(70, .56, "γL = 0,5 (iluminação atenuada)", fontsize=8, color=DIM)
    ax[1].text(40, 1.62, "γH = 1,75 (detalhe realçado)", fontsize=8, color=DIM)
    ax[1].axvline(30, color=CY, ls="--"); ax[1].text(31, .9, "D0 = 30", color=CY, fontsize=8)
    ax[1].set_xlabel("distância ao centro D(u,v)"); ax[1].set_ylabel("ganho"); ax[1].set_title("Perfil radial do filtro")
    return save(fig, "teoria_homo")


def fig_segment_theory():
    from laringopdi.pdi.segmentation import dice, segment
    from laringopdi.pdi.enhance import clahe
    g, gt = synth()
    names = [("otsu", "Otsu global"), ("adaptive", "Adaptativo"), ("otsu_morph", "Otsu + morfologia"),
             ("region_grow", "Crescimento de região"), ("watershed", "Watershed")]
    fig, axes = plt.subplots(1, 6, figsize=(13, 2.6))
    axes[0].imshow(g, cmap="gray"); axes[0].contour(gt, [.5], colors=[GR], linewidths=1.2); axes[0].set_title("Frame + glote (verde)", fontsize=8.5)
    for a, (m, t) in zip(axes[1:], names):
        pm = segment(clahe(g, {}) if m == "region_grow" else g, m)
        a.imshow(g, cmap="gray"); a.imshow(np.ma.masked_where(~pm, pm), cmap="cool", alpha=.55)
        a.set_title(f"{t}\nDice {dice(pm, gt):.2f}", fontsize=8.5)
    for a in axes:
        a.axis("off")
    return save(fig, "teoria_segmentacao")


def fig_texture_theory():
    from laringopdi.pdi.texture import edges, lbp, m1
    g, _ = synth()
    e, l, mm = edges(g, {}), lbp(g, {}), m1(g, {})
    fig, axes = plt.subplots(1, 5, figsize=(12, 2.6))
    for a, im, t in zip(axes, [g, e[..., 1], e[..., 2], l, mm], ["Cinza", "Sobel (magnitude)", "Canny automático", "LBP uniforme", "M1 = [CLAHE, Sobel, LBP]"]):
        a.imshow(im, cmap=None if im.ndim == 3 else "gray"); a.set_title(t, fontsize=9); a.axis("off")
    return save(fig, "teoria_textura")


def fig_neuron():
    fig, ax = plt.subplots(1, 2, figsize=(10, 3), gridspec_kw={"width_ratios": [1.5, 1]})
    a = ax[0]; a.axis("off"); a.set_xlim(0, 10); a.set_ylim(0, 6)
    for i, y in enumerate([5, 3.5, 2, .5]):
        a.add_patch(plt.Circle((1, y), .35, fc="#DDF4F7", ec=CY)); a.text(1, y, f"x{i + 1}" if i < 3 else "1", ha="center", va="center", fontsize=9)
        a.annotate("", (5.2, 2.8), (1.4, y), arrowprops=dict(arrowstyle="->", color=DIM))
        a.text(3, (y + 2.8) / 2 + .15, f"w{i + 1}" if i < 3 else "b", fontsize=8.5, color=OR)
    a.add_patch(plt.Circle((5.9, 2.8), .75, fc="#EEE6FB", ec=VI, lw=1.5)); a.text(5.9, 2.8, "Σ  →  f", ha="center", va="center", fontsize=10)
    a.annotate("", (8.6, 2.8), (6.7, 2.8), arrowprops=dict(arrowstyle="->", color=DIM)); a.text(8.7, 2.7, "y", fontsize=11)
    a.text(5.9, 1.4, "y = f( Σ wᵢxᵢ + b )", ha="center", fontsize=9.5, color=INK)
    z = np.linspace(-4, 4, 200)
    ax[1].plot(z, np.maximum(z, 0), color=CY, lw=2, label="ReLU")
    ax[1].plot(z, 1 / (1 + np.exp(-z)), color=VI, lw=2, label="sigmoide")
    ax[1].legend(frameon=False); ax[1].set_title("Funções de ativação")
    return save(fig, "teoria_neuronio")


# ============================================================================ figuras de resultados
def fig_dataset():
    s = rj(ROOT / "logs" / "index_summary.json")
    fam = s["videos_per_family"]
    order = ["healthy", "functional", "structural", "unknown"]
    pt = {"healthy": "saudável", "functional": "funcional", "structural": "estrutural", "unknown": "excluído"}
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    vals = [fam.get(k, 0) for k in order]
    b = ax[0].bar([pt[k] for k in order], vals, color=[GR, CY, OR, "#B8C0D4"])
    ax[0].bar_label(b, fontsize=9); ax[0].set_title("Vídeos por família (BAGLS)")
    tv = s["task_videos"]
    lab = {"A": ["saudável", "com distúrbio"], "B": ["saudável", "estrutural"], "C": ["saudável", "funcional", "estrutural"]}
    x = 0
    for t in "ABC":
        for i, n in enumerate(lab[t]):
            v = tv[t].get(str(i), 0)
            ax[1].bar(x, v, color=[GR, OR if t != "C" else CY, OR][i]); ax[1].text(x, v + 4, str(v), ha="center", fontsize=8)
            ax[1].text(x, -38, n, ha="center", fontsize=7.5, rotation=0, color=DIM)
            x += 1
        ax[1].text(x - len(lab[t]) / 2 - .5, max(tv[t].values()) + 40, f"Tarefa {t}", ha="center", fontsize=9, color=INK, fontweight="bold")
        x += .8
    ax[1].set_xticks([]); ax[1].set_title("Vídeos por tarefa"); ax[1].set_ylim(-50, 450)
    return save(fig, "dataset")


def fig_quality():
    q = pd.read_csv(ROOT / "results" / "pdi_quality.csv").set_index("exp")
    cfg = C.load()
    lab = [f"{e}\n{cfg['experiments'][e]['pipeline']}" for e in q.index]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    for a, col, t, c in zip(axes, ["entropy", "rms_contrast", "spec_frac"], ["Entropia (bits)", "Contraste RMS", "Fração especular"], [CY, VI, OR]):
        a.barh(lab, q[col], color=c); a.set_title(t); a.invert_yaxis(); a.tick_params(axis="y", labelsize=7.5)
        a.axvline(q.loc["E00", col], color=INK, ls=":", lw=1)
    return save(fig, "qualidade")


def fig_seg():
    s = pd.read_csv(T / "segmentation_summary.csv")
    piv = s.pivot_table(index="method", columns="preprocess", values="dice_mean").reindex(columns=["raw", "clahe", "homo", "bilat"])
    nomes = {"otsu": "Otsu global", "adaptive": "Adaptativo", "otsu_morph": "Otsu + morfologia", "region_grow": "Cresc. de região", "watershed": "Watershed"}
    piv.index = [nomes[i] for i in piv.index]
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 3.8), gridspec_kw={"width_ratios": [1, 1.1], "wspace": .95})
    im = ax[0].imshow(piv.values, cmap="viridis", vmin=0, vmax=.65)
    ax[0].set_xticks(range(4), piv.columns); ax[0].set_yticks(range(len(piv)), piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax[0].text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center", color="white" if piv.values[i, j] < .4 else INK, fontsize=9)
    ax[0].set_title("Dice médio no teste (5 folds)"); fig.colorbar(im, ax=ax[0], fraction=.046, pad=.03)
    h = pd.read_csv(T / "seg_h4.csv")
    h["lbl"] = h["method"].map(nomes) + " · " + h["preprocess"]
    h = h.sort_values("delta")
    cols = [GR if v == "melhora" else RD if v == "piora" else "#B8C0D4" for v in h["verdict"]]
    ax[1].barh(h["lbl"], h["delta"], color=cols); ax[1].axvline(0, color=INK, lw=.8)
    ax[1].set_title("H4: Δ Dice do realce vs raw (por vídeo)"); ax[1].tick_params(axis="y", labelsize=7.5)
    return save(fig, "segmentacao")


def fig_stage1():
    r = pd.read_csv(T / "stage1_ranking.csv")
    cfg = C.load()
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.4), sharex=True)
    for i, m in enumerate(["probe", "simplecnn_fast"]):
        for j, t in enumerate("AB"):
            a = axes[i, j]
            d = r[(r.model == m) & (r.task == t)].set_index("exp").reindex(cfg["stage1"]["experiments"])
            x = np.arange(len(d))
            a.bar(x - .2, d["val_f1"], .4, color=CY, label="validação (escolha)")
            a.bar(x + .2, d["test_f1"], .4, color=VI, alpha=.75, label="teste")
            a.axhline(d.loc["E00", "test_f1"], color=OR, ls="--", lw=1.2, label="E00 (teste)")
            a.set_ylim(.5, .9); a.set_title(f"{'ResNet-18 congelada (probe)' if m == 'probe' else 'SimpleCNN rápida'} · Tarefa {t}")
            a.set_xticks(x, [f"{e}\n{cfg['experiments'][e]['pipeline']}" for e in d.index], fontsize=6.6, rotation=90)
            if i == 0 and j == 0:
                a.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    fig.supylabel("F1-macro por vídeo")
    return save(fig, "estagio1")


def fig_forest():
    cfg = C.load()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for a, m, title in zip(axes, ["probe", "simplecnn_fast"], ["ResNet-18 congelada (probe)", "SimpleCNN rápida"]):
        v = pd.read_csv(T / f"verdicts_{m}.csv")
        exps = cfg["stage1"]["experiments"][1:]
        for k, t in enumerate("AB"):
            d = v[v.task == t].set_index("exp").reindex(exps)
            y = np.arange(len(exps)) + (k - .5) * .32
            a.errorbar(d["oof_delta_f1"], y, xerr=[d["oof_delta_f1"] - d["ci_low"], d["ci_high"] - d["oof_delta_f1"]],
                       fmt="o", color=[CY, OR][k], ms=4, capsize=2, lw=1.1, label=f"Tarefa {t}")
        a.axvline(0, color=INK, lw=.8); a.set_yticks(range(len(exps)), [f"{e} {cfg['experiments'][e]['pipeline']}" for e in exps], fontsize=8)
        a.invert_yaxis(); a.set_xlabel("Δ F1-macro por vídeo vs E00 (IC 95% bootstrap)"); a.set_title(title); a.legend(frameon=False, fontsize=8)
    return save(fig, "vereditos")


def fig_classic():
    c = pd.read_csv(T / "classic_vs_cnn.csv")
    runs = pd.read_csv(ROOT / "results" / "summary.csv")
    s = runs[(runs.exp == "E00") & (runs.model == "simplecnn_fast")].groupby("task")["video_f1_macro"].agg(["mean", "std"]).reset_index()
    rows = [(f"{r.exp} {r.model}", r.task, r.f1, r.f1_sd) for r in c.itertuples()] + \
           [("E00 simplecnn_fast", r.task, r["mean"], r["std"]) for _, r in s.iterrows()]
    d = pd.DataFrame(rows, columns=["nome", "task", "f1", "sd"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    for a, t in zip(axes, "AB"):
        x = d[d.task == t].sort_values("f1")
        cols = [VI if "C" == n[0] else CY for n in x["nome"]]
        a.barh(x["nome"], x["f1"], xerr=x["sd"], color=cols, capsize=2); a.set_xlim(.4, .95); a.set_title(f"Tarefa {t}")
        a.set_xlabel("F1-macro por vídeo (média ± dp, 5 folds)")
    nomes = "C1 forma+textura MLP · C2 idem na ROI · C3 SVM · C4 RF · C5 forma pelo GT + MLP"
    fig.text(.5, -.04, nomes, ha="center", fontsize=8, color=DIM)
    return save(fig, "classico")


def fig_perm():
    rows = []
    for p in (ROOT / "results" / "runs").glob("C*/*/*/fold*_seed0/metrics.json"):
        m = rj(p)
        for fam, v in m.get("perm_importance", {}).items():
            rows.append({"exp": p.parts[-5], "task": p.parts[-4], "fam": fam, "drop": v["mean"]})
    d = pd.DataFrame(rows).groupby(["exp", "task", "fam"])["drop"].mean().unstack("fam")
    fam_pt = {"shape": "forma", "tex": "textura", "hog": "HOG", "edge": "bordas"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), sharey=True)
    for a, t in zip(axes, "AB"):
        x = d.xs(t, level="task")
        x.rename(columns=fam_pt).plot.bar(ax=a, color=[OR, CY, VI, GR], width=.8, legend=t == "A")
        a.axhline(0, color=INK, lw=.7); a.set_title(f"Importância por família (permutação) · Tarefa {t}"); a.set_xlabel("")
        a.tick_params(axis="x", rotation=0)
    axes[0].set_ylabel("queda da acurácia balanceada")
    return save(fig, "importancia")


def fig_curves():
    rows = []
    for p in (ROOT / "results" / "runs").glob("*/*/simplecnn_fast/fold*_seed0/metrics.json"):
        m = rj(p)
        for h in m.get("history", []):
            rows.append({"exp": p.parts[-5], "task": p.parts[-4], "epoch": h["epoch"] + 1, "f1": h["val_video_f1"], "loss": h["loss"]})
    d = pd.DataFrame(rows)
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.1))
    for t, c in zip("AB", [CY, OR]):
        g = d[d.task == t].groupby("epoch")
        ax[0].plot(g["f1"].mean(), color=c, lw=2, label=f"Tarefa {t}")
        ax[0].fill_between(g["f1"].mean().index, g["f1"].quantile(.25), g["f1"].quantile(.75), color=c, alpha=.15)
        ax[1].plot(g["loss"].mean(), color=c, lw=2, label=f"Tarefa {t}")
    ax[0].set_title("F1 por vídeo na validação (mediana dos jobs, faixa interquartil)"); ax[0].set_xlabel("época"); ax[0].legend(frameon=False)
    ax[1].set_title("Custo (perda) no fim de cada época"); ax[1].set_xlabel("época")
    return save(fig, "curvas")


def fig_stage2():
    cfg = C.load()
    ms = [(m, n) for m, n in (("resnet18", "ResNet-18 (fine-tuning)"), ("simplecnn", "SimpleCNN (treino completo)"))
          if (T / f"verdicts_{m}.csv").exists()]
    fig, axes = plt.subplots(1, len(ms), figsize=(6 * len(ms), 3.8), sharey=True, squeeze=False)
    for a, (m, title) in zip(axes[0], ms):
        v = pd.read_csv(T / f"verdicts_{m}.csv")
        exps = list(dict.fromkeys(v["exp"]))
        for k, t in enumerate("AB"):
            d = v[v.task == t].set_index("exp").reindex(exps)
            y = np.arange(len(exps)) + (k - .5) * .32
            col = [GR if x == "melhora" else RD if x == "piora" else [CY, OR][k] for x in d["verdict"].fillna("neutra")]
            a.errorbar(d["oof_delta_f1"], y, xerr=[d["oof_delta_f1"] - d["ci_low"], d["ci_high"] - d["oof_delta_f1"]],
                       fmt="none", ecolor=[CY, OR][k], capsize=2, lw=1.1)
            a.scatter(d["oof_delta_f1"], y, c=col, s=22, zorder=3, label=f"Tarefa {t}")
        a.axvline(0, color=INK, lw=.8)
        a.set_yticks(range(len(exps)), [f"{e} {cfg['experiments'][e]['pipeline']}" + (" (1 sem.)" if e in cfg["stage2_hyp"]["experiments"] else "")
                                         for e in exps], fontsize=8)
        a.invert_yaxis(); a.set_xlabel("Δ F1-macro por vídeo vs E00 (IC 95% bootstrap)"); a.set_title(title); a.legend(frameon=False, fontsize=8)
    return save(fig, "estagio2")


def fig_attention():
    d = pd.read_csv(T / "attention.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), sharey=True)
    for a, t in zip(axes, "AB"):
        x = d[d.task == t].copy()
        x["nome"] = x["exp"] + " " + x["model"].map({"resnet18": "RN18", "simplecnn": "SCNN"})
        x = x.sort_values(["model", "exp"])
        i = np.arange(len(x))
        a.bar(i - .2, x["att_glottis"], .4, color=CY, label="energia na glote")
        a.bar(i + .2, x["att_spec"].fillna(0), .4, color=OR, label="energia nos reflexos")
        a.scatter(i - .2, x["area_glottis"], marker="_", s=120, color=INK, zorder=3, label="área (acaso)")
        a.scatter(i + .2, x["area_spec"].fillna(0), marker="_", s=120, color=INK, zorder=3)
        a.set_xticks(i, x["nome"], rotation=90, fontsize=7.5); a.set_title(f"Grad-CAM · Tarefa {t}")
    axes[0].set_ylabel("fração da energia"); axes[0].legend(frameon=False, fontsize=8)
    return save(fig, "atencao")


# ============================================================================ PDF
def build(out: Path, publico: bool) -> None:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                    TableStyle)
    from reportlab.lib.fonts import addMapping

    fdir = Path("C:/Windows/Fonts")
    for n, f in (("Arial", "arial.ttf"), ("Arial-Bold", "arialbd.ttf"), ("Arial-Italic", "ariali.ttf"), ("Arial-BoldItalic", "arialbi.ttf")):
        pdfmetrics.registerFont(TTFont(n, str(fdir / f)))
    addMapping("Arial", 0, 0, "Arial"); addMapping("Arial", 1, 0, "Arial-Bold"); addMapping("Arial", 0, 1, "Arial-Italic"); addMapping("Arial", 1, 1, "Arial-BoldItalic")

    cfg = C.load()
    st = {
        "title": ParagraphStyle("t", fontName="Arial-Bold", fontSize=24, leading=29, textColor=colors.HexColor(INK)),
        "sub": ParagraphStyle("s", fontName="Arial", fontSize=12.5, leading=17, textColor=colors.HexColor(DIM)),
        "h1": ParagraphStyle("h1", fontName="Arial-Bold", fontSize=16, leading=20, spaceBefore=6, spaceAfter=8, textColor=colors.HexColor(INK)),
        "h2": ParagraphStyle("h2", fontName="Arial-Bold", fontSize=12, leading=15, spaceBefore=10, spaceAfter=4, textColor=colors.HexColor("#0B6F80")),
        "p": ParagraphStyle("p", fontName="Arial", fontSize=9.8, leading=14.2, alignment=TA_JUSTIFY, spaceAfter=6, textColor=colors.HexColor("#222B3D")),
        "b": ParagraphStyle("b", fontName="Arial", fontSize=9.8, leading=14, leftIndent=12, bulletIndent=2, spaceAfter=3, textColor=colors.HexColor("#222B3D")),
        "cap": ParagraphStyle("c", fontName="Arial-Italic", fontSize=8.4, leading=11, alignment=TA_CENTER, spaceAfter=10, textColor=colors.HexColor(DIM)),
        "cell": ParagraphStyle("cell", fontName="Arial", fontSize=8.2, leading=10.4, textColor=colors.HexColor("#222B3D")),
        "cellb": ParagraphStyle("cellb", fontName="Arial-Bold", fontSize=8.2, leading=10.4, textColor=colors.white),
        "note": ParagraphStyle("n", fontName="Arial", fontSize=9, leading=12.5, textColor=colors.HexColor("#6B4A12"),
                               backColor=colors.HexColor("#FFF4E5"), borderPadding=7, spaceBefore=4, spaceAfter=10),
    }
    W = A4[0] - 4 * cm
    story = []
    P = lambda t, s="p": story.append(Paragraph(t, st[s]))
    B = lambda items: [story.append(Paragraph(i, st["b"], bulletText="•")) for i in items]

    def fig(path, caption, width=1.0, private=False):
        if private and publico:
            story.append(Paragraph(f"[Figura omitida na versão pública: contém frames de pacientes do BAGLS (CC BY-NC-SA). "
                                   f"Disponível na versão completa, local.] {escape(caption)}", st["note"]))
            return
        p = Path(path)
        if not p.exists():
            return
        w, h = cv2.imread(str(p)).shape[1::-1] if p.suffix.lower() in (".png", ".jpg") else (1, 1)
        iw = W * width
        story.append(KeepTogether([Image(str(p), width=iw, height=iw * h / w), Paragraph(caption, st["cap"])]))

    def table(rows, widths, header=True, zebra=True):
        data = [[Paragraph(str(c), st["cellb" if (header and i == 0) else "cell"]) for c in r] for i, r in enumerate(rows)]
        t = Table(data, colWidths=[W * x for x in widths], repeatRows=1 if header else 0)
        style = [("GRID", (0, 0), (-1, -1), .3, colors.HexColor("#C9CFDD")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                 ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
        if header:
            style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1B2A4A")))
        if zebra:
            for i in range(1, len(rows)):
                if i % 2 == 0:
                    style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#F2F5FA")))
        t.setStyle(TableStyle(style))
        story.append(t); story.append(Spacer(1, 8))

    # ------------------------------------------------------------------ dados para o texto
    idx = rj(ROOT / "logs" / "index_summary.json")
    fj = rj(ROOT / "data" / "splits" / "frames.json")["meta"]
    best = rj(T / "segmentation_best.json")
    segs = pd.read_csv(T / "segmentation_summary.csv")
    rg = segs.set_index(["method", "preprocess"])
    h4 = pd.read_csv(T / "seg_h4.csv")
    rk = pd.read_csv(T / "stage1_ranking.csv")
    vp = pd.read_csv(T / "verdicts_probe.csv"); vs = pd.read_csv(T / "verdicts_simplecnn_fast.csv")
    cvc = pd.read_csv(T / "classic_vs_cnn.csv")
    summ = pd.read_csv(ROOT / "results" / "summary.csv")
    e00 = summ[summ.exp == "E00"].groupby(["model", "task"])["video_f1_macro"].mean()
    top4 = [e for e in rk.drop_duplicates("exp")["exp"] if e not in ("E00", "E07")][:4]
    fam = idx["videos_per_family"]
    n_runs = len(summ)
    git = C.git_sha()
    s2 = (T / "verdicts_resnet18.csv").exists()
    v2 = pd.concat([pd.read_csv(T / f"verdicts_{m}.csv") for m in ("resnet18", "simplecnn") if (T / f"verdicts_{m}.csv").exists()]) if s2 else None
    hyp = rj(T / "hypotheses.json") if (T / "hypotheses.json").exists() else {}
    att = (T / "attention.csv").exists()
    e14 = cfg["experiments"]["E14"]["pipeline"]

    # ------------------------------------------------------------------ capa
    story += [Spacer(1, 3.2 * cm), Paragraph("LaringoPDI", st["title"]), Spacer(1, 6),
              Paragraph("Efeito de técnicas de Processamento Digital de Imagens na segmentação da glote e na "
                        "classificação de distúrbios da voz em videoendoscopia laríngea", st["sub"]), Spacer(1, 18),
              Paragraph("Relatório técnico de andamento", ParagraphStyle("x", parent=st["sub"], fontName="Arial-Bold", textColor=colors.HexColor(CY))),
              Spacer(1, 30)]
    fig(fig_pipeline(), "Figura 1. Desenho do estudo: o PDI é a variável independente; o classificador, o protocolo e os splits ficam fixos.")
    story += [Spacer(1, 26)]
    table([["Disciplina", "Processamento Digital de Imagens (PDI), UNIFOR. Prof.ª Lyndainês Santos. Projeto extensionista."],
           ["Equipe", "Nathan (2220227, representante) · Marcos Vinicius · Pedro Nobre · Marcos Mendes · Tales"],
           ["Data", date.today().strftime("%d/%m/%Y") + f" · versão do código {git} · {'versão pública (sem frames de pacientes)' if publico else 'versão completa (uso interno da equipe)'}"],
           ["Aviso", "Ferramenta de apoio acadêmico. Não substitui avaliação médica."]], [.18, .82], header=False, zebra=False)
    story.append(PageBreak())

    # ------------------------------------------------------------------ 1. resumo
    P("1. Resumo executivo", "h1")
    pv = (vp["verdict"] != "neutra").sum() + (vs["verdict"] != "neutra").sum()
    P(f"Este documento consolida o que o projeto LaringoPDI produziu até {date.today().strftime('%d/%m/%Y')}: a fundamentação "
      "teórica de PDI e de Inteligência Artificial, os métodos aplicados, os resultados obtidos, os erros encontrados (e como "
      "foram corrigidos) e o próximo passo. A pergunta central é <b>quais técnicas de PDI aplicadas a frames de "
      "videoendoscopia laríngea melhoram, não alteram ou pioram</b> (a) a segmentação clássica da glote e (b) a "
      "classificação de distúrbios da voz por redes neurais, e por quê.")
    B([f"<b>Dados:</b> BAGLS, {idx['n_frames']:,} frames de {idx['n_videos']} vídeos de alta velocidade; {fj['n_frames']:,} frames "
       f"amostrados de {fj['n_videos']} vídeos rotulados (saudável {fam['healthy']}, funcional {fam['functional']}, estrutural "
       f"{fam['structural']}); 5 folds agrupados por vídeo, sem vazamento.".replace(",", "."),
       f"<b>Segmentação clássica:</b> o melhor método foi o crescimento de região sobre CLAHE (Dice "
       f"{rg.loc[(best['method'], best['preprocess']), 'dice_mean']:.3f}, contra {rg.loc[(best['method'], 'raw'), 'dice_mean']:.3f} "
       f"sem realce). O efeito do realce depende do par método × técnica: {int((h4.verdict == 'melhora').sum())} de {len(h4)} "
       "comparações melhoram significativamente.",
       f"<b>Classificação, Estágio 1 (triagem):</b> {n_runs} execuções concluídas. Com ResNet-18 congelada e SimpleCNN, "
       f"<b>nenhuma das 13 técnicas</b> mudou o F1 por vídeo de forma estatisticamente significativa ({pv} vereditos "
       "diferentes de 'neutra' em 52). As diferenças ficam em torno de ±0,03, com intervalos que cruzam zero.",
       f"<b>Braço clássico:</b> características de forma glótica, textura, HOG e bordas com MLP/SVM/RF chegaram a F1 por vídeo "
       f"de {cvc[cvc.task == 'A']['f1'].max():.2f} (Tarefa A) e {cvc[cvc.task == 'B']['f1'].max():.2f} (Tarefa B), "
       "no mesmo patamar das CNNs.",
       "<b>Erros relevantes encontrados e corrigidos:</b> o principal foi a augmentation corrompida em precisão mista "
       "na GPU, detectada pela telemetria visual do treino. Os 22 jobs afetados foram descartados e o Estágio 1 foi refeito.",
       *([f"<b>Classificação, Estágio 2 (fine-tuning, 3 sementes × 5 folds):</b> ResNet-18 no E00 com F1 por vídeo "
          f"{e00.get(('resnet18', 'A'), float('nan')):.3f} (A) e {e00.get(('resnet18', 'B'), float('nan')):.3f} (B); "
          f"{int((v2.verdict != 'neutra').sum())} de {len(v2)} vereditos diferentes de 'neutra' "
          f"({', '.join(f'{r.exp} {r.pipeline} {r.task} {r.model}: {r.verdict}' for r in v2[v2.verdict != 'neutra'].itertuples()) or 'nenhum'}). "
          f"E14 (guloso) = <i>{escape(str(e14))}</i>."] if s2 else []),
       ("<b>Hipóteses:</b> " + "; ".join(f"{k} {v}" for k, v in hyp.items()) + ". Detalhes em <i>docs/hipoteses.md</i>.") if s2 and hyp else
       "<b>Próximo passo:</b> o Estágio 2 (fine-tuning de E00, do top-4 e do E14; 360 jobs, ~16 h na RTX 3060), seguido de "
       "Grad-CAM, relatório estatístico final e conclusão de H1–H6. As instruções estão em <i>docs/PROXIMOS_PASSOS.md</i>."])

    # ------------------------------------------------------------------ 2. contexto
    P("2. Contexto, objetivo e desenho", "h1")
    P("O projeto é extensionista e tem três dimensões. A <b>social</b> são os profissionais da voz, sobretudo professores, "
      "como beneficiários, e os profissionais de saúde em triagem como usuários. A <b>tecnológica</b> é um protótipo de apoio "
      "à decisão. A <b>científica</b> é uma comparação controlada de técnicas de PDI. A regra de ouro da disciplina guiou "
      "todas as escolhas: <b>o PDI é o protagonista</b>. Uma rede recebendo a imagem crua não basta; cada técnica precisa "
      "mostrar o que faz na imagem e por que importa na laringoscopia.")
    P("O desenho experimental trata o classificador como <b>instrumento de medida fixo</b>. Arquitetura, hiperparâmetros, "
      "splits, sementes e augmentation são idênticos entre as variantes; muda apenas o pipeline de PDI aplicado antes. "
      "Resultados negativos, em que a técnica não muda nada, são resultados válidos e são reportados.")
    P("Hipóteses pré-registradas (escritas antes de qualquer resultado)", "h2")
    table([["ID", "Hipótese"], ["H1", "Remover reflexos especulares mantém ou melhora o F1 e reduz a atenção do Grad-CAM sobre os reflexos."],
           ["H2", "CLAHE e filtragem homomórfica superam a equalização global (classificação e segmentação)."],
           ["H3", "O PDI ajuda mais a SimpleCNN (treinada do zero) do que a ResNet-18 pré-treinada."],
           ["H4", "O realce melhora o Dice da segmentação clássica da glote em relação à imagem crua."],
           ["H5", "O recorte pela ROI ajuda mais na Tarefa B (lesões estruturais) e o roi clássico se aproxima do limite roi_gt."],
           ["H6", "Forma glótica + MLP é competitiva com a CNN na Tarefa A, mas inferior na Tarefa B."]], [.08, .92])
    story.append(PageBreak())

    # ------------------------------------------------------------------ 3. fundamentos de PDI
    P("3. Fundamentos de Processamento Digital de Imagens", "h1")
    P("Uma imagem digital em tons de cinza é uma matriz f(x, y) de intensidades inteiras entre 0 e 255. O PDI transforma "
      "essa matriz para realçar o que interessa (contraste, bordas, textura), suprimir o que atrapalha (ruído, reflexos, "
      "iluminação desigual) ou extrair regiões e medidas. Em videoendoscopia laríngea, os problemas típicos são três: a "
      "<b>iluminação pontual</b> (o centro claro e as bordas escuras, chamado vinhetamento), os <b>reflexos especulares</b> "
      "da mucosa úmida e o <b>ruído</b> de sensor e de compressão. Todas as técnicas trabalham em cinza: 618 dos 640 vídeos "
      "do BAGLS já são monocromáticos, e os 22 coloridos são convertidos.")
    P("3.1 Histograma e realce de contraste", "h2")
    P("O histograma conta quantos pixels há em cada nível de cinza. A <b>equalização global (HE)</b> redistribui os níveis "
      "pela função de distribuição acumulada, espalhando o histograma por toda a faixa: aumenta o contraste global, mas "
      "amplifica ruído e estoura regiões já claras. O <b>CLAHE</b> (Contrast Limited Adaptive Histogram Equalization) "
      "equaliza em blocos (8×8) e limita a altura do histograma (clip 2,0) antes de redistribuir. O resultado é realce "
      "local, sem saturar o centro iluminado, o que preserva a borda livre das pregas vocais.")
    fig(fig_enhance(), "Figura 2. Frame sintético com vinheta: a HE espalha o histograma inteiro, o CLAHE realça localmente e a homomórfica achata a iluminação.")
    P("3.2 Domínio da frequência e filtragem homomórfica", "h2")
    P("O modelo de iluminação–refletância descreve a imagem como f = i · r: a iluminação i varia devagar (baixa frequência) "
      "e a refletância r carrega os detalhes (alta frequência). Aplicando o logaritmo, o produto vira soma, "
      "ln f = ln i + ln r, e os dois termos ficam separáveis no domínio de Fourier. O filtro gaussiano de alta ênfase "
      "H(u,v) = (γH − γL)(1 − e<super>−D²/D0²</super>) + γL atenua a iluminação (γL = 0,5) e realça a refletância "
      "(γH = 1,75). A transformada inversa e a exponencial devolvem a imagem com iluminação mais uniforme.")
    fig(fig_homo_theory(), "Figura 3. Filtro homomórfico: H(u,v) no plano de frequências e seu perfil radial (D0 = 30 px em imagens com lado de até 512 px).", .85)
    P("3.3 Filtragem espacial, reflexos e inpainting", "h2")
    P("O <b>filtro bilateral</b> faz uma média ponderada pela proximidade espacial <i>e</i> pela semelhança de intensidade: "
      "suaviza o ruído sem borrar bordas. O <b>filtro da mediana</b> substitui cada pixel pela mediana da vizinhança 5×5, "
      "eficaz contra ruído impulsivo, mas pode apagar detalhes finos. Os <b>reflexos especulares</b> são detectados por um "
      "limiar de intensidade: Otsu sobre a cauda do histograma acima do percentil 90, com piso 200. A máscara é dilatada e "
      "preenchida por <b>inpainting de Telea</b>, que propaga as intensidades vizinhas para dentro da região.")
    P("3.4 Segmentação: limiarização, morfologia, região e watershed", "h2")
    P("A glote aberta é a região mais escura do frame. O <b>método de Otsu</b> escolhe o limiar que maximiza a variância "
      "entre as classes claro e escuro. O <b>limiar adaptativo</b> compara cada pixel com a média gaussiana da sua vizinhança "
      "e tolera a iluminação desigual. A <b>morfologia matemática</b> remove ruído (abertura = erosão seguida de dilatação) "
      "e une regiões quebradas (fechamento). O <b>crescimento de região</b> parte da semente mais escura e anexa vizinhos "
      "dentro de uma tolerância de intensidade. O <b>watershed</b> trata a imagem como relevo e inunda a partir de "
      "marcadores derivados da transformada de distância, o que separa regiões escuras que se tocam.")
    fig(fig_segment_theory(), "Figura 4. Os cinco métodos clássicos num frame sintético (o Dice compara com a glote verdadeira, em verde).")
    P("3.5 Bordas e textura", "h2")
    P("O operador de <b>Sobel</b> aproxima o gradiente por convolução com máscaras 3×3; sua magnitude destaca bordas. O "
      "<b>Canny</b> refina com supressão de não máximos e histerese, com limiares automáticos de (1 ∓ 0,33) × mediana. O "
      "<b>LBP</b> (Local Binary Pattern) codifica cada pixel comparando-o com 8 vizinhos; o padrão uniforme é invariante a "
      "mudanças monotônicas de brilho e descreve micro-textura (mucosa lisa × rugosa). A <b>GLCM</b> (matriz de "
      "co-ocorrência) mede contraste, homogeneidade, energia e correlação entre pares de pixels, e os filtros de "
      "<b>Gabor</b> respondem a texturas em frequências e orientações específicas.")
    fig(fig_texture_theory(), "Figura 5. Representações de bordas e textura usadas como entrada (E10–E12) e no braço clássico.")
    story.append(PageBreak())

    # ------------------------------------------------------------------ 4. fundamentos de IA
    P("4. Fundamentos de Inteligência Artificial", "h1")
    P("4.1 O neurônio artificial e a rede", "h2")
    P("Um neurônio calcula uma soma ponderada das entradas mais um viés e aplica uma função de ativação não linear, "
      "y = f(Σ wᵢxᵢ + b). Empilhando camadas desses neurônios, a rede aprende representações cada vez mais abstratas. "
      "Os <b>pesos w</b> são os parâmetros ajustados durante o treino. A função <b>ReLU</b> (max(0, z)) é a ativação "
      "interna padrão; na saída, a <b>softmax</b> transforma os escores em probabilidades por classe.")
    fig(fig_neuron(), "Figura 6. O neurônio artificial e duas funções de ativação.", .9)
    P("4.2 Redes convolucionais (CNN)", "h2")
    P("Numa CNN, cada neurônio olha só uma pequena janela da imagem e compartilha os pesos (o <b>filtro</b>, ou kernel) "
      "por toda ela. Uma convolução é, portanto, exatamente a filtragem espacial do PDI, só que com <b>pesos aprendidos</b> "
      "em vez de escolhidos à mão. Camadas de <b>pooling</b> reduzem a resolução; o <b>Global Average Pooling</b> resume "
      "cada mapa num número antes da camada de decisão. Usamos duas arquiteturas fixas:")
    B(["<b>SimpleCNN</b>, treinada do zero: 4 blocos [convolução 3×3 → BatchNorm → ReLU → max-pool] com 32/64/128/256 "
       "filtros, GAP, dropout 0,3 e camada linear (~390 mil pesos).",
       "<b>ResNet-18</b>, pré-treinada no ImageNet: 18 camadas com conexões residuais (~11 milhões de pesos). Também é usada "
       "<b>congelada</b> no <i>linear probe</i>: as características de 512 dimensões vão para uma regressão logística, um "
       "teste rápido de quanto a representação já existente separa as classes."])
    P("4.3 Como a rede aprende", "h2")
    P("O treino mede o erro com a <b>entropia cruzada ponderada</b> (pesos inversos à frequência de cada classe, porque os "
      "saudáveis são maioria). A <b>retropropagação</b> calcula, pela regra da cadeia, quanto cada peso contribuiu para o "
      "erro (o gradiente), e o otimizador <b>AdamW</b> ajusta os pesos na direção que reduz o erro. A <b>taxa de "
      "aprendizado</b> decai em curva de cosseno. A cada época, a rede é avaliada na <b>validação interna</b> (vídeos "
      "separados do treino); o <b>early stopping</b> guarda o melhor ponto e para se não houver melhora em 7 épocas. A "
      "<b>augmentation</b> é só geométrica (espelhamento, giro de ±15°, escala de 0,9 a 1,1). Variações de intensidade "
      "são proibidas, porque mudariam justamente a variável em estudo (o PDI).")
    P("4.4 Aprendizado de máquina clássico", "h2")
    P("O braço clássico, ponte com a disciplina de IA Computacional, extrai características interpretáveis. São elas: a "
      "<b>forma da glote</b> (área, perímetro, excentricidade, solidez, razão de aspecto, compacidade e simetria "
      "esquerda/direita), a <b>textura</b> (GLCM, LBP, Gabor), o <b>HOG</b> e as <b>bordas</b>. Elas são agregadas por "
      "vídeo (média e desvio-padrão dos 10 frames) e classificadas por um <b>MLP</b> (perceptron multicamadas), com "
      "<b>SVM-RBF</b> e <b>Random Forest</b> como referência.")
    P("4.5 Métricas, estatística e interpretabilidade", "h2")
    B(["<b>F1-macro</b> (média do F1 de cada classe), <b>acurácia balanceada</b>, <b>sensibilidade</b>, "
       "<b>especificidade</b> e <b>AUC</b>, por frame e <b>por vídeo</b> (média das probabilidades dos frames). O nível de "
       "vídeo é o principal.",
       "<b>Wilcoxon pareado</b> por fold × semente, com <b>correção de Holm</b> para comparações múltiplas; <b>McNemar</b> "
       "sobre as predições out-of-fold por vídeo; <b>IC 95% por bootstrap</b> da diferença de F1. Veredito: a técnica "
       "'melhora' ou 'piora' só se o McNemar corrigido for menor que 0,05 e o IC excluir zero; caso contrário, 'neutra'.",
       "<b>Grad-CAM</b>: combina os gradientes da classe prevista com os mapas da última convolução e mostra onde a decisão "
       "mais pesou. Medimos a fração dessa atenção dentro da glote (máscara do especialista)."])
    story.append(PageBreak())

    # ------------------------------------------------------------------ 5. dados
    P("5. Dados", "h1")
    P("5.1 Troca de dataset: Laryngoscope8 → BAGLS", "h2")
    P("O projeto foi especificado para o Laryngoscope8 (3.057 imagens, 8 classes). O único link oficial, no Baidu Pan, "
      "estava inativo desde 2023, com 17 pedidos sem resposta no repositório dos autores, e não há espelho público. O "
      "VoFoCD também não é público, e o CE-NBI é NBI de contato (fora do escopo). O projeto passou então ao <b>BAGLS</b> "
      "(Gómez et al., <i>Scientific Data</i>, 2020; CC BY-NC-SA 4.0), que traz frames de videoendoscopia de alta "
      "velocidade de vários hospitais, <b>máscaras da glote feitas por especialistas</b> e o status do distúrbio de "
      "cada sujeito. A mudança abriu um eixo novo e forte para o PDI: a segmentação clássica com ground truth.")
    P("5.2 Rótulos a partir de texto livre", "h2")
    P(f"O campo de status é texto livre, com {idx['n_distinct_status']} valores distintos (ex.: \"Muscle tension dysphonia with "
      "nodules\", \"Reinke's edema right vocal fold ...\", \"None (half year post-phonomicrosurgery ...)\"). Regras "
      "ordenadas por palavras-chave (<i>configs/label_map.yaml</i>) atribuem cada vídeo a uma família. Qualquer lesão "
      "estrutural vence o componente funcional. Os status vazios e os 'sem distúrbio' pós-cirúrgicos foram excluídos. A "
      "família estrutural tinha 47 vídeos (abaixo do mínimo de 50) e, por decisão de método, passou a incluir cicatriz, "
      "sulco/vergeture, rigidez e irregularidade de mucosa, chegando a 54.")
    fig(fig_dataset(), "Figura 7. Distribuição dos vídeos por família e por tarefa (A: saudável × distúrbio; B: saudável × estrutural; C: três classes).")
    fig(ROOT / "docs" / "figures" / "eda_samples.png", "Figura 8. Exemplos de frames do BAGLS por família.", .62, private=True)
    P("5.3 Anti-vazamento e amostragem", "h2")
    B(["Frames de um mesmo vídeo são quase idênticos. A <b>unidade de separação é o vídeo</b>: StratifiedGroupKFold com 5 "
       "folds, estratificado pelo rótulo de 3 classes. O split oficial do BAGLS (feito para segmentação) foi ignorado.",
       "Os vídeos possivelmente do mesmo sujeito foram verificados pelo pHash do frame central, confirmado por SSIM: 1 par "
       "candidato e nenhum confirmado.",
       f"Até <b>10 frames por vídeo</b>, uniformemente espaçados e só com a glote visível: {fj['n_frames']:,} frames, a "
       "mesma amostra para todos os experimentos.".replace(",", "."),
       "A validação interna para o early stopping sai de dentro do treino; nada é escolhido olhando o teste."])
    story.append(PageBreak())

    # ------------------------------------------------------------------ 6. métodos
    P("6. Métodos aplicados", "h1")
    P("Matriz de experimentos", "h2")
    desc = {e: v["desc"] for e, v in cfg["experiments"].items()}
    table([["ID", "Pipeline", "Descrição"]] + [[e, escape(str(v["pipeline"])), escape(desc[e])] for e, v in cfg["experiments"].items()],
          [.08, .2, .72])
    P("Todo frame é convertido para cinza e reduzido a lado máximo de 512 px antes do PDI, o que fixa a escala dos "
      "parâmetros. Ao final, é redimensionado sem distorção para 224×224 e replicado em 3 canais. Os resultados de cada "
      "pipeline ficam em cache em disco, e as métricas de qualidade da própria imagem são registradas.")
    fig(fig_quality(), "Figura 9. Qualidade da imagem após cada pipeline (linha pontilhada: E00). A HE estoura o contraste; o spec reduz a fração especular.")
    fig(ROOT / "docs" / "figures" / "gallery_grid.png", "Figura 10. Galeria: cada linha é um experimento, cada coluna um frame de uma família.", .62, private=True)
    fig(ROOT / "docs" / "figures" / "gallery_spec_steps.png", "Figura 11. Remoção de reflexo especular passo a passo.", .8, private=True)
    fig(ROOT / "docs" / "figures" / "gallery_homo.png", "Figura 12. Filtragem homomórfica num frame real: espectro, H(u,v) e resultado.", 1.0, private=True)
    P("Estágios de classificação", "h2")
    B(["<b>Estágio 1 (triagem):</b> todas as 14 variantes × tarefas A e B × 5 folds, com ResNet-18 congelada (probe, CPU) "
       "e SimpleCNN rápida (10 épocas, GPU).",
       f"<b>Estágio 2{'' if s2 else ' (pendente)'}:</b> fine-tuning completo (até 30 épocas, 3 sementes × 5 folds) de E00, do top-4 do "
       f"Estágio 1 ({', '.join(top4)}) e do E14, uma combinação gulosa escolhida só pela validação.",
       "<b>Braço clássico:</b> C1 (forma + textura + MLP), C2 (idem na ROI), C3 (SVM), C4 (Random Forest) e C5 (forma pelo "
       "ground truth + MLP, como referência superior)."])
    story.append(PageBreak())

    # ------------------------------------------------------------------ 7. resultados
    P("7. Resultados", "h1")
    P("7.1 Segmentação clássica da glote (H2 e H4)", "h2")
    P(f"Cinco métodos × quatro pré-processamentos × uma grade de parâmetros rodaram em todos os {fj['n_frames']:,} frames. "
      "Os parâmetros foram escolhidos só no treino de cada fold e avaliados no teste. ".replace(",", ".") +
      f"O melhor foi o <b>crescimento de região sobre CLAHE</b> (tol = {best['params'].get('tol')}): Dice "
      f"{rg.loc[(best['method'], best['preprocess']), 'dice_mean']:.3f}, falha de "
      f"{100 * rg.loc[(best['method'], best['preprocess']), 'fail_rate']:.1f}% e HD95 mediano de "
      f"{rg.loc[(best['method'], best['preprocess']), 'hd95_median']:.1f} px. Ele passou a alimentar a técnica <i>roi</i> e o M2. "
      "O CLAHE quase dobra o acerto do crescimento de região e reduz sua falha de ~20% para 2%. A homomórfica ajuda o "
      "watershed e o Otsu com morfologia. Já nos métodos de limiar global o realce piora, porque aumenta o contraste "
      "também fora da glote. <b>H4 é apoiada de forma condicional</b>: o realce certo para o método certo.")
    fig(fig_seg(), "Figura 13. Dice médio no teste por método × pré-processamento (esq.) e Δ Dice por vídeo vs raw, Wilcoxon com Holm (dir.; verde = melhora, vermelho = piora).")
    fig(ROOT / "docs" / "figures" / "segmentation_examples.png", "Figura 14. Exemplos de teste: original, ground truth e os três melhores métodos.", .9, private=True)
    fig(ROOT / "docs" / "figures" / "gallery_roi.png", "Figura 15. ROI: caixa pelo ground truth (verde) vs caixa pela segmentação clássica (vermelho).", .85, private=True)
    P("7.2 Classificação, Estágio 1", "h2")
    P(f"No baseline E00, o F1 por vídeo no teste foi {e00.get(('probe', 'A'), float('nan')):.3f} (A) e "
      f"{e00.get(('probe', 'B'), float('nan')):.3f} (B) no probe, e {e00.get(('simplecnn_fast', 'A'), float('nan')):.3f} (A) e "
      f"{e00.get(('simplecnn_fast', 'B'), float('nan')):.3f} (B) na SimpleCNN rápida. A Tarefa A é mais difícil: os "
      "distúrbios funcionais (como a disfonia por tensão muscular) são dinâmicos e pouco visíveis num frame isolado, como "
      "havia sido previsto. A Tarefa B, com lesões visíveis, rende mais.")
    fig(fig_stage1(), "Figura 16. F1 por vídeo de cada variante: validação (usada na escolha) e teste (só registro). Linha laranja: E00 no teste.")
    P("Teste estatístico de cada técnica contra o E00 (McNemar com Holm e IC bootstrap sobre as predições out-of-fold):", "p")
    fig(fig_forest(), "Figura 17. Δ F1 de cada técnica vs E00, com IC 95%. Todos os intervalos cruzam zero: os 52 vereditos são 'neutra'.")
    P("<b>Leitura:</b> no regime de triagem, <b>nenhuma técnica de PDI mudou de forma significativa a classificação</b>, nem "
      "na ResNet congelada nem na SimpleCNN. Há tendências: <i>lbp</i> e <i>edges</i> sobem na Tarefa B, e <i>roi_gt</i>/"
      "<i>roi</i> caem na B da SimpleCNN, possivelmente porque o recorte remove contexto útil. Essas tendências guiaram o "
      "top-4, mas precisam do Estágio 2 (15 pares por comparação e treino completo) para virar conclusão. Um resultado "
      "neutro também é informativo: as redes já extraem do frame cru, sozinhas, o que o PDI de realce entregaria.")
    if s2:
        P("7.3 Classificação, Estágio 2 (fine-tuning)", "h2")
        P(f"ResNet-18 (ImageNet) e SimpleCNN treinadas por completo (até 30 épocas, early stopping no F1 por vídeo da validação "
          f"interna) em E00, no top-4 do Estágio 1 ({', '.join(top4)}) e no E14 (<i>{escape(str(e14))}</i>), com 3 sementes × 5 folds "
          f"= 15 pares por comparação. E01, E07 e E08 rodaram com 1 semente só para H1 e H5. No E00, o F1 por vídeo foi "
          f"{e00.get(('resnet18', 'A'), float('nan')):.3f} (A) e {e00.get(('resnet18', 'B'), float('nan')):.3f} (B) na ResNet-18, e "
          f"{e00.get(('simplecnn', 'A'), float('nan')):.3f} (A) e {e00.get(('simplecnn', 'B'), float('nan')):.3f} (B) na SimpleCNN.")
        fig(fig_stage2(), "Figura. Δ F1 de cada técnica vs E00 no Estágio 2 (IC 95% bootstrap sobre o ensemble das sementes; verde = melhora, vermelho = piora).")
        rows = [["Técnica", "Modelo", "Tarefa", "ΔF1 (IC 95%)", "Wilcoxon p (Holm)", "McNemar p (Holm)", "Veredito"]]
        rows += [[f"{r.exp} {escape(str(r.pipeline))}", r.model, r.task, f"{r.oof_delta_f1:+.3f} [{r.ci_low:+.3f}; {r.ci_high:+.3f}]",
                  f"{r.wilcoxon_p_holm:.2g}", f"{r.mcnemar_p_holm:.2g}", f"<b>{r.verdict}</b>"] for r in v2.itertuples()]
        table(rows, [.2, .12, .08, .22, .13, .13, .12])
        P(f"<b>Leitura:</b> {int((v2.verdict == 'melhora').sum())} melhora(s), {int((v2.verdict == 'piora').sum())} piora(s) e "
          f"{int((v2.verdict == 'neutra').sum())} neutra(s). Com 15 pares, o Wilcoxon passa a ter poder; um veredito exige também "
          "que o IC do ΔF1 exclua zero (regra fixada antes dos resultados, em <i>DECISIONS.md</i>).")
    P("7.3 Braço clássico × CNN (H6)" if not s2 else "7.4 Braço clássico × CNN (H6)", "h2")
    fig(fig_classic(), "Figura 18. F1 por vídeo do braço clássico (roxo) e das CNNs no E00 (ciano).")
    fig(fig_perm(), "Figura 19. Importância por família de características (queda da acurácia balanceada ao permutar a família inteira).")
    P("As características clássicas, que são interpretáveis, ficam no mesmo patamar das CNNs. O Random Forest (C4) é o "
      "melhor na Tarefa B. Até aqui, H6 é <b>parcialmente apoiada</b> (competitivo em A), mas a parte 'inferior em B' "
      "<b>não se confirmou</b> contra a CNN de triagem." + (f" Contra a ResNet-18 do Estágio 2 a conclusão é: <b>{escape(hyp.get('H6', '—'))}</b>."
      if s2 and hyp else " A comparação definitiva usa a ResNet-18 do Estágio 2."))
    P("7.4 Aprendizado e interpretabilidade (preliminar)" if not s2 else "7.5 Aprendizado e interpretabilidade", "h2")
    if att:
        fig(fig_attention(), "Figura. Fração da energia do Grad-CAM na glote (máscara do especialista) e sobre os reflexos especulares; o traço preto é a área da região (o que um mapa uniforme daria).")
    fig(fig_curves(), "Figura 20. Curvas de aprendizado da SimpleCNN rápida (todas as variantes): F1 na validação e custo por época.")
    live = ROOT / "results" / "live"
    fig(live / "val.jpg", "Figura 21. Grad-CAM ao fim de uma época (exemplo do painel de telemetria): contorno verde e caixa = glote do especialista; a porcentagem é a atenção dentro dela.", .75, private=True)
    fig(live / "filters.jpg", "Figura 22. Filtros 3×3 aprendidos na 1ª camada da SimpleCNN (laranja = peso positivo, ciano = negativo): detectores de borda e contraste.", .45)
    P("Os mapas mostram que, no treino rápido, a rede dedica pouca atenção à glote (tipicamente menos de 5%) e responde "
      "também à moldura e a reflexos. É um primeiro indício para H1 e H5, a ser medido de forma sistemática no Estágio 2.")
    story.append(PageBreak())

    # ------------------------------------------------------------------ 8. hipóteses
    P("8. Estado das hipóteses", "h1")
    if hyp:
        P("Conclusões geradas pelo comando <i>report</i> a partir das regras pré-registradas (evidência completa em <i>docs/hipoteses.md</i>):")
        tests = {"H1": "F1 de E01 × E00 e energia do Grad-CAM sobre os reflexos (Wilcoxon por frame).",
                 "H2": "E03/E04 contra o E02 (McNemar + IC) e Δ Dice na segmentação.",
                 "H3": "ΔF1 da SimpleCNN × ΔF1 da ResNet nas mesmas técnicas (Wilcoxon).",
                 "H4": "Δ Dice por vídeo com realce × raw, 5 métodos × 3 realces (Wilcoxon, Holm).",
                 "H5": "ΔF1 do roi em B × A; roi × roi_gt (IC bootstrap).",
                 "H6": "C1 (forma + textura + MLP) × ResNet-18 no E00 (IC bootstrap)."}
        table([["Hipótese", "Teste", "Conclusão"], *[[k, tests.get(k, ""), escape(v)] for k, v in hyp.items()]], [.12, .5, .38])
    if not hyp:
        table([["Hipótese", "Evidência até agora", "Estado"],
           ["H1 spec", "F1: neutro nos dois modelos (Estágio 1). Atenção sobre reflexos: ainda não medida.", "Parcial (F1 mantém)"],
           ["H2 CLAHE/homo > HE", "Classificação: sem diferença significativa. Segmentação: CLAHE e homo ajudam conforme o método (HE não testada na segmentação).", "Não apoiada na classificação"],
           ["H3 PDI ajuda mais a SimpleCNN", "Estágio 1: todos neutros nos dois modelos.", "Pendente (Estágio 2)"],
           ["H4 realce melhora o Dice", f"{int((h4.verdict == 'melhora').sum())}/{len(h4)} comparações melhoram; o melhor par (região + CLAHE) +0,15.", "Apoiada condicionalmente"],
           ["H5 ROI ajuda mais em B", "Tendência oposta na SimpleCNN rápida (o recorte reduz o F1 em B); neutro no probe.", "Pendente / tendência contrária"],
           ["H6 clássico competitivo em A", "Competitivo em A e também em B contra a CNN de triagem.", "Parcial; definição no Estágio 2"]],
          [.2, .55, .25])

    # ------------------------------------------------------------------ 9. erros
    P("9. Erros encontrados, correções e melhorias", "h1")
    P("Registrar os erros faz parte do rigor do projeto; todos estão em <i>docs/DECISIONS.md</i>.")
    table([["Problema", "Impacto", "Correção"],
           ["Laryngoscope8 inacessível", "Bloqueava todo o projeto", "Troca para o BAGLS, mais rico (máscaras + status); a especificação foi reescrita."],
           ["Frames do BAGLS commitados por engano", "Dado de paciente no repositório privado", "Histórico reescrito (filter-repo + force push); .gitignore barra figuras com frames."],
           ["Watershed semeando só a maior região escura", "Dice ≈ 0,005 (método inútil)", "Núcleos por componente; Dice de 0,18 a 0,44. Correção feita antes da escolha do vencedor, que não mudou."],
           ["Chave 'classic' sobrescrevia hiperparâmetros", "MLP/SVM/RF sem configuração", "Renomeada para classic_experiments; teste de regressão."],
           ["Gabor do scikit-image", "8 s/frame (~13 h de extração)", "OpenCV getGaborKernel: 35 ms/frame (~240× mais rápido)."],
           ["Augmentation sorteada por lote", "32 imagens com a mesma transformação", "Sorteio por imagem."],
           ["<b>Augmentation sob AMP (GPU)</b>", "<b>~22% dos pixels saturados; 22 jobs inválidos</b>", "<b>Augmentation em float32 fora do autocast; teste de regressão; jobs em quarentena; Estágio 1 refeito.</b>"],
           ["Ranking pelo teste (risco)", "Viés de seleção", "Ranking e E14 usam só a validação interna."],
           ["Cache montado inteiro na RAM", "Processo do Estágio 2 encerrado por falta de memória", "Escrita em blocos num memmap (pico ~1 bloco)."]],
          [.27, .3, .43])
    P("Melhorias de infraestrutura", "h2")
    B(["Uma única CLI (<i>python -m laringopdi</i>) e a matriz inteira em YAML; jobs concluídos são pulados; cada execução "
       "registra config, semente, git e hash do pipeline.",
       "A RTX 3060 local substituiu o Colab: Estágio 1 em ~3,5 h e Estágio 2 estimado em ~16 h.",
       "65 testes automatizados, incluindo um ponta a ponta em CPU (~1 min) sobre um BAGLS sintético.",
       "O painel <b>LaringoMonitor</b> (janela nativa) mostra progresso e tempo previsto, CPU/GPU, temperatura, fan e a rede "
       "treinando ao vivo (lote atual, Grad-CAM, ativações, ajuste dos pesos, matriz de confusão). Foi ele que revelou o "
       "bug de AMP."])
    fig(live / "batch.jpg", "Figura 23. Lote de treino visto pelo painel após a correção: frames com augmentation, rótulo real e palpite da rede.", .7, private=True)

    # ------------------------------------------------------------------ 10. limitações e próximos passos
    P("10. Limitações", "h1")
    B(["Frames isolados: a dinâmica vibratória, essencial nos distúrbios funcionais, não é usada (análise temporal está fora do escopo).",
       "Classe estrutural pequena (54 vídeos, ~11 por fold de teste): intervalos de confiança largos na Tarefa B.",
       "Rótulos derivados de texto livre por regras; os casos ambíguos aguardam revisão humana.",
       "No Estágio 1 há só 5 pares por comparação; o Wilcoxon não alcança p < 0,05, por isso os vereditos usam McNemar + bootstrap.",
       "A HE não foi incluída no estudo de segmentação, o que deixa H2 incompleta nessa parte."])
    P("11. Próximo passo" if not s2 else "11. Próximos passos", "h1")
    P("O roteiro executável completo está em <i>docs/PROXIMOS_PASSOS.md</i>. Em resumo:")
    B([*([] if s2 else ["<b>Estágio 2:</b> busca gulosa do E14 → atualizar a matriz → 360 jobs de fine-tuning (ResNet-18 e SimpleCNN, 3 sementes × 5 folds), ~16 h na RTX 3060."]),
       *([] if att else ["<b>Fase 7:</b> Grad-CAM sistemático (atenção na glote e nos reflexos) e <i>report</i> final, com os vereditos do Estágio 2 e a conclusão de H1–H6.",
       ]),
       "<b>Fase 8:</b> protótipo Streamlit com os modelos treinados e reprodução a partir de um clone limpo.",
       "<b>Equipe:</b> revisão visual da galeria e da segmentação, revisão da tabela de rótulos, aviso à professora sobre a troca de dataset e colaboradores no GitHub."])
    P("Referências principais", "h2")
    refs = ["Gómez P. et al. BAGLS, a multihospital Benchmark for Automatic Glottis Segmentation. Scientific Data 7, 186 (2020).",
            "Gonzalez R. C., Woods R. E. Digital Image Processing. 4ª ed. Pearson, 2018.",
            "Otsu N. A threshold selection method from gray-level histograms. IEEE TSMC 9(1), 1979.",
            "Zuiderveld K. Contrast Limited Adaptive Histogram Equalization. Graphics Gems IV, 1994.",
            "Tomasi C., Manduchi R. Bilateral filtering for gray and color images. ICCV, 1998.",
            "Telea A. An image inpainting technique based on the fast marching method. J. Graphics Tools, 2004.",
            "Ojala T., Pietikäinen M., Mäenpää T. Multiresolution gray-scale and rotation invariant texture classification with LBP. IEEE TPAMI, 2002.",
            "Vincent L., Soille P. Watersheds in digital spaces. IEEE TPAMI, 1991.",
            "He K. et al. Deep Residual Learning for Image Recognition. CVPR, 2016.",
            "Selvaraju R. R. et al. Grad-CAM: Visual Explanations from Deep Networks. ICCV, 2017.",
            "Loshchilov I., Hutter F. Decoupled Weight Decay Regularization (AdamW). ICLR, 2019.",
            "Yin L. et al. Laryngoscope8: Laryngeal image dataset and classification of laryngeal disease based on attention mechanism. Pattern Recognition Letters 150, 2021."]
    for r in refs:
        story.append(Paragraph(escape(r), st["b"], bulletText="–"))

    def on_page(canvas, doc):
        canvas.saveState()
        canvas.setFont("Arial", 7.5); canvas.setFillColor(colors.HexColor(DIM))
        canvas.drawString(2 * cm, 1.2 * cm, "LaringoPDI · relatório técnico de andamento" + (" · versão pública" if publico else " · versão completa (uso interno)"))
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"{doc.page}")
        canvas.setStrokeColor(colors.HexColor(CY)); canvas.setLineWidth(1.2); canvas.line(2 * cm, A4[1] - 1.3 * cm, A4[0] - 2 * cm, A4[1] - 1.3 * cm)
        canvas.restoreState()

    out.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                            title="LaringoPDI — relatório técnico", author="Equipe LaringoPDI (UNIFOR)")
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--publico", action="store_true")
    a = ap.parse_args()
    build(Path(a.out), a.publico)
    print("ok", a.out)

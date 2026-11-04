"""Relatório final da Etapa 2 do LaringoPDI: os 9 itens do enunciado, no vocabulário da apresentação da Etapa 1.

Uso:
  python tools/relatorio_etapa2.py --out docs/relatorio_etapa2.pdf --publico       # sem frames do BAGLS (vai para o Git)
  python tools/relatorio_etapa2.py --out entrega/LaringoPDI_relatorio_etapa2.pdf   # com frames (fora do Git)

Todos os números saem de results/tables, logs/index_summary.json e data/splits/frames.json; nada é digitado à mão.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from laringopdi import config as C  # noqa: E402

T = ROOT / "results" / "tables"
FIG = ROOT / "docs" / "figures" / "relatorio_etapa2"
LOGO = ROOT / "LOGO_UNIFOR_3_page-0001-removebg-preview.png"
BAGLS = ROOT / "data" / "raw" / "bagls" / "test"
AVISO = "Ferramenta de apoio acadêmico. Não substitui avaliação médica."
EQUIPE = [("Nathan Linhares Dias Malheiros (representante)", "2220227"), ("Marcos Mendes Carneiro Furtado Bezerra", "2310532"),
          ("Marcos Vinícius dos Santos e Silva", "2220232"), ("Pedro Nobre Costa", "2210380"),
          ("Tales Rodrigues da Costa Sales Rios", "2220242")]

INK, DIM, GRID, BLUE, ORANGE, GREEN, GRAY = "#1B2A4A", "#5B6785", "#DDE2EC", "#2a78d6", "#eb6834", "#1baf7a", "#8A94AD"
plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.edgecolor": "#C9CFDD", "axes.labelcolor": DIM, "xtick.color": DIM, "ytick.color": INK,
                     "axes.titleweight": "bold", "axes.titlesize": 10, "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 200, "savefig.bbox": "tight"})

REALCE = {"raw": "sem realce", "stretch": "alargamento", "he": "equalização global", "clahe": "CLAHE"}
METODO = {"otsu": "Otsu", "otsu_morph": "Otsu + morfologia", "adaptive": "adaptativa", "adaptive_morph": "adaptativa + morfologia",
          "region_grow": "crescimento de região", "watershed": "watershed"}
ENTRADA = {"C6": "18 características + MLP", "E00": "CNN: frame original", "E01": "CNN: remoção de reflexo", "E02": "CNN: equalização global", "E03": "CNN: CLAHE",
           "E04": "CNN: homomórfica", "E06": "CNN: mediana"}
M18_SYSTEM = "C6"     # as 18 características prometidas; C1 é a variante maior, citada só nos testes adicionais
FAMILIA = {"f18shape": "forma", "f18int": "intensidade", "f18tex": "textura"}


def n(v: float, d: int = 3) -> str:
    return f"{v:.{d}f}".replace(".", ",")


def pc(v: float, d: int = 1) -> str:
    return f"{v:.{d}%}".replace(".", ",")


def fp(p: float) -> str:
    return "&lt; 0,001" if p < 0.001 else n(p)


def save(fig, name: str) -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    p = FIG / f"{name}.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def tab(name: str) -> pd.DataFrame:
    return pd.read_csv(T / name)


# ----------------------------------------------------------------------------- figuras sem imagens de pacientes
def fig_arquitetura() -> Path:
    fig, ax = plt.subplots(figsize=(10, 3.1))
    ax.axis("off")
    ax.set_xlim(0, 10.4)
    ax.set_ylim(0, 3.1)

    def box(x, y, w, t, c="#EEF2F9", ec="#9AA6C2"):
        ax.add_patch(plt.Rectangle((x, y), w, .72, fc=c, ec=ec, lw=1))
        ax.text(x + w / 2, y + .36, t, ha="center", va="center", fontsize=8.6, color=INK)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="->", color=DIM, lw=1.2))

    box(0, 1.2, 1.3, "Frame BAGLS\n(escala de cinza)")
    box(1.75, 1.2, 1.3, "Realce e\nfiltragem", "#DDEAFB", BLUE)
    arrow(1.3, 1.56, 1.75, 1.56)
    arrow(3.05, 1.72, 3.5, 2.4)
    arrow(3.05, 1.4, 3.5, .56)
    ax.text(3.5, 2.98, "Braço 1: PDI + MLP", fontsize=9, fontweight="bold", color=INK)
    x = 3.5
    for w, t in ((1.45, "Segmentação\nda glote"), (1.45, "18\ncaracterísticas"), (1.45, "MLP\n18 → 32 → 16 → 2")):
        box(x, 2.05, w, t)
        if x > 3.5:
            arrow(x - .2, 2.41, x, 2.41)
        x += w + .2
    ax.text(3.5, 1.04, "Braço 2: CNN", fontsize=9, fontweight="bold", color=INK)
    box(3.5, .2, 3.1, "ResNet-18 pré-treinada\n(aprende os próprios filtros)")
    box(6.8, .2, 1.45, "Grad-CAM\n(onde a rede olhou)")
    arrow(6.6, .56, 6.8, .56)
    box(8.9, 1.2, 1.45, "Saudável ou\nalterado", "#E4F6EF", GREEN)
    arrow(8.25, 2.41, 9.2, 1.92)
    arrow(8.25, .56, 9.2, 1.2)
    return save(fig, "arquitetura")


def fig_realce() -> Path:
    q = tab("enhance_quality.csv").set_index("technique").reindex(list(REALCE))
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.1), sharey=True)
    for a, (col, t, fmt) in zip(axes, (("entropy_mean", "Entropia (bits)", lambda v: n(v, 2)), ("rms_contrast_mean", "Contraste RMS (0 a 1)", lambda v: n(v, 2)),
                                       ("saturated_frac_mean", "Pixels em 0 ou 255", lambda v: pc(v)))):
        b = a.barh(list(REALCE.values()), q[col], color=[GRAY, GRAY, GRAY, BLUE], height=.62)
        a.bar_label(b, [fmt(v) for v in q[col]], padding=3, fontsize=8.5, color=INK)
        a.set_title(t, loc="left")
        a.set_xticks([])
        a.spines["bottom"].set_visible(False)
        a.set_xlim(0, q[col].max() * 1.28)
    axes[0].invert_yaxis()
    return save(fig, "realce")


def fig_filtragem() -> Path:
    d = tab("denoise_summary.csv")
    fs = {"none": "sem filtro", "mean3": "média 3×3", "mean7": "média 7×7", "mean15": "média 15×15", "median3": "mediana 3×3", "median5": "mediana 5×5", "lpf15": "passa-baixa σ = 15", "lpf40": "passa-baixa σ = 40"}
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.9), sharey=True)
    for a, (noise, title) in zip(axes, (("sp_5", "Sal e pimenta 5%"), ("sp_10", "Sal e pimenta 10%"), ("gauss_s20", "Gaussiano σ = 20"))):
        x = d[d["noise"] == noise].set_index("filter").reindex(list(fs))
        cols = [ORANGE if k.startswith("median") else BLUE if k.startswith("mean") else GREEN if k.startswith("lpf") else GRAY for k in fs]
        b = a.barh(list(fs.values()), x["psnr_mean"], color=cols, height=.66)
        a.bar_label(b, [n(v, 1) for v in x["psnr_mean"]], padding=3, fontsize=8, color=INK)
        a.set_title(title + " (PSNR, dB)", loc="left")
        a.set_xticks([])
        a.spines["bottom"].set_visible(False)
        a.set_xlim(0, d["psnr_mean"].max() * 1.2)
    axes[0].invert_yaxis()
    return save(fig, "filtragem")


def seg_pivot() -> pd.DataFrame:
    s = pd.concat([tab("segmentation_etapa1_summary.csv"), tab("segmentation_adaptive_morph_summary.csv")])
    p = s.pivot_table(index="method", columns="preprocess", values="dice_mean").reindex(index=list(METODO), columns=list(REALCE))
    return p


def fig_seg() -> Path:
    p = seg_pivot()
    fig, ax = plt.subplots(figsize=(7.6, 2.9))
    ax.imshow(p.values, cmap="Blues", vmin=0, vmax=.75, aspect="auto")
    ax.set_xticks(range(4), list(REALCE.values()))
    ax.set_yticks(range(len(p)), [METODO[i] for i in p.index])
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            v = p.values[i, j]
            ax.text(j, i, n(v, 3), ha="center", va="center", fontsize=9, color="white" if v > .45 else INK)
    for s in ax.spines.values():
        s.set_visible(False)
    return save(fig, "segmentacao")


def fig_sens() -> Path:
    c = tab("classif_aula.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.5), sharey=True)
    order = ["C6", "E00", "E02", "E03", "E06"]
    for a, task in zip(axes, "AB"):
        x = c[c["task"] == task].set_index("exp").reindex(order)
        b = a.barh([ENTRADA[e] for e in order], x["sensitivity"], color=[ORANGE, GRAY, BLUE, BLUE, BLUE], height=.62)
        a.bar_label(b, [n(v, 2) for v in x["sensitivity"]], padding=3, fontsize=8.5, color=INK)
        a.set_title(f"Tarefa {task}: sensibilidade por vídeo", loc="left")
        a.set_xlim(0, 1)
        a.set_xticks([])
        a.spines["bottom"].set_visible(False)
    axes[0].invert_yaxis()
    return save(fig, "sensibilidade")


def fig_importancia() -> Path:
    d = tab("importance_c6.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 1.8), sharey=True)
    for a, task in zip(axes, "AB"):
        x = d[d["task"] == task].set_index("family").reindex(list(FAMILIA))
        b = a.barh(list(FAMILIA.values()), x["drop_mean"], color=[ORANGE, GRAY, BLUE], height=.6)
        a.bar_label(b, [n(v, 3) for v in x["drop_mean"]], padding=3, fontsize=8.5, color=INK)
        a.set_title(f"Tarefa {task}: queda da acurácia ao embaralhar", loc="left")
        a.set_xlim(0, d["drop_mean"].max() * 1.25)
        a.set_xticks([])
        a.spines["bottom"].set_visible(False)
    axes[0].invert_yaxis()
    return save(fig, "importancia")


def fig_atencao() -> Path:
    d = tab("attention_aula.csv")
    labels = {"E00": "frame original", "E02": "equalização global", "E03": "CLAHE", "E06": "mediana"}
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.2), sharey=True)
    for a, task in zip(axes, "AB"):
        x = d[d["task"] == task].set_index("exp")
        vals = [x["ratio_E00"].iloc[0], *[x.loc[e, "ratio_exp"] for e in ("E02", "E03", "E06")]]
        b = a.barh(list(labels.values()), vals, color=[GRAY, BLUE, BLUE, BLUE], height=.62)
        a.bar_label(b, [n(v, 2) for v in vals], padding=3, fontsize=8.5, color=INK)
        a.axvline(1, color=INK, lw=.9, ls=":")
        a.set_title(f"Tarefa {task}: atenção na glote (1 = acaso)", loc="left")
        a.set_xticks([])
        a.spines["bottom"].set_visible(False)
        a.set_xlim(0, max(vals + [1]) * 1.22)
    axes[0].invert_yaxis()
    return save(fig, "atencao")


# ----------------------------------------------------------------------------- figuras com frames do BAGLS (só na versão completa)
def _frame(name: str):
    from laringopdi.pdi.registry import to_work_size
    cfg = C.load()
    g = to_work_size(cv2.imread(str(BAGLS / f"{name}.png"), cv2.IMREAD_GRAYSCALE), cfg["data"]["work_size"])
    m = cv2.resize((cv2.imread(str(BAGLS / f"{name}_seg.png"), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8), g.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
    return g, m


def _show(a, im, title: str, gt=None, pred=None) -> None:
    a.imshow(im, cmap="gray", vmin=0, vmax=255)
    if gt is not None:
        a.contour(gt, [.5], colors=[GREEN], linewidths=1.1)
    if pred is not None and pred.any():
        a.contour(pred, [.5], colors=[ORANGE], linewidths=1.1)
    a.set_title(title, fontsize=8.8, fontweight="normal")
    a.axis("off")


def fig_ex_realce() -> Path:
    from laringopdi.pdi.registry import DEFAULTS, TECHNIQUES
    g, _ = _frame("1881")
    ims = [("original", g), *[(REALCE[k], TECHNIQUES[k](g, {}, **DEFAULTS[k])) for k in ("stretch", "he", "clahe")]]
    fig, axes = plt.subplots(2, 4, figsize=(10, 3.9), gridspec_kw={"height_ratios": [3, 1]})
    for j, (t, im) in enumerate(ims):
        _show(axes[0, j], im, t)
        axes[1, j].hist(im.ravel(), 64, range=(0, 255), color=BLUE if t == "CLAHE" else GRAY)
        axes[1, j].set_yticks([])
        axes[1, j].tick_params(axis="x", labelsize=7, labelcolor=DIM)
    return save(fig, "ex_realce")


def fig_ex_filtro() -> Path:
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from laringopdi.etapa1 import noise_image
    from laringopdi.pdi.enhance import gauss_lpf_transfer
    from laringopdi.pdi.registry import TECHNIQUES
    g, _ = _frame("21")
    r = noise_image(g, "sp", .05, np.random.default_rng(0))
    a3, m3 = TECHNIQUES["mean"](r, {}, ksize=3), TECHNIQUES["median"](r, {}, ksize=3)
    spec = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(g.astype(float)))))
    fig, axes = plt.subplots(1, 6, figsize=(10, 3.6))
    _show(axes[0], r, f"sal e pimenta 5%\n{n(psnr(g, r, data_range=255), 1)} dB")
    _show(axes[1], a3, f"média 3×3\n{n(psnr(g, a3, data_range=255), 1)} dB")
    _show(axes[2], m3, f"mediana 3×3\n{n(psnr(g, m3, data_range=255), 1)} dB")
    axes[3].imshow(spec, cmap="gray")
    axes[3].set_title("espectro\nde Fourier", fontsize=8.8, fontweight="normal")
    axes[3].axis("off")
    axes[4].imshow(gauss_lpf_transfer(g.shape, 40), cmap="gray")
    axes[4].set_title("filtro H(u,v)\nσ = 40", fontsize=8.8, fontweight="normal")
    axes[4].axis("off")
    _show(axes[5], TECHNIQUES["gauss_lpf"](g, {}, sigma=40), "passa-baixa\nσ = 40")
    return save(fig, "ex_filtro")


def fig_ex_seg() -> Path:
    from laringopdi.pdi.segmentation import adaptive_morph_steps, apply_best, best_config, dice, segment
    g, gt = _frame("21")
    s = adaptive_morph_steps(g)
    otsu = segment(g, "otsu")
    fig, axes = plt.subplots(1, 6, figsize=(10, 3.6))
    _show(axes[0], g, "entrada\n(verde: referência)", gt)
    for a, (k, t) in zip(axes[1:4], (("limiar", "limiar adaptativo\njanela 51×51"), ("abertura", "abertura\ntira os pontinhos"), ("fechamento", "fechamento\nfecha os buracos"))):
        _show(a, s[k].astype(np.uint8) * 255, t)
    _show(axes[4], g, f"adaptativa + morf.\nDice {n(dice(s['componente'], gt), 2)}", gt, s["componente"])
    _show(axes[5], g, f"Otsu\nDice {n(dice(otsu, gt), 2)}", gt, otsu)
    p1 = save(fig, "ex_seg")
    g, gt = _frame("1881")
    bc = best_config()
    a_, b_ = apply_best(g, bc["method"], bc["params"], "raw"), apply_best(g, bc["method"], bc["params"], "clahe")
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.8))
    _show(axes[0], g, f"sem realce: Dice {n(dice(a_, gt), 2)}", gt, a_)
    _show(axes[1], g, f"com CLAHE: Dice {n(dice(b_, gt), 2)}", gt, b_)
    save(fig, "ex_ideia")
    return p1


def _trim(path: Path) -> Path:
    """Tira a faixa branca que sobra no fim da captura."""
    im = cv2.imread(str(path))
    rows = np.where((im[:, 400:] < 245).any(axis=(1, 2)))[0]
    return _write(path.stem, im[: rows.max() + 20])


def _write(name: str, im: np.ndarray) -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / f"{name}.png"
    cv2.imwrite(str(out), im)
    return out


# ----------------------------------------------------------------------------- PDF
def build(out: Path, publico: bool) -> None:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    from reportlab.lib.fonts import addMapping
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    fd = Path("C:/Windows/Fonts")
    for name, f in (("Arial", "arial.ttf"), ("Arial-Bold", "arialbd.ttf"), ("Arial-Italic", "ariali.ttf"), ("Arial-BoldItalic", "arialbi.ttf")):
        pdfmetrics.registerFont(TTFont(name, str(fd / f)))
    for b_, i_, name in ((0, 0, "Arial"), (1, 0, "Arial-Bold"), (0, 1, "Arial-Italic"), (1, 1, "Arial-BoldItalic")):
        addMapping("Arial", b_, i_, name)
    ink = colors.HexColor("#222B3D")
    st = {"h1": ParagraphStyle("h1", fontName="Arial-Bold", fontSize=13.5, leading=17, spaceBefore=10, spaceAfter=6, keepWithNext=1, textColor=colors.HexColor(INK)),
          "h2": ParagraphStyle("h2", fontName="Arial-Bold", fontSize=10.8, leading=14, spaceBefore=8, spaceAfter=3, keepWithNext=1, textColor=colors.HexColor(INK)),
          "p": ParagraphStyle("p", fontName="Arial", fontSize=10, leading=14.2, alignment=TA_JUSTIFY, spaceAfter=6, textColor=ink),
          "b": ParagraphStyle("b", fontName="Arial", fontSize=10, leading=13.8, leftIndent=13, bulletIndent=3, spaceAfter=3, textColor=ink),
          "cap": ParagraphStyle("cap", fontName="Arial", fontSize=8.5, leading=11, alignment=TA_CENTER, spaceBefore=3, spaceAfter=9, textColor=colors.HexColor(DIM)),
          "c": ParagraphStyle("c", fontName="Arial", fontSize=8.6, leading=11, textColor=ink),
          "ch": ParagraphStyle("ch", fontName="Arial-Bold", fontSize=8.6, leading=11, textColor=colors.white),
          "cv": ParagraphStyle("cv", fontName="Arial", fontSize=11.5, leading=16, alignment=TA_CENTER, textColor=ink),
          "cvt": ParagraphStyle("cvt", fontName="Arial-Bold", fontSize=16, leading=22, alignment=TA_CENTER, textColor=colors.HexColor(INK)),
          "note": ParagraphStyle("note", fontName="Arial", fontSize=8.8, leading=12, textColor=colors.HexColor("#6B4A12"), backColor=colors.HexColor("#FFF4E5"), borderPadding=6, spaceBefore=4, spaceAfter=10)}
    W = A4[0] - 5 * cm
    story: list = []
    P = lambda t, s="p": story.append(Paragraph(t, st[s]))
    B = lambda items: [story.append(Paragraph(i, st["b"], bulletText="•")) for i in items]
    count = {"fig": 0, "q": 0}

    def fig(path, caption: str, width: float = 1.0, private: bool = False) -> None:
        count["fig"] += 1
        cap = f"Figura {count['fig']}: {caption}"
        if private and publico:
            story.append(Paragraph(f"[{cap} Omitida nesta versão pública: mostra frames do BAGLS; está na versão entregue no AVA.]", st["note"]))
            return
        p = Path(path)
        h, w = cv2.imread(str(p)).shape[:2]
        iw = W * width
        story.append(KeepTogether([Image(str(p), width=iw, height=iw * h / w), Paragraph(cap, st["cap"])]))

    def quadro(title: str, rows: list, widths: list[float]) -> None:
        count["q"] += 1
        data = [[Paragraph(str(c), st["ch" if i == 0 else "c"]) for c in r] for i, r in enumerate(rows)]
        t = Table(data, colWidths=[W * x for x in widths], repeatRows=1)
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .3, colors.HexColor("#C9CFDD")), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(INK)),
                               ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4), ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
        story.append(Paragraph(f"Quadro {count['q']}: {title}", ParagraphStyle("qt", parent=st["c"], fontName="Arial-Bold", spaceBefore=4, spaceAfter=3, keepWithNext=1)))
        story.append(t)
        story.append(Spacer(1, 9))

    # ------------------------------------------------------------- dados
    cfg = C.load()
    idx = json.loads((ROOT / "logs" / "index_summary.json").read_text(encoding="utf-8"))
    fj = json.loads((ROOT / "data" / "splits" / "frames.json").read_text(encoding="utf-8"))["meta"]
    hyp = json.loads((T / "hypotheses_etapa1.json").read_text(encoding="utf-8"))
    fam, tv = idx["videos_per_family"], idx["task_videos"]
    q = tab("enhance_quality.csv").set_index("technique")
    den = tab("denoise_summary.csv").set_index(["noise", "filter"])
    piv = seg_pivot()
    am = tab("seg_adaptive_morph.csv").set_index("comparison")
    p1 = tab("seg_p1.csv")
    cl = tab("classif_aula.csv").set_index(["exp", "task"])
    m18 = tab("mlp18_p4.csv")
    m18 = m18[(m18["task"] == "B") & (m18["system"] == M18_SYSTEM)].iloc[0]
    sens = tab("sens_c6_resnet.csv").set_index("task")
    ver = tab("verdicts_aula_resnet18.csv")
    v_old = tab("verdicts_resnet18.csv").set_index(["exp", "task"])
    hc = tab("he_clahe_resnet18.csv").set_index("task")
    att = tab("attention_aula.csv")
    tm = tab("timing.csv").set_index("name")["median_ms"]
    imp = tab("importance_c6.csv").set_index(["task", "family"])["drop_mean"]
    e01 = ver[(ver["exp"] == "E01") & (ver["task"] == "A")].iloc[0]
    melhoras = ver[ver["verdict"] != "neutra"]
    outras = ver[ver["exp"].isin(["E02", "E03", "E06"])]

    if not publico:
        fig_ex_seg()

    # ------------------------------------------------------------- capa
    if LOGO.exists():
        h_, w_ = cv2.imread(str(LOGO)).shape[:2]
        logo = Image(str(LOGO), width=4.6 * cm, height=4.6 * cm * h_ / w_)
        logo.hAlign = "CENTER"
        story += [logo, Spacer(1, 12)]
    story += [Paragraph("UNIVERSIDADE DE FORTALEZA", st["cv"]), Paragraph("PROCESSAMENTO DIGITAL DE IMAGENS", st["cv"]), Paragraph("Prof.ª Lyndainês Santos", st["cv"]), Spacer(1, 2 * cm),
              Paragraph("LARINGOPDI: PROCESSAMENTO DIGITAL DE IMAGENS APLICADO À LARINGOSCOPIA", st["cvt"]), Spacer(1, 10),
              Paragraph("Relatório final do projeto extensionista – Etapa 2", st["cv"]), Spacer(1, 2 * cm)]
    for nome, mat in EQUIPE:
        story.append(Paragraph(f"{nome} ({mat})", st["cv"]))
    story += [Spacer(1, 2 * cm), Paragraph("FORTALEZA", st["cv"]), Paragraph("2026", st["cv"]), Spacer(1, 18), Paragraph(AVISO, ParagraphStyle("av", parent=st["cv"], fontName="Arial-Bold", fontSize=10)), PageBreak()]

    # ------------------------------------------------------------- 1
    P("1 Problema social e público-alvo", "h1")
    P("Professores e demais profissionais da voz estão mais expostos a alterações nas pregas vocais, como nódulos, pólipos, cistos e edema de Reinke. O distúrbio de voz é o principal "
      "problema de saúde associado ao afastamento de professores da sala de aula (17,7%; Medeiros e Vieira, 2019). A prevalência ao longo da vida é de 63% entre professores e 35,8% entre não "
      "professores (Behlau et al., 2012), e o Ministério da Saúde reconhece o Distúrbio de Voz Relacionado ao Trabalho como agravo à saúde do trabalhador (Brasil, 2018).")
    P("A identificação dessas alterações depende da leitura visual da laringoscopia, e a imagem tem limitações recorrentes: reflexo de luz, quadros escuros, baixo contraste e uma glote pequena no quadro. "
      "Os <b>beneficiários</b> são os professores e demais profissionais da voz. Os <b>usuários</b> são quem interpreta o exame: otorrinolaringologistas, fonoaudiólogos e residentes. "
      "A ferramenta apoia a análise de imagens estáticas em contexto acadêmico de triagem; não faz diagnóstico nem substitui a avaliação médica.")

    # ------------------------------------------------------------- 2
    P("2 Objetivos", "h1")
    P("<b>Objetivo geral:</b> desenvolver e avaliar um protótipo baseado em técnicas de Processamento Digital de Imagens para realçar, segmentar e caracterizar a glote em imagens de laringoscopia, "
      "investigando a contribuição de cada técnica para a delimitação da região e para a distinção entre pregas vocais saudáveis e alteradas.")
    quadro("Objetivos específicos e o que foi entregue", [["Objetivo específico (Etapa 1)", "O que foi feito", "Seção"],
           ["1. Investigar realce e redução de ruído", "Alargamento, equalização global e CLAHE; média, mediana e passa-baixa gaussiano. Medidos com entropia, contraste RMS, PSNR e SSIM.", "6.1 e 6.2"],
           ["2. Implementar a segmentação por Otsu e adaptativa, com morfologia", "Implementadas, com abertura, fechamento e seleção do componente central.", "6.3"],
           ["3. Avaliar contra as máscaras de referência", "Dice e IoU por vídeo, em cinco partições.", "6.3"],
           ["4. Extrair características e treinar uma MLP", "18 características (6 de forma, 6 de intensidade, 6 de textura) e MLP 18–32–16–2.", "6.4"],
           ["5. Comparar com uma CNN pré-treinada, com validação agrupada por vídeo", "ResNet-18 com transferência de aprendizado, com o frame original e com o frame tratado por PDI, e Grad-CAM.", "6.4 e 6.5"]], [.36, .52, .12])

    # ------------------------------------------------------------- 3
    P("3 Arquitetura da solução e ferramentas", "h1")
    fig(fig_arquitetura(), "os dois braços comparados nas mesmas condições.")
    P(f"<b>Dados.</b> BAGLS (Gómez et al., 2020; CC BY-NC-SA 4.0): {idx['n_frames']:,} frames de {idx['n_videos']} vídeos de sete hospitais, com máscara da glote feita por especialistas. ".replace(",", ".") +
      f"Usamos até {fj['frames_per_video']} frames por vídeo, uniformemente espaçados, nos {fj['n_videos']} vídeos com rótulo definido ({fj['n_frames']:,} frames; ".replace(",", ".") +
      f"{fam['healthy']} saudáveis, {fam['functional']} funcionais e {fam['structural']} com lesão estrutural). Os {fam['unknown']} vídeos de estado desconhecido ou fora dessas categorias ficaram de fora.")
    P(f"<b>Tarefas.</b> Tarefa A: saudável × alterado ({tv['A']['0']} × {tv['A']['1']} vídeos). Tarefa B: saudável × lesão estrutural ({tv['B']['0']} × {tv['B']['1']} vídeos).")
    P("<b>Validação.</b> Cinco partições agrupadas por vídeo: frames de um mesmo vídeo nunca ficam ao mesmo tempo no treino e no teste. A nota é calculada por vídeo, com a média das probabilidades dos frames. "
      "Parâmetros são escolhidos só no treino. Para comparar técnicas, usamos o teste de McNemar e intervalos de confiança de 95%. Quando há várias comparações ao mesmo tempo, os valores de p são corrigidos pelo método de Holm, para "
      "não contar sorte como resultado. As regras de veredito foram registradas antes dos resultados.")
    P("<b>Ferramentas.</b> Python 3.11, OpenCV e scikit-image (PDI), scikit-learn (MLP), PyTorch (ResNet-18), Streamlit (protótipo). O projeto roda em um computador comum; a GPU só acelera o treino da CNN.")

    # ------------------------------------------------------------- 4
    P("4 Técnicas de PDI implementadas", "h1")
    P("A regra da Etapa 1 foi mantida: começar pela técnica vista em aula e usar técnica externa só como comparação ou quando a de aula não bastar.")
    quadro("Técnicas e origem", [["Técnica", "O que faz na imagem", "Origem"],
           ["Histograma, alargamento, equalização global, CLAHE (blocos 8×8, limite 2,0)", "Recupera detalhe em quadro escuro. O CLAHE trabalha por blocos e limita o ganho.", "Aula"],
           ["Média e mediana", "Reduzem ruído. A mediana ignora valores extremos.", "Aula"],
           ["Transformada de Fourier e passa-baixa gaussiano", "Corta altas frequências; o σ controla quanto.", "Aula"],
           ["Abertura, fechamento, maior componente (elipse 5×5)", "Limpa a máscara e mantém a região central.", "Aula"],
           ["Otsu e limiarização adaptativa (janela padrão 51×51, ajustada só no treino)", "Separa a glote, que é escura, do tecido.", "Lista da disciplina"],
           ["Sobel e Canny", "Mostram o contorno das pregas; alimentam as características de textura.", "Lista da disciplina"],
           ["Extração de 18 características", "Transforma a região em números para a MLP.", "Lista da disciplina"],
           ["Crescimento de região, watershed, remoção de reflexo, homomórfica, LBP, GLCM", "Comparações previstas no Quadro 3 da Etapa 1.", "Literatura"],
           ["MLP, ResNet-18, Grad-CAM", "Decidem saudável ou alterado; o Grad-CAM mostra onde a rede olhou.", "Lista; IA Computacional"]], [.42, .42, .16])
    if not publico:
        fig(fig_ex_realce(), "realce em um frame escuro (test/1881), com o histograma de cada versão.", private=True)
        fig(fig_ex_filtro(), "média × mediana em sal e pimenta (test/21), e filtragem no domínio da frequência.", private=True)
    else:
        fig(None, "realce em um frame escuro, com histogramas.", private=True)
        fig(None, "média × mediana em sal e pimenta e filtragem na frequência.", private=True)

    # ------------------------------------------------------------- 5
    P("5 Demonstração do protótipo", "h1")
    P("O protótipo é uma aplicação Streamlit que recebe um frame do BAGLS ou um arquivo enviado e mostra cada etapa do processamento até a classificação, sempre com o aviso de que não substitui a avaliação médica. "
      "As abas seguem a ordem da apresentação da Etapa 1:")
    quadro("Abas do protótipo", [["Aba", "O que mostra"], ["1. Realce", "Original, alargamento, equalização global e CLAHE, com histograma, contraste RMS e entropia."],
           ["2. Filtragem e Fourier", "Ruído à escolha; média × mediana com PSNR e SSIM; espectro e passa-baixa gaussiano com σ ajustável."],
           ["3. Segmentação", "Passo a passo da adaptativa + morfologia; comparação com Otsu e crescimento de região, com Dice e IoU; Sobel e Canny."],
           ["4. PDI + MLP × CNN", "Braço 1: as 18 características e a saída da MLP. Braço 2: ResNet-18, com Grad-CAM."],
           ["5. Resultados", "As seis hipóteses e as tabelas do estudo."]], [.26, .74])
    for i, (t_) in enumerate(("aba 1, Realce.", "aba 3, Segmentação.", "aba 4, PDI + MLP × CNN.")):
        k = (0, 2, 3)[i]
        pth = ROOT / "docs" / "figures" / "app" / f"aba{k}.png"
        fig(_trim(pth) if pth.exists() and not publico else None, t_, .78, private=True)

    # ------------------------------------------------------------- 6
    P("6 Resultados", "h1")
    P("Os valores abaixo são médias do conjunto de teste das cinco partições, não exemplos de uma imagem. Na apresentação da Etapa 1, os números de exemplo eram de imagens isoladas; aqui eles são medidos no conjunto inteiro, e isso muda várias conclusões.")
    P("6.1 Realce (H1)", "h2")
    fig(fig_realce(), f"efeito do realce em {int(q.loc['raw', 'n_frames']):,} frames. ".replace(",", ".") + "A barra azul marca o CLAHE.")
    he = p1[(p1["comparison"] == "clahe_vs_he")].set_index("method")
    P(f"O CLAHE teve a maior entropia ({n(q.loc['clahe', 'entropy_mean'], 2)} bits, contra {n(q.loc['raw', 'entropy_mean'], 2)} sem realce) e a equalização global o maior contraste ({n(q.loc['he', 'rms_contrast_mean'], 2)}). "
      f"A expectativa de que a equalização \"estoura\" o tecido não se sustentou: o frame sem realce já tem {pc(q.loc['raw', 'saturated_frac_mean'])} de pixels em 0 ou 255, quase todos do fundo preto, e a equalização quase não muda isso ({pc(q.loc['he', 'saturated_frac_mean'])}). "
      f"O CLAHE baixa para {pc(q.loc['clahe', 'saturated_frac_mean'])} porque tira esses pixels do zero. Na segmentação, o melhor par usa CLAHE, mas ele venceu a equalização em só {hyp['H1']['numbers']['methods_clahe_wins']} dos 5 métodos comparados; a equalização venceu em {hyp['H1']['numbers']['methods_he_wins']}. <b>H1: {hyp['H1']['verdict']}.</b>")
    P("6.2 Filtragem e Fourier (H2)", "h2")
    fig(fig_filtragem(), "PSNR de cada filtro em ruído adicionado de propósito (quanto maior, melhor).")
    s5 = den.loc["sp_5"]
    P(f"Em sal e pimenta a 5%, a mediana 3×3 chegou a {n(s5.loc['median3', 'psnr_mean'], 1)} dB e SSIM {n(s5.loc['median3', 'ssim_mean'], 2)}, contra {n(s5.loc['mean3', 'psnr_mean'], 1)} dB e {n(s5.loc['mean3', 'ssim_mean'], 2)} da média 3×3. <b>H2: {hyp['H2']['verdict']}.</b> "
      f"Em ruído gaussiano, a vantagem quase some: a σ = 20, a mediana 3×3 e a média 3×3 ficam praticamente iguais ({n(den.loc[('gauss_s20', 'median3'), 'psnr_mean'], 1)} e {n(den.loc[('gauss_s20', 'mean3'), 'psnr_mean'], 1)} dB). A grande vantagem da mediana 3×3 é específica do sal e pimenta. No domínio da frequência, o passa-baixa reduz o ruído gaussiano ({n(den.loc[('gauss_s20', 'none'), 'psnr_mean'], 1)} dB sem filtro, {n(den.loc[('gauss_s20', 'lpf40'), 'psnr_mean'], 1)} dB com σ = 40), como previsto.")
    P("6.3 Segmentação da glote (H3)", "h2")
    fig(fig_seg(), "Dice médio por método e realce. O crescimento de região e o watershed são técnicas externas, usadas como comparação.", .84)
    if publico:
        fig(None, "passo a passo da limiarização adaptativa com morfologia.", private=True)
        fig(None, "o exemplo da Etapa 1 reproduzido.", private=True)
    else:
        fig(FIG / "ex_seg.png", "passo a passo da adaptativa + morfologia (test/21). Verde: referência; laranja: segmentação obtida.", private=True)
    o = am.loc["adaptive_morph_vs_otsu_morph"]
    rg = am.loc["region_grow_clahe_vs_best_adaptive_morph"]
    P(f"A limiarização adaptativa venceu o Otsu com folga sobre o frame sem realce (Dice {n(piv.loc['adaptive', 'raw'])} × {n(piv.loc['otsu', 'raw'])}). <b>H3: {hyp['H3']['verdict']}.</b> "
      f"O método prometido na Etapa 1, adaptativa + morfologia, ficou em {n(piv.loc['adaptive_morph', 'raw'])}, sem diferença detectável para Otsu + morfologia ({n(o['dice_a'])}; p corrigido {fp(o['p_holm'])}). "
      f"Portanto, o 0,97 do slide 9 era o exemplo de uma imagem, e a média no conjunto é bem menor.")
    P(f"Como as técnicas de aula não bastaram, valeu a regra da Etapa 1 e entrou a comparação externa: <b>crescimento de região com CLAHE foi o melhor método</b>, com Dice {n(rg['dice_b'])}, acima da melhor versão da adaptativa + morfologia ({n(rg['dice_a'])}; p {fp(rg['p_holm'])}). "
      f"Sobre a adaptativa + morfologia, o CLAHE atrapalhou ({n(am.loc['clahe_vs_raw_in_adaptive_morph', 'dice_a'])} → {n(am.loc['clahe_vs_raw_in_adaptive_morph', 'dice_b'])}): o efeito do realce depende do método.")
    if not publico:
        fig(FIG / "ex_ideia.png", "o caso escuro de pólipo (test/1881) com o método final. É um exemplo de uma imagem, como na Etapa 1.", .56, private=True)
    P("6.4 Classificação: PDI + MLP × CNN (H4 e H6)", "h2")
    P("A sensibilidade é a métrica prioritária, porque deixar passar um caso alterado é o erro mais grave numa triagem.")
    rows = [["Sistema", "Tarefa", "Sensibilidade", "Especificidade", "F1", "AUC"]]
    for task in "AB":
        for e in ("C6", "E00", "E02", "E03", "E06"):
            r = cl.loc[(e, task)]
            rows.append([ENTRADA[e], task, n(r["sensitivity"], 2), n(r["specificity"], 2), n(r["f1"], 2), n(r["auc"], 2)])
    quadro("desempenho por vídeo (média das cinco partições, mesma semente)", rows, [.38, .09, .17, .18, .09, .09])
    fig(fig_sens(), "sensibilidade nas duas tarefas. Laranja: MLP; cinza: CNN sem PDI; azul: CNN com PDI de aula.")
    sa, sb = sens.loc["A"], sens.loc["B"]
    P(f"Na <b>Tarefa B</b>, o F1 da MLP foi {n(m18['f1_classic'])} contra {n(m18['f1_resnet18'])} da ResNet-18, com intervalo de confiança da diferença de {n(m18['ci_low'])} a {n(m18['ci_high'])}. Como o intervalo contém o zero, a regra dá <b>H4: {hyp['H4']['verdict']}</b>, mas é uma confirmação frágil: o intervalo é quase todo negativo, e a acurácia da ResNet-18 é maior ({n(m18['acc_resnet18'])} × {n(m18['acc_classic'])}; McNemar {fp(m18['mcnemar_p'])}). "
      f"Na sensibilidade, o quadro se inverte: contando todos os vídeos de uma vez, a MLP encontrou mais casos ({n(sb['sens_c6'], 2)} × {n(sb['sens_resnet18'], 2)}; p corrigido {fp(sb['p_holm'])}), à custa de mais alarmes falsos em saudáveis. Com poucos vídeos estruturais ({int(sb['n_altered'])}), essa diferença tem incerteza grande (intervalo de {n(sb['ci_low'], 2)} a {n(sb['ci_high'], 2)}).")
    P(f"Na <b>Tarefa A</b>, a ResNet-18 foi mais sensível ({n(sa['sens_resnet18'], 2)} × {n(sa['sens_c6'], 2)}; p corrigido {fp(sa['p_holm'])}). <b>Nenhum sistema chegou a uma sensibilidade adequada para triagem nessa tarefa.</b> É um resultado esperado em parte: distúrbios funcionais são dinâmicos e podem não aparecer numa imagem parada.")
    fig(fig_importancia(), "importância de cada grupo de características na MLP.", .86)
    P(f"Ao contrário do esperado, a forma não foi o grupo mais importante: na Tarefa B, a queda foi de {n(imp.loc[('B', 'f18shape')])} para forma, {n(imp.loc[('B', 'f18int')])} para intensidade e {n(imp.loc[('B', 'f18tex')])} para textura. <b>H6: {hyp['H6']['verdict']}.</b> A segmentação ainda erra bastante (Dice {n(piv.loc['region_grow', 'clahe'])} no melhor caso), e isso pode prejudicar as medidas de forma; este estudo não separa as duas explicações.")
    P("6.5 O PDI ajuda a CNN? (H5)", "h2")
    labelv = {"E01": "remoção de reflexo", "E02": "equalização global", "E03": "CLAHE", "E04": "homomórfica", "E06": "mediana"}
    rows = [["Técnica", "Tarefa", "F1 sem PDI", "F1 com PDI", "Diferença (IC 95%)", "p corrigido", "Veredito"]]
    for _, r in ver.iterrows():
        rows.append([labelv[r["exp"]] + (f" ({int(r['n_seeds'])} sementes)" if r["n_seeds"] > 1 else ""), r["task"], n(r["f1_base"]), n(r["f1_exp"]), f"{n(r['oof_delta_f1'])} ({n(r['ci_low'])} a {n(r['ci_high'])})", fp(r["mcnemar_p_holm"]), r["verdict"]])
    quadro("técnicas de PDI como entrada da ResNet-18, contra o frame original", rows, [.28, .08, .1, .1, .22, .1, .12])
    nv = len(outras[outras["verdict"] == "neutra"])
    P(f"As técnicas de aula testadas na CNN (equalização global, CLAHE e mediana) não mudaram a classificação de forma detectável: as {nv} comparações ficaram neutras. Entre CLAHE e equalização também não houve diferença (McNemar {fp(hc.loc['A', 'mcnemar_p'])} na Tarefa A e {fp(hc.loc['B', 'mcnemar_p'])} na B).")
    if len(melhoras):
        P(f"A única comparação fora de neutra foi a remoção de reflexo, técnica da literatura, na Tarefa A (F1 {n(e01['f1_base'])} → {n(e01['f1_exp'])}; p corrigido {n(e01['mcnemar_p_holm'])}). <b>O resultado é fraco e precisa de confirmação:</b> usa uma semente só e depende da família de comparações. "
          f"Os mesmos dados, corrigidos numa família maior de oito técnicas, ficam neutros (p corrigido {n(v_old.loc[('E01', 'A'), 'mcnemar_p_holm'])}). Uma explicação provável é que a CNN pré-treinada aprende a lidar com a imagem crua e, por isso, ganha pouco com o realce; este estudo não testa essa explicação.")
    fig(fig_atencao(), "atenção do Grad-CAM na glote, normalizada pela área da glote, na ResNet-18.", .92)
    ae = att.set_index(["exp", "task"])
    sig = att[att["p_holm"] < .05]
    desc = "; ".join(f"{labelv[r['exp']]} na Tarefa {r['task']} ({n(r['ratio_E00'], 2)} → {n(r['ratio_exp'], 2)})" for _, r in sig.iterrows()) or "nenhuma"
    P(f"Quanto à atenção, as mudanças com significância nas técnicas de aula foram: {desc}. A regra pré-registrada avalia 24 combinações, que incluem variantes além das apresentadas: {hyp['H5']['numbers']['up']} sobem e {hyp['H5']['numbers']['down']} caem. <b>H5: {hyp['H5']['verdict']}.</b> Atenção maior na glote também não é o mesmo que nota maior: o CLAHE aumentou a atenção nas duas tarefas sem mudar a classificação de forma detectável.")
    P("6.6 Tempo de processamento", "h2")
    P(f"O CLAHE leva {n(tm['E03 (clahe)'], 1)} ms por frame, a segmentação final {n(tm['region_grow + clahe'], 1)} ms, as 18 características {n(tm['18 características (C6)'], 1)} ms e a MLP {n(tm['MLP C6 (1 vetor)'], 1)} ms. A ResNet-18 leva {n(tm['resnet18 (CPU, batch 1)'], 1)} ms por imagem em CPU. Todo o pipeline cabe num computador comum, como proposto.")
    P("6.7 Resumo das hipóteses", "h2")
    quadro("hipóteses apresentadas na Etapa 1, com as regras registradas antes dos resultados", [["", "Hipótese", "Veredito", "O que encontramos"],
           *[[k, r["statement"], f"<b>{r['verdict']}</b>", r["plain"]] for k, r in hyp.items()]], [.06, .2, .17, .57])

    # ------------------------------------------------------------- 7
    P("7 Limitações", "h1")
    B([f"<b>Sensibilidade baixa.</b> Na Tarefa A, a melhor das CNNs do Quadro 4 encontra cerca de {n(max(cl.loc[(e, 'A'), 'sensitivity'] for e in ('E00', 'E02', 'E03', 'E06')), 2)} dos casos alterados. Isso não basta para triagem, e a ferramenta não deve ser usada para isso sem novos estudos.",
       "<b>Imagem parada.</b> Os frames são tratados um a um. A vibração das pregas, importante nos distúrbios funcionais, não é usada.",
       f"<b>Poucos vídeos estruturais.</b> São {fam['structural']} vídeos na Tarefa B, cerca de 11 por partição de teste, e os intervalos de confiança ficam largos.",
       f"<b>Segmentação ainda limitada.</b> Dice médio de {n(piv.loc['region_grow', 'clahe'])} no melhor método. O protótipo funciona melhor em imagens do BAGLS; em outras origens o resultado pode ser pior.",
       "<b>Técnicas na CNN com uma semente.</b> Equalização, CLAHE, mediana e remoção de reflexo foram medidas com uma semente; diferenças pequenas podem mudar com outra.",
       "<b>Sobel e Canny.</b> Aparecem no protótipo e nas características, mas não foram avaliados com Dice como método próprio de segmentação.",
       "<b>Sem validação clínica.</b> Os dados vêm de hospitais da Europa e dos EUA, e não houve teste com profissionais nem com pacientes."])

    # ------------------------------------------------------------- 8
    P("8 Contribuição potencial para o público-alvo", "h1")
    P("Para o <b>usuário</b> (quem lê o exame), o protótipo mostra o que cada técnica faz na imagem e indica, com medidas, quais ajudam a delimitar a glote: um frame escuro fica legível com o realce, e a glote aparece contornada ao lado da referência. "
      "Para a <b>formação</b> de residentes e estudantes, serve como material para entender por que uma imagem é difícil de ler. Para os <b>professores</b>, que são os beneficiários, a contribuição é indireta: leitura mais clara do exame e um ponto de partida medido para futuras ferramentas de triagem. "
      "O projeto não entrega uma triagem pronta, e dizer isso também faz parte da contribuição.")

    # ------------------------------------------------------------- 9
    P("9 Conclusões e possibilidades de melhoria", "h1")
    P("Voltando à pergunta de pesquisa: <b>a combinação que melhor delimitou a glote foi CLAHE com crescimento de região</b>, e o efeito do realce depende do método, podendo até piorar a segmentação. Entre as técnicas de aula, a mediana é a escolha certa contra sal e pimenta, e a limiarização adaptativa supera o Otsu. "
      "As características extraídas da glote distinguem saudáveis de lesões estruturais com F1 próximo ao da CNN, mas <b>não distinguem bem saudáveis de alterados em geral</b>. Na classificação com a CNN, o realce de aula não mudou o resultado de forma detectável. Nos nossos dados, o PDI fez mais diferença para um método de regras fixas do que para uma rede que aprende os próprios filtros. "
      "Nem todas as hipóteses se confirmaram, e isso é resultado válido, porque as regras foram registradas antes.")
    P("<b>Melhorias possíveis:</b>")
    B(["ajustar o limiar de decisão para priorizar a sensibilidade, escolhendo-o só na validação;", "repetir as técnicas de aula e a remoção de reflexo com mais sementes, para confirmar ou descartar o efeito na Tarefa A;",
       "usar uma sequência de frames do vídeo, para captar a vibração das pregas;", "obter mais vídeos de lesão estrutural;", "avaliar Sobel e Canny como método próprio de segmentação;", "testar o protótipo com profissionais de saúde."])

    # ------------------------------------------------------------- mudanças
    P("10 Mudanças em relação à Etapa 1", "h1")
    quadro("mudanças e justificativas", [["Item", "Etapa 1", "Etapa 2", "Justificativa"],
           ["Método final de segmentação", "Otsu e adaptativa + morfologia", "Os dois foram feitos; o melhor foi crescimento de região com CLAHE", "O Quadro 3 da Etapa 1 previa técnica externa se as de aula não bastassem; a melhor adaptativa + morfologia ficou bem abaixo."],
           ["Tarefa B", "Saudável × lesão estrutural", f"{fam['structural']} vídeos, incluindo cicatriz, sulco, rigidez e irregularidade de mucosa", "Com só as lesões listadas, a classe ficava abaixo do mínimo de 50 vídeos adotado no método."],
           ["Nota por vídeo", "Validação agrupada por vídeo", "Além do agrupamento, a nota principal é por vídeo", "Os frames de um vídeo não são independentes, e a decisão é sobre o paciente."]], [.2, .2, .28, .32])
    c1 = tab("mlp18_p4.csv").set_index(["system", "task"]).loc[("C1", "B")]
    P("<b>Testes adicionais.</b> Além do apresentado, avaliamos outras variantes: uma segunda CNN treinada do zero, recortes pela região da glote, entradas de bordas e textura, e um conjunto maior de características. "
      f"Na família maior de comparações, nenhuma técnica mudou o F1 de forma significativa. Com mais características, a MLP chegou a F1 {n(c1['f1_classic'])} na Tarefa B, contra {n(c1['f1_resnet18'])} da ResNet-18, o que vai na mesma direção da H4. "
      "Essas variantes ficam fora do protótipo; o registro completo está no repositório (<i>docs/DECISIONS.md</i> e <i>docs/registro_exploratorio.pdf</i>).")

    P("Referências", "h1")
    for r in ("BEHLAU, M. et al. Epidemiology of voice disorders in teachers and nonteachers in Brazil: prevalence and adverse effects. <b>Journal of Voice</b>, v. 26, n. 5, p. 665.e9-665.e18, 2012.",
              "BRASIL. Ministério da Saúde. <b>Distúrbio de Voz Relacionado ao Trabalho – DVRT</b>. Brasília: Ministério da Saúde, 2018.",
              "GÓMEZ, P. et al. BAGLS, a multihospital Benchmark for Automatic Glottis Segmentation. <b>Scientific Data</b>, v. 7, art. 186, 2020.",
              "MARTINS, R. H. G. et al. Voice disorders in teachers: a review. <b>Journal of Voice</b>, v. 28, n. 6, p. 716-724, 2014.",
              "MEDEIROS, A. M.; VIEIRA, M. T. Ausência ao trabalho por distúrbio vocal de professores da Educação Básica no Brasil. <b>Cadernos de Saúde Pública</b>, v. 35, supl. 1, e00171717, 2019."):
        story.append(Paragraph(r, ParagraphStyle("r", parent=st["p"], alignment=0, fontSize=9.4, leading=12.6)))

    def page(c, d):
        if d.page > 1:
            c.setFont("Arial", 8.5)
            c.setFillColor(colors.HexColor(DIM))
            c.drawRightString(A4[0] - 2.5 * cm, 1.4 * cm, str(d.page))
            c.drawString(2.5 * cm, 1.4 * cm, "LaringoPDI · Etapa 2 · " + AVISO)

    out.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(str(out), pagesize=A4, leftMargin=2.5 * cm, rightMargin=2.5 * cm, topMargin=2.2 * cm, bottomMargin=2.2 * cm, title="LaringoPDI: relatório final da Etapa 2", author="Equipe LaringoPDI").build(story, onFirstPage=page, onLaterPages=page)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--publico", action="store_true", help="omite as figuras com frames do BAGLS")
    a = ap.parse_args()
    build(ROOT / a.out if not Path(a.out).is_absolute() else Path(a.out), a.publico)
    print(a.out)


if __name__ == "__main__":
    main()

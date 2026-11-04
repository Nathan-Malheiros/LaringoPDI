"""Relatório completo do LaringoPDI (PDF, uso interno: contém frames do BAGLS).

Uso: python tools/relatorio_completo.py --out "<Área de Trabalho>/LaringoPDI_relatorio_completo.pdf"

Reaproveita as figuras de tools/relatorio_pdf.py e acrescenta: uma ficha por técnica de PDI
(frame real antes/depois, mapa de diferença, histogramas, teoria, métricas medidas no conjunto e
efeito na classificação), Grad-CAM em frames reais, capturas do protótipo e telemetria do treino.
Todos os números saem de results/ (nada é digitado à mão).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import relatorio_pdf as R  # noqa: E402  (estilo dos gráficos e figuras já existentes)
from laringopdi import config as C  # noqa: E402
from laringopdi.pdi import parse, run_pipeline  # noqa: E402
from laringopdi.pdi.registry import TECHNIQUES  # noqa: E402

T = ROOT / "results" / "tables"
FIG = ROOT / "docs" / "figures" / "relatorio_completo"
LOGO = ROOT / "LOGO_UNIFOR_3_page-0001-removebg-preview.png"
BLUE, INK, GRAY, LIGHT, OR = "#1A47E0", "#101828", "#5B6478", "#EEF2FD", "#E07B2E"
MODELS = [("probe", "ResNet-18 congelada (probe)"), ("simplecnn_fast", "SimpleCNN rápida"),
          ("resnet18", "ResNet-18 fine-tuning"), ("simplecnn", "SimpleCNN completa")]
NAME = {"raw": "Linha de base", "spec": "Remoção de reflexo especular", "he": "Equalização global de histograma",
        "clahe": "CLAHE", "homo": "Filtragem homomórfica", "bilat": "Filtro bilateral", "median": "Filtro da mediana",
        "roi_gt": "Recorte pela máscara do especialista", "roi": "Recorte pela glote segmentada", "clahe>roi": "CLAHE + recorte",
        "edges": "Mapa de bordas", "lbp": "Padrões binários locais (LBP)", "m1": "Fusão M1", "m2": "Fusão M2"}


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    p = FIG / f"{name}.png"
    fig.savefig(p, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return p


# ============================================================================ frames de exemplo
def pick_frames(cfg):
    """Um frame por família, do teste do fold 0, com a maior glote visível (para a figura ler bem)."""
    idx = pd.read_csv(ROOT / "data" / "index.csv", dtype={"video_id": str})
    fr = set(json.loads((ROOT / "data" / "splits" / "frames.json").read_text(encoding="utf-8"))["frames"])
    from laringopdi import data as D
    test0 = {str(v) for v in D.load_splits(cfg)[0]["test"]}
    d = idx[idx.path.isin(fr) & idx.video_id.isin(test0) & ~idx.is_rgb.astype(bool)]
    out = {}
    for fam in ("healthy", "functional", "structural"):
        g = d[d.family == fam].sort_values("seg_area", ascending=False)
        if len(g):
            out[fam] = g.iloc[min(3, len(g) - 1)]
    return out


def load_frame(row):
    base = ROOT / "data" / "raw" / "bagls"
    g = cv2.imread(str(base / row.path), cv2.IMREAD_GRAYSCALE)
    m = cv2.imread(str(base / row.seg_path), cv2.IMREAD_GRAYSCALE) > 127
    return g, m


def view(o):
    return o[..., :3] if o.shape[2] > 1 and not np.array_equal(o[..., 0], o[..., 1]) else o[..., 0]


def fig_tech(cfg, e, g, m):
    """Ficha visual: original | resultado | onde mudou (ou canais / glote) | histogramas."""
    pipe = cfg["experiments"][e]["pipeline"]
    o0, _ = run_pipeline(g, "raw", None, 224, cfg["data"]["work_size"], gt_mask=m)
    o, ctx = run_pipeline(g, pipe, cfg["experiments"][e].get("params"), 224, cfg["data"]["work_size"], gt_mask=m)
    multi = not np.array_equal(o[..., 0], o[..., 1])
    crop = "roi" in pipe
    panels = [(o0[..., 0], "original (E00)", "gray"), (view(o), "resultado" + (" (falsa cor RGB)" if multi else ""), None if multi else "gray")]
    if multi:
        panels += [(o[..., 1], "canal 2", "gray"), (o[..., 2], "canal 3", "gray")]
    elif crop:
        v = cv2.cvtColor(o0[..., 0], cv2.COLOR_GRAY2RGB)
        mm = cv2.resize(m.astype(np.uint8), (224, 224), interpolation=cv2.INTER_NEAREST)
        g0, _ = run_pipeline(np.where(m, 255, 0).astype(np.uint8), "raw", None, 224, cfg["data"]["work_size"])
        cnts, _ = cv2.findContours((g0[..., 0] > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(v, cnts, -1, (46, 158, 106), 2)
        panels.append((v, "glote do especialista (verde)", None))
    elif e != "E00":
        d = np.abs(o[..., 0].astype(float) - o0[..., 0].astype(float))
        d = np.clip(d / max(np.percentile(d, 99), 1) * 255, 0, 255).astype(np.uint8)   # normalizado: mostra onde mudou
        panels.append((d, "onde mudou (|Δ| normalizado)", "inferno"))
    n = len(panels) + 1
    fig, axes = plt.subplots(1, n, figsize=(2.55 * n, 2.75), gridspec_kw={"width_ratios": [1] * (n - 1) + [1.35]})
    for a, (im, title, cmap) in zip(axes, panels):
        a.imshow(im, cmap=cmap, vmin=0, vmax=255) if cmap else a.imshow(im)
        a.set_title(title, fontsize=8.5, color=INK, fontweight="normal"); a.axis("off")
    a = axes[-1]
    for img, col, lab in ((o0[..., 0], "#9AA3B5", "original"), (o[..., 0], BLUE, "resultado")):
        v = img[img > 0]
        a.hist(v, bins=48, range=(0, 255), color=col, alpha=.6, label=lab, density=True)
    a.set_yticks([]); a.set_xticks([0, 128, 255]); a.tick_params(labelsize=7)
    a.set_title("histograma (canal 1)", fontsize=8.5, color=INK, fontweight="normal"); a.legend(frameon=False, fontsize=7)
    fig.tight_layout(w_pad=.6)
    return save(fig, f"ficha_{e}")


def fig_families(cfg, frames):
    """As 14 variantes × 3 famílias (saudável, funcional, estrutural)."""
    exps = [e for e, v in cfg["experiments"].items() if v.get("pipeline")]
    fams = [f for f in ("healthy", "functional", "structural") if f in frames]
    fig, axes = plt.subplots(len(fams), len(exps), figsize=(1.3 * len(exps), 1.45 * len(fams)))
    for i, f in enumerate(fams):
        g, m = load_frame(frames[f])
        for j, e in enumerate(exps):
            o, _ = run_pipeline(g, cfg["experiments"][e]["pipeline"], cfg["experiments"][e].get("params"), 224, cfg["data"]["work_size"], gt_mask=m)
            a = axes[i, j]; v = view(o)
            a.imshow(v, cmap="gray" if v.ndim == 2 else None, vmin=0, vmax=255); a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values():
                s.set_visible(False)
            if i == 0:
                a.set_title(f"{e}\n{cfg['experiments'][e]['pipeline']}", fontsize=6.6, color=INK, fontweight="normal")
            if j == 0:
                a.set_ylabel({"healthy": "saudável", "functional": "funcional", "structural": "estrutural"}[f], fontsize=8)
    fig.tight_layout(pad=.3, w_pad=.15, h_pad=.3)
    return save(fig, "familias")


def fig_gradcam(cfg, frames):
    """Grad-CAM real (ResNet-18, Tarefa A, fold 0) nos mesmos frames, para E00, E01 e E04."""
    from laringopdi.evaluate import gradcam_maps, load_model
    exps = [e for e in ("E00", "E01", "E04", "E14") if load_model(cfg, e, "A", "resnet18", 0) is not None]
    if not exps:
        return None
    fams = [f for f in ("healthy", "functional", "structural") if f in frames]
    fig, axes = plt.subplots(len(fams), len(exps) + 1, figsize=(2.3 * (len(exps) + 1), 2.35 * len(fams)))
    models = {e: load_model(cfg, e, "A", "resnet18", 0) for e in exps}
    for i, f in enumerate(fams):
        g, m = load_frame(frames[f])
        o0, _ = run_pipeline(g, "raw", None, 224, cfg["data"]["work_size"])
        gm, _ = run_pipeline(np.where(m, 255, 0).astype(np.uint8), "raw", None, 224, cfg["data"]["work_size"])
        cnts, _ = cv2.findContours((gm[..., 0] > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        v = cv2.cvtColor(o0[..., 0], cv2.COLOR_GRAY2RGB); cv2.drawContours(v, cnts, -1, (46, 158, 106), 2)
        axes[i, 0].imshow(v); axes[i, 0].set_title("frame + glote" if i == 0 else "", fontsize=8.5, fontweight="normal")
        axes[i, 0].set_ylabel({"healthy": "saudável", "functional": "funcional", "structural": "estrutural"}[f], fontsize=8.5)
        for j, e in enumerate(exps):
            o, _ = run_pipeline(g, cfg["experiments"][e]["pipeline"], None, 224, cfg["data"]["work_size"], gt_mask=m)
            cam = gradcam_maps(models[e], o[None, ..., :3])[0]
            heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1]
            ov = (0.55 * cv2.cvtColor(o[..., 0], cv2.COLOR_GRAY2RGB) + 0.45 * heat).astype(np.uint8)
            cv2.drawContours(ov, cnts, -1, (255, 255, 255), 1)
            axes[i, j + 1].imshow(ov)
            if i == 0:
                axes[i, j + 1].set_title(f"{e} · {cfg['experiments'][e]['pipeline']}", fontsize=8.5, fontweight="normal")
    for a in axes.ravel():
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)
    fig.tight_layout(pad=.3)
    return save(fig, "gradcam_real")


# ============================================================================ texto das técnicas
def doc_parts(step):
    d = re.sub(r"\s+", " ", TECHNIQUES[step].__doc__ or "")
    keys = ["O que faz:", "Por que importa:", "Parâmetros padrão:", "Referência:"]
    out, pos = {}, [(d.find(k), k) for k in keys if d.find(k) >= 0]
    pos.sort()
    for i, (p, k) in enumerate(pos):
        end = pos[i + 1][0] if i + 1 < len(pos) else len(d)
        out[k[:-1]] = d[p + len(k):end].strip()
    return out


def verdict_rows(e):
    rows = []
    for m, label in MODELS:
        p = T / f"verdicts_{m}.csv"
        if not p.exists():
            continue
        v = pd.read_csv(p)
        for t in ("A", "B"):
            r = v[(v.exp == e) & (v.task == t)]
            if len(r):
                r = r.iloc[0]
                rows.append([label, t, f"{r.f1_base:.3f} → {r.f1_exp:.3f}", f"{r.oof_delta_f1:+.3f}  [{r.ci_low:+.3f}; {r.ci_high:+.3f}]",
                             f"{int(r.n_pairs)}", r.verdict])
    return rows


# ============================================================================ PDF
def build(out: Path):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
    from reportlab.lib.fonts import addMapping
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, NextPageTemplate, PageBreak, PageTemplate,
                                    Paragraph, Spacer, Table, TableStyle)
    from reportlab.platypus.tableofcontents import TableOfContents

    F = Path("C:/Windows/Fonts")
    for n, f in (("Arial", "arial.ttf"), ("Arial-Bold", "arialbd.ttf"), ("Arial-Italic", "ariali.ttf"), ("Arial-BoldItalic", "arialbi.ttf"),
                 ("Georgia", "georgia.ttf"), ("Georgia-Bold", "georgiab.ttf"), ("Georgia-Italic", "georgiai.ttf")):
        pdfmetrics.registerFont(TTFont(n, str(F / f)))
    addMapping("Arial", 0, 0, "Arial"); addMapping("Arial", 1, 0, "Arial-Bold"); addMapping("Arial", 0, 1, "Arial-Italic"); addMapping("Arial", 1, 1, "Arial-BoldItalic")
    addMapping("Georgia", 0, 0, "Georgia"); addMapping("Georgia", 1, 0, "Georgia-Bold"); addMapping("Georgia", 0, 1, "Georgia-Italic")
    c = colors.HexColor
    st = {
        "h1": ParagraphStyle("h1", fontName="Georgia", fontSize=21, leading=26, textColor=c(INK), spaceAfter=10, spaceBefore=4, keepWithNext=1),
        "kick": ParagraphStyle("k", fontName="Arial-Bold", fontSize=8.5, leading=11, textColor=c(BLUE), spaceAfter=2),
        "h2": ParagraphStyle("h2", fontName="Georgia-Bold", fontSize=12.5, leading=16, textColor=c(INK), spaceBefore=12, spaceAfter=5),
        "h3": ParagraphStyle("h3", fontName="Arial-Bold", fontSize=9.6, leading=12.5, textColor=c(BLUE), spaceBefore=8, spaceAfter=3),
        "p": ParagraphStyle("p", fontName="Georgia", fontSize=10.2, leading=15.2, alignment=TA_JUSTIFY, textColor=c("#1F2937"), spaceAfter=7),
        "lead": ParagraphStyle("lead", fontName="Georgia-Italic", fontSize=11.5, leading=17, textColor=c(GRAY), spaceAfter=10),
        "cap": ParagraphStyle("cap", fontName="Arial", fontSize=8, leading=10.6, textColor=c(GRAY), spaceAfter=12, alignment=TA_LEFT),
        "cell": ParagraphStyle("cell", fontName="Arial", fontSize=8.1, leading=10.4, textColor=c("#1F2937")),
        "cellh": ParagraphStyle("cellh", fontName="Arial-Bold", fontSize=8.1, leading=10.4, textColor=c(INK)),
        "num": ParagraphStyle("num", fontName="Georgia", fontSize=24, leading=27, textColor=c(BLUE)),
        "numl": ParagraphStyle("numl", fontName="Arial", fontSize=8, leading=10.5, textColor=c(GRAY)),
        "toc1": ParagraphStyle("toc1", fontName="Georgia", fontSize=10.5, leading=17, textColor=c(INK)),
    }
    W = A4[0] - 4.2 * cm
    story = []
    nfig = [0]

    def P(t, s="p"):
        story.append(Paragraph(t, st[s]))

    def H1(kicker, title):
        story.append(PageBreak())
        story.append(Paragraph(kicker.upper(), st["kick"]))
        story.append(Paragraph(title, st["h1"]))

    def fig(path, caption, width=1.0, maxh=None):
        p = Path(path) if path else None
        if p is None or not p.exists():
            return
        h, w = cv2.imread(str(p)).shape[:2]
        iw = W * width; ih = iw * h / w
        if maxh and ih > maxh:
            ih = maxh; iw = ih * w / h
        nfig[0] += 1
        story.append(KeepTogether([Image(str(p), width=iw, height=ih),
                                   Paragraph(f"<b>Figura {nfig[0]}.</b> {caption}", st["cap"])]))

    def table(rows, widths, head=True, font=None):
        data = [[Paragraph(str(x), st["cellh" if head and i == 0 else "cell"]) for x in r] for i, r in enumerate(rows)]
        t = Table(data, colWidths=[W * x for x in widths], repeatRows=1 if head else 0)
        sty = [("LINEBELOW", (0, 0), (-1, 0), .8, c(INK)) if head else ("LINEABOVE", (0, 0), (-1, 0), .4, c("#D0D5DD")),
               ("LINEBELOW", (0, -1), (-1, -1), .4, c("#D0D5DD")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
               ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
               ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
        for i in range(1 if head else 0, len(rows)):
            if (i % 2 == 0):
                sty.append(("BACKGROUND", (0, i), (-1, i), c("#F7F8FA")))
        t.setStyle(TableStyle(sty))
        story.append(t); story.append(Spacer(1, 9))

    def numbers(items):
        cells = [[Paragraph(v, st["num"]) for v, _ in items], [Paragraph(l, st["numl"]) for _, l in items]]
        t = Table(cells, colWidths=[W / len(items)] * len(items))
        t.setStyle(TableStyle([("LINEABOVE", (0, 0), (-1, 0), 1.2, c(BLUE)), ("TOPPADDING", (0, 0), (-1, 0), 7),
                               ("LEFTPADDING", (0, 0), (-1, -1), 0), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        story.append(t); story.append(Spacer(1, 12))

    # ------------------------------------------------------------------ dados
    cfg = C.load()
    idx = json.loads((ROOT / "logs" / "index_summary.json").read_text(encoding="utf-8"))
    fj = json.loads((ROOT / "data" / "splits" / "frames.json").read_text(encoding="utf-8"))["meta"]
    fam = idx["videos_per_family"]
    segs = pd.read_csv(T / "segmentation_summary.csv").set_index(["method", "preprocess"])
    best = json.loads((T / "segmentation_best.json").read_text(encoding="utf-8"))
    h4 = pd.read_csv(T / "seg_h4.csv")
    q = pd.read_csv(ROOT / "results" / "pdi_quality.csv").set_index("exp")
    summ = pd.read_csv(ROOT / "results" / "summary.csv")
    e00 = summ[summ.exp == "E00"].groupby(["model", "task"])["video_f1_macro"].mean()
    hyp = json.loads((T / "hypotheses.json").read_text(encoding="utf-8")) if (T / "hypotheses.json").exists() else {}
    V = {m: pd.read_csv(T / f"verdicts_{m}.csv") for m, _ in MODELS if (T / f"verdicts_{m}.csv").exists()}
    nverd = sum(len(v) for v in V.values()); nneu = sum((v.verdict == "neutra").sum() for v in V.values())
    cvc = pd.read_csv(T / "classic_vs_cnn.csv")
    e14 = cfg["experiments"]["E14"]["pipeline"]
    frames = pick_frames(cfg)
    exps = [e for e, v in cfg["experiments"].items() if v.get("pipeline")]
    print("figuras...", flush=True)
    g_s, m_s = load_frame(frames["structural"])
    fichas = {e: fig_tech(cfg, e, g_s, m_s) for e in exps}
    p_fam = fig_families(cfg, frames)
    p_cam = fig_gradcam(cfg, frames)
    rf = {k: getattr(R, k)() for k in ("fig_pipeline", "fig_enhance", "fig_homo_theory", "fig_segment_theory", "fig_texture_theory",
                                       "fig_neuron", "fig_dataset", "fig_quality", "fig_seg", "fig_stage1", "fig_forest",
                                       "fig_classic", "fig_perm", "fig_curves")}
    rf["fig_stage2"] = R.fig_stage2() if "resnet18" in V else None
    rf["fig_attention"] = R.fig_attention() if (T / "attention.csv").exists() else None
    DF = ROOT / "docs" / "figures"
    print("pdf...", flush=True)

    # ------------------------------------------------------------------ capa
    story.append(Image(str(LOGO), width=3.6 * cm, height=3.6 * cm * 341 / 394, hAlign="LEFT"))
    story.append(Spacer(1, 2.6 * cm))
    story.append(Paragraph("PROCESSAMENTO DIGITAL DE IMAGENS · PROJETO EXTENSIONISTA", st["kick"]))
    story.append(Paragraph("LaringoPDI", ParagraphStyle("t", fontName="Georgia", fontSize=40, leading=46, textColor=c(INK), spaceAfter=10)))
    story.append(Paragraph("O que o processamento de imagens faz com um frame de videoendoscopia laríngea — "
                           "e o que isso muda para a segmentação da glote e para o reconhecimento de distúrbios da voz",
                           ParagraphStyle("s", fontName="Georgia-Italic", fontSize=14, leading=20, textColor=c(GRAY))))
    story.append(Spacer(1, 2.2 * cm))
    fig(rf["fig_pipeline"], "Desenho do estudo: o PDI é a variável independente; classificador, protocolo e divisões dos dados ficam fixos.", .92)
    story.append(Spacer(1, 1.2 * cm))
    tc = Table([[Paragraph("<b>Disciplina</b>", st["cell"]), Paragraph("Processamento Digital de Imagens — Universidade de Fortaleza (UNIFOR). Prof.ª Lyndainês Santos.", st["cell"])],
                [Paragraph("<b>Equipe</b>", st["cell"]), Paragraph("Nathan (2220227) · Marcos Vinicius · Pedro Nobre · Marcos Mendes · Tales", st["cell"])],
                [Paragraph("<b>Data</b>", st["cell"]), Paragraph(f"{date.today().strftime('%d/%m/%Y')} · código {C.git_sha()}", st["cell"])],
                [Paragraph("<b>Uso</b>", st["cell"]), Paragraph("Documento interno da equipe: contém frames do BAGLS (CC BY-NC-SA 4.0). "
                                                                "Ferramenta de apoio acadêmico; não substitui avaliação médica.", st["cell"])]],
               colWidths=[W * .16, W * .84])
    tc.setStyle(TableStyle([("LINEABOVE", (0, 0), (-1, 0), 1.2, c(BLUE)), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 0), ("TOPPADDING", (0, 0), (-1, -1), 4)]))
    story.append(tc)
    story.append(NextPageTemplate("body"))

    # ------------------------------------------------------------------ sumário
    story.append(PageBreak())
    story.append(Paragraph("SUMÁRIO", st["kick"]))
    story.append(Paragraph("Conteúdo", ParagraphStyle("h1toc", parent=st["h1"])))
    toc = TableOfContents(); toc.levelStyles = [st["toc1"]]; toc.dotsMinLevel = 0
    story.append(toc)

    # ------------------------------------------------------------------ 1. em uma página
    H1("1", "O estudo em uma página")
    P("Laringoscopia de alta velocidade produz milhares de frames por segundo das pregas vocais. O projeto pergunta "
      "algo simples de enunciar e trabalhoso de responder: <i>quando aplicamos uma técnica clássica de Processamento Digital "
      "de Imagens antes da análise, o resultado melhora, não muda ou piora?</i> A resposta foi medida em dois usos do frame — "
      "a segmentação da glote por métodos clássicos e a classificação de distúrbios da voz por redes neurais — mantendo "
      "fixo todo o resto (rede, hiperparâmetros, divisão dos dados, sementes). O PDI é a única coisa que muda.", "lead")
    numbers([(f"{fj['n_videos']}", "vídeos rotulados (BAGLS)"), (f"{fj['n_frames']:,}".replace(",", "."), "frames analisados"),
             ("14", "variantes de PDI"), (f"{len(summ)}", "execuções de treino e avaliação")])
    P(f"<b>Na segmentação clássica, o PDI decide.</b> O melhor método (crescimento de região) sobe de Dice "
      f"{segs.loc[(best['method'], 'raw'), 'dice_mean']:.2f} no frame cru para {segs.loc[(best['method'], best['preprocess']), 'dice_mean']:.2f} "
      f"depois do CLAHE. O mesmo realce, porém, piora os métodos de limiar global: {int((h4.verdict == 'melhora').sum())} das "
      f"{len(h4)} combinações método × realce melhoram e {int((h4.verdict == 'piora').sum())} pioram. Realce não é bom ou ruim em si; "
      "depende de quem vai ler a imagem depois.")
    P(f"<b>Na classificação por redes neurais, o PDI não mudou o resultado.</b> Dos {nverd} confrontos técnica × modelo × tarefa, "
      f"{nneu} terminaram como <i>neutros</i>: nenhuma técnica melhorou ou piorou o F1 por vídeo de forma estatisticamente "
      "detectável, nem com a rede congelada, nem com treino completo, nem com a rede pequena treinada do zero. As redes já "
      "extraem do frame cru o que o realce entregaria. É um resultado negativo, medido com 15 repetições por comparação, e "
      "por isso informativo.")
    P(f"<b>Características clássicas competem com as redes.</b> Forma da glote, textura, HOG e bordas, com MLP, SVM ou Random Forest, "
      f"chegaram a F1 por vídeo de {cvc[cvc.task == 'A'].f1.max():.2f} na Tarefa A e {cvc[cvc.task == 'B'].f1.max():.2f} na B, no mesmo "
      "patamar das CNNs — com a vantagem de serem interpretáveis.")
    if hyp:
        table([["Hipótese pré-registrada", "Conclusão"]] + [[k, escape(v)] for k, v in hyp.items()], [.3, .7])

    # ------------------------------------------------------------------ 2. contexto
    H1("2", "Contexto: por que olhar para a glote")
    P("Professores estão entre os profissionais com maior prevalência de distúrbios da voz: usam a voz por horas, em salas "
      "ruidosas, muitas vezes sem preparo vocal. A avaliação passa pela laringoscopia, em que um especialista observa a "
      "vibração das pregas vocais e a abertura entre elas — a <i>glote</i>. Lesões estruturais (nódulos, pólipos, cistos, "
      "edema) costumam aparecer na imagem; distúrbios funcionais, como a disfonia por tensão muscular, estão sobretudo no "
      "<i>movimento</i>, e um frame isolado mostra pouco deles.")
    P("O projeto tem três dimensões. A <b>social</b>: profissionais da voz como beneficiários e profissionais de saúde em "
      "triagem como usuários. A <b>tecnológica</b>: um protótipo que mostra, passo a passo, o que cada técnica faz no frame. "
      "A <b>científica</b>: uma comparação controlada, com hipóteses escritas antes dos resultados e estatística corrigida "
      "para múltiplas comparações. Em todos os casos, a ferramenta é de apoio à decisão e nunca de diagnóstico.")
    P("O dataset planejado, o Laryngoscope8, estava inacessível (link único em serviço chinês, sem resposta dos autores). Ele "
      "foi substituído pelo <b>BAGLS</b> (Gómez et al., 2020): 59.250 frames de 640 vídeos de alta velocidade, de vários "
      "hospitais, cada frame com a máscara da glote desenhada por especialistas e metadados com o estado da voz. A troca "
      "trouxe algo que o dataset original não tinha: um gabarito de segmentação para medir o PDI diretamente.")
    fig(DF / "eda_samples.png", "Frames do BAGLS por família diagnóstica, em escala de cinza (o estudo inteiro é feito em cinza).", .95, maxh=9 * cm)

    # ------------------------------------------------------------------ 3. dados
    H1("3", "Dados e cuidados contra vazamento")
    P(f"O campo de estado da voz é texto livre, com {idx['n_distinct_status']} valores distintos (de \"healthy\" a \"Muscle tension "
      "dysphonia with...\"). Uma tabela de regras (<i>configs/label_map.yaml</i>) traduz cada texto em uma família: saudável, "
      f"funcional ou estrutural. Ficaram {fam['healthy']} vídeos saudáveis, {fam['functional']} funcionais e {fam['structural']} "
      f"estruturais; {fam['unknown']} vídeos sem informação suficiente foram excluídos. Daí saem as tarefas: <b>A</b>, saudável × "
      "qualquer distúrbio (principal); <b>B</b>, saudável × lesão estrutural.")
    fig(rf["fig_dataset"], "Vídeos por família e por tarefa.", .9)
    P("O cuidado mais importante é o <b>vazamento</b>: frames do mesmo vídeo são quase idênticos, e se um vai para o treino e "
      "outro para o teste, a rede \"decora\" o paciente e o resultado fica otimista. Por isso a divisão é feita por vídeo "
      "(StratifiedGroupKFold, 5 folds), vídeos quase duplicados são detectados por pHash + SSIM, e de cada vídeo saem até 10 "
      "frames uniformemente espaçados, sempre os mesmos (<i>frames.json</i>). Toda estatística ajustada aos dados — limiares, "
      "parâmetros de segmentação, escolha de técnicas — usa só o treino de cada fold. O teste nunca é usado para decidir.")
    fig(DF / "eda_sizes.png", "Resoluções dos vídeos do BAGLS: a variedade de câmeras explica por que todo frame é levado a um lado máximo de 512 px antes do PDI.", .8, maxh=7 * cm)

    # ------------------------------------------------------------------ 4. fundamentos
    H1("4", "Fundamentos de PDI usados no projeto")
    P("Uma imagem em tons de cinza é uma matriz f(x, y) de intensidades de 0 a 255. Todas as técnicas do estudo recebem essa "
      "matriz e devolvem outra do mesmo tipo — ou, nas representações, três matrizes empilhadas que a rede lê como canais.")
    story.append(Paragraph("Histograma e realce de contraste", st["h2"]))
    P("O histograma conta quantos pixels há em cada nível de cinza. A <b>equalização global</b> (HE) usa o histograma acumulado "
      "como função de transferência e espalha os níveis por toda a faixa; é simples, mas amplifica ruído e estoura regiões já "
      "claras. O <b>CLAHE</b> faz a equalização por blocos (8×8) e limita a inclinação da função (clip 2,0), o que realça "
      "detalhes locais sem estourar o frame inteiro.")
    fig(rf["fig_enhance"], "Frame sintético com vinheta: a HE espalha o histograma inteiro, o CLAHE realça localmente e a homomórfica achata a iluminação.")
    story.append(Paragraph("Domínio da frequência", st["h2"]))
    P("No modelo iluminação–refletância, f = i · r. A iluminação da fonte pontual do endoscópio varia devagar (baixa frequência); "
      "os detalhes da mucosa variam rápido (alta frequência). Aplicando o logaritmo, o produto vira soma, e um filtro no "
      "espectro de Fourier pode tratar as duas partes separadamente: H(u,v) = (γH − γL)(1 − e<super>−D²/D0²</super>) + γL, com "
      "γL = 0,5 atenuando a iluminação e γH = 1,75 realçando a refletância.")
    fig(rf["fig_homo_theory"], "O filtro homomórfico no plano de frequências e seu perfil radial (D0 = 30 px).", .8)
    story.append(Paragraph("Filtragem espacial, segmentação, bordas e textura", st["h2"]))
    P("O <b>filtro bilateral</b> pondera vizinhos pela distância <i>e</i> pela semelhança de intensidade, suavizando sem borrar "
      "bordas; a <b>mediana</b> substitui cada pixel pela mediana da janela e elimina ruído impulsivo. Na segmentação, a glote "
      "aberta é a região mais escura: Otsu escolhe o limiar que maximiza a variância entre classes; o limiar adaptativo "
      "decide pixel a pixel; a morfologia limpa o resultado; o crescimento de região parte de uma semente escura e agrega "
      "vizinhos parecidos; o watershed trata a imagem como relevo. Sobel e Canny destacam bordas; o LBP codifica a textura "
      "local comparando cada pixel com seus 8 vizinhos.")
    fig(rf["fig_segment_theory"], "Os cinco métodos clássicos de segmentação num frame sintético.")
    fig(rf["fig_texture_theory"], "Representações de bordas e textura usadas como entrada da rede e no braço clássico.")

    # ------------------------------------------------------------------ 5. catálogo
    H1("5", "Catálogo das técnicas: o que cada uma faz no frame")
    P("Cada ficha a seguir mostra a mesma imagem — um frame de teste de um vídeo com lesão estrutural — antes e depois da "
      "técnica, onde a técnica mexeu (diferença absoluta, em escala <i>inferno</i>: quanto mais claro, maior a mudança) e o "
      "histograma antes e depois. Em seguida vêm a descrição técnica (tirada da própria documentação do código), as métricas "
      f"medidas nos {fj['n_frames']:,} frames do estudo e o efeito na classificação em cada modelo.".replace(",", "."), "lead")
    P("Leitura das métricas: <b>entropia</b> mede quanta informação de intensidade a imagem carrega (mais níveis usados = maior); "
      "<b>contraste RMS</b> é o desvio-padrão das intensidades; <b>nitidez</b> é a variância do Laplaciano (bordas fortes = maior); "
      "<b>fração especular</b> é a parte do frame ocupada por reflexos saturados da luz. Na tabela de classificação, o ΔF1 é a "
      "diferença de F1-macro por vídeo contra o E00, com intervalo de confiança de 95%; o veredito só é \"melhora\" ou "
      "\"piora\" se o intervalo excluir zero e o teste corrigido por Holm for significativo.")
    qcols = [("entropy", "Entropia", "{:.2f}"), ("rms_contrast", "Contraste RMS", "{:.3f}"), ("lap_var", "Nitidez", "{:.0f}"), ("spec_frac", "Fração especular", "{:.2%}")]
    for e in exps:
        pipe = cfg["experiments"][e]["pipeline"]
        steps = parse(pipe)
        block = [Paragraph(f"{e} · <font name='Arial'>{escape(pipe)}</font>", st["kick"]),
                 Paragraph(NAME.get(pipe, cfg["experiments"][e]["desc"]), st["h2"])]
        story.append(PageBreak()); [story.append(b) for b in block]
        fig(fichas[e], f"{e} ({pipe}) no mesmo frame de teste. Painéis: original, resultado, " +
            ("canais da representação" if "edges" in pipe or "lbp" in pipe or pipe in ("m1", "m2") else
             "glote do especialista" if "roi" in pipe else "diferença absoluta" if e != "E00" else "") + " e histogramas.")
        rows = []
        for s in steps:
            d = doc_parts(s)
            for k, label in (("O que faz", "O que faz"), ("Por que importa", "Por que importa na laringoscopia"),
                             ("Parâmetros padrão", "Parâmetros (literatura)"), ("Referência", "Referência")):
                if d.get(k):
                    rows.append([label + (f" — {s}" if len(steps) > 1 else ""), escape(d[k])])
        if rows:
            table(rows, [.24, .76], head=False)
        if e in q.index:
            qr = [["Métrica (média no conjunto)", "E00", e, "Δ"]]
            for k, lab, f in qcols:
                a, b = q.loc["E00", k], q.loc[e, k]
                if pd.notna(a) and pd.notna(b):
                    qr.append([lab, f.format(a), f.format(b), (f"{b - a:+.2%}" if "%" in f else ("+" if b >= a else "") + f.format(b - a))])
            if e != "E00":
                table(qr, [.4, .2, .2, .2])
        vr = verdict_rows(e)
        if vr:
            story.append(Paragraph("Efeito na classificação (F1-macro por vídeo)", st["h3"]))
            table([["Modelo", "Tarefa", "F1 E00 → técnica", "ΔF1 [IC 95%]", "Pares", "Veredito"]] + vr, [.27, .08, .19, .26, .08, .12])
        elif e == "E00":
            P(f"O E00 é a linha de base de todas as comparações: só converte para cinza, reduz o lado maior a 224 px preservando a "
              f"proporção e completa com bordas pretas. F1 por vídeo no E00: ResNet-18 fine-tuning {e00.get(('resnet18', 'A'), float('nan')):.3f} (A) "
              f"e {e00.get(('resnet18', 'B'), float('nan')):.3f} (B); SimpleCNN {e00.get(('simplecnn', 'A'), float('nan')):.3f} (A) e "
              f"{e00.get(('simplecnn', 'B'), float('nan')):.3f} (B).")

    # ------------------------------------------------------------------ 6. comparativo
    H1("6", "Comparativo entre técnicas")
    P("Vistas lado a lado, as técnicas se separam em quatro grupos. As de <b>realce</b> (HE, CLAHE, homomórfica) mudam a "
      "distribuição de intensidades; as de <b>suavização</b> (bilateral, mediana) e a <b>remoção de reflexos</b> mexem pouco e "
      "localmente; os <b>recortes</b> mudam a geometria e aproximam a glote; as <b>representações</b> (bordas, LBP, fusões) "
      "trocam a intensidade por outra grandeza, e a imagem deixa de parecer uma fotografia.")
    fig(p_fam, "As 14 variantes aplicadas a um frame de cada família (saudável, funcional, estrutural).", 1.0)
    fig(rf["fig_quality"], "Métricas de qualidade da imagem em todo o conjunto, por técnica (linha pontilhada: E00).", 1.0)
    rows = [["Técnica", "Entropia", "Contraste", "Nitidez", "Reflexos"] + [f"ΔF1 {lab.split(' ')[0]} {t}" for m, lab in MODELS[2:] for t in "AB" if m in V]]
    for e in exps:
        r = [f"{e} {cfg['experiments'][e]['pipeline']}"] + [f.format(q.loc[e, k]) if e in q.index and pd.notna(q.loc[e, k]) else "—" for k, _, f in qcols]
        for m, _ in MODELS[2:]:
            if m in V:
                for t in "AB":
                    x = V[m][(V[m].exp == e) & (V[m].task == t)]
                    r.append("base" if e == "E00" else (f"{x.oof_delta_f1.iloc[0]:+.3f}" if len(x) else "—"))
        rows.append(r)
    nc = len(rows[0])
    table(rows, [.2] + [(.8 / (nc - 1))] * (nc - 1))
    P("A tabela resume o padrão do estudo: as técnicas mudam muito a imagem — a homomórfica multiplica a nitidez por quase dez, "
      "a HE estoura o contraste, a remoção de reflexos corta a fração especular quase pela metade —, mas nenhuma dessas mudanças "
      "se traduz em diferença significativa de F1. A imagem muda; a decisão da rede, não.")

    # ------------------------------------------------------------------ 7. frequência
    H1("7", "Um olhar no domínio da frequência")
    P("A filtragem homomórfica merece uma seção própria porque é a técnica mais \"de PDI\" do estudo: ela não mexe nos pixels "
      "diretamente, e sim no espectro de Fourier do logaritmo da imagem. A figura mostra o caminho completo num frame real: o "
      "espectro concentra energia no centro (iluminação lenta), o filtro atenua esse centro e reforça o anel externo, e o "
      "resultado devolve uma imagem de iluminação uniforme e textura acentuada.")
    fig(DF / "gallery_homo.png", "Filtragem homomórfica num frame real: imagem, espectro, H(u,v) e resultado.", 1.0)
    fig(ROOT / "docs" / "figures" / "app" / "aba3.png", "A mesma análise no protótipo (aba \"Domínio da frequência\"), com o perfil radial de H e o espectro médio por raio antes e depois.", 1.0, maxh=11 * cm)
    P("Por que isso não ajudou a rede? Uma hipótese é que a primeira camada de uma CNN já aprende filtros passa-altas; a "
      "filtragem homomórfica entrega um resultado que a rede consegue construir sozinha. A ResNet-18 com homomórfica foi a "
      f"combinação com maior ganho bruto na Tarefa A, mas o intervalo de confiança cruzou o zero.")

    # ------------------------------------------------------------------ 8. segmentação
    H1("8", "Segmentação clássica da glote")
    P(f"Cinco métodos × quatro pré-processamentos (cru, CLAHE, homomórfica, bilateral) × uma grade de parâmetros rodaram em "
      f"todos os {fj['n_frames']:,} frames. Para cada fold, os parâmetros foram escolhidos no treino e avaliados no teste, contra a "
      "máscara dos especialistas. Métricas: Dice e IoU (sobreposição), HD95 (distância entre contornos, em pixels) e taxa de "
      "falha (máscara vazia quando havia glote).".replace(",", "."))
    fig(rf["fig_seg"], "Dice médio por método × pré-processamento (esq.) e Δ Dice por vídeo contra o frame cru, Wilcoxon com Holm (dir.).", 1.0)
    rows = [["Método", "Pré", "Dice", "IoU", "HD95 (px)", "Falha"]]
    for (mt, pr), r in segs.sort_values("dice_mean", ascending=False).head(8).iterrows():
        rows.append([mt, pr, f"{r.dice_mean:.3f}", f"{r.get('iou_mean', float('nan')):.3f}", f"{r.get('hd95_mean', float('nan')):.1f}",
                     f"{r.get('fail_mean', float('nan')):.1%}"])
    table(rows, [.22, .14, .14, .14, .18, .18])
    fig(DF / "segmentation_examples.png", "Exemplos de teste: original, máscara do especialista e os três melhores métodos.", .95, maxh=12 * cm)
    P("O vencedor — crescimento de região sobre CLAHE — virou o segmentador oficial do projeto: é ele que gera o recorte "
      "<i>roi</i> (E08, E09) e a máscara glótica da fusão M2. O resultado mais interessante, porém, é o da tabela de H4: o "
      "CLAHE ajuda o crescimento de região (+0,15 de Dice) e atrapalha o Otsu com morfologia e o watershed (≈ −0,15), porque "
      "realça o contraste também fora da glote e cria falsas regiões escuras. A homomórfica faz o oposto: ajuda os métodos "
      "de limiar e atrapalha o crescimento de região.")
    rows = [["Método", "Pré", "Dice cru", "Dice com pré", "Δ", "Veredito"]] + \
           [[r.method, r.preprocess, f"{r.dice_raw:.3f}", f"{r.dice_pre:.3f}", f"{r.delta:+.3f}", r.verdict] for r in h4.itertuples()]
    table(rows, [.2, .12, .15, .17, .14, .22])
    fig(DF / "gallery_roi.png", "Recorte da ROI: caixa pela máscara do especialista (verde) e pela segmentação clássica (vermelho).", .85, maxh=9 * cm)

    # ------------------------------------------------------------------ 9. classificação
    H1("9", "Classificação: protocolo e resultados")
    P("O classificador é o instrumento de medida fixo. Foram usados dois: a <b>ResNet-18</b> pré-treinada no ImageNet e uma "
      "<b>SimpleCNN</b> de quatro blocos treinada do zero. Entrada 224×224, cinza replicado em três canais, augmentation só "
      "geométrico (espelhamento, ±15°, escala 0,9–1,1) — qualquer augmentation de intensidade competiria com o próprio PDI e "
      "foi proibido. Entropia cruzada ponderada, AdamW, cosine, até 30 épocas com early stopping pelo F1 por vídeo da "
      "validação interna. A avaliação principal é por vídeo, com a média das probabilidades dos frames.")
    story.append(Paragraph("Estágio 1 — triagem", st["h2"]))
    P("As 14 variantes passaram por uma triagem barata: ResNet congelada com regressão logística (probe) e SimpleCNN com 10 "
      "épocas. O ranking usou só a validação interna; o teste foi apenas registrado.")
    fig(rf["fig_stage1"], "F1 por vídeo de cada variante no Estágio 1: validação (usada na escolha) e teste (registro).", 1.0)
    fig(rf["fig_forest"], "ΔF1 contra o E00 no Estágio 1, com IC 95%. Todos os intervalos cruzam zero.", 1.0)
    story.append(Paragraph("Estágio 2 — treino completo", st["h2"]))
    P(f"As quatro melhores pela validação, o E00 e a combinação gulosa E14 (<i>{escape(str(e14))}</i>) foram treinadas por completo, "
      "com 3 sementes × 5 folds = 15 repetições por comparação. E01, E07 e E08 rodaram com uma semente para testar as hipóteses "
      "sobre reflexos e recorte.")
    fig(rf["fig_stage2"], "ΔF1 contra o E00 no Estágio 2 (IC 95% sobre o ensemble das sementes).", 1.0)
    fig(rf["fig_curves"], "Curvas de aprendizado da SimpleCNN rápida: F1 de validação e perda por época.", .95)
    story.append(Paragraph("Braço clássico", st["h2"]))
    P("Como ponte com a disciplina de IA Computacional, um braço sem redes neurais extrai características interpretáveis de "
      "cada frame — forma da glote (área, perímetro, excentricidade, solidez, simetria), textura (GLCM, LBP, Gabor), HOG e "
      "bordas —, agrega por vídeo (média e desvio) e classifica com MLP, SVM ou Random Forest nos mesmos folds.")
    fig(rf["fig_classic"], "F1 por vídeo do braço clássico (roxo) e das CNNs no E00 (ciano).", 1.0)
    fig(rf["fig_perm"], "Importância de cada família de características, medida pela queda de desempenho ao embaralhá-la.", 1.0)

    # ------------------------------------------------------------------ 10. interpretabilidade
    H1("10", "Para onde a rede olha")
    P("O Grad-CAM pondera os mapas de ativação da última camada convolucional pelo gradiente da classe prevista e mostra quais "
      "regiões pesaram na decisão. Para cada frame de teste foi medida a fração da energia do mapa dentro da glote (máscara "
      "do especialista) e sobre os reflexos especulares. Um mapa uniforme teria fração igual à área da região — esse é o nível "
      "do acaso.")
    fig(p_cam, "Grad-CAM real da ResNet-18 (Tarefa A, fold 0) nos mesmos três frames, para o E00 e para técnicas do Estágio 2. Contorno: glote do especialista.", 1.0)
    fig(rf["fig_attention"], "Fração da energia do Grad-CAM na glote e nos reflexos, por técnica (traço preto: nível do acaso).", 1.0)
    P("A rede dedica à glote uma fração de atenção pouco acima da área que ela ocupa, e aos reflexos uma fração próxima do "
      "acaso já no E00. Isso explica por que remover os reflexos (E01) mexe pouco na decisão: a rede já não se apoiava "
      "neles. A atenção também se espalha pelas pregas vocais e pela mucosa ao redor, o que é coerente com lesões estruturais "
      "que ficam na borda das pregas, e não dentro da glote.")

    # ------------------------------------------------------------------ 11. protótipo
    H1("11", "O protótipo")
    P("O aplicativo (Streamlit, roda em CPU no Windows) é a dimensão tecnológica do projeto. Ele não esconde o PDI atrás de "
      "uma predição: cada aba mostra uma etapa do que o estudo fez, sobre qualquer frame enviado.")
    caps = ["Técnicas de aula: realce com histogramas, ruído com PSNR e SSIM ao vivo e passa-baixa gaussiano com σ ajustável.",
            "Pipeline passo a passo: cada etapa com seu histograma e histograma acumulado, e as métricas comparadas com o E00.",
            "Laboratório: as 14 técnicas no mesmo frame, com tabela de métricas.",
            "Domínio da frequência: espectro, H(u,v), resultado e perfis radiais.",
            "Segmentação: pré-processamento, resultado clássico e contorno do especialista, com Dice, IoU e HD95.",
            "Rede neural: probabilidades, Grad-CAM, o veredito daquela técnica no estudo e as 18 características com a MLP-18.",
            "Resultados: hipóteses P1–P6, Dice por método × realce, vereditos e clássico × CNN, lidos das tabelas."]
    for i, cp in enumerate(caps):
        fig(ROOT / "docs" / "figures" / "app" / f"aba{i}.png", cp, .9, maxh=11.5 * cm)

    # ------------------------------------------------------------------ 12. telemetria
    H1("12", "Bastidores do treino")
    P("Durante o treino, um painel próprio (LaringoMonitor) mostrava a rede aprendendo em tempo real: o lote atual com o "
      "palpite da rede, o Grad-CAM ao fim de cada época, os filtros da primeira camada e os mapas de ativação. Foi esse painel "
      "que revelou o erro mais sério do projeto: sob precisão mista na GPU, a augmentation saturava cerca de 22% dos pixels. "
      "O bug foi corrigido, 22 execuções foram descartadas e o Estágio 1 refeito do zero.")
    LV = ROOT / "results" / "live"
    fig(LV / "batch.jpg", "Lote de treino visto pelo painel após a correção: augmentation geométrica, rótulo real e palpite da rede.", .8, maxh=10 * cm)
    fig(LV / "filters.jpg", "Filtros 3×3 aprendidos na primeira camada da SimpleCNN: detectores de borda e contraste — PDI aprendido pela rede.", .5)
    fig(LV / "fmaps.jpg", "Mapas de ativação das camadas intermediárias para um frame.", .8, maxh=9 * cm)
    table([["Problema", "Impacto", "Correção"],
           ["Dataset original inacessível", "Bloqueava o projeto", "Troca para o BAGLS, que trouxe máscaras de especialista"],
           ["Watershed semeando uma única região", "Dice ≈ 0,005", "Núcleos por componente; corrigido antes da escolha do vencedor"],
           ["Augmentation sob AMP", "~22% dos pixels saturados", "Augmentation em float32; teste de regressão; Estágio 1 refeito"],
           ["Cache inteiro na RAM", "Processo encerrado", "Escrita em blocos (memmap)"],
           ["Gabor do scikit-image", "~13 h de extração", "Kernel do OpenCV: 240× mais rápido"]], [.3, .25, .45])

    # ------------------------------------------------------------------ 13–18. fechamento da Etapa 1
    from secoes_etapa1 import secoes
    secoes(lambda n, t: H1(str(n), t), P, table, lambda p, cap, w=1.0, priv=False: fig(p, cap, w, maxh=20 * cm), ROOT, 13)

    # ------------------------------------------------------------------ 19. discussão
    H1("19", "Discussão, limitações e o que mais o PDI pode oferecer")
    P("O achado central não é \"o PDI não serve\". É mais preciso: <b>o PDI importa quando quem lê a imagem é um algoritmo de "
      "regras fixas</b> — um limiar, um crescimento de região —, e importa pouco quando quem lê é uma rede que aprende seus "
      "próprios filtros com dados suficientes. Na segmentação clássica, escolher o realce certo vale tanto quanto escolher o "
      "método. Na rede neural, as mesmas técnicas se tornam redundantes.")
    story.append(Paragraph("Limitações", st["h2"]))
    P("Os frames são tratados como imagens estáticas, e a dinâmica vibratória — essencial nos distúrbios funcionais — fica de "
      "fora; isso explica o teto modesto da Tarefa A. A classe estrutural é pequena (54 vídeos, cerca de 11 por fold de "
      "teste), o que alarga os intervalos de confiança da Tarefa B. Os rótulos vêm de texto livre por regras e aguardam "
      "revisão humana. Com efeitos de ±0,03 no F1 e cerca de 118 vídeos de teste por fold, o estudo tem poder para detectar "
      "apenas efeitos grandes: \"neutro\" significa \"sem diferença detectável neste desenho\", não \"efeito nulo provado\".")
    story.append(Paragraph("Para onde o PDI pode ir", st["h2"]))
    P("Ficaram fora do escopo, por decisão de projeto, caminhos em que o PDI tende a ter mais a dizer: a <b>análise temporal</b> "
      "(quimografia, que transforma uma linha do frame ao longo do tempo numa imagem da vibração); a medição de <b>parâmetros "
      "glóticos dinâmicos</b> a partir da segmentação frame a frame (área glótica ao longo do ciclo, quociente de abertura, "
      "simetria entre as pregas); a <b>cor</b> e a imagem de banda estreita (NBI) para lesões vasculares; e o uso do PDI como "
      "<b>controle de qualidade</b> do exame — detectar frames borrados, superexpostos ou sem glote visível antes de qualquer "
      "análise. Em todos esses casos o PDI deixa de ser um pré-processamento opcional e passa a ser a própria medida.")

    # ------------------------------------------------------------------ 14. reprodução
    H1("20", "Reprodutibilidade e referências")
    P("Todo o projeto roda com um único ponto de entrada (<i>python -m laringopdi &lt;comando&gt;</i>), a matriz de experimentos é "
      "um arquivo YAML, cada execução salva configuração, semente, versão do código e hash do pipeline, e execuções concluídas "
      "são puladas. Todos os números e figuras deste documento são gerados a partir de <i>results/</i>.")
    table([["Etapa", "Comando"], ["Índice, divisões, EDA", "index · splits · eda"], ["Segmentação clássica", "seg-study"],
           ["PDI em cache e galeria", "cache · gallery"], ["Triagem", "probe · train --stage stage1"],
           ["Treino completo", "tools/stage2_chain.py"], ["Atenção, estatística, relatório", "tools/final_chain.py"],
           ["Protótipo", "streamlit run app/streamlit_app.py"]], [.4, .6])
    refs = ["Gómez P. et al. BAGLS, a multihospital Benchmark for Automatic Glottis Segmentation. Scientific Data 7, 186, 2020.",
            "Gonzalez R. C.; Woods R. E. Digital Image Processing. 4. ed. Pearson, 2018.",
            "Otsu N. A threshold selection method from gray-level histograms. IEEE TSMC 9(1), 1979.",
            "Zuiderveld K. Contrast Limited Adaptive Histogram Equalization. Graphics Gems IV, 1994.",
            "Tomasi C.; Manduchi R. Bilateral filtering for gray and color images. ICCV, 1998.",
            "Telea A. An image inpainting technique based on the fast marching method. Journal of Graphics Tools, 2004.",
            "Ojala T.; Pietikäinen M.; Mäenpää T. Multiresolution gray-scale and rotation invariant texture classification with LBP. IEEE TPAMI, 2002.",
            "Vincent L.; Soille P. Watersheds in digital spaces. IEEE TPAMI, 1991.",
            "He K. et al. Deep Residual Learning for Image Recognition. CVPR, 2016.",
            "Selvaraju R. R. et al. Grad-CAM: Visual Explanations from Deep Networks. ICCV, 2017.",
            "Loshchilov I.; Hutter F. Decoupled Weight Decay Regularization. ICLR, 2019."]
    story.append(Paragraph("Referências", st["h2"]))
    for r in refs:
        story.append(Paragraph(escape(r), ParagraphStyle("ref", parent=st["p"], fontSize=9, leading=12.5, leftIndent=12, firstLineIndent=-12, alignment=TA_LEFT)))

    # ------------------------------------------------------------------ montagem
    class Doc(BaseDocTemplate):
        def afterFlowable(self, f):
            if isinstance(f, Paragraph) and f.style.name == "h1":
                self.notify("TOCEntry", (0, f.getPlainText(), self.page))

    def cover(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(c(BLUE)); canvas.rect(0, 0, .55 * cm, A4[1], stroke=0, fill=1)
        canvas.restoreState()

    def body(canvas, doc):
        canvas.saveState()
        canvas.drawImage(str(LOGO), A4[0] - 3.3 * cm, A4[1] - 1.75 * cm, width=1.15 * cm, height=1.15 * cm * 341 / 394, mask="auto")
        canvas.setStrokeColor(c("#D0D5DD")); canvas.setLineWidth(.5)
        canvas.line(2.1 * cm, A4[1] - 1.85 * cm, A4[0] - 2.1 * cm, A4[1] - 1.85 * cm)
        canvas.setFont("Arial", 7.5); canvas.setFillColor(c(GRAY))
        canvas.drawString(2.1 * cm, A4[1] - 1.6 * cm, "LaringoPDI · Processamento Digital de Imagens · UNIFOR")
        canvas.drawString(2.1 * cm, 1.2 * cm, "Documento interno da equipe · contém frames do BAGLS (CC BY-NC-SA 4.0) · não substitui avaliação médica")
        canvas.setFillColor(c(BLUE)); canvas.setFont("Arial-Bold", 8.5)
        canvas.drawRightString(A4[0] - 2.1 * cm, 1.2 * cm, str(doc.page))
        canvas.restoreState()

    out.parent.mkdir(parents=True, exist_ok=True)
    doc = Doc(str(out), pagesize=A4, title="LaringoPDI — relatório completo", author="Equipe LaringoPDI (UNIFOR)")
    fr = Frame(2.1 * cm, 1.9 * cm, W, A4[1] - 4.2 * cm, id="f")
    doc.addPageTemplates([PageTemplate("cover", [Frame(2.1 * cm, 1.9 * cm, W, A4[1] - 3.6 * cm)], onPage=cover),
                          PageTemplate("body", [fr], onPage=body)])
    doc.multiBuild(story)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    build(Path(ap.parse_args().out))
    print("ok", flush=True)

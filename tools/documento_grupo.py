"""Gera o PDF explicativo do grupo. Tabelas e figuras vêm de results/ e docs/figures/.

Uso: python tools/documento_grupo.py            # versão sem frames, em docs/, enviada ao GitHub
     python tools/documento_grupo.py --frames   # versão com frames do BAGLS, em entrega/, fora do git
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, Spacer, Table, TableStyle
from reportlab.platypus.doctemplate import BaseDocTemplate, Frame, PageTemplate
from reportlab.platypus.tableofcontents import TableOfContents

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.documento_textos import T  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TAB, FIG = ROOT / "results" / "tables", ROOT / "docs" / "figures"
WITH_FRAMES = "--frames" in sys.argv
OUT = ROOT / "entrega" / "Documento_explicativo_LaringoPDI_com_imagens.pdf" if WITH_FRAMES else ROOT / "docs" / "Documento_explicativo_LaringoPDI.pdf"

for name, f in (("A", "arial.ttf"), ("A-B", "arialbd.ttf"), ("A-I", "ariali.ttf"), ("A-BI", "arialbi.ttf")):
    pdfmetrics.registerFont(TTFont(name, f"C:/Windows/Fonts/{f}"))
pdfmetrics.registerFontFamily("A", normal="A", bold="A-B", italic="A-I", boldItalic="A-BI")

INK, DIM, LINE, BLUE = colors.HexColor("#1b2a4a"), colors.HexColor("#56627f"), colors.HexColor("#c9cfdd"), colors.HexColor("#2a78d6")
GOOD, BAD, NEUT, HEAD = colors.HexColor("#d9f0e0"), colors.HexColor("#f9dcd5"), colors.HexColor("#f1f3f8"), colors.HexColor("#e6ecf8")
S = {
    "body": ParagraphStyle("body", fontName="A", fontSize=10, leading=14.5, textColor=INK, spaceAfter=7, alignment=TA_LEFT),
    "h1": ParagraphStyle("h1", fontName="A-B", fontSize=16, leading=20, textColor=INK, spaceBefore=8, spaceAfter=9, keepWithNext=1),
    "h2": ParagraphStyle("h2", fontName="A-B", fontSize=11.5, leading=15, textColor=INK, spaceBefore=9, spaceAfter=4, keepWithNext=1),
    "title": ParagraphStyle("title", fontName="A-B", fontSize=25, leading=30, textColor=INK, spaceAfter=10),
    "sub": ParagraphStyle("sub", fontName="A", fontSize=12, leading=17, textColor=DIM, spaceAfter=18),
    "cap": ParagraphStyle("cap", fontName="A", fontSize=8.5, leading=11.5, textColor=DIM, spaceBefore=3, spaceAfter=10),
    "cell": ParagraphStyle("cell", fontName="A", fontSize=8, leading=10, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="A-B", fontSize=8, leading=10, textColor=INK),
    "bullet": ParagraphStyle("bullet", fontName="A", fontSize=10, leading=14.5, textColor=INK, leftIndent=13, bulletIndent=2, spaceAfter=4),
    "box": ParagraphStyle("box", fontName="A", fontSize=10, leading=14.5, textColor=INK),
    "toc1": ParagraphStyle("toc1", fontName="A", fontSize=10, leading=17, textColor=INK),
}
W = A4[0] - 4.4 * cm


def num(v, d=3):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "–"
    return f"{float(v):.{d}f}".replace(".", ",")


class Doc(BaseDocTemplate):
    def __init__(self, path):
        super().__init__(str(path), pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
                         title="LaringoPDI: documento explicativo", author="Equipe LaringoPDI")
        self.addPageTemplates(PageTemplate("p", [Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="f", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)], onPage=self._footer))

    def _footer(self, c, d):
        c.setFont("A", 8)
        c.setFillColor(DIM)
        c.drawString(self.leftMargin, 1.15 * cm, "LaringoPDI: documento explicativo do grupo")
        c.drawRightString(A4[0] - self.rightMargin, 1.15 * cm, str(d.page))

    def afterFlowable(self, f):
        if isinstance(f, Paragraph) and f.style.name == "h1":
            self.notify("TOCEntry", (0, f.getPlainText(), self.page))


def P(text, style="body"):
    return Paragraph(text, S[style])


def paras(text):
    out = []
    for block in text.strip().split("\n\n"):
        b = block.strip()
        if b.startswith("## "):
            out.append(P(b[3:], "h2"))
        elif b.startswith("- "):
            out += [Paragraph(x[2:].strip(), S["bullet"], bulletText="•") for x in b.split("\n") if x.strip()]
        else:
            out.append(P(" ".join(b.split("\n"))))
    return out


def table(rows, widths, header=1, tint=None, align_num=None):
    data = [[Paragraph(str(c), S["cellb" if i < header else "cell"]) for c in r] for i, r in enumerate(rows)]
    total = sum(widths)
    t = Table(data, colWidths=[w / total * W for w in widths], repeatRows=header)
    st = [("BACKGROUND", (0, 0), (-1, header - 1), HEAD), ("LINEBELOW", (0, 0), (-1, -1), .4, LINE), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 3.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2), ("LEFTPADDING", (0, 0), (-1, -1), 3.5), ("RIGHTPADDING", (0, 0), (-1, -1), 3.5)]
    if tint:
        for i, r in enumerate(rows):
            if i < header:
                continue
            for j, c in enumerate(r):
                k = str(c).lower()
                if k.startswith(("melhora", "aumenta", "converge antes", "mais barata", "mlp melhor")):
                    st.append(("BACKGROUND", (j, i), (j, i), GOOD))
                elif k.startswith(("piora", "diminui", "converge depois", "mais cara", "clássico melhor", "não melhora")):
                    st.append(("BACKGROUND", (j, i), (j, i), BAD))
    t.setStyle(TableStyle(st))
    return t


def figure(name, caption, width=None, folder=FIG):
    p = Path(folder) / name
    if not p.exists():
        return [P(f"<i>Figura ausente: {escape(name)}</i>", "cap")]
    from PIL import Image as PI

    w, h = PI.open(p).size
    width = width or W
    return [KeepTogether([Image(str(p), width=width, height=width * h / w), P(caption, "cap")])]


def box(text):
    t = Table([[Paragraph(text, S["box"])]], colWidths=[W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), HEAD), ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10), ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 9)]))
    return [t, Spacer(1, 8)]


NAMES = {"raw": "sem PDI", "stretch": "alargamento", "he": "equalização", "clahe": "CLAHE", "mean": "média", "median": "mediana", "gauss_lpf": "passa-baixa", "spec": "remoção de reflexo", "homo": "homomórfica"}


def tab_visual():
    d = pd.read_csv(TAB / "pdi" / "tabela_unica.csv")
    rows = [["Técnica", "Entropia", "Contraste RMS", "Nitidez (Laplaciano)", "Saturação", "Reflexo", "Veredito visual", "ms/frame"]]
    for _, r in d.iterrows():
        rows.append([NAMES[r["tecnica"]], num(r["entropia"], 2), num(r["contraste_rms"], 3), num(r["laplaciano"], 0), num(r["saturacao"], 3), num(r["fracao_especular"], 3), r["veredito_visual"], num(r["ms_tecnica"], 2)])
    return table(rows, [2.6, 1.3, 1.4, 1.5, 1.3, 1.1, 1.9, 1.1], tint=True)


def tab_ruido():
    d = pd.read_csv(TAB / "pdi" / "tabela_unica.csv")
    d = d[d["psnr_sal_pimenta5"].notna()]
    rows = [["Filtro", "PSNR sal e pimenta", "SSIM sal e pimenta", "PSNR gaussiano", "SSIM gaussiano"]]
    for _, r in d.iterrows():
        rows.append([NAMES[r["tecnica"]], num(r["psnr_sal_pimenta5"], 1), num(r["ssim_sal_pimenta5"], 3), num(r["psnr_gauss10"], 1), num(r["ssim_gauss10"], 3)])
    return table(rows, [2, 1.5, 1.5, 1.5, 1.5])


def tab_seg():
    d = pd.read_csv(TAB / "pdi" / "tabela_unica.csv")
    rows = [["Técnica", "Dice", "IoU", "HD95 (px)", "Falha", "Δ Dice", "Veredito", "ms total/frame"]]
    for _, r in d.iterrows():
        rows.append([NAMES[r["tecnica"]], num(r["dice"]), num(r["iou"]), num(r["hd95_mediana"], 1), f"{r['taxa_falha'] * 100:.1f}%".replace(".", ","), num(r["delta_dice_vs_raw"]), r["veredito_seg"], num(r["ms_total"], 2)])
    return table(rows, [2.6, 1, 1, 1.1, 1, 1, 1.5, 1.3], tint=True)


def tab_mlp_seg():
    d = pd.read_csv(TAB / "mlp" / "segmentacao.csv")
    rows = [["Técnica", "Dice MLP", "Δ Dice", "Veredito do PDI", "Dice clássico", "MLP × clássico", "Época", "Custo (s)"]]
    for _, r in d.iterrows():
        rows.append([NAMES[r["tecnica"]], num(r["dice"]), num(r["delta_dice_vs_raw"]), r["veredito_seg"], num(r["dice_classico"]), r["veredito_vs_classico"], num(r["epoca_melhor"], 1), num(r["custo_total_s"], 1)])
    return table(rows, [2.5, 1, 1, 1.4, 1.1, 1.6, .9, 1], tint=True)


def tab_class(path, model=None, task="A"):
    d = pd.read_csv(path)
    if model:
        d = d[d["modelo"] == model]
    d = d[d["tarefa"] == task]
    rows = [["Técnica", "F1", "Sens.", "Espec.", "AUC", "Veredito F1", "Veredito sens.", "Melhor época", "Convergência", "Custo (s)", "Custo"]]
    for _, r in d.iterrows():
        rows.append([NAMES.get(r["tecnica"], r["tecnica"]), num(r["f1"]), num(r["sensibilidade"]), num(r["especificidade"]), num(r["auc"]), r["veredito_f1"], r["veredito_sensibilidade"], num(r["epoca_melhor"], 1), r["veredito_convergencia"], num(r["custo_total_s"], 0), r["veredito_custo"]])
    return table(rows, [2.3, .85, .85, .85, .85, 1.25, 1.25, .95, 1.4, .85, 1.25], tint=True)


def tab_attention():
    p = TAB / "cnn" / "atencao.csv"
    if not p.exists():
        return P("<i>Tabela de atenção ausente.</i>")
    d = pd.read_csv(p)
    rows = [["Técnica", "Atenção A", "Veredito registrado A", "Razão A", "Veredito da razão A", "Atenção B", "Veredito registrado B", "Razão B", "Veredito da razão B"]]
    for e, g in d.groupby("exp", sort=False):
        a, b = g[g["task"] == "A"].iloc[0], g[g["task"] == "B"].iloc[0]
        rows.append([NAMES[a["tecnica"]], num(a["att_glote"], 4), a["veredito_atencao"], num(a["razao_att_area"], 2), a["veredito_razao"], num(b["att_glote"], 4), b["veredito_atencao"], num(b["razao_att_area"], 2), b["veredito_razao"]])
    return table(rows, [2.2, 1, 1.3, .8, 1.3, 1, 1.3, .8, 1.3], tint=True)


def tab_family():
    d = pd.read_csv(TAB / "mlp" / "importancia_familias.csv")
    names = {"shape": "forma", "int": "intensidade", "tex": "textura"}
    rows = [["Família", "Queda na tarefa A", "Queda na tarefa B"]]
    for f in ("shape", "int", "tex"):
        rows.append([names[f], num(d[(d["familia"] == f) & (d["tarefa"] == "A")]["queda_media"].iloc[0]), num(d[(d["familia"] == f) & (d["tarefa"] == "B")]["queda_media"].iloc[0])])
    return table(rows, [2, 1.5, 1.5])


def tab_costs():
    d = pd.read_csv(TAB / "custo" / "por_fase.csv")
    rows = [["Fase", "Tempo (h)", "CPU (núcleos)", "RAM pico (GB)", "GPU?", "VRAM pico (MB)", "GPU média (W)", "Energia GPU (Wh)"]]
    for _, r in d.iterrows():
        rows.append([r["fase"], num(r["duracao_h"], 2), num(r["cpu_nucleos_medio"], 1), num(r["ram_processo_pico_gb"], 1), r["gpu_usada"], num(r["gpu_mem_pico_mb"], 0), num(r["gpu_potencia_media_w"], 0), num(r["gpu_energia_wh"], 0)])
    rows.append(["<b>Total</b>", f"<b>{num(d['duracao_h'].sum(), 2)}</b>", "", "", "", "", "", f"<b>{num(d['gpu_energia_wh'].sum(), 0)}</b>"])
    return table(rows, [3, 1, 1, 1.1, .7, 1.2, 1.1, 1.2])


def tab_technique_cost():
    d = pd.read_csv(TAB / "sintese" / "por_tecnica.csv")
    rows = [["Técnica", "PDI (ms/frame)", "Custo MLP A (s)", "Custo MLP B (s)", "Custo CNN A (s)", "Custo CNN B (s)"]]
    for _, r in d.iterrows():
        rows.append([r["nome"], num(r["ms_tecnica"], 2), num(r["mlp_A_custo_s"], 0), num(r["mlp_B_custo_s"], 0), num(r["cnn_A_custo_s"], 0), num(r["cnn_B_custo_s"], 0)])
    return table(rows, [2.5, 1.2, 1.2, 1.2, 1.2, 1.2])


def tab_synthesis():
    d = pd.read_csv(TAB / "sintese" / "por_tecnica.csv")
    rows = [["Técnica", "Imagem", "Seg. clássica", "Seg. MLP", "MLP A sens.", "MLP B sens.", "CNN A sens.", "CNN B sens.", "Atenção A", "Atenção B"]]
    for _, r in d.iterrows():
        rows.append([r["nome"], r["visual"], r["seg_classica"], r["seg_mlp"], r["mlp_A_v_sens"], r["mlp_B_v_sens"], r["cnn_A_v_sens"], r["cnn_B_v_sens"], r.get("razao_A", "–"), r.get("razao_B", "–")])
    return table(rows, [2, 1.25, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1], tint=True)


def tab_conclusions():
    rows = [["Técnica", "Conclusão"]] + [[k, v] for k, v in T["conclusoes"]]
    return table(rows, [2, 8])


def tab_hypotheses():
    h = json.loads((TAB / "sintese" / "hipoteses.json").read_text(encoding="utf-8"))
    rows = [["Hipótese", "Veredito", "Evidência"]]
    for k, text, ev in T["hipoteses"]:
        rows.append([f"<b>{k}.</b> {text}", h[k]["veredito"], ev])
    return table(rows, [3, 1.5, 6])


def frames_figures():
    """Figuras com frames do BAGLS, gravadas só em entrega/, que o git ignora."""
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from laringopdi import config as C
    from laringopdi import data as D
    from laringopdi.estudo_pdi import _apply
    from laringopdi.evaluate import gradcam_maps, load_model
    from laringopdi.features import _unpack
    from laringopdi.pdi.registry import to_work_size
    from laringopdi.pdi.segmentation import segment

    cfg = C.load_config() if hasattr(C, "load_config") else C.load()
    out = ROOT / "entrega" / "figuras"
    out.mkdir(parents=True, exist_ok=True)
    fr = D.load_frames(cfg)
    test = {str(v) for v in D.load_splits(cfg)[0]["test"]}
    fr = fr[fr["video_id"].astype(str).isin(test)]
    raw = C.path(cfg, "raw")
    best, area = None, -1.0
    for _, r in fr.sample(40, random_state=0).iterrows():
        m = D.imread(raw / r["seg_path"], gray=True) > 127
        if .004 < m.mean() < .04 and m.mean() > area:
            best, area = r, m.mean()
    g = to_work_size(D.imread(raw / best["path"], gray=True), cfg["data"]["work_size"])
    gt = cv2.resize((D.imread(raw / best["seg_path"], gray=True) > 127).astype(np.uint8), g.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for ax, t in zip(axes.ravel(), NAMES):
        p = _apply(g, t)
        ax.imshow(p, cmap="gray", vmin=0, vmax=255)
        ax.contour(segment(p, "region_grow", tol=20), levels=[.5], colors="#eb6834", linewidths=1.2)
        ax.contour(gt, levels=[.5], colors="#3ddc97", linewidths=.9)
        ax.set_title(NAMES[t], fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "tecnicas.png", dpi=170)
    plt.close(fig)
    size = cfg["data"]["image_size"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, (e, name) in zip(axes, (("E00", "sem PDI"), ("E03", "CLAHE"))):
        X, paths, _ = D.load_cache(cfg, e)
        i = list(paths).index(best["path"])
        cam = gradcam_maps(load_model(cfg, e, "A", "resnet18", 0), X[i:i + 1], device="cpu")[0]
        mask = _unpack(np.load(D.cache_paths(cfg, e)["masks"])["gt_mask_lb"], size)[i]
        ax.imshow(X[i, ..., 0], cmap="gray")
        ax.imshow(cam, cmap="inferno", alpha=.45)
        ax.contour(mask, levels=[.5], colors="#3ddc97", linewidths=1)
        ax.set_title(f"Grad-CAM: {name}", fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "gradcam.png", dpi=170)
    plt.close(fig)
    return out


def build():
    story = [Spacer(1, 3.5 * cm), P("LaringoPDI", "title"), P("Documento explicativo do projeto: o que fizemos, por que fizemos e o que os resultados significam", "sub")]
    story += box(T["capa"])
    story += [Spacer(1, .4 * cm), P(T["data_e_versao"]), PageBreak()]
    toc = TableOfContents()
    toc.levelStyles = [S["toc1"]]
    story += [P("Sumário", "h2"), toc, PageBreak()]

    def section(title, text=None):
        story.append(P(title, "h1"))
        if text:
            story.extend(paras(text))

    section("1. Resumo", T["resumo"])
    story.append(PageBreak())
    section("2. O problema e a pergunta", T["problema"])
    section("3. Os dados", T["dados"])
    section("4. O caminho do estudo", T["fluxo"])
    story.append(table([["Etapa", "Pergunta", "Como medimos"]] + T["etapas"], [2.3, 3.6, 4]))
    story.append(Spacer(1, 8))
    section("5. Técnicas de PDI", T["tecnicas_intro"])
    story.append(table([["Técnica", "Origem", "O que faz", "Por que usar na laringoscopia"]] + T["tecnicas"], [1.8, 1, 3.8, 3.8]))
    story.append(Spacer(1, 8))
    if WITH_FRAMES:
        out = frames_figures()
        story += figure("tecnicas.png", "Figura 1. Um frame do BAGLS em cada técnica. Verde: máscara do especialista. Laranja: crescimento de região com tolerância 20 sobre a imagem da técnica.", 14 * cm, out)
    else:
        story.append(P("<i>A versão com imagens deste documento, fora do GitHub, mostra um frame em cada técnica.</i>", "cap"))
    story.append(P("Métodos de segmentação", "h2"))
    story.append(table([["Método", "Ideia"]] + T["metodos_seg"], [2.2, 8]))
    story.append(Spacer(1, 8))
    section("6. Como ler as métricas", T["metricas_intro"])
    story.append(table([["Métrica", "O que diz", "Melhor quando"]] + T["metricas"], [2.2, 6.5, 1.5]))
    story.append(Spacer(1, 8))
    story.extend(paras(T["veredito"]))

    section("7. Resultado 1: o PDI na própria imagem", T["r_visual_1"])
    story.append(tab_visual())
    story.append(P("Tabela 1. Medidas da imagem com 5.870 frames. Tempo em 1 núcleo da CPU. Média, mediana e passa-baixa são avaliadas pelo efeito no ruído.", "cap"))
    story.append(tab_ruido())
    story.append(P("Tabela 2. Remoção de ruído artificial: sal e pimenta a 5% e gaussiano com desvio 10.", "cap"))
    story.extend(paras(T["r_visual_2"]))

    section("8. Resultado 2: segmentação clássica", T["r_seg_1"])
    story.append(tab_seg())
    story.append(P("Tabela 3. Melhor método de cada técnica, no teste dos 5 folds. Veredito contra a imagem sem PDI.", "cap"))
    story += figure("pdi_segmentacao_custo.png", "Figura. Esquerda: Dice por método e técnica. Direita: Dice e custo. O efeito de uma técnica depende do método.")
    story.extend(paras(T["r_seg_2"]))

    section("9. Resultado 3: MLP na segmentação", T["r_mlp_seg_1"])
    story.append(tab_mlp_seg())
    story.append(P("Tabela 4. MLP por pixel, 1 semente e 5 folds. Custo: PDI nos frames de treino + treino até a melhor validação.", "cap"))
    story += figure("mlp_segmentacao.png", "Figura. MLP de segmentação por técnica.")
    story.extend(paras(T["r_mlp_seg_2"]))

    section("10. Resultado 4: MLP na classificação", T["r_mlp_1"])
    story.append(tab_class(TAB / "mlp" / "classificacao.csv", "mlp18", "A"))
    story.append(P("Tabela 5. MLP de 18 características, Tarefa A, nível de vídeo. Custo em segundos por treino.", "cap"))
    story.append(tab_class(TAB / "mlp" / "classificacao.csv", "mlp18", "B"))
    story.append(P("Tabela 6. MLP de 18 características, Tarefa B.", "cap"))
    story.append(tab_family())
    story.append(P("Tabela 7. Importância por família: quanto a acurácia balanceada cai ao embaralhar a família (imagem sem PDI).", "cap"))
    story += figure("mlp_curvas.png", "Figura. Curvas de treino da MLP, sem PDI e com CLAHE.")
    story.extend(paras(T["r_mlp_2"]))

    section("11. Resultado 5: CNN na classificação", T["r_cnn_1"])
    story.append(tab_class(TAB / "cnn" / "classificacao.csv", None, "A"))
    story.append(P("Tabela 8. ResNet-18, Tarefa A, nível de vídeo, 3 sementes × 5 folds.", "cap"))
    story.append(tab_class(TAB / "cnn" / "classificacao.csv", None, "B"))
    story.append(P("Tabela 9. ResNet-18, Tarefa B.", "cap"))
    story += figure("sintese_deltas.png", "Figura. O que cada técnica muda contra a imagem sem PDI, na ResNet-18 e na MLP. Barras: IC 95% por bootstrap. O veredito exige também o teste corrigido.")
    story += figure("cnn_curvas.png", "Figura. Curvas de treino da ResNet-18, sem PDI e com CLAHE.")
    story.extend(paras(T["r_cnn_2"]))
    story.append(P("Para onde a CNN olha (Grad-CAM)", "h2"))
    story.extend(paras(T["r_gradcam_1"]))
    story.append(tab_attention())
    story.append(P("Tabela 10. Atenção: parcela da energia do Grad-CAM dentro da glote. Razão: atenção dividida pela área da glote; acima de 1, a rede olha para a glote mais do que o acaso.", "cap"))
    if WITH_FRAMES:
        story += figure("gradcam.png", "Figura. Exemplo de Grad-CAM. Verde: máscara do especialista.", 13 * cm, ROOT / "entrega" / "figuras")
    story.extend(paras(T["r_gradcam_2"]))

    section("12. Tempo, custo e hardware", T["r_custo_1"])
    story.append(tab_costs())
    story.append(P("Tabela 11. Medições por fase. Energia: placa de vídeo inteira, sem CPU nem o restante do PC. Índice e divisão dos folds não foram medidos.", "cap"))
    story += figure("custo_fases.png", "Figura. Tempo por fase do estudo.", 13 * cm)
    story.append(tab_technique_cost())
    story.append(P("Tabela 12. Custo por técnica: PDI nos frames de treino + treino até a melhor validação, por treino.", "cap"))
    story += figure("sintese_custo_beneficio.png", "Figura. Sensibilidade e custo da ResNet-18.")
    story.extend(paras(T["r_custo_2"]))

    section("13. Síntese técnica por técnica", T["sintese_1"])
    story.append(tab_synthesis())
    story.append(P("Tabela 13. Vereditos de todas as etapas. Classificação com sensibilidade, a métrica prioritária.", "cap"))
    story.append(tab_conclusions())
    story.append(Spacer(1, 8))
    section("14. As hipóteses da Etapa 1", T["hipoteses_intro"])
    story.append(tab_hypotheses())
    story.append(Spacer(1, 8))
    section("15. Limites e correções", T["limites"])
    sb_p = TAB / "sintese" / "sem_bordas.csv"
    if sb_p.exists():
        sb = pd.read_csv(sb_p)
        rows = [["Modelo", "Tarefa", "F1 com → sem bordas", "Sens. com → sem", "AUC com → sem", "Veredito F1", "Veredito sens."]]
        for _, r in sb.iterrows():
            rows.append(["MLP" if r["modelo"] == "mlp18" else "ResNet-18", r["tarefa"], f"{num(r['f1_com_bordas'])} → {num(r['f1_sem_bordas'])}",
                         f"{num(r['sensibilidade_com_bordas'])} → {num(r['sensibilidade_sem_bordas'])}", f"{num(r['auc_com_bordas'])} → {num(r['auc_sem_bordas'])}",
                         r["veredito_f1"], r["veredito_sensibilidade"]])
        story.append(table(rows, [1.3, .8, 1.8, 1.8, 1.8, 1.2, 1.2], tint=True))
        story.append(P("Tabela 14. Controle do formato do vídeo: imagem sem PDI com e sem as bordas pretas, 5 folds × 3 sementes. "
                       "Aqui, \"piora\" significa que o modelo perdeu desempenho ao perder o formato.", "cap"))
    section("16. O que ainda falta e onde está cada coisa", T["faltando"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    assert not any("Pendente:" in str(v) for v in T.values()), "há texto pendente"
    Doc(OUT).multiBuild(story)
    print(OUT, round(OUT.stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    build()

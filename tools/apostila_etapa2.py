"""Apostila da apresentação da Etapa 2, no formato da apostila da Etapa 1. Os números saem de results/tables.

Uso: python tools/apostila_etapa2.py --out entrega/LaringoPDI_Apostila_Etapa2.pdf
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))

import relatorio_etapa2 as R  # noqa: E402
from relatorio_etapa2 import fp, n, pc, tab  # noqa: E402

AVISO = R.AVISO


def numbers() -> dict:
    """Tudo o que as falas citam, lido das tabelas."""
    q = tab("enhance_quality.csv").set_index("technique")
    den = tab("denoise_summary.csv").set_index(["noise", "filter"])
    piv = R.seg_pivot()
    am = tab("seg_adaptive_morph.csv").set_index("comparison")
    cl = tab("classif_aula.csv").set_index(["exp", "task"])
    m = tab("mlp18_p4.csv")
    ver = tab("verdicts_aula_resnet18.csv").set_index(["exp", "task"])
    old = tab("verdicts_resnet18.csv").set_index(["exp", "task"])
    tm = tab("timing.csv").set_index("name")["median_ms"]
    imp = tab("importance_c6.csv").set_index(["task", "family"])["drop_mean"]
    from laringopdi.pdi.segmentation import apply_best, best_config, dice
    g, gt = R._frame("1881")
    bc = best_config()
    return {"q": q, "den": den, "piv": piv, "am": am, "cl": cl, "m4": m[(m["system"] == "C6") & (m["task"] == "B")].iloc[0], "sens": tab("sens_c6_resnet.csv").set_index("task"),
            "ver": ver, "old": old, "tm": tm, "imp": imp, "att": tab("attention_aula.csv").set_index(["exp", "task"]), "hc": tab("he_clahe_resnet18.csv").set_index("task"),
            "hyp": json.loads((R.T / "hypotheses_etapa1.json").read_text(encoding="utf-8")),
            "ex_raw": dice(apply_best(g, bc["method"], bc["params"], "raw"), gt), "ex_clahe": dice(apply_best(g, bc["method"], bc["params"], "clahe"), gt)}


def build(out: Path) -> None:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.fonts import addMapping
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    fd = Path("C:/Windows/Fonts")
    for name, f in (("Arial", "arial.ttf"), ("Arial-Bold", "arialbd.ttf"), ("Arial-Italic", "ariali.ttf"), ("Arial-BoldItalic", "arialbi.ttf")):
        pdfmetrics.registerFont(TTFont(name, str(fd / f)))
    for b_, i_, name in ((0, 0, "Arial"), (1, 0, "Arial-Bold"), (0, 1, "Arial-Italic"), (1, 1, "Arial-BoldItalic")):
        addMapping("Arial", b_, i_, name)
    ink, navy = colors.HexColor("#222B3D"), colors.HexColor(R.INK)
    st = {"h1": ParagraphStyle("h1", fontName="Arial-Bold", fontSize=15, leading=19, spaceBefore=12, spaceAfter=7, keepWithNext=1, textColor=navy),
          "h2": ParagraphStyle("h2", fontName="Arial-Bold", fontSize=11.2, leading=14, spaceBefore=8, spaceAfter=3, keepWithNext=1, textColor=navy),
          "p": ParagraphStyle("p", fontName="Arial", fontSize=10.2, leading=14.4, spaceAfter=6, textColor=ink),
          "b": ParagraphStyle("b", fontName="Arial", fontSize=10.2, leading=14, leftIndent=13, bulletIndent=3, spaceAfter=3, textColor=ink),
          "fala": ParagraphStyle("fala", fontName="Arial", fontSize=10.2, leading=14.6, leftIndent=9, rightIndent=4, spaceBefore=3, spaceAfter=8, borderPadding=(5, 6, 5, 7), backColor=colors.HexColor("#EEF3FB"), textColor=ink),
          "c": ParagraphStyle("c", fontName="Arial", fontSize=9, leading=11.8, textColor=ink),
          "ch": ParagraphStyle("ch", fontName="Arial-Bold", fontSize=9, leading=11.8, textColor=colors.white),
          "tt": ParagraphStyle("tt", fontName="Arial-Bold", fontSize=30, leading=36, alignment=TA_CENTER, textColor=navy),
          "cv": ParagraphStyle("cv", fontName="Arial", fontSize=12, leading=17, alignment=TA_CENTER, textColor=ink)}
    W = A4[0] - 4.4 * cm
    story: list = []
    P = lambda t, s="p": story.append(Paragraph(t, st[s]))
    B = lambda items: [story.append(Paragraph(i, st["b"], bulletText="•")) for i in items]
    Fala = lambda t: story.append(Paragraph(f"<b>Fale:</b> {t}", st["fala"]))

    def T_(rows, widths):
        star = lambda c: f"<b>{c[2:]}</b>" if str(c).startswith("★ ") else str(c)      # a fonte não tem a estrela: vira negrito
        t = Table([[Paragraph(star(c), st["ch" if i == 0 else "c"]) for c in r] for i, r in enumerate(rows)], colWidths=[W * x for x in widths], repeatRows=1)
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .3, colors.HexColor("#C9CFDD")), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("BACKGROUND", (0, 0), (-1, 0), navy),
                               ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4), ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
        story.extend([t, Spacer(1, 9)])

    d = numbers()
    q, den, piv, am, cl, m4, sens, ver, old, tm, imp, att, hc, hyp = (d[k] for k in ("q", "den", "piv", "am", "cl", "m4", "sens", "ver", "old", "tm", "imp", "att", "hc", "hyp"))
    sp5, sp10 = den.loc["sp_5"], den.loc["sp_10"]
    rg, amb = n(piv.loc["region_grow", "clahe"]), n(piv.loc["adaptive_morph", "raw"])
    ao = am.loc["adaptive_morph_vs_otsu_morph"]
    sa, sb = sens.loc["A"], sens.loc["B"]
    e01 = ver.loc[("E01", "A")]
    a3a, a3b, a2b = att.loc[("E03", "A")], att.loc[("E03", "B")], att.loc[("E02", "B")]
    v = lambda k: hyp[k]["verdict"]

    # ------------------------------------------------------------- capa
    story += [Spacer(1, 5 * cm), Paragraph("LaringoPDI", st["tt"]), Spacer(1, 10), Paragraph("Apostila da apresentação da Etapa 2", st["cv"]), Paragraph("15 minutos, com demonstração do protótipo", st["cv"]), Spacer(1, 2 * cm)]
    for nome in ("Nathan Malheiros", "Tales Rios", "Pedro Nobre", "Marcos Vinicius", "Marcos Mendes"):
        story.append(Paragraph(nome, st["cv"]))
    story += [Spacer(1, 2 * cm), Paragraph("UNIFOR · Prof.ª Lyndainês Santos", st["cv"]), Paragraph(AVISO, st["cv"]), PageBreak()]

    # ------------------------------------------------------------- 1
    P("1. O projeto em 1 minuto", "h1")
    P("Todos precisam saber explicar o projeto sem slide. Decorem a ideia, não as palavras. <b>Agora falem sempre no passado:</b> \"segmentamos\", \"treinamos\", \"medimos\". A Etapa 1 era a promessa; a Etapa 2 mostra o que aconteceu.")
    B([f"<b>O problema.</b> Professores adoecem da voz, e o exame que mostra as pregas vocais é lido a olho, numa imagem com reflexo, pouco contraste e quadros escuros.",
       f"<b>O que fizemos.</b> Usamos PDI para realçar a imagem e segmentar a glote, medimos 18 características e treinamos uma MLP para decidir entre saudável e alterado. Comparamos com uma CNN, a ResNet-18.",
       f"<b>O que descobrimos.</b> Na segmentação, o PDI faz muita diferença, e o efeito depende do método: o melhor foi crescimento de região com CLAHE, com Dice {rg}. Na classificação com a CNN, as técnicas de aula não mudaram a nota de forma detectável. A MLP ficou perto da CNN nas lesões estruturais, mas a sensibilidade em saudável × alterado ainda é baixa.",
       "<b>A mensagem honesta.</b> Nem todas as hipóteses se confirmaram, e isso é resultado válido, porque as regras foram registradas antes."])
    P("<b>Três frases para repetir:</b> \"O PDI é o objeto de estudo.\" \"Começamos sempre pela técnica vista em aula.\" \"É apoio à triagem, não diagnóstico.\"")

    # ------------------------------------------------------------- 2
    P("2. Roteiro geral: 15 minutos", "h1")
    T_([["Bloco", "Assunto (item do enunciado)", "Quem", "Tempo"], ["1", "Equipe, problema social e público-alvo", "Tales", "1 min 30 s"], ["—", "Depoimento (se autorizado)", "Marcos Mendes", "40 s"],
        ["2", "Delimitação e dados", "Marcos Vinicius", "1 min"], ["3", "Objetivos, arquitetura e ferramentas", "Nathan", "1 min 30 s"],
        ["4", "Técnicas e demonstração: realce e filtragem (abas 1 e 2)", "Marcos Mendes", "2 min 30 s"], ["5", "Técnicas e demonstração: segmentação (aba 3)", "Pedro", "2 min"],
        ["6", "Demonstração: PDI + MLP × CNN (aba 4)", "Pedro", "2 min"], ["7", "Resultados e hipóteses (aba 5)", "Nathan", "1 min 30 s"], ["8", "Limitações e contribuição", "Tales", "1 min 10 s"], ["9", "Mudanças, conclusões e melhorias", "Nathan", "50 s"],
        ["", "<b>Total</b>", "", "<b>~14 min 50 s</b>"]], [.09, .57, .19, .15])
    P("Esta divisão segue a da Etapa 1 e pode ser ajustada pela equipe. O depoimento só entra com autorização. Quem não está falando opera o computador.")
    P("<b>Cuidado com os números da Etapa 1.</b> Os valores da apresentação anterior eram exemplos de uma imagem. Agora vocês têm a média do conjunto; não misturem os dois.")
    T_([["O que era", "Exemplo (Etapa 1)", "Média do conjunto (Etapa 2)"],
        ["Dice adaptativa + morfologia", "0,97", amb], ["Dice Otsu", "≈ 0,10", n(piv.loc["otsu", "raw"])], ["Entropia sem realce → CLAHE", "5,67 → 6,29", f"{n(q.loc['raw', 'entropy_mean'], 2)} → {n(q.loc['clahe', 'entropy_mean'], 2)}"],
        ["Média × mediana 3×3, sal e pimenta (dB)", "27,3 × 36,3", f"{n(sp5.loc['mean3', 'psnr_mean'], 1)} × {n(sp5.loc['median3', 'psnr_mean'], 1)} (a 5%)"]], [.46, .22, .32])

    # ------------------------------------------------------------- falas
    P("3. Falas de cada bloco", "h1")
    P("Bloco 1. Tales: problema e público (1 min 30 s)", "h2")
    Fala("Somos a equipe do LaringoPDI. O distúrbio de voz é a principal causa de afastamento por saúde entre professores, 17,7%, e o Ministério da Saúde o reconhece como doença do trabalho. Quem se beneficia são os professores; quem usa a ferramenta é o profissional que lê a laringoscopia. O exame é lido a olho, numa imagem difícil. Na Etapa 1 prometemos medir quanto o PDI ajuda nisso. Hoje mostramos o que encontramos.")
    P("Passagem: \"Esses números parecem distantes, mas não são. Para o Marcos, isso aconteceu dentro de casa.\" Depois do depoimento, o Marcos Vinicius assume.")
    P("Bloco 2. Marcos Vinicius: dados (1 min)", "h2")
    Fala("Usamos o BAGLS: 59.250 frames de 640 vídeos, de sete hospitais, com a máscara da glote feita por especialistas. Ficamos com até 10 frames por vídeo, em 587 vídeos com rótulo. Frames do mesmo vídeo são quase iguais; por isso dividimos os dados por vídeo, em cinco rodadas, para a nota não ficar inflada. A ferramenta apoia a análise de imagens estáticas em contexto acadêmico: não dá diagnóstico.")
    P("Bloco 3. Nathan: objetivos e arquitetura (1 min 30 s)", "h2")
    Fala("O objetivo geral foi desenvolver e avaliar um protótipo de PDI para realçar, segmentar e caracterizar a glote, medindo quanto cada técnica ajuda. Cumprimos os cinco objetivos específicos: investigar, implementar, avaliar contra a referência, extrair características e treinar a MLP, e comparar com a CNN. A arquitetura tem dois braços. No primeiro, o frame passa por realce, segmentação e 18 medidas, que entram numa MLP 18–32–16–2. No segundo, a ResNet-18 recebe a imagem e aprende sozinha. Tudo em Python, com OpenCV, scikit-learn, PyTorch e Streamlit.")
    P("Bloco 4. Marcos Mendes: realce e filtragem (2 min 30 s)", "h2")
    P("<b>Faça:</b> abra o app no frame 1881, aba 1; depois a aba 2, com sal e pimenta a 5%, e mova o σ de Fourier.")
    Fala(f"Este frame é escuro. O alargamento estica o histograma, a equalização global dá o maior contraste e o CLAHE trabalha por blocos. Em todo o conjunto, o CLAHE teve a maior entropia, {n(q.loc['clahe', 'entropy_mean'], 2)} bits. A nossa hipótese era que a equalização estourava o tecido, e isso não se sustentou: a imagem sem realce já tem {pc(q.loc['raw', 'saturated_frac_mean'])} de pixels nos extremos, quase todos do fundo preto. Por isso a H1 ficou parcial.")
    Fala(f"Na filtragem, a H2 se confirmou: em sal e pimenta a 5%, a mediana chegou a {n(sp5.loc['median3', 'psnr_mean'], 1)} dB contra {n(sp5.loc['mean3', 'psnr_mean'], 1)} da média. Em ruído gaussiano, a diferença entre elas quase some. Em Fourier, um σ pequeno corta mais frequência e borra mais.")
    P("Bloco 5. Pedro: segmentação (2 min)", "h2")
    P("<b>Faça:</b> aba 3. Mostre o passo a passo e, em seguida, a comparação dos métodos.")
    Fala(f"Aqui está o passo a passo que prometemos: limiar adaptativo, abertura, fechamento e componente central. A H3 se confirmou: sem realce, a adaptativa deu Dice {n(piv.loc['adaptive', 'raw'])} contra {n(piv.loc['otsu', 'raw'])} do Otsu. Mas atenção: o 0,97 da Etapa 1 era o exemplo de uma imagem. No conjunto inteiro, a adaptativa com morfologia ficou em {amb}, sem diferença detectável para o Otsu com morfologia. Como isso não bastou, seguimos a nossa regra e testamos técnicas externas. O melhor foi crescimento de região com CLAHE, com Dice {rg}. Neste frame escuro, o Dice vai de {n(d['ex_raw'], 2)} para {n(d['ex_clahe'], 2)}. O realce também pode atrapalhar: na adaptativa com morfologia, o CLAHE derrubou o Dice para {n(piv.loc['adaptive_morph', 'clahe'])}.")
    P("Bloco 6. Pedro: PDI + MLP × CNN (2 min)", "h2")
    P("<b>Faça:</b> aba 4. Mostre as 18 características, a saída da MLP e depois o Grad-CAM da ResNet-18.")
    Fala(f"No Braço 1, estas são as 18 medidas de forma, intensidade e textura, que entram na MLP. No Braço 2, a ResNet-18 decide sozinha, e o Grad-CAM mostra onde ela olhou. A sensibilidade era a nossa métrica prioritária. Em lesões estruturais, a MLP encontrou mais casos: {n(sb['sens_c6'], 2)} contra {n(sb['sens_resnet18'], 2)}, com mais alarmes falsos. O F1 foi {n(m4['f1_classic'])} contra {n(m4['f1_resnet18'])}. Pela regra, a H4 se confirma, mas de forma frágil. Em saudável × alterado, a CNN foi mais sensível, {n(sa['sens_resnet18'], 2)} contra {n(sa['sens_c6'], 2)}, e nenhum dos dois chega a um nível adequado para triagem.")
    P("Bloco 7. Nathan: resultados (1 min 30 s)", "h2")
    P("<b>Faça:</b> aba 5, na tabela das hipóteses.")
    Fala(f"As seis hipóteses: H1 parcial, H2 e H3 confirmadas, H4 confirmada de forma frágil, H5 parcial e H6 refutada: a forma não foi o grupo mais importante, e sim a textura. Na CNN, a equalização, o CLAHE e a mediana não mudaram a classificação de forma detectável. O CLAHE aumentou a atenção da rede na glote nas duas tarefas ({n(a3a['ratio_E00'], 2)} para {n(a3a['ratio_exp'], 2)} na A), sem ganho na nota. A remoção de reflexo, técnica externa, foi a única que deu melhora, só na Tarefa A, e o resultado é fraco: usa uma semente e não se sustenta numa família maior de comparações.")
    P("Bloco 8. Tales: limitações e contribuição (1 min 10 s)", "h2")
    Fala("As limitações principais: a sensibilidade na Tarefa A é baixa; os frames são imagens paradas, e distúrbios funcionais dependem da vibração; há só 54 vídeos estruturais; e não houve validação clínica. Por isso não entregamos uma triagem pronta. A contribuição é mostrar a quem lê o exame o que cada técnica faz na imagem e quais realmente ajudam a delimitar a glote, além de servir como material de formação. Para os professores, o ganho é indireto: um ponto de partida medido para ferramentas futuras.")
    P("Bloco 9. Nathan: mudanças e conclusão (50 s)", "h2")
    Fala("Em relação à Etapa 1, a principal mudança foi o método final de segmentação, previsto como técnica externa caso as de aula não bastassem. Nossa conclusão: o PDI fez mais diferença para um método de regras fixas do que para uma rede que aprende os próprios filtros. Como melhorias, ajustaríamos o limiar para priorizar a sensibilidade, usaríamos mais sementes e a sequência de frames do vídeo. O PDI é o objeto de estudo. Obrigado.")

    # ------------------------------------------------------------- perguntas
    P("4. Perguntas prováveis e respostas", "h1")
    P("Respostas curtas. A pessoa indicada responde primeiro; qualquer um pode completar.")
    T_([["Pergunta", "Resposta", "Quem"],
        ["Por que o 0,97 virou 0,28?", f"O 0,97 era o exemplo de uma imagem. A média em 587 vídeos é {amb}. Medir no conjunto era justamente o objetivo.", "Pedro"],
        ["O método prometido falhou?", f"Funciona, mas tem limite no conjunto: {amb}, igual ao Otsu com morfologia ({n(ao['dice_a'])}; p {fp(ao['p_holm'])}). A regra previa técnica externa nesse caso.", "Pedro"],
        ["CLAHE é melhor que a equalização?", f"Depende. No método final, {n(hyp['H1']['numbers']['dice_clahe'])} × {n(hyp['H1']['numbers']['dice_he'])}, mas o CLAHE venceu em {hyp['H1']['numbers']['methods_clahe_wins']} de 5 métodos, e a equalização em {hyp['H1']['numbers']['methods_he_wins']}. Na CNN, não houve diferença detectável.", "M. Mendes"],
        ["Por que a equalização não \"estourou\"?", f"O frame sem realce já tem {pc(q.loc['raw', 'saturated_frac_mean'])} de pixels nos extremos, quase todos do fundo preto. A equalização fica em {pc(q.loc['he', 'saturated_frac_mean'])}.", "M. Mendes"],
        ["A MLP empata com a CNN?", f"Pela regra, sim: o intervalo do F1 vai de {n(m4['ci_low'])} a {n(m4['ci_high'])} e contém o zero. Mas é frágil, e a acurácia da CNN é maior ({n(m4['acc_resnet18'])} × {n(m4['acc_classic'])}).", "Pedro"],
        ["Se a MLP é mais sensível na B, ela é melhor?", f"Não necessariamente. Ela encontra mais casos ({n(sb['sens_c6'], 2)} × {n(sb['sens_resnet18'], 2)}), mas libera menos saudáveis (especificidade {n(cl.loc[('C6', 'B'), 'specificity'], 2)} × {n(cl.loc[('E00', 'B'), 'specificity'], 2)}). Com {int(sb['n_altered'])} vídeos, a incerteza é grande.", "Nathan"],
        ["Por que a sensibilidade é baixa na Tarefa A?", "A Tarefa A inclui distúrbios funcionais, que podem não aparecer numa imagem parada. Já esperávamos desempenho modesto.", "Nathan"],
        ["Por que o PDI não ajudou a CNN?", "Uma explicação provável é que a rede pré-treinada aprende a lidar com a imagem crua. Nosso estudo mede o efeito, mas não testa essa causa.", "Nathan"],
        ["A remoção de reflexo melhora?", f"Só na Tarefa A, com p corrigido {n(e01['mcnemar_p_holm'])}. Com oito técnicas na correção, fica {n(old.loc[('E01', 'A'), 'mcnemar_p_holm'])}. Com uma semente, precisa ser confirmado.", "Nathan"],
        ["O que significa atenção 1,64?", f"É a parte do Grad-CAM dentro da glote dividida pela área da glote: 1 seria o acaso. O CLAHE foi de {n(a3a['ratio_E00'], 2)} para {n(a3a['ratio_exp'], 2)} (A) e de {n(a3b['ratio_E00'], 2)} para {n(a3b['ratio_exp'], 2)} (B). A equalização reduziu na B ({n(a2b['ratio_E00'], 2)} para {n(a2b['ratio_exp'], 2)}).", "Pedro"],
        ["Por que a forma não foi a mais importante?", f"Na Tarefa B, textura {n(imp.loc[('B', 'f18tex')])}, intensidade {n(imp.loc[('B', 'f18int')])}, forma {n(imp.loc[('B', 'f18shape')])}. A segmentação imperfeita pode prejudicar as medidas de forma, mas não separamos essa explicação.", "Pedro"],
        ["O que é McNemar? E Holm?", "McNemar compara os erros de dois classificadores nos mesmos vídeos. Holm corrige o p quando há várias comparações, para não contar sorte como resultado.", "Nathan"],
        ["Isso é PDI ou IA?", "PDI. O objeto de estudo é o efeito de cada técnica. A comparação com a CNN está na lista da própria disciplina.", "Nathan"],
        ["Fizeram mais coisas que não mostraram?", "Sim: outra CNN, recortes pela glote e mais características. Está registrado no repositório. Mostramos aqui o que prometemos.", "Nathan"],
        ["Por que usar técnicas externas?","O critério da Etapa 1: começar pela de aula e recorrer à externa como comparação ou quando a de aula não bastar.", "Nathan"],
        ["Funciona em imagem de outra origem?", "O app aceita upload, mas a segmentação foi calibrada no BAGLS, e o resultado pode ser pior.", "Pedro"],
        ["Quanto tempo leva?", f"CLAHE {n(tm['E03 (clahe)'], 1)} ms; segmentação {n(tm['region_grow + clahe'], 1)} ms; 18 características {n(tm['18 características (C6)'], 1)} ms; MLP {n(tm['MLP C6 (1 vetor)'], 1)} ms; ResNet-18 {n(tm['resnet18 (CPU, batch 1)'], 1)} ms em CPU.", "Nathan"],
        ["É diagnóstico?", AVISO, "Qualquer um"]], [.3, .56, .14])
    P("<b>Se não souber a resposta:</b> \"Não medimos isso, professora.\" Nunca invente um número.")

    # ------------------------------------------------------------- cola
    P("5. Cola rápida", "h1")
    P("Todos devem saber os números em negrito.")
    T_([["Número", "O que é"], ["★ 17,7%", "afastamento de professores por distúrbio de voz"], ["★ 59.250 / 640 / 7", "frames / vídeos / hospitais do BAGLS"], ["★ 5.870 / 587", "frames e vídeos usados"], ["★ 18 → 32 → 16 → 2", "arquitetura da MLP"],
        [f"★ {rg}", "Dice: crescimento de região + CLAHE (melhor método)"], [f"★ {amb}", "Dice: adaptativa + morfologia (método prometido)"], [f"{n(piv.loc['adaptive', 'raw'])} × {n(piv.loc['otsu', 'raw'])}", "Dice: adaptativa × Otsu, sem realce"],
        [f"{n(d['ex_raw'], 2)} → {n(d['ex_clahe'], 2)}", "Dice do frame 1881, sem realce → CLAHE (exemplo de 1 imagem)"], [f"{n(sp5.loc['median3', 'psnr_mean'], 1)} × {n(sp5.loc['mean3', 'psnr_mean'], 1)} dB", "mediana × média 3×3, sal e pimenta 5%"], [f"{n(sp10.loc['median3', 'psnr_mean'], 1)} × {n(sp10.loc['mean3', 'psnr_mean'], 1)} dB", "mediana × média 3×3, sal e pimenta 10%"],
        [f"★ {n(m4['f1_classic'])} × {n(m4['f1_resnet18'])}", "F1 Tarefa B: MLP × ResNet-18"], [f"★ {n(sb['sens_c6'], 2)} × {n(sb['sens_resnet18'], 2)}", "sensibilidade Tarefa B: MLP × ResNet-18"], [f"★ {n(sa['sens_c6'], 2)} × {n(sa['sens_resnet18'], 2)}", "sensibilidade Tarefa A: MLP × ResNet-18"],
        ["H1 a H6", f"{v('H1')}; {v('H2')}; {v('H3')}; {v('H4')} (frágil); {v('H5')}; {v('H6')}"]], [.28, .72])

    # ------------------------------------------------------------- checklist
    P("6. Checklist antes de apresentar", "h1")
    B(["Todos leram a apostila inteira e o relatório.", "Um ensaio completo com cronômetro; o limite é de 15 minutos.", "Marcos Mendes pediu autorização e ensaiou o depoimento, ou o bloco saiu do roteiro.",
       "O app foi aberto antes, no computador da sala, porque a primeira carga é lenta. Frame 1881 selecionado.", "As capturas de <i>docs/figures/app/</i> estão à mão, como plano B.", "O código e o material da apresentação foram enviados ao AVA no prazo: atraso custa 40% da nota.", "O aviso está visível: " + AVISO])

    def page(c, doc):
        if doc.page > 1:
            c.setFont("Arial", 8.5)
            c.setFillColor(colors.HexColor(R.DIM))
            c.drawString(2.2 * cm, 1.3 * cm, "LaringoPDI · Apostila da apresentação da Etapa 2")
            c.drawRightString(A4[0] - 2.2 * cm, 1.3 * cm, str(doc.page))

    out.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(str(out), pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm, topMargin=2 * cm, bottomMargin=2 * cm, title="LaringoPDI: apostila da Etapa 2", author="Equipe LaringoPDI").build(story, onFirstPage=page, onLaterPages=page)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    build(ROOT / a.out if not Path(a.out).is_absolute() else Path(a.out))
    print(a.out)

"""Apostila da apresentação final: roteiro, falas, perguntas prováveis e cola de números. Os números saem de results/tables.

Uso: python tools/apostila.py            # entrega/LaringoPDI_Apostila.pdf (fora do git, junto com o material da apresentação)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer  # noqa: E402

from tools.documento_grupo import DIM, HEAD, INK, S, W, table  # noqa: E402

OUT = ROOT / "entrega" / "LaringoPDI_Apostila.pdf"
TAB = ROOT / "results" / "tables"
AVISO = "Ferramenta de apoio acadêmico. Não substitui avaliação médica."
FALA = ParagraphStyle("fala", parent=S["body"], leftIndent=9, rightIndent=4, spaceBefore=3, spaceAfter=8, borderPadding=(5, 6, 5, 7), backColor=HEAD)
CAPA = ParagraphStyle("capa", parent=S["sub"], alignment=1)
TIT = ParagraphStyle("tit", parent=S["title"], alignment=1)


def n(v, d=3):
    return f"{float(v):.{d}f}".replace(".", ",")


def pc(v):
    return f"{float(v) * 100:.0f}%"


def numbers() -> dict:
    """Tudo o que as falas citam, lido das tabelas finais."""
    import cv2

    from laringopdi import config as C
    from laringopdi import data as D
    from laringopdi.pdi.registry import to_work_size
    from laringopdi.pdi.segmentation import apply_best, best_config, dice

    pdi = pd.read_csv(TAB / "pdi" / "tabela_unica.csv").set_index("tecnica")
    seg = pd.read_csv(TAB / "pdi" / "segmentacao.csv").set_index(["method", "preprocess"])
    ms = pd.read_csv(TAB / "mlp" / "segmentacao.csv").set_index("tecnica")
    mc = pd.read_csv(TAB / "mlp" / "classificacao.csv")
    cnn = pd.read_csv(TAB / "cnn" / "classificacao.csv").set_index(["tarefa", "tecnica"])
    m18 = mc[mc["modelo"] == "mlp18"].set_index(["tarefa", "tecnica"])
    pix = mc[mc["modelo"] == "mlp_pixels"].set_index("tarefa")
    att = pd.read_csv(TAB / "cnn" / "atencao.csv").set_index(["task", "tecnica"])
    fase = pd.read_csv(TAB / "custo" / "por_fase.csv").set_index("fase")
    hyp = json.loads((TAB / "sintese" / "hipoteses.json").read_text(encoding="utf-8"))
    conf = json.loads((TAB / "sintese" / "confusao_formato.json").read_text(encoding="utf-8"))
    sb_p = TAB / "sintese" / "sem_bordas.csv"
    sb = pd.read_csv(sb_p).set_index(["modelo", "tarefa"]) if sb_p.exists() else None
    imp = pd.read_csv(TAB / "mlp" / "importancia_familias.csv").set_index(["tarefa", "familia"])["queda_media"]
    cfg = C.load()
    raw = C.path(cfg, "raw") / "test"
    g = to_work_size(D.imread(raw / "1881.png", gray=True), cfg["data"]["work_size"])
    gt = cv2.resize((D.imread(raw / "1881_seg.png", gray=True) > 127).astype("uint8"), g.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
    bc = best_config()
    ex = {p: dice(apply_best(g, bc["method"], bc["params"], p), gt) for p in ("raw", "clahe")}
    v = pd.concat([cnn.reset_index()[["veredito_f1", "veredito_sensibilidade"]], m18.reset_index()[["veredito_f1", "veredito_sensibilidade"]]]).melt()["value"]
    return {"pdi": pdi, "seg": seg, "ms": ms, "cnn": cnn, "m18": m18, "pix": pix, "att": att, "fase": fase, "hyp": hyp, "conf": conf, "sb": sb, "imp": imp, "ex": ex,
            "neutros": int((v == "neutra").sum()), "vereditos": int(v.isin(["neutra", "melhora", "piora"]).sum())}


def build() -> None:
    d = numbers()
    pdi, seg, ms, cnn, m18, pix, att, fase, hyp, conf, sb, imp, ex = (d[k] for k in ("pdi", "seg", "ms", "cnn", "m18", "pix", "att", "fase", "hyp", "conf", "sb", "imp", "ex"))
    rg, rg0 = n(pdi.loc["clahe", "dice"]), n(pdi.loc["raw", "dice"])
    story: list = []
    P = lambda t, s="body": story.append(Paragraph(t, S[s]))  # noqa: E731
    B = lambda items: [story.append(Paragraph(i, S["bullet"], bulletText="•")) for i in items]  # noqa: E731
    Fala = lambda t: story.append(Paragraph(f"<b>Fale:</b> {t}", FALA))  # noqa: E731

    def T_(rows, widths):
        story.extend([table(rows, widths, tint=True), Spacer(1, 9)])

    h = lambda k: hyp[k]["veredito"]  # noqa: E731
    h4 = hyp["H4"]
    if sb is not None:
        cb, mb = sb.loc[("resnet18", "B")], sb.loc[("mlp18", "B")]
        ca = sb.loc[("resnet18", "A")]
        txt_sb = (f"Para medir isso, recortamos as bordas pretas e treinamos de novo. A MLP piorou na Tarefa B: a sensibilidade caiu de {n(mb['sensibilidade_com_bordas'], 2)} "
                  f"para {n(mb['sensibilidade_sem_bordas'], 2)}. A CNN quase não mudou: AUC de {n(cb['auc_com_bordas'], 2)} para {n(cb['auc_sem_bordas'], 2)} na B e de "
                  f"{n(ca['auc_com_bordas'], 2)} para {n(ca['auc_sem_bordas'], 2)} na A. A causa está nos dados: uma das câmeras tem 65% de vídeos alterados, as outras, 30%.")
    else:
        txt_sb = "O controle sem bordas ainda não terminou: não citem números dele."

    # ------------------------------------------------------------- capa
    story += [Spacer(1, 5 * cm), Paragraph("LaringoPDI", TIT), Spacer(1, 10), Paragraph("Apostila da apresentação final", CAPA), Paragraph("15 minutos, com demonstração do protótipo", CAPA), Spacer(1, 1.6 * cm)]
    for nome in ("Nathan Malheiros", "Tales Rios", "Pedro Nobre", "Marcos Vinicius", "Marcos Mendes"):
        story.append(Paragraph(nome, CAPA))
    story += [Spacer(1, 1.6 * cm), Paragraph("UNIFOR · Prof.ª Lyndainês Santos", CAPA), Paragraph(AVISO, CAPA), PageBreak()]

    # ------------------------------------------------------------- 1
    P("1. O projeto em 1 minuto", "h1")
    P("Todos precisam saber explicar o projeto sem slide. Decorem a ideia, não as palavras. Falem no passado: \"segmentamos\", \"treinamos\", \"medimos\". "
      "O documento explicativo (<i>docs/Documento_explicativo_LaringoPDI.pdf</i>) tem todos os detalhes; esta apostila é só para a apresentação.")
    B(["<b>O problema.</b> Professores adoecem da voz, e o exame que mostra as pregas vocais é lido a olho, numa imagem com reflexo, pouco contraste e quadros escuros.",
       "<b>O que fizemos.</b> Aplicamos 9 versões da imagem (8 técnicas e a original) em 587 vídeos e medimos o efeito em quatro lugares: na própria imagem, na segmentação da glote, "
       "numa MLP e numa CNN (ResNet-18). Tudo com o mesmo protocolo e com tempo e custo medidos.",
       f"<b>O que descobrimos.</b> Na segmentação, o PDI faz muita diferença: CLAHE com crescimento de região levou o Dice de {rg0} para {rg}. Na classificação, quase não muda: "
       f"{d['neutros']} de {d['vereditos']} vereditos neutros. A MLP simples empatou com a CNN em F1, a uma fração do custo.",
       "<b>A mensagem honesta.</b> Melhorar a imagem aos nossos olhos não garantiu ganho para o modelo, e o melhor PDI depende do algoritmo. Resultado negativo é resultado."])
    P("<b>Três frases para repetir:</b> \"O PDI é o objeto de estudo.\" \"O melhor PDI depende de quem vai usar a imagem.\" \"É apoio à triagem, não diagnóstico.\"")

    # ------------------------------------------------------------- 2
    P("2. Roteiro geral: 15 minutos", "h1")
    T_([["Bloco", "Assunto", "Quem", "Tempo"], ["1", "Equipe, problema social e público-alvo", "Tales", "1 min 30 s"], ["–", "Depoimento (se autorizado)", "Marcos Mendes", "40 s"],
        ["2", "Dados e cuidado contra vazamento", "Marcos Vinicius", "1 min"], ["3", "Objetivos, fluxo do estudo e ferramentas", "Nathan", "1 min 30 s"],
        ["4", "Demonstração: realce e filtragem (abas 1 e 2)", "Marcos Mendes", "2 min 30 s"], ["5", "Demonstração: segmentação clássica e MLP (aba 3)", "Pedro", "2 min"],
        ["6", "Demonstração: PDI + MLP × CNN (aba 4)", "Pedro", "2 min"], ["7", "Resultados, custo e hipóteses (aba 5)", "Nathan", "1 min 30 s"],
        ["8", "Limitações e contribuição", "Tales", "1 min 10 s"], ["9", "Mudanças, conclusões e melhorias", "Nathan", "50 s"], ["", "<b>Total</b>", "", "<b>~14 min 50 s</b>"]], [.09, .57, .19, .15])
    P("A divisão segue a da Etapa 1 e pode ser ajustada. O depoimento só entra com autorização. Quem não está falando opera o computador.")

    # ------------------------------------------------------------- 3
    P("3. Falas de cada bloco", "h1")
    P("Bloco 1. Tales: problema e público (1 min 30 s)", "h2")
    Fala("Somos a equipe do LaringoPDI. O distúrbio de voz é a principal causa de afastamento por saúde entre professores, 17,7%, e o Ministério da Saúde o reconhece como doença do "
         "trabalho. Quem se beneficia são os professores; quem usa a ferramenta é o profissional que lê a laringoscopia. O exame é lido a olho, numa imagem difícil. Nossa pergunta foi: "
         "processar essa imagem antes ajuda, e a que custo?")
    P("Bloco 2. Marcos Vinicius: dados (1 min)", "h2")
    Fala("Usamos o BAGLS: 59.250 frames de 640 vídeos, de vários hospitais, com a máscara da glote feita por especialistas. Ficamos com até 10 frames por vídeo, em 587 vídeos com "
         "rótulo: 5.870 frames. Frames do mesmo vídeo são quase iguais; por isso dividimos os dados por vídeo, em cinco rodadas, e o teste nunca influenciou nenhuma escolha. "
         "Duas tarefas: saudável × alterado, e saudável × lesão estrutural.")
    P("Bloco 3. Nathan: objetivos e fluxo (1 min 30 s)", "h2")
    Fala("O PDI é a variável do estudo; os modelos são só o instrumento de medida, sempre iguais. Fizemos seis etapas: a técnica na própria imagem; a segmentação clássica da glote; "
         "uma MLP que segmenta pixel a pixel; uma MLP com 18 medidas da glote; a ResNet-18; e a síntese de resultado contra custo. Cada técnica recebeu veredito de melhora, neutra "
         "ou piora contra a imagem original, com testes estatísticos corrigidos. Tudo em Python, com OpenCV, scikit-learn, PyTorch e Streamlit, numa RTX 3060.")
    P("Bloco 4. Marcos Mendes: realce e filtragem (2 min 30 s)", "h2")
    P("<b>Faça:</b> abra o app no frame 1881, aba 1; depois a aba 2, com sal e pimenta a 5%, e mova o σ de Fourier.")
    Fala(f"Este frame é escuro. O alargamento estica o histograma, a equalização global dá o maior contraste e o CLAHE trabalha por blocos. No conjunto, o CLAHE foi a única "
         f"técnica a melhorar a imagem pelos três critérios: mais contraste, entropia de {n(pdi.loc['raw', 'entropia'], 2)} para {n(pdi.loc['clahe', 'entropia'], 2)} bits e saturação "
         f"de {pc(pdi.loc['raw', 'saturacao'])} para {pc(pdi.loc['clahe', 'saturacao'])}. A equalização aumenta o contraste mas perde informação.")
    Fala(f"Na filtragem, a mediana venceu a média em sal e pimenta: {n(pdi.loc['median', 'psnr_sal_pimenta5'], 1)} dB contra {n(pdi.loc['mean', 'psnr_sal_pimenta5'], 1)}. "
         "Em Fourier, um σ pequeno corta mais frequência e borra mais a borda das pregas.")
    P("Bloco 5. Pedro: segmentação (2 min)", "h2")
    P("<b>Faça:</b> aba 3. Passo a passo, comparação dos métodos e, embaixo, a MLP de segmentação.")
    Fala(f"Aqui está o passo a passo prometido: limiar adaptativo, abertura, fechamento e componente central. No conjunto, o melhor foi crescimento de região com CLAHE: Dice "
         f"{rg} contra {rg0} sem PDI, e a falha caiu de {pc(pdi.loc['raw', 'taxa_falha'])} para {pc(pdi.loc['clahe', 'taxa_falha'])}, por menos de meio milissegundo a mais. "
         f"Neste frame, o Dice vai de {n(ex['raw'], 2)} para {n(ex['clahe'], 2)}. Mas o efeito depende do método: com a adaptativa e morfologia, o CLAHE derrubou o Dice de "
         f"{n(seg.loc[('adaptive_morph', 'raw'), 'dice_mean'])} para {n(seg.loc[('adaptive_morph', 'clahe'), 'dice_mean'])}.")
    Fala(f"Embaixo está a MLP de segmentação, que decide cada pixel olhando a vizinhança. Para ela, o CLAHE não mudou nada ({n(ms.loc['clahe', 'dice'])}); a melhor foi a homomórfica, "
         f"{n(ms.loc['homo', 'dice'])} contra {n(ms.loc['raw', 'dice'])}. Ela não superou o melhor clássico. A lição: o melhor PDI depende do algoritmo seguinte.")
    P("Bloco 6. Pedro: PDI + MLP × CNN (2 min)", "h2")
    P("<b>Faça:</b> aba 4, com CLAHE na barra lateral. Mostre as 18 medidas, a saída da MLP e o Grad-CAM. Troque para \"sem PDI\" e mostre que pouco muda.")
    Fala(f"No braço 1, estas são as 18 medidas de forma, intensidade e textura, que entram numa MLP. No braço 2, a ResNet-18 decide sozinha, e o Grad-CAM mostra onde ela olhou. "
         f"A sensibilidade é a nossa métrica prioritária. Na Tarefa B, sem PDI, a MLP encontrou mais casos: {n(h4['sens_mlp'], 2)} contra {n(h4['sens_cnn'], 2)}, e o F1 ficou "
         f"empatado, {n(h4['f1_mlp'])} contra {n(h4['f1_cnn'])}. Mas, como o Tales vai mostrar, essa vantagem da MLP vinha em parte do formato do vídeo. "
         "A CNN custou 8,6 horas de GPU; o estudo inteiro da MLP de classificação, uns 17 minutos.")
    P("Bloco 7. Nathan: resultados (1 min 30 s)", "h2")
    P("<b>Faça:</b> aba 5. Mostre a frase-resumo, a tabela por técnica e a das hipóteses.")
    Fala(f"Na classificação, o PDI quase não mudou nada: {d['neutros']} de {d['vereditos']} vereditos neutros, uma melhora e uma piora, as duas do alargamento e só em F1. "
         f"Nenhuma técnica fez o treino convergir mais cedo. As mais caras, passa-baixa e homomórfica, não trouxeram ganho. Hipóteses: H1 {h('H1')}, H2 {h('H2')}, H3 {h('H3')}, "
         f"H4 {h('H4')}, H5 {h('H5')} e H6 {h('H6')}: a textura pesou mais que a forma. Recomendação: CLAHE para segmentar com método clássico; para classificar frames isolados, "
         "o PDI não se justifica pelo ganho.")
    P("Bloco 8. Tales: limitações e contribuição (1 min 10 s)", "h2")
    Fala(f"A principal limitação: o formato do vídeo, que muda com o hospital e o equipamento, sozinho já separa as classes, com AUC de {n(conf['auc_formato_B'], 2)} na Tarefa B. "
         f"Parte do que os modelos aprenderam pode ser o equipamento, e não a voz. {txt_sb} Além disso, usamos imagens paradas, e distúrbios funcionais dependem da vibração; há só "
         "54 vídeos de lesão estrutural; e não houve validação clínica. A contribuição é mostrar, com medida, o que cada técnica faz e quando ela realmente ajuda.")
    P("Bloco 9. Nathan: mudanças e conclusão (50 s)", "h2")
    Fala("Em relação à Etapa 1, o método final de segmentação foi o crescimento de região, previsto como técnica externa; acrescentamos a MLP de segmentação e medimos tempo, custo "
         "e convergência de tudo. Nossa conclusão: o PDI fez diferença para quem segmenta com regras fixas, não para as redes que classificam. Como melhorias: separar por "
         "hospital, usar a sequência de frames e mais vídeos de lesão. O PDI é o objeto de estudo. Obrigado.")

    # ------------------------------------------------------------- 4
    P("4. Perguntas prováveis e respostas", "h1")
    P("Respostas curtas. A pessoa indicada responde primeiro; qualquer um pode completar.")
    a_c, a_r = att.loc[("A", "clahe")], att.loc[("A", "raw")]
    T_([["Pergunta", "Resposta", "Quem"],
        ["Por que o PDI não ajudou a classificação?", "Não sabemos ao certo. A rede pré-treinada lida bem com a imagem crua, e boa parte da decisão parece vir da textura e do formato do vídeo, que o PDI pouco muda. Medimos o efeito, não a causa.", "Nathan"],
        ["O CLAHE é melhor que a equalização (H1)?", f"Depende do método. No crescimento de região, {n(hyp['H1']['dice_region_grow_clahe'])} × {n(hyp['H1']['dice_region_grow_he'])}; mas o CLAHE vence em {hyp['H1']['metodos_clahe_melhor']} de {hyp['H1']['metodos']} métodos e a equalização em {hyp['H1']['metodos_he_melhor']}. Por isso, parcial.", "M. Mendes"],
        ["A MLP empata com a CNN?", f"Em F1, sim: o intervalo da diferença vai de {n(h4['ic_f1'][0])} a {n(h4['ic_f1'][1])} e contém o zero. A sensibilidade maior da MLP na B não se sustentou sem as bordas pretas: as duas ficaram perto de 0,50.", "Pedro"],
        ["O que é esse teste sem bordas?", "Recortamos o quadrado central de cada frame, para tirar as bordas e a proporção do vídeo, que dependem da câmera, e treinamos de novo com o mesmo protocolo. A CNN se manteve, e a MLP caiu na Tarefa B.", "Nathan"],
        ["Por que a sensibilidade é baixa na Tarefa A?", f"A Tarefa A inclui distúrbios funcionais, que aparecem no movimento, e usamos imagens paradas. Já esperávamos: a sensibilidade da CNN ficou entre {n(cnn.xs('A')['sensibilidade'].min(), 2)} e {n(cnn.xs('A')['sensibilidade'].max(), 2)}.", "Nathan"],
        ["E a homomórfica, que teve os maiores valores?", f"Na Tarefa A, sensibilidade {n(cnn.loc[('A', 'homo'), 'sensibilidade'], 2)} contra {n(cnn.loc[('A', 'raw'), 'sensibilidade'], 2)}, mas os testes corrigidos não confirmam. Vale testar de novo; não é resultado comprovado.", "Nathan"],
        ["O que significa a atenção do Grad-CAM?", f"É a parte do mapa dentro da glote. Dividida pela área, dá {n(a_r['razao_att_area'], 2)} sem PDI: a rede olha a glote um pouco mais que o acaso, mas a maior parte da atenção fica fora. Com CLAHE, {n(a_c['razao_att_area'], 2)}.", "Pedro"],
        ["Por que a forma não foi a mais importante (H6)?", f"Na Tarefa B, a textura derrubou {n(imp.loc[('B', 'tex')])} da acurácia, a forma {n(imp.loc[('B', 'shape')])}. A textura também se correlaciona com o formato do vídeo, então pode estar medindo o equipamento.", "Pedro"],
        ["Vocês acharam algum erro?", "Sim, e corrigimos antes destes números: a máscara da MLP estava vazia em boa parte dos frames, e a medida de tempo incluía a carga das bibliotecas. Está tudo registrado.", "Nathan"],
        ["O que é McNemar? E Holm?", "McNemar compara os erros de dois modelos nos mesmos vídeos. Holm corrige o p quando há muitas comparações, para não contar sorte como resultado.", "Nathan"],
        ["Quanto custa?", f"CLAHE {n(pdi.loc['clahe', 'ms_tecnica'], 2)} ms por frame; homomórfica {n(pdi.loc['homo', 'ms_tecnica'], 1)} ms. A CNN levou {n(fase.loc['CNN: 270 treinos', 'duracao_h'], 1)} h e {n(fase.loc['CNN: 270 treinos', 'gpu_energia_wh'] / 1000, 2)} kWh na GPU; o estudo inteiro, {n(fase['duracao_h'].sum(), 1)} h.", "Nathan"],
        ["Isso é PDI ou IA?", "PDI. O objeto de estudo é o efeito de cada técnica; os modelos são o instrumento de medida, sempre iguais.", "Nathan"],
        ["Funciona em imagem de outra origem?", "O app aceita upload, mas tudo foi calibrado no BAGLS, e o resultado pode ser pior.", "Pedro"],
        ["É diagnóstico?", AVISO, "Qualquer um"]], [.27, .59, .14])
    P("<b>Se não souber a resposta:</b> \"Não medimos isso, professora.\" Nunca invente um número.")

    # ------------------------------------------------------------- 5
    P("5. Cola rápida", "h1")
    P("Todos devem saber os números em negrito.")
    T_([["Número", "O que é"], ["<b>17,7%</b>", "afastamento de professores por distúrbio de voz"], ["<b>59.250 / 640</b>", "frames / vídeos do BAGLS"],
        ["<b>5.870 / 587</b>", "frames e vídeos usados"], ["9", "versões da imagem: 8 técnicas + original"],
        [f"<b>{rg} × {rg0}</b>", "Dice: crescimento de região com CLAHE × sem PDI (melhor do estudo)"], [f"{n(ex['raw'], 2)} → {n(ex['clahe'], 2)}", "Dice do frame 1881, sem PDI → CLAHE (exemplo de 1 imagem)"],
        [f"{n(ms.loc['homo', 'dice'])} × {n(ms.loc['raw', 'dice'])}", "Dice da MLP de segmentação: homomórfica × sem PDI"],
        [f"<b>{d['neutros']} de {d['vereditos']}</b>", "vereditos neutros na classificação"],
        [f"<b>{n(h4['f1_mlp'])} × {n(h4['f1_cnn'])}</b>", "F1 na Tarefa B, sem PDI: MLP × ResNet-18"], [f"<b>{n(h4['sens_mlp'], 2)} × {n(h4['sens_cnn'], 2)}</b>", "sensibilidade na Tarefa B: MLP × ResNet-18"],
        [f"{n(cnn.loc[('A', 'raw'), 'sensibilidade'], 2)} × {n(m18.loc[('A', 'raw'), 'sensibilidade'], 2)}", "sensibilidade na Tarefa A: ResNet-18 × MLP"],
        [f"<b>{n(fase.loc['CNN: 270 treinos', 'duracao_h'], 1)} h</b>", "270 treinos da ResNet-18 na RTX 3060"], [f"<b>{n(conf['auc_formato_B'], 2)}</b>", "AUC do formato do vídeo sozinho, Tarefa B"],
        ["H1 a H6", f"{h('H1')}; {h('H2')}; {h('H3')}; {h('H4')}; {h('H5')}; {h('H6')}"]], [.26, .74])

    # ------------------------------------------------------------- 6
    P("6. Checklist antes de apresentar", "h1")
    B(["Todos leram esta apostila e o documento explicativo.", "Um ensaio completo com cronômetro; o limite é de 15 minutos.",
       "Marcos Mendes pediu autorização e ensaiou o depoimento, ou o bloco saiu do roteiro.",
       "O app foi aberto antes (<i>tools/abrir_app.bat</i>), porque a primeira carga é lenta. Frame 1881 selecionado, CLAHE na barra lateral.",
       "O código e o material foram enviados ao AVA no prazo: atraso custa 40% da nota.", "O aviso está visível: " + AVISO])

    def page(c, doc):
        if doc.page > 1:
            c.setFont("A", 8.5)
            c.setFillColor(DIM)
            c.drawString(2.2 * cm, 1.3 * cm, "LaringoPDI · Apostila da apresentação final")
            c.drawRightString(A4[0] - 2.2 * cm, 1.3 * cm, str(doc.page))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
                      title="LaringoPDI: apostila da apresentação final", author="Equipe LaringoPDI").build(story, onFirstPage=page, onLaterPages=page)
    print(OUT, round(OUT.stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    _ = (INK, W)
    build()

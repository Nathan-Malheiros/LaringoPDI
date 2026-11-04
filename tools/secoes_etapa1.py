"""Seções do fechamento da Etapa 1, compartilhadas pelos dois PDFs. Todos os números vêm das tabelas."""
from __future__ import annotations

import json
import re
from html import escape
from pathlib import Path

import pandas as pd


def _n(v: float, d: int = 3) -> str:
    return f"{v:.{d}f}".replace(".", ",")


def _pc(v: float, d: int = 1) -> str:
    return f"{v:.{d}%}".replace(".", ",")


def _md(text: str) -> str:
    """Markdown de uma linha → marcação do reportlab, com o texto escapado."""
    t = escape(text.strip(), quote=False)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    return re.sub(r"`(.+?)`", r"<i>\1</i>", t)


def md_tables(path: Path) -> list[list[list[str]]]:
    """Tabelas de um arquivo Markdown, sem a linha separadora."""
    tables, cur = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("|"):
            cells = [c for c in line.strip().strip("|").split("|")]
            if not all(set(c.strip()) <= {"-", ":"} for c in cells):
                cur.append([_md(c) for c in cells])
        elif cur:
            tables.append(cur)
            cur = []
    if cur:
        tables.append(cur)
    return tables


def secoes(h1, P, table, fig, root: Path, start: int) -> None:
    """h1(numero, titulo); P(texto, estilo="p"); table(linhas, larguras); fig(caminho, legenda, largura, private)."""
    T = root / "results" / "tables"
    F = root / "docs" / "figures"
    R = F / "relatorio"
    # Este registro interno já usa H1–H6 para as hipóteses do desenho; aqui as da apostila seguem como P1–P6 (mesma ordem).
    hyp = {k.replace("H", "P"): v for k, v in json.loads((T / "hypotheses_etapa1.json").read_text(encoding="utf-8")).items()}
    tag = lambda k: f"<b>{hyp[k]['verdict']}</b>"

    # ---------------------------------------------------------------- técnicas de aula
    h1(start, "Técnicas de aula: realce e ruído (P1, P2)")
    d = pd.read_csv(T / "denoise_summary.csv").set_index(["noise", "filter"])
    p2 = pd.read_csv(T / "denoise_p2.csv")
    q = pd.read_csv(T / "enhance_quality.csv").set_index("technique")
    s = pd.read_csv(T / "seg_p1.csv")
    c = s[s["comparison"] == "clahe_vs_he"].set_index("method")
    P("<b>Ruído.</b> Em 300 frames, com no máximo um por vídeo, aplicamos quatro ruídos sintéticos e comparamos os filtros vistos em aula. "
      f"Em sal e pimenta a 5%, a mediana 3×3 chega a {_n(d.loc[('sp_5', 'median3'), 'psnr_mean'], 1)} dB e SSIM {_n(d.loc[('sp_5', 'median3'), 'ssim_mean'], 2)}, "
      f"contra {_n(d.loc[('sp_5', 'mean3'), 'psnr_mean'], 1)} dB e {_n(d.loc[('sp_5', 'mean3'), 'ssim_mean'], 2)} da média 3×3; ela também preserva a borda "
      f"(F1 do Canny {_n(d.loc[('sp_5', 'median3'), 'edge_f1_mean'], 2)} × {_n(d.loc[('sp_5', 'mean3'), 'edge_f1_mean'], 2)}). P2: {tag('P2')}. "
      f"No ruído gaussiano a mediana 3×3 não perdeu da média 3×3, ao contrário do esperado (σ=10: {_n(d.loc[('gauss_s10', 'median3'), 'psnr_mean'], 1)} × "
      f"{_n(d.loc[('gauss_s10', 'mean3'), 'psnr_mean'], 1)} dB). Só o passa-baixa σ40 superou a mediana 3×3, e só em σ=20. "
      "Limitação: a referência é o próprio frame do BAGLS, que já tem ruído; o PSNR mede fidelidade ao original, não a um sinal verdadeiro.")
    fig(R / "denoise.png", "PSNR e SSIM por filtro, em cada ruído sintético (média de 300 frames).", 1.0, False)
    fig(F / "denoise_exemplo.png", "Um frame com sal e pimenta a 5%: média, mediana e passa-baixa.", 1.0, True)
    P("<b>Realce.</b> Medimos as métricas nos 5.870 frames e repetimos o estudo de segmentação com alargamento e equalização global (HE), sem alterar o método já fixado. "
      f"O CLAHE supera o HE com significância em {int(((c['delta_b_minus_a'] > 0) & (c['p_holm'] < .05)).sum())} de 5 métodos, inclusive o escolhido "
      f"(<i>region_grow</i>: Dice {_n(c.loc['region_grow', 'dice_b'])} × {_n(c.loc['region_grow', 'dice_a'])}), mas perde nos métodos de limiar com morfologia "
      f"(<i>otsu_morph</i>: {_n(c.loc['otsu_morph', 'dice_b'])} × {_n(c.loc['otsu_morph', 'dice_a'])}; <i>watershed</i>: {_n(c.loc['watershed', 'dice_b'])} × {_n(c.loc['watershed', 'dice_a'])}). "
      f"P1: {tag('P1')}. O efeito do realce depende do método. "
      f"Ressalva: a fração saturada de HE ({_pc(q.loc['he', 'saturated_frac_mean'])}) é maior que a de CLAHE ({_pc(q.loc['clahe', 'saturated_frac_mean'])}), mas o frame sem realce já tem "
      f"{_pc(q.loc['raw', 'saturated_frac_mean'])}, quase todo preto do fundo. O HE pouco acrescenta, e o CLAHE fica baixo porque tira esses pixels do zero. "
      "A métrica não demonstra, portanto, o \"estouro\" do tecido que a hipótese supunha, embora ele seja visível na imagem.")
    fig(R / "realce.png", "Entropia, contraste e fração saturada por técnica (5.870 frames) e Dice por método × realce (teste, 5 folds).", 1.0, False)
    fig(F / "realce_exemplo.png", "Um frame escuro e os histogramas após cada realce.", .9, True)

    # ---------------------------------------------------------------- MLP-18
    h1(start + 1, "MLP-18 (objetivo específico 4, P4, P6)")
    p4 = pd.read_csv(T / "mlp18_p4.csv").set_index(["system", "task"])
    ms = json.loads((T / "mlp18_summary.json").read_text(encoding="utf-8"))
    imp = pd.read_csv(T / "importance_c6.csv")
    b = p4.loc[("C6", "B")]
    P("O C6 cumpre o que a Etapa 1 prometeu: 18 características (6 de forma, 6 de intensidade do tecido e 6 de textura), média por vídeo e MLP 18–32–16–2, nos mesmos folds dos demais experimentos. "
      f"A máscara clássica fica vazia em {_pc(ms['empty_mask_frame_rate'], 2)} dos frames. "
      f"Na Tarefa B, o F1 é {_n(b['f1_classic'])} contra {_n(b['f1_resnet18'])} da ResNet-18 (semente 0), com IC 95% do ΔF1 de {_n(b['ci_low'])} a {_n(b['ci_high'])}. P4: {tag('P4')} pela regra, "
      f"mas <b>a confirmação é frágil</b>: o IC é quase todo negativo, e a acurácia ({_n(b['acc_classic'])} × {_n(b['acc_resnet18'])}) e o McNemar favorecem a ResNet-18. "
      f"\"Competitiva\" aqui não significa equivalente. O C1, com 187 características, empata de fato ({_n(p4.loc[('C1', 'B'), 'f1_classic'])}). "
      f"P6: {tag('P6')}: a família forma não foi a mais importante em nenhuma das duas tarefas; a textura foi a primeira.")
    rows = [["Sistema", "Tarefa", "F1 clássico", "F1 ResNet-18", "ΔF1 [IC 95%]", "Acurácia (clássico × ResNet-18)", "McNemar p"]]
    for (sy, t), r in p4.iterrows():
        rows.append([sy, t, _n(r["f1_classic"]), _n(r["f1_resnet18"]), f"{_n(r['delta'])} [{_n(r['ci_low'])}; {_n(r['ci_high'])}]",
                     f"{_n(r['acc_classic'])} × {_n(r['acc_resnet18'])}", "&lt; 0,001" if r["mcnemar_p"] < .001 else _n(r["mcnemar_p"])])
    table(rows, [.1, .08, .12, .13, .25, .2, .12])
    fig(R / "importancia_c6.png", "Queda da acurácia balanceada ao permutar cada família do C6 (média e desvio entre os 5 folds).", .85, False)

    # ---------------------------------------------------------------- atenção
    h1(start + 2, "Atenção na glote (P5)")
    a = pd.read_csv(T / "attention_p5.csv")
    n = a[a["family"] == "sem_recorte"]
    num = hyp["P5"]["numbers"]
    e = n[n["model"] == "resnet18"].groupby("task")["ratio_E00"].mean()
    P("A métrica é a razão entre a fração da energia do Grad-CAM na glote e a fração de área da glote: 1 é o nível do acaso. Foi calculada por vídeo, sem rodar o Grad-CAM de novo. "
      f"Das {num['n']} combinações sem recorte, {num['up']} sobem e {num['down']} caem com significância. P5: {tag('P5')}; "
      "o veredito segue a regra, mas o sinal do efeito muda com técnica, modelo e tarefa, e mais combinações caem do que sobem. "
      f"Na ResNet-18, o E00 já olha para a glote acima do acaso (razão {_n(e['A'], 2)} na Tarefa A e {_n(e['B'], 2)} na B). "
      f"Desvio declarado: frames com Grad-CAM sem energia, frequentes na SimpleCNN, saíram da análise principal depois de vistos os resultados; contá-los como atenção 0 dá {num['sens_up']} subidas e {num['sens_down']} quedas, com o mesmo veredito.")
    fig(R / "atencao_p5.png", "Diferença pareada da razão de atenção em relação ao E00; * indica p_Holm &lt; 0,05.", .95, False)

    # ---------------------------------------------------------------- tempo
    h1(start + 3, "Tempo de processamento")
    t = pd.read_csv(T / "timing.csv", encoding="utf-8")
    P(f"Mediana por frame em 100 frames, numa CPU ({escape(str(t['cpu'].iloc[0]))}; PyTorch com {int(t['torch_threads'].iloc[0])} threads), num processo só, descartando 5 aquecimentos. "
      "Todo o caminho (pipeline, segmentação, características e inferência) fica bem abaixo de um décimo de segundo por frame, o que viabiliza a demonstração ao vivo sem GPU.")
    pipes = t[t["kind"] == "pipeline"]
    sel = t[t["name"].isin(["stretch", "he", "clahe", "mean", "median", "gauss_lpf", "homo", "bilat"]) & (t["kind"] == "técnica")]
    rows = [["Etapa", "Mediana (ms)"], *[[f"Técnica: {r['name']}", _n(r["median_ms"], 1)] for _, r in sel.iterrows()],
            ["Pipelines E00–E14 (mínimo a máximo)", f"{_n(pipes['median_ms'].min(), 1)} a {_n(pipes['median_ms'].max(), 1)}"],
            *[[r["name"], _n(r["median_ms"], 1)] for _, r in t[t["kind"].isin(["segmentação", "características", "inferência"])].iterrows()]]
    table(rows, [.7, .3])

    # ---------------------------------------------------------------- hipóteses
    h1(start + 4, "Hipóteses da Etapa 1 (P1–P6)")
    P("As regras de veredito foram fixadas em <i>docs/DECISIONS.md</i> (2026-11-04) antes de gerar os números novos. P1, P3, P4 e P6 já tinham evidência parcial conhecida, como aquela entrada registra. "
      "O texto completo, gerado por comando, está em <i>docs/hipoteses_etapa1.md</i>.")
    table([["ID", "Hipótese", "Veredito", "Evidência"], *[[k, escape(r["statement"]), f"<b>{r['verdict']}</b>", _md(r["evidence"])] for k, r in hyp.items()]], [.05, .2, .13, .62])

    # ---------------------------------------------------------------- mudanças
    h1(start + 5, "Mudanças em relação à Etapa 1")
    P("A orientação da Etapa 2 pede que mudanças relevantes em relação à proposta sejam justificadas. Texto integral em <i>docs/MUDANCAS_ETAPA1.md</i>.")
    tbs = md_tables(root / "docs" / "MUDANCAS_ETAPA1.md")
    for tb in tbs:
        k = len(tb[0])
        table(tb, {5: [.15, .16, .25, .27, .17], 4: [.16, .18, .42, .24], 3: [.15, .5, .35], 2: [.35, .65]}[k])

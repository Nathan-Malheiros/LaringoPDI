"""Textos do documento explicativo. As conclusões saem das tabelas finais e são escritas depois de lê-las."""

PENDENTE = "<i>Pendente: aguardando o resultado final.</i>"

T = {
    "capa": "Este documento é para quem entrou agora no assunto. Ele explica o projeto do zero e dá mais espaço aos resultados que à teoria. "
            "<b>Nada aqui é ferramenta de diagnóstico:</b> o trabalho estuda apoio à decisão em um conjunto público de dados.",
    "data_e_versao": "Disciplina de Processamento Digital de Imagens, UNIFOR, Prof.ª Lyndainês Santos. Equipe: Nathan, Marcos Vinicius, Pedro Nobre, Marcos Mendes e Tales. "
                      "Estudo concluído em 6 de novembro de 2026.",
    "resumo": PENDENTE,
    "problema": """
A laringoscopia filma as pregas vocais. A <b>glote</b> é a abertura entre elas, e sua forma e seu movimento ajudam a avaliar a voz. Essas imagens costumam ter iluminação desigual, reflexos e pouco contraste, então é natural perguntar se processá-las antes ajuda um algoritmo.

Na nossa disciplina, o <b>PDI é o protagonista</b>. A rede neural é só o instrumento de medida: arquitetura, sementes, divisão dos dados e treino ficam fixos, e muda apenas a técnica aplicada à imagem. Assim, qualquer diferença pode ser atribuída ao PDI. Resultado negativo também é resultado.

<b>Pergunta de pesquisa:</b> quais técnicas de PDI melhoram, não alteram ou pioram a segmentação da glote e a classificação de distúrbios da voz, e a que custo?

O público beneficiado são profissionais da voz, sobretudo professores, e os usuários seriam profissionais de saúde em triagem.
""",
    "dados": """
Usamos o <b>BAGLS</b> (Gómez et al., 2020; licença CC BY-NC-SA 4.0), com 59.250 frames de 640 vídeos de vários hospitais e máscaras da glote feitas por especialistas. Excluímos os 53 vídeos com status desconhecido e ficamos com <b>587 vídeos</b>: 377 saudáveis, 156 com distúrbio funcional e 54 com lesão estrutural. De cada vídeo tomamos até 10 frames espaçados, num total de <b>5.870 frames</b>.

Há duas tarefas de classificação:

- <b>Tarefa A:</b> saudável × com alteração (a principal).
- <b>Tarefa B:</b> saudável × lesão estrutural (nódulo, pólipo, cisto e outras).

## Como evitamos resultado enganoso

Frames do mesmo vídeo são quase iguais. Se um fosse para o treino e outro para o teste, a nota subiria sem o modelo ter aprendido nada útil. Por isso:

- a divisão é feita <b>por vídeo</b>, em 5 partes (validação cruzada de 5 folds), mantendo a proporção de classes;
- cada fold separa treino, validação (para saber quando parar) e teste, que nunca influencia nenhuma escolha;
- os parâmetros são definidos pela literatura ou escolhidos só no treino;
- a nota principal é <b>por vídeo</b> (média das probabilidades dos frames), porque a decisão real é sobre a pessoa.
""",
    "fluxo": """
Cada etapa usa as mesmas nove versões da imagem, sempre comparadas com a imagem sem PDI (a referência). Os modelos foram treinados no mesmo PC, descrito na seção 12.
""",
    "etapas": [
        ["1. PDI isolado", "A técnica melhora a imagem?", "Contraste, entropia, nitidez, saturação; PSNR e SSIM com ruído; tempo"],
        ["2. Segmentação clássica", "Qual técnica e método encontram melhor a glote?", "Dice, IoU, HD95, falha, tempo"],
        ["3. MLP na segmentação", "A MLP supera os métodos clássicos? O PDI a ajuda?", "Dice, curva e tempo de treino"],
        ["4. MLP na classificação", "O PDI ajuda a identificar o distúrbio com 18 medidas?", "Sensibilidade, especificidade, F1, AUC, custo"],
        ["5. CNN na classificação", "O PDI ajuda a ResNet-18? Para onde ela olha?", "As mesmas + convergência e Grad-CAM"],
        ["6. Síntese", "O ganho compensa o custo?", "Resultado × custo por técnica"],
    ],
    "tecnicas_intro": """
Todas funcionam em escala de cinza, com parâmetros padrão da literatura, nunca escolhidos pela nota do classificador. As duas últimas não foram vistas em aula, estavam previstas na Etapa 1 e aparecem aqui só no essencial.
""",
    "tecnicas": [
        ["Sem PDI", "Referência", "Só redimensiona para 224×224, preservando a proporção, com bordas pretas.", "Linha de base de todas as comparações."],
        ["Alargamento", "Aula", "Estica os níveis de cinza para ocupar toda a faixa.", "Imagens escuras usam só parte da faixa disponível."],
        ["Equalização (HE)", "Aula", "Redistribui os níveis pelo histograma da imagem inteira.", "Aumenta o contraste, mas pode exagerar reflexos e ruído."],
        ["CLAHE", "Aula", "Equaliza por regiões, com limite de contraste (clip 2,0; grade 8×8).", "Lida com a luz desigual do endoscópio sem estourar tanto a imagem."],
        ["Média", "Aula", "Troca cada pixel pela média da vizinhança.", "Reduz ruído, mas borra a borda da glote."],
        ["Mediana", "Aula", "Troca cada pixel pela mediana da vizinhança.", "Remove pontos isolados preservando melhor as bordas."],
        ["Passa-baixa (Fourier)", "Aula", "Atenua altas frequências com filtro gaussiano.", "Suaviza a imagem no domínio da frequência; é a mais lenta entre as de aula."],
        ["Remoção de reflexo", "Literatura", "Detecta os pontos muito claros e preenche por inpainting.", "Reflexos da luz atrapalham a leitura da mucosa."],
        ["Homomórfica", "Literatura", "Filtra na frequência separando iluminação e detalhe.", "Compensa a iluminação irregular."],
    ],
    "metodos_seg": [
        ["Otsu", "Um único limiar para a imagem inteira. Escuro vira glote."],
        ["Adaptativa", "Um limiar para cada vizinhança, tolerando a luz desigual."],
        ["Otsu + morfologia", "Otsu com abertura, fechamento e escolha do componente central."],
        ["Adaptativa + morfologia", "A mesma limpeza, sobre o limiar adaptativo."],
        ["Crescimento de região", "Começa no ponto mais escuro do centro e cresce enquanto o nível de cinza for parecido."],
        ["Watershed", "Trata a imagem como relevo e separa regiões a partir de marcadores."],
    ],
    "metricas_intro": "Estas são as medidas que aparecem nas tabelas, em linguagem simples.",
    "metricas": [
        ["Dice", "Sobreposição entre a máscara prevista e a do especialista, de 0 a 1.", "Alto"],
        ["IoU", "Parecido com o Dice, porém mais rigoroso.", "Alto"],
        ["HD95", "O quanto a borda prevista se afasta da borda real, em pixels.", "Baixo"],
        ["Taxa de falha", "Frames em que o método não devolveu nenhuma máscara.", "Baixa"],
        ["PSNR e SSIM", "Quanto a imagem filtrada se aproxima da original sem ruído.", "Altos"],
        ["Sensibilidade", "Das pessoas com alteração, quantas o modelo encontrou. É a <b>prioritária</b>: numa triagem, deixar passar é o pior erro.", "Alta"],
        ["Especificidade", "Das pessoas saudáveis, quantas foram reconhecidas como saudáveis.", "Alta"],
        ["F1-macro", "Equilíbrio entre acertos nas duas classes, mesmo com classes desiguais.", "Alto"],
        ["AUC", "Capacidade de ordenar alterados acima de saudáveis. 0,5 é o acaso.", "Alta"],
        ["Melhor época", "Em que ponto do treino o modelo atingiu a melhor validação. Mais cedo é melhor.", "Baixa"],
        ["Custo total", "Tempo de PDI nos frames de treino + tempo de treino até a melhor validação.", "Baixo"],
    ],
    "veredito": """
## Como decidimos "melhora", "neutra" ou "piora"

Uma diferença pequena pode ser só sorte, então cada comparação passa por testes:

- <b>Wilcoxon pareado:</b> compara a técnica com a referência no mesmo fold e na mesma semente.
- <b>McNemar:</b> olha os vídeos em que os dois modelos discordaram.
- <b>Bootstrap:</b> reamostra os vídeos para dar um intervalo de confiança de 95% (IC) da diferença.
- <b>Correção de Holm:</b> ajusta os testes, porque testar muitas técnicas aumenta a chance de um acerto casual.

Na classificação, só há "melhora" ou "piora" quando o IC exclui zero <b>e</b> um dos testes corrigidos fica abaixo de 0,05. Em qualquer outro caso, é <b>neutra</b>: não conseguimos distinguir o efeito do acaso, o que não prova que ele seja zero.
""",
    "r_visual_1": """
A pergunta aqui é só sobre a imagem, sem modelo nenhum. Nos realces, consideramos melhora um contraste maior, sem perder entropia e sem aumentar a saturação. Nos filtros, medimos a capacidade de remover ruído artificial.
""",
    "r_visual_2": """
<b>O que isso significa:</b> só o CLAHE melhora a imagem pelos três critérios. O alargamento e a equalização aumentam o contraste, mas perdem entropia. A mediana é claramente melhor que a média em sal e pimenta (PSNR de 39,1 contra 25,9), como esperado, porque ignora valores extremos.

A melhora visual não garante ganho para o modelo. É isso que as etapas seguintes testam.
""",
    "r_seg_1": """
Rodamos os 6 métodos nas 9 versões da imagem e comparamos com a máscara do especialista. Os parâmetros de cada método foram escolhidos só com o treino de cada fold.
""",
    "r_seg_2": """
<b>O que isso significa:</b> a combinação de <b>crescimento de região com CLAHE</b> foi a melhor do estudo, com Dice de 0,614 contra 0,467 sem PDI, a um custo extra de menos de 0,5 ms por frame, e ainda reduziu a falha de quase 20% para 2%. O alargamento também aumenta o Dice (0,484), mas piora a precisão da borda (HD95). A equalização fica neutra. Média, mediana, passa-baixa, reflexo e homomórfica pioram.

O efeito <b>depende do método</b>: no mapa de calor, o CLAHE ajuda o crescimento de região e prejudica os outros métodos. Não existe técnica boa em si; existe técnica boa para um uso.
""",
    "r_mlp_seg_1": """
Aqui uma rede simples (MLP) decide, pixel a pixel, se ele pertence à glote, olhando uma vizinhança de 15×15 (225 valores, rede 225–64–32–2). A MLP de segmentação é um acréscimo em relação à Etapa 1 e foi declarada como tal.
""",
    "r_mlp_seg_2": """
<b>O que isso significa:</b> o PDI ajuda a MLP, mas de forma diferente. As melhores foram <b>homomórfica (0,532)</b>, equalização (0,500) e alargamento (0,481), contra 0,449 sem PDI. O CLAHE, campeão no método clássico, não muda nada aqui (0,451). Isso reforça que o melhor PDI depende do algoritmo seguinte.

A MLP não superou o melhor resultado clássico. Com a mesma técnica, ela só ganhou com homomórfica e passa-baixa, empatou na maioria e perdeu com CLAHE (0,451 contra 0,614). Não comparamos o HD95 entre as duas tabelas, porque MLP e método clássico foram avaliados em resoluções diferentes.
""",
    "r_mlp_1": """
A técnica é aplicada, a máscara é obtida por crescimento de região, e extraímos <b>18 características</b>: 6 de forma, 6 de intensidade e 6 de textura. A média por vídeo entra numa rede 18–32–16–2. Como referência sem nenhum PDI, treinamos também uma MLP sobre a imagem reduzida a 32×32.
""",
    "r_mlp_2": PENDENTE,
    "r_cnn_1": """
A <b>ResNet-18</b> é uma rede convolucional pré-treinada no ImageNet, ajustada aqui para nossos dados. Foram <b>270 treinos</b> (9 técnicas × 2 tarefas × 5 folds × 3 sementes), com o mesmo protocolo em todos: entrada 224, aumento de dados apenas geométrico, AdamW 1e-4, até 30 épocas e parada com paciência 7.
""",
    "r_cnn_2": PENDENTE,
    "r_gradcam_1": """
O <b>Grad-CAM</b> gera um mapa de quais regiões mais pesaram na decisão da rede. Medimos que parcela desse mapa cai dentro da glote, nos frames de teste, usando os pesos da semente 0.
""",
    "r_gradcam_2": PENDENTE,
    "r_custo_1": """
<b>Hardware:</b> AMD Ryzen 5 5500 (12 threads), 15,9 GB de RAM, NVIDIA GeForce RTX 3060 de 12 GB, Windows 11, Python 3.11 e PyTorch 2.14.1 com CUDA 13.0. Tudo foi medido nessa máquina. A MLP de classificação rodou na CPU; a MLP de segmentação e a CNN, na GPU.
""",
    "r_custo_2": PENDENTE,
    "sintese_1": PENDENTE,
    "conclusoes": [["–", PENDENTE]],
    "hipoteses_intro": "As hipóteses foram apresentadas na Etapa 1, e as regras de decisão foram registradas antes destes números finais.",
    "hipoteses": [
        ["H1", "CLAHE > equalização global", PENDENTE], ["H2", "Mediana > média em sal e pimenta", PENDENTE], ["H3", "Limiar adaptativo > Otsu", PENDENTE],
        ["H4", "MLP ≈ CNN na Tarefa B", PENDENTE], ["H5", "PDI aumenta a atenção da CNN na glote", PENDENTE], ["H6", "Forma é a família mais importante", PENDENTE],
    ],
    "limites": PENDENTE,
    "faltando": PENDENTE,
}

from tools.documento_resultados import R  # noqa: E402

T.update(R)

"""Textos escritos depois de ler as tabelas finais (6 de novembro de 2026). Se os resultados mudarem, reveja estes números."""

R = {
    "resumo": """
<b>A pergunta:</b> processar a imagem de laringoscopia antes de um algoritmo ajuda? Testamos 9 versões da imagem em 587 vídeos do BAGLS: na própria imagem, na segmentação da glote e na classificação de distúrbios, com MLP e com a CNN ResNet-18.

## A resposta em quatro pontos

- <b>Na segmentação, o PDI faz muita diferença.</b> O melhor resultado foi crescimento de região com CLAHE: Dice de 0,614 contra 0,467 sem PDI, com falha caindo de 20% para 2%, por menos de 0,5 ms a mais por frame.
- <b>O melhor PDI depende do algoritmo.</b> O CLAHE, campeão no método clássico, não mudou nada na MLP de segmentação, em que a melhor foi a homomórfica (Dice de 0,532 contra 0,449). Melhorar a imagem aos nossos olhos também não garantiu ganho para o modelo.
- <b>Na classificação, o PDI quase não mudou o resultado.</b> Dos 64 vereditos de F1 e sensibilidade, 62 foram neutros, 1 foi melhora (alargamento, CNN, Tarefa A, só em F1) e 1 foi piora (alargamento, MLP, Tarefa B, só em F1). Nenhuma técnica fez o treino convergir significativamente mais cedo.
- <b>A MLP simples empatou com a CNN em F1, por uma fração do custo.</b> Na Tarefa B, sem PDI, a MLP teve F1 de 0,758 contra 0,783, sem diferença clara, e sensibilidade maior (0,705 contra 0,577). O treino da CNN consumiu 8,6 h de GPU; o estudo inteiro da MLP de classificação, cerca de 17 minutos.

## O que recomendamos

Para segmentar a glote com método clássico, use <b>CLAHE com crescimento de região</b>. Para classificar distúrbios a partir de frames isolados, os dados não justificam incluir PDI na entrada só pelo ganho esperado.

<b>Cuidado importante:</b> a classificação pode estar aprendendo em parte o formato do vídeo. Esse formato sozinho já separa as classes com AUC de 0,76 na Tarefa B (seção 15). Por isso, as notas absolutas não devem ser lidas como capacidade clínica. As comparações entre técnicas continuam válidas como comparações, porque esse fator é o mesmo em todas.
""",
    "r_mlp_2": """
<b>O que isso significa:</b> nenhuma técnica mudou a sensibilidade da MLP nas duas tarefas. Em F1, só o alargamento piorou, na Tarefa B (0,729 contra 0,758). A convergência foi igual em todas. Quase todas as técnicas encarecem um pouco o treino, e as mais caras foram passa-baixa e homomórfica, de 2,2 a 2,4 vezes o custo sem PDI.

A correção da máscara (seção 15) melhorou muito as máscaras, mas não mudou a conclusão: o PDI continua sem ganho claro aqui. Um detalhe merece atenção: a técnica também muda a qualidade da máscara. Sem PDI, 24% dos frames ficam sem máscara; com equalização, nenhum. Mesmo assim, a classificação não melhorou.

<b>Com ou sem características:</b> a MLP sobre a imagem reduzida a 32×32, sem nenhum PDI, teve F1 de 0,634 e sensibilidade de 0,540 na Tarefa A, e 0,781 e 0,614 na Tarefa B. Das 36 comparações com a MLP de características, 34 foram neutras, e 2 favoreceram as características, ambas em sensibilidade na Tarefa B.

<b>Quais características pesam mais:</b> a textura, nas duas tarefas. A forma da glote, que esperávamos liderar, ficou por último. A seção 15 explica por que esse resultado deve ser lido com cautela.
""",
    "r_cnn_2": """
<b>O que isso significa:</b> a Tarefa A ficou modesta, como esperado: F1 de 0,630 a 0,680 e AUC de 0,718 a 0,758. Isso é plausível, porque distúrbios funcionais aparecem no movimento, e aqui cada frame é tratado como imagem parada. A Tarefa B tem F1 maior (0,763 a 0,783) e especificidade alta (0,94 a 0,97), mas sensibilidade baixa (0,545 a 0,590): a rede raramente alarma em falso, e também deixa passar cerca de 4 em cada 10 casos.

O único veredito de melhora foi o <b>alargamento na Tarefa A, em F1</b> (0,664 contra 0,643), sem ganho de sensibilidade. A <b>homomórfica</b> teve os maiores valores da Tarefa A (F1 de 0,680 e sensibilidade de 0,571 contra 0,505). O intervalo da sensibilidade até exclui zero, mas os testes corrigidos não confirmam, então o veredito é neutro. Vale testá-la de novo em outro estudo, sem tratá-la como comprovada.

<b>Treino e convergência:</b> a melhor validação chega cedo, entre as épocas 4 e 8, e a diferença entre treino e validação fica em torno de 0,30 na Tarefa A e 0,15 na Tarefa B. A rede decora o treino depressa. Nenhuma técnica mudou a convergência de forma significativa.
""",
    "r_gradcam_2": """
<b>O que isso significa:</b> a rede dá à glote uma atenção de 1,2 a 1,7 vez o que sua área explicaria. Ela olha para a glote um pouco mais do que o acaso, mas cerca de 99% da energia do mapa fica fora dela. Isso é coerente com a textura ter pesado tanto na MLP e não mostra que a rede decide pela glote.

<b>Duas leituras, porque a medida exige cuidado.</b> A regra registrada usa a atenção direta. Por ela, só a média aumenta (Tarefa B), e 8 combinações diminuem. Mas a parte válida da imagem muda com a técnica, o que altera a parcela da glote mesmo sem a rede mudar. Corrigindo pela área, os resultados mudam: <b>CLAHE aumenta nas duas tarefas</b>, a homomórfica aumenta na A, a média na B, e só a equalização diminui, na B. Como essa correção surgiu depois de vermos os números, tratamos a segunda leitura como análise exploratória.
""",
    "r_custo_2": """
<b>O que isso significa:</b> a CNN responde por quase 80% do tempo e por quase toda a energia medida da GPU, cerca de 1,08 kWh nos 270 treinos, a uma média de 115 s por treino. Cada MLP de classificação leva menos de 1 s até a melhor validação; o maior custo dela é preparar as imagens, cerca de 10 ms por frame para máscara e características.

Entre as técnicas, a diferença de custo aparece principalmente nas duas que usam Fourier (12 a 14 ms por frame). O CLAHE custa só 0,46 ms e a equalização, 0,12 ms. Na CNN, o custo não mudou de forma significativa para a maioria; passa-baixa e homomórfica ficaram mais caras nas duas tarefas e a remoção de reflexo, na B.

<b>Compensa?</b> Na segmentação clássica, o CLAHE compensa com folga. Na classificação, nenhuma técnica mostrou ganho que justifique custo extra. A troca da CNN pela MLP é a maior economia do estudo, sem perda clara em F1.
""",
    "sintese_1": """
A tabela reúne os vereditos, e o quadro seguinte traduz cada linha em uma recomendação. "Neutra" significa que não detectamos diferença, e não que a técnica é inútil em qualquer situação. As colunas de atenção usam a medida corrigida pela área, que é exploratória.
""",
    "conclusoes": [
        ["Sem PDI", "Referência. Com crescimento de região, falha em cerca de 20% dos frames. Serve bem à classificação, que não melhorou com nenhuma técnica."],
        ["Alargamento", "<b>Resultado misto.</b> Aumenta contraste e Dice, mas piora a borda (HD95). Ajuda a MLP de segmentação. Na classificação, melhora o F1 da CNN (A) e piora o da MLP (B). Não é uma escolha segura."],
        ["Equalização", "<b>Útil só na MLP de segmentação</b> (Dice de 0,500). É a mais barata. Neutra no método clássico e na classificação."],
        ["CLAHE", "<b>A recomendada para segmentação clássica</b>: única a melhorar imagem e segmentação, a custo mínimo. Neutra na classificação e na MLP de segmentação."],
        ["Média", "<b>Não recomendada.</b> Reduz ruído gaussiano, mas borra e piora a segmentação clássica. Sem ganho na classificação."],
        ["Mediana", "<b>Só com ruído impulsivo real.</b> É a melhor contra sal e pimenta, mas piora a segmentação clássica e não muda a classificação."],
        ["Passa-baixa", "<b>Não recomendada.</b> Pior Dice clássico (0,419), custo alto e nenhum ganho na classificação."],
        ["Remoção de reflexo", "<b>Não recomendada no fluxo atual.</b> Reduz o reflexo, mas não trouxe ganho em nenhuma etapa e tem custo de 3,4 ms por frame."],
        ["Homomórfica", "<b>Promissora para modelos, mas cara.</b> Melhor na MLP de segmentação (Dice de 0,532), pior que a referência no método clássico. Maiores valores na CNN (A), sem confirmação estatística."],
    ],
    "hipoteses": [
        ["H1", "CLAHE > equalização global", "No crescimento de região, 0,614 contra 0,476 (p &lt; 0,001), e menos saturação (0,015 contra 0,182). Mas o CLAHE vence em só 2 dos 6 métodos, e a equalização em 4."],
        ["H2", "Mediana > média em sal e pimenta", "Filtros 3×3: PSNR de 41,4 contra 25,9 a 5% de ruído, e 39,2 contra 22,5 a 10%. SSIM de 0,97 contra 0,44 e 0,97 contra 0,27."],
        ["H3", "Limiar adaptativo > Otsu", "Dice de 0,253 contra 0,023 sem PDI (587 vídeos, p &lt; 0,001). Os dois ficam bem abaixo do crescimento de região."],
        ["H4", "MLP ≈ CNN na Tarefa B", "F1 de 0,758 contra 0,783, com IC 95% de −0,069 a +0,059. A MLP ainda teve sensibilidade maior (0,705 contra 0,577; IC de +0,074 a +0,315). Ver a ressalva sobre o formato do vídeo."],
        ["H5", "PDI aumenta a atenção da CNN na glote", "Na regra registrada, 1 de 16 combinações aumenta e 8 diminuem. Na análise exploratória corrigida pela área, 4 aumentam e 1 diminui."],
        ["H6", "Forma é a família mais importante", "A textura lidera (queda de 0,089 na A e 0,184 na B). A forma ficou por último (0,017 e −0,018)."],
    ],
    "limites": """
## O formato do vídeo pode estar influenciando a classificação

O BAGLS reúne vários hospitais e equipamentos, e a parte útil da imagem varia entre eles. Verificamos que <b>essa proporção sozinha, sem olhar a laringe, separa as classes com AUC de 0,657 na Tarefa A e 0,763 na Tarefa B</b>. Além disso, características de textura se correlacionam com ela (energia GLCM, ρ = −0,62; Sobel, ρ = 0,54), e a rede também vê as bordas pretas.

Na prática, parte do que os modelos aprenderam pode ser "de que equipamento veio o vídeo", e não "como está a voz". Isso pode explicar a liderança da textura e a boa nota da MLP na Tarefa B. Não medimos quanto da nota vem disso; o próximo passo seria testar sem as bordas ou separar os dados por hospital. A verificação foi feita depois dos resultados e não estava prevista.

## Erros encontrados e corrigidos antes destes números

- <b>Máscara da MLP de classificação:</b> a semente do crescimento de região caía na borda preta, deixando 66% a 87% das máscaras vazias. O erro foi corrigido, com teste automático, e todas as MLPs de classificação foram treinadas de novo. Agora as máscaras vazias vão de 0% a 25%, em linha com a etapa de segmentação.
- <b>Tempo das características:</b> a primeira medição incluía a carga das bibliotecas e fazia a imagem sem PDI parecer mais cara. O tempo foi medido de novo, sem esse efeito, e os vereditos de custo foram refeitos.
- <b>Atenção do Grad-CAM:</b> identificamos que a parcela da glote depende da área válida da imagem. Mantivemos a regra registrada e acrescentamos a leitura corrigida, como análise exploratória.

## Outros limites

- <b>Neutro não é zero.</b> Com 54 vídeos de lesão estrutural, efeitos pequenos podem passar despercebidos.
- <b>Frames isolados.</b> O estudo não usa o movimento, que é central nos distúrbios funcionais.
- <b>Máscaras e técnica.</b> Na MLP de classificação, a técnica muda a imagem e a máscara, então os dois efeitos se misturam.
- <b>MLP de segmentação e Grad-CAM.</b> A primeira usa 1 semente; o segundo, os pesos da semente 0. A MLP de segmentação é avaliada em resolução diferente, por isso não comparamos o HD95 com o método clássico.
- <b>Tempos.</b> Medidos em um único PC, em uso normal. Análises curtas rodaram durante parte do treino da CNN, o que pode ter alongado um pouco alguns treinos. Por isso, os custos servem para comparar, não como valores gerais.
- <b>Regras das hipóteses.</b> Algumas foram escritas com evidência parcial já conhecida (registrado em DECISIONS.md). A H1 foi adaptada de 5 para 6 métodos e a H5, de 24 para 16 combinações, porque o estudo mudou. A H2 compara médias da tabela de ruído.
- <b>Uso.</b> O estudo não tem validação clínica. A ferramenta é apoio à decisão, nunca diagnóstico.
""",
    "faltando": """
## O que ainda falta

- <b>Tales:</b> relatório final em ABNT, a partir deste documento.
- <b>Protótipo e apostila:</b> ainda não foram atualizados com estes resultados.
- <b>Marcos Vinicius:</b> revisar as técnicas e a Figura 1 da versão com imagens.
- <b>Pedro Nobre:</b> revisar os resultados da MLP e testar o app quando for atualizado.
- <b>Marcos Mendes e Tales:</b> revisar as figuras de segmentação; Marcos Mendes, também a literatura.
- <b>Decisão do grupo:</b> vale fazer o teste sem as bordas pretas antes da entrega? Ele mediria o peso do formato do vídeo.

## Onde está cada coisa

- <b>Código:</b> <font face="A-I">src/laringopdi/</font>. Tudo roda por <font face="A-I">python -m laringopdi &lt;comando&gt;</font>.
- <b>Refazer o estudo:</b> <font face="A-I">iniciar_estudo.bat</font>, que pula o que já terminou.
- <b>Tabelas:</b> <font face="A-I">results/tables/</font>, nas pastas <font face="A-I">pdi, mlp, cnn, sintese</font> e <font face="A-I">custo</font>.
- <b>Figuras:</b> <font face="A-I">docs/figures/</font>.
- <b>Decisões e histórico:</b> <font face="A-I">docs/DECISIONS.md</font>, <font face="A-I">docs/LOG.md</font> e <font face="A-I">docs/MUDANCAS_ETAPA1.md</font>.
- <b>Só no PC do Nathan:</b> pesos e resultados de cada treino, além da versão deste documento com imagens, porque imagens de pacientes não vão para o repositório.
""",
}

# Mudanças em relação ao escopo aprovado na Etapa 1

A orientação da Etapa 2 pede que mudanças relevantes na proposta original sejam justificadas. Esta tabela reúne todas elas; o histórico completo está em `docs/DECISIONS.md`.

**O que foi prometido e está entregue:** BAGLS; técnicas de aula (histograma, alargamento, equalização, CLAHE, média, mediana, Fourier com passa-baixa gaussiano); segmentação Otsu × adaptativa + morfologia com Dice e IoU; PSNR, SSIM, contraste RMS, entropia e tempo de processamento; MLP 18–32–16–2; comparação com CNN pré-treinada em validação cruzada agrupada por vídeo; as hipóteses P1–P6, com veredito em `docs/hipoteses_etapa1.md`.

## Acréscimos e ampliações

| Mudança | Etapa 1 | Etapa 2 | Motivo | Evidência / arquivo |
|---|---|---|---|---|
| Características clássicas | 18 características e MLP 18–32–16–2 | As duas versões: C6 cumpre o prometido; C1–C5 usam 187 características (média e desvio por vídeo), com MLP 256–64, SVM e RF | Medir se um vetor maior muda o resultado, sem abandonar o prometido | `results/runs/C6/`, `results/tables/mlp18_p4.csv`, `classic_vs_cnn.csv` |
| Técnicas de PDI | As técnicas de aula | + `spec`, `homo`, `bilat`, `lbp`, `edges`, M1/M2, `roi`/`roi_gt` e a combinação gulosa E14 | É a "pesquisa externa" prevista no Quadro 3 da Etapa 1; as técnicas de aula continuam como ponto de partida | `docs/pdi_tecnicas.md`, `configs/experiments.yaml` |
| Experimento da CNN | CNN pré-treinada | + SimpleCNN do zero, linear probe, dois estágios (triagem e fine-tuning) e 3 sementes | Testar se o PDI ajuda mais uma rede sem pré-treino e controlar a variação entre sementes | `results/tables/stage1_ranking.csv`, `verdicts_*.csv` |
| Hipóteses | P1–P6 | P1–P6 + H1–H6 internas | As H1–H6 detalham a pergunta de pesquisa para o desenho do experimento; a correspondência está no relatório das P1–P6 | `docs/hipoteses.md`, `docs/hipoteses_etapa1.md` |
| Tarefa B | 47 vídeos estruturais | 54 vídeos | 47 ficava abaixo do mínimo de 50 do método; entraram 7 vídeos de alterações estruturais benignas (cicatriz, sulco, rigidez, irregularidade de mucosa) | `configs/label_map.yaml`, `DECISIONS.md` (2026-10-23) |
| Unidade de avaliação | Validação agrupada por vídeo | Métricas por frame e **por vídeo** (média das probabilidades), e o nível de vídeo é o principal | O diagnóstico é do paciente, e os frames de um vídeo não são independentes | `src/laringopdi/evaluate.py` |
| Grad-CAM | Hipótese de atenção (P5) | + métricas formais: energia na glote, energia nos reflexos e razão normalizada pela área | Sem normalizar pela área, as variantes com recorte não seriam comparáveis | `results/tables/attention.csv`, `attention_p5.csv` |

## O que não foi feito, e por quê

| Item | Decisão | Motivo | Evidência / arquivo |
|---|---|---|---|
| Laryngoscope8 | Não usado | O dataset ficou inacessível; a troca para o BAGLS ocorreu **antes** da entrega da Etapa 1 e não é mudança de escopo | `DECISIONS.md` (2026-10-22) |
| Imagens coloridas | Tudo em escala de cinza | O BAGLS é quase todo em cinza (só 25 vídeos RGB), o que não sustenta um estudo de técnicas de cor. As técnicas `lab`, `hsv` e `vessel` e as características de cor saíram | `DECISIONS.md` (2026-10-22 e 2026-10-23) |

## Desvios a declarar no fechamento

| Item | O que aconteceu | Tratamento |
|---|---|---|
| P5 | Frames com Grad-CAM sem energia não estavam previstos no pré-registro e saíram da análise principal depois de vistos os resultados | Sensibilidade contando-os como atenção 0: o veredito não muda (`attention_p5_sensitivity.csv`) |
| P1, P3, P4 e P6 | As regras foram escritas com evidência parcial já conhecida | Declarado na entrada de 2026-11-04 do `DECISIONS.md` |

# Mudanças em relação à Etapa 1

A orientação da Etapa 2 pede que mudanças relevantes na proposta original sejam justificadas. O histórico completo está em `docs/DECISIONS.md`.

**O que foi prometido e está entregue:** BAGLS; técnicas de aula (histograma, alargamento, equalização, CLAHE, média, mediana, Fourier com passa-baixa gaussiano); segmentação por Otsu e adaptativa com morfologia, com Dice e IoU; PSNR, SSIM, contraste RMS, entropia e tempo de processamento; 18 características e MLP 18–32–16–2; comparação com ResNet-18 pré-treinada e Grad-CAM, em validação cruzada agrupada por vídeo; hipóteses H1–H6, com veredito em `docs/hipoteses_etapa1.md`.

## Mudanças

| Item | Etapa 1 | Etapa 2 | Justificativa | Arquivo |
|---|---|---|---|---|
| Método final de segmentação | Otsu e adaptativa + morfologia | As duas foram feitas; o melhor resultado foi crescimento de região com CLAHE | O Quadro 3 da Etapa 1 previa técnica externa caso as de aula não bastassem. A média da adaptativa + morfologia ficou bem abaixo | `results/tables/seg_adaptive_morph.csv` |
| Tarefa B | Saudável × lesão estrutural (47 vídeos) | 54 vídeos | 47 ficava abaixo do mínimo de 50 do método; entraram 7 vídeos de alterações estruturais benignas (cicatriz, sulco, rigidez, irregularidade de mucosa) | `configs/label_map.yaml`, `DECISIONS.md` (2026-10-23) |
| Unidade de avaliação | Validação agrupada por vídeo | Além do agrupamento, a nota principal é por vídeo (média das probabilidades) | Os frames de um vídeo não são independentes, e a decisão é sobre o paciente | `src/laringopdi/evaluate.py` |

## Testes além do prometido

Ficam no repositório, fora do protótipo e da apresentação: uma segunda CNN treinada do zero (SimpleCNN), triagem com a ResNet congelada, recortes pela região da glote, entradas de bordas e textura, combinações de técnicas, e um vetor maior de características com outros classificadores. Eles não substituem o que foi prometido. Registro em `docs/DECISIONS.md` e `docs/registro_exploratorio.pdf`.

## Prometido, mas parcial

| Item | O que aconteceu |
|---|---|
| Sobel e Canny como abordagem complementar | Aparecem no protótipo e nas características, mas não foram avaliados com Dice como método próprio de segmentação |

## Desvios a declarar

| Item | O que aconteceu | Tratamento |
|---|---|---|
| H5 | Frames com Grad-CAM sem energia não estavam previstos no pré-registro e saíram da análise principal depois de vistos os resultados | Sensibilidade contando-os como atenção 0: o veredito não muda (`attention_p5_sensitivity.csv`) |
| H1, H3, H4 e H6 | As regras foram escritas com evidência parcial já conhecida | Declarado em `DECISIONS.md` (2026-11-04) |
| Técnicas na ResNet-18 | Remoção de reflexo, equalização, CLAHE e mediana têm 1 semente; a homomórfica tem 3 | Declarado no relatório. A "melhora" da remoção de reflexo na Tarefa A depende da família de comparações (p corrigido 0,033 com 5 técnicas; 0,053 com 8) |

**Nomes das hipóteses:** H1–H6 são as da apresentação da Etapa 1. O arquivo `docs/hipoteses.md` traz outro conjunto, de uso interno no desenho do experimento, com a mesma numeração; a nota sobre isso está em `DECISIONS.md` (2026-11-04).

## Acréscimos da reestruturação final

| Item | O que mudou | Justificativa |
|---|---|---|
| MLP de segmentação | Acrescentada: classifica cada pixel a partir da vizinhança 15×15 | Pedido de levar a MLP também à segmentação, com e sem PDI |
| Treino local | Tudo na RTX 3060 do Nathan, sem nuvem | Ter hardware, tempo e custo medidos nas mesmas condições |
| Critérios de convergência e custo | Cada técnica recebe veredito também nessas medidas | Convergir antes é um resultado válido, mesmo com custo maior em outro ponto |
| Controle do formato do vídeo | Imagem sem PDI sem as bordas (E18), fora do estudo principal | Verificar um fator de confusão encontrado nos resultados: câmera e diagnóstico andam juntos no BAGLS |

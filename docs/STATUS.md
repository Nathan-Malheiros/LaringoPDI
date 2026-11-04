# Estado do projeto — 2026-11-04

> Relatório: `docs/relatorio_etapa2.pdf` (público; a versão com frames fica em `entrega/`). **Próximo passo: a apresentação final; ver `docs/PROXIMOS_PASSOS.md`.**

## Concluído
- **Dados, PDI, segmentação, classificação, Grad-CAM e protótipo:** 5.870 frames, 587 vídeos, 5 folds por vídeo.
- **Entrega fiel à Etapa 1 (branch `etapa2/fechamento`):**
  - adaptativa + morfologia, sensibilidade nas tabelas, e equalização, CLAHE e mediana na ResNet-18 (30 jobs, semente 0);
  - hipóteses H1–H6 da apostila (`docs/hipoteses_etapa1.md`);
  - app com 5 abas, na ordem da apresentação (`docs/ROTEIRO_DEMO.md`);
  - relatório curto (`tools/relatorio_etapa2.py`) e apostila (`tools/apostila_etapa2.py`);
  - pacote de entrega (`tools/pacote_entrega.py`).

## Resultados principais (todos em `results/tables/`)
- **Segmentação:** o PDI tem efeito forte e dependente do método. O melhor foi crescimento de região + CLAHE (Dice 0,614); a adaptativa + morfologia ficou em 0,280.
- **Classificação:** MLP e ResNet-18 ficam próximas na Tarefa B (F1 0,713 × 0,762). Sensibilidade baixa na Tarefa A.
- **PDI na ResNet-18:** equalização, CLAHE e mediana ficaram neutras. A remoção de reflexo melhora na Tarefa A, com evidência fraca (1 semente; depende da família de comparações).
- **Hipóteses:** H1 parcial, H2 confirmada, H3 confirmada, H4 confirmada (frágil), H5 parcial, H6 refutada.

## Intocado
Splits, caches E00–E14, `segmentation_best.json`, `stage1_ranking.csv` e os `verdicts_*.csv` anteriores. O único treino novo foi o de 30 jobs, com o protocolo fixo.

## Pendências
Humanas, listadas em `docs/PROXIMOS_PASSOS.md`. A branch aguarda o pedido de merge na `main`.

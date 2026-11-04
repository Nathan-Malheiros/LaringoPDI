# Estado do projeto — 2026-11-04

> Relatório técnico: `docs/relatorio_tecnico.pdf` (público). **Próximo passo: a apresentação final; ver `docs/PROXIMOS_PASSOS.md`.**

## Concluído
- **Fases 0–8:** dados (5.870 frames, 587 vídeos, 5 folds por vídeo), PDI, segmentação (`region_grow` + `clahe`, Dice 0,614), Estágios 1 e 2 (420 jobs de CNN, probe e clássico), Grad-CAM, relatório e protótipo.
- **Fechamento da Etapa 2 (`docs/ETAPA2_PLANO.md`, branch `etapa2/fechamento`):**
  - técnicas de aula `stretch`, `mean` e `gauss_lpf`;
  - estudos de ruído (PSNR/SSIM), realce (saturação e segmentação com HE e alargamento), MLP 18–32–16–2 (C6), atenção normalizada e tempo de processamento;
  - relatório P1–P6 (`docs/hipoteses_etapa1.md`) e tabela de mudanças (`docs/MUDANCAS_ETAPA1.md`);
  - app com 7 abas e roteiro de demonstração (`docs/ROTEIRO_DEMO.md`);
  - pacote de entrega (`tools/pacote_entrega.py` → `entrega/`).

## Resultados principais
- **Classificação:** nenhuma técnica de PDI muda o F1 por vídeo de forma significativa (todos os vereditos do Estágio 2 são neutros).
- **Segmentação:** o PDI tem efeito forte e dependente do método.
- **Hipóteses da Etapa 1:** P1 parcial, P2 confirmada, P3 confirmada, P4 confirmada (com confirmação frágil), P5 parcial (efeito misto), P6 refutada.

## Intocado neste fechamento
`results/runs/` dos experimentos existentes, caches E00–E14, `data/splits/*`, `segmentation_best.json`, `segmentation_summary.csv`, `stage1_ranking.csv`, `verdicts_*.csv`. Nenhuma CNN foi treinada.

## Pendências
Humanas, listadas em `docs/PROXIMOS_PASSOS.md`. A branch aguarda o pedido de merge na `main`.

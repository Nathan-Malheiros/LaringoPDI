# Diário das fases

Resumos curtos (≤ 10 linhas) ao fim de cada fase. Detalhes em `DECISIONS.md` e nos logs.

## Fase 0 — Setup (2026-10-22)
- Estrutura `src/laringopdi`, `configs/base.yaml` + `experiments.yaml` (E00–E16, estágios, C1–C4), `requirements.txt`, `.gitignore`, README, venv Python 3.11 com torch CPU.
- CLI única `python -m laringopdi <cmd>` com os 14 comandos; implementados: `index`, `splits`, `eda`, `gallery`, `cache`.
- Adiantado: dedup pHash+SSIM, StratifiedGroupKFold com checagem de vazamento, as 16 técnicas de PDI + `quality`, EDA e galeria.
- Testes sintéticos: 29 passam, 1 pulado (splits reais).
- **Bloqueio:** o dataset não está em `data/raw/laryngoscope8/`. O download é só pelo Baidu Pan, com login; não há espelho. O aceite (`index.csv` com 3057 linhas, contagem por classe, ID de paciente) fica pendente.

## Troca para o BAGLS — código adaptado (2026-10-22)
- ESPECIFICACAO.md reescrito para o BAGLS; decisões em DECISIONS.md.
- Código: `BaglsAdapter` (lê os metadados de `.meta`/`.json`, a ser confirmado nos arquivos reais) e o stub `Laryngoscope8Adapter`.
- Amostragem de até 10 frames por vídeo, grupos por vídeo (pHash + SSIM), folds por vídeo e cache com a máscara GT.
- PDI em cinza: saem `lab`, `hsv` e `vessel`; `spec` por limiar de intensidade (piso 200); entram `roi_gt`, `roi` (melhor segmentação clássica) e M2 = [cinza, máscara, bordas].
- Novo `pdi/segmentation.py` com 5 métodos e métricas Dice, IoU, HD95 e falha; `seg-study` escolhe os parâmetros só no treino de cada fold.
- Testes: 45 passam, 1 pulado (splits reais), incluindo um teste de fumaça ponta a ponta num BAGLS sintético.
- **Bloqueio:** o BAGLS ainda não está em `data/raw/bagls/`; o Passo 0 (inspeção dos metadados reais) aguarda o download.

## Passo 0 — BAGLS real (2026-10-23)
- Zips validados por md5 e extraídos em `data/raw/bagls/{training,test}`: 59.250 frames, 640 vídeos.
- `.meta` confirmado (JSON). `Video Id` único entre os splits; não há ID de sujeito. Adaptador corrigido para `Color` booleano.
- O status é texto livre (55 valores) → `configs/label_map.yaml` (regras ordenadas) + tabela de revisão em `dataset_report.md`.
- Vídeos: saudável 377 · funcional 156 · estrutural **47** · desconhecido 53 · outros 7. Tarefas A 377×203; B 377×47; C 377/156/47. RGB: 25.
- **Parada:** a família estrutural tem menos de 50 vídeos (regra do ESPECIFICACAO.md) → decisão de método com o usuário antes dos splits.

## Fase 1 — Dados (2026-10-23)
- Tarefa B ampliada para 54 vídeos estruturais (decisão do usuário). Vídeos: saudável 377 · funcional 156 · estrutural 54 · excluídos 53.
- Amostragem: 5.870 frames de 587 vídeos (10 por vídeo, só frames com glote visível) em `data/splits/frames.json`.
- Quase-duplicatas de vídeo: 1 par candidato (pHash ≤ 6), nenhum confirmado por SSIM ≥ 0,85 → 587 grupos de 1 vídeo.
- Folds (StratifiedGroupKFold por vídeo, estratificados por C): teste com ~118 vídeos por fold (75–76 saudáveis / 31–32 funcionais / 10–11 estruturais); validação interna com ~67 vídeos.
- O teste de vazamento nos splits reais passa. `dataset_report.md` regenerado com figuras e com a tabela status → família.
- Histórico do Git limpo de arquivos do BAGLS (filter-repo + force push, aprovado).

## Fase 3 — Segmentação clássica (2026-10-23)
- `seg-study`: 5 métodos × 4 pré-processamentos × grade de parâmetros em 5.870 frames; ajuste no treino de cada fold, avaliação no teste.
- Melhor: crescimento de região + CLAHE (Dice 0,614 · falha 2,1% · HD95 9,7 px), seguido de crescimento de região sobre o `raw` (0,467) e sobre a homomórfica (0,440). O Otsu global é linha de base (Dice 0,02).
- Watershed corrigido (núcleos por componente) antes da escolha final; o vencedor não mudou (DECISIONS.md).
- Saídas: `results/tables/segmentation.csv`, `segmentation_summary.csv`, `segmentation_best.json`; a figura `segmentation_examples.png` (local, não versionada) vai para a revisão de Marcos Mendes e Tales.

## Fase 2 — PDI (2026-10-23)
- 14 técnicas em cinza + 5 segmentadores. `docs/pdi_tecnicas.md` é gerado das docstrings pelo `gallery`. A galeria (grade técnica × família; passo a passo de `spec`, `homo` com espectro e H(u,v), e `roi` × `roi_gt`) fica em `docs/figures/`, local e não versionada, para revisão do Marcos Vinicius.
- Caches E00–E13: 5.870 × 224 × 224 × 3 cada (~12 GB em `data/processed/cache`, fora do Git).
- `results/pdi_quality.csv`:
  - `spec` reduz a fração especular de 1,7% para 0,9%;
  - HE estoura o contraste (entropia 7,7; contraste RMS 0,31) frente ao CLAHE (6,5; 0,22);
  - a homomórfica eleva a nitidez (variância do Laplaciano 130 → 1.185);
  - fallback da ROI: 2,1% (`roi`), 0,3% (`clahe>roi`), 0% (`roi_gt`).
- Limitação: as métricas de qualidade usam o canal 0. Em `edges` e M2, que começam pelo cinza, elas coincidem com E00.
- Testes: 55 passam (inclui o teste de vazamento nos splits reais).

## Fases 4–6 (parte local, CPU) — 2026-10-23
- **Fase 4 (código):**
  - `models.py` (ResNet-18 ImageNet, SimpleCNN 32/64/128/256);
  - `train.py`: jobs com early stopping no F1-macro **por vídeo da validação interna**, só aug geométrico, AMP na GPU, jobs concluídos pulados;
  - `shards` (stage1: 140 jobs em 5 shards de 2–3 experimentos cada) e `notebooks/colab_runner.ipynb`.
  - Medição local: SimpleCNN ≈ 121 s/época na CPU. A medição no T4 está **pendente** (humano).
- **Fase 5 (probe em CPU):** 14 exp × 2 tarefas × 5 folds = 140 jobs.
  - F1 por vídeo no teste: A 0,61–0,67; B 0,72–0,80.
  - Top-4 provisório, escolhido **só pela validação**: E05 `bilat`, E04 `homo`, E08 `roi`, E06 `median`. É definitivo só depois da SimpleCNN rápida no Colab.
- **Fase 6 (clássico):** C1–C5 com características por vídeo (média e desvio de 187 características × 10 frames).
  - F1 por vídeo, A: C1 0,642 · C2 0,662 · C3 0,664 · C4 0,664 · C5 0,658.
  - F1 por vídeo, B: C1 0,762 · C2 0,737 · C3 0,757 · C4 0,811 · C5 0,762.
  - Competitivo com o probe da ResNet (E00: A 0,656; B 0,748).
- **Estatística (`report`):**
  - Probe: todos os 26 vereditos são **neutros** (McNemar OOF com Holm + IC bootstrap). Os ΔF1 ficam em ±0,03, com ICs cruzando 0.
  - H4 (segmentação): o efeito do realce depende do par método × pré-processamento. Crescimento de região + CLAHE dá +0,148 de Dice e watershed/Otsu+morfologia + homomórfica dão +0,08 a +0,10; CLAHE piora os métodos baseados em limiar global.
  - Os p muito pequenos vêm de ~587 vídeos pareados; no Otsu o efeito é desprezível (|Δ| ≤ 0,002) apesar do p.
- Correção de bug: a chave `classic` do `experiments.yaml` sobrescrevia os hiperparâmetros do `base.yaml`; foi renomeada para `classic_experiments`.

## Fases 7–8 (código) — 2026-10-23
- `gradcam`: energia do Grad-CAM dentro da glote (GT de cada cache) e sobre os reflexos (máscara `spec` do E00, só em pipelines sem recorte). Usa `model.pt` da semente 0, agora salvo pelo `train`. Roda quando o Estágio 2 terminar.
- `report`: vereditos (McNemar OOF + Holm + IC bootstrap; Wilcoxon reportado), teste de H4, clássico × CNN, tabelas CSV + LaTeX e `docs/hipoteses.md`.
- App Streamlit: upload → etapas do PDI → segmentação → métricas → predição + Grad-CAM (quando houver pesos), com o aviso obrigatório.
- `tests/test_e2e.py`: ponta a ponta em CPU em 64 s. Suíte completa: 64 testes passando.

## Telemetria do treino + bug de AMP (2026-10-23)
- `live.py`: telemetria só de observação para o painel LaringoMonitor: lote atual, Grad-CAM com a caixa da glote, ativação e ‖ΔW‖/‖W‖ por camada, filtros, mapas de ativação e PDI passo a passo. Liga só com `results/live/enabled`.
- A telemetria revelou augmentation corrompida sob AMP (DECISIONS.md). Corrigido, com teste de regressão; os 22 runs afetados foram para quarentena; o Estágio 1 foi reiniciado.

## Estágio 1 na GPU + relatório técnico (2026-10-23)
- Estágio 1 refeito após a correção de AMP: 140 jobs da SimpleCNN rápida (RTX 3060, ~80 s/job). Top-4 pela validação: E04 `homo`, E11 `lbp`, E05 `bilat`, E10 `edges`.
- `report`: probe e SimpleCNN somam **52/52 vereditos neutros** (nenhuma técnica muda o F1 por vídeo de forma significativa no Estágio 1).
- 1ª tentativa do Estágio 2 encerrada por falta de RAM, ainda na busca gulosa do E14 (nada foi gravado). `build_cache` agora grava em blocos (memmap).
- Relatório técnico em PDF (`tools/relatorio_pdf.py`): 16 p. na versão pública (Git) e 19 p. na completa (Área de Trabalho). Próximo passo em `docs/PROXIMOS_PASSOS.md`.

## Estágio 2 em execução + fechamento automático (2026-10-24)
- E14 guloso = `homo>lbp` (F1 de validação do probe 0,712). Estágio 2 (360 jobs) rodando em `tools/stage2_chain.py`.
- `tools/final_chain.py` espera o Estágio 2 e roda `stage2_hyp` (E01/E07/E08, 1 semente, para H1 e H5) → `gradcam` → `report` → PDFs.
- Regra de veredito do Estágio 2 fixada antes dos resultados (DECISIONS.md): OOF por ensemble das sementes; IC exclui 0 e McNemar ou Wilcoxon (≥ 10 pares) com Holm.
- `report` agora conclui H1–H6 de forma explícita (`docs/hipoteses.md` + `results/tables/hypotheses.json`): H2 com confronto direto contra o HE, H3 por Wilcoxon sobre os ΔF1, H5 com `roi` × `roi_gt`, H6 com C1 × CNN pareado.
- Com os dados atuais (Estágio 1): H2 e H3 não suportadas; H4, H5 e H6 parciais; H1 aguarda o Grad-CAM.
- PDF: seção do Estágio 2, figura de atenção e tabela de hipóteses entram sozinhas quando os dados existirem. 67 testes passando.


## Fechamento automático (tools/final_chain.py, 2026-10-25 00:47)
- Estágio 2 + stage2_hyp concluídos; gradcam, report e PDFs regenerados. Conclusões (`docs/hipoteses.md`):
  - H1: parcialmente suportada
  - H2: não suportada (sem diferença detectável na classificação)
  - H3: não suportada
  - H4: parcialmente suportada (depende do par método × realce: 6 melhoram, 9 pioram; melhor = region_grow + clahe, Δ +0.148)
  - H5: parcialmente suportada
  - H6: parcialmente suportada (A: competitiva; B: competitiva)

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

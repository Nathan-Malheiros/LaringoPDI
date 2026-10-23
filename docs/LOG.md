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

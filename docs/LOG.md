# Diário das fases

Resumos curtos (≤ 10 linhas) ao fim de cada fase. Detalhes em `DECISIONS.md` e nos logs.

## Fase 0 — Setup (2026-10-22)
- Estrutura `src/laringopdi`, `configs/base.yaml` + `experiments.yaml` (E00–E16, estágios, C1–C4), `requirements.txt`, `.gitignore`, README, venv Python 3.11 com torch CPU.
- CLI única `python -m laringopdi <cmd>` com os 14 comandos; implementados: `index`, `splits`, `eda`, `gallery`, `cache`.
- Adiantado: dedup pHash+SSIM, StratifiedGroupKFold com checagem de vazamento, as 16 técnicas de PDI + `quality`, EDA e galeria.
- Testes sintéticos: 29 passam, 1 pulado (splits reais).
- **Bloqueio:** o dataset não está em `data/raw/laryngoscope8/`. O download é só pelo Baidu Pan, com login; não há espelho. O aceite (`index.csv` com 3057 linhas, contagem por classe, ID de paciente) fica pendente.

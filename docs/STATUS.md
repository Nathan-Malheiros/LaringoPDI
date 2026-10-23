# Estado do projeto — 2026-10-23

## Concluído
- **Passo 0:** BAGLS (Zenodo 3762320) validado por md5 e extraído em `data/raw/bagls/{training,test}`: 59.250 frames, 640 vídeos.
- **Rótulos:** `configs/label_map.yaml` (texto livre → família). Vídeos: saudável 377 · funcional 156 · estrutural 54 · excluídos 53.
- **Fase 1:** 5.870 frames (10 por vídeo), 587 vídeos, 5 folds por vídeo sem vazamento; `docs/dataset_report.md`.
- **Fase 3:** segmentação clássica; melhor = crescimento de região + CLAHE (Dice 0,614); `roi` e `m2` fixados.
- **Fase 2:** galeria, `docs/pdi_tecnicas.md`, caches E00–E13, `results/pdi_quality.csv`.

## Próximo (Fase 4 — infra de GPU)
- `train`, `models.py`, `shards`, `notebooks/colab_runner.ipynb`; depois o `probe` local em CPU (Fase 5).
- Depende de humanos: medir 1 job no T4 para recalcular o orçamento.

## Pendências humanas
- Revisar as figuras locais em `docs/figures/` (não versionadas: contêm frames do BAGLS):
  - Marcos Vinicius: `gallery_*.png`;
  - Marcos Mendes e Tales: `segmentation_examples.png`.
- Revisar a tabela "texto → família" em `docs/dataset_report.md`, sobretudo os casos ambíguos (`None (…)`, "swelling", cicatriz/sulco).
- E-mail aos autores do Laryngoscope8; avisar a Prof.ª Lyndainês sobre a troca de dataset.
- Adicionar Marcos Vinicius, Pedro Nobre, Marcos Mendes e Tales como colaboradores do repositório `PDIZip` (preciso dos usernames do GitHub).
- Os zips `data/test.zip` e `data/training.zip` (3,7 GB) já foram extraídos e podem ser movidos ou apagados por você.

# Estado do projeto — 2026-10-23

## Concluído
- **Passo 0:** BAGLS (Zenodo 3762320) validado por md5 e extraído em `data/raw/bagls/{training,test}`: 59.250 frames, 640 vídeos.
- **Rótulos:** `configs/label_map.yaml` (texto livre → família). Vídeos: saudável 377 · funcional 156 · estrutural 54 · excluídos 53.
- **Fase 1:** 5.870 frames (10 por vídeo), 587 vídeos, 5 folds por vídeo sem vazamento; `docs/dataset_report.md`.
- **Fase 3:** segmentação clássica; melhor = crescimento de região + CLAHE (Dice 0,614); `roi` e `m2` fixados.
- **Fase 2:** galeria, `docs/pdi_tecnicas.md`, caches E00–E13, `results/pdi_quality.csv`.

- **Fase 4 (código), Fase 5 (probe) e Fase 6 (clássico C1–C5)** rodaram na CPU; `report` gera `docs/hipoteses.md` + tabelas em `results/tables/`.

## Próximo — depende do Colab (humano)
1. **Nathan prepara o Drive** `MyDrive/LaringoPDI/`:
   - `code/`: `git archive --format=zip HEAD -o laringopdi_code.zip`, descompactado em `LaringoPDI/code/`;
   - `cache/`: os arquivos de `data/processed/cache/` (`E*_*.npy`, `E*_*.json`, `E*_*.masks.npz` e `frame_labels.csv`; ~12 GB). Os `.probe.npy` não são necessários;
   - compartilhar a pasta com os 4 colegas (só a equipe: são dados do BAGLS).
2. **Medir o orçamento:** abrir `notebooks/colab_runner.ipynb` com `MEASURE_ONLY = True` (T4) e anotar s/época da SimpleCNN e da ResNet-18. Recalculo o orçamento em DECISIONS.md.
3. **Estágio 1:** cada pessoa roda o notebook com o seu `SHARD_ID` (0–4; 140 jobs no total). Depois: copiar `LaringoPDI/runs/` para `results/runs/` e rodar `summary` → top-4 definitivo + E14 → `shards --stage stage2`.
4. Estágio 2 (fine-tuning, 3 sementes) → `gradcam` (Fase 7) → `report` final → app Streamlit (Fase 8).

## Pendências humanas
- Revisar as figuras locais em `docs/figures/` (não versionadas: contêm frames do BAGLS):
  - Marcos Vinicius: `gallery_*.png`;
  - Marcos Mendes e Tales: `segmentation_examples.png`.
- Revisar a tabela "texto → família" em `docs/dataset_report.md`, sobretudo os casos ambíguos (`None (…)`, "swelling", cicatriz/sulco).
- E-mail aos autores do Laryngoscope8; avisar a Prof.ª Lyndainês sobre a troca de dataset.
- Adicionar Marcos Vinicius, Pedro Nobre, Marcos Mendes e Tales como colaboradores do repositório `PDIZip` (preciso dos usernames do GitHub).
- Os zips `data/test.zip` e `data/training.zip` (3,7 GB) já foram extraídos e podem ser movidos ou apagados por você.

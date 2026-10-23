# LaringoPDI

Projeto extensionista de Processamento Digital de Imagens (PDI, UNIFOR): quais técnicas de PDI
aplicadas a imagens de laringoscopia de luz branca melhoram, não alteram ou pioram a classificação
de alterações das pregas vocais por redes neurais, e por quê? O classificador é o instrumento fixo;
o PDI é a variável independente. **Ferramenta de apoio acadêmico. Não substitui avaliação médica.**

A especificação completa está em `docs/ESPECIFICACAO.md`. Decisões de método ficam em `docs/DECISIONS.md` e o
diário das fases em `docs/LOG.md`.

## Instalação (Windows, CPU, Python 3.11)

```powershell
uv venv --python 3.11 .venv            # ou: py -3.11 -m venv .venv
uv pip install -p .venv torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install -p .venv -r requirements.txt
uv pip install -p .venv -e .
```

## Dados (download manual) — BAGLS

1. Baixe o BAGLS (Gómez et al., Scientific Data, 2020; CC BY-NC-SA 4.0) por bagls.org, Zenodo ou Kaggle.
2. Extraia em `data/raw/bagls/`, mantendo as pastas `training/` e `test/` com `<id>.png`,
   `<id>_seg.png` e os metadados por frame.

Cite: Gómez, P. et al. *BAGLS, a multihospital Benchmark for Automatic Glottis Segmentation*. Sci Data 7, 186 (2020).

## Fluxo

```powershell
.venv\Scripts\python -m laringopdi index     > logs\index.log 2>&1     # data/index.csv
.venv\Scripts\python -m laringopdi splits    > logs\splits.log 2>&1    # frames.json + grupos + folds.json (uma vez)
.venv\Scripts\python -m laringopdi eda       > logs\eda.log 2>&1       # docs/dataset_report.md
.venv\Scripts\python -m laringopdi seg-study > logs\seg.log 2>&1       # results/tables/segmentation*.csv
.venv\Scripts\python -m laringopdi gallery   > logs\gallery.log 2>&1   # docs/figures/gallery_*.png
.venv\Scripts\python -m laringopdi cache --exp stage1                   # data/processed/cache + results/pdi_quality.csv
.venv\Scripts\python -m pytest -q --no-header -x
```

Os comandos `probe`, `train`, `shards`, `classic`, `gradcam`, `summary` e `report` entram nas fases
seguintes. A matriz de experimentos fica em `configs/experiments.yaml`; os parâmetros fixos, em
`configs/base.yaml`.

## Estrutura

```
src/laringopdi/   cli, config, data, evaluate, report, pdi/ (registry, técnicas, segmentation, quality)
configs/          base.yaml, experiments.yaml
data/             raw/ (ignorado), index.csv, groups.csv, splits/ (frames.json, folds.json)
docs/             DECISIONS.md, LOG.md, dataset_report.md, figures/
results/          runs/ (ignorado), pdi_quality.csv, tabelas
```

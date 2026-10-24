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

## Dados

O dataset **não fica neste repositório**: são imagens de pacientes sob licença CC BY-NC-SA 4.0.
Cada integrante baixa o BAGLS manualmente.

1. Baixe `training.zip` e `test.zip` do Zenodo, record 3762320
   (https://zenodo.org/records/3762320), e confira o md5 de cada arquivo. O `raw.zip` não é necessário.
2. Extraia em `data/raw/bagls/training/` e `data/raw/bagls/test/`. Cada frame tem `<id>.png`,
   `<id>_seg.png` (máscara da glote) e `<id>.meta` (metadados em JSON).

- **Licença do dataset:** Creative Commons BY-NC-SA 4.0 (uso não comercial, com atribuição e
  compartilhamento pela mesma licença).
- **Citação:** Gómez, P., Kist, A. M., Schlegel, P. et al. *BAGLS, a multihospital Benchmark for
  Automatic Glottis Segmentation*. Scientific Data 7, 186 (2020). https://doi.org/10.1038/s41597-020-0526-3

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

Depois dos caches:

```powershell
.venv\Scripts\python -m laringopdi probe                 # Estágio 1, linear probe em CPU (ranking pela validação)
.venv\Scripts\python -m laringopdi classic               # braço clássico C1–C5 (por vídeo)
.venv\Scripts\python -m laringopdi shards --stage stage1 # divide os jobs da SimpleCNN entre 5 contas do Colab
.venv\Scripts\python -m laringopdi summary               # results/summary.csv + stage1_ranking.csv
.venv\Scripts\python -m laringopdi gradcam               # atenção na glote/reflexos (precisa dos pesos do Estágio 2)
.venv\Scripts\python -m laringopdi report                # vereditos, H4, clássico × CNN, docs/hipoteses.md
streamlit run app\streamlit_app.py                        # protótipo (apoio acadêmico; não é diagnóstico)
```

Estágio 2 e fechamento (GPU local; retomáveis, jobs concluídos são pulados):

```powershell
.venv\Scripts\python -m laringopdi train --stage stage1 --shard 0   # … shards 0–4 da SimpleCNN rápida
.venv\Scripts\python tools\stage2_chain.py > logs\stage2_chain.log 2>&1   # E14 guloso → cache → 360 jobs do Estágio 2
.venv\Scripts\python tools\final_chain.py  > logs\final_chain.log 2>&1    # stage2_hyp (E01/E07/E08) → gradcam → report → PDFs
```

Treinos de CNN também rodam no Colab com `notebooks/colab_runner.ipynb` (escolher `SHARD_ID` → Executar tudo).
Um job isolado: `python -m laringopdi train --exp E03 --task A --model resnet18 --fold 0 --seed 0`.

## Estrutura

```
src/laringopdi/   cli, config, data, features, models, train, evaluate, report,
                  pdi/ (registry, técnicas, segmentation, quality, attention)
app/              streamlit_app.py
notebooks/        colab_runner.ipynb
configs/          base.yaml, experiments.yaml
data/             raw/ (ignorado), index.csv, groups.csv, splits/ (frames.json, folds.json)
docs/             DECISIONS.md, LOG.md, dataset_report.md, figures/
results/          runs/ (ignorado), pdi_quality.csv, tabelas
```

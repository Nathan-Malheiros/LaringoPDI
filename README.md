# LaringoPDI

Projeto extensionista de Processamento Digital de Imagens (PDI, UNIFOR): qual combinação de técnicas
de realce, filtragem e segmentação delimita melhor a glote em imagens de laringoscopia, e em que medida
as características extraídas dela distinguem pregas vocais saudáveis de alteradas? **Ferramenta de apoio acadêmico. Não substitui avaliação médica.**

A especificação completa está em `docs/ESPECIFICACAO.md`. Decisões de método ficam em `docs/DECISIONS.md` e o
diário das fases em `docs/LOG.md`.

## Como executar o protótipo (sem GPU e sem o BAGLS)

Use o pacote `LaringoPDI_codigo_com_pesos.zip`, que já traz os pesos da ResNet-18 (frame original e CLAHE) e da MLP de 18 características:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -e .
.venv\Scripts\python -m streamlit run app\streamlit_app.py
```

Na barra lateral, **Enviar arquivo** aceita um frame PNG ou JPG, que é convertido para cinza. A segmentação foi calibrada no BAGLS, então imagens de outra origem podem dar resultado pior. As abas seguem a ordem da apresentação: realce, filtragem e Fourier, segmentação, PDI + MLP × CNN e resultados. Na barra lateral, a entrada da CNN oferece as técnicas com pesos disponíveis.

- **Pesos:** `results/runs/{E00,E03}/{A,B}/resnet18/fold0_seed0/model.pt` e `results/runs/C6/{A,B}/mlp18/fold0_seed0/model.joblib`.
- **Relatório:** `docs/relatorio_etapa2.pdf`. **Hipóteses:** `docs/hipoteses_etapa1.md`. **Roteiro da demonstração:** `docs/ROTEIRO_DEMO.md`.
- **Dataset:** o BAGLS **não é redistribuído** (CC BY-NC-SA 4.0), nem no repositório nem no pacote. Para rodar os estudos, baixe-o conforme a seção Dados.

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

Estudos das hipóteses H1–H6 da Etapa 1 (precisam do BAGLS e dos caches, nesta ordem):

```powershell
.venv\Scripts\python -m laringopdi denoise-study > logs\denoise.log 2>&1      # H2: PSNR, SSIM e borda com ruído sintético
.venv\Scripts\python -m laringopdi enhance-study > logs\enhance.log 2>&1      # H1: saturação e segmentação com alargamento e HE
.venv\Scripts\python -m laringopdi mlp18-study   > logs\mlp18.log 2>&1        # C6 (MLP 18–32–16–2), H4 e H6
.venv\Scripts\python -m laringopdi attention-p5  > logs\attention_p5.log 2>&1 # H5, a partir do Grad-CAM já calculado
.venv\Scripts\python -m laringopdi timing        > logs\timing.log 2>&1       # tempo por técnica, pipeline e modelo
.venv\Scripts\python -m laringopdi adaptive-morph-study > logs\adaptive.log 2>&1 # H3: adaptativa + morfologia
.venv\Scripts\python -m laringopdi train --stage stage_aula --shard 0 > logs\stage_aula.log 2>&1  # 30 jobs: E02, E03 e E06 na ResNet-18 (GPU)
.venv\Scripts\python -m laringopdi aula-study   > logs\aula.log 2>&1         # vereditos, sensibilidade e Grad-CAM desses jobs
.venv\Scripts\python -m laringopdi etapa1-report > logs\etapa1_report.log 2>&1 # docs/hipoteses_etapa1.md
.venv\Scripts\python tools\capturas_app.py                                   # capturas das abas
.venv\Scripts\python tools\relatorio_etapa2.py --out docs\relatorio_etapa2.pdf --publico
.venv\Scripts\python tools\relatorio_etapa2.py --out entrega\LaringoPDI_relatorio_etapa2.pdf
.venv\Scripts\python tools\apostila_etapa2.py --out entrega\LaringoPDI_Apostila_Etapa2.pdf
.venv\Scripts\python tools\pacote_entrega.py --with-weights                  # entrega/LaringoPDI_codigo_com_pesos.zip
```

Treinos de CNN também rodam no Colab com `notebooks/colab_runner.ipynb` (escolher `SHARD_ID` → Executar tudo).
Um job isolado: `python -m laringopdi train --exp E03 --task A --model resnet18 --fold 0 --seed 0`.

## Estrutura

```
src/laringopdi/   cli, config, data, features, models, train, evaluate, report, etapa1,
                  pdi/ (registry, técnicas, segmentation, quality, attention)
app/              streamlit_app.py
notebooks/        colab_runner.ipynb
configs/          base.yaml, experiments.yaml
data/             raw/ (ignorado), index.csv, groups.csv, splits/ (frames.json, folds.json)
docs/             relatorio_etapa2.pdf, hipoteses_etapa1.md, MUDANCAS_ETAPA1.md, ROTEIRO_DEMO.md, DECISIONS.md, LOG.md, dataset_report.md, figures/
results/          runs/ (ignorado), pdi_quality.csv, tabelas
```

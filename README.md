# LaringoPDI

Projeto extensionista de Processamento Digital de Imagens (PDI, UNIFOR, Prof.ª Lyndainês Santos). Pergunta: quais técnicas de PDI aplicadas a frames de laringoscopia melhoram, não alteram ou pioram a segmentação da glote e a classificação de distúrbios da voz, e a que custo? **Ferramenta de apoio acadêmico. Não substitui avaliação médica.**

**Para entender o projeto e os resultados, comece por `docs/Documento_explicativo_LaringoPDI.pdf`.** A especificação está em `docs/ESPECIFICACAO.md`, as decisões de método em `docs/DECISIONS.md` e o diário das fases em `docs/LOG.md`.

## Resultado em uma frase

O PDI muda muito a segmentação (CLAHE + crescimento de região: Dice 0,614 contra 0,467 sem PDI) e quase nada a classificação (62 de 64 vereditos neutros). A MLP de 18 características empatou com a ResNet-18 em F1, a uma fração do custo. O formato do vídeo é um possível fator de confusão (seção 15 do documento).

## Como executar o protótipo (sem GPU)

Use o pacote `LaringoPDI_codigo_com_pesos.zip` (`python tools/pacote_entrega.py --with-weights`), que traz os pesos do fold 0 sem PDI e com CLAHE:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -e .
.venv\Scripts\python -m streamlit run app\streamlit_app.py      # ou: tools\abrir_app.bat
```

Os passos seguem a apresentação: realce, filtragem e Fourier, segmentação (clássica e MLP), classificação (MLP × CNN) e resultados do estudo. A barra lateral traz as **cenas da apresentação**: cada cena ajusta frame, técnica, tarefa e passo. O app só usa frames de vídeos do teste do fold 0, que nenhum modelo do app viu no treino; o BAGLS precisa estar na máquina.

- **Pesos:** `results/runs/<exp>/<A|B>/{resnet18,mlp18}/fold0_seed0/model.pt` e `results/runs/<exp>/seg/mlp_seg/fold0_seed0/model.pt`.
- **Apresentação:** apostila em `entrega/LaringoPDI_Apostila.pdf` (gerada por `tools/apostila.py`) e roteiro da demonstração em `docs/ROTEIRO_DEMO.md`.
- **Dataset:** o BAGLS **não é redistribuído** (CC BY-NC-SA 4.0), nem no repositório nem no pacote.

## Dados

Baixe o BAGLS manualmente (Gómez et al., 2020) e extraia em `data/raw/bagls/`, com as pastas `training/` e `test/` (`<id>.png`, `<id>_seg.png` e os metadados). Dados, caches, pesos e resultados ficam no HD (`D:`).

## Reproduzir o estudo (Windows, GPU)

O jeito simples: `iniciar_estudo.bat`. Ele roda ou retoma tudo, pulando o que terminou, e pode ser pausado pelo LaringoMonitor. Os comandos, na ordem:

```powershell
.venv\Scripts\python -m laringopdi index                 # índice do BAGLS
.venv\Scripts\python -m laringopdi splits                # 5 folds por vídeo, salvos uma vez
.venv\Scripts\python -m laringopdi pdi-study             # etapa 1 e 2: PDI isolado e segmentação clássica
.venv\Scripts\python -m laringopdi cache --exp study     # imagens de cada técnica, em cache
.venv\Scripts\python -m laringopdi mlp-class --jobs 10   # etapa 4: MLP de classificação (CPU)
.venv\Scripts\python -m laringopdi mlp-seg               # etapa 3: MLP de segmentação (GPU)
.venv\Scripts\python -m laringopdi train --all           # etapa 5: 270 treinos da ResNet-18 (GPU)
.venv\Scripts\python -m laringopdi gradcam               # atenção na glote
.venv\Scripts\python -m laringopdi cnn                   # vereditos da CNN
.venv\Scripts\python -m laringopdi sintese               # etapa 6: síntese, hipóteses e custo por fase
.venv\Scripts\python tools\teste_sem_bordas.py           # controle do formato do vídeo
.venv\Scripts\python tools\documento_grupo.py            # documento explicativo (--frames: versão com imagens, fora do git)
.venv\Scripts\python tools\apostila.py                   # apostila da apresentação
.venv\Scripts\python -m pytest -q
```

Todo comando grava o tempo, a CPU, a RAM, a GPU e a energia em `results/tables/custo/etapas.csv`.

## Onde está cada coisa

```
src/laringopdi/    código (cli, data, pdi/, features, mlp, train, evaluate, sintese, custo, pausa)
configs/           base.yaml e experiments.yaml: a matriz inteira do estudo
app/               streamlit_app.py
tools/             documento, apostila, controle sem bordas, pacote de entrega
results/tables/    pdi/, mlp/, cnn/, sintese/, custo/
docs/              Documento_explicativo_LaringoPDI.pdf, DECISIONS.md, LOG.md, MUDANCAS_ETAPA1.md, ROTEIRO_DEMO.md, figures/
```

O que saiu do fluxo está em `D:\DocumentosHD\Unifor\PDI\Projeto em desuso LaringoPDI` (`MANIFESTO.json`).

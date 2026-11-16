# LaringoPDI — especificação do projeto

Documento de referência do método: dados, técnicas, protocolo, estudo, arquitetura do código e fases. As decisões tomadas ao longo do projeto estão em `docs/DECISIONS.md` e o diário das fases em `docs/LOG.md`.

> Dataset trocado do Laryngoscope8 (inacessível) para o **BAGLS** em 2026-10-22; histórico em `docs/DECISIONS.md`.

---

## 1. O projeto
- **Disciplina:** Processamento Digital de Imagens (PDI), UNIFOR, Prof.ª Lyndainês Santos. É um projeto extensionista com três dimensões obrigatórias: social, tecnológica e científica.
- **Regra de ouro:** o **PDI é o protagonista**. Uma rede neural recebendo a imagem crua não satisfaz a disciplina. Toda técnica precisa mostrar *o que faz na imagem* e *por que importa na laringoscopia*.
- **Pergunta de pesquisa:** quais técnicas de PDI aplicadas a frames de videoendoscopia laríngea melhoram, não alteram ou pioram (a) a segmentação clássica da glote e (b) a classificação de distúrbios da voz por redes neurais, e por quê?
- **Desenho:** o classificador é o **instrumento de medida fixo** e o PDI é a **variável independente**. Arquitetura, hiperparâmetros, splits, sementes e augmentation ficam fixos entre as variantes. Resultados negativos são válidos e devem ser reportados.
- **Público-alvo:** profissionais da voz (sobretudo professores) como beneficiários; profissionais de saúde em triagem como usuários. A ferramenta é **apoio à decisão, nunca diagnóstico**.

## 2. Dados — BAGLS
- **BAGLS** (Gómez et al., *BAGLS, a multihospital Benchmark for Automatic Glottis Segmentation*, Scientific Data, 2020; CC BY-NC-SA 4.0): ~59.250 frames de 640 vídeos de alta velocidade, com máscara da glote feita por especialistas (`<id>.png` + `<id>_seg.png`) e metadados por frame (vídeo, idade, sexo, status do distúrbio). ~618 vídeos em cinza e ~22 em RGB. O download é **manual**, feito pelo usuário, para `data/raw/bagls/`. Não baixe sozinho nem contorne bloqueios. O BAGLS-VF só entra se o usuário fornecer. Leitura via `DatasetAdapter` em `data.py` (`BaglsAdapter`; `Laryngoscope8Adapter` é um stub).
- **Tarefas** (excluindo os vídeos "desconhecido" e "outros"):
  - A (principal) = saudável × com distúrbio.
  - B = saudável × lesão estrutural (edema, nódulos, pólipo, cisto, granuloma, papiloma, leucoplasia, carcinoma + cicatriz, sulco/vergeture, rigidez e irregularidade de mucosa; 54 vídeos). Rótulos por `configs/label_map.yaml`.
  - C (opcional) = saudável / funcional / estrutural. Qualquer lesão estrutural → estrutural.
- **Anti-vazamento (inegociável):**
  - `StratifiedGroupKFold` com 5 folds, grupo = vídeo, estratificado pelo rótulo do vídeo. Vídeos quase-duplicados (pHash do frame central + SSIM) são fundidos. O split oficial do BAGLS é ignorado.
  - Até 10 frames com máscara por vídeo, uniformemente espaçados, numa amostra fixa salva em `data/splits/frames.json`.
  - Splits salvos uma vez em `data/splits/*.json`. A validação interna (early stopping) sai de dentro do treino.
  - Toda estatística ajustada aos dados é calculada só no treino. Nunca escolha nada olhando o teste.
- **Avaliação em dois níveis:** por frame e **por vídeo** (média das probabilidades). O nível de vídeo é o principal.

## 3. Técnicas de PDI (tudo em escala de cinza)
Registry com pipelines encadeáveis. Cada técnica tem a assinatura `f(img: np.ndarray, ctx: dict, **params) -> np.ndarray`, sobre cinza uint8. A docstring em PT-BR tem quatro partes: **o que faz / por que importa na laringoscopia / parâmetros padrão / referência**. Os parâmetros vêm da literatura, nunca do desempenho do classificador. Saída com 1 canal → replicada 3× para a CNN.

| ID | Código | Técnica | Origem |
|---|---|---|---|
| E00 | `raw` | Sem realce nem filtro; resize preservando a proporção + pad 224 (referência) | Aula |
| E15 | `stretch` | Alargamento de contraste | Aula |
| E02 | `he` | Equalização global | Aula |
| E03 | `clahe` | CLAHE (clip 2.0, tiles 8×8) | Aula |
| E16 | `mean` | Filtro da média | Aula |
| E06 | `median` | Filtro da mediana | Aula |
| E17 | `gauss_lpf` | Fourier + passa-baixa gaussiano | Aula |
| E01 | `spec` | Remoção de reflexo especular (limiar + dilatação + inpaint TELEA) | Literatura, prevista na Etapa 1 |
| E04 | `homo` | Filtragem homomórfica via FFT (γL≈0.5, γH≈1.5–2, D0≈30) | Literatura, prevista na Etapa 1 |

Métricas da própria imagem: entropia, contraste RMS, variância do Laplaciano, fração saturada e fração especular; com ruído sintético, PSNR, SSIM e preservação de borda.

## 4. Modelos e protocolo (fixos)
- **CNN:** ResNet-18 (ImageNet, transferência de aprendizado).
- **MLP de classificação:** 18 características (forma, intensidade e textura; média por vídeo) → 18–32–16–2.
- **MLP de segmentação:** classifica cada pixel como glote ou não, a partir da vizinhança. Acréscimo a declarar em `docs/MUDANCAS_ETAPA1.md`.
- **Treino da CNN:**
  - Entrada 224, em cinza replicado 3×. Augmentation **só geométrico** (flip, ±15°, escala 0.9–1.1); **proibido jitter de intensidade, blur ou equalização**. PDI aplicado antes, com cache em disco.
  - Entropia cruzada ponderada, AdamW 1e-4, cosine, batch 32, até 30 épocas, early stopping com paciência 7 no F1-macro, precisão mista, imagens pré-carregadas em memória.
  - 3 sementes × 5 folds. A semente 0 termina inteira primeiro.
- **Toda execução salva:** métricas por época no treino e na validação, tempo de treino, tempo até a melhor época, épocas até convergir, tempo de inferência, pico de memória e **especificação do hardware** (`config.hardware()`).

## 5. Estudo (regras em `docs/DECISIONS.md`, 2026-11-06)
O PDI é o protagonista. Cada técnica é comparada com a referência `raw`, nas mesmas condições.

| Etapa | Pergunta | Métricas |
|---|---|---|
| 1. PDI isolado | Qual técnica melhora a imagem, e por quê? | Contraste RMS, entropia, Laplaciano, saturação; PSNR e SSIM com ruído; tempo |
| 2. Segmentação clássica | Qual técnica segmenta melhor a glote, e a que custo? | Dice, IoU, HD95, falha, tempo |
| 3. MLP na segmentação | A MLP supera os métodos clássicos? O PDI a ajuda? | Dice, IoU, curva e tempo de treino |
| 4. MLP na classificação | Quanto cada técnica ajuda, não muda ou piora? | Sensibilidade, especificidade, F1, AUC, curva e tempo de treino |
| 5. CNN na classificação | Idem | Idem + Grad-CAM (atenção na glote) |
| 6. Síntese | O ganho compensa o custo? | Custo de PDI + custo de treino × resultado, por técnica |

- **Tarefas:** A = saudável × alterado; B = saudável × lesão estrutural. A sensibilidade é a métrica prioritária.
- **Estatística:** Wilcoxon pareado (fold×semente) com correção de Holm, McNemar sobre predições out-of-fold (nível de vídeo), IC 95% por bootstrap. Cada técnica recebe o veredito melhora / neutra / piora, inclusive no custo e na convergência.
- **Hipóteses apresentadas na Etapa 1:** H1: CLAHE > equalização global. H2: mediana > média em sal e pimenta. H3: limiar adaptativo > Otsu. H4: MLP ≈ CNN na Tarefa B. H5: PDI aumenta a atenção da CNN na glote. H6: forma é a família de características mais importante.
- **Expectativa honesta:** distúrbios funcionais são dinâmicos, então um desempenho modesto na Tarefa A é plausível. Resultados negativos são válidos e devem ser reportados.

## 6. Arquitetura do código (enxuta, orientada a configuração)
**Um único ponto de entrada:** `python -m laringopdi <comando>`, com os comandos `index, eda, splits, gallery, seg-study, cache, train, gradcam, summary, denoise-study, pdi-study, mlp-class, mlp-seg, cnn, sintese`.

```
src/laringopdi/
  cli.py        # subcomandos
  config.py     # configs + hardware()
  data.py       # adaptador BAGLS, índice, amostragem de frames, grupos, splits, cache
  pdi/          # registry.py + enhance.py + filters.py + texture.py + segmentation.py + quality.py + attention.py
  features.py   # as 18 características
  models.py     # resnet18
  train.py      # treino da CNN, um job por vez ou `--all`
  evaluate.py   # métricas (frame e vídeo), estatística, Grad-CAM, estudo de segmentação
  etapa1.py     # estudo de ruído
  report.py     # EDA, galeria e documentação das técnicas
  live.py       # observação do treino ao vivo
configs/base.yaml, configs/experiments.yaml   # a matriz inteira vive aqui
app/streamlit_app.py
tests/
```

- **Execução autônoma:** `iniciar_estudo.bat` roda ou retoma tudo, pulando o que terminou. No LaringoMonitor (atalho na área de trabalho), Pausar termina o job atual e espera; Retomar continua ou relança o `.bat`.
- Rodar um job = `python -m laringopdi train --exp E03 --task A --model resnet18 --fold 0 --seed 0`; o estudo inteiro = `train --all`.
- Use bibliotecas em vez de reescrever: `scikit-learn`, `scikit-image`, `torchvision`, `grad-cam`, `scipy.stats`.
- Resultados em `results/runs/<exp>/<task>/<model>/fold{k}_seed{s}/` (`metrics.json`, `preds.csv`, `config.yaml`). Jobs já concluídos são **pulados**. O `summary.csv` é sempre gerado por comando, nunca editado à mão.

## 7. Fases (ao fim de cada uma: resumo em `docs/LOG.md` + commit)
| Fase | Entrega | Aceite |
|---|---|---|
| **A. Reestruturação** | Código e dados fora do fluxo arquivados; regras registradas | Testes passando |
| **B. PDI + segmentação** | Tabela única por técnica: visual, Dice e custo | `results/tables/` + figura; todas as técnicas com veredito |
| **C. MLP** | Segmentação e classificação, com e sem PDI | Tabelas com treino, custo e veredito por técnica |
| **D. CNN** | `train --all` na GPU + Grad-CAM | Tabelas com treino, custo, convergência e veredito por técnica |
| **E. Síntese e entrega** | Hipóteses H1–H6, relatório, apostila e protótipo atualizados | Conclusão explícita por técnica; app roda no Windows |

## 8. Equipe (quem faz o quê fora do código)
| Integrante | Papel |
|---|---|
| **Nathan** (2220227) | Representante; desenvolvimento e integração; roda os treinos |
| **Marcos Vinicius** | Revisa as técnicas de PDI e a galeria (validação visual) |
| **Pedro Nobre** | Testa o app e revisa os resultados da MLP |
| **Marcos Mendes** | Evidências do público-alvo e literatura; revisa as figuras de segmentação |
| **Tales** | Relatório final (ABNT); revisa as figuras de segmentação |

## 9. Regras gerais
- Python 3.11 (`.venv`). Código em inglês; docstrings, comentários e docs em PT-BR.
- A máquina é Windows com GPU (RTX 3060, 12 GB): use `pathlib` e `num_workers=0`. Dados, caches, pesos e resultados ficam no HD (`D:`), nunca no SSD. Dependências: `torch`, `torchvision`, `opencv-python`, `scikit-image`, `scikit-learn`, `scipy`, `numpy`, `pandas`, `matplotlib`, `pyyaml`, `tqdm`, `imagehash`, `grad-cam`, `streamlit`, `pytest`.
- Commits pequenos, com mensagens em PT-BR. Decisões de método ou escopo vão em `docs/DECISIONS.md`.
- Não adicione técnicas, modelos ou tarefas fora deste escopo sem decisão da equipe.
- **Fora do escopo:** treinar U-Net, SimpleCNN, recortes pela glote, combinações de técnicas, análise temporal de vídeo (os frames são tratados como imagens estáticas), NBI, detecção de lesões com caixas, validação clínica, deploy.

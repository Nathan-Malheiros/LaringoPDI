# Decisões de método e escopo

Registro cronológico. Cada decisão: data, contexto, decisão, motivo.

## 2026-10-22 — Fase 0
- **Ambiente:** Python 3.11 em `.venv` (via `uv`), torch CPU. A máquina tinha só o Python 3.14 instalado; a especificação exige 3.11.
- **Escala de trabalho do PDI:** antes de qualquer técnica, o lado maior é reduzido para 512 px (`data.work_size`). Assim os parâmetros espaciais da literatura (D0=30 na homomórfica, kernels morfológicos, bloco adaptativo 51) têm o mesmo significado em imagens de resoluções diferentes. O `raw` (letterbox 224) é sempre o último passo.
- **`fov`:** limiar = min(Otsu, 60) para que o Otsu não caia dentro do tecido em imagens sem moldura; fallback para a imagem inteira se a máscara cobre < 20%.
- **`roi`:** a fenda glótica (componente escuro mais central) serve de âncora; a caixa é expandida com margem de 60% e lado mínimo de 35% para incluir as pregas. Fallback: recorte central de 60%. A validação contra anotação humana acontece na Fase 3.
- **`vessel`:** saída montada como (G', B', B'), o mapeamento de cores do NBI, onde G' e B' = CLAHE − black-hat.
- **`edges`:** 3 canais [cinza, Sobel, Canny automático] para manter a compatibilidade com a ResNet.
- **Estratificação dos splits:** pelas 8 classes (cobre as tarefas A, B e C); validação interna = 1 fold de um StratifiedGroupKFold interno (~15%) dentro do treino.
- **Dataset:** o Laryngoscope8 só está disponível pelo Baidu Pan (exige login) e não há espelho público; o download continua manual.

## 2026-10-22 — Bloqueio de dados (decisão pendente com o usuário)
- O link do Baidu Pan do Laryngoscope8 retorna "arquivo compartilhado excluído". As issues #1–#17 do repositório pedem um novo link desde 2023-08 e os autores não respondem.
- Não há espelho público (busca em Kaggle, Hugging Face e Zenodo).
- VoFoCD (Dao et al., 2024; 1724 imagens, 4 classes) **não é público**: as imagens pertencem ao Cho Ray Hospital.
- CE-NBI (Scientific Data, 2023) é público, mas é NBI de contato, que está fora do escopo.

## 2026-10-22 — Troca de dataset: Laryngoscope8 → BAGLS (decisão do usuário)
- Motivo: indisponibilidade registrada na entrada anterior (commit 4da9a12). E-mail aos autores e consulta à professora seguem em paralelo, sem bloquear nada; a leitura da revisão de 2024 foi cancelada.
- Dataset principal: **BAGLS** (Gómez et al., Scientific Data, 2020; CC BY-NC-SA 4.0), com download manual para `data/raw/bagls/`. O `Laryngoscope8Adapter` vira um stub.
- **Tarefas:** A = saudável × com distúrbio; B = saudável × lesão estrutural; C = saudável / funcional / estrutural (qualquer lesão estrutural → estrutural). Vídeos "desconhecido" ou "outros" ficam de fora.
- **Agrupamento:** a unidade é o vídeo. StratifiedGroupKFold com 5 folds, grupo = vídeo, estratificado pelo rótulo C do vídeo (cobre A e B). Vídeos quase-duplicados (pHash do frame central, Hamming ≤ 6, confirmado por SSIM ≥ 0,85) são fundidos num grupo só. O **split oficial treino/teste do BAGLS é ignorado**: foi feito para segmentação e não é estratificado pelo status.
- **Amostragem:** até 10 frames por vídeo (`data.frames_per_video`), uniformemente espaçados, só entre frames com máscara. A amostra é fixa (`data/splits/frames.json`) e a mesma em todos os experimentos.
- **Avaliação:** em dois níveis, por frame e por vídeo (média das probabilidades). O nível de vídeo é o principal.
- **Escala de cinza:** todas as imagens, inclusive os ~22 vídeos RGB, viram cinza; para as CNNs, o canal é replicado 3×. Saem `lab`, `hsv` e `vessel`. O `spec` passa a usar limiar de intensidade (Otsu sobre a cauda acima do percentil 90, com piso de 200) + dilatação + TELEA. O `fov` segue disponível, mas fora da matriz; se a EDA mostrar que não há campo circular, ele fica documentado como identidade.
- **Fase 3 (nova):** estudo de segmentação clássica da glote contra o ground truth (Otsu, adaptativo, Otsu+morfologia, crescimento de região, watershed) × pré-processamento (raw, clahe, homo, bilat). Os parâmetros são ajustados só nos vídeos de treino de cada fold. Para a técnica `roi` usada no cache de classificação, método e parâmetros vêm do treino do fold 0. Esse ajuste usa só as máscaras de glote, nunca os rótulos de classe, e é a única exceção ao ajuste por fold. A anotação manual da ROI foi eliminada.
- **Matriz nova:** E00–E14 (ESPECIFICACAO.md, Seção 5). `roi_gt` (E07) é o limite superior, não uma técnica utilizável na prática.
- **Braço clássico:** saem as características de cor; entram as de forma glótica (máscara clássica e, como referência, a do ground truth). As características são agregadas por vídeo (média e desvio-padrão).
- **Hipóteses H1–H6 reescritas antes de qualquer resultado** (nenhum treino ou métrica de classificação existia em 2026-10-22). Expectativa registrada: distúrbios funcionais são dinâmicos, e um desempenho modesto na Tarefa A é plausível e válido; o efeito mais claro do PDI é esperado na Tarefa B e na segmentação.

## 2026-10-23 — Passo 0 (BAGLS real)
- **Formato confirmado:** `<id>.png`, `<id>_seg.png` e `<id>.meta` (JSON com 12 chaves, entre elas `Video Id` inteiro, `Color` booleano, `Camera`, `Sampling rate (Hz)`, `Subject disorder status`, `Segmenter`). O `Video Id` (0–639) é único entre `training/` e `test/`, então serve como `video_id`. Nenhum vídeo tem mais de um status. Não há ID de sujeito.
- **Rótulos:** o status é texto livre, com 55 valores distintos. O mapeamento usa regras ordenadas em `configs/label_map.yaml`, e a tabela completa texto → família fica em `results/tables/status_map.csv` e `docs/dataset_report.md` para revisão humana. Casos decididos:
  - vazio → excluído;
  - `None (higher phonation)` → saudável;
  - os demais `None (…)` (pós-cirurgia ou histórico de lesão) → excluídos;
  - "swelling" mesomembranoso → estrutural;
  - cicatriz, vergeture/sulco, rigidez e irregularidade de mucosa → "outros" (excluídos);
  - espasmódica, laringite e paresia → funcional;
  - qualquer lesão estrutural vence o componente funcional.
- **Contagem (vídeos):** saudável 377, funcional 156, estrutural 47, desconhecido 53, outros 7. RGB: 25 vídeos (esperado ~22; a contagem usa o `Color` do 1º frame). Frames com glote visível: 55.286 de 59.250.

## 2026-10-23 — Tarefa B ampliada (decisão do usuário)
- Com a lista original, a família estrutural tinha 47 vídeos (< 50). Por decisão do usuário, entraram nela os 7 vídeos de alterações estruturais benignas da lâmina própria: cicatriz, sulco/vergeture, rigidez/fibrose e irregularidade de mucosa. Antes ficavam em "outros" e eram excluídos. Total: **54 vídeos**. A família "outros" fica vazia no BAGLS.
- Higiene do repositório: `eda_samples.png` (frames reais) e `data/index.csv` foram commitados por engano. Saíram de todo o histórico (git filter-repo + force push, com aprovação do usuário). O `.gitignore` agora barra figuras com frames reais e tabelas por frame.

## 2026-10-23 — Fase 3: segmentação clássica (método fixado para `roi`/`m2`)
- **Correção de implementação no watershed (antes da escolha final):** na 1ª rodada, o núcleo dos marcadores era `dist > 0,5 × máximo global`, então só a maior região escura (fundo/canto) recebia semente e a glote nunca. Resultado: Dice ≈ 0,005 em todos os pré-processamentos (tabela preservada em `results/tables/segmentation_summary_v1_watershed_bug.csv`). Agora o núcleo usa o máximo **de cada componente**. Isso corrige um defeito, não ajusta a métrica, e o vencedor não mudou entre as rodadas.
- **Escolhido (treino+val do fold 0, só máscaras):** crescimento de região sobre CLAHE, `tol=20`. Dice de treino 0,623; no teste dos 5 folds, Dice 0,614, falha 2,1% e HD95 mediano 9,7 px. Fica em `results/tables/segmentation_best.json` e passa a ser usado por `roi` (E08, E09) e `m2` (E13).
- **Leitura para H4 (realce melhora a segmentação):**
  - CLAHE eleva o crescimento de região de 0,467 para 0,614 de Dice e reduz a falha de 19,8% para 2,1%.
  - A homomórfica melhora o Otsu+morfologia (0,324 → 0,408) e o watershed (0,334 → 0,437).
  - Em métodos globais (Otsu, adaptativo), o realce não ajuda ou piora.
  - O teste estatístico formal fica para o `report`.

## 2026-10-23 — Treino na GPU local (RTX 3060) e augmentation por imagem
- O PC do Nathan tem uma **RTX 3060 (12 GB)**. O torch passou a usar CUDA 13 (2.14.1+cu130). Medição no protocolo: SimpleCNN rápida a 6,6 s/época (85 s por job); ResNet-18 a 5,4 s/época (152 s, 28 épocas).
- **Orçamento:** Estágio 1 ≈ 3,5 h (140 jobs); Estágio 2 ≈ 16 h (360 jobs). Tudo roda localmente. O Colab vira opcional e serve só para paralelizar; o notebook e os shards continuam válidos.
- **Correção:** as transformações geométricas eram sorteadas uma vez por lote (v2 sobre o tensor 4-D) e agora são sorteadas por imagem. Os 2 jobs de medição feitos antes da correção foram descartados e refeitos no Estágio 1.
- **E14:** seleção gulosa entre as técnicas do top-4 do Estágio 1. Parte da melhor; acrescenta cada candidata na ordem canônica (ruído/reflexo → realce → recorte → representação) e aceita se o F1-macro por vídeo da **validação interna** do probe (média em A, B × 5 folds) aumentar. Saída em `results/tables/e14_greedy.json`.

## 2026-10-23 — Bug: augmentation corrompida sob AMP (corrigido; Estágio 1 reiniciado)
- **Sintoma:** encontrado pela telemetria visual do treino (`live.py`). Os lotes de treino na GPU apareciam quase binarizados.
- **Causa:** o `RandomAffine` (torchvision v2) rodava dentro do `torch.autocast` (AMP) e saturava ~21,6% dos pixels, contra 0,9% sem autocast.
- **Alcance:** só treinos de CNN na GPU com augmentation (22 jobs do Estágio 1). O probe (sem augmentation), o braço clássico (CPU) e a segmentação não foram afetados.
- **Correção:** a augmentation roda com o autocast desligado, em float32; há teste de regressão em `tests/test_models.py`. Os 22 runs foram para `results/runs_quarantine_amp_aug/` (fora da análise), e o Estágio 1 foi refeito do zero.

## 2026-10-24 — Regra de veredito do Estágio 2 (fixada antes de qualquer resultado do Estágio 2)
- **OOF por vídeo:** com 3 sementes, as predições out-of-fold usam a **média das probabilidades das sementes** (ensemble) antes do McNemar e do bootstrap. Com 1 semente (probe, SimpleCNN rápida) nada muda.
- **Significância:** o IC 95% bootstrap do ΔF1 precisa excluir 0 **e** pelo menos um teste corrigido por Holm (por tarefa) dar p < 0,05: o McNemar OOF ou, quando houver ≥ 10 pares fold×semente, o Wilcoxon pareado. Com 5 pares (Estágio 1) o Wilcoxon não alcança p < 0,05, então a regra do Estágio 1 fica idêntica à já aplicada.
- **H3:** compara o ΔF1 (técnica − E00) da SimpleCNN com o da ResNet-18 nas mesmas técnicas (Estágio 2: 5 técnicas; Estágio 1: SimpleCNN rápida × probe, 13 técnicas), por Wilcoxon pareado sobre as técnicas × tarefas. Suporte se a mediana da diferença for > 0 e p < 0,05.
- **H1 (atenção):** fração da energia do Grad-CAM sobre os reflexos, E01 × E00, por frame pareado (Wilcoxon). Só se E01 tiver pesos (não está no top-4; se ficar de fora, H1 de atenção é reportada como não testada no Estágio 2 e a parte do F1 vem do Estágio 1).
- **Complemento `stage2_hyp`:** E01, E07 e E08 com fine-tuning (ResNet-18 e SimpleCNN, 1 semente, 60 jobs ≈ 2,5 h) só para testar H1 (Grad-CAM sobre reflexos exige pesos do E01) e H5 (`roi` × `roi_gt` com fine-tuning). Não entram no ranking nem no E14; nos vereditos, o par com E00 usa só a semente 0 (5 pares, McNemar OOF).

## 2026-11-04 — Pré-registro das regras de veredito de P1–P6 (hipóteses da apresentação da Etapa 1)
Fixadas **antes** de gerar qualquer número novo dos Blocos 2–5 do `docs/ETAPA2_PLANO.md`.

**Transparência sobre o que já era conhecido.** As regras foram escritas com evidência parcial já vista: P3 (o estudo de segmentação já existia, e `region_grow` + `clahe` já era o melhor); P4 (C1–C5 empatam com a CNN, RF 0,811 na Tarefa B); P6 (nas 187 características, o HOG domina e a forma fica perto de 0). P1 também tinha contexto parcial: E02 × E03 neutros na classificação e o efeito do CLAHE no Dice em relação ao `raw`. Ainda não existiam números de HE na segmentação nem de fração saturada. P2 e P5 (a razão normalizada por área, com teste formal) não tinham resultado visto, embora a tabela `attention.csv` já existisse com as frações brutas.

### Regras (cópia da Seção 4 do plano)
Nível α = 0,05, com correção de Holm dentro de cada família de testes, como descrito em cada item.

- **P1 (CLAHE > HE):**
  - **Confirmada** se (a) o Dice por vídeo de CLAHE for maior que o de HE com p_Holm < 0,05 no método escolhido (`region_grow`) **e** (b) a fração saturada média de HE for maior que a de CLAHE.
  - **Parcialmente confirmada** se só (a) ou só (b) valer, ou se (a) valer em menos da metade dos 5 métodos.
  - **Refutada** se HE > CLAHE com significância no `region_grow`.
  - O resultado de classificação (E02 × E03) é reportado como contexto e não entra no veredito.
- **P2 (mediana > média em sal e pimenta):**
  - **Confirmada** se a mediana 3×3 superar a média 3×3 em PSNR **e** SSIM, com p_Holm < 0,05, nas duas densidades.
  - **Parcialmente confirmada** se valer só em parte.
  - **Refutada** se a média superar a mediana com significância.
- **P3 (adaptativo > Otsu):**
  - **Confirmada** se o Dice por vídeo do `adaptive` for maior que o do `otsu` (ambos sobre `raw`) com p < 0,05 **e** Δ ≥ 0,05.
  - Reporte, sem mudar o veredito, que `otsu_morph` e `region_grow` superam ambos.
- **P4 (MLP ≈ CNN na Tarefa B):**
  - **Confirmada** ("competitiva") se o IC 95% bootstrap do ΔF1 por vídeo (C6 − ResNet-18 E00) contiver 0 ou estiver acima de 0.
  - **Refutada** se estiver todo abaixo de 0.
  - Reporte C1 da mesma forma, como extensão.
- **P5 (PDI aumenta a atenção na glote):** conte as combinações técnica × modelo × tarefa sem recorte (6 × 2 × 2 = 24) em que a razão de atenção sobe com p_Holm < 0,05.
  - **Confirmada** se ≥ 50% das combinações subirem e nenhuma cair com significância.
  - **Parcialmente confirmada** se 1 ou mais subirem.
  - **Refutada** se nenhuma subir e alguma cair.
  - **Inconclusiva** se nenhuma for significativa.
- **P6 (forma é a família mais importante):**
  - **Confirmada** se, no C6, a queda média por permutação da família forma for a maior entre as 3 famílias **nas duas tarefas**.
  - **Parcialmente confirmada** se for a maior em uma tarefa.
  - **Refutada** caso contrário.
  - Reporte também o C1–C5 (187 características), onde o HOG domina, como evidência complementar.

### Esclarecimentos (o plano não define estes casos; fixados agora, sem resultado à vista)
- **P1, "metade dos métodos":** o Holm é aplicado aos 5 métodos. Para **confirmar**, (a) e (b) precisam valer no `region_grow` **e** CLAHE > HE com p_Holm < 0,05 em pelo menos 3 dos 5 métodos; com 1 ou 2 métodos, é parcial. Se nem (a) nem (b) valerem e HE não superar CLAHE com significância no `region_grow`: **inconclusiva**. Se (b) valer e HE > CLAHE com significância no `region_grow`, a regra de refutação prevalece.
- **P2:** Holm sobre os 4 testes (PSNR e SSIM × 5% e 10%). Se a média vencer com significância em algum teste e a mediana em nenhum: **refutada**. Se houver vitórias significativas dos dois lados: **parcialmente confirmada**. Se nenhum teste for significativo: **inconclusiva**.
- **P3:** um único teste, sem Holm. Se adaptativo > Otsu com p < 0,05 mas Δ < 0,05: **parcialmente confirmada**. Se Otsu > adaptativo com p < 0,05: **refutada**. Nos demais casos: **inconclusiva**.
- **P4:**
  - `report.paired_compare` usa as sementes comuns. Como o C6 só tem a semente 0, a comparação é contra a semente 0 da ResNet-18 (sem o ensemble de 3 sementes). O mesmo vale para o C1.
  - **Limitação declarada:** a regra é permissiva. "O IC contém 0" não prova equivalência, e um IC largo também confirma a hipótese. Por isso, o relatório mostra a largura do IC e não trata "competitiva" como "equivalente".
- **P5:**
  - **Vídeos considerados:** só os que têm rótulo na tarefa. O Grad-CAM foi calculado em todos os 587 vídeos de teste, inclusive nos 156 funcionais sob os modelos da Tarefa B, que não pertencem a ela; eles ficam fora da análise da Tarefa B.
  - **Frames excluídos:** os de área glótica 0, porque a razão fica indefinida (27 frames, ou 185 no E08).
  - **Agregação:** razão por frame, depois média por vídeo.
  - **Pareamento:** por vídeo presente nas duas variantes.
  - **Teste:** Wilcoxon bilateral; "sobe" significa p_Holm < 0,05 e Δ médio > 0. Holm sobre as 6 técnicas, dentro de cada modelo × tarefa.
  - **Limitação de medida:** o `gradcam` definiu a área válida como `X > 0`. Nas variantes que zeram pixels da imagem (`homo`, LBP), parte dos pixels escuros da glote sai da energia, mas continua contada na área, o que pode **reduzir** a razão artificialmente. O `gradcam` não será refeito. O relatório informa, por experimento, a fração dos pixels da glote com `X = 0`, e o veredito deve ser lido junto com ela.
- **P6:** "maior" compara as médias sobre os 5 folds, sem teste de significância, como manda a regra. Se a família forma tiver queda média ≤ 0, ela não conta como "a maior", mesmo sendo numericamente a primeira.
- **Vocabulário:** os vereditos possíveis são sempre confirmada, parcialmente confirmada, refutada e inconclusiva.

## 2026-11-04 — MLP-18 (C6): decisões de implementação
- **Agregação por vídeo:** só a média dos 10 frames, para manter exatamente 18 entradas (o C1 usa média e desvio).
- **Máscara e imagem:** E00 (cinza 224) + `glottis_mask_lb` do E13, as mesmas do C1. "Tecido" são os pixels no retângulo da imagem (fora do pad), fora da máscara. GLCM, Canny e Sobel usam a imagem 224 inteira, como `texture_features` e `edge_hog_features` já faziam.
- **Protocolo do C1 mantido:** mesmos folds, z-score do treino, sobreamostragem aleatória do treino. Isso herda uma limitação do C1: o `early_stopping` do scikit-learn separa 10% do treino já sobreamostrado, de modo que cópias podem cair nos dois lados; afeta só o ponto de parada, não o teste.
- **Separação do relatório interno:** o `report` (H1–H6) ignora o C6, para que seu resultado não mude ao reexecutar o comando.

## 2026-11-04 — P5: desvio do pré-registro (frames com Grad-CAM de energia zero)
O pré-registro só previa excluir frames de área glótica 0. Ao rodar o teste, apareceu um segundo caso de razão indefinida: mapas sem energia, com `att_glottis` NaN (zero na ResNet-18; na SimpleCNN, até 38,9% no E11 e 29,1% no E14). A análise principal os excluiu, **decisão tomada depois de ver os resultados**. Para não depender dela, `attention_p5_sensitivity.csv` conta esses frames como atenção 0. O veredito não muda: principal, 5 sobem e 9 caem; sensibilidade, 4 sobem e 10 caem.

## 2026-11-04 — Entrega fiel ao apresentado na Etapa 1 (decisão do Nathan)
Registrado **antes** de gerar os números novos desta entrada. Supera a proibição de treino do `docs/ETAPA2_PLANO.md` só para os 30 jobs abaixo, com autorização do Nathan.

**Escopo da entrega.** Relatório, app e apostila seguem o documento e os slides da Etapa 1. Os experimentos que não foram apresentados (187 características, HOG, Gabor, SimpleCNN, probe como resultado, bilateral, M1/M2, E14, ROI) continuam no repositório, inalterados, e ficam fora da apresentação. O relatório declara a existência deles em um parágrafo, para não haver relato seletivo.

**Numeração.** A entrega usa os nomes da apostila: H1 = CLAHE > equalização global; H2 = mediana > média; H3 = adaptativo > Otsu; H4 = MLP ≈ CNN na Tarefa B; H5 = PDI aumenta a atenção na glote; H6 = forma é a família mais importante. Correspondem, na mesma ordem, às P1–P6 das entradas anteriores; as regras não mudam. As H1–H6 antigas de `docs/hipoteses.md` são internas e ficam fora do pacote de entrega.

**Complementos fixados agora; nenhum altera os vereditos de H1–H6:**
1. **Adaptativa + morfologia (slide 9, objetivo específico 2), que não existia no código:** limiar adaptativo (mesma grade do `adaptive`, inclusive a janela 51 apresentada), abertura e fechamento com elipse 5×5, componente central. Roda nos 6 realces com parâmetros escolhidos no treino de cada fold. Testes em `raw`, Dice por vídeo: contra `otsu`, `adaptive` e `otsu_morph` (Wilcoxon, Holm sobre os 3); e CLAHE × raw no próprio método. O veredito de H3 continua pela regra original (`adaptive` × `otsu`). **`segmentation_best.json` não muda, mesmo que este método vença:** ele já alimentou a máscara do C6 e os experimentos; se vencer, isso será reportado.
2. **Sensibilidade (métrica prioritária do Quadro 4):** tabela por sistema (média dos folds, semente 0). C6 × ResNet-18 E00 em cada tarefa: OOF, semente 0, ΔSensibilidade com IC 95% bootstrap por vídeo e McNemar exato só nos vídeos alterados, com Holm sobre as 2 tarefas.
3. **CLAHE, equalização e mediana na ResNet-18:** E02, E03 e E06, Tarefas A e B, 5 folds, semente 0, com o protocolo fixo (30 jobs). Vereditos pela regra de 2026-10-24, com Holm por tarefa na família das 5 técnicas apresentadas que têm ResNet-18 (E01, E02, E03, E04, E06). E01 e E04 já tinham resultado conhecido, neutro. `verdicts_*.csv` existentes não mudam: a tabela nova é `verdicts_aula_resnet18.csv`, e o `report` ignora esses jobs.
4. **Atenção (H5), complemento:** Grad-CAM de E02, E03 e E06 na ResNet-18, razão por vídeo contra o E00 (Holm sobre os 3, por tarefa). Reportado à parte: as 24 combinações da regra de H5 não mudam.


## 2026-11-06 — Reestruturação e regras do novo estudo (antes de qualquer número novo)
**Por quê:** o projeto se afastou do que foi apresentado na Etapa 1 e ficou difícil de acompanhar e explicar. O fluxo volta a ser o da apresentação, com o PDI como objeto de estudo.

**Saiu do fluxo (arquivado, nada apagado):** SimpleCNN, linear probe, recortes pela glote, bordas/LBP/fusões como entrada, combinação gulosa, bilateral, vetor de 187 características, SVM e RF. Treinos passam do Colab para o PC do Nathan (RTX 3060), para que o custo seja comparável.

**Técnicas:** `raw` (referência), `stretch`, `he`, `clahe`, `mean`, `median`, `gauss_lpf`, `spec`, `homo`, com os parâmetros padrão já registrados. Nada é escolhido olhando o teste.

**Etapa 1, PDI isolado.** Realce (`stretch`, `he`, `clahe`, `spec`, `homo`): **melhora visual** = contraste RMS maior, entropia não menor e fração saturada até 1 ponto percentual acima da referência, com Wilcoxon pareado por vídeo e Holm. Filtros (`mean`, `median`, `gauss_lpf`): PSNR e SSIM com ruído sintético. As métricas são medidas objetivas de imagem, sem validação clínica.

**Etapa 2, segmentação clássica.** Métodos: `otsu`, `adaptive`, `otsu_morph`, `adaptive_morph`, `region_grow`, `watershed`, cada um × as 9 técnicas. Parâmetros ajustados só no treino. Veredito por método: Dice por vídeo contra `raw`, com Wilcoxon e Holm. Custo: tempo por frame de técnica + segmentação.

**Etapa 3, MLP na segmentação (acréscimo).** Classifica cada pixel como glote ou não a partir de uma vizinhança de 15×15 pixels da imagem 224×224, seguida da seleção do componente central usada nos métodos clássicos. Pixels de treino balanceados, só de vídeos do treino do fold. Mesma arquitetura para todas as técnicas. Métricas: Dice, IoU, curva e tempo de treino.

**Etapa 4, MLP na classificação.** Técnica T → segmentação pelo método fixado em `segmentation_best.json` (crescimento de região) sobre a imagem com T → 18 características sobre a imagem com T → média por vídeo → MLP 18–32–16–2. Com `raw`, é a comparação principal "sem realce nem filtro". A segmentação é ela própria PDI, então um "sem PDI" absoluto não existe nesse formato; a **referência sem nenhum PDI** é uma MLP sobre os pixels da imagem reduzida a 32×32, cinza, com arquitetura fixa 1024–256–64–2.

**Etapa 5, CNN.** Protocolo fixo da Seção 4 do `docs/ESPECIFICACAO.md`, 3 sementes × 5 folds.

**Vereditos de classificação (Etapas 4 e 5).** F1-macro por vídeo contra `raw`; melhora ou piora só se o IC 95% excluir 0 e o McNemar (ou, com pelo menos 10 pares, o Wilcoxon) corrigido por Holm for < 0,05. A mesma regra vale para a sensibilidade, métrica prioritária, reportada ao lado.

**Vereditos de custo e convergência.** Medidas: épocas até a melhor validação, segundos de treino até a melhor validação, segundos por época, milissegundos de PDI por frame e custo total (PDI × nº de frames de treino + treino até a melhor validação). Wilcoxon pareado (fold×semente) com Holm dentro de cada tarefa e modelo: "converge antes", "igual" ou "converge depois"; "mais barata", "igual" ou "mais cara". Um PDI mais caro pode compensar se reduzir o treino: é isso que o custo total mede.

**Curvas de treino.** Perda e F1 no treino e na validação, por época, para medir sobreajuste (distância entre as curvas) por técnica.

**Hardware.** Cada job registra CPU, RAM, GPU e versões. Tempos de PDI e MLP medidos na CPU; os da CNN, na GPU. Tempos de máquinas diferentes não são comparados.

**Declaração de evidência já vista:** os resultados anteriores, com menos técnicas e sementes, indicavam classificação majoritariamente neutra e efeito forte do PDI na segmentação. As regras acima foram escritas conhecendo isso. As curvas, o custo e a MLP por técnica ainda não tinham sido vistos.

**Complemento de 2026-11-06, antes de rodar a Fase B:** filtro `mean`, `median` ou `gauss_lpf` "reduz o ruído" quando PSNR e SSIM aumentam em relação a sem filtro (Wilcoxon por frame; Holm entre filtros em cada tipo de ruído). Na tabela única, o melhor método de cada técnica é escolhido pelo Dice no treino, e os números mostrados são do teste. O `segmentation_best.json` não é alterado por este estudo. Tempo medido em um único núcleo da CPU, na imagem em tamanho de trabalho. A segmentação clássica e as técnicas usam OpenCV/NumPy, que rodam na CPU; a GPU fica para as redes.

**Complemento de 2026-11-06, custo por etapa:** todo comando e cada etapa interna dos estudos registram tempo, CPU (núcleos usados pelo processo e uso do sistema), RAM, GPU (utilização, memória, potência e energia) e variação de disco em `results/tables/custo/etapas.csv`, com a máquina em `hardware.json`. Cada treino da CNN guarda o mesmo no `metrics.json`. O uso de CPU do sistema mostra se havia outros programas competindo durante a medição. Etapas antigas sem medição serão refeitas ou declaradas como não medidas.

**Correção de 2026-11-06, GPU no custo:** a leitura do nvidia-smi é da placa inteira. O estudo do PDI roda só na CPU, mas registrou 28% de GPU e 31 Wh; com ele parado, a placa seguia em 39% e 31 W, por causa do painel LaringoMonitor e de outros programas. As colunas passam a se chamar `gpu_sistema_*`, e `gpu_usada_pelo_processo` indica se a etapa usou a GPU (lista de computação do nvidia-smi, conferida com um processo real). Só nesse caso há `gpu_energia_liquida_wh`, que desconta a potência lida no início da etapa. As 5 linhas já medidas foram migradas sem alterar os valores, com `gpu_usada_pelo_processo = não`, porque o estudo não aloca memória na GPU.

**Complemento de 2026-11-06, antes de rodar a Fase C:** detalhes que as regras acima não fixavam.
- **Implementação:** MLPs em PyTorch, e não no scikit-learn, para ter as curvas por época e usar a validação interna do fold no early stopping. Isso também elimina a limitação registrada em 2026-11-04 (cópias sobreamostradas nos dois lados da validação). As classes são tratadas por entropia cruzada ponderada, como na CNN.
- **Classificação:** CPU, 1 núcleo; AdamW 1e-3, cosine, lote 32, até 300 épocas, paciência 20 no F1-macro por vídeo; z-score só com o treino; 3 sementes × 5 folds. Máscara: método e parâmetros de `segmentation_best.json` (crescimento de região, tol 20), aplicados diretamente na imagem de cada técnica, sem o pré-processamento CLAHE desse arquivo. Só assim `raw` fica de fato sem realce.
- **Referência em pixels:** um vetor por frame; o vídeo recebe a média das probabilidades, como na CNN.
- **Comparações:** cada técnica contra `raw` (veredito principal) e contra a referência em pixels (`classificacao_vs_pixels.csv`, com Holm próprio).
- **Segmentação:** GPU, porque a inferência passa por todos os pixels; isso ajusta a frase "tempos de MLP medidos na CPU" acima, valendo só para a classificação. Arquitetura 225–64–32–2; até 200 pixels de glote e 200 de fora por frame de treino, fora do pad; AdamW 1e-3, cosine, lote 4096, até 30 épocas, paciência 5 no Dice de até 100 frames de validação; limiar 0,5. Uma semente × 5 folds, porque o teste estatístico é pareado por vídeo.
- **MLP × clássico:** Dice por vídeo contra o melhor método clássico da mesma técnica, com Wilcoxon e Holm sobre as 9 técnicas: "MLP melhor", "clássico melhor" ou "empate".
- **Custo:** na classificação, o PDI por frame inclui técnica, máscara e 18 características em 1 núcleo; na segmentação, só a técnica. Custo total como definido acima. Os tempos de CPU e de GPU não são somados como se fossem o mesmo recurso sem indicação: a tabela mostra as parcelas.
- **H6:** queda da acurácia balanceada ao permutar cada família, repetida 10 vezes, em `raw` (5 folds, semente 0), pela regra de P6.
- **Caches:** as 9 técnicas são regravadas agora, com medição; o conteúdo é conferido contra os caches antigos de E00 e E03.

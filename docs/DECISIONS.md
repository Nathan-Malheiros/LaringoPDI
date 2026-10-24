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

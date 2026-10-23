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

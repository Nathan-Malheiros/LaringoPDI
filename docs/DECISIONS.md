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

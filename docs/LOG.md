# Diário das fases

Resumos curtos (≤ 10 linhas) ao fim de cada fase. Detalhes em `DECISIONS.md` e nos logs.

## Fase 0 — Setup (2026-10-22)
- Estrutura `src/laringopdi`, `configs/base.yaml` + `experiments.yaml` (E00–E16, estágios, C1–C4), `requirements.txt`, `.gitignore`, README, venv Python 3.11 com torch CPU.
- CLI única `python -m laringopdi <cmd>` com os 14 comandos; implementados: `index`, `splits`, `eda`, `gallery`, `cache`.
- Adiantado: dedup pHash+SSIM, StratifiedGroupKFold com checagem de vazamento, as 16 técnicas de PDI + `quality`, EDA e galeria.
- Testes sintéticos: 29 passam, 1 pulado (splits reais).
- **Bloqueio:** o dataset não está em `data/raw/laryngoscope8/`. O download é só pelo Baidu Pan, com login; não há espelho. O aceite (`index.csv` com 3057 linhas, contagem por classe, ID de paciente) fica pendente.

## Troca para o BAGLS — código adaptado (2026-10-22)
- ESPECIFICACAO.md reescrito para o BAGLS; decisões em DECISIONS.md.
- Código: `BaglsAdapter` (lê os metadados de `.meta`/`.json`, a ser confirmado nos arquivos reais) e o stub `Laryngoscope8Adapter`.
- Amostragem de até 10 frames por vídeo, grupos por vídeo (pHash + SSIM), folds por vídeo e cache com a máscara GT.
- PDI em cinza: saem `lab`, `hsv` e `vessel`; `spec` por limiar de intensidade (piso 200); entram `roi_gt`, `roi` (melhor segmentação clássica) e M2 = [cinza, máscara, bordas].
- Novo `pdi/segmentation.py` com 5 métodos e métricas Dice, IoU, HD95 e falha; `seg-study` escolhe os parâmetros só no treino de cada fold.
- Testes: 45 passam, 1 pulado (splits reais), incluindo um teste de fumaça ponta a ponta num BAGLS sintético.
- **Bloqueio:** o BAGLS ainda não está em `data/raw/bagls/`; o Passo 0 (inspeção dos metadados reais) aguarda o download.

## Passo 0 — BAGLS real (2026-10-23)
- Zips validados por md5 e extraídos em `data/raw/bagls/{training,test}`: 59.250 frames, 640 vídeos.
- `.meta` confirmado (JSON). `Video Id` único entre os splits; não há ID de sujeito. Adaptador corrigido para `Color` booleano.
- O status é texto livre (55 valores) → `configs/label_map.yaml` (regras ordenadas) + tabela de revisão em `dataset_report.md`.
- Vídeos: saudável 377 · funcional 156 · estrutural **47** · desconhecido 53 · outros 7. Tarefas A 377×203; B 377×47; C 377/156/47. RGB: 25.
- **Parada:** a família estrutural tem menos de 50 vídeos (regra do ESPECIFICACAO.md) → decisão de método com o usuário antes dos splits.

## Fase 1 — Dados (2026-10-23)
- Tarefa B ampliada para 54 vídeos estruturais (decisão do usuário). Vídeos: saudável 377 · funcional 156 · estrutural 54 · excluídos 53.
- Amostragem: 5.870 frames de 587 vídeos (10 por vídeo, só frames com glote visível) em `data/splits/frames.json`.
- Quase-duplicatas de vídeo: 1 par candidato (pHash ≤ 6), nenhum confirmado por SSIM ≥ 0,85 → 587 grupos de 1 vídeo.
- Folds (StratifiedGroupKFold por vídeo, estratificados por C): teste com ~118 vídeos por fold (75–76 saudáveis / 31–32 funcionais / 10–11 estruturais); validação interna com ~67 vídeos.
- O teste de vazamento nos splits reais passa. `dataset_report.md` regenerado com figuras e com a tabela status → família.
- Histórico do Git limpo de arquivos do BAGLS (filter-repo + force push, aprovado).

## Fase 3 — Segmentação clássica (2026-10-23)
- `seg-study`: 5 métodos × 4 pré-processamentos × grade de parâmetros em 5.870 frames; ajuste no treino de cada fold, avaliação no teste.
- Melhor: crescimento de região + CLAHE (Dice 0,614 · falha 2,1% · HD95 9,7 px), seguido de crescimento de região sobre o `raw` (0,467) e sobre a homomórfica (0,440). O Otsu global é linha de base (Dice 0,02).
- Watershed corrigido (núcleos por componente) antes da escolha final; o vencedor não mudou (DECISIONS.md).
- Saídas: `results/tables/segmentation.csv`, `segmentation_summary.csv`, `segmentation_best.json`; a figura `segmentation_examples.png` (local, não versionada) vai para a revisão de Marcos Mendes e Tales.

## Fase 2 — PDI (2026-10-23)
- 14 técnicas em cinza + 5 segmentadores. `docs/pdi_tecnicas.md` é gerado das docstrings pelo `gallery`. A galeria (grade técnica × família; passo a passo de `spec`, `homo` com espectro e H(u,v), e `roi` × `roi_gt`) fica em `docs/figures/`, local e não versionada, para revisão do Marcos Vinicius.
- Caches E00–E13: 5.870 × 224 × 224 × 3 cada (~12 GB em `data/processed/cache`, fora do Git).
- `results/pdi_quality.csv`:
  - `spec` reduz a fração especular de 1,7% para 0,9%;
  - HE estoura o contraste (entropia 7,7; contraste RMS 0,31) frente ao CLAHE (6,5; 0,22);
  - a homomórfica eleva a nitidez (variância do Laplaciano 130 → 1.185);
  - fallback da ROI: 2,1% (`roi`), 0,3% (`clahe>roi`), 0% (`roi_gt`).
- Limitação: as métricas de qualidade usam o canal 0. Em `edges` e M2, que começam pelo cinza, elas coincidem com E00.
- Testes: 55 passam (inclui o teste de vazamento nos splits reais).

## Fases 4–6 (parte local, CPU) — 2026-10-23
- **Fase 4 (código):**
  - `models.py` (ResNet-18 ImageNet, SimpleCNN 32/64/128/256);
  - `train.py`: jobs com early stopping no F1-macro **por vídeo da validação interna**, só aug geométrico, AMP na GPU, jobs concluídos pulados;
  - `shards` (stage1: 140 jobs em 5 shards de 2–3 experimentos cada) e `notebooks/colab_runner.ipynb`.
  - Medição local: SimpleCNN ≈ 121 s/época na CPU. A medição no T4 está **pendente** (humano).
- **Fase 5 (probe em CPU):** 14 exp × 2 tarefas × 5 folds = 140 jobs.
  - F1 por vídeo no teste: A 0,61–0,67; B 0,72–0,80.
  - Top-4 provisório, escolhido **só pela validação**: E05 `bilat`, E04 `homo`, E08 `roi`, E06 `median`. É definitivo só depois da SimpleCNN rápida no Colab.
- **Fase 6 (clássico):** C1–C5 com características por vídeo (média e desvio de 187 características × 10 frames).
  - F1 por vídeo, A: C1 0,642 · C2 0,662 · C3 0,664 · C4 0,664 · C5 0,658.
  - F1 por vídeo, B: C1 0,762 · C2 0,737 · C3 0,757 · C4 0,811 · C5 0,762.
  - Competitivo com o probe da ResNet (E00: A 0,656; B 0,748).
- **Estatística (`report`):**
  - Probe: todos os 26 vereditos são **neutros** (McNemar OOF com Holm + IC bootstrap). Os ΔF1 ficam em ±0,03, com ICs cruzando 0.
  - H4 (segmentação): o efeito do realce depende do par método × pré-processamento. Crescimento de região + CLAHE dá +0,148 de Dice e watershed/Otsu+morfologia + homomórfica dão +0,08 a +0,10; CLAHE piora os métodos baseados em limiar global.
  - Os p muito pequenos vêm de ~587 vídeos pareados; no Otsu o efeito é desprezível (|Δ| ≤ 0,002) apesar do p.
- Correção de bug: a chave `classic` do `experiments.yaml` sobrescrevia os hiperparâmetros do `base.yaml`; foi renomeada para `classic_experiments`.

## Fases 7–8 (código) — 2026-10-23
- `gradcam`: energia do Grad-CAM dentro da glote (GT de cada cache) e sobre os reflexos (máscara `spec` do E00, só em pipelines sem recorte). Usa `model.pt` da semente 0, agora salvo pelo `train`. Roda quando o Estágio 2 terminar.
- `report`: vereditos (McNemar OOF + Holm + IC bootstrap; Wilcoxon reportado), teste de H4, clássico × CNN, tabelas CSV + LaTeX e `docs/hipoteses.md`.
- App Streamlit: upload → etapas do PDI → segmentação → métricas → predição + Grad-CAM (quando houver pesos), com o aviso obrigatório.
- `tests/test_e2e.py`: ponta a ponta em CPU em 64 s. Suíte completa: 64 testes passando.

## Telemetria do treino + bug de AMP (2026-10-23)
- `live.py`: telemetria só de observação para o painel LaringoMonitor: lote atual, Grad-CAM com a caixa da glote, ativação e ‖ΔW‖/‖W‖ por camada, filtros, mapas de ativação e PDI passo a passo. Liga só com `results/live/enabled`.
- A telemetria revelou augmentation corrompida sob AMP (DECISIONS.md). Corrigido, com teste de regressão; os 22 runs afetados foram para quarentena; o Estágio 1 foi reiniciado.

## Estágio 1 na GPU + relatório técnico (2026-10-23)
- Estágio 1 refeito após a correção de AMP: 140 jobs da SimpleCNN rápida (RTX 3060, ~80 s/job). Top-4 pela validação: E04 `homo`, E11 `lbp`, E05 `bilat`, E10 `edges`.
- `report`: probe e SimpleCNN somam **52/52 vereditos neutros** (nenhuma técnica muda o F1 por vídeo de forma significativa no Estágio 1).
- 1ª tentativa do Estágio 2 encerrada por falta de RAM, ainda na busca gulosa do E14 (nada foi gravado). `build_cache` agora grava em blocos (memmap).
- Relatório técnico em PDF (`tools/relatorio_pdf.py`): 16 p. na versão pública (Git) e 19 p. na completa (Área de Trabalho). Próximo passo em `docs/PROXIMOS_PASSOS.md`.

## Estágio 2 em execução + fechamento automático (2026-10-24)
- E14 guloso = `homo>lbp` (F1 de validação do probe 0,712). Estágio 2 (360 jobs) rodando em `tools/stage2_chain.py`.
- `tools/final_chain.py` espera o Estágio 2 e roda `stage2_hyp` (E01/E07/E08, 1 semente, para H1 e H5) → `gradcam` → `report` → PDFs.
- Regra de veredito do Estágio 2 fixada antes dos resultados (DECISIONS.md): OOF por ensemble das sementes; IC exclui 0 e McNemar ou Wilcoxon (≥ 10 pares) com Holm.
- `report` agora conclui H1–H6 de forma explícita (`docs/hipoteses.md` + `results/tables/hypotheses.json`): H2 com confronto direto contra o HE, H3 por Wilcoxon sobre os ΔF1, H5 com `roi` × `roi_gt`, H6 com C1 × CNN pareado.
- Com os dados atuais (Estágio 1): H2 e H3 não suportadas; H4, H5 e H6 parciais; H1 aguarda o Grad-CAM.
- PDF: seção do Estágio 2, figura de atenção e tabela de hipóteses entram sozinhas quando os dados existirem. 67 testes passando.


## Fechamento automático (tools/final_chain.py, 2026-10-25 00:47)
- Estágio 2 + stage2_hyp concluídos; gradcam, report e PDFs regenerados. Conclusões (`docs/hipoteses.md`):
  - H1: parcialmente suportada
  - H2: não suportada (sem diferença detectável na classificação)
  - H3: não suportada
  - H4: parcialmente suportada (depende do par método × realce: 6 melhoram, 9 pioram; melhor = region_grow + clahe, Δ +0.148)
  - H5: parcialmente suportada
  - H6: parcialmente suportada (A: competitiva; B: competitiva)
- **Resultado principal:** 420/420 jobs. Estágio 2: os **32 vereditos são neutros** (ResNet-18 e SimpleCNN, A e B). O maior ganho bruto (E04 `homo`, ResNet-18, A: 0,643 → 0,680) não passou no teste. Somando os estágios, nenhuma técnica de PDI mudou a classificação de forma significativa; o PDI só teve efeito na segmentação (H4).
- **H1 (atenção):** a energia sobre os reflexos já fica no nível do acaso (~3%) no E00. O `spec` a reduz na ResNet-18 (0,036 → 0,034), mas a mediana por frame é 0, porque a maioria dos frames quase não tem reflexo.
- **Pendências:** revisar `docs/hipoteses.md` e o PDF (cópia completa na Área de Trabalho do OneDrive); Fase 8 (app com os pesos `fold0_seed0`); o `gradcam` roda na CPU (2,5 h) e poderia usar a GPU.

## Etapa 2 — Bloco 0: preparação (2026-11-04)
- Branch `etapa2/fechamento` criada a partir da `main` (já atualizada). Os dois documentos do plano, ainda sem commit, entraram **na branch**, para não adicionar nada à `main`.
- Verificado: `data/raw/bagls/{training,test}`; caches E00, E08 e E13 (o nome leva o hash do pipeline: `data/processed/cache/E00_ff5107162d.*`, e não uma pasta `E00/` como o plano supunha); pesos `fold0_seed0/model.pt` de E00 e E04 (A e B × resnet18 e simplecnn); `attention_frames.csv.gz` com as colunas `exp,model,task,fold,path,video_id,att_glottis,area_glottis,att_spec,area_spec`.
- Cobertura da atenção: 9 experimentos × 2 modelos × 2 tarefas, 5.870 frames e 587 vídeos em cada combinação. Frames com área glótica 0: 27 (0 no E07; 185 no E08).
- `pytest`: 67 passando.

## Etapa 2 — Bloco 1: técnicas de aula (2026-11-04)
- `stretch` e `gauss_lpf` em `pdi/enhance.py`; `mean` em `pdi/filters.py`; `saturated_frac` em `pdi/quality.py` (coluna nova, sem mexer nas existentes). As técnicas novas ficam fora da matriz E00–E14 e os hashes dos caches não mudam.
- `homo_transfer` e `gauss_lpf_transfer` agora compartilham `_freq_d2`, com a fórmula inalterada. Conferido antes da refatoração: `homo` reproduz o frame 0 do cache E04 byte a byte.
- `docs/pdi_tecnicas.md` regenerado só pelo `pdi_doc`, sem refazer a galeria.

## Etapa 2 — Bloco 2: redução de ruído (P2) (2026-11-04)
- `python -m laringopdi denoise-study`: 300 frames, 1 por vídeo (193 saudáveis, 80 funcionais, 27 estruturais, na proporção dos vídeos), 4 ruídos × 9 filtros. Saídas: `denoise_summary.csv/.tex`, `denoise_p2.csv`, `denoise_sample.json`, `docs/figures/relatorio/denoise.png`; `denoise_frames.csv.gz` e `denoise_exemplo.png` ficam fora do Git.
- **P2, sal e pimenta (mediana 3 × média 3):** 5%: PSNR 41,4 × 25,9 dB, SSIM 0,97 × 0,44. 10%: PSNR 39,2 × 22,5 dB, SSIM 0,97 × 0,27. Os 4 testes têm p_Holm < 0,001.
- **Preservação de borda (F1 do Canny), sal e pimenta:** mediana 3 fica em 0,67 e 0,62; média 3 em 0,11 e 0,07.
- **Contraste gaussiano, contra a expectativa do plano:** a mediana 3 não perde da média 3 (σ=10: 35,3 × 35,0 dB; σ=20: 30,22 × 30,20 dB, praticamente igual). Só o passa-baixa σ40 supera a mediana 3, e só em σ=20 (31,3 × 30,2 dB; SSIM 0,64 × 0,56).
- **Limitação:** a referência é o próprio frame do BAGLS, que já tem ruído; o PSNR mede fidelidade ao original, não a um sinal verdadeiro.
- `seg_per_fold` foi extraída do `seg_study`, sem mudança de comportamento (conferida no Bloco 3 contra o `segmentation_summary.csv`).

## Etapa 2 — Bloco 3: realce (P1) (2026-11-04)
- `python -m laringopdi enhance-study`: qualidade nos 5.870 frames (`enhance_quality.csv`) e segmentação com `stretch` e `he` somados a raw/clahe/homo/bilat (`segmentation_etapa1*.csv`, `seg_p1.csv`, `realce_classif_context.csv`, `docs/figures/relatorio/realce.png`). `segmentation_best.json` e `segmentation_summary.csv` não foram tocados.
- **Conferências:** o novo resumo reproduz o `segmentation_summary.csv` (20 linhas, diferença 0,0) e 60 frames recalculados batem com os salvos (2.280 linhas, diferença 0,0).
- **CLAHE × HE no Dice por vídeo:** `region_grow` 0,614 × 0,476 e `adaptive` 0,150 × 0,136 favorecem o CLAHE; `otsu_morph` 0,172 × 0,368, `watershed` 0,182 × 0,386 e `otsu` (0,023 × 0,024) favorecem o HE. Todos com p_Holm < 0,01. O efeito **depende do método**.
- **Fração saturada:** raw 18,1%, HE 18,2%, alargamento 18,9%, CLAHE 1,5%, homomórfica 1,7%. **Ressalva:** quase tudo é preto que já existia no frame (raw já tem 16,6% em 0). O HE só soma 0,1 ponto; o CLAHE fica baixo por tirar esses pixels do zero. A condição (b) do P1 é atendida, mas não demonstra o "estouro" suposto na hipótese.
- **Classificação (contexto):** E02 × E03 sem diferença (McNemar p ≥ 0,076, todos os IC contêm 0).

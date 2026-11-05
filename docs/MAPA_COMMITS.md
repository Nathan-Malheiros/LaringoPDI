# Mapa de commits (histórico reescrito em 2026-11-05)

O histórico foi reescrito ao renomear o repositório de PDIZip para LaringoPDI: só as mensagens mudaram; o conteúdo de cada commit é idêntico. Registros gerados antes disso (`git_sha` em `data/splits/folds.json`, `results/tables/*.json` e cabeçalhos em `docs/`) citam o SHA antigo. Use esta tabela para encontrar o commit correspondente.

| SHA antigo | SHA novo | Commit |
|---|---|---|
| c96d1f7 | 7daf276 | Fase 0: estrutura, configs, CLI, PDI e testes sintéticos |
| 4da9a12 | 5159f3b | Registra indisponibilidade do Laryngoscope8 |
| dcdd219 | 81237e8 | Troca do dataset para o BAGLS: adaptador, PDI em cinza, segmentação clássica |
| 65111c1 | 4ada811 | docs: registra estado e pendências do dataset (retomar amanhã) |
| 3feb4b0 | a84d4ed | chore: ajusta .gitignore, README (dados BAGLS) e pendências de colaboradores |
| a9a65d3 | 8c6ad88 | Passo 0: índice do BAGLS real, label_map por palavras-chave e relatório do dataset |
| c938262 | 9b43e50 | Remove do rastreamento as figuras com frames do BAGLS e o index.csv |
| da17496 | 1e7ed06 | Fase 1: amostragem de frames, grupos por vídeo, folds e EDA do BAGLS |
| 34c340b | d10f748 | Fase 3: estudo de segmentação clássica da glote; watershed corrigido; roi fixado em crescimento de região + CLAHE |
| f6d3683 | c537e28 | Fase 2: galeria, doc das técnicas gerado das docstrings, caches E00–E13 e qualidade do PDI |
| ff34af4 | 573c3cb | Fase 4: train/probe/shards/classic, métricas frame+vídeo, notebook do Colab |
| b941be6 | 4125d49 | Fases 5–6 (CPU): probe, braço clássico C1–C5, estatística (vereditos, H4) e hipoteses.md |
| f1a7869 | 715c143 | Fases 7–8 (código): Grad-CAM e atenção, report, app Streamlit, teste ponta a ponta |
| 13938f6 | a11eb63 | Corrige augmentation sob AMP (pixels saturados); telemetria visual do treino (live.py) |
| 339c4ca | 7cf7ff3 | Telemetria do treino: acerto por lote, taxa de aprendizado, matriz de confusão, confiança, ajuste por camada e erros mais confiantes |
| 9f32f70 | b3fd153 | Estágio 1 concluído (140 jobs); E14 guloso aceita no máximo uma representação multicanal |
| d248c71 | 07b0a0e | Relatório técnico em PDF (versão pública), próximos passos, cache em blocos e script do Estágio 2 |
| bb6aa94 | 82fd8ba | Estágio 2 em execução (E14 = homo>lbp); regra de veredito com sementes; conclusões H1–H6 no report; seção do Estágio 2 no PDF; cadeia final automática |
| a8cb07c | ccb1bd3 | final_chain: registra conclusões no LOG e faz commit local ao terminar |
| 4d26931 | 30c3f39 | Atalho para pausar/retomar o treino (tools/pausar_retomar.bat) |
| a9188df | 6dc99dc | Estágio 2 e fechamento: vereditos, Grad-CAM, conclusões H1–H6 e relatório (gerado por tools/final_chain.py) |
| bf330a1 | 15a7595 | LOG: resumo do fechamento do Estágio 2 |
| 67585e0 | c138003 | Corrige caminho no pausar_retomar.bat |
| d3723af | 041de80 | App redesenhado com foco em PDI: pipeline com histogramas, laboratório das 14 técnicas, domínio da frequência, segmentação e Grad-CAM com o veredito do estudo |
| 6d7421b | 5cf3005 | Limpeza de disco: zips do BAGLS e caches da busca gulosa |
| 5458b38 | 8ed09e9 | Relatório completo em PDF (fichas por técnica, Grad-CAM real, capturas do app); app em tema claro e com ?aba= |
| 41e2457 | fe8495f | Logo da UNIFOR (usada pelo relatório completo) |
| c5c8ff3 | 19962d4 | Plano de fechamento da Etapa 2 |
| 693ca7b | cb2170f | Bloco 0: verificação do ambiente para o fechamento da Etapa 2 |
| 327e545 | 8ee49a1 | PDI: alargamento, média e passa-baixa gaussiano (técnicas de aula da Etapa 1) |
| 29dd0e1 | 7eb108b | Pré-registro das regras de veredito de P1–P6 (antes dos números novos) |
| e96e47d | 9fa9e92 | Estudo de ruído (P2): mediana 3 supera a média 3 em sal e pimenta (PSNR 41,4 × 25,9 dB a 5%; SSIM 0,97 × 0,44), p_Holm < 0,001 |
| d85c0af | 3c52762 | Estudo de realce (P1): CLAHE × HE depende do método (region_grow 0,614 × 0,476; otsu_morph 0,172 × 0,368) |
| a0f8c96 | e14dc5e | MLP 18–32–16–2 (C6): F1 0,712 na Tarefa B, IC do ΔF1 contra a ResNet-18 [−0,104; +0,009] |
| d97debf | 3c1acfd | Atenção na glote (P5): 5 combinações sobem e 9 caem; sensibilidade para Grad-CAM sem energia |
| 4d2c922 | a2820a6 | Tempo de processamento: de 1,1 a 28,5 ms por pipeline; ResNet-18 em 24,5 ms na CPU |
| 1117df8 | 96b5ce2 | Relatório P1–P6 e tabela de mudanças em relação à Etapa 1 |
| 42e495d | 8175ec5 | App: abas Técnicas de aula e Resultados, MLP-18 na classificação e roteiro da demonstração |
| 72ca7a8 | 02cd9e9 | Relatório técnico com o fechamento da Etapa 1, README, pacote de entrega e dependências no pyproject |
| 0b0730e | 9295aa3 | Reprodução em clone limpo: teste de splits pula sem o groups.csv; registro do Bloco 9 |
| 768ea9e | 28523ae | Relatórios: nomes completos e matrículas da equipe nas capas |
| f010e5a | 1d95056 | Pré-registro: entrega fiel à Etapa 1 e complementos (adaptativa + morfologia, sensibilidade, CLAHE na ResNet-18) |
| 1b38b8e | 4924012 | Adaptativa + morfologia (Dice 0,280, sem diferença contra Otsu + morfologia); app em 5 abas na ordem dos slides; H1–H6 da apostila |
| 611958a | b49a1dc | Técnicas de aula na ResNet-18, relatório curto e apostila da Etapa 2 |
| 7842211 | 6de8e27 | Registro da reprodução do pacote da Etapa 2 |

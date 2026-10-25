# Estado do projeto — 2026-10-24

> Relatório técnico: `docs/relatorio_tecnico.pdf` (público). **Próximo passo executável: `docs/PROXIMOS_PASSOS.md`.**

## Concluído
- **Passo 0:** BAGLS (Zenodo 3762320) validado por md5 e extraído em `data/raw/bagls/{training,test}`: 59.250 frames, 640 vídeos.
- **Rótulos:** `configs/label_map.yaml` (texto livre → família). Vídeos: saudável 377 · funcional 156 · estrutural 54 · excluídos 53.
- **Fase 1:** 5.870 frames (10 por vídeo), 587 vídeos, 5 folds por vídeo sem vazamento; `docs/dataset_report.md`.
- **Fase 3:** segmentação clássica; melhor = crescimento de região + CLAHE (Dice 0,614); `roi` e `m2` fixados.
- **Fase 2:** galeria, `docs/pdi_tecnicas.md`, caches E00–E13, `results/pdi_quality.csv`.

- **Fase 4 (código), Fase 5 (probe) e Fase 6 (clássico C1–C5)** rodaram na CPU; `report` gera `docs/hipoteses.md` + tabelas em `results/tables/`.
- **Fases 7–8 (código):** `gradcam`, `report`, app Streamlit e `tests/test_e2e.py` prontos; 64 testes passando. Falta rodar com os modelos do Colab.

- **Estágio 1** (probe + SimpleCNN rápida, 140 + 140 jobs) concluído na RTX 3060; top-4 pela validação: E04, E11, E05, E10.

## Em execução (2026-10-24, sem intervenção)
1. `tools/stage2_chain.py`: E14 guloso = **`homo>lbp`** (F1 de validação 0,712) → cache → 360 jobs do Estágio 2 (~16 h). Log: `logs/stage2_chain.log` e `logs/stage2_shard*.log`.
2. `tools/final_chain.py` (espera o item 1): completa jobs faltantes → `stage2_hyp` (E01/E07/E08, 1 semente, 60 jobs, para H1 e H5) → `gradcam` → `report` → PDFs (público em `docs/`, completo na Área de Trabalho). Log: `logs/final_chain.log`.
3. Ambos aparecem no painel LaringoMonitor (`stage2`, `final`). Se a máquina reiniciar, basta rodar de novo os dois comandos do README: os jobs concluídos são pulados.

## Depois disso
- Revisar `docs/hipoteses.md` (conclusões de H1–H6 geradas pelo `report`) e o PDF.
- Fase 8: `streamlit run app\streamlit_app.py` com os pesos do Estágio 2; reprodução a partir de um clone limpo.

## Pendências humanas
- Revisar as figuras locais em `docs/figures/` (não versionadas: contêm frames do BAGLS):
  - Marcos Vinicius: `gallery_*.png`;
  - Marcos Mendes e Tales: `segmentation_examples.png`.
- Revisar a tabela "texto → família" em `docs/dataset_report.md`, sobretudo os casos ambíguos (`None (…)`, "swelling", cicatriz/sulco).
- E-mail aos autores do Laryngoscope8; avisar a Prof.ª Lyndainês sobre a troca de dataset.
- Adicionar Marcos Vinicius, Pedro Nobre, Marcos Mendes e Tales como colaboradores do repositório `PDIZip` (preciso dos usernames do GitHub).

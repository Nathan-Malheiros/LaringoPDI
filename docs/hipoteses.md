# Hipóteses — estado da evidência

Gerado por `python -m laringopdi report` (git ff34af4). Não edite à mão. Modelos com resultados: probe. Hipóteses pré-registradas em DECISIONS.md (2026-10-22).

## H1 — `spec` mantém/melhora o F1 e reduz a atenção nos reflexos
- Probe (E01 vs E00): A ΔF1=+0.004 [IC -0.038; +0.049], neutra; B ΔF1=+0.025 [IC -0.032; +0.076], neutra.
- Atenção (Grad-CAM): **pendente** (requer CNNs do Colab).

## H2 — CLAHE/homomórfica > HE global (classificação e segmentação)
- Probe E02: A ΔF1=-0.048 [IC -0.097; +0.002], neutra; B ΔF1=+0.003 [IC -0.052; +0.057], neutra.
- Probe E03: A ΔF1=-0.028 [IC -0.079; +0.020], neutra; B ΔF1=+0.003 [IC -0.047; +0.049], neutra.
- Probe E04: A ΔF1=+0.000 [IC -0.051; +0.052], neutra; B ΔF1=+0.014 [IC -0.043; +0.067], neutra.
- Segmentação, Δ Dice vs raw (crescimento de região): clahe +0.148, homo -0.027; HE não faz parte do estudo de segmentação.

## H3 — PDI ajuda mais a SimpleCNN do que a ResNet
- **Pendente** (Estágios 1–2 no Colab).

## H4 — o realce melhora o Dice da segmentação clássica

| método | pré | Dice raw | Dice pré | Δ | p (Holm) | veredito |
|---|---|---:|---:|---:|---:|---|
| adaptive | bilat | 0.253 | 0.321 | +0.068 | 1e-58 | melhora |
| adaptive | clahe | 0.253 | 0.149 | -0.104 | 5.8e-64 | piora |
| adaptive | homo | 0.253 | 0.103 | -0.150 | 1.2e-93 | piora |
| otsu | bilat | 0.023 | 0.022 | -0.001 | 6e-63 | piora |
| otsu | clahe | 0.023 | 0.023 | +0.001 | 1.4e-22 | melhora |
| otsu | homo | 0.023 | 0.025 | +0.002 | 1.1e-24 | melhora |
| otsu_morph | bilat | 0.324 | 0.299 | -0.025 | 6.7e-14 | piora |
| otsu_morph | clahe | 0.324 | 0.172 | -0.153 | 7e-35 | piora |
| otsu_morph | homo | 0.324 | 0.407 | +0.083 | 1.4e-06 | melhora |
| region_grow | bilat | 0.467 | 0.358 | -0.109 | 5.4e-76 | piora |
| region_grow | clahe | 0.467 | 0.614 | +0.148 | 8.7e-30 | melhora |
| region_grow | homo | 0.467 | 0.440 | -0.027 | 0.00047 | piora |
| watershed | bilat | 0.334 | 0.315 | -0.018 | 3.5e-11 | piora |
| watershed | clahe | 0.334 | 0.182 | -0.152 | 6.9e-32 | piora |
| watershed | homo | 0.334 | 0.437 | +0.103 | 5.9e-10 | melhora |

## H5 — ROI ajuda mais em B do que em A; `roi` se aproxima de `roi_gt`
- Probe E08 (`roi`): A ΔF1=-0.031 [IC -0.084; +0.025], neutra; B ΔF1=-0.018 [IC -0.079; +0.039], neutra.
- Probe E07 (`roi_gt`): A ΔF1=-0.025 [IC -0.077; +0.029], neutra; B ΔF1=+0.021 [IC -0.050; +0.082], neutra.

## H6 — forma glótica + MLP competitiva com a CNN em A, inferior em B

| exp | modelo | tarefa | F1 vídeo | dp | AUC |
|---|---|---|---:|---:|---:|
| C1 | mlp | A | 0.642 | 0.049 | 0.707 |
| C1 | mlp | B | 0.762 | 0.065 | 0.834 |
| C2 | mlp | A | 0.662 | 0.027 | 0.716 |
| C2 | mlp | B | 0.737 | 0.074 | 0.845 |
| C3 | svm | A | 0.664 | 0.044 | 0.712 |
| C3 | svm | B | 0.757 | 0.075 | 0.823 |
| C4 | rf | A | 0.664 | 0.041 | 0.747 |
| C4 | rf | B | 0.811 | 0.039 | 0.865 |
| C5 | mlp | A | 0.658 | 0.081 | 0.718 |
| C5 | mlp | B | 0.762 | 0.077 | 0.829 |
| E00 | probe | A | 0.656 | 0.020 | 0.721 |
| E00 | probe | B | 0.748 | 0.073 | 0.846 |

- A comparação final usa a ResNet-18 do Estágio 2 (pendente); o probe é só referência provisória.

> Com 5 pares (probe: 5 folds × 1 semente), o Wilcoxon não alcança p < 0,05; os vereditos usam McNemar OOF (Holm) + IC bootstrap.
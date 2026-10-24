# Hipóteses — estado da evidência

Gerado por `python -m laringopdi report` (git d248c71). Não edite à mão. Modelos com resultados: probe (ResNet congelada), SimpleCNN rápida. Hipóteses pré-registradas e regras de veredito em DECISIONS.md.

Veredito de cada técnica contra o E00 (F1-macro por vídeo): **melhora/piora** só se o IC 95% bootstrap excluir 0 e o McNemar OOF (ou, com ≥ 10 pares fold×semente, o Wilcoxon) corrigido por Holm der p < 0,05; senão **neutra**.

## Resumo dos vereditos

| modelo | tarefa | melhora | neutra | piora |
|---|---|---:|---:|---:|
| probe (ResNet congelada) | A | 0 | 13 | 0 |
| probe (ResNet congelada) | B | 0 | 13 | 0 |
| SimpleCNN rápida | A | 0 | 13 | 0 |
| SimpleCNN rápida | B | 0 | 13 | 0 |

## H1 — `spec` mantém ou melhora o F1 e reduz a atenção sobre os reflexos
- F1 (E01 × E00):
  - probe (ResNet congelada): A ΔF1=+0.004 [IC -0.038; +0.049], 5 pares, **neutra**; B ΔF1=+0.025 [IC -0.032; +0.076], 5 pares, **neutra**.
  - SimpleCNN rápida: A ΔF1=-0.026 [IC -0.057; +0.005], 5 pares, **neutra**; B ΔF1=-0.015 [IC -0.038; +0.005], 5 pares, **neutra**.
- Atenção (Grad-CAM): sem pares E00 × E01 (`python -m laringopdi gradcam` com pesos do E01).

## H2 — CLAHE/homomórfica > HE global (classificação e segmentação)
- Contra o E00:
  - probe (ResNet congelada): A ΔF1=-0.048 [IC -0.097; +0.002], 5 pares, **neutra**; B ΔF1=+0.003 [IC -0.052; +0.057], 5 pares, **neutra**.
  - SimpleCNN rápida: A ΔF1=-0.021 [IC -0.055; +0.014], 5 pares, **neutra**; B ΔF1=-0.009 [IC -0.050; +0.032], 5 pares, **neutra**.
  - probe (ResNet congelada): A ΔF1=-0.028 [IC -0.079; +0.020], 5 pares, **neutra**; B ΔF1=+0.003 [IC -0.047; +0.049], 5 pares, **neutra**.
  - SimpleCNN rápida: A ΔF1=-0.009 [IC -0.045; +0.026], 5 pares, **neutra**; B ΔF1=+0.018 [IC -0.013; +0.050], 5 pares, **neutra**.
  - probe (ResNet congelada): A ΔF1=+0.000 [IC -0.051; +0.052], 5 pares, **neutra**; B ΔF1=+0.014 [IC -0.043; +0.067], 5 pares, **neutra**.
  - SimpleCNN rápida: A ΔF1=-0.032 [IC -0.070; +0.007], 5 pares, **neutra**; B ΔF1=+0.027 [IC -0.008; +0.064], 5 pares, **neutra**.
- Confronto direto contra o HE (E02 como baseline):
  - probe (ResNet congelada) A, clahe − he: ΔF1=+0.019 [IC -0.029; +0.069], **neutra**.
  - probe (ResNet congelada) A, homo − he: ΔF1=+0.048 [IC -0.001; +0.098], **neutra**.
  - probe (ResNet congelada) B, clahe − he: ΔF1=+0.000 [IC -0.043; +0.043], **neutra**.
  - probe (ResNet congelada) B, homo − he: ΔF1=+0.011 [IC -0.056; +0.075], **neutra**.
  - SimpleCNN rápida A, clahe − he: ΔF1=+0.012 [IC -0.025; +0.050], **neutra**.
  - SimpleCNN rápida A, homo − he: ΔF1=-0.011 [IC -0.048; +0.026], **neutra**.
  - SimpleCNN rápida B, clahe − he: ΔF1=+0.027 [IC -0.018; +0.069], **neutra**.
  - SimpleCNN rápida B, homo − he: ΔF1=+0.036 [IC -0.004; +0.078], **neutra**.
- Segmentação (crescimento de região, método escolhido): Δ Dice vs raw — clahe +0.148, homo -0.027. O HE não entrou no estudo de segmentação (escopo da Seção 5).

## H3 — o PDI ajuda mais a SimpleCNN do que a ResNet
Diferença de ΔF1 (técnica − E00) entre a SimpleCNN e a ResNet nas mesmas técnicas × tarefas (Wilcoxon pareado).
- SimpleCNN rápida × probe (ResNet congelada): ΔF1 médio -0.012 × -0.002; mediana da diferença -0.006; p=0.35; 26 pares técnica×tarefa.

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

- Com ~587 vídeos pareados, p pequenos acompanham efeitos desprezíveis (Otsu, |Δ| ≤ 0,002); leia o Δ.

## H5 — a ROI ajuda mais na Tarefa B do que na A, e `roi` se aproxima de `roi_gt`
- E08 (`roi`) × E00:
  - probe (ResNet congelada): A ΔF1=-0.031 [IC -0.084; +0.025], 5 pares, **neutra**; B ΔF1=-0.018 [IC -0.079; +0.039], 5 pares, **neutra**.
  - SimpleCNN rápida: A ΔF1=-0.027 [IC -0.068; +0.013], 5 pares, **neutra**; B ΔF1=-0.094 [IC -0.155; -0.037], 5 pares, **neutra**.
- E07 (`roi_gt`) × E00:
  - probe (ResNet congelada): A ΔF1=-0.025 [IC -0.077; +0.029], 5 pares, **neutra**; B ΔF1=+0.021 [IC -0.050; +0.082], 5 pares, **neutra**.
  - SimpleCNN rápida: A ΔF1=-0.039 [IC -0.082; +0.004], 5 pares, **neutra**; B ΔF1=-0.131 [IC -0.206; -0.054], 5 pares, **neutra**.
- `roi` × `roi_gt` (E07 como referência):
  - probe (ResNet congelada): ΔF1(B) − ΔF1(A) do `roi` = +0.013.
  - probe (ResNet congelada) A: roi_gt → roi F1 0.632 → 0.626, Δ=-0.006 [IC -0.051; +0.042], McNemar p=0.65.
  - probe (ResNet congelada) B: roi_gt → roi F1 0.769 → 0.730, Δ=-0.039 [IC -0.102; +0.022], McNemar p=0.26.
  - SimpleCNN rápida: ΔF1(B) − ΔF1(A) do `roi` = -0.067.
  - SimpleCNN rápida A: roi_gt → roi F1 0.611 → 0.622, Δ=+0.012 [IC -0.026; +0.048], McNemar p=0.4.
  - SimpleCNN rápida B: roi_gt → roi F1 0.612 → 0.649, Δ=+0.037 [IC -0.042; +0.110], McNemar p=0.91.
- Ressalva: "ajuda mais em B" compara efeitos que, isoladamente, podem ser neutros; "se aproxima" é a leitura de um IC que cruza 0 (ausência de diferença detectável, não equivalência demonstrada).

## H6 — forma glótica + MLP é competitiva com a CNN na Tarefa A e inferior na B
Referência CNN: E00 com probe (ResNet congelada). Competitiva = IC do ΔF1 (clássico − CNN) contém 0 ou é > 0; inferior = IC < 0.

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

- C1 (forma + textura + MLP) × E00 probe (ResNet congelada), A: F1 0.657 → 0.642, Δ=-0.015 [IC -0.067; +0.037], McNemar p=0.39.
- C1 (forma + textura + MLP) × E00 probe (ResNet congelada), B: F1 0.749 → 0.764, Δ=+0.016 [IC -0.050; +0.080], McNemar p=0.88.
- Provisório: a comparação final usa a ResNet-18 do Estágio 2.

## Conclusões

| hipótese | conclusão |
|---|---|
| H1 | não testada (atenção) |
| H2 | não suportada (sem diferença detectável na classificação) |
| H3 | não suportada |
| H4 | parcialmente suportada (depende do par método × realce: 6 melhoram, 9 pioram; melhor = region_grow + clahe, Δ +0.148) |
| H5 | parcialmente suportada |
| H6 | parcialmente suportada (A: competitiva; B: competitiva) |

> Resultado neutro é resultado: com ~118 vídeos de teste por fold e efeitos de ±0,03 no F1, o estudo só tem poder para efeitos grandes. As conclusões descrevem a evidência deste desenho, não um efeito nulo provado.
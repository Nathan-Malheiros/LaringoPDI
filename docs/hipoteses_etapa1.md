# Hipóteses da apresentação da Etapa 1 (P1–P6)

> Gerado por `python -m laringopdi etapa1-report` (git 4d2c922). **Não edite à mão.**
> As regras foram fixadas em `docs/DECISIONS.md` (2026-11-04) antes de gerar os números novos. P1, P3, P4 e P6 já tinham evidência parcial vista, como a entrada registra.

## Resumo

| ID | Hipótese | Veredito |
|---|---|---|
| P1 | CLAHE > equalização global | **parcialmente confirmada** |
| P2 | mediana > média em sal e pimenta | **confirmada** |
| P3 | limiar adaptativo > global (Otsu) | **confirmada** |
| P4 | MLP ≈ CNN em saudável × lesão estrutural (Tarefa B) | **confirmada** |
| P5 | PDI na entrada aumenta a atenção da CNN na glote | **parcialmente confirmada** |
| P6 | a forma da glote é o grupo de características mais importante | **refutada** |

## Detalhe

### P1: CLAHE > equalização global

- **Por que esperávamos isso:** a iluminação é desigual e a equalização global "estoura" o tecido.
- **Regra pré-registrada:** Confirmada: CLAHE > HE no Dice do `region_grow` (p_Holm < 0,05), em pelo menos 3 dos 5 métodos, e fração saturada de HE > CLAHE. Parcial: vale só parte disso. Refutada: HE > CLAHE com significância no `region_grow`.
- **Evidência:** `region_grow`: Dice CLAHE 0,614 × HE 0,476, p_Holm < 0,001. CLAHE vence com significância em 2 de 5 métodos; HE vence em 3. Fração saturada: HE 18,20% × CLAHE 1,53%. Ressalva: a imagem sem realce já tem 18,10% de pixels em 0 ou 255, quase todos pretos (fundo do frame); o HE quase não acrescenta nada a isso, e o CLAHE fica baixo porque tira esses pixels do zero (16,57% → 0,00%). Logo, a condição (b) é atendida, mas não demonstra que o HE "estoura" o tecido. Contexto, fora do veredito: na classificação do Estágio 1, ΔF1 (CLAHE − HE) varia de 0,000 a 0,027, com McNemar mínimo 0,075.
- **Veredito: parcialmente confirmada.**

### P2: mediana > média em sal e pimenta

- **Por que esperávamos isso:** a mediana ignora valores extremos e preserva a borda.
- **Regra pré-registrada:** Confirmada: mediana 3×3 > média 3×3 em PSNR e SSIM nas duas densidades (p_Holm < 0,05). Parcial: vale só em parte. Refutada: a média vence com significância e a mediana em nenhum teste.
- **Evidência:** sal e pimenta 5%: PSNR mediana 41,4 × média 25,9 dB, SSIM 0,97 × 0,44; sal e pimenta 10%: PSNR mediana 39,2 × média 22,5 dB, SSIM 0,97 × 0,27. Mediana vence com significância em 4 de 4 testes (maior p_Holm < 0,001; n = 300 frames). A referência é o próprio frame, que já tem ruído.
- **Veredito: confirmada.**

### P3: limiar adaptativo > global (Otsu)

- **Por que esperávamos isso:** o reflexo desloca o limiar único.
- **Regra pré-registrada:** Confirmada: Dice por vídeo de `adaptive` > `otsu` sobre `raw`, com p < 0,05 e Δ ≥ 0,05. Parcial: p < 0,05 com Δ < 0,05. Refutada: Otsu > adaptativo com p < 0,05.
- **Evidência:** Sobre `raw`, por vídeo (n = 587): `adaptive` 0,253 × `otsu` 0,023, Δ = 0,230, p < 0,001. Nuance, fora do veredito: `otsu_morph` 0,324 e `region_grow` 0,467. O Otsu puro praticamente não acerta a glote, e o adaptativo também é fraco; o resultado mostra a superioridade do adaptativo, mas não testa se a causa é o reflexo.
- **Veredito: confirmada.**

### P4: MLP ≈ CNN em saudável × lesão estrutural (Tarefa B)

- **Por que esperávamos isso:** a lesão deforma a glote.
- **Regra pré-registrada:** Confirmada ("competitiva"): o IC 95% do ΔF1 por vídeo (C6 − ResNet-18 E00) contém 0 ou está acima de 0. Refutada: está todo abaixo de 0.
- **Evidência:** C6, Tarefa B (431 vídeos): F1 0,713 × ResNet-18 0,762, ΔF1 -0,049 [IC 95% -0,104; 0,009], McNemar p < 0,001. Largura do IC: 0,114; "competitiva" não significa equivalente. O veredito segue a regra do IC, mas o McNemar aponta outra direção: a acurácia é 0,842 (C6) × 0,903 (ResNet-18), e só a ResNet-18 acerta 40 vídeos em que o C6 erra, contra 14 no sentido inverso. Trate esta confirmação como frágil. Contexto: C6, Tarefa A: F1 0,613 × ResNet-18 0,623, ΔF1 -0,010 [IC 95% -0,060; 0,041], McNemar p 0,385. Extensão C1 (187 características), Tarefa B: F1 0,764 × ResNet-18 0,762, ΔF1 0,002 [IC 95% -0,057; 0,063], McNemar p 0,405. Comparação contra a semente 0 da ResNet-18, porque o C6 tem uma semente só.
- **Veredito: confirmada.**

### P5: PDI na entrada aumenta a atenção da CNN na glote

- **Por que esperávamos isso:** menos reflexo, mais contraste.
- **Regra pré-registrada:** Das 24 combinações técnica × modelo × tarefa sem recorte: confirmada se ≥ 50% sobem (p_Holm < 0,05) e nenhuma cai; parcial se 1 ou mais sobem; refutada se nenhuma sobe e alguma cai; inconclusiva se nenhuma é significativa.
- **Evidência:** Sobem com significância: 5 de 24 combinações (E04, E10, E14); caem: 9 (E01, E04, E05, E11, E14). Por modelo, sobem resnet18: 3, simplecnn: 2; caem resnet18: 1, simplecnn: 8. Mais combinações caem do que sobem, e o efeito muda de sinal conforme técnica, modelo e tarefa. Razão média no E00: 1,17 (1 = acaso). Desvio do pré-registro: frames com mapa de energia zero (até 0,0% na ResNet-18 e 38,9% na SimpleCNN) têm razão indefinida e saíram da análise principal, decisão tomada depois de ver os resultados, e por isso a queda no número de vídeos pareados da SimpleCNN (mínimo 299). Sensibilidade, contando esses frames como atenção 0: sobem 4, caem 10. Pixels da glote com X = 0, que a medida exclui da energia: E00 0,00%, E01 0,00%, E04 0,38%, E05 0,00%, E10 0,00%, E11 0,00%, E14 0,00%. Com recorte, à parte e com a ressalva da área: E07 1,51 (E00 1,20); E08 1,29 (E00 1,18).
- **Veredito: parcialmente confirmada.**

### P6: a forma da glote é o grupo de características mais importante

- **Por que esperávamos isso:** nódulo e pólipo mudam o contorno.
- **Regra pré-registrada:** No C6, a queda média por permutação da família forma é a maior das 3 (e positiva): confirmada nas duas tarefas; parcial em uma; refutada em nenhuma.
- **Evidência:** C6, queda média por permutação. Tarefa A: forma 0,027, intensidade 0,027, textura 0,036. Tarefa B: forma 0,017, intensidade 0,037, textura 0,061. Complementar, média de C1–C5 (187 características): Tarefa A: hog 0,147, tex 0,016, shape 0,008, edge 0,001; Tarefa B: hog 0,256, tex 0,032, shape 0,000, edge -0,008.
- **Veredito: refutada.**

## Correspondência com as hipóteses internas (`docs/hipoteses.md`)

| Apresentada | Interna |
|---|---|
| P1 | H2 |
| P2 | sem par |
| P3 | H4, em parte |
| P4 | H6, em parte |
| P5 | H1 (atenção) |
| P6 | sem par |

# Técnicas de PDI — LaringoPDI (BAGLS, escala de cinza)

Gerado por `python -m laringopdi gallery` a partir das docstrings (git da17496). Não edite à mão.

Toda imagem é convertida para cinza e reduzida a lado maior ≤ 512 px antes do PDI; ao final, letterbox 224×224 e replicação para 3 canais. Parâmetros vêm da literatura, nunca do classificador.

## Matriz de experimentos

| ID | Pipeline | Descrição |
|---|---|---|
| E00 | `raw` | baseline (cinza replicado 3×) |
| E01 | `spec` | remoção de reflexo especular |
| E02 | `he` | equalização global |
| E03 | `clahe` | CLAHE |
| E04 | `homo` | filtragem homomórfica (FFT) |
| E05 | `bilat` | filtro bilateral |
| E06 | `median` | filtro da mediana |
| E07 | `roi_gt` | recorte pela máscara de ground truth (limite superior) |
| E08 | `roi` | recorte pela segmentação clássica (melhor método da Fase 3) |
| E09 | `clahe>roi` | CLAHE + recorte clássico |
| E10 | `edges` | [cinza, Sobel, Canny] |
| E11 | `lbp` | LBP uniforme |
| E12 | `m1` | M1 = [cinza com CLAHE, bordas, LBP] |
| E13 | `m2` | M2 = [cinza, máscara glótica clássica, bordas] |
| E14 | `None` | combinação gulosa (definida no Estágio 2 só pela validação interna) |

## Técnicas

### `raw`

- **O que faz:** converte para cinza, redimensiona preservando a proporção, completa com preto até 224×224 e replica o canal 3× para a CNN.
- **Por que importa na laringoscopia:** baseline sem PDI; não distorce a geometria da glote e das pregas vocais.
- **Parâmetros padrão:** size=224 (entrada da ResNet-18).
- **Referência:** He et al., 2016 (ResNet); prática padrão de letterboxing.

### `spec`

- **O que faz:** detecta reflexos especulares por limiar de intensidade (Otsu sobre a cauda do histograma acima do percentil 90, com piso 200), dilata a máscara e preenche por inpainting de Telea.
- **Por que importa na laringoscopia:** a mucosa úmida das pregas reflete a luz do endoscópio; os brilhos saturados escondem a borda livre e podem ser confundidos com lesões esbranquiçadas.
- **Parâmetros padrão:** percentil 90, piso 200, dilatação 2 px, raio de inpaint 5.
- **Referência:** Arnold et al., 2010 (reflexos em endoscopia); Telea, 2004.

### `he`

- **O que faz:** equalização global do histograma.
- **Por que importa na laringoscopia:** corrige subexposição global, mas amplifica ruído e satura os reflexos; serve de contraste para H2 (CLAHE/homomórfica > HE).
- **Parâmetros padrão:** nenhum.
- **Referência:** Gonzalez & Woods, Digital Image Processing, cap. 3.

### `clahe`

- **O que faz:** equalização adaptativa por blocos com limite de contraste (CLAHE).
- **Por que importa na laringoscopia:** a iluminação do endoscópio é central e decai para as bordas; o CLAHE realça o contraste local (borda livre das pregas, contorno da glote) sem estourar o centro.
- **Parâmetros padrão:** clip 2,0, grade 8×8.
- **Referência:** Zuiderveld, 1994; Pizer et al., 1987.

### `homo`

- **O que faz:** filtragem homomórfica — log → FFT → filtro gaussiano de alta ênfase H(u,v) = (γH−γL)(1−e^{−cD²/D0²})+γL → IFFT → exp → reescala pelos percentis 0,5–99,5.
- **Por que importa na laringoscopia:** separa a iluminação (baixa frequência, atenuada por γL) da refletância (alta frequência, realçada por γH); compensa o vinhetamento da fonte de luz pontual.
- **Parâmetros padrão:** γL=0,5; γH=1,75; D0=30 (lado maior ≤ 512); c=1.
- **Referência:** Gonzalez & Woods, Digital Image Processing, seção 4.9.

### `bilat`

- **O que faz:** filtro bilateral — média ponderada por proximidade espacial e de intensidade.
- **Por que importa na laringoscopia:** suaviza ruído de sensor e compressão JPEG da mucosa preservando as bordas das pregas e das lesões (que o filtro gaussiano borraria).
- **Parâmetros padrão:** d=9, σintensidade=75, σespaço=75.
- **Referência:** Tomasi & Manduchi, 1998.

### `median`

- **O que faz:** filtro da mediana em janela ksize×ksize por canal.
- **Por que importa na laringoscopia:** remove ruído impulsivo e pequenos pontos brilhantes, mas pode apagar detalhes finos (vasos, pequenos nódulos) — o experimento mede esse custo.
- **Parâmetros padrão:** ksize=5.
- **Referência:** Gonzalez & Woods, Digital Image Processing, seção 3.5.

### `fov`

- **O que faz:** segmenta o campo de visão (limiar = min(Otsu, 40) → maior componente → preenchimento de buracos), zera o fundo e recorta a caixa envolvente. Sem borda escura (máscara fora de 20–97% da imagem) vira identidade.
- **Por que importa na laringoscopia:** vinheta e moldura preta não são anatomia e podem virar atalho para a rede.
- **Parâmetros padrão:** blur 5 (mediana), teto do limiar 40.
- **Referência:** Otsu, 1979; Bernal et al., 2012 (pré-processamento endoscópico).

### `roi_gt`

- **O que faz:** recorta em torno da máscara de glote do especialista (ground truth), com margem de 50% e lado mínimo de 30% da imagem.
- **Por que importa na laringoscopia:** é o **limite superior** do recorte — mostra quanto um ROI perfeito ajudaria; não é utilizável na prática, pois exige anotação.
- **Parâmetros padrão:** margin 0,5; min_side 0,3; fallback central 60% (máscara vazia).
- **Referência:** Gómez et al., 2020 (BAGLS).

### `roi`

- **O que faz:** segmenta a glote com o melhor método clássico da Fase 3 (resolvido em `results/tables/segmentation_best.json`) e recorta em torno dela com margem de 50%. Sem glote detectada: recorte central de 60%.
- **Por que importa na laringoscopia:** as lesões estruturais (nódulos, pólipos, cistos, edema) ficam na borda livre das pregas, ao redor da fenda glótica; o recorte descarta o contexto irrelevante.
- **Parâmetros padrão:** método = melhor da Fase 3; margin 0,5; min_side 0,3.
- **Referência:** Otsu, 1979; Gonzalez & Woods (morfologia); Gómez et al., 2020.

### `edges`

- **O que faz:** empilha [cinza, magnitude de Sobel, Canny automático] em 3 canais (limiares do Canny = (1∓σ)·mediana, sobre a imagem suavizada).
- **Por que importa na laringoscopia:** o contorno da glote e a borda livre das pregas (irregular no carcinoma, abaulada no pólipo e no nódulo) são critérios do laringologista; testa se explicitar bordas ajuda a rede.
- **Parâmetros padrão:** blur gaussiano 5, σ=0,33.
- **Referência:** Sobel, 1968; Canny, 1986; regra da mediana (Rosebrock, 2015).

### `lbp`

- **O que faz:** mapa LBP uniforme (P=8 vizinhos, raio 1).
- **Por que importa na laringoscopia:** codifica a micro-textura da mucosa (lisa × rugosa/queratinizada) e é invariante a mudanças monotônicas de brilho da iluminação.
- **Parâmetros padrão:** P=8, R=1, "uniform".
- **Referência:** Ojala, Pietikäinen & Mäenpää, 2002.

### `m1`

- **O que faz:** fusão multicanal M1 = [cinza com CLAHE, magnitude de Sobel, LBP uniforme].
- **Por que importa na laringoscopia:** entrega à rede, em 3 canais compatíveis com a ResNet, luminância realçada + forma + textura — descritores clássicos do PDI.
- **Parâmetros padrão:** CLAHE 2,0/8×8; Sobel 3×3; LBP P=8, R=1.
- **Referência:** combinação das referências de `clahe`, `edges` e `lbp`.

### `m2`

- **O que faz:** fusão multicanal M2 = [cinza, máscara glótica da segmentação clássica, magnitude de Sobel].
- **Por que importa na laringoscopia:** dá à rede, explicitamente, a forma da glote (que reflete o fechamento glótico e a presença de massas na borda das pregas) junto da imagem e das bordas.
- **Parâmetros padrão:** segmentação = melhor método da Fase 3; Sobel 3×3 após blur 5.
- **Referência:** combinação das referências de `roi` e `edges`.

## Segmentação clássica da glote (Fase 3)

Melhor configuração atual (usada por `roi` e `m2`): **otsu_morph** sobre **raw**, parâmetros `{}` (padrão, antes da Fase 3).

### `otsu`

- **O que faz:** limiar global de Otsu sobre a imagem suavizada; pixels escuros = glote.
- **Por que importa na laringoscopia:** a glote aberta é a região mais escura do frame (via aérea sem retorno de luz); é o método mais simples, sem nenhuma restrição espacial — linha de base da segmentação.
- **Parâmetros padrão:** blur gaussiano 5. · grade de ajuste: `[{"blur": 3}, {"blur": 5}, {"blur": 9}]`
- **Referência:** Otsu, 1979.

### `adaptive`

- **O que faz:** limiar adaptativo gaussiano (pixel escuro em relação à vizinhança block×block).
- **Por que importa na laringoscopia:** tolera o decaimento da iluminação do endoscópio do centro para a borda, que faz o Otsu global incluir cantos escuros.
- **Parâmetros padrão:** bloco 51, C=5, blur 5. · grade de ajuste: `[{"block": 31, "C": 5}, {"block": 31, "C": 10}, {"block": 51, "C": 5}, {"block": 51, "C": 10}, {"block": 101, "C": 5}, {"block": 101, "C": 10}]`
- **Referência:** Gonzalez & Woods, Digital Image Processing, seção 10.3.

### `otsu_morph`

- **O que faz:** Otsu + abertura (remove ruído) + fechamento (une a fenda) + componente conexo central de área plausível + preenchimento de buracos.
- **Por que importa na laringoscopia:** incorpora o conhecimento anatômico de que a glote é uma região única, alongada e central no frame da videoendoscopia.
- **Parâmetros padrão:** blur 5, abertura 3, fechamento 7 (elípticos). · grade de ajuste: `[{"open_k": 3, "close_k": 7}, {"open_k": 3, "close_k": 15}, {"open_k": 5, "close_k": 7}, {"open_k": 5, "close_k": 15}]`
- **Referência:** Otsu, 1979; Serra, 1982 (morfologia matemática).

### `region_grow`

- **O que faz:** crescimento de região (flood fill de faixa fixa ±tol) a partir do pixel mais escuro da janela central.
- **Por que importa na laringoscopia:** a glote é um "poço" escuro e conexo; crescer a partir da semente evita capturar regiões escuras desconexas (sombras, bordas).
- **Parâmetros padrão:** tol=10 níveis de cinza, blur 5, janela central de 50%. · grade de ajuste: `[{"tol": 5}, {"tol": 10}, {"tol": 20}]`
- **Referência:** Adams & Bischof, 1994 (seeded region growing).

### `watershed`

- **O que faz:** watershed com marcadores — Otsu escuro → transformada de distância → núcleos (dist > fg_frac·máx) como marcadores de objeto, dilatação como fundo certo — e escolhe a bacia central.
- **Por que importa na laringoscopia:** separa a glote de regiões escuras vizinhas que tocam nela (ventrículos, sombras das aritenoides), algo que o limiar sozinho não consegue.
- **Parâmetros padrão:** blur 5, fg_frac 0,5. · grade de ajuste: `[{"fg_frac": 0.3}, {"fg_frac": 0.5}, {"fg_frac": 0.7}]`
- **Referência:** Vincent & Soille, 1991; Meyer, 1994 (watershed com marcadores).

## Figuras para revisão (Marcos Vinicius; segmentação: Marcos Mendes e Tales)

Geradas localmente em `docs/figures/` e **não versionadas** (contêm frames do BAGLS): `gallery_grid.png`, `gallery_spec_steps.png`, `gallery_homo.png`, `gallery_roi.png`, `segmentation_examples.png`.

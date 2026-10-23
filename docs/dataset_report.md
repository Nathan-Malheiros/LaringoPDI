# Relatório do dataset — BAGLS

Gerado por `python -m laringopdi eda` (git 3feb4b0). Não edite à mão.

## Contagens

- Frames: **59250** (esperado ~59.250); vídeos: **640** (esperado 640)
- Vídeos RGB: 25 (todos são convertidos para cinza)
- Frames com máscara: 59250; com glote visível (máscara não vazia): 55286
- ID de sujeito nos metadados: não (só ID de vídeo)
- Split oficial (vídeos): {'training': 570, 'test': 70} — ignorado nos folds

### Por família

| family | videos | frames |
|---|---|---|
| healthy | 377 | 35500 |
| functional | 156 | 15150 |
| structural | 47 | 4300 |
| other | 7 | 600 |
| unknown | 53 | 3700 |

### Texto do status → família (revisão humana; regras em `configs/label_map.yaml`)

| disorder_raw | family | videos |
|---|---|---|
| `Muscle tension dysphonia` | functional | 105 |
| `Muscle tension dysphonia with M. thyroarythaenoideus atrophy` | functional | 10 |
| `Muscle tension dysphonia with vocal insufficiency and M. thyroarythaenoideus atrophy` | functional | 8 |
| `Vocal insufficiency and M. thyroarythaenoideus atrophy` | functional | 6 |
| `functional` | functional | 6 |
| `Muscle tension dysphonia with vocal insufficiency` | functional | 3 |
| `Laryngitis` | functional | 2 |
| `Paresis` | functional | 2 |
| `Posterior insufficient glottic closure` | functional | 2 |
| `laryngitis` | functional | 2 |
| `paresis` | functional | 2 |
| `spasmodic dysphonia` | functional | 2 |
| `Anterior insufficient glottic closure` | functional | 1 |
| `Hourglass-shaped insufficient glottic closure (high phonation)` | functional | 1 |
| `Insufficient glottic closure: posterior` | functional | 1 |
| `Irregular vibration anterior and middle portion of both vocal folds` | functional | 1 |
| `Posterior insufficient glottic closure (high phonation)` | functional | 1 |
| `Vocal fold atrophy with complete glottic closure` | functional | 1 |
| `healthy` | healthy | 376 |
| `None (higher phonation)` | healthy | 1 |
| `Minimal anterior mucosal irregularity right vocal fold ` | other | 2 |
| `scar` | other | 2 |
| `Bilateral vergeture with bowed insufficient glottic closure` | other | 1 |
| `Bowed/oval insufficient glottic closure (left more than right); stifness right vocal fold` | other | 1 |
| `Fibrotic/stiff right vocal fold` | other | 1 |
| `Muscle tension dysphonia with nodules` | structural | 10 |
| `Muscle tension dysphonia with contact granuloma` | structural | 4 |
| `edema` | structural | 4 |
| `Polyp` | structural | 3 |
| `Bilateral Reinke's edema (right more than left)` | structural | 2 |
| `Midmembranous swelling (mainly right vocal fold) with hourglass-shaped insufficient glottic closure` | structural | 2 |
| `Reinke's edema right vocal fold with hourglass-shaped insufficient glottic closure` | structural | 2 |
| `polyp` | structural | 2 |
| `Bilateral Reinke's edema` | structural | 1 |
| `Cyst vocal fold left (posterior insufficient glottic closure) ` | structural | 1 |
| `Cyst vocal fold left with posterior insufficient glottic closure (high phonation) ` | structural | 1 |
| `Extreme polipoid pendulating edema obstructing almost complete glottis` | structural | 1 |
| `Lateral-posterior vocal fold cyst` | structural | 1 |
| `Lateral-posterior vocal fold cyst (high phonation)` | structural | 1 |
| `Post-resection of extreme polipoid pendulating edema right vocal folds; extreme polipoid pendulating edema obstructing almost complete glottis left vocal folds` | structural | 1 |
| `Reinke's edema right vocal fold (earlier it was bilateral)` | structural | 1 |
| `Right vocal fold polyp (with hourglass-shaped insufficient glottic closure)` | structural | 1 |
| `Right vocal fold polyp with contraleral edema ` | structural | 1 |
| `Vocal fold nodules` | structural | 1 |
| `Vocal fold nodules (high phonation)` | structural | 1 |
| `Vocal fold nodules with hourglass insufficient glottic closure and teleangectasia` | structural | 1 |
| `Vocal insufficiency and contact granuloma` | structural | 1 |
| `carcinoma` | structural | 1 |
| `cyst` | structural | 1 |
| `leucoplacia` | structural | 1 |
| `papilloma` | structural | 1 |
| *(vazio)* | unknown | 50 |
| `None (half year after diagnosis of small vocal nodules)` | unknown | 1 |
| `None (half year post-phonomicrosurgery for polipoid mid-membranous lesions)` | unknown | 1 |
| `None (one year after presumption of a pseudocyst/sulcus in left vocal fold)` | unknown | 1 |

### Tarefas (vídeos)

| tarefa | healthy | disorder | structural | functional |
|---|---|---|---|---|
| A | 377 | 203.0 | 0.0 | 0.0 |
| B | 377 | 0.0 | 47.0 | 0.0 |
| C | 377 | 0.0 | 47.0 | 156.0 |

![visão geral](figures/eda_overview.png)

![resoluções](figures/eda_sizes.png)

![amostras](figures/eda_samples.png)

### Intensidade (até 3 frames por vídeo)

| family | mean | spec_frac | seg_area |
|---|---|---|---|
| functional | 56.2852 | 0.0168 | 0.0067 |
| healthy | 55.1673 | 0.0163 | 0.008 |
| structural | 89.2533 | 0.0149 | 0.0091 |

### Por `meta_camera` (vídeos)

| meta_camera | functional | healthy | other | structural | unknown |
|---|---|---|---|---|---|
| FASTCAM Mini AX100 type 540K-C-16GB | 0 | 0 | 0 | 0 | 20 |
| HERS 5562 Endocam Wolf | 34 | 28 | 0 | 17 | 0 |
| KayPentax HSV 9700 (Photron) | 0 | 16 | 0 | 0 | 0 |
| KayPentax HSV 9710 (Photron) | 122 | 333 | 7 | 30 | 3 |
| Phantom v210 | 0 | 0 | 0 | 0 | 30 |

## Famílias com menos de 50 vídeos

Sinalizadas: structural. **Decisão de método pendente com o usuário.**

# Estado do projeto — retomar em 2026-10-23

## Fase atual
- Código adaptado ao BAGLS (commit `dcdd219`). O **Passo 0 ainda NÃO rodou**.
- Testes: 45 passam, 1 pulado (splits reais), incluindo o teste de fumaça ponta a ponta sobre um BAGLS sintético.

## Dataset
- BAGLS, Zenodo record 3762320, licença CC BY-NC-SA 4.0.
- `C:\Users\natha\Downloads\test.zip`: completo, 207.236.572 bytes, md5 `4ed918e5adfd0b8e0bebdc3aa1265afd`.
- `C:\Users\natha\Downloads\training.zip`: baixando durante a noite (~3,5 GB, ~0,5 MB/s). **Antes de extrair**, confira o tamanho final e o md5 contra zenodo.org/records/3762320. Se estiver incompleto, rebaixe.
- `raw.zip` (13,4 GB): propositalmente não baixado.

## Formato confirmado
- `<id>.png`, `<id>_seg.png`, `<id>.meta` (JSON) com `"Video Id"`, `"Color"`, `"Subject disorder status"`, `"Age range (yrs)"`, `"Subject sex"`.
- O `BaglsAdapter` já lê `.meta` em JSON e normaliza as chaves (`video_id`, `color`, `subject_disorder_status` …). Falta validar contra os arquivos reais.

## Achado importante: status em texto livre
- `"Subject disorder status"` é **texto livre**. Exemplos: `'healthy'`, `''`, `'Vocal fold nodules (high phonation)'`, `"Reinke's edema right vocal fold ..."`, `'None (half year post-phonomicrosurgery ...)'`.
- No split de teste: 70 vídeos, sendo 29 healthy, 20 com status vazio e 11 coloridos.
- **A fazer:** criar `configs/label_map.yaml` por palavras-chave (saudável / funcional / estrutural / excluir), substituindo a seção `disorders` de `configs/base.yaml` hoje usada por `data.disorder_family`. Incluir no `dataset_report.md` a tabela "texto → rótulo" para revisão humana.
- Atenção a casos como `'None (pós-cirurgia …)'`: hoje caem em `other` (excluído) porque `unknown` só reconhece correspondência exata. Decidir explicitamente no label_map. Pós-operatório não é saudável por padrão.
- Ver também `Color`: hoje o adaptador trata texto contendo "rgb"/"colo" como RGB. Se o valor for booleano, ajustar.

## Próximos passos (em ordem)
1. Extrair `test.zip` e `training.zip` em `data/raw/bagls/{test,training}/`, depois de validar o md5 do training.
2. Criar o `label_map.yaml` e validar o leitor em 2–3 `.meta` reais.
3. Rodar o Passo 0 completo (`index`, contagens reais por status em `docs/dataset_report.md`).
4. Fases 1–3 automáticas, conforme o ESPECIFICACAO.md (`splits` → `eda` → `seg-study` → `gallery` → `cache`).

## Pendências humanas
- E-mail aos autores do Laryngoscope8.
- Avisar a Prof.ª Lyndainês sobre a troca de dataset.
- Adicionar Marcos Vinicius, Pedro Nobre, Marcos Mendes e Tales como colaboradores do repositório privado `PDIZip` (preciso dos usernames do GitHub).

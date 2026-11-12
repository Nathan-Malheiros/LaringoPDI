"""Roteiro do app: cenas da demonstração, textos de cada passo e o glossário das métricas.

Fica separado do app para que `tools/gerar_amostras.py` saiba quais frames as cenas usam."""

# Cada cena ajusta o passo, o frame e os controles. Todos os frames são de vídeos do teste do fold 0.
CENAS = [
    {"titulo": "Realce num frame escuro", "nota": "Compare o original com o CLAHE: a glote aparece sem estourar o tecido ao redor.",
     "passo": 0, "frame": "training/12188.png", "exp": "E03", "task": "A", "realce": "clahe"},
    {"titulo": "Média × mediana no ruído", "nota": "Sal e pimenta a 5%: a mediana limpa os pontos; a média só os espalha.",
     "passo": 1, "frame": "training/12068.png", "filtro_vista": "ruido", "ruido_tipo": "sal e pimenta", "ruido_dens": 5, "k_media": 3, "k_mediana": 3},
    {"titulo": "Segmentação passo a passo", "nota": "Arraste a etapa: limiar adaptativo, abertura, fechamento e, no fim, só a glote.",
     "passo": 2, "frame": "training/27990.png", "seg_clahe": False, "seg_vista": "passos", "seg_etapa": "entrada"},
    {"titulo": "O CLAHE salva a segmentação", "nota": "Frame escuro: com o CLAHE, o crescimento de região vai de Dice 0,00 a 0,95.",
     "passo": 2, "frame": "training/12188.png", "seg_clahe": True, "seg_vista": "metodos", "exp": "E03"},
    {"titulo": "Quando o CLAHE atrapalha", "nota": "Frame claro: com o CLAHE, o Dice cai de 0,81 para 0,12. Realce não é ganho garantido.",
     "passo": 2, "frame": "training/11116.png", "seg_clahe": True, "seg_vista": "metodos"},
    {"titulo": "Lesão estrutural: MLP × CNN", "nota": "Tarefa B com CLAHE. O Grad-CAM mostra onde a rede olhou para decidir.",
     "passo": 3, "frame": "training/7957.png", "exp": "E03", "task": "B"},
    {"titulo": "Vídeo saudável, sem PDI", "nota": "Troque a técnica na barra lateral: a decisão quase não muda.",
     "passo": 3, "frame": "training/12068.png", "exp": "E00", "task": "A"},
    {"titulo": "Resultados do estudo", "nota": "Hipóteses, veredito por técnica, custo e o teste sem as bordas pretas.", "passo": 4},
]

PASSOS = [
    {"nome": "Realce", "porque": "Frames de laringoscopia saem escuros e com contraste irregular. O realce redistribui os tons de cinza "
                                 "para a glote e as pregas vocais ficarem visíveis."},
    {"nome": "Filtragem e Fourier", "porque": "A câmera de alta velocidade gera ruído. Filtros suavizam a imagem, mas também podem "
                                              "borrar a borda das pregas, que é justamente o que importa."},
    {"nome": "Segmentação da glote", "porque": "Separar a glote (a abertura entre as pregas) do resto é o primeiro passo para medir "
                                               "forma e simetria. A máscara do especialista é o gabarito."},
    {"nome": "Classificação", "porque": "Dois modelos dizem se o vídeo tende a saudável ou alterado: uma rede pequena com 18 medidas "
                                        "e uma ResNet-18 que vê a imagem inteira. É apoio à decisão, não diagnóstico."},
    {"nome": "Resultados do estudo", "porque": "Os números do estudo completo: 587 vídeos, 5 folds, 3 sementes. Não dependem do frame escolhido."},
]

AJUDA = {
    "rms": "Quanto os tons de cinza variam em torno da média. Mais alto = mais contraste.",
    "entropia": "Quantidade de informação nos tons de cinza, em bits. Mais alta = tons mais bem distribuídos.",
    "psnr": "Quanto a imagem se parece com o original, em decibéis. Mais alto = mais parecida; acima de ~30 dB a diferença quase não se vê.",
    "ssim": "Semelhança de estrutura (bordas e texturas) com o original, de 0 a 1. 1 = idêntica.",
    "dice": "Quanto a área encontrada coincide com a marcada pelo especialista, de 0 a 1. 1 = coincide toda; 0 = não se tocam.",
    "iou": "Como o Dice, porém mais rigoroso: área em comum dividida pela área das duas máscaras somadas.",
    "sigma": "Largura do filtro gaussiano H(u,v) = exp(−D²/2σ²) no espectro. σ pequeno corta mais altas frequências e borra mais.",
    "tecnica": "O PDI aplicado à imagem antes de ela entrar nos modelos. Cada técnica tem seus próprios modelos treinados (fold 0).",
    "tarefa": "A: saudável × qualquer alteração. B: saudável × lesão estrutural (nódulo, pólipo, cisto...).",
}

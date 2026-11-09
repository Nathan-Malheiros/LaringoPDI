# Roteiro da demonstração (cerca de 6 minutos, dentro da apresentação de 15)

**Antes de começar:** abra o app pelo atalho (`tools\abrir_app.bat`) e passe uma vez por todas as cenas, porque a primeira carga de cada modelo é lenta. As falas completas e os números estão em `entrega/LaringoPDI_Apostila.pdf`.

**Como usar:** a barra lateral tem as **Cenas da apresentação**. Cada cena ajusta sozinha o frame, a técnica, a tarefa e o passo. Basta clicar na cena e falar. Depois, tudo pode ser mudado à mão em **Entrada** e **Classificação**.

Todos os frames são de vídeos do **teste do fold 0**: nenhum modelo do app os viu no treino.

| Cena | Quem | O que aparece | O que dizer |
|---|---|---|---|
| 1. Realce num frame escuro | Marcos Mendes | `training/12188`, os quatro realces com histograma | O CLAHE foi a única técnica que melhorou a imagem pelos três critérios |
| 2. Média × mediana no ruído | Marcos Mendes | Sal e pimenta a 5%; mova o σ de Fourier | A mediana descarta os extremos; σ menor corta mais frequência e borra mais |
| 3. Segmentação passo a passo | Pedro | `training/3672`, com reflexo | Limiar adaptativo, abertura, fechamento e componente central |
| 4. O CLAHE salva a segmentação | Pedro | `training/12188` com CLAHE; desligue a chave para comparar | O crescimento de região vai de Dice 0,00 a 0,95; no estudo, 0,467 → 0,614 |
| 5. Quando o CLAHE atrapalha | Pedro | `training/11116`, frame claro | O Dice cai de 0,81 para 0,12: o efeito depende da imagem e do método |
| 6. Lesão estrutural: MLP × CNN | Pedro | `training/7957`, Tarefa B, CLAHE, Grad-CAM | A MLP recebe 18 medidas; a CNN aprende sozinha, e o Grad-CAM mostra onde olhou |
| 7. Vídeo saudável, sem PDI | Pedro | `training/12068`, Tarefa A | Trocar a técnica quase não muda a decisão: 62 de 64 vereditos neutros |
| 8. Resultados do estudo | Nathan | Frase-resumo, hipóteses, técnicas, formato do vídeo | H1 parcial, H2 a H4 confirmadas, H5 parcial, H6 refutada |

Os frames são exemplos de uma imagem; os números do estudo são médias de 587 vídeos. A MLP de classificação foi treinada com a média de 10 frames por vídeo, por isso, num único frame, ela fica perto de 50%.

## Se algo falhar

- **App lento:** espere a primeira carga; depois, os modelos ficam em memória.
- **Navegador não abriu:** abra `http://localhost:8501` com a janela preta do app ainda aberta.
- **Técnica sem pesos:** a barra lateral só mostra as técnicas com pesos na máquina. O pacote de entrega traz "sem PDI" e CLAHE; nesse caso, as cenas com outra técnica usam a que existir.
- **Sem o BAGLS na máquina:** o app avisa e para. Os frames do BAGLS não são distribuídos com o código.

Mantenha o aviso visível: **Ferramenta de apoio acadêmico. Não substitui avaliação médica.**

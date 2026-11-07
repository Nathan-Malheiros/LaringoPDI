# Roteiro da demonstração (cerca de 6 minutos, dentro da apresentação de 15)

**Antes de começar:** abra `tools\abrir_app.bat` e deixe a primeira execução terminar, porque a primeira carga é lenta. Abra `http://localhost:8501/?frame=1881` e escolha **CLAHE** na barra lateral. As falas completas e os números estão em `entrega/LaringoPDI_Apostila.pdf`.

**Frames usados** (primeiros do seletor `Frame`):

| Frame | Por que foi escolhido |
|---|---|
| `test/1881.png` | Escuro, de pólipo. O crescimento de região com CLAHE vai de Dice 0,00 para 0,93 |
| `test/21.png` | Reflexos visíveis, saudável; bom para o passo a passo |

O frame 1881 é o exemplo de uma imagem. Os números do estudo são médias de 587 vídeos.

## Passo a passo

| Quem | Aba | Ação | O que dizer |
|---|---|---|---|
| Marcos Mendes | 1. Realce | Frame 1881; compare os histogramas | O CLAHE foi a única técnica que melhorou a imagem pelos três critérios |
| Marcos Mendes | 2. Filtragem e Fourier | Sal e pimenta, 5%; depois, mova o σ | A mediana ignora os extremos; σ menor corta mais frequência e borra mais |
| Pedro | 3. Segmentação | Passo a passo, comparação dos métodos e MLP de segmentação | Crescimento de região com CLAHE é o melhor (0,614); para a MLP, o CLAHE não muda nada: o melhor PDI depende do algoritmo |
| Pedro | 4. PDI + MLP × CNN | As 18 características, as duas saídas e o Grad-CAM; troque para "sem PDI" | Os dois braços quase não mudam com a técnica; a MLP empata com a CNN em F1 |
| Nathan | 5. Resultados | Frase-resumo, tabela por técnica, hipóteses e o aviso do formato do vídeo | 62 de 64 vereditos neutros; H1 parcial, H2 a H4 confirmadas, H5 parcial, H6 refutada |

## Se algo falhar

- **App lento ou travado:** recarregue a página; os resultados ficam em cache depois da primeira execução.
- **Técnica sem pesos:** a barra lateral só mostra as técnicas com pesos na máquina. O pacote de entrega traz "sem PDI" e CLAHE.
- **Sem o BAGLS na máquina:** use **Enviar arquivo** com um frame seu.
- **Imagem de outra origem:** tudo foi calibrado no BAGLS, e o resultado pode ser pior.

Mantenha o aviso visível: **Ferramenta de apoio acadêmico. Não substitui avaliação médica.**

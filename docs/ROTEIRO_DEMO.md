# Roteiro da demonstração (cerca de 6 minutos, dentro da apresentação de 15)

**Antes de começar:** rode `streamlit run app/streamlit_app.py` e deixe a primeira execução terminar, porque a primeira carga é lenta. Abra `http://localhost:8501/?frame=1881`. As falas completas e os números estão em `entrega/LaringoPDI_Apostila_Etapa2.pdf`.

**Frames usados** (primeiros do seletor `Frame`):

| Frame | Por que foi escolhido |
|---|---|
| `test/1881.png` | Escuro, de pólipo. O crescimento de região com CLAHE vai de Dice 0,00 para 0,93 |
| `test/21.png` | Reflexos visíveis, saudável; bom para o passo a passo |

O caso "Dice 0,00 → 0,93" da Etapa 1 não está identificado nos arquivos. `test/1881.png` é o frame de teste mais próximo desses valores. Se perguntarem, explique assim, e lembre que é o exemplo de uma imagem.

## Passo a passo

| Quem | Aba | Ação | O que dizer |
|---|---|---|---|
| Marcos Mendes | 1. Realce | Frame 1881; compare os histogramas | A equalização dá o maior contraste; o CLAHE trabalha por blocos e dá a maior entropia |
| Marcos Mendes | 2. Filtragem e Fourier | Sal e pimenta, 5%; depois, mova o σ | A mediana ignora os extremos; σ menor corta mais frequência e borra mais |
| Pedro | 3. Segmentação | Passo a passo; depois, a comparação dos métodos | Limiar, abertura, fechamento, componente central. No conjunto, o melhor foi crescimento de região com CLAHE |
| Pedro | 4. PDI + MLP × CNN | As 18 características; depois, o Grad-CAM | A MLP recebe as medidas; a CNN aprende sozinha, e o Grad-CAM mostra onde ela olhou |
| Nathan | 5. Resultados | Tabela das hipóteses | H1 parcial, H2 e H3 confirmadas, H4 frágil, H5 parcial, H6 refutada |

## Se algo falhar

- **App lento ou travado:** recarregue a página; os resultados ficam em cache depois da primeira execução.
- **Sem o BAGLS na máquina:** use **Enviar arquivo** com um frame seu. As capturas em `docs/figures/app/` servem de plano B, na máquina em que foram geradas.
- **Imagem de outra origem:** a segmentação foi calibrada no BAGLS, e o resultado pode ser pior.

Mantenha o aviso visível: **Ferramenta de apoio acadêmico. Não substitui avaliação médica.**

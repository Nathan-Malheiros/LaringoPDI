# Roteiro da demonstração (3 a 4 minutos)

**Antes de começar:** rode `streamlit run app/streamlit_app.py` e deixe a primeira execução terminar, porque a primeira carga é lenta. Abra `http://localhost:8501/?frame=1881`, em tela cheia, na aba ① Técnicas de aula.

**Frames usados** (também listados no início do seletor `Frame`):

| Frame | Por que foi escolhido | Número do estudo |
|---|---|---|
| `test/1881.png` | Escuro, família estrutural | `region_grow`: Dice 0,000 sem realce → 0,929 com CLAHE |
| `test/21.png` | Reflexos visíveis, família saudável | Fração especular 3,5%; Dice 0,868 com CLAHE |

O caso "Dice 0,00 → 0,93" da Etapa 1 não está identificado nos arquivos. `test/1881.png` é o frame de teste mais próximo desses valores; se aparecer a pergunta, explique assim.

## Passo a passo

| Tempo | Ação | Fala |
|---|---|---|
| 0:00 | Mostre a aba ①, seção **Realce**, com o frame 1881 | "Este frame é escuro. A equalização global mostra a estrutura, mas repare como estoura o tecido. O CLAHE mantém o tecido e realça o local. O histograma mostra o porquê." |
| 0:35 | Role até **Ruído**: sal e pimenta, densidade 5%, janelas 3 | "Agora adicionamos ruído de propósito. A média espalha cada ponto; a mediana o elimina. No estudo com 300 frames, foram 41,4 dB contra 25,9 dB: é a nossa hipótese P2, confirmada." |
| 1:00 | Mude para **gaussiano** | "Com ruído gaussiano, a vantagem some: aqui, nenhum filtro é o certo para todo tipo de ruído." |
| 1:15 | Role até **Fourier**: leve o σ de 15 para 60 | "No domínio da frequência, o centro é a baixa frequência. Com σ pequeno, cortamos o ruído e também a borda da prega; é um compromisso que dá para ver." |
| 1:40 | Abra a aba ⑤ **Segmentação da glote** | "Com CLAHE, o crescimento de região encontra a glote; verde é o especialista, laranja é o método clássico. Neste frame, sem o realce, o Dice era zero. Mas atenção: o mesmo CLAHE piora os métodos de limiar. O efeito do PDI depende do método." |
| 2:15 | Na barra lateral, escolha o frame **21.png** e o experimento **E01 · spec**; abra a aba ② **Pipeline** | "Aqui há reflexos. A técnica detecta os brilhos e os preenche; as métricas abaixo mostram o que mudou." |
| 2:40 | Abra a aba ⑥ **Rede neural**: Tarefa B, ResNet-18, depois role até a **MLP-18** | "O mapa mostra onde a rede olhou. Abaixo estão as 18 características que prometemos: forma, intensidade e textura. Quando os dois sistemas discordam, é exatamente por isso que esta é uma ferramenta de apoio, e não um diagnóstico." |
| 3:15 | Abra a aba ⑦ **Resultados do estudo** e aponte a tabela das hipóteses | "O resultado principal: na classificação, nenhuma técnica mudou o F1 com significância. Na segmentação, o efeito é forte. Das seis hipóteses, a P6 foi refutada: a forma da glote não foi o grupo mais importante." |

## Se perguntarem

- **"A MLP empata com a CNN?"** Pela regra pré-registrada, sim, mas é uma confirmação frágil. O IC do ΔF1 vai de −0,104 a +0,009, e a ResNet-18 tem acurácia maior (0,903 × 0,842). Veja `docs/hipoteses_etapa1.md`.
- **"O PDI aumenta a atenção na glote?"** Só em parte, e o efeito é misto: 5 combinações sobem e 9 caem, de 24.
- **"E se eu enviar outra imagem?"** Funciona em **Enviar arquivo**, mas a segmentação foi calibrada no BAGLS, então o resultado pode ser pior.

## Se algo falhar

- **App lento ou travado:** recarregue a página; os resultados ficam em cache depois da primeira execução.
- **Sem o BAGLS na máquina:** use **Enviar arquivo** com um frame seu. As capturas em `docs/figures/app/` servem de plano B, na máquina em que foram geradas.

Mantenha o aviso visível: **Ferramenta de apoio acadêmico. Não substitui avaliação médica.**

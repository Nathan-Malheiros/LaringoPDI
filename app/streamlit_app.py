"""LaringoPDI: protótipo de demonstração do estudo. Realce → filtragem e Fourier → segmentação da glote (clássica e MLP) →
classificação (PDI + MLP × CNN) → resultados do estudo.

Três modos: Roteiro guiado (cenas com Anterior/Próxima), Explorar (todos os controles) e Galeria (escolher a imagem).
As imagens vêm de `app/amostras/` (gerada por `tools/gerar_amostras.py`), então o app roda sem o BAGLS no computador.
Todas são de vídeos do teste do fold 0, que nenhum modelo do app viu no treino. Rodar: `tools\\abrir_app.bat`."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from laringopdi import config as C
from laringopdi import data as D
from laringopdi.etapa1 import noise_image
from laringopdi.pdi import run_pipeline
from laringopdi.pdi.enhance import gauss_lpf_transfer
from laringopdi.pdi.quality import image_quality
from laringopdi.pdi.registry import DEFAULTS, TECHNIQUES, to_work_size
from laringopdi.pdi.segmentation import adaptive_morph_steps, apply_best, best_config, dice, iou, segment
from laringopdi.pdi.texture import auto_canny, sobel_mag

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))
from roteiro import AJUDA, CENAS, PASSOS  # noqa: E402

AVISO = "Ferramenta de apoio acadêmico. Não substitui avaliação médica."
DIM, CY, CINZA = "#5B6785", "#0E9FB5", "#8A94AD"
VERDE, LARANJA, CIANO = (46, 158, 106), (224, 123, 46), (14, 159, 181)      # especialista, segmentação obtida, máscara
ENTRADAS = {"E00": "sem PDI", "E15": "alargamento", "E02": "equalização global", "E03": "CLAHE", "E16": "média", "E06": "mediana",
            "E17": "passa-baixa (Fourier)", "E01": "remoção de reflexo", "E04": "homomórfica"}
NOMES = {"raw": "sem PDI", "stretch": "alargamento", "he": "equalização", "clahe": "CLAHE", "mean": "média", "median": "mediana",
         "gauss_lpf": "passa-baixa", "spec": "remoção de reflexo", "homo": "homomórfica"}
FAMILIA = {"healthy": "saudável", "functional": "distúrbio funcional", "structural": "lesão estrutural"}
ROTULO = {"healthy": "saudável", "disorder": "alterado", "structural": "lesão estrutural"}
TAREFAS = {"A": "A · saudável × alterado", "B": "B · saudável × lesão estrutural"}
MODOS = {"roteiro": ":material/slideshow: Roteiro guiado", "explorar": ":material/tune: Explorar", "galeria": ":material/photo_library: Galeria"}
POR_PAGINA = 24

st.set_page_config(page_title="LaringoPDI", page_icon="🔬", layout="wide")
st.markdown("<style>.block-container { padding-top: 1.6rem; max-width: 1400px; }</style>", unsafe_allow_html=True)

cfg = C.load()
SIZE, WORK = cfg["data"]["image_size"], cfg["data"]["work_size"]
RUNS, TABLES = C.ROOT / "results" / "runs", C.ROOT / "results" / "tables"
ss = st.session_state


# ---------------------------------------------------------------------------------------- dados
@st.cache_data(show_spinner=False)
def carregar_frames() -> pd.DataFrame:
    """Frames do app: as amostras embutidas; sem elas, os frames do teste do fold 0 do BAGLS local."""
    cols = ["path", "video_id", "family", "img", "seg"]
    man = APP_DIR / "amostras" / "amostras.json"
    if man.exists():
        fr = pd.DataFrame(json.loads(man.read_text(encoding="utf-8"))["frames"])
        fr["img"] = [str(APP_DIR / "amostras" / f) for f in fr["arquivo"]]
        fr["seg"] = [str(APP_DIR / "amostras" / f) for f in fr["mascara"]]
        return fr.astype({"video_id": str})[cols]
    raw = C.ROOT / cfg["paths"]["raw"]
    if not raw.exists():
        return pd.DataFrame(columns=cols)
    fr = D.load_frames(cfg)
    test = {str(v) for v in D.load_splits(cfg)[0]["test"]}
    fr = fr[fr["video_id"].astype(str).isin(test)]
    fr = fr[[(raw / p).exists() for p in fr["path"]]].copy()
    fr["img"], fr["seg"] = [str(raw / p) for p in fr["path"]], [str(raw / p) for p in fr["seg_path"]]
    return fr.astype({"video_id": str})[cols].reset_index(drop=True)


@st.cache_data(show_spinner=False, max_entries=400)
def miniatura(path: str, lado: int = 160) -> np.ndarray:
    """Frame reduzido dentro de um quadrado preto, para a grade da galeria ficar alinhada."""
    g = D.imread(path, gray=True)
    f = lado / max(g.shape)
    r = cv2.resize(g, (max(1, round(g.shape[1] * f)), max(1, round(g.shape[0] * f))), interpolation=cv2.INTER_AREA)
    q = np.zeros((lado, lado), np.uint8)
    y, x = (lado - r.shape[0]) // 2, (lado - r.shape[1]) // 2
    q[y:y + r.shape[0], x:x + r.shape[1]] = r
    return q


def has_weights(exp_: str) -> bool:
    return (RUNS / exp_ / "A" / "resnet18" / "fold0_seed0" / "model.pt").exists()


@st.cache_resource(show_spinner=False)
def mlp_model(exp_: str, task_: str, kind: str):
    from laringopdi.mlp import load_mlp
    return load_mlp(cfg, exp_, task_, kind)


@st.cache_resource(show_spinner=False)
def cnn_model(exp_: str, task_: str):
    from laringopdi.evaluate import load_model
    return load_model(cfg, exp_, task_, "resnet18", 0)


def fid(path: str) -> str:
    return path.split("/")[-1].split(".")[0]


FR = carregar_frames()
if FR.empty:
    st.title("LaringoPDI")
    st.error("Nenhuma imagem encontrada. O app usa as amostras de `app/amostras/` (gere com `tools/gerar_amostras.py`) "
             "ou o BAGLS em data/raw/bagls, que não é distribuído com o código.")
    st.stop()
FRAMES = FR["path"].tolist()
POR_PATH = FR.set_index("path")
DISP = [e for e in ENTRADAS if has_weights(e)] or ["E00"]
NA_CENA = {c["frame"] for c in CENAS if "frame" in c}

# ---------------------------------------------------------------------------------------- estado
PADRAO = {"modo": "roteiro", "cena": 0, "passo": 0, "frame": FRAMES[0], "exp": "E03" if "E03" in DISP else DISP[0], "task": "A",
          "realce": "clahe", "filtro_vista": "ruido", "ruido_tipo": "sal e pimenta", "ruido_dens": 5, "ruido_sigma": 10, "k_media": 3,
          "k_mediana": 3, "lpf_sigma": 15, "seg_clahe": False, "seg_vista": "passos", "seg_etapa": "entrada", "galeria_familia": "todas",
          "galeria_pagina": 0}
CHAVES_CENA = [k for k in PADRAO if k not in ("modo", "cena", "galeria_familia", "galeria_pagina")]


def aplicar_cena(i: int) -> None:
    """Ajusta todos os controles de uma vez, para a demonstração sair igual ao ensaio."""
    i = max(0, min(i, len(CENAS) - 1))
    ss["cena"] = i
    for k, v in CENAS[i].items():
        if k in CHAVES_CENA and (k != "frame" or v in FRAMES) and (k != "exp" or v in DISP):
            ss[k] = v


def cena_alterada() -> bool:
    return any(k in ss and ss[k] != v for k, v in CENAS[ss["cena"]].items()
               if k in CHAVES_CENA and (k != "frame" or v in FRAMES) and (k != "exp" or v in DISP))


def ao_trocar_modo(anterior: str) -> None:
    if ss["modo"] == "roteiro" and anterior != "roteiro":
        aplicar_cena(ss["cena"])                              # o roteiro sempre volta exatamente como ensaiado


def usar_frame(path: str) -> None:
    ss["frame"], ss["modo"] = path, "explorar"
    if ss["passo"] == 4:
        ss["passo"] = 0


def ir_galeria() -> None:
    ss["modo"] = "galeria"


if "iniciado" not in ss:                                     # primeira carga: cena 1, ou ?aba=N&frame=ID (capturas)
    ss["iniciado"] = True
    for k, v in PADRAO.items():
        ss[k] = v
    aplicar_cena(0)
    if "aba" in st.query_params:
        ss["modo"], ss["passo"] = "explorar", int(st.query_params["aba"]) % len(PASSOS)
    if st.query_params.get("modo") in MODOS:
        ss["modo"] = st.query_params["modo"]
    want = st.query_params.get("frame")
    if want:
        ss["frame"] = next((p for p in FRAMES if fid(p) == want), ss["frame"])
for k, v in PADRAO.items():                                  # widgets que não foram desenhados perdem o valor: volta o padrão
    ss.setdefault(k, v)
if ss["frame"] not in POR_PATH.index:
    ss["frame"] = FRAMES[0]


def segmentado(label: str, opcoes: dict, chave: str, depois=None, **kw):
    """Controle segmentado que não fica vazio ao clicar de novo na opção marcada; o valor vive em `ss[chave]`."""
    nav = f"{chave}__nav"
    ss[nav] = ss[chave]

    def _sync():
        anterior = ss[chave]
        if ss[nav] is not None:
            ss[chave] = ss[nav]
            if depois:
                depois(anterior)

    st.segmented_control(label, list(opcoes), key=nav, format_func=lambda o: opcoes[o], on_change=_sync, **kw)
    return ss[chave]


# ---------------------------------------------------------------------------------------- imagem atual
row = POR_PATH.loc[ss["frame"]]
gray = D.imread(row["img"], gray=True)
gt_full = D.imread(row["seg"], gray=True) > 127
base = to_work_size(gray, WORK)
gt = cv2.resize(gt_full.astype(np.uint8), base.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)


def tech(name: str, g: np.ndarray | None = None, **params) -> np.ndarray:
    return TECHNIQUES[name](base if g is None else g, {}, **(params or DEFAULTS[name]))


def mostrar_hist(alvo, img: np.ndarray, color: str = CY) -> None:
    v = img[img > 0] if (img > 0).mean() < .98 else img.ravel()
    fig, ax = plt.subplots(figsize=(4, 1.3), dpi=120)
    ax.hist(v.ravel(), bins=64, range=(0, 255), color=color, alpha=.9)
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([0, 128, 255])
    ax.tick_params(labelsize=8, colors=DIM, length=0)
    fig.tight_layout(pad=.2)
    alvo.pyplot(fig, width="stretch")
    plt.close(fig)


def norm8(a: np.ndarray) -> np.ndarray:
    return ((a - a.min()) / (np.ptp(a) or 1) * 255).astype(np.uint8)


def psnr_ssim(ref: np.ndarray, img: np.ndarray) -> tuple[float, float]:
    return float(peak_signal_noise_ratio(ref, img, data_range=255)), float(structural_similarity(ref, img, data_range=255))


def legenda(c_, titulo: str, sub: str = "") -> None:
    c_.markdown(f"**{titulo}**")
    if sub:
        c_.caption(sub)


def outline(g: np.ndarray, masks: list[tuple[np.ndarray, tuple]]) -> np.ndarray:
    v = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)
    for mk, col in masks:
        cnts, _ = cv2.findContours(mk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(v, cnts, -1, col, 2)
    return v


def pintar(g: np.ndarray, mk: np.ndarray) -> np.ndarray:
    """Máscara em ciano translúcido sobre o frame em cinza."""
    v = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB).astype(np.float32)
    v[mk] = 0.45 * v[mk] + 0.55 * np.array(CIANO, np.float32)
    return v.astype(np.uint8)


def table(name: str) -> pd.DataFrame | None:
    p = TABLES / name
    return pd.read_csv(p) if p.exists() else None


def rotulo_real(task_: str) -> str | None:
    fam = row["family"]
    if task_ == "A":
        return "healthy" if fam == "healthy" else "disorder"
    return {"healthy": "healthy", "structural": "structural"}.get(fam)


def resultado(names: list[str], p: np.ndarray, real: str | None) -> str:
    """Tendência do modelo, conferida com o rótulo do especialista, e as probabilidades em barras."""
    top = names[int(np.argmax(p))]
    marca = "" if real is None else (" · :green[confere com o especialista]" if top == real else " · :orange[difere do especialista]")
    st.markdown(f"Tendência: **{ROTULO.get(top, top)}** ({p.max():.0%}){marca}")
    for n, pv in sorted(zip(names, p), key=lambda t: -t[1]):
        st.progress(float(pv), text=f"{ROTULO.get(n, n)}: {pv:.1%}")
    return top


def contexto() -> None:
    fam = FAMILIA.get(row["family"], row["family"])
    st.caption(f":material/image: Vídeo **{fam}** · frame {fid(ss['frame'])} &nbsp;·&nbsp; :material/memory: modelos com **{ENTRADAS[ss['exp']]}**"
               f" &nbsp;·&nbsp; tarefa {TAREFAS[ss['task']]}")


# ---------------------------------------------------------------------------------------- barra lateral
with st.sidebar:
    st.markdown("#### Imagem em uso")
    st.image(miniatura(row["img"], 220), width="stretch")
    st.caption(f"Vídeo {FAMILIA.get(row['family'], row['family'])} · frame {fid(ss['frame'])}")
    st.button("Trocar imagem na galeria", icon=":material/photo_library:", on_click=ir_galeria, width="stretch",
              disabled=ss["modo"] == "galeria")
    st.markdown("#### Modelos")
    st.selectbox("Técnica de PDI na entrada", DISP, key="exp", format_func=lambda e: ENTRADAS[e], help=AJUDA["tecnica"])
    st.radio("Tarefa", list(TAREFAS), key="task", format_func=TAREFAS.get, help=AJUDA["tarefa"])
    with st.expander("Sobre o protótipo"):
        st.markdown(f"Imagens do **BAGLS** (Gómez et al., 2020; CC BY-NC-SA 4.0): {len(FR)} frames de {FR['video_id'].nunique()} vídeos "
                    "do teste do fold 0, que nenhum modelo deste app viu no treino. A máscara verde é a do especialista.")
        st.caption(AVISO)

# ---------------------------------------------------------------------------------------- cabeçalho
st.title("LaringoPDI")
st.caption(f"Como o processamento digital de imagens muda a análise da glote na laringoscopia · UNIFOR · :material/info: {AVISO}")
modo = segmentado("Modo", MODOS, "modo", depois=ao_trocar_modo, label_visibility="collapsed")

if modo == "roteiro":
    cena = ss["cena"]
    c = CENAS[cena]
    ss["passo"] = c["passo"]
    with st.container(border=True):
        a, meio, b = st.columns([1.1, 6, 1.1], vertical_alignment="center")
        a.button("Anterior", key="cena_ant", icon=":material/arrow_back:", on_click=aplicar_cena, args=(cena - 1,),
                 disabled=cena == 0, width="stretch")
        with meio:
            st.caption(f"Cena {cena + 1} de {len(CENAS)} · Passo {c['passo'] + 1}: {PASSOS[c['passo']]['nome']}")
            st.markdown(f"### {c['titulo']}")
            st.markdown(f"**O que observar:** {c['nota']}")
        b.button("Próxima", key="cena_prox", icon=":material/arrow_forward:", type="primary", on_click=aplicar_cena, args=(cena + 1,),
                 disabled=cena == len(CENAS) - 1, width="stretch")
        st.progress((cena + 1) / len(CENAS))
        x, y, z = st.columns([2, 2, 5], vertical_alignment="center")
        with x.popover("Todas as cenas", icon=":material/list:", width="stretch"):
            for i, ci in enumerate(CENAS):
                st.button(f"{i + 1}. {ci['titulo']}", key=f"ir_cena_{i}", on_click=aplicar_cena, args=(i,), width="stretch",
                          type="primary" if i == cena else "tertiary")
        if cena_alterada():
            y.button("Restaurar a cena", key="cena_restaurar", icon=":material/undo:", on_click=aplicar_cena, args=(cena,), width="stretch")
            z.caption("Você mudou algum controle; os números podem diferir da nota da cena.")
        else:
            z.caption(PASSOS[c["passo"]]["porque"])
elif modo == "explorar":
    segmentado("Passo", {i: f"{i + 1}. {p['nome']}" for i, p in enumerate(PASSOS)}, "passo", label_visibility="collapsed", width="stretch")
    st.caption(PASSOS[ss["passo"]]["porque"])

passo = ss["passo"]

# ---------------------------------------------------------------------------------------- galeria
if modo == "galeria":
    st.markdown("Escolha um frame para analisar. São só vídeos do **teste do fold 0**: nenhum modelo do app os viu no treino. "
                "A estrela marca os frames usados no roteiro.")
    cont = FR["family"].value_counts()
    fams = {"todas": f"Todas ({len(FR)})", **{k: f"{FAMILIA[k].capitalize()} ({cont.get(k, 0)})" for k in FAMILIA if cont.get(k, 0)}}
    f = segmentado("Filtrar por classe do vídeo", fams, "galeria_familia", depois=lambda _: ss.update(galeria_pagina=0))
    sub = FR if f == "todas" else FR[FR["family"] == f]
    npag = max(1, math.ceil(len(sub) / POR_PAGINA))
    pag = min(ss["galeria_pagina"], npag - 1)
    for ini in range(pag * POR_PAGINA, min(len(sub), (pag + 1) * POR_PAGINA), 6):
        cols = st.columns(6)
        for c_, r in zip(cols, sub.iloc[ini:ini + 6].itertuples()):
            with c_.container(border=True):
                st.image(miniatura(r.img), width="stretch")
                atual = r.path == ss["frame"]
                st.caption(f"{'★ ' if r.path in NA_CENA else ''}{FAMILIA.get(r.family, r.family)} · {fid(r.path)}")
                st.button("Em uso" if atual else "Abrir", key=f"gal_{r.path}", on_click=usar_frame, args=(r.path,), disabled=atual,
                          type="secondary" if atual else "primary", width="stretch")
    if npag > 1:
        a, meio, b = st.columns([1, 3, 1], vertical_alignment="center")
        a.button("Anterior", key="gal_ant", icon=":material/arrow_back:", disabled=pag == 0, width="stretch",
                 on_click=lambda: ss.update(galeria_pagina=pag - 1))
        meio.markdown(f"<div style='text-align:center'>Página {pag + 1} de {npag}</div>", unsafe_allow_html=True)
        b.button("Próxima", key="gal_prox", icon=":material/arrow_forward:", disabled=pag >= npag - 1, width="stretch",
                 on_click=lambda: ss.update(galeria_pagina=pag + 1))

# ---------------------------------------------------------------------------------------- 1 realce
elif passo == 0:
    contexto()
    OPC = {"stretch": "Alargamento", "he": "Equalização global", "clahe": "CLAHE"}
    DESCR = {"stretch": "Estica os tons de cinza para ocuparem de 0 a 255. Simples e previsível, mas ajuda pouco quando a imagem já usa "
                        "quase toda a faixa.",
             "he": "Redistribui os tons pela imagem inteira. Revela as regiões escuras, mas pode estourar o tecido claro e realçar ruído.",
             "clahe": "Equaliza em blocos de 8×8 e limita o ganho (clip 2,0). Mostra a glote no escuro sem estourar o tecido. "
                      "No estudo, foi a única técnica que melhorou os três critérios de qualidade."}
    t = segmentado("Técnica de realce", OPC, "realce")
    im = tech(t)
    q0, q1 = image_quality(base), image_quality(im)
    a, b, info = st.columns([1, 1, 1.1])
    a.image(base, width="stretch", clamp=True)
    legenda(a, "Original")
    mostrar_hist(a, base, CINZA)
    b.image(im, width="stretch", clamp=True)
    legenda(b, OPC[t])
    mostrar_hist(b, im)
    with info:
        st.markdown(f"#### {OPC[t]}")
        st.markdown(DESCR[t])
        m1, m2 = st.columns(2)
        m1.metric("Contraste RMS", f"{q1['rms_contrast']:.3f}", f"{q1['rms_contrast'] - q0['rms_contrast']:+.3f} vs. original", help=AJUDA["rms"])
        m2.metric("Entropia", f"{q1['entropy']:.2f}", f"{q1['entropy'] - q0['entropy']:+.2f} vs. original", help=AJUDA["entropia"])
        st.caption("O histograma mostra quantos pixels há em cada tom de cinza (0 = preto, 255 = branco), sem o fundo preto. "
                   "Um realce bom espalha os tons pela faixa.")
    with st.expander("Ver as quatro versões lado a lado"):
        cols, rows = st.columns(4), []
        for c_, (t_, lab) in zip(cols, (("raw", "original"), ("stretch", "alargamento"), ("he", "equalização global"), ("clahe", "CLAHE"))):
            im_ = base if t_ == "raw" else tech(t_)
            qa = image_quality(im_)
            c_.image(im_, width="stretch", clamp=True)
            legenda(c_, lab)
            rows.append({"versão": lab, "contraste RMS": qa["rms_contrast"], "entropia": qa["entropy"]})
        st.dataframe(pd.DataFrame(rows).set_index("versão").style.format({"contraste RMS": "{:.3f}", "entropia": "{:.2f}"}), width="stretch")

# ---------------------------------------------------------------------------------------- 2 filtragem
elif passo == 1:
    contexto()
    vista = segmentado("Experimento", {"ruido": "Ruído: média × mediana", "fourier": "Fourier: passa-baixa"}, "filtro_vista")
    if vista == "ruido":
        st.markdown("Colocamos ruído artificial no frame e vemos qual filtro **recupera melhor o original**.")
        c = st.columns(4)
        kind = c[0].radio("Tipo de ruído", ["sal e pimenta", "gaussiano"], key="ruido_tipo",
                          help="Sal e pimenta: pixels isolados brancos e pretos. Gaussiano: todos os pixels variam um pouco.")
        if kind == "gaussiano":
            lvl = c[1].slider("Intensidade (σ)", 5, 40, key="ruido_sigma")
            noisy = noise_image(base, "gauss", lvl, np.random.default_rng(0))
        else:
            lvl = c[1].slider("Densidade (%)", 1, 20, key="ruido_dens")
            noisy = noise_image(base, "sp", lvl / 100, np.random.default_rng(0))
        km = c[2].select_slider("Janela da média", [3, 7, 15], key="k_media", help="Tamanho da vizinhança. Maior = mais suave e mais borrado.")
        kd = c[3].select_slider("Janela da mediana", [3, 5], key="k_mediana", help="Tamanho da vizinhança da mediana.")
        fm, fd = tech("mean", noisy, ksize=km), tech("median", noisy, ksize=kd)
        p_ruido = psnr_ssim(base, noisy)
        c = st.columns(4)
        res = {}
        for c_, im, lab in zip(c, (base, noisy, fm, fd), ("original (referência)", "com ruído", f"média {km}×{km}", f"mediana {kd}×{kd}")):
            c_.image(im, width="stretch", clamp=True)
            legenda(c_, lab)
            if im is not base:
                ps, sm = psnr_ssim(base, im)
                res[lab] = ps
                d = None if im is noisy else (f"{ps - p_ruido[0]:+.1f} dB", f"{sm - p_ruido[1]:+.3f}")
                c_.metric("PSNR", f"{ps:.1f} dB", d and d[0], help=AJUDA["psnr"])
                c_.metric("SSIM", f"{sm:.3f}", d and d[1], help=AJUDA["ssim"])
        lm, ld = f"média {km}×{km}", f"mediana {kd}×{kd}"
        venc, perd = (ld, lm) if res[ld] >= res[lm] else (lm, ld)
        porque = ("A mediana descarta os valores extremos da vizinhança; a média os espalha." if kind == "sal e pimenta"
                  else "No ruído gaussiano, todo pixel varia um pouco, e a média, que suaviza tudo, compete com a mediana.")
        st.success(f"Neste ruído, a **{venc}** ficou mais perto do original: PSNR {res[venc]:.1f} dB contra {res[perd]:.1f} dB. {porque}")
    else:
        st.markdown("Na transformada de Fourier, o **centro do espectro são os tons que variam devagar** e as bordas são os detalhes finos. "
                    "O passa-baixa mantém o centro e corta o resto.")
        sigma = st.slider("Corte do filtro (σ): menor = mais borrado", 5, 80, key="lpf_sigma", help=AJUDA["sigma"])
        lpf = tech("gauss_lpf", sigma=sigma)
        spec_f = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(base.astype(np.float64)))))
        c = st.columns(4)
        c[0].image(base, caption="original", width="stretch", clamp=True)
        c[1].image(cv2.applyColorMap(norm8(spec_f), cv2.COLORMAP_MAGMA)[..., ::-1], caption="espectro |F(u,v)|: o centro é a baixa frequência",
                   width="stretch")
        c[2].image(cv2.applyColorMap(norm8(gauss_lpf_transfer(base.shape, sigma)), cv2.COLORMAP_VIRIDIS)[..., ::-1],
                   caption=f"filtro H(u,v), σ = {sigma}: claro passa, escuro corta", width="stretch")
        c[3].image(lpf, caption="resultado", width="stretch", clamp=True)
        g1, g2 = st.columns([1, 3], vertical_alignment="center")
        g1.metric("PSNR com este σ", f"{psnr_ssim(base, lpf)[0]:.1f} dB", help=AJUDA["psnr"])
        curva = pd.DataFrame({"σ": list(range(5, 81, 5)), "PSNR (dB)": [psnr_ssim(base, tech("gauss_lpf", sigma=s))[0] for s in range(5, 81, 5)]})
        g2.line_chart(curva, x="σ", y="PSNR (dB)", height=180, color=CY)
        st.caption("PSNR em relação ao original: quanto menor o σ, mais alta frequência é cortada e mais borrada fica a borda das pregas.")

# ---------------------------------------------------------------------------------------- 3 segmentação
elif passo == 2:
    contexto()
    t1, t2 = st.columns([2, 3], vertical_alignment="center")
    usar_clahe = t1.toggle("Aplicar CLAHE antes de segmentar", key="seg_clahe",
                           help="Realça o frame antes da segmentação. Ajuda em frames escuros e pode atrapalhar nos claros.")
    t2.caption(":green[Verde]: máscara do especialista (gabarito) · :orange[Laranja]: segmentação obtida")
    vista = segmentado("Vista", {"passos": "Passo a passo", "metodos": "Métodos × especialista", "mlp": "MLP de pixels", "bordas": "Bordas"},
                       "seg_vista")
    gs = tech("clahe") if usar_clahe else base
    bc = best_config()
    if vista == "passos":
        s_ = adaptive_morph_steps(gs)
        ETAPAS = {"entrada": ("Entrada", f"O frame que vai ser segmentado, {'com CLAHE' if usar_clahe else 'sem realce'}. A glote é a região escura no centro."),
                  "limiar": ("Limiar adaptativo", "Cada pixel é comparado com a média da vizinhança 51×51: o que for mais escuro vira candidato "
                                                  "a glote. Funciona mesmo com iluminação desigual."),
                  "abertura": ("Abertura", "Erosão seguida de dilatação: some com os pontinhos isolados sem encolher as regiões grandes."),
                  "fechamento": ("Fechamento", "Dilatação seguida de erosão: fecha os buracos dentro da região."),
                  "componente": ("Componente central", "Fica só a região conectada mais perto do centro: a glote. Agora dá para comparar com o especialista.")}
        etapa = st.select_slider("Etapa", options=list(ETAPAS), format_func=lambda k: ETAPAS[k][0], key="seg_etapa")
        img_col, txt = st.columns([1.3, 1])
        if etapa == "entrada":
            img_col.image(gs, width="stretch", clamp=True)
        elif etapa == "componente":
            img_col.image(outline(gs, [(gt, VERDE), (s_["componente"], LARANJA)]), width="stretch")
        else:
            img_col.image(pintar(gs, s_[etapa].astype(bool)), width="stretch")
        with txt:
            st.markdown(f"#### {ETAPAS[etapa][0]}")
            st.markdown(ETAPAS[etapa][1])
            ordem = list(ETAPAS)
            st.markdown("\n".join(f"{i + 1}. {'**' + ETAPAS[k][0] + '** ←' if k == etapa else ETAPAS[k][0]}" for i, k in enumerate(ordem)))
            if etapa == "componente":
                d1, d2 = st.columns(2)
                d1.metric("Dice", f"{dice(s_['componente'], gt):.3f}", help=AJUDA["dice"])
                d2.metric("IoU", f"{iou(s_['componente'], gt):.3f}", help=AJUDA["iou"])
            elif etapa != "entrada":
                st.caption("Em ciano: os pixels marcados como glote nesta etapa.")
        with st.expander("Ver as cinco etapas lado a lado"):
            c = st.columns(5)
            c[0].image(gs, width="stretch", clamp=True)
            legenda(c[0], "entrada")
            for c_, k in zip(c[1:], ("limiar", "abertura", "fechamento", "componente")):
                c_.image(s_[k].astype(np.uint8) * 255, width="stretch")
                legenda(c_, ETAPAS[k][0].lower())
    elif vista == "metodos":
        s_ = adaptive_morph_steps(gs)
        pre = "clahe" if usar_clahe else "raw"
        metodos = [("Otsu global", segment(gs, "otsu")), ("Otsu + morfologia", segment(gs, "otsu_morph")),
                   ("adaptativa + morfologia", s_["componente"]), ("crescimento de região (método do estudo)", apply_best(base, bc["method"], bc["params"], pre))]
        dices = [dice(mk, gt) for _, mk in metodos]
        melhor = int(np.argmax(dices))
        c = st.columns(4)
        for i, (c_, (lab, mk)) in enumerate(zip(c, metodos)):
            c_.image(outline(base, [(gt, VERDE), (mk, LARANJA)]), width="stretch")
            legenda(c_, lab, ":material/star: melhor neste frame" if i == melhor else "")
            c_.metric("Dice", f"{dices[i]:.3f}", help=AJUDA["dice"])
            c_.metric("IoU", f"{iou(mk, gt):.3f}", help=AJUDA["iou"])
        st.markdown("#### O efeito do CLAHE no método do estudo")
        d_raw = dice(apply_best(base, bc["method"], bc["params"], "raw"), gt)
        d_cl = dice(apply_best(base, bc["method"], bc["params"], "clahe"), gt)
        e1, e2, e3 = st.columns([1, 1, 2], vertical_alignment="center")
        e1.metric("Crescimento de região sem realce", f"{d_raw:.2f}", help=AJUDA["dice"])
        e2.metric("Com CLAHE", f"{d_cl:.2f}", f"{d_cl - d_raw:+.2f}", help=AJUDA["dice"])
        e3.caption("Dice neste frame. No estudo inteiro, o CLAHE levou o crescimento de região de 0,467 para 0,614 em média, "
                   "mas há frames claros em que ele atrapalha.")
    elif vista == "mlp":
        st.markdown(f"Uma rede pequena olha a vizinhança 15×15 de cada pixel e decide se ele é glote. Entrada: **{ENTRADAS[ss['exp']]}** "
                    "(troque na barra lateral).")
        mseg = mlp_model(ss["exp"], "seg", "mlp_seg")
        if mseg is None:
            st.info(f"Sem pesos da MLP de segmentação para {ENTRADAS[ss['exp']]}.")
        else:
            from laringopdi.mlp import _segment_region, segment_frame_mlp
            with st.spinner("Classificando cada pixel…"):
                g224, ctx224 = run_pipeline(gray, cfg["experiments"][ss["exp"]]["pipeline"], None, SIZE, WORK, gt_mask=gt_full)
                g224 = np.ascontiguousarray(g224[..., 0])
                gt224 = ctx224["gt_mask_lb"]
                pares = [("MLP de pixels (vizinhança 15×15)", segment_frame_mlp(cfg, mseg, g224)),
                         ("crescimento de região na mesma imagem", _segment_region(g224, bc["method"], bc["params"]))]
            c = st.columns(4)
            for c_, (lab, mk) in zip(c, pares):
                c_.image(outline(g224, [(gt224, VERDE), (mk, LARANJA)]), width="stretch")
                legenda(c_, lab)
                c_.metric("Dice", f"{dice(mk, gt224):.3f}", help=AJUDA["dice"])
            with c[2]:
                st.markdown("**No estudo inteiro**")
                st.markdown("A melhor MLP de segmentação foi com a homomórfica (Dice 0,532). O melhor resultado geral continua sendo "
                            "o crescimento de região com CLAHE (0,614): mais simples e sem treino.")
    else:
        st.markdown("Em vez de separar regiões, os detectores de borda marcam onde o tom muda de repente, como o contorno das pregas.")
        blur = cv2.GaussianBlur(base, (5, 5), 0)
        c = st.columns(3)
        c[0].image(base, caption="original", width="stretch", clamp=True)
        c[1].image(sobel_mag(blur), caption="Sobel: força da borda (mais claro = mudança mais forte)", width="stretch", clamp=True)
        c[2].image(auto_canny(blur), caption="Canny: linhas finas e contínuas", width="stretch", clamp=True)
        st.caption("As bordas sozinhas não fecham a glote, mas descrevem a textura: a densidade do Canny e o gradiente médio do Sobel "
                   "estão entre as 18 medidas da MLP de classificação.")

# ---------------------------------------------------------------------------------------- 4 classificação
elif passo == 3:
    contexto()
    F18_PT = {"shape_area_rel": "área relativa", "shape_perimeter_rel": "perímetro", "shape_eccentricity": "excentricidade",
              "shape_solidity": "solidez", "shape_aspect": "razão de aspecto", "shape_symmetry": "simetria", "int_mean": "média",
              "int_std": "desvio", "int_entropy": "entropia", "int_skew": "assimetria", "int_kurtosis": "curtose",
              "int_contrast": "contraste glote × tecido", "tex_canny_density": "densidade de bordas (Canny)", "tex_sobel_mean": "gradiente médio (Sobel)",
              "tex_glcm_contrast": "GLCM contraste", "tex_glcm_homogeneity": "GLCM homogeneidade", "tex_glcm_energy": "GLCM energia",
              "tex_lbp_entropy": "entropia do LBP"}
    exp, task = ss["exp"], ss["task"]
    names = D.task_names(cfg, task)
    real = rotulo_real(task)
    if real is None:
        st.warning(f"Este vídeo é de **{FAMILIA.get(row['family'], row['family'])}** e não entra na Tarefa B. O resultado abaixo é só ilustrativo.")
    else:
        st.markdown(f"Rótulo do especialista para este vídeo: **{ROTULO[real]}**. Abaixo, o que cada modelo diria olhando **só este frame**.")
    o_, _ = run_pipeline(gray, cfg["experiments"][exp]["pipeline"], None, SIZE, WORK)
    esquerda, direita = st.columns(2)
    tops = {}
    with esquerda.container(border=True):
        st.markdown("#### Braço 1 · PDI + 18 características + MLP")
        st.caption("Mede 18 características de forma, intensidade e textura da glote e as passa a uma rede pequena (18 → 32 → 16 → 2).")
        c6 = mlp_model(exp, task, "mlp18")
        if c6 is None:
            st.info(f"Sem pesos da MLP para {ENTRADAS[exp]} na Tarefa {task}.")
        else:
            from laringopdi.mlp import classify_frame18
            p6, vec = classify_frame18(cfg, c6, np.ascontiguousarray(o_[..., 0]))
            tops["MLP"] = resultado(names, p6, real)
            st.caption("A MLP foi treinada com a média de 10 frames por vídeo; com um único frame, costuma ficar perto de 50%.")
            with st.expander("As 18 medidas que a MLP recebeu"):
                cols = st.columns(3)
                for c_, (fam, title) in zip(cols, (("f18shape_", "Forma"), ("f18int_", "Intensidade"), ("f18tex_", "Textura"))):
                    legenda(c_, title)
                    c_.dataframe(pd.DataFrame([{"medida": F18_PT.get(k.replace("f18", ""), k), "valor": v} for k, v in vec.items() if k.startswith(fam)])
                                 .set_index("medida").style.format({"valor": "{:.3f}"}), width="stretch")
    with direita.container(border=True):
        st.markdown(f"#### Braço 2 · ResNet-18 sobre {ENTRADAS[exp]}")
        st.caption("Uma rede convolucional vê a imagem inteira e aprende sozinha o que observar.")
        with st.spinner("Carregando a ResNet-18 e calculando o Grad-CAM…"):
            model = cnn_model(exp, task)
            if model is not None:
                import torch

                from laringopdi.evaluate import gradcam_maps
                from laringopdi.train import _prep
                with torch.no_grad():
                    prob = torch.softmax(model(_prep(torch.from_numpy(o_[None, ..., :3].copy()), torch.device("cpu"))), 1)[0].numpy()
                cam = gradcam_maps(model, o_[None, ..., :3])[0]
        if model is None:
            st.info(f"Sem pesos da ResNet-18 para {ENTRADAS[exp]} na Tarefa {task}.")
        else:
            tops["ResNet-18"] = resultado(names, prob, real)
            heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1]
            c1, c2 = st.columns(2)
            c1.image(o_[..., 0], caption="entrada da rede", width="stretch", clamp=True)
            c2.image((0.55 * o_[..., :3] + 0.45 * heat).astype(np.uint8), caption="Grad-CAM: vermelho = onde a rede mais olhou", width="stretch")
    if len(tops) == 2:
        if len(set(tops.values())) == 1:
            st.info(f"Os dois braços concordam: **{ROTULO.get(tops['MLP'], tops['MLP'])}**.", icon=":material/check_circle:")
        else:
            st.warning("Os dois braços discordam. Numa triagem real, a decisão é do profissional.", icon=":material/call_split:")
    st.caption("Modelos do fold 0. Saída de apoio, não diagnóstico.")

# ---------------------------------------------------------------------------------------- 5 resultados
else:
    st.caption("Números do estudo inteiro (587 vídeos, 5 folds, 3 sementes; classificação por vídeo), lidos das tabelas geradas pelo código. "
               "Não dependem do frame escolhido.")
    k1, k2, k3 = st.columns(3)
    k1.metric("Dice da segmentação: CLAHE + crescimento de região", "0,614", "+0,147 vs. sem PDI (0,467)", help=AJUDA["dice"])
    k2.metric("Vereditos neutros na classificação", "62 de 64", help="Melhora ou piora só conta com IC 95% fora do zero e teste corrigido por Holm.")
    k3.metric("MLP simples × ResNet-18 em F1", "empate", help="A MLP de 18 características custa uma fração do treino da ResNet-18.")
    st.success("**Em uma frase:** o PDI muda muito a segmentação (CLAHE + crescimento de região: Dice 0,614 contra 0,467) e quase nada a classificação "
               "(62 de 64 vereditos neutros). A MLP simples empatou com a ResNet-18 em F1, a uma fração do custo.")
    sys.path.insert(0, str(C.ROOT))
    hp = TABLES / "sintese" / "hipoteses.json"
    if hp.exists():
        from tools.documento_resultados import R as _R
        st.subheader("Hipóteses da Etapa 1")
        st.caption("Como ler: cada hipótese foi formulada antes dos experimentos; o veredito diz se os dados a sustentaram.")
        hyp = json.loads(hp.read_text(encoding="utf-8"))
        st.dataframe(pd.DataFrame([{"ID": k, "hipótese": txt, "veredito": hyp[k]["veredito"], "evidência": ev.replace("&lt;", "<")} for k, txt, ev in _R["hipoteses"]]).set_index("ID"),
                     width="stretch")

    pt_ = table("sintese/por_tecnica.csv")
    if pt_ is not None:
        st.subheader("Veredito de cada técnica, em cada etapa")
        st.caption("Como ler: cada célula compara a técnica com a imagem sem PDI. Melhora ou piora só com IC 95% fora do zero e teste corrigido "
                   "por Holm abaixo de 0,05. Neutra não prova efeito zero.")
        cols_ = {"nome": "técnica", "visual": "imagem", "seg_classica": "segmentação clássica", "seg_mlp": "segmentação MLP", "mlp_A_v_sens": "MLP A (sens.)",
                 "mlp_B_v_sens": "MLP B (sens.)", "cnn_A_v_sens": "CNN A (sens.)", "cnn_B_v_sens": "CNN B (sens.)", "cnn_A_v_f1": "CNN A (F1)", "mlp_B_v_f1": "MLP B (F1)"}
        st.dataframe(pt_[list(cols_)].rename(columns=cols_).set_index("técnica"), width="stretch")

    sg = table("pdi/segmentacao.csv")
    if sg is not None:
        st.subheader("Segmentação: Dice médio por método e técnica")
        st.caption("Como ler: quanto mais escura a célula, mais a segmentação coincide com o especialista (1 = perfeita).")
        metodos = {"otsu": "Otsu", "adaptive": "adaptativa", "otsu_morph": "Otsu + morfologia", "adaptive_morph": "adaptativa + morfologia",
                   "region_grow": "crescimento de região", "watershed": "watershed"}
        w = sg.pivot(index="method", columns="preprocess", values="dice_mean")
        w = w.loc[[m for m in metodos if m in w.index], [t for t in NOMES if t in w.columns]].rename(index=metodos, columns=NOMES)
        st.dataframe(w.style.format("{:.3f}").background_gradient(cmap="Blues", vmin=0, vmax=0.7), width="stretch")

    cn, ml = table("cnn/classificacao.csv"), table("mlp/classificacao.csv")
    if cn is not None and ml is not None:
        st.subheader("Classificação: sensibilidade em primeiro lugar")
        st.caption("Como ler: sensibilidade = fração dos vídeos alterados que o modelo pegou (o que mais importa numa triagem); "
                   "especificidade = fração dos saudáveis reconhecidos como saudáveis.")
        cl = pd.concat([cn.assign(sistema="ResNet-18"), ml.assign(sistema=np.where(ml["modelo"] == "mlp18", "MLP (18 características)", "MLP (pixels, sem PDI)"))])
        cl["técnica"] = cl["tecnica"].map(NOMES).fillna(cl["tecnica"])
        st.dataframe(cl[["tarefa", "sistema", "técnica", "sensibilidade", "especificidade", "f1", "auc", "veredito_sensibilidade", "custo_total_s"]]
                     .rename(columns={"f1": "F1", "auc": "AUC", "veredito_sensibilidade": "veredito (sens.)", "custo_total_s": "custo por treino (s)"})
                     .set_index(["tarefa", "sistema", "técnica"]).style.format({"sensibilidade": "{:.3f}", "especificidade": "{:.3f}", "F1": "{:.3f}", "AUC": "{:.3f}",
                                                                                "custo por treino (s)": "{:.0f}"}), width="stretch")
        st.caption("Média de 5 folds × 3 sementes, por vídeo. Custo: PDI nos frames de treino + treino até a melhor validação.")

    st.subheader("Cuidado: o formato do vídeo")
    st.warning("A proporção da imagem, que muda com a câmera, sozinha separa as classes com AUC 0,66 na Tarefa A e 0,76 na B. "
               "Uma das câmeras do BAGLS tem 65% de vídeos alterados; as outras, 30%. Sem as bordas pretas, a MLP piora na Tarefa B, "
               "e a ResNet-18 quase não muda.")
    sb = table("sintese/sem_bordas.csv")
    if sb is not None:
        num_cols = ["f1_com_bordas", "f1_sem_bordas", "sensibilidade_com_bordas", "sensibilidade_sem_bordas", "auc_com_bordas", "auc_sem_bordas"]
        st.dataframe(sb[["modelo", "tarefa", *num_cols, "veredito_f1", "veredito_sensibilidade"]].set_index(["modelo", "tarefa"]).style.format("{:.3f}", subset=num_cols),
                     width="stretch")
        st.caption("Teste de controle: a mesma imagem sem PDI, recortada no quadrado central, sem as bordas pretas.")

    cf = table("custo/por_fase.csv")
    if cf is not None:
        st.subheader("Tempo e custo de cada fase (AMD Ryzen 5 5500, 15,9 GB, RTX 3060 12 GB)")
        st.dataframe(cf[["fase", "duracao_h", "cpu_nucleos_medio", "ram_processo_pico_gb", "gpu_usada", "gpu_energia_wh"]]
                     .rename(columns={"duracao_h": "horas", "cpu_nucleos_medio": "núcleos de CPU", "ram_processo_pico_gb": "RAM pico (GB)", "gpu_usada": "GPU",
                                      "gpu_energia_wh": "energia da GPU (Wh)"}).set_index("fase").style.format(precision=2), width="stretch")

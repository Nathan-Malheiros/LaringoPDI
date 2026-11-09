"""LaringoPDI — protótipo da apresentação: realce → filtragem e Fourier → segmentação da glote (clássica e MLP) →
classificação (PDI + MLP × CNN) → resultados do estudo.

Só usa frames de teste do fold 0: vídeos que nenhum modelo do app viu no treino. A barra lateral traz as cenas
da demonstração; cada cena ajusta frame, técnica, tarefa e passo. Rodar: `tools\\abrir_app.bat`."""
from __future__ import annotations

import json
import sys

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

AVISO = "Ferramenta de apoio acadêmico. Não substitui avaliação médica."
DIM, CY = "#5B6785", "#0E9FB5"
VERDE, LARANJA = (46, 158, 106), (224, 123, 46)                # contornos: especialista e segmentação obtida
ENTRADAS = {"E00": "sem PDI", "E15": "alargamento", "E02": "equalização global", "E03": "CLAHE", "E16": "média", "E06": "mediana",
            "E17": "passa-baixa (Fourier)", "E01": "remoção de reflexo", "E04": "homomórfica"}
NOMES = {"raw": "sem PDI", "stretch": "alargamento", "he": "equalização", "clahe": "CLAHE", "mean": "média", "median": "mediana",
         "gauss_lpf": "passa-baixa", "spec": "remoção de reflexo", "homo": "homomórfica"}
FAMILIA = {"healthy": "saudável", "functional": "distúrbio funcional", "structural": "lesão estrutural"}
PASSOS = ["Realce", "Filtragem e Fourier", "Segmentação da glote", "PDI + MLP × CNN", "Resultados do estudo"]
TAREFAS = {"A": "A · saudável × alterado", "B": "B · saudável × lesão estrutural"}

# Cenas da demonstração (docs/ROTEIRO_DEMO.md). Todos os frames são de vídeos do teste do fold 0.
CENAS = [
    {"titulo": "Realce num frame escuro", "nota": "O CLAHE mostra a glote sem estourar o tecido.", "passo": 0, "frame": "training/12188.png", "exp": "E03", "task": "A"},
    {"titulo": "Média × mediana no ruído", "nota": "Sal e pimenta a 5%: a mediana limpa, a média espalha.", "passo": 1, "frame": "training/12068.png",
     "ruido_tipo": "sal e pimenta", "ruido_dens": 5, "k_media": 3, "k_mediana": 3, "lpf_sigma": 15},
    {"titulo": "Segmentação passo a passo", "nota": "Limiar adaptativo, abertura, fechamento e componente central.", "passo": 2, "frame": "training/3672.png", "seg_clahe": False},
    {"titulo": "O CLAHE salva a segmentação", "nota": "Frame escuro: o crescimento de região vai de Dice 0,00 a 0,95.", "passo": 2, "frame": "training/12188.png", "seg_clahe": True, "exp": "E03"},
    {"titulo": "Quando o CLAHE atrapalha", "nota": "Frame claro: o Dice cai de 0,81 para 0,12.", "passo": 2, "frame": "training/11116.png", "seg_clahe": True},
    {"titulo": "Lesão estrutural: MLP × CNN", "nota": "Tarefa B com CLAHE; o Grad-CAM mostra onde a rede olhou.", "passo": 3, "frame": "training/7957.png", "exp": "E03", "task": "B"},
    {"titulo": "Vídeo saudável, sem PDI", "nota": "Trocar a técnica quase não muda a decisão.", "passo": 3, "frame": "training/12068.png", "exp": "E00", "task": "A"},
    {"titulo": "Resultados do estudo", "nota": "Hipóteses, veredito por técnica, custo e o teste sem bordas.", "passo": 4},
]

st.set_page_config(page_title="LaringoPDI", page_icon="🔬", layout="wide")
st.markdown("<style>.block-container { padding-top: 2rem; max-width: 1400px; }</style>", unsafe_allow_html=True)

cfg = C.load()
SIZE, WORK = cfg["data"]["image_size"], cfg["data"]["work_size"]
RUNS, TABLES, RAW = C.ROOT / "results" / "runs", C.ROOT / "results" / "tables", C.path(cfg, "raw")


@st.cache_data(show_spinner=False)
def frames_fold0() -> pd.DataFrame:
    """Frames do teste do fold 0 que existem no disco, com a classe do vídeo."""
    fr = D.load_frames(cfg)
    test = {str(v) for v in D.load_splits(cfg)[0]["test"]}
    fr = fr[fr["video_id"].astype(str).isin(test)]
    fr = fr[[(RAW / p).exists() for p in fr["path"]]]
    return fr[["path", "seg_path", "video_id", "family"]].reset_index(drop=True)


def has_weights(exp_: str) -> bool:
    return (RUNS / exp_ / "A" / "resnet18" / "fold0_seed0" / "model.pt").exists()


st.title("LaringoPDI")
st.caption("Processamento Digital de Imagens aplicado à laringoscopia: realce, segmentação e caracterização da glote. UNIFOR, Prof.ª Lyndainês Santos.")
st.warning(AVISO)

FR = frames_fold0()
if FR.empty:
    st.error("O BAGLS não está em data/raw/bagls. O app usa os frames de teste do estudo, que não são distribuídos com o código.")
    st.stop()
FRAMES = FR["path"].tolist()
DISP = [e for e in ENTRADAS if has_weights(e)] or ["E00"]
WIDGETS = ("passo", "frame", "exp", "task", "ruido_tipo", "ruido_dens", "ruido_sigma", "k_media", "k_mediana", "lpf_sigma", "seg_clahe")


def aplicar_cena(i: int) -> None:
    """Ajusta todos os controles de uma vez, para a demonstração sair igual ao ensaio."""
    c = CENAS[i]
    st.session_state["cena"] = i
    for k in WIDGETS:
        if k in c and (k != "frame" or c[k] in FRAMES) and (k != "exp" or c[k] in DISP):
            st.session_state[k] = c[k]


if "passo" not in st.session_state:                          # primeira carga: cena 1, ou ?aba=N&frame=ID (capturas)
    aplicar_cena(0)
    if "aba" in st.query_params:
        st.session_state["passo"] = int(st.query_params["aba"]) % len(PASSOS)
    want = st.query_params.get("frame")
    if want:
        st.session_state["frame"] = next((p for p in FRAMES if p.split("/")[-1].split(".")[0] == want), st.session_state["frame"])
for k, v in {"exp": "E03" if "E03" in DISP else DISP[0], "task": "A", "seg_clahe": False}.items():
    st.session_state.setdefault(k, v)

# ---------------------------------------------------------------------------------------- barra lateral
with st.sidebar:
    st.markdown("### Cenas da apresentação")
    st.caption("Clique numa cena: ela escolhe o frame, a técnica, a tarefa e o passo.")
    for i, c in enumerate(CENAS):
        st.button(f"{i + 1}. {c['titulo']}", key=f"cena_{i}", on_click=aplicar_cena, args=(i,), width="stretch",
                  type="primary" if st.session_state.get("cena") == i else "secondary")
    st.markdown("### Entrada")
    familia = dict(zip(FR["path"], FR["family"]))
    st.selectbox("Frame (teste do fold 0)", FRAMES, key="frame", format_func=lambda p: f"{p.split('/')[-1]} · {FAMILIA.get(familia[p], familia[p])}")
    st.caption("Só vídeos que os modelos do app não viram no treino.")
    st.markdown("### Classificação")
    st.radio("Tarefa", list(TAREFAS), key="task", format_func=TAREFAS.get)
    st.selectbox("Técnica de PDI na entrada dos modelos", DISP, key="exp", format_func=lambda e: ENTRADAS[e])
    st.caption("Vale para a MLP de segmentação, a MLP de classificação e a ResNet-18. Os modelos são os do fold 0.")

row = FR[FR["path"] == st.session_state["frame"]].iloc[0]
exp, task = st.session_state["exp"], st.session_state["task"]
gray = D.imread(RAW / row["path"], gray=True)
gt_full = D.imread(RAW / row["seg_path"], gray=True) > 127
base = to_work_size(gray, WORK)
gt = cv2.resize(gt_full.astype(np.uint8), base.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)

if st.session_state.get("cena") is not None:
    c = CENAS[st.session_state["cena"]]
    st.info(f"**Cena {st.session_state['cena'] + 1}: {c['titulo']}.** {c['nota']}")


@st.cache_resource(show_spinner=False)
def mlp_model(exp_: str, task_: str, kind: str):
    from laringopdi.mlp import load_mlp
    return load_mlp(cfg, exp_, task_, kind)


def tech(name: str, g: np.ndarray = base, **params) -> np.ndarray:
    return TECHNIQUES[name](g, {}, **(params or DEFAULTS[name]))


def hist_fig(img: np.ndarray, color: str = CY):
    v = img[img > 0] if (img > 0).mean() < .98 else img.ravel()
    fig, ax = plt.subplots(figsize=(3.2, 1.2), dpi=110)
    ax.hist(v.ravel(), bins=64, range=(0, 255), color=color, alpha=.9)
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([0, 128, 255])
    ax.tick_params(labelsize=6, colors=DIM, length=0)
    fig.tight_layout(pad=.2)
    return fig


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


def probs(names: list[str], p: np.ndarray) -> None:
    pt = {"healthy": "saudável", "disorder": "alterado", "structural": "lesão estrutural"}
    for n, pv in sorted(zip(names, p), key=lambda t: -t[1]):
        st.markdown(f"**{pt.get(n, n)}**: {pv:.1%}")
        st.progress(float(pv))


def table(name: str) -> pd.DataFrame | None:
    p = TABLES / name
    return pd.read_csv(p) if p.exists() else None


passo = st.segmented_control("Passo", list(range(len(PASSOS))), key="passo", label_visibility="collapsed", width="stretch",
                             format_func=lambda i: f"{i + 1}. {PASSOS[i]}")
if passo is None:                                            # clicar no passo já aberto não fecha a página
    passo = st.session_state.get("ultimo_passo", 0)
st.session_state["ultimo_passo"] = passo
st.caption(f"Frame `{row['path']}` · vídeo: {FAMILIA.get(row['family'], row['family'])} · técnica nos modelos: {ENTRADAS[exp]} · tarefa {TAREFAS[task]}")

# ---------------------------------------------------------------------------------------- 1 realce
if passo == 0:
    st.subheader("Realce")
    st.caption("Mesmo frame, quatro versões. O histograma omite os pixels pretos do fundo.")
    cols, rows = st.columns(4), []
    for c_, (t_, lab) in zip(cols, (("raw", "original"), ("stretch", "alargamento"), ("he", "equalização global"), ("clahe", "CLAHE"))):
        im = base if t_ == "raw" else tech(t_)
        qa = image_quality(im)
        c_.image(im, width="stretch", clamp=True)
        legenda(c_, lab)
        c_.pyplot(hist_fig(im, "#8A94AD" if t_ == "raw" else CY), width="stretch")
        rows.append({"versão": lab, "contraste RMS": qa["rms_contrast"], "entropia": qa["entropy"]})
    st.dataframe(pd.DataFrame(rows).set_index("versão").style.format({"contraste RMS": "{:.3f}", "entropia": "{:.2f}"}), width="stretch")
    st.caption("Contraste RMS mede o quanto a imagem varia; entropia mede a quantidade de informação. No estudo, só o CLAHE melhorou os três critérios.")

# ---------------------------------------------------------------------------------------- 2 filtragem
elif passo == 1:
    st.subheader("Média × mediana")
    c = st.columns(4)
    kind = c[0].radio("Tipo de ruído", ["sal e pimenta", "gaussiano"], key="ruido_tipo")
    if kind == "gaussiano":
        lvl = c[1].slider("Intensidade (σ)", 5, 40, 10, key="ruido_sigma")
        noisy = noise_image(base, "gauss", lvl, np.random.default_rng(0))
    else:
        lvl = c[1].slider("Densidade (%)", 1, 20, 5, key="ruido_dens")
        noisy = noise_image(base, "sp", lvl / 100, np.random.default_rng(0))
    km = c[2].select_slider("Janela da média", [3, 7, 15], 3, key="k_media")
    kd = c[3].select_slider("Janela da mediana", [3, 5], 3, key="k_mediana")
    fm, fd = tech("mean", noisy, ksize=km), tech("median", noisy, ksize=kd)
    c = st.columns(4)
    for c_, im, lab in zip(c, (base, noisy, fm, fd), ("original (referência)", "com ruído", f"média {km}×{km}", f"mediana {kd}×{kd}")):
        c_.image(im, width="stretch", clamp=True)
        legenda(c_, lab)
        if im is not base:
            ps, ss = psnr_ssim(base, im)
            c_.metric("PSNR", f"{ps:.1f} dB")
            c_.metric("SSIM", f"{ss:.3f}")

    st.subheader("Fourier: passa-baixa gaussiano")
    sigma = st.slider("σ de H(u,v) = exp(−D²/2σ²)", 5, 80, 15, key="lpf_sigma")
    lpf = tech("gauss_lpf", sigma=sigma)
    spec_f = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(base.astype(np.float64)))))
    c = st.columns(4)
    c[0].image(base, caption="original", width="stretch", clamp=True)
    c[1].image(cv2.applyColorMap(norm8(spec_f), cv2.COLORMAP_MAGMA)[..., ::-1], caption="espectro |F(u,v)|: o centro é a baixa frequência", width="stretch")
    c[2].image(cv2.applyColorMap(norm8(gauss_lpf_transfer(base.shape, sigma)), cv2.COLORMAP_VIRIDIS)[..., ::-1], caption=f"filtro H(u,v), σ = {sigma}", width="stretch")
    c[3].image(lpf, caption="resultado", width="stretch", clamp=True)
    c = st.columns(3)
    c[0].metric(f"PSNR com σ = {sigma}", f"{psnr_ssim(base, lpf)[0]:.1f} dB")
    c[1].metric("PSNR com σ = 15", f"{psnr_ssim(base, tech('gauss_lpf', sigma=15))[0]:.1f} dB")
    c[2].metric("PSNR com σ = 40", f"{psnr_ssim(base, tech('gauss_lpf', sigma=40))[0]:.1f} dB")
    st.caption("PSNR em relação ao frame original: quanto menor o σ, mais alta frequência é cortada e mais borrada fica a borda das pregas.")

# ---------------------------------------------------------------------------------------- 3 segmentação
elif passo == 2:
    usar_clahe = st.toggle("Aplicar CLAHE antes de segmentar", key="seg_clahe")
    gs = tech("clahe") if usar_clahe else base
    st.subheader("Passo a passo: limiar adaptativo + morfologia")
    s_ = adaptive_morph_steps(gs)
    c = st.columns(5)
    c[0].image(gs, width="stretch", clamp=True)
    legenda(c[0], "entrada", "com CLAHE" if usar_clahe else "sem realce")
    for c_, (k, lab, sub) in zip(c[1:], (("limiar", "limiar adaptativo", "janela 51×51"), ("abertura", "abertura", "tira os pontinhos"),
                                         ("fechamento", "fechamento", "fecha os buracos"), ("componente", "componente central", "fica só a glote"))):
        c_.image(s_[k].astype(np.uint8) * 255, width="stretch")
        legenda(c_, lab, sub)

    st.subheader("Comparação dos métodos neste frame")
    bc = best_config()
    pre = "clahe" if usar_clahe else "raw"
    metodos = [("Otsu global", segment(gs, "otsu")), ("Otsu + morfologia", segment(gs, "otsu_morph")), ("adaptativa + morfologia", s_["componente"]),
               ("crescimento de região (método do estudo)", apply_best(base, bc["method"], bc["params"], pre))]
    c = st.columns(4)
    for c_, (lab, mk) in zip(c, metodos):
        c_.image(outline(base, [(gt, VERDE), (mk, LARANJA)]), width="stretch")
        legenda(c_, lab)
        c_.metric("Dice", f"{dice(mk, gt):.3f}")
        c_.metric("IoU", f"{iou(mk, gt):.3f}")
    st.caption("Verde: máscara de referência (especialista). Laranja: segmentação obtida.")

    st.subheader(f"MLP de segmentação ({ENTRADAS[exp]})")
    mseg = mlp_model(exp, "seg", "mlp_seg")
    if mseg is None:
        st.info(f"Sem pesos da MLP de segmentação para {ENTRADAS[exp]}.")
    else:
        from laringopdi.mlp import _segment_region, segment_frame_mlp
        g224, ctx224 = run_pipeline(gray, cfg["experiments"][exp]["pipeline"], None, SIZE, WORK, gt_mask=gt_full)
        g224 = np.ascontiguousarray(g224[..., 0])
        gt224 = ctx224["gt_mask_lb"]
        pares = [("MLP de pixels (vizinhança 15×15)", segment_frame_mlp(cfg, mseg, g224)),
                 ("crescimento de região na mesma imagem", _segment_region(g224, bc["method"], bc["params"]))]
        c = st.columns(4)
        for c_, (lab, mk) in zip(c, pares):
            c_.image(outline(g224, [(gt224, VERDE), (mk, LARANJA)]), width="stretch")
            legenda(c_, lab)
            c_.metric("Dice", f"{dice(mk, gt224):.3f}")
        st.caption("No estudo, a melhor MLP de segmentação foi com a homomórfica (Dice 0,532); o melhor resultado geral continua sendo o "
                   "crescimento de região com CLAHE (0,614).")

    st.subheader("Bordas: abordagem complementar")
    blur = cv2.GaussianBlur(base, (5, 5), 0)
    c = st.columns(3)
    c[0].image(base, caption="original", width="stretch", clamp=True)
    c[1].image(sobel_mag(blur), caption="Sobel: gradiente", width="stretch", clamp=True)
    c[2].image(auto_canny(blur), caption="Canny: linhas finas e contínuas", width="stretch", clamp=True)

# ---------------------------------------------------------------------------------------- 4 classificação
elif passo == 3:
    F18_PT = {"shape_area_rel": "área relativa", "shape_perimeter_rel": "perímetro", "shape_eccentricity": "excentricidade",
              "shape_solidity": "solidez", "shape_aspect": "razão de aspecto", "shape_symmetry": "simetria", "int_mean": "média",
              "int_std": "desvio", "int_entropy": "entropia", "int_skew": "assimetria", "int_kurtosis": "curtose",
              "int_contrast": "contraste glote × tecido", "tex_canny_density": "densidade de bordas (Canny)", "tex_sobel_mean": "gradiente médio (Sobel)",
              "tex_glcm_contrast": "GLCM contraste", "tex_glcm_homogeneity": "GLCM homogeneidade", "tex_glcm_energy": "GLCM energia",
              "tex_lbp_entropy": "entropia do LBP"}
    from laringopdi.data import task_names
    names = task_names(cfg, task)
    o_, _ = run_pipeline(gray, cfg["experiments"][exp]["pipeline"], None, SIZE, WORK)
    esquerda, direita = st.columns(2)
    with esquerda:
        st.subheader("Braço 1: PDI + 18 características + MLP (18 → 32 → 16 → 2)")
        c6 = mlp_model(exp, task, "mlp18")
        if c6 is None:
            st.info(f"Sem pesos da MLP para {ENTRADAS[exp]} na Tarefa {task}.")
        else:
            from laringopdi.mlp import classify_frame18
            p6, vec = classify_frame18(cfg, c6, np.ascontiguousarray(o_[..., 0]))
            probs(names, p6)
            cols = st.columns(3)
            for c_, (fam, title) in zip(cols, (("f18shape_", "Forma"), ("f18int_", "Intensidade"), ("f18tex_", "Textura"))):
                legenda(c_, title)
                c_.dataframe(pd.DataFrame([{"medida": F18_PT.get(k.replace("f18", ""), k), "valor": v} for k, v in vec.items() if k.startswith(fam)])
                             .set_index("medida").style.format({"valor": "{:.3f}"}), width="stretch")
            st.caption("A MLP foi treinada com a média de 10 frames por vídeo; aqui recebe um único frame, por isso fica perto de 50%.")
    with direita:
        st.subheader(f"Braço 2: ResNet-18 sobre {ENTRADAS[exp]}")
        from laringopdi.evaluate import gradcam_maps, load_model
        model = load_model(cfg, exp, task, "resnet18", 0)
        if model is None:
            st.info(f"Sem pesos da ResNet-18 para {ENTRADAS[exp]} na Tarefa {task}.")
        else:
            import torch

            from laringopdi.train import _prep
            with torch.no_grad():
                prob = torch.softmax(model(_prep(torch.from_numpy(o_[None, ..., :3].copy()), torch.device("cpu"))), 1)[0].numpy()
            probs(names, prob)
            cam = gradcam_maps(model, o_[None, ..., :3])[0]
            heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1]
            c1, c2 = st.columns(2)
            c1.image(o_[..., 0], caption="entrada da rede", width="stretch", clamp=True)
            c2.image((0.55 * o_[..., :3] + 0.45 * heat).astype(np.uint8), caption="Grad-CAM: onde a rede olhou", width="stretch")
    st.caption("Modelos do fold 0. Saída de apoio, não diagnóstico; quando os dois braços discordam, a decisão é do profissional.")

# ---------------------------------------------------------------------------------------- 5 resultados
else:
    st.caption("Números do estudo inteiro (587 vídeos, 5 folds, 3 sementes; classificação por vídeo), lidos das tabelas geradas pelo código. Não dependem do frame escolhido.")
    st.success("**Em uma frase:** o PDI muda muito a segmentação (CLAHE + crescimento de região: Dice 0,614 contra 0,467) e quase nada a classificação "
               "(62 de 64 vereditos neutros). A MLP simples empatou com a ResNet-18 em F1, a uma fração do custo.")
    sys.path.insert(0, str(C.ROOT))
    hp = TABLES / "sintese" / "hipoteses.json"
    if hp.exists():
        from tools.documento_resultados import R as _R
        st.subheader("Hipóteses da Etapa 1")
        hyp = json.loads(hp.read_text(encoding="utf-8"))
        st.dataframe(pd.DataFrame([{"ID": k, "hipótese": txt, "veredito": hyp[k]["veredito"], "evidência": ev.replace("&lt;", "<")} for k, txt, ev in _R["hipoteses"]]).set_index("ID"),
                     width="stretch")

    pt_ = table("sintese/por_tecnica.csv")
    if pt_ is not None:
        st.subheader("Veredito de cada técnica, em cada etapa")
        cols_ = {"nome": "técnica", "visual": "imagem", "seg_classica": "segmentação clássica", "seg_mlp": "segmentação MLP", "mlp_A_v_sens": "MLP A (sens.)",
                 "mlp_B_v_sens": "MLP B (sens.)", "cnn_A_v_sens": "CNN A (sens.)", "cnn_B_v_sens": "CNN B (sens.)", "cnn_A_v_f1": "CNN A (F1)", "mlp_B_v_f1": "MLP B (F1)"}
        st.dataframe(pt_[list(cols_)].rename(columns=cols_).set_index("técnica"), width="stretch")
        st.caption("Contra a imagem sem PDI. Melhora ou piora só com IC 95% fora do zero e teste corrigido por Holm abaixo de 0,05. Neutra não prova efeito zero.")

    sg = table("pdi/segmentacao.csv")
    if sg is not None:
        st.subheader("Segmentação: Dice médio por método e técnica")
        metodos = {"otsu": "Otsu", "adaptive": "adaptativa", "otsu_morph": "Otsu + morfologia", "adaptive_morph": "adaptativa + morfologia",
                   "region_grow": "crescimento de região", "watershed": "watershed"}
        w = sg.pivot(index="method", columns="preprocess", values="dice_mean")
        w = w.loc[[m for m in metodos if m in w.index], [t for t in NOMES if t in w.columns]].rename(index=metodos, columns=NOMES)
        st.dataframe(w.style.format("{:.3f}").background_gradient(cmap="Blues", vmin=0, vmax=0.7), width="stretch")

    cn, ml = table("cnn/classificacao.csv"), table("mlp/classificacao.csv")
    if cn is not None and ml is not None:
        st.subheader("Classificação: sensibilidade em primeiro lugar")
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

st.divider()
st.caption(f"LaringoPDI · UNIFOR · Processamento Digital de Imagens · {AVISO}")

"""LaringoPDI — protótipo, na ordem da apresentação: frame → realce → filtragem e Fourier →
segmentação da glote (clássica e MLP) → classificação (PDI + MLP × CNN) → resultados do estudo.
Rodar: `streamlit run app/streamlit_app.py`."""
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
from laringopdi.etapa1 import noise_image
from laringopdi.pdi import run_pipeline
from laringopdi.pdi.enhance import gauss_lpf_transfer
from laringopdi.pdi.quality import image_quality
from laringopdi.pdi.registry import DEFAULTS, TECHNIQUES, to_work_size
from laringopdi.pdi.segmentation import adaptive_morph_steps, apply_best, best_config, dice, iou, segment
from laringopdi.pdi.texture import auto_canny, sobel_mag

AVISO = "Ferramenta de apoio acadêmico. Não substitui avaliação médica."
INK, DIM, CY, OR, GR = "#14213D", "#5B6785", "#0E9FB5", "#E07B2E", "#2E9E6A"
ENTRADAS = {"E00": "sem PDI", "E15": "alargamento", "E02": "equalização global", "E03": "CLAHE", "E16": "média", "E06": "mediana",
            "E17": "passa-baixa (Fourier)", "E01": "remoção de reflexo", "E04": "homomórfica"}
NOMES = {"raw": "sem PDI", "stretch": "alargamento", "he": "equalização", "clahe": "CLAHE", "mean": "média", "median": "mediana",
         "gauss_lpf": "passa-baixa", "spec": "remoção de reflexo", "homo": "homomórfica"}

st.set_page_config(page_title="LaringoPDI", page_icon="🔬", layout="wide")
st.markdown("""
<style>
.block-container { padding-top: 1.6rem; max-width: 1400px; }
.hero { background: linear-gradient(120deg, #14213D 0%, #0B4F63 60%, #0E9FB5 100%); color: #fff; border-radius: 18px; padding: 24px 30px; margin-bottom: 14px; }
.hero h1 { font-size: 2.1rem; margin: 0; font-weight: 700; letter-spacing: -.02em; color: #fff; }
.hero p { margin: 6px 0 0; opacity: .9; font-size: 1.02rem; }
.step { font-weight: 600; color: #14213D; font-size: .92rem; margin: 4px 0 0; }
.sub { color: #5B6785; font-size: .8rem; margin: 0 0 6px; }
.card { border: 1px solid #E2E7F0; border-radius: 14px; padding: 14px 16px; background: #FBFCFE; margin-bottom: 12px; }
</style>""", unsafe_allow_html=True)

cfg = C.load()
SIZE, WORK = cfg["data"]["image_size"], cfg["data"]["work_size"]
RUNS, TABLES = C.ROOT / "results" / "runs", C.ROOT / "results" / "tables"
st.markdown('<div class="hero"><h1>LaringoPDI</h1><p>Processamento Digital de Imagens aplicado à laringoscopia: realce, segmentação e '
            'caracterização da glote para apoiar a triagem de alterações nas pregas vocais.</p></div>', unsafe_allow_html=True)
st.warning(AVISO, icon="⚠️")

bagls = C.ROOT / "data" / "raw" / "bagls" / "test"
examples = sorted((p for p in bagls.glob("*.png") if not p.stem.endswith("_seg")), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)[:60] if bagls.exists() else []
demo = [bagls / f"{d}.png" for d in cfg.get("app", {}).get("demo_frames", []) if (bagls / f"{d}.png").exists()]
examples = demo + [p for p in examples if p not in demo]


def has_weights(exp_: str) -> bool:
    return (RUNS / exp_ / "A" / "resnet18" / "fold0_seed0" / "model.pt").exists()


with st.sidebar:
    st.markdown("### Entrada")
    src = st.radio("Origem do frame", ["Exemplo local (BAGLS)", "Enviar arquivo"] if examples else ["Enviar arquivo"])
    st.markdown("### Classificação")
    task = st.radio("Tarefa", ["A", "B"], format_func=lambda t: {"A": "A · saudável × alterado", "B": "B · saudável × lesão estrutural"}[t])
    disp = [e for e in ENTRADAS if has_weights(e)] or ["E00"]
    exp = st.selectbox("Técnica de PDI na entrada dos modelos", disp, index=disp.index("E03") if "E03" in disp else 0, format_func=lambda e: ENTRADAS[e])
    st.caption("Vale para a MLP de segmentação, a MLP de classificação e a ResNet-18. Os modelos são os do fold 0.")


def _decode(b: bytes, flag):
    return cv2.imdecode(np.frombuffer(b, np.uint8), flag)


if src.startswith("Exemplo"):
    with st.sidebar:
        want = st.query_params.get("frame")          # ?frame=1881 abre direto num frame (capturas e demonstração)
        pick = st.selectbox("Frame", examples, index=next((i for i, p in enumerate(examples) if p.stem == want), 0), format_func=lambda p: p.name)
        st.caption("Exemplos lidos do BAGLS local (não distribuídos com o app).")
    img_b = pick.read_bytes()
    seg_p = pick.with_name(f"{pick.stem}_seg.png")
    mask_b = seg_p.read_bytes() if seg_p.exists() else None
else:
    up = st.file_uploader("Frame de laringoscopia (PNG/JPG)", type=["png", "jpg", "jpeg"])
    st.caption("A imagem é convertida para escala de cinza. A segmentação foi calibrada no BAGLS: em imagens de outra origem, "
               "o resultado pode ser pior que o medido no estudo.")
    up_mask = st.file_uploader("Máscara da glote (opcional; necessária para calcular o Dice)", type=["png"])
    if up is None:
        st.info("Envie um frame para começar. Os dados do BAGLS não são distribuídos com o app.")
        st.caption(AVISO)
        st.stop()
    img_b, mask_b = up.getvalue(), (up_mask.getvalue() if up_mask is not None else None)

gray = _decode(img_b, cv2.IMREAD_GRAYSCALE)
base = to_work_size(gray, WORK)
gt = gt_full = None
if mask_b is not None:
    gt_full = _decode(mask_b, cv2.IMREAD_GRAYSCALE) > 127
    gt = cv2.resize(gt_full.astype(np.uint8), base.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)


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


def step(c_, title: str, sub: str = "") -> None:
    c_.markdown(f'<p class="step">{title}</p><p class="sub">{sub}</p>', unsafe_allow_html=True)


def outline(g: np.ndarray, masks: list[tuple[np.ndarray, tuple]]) -> np.ndarray:
    v = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)
    for mk, col in masks:
        cnts, _ = cv2.findContours(mk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(v, cnts, -1, col, 2)
    return v


def table(name: str) -> pd.DataFrame | None:
    p = TABLES / name
    return pd.read_csv(p) if p.exists() else None


TAB_NAMES = ["① Realce", "② Filtragem e Fourier", "③ Segmentação da glote", "④ PDI + MLP × CNN", "⑤ Resultados do estudo"]
first = int(st.query_params.get("aba", 0)) % len(TAB_NAMES)   # ?aba=N abre direto numa aba (capturas)
order = list(range(first, len(TAB_NAMES))) + list(range(first))
tabs = dict(zip(order, st.tabs([TAB_NAMES[i] for i in order])))

# ---------------------------------------------------------------------------------------- ① realce
with tabs[0]:
    st.caption("Mesmo frame, quatro versões. O histograma omite os pixels pretos do fundo.")
    cols, rows = st.columns(4), []
    for c_, (t_, lab) in zip(cols, (("raw", "original"), ("stretch", "alargamento"), ("he", "equalização global"), ("clahe", "CLAHE"))):
        im = base if t_ == "raw" else tech(t_)
        qa = image_quality(im)
        c_.image(im, width="stretch", clamp=True)
        step(c_, lab)
        c_.pyplot(hist_fig(im, "#8A94AD" if t_ == "raw" else CY), width="stretch")
        rows.append({"versão": lab, "contraste RMS": qa["rms_contrast"], "entropia": qa["entropy"]})
    st.dataframe(pd.DataFrame(rows).set_index("versão").style.format({"contraste RMS": "{:.3f}", "entropia": "{:.2f}"}), width="stretch")
    st.caption("Contraste RMS mede o quanto a imagem varia; entropia mede a quantidade de informação.")

# ---------------------------------------------------------------------------------------- ② filtragem
with tabs[1]:
    st.markdown("#### Média × mediana")
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
        step(c_, lab)
        if im is not base:
            ps, ss = psnr_ssim(base, im)
            c_.metric("PSNR", f"{ps:.1f} dB")
            c_.metric("SSIM", f"{ss:.3f}")

    st.markdown("#### Fourier: passa-baixa gaussiano")
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

# ---------------------------------------------------------------------------------------- ③ segmentação
with tabs[2]:
    usar_clahe = st.toggle("Aplicar CLAHE antes de segmentar", value=False, key="seg_clahe")
    gs = tech("clahe") if usar_clahe else base
    st.markdown("#### Passo a passo: limiar adaptativo + morfologia")
    s_ = adaptive_morph_steps(gs)
    c = st.columns(5)
    c[0].image(gs, width="stretch", clamp=True)
    step(c[0], "entrada", "com CLAHE" if usar_clahe else "sem realce")
    for c_, (k, lab, sub) in zip(c[1:], (("limiar", "limiar adaptativo", "janela 51×51"), ("abertura", "abertura", "tira os pontinhos"),
                                         ("fechamento", "fechamento", "fecha os buracos"), ("componente", "componente central", "fica só a glote"))):
        c_.image(s_[k].astype(np.uint8) * 255, width="stretch")
        step(c_, lab, sub)

    st.markdown("#### Comparação dos métodos neste frame")
    bc = best_config()
    pre = bc.get("preprocess", "raw")
    metodos = [("Otsu global", segment(gs, "otsu")), ("Otsu + morfologia", segment(gs, "otsu_morph")), ("adaptativa + morfologia", s_["componente"]),
               (f"crescimento de região + {pre.upper()} (método do estudo)", apply_best(base, bc["method"], bc["params"], pre))]
    c = st.columns(4)
    for c_, (lab, mk) in zip(c, metodos):
        layers = ([(gt, (46, 158, 106))] if gt is not None else []) + [(mk, (224, 123, 46))]
        c_.image(outline(base, layers), width="stretch")
        step(c_, lab)
        if gt is not None:
            c_.metric("Dice", f"{dice(mk, gt):.3f}")
            c_.metric("IoU", f"{iou(mk, gt):.3f}")
    st.caption("Verde: máscara de referência (especialista). Laranja: segmentação obtida." if gt is not None
               else "Laranja: segmentação obtida. Envie a máscara de referência para calcular Dice e IoU.")

    st.markdown(f"#### MLP de segmentação: cada pixel decidido pela vizinhança 15×15 ({ENTRADAS[exp]})")
    mseg = mlp_model(exp, "seg", "mlp_seg")
    if mseg is None:
        st.info(f"Sem pesos da MLP de segmentação para {ENTRADAS[exp]}.")
    else:
        g224, ctx224 = run_pipeline(gray, cfg["experiments"][exp]["pipeline"], None, SIZE, WORK, gt_mask=gt_full)
        g224 = np.ascontiguousarray(g224[..., 0])
        from laringopdi.mlp import segment_frame_mlp
        from laringopdi.pdi.segmentation import best_config as _bc
        from laringopdi.mlp import _segment_region
        _b = _bc()
        pares = [("MLP de pixels", segment_frame_mlp(cfg, mseg, g224)), ("crescimento de região na mesma imagem", _segment_region(g224, _b["method"], _b["params"]))]
        gt224 = ctx224.get("gt_mask_lb")
        c = st.columns(2)
        for c_, (lab, mk) in zip(c, pares):
            c_.image(outline(g224, ([(gt224, (46, 158, 106))] if gt224 is not None else []) + [(mk, (224, 123, 46))]), width="stretch")
            step(c_, lab)
            if gt224 is not None:
                c_.metric("Dice", f"{dice(mk, gt224):.3f}")
        st.caption("No estudo, a melhor MLP de segmentação foi com a homomórfica (Dice 0,532); o melhor resultado geral continua sendo o "
                   "crescimento de região com CLAHE (0,614).")

    st.markdown("#### Bordas: abordagem complementar")
    blur = cv2.GaussianBlur(base, (5, 5), 0)
    c = st.columns(3)
    c[0].image(base, caption="original", width="stretch", clamp=True)
    c[1].image(sobel_mag(blur), caption="Sobel: gradiente", width="stretch", clamp=True)
    c[2].image(auto_canny(blur), caption="Canny: linhas finas e contínuas", width="stretch", clamp=True)


# ---------------------------------------------------------------------------------------- ④ classificação
F18_PT = {"shape_area_rel": "área relativa", "shape_perimeter_rel": "perímetro", "shape_eccentricity": "excentricidade",
          "shape_solidity": "solidez", "shape_aspect": "razão de aspecto", "shape_symmetry": "simetria", "int_mean": "média",
          "int_std": "desvio", "int_entropy": "entropia", "int_skew": "assimetria", "int_kurtosis": "curtose",
          "int_contrast": "contraste glote × tecido", "tex_canny_density": "densidade de bordas (Canny)", "tex_sobel_mean": "gradiente médio (Sobel)",
          "tex_glcm_contrast": "GLCM contraste", "tex_glcm_homogeneity": "GLCM homogeneidade", "tex_glcm_energy": "GLCM energia",
          "tex_lbp_entropy": "entropia do LBP"}

with tabs[3]:
    from laringopdi.data import task_names
    names = task_names(cfg, task)
    pt = {"healthy": "saudável", "disorder": "alterado", "structural": "lesão estrutural"}
    esquerda, direita = st.columns(2)
    with esquerda:
        st.markdown(f"#### Braço 1: {ENTRADAS[exp]} + 18 características + MLP (18 → 32 → 16 → 2)")
        c6 = mlp_model(exp, task, "mlp18")
        if c6 is None:
            st.warning(f"Modelo da MLP para {ENTRADAS[exp]} na Tarefa {task} não encontrado. Rode `python -m laringopdi mlp-class` ou use o pacote com pesos.")
        else:
            from laringopdi.mlp import classify_frame18
            o_, _ = run_pipeline(gray, cfg["experiments"][exp]["pipeline"], None, SIZE, WORK)
            p6, vec = classify_frame18(cfg, c6, np.ascontiguousarray(o_[..., 0]))
            for n, pv in sorted(zip(names, p6), key=lambda t: -t[1]):
                st.markdown(f"**{pt.get(n, n)}**: {pv:.1%}")
                st.progress(float(pv))
            cols = st.columns(3)
            for c_, (fam, title) in zip(cols, (("f18shape_", "Forma"), ("f18int_", "Intensidade"), ("f18tex_", "Textura"))):
                step(c_, title)
                c_.dataframe(pd.DataFrame([{"medida": F18_PT.get(k.replace("f18", ""), k), "valor": v} for k, v in vec.items() if k.startswith(fam)])
                             .set_index("medida").style.format({"valor": "{:.3f}"}), width="stretch")
            st.caption("A MLP foi treinada com a média de 10 frames por vídeo; aqui recebe um único frame.")
    with direita:
        st.markdown(f"#### Braço 2: ResNet-18 sobre {ENTRADAS[exp]}")
        from laringopdi.evaluate import gradcam_maps, load_model
        model = load_model(cfg, exp, task, "resnet18", 0)
        if model is None:
            st.info(f"Sem pesos da ResNet-18 para {ENTRADAS[exp]} na Tarefa {task}.")
        else:
            import torch

            from laringopdi.train import _prep
            out, _ = run_pipeline(gray, cfg["experiments"][exp]["pipeline"], None, SIZE, WORK)
            with torch.no_grad():
                prob = torch.softmax(model(_prep(torch.from_numpy(out[None, ..., :3].copy()), torch.device("cpu"))), 1)[0].numpy()
            for n, pv in sorted(zip(names, prob), key=lambda t: -t[1]):
                st.markdown(f"**{pt.get(n, n)}**: {pv:.1%}")
                st.progress(float(pv))
            cam = gradcam_maps(model, out[None, ..., :3])[0]
            heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1]
            c1, c2 = st.columns(2)
            c1.image(out[..., 0], caption="entrada da rede", width="stretch", clamp=True)
            c2.image((0.55 * out[..., :3] + 0.45 * heat).astype(np.uint8), caption="Grad-CAM: onde a rede olhou", width="stretch")
    st.caption("Modelos do fold 0. Saída de apoio, não diagnóstico; quando os dois braços discordam, é exatamente por isso que a decisão é do profissional.")

# ---------------------------------------------------------------------------------------- ⑤ resultados
with tabs[4]:
    st.caption("Números do estudo inteiro (587 vídeos, 5 folds; classificação por vídeo), lidos das tabelas geradas pelo código. Não dependem do frame escolhido.")
    st.info("**Em uma frase:** o PDI muda muito a segmentação (CLAHE + crescimento de região: Dice 0,614 contra 0,467) e quase nada a classificação "
            "(62 de 64 vereditos neutros). A MLP simples empatou com a ResNet-18 em F1, a uma fração do custo.")
    sys.path.insert(0, str(C.ROOT))
    hp = TABLES / "sintese" / "hipoteses.json"
    if hp.exists():
        from tools.documento_resultados import R as _R
        st.markdown("#### Hipóteses da Etapa 1")
        hyp = json.loads(hp.read_text(encoding="utf-8"))
        st.dataframe(pd.DataFrame([{"ID": k, "hipótese": txt, "veredito": hyp[k]["veredito"], "evidência": ev.replace("&lt;", "<")} for k, txt, ev in _R["hipoteses"]]).set_index("ID"),
                     width="stretch")

    pt_ = table("sintese/por_tecnica.csv")
    if pt_ is not None:
        st.markdown("#### Veredito de cada técnica, em cada etapa (contra a imagem sem PDI)")
        cols_ = {"nome": "técnica", "visual": "imagem", "seg_classica": "segmentação clássica", "seg_mlp": "segmentação MLP", "mlp_A_v_sens": "MLP A (sens.)",
                 "mlp_B_v_sens": "MLP B (sens.)", "cnn_A_v_sens": "CNN A (sens.)", "cnn_B_v_sens": "CNN B (sens.)", "cnn_A_v_f1": "CNN A (F1)", "mlp_B_v_f1": "MLP B (F1)"}
        st.dataframe(pt_[list(cols_)].rename(columns=cols_).set_index("técnica"), width="stretch")
        st.caption("Melhora ou piora só com IC 95% fora do zero e teste corrigido por Holm abaixo de 0,05. Neutra não prova efeito zero.")

    sg = table("pdi/segmentacao.csv")
    if sg is not None:
        st.markdown("#### Segmentação clássica: Dice médio no teste, por método e técnica")
        metodos = {"otsu": "Otsu", "adaptive": "adaptativa", "otsu_morph": "Otsu + morfologia", "adaptive_morph": "adaptativa + morfologia",
                   "region_grow": "crescimento de região", "watershed": "watershed"}
        w = sg.pivot(index="method", columns="preprocess", values="dice_mean")
        w = w.loc[[m for m in metodos if m in w.index], [t for t in NOMES if t in w.columns]].rename(index=metodos, columns=NOMES)
        st.dataframe(w.style.format("{:.3f}").background_gradient(cmap="Blues", vmin=0, vmax=0.7), width="stretch")

    cn, ml = table("cnn/classificacao.csv"), table("mlp/classificacao.csv")
    if cn is not None and ml is not None:
        st.markdown("#### Classificação: sensibilidade em primeiro lugar")
        cl = pd.concat([cn.assign(sistema="ResNet-18"), ml.assign(sistema=np.where(ml["modelo"] == "mlp18", "MLP (18 características)", "MLP (pixels, sem PDI)"))])
        cl["técnica"] = cl["tecnica"].map(NOMES).fillna(cl["tecnica"])
        st.dataframe(cl[["tarefa", "sistema", "técnica", "sensibilidade", "especificidade", "f1", "auc", "veredito_sensibilidade", "custo_total_s"]]
                     .rename(columns={"f1": "F1", "auc": "AUC", "veredito_sensibilidade": "veredito (sens.)", "custo_total_s": "custo por treino (s)"})
                     .set_index(["tarefa", "sistema", "técnica"]).style.format({"sensibilidade": "{:.3f}", "especificidade": "{:.3f}", "F1": "{:.3f}", "AUC": "{:.3f}",
                                                                                "custo por treino (s)": "{:.0f}"}), width="stretch")
        st.caption("Média de 5 folds × 3 sementes, por vídeo. Custo: PDI nos frames de treino + treino até a melhor validação.")

    sb = table("sintese/sem_bordas.csv")
    st.markdown("#### Cuidado: o formato do vídeo")
    st.warning("A proporção da imagem (que muda com o hospital e o equipamento) sozinha separa as classes com AUC 0,66 na Tarefa A e 0,76 na B. "
               "Parte do que os modelos aprendem pode ser o equipamento, e não a voz.")
    if sb is not None:
        num_cols = ["f1_com_bordas", "f1_sem_bordas", "sensibilidade_com_bordas", "sensibilidade_sem_bordas", "auc_com_bordas", "auc_sem_bordas"]
        st.dataframe(sb[["modelo", "tarefa", *num_cols, "veredito_f1", "veredito_sensibilidade"]].set_index(["modelo", "tarefa"]).style.format("{:.3f}", subset=num_cols),
                     width="stretch")
        st.caption("Controle: a mesma imagem sem PDI, recortada no quadrado central, sem bordas pretas nem proporção do vídeo. A MLP piora na Tarefa B; "
                   "a ResNet-18 quase não muda. Uma das câmeras do BAGLS tem 65% de vídeos alterados; as outras, 30%.")

    cf = table("custo/por_fase.csv")
    if cf is not None:
        st.markdown("#### Tempo e custo de cada fase (AMD Ryzen 5 5500, 15,9 GB, RTX 3060 12 GB)")
        st.dataframe(cf[["fase", "duracao_h", "cpu_nucleos_medio", "ram_processo_pico_gb", "gpu_usada", "gpu_energia_wh"]]
                     .rename(columns={"duracao_h": "horas", "cpu_nucleos_medio": "núcleos de CPU", "ram_processo_pico_gb": "RAM pico (GB)", "gpu_usada": "GPU",
                                      "gpu_energia_wh": "energia da GPU (Wh)"}).set_index("fase").style.format(precision=2), width="stretch")

st.divider()
st.caption(f"LaringoPDI · UNIFOR · Processamento Digital de Imagens · {AVISO}")

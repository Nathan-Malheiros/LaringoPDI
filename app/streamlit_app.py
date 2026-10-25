"""LaringoPDI — protótipo com foco em PDI: frame → pipeline passo a passo (com histogramas) →
laboratório com as 14 técnicas → domínio da frequência → segmentação clássica da glote →
predição + Grad-CAM com o veredito do estudo. Rodar: `streamlit run app/streamlit_app.py`."""
from __future__ import annotations

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from laringopdi import config as C
from laringopdi.pdi import parse, run_pipeline
from laringopdi.pdi.enhance import homo_transfer
from laringopdi.pdi.quality import image_quality
from laringopdi.pdi.registry import DEFAULTS, TECHNIQUES, to_work_size
from laringopdi.pdi.segmentation import apply_best, best_config

AVISO = "Ferramenta de apoio acadêmico. Não substitui avaliação médica."
INK, DIM, CY, OR, GR, RD = "#14213D", "#5B6785", "#0E9FB5", "#E07B2E", "#2E9E6A", "#D6455D"
STEP_PT = {"raw": "redimensiona + pad 224", "fov": "campo de visão", "spec": "remove reflexos", "he": "equalização global",
           "clahe": "CLAHE", "homo": "homomórfica (FFT)", "bilat": "bilateral", "median": "mediana", "roi_gt": "recorte pelo GT",
           "roi": "recorte pela glote segmentada", "edges": "[cinza, Sobel, Canny]", "lbp": "LBP uniforme",
           "m1": "fusão M1", "m2": "fusão M2"}

st.set_page_config(page_title="LaringoPDI", page_icon="🔬", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500&display=swap');
html, body, [class*="css"], .stMarkdown, .stText { font-family: 'IBM Plex Sans', sans-serif; }
.block-container { padding-top: 1.6rem; max-width: 1400px; }
.hero { background: linear-gradient(120deg, #14213D 0%, #0B4F63 60%, #0E9FB5 100%); color: #fff; border-radius: 18px;
        padding: 26px 30px; margin-bottom: 14px; }
.hero h1 { font-size: 2.1rem; margin: 0; font-weight: 700; letter-spacing: -.02em; color: #fff; }
.hero p { margin: 6px 0 0; opacity: .88; font-size: 1.02rem; }
.aviso { background: #FFF6E9; border-left: 4px solid #E07B2E; padding: 9px 14px; border-radius: 8px; color: #7A3E0C;
         font-size: .92rem; margin-bottom: 10px; }
.chip { display: inline-block; font-family: 'IBM Plex Mono', monospace; font-size: .78rem; padding: 3px 9px; border-radius: 99px;
        background: #E6F6F8; color: #0B6F80; margin: 0 4px 4px 0; }
.step { font-weight: 600; color: #14213D; font-size: .92rem; margin: 4px 0 0; }
.sub { color: #5B6785; font-size: .8rem; margin: 0 0 6px; }
.card { border: 1px solid #E2E7F0; border-radius: 14px; padding: 14px 16px; background: #FBFCFE; }
.badge { display: inline-block; padding: 4px 12px; border-radius: 99px; font-weight: 600; font-size: .85rem; }
.b-neutra { background: #EEF1F6; color: #3A4660; } .b-melhora { background: #E3F5EC; color: #1D7A50; }
.b-piora { background: #FCE8EC; color: #A3283F; }
div[data-testid="stMetricValue"] { font-size: 1.45rem; }
</style>""", unsafe_allow_html=True)

cfg = C.load()
SIZE, WORK = cfg["data"]["image_size"], cfg["data"]["work_size"]
st.markdown('<div class="hero"><h1>LaringoPDI</h1><p>Processamento Digital de Imagens em videoendoscopia laríngea: '
            'o que cada técnica faz no frame, e o que isso muda para a segmentação da glote e para a rede neural.</p></div>',
            unsafe_allow_html=True)
st.warning(AVISO, icon="⚠️")

exps = {e: v for e, v in cfg["experiments"].items() if v.get("pipeline")}
bagls = C.ROOT / "data" / "raw" / "bagls" / "test"
examples = sorted((p for p in bagls.glob("*.png") if not p.stem.endswith("_seg")), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)[:60] \
    if bagls.exists() else []

with st.sidebar:
    st.markdown("### Entrada")
    src = st.radio("Origem do frame", ["Exemplo local (BAGLS)", "Enviar arquivo"] if examples else ["Enviar arquivo"])
    st.markdown("### Pipeline de PDI")
    exp = st.selectbox("Experimento", list(exps), index=list(exps).index("E04") if "E04" in exps else 0,
                       format_func=lambda e: f"{e} · {exps[e]['pipeline']}")
    st.caption(exps[exp]["desc"])
    st.markdown("### Classificador")
    task = st.radio("Tarefa", ["A", "B"], format_func=lambda t: {"A": "A · saudável × distúrbio", "B": "B · saudável × lesão estrutural"}[t])
    model_name = st.selectbox("Modelo", ["resnet18", "simplecnn"], format_func=lambda m: {"resnet18": "ResNet-18 (fine-tuning)",
                                                                                          "simplecnn": "SimpleCNN (do zero)"}[m])


def _decode(b: bytes, flag):
    return cv2.imdecode(np.frombuffer(b, np.uint8), flag)


if src.startswith("Exemplo"):
    with st.sidebar:
        pick = st.selectbox("Frame", examples, format_func=lambda p: p.name)
        st.caption("Exemplos lidos do BAGLS local (não distribuídos com o app).")
    img_b = pick.read_bytes()
    seg_p = pick.with_name(f"{pick.stem}_seg.png")
    mask_b = seg_p.read_bytes() if seg_p.exists() else None
else:
    up = st.file_uploader("Frame de laringoscopia (PNG/JPG)", type=["png", "jpg", "jpeg", "bmp"])
    up_mask = st.file_uploader("Máscara da glote (opcional; necessária para `roi_gt` e para o Dice)", type=["png"])
    if up is None:
        st.info("Envie um frame para começar. Os dados do BAGLS não são distribuídos com o app.")
        st.caption(AVISO)
        st.stop()
    img_b, mask_b = up.getvalue(), (up_mask.getvalue() if up_mask is not None else None)

gray = _decode(img_b, cv2.IMREAD_GRAYSCALE)
gt = _decode(mask_b, cv2.IMREAD_GRAYSCALE) > 127 if mask_b is not None else None
pipe = exps[exp]["pipeline"]
if "roi_gt" in pipe and gt is None:
    st.error("`roi_gt` exige a máscara de ground truth.")
    st.stop()


@st.cache_data(show_spinner=False)
def run(img_bytes: bytes, mask_bytes: bytes | None, pipeline: str, params: dict | None):
    g = _decode(img_bytes, cv2.IMREAD_GRAYSCALE)
    m = _decode(mask_bytes, cv2.IMREAD_GRAYSCALE) > 127 if mask_bytes is not None else None
    out, ctx = run_pipeline(g, pipeline, params, SIZE, WORK, gt_mask=m)
    q = image_quality(out, ctx)
    keep = {k: v for k, v in ctx.items() if k in ("spec_mask", "roi_bbox", "roi_fallback")}
    return out, q, keep


def view(out: np.ndarray) -> np.ndarray:
    """Multicanal (bordas, LBP, fusões) aparece em falsa cor; cinza replicado aparece em cinza."""
    return out[..., :3] if out.shape[2] > 1 and not np.array_equal(out[..., 0], out[..., 1]) else out[..., 0]


def hist_fig(img: np.ndarray, color: str = CY):
    v = img[img > 0] if (img > 0).mean() < .98 else img.ravel()
    fig, ax = plt.subplots(figsize=(3.2, 1.25), dpi=110)
    ax.hist(v.ravel(), bins=64, range=(0, 255), color=color, alpha=.85)
    cdf = np.cumsum(np.histogram(v, 64, (0, 255))[0]); ax2 = ax.twinx()
    ax2.plot(np.linspace(0, 255, 64), cdf / max(cdf[-1], 1), color=INK, lw=1)
    for a in (ax, ax2):
        a.set_yticks([]); [s.set_visible(False) for s in a.spines.values()]
    ax.set_xticks([0, 128, 255]); ax.tick_params(labelsize=6, colors=DIM, length=0)
    fig.tight_layout(pad=.2)
    return fig


def metric_row(q: dict, ref: dict | None = None):
    names = [("entropy", "Entropia", "{:.2f}"), ("rms_contrast", "Contraste RMS", "{:.3f}"),
             ("lap_var", "Nitidez (var. Laplaciano)", "{:.0f}"), ("spec_frac", "Fração especular", "{:.2%}")]
    cols = st.columns(len(names))
    for c, (k, label, f) in zip(cols, names):
        if k in q and q[k] is not None and not (isinstance(q[k], float) and np.isnan(q[k])):
            d = None if ref is None or k not in ref else q[k] - ref[k]
            c.metric(label, f.format(q[k]), None if d is None else f.format(d) if "%" not in f else f"{d:+.2%}",
                     delta_color="off" if k == "spec_frac" else "normal")


tabs = st.tabs(["① Pipeline passo a passo", "② Laboratório de técnicas", "③ Domínio da frequência",
                "④ Segmentação da glote", "⑤ Rede neural + Grad-CAM"])

# ---------------------------------------------------------------------------------------- ① pipeline
with tabs[0]:
    steps = [s for s in parse(pipe)]
    st.markdown("".join(f'<span class="chip">{s}</span>' for s in ["cinza", *steps]), unsafe_allow_html=True)
    st.caption("Cada coluna aplica um passo a mais. Embaixo: histograma (barras) e histograma acumulado (linha) — "
               "é nele que se vê o que o realce faz com a distribuição de intensidades.")
    cols = st.columns(len(steps) + 1)
    base = to_work_size(gray, WORK)
    with cols[0]:
        st.image(base, width="stretch", clamp=True)
        st.markdown('<p class="step">original</p><p class="sub">cinza, lado ≤ 512 px</p>', unsafe_allow_html=True)
        st.pyplot(hist_fig(base, "#8A94AD"), width="stretch")
    for i, s in enumerate(steps):
        o, _, _ = run(img_b, mask_b, ">".join(steps[:i + 1]), exps[exp].get("params"))
        with cols[i + 1]:
            st.image(view(o), width="stretch", clamp=True)
            st.markdown(f'<p class="step">+ {s}</p><p class="sub">{STEP_PT.get(s, "")}</p>', unsafe_allow_html=True)
            st.pyplot(hist_fig(o[..., 0], OR if i == len(steps) - 1 else CY), width="stretch")
    out, q, ctx = run(img_b, mask_b, pipe, exps[exp].get("params"))
    _, q0, _ = run(img_b, mask_b, "raw", None)
    st.markdown("#### Qualidade da imagem final (Δ em relação ao E00 `raw`)")
    metric_row(q, q0)
    doc = (TECHNIQUES.get(steps[0]).__doc__ or "") if steps and steps[0] in TECHNIQUES else ""
    if doc:
        with st.expander(f"Como funciona `{steps[0]}` (docstring da técnica)"):
            st.markdown(doc.replace("    ", "").replace("O que faz:", "**O que faz:**").replace("Por que importa:", "\n\n**Por que importa:**")
                        .replace("Parâmetros padrão:", "\n\n**Parâmetros padrão:**").replace("Referência:", "\n\n**Referência:**"))

# ---------------------------------------------------------------------------------------- ② laboratório
with tabs[1]:
    st.caption("As 14 variantes do estudo aplicadas ao mesmo frame. Mesmo frame, mesma rede: só o PDI muda.")
    rows, items = [], [(e, v) for e, v in exps.items() if not ("roi_gt" in v["pipeline"] and gt is None)]
    for chunk in range(0, len(items), 5):
        cols = st.columns(5)
        for c, (e, v) in zip(cols, items[chunk:chunk + 5]):
            o, qq, _ = run(img_b, mask_b, v["pipeline"], v.get("params"))
            c.image(view(o), width="stretch", clamp=True)
            c.markdown(f'<p class="step">{e} · {v["pipeline"]}</p>', unsafe_allow_html=True)
            rows.append({"exp": e, "pipeline": v["pipeline"], **{k: qq.get(k) for k in ("entropy", "rms_contrast", "lap_var", "spec_frac")}})
    df = pd.DataFrame(rows).set_index("exp")
    st.markdown("#### Métricas da imagem")
    st.dataframe(df.style.format({"entropy": "{:.2f}", "rms_contrast": "{:.3f}", "lap_var": "{:.0f}", "spec_frac": "{:.2%}"}, na_rep="—")
                 .background_gradient(cmap="GnBu", subset=["entropy", "rms_contrast", "lap_var"]), width="stretch")

# ---------------------------------------------------------------------------------------- ③ frequência
with tabs[2]:
    p = DEFAULTS["homo"]
    w = to_work_size(gray, WORK).astype(np.float64)
    F = np.fft.fftshift(np.fft.fft2(np.log1p(w)))
    H = homo_transfer(w.shape, **p)
    spec_in = np.log1p(np.abs(F)); spec_out = np.log1p(np.abs(F * H))
    homo_img = TECHNIQUES["homo"](w.astype(np.uint8), {}, **p)
    st.caption("A filtragem homomórfica trabalha no espectro de ln(f): f = iluminação × refletância vira uma soma; "
               "a iluminação (baixa frequência, centro) é atenuada por γL e os detalhes (altas frequências) realçados por γH.")
    c = st.columns(4)
    norm = lambda a: ((a - a.min()) / (np.ptp(a) or 1) * 255).astype(np.uint8)
    c[0].image(w.astype(np.uint8), caption="original", width="stretch")
    c[1].image(cv2.applyColorMap(norm(spec_in), cv2.COLORMAP_MAGMA)[..., ::-1], caption="|F(u,v)| de ln f (log)", width="stretch")
    c[2].image(cv2.applyColorMap(norm(H), cv2.COLORMAP_VIRIDIS)[..., ::-1], caption=f"H(u,v): γL={p['gl']}, γH={p['gh']}, D0={p['d0']}", width="stretch")
    c[3].image(homo_img, caption="resultado (exp → reescala)", width="stretch")
    fig, ax = plt.subplots(1, 2, figsize=(10, 2.4), dpi=110)
    r = np.arange(0, min(w.shape) // 2)
    ax[0].plot(r, (p["gh"] - p["gl"]) * (1 - np.exp(-p["c"] * r ** 2 / p["d0"] ** 2)) + p["gl"], color=CY, lw=2)
    ax[0].axhline(1, color=DIM, ls=":", lw=1); ax[0].axvline(p["d0"], color=OR, ls="--", lw=1)
    ax[0].set_title("Perfil radial de H (D0 em laranja)", fontsize=9); ax[0].set_xlabel("D (distância ao centro)", fontsize=8)
    rad = np.hypot(*np.meshgrid(np.arange(w.shape[1]) - w.shape[1] / 2, np.arange(w.shape[0]) - w.shape[0] / 2)).astype(int)
    e_in = np.bincount(rad.ravel(), spec_in.ravel())[:len(r)] / np.maximum(np.bincount(rad.ravel())[:len(r)], 1)
    e_out = np.bincount(rad.ravel(), spec_out.ravel())[:len(r)] / np.maximum(np.bincount(rad.ravel())[:len(r)], 1)
    ax[1].plot(r, e_in, color="#8A94AD", lw=1.5, label="antes"); ax[1].plot(r, e_out, color=OR, lw=1.5, label="depois de H")
    ax[1].set_title("Espectro médio por raio (log)", fontsize=9); ax[1].legend(frameon=False, fontsize=8); ax[1].set_xlabel("D", fontsize=8)
    for a in ax:
        a.tick_params(labelsize=7); a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")

# ---------------------------------------------------------------------------------------- ④ segmentação
with tabs[3]:
    b = best_config()
    pre = b.get("preprocess", "raw")
    w8 = to_work_size(gray, WORK)
    m = apply_best(w8, b["method"], b["params"], pre)
    pre_img = TECHNIQUES[pre](w8, {}, **DEFAULTS[pre]) if pre != "raw" else w8

    def overlay(base, mask, color, alpha=.45):
        v = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB).astype(np.float32)
        v[mask] = (1 - alpha) * v[mask] + alpha * np.array(color, np.float32)
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        v = v.astype(np.uint8); cv2.drawContours(v, cnts, -1, color, 2)
        return v

    st.caption(f"Melhor método da Fase 3: **{b['method']}** sobre **{pre}** (escolhido no treino de cada fold; Dice médio 0,614 no teste).")
    c = st.columns(4 if gt is not None else 3)
    c[0].image(w8, caption="original", width="stretch")
    c[1].image(pre_img, caption=f"pré-processamento: {pre}", width="stretch")
    c[2].image(overlay(w8, m, (14, 159, 181)), caption=f"segmentação clássica — área {m.mean():.1%}", width="stretch")
    if gt is not None:
        from laringopdi.pdi.segmentation import seg_metrics
        g2 = cv2.resize(gt.astype(np.uint8), w8.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
        v = cv2.cvtColor(w8, cv2.COLOR_GRAY2RGB)
        for mk, col in ((g2, (46, 158, 106)), (m, (224, 123, 46))):
            cnts, _ = cv2.findContours(mk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(v, cnts, -1, col, 2)
        c[3].image(v, caption="verde = especialista · laranja = clássico", width="stretch")
        sm = seg_metrics(m, g2)
        k = st.columns(3)
        k[0].metric("Dice", f"{sm['dice']:.3f}"); k[1].metric("IoU", f"{sm.get('iou', float('nan')):.3f}")
        k[2].metric("HD95 (px)", f"{sm.get('hd95', float('nan')):.1f}")

# ---------------------------------------------------------------------------------------- ⑤ rede neural
with tabs[4]:
    from laringopdi.evaluate import gradcam_maps, load_model

    vt = C.ROOT / "results" / "tables" / f"verdicts_{model_name}.csv"
    if exp == "E00":
        st.markdown('<div class="card"><b>E00 é a linha de base</b>: as outras técnicas são comparadas contra ele.</div>', unsafe_allow_html=True)
    elif vt.exists():
        v = pd.read_csv(vt)
        r = v[(v.exp == exp) & (v.task == task)]
        if not r.empty:
            r = r.iloc[0]
            st.markdown(f'<div class="card">No estudo (5 folds × {int(r.n_seeds)} semente(s)), <b>{exp} · {r.pipeline}</b> na Tarefa {task}: '
                        f'ΔF1 por vídeo = <b>{r.oof_delta_f1:+.3f}</b> [IC 95% {r.ci_low:+.3f}; {r.ci_high:+.3f}] → '
                        f'<span class="badge b-{r.verdict}">{r.verdict}</span></div>', unsafe_allow_html=True)
    model = load_model(cfg, exp, task, model_name, 0)
    if model is None:
        st.info(f"Sem pesos treinados para {exp} / Tarefa {task} / {model_name} (o Estágio 2 treinou E00, top-4, E14, E01, E07 e E08).")
    else:
        import torch

        from laringopdi.data import task_names
        from laringopdi.train import _prep
        with torch.no_grad():
            prob = torch.softmax(model(_prep(torch.from_numpy(out[None, ..., :3].copy()), torch.device("cpu"))), 1)[0].numpy()
        names = task_names(cfg, task)
        cam = gradcam_maps(model, out[None, ..., :3])[0]
        heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1]
        c1, c2, c3 = st.columns([1, 1, 1.1])
        c1.image(view(out), caption="entrada da rede (224×224)", width="stretch", clamp=True)
        c2.image((0.55 * out[..., :3] + 0.45 * heat).astype(np.uint8), caption="Grad-CAM: onde a rede olhou", width="stretch")
        with c3:
            st.markdown("#### Probabilidades")
            for n, pv in sorted(zip(names, prob), key=lambda t: -t[1]):
                st.markdown(f"**{n}** — {pv:.1%}"); st.progress(float(pv))
            st.caption("Modelo do fold 0, semente 0. Saída de apoio, não diagnóstico.")

st.divider()
st.caption(f"LaringoPDI · UNIFOR · Processamento Digital de Imagens · {AVISO}")

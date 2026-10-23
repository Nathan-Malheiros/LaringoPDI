"""LaringoPDI — protótipo: upload de um frame → etapas do PDI lado a lado → segmentação clássica
da glote → métricas da imagem → predição + Grad-CAM. Rodar: `streamlit run app/streamlit_app.py`."""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from laringopdi import config as C
from laringopdi.pdi import parse, run_pipeline
from laringopdi.pdi.quality import image_quality
from laringopdi.pdi.registry import to_work_size
from laringopdi.pdi.segmentation import apply_best, best_config

AVISO = "Ferramenta de apoio acadêmico. Não substitui avaliação médica."

st.set_page_config(page_title="LaringoPDI", layout="wide")
cfg = C.load()
st.title("LaringoPDI — PDI em videoendoscopia laríngea")
st.warning(AVISO, icon="⚠️")

exps = {e: v for e, v in cfg["experiments"].items() if v.get("pipeline")}
with st.sidebar:
    exp = st.selectbox("Pipeline de PDI", list(exps), format_func=lambda e: f"{e} · {exps[e]['pipeline']} — {exps[e]['desc']}")
    task = st.radio("Tarefa", ["A", "B"], format_func=lambda t: {"A": "A: saudável × distúrbio",
                                                                  "B": "B: saudável × lesão estrutural"}[t])
    model_name = st.selectbox("Modelo", ["resnet18", "simplecnn"])
    st.caption("Os pesos vêm de `results/runs/<exp>/<tarefa>/<modelo>/fold0_seed0/model.pt` (Estágio 2).")

up = st.file_uploader("Frame de laringoscopia (PNG/JPG)", type=["png", "jpg", "jpeg", "bmp"])
up_mask = st.file_uploader("Máscara da glote (opcional; necessária para `roi_gt`)", type=["png"])
if up is None:
    st.info("Envie um frame para começar. Os dados do BAGLS não são distribuídos com o app.")
    st.stop()


def _read(f, flag):
    return cv2.imdecode(np.frombuffer(f.getvalue(), np.uint8), flag)


gray = _read(up, cv2.IMREAD_GRAYSCALE)
gt = _read(up_mask, cv2.IMREAD_GRAYSCALE) > 127 if up_mask is not None else None
pipe = exps[exp]["pipeline"]
if "roi_gt" in pipe and gt is None:
    st.error("`roi_gt` exige a máscara de ground truth.")
    st.stop()

st.subheader("1. Etapas do PDI")
steps = parse(pipe)
cols = st.columns(len(steps) + 1)
cols[0].image(gray, caption="original (cinza)", clamp=True)
ctx = {}
for i, s in enumerate(steps):
    out, ctx = run_pipeline(gray, ">".join(steps[:i + 1]), exps[exp].get("params"), cfg["data"]["image_size"],
                            cfg["data"]["work_size"], gt_mask=gt)
    shown = out[..., :3] if out.shape[2] > 1 and not np.array_equal(out[..., 0], out[..., 1]) else out[..., 0]
    cols[i + 1].image(shown, caption=f"após `{s}`", clamp=True)

st.subheader("2. Segmentação clássica da glote")
b = best_config()
w = to_work_size(gray, cfg["data"]["work_size"])
m = apply_best(w, b["method"], b["params"], b.get("preprocess", "raw"))
vis = cv2.cvtColor(w, cv2.COLOR_GRAY2RGB)
vis[m] = (0.5 * vis[m] + [0, 127, 0]).astype(np.uint8)
c1, c2 = st.columns(2)
c1.image(vis, caption=f"{b['method']} sobre {b.get('preprocess', 'raw')} — área {m.mean():.1%} do frame")
if gt is not None:
    from laringopdi.pdi.segmentation import seg_metrics
    g2 = cv2.resize(gt.astype(np.uint8), w.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
    c2.metric("Dice vs máscara enviada", f"{seg_metrics(m, g2)['dice']:.3f}")

st.subheader("3. Métricas da imagem processada")
st.dataframe(pd.DataFrame([image_quality(out, ctx)]).round(4), hide_index=True)

st.subheader("4. Predição e Grad-CAM")
from laringopdi.evaluate import gradcam_maps, load_model  # noqa: E402

model = load_model(cfg, exp, task, model_name, 0)
if model is None:
    st.info(f"Sem pesos treinados para {exp}/{task}/{model_name} (o Estágio 2 roda no Colab).")
else:
    import torch

    from laringopdi.data import task_names
    from laringopdi.train import _prep
    with torch.no_grad():
        prob = torch.softmax(model(_prep(torch.from_numpy(out[None, ..., :3].copy()), torch.device("cpu"))), 1)[0].numpy()
    names = task_names(cfg, task)
    st.bar_chart(pd.DataFrame({"probabilidade": prob}, index=names))
    cam = gradcam_maps(model, out[None, ..., :3])[0]
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
    st.image((0.6 * out[..., :3] + 0.4 * heat).astype(np.uint8), caption=f"Grad-CAM — classe prevista: {names[int(prob.argmax())]}")
st.caption(AVISO)

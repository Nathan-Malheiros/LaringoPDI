import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
APP = str(ROOT / "app" / "streamlit_app.py")
AMOSTRAS = ROOT / "app" / "amostras" / "amostras.json"
AVISO = "Não substitui avaliação médica"


def _app():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(APP, default_timeout=180)       # a primeira carga inclui os modelos


def _has_frames():
    return AMOSTRAS.exists() or (ROOT / "data" / "raw" / "bagls").exists()


def _text(at) -> str:
    parts = [*at.markdown, *at.title, *at.subheader, *at.caption, *at.info, *at.success, *at.warning]
    return " ".join(str(m.value) for m in parts)


def _ok(at, onde=""):
    assert not at.exception, f"{onde}: {[e.value for e in at.exception]}"


def test_app_without_images_explains_instead_of_crashing():
    if _has_frames():
        pytest.skip("há imagens nesta máquina")
    at = _app().run()
    assert not at.exception and any("BAGLS" in e.value for e in at.error)


@pytest.mark.skipif(not _has_frames(), reason="sem imagens")
def test_explore_mode_renders_every_step_and_view():
    at = _app().run()
    _ok(at, "início")
    assert AVISO in _text(at) and at.session_state["modo"] == "roteiro"
    at.button_group(key="modo__nav").set_value("explorar").run()
    seen, text = set(), ""
    vistas = {1: ("filtro_vista", ["ruido", "fourier"]), 2: ("seg_vista", ["passos", "metodos", "mlp", "bordas"])}
    for i in range(5):
        at.button_group(key="passo__nav").set_value(i).run()
        _ok(at, f"passo {i}")
        chave, opcoes = vistas.get(i, (None, [None]))
        for v in opcoes:
            if chave:
                at.button_group(key=f"{chave}__nav").set_value(v).run()
                _ok(at, f"passo {i}, vista {v}")
            seen |= {m.label for m in at.metric}
            text += _text(at)
    assert {"PSNR", "SSIM", "Dice", "IoU", "Contraste RMS"} <= seen
    for needle in ("Realce", "limiar adaptativo", "18 características + MLP", "Hipóteses da Etapa 1", "sensibilidade em primeiro lugar", "formato do vídeo"):
        assert needle.lower() in text.lower(), needle
    for gone in ("Enviar arquivo", "SimpleCNN", "HOG"):
        assert gone not in text


@pytest.mark.skipif(not _has_frames(), reason="sem imagens")
def test_scene_navigation_sets_every_control_and_restores():
    at = _app().run()
    for _ in range(3):
        at.button(key="cena_prox").click().run()
    _ok(at, "cena 4")
    s = at.session_state
    assert s["cena"] == 3 and s["passo"] == 2 and s["seg_clahe"] is True and s["frame"].endswith("12188.png") and s["seg_vista"] == "metodos"
    assert "CLAHE" in _text(at)
    at.toggle(key="seg_clahe").set_value(False).run()
    at.button(key="cena_restaurar").click().run()
    _ok(at, "restaurar")
    assert at.session_state["seg_clahe"] is True
    at.button(key="ir_cena_5").click().run()                  # lesão estrutural, Tarefa B
    _ok(at, "cena 6")
    assert at.session_state["task"] == "B" and at.session_state["passo"] == 3
    at.button(key="cena_ant").click().run()
    assert at.session_state["cena"] == 4


@pytest.mark.skipif(not _has_frames(), reason="sem imagens")
def test_noise_controls_change_psnr():
    at = _app().run()
    at.button(key="ir_cena_1").click().run()
    before = [m.value for m in at.metric if m.label == "PSNR"]
    at.radio(key="ruido_tipo").set_value("gaussiano").run()
    _ok(at, "ruído gaussiano")
    assert before and [m.value for m in at.metric if m.label == "PSNR"] != before


@pytest.mark.skipif(not _has_frames(), reason="sem imagens")
def test_gallery_opens_a_frame_in_explore_mode():
    at = _app().run()
    at.button_group(key="modo__nav").set_value("galeria").run()
    _ok(at, "galeria")
    botoes = [b for b in at.button if (b.key or "").startswith(("gal_training", "gal_test"))]
    alvo = next(b for b in botoes if not b.disabled)
    alvo.click().run()
    _ok(at, "abrir frame")
    assert at.session_state["modo"] == "explorar" and at.session_state["frame"] == alvo.key[len("gal_"):]
    at.button_group(key="modo__nav").set_value("galeria").run()
    at.button_group(key="galeria_familia__nav").set_value("structural").run()
    _ok(at, "filtro")


@pytest.mark.skipif(not AMOSTRAS.exists(), reason="sem app/amostras")
def test_bundled_frames_are_fold0_test_only_and_cover_the_scenes():
    """As amostras embutidas são de vídeos que nenhum modelo do fold 0 viu no treino, e incluem os frames das cenas."""
    import sys

    from laringopdi import config as C
    sys.path.insert(0, str(ROOT / "app"))
    from roteiro import CENAS

    splits = ROOT / C.load()["paths"]["splits"] / "folds.json"
    if not splits.exists():
        pytest.skip("sem os splits")
    test = {str(v) for v in json.loads(splits.read_text(encoding="utf-8"))["folds"][0]["test"]}
    frames = json.loads(AMOSTRAS.read_text(encoding="utf-8"))["frames"]
    assert frames and all(f["video_id"] in test for f in frames)
    assert {c["frame"] for c in CENAS if "frame" in c} <= {f["path"] for f in frames}
    assert all((AMOSTRAS.parent / f["arquivo"]).exists() and (AMOSTRAS.parent / f["mascara"]).exists() for f in frames)

from pathlib import Path

import pytest

APP = str(Path(__file__).parents[1] / "app" / "streamlit_app.py")
AVISO = "Não substitui avaliação médica"


def _app():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(APP, default_timeout=180)       # a primeira carga inclui os modelos


def _has_bagls():
    return (Path(__file__).parents[1] / "data" / "raw" / "bagls").exists()


def _text(at) -> str:
    parts = [*at.markdown, *at.title, *at.subheader, *at.caption, *at.info, *at.success, *at.warning]
    return " ".join(str(m.value) for m in parts)


def test_app_without_dataset_explains_instead_of_crashing():
    if _has_bagls():
        pytest.skip("o BAGLS está nesta máquina")
    at = _app().run()
    assert not at.exception and any("BAGLS" in e.value for e in at.error)


@pytest.mark.skipif(not _has_bagls(), reason="sem o BAGLS local")
def test_app_all_steps_render():
    at = _app().run()
    assert not at.exception and AVISO in _text(at)
    seen_metrics, text = set(), ""
    for i in range(5):
        at.button_group(key="passo").set_value(i).run()
        assert not at.exception, f"passo {i}: {[e.value for e in at.exception]}"
        seen_metrics |= {m.label for m in at.metric}
        text += _text(at)
    assert {"PSNR", "SSIM", "Dice", "IoU"} <= seen_metrics
    for needle in ("Realce", "limiar adaptativo", "18 características + MLP", "Hipóteses da Etapa 1", "sensibilidade em primeiro lugar", "formato do vídeo"):
        assert needle in text
    for gone in ("Enviar arquivo", "SimpleCNN", "HOG"):
        assert gone not in text


@pytest.mark.skipif(not _has_bagls(), reason="sem o BAGLS local")
def test_scenes_set_every_control():
    at = _app().run()
    at.button(key="cena_3").click().run()                    # "O CLAHE salva a segmentação"
    assert not at.exception
    s = at.session_state
    assert s["passo"] == 2 and s["seg_clahe"] is True and s["frame"].endswith("12188.png") and s["cena"] == 3
    at.button(key="cena_5").click().run()                    # lesão estrutural, Tarefa B
    assert not at.exception and at.session_state["task"] == "B" and at.session_state["passo"] == 3


@pytest.mark.skipif(not _has_bagls(), reason="sem o BAGLS local")
def test_noise_controls_change_psnr():
    at = _app().run()
    at.button_group(key="passo").set_value(1).run()
    before = [m.value for m in at.metric if m.label == "PSNR"]
    at.radio(key="ruido_tipo").set_value("gaussiano").run()
    assert not at.exception
    assert [m.value for m in at.metric if m.label == "PSNR"] != before


@pytest.mark.skipif(not _has_bagls(), reason="sem o BAGLS local")
def test_frames_are_fold0_test_only():
    """Os frames do app são de vídeos que nenhum modelo do fold 0 viu no treino."""
    from laringopdi import config as C
    from laringopdi import data as D

    cfg = C.load()
    test = {str(v) for v in D.load_splits(cfg)[0]["test"]}
    fr = D.load_frames(cfg)
    vid = dict(zip(fr["path"], fr["video_id"].astype(str)))
    at = _app().run()
    assert len(at.selectbox(key="frame").options) == sum(v in test for v in vid.values())
    for i in range(7):                                       # as cenas com frame (a 8ª é só a aba de resultados)
        at.button(key=f"cena_{i}").click().run()
        assert vid[at.session_state["frame"]] in test, f"cena {i + 1}"

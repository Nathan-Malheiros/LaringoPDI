from pathlib import Path

import pytest


def test_app_loads_without_upload():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(Path(__file__).parents[1] / "app" / "streamlit_app.py"), default_timeout=60).run()
    assert not at.exception
    assert any("Não substitui avaliação médica" in w.value for w in at.warning)


def _app(default_timeout=180):
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(str(Path(__file__).parents[1] / "app" / "streamlit_app.py"), default_timeout=default_timeout)


def _has_bagls():
    return (Path(__file__).parents[1] / "data" / "raw" / "bagls" / "test").exists()


@pytest.mark.skipif(not _has_bagls(), reason="sem o BAGLS local")
def test_app_all_tabs_with_example_frame():
    at = _app().run()
    assert not at.exception
    labels = [t.label for t in at.tabs]
    assert labels[0].endswith("Técnicas de aula") and labels[-1].endswith("Resultados do estudo") and len(labels) == 7
    metrics = {m.label for m in at.metric}
    assert {"PSNR", "SSIM", "Dice"} <= metrics
    assert any(m.label.startswith("PSNR com σ") for m in at.metric)
    assert any("Hipóteses apresentadas" in m.value for m in at.markdown)
    assert any("MLP-18" in m.value for m in at.markdown)


@pytest.mark.skipif(not _has_bagls(), reason="sem o BAGLS local")
def test_app_noise_controls_change_psnr():
    at = _app().run()
    before = [m.value for m in at.metric if m.label == "PSNR"]
    at.radio(key="ruido_tipo").set_value("gaussiano").run()
    assert not at.exception
    assert [m.value for m in at.metric if m.label == "PSNR"] != before

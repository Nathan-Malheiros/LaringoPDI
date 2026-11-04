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


def test_app_upload_mode_works_without_dataset_frame():
    """O modo usado na entrega, sem o BAGLS: o uploader é trocado por um PNG sintético."""
    import cv2
    import numpy as np
    from unittest.mock import patch

    g = np.full((256, 320), 90, np.uint8)
    cv2.ellipse(g, (160, 128), (20, 70), 0, 0, 360, 15, -1)
    png = cv2.imencode(".png", g)[1].tobytes()

    class Upload:
        def getvalue(self):
            return png

    with patch("streamlit.file_uploader", lambda label, **kw: Upload() if label.startswith("Frame") else None):
        at = _app().run()
        src = at.sidebar.radio[0]
        if src.value != "Enviar arquivo":
            src.set_value("Enviar arquivo")
            at.run()
    assert not at.exception
    assert any("calibrada no BAGLS" in c.value for c in at.caption)
    assert any(m.label == "PSNR" for m in at.metric)
    assert any("Não substitui avaliação médica" in w.value for w in at.warning)

from pathlib import Path


def test_app_loads_without_upload():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(Path(__file__).parents[1] / "app" / "streamlit_app.py"), default_timeout=60).run()
    assert not at.exception
    assert any("Não substitui avaliação médica" in w.value for w in at.warning)

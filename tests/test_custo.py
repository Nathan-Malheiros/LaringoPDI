import time

import pandas as pd
import pytest

from laringopdi import config as C
from laringopdi.custo import COLUMNS, Medidor, latest_stages, medir


def test_medir_records_time_cpu_ram_and_items(tmp_cfg, monkeypatch):
    monkeypatch.setattr("laringopdi.custo.INTERVAL_S", 0.1)
    with medir(tmp_cfg, "etapa_a", "teste", n_itens=10):
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 0.3:
            sum(i * i for i in range(1000))
    d = C.path(tmp_cfg, "tables") / "custo"
    row = pd.read_csv(d / "etapas.csv").iloc[0]
    assert list(pd.read_csv(d / "etapas.csv").columns) == COLUMNS
    assert row["status"] == "ok" and row["duracao_s"] >= 0.3 and row["ms_por_item"] == pytest.approx(row["duracao_s"] * 100, rel=0.01)
    assert (d / "hardware.json").exists()


def test_failure_is_recorded_and_latest_ignores_it(tmp_cfg):
    with medir(tmp_cfg, "etapa_b"):
        pass
    with pytest.raises(ValueError):
        with medir(tmp_cfg, "etapa_b"):
            raise ValueError("falhou")
    all_rows = pd.read_csv(C.path(tmp_cfg, "tables") / "custo" / "etapas.csv")
    assert list(all_rows["status"]) == ["ok", "erro"]
    latest = latest_stages(tmp_cfg)
    assert len(latest) == 1 and latest.iloc[0]["status"] == "ok"


def test_sampler_measures_busy_process():
    m = Medidor(interval=0.1).start()
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 0.5:
        sum(i * i for i in range(1000))
    r = m.stop()
    assert r["amostras"] >= 2 and r["cpu_nucleos_medio"] > 0 and r["ram_processo_pico_gb"] > 0

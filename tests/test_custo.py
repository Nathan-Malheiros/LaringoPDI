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


def test_gpu_of_other_programs_is_not_charged_to_the_stage(monkeypatch):
    monkeypatch.setattr("laringopdi.custo._gpu_sample", lambda: (40.0, 900.0, 30.0))
    monkeypatch.setattr("laringopdi.custo._gpu_pids", lambda: {999999})
    r = Medidor(interval=0.05).start().stop()
    assert r["gpu_usada_pelo_processo"] == "não" and r["gpu_energia_liquida_wh"] is None
    assert r["gpu_sistema_util_pct_media"] == 40.0


def test_gpu_used_by_the_process_gets_net_energy(monkeypatch):
    import os
    samples = iter([(5.0, 900.0, 30.0)])
    monkeypatch.setattr("laringopdi.custo._gpu_sample", lambda: next(samples, (90.0, 3000.0, 130.0)))
    monkeypatch.setattr("laringopdi.custo._gpu_pids", lambda: {os.getpid()})
    m = Medidor(interval=0.05).start()
    time.sleep(0.2)
    r = m.stop()
    assert r["gpu_usada_pelo_processo"] == "sim" and r["gpu_potencia_inicio_w"] == 30.0
    assert r["gpu_energia_liquida_wh"] == pytest.approx(100 * r["duracao_s"] / 3600, abs=0.002)

import threading
import time

from laringopdi import custo, pausa


def test_no_flag_does_not_wait(tmp_cfg):
    assert pausa.esperar_se_pausado(tmp_cfg) == 0.0


def test_waits_until_flag_is_removed_and_discounts_from_stage(tmp_cfg):
    d = pausa._dir(tmp_cfg)
    d.mkdir(parents=True)
    flag = d / "pausa.flag"
    flag.touch()
    seen = {}

    def resume():
        time.sleep(0.3)
        seen["ativa"] = (d / "pausa.ativa").exists()
        flag.unlink()

    m = custo.Medidor(interval=0.05).start()
    threading.Thread(target=resume).start()
    waited = pausa.esperar_se_pausado(tmp_cfg, intervalo=0.02)
    r = m.stop()
    assert seen["ativa"] and waited >= 0.25 and not (d / "pausa.ativa").exists()
    assert r["pausado_s"] >= 0.25 and r["duracao_s"] < waited + 0.5 and r["duracao_s"] < 0.3

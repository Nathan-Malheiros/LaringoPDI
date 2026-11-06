"""Pausa segura entre jobs, acionada pelo painel LaringoMonitor.

O painel cria o arquivo `pausa.flag`; o treino termina o job atual e espera, com `pausa.ativa` indicando que parou.
Pausar entre jobs mantém corretos os tempos de cada treino, e o tempo de espera é descontado da etapa (laringopdi.custo).
"""
from __future__ import annotations

import time

from laringopdi import config as C

segundos_pausados = 0.0     # acumulado neste processo
em_espera = False           # o medidor não amostra durante a espera


def _dir(cfg: dict):
    return C.path(cfg, "cache").parent / "controle"


def esperar_se_pausado(cfg: dict, intervalo: float = 2.0) -> float:
    """Se houver pedido de pausa, espera até ele ser retirado. Devolve os segundos de espera."""
    global segundos_pausados, em_espera
    d = _dir(cfg)
    flag, ativa = d / "pausa.flag", d / "pausa.ativa"
    if not flag.exists():
        return 0.0
    t0 = time.perf_counter()
    em_espera = True
    ativa.touch()
    print("\nPAUSADO: treino atual concluído. Clique em Retomar no painel para continuar.", flush=True)
    try:
        while flag.exists():
            time.sleep(intervalo)
    finally:
        ativa.unlink(missing_ok=True)
        em_espera = False
    waited = time.perf_counter() - t0
    segundos_pausados += waited
    print("Retomado.", flush=True)
    return waited

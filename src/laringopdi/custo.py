"""Custo computacional de cada etapa: tempo, CPU, RAM, GPU, energia da GPU e disco.

Todo comando da CLI é medido, e os estudos medem também suas etapas internas. Cada medição vira uma
linha em results/tables/custo/etapas.csv; a máquina fica em results/tables/custo/hardware.json.
Medições repetidas acrescentam linhas: a mais recente de cada etapa é a que vale (`latest_stages`).
"""
from __future__ import annotations

import csv
import shutil
import subprocess
import threading
import time
from contextlib import contextmanager
from datetime import datetime

import psutil

from laringopdi import config as C

INTERVAL_S = 3.0
GPU_QUERY = "utilization.gpu,memory.used,power.draw"
COLUMNS = ["etapa", "descricao", "status", "inicio", "fim", "duracao_s", "n_itens", "ms_por_item", "cpu_nucleos_medio",
           "cpu_nucleos_pico", "cpu_sistema_pct_medio", "ram_processo_pico_gb", "ram_sistema_pico_gb", "gpu_util_pct_media",
           "gpu_util_pct_pico", "gpu_mem_pico_mb", "gpu_potencia_w_media", "gpu_energia_wh", "disco_delta_gb", "amostras", "git_sha"]


def _gpu_sample() -> tuple[float, float, float] | None:
    """Utilização (%), memória (MB) e potência (W) da GPU; None se não houver GPU."""
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"], capture_output=True,
                             text=True, timeout=5, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        vals = [float(v) for v in out.stdout.splitlines()[0].split(",")]
        return vals[0], vals[1], vals[2]
    except Exception:
        return None


class Medidor:
    """Amostra, em uma thread, o processo atual com seus filhos, o sistema e a GPU."""

    def __init__(self, interval: float = INTERVAL_S):
        self.interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._me = psutil.Process()
        self._procs: dict[int, psutil.Process] = {}
        self.cpu, self.cpu_sys, self.rss, self.ram, self.gpu = [], [], [], [], []

    def _tree_sample(self) -> tuple[float, float]:
        cores, rss = 0.0, 0.0
        try:
            tree = [self._me, *self._me.children(recursive=True)]
        except psutil.Error:
            tree = [self._me]
        for p in tree:
            q = self._procs.setdefault(p.pid, p)        # o mesmo objeto guarda a leitura anterior de CPU
            try:
                cores += q.cpu_percent(None) / 100
                rss += q.memory_info().rss
            except psutil.Error:
                pass
        return cores, rss / 2**30

    def _sample(self) -> None:
        cores, rss = self._tree_sample()
        self.cpu.append(cores)
        self.rss.append(rss)
        self.cpu_sys.append(psutil.cpu_percent(None))
        self.ram.append(psutil.virtual_memory().used / 2**30)
        g = _gpu_sample()
        if g:
            self.gpu.append(g)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self._sample()

    def start(self) -> "Medidor":
        self.t0, self.started = time.perf_counter(), datetime.now()
        self._disk0 = shutil.disk_usage(C.ROOT).used
        self._gpu0 = _gpu_sample()                      # memória da GPU antes da etapa
        self._tree_sample()
        psutil.cpu_percent(None)                        # primeira leitura só zera os contadores
        self._thread.start()
        return self

    def stop(self) -> dict:
        self._stop.set()
        self._thread.join(timeout=self.interval + 6)
        dur = time.perf_counter() - self.t0
        mean = lambda v: round(sum(v) / len(v), 2) if v else None
        peak = lambda v: round(max(v), 2) if v else None
        util, mem, power = zip(*self.gpu) if self.gpu else ((), (), ())
        return {"inicio": self.started.isoformat(timespec="seconds"), "fim": datetime.now().isoformat(timespec="seconds"),
                "duracao_s": round(dur, 2), "cpu_nucleos_medio": mean(self.cpu), "cpu_nucleos_pico": peak(self.cpu),
                "cpu_sistema_pct_medio": mean(self.cpu_sys), "ram_processo_pico_gb": peak(self.rss),
                "ram_sistema_pico_gb": peak(self.ram), "gpu_util_pct_media": mean(util), "gpu_util_pct_pico": peak(util),
                "gpu_mem_pico_mb": peak(mem), "gpu_potencia_w_media": mean(power),
                "gpu_energia_wh": round(sum(power) / len(power) * dur / 3600, 3) if power else None,
                "disco_delta_gb": round((shutil.disk_usage(C.ROOT).used - self._disk0) / 2**30, 3), "amostras": len(self.cpu)}


def _record(cfg: dict, row: dict) -> None:
    d = C.path(cfg, "tables") / "custo"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "etapas.csv"
    is_new = not p.exists()
    with open(p, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, COLUMNS)
        if is_new:
            w.writeheader()
        w.writerow({k: row.get(k) for k in COLUMNS})
    C.write_json(d / "hardware.json", C.hardware())


@contextmanager
def medir(cfg: dict, etapa: str, descricao: str = "", n_itens: int | None = None):
    """Mede o bloco e grava uma linha, inclusive se ele falhar. O objeto entregue aceita `.n_itens = N`."""
    class Info:
        pass
    info = Info()
    info.n_itens = n_itens
    m = Medidor().start()
    status = "ok"
    try:
        yield info
    except BaseException:
        status = "erro"
        raise
    finally:
        r = m.stop()
        n = info.n_itens
        _record(cfg, {**r, "etapa": etapa, "descricao": descricao, "status": status, "n_itens": n,
                      "ms_por_item": round(r["duracao_s"] * 1000 / n, 3) if n else None, "git_sha": C.git_sha()})


def latest_stages(cfg: dict):
    """A medição bem-sucedida mais recente de cada etapa."""
    import pandas as pd

    p = C.path(cfg, "tables") / "custo" / "etapas.csv"
    if not p.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(p)
    return df[df["status"] == "ok"].drop_duplicates("etapa", keep="last").reset_index(drop=True)

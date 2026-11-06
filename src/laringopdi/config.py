"""Carregamento de configuração e utilidades de reprodutibilidade."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
# Pesos pré-treinados ficam no HD, junto do projeto, e não no SSD (cache padrão do PyTorch fica em C:).
os.environ.setdefault("TORCH_HOME", str(ROOT / "data" / "processed" / "torch_home"))


def hardware() -> dict:
    """Máquina em que o job rodou: o custo computacional só é comparável com isto registrado."""
    import platform

    cpu, ram_gb = platform.processor(), None
    if sys.platform == "win32":
        import ctypes
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
            cpu = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()

        class _Mem(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        mem = _Mem()
        mem.dwLength = ctypes.sizeof(_Mem)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem)):
            ram_gb = round(mem.ullTotalPhys / 2**30, 1)
    out = {"cpu": cpu, "cpu_threads": os.cpu_count(), "ram_gb": ram_gb, "gpu": None, "gpu_mem_gb": None,
           "torch": None, "cuda": None, "os": platform.platform(), "python": platform.python_version()}
    try:
        import torch

        out["torch"], out["cuda"] = torch.__version__, torch.version.cuda
        if torch.cuda.is_available():
            out["gpu"] = torch.cuda.get_device_name(0)
            out["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    except ImportError:
        pass
    return out


def load(base: str = "configs/base.yaml", experiments: str = "configs/experiments.yaml",
         label_map: str = "configs/label_map.yaml") -> dict:
    """Carrega base.yaml, experiments.yaml e label_map.yaml (em `cfg["label_map"]`)."""
    cfg = yaml.safe_load((ROOT / base).read_text(encoding="utf-8"))
    cfg.update(yaml.safe_load((ROOT / experiments).read_text(encoding="utf-8")))
    cfg["label_map"] = yaml.safe_load((ROOT / label_map).read_text(encoding="utf-8"))
    return cfg


def path(cfg: dict, key: str) -> Path:
    """Caminho absoluto de `paths.<key>`; cria diretórios-pai quando necessário."""
    p = ROOT / cfg["paths"][key]
    (p if p.suffix == "" else p.parent).mkdir(parents=True, exist_ok=True)
    return p


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "nogit"


def stable_hash(obj) -> str:
    """Hash curto e determinístico de um objeto serializável (ex.: pipeline + parâmetros)."""
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:10]


def write_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

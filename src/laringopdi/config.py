"""Carregamento de configuração e utilidades de reprodutibilidade."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load(base: str = "configs/base.yaml", experiments: str = "configs/experiments.yaml") -> dict:
    """Carrega base.yaml e experiments.yaml num único dicionário."""
    cfg = yaml.safe_load((ROOT / base).read_text(encoding="utf-8"))
    cfg.update(yaml.safe_load((ROOT / experiments).read_text(encoding="utf-8")))
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

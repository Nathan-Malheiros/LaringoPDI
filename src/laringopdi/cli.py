"""Ponto de entrada único: `python -m laringopdi <comando> [opções]`.

Cada comando grava os resultados em disco e imprime só um resumo curto (≤ 10 linhas)."""
from __future__ import annotations

import argparse
import json
import sys

from laringopdi import config as C

COMMANDS = ["index", "eda", "splits", "gallery", "seg-study", "cache", "train", "gradcam", "summary", "denoise-study"]


def _exps(cfg: dict, arg: str | None) -> list[str]:
    if not arg or arg == "study":
        return cfg["study"]["experiments"]
    if arg == "all":
        return [e for e, v in cfg["experiments"].items() if v.get("pipeline")]
    return arg.split(",")


def _print(obj) -> None:
    txt = json.dumps(obj, ensure_ascii=False, default=str)
    print(txt if len(txt) < 1500 else txt[:1500] + " …")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="laringopdi", description=__doc__)
    ap.add_argument("command", choices=COMMANDS)
    ap.add_argument("--exp", help="E00, lista E01,E02, 'study' ou 'all'")
    ap.add_argument("--task", default="A", choices=["A", "B", "C"])
    ap.add_argument("--model", default="resnet18")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all", action="store_true", help="train: roda todos os jobs do estudo; os concluídos são pulados")
    ap.add_argument("--seed-only", type=int, help="train --all: só esta semente")
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--epochs", type=int, help="sobrescreve o nº máximo de épocas (só para teste/medição)")
    ap.add_argument("--force", action="store_true", help="sobrescreve saídas existentes (cuidado)")
    a = ap.parse_args(argv)
    cfg = C.load()

    if a.command == "index":
        from laringopdi.data import build_index
        s = build_index(cfg)
        _print({k: s[k] for k in ("n_frames", "n_videos", "videos_per_family", "task_videos", "videos_rgb",
                                  "subject_id_column", "small_classes_videos")})
    elif a.command == "splits":
        from laringopdi.data import build_groups, make_splits, sample_frames
        _print({"frames": sample_frames(cfg), "dedup": build_groups(cfg), "splits": make_splits(cfg, a.force)})
    elif a.command == "eda":
        from laringopdi.report import eda
        _print(eda(cfg))
    elif a.command == "gallery":
        from laringopdi.report import gallery, pdi_doc
        _print({**gallery(cfg), "doc": pdi_doc(cfg)})
    elif a.command == "seg-study":
        from laringopdi.evaluate import seg_study
        _print(seg_study(cfg, a.jobs, a.force))
    elif a.command == "cache":
        from laringopdi.data import build_cache, quality_summary
        out = [build_cache(cfg, e, a.jobs, a.force) for e in _exps(cfg, a.exp)]
        quality_summary(cfg)
        _print(out)
    else:
        mod = {"train": "train", "gradcam": "evaluate", "summary": "evaluate", "denoise-study": "etapa1"}[a.command]
        import importlib
        try:
            fn = getattr(importlib.import_module(f"laringopdi.{mod}"), "cmd_" + a.command.replace("-", "_"), None)
        except ModuleNotFoundError:
            fn = None
        if fn is None:
            sys.exit(f"comando `{a.command}` ainda não implementado.")
        _print(fn(cfg, a))


if __name__ == "__main__":
    main()

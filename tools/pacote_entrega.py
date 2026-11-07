"""Pacote de entrega do código: entrega/LaringoPDI_codigo.zip (com --with-weights: LaringoPDI_codigo_com_pesos.zip).

Contém o código, as configs, os testes, o app, o README, docs/*.md (menos os de uso interno), o relatório da Etapa 2,
results/tables/*.csv|json|tex e data/splits/*.json. Nunca contém data/raw, data/processed, results/runs (exceto os pesos pedidos), logs, .venv
nem figuras: várias têm frames do BAGLS, que é CC BY-NC-SA e não é redistribuído.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_WEIGHTS_MB = 200
CODE_PREFIXES = ("src/", "configs/", "tests/", "app/", "tools/", "notebooks/", ".streamlit/")
ROOT_FILES = ("README.md", "pyproject.toml", "requirements.txt", ".gitignore")
WEIGHTS = [f"results/runs/{e}/{t}/{m}/fold0_seed0/model.pt" for e in ("E00", "E03") for t in ("A", "B") for m in ("resnet18", "mlp18")] + \
          [f"results/runs/{e}/seg/mlp_seg/fold0_seed0/model.pt" for e in ("E00", "E03")]
FORBIDDEN = ("data/raw/", "data/processed/", "results/runs/", "logs/", ".venv/", "docs/figures/", "entrega/")
INTERNAL_DOCS = ("docs/PROMPT_INTERNO_ETAPA2.md",)
DOC_PDF = "docs/Documento_explicativo_LaringoPDI.pdf"


def tracked() -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True).stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


def select_files(files: list[str]) -> list[str]:
    """Filtra os arquivos versionados pelas regras do pacote."""
    keep = []
    for f in files:
        if f.startswith(FORBIDDEN) or f in INTERNAL_DOCS:
            continue
        p = Path(f)
        if (f.startswith(CODE_PREFIXES) or f in ROOT_FILES
                or (p.parent.as_posix() == "docs" and p.suffix == ".md") or f == DOC_PDF
                or (p.parent.as_posix() == "results/tables" and p.suffix in (".csv", ".json", ".tex"))
                or (p.parent.as_posix() == "data/splits" and p.suffix == ".json")):
            keep.append(f)
    return keep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--with-weights", action="store_true", help="inclui só os pesos sem PDI e com CLAHE usados pelo app (ResNet-18 e MLPs, fold 0)")
    a = ap.parse_args()
    files = select_files(tracked())
    if a.with_weights:
        missing = [w for w in WEIGHTS if not (ROOT / w).exists()]
        if missing:
            sys.exit(f"pesos ausentes: {missing}")
        sizes = {w: (ROOT / w).stat().st_size / 1e6 for w in WEIGHTS}
        total = sum(sizes.values())
        print(f"pesos: {total:.1f} MB")
        if total > MAX_WEIGHTS_MB:
            for w, mb in sizes.items():
                print(f"  {mb:8.1f} MB  {w}")
            sys.exit(f"o total passa de {MAX_WEIGHTS_MB} MB: pacote não gerado.")
        files += WEIGHTS
    out = ROOT / "entrega" / ("LaringoPDI_codigo_com_pesos.zip" if a.with_weights else "LaringoPDI_codigo.zip")
    out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(ROOT / f, f"LaringoPDI/{f}")
    print(f"{out.relative_to(ROOT)}: {len(files)} arquivos, {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
from pacote_entrega import select_files  # noqa: E402


def test_package_keeps_code_and_tables_only():
    files = ["src/laringopdi/cli.py", "README.md", "docs/LOG.md", "docs/relatorio_tecnico.pdf", "results/tables/seg_p1.csv", "data/splits/folds.json",
             "data/raw/bagls/test/1.png", "data/processed/cache/E00.npy", "results/runs/E00/A/resnet18/fold0_seed0/model.pt", "logs/a.log",
             ".venv/x.py", "docs/figures/gallery_grid.png", "docs/figures/relatorio/denoise.png", "results/tables/segmentation_frames.csv.gz",
             "entrega/a.zip", "results/summary.csv"]
    assert select_files(files) == files[:6]

"""Gera `app/amostras/`: as imagens que o app carrega consigo, para rodar sem o BAGLS no computador.

Um frame por vídeo do teste do fold 0 (o do meio da amostra fixa de `frames.json`) mais os frames das cenas,
cada um com a máscara do especialista. Só vídeos que nenhum modelo do app viu no treino.
Rodar: `.venv\\Scripts\\python.exe tools\\gerar_amostras.py` (precisa do BAGLS em data/raw/bagls)."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from laringopdi import config as C  # noqa: E402
from laringopdi import data as D  # noqa: E402
from roteiro import CENAS  # noqa: E402

ORDEM = {"healthy": 0, "functional": 1, "structural": 2}
FONTE = ("BAGLS: Gómez et al., BAGLS, a multihospital Benchmark for Automatic Glottis Segmentation, Scientific Data 7, 186 (2020). "
         "Licença CC BY-NC-SA 4.0: uso não comercial, com atribuição.")


def main() -> None:
    cfg = C.load()
    raw = C.ROOT / cfg["paths"]["raw"]
    fr = D.load_frames(cfg)
    test = {str(v) for v in D.load_splits(cfg)[0]["test"]}
    fr = fr[fr["video_id"].astype(str).isin(test)].copy()
    meio = set(fr.groupby("video_id")["path"].agg(lambda s: s.iloc[len(s) // 2]))
    cenas = {c["frame"] for c in CENAS if "frame" in c}
    fora = cenas - set(fr["path"])
    if fora:
        raise SystemExit(f"frames de cena fora do teste do fold 0: {sorted(fora)}")
    sel = fr[fr["path"].isin(meio | cenas)].copy()
    sel["o"] = sel["family"].map(ORDEM).fillna(9)
    sel = sel.sort_values(["o", "video_id", "path"])

    out = ROOT / "app" / "amostras"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    itens = []
    for r in sel.itertuples():
        img, seg = r.path.replace("/", "_"), r.seg_path.replace("/", "_")
        shutil.copy2(raw / r.path, out / img)
        shutil.copy2(raw / r.seg_path, out / seg)
        itens.append({"path": r.path, "video_id": str(r.video_id), "family": r.family, "arquivo": img, "mascara": seg})
    (out / "amostras.json").write_text(json.dumps({"fonte": FONTE, "regra": "teste do fold 0; um frame por vídeo + frames das cenas",
                                                   "frames": itens}, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "LEIAME.txt").write_text(f"Imagens de amostra do LaringoPDI.\n\n{FONTE}\n", encoding="utf-8")
    mb = sum(p.stat().st_size for p in out.iterdir()) / 2**20
    print(f"{len(itens)} frames ({sel['video_id'].nunique()} vídeos) em {out} · {mb:.1f} MB · "
          + ", ".join(f"{k}: {v}" for k, v in sel["family"].value_counts().items()))


if __name__ == "__main__":
    main()

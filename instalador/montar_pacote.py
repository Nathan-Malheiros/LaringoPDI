"""Monta `build/pacote/` (tudo que o app instalado precisa) e gera o instalador com o Inno Setup.

Etapas: 1 Python base · 2 PyTorch de CPU · 3 demais bibliotecas · 4 limpeza · 5 projeto, modelos e amostras ·
6 ícone e lançador · 7 teste do pacote isolado · 8 instalador. Cada etapa concluída grava `logs/instalador_etapas/etapaN.ok`
(o LaringoMonitor conta esses arquivos). A saída do uv vai para `logs/instalador_detalhe.log`; o resumo, para
`logs/instalador_resumo.json`. Rodar: `instalador\\montar_instalador.bat` (use `--limpo` para refazer o Python do zero)."""
from __future__ import annotations

import importlib.metadata as md
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
PACOTE = BUILD / "pacote"
PY = PACOTE / "python" / "python.exe"
LOGS = ROOT / "logs"
ETAPAS = LOGS / "instalador_etapas"
DETALHE = LOGS / "instalador_detalhe.log"
ISCC = Path(os.environ["LOCALAPPDATA"]) / "Programs" / "Inno Setup 6" / "ISCC.exe"
VERSAO = "1.0.0"
EXPS = ["E00", "E15", "E02", "E03", "E16", "E06", "E17", "E01", "E04"]       # as técnicas do app
LIBS = ["numpy", "opencv-python", "scikit-image", "scikit-learn", "scipy", "pandas", "matplotlib", "PyYAML", "streamlit",
        "grad-cam", "psutil", "pillow", "pyarrow", "tqdm"]
TOTAL = 8
resumo: dict = {"inicio": time.strftime("%Y-%m-%d %H:%M:%S"), "etapas": {}}


def etapa(n: int, nome: str):
    """Decorador simples: mostra a etapa, mede o tempo e grava o marcador de concluída."""
    def deco(f):
        def run(*a, **k):
            print(f"\n[{n}/{TOTAL}] {nome}", flush=True)
            t0 = time.time()
            out = f(*a, **k)
            resumo["etapas"][f"{n}. {nome}"] = round(time.time() - t0, 1)
            (ETAPAS / f"etapa{n}.ok").write_text("ok", encoding="utf-8")
            print(f"      concluída em {time.time() - t0:.0f} s", flush=True)
            return out
        return run
    return deco


def sh(cmd: list, **kw) -> None:
    env = {**os.environ, "UV_CACHE_DIR": str(BUILD / "uv_cache"), "UV_LINK_MODE": "copy"}
    env.pop("VIRTUAL_ENV", None)
    with DETALHE.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {' '.join(map(str, cmd))}\n")
        log.flush()
        r = subprocess.run(list(map(str, cmd)), stdout=log, stderr=subprocess.STDOUT, env=env, **kw)
    if r.returncode:
        raise SystemExit(f"falhou: {' '.join(map(str, cmd[:4]))}... (veja {DETALHE})")


def mb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 2**20 if p.exists() else 0.0


def copiar(pares: list[tuple[Path, Path]], desc: str) -> None:
    for src, dst in tqdm(pares, desc=f"      {desc}", unit="arq", ncols=90):
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def deps_ok() -> bool:
    sp = PACOTE / "python" / "Lib" / "site-packages"
    return PY.exists() and (sp / "torch").exists() and (sp / "streamlit").exists() and (sp / "pytorch_grad_cam").exists()


@etapa(1, "Python base (cópia do Python 3.11 do uv)")
def python_base() -> None:
    home = next(ln.split("=", 1)[1].strip() for ln in (ROOT / ".venv" / "pyvenv.cfg").read_text().splitlines() if ln.startswith("home"))
    shutil.copytree(home, PACOTE / "python", ignore=shutil.ignore_patterns("__pycache__", "test", "tests", "idlelib", "turtledemo"))
    (PACOTE / "python" / "Lib" / "EXTERNALLY-MANAGED").unlink(missing_ok=True)


@etapa(2, "PyTorch de CPU (download na primeira vez)")
def torch_cpu() -> None:
    v_t, v_tv = md.version("torch").split("+")[0], md.version("torchvision").split("+")[0]
    sh(["uv", "pip", "install", "--python", PY, "--index-url", "https://download.pytorch.org/whl/cpu", f"torch=={v_t}", f"torchvision=={v_tv}"])


@etapa(3, "Demais bibliotecas, nas mesmas versões do .venv")
def bibliotecas() -> None:
    sh(["uv", "pip", "install", "--python", PY, *[f"{p}=={md.version(p)}" for p in LIBS]])


@etapa(4, "Limpeza (cabeçalhos, .lib, .pdb, caches)")
def limpeza() -> None:
    sp = PACOTE / "python" / "Lib" / "site-packages"
    antes = mb(PACOTE / "python")
    for d in [sp / "torch" / "include", sp / "torch" / "share", *sp.rglob("__pycache__")]:
        shutil.rmtree(d, ignore_errors=True)
    for pat in ("*.lib", "*.pdb"):
        for f in sp.rglob(pat):
            f.unlink(missing_ok=True)
    print(f"      Python + bibliotecas: {antes:.0f} → {mb(PACOTE / 'python'):.0f} MB")


@etapa(5, "Projeto, modelos do fold 0, tabelas e amostras")
def projeto() -> None:
    for d in ("src", "configs", "app", "results", "tools", "data"):
        shutil.rmtree(PACOTE / d, ignore_errors=True)
    if not (ROOT / "app" / "amostras" / "amostras.json").exists():
        raise SystemExit("faltam as amostras: rode tools\\gerar_amostras.py")
    pares = [(f, PACOTE / f.relative_to(ROOT)) for f in (ROOT / "src" / "laringopdi").rglob("*.py")]
    pares += [(f, PACOTE / f.relative_to(ROOT)) for f in (ROOT / "configs").glob("*.yaml")]
    pares += [(ROOT / "app" / n, PACOTE / "app" / n) for n in ("streamlit_app.py", "roteiro.py")]
    pares += [(f, PACOTE / f.relative_to(ROOT)) for f in (ROOT / "app" / "amostras").iterdir()]
    pares += [(ROOT / "tools" / "documento_resultados.py", PACOTE / "tools" / "documento_resultados.py")]
    pares += [(f, PACOTE / f.relative_to(ROOT)) for f in (ROOT / "results" / "tables").rglob("*") if f.suffix in (".csv", ".json")]
    runs = ROOT / "results" / "runs"
    for e in EXPS:
        for rel in [f"{t}/{m}/fold0_seed0/model.pt" for t in ("A", "B") for m in ("resnet18", "mlp18")] + ["seg/mlp_seg/fold0_seed0/model.pt"]:
            if (runs / e / rel).exists():
                pares.append((runs / e / rel, PACOTE / "results" / "runs" / e / rel))
    copiar(pares, "copiando")
    resumo["modelos"] = sum(1 for _, d in pares if d.name == "model.pt")


@etapa(6, "Ícone, lançador e LEIAME")
def extras() -> None:
    from PIL import Image, ImageDraw
    n = 256
    im = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((8, 8, n - 8, n - 8), radius=56, fill=(14, 116, 144, 255))
    d.ellipse((58, 34, 198, 222), fill=(236, 246, 249, 255))                 # pregas vocais
    d.ellipse((104, 62, 152, 194), fill=(16, 42, 56, 255))                   # glote
    d.line((128, 44, 128, 212), fill=(14, 116, 144, 255), width=4)
    im.save(PACOTE / "LaringoPDI.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    shutil.copy2(ROOT / "instalador" / "lancador.py", PACOTE / "lancador.py")
    shutil.copy2(ROOT / "instalador" / "LEIAME.txt", PACOTE / "LEIAME.txt")


TESTE = r'''
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("app/streamlit_app.py", default_timeout=600).run()
assert not at.exception, at.exception
at.button_group(key="modo__nav").set_value("explorar").run()
for i in range(5):
    at.button_group(key="passo__nav").set_value(i).run()
    assert not at.exception, (i, [e.value for e in at.exception])
at.button_group(key="passo__nav").set_value(2).run()
at.button_group(key="seg_vista__nav").set_value("mlp").run()
assert not at.exception, at.exception
at.button_group(key="modo__nav").set_value("galeria").run()
assert not at.exception, at.exception
import sys, torch
print("ok", sys.executable, torch.__version__, "cuda" if torch.cuda.is_available() else "cpu")
'''


@etapa(7, "Teste do pacote isolado (sem .venv e sem o BAGLS)")
def teste() -> None:
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")}
    env.update(PYTHONPATH=str(PACOTE / "src"), PYTHONUTF8="1")
    r = subprocess.run([str(PY), "-c", TESTE], cwd=PACOTE, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    with DETALHE.open("a", encoding="utf-8") as log:
        log.write("\n$ teste do pacote\n" + r.stdout + r.stderr)
    if r.returncode or "ok" not in r.stdout:
        raise SystemExit(f"o app do pacote falhou (veja {DETALHE}):\n{r.stderr[-1500:]}")
    resumo["teste"] = r.stdout.strip().splitlines()[-1]
    for d in PACOTE.rglob("__pycache__"):                    # o teste gerou caches; não vão para o instalador
        shutil.rmtree(d, ignore_errors=True)


@etapa(8, "Instalador (Inno Setup; a compressão demora)")
def instalador() -> None:
    if not ISCC.exists():
        raise SystemExit(f"Inno Setup não encontrado em {ISCC}")
    r = subprocess.run([str(ISCC), f"/DVersao={VERSAO}", str(ROOT / "instalador" / "laringopdi.iss")])
    if r.returncode:
        raise SystemExit("o ISCC falhou (mensagem acima)")
    exe = BUILD / "instalador" / f"LaringoPDI-{VERSAO}-instalador.exe"
    resumo["instalador"] = str(exe)
    resumo["instalador_mb"] = round(exe.stat().st_size / 2**20)


def main() -> None:
    limpo = "--limpo" in sys.argv
    LOGS.mkdir(exist_ok=True)
    shutil.rmtree(ETAPAS, ignore_errors=True)
    ETAPAS.mkdir(parents=True)
    DETALHE.write_text("", encoding="utf-8")
    t0 = time.time()
    print(f"LaringoPDI · montando o instalador {VERSAO}\nPasta do pacote: {PACOTE}")
    if limpo:
        shutil.rmtree(PACOTE, ignore_errors=True)
    PACOTE.mkdir(parents=True, exist_ok=True)
    if deps_ok():
        print("\nPython e bibliotecas já estão no pacote: etapas 1 a 4 puladas (use --limpo para refazer).")
        for n in range(1, 5):
            (ETAPAS / f"etapa{n}.ok").write_text("pulada", encoding="utf-8")
    else:
        shutil.rmtree(PACOTE / "python", ignore_errors=True)
        python_base()
        torch_cpu()
        bibliotecas()
        limpeza()
    projeto()
    extras()
    teste()
    resumo["pacote_mb"] = round(mb(PACOTE))
    instalador()
    resumo["total_min"] = round((time.time() - t0) / 60, 1)
    (LOGS / "instalador_resumo.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nPronto em {resumo['total_min']} min. Instalador: {resumo['instalador']} ({resumo['instalador_mb']} MB)")


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        (LOGS / "instalador_resumo.json").write_text(json.dumps({**resumo, "erro": str(e)}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nERRO: {e}")
        sys.exit(1)

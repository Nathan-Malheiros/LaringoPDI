"""Abre o LaringoPDI instalado: sobe o app desta pasta e abre o navegador quando ele responde.

Fechar esta janela encerra o app. Se o app já estiver aberto, só abre o navegador de novo."""
import os
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PORTA = 8517                                                 # diferente da 8501 do desenvolvimento, para não colidir
URL = f"http://localhost:{PORTA}"


def no_ar() -> bool:
    try:
        urllib.request.urlopen(URL + "/_stcore/health", timeout=1)
        return True
    except Exception:
        return False


def main() -> None:
    os.system("title LaringoPDI")
    if no_ar():
        webbrowser.open(URL)
        return
    print("LaringoPDI\n")
    print("Abrindo o app. A primeira carga leva alguns segundos; o navegador abre sozinho quando estiver pronto.")
    print("Mantenha esta janela aberta enquanto usar o app. Para encerrar, feche esta janela.\n")
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")}
    env.update(PYTHONPATH=str(ROOT / "src"), PYTHONUTF8="1", STREAMLIT_BROWSER_GATHER_USAGE_STATS="false")
    p = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "streamlit_app.py"), "--server.port", str(PORTA),
                          "--server.headless", "true", "--global.developmentMode", "false", "--server.fileWatcherType", "none"],
                         cwd=ROOT, env=env)
    for _ in range(300):
        if p.poll() is not None:
            break
        if no_ar():
            webbrowser.open(URL)
            print(f"Pronto: {URL}")
            break
        time.sleep(1)
    try:
        p.wait()
    except KeyboardInterrupt:
        p.terminate()
    if p.returncode:
        input("\nO app parou com erro (mensagem acima). Tecle Enter para fechar.")


if __name__ == "__main__":
    main()

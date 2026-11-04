"""Capturas das abas do protótipo em docs/figures/app/ (fora do Git: têm frames do BAGLS).

Sobe o Streamlit sem interface e usa o Chrome headless em cada aba (`?aba=N&frame=ID`).
Uso: python tools/capturas_app.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures" / "app"
PORT = 8599
N_TABS = 5
MIN_BYTES = 50_000
CHROME = [r"C:\Program Files\Google\Chrome\Application\chrome.exe", r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    browser = next(c for c in CHROME if Path(c).exists())
    server = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "streamlit_app.py"), "--server.headless", "true",
                               "--server.port", str(PORT), "--browser.gatherUsageStats", "false"], cwd=ROOT,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(f"http://localhost:{PORT}/_stcore/health", timeout=2)
                break
            except OSError:
                time.sleep(1)
        for i in range(N_TABS):
            shot = OUT / f"aba{i}.png"
            for _ in range(4):        # enquanto o servidor aquece, a captura sai quase em branco
                shot.unlink(missing_ok=True)
                subprocess.run([browser, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--window-size=1500,1500", f"--screenshot={shot}",
                                "--virtual-time-budget=60000", f"http://localhost:{PORT}/?aba={i}&frame=1881"], check=False, timeout=180,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if shot.exists() and shot.stat().st_size > MIN_BYTES:
                    break
            print(shot.name, shot.stat().st_size if shot.exists() else "falhou")
    finally:
        server.terminate()


if __name__ == "__main__":
    main()

@echo off
REM LaringoPDI - abre o app (Streamlit) no navegador quando ele estiver pronto.
title LaringoPDI - app
cd /d "%~dp0.."
echo Abrindo o LaringoPDI. A primeira carga demora; o navegador abre sozinho quando o app estiver pronto.
echo Mantenha esta janela aberta enquanto usar o app. Para fechar o app, feche esta janela.
start "" /min powershell -NoProfile -WindowStyle Hidden -Command "for ($i = 0; $i -lt 120; $i++) { try { Invoke-WebRequest http://localhost:8501/_stcore/health -UseBasicParsing -TimeoutSec 2 | Out-Null; Start-Process http://localhost:8501; break } catch { Start-Sleep 1 } }"
.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py --server.port 8501 --server.headless true --browser.gatherUsageStats false
if errorlevel 1 (echo. & echo O app parou com erro. Veja a mensagem acima. & pause >nul)

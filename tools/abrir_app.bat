@echo off
REM LaringoPDI - abre o app (Streamlit) no navegador.
cd /d D:\DocumentosHD\Projetos\ProjetoPDIZIP
start "" http://localhost:8501
.venv\Scripts\streamlit run app\streamlit_app.py --server.port 8501 --server.headless true --browser.gatherUsageStats false

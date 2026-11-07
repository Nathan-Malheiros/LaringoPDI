@echo off
rem Controle do formato do video (E00 x E18). O que ja terminou e pulado; Pausar no painel funciona.
title LaringoPDI - Teste sem bordas
cd /d "%~dp0.."
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
.venv\Scripts\python.exe -u tools\teste_sem_bordas.py --jobs 10 2> logs\sem_bordas.err
if errorlevel 1 (echo. & echo ERRO: veja logs\sem_bordas.err & pause >nul & exit /b 1)
echo.
echo TESTE CONCLUIDO.
exit /b 0

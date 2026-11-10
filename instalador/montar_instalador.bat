@echo off
REM LaringoPDI - monta build\pacote e gera build\instalador\LaringoPDI-<versao>-instalador.exe
REM Uso: montar_instalador.bat [--limpo]
title LaringoPDI - montando o instalador
cd /d "%~dp0.."
.venv\Scripts\python.exe instalador\montar_pacote.py %*
echo.
pause

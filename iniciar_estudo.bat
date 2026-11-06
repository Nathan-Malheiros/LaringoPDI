@echo off
rem Roda ou retoma o estudo inteiro. O que ja terminou e pulado. Tambem e chamado pelo botao Retomar do LaringoMonitor.
title LaringoPDI - Estudo
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
echo [1/5] Caches das tecnicas
.venv\Scripts\python.exe -u -m laringopdi cache --exp study --jobs 10 2> logs\cache.err || goto erro
echo [2/5] MLP de classificacao
.venv\Scripts\python.exe -u -m laringopdi mlp-class --jobs 10 2> logs\mlp-class.err || goto erro
echo [3/5] MLP de segmentacao, GPU
.venv\Scripts\python.exe -u -m laringopdi mlp-seg 2> logs\mlp-seg.err || goto erro
echo [4/5] CNN, GPU
.venv\Scripts\python.exe -u -m laringopdi train --all 2> logs\train.err || goto erro
echo [5/5] Grad-CAM
.venv\Scripts\python.exe -u -m laringopdi gradcam 2> logs\gradcam.err || goto erro
echo.
echo ESTUDO CONCLUIDO. Pode fechar esta janela.
pause >nul
exit /b 0
:erro
echo.
echo ERRO: veja o arquivo mais recente em logs\*.err. Os treinos prontos estao salvos.
pause >nul
exit /b 1

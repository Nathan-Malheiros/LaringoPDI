@echo off
REM LaringoPDI - pausa ou retoma o treino. A logica fica em tools\pausar_retomar.ps1.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0pausar_retomar.ps1"

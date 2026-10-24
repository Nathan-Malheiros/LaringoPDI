# LaringoPDI - pausa ou retoma o treino (Estagio 2 + fechamento). Jobs concluidos ficam salvos.
$root = 'D:\DocumentosHD\Projetos\ProjetoPDIZIP'
Set-Location $root
$pat = 'stage2_chain|final_chain|laringopdi train'
$procs = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match $pat })
$feitos = @(Get-ChildItem results\runs -Recurse -Filter metrics.json | Where-Object { $_.FullName -match '\\(resnet18|simplecnn)\\' }).Count

Write-Host ""
Write-Host " LaringoPDI - jobs de fine-tuning concluidos: $feitos de 420"
Write-Host ""
if ($procs.Count -gt 0) {
    Write-Host " Estado: RODANDO" -ForegroundColor Green
    $r = Read-Host " Pausar o treino agora? Perde no maximo o job em andamento (~2 min). [S/N]"
    if ($r -match '^[sS]') {
        $procs | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
        Write-Host ""
        Write-Host " Treino PAUSADO. Rode este arquivo de novo para retomar." -ForegroundColor Yellow
    }
} else {
    Write-Host " Estado: PAUSADO" -ForegroundColor Yellow
    $r = Read-Host " Retomar o treino agora? [S/N]"
    if ($r -match '^[sS]') {
        Start-Process cmd.exe -ArgumentList '/c', '.venv\Scripts\python tools\final_chain.py >> logs\final_chain.log 2>&1' -WorkingDirectory $root -WindowStyle Hidden
        Write-Host ""
        Write-Host " Treino RETOMADO em segundo plano. Pode fechar esta janela." -ForegroundColor Green
    }
}
Write-Host ""
Read-Host " Enter para fechar"

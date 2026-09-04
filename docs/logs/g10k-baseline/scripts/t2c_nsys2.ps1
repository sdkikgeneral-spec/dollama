# G-10k T2c nsys capture #2 : identical env to capture #1, plus -w true (--show-output)
# so the target's own stdout ([S4] config ... fast=1 / per-image sec) lands in the log.
# Capture #1 lost it (nsys swallowed the child's stdout when redirected).
$ErrorActionPreference = 'Continue'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$repo = 'E:\Develop\Projects\dollama'
$OUT  = 'E:\Develop\logs\g10k-t2c'
$NSD  = Join-Path $OUT 'nsys'
$nsys = 'C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.1.3\target-windows-x64\nsys.exe'
$exe  = Join-Path $repo 'build\src\prof_arena_e2e.exe'
$rep  = Join-Path $NSD  'g10k_t2c_nsys2'
$log  = Join-Path $OUT  't2c_nsys2_stats.log'

$env:PROF_IMAGES = '2'
$env:PROF_STEPS  = '20'
$env:PROF_G      = '7.5'
$env:PROF_FAST   = '1'
Remove-Item Env:DOLLAMA_PROFILE       -ErrorAction SilentlyContinue
Remove-Item Env:DOLLAMA_POOL          -ErrorAction SilentlyContinue
Remove-Item Env:DOLLAMA_ARENA_RELEASE -ErrorAction SilentlyContinue
Remove-Item Env:PROF_SAMPLE_MS        -ErrorAction SilentlyContinue

Write-Output "=== T2c nsys capture #2 meta ==="
Write-Output ("start_local = " + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss K'))
Write-Output ("exe_sha256 = " + (Get-FileHash $exe -Algorithm SHA256).Hash)
Get-ChildItem Env: | Where-Object { $_.Name -like 'PROF_*' -or $_.Name -like 'DOLLAMA_*' } |
    Sort-Object Name | ForEach-Object { Write-Output ("  " + $_.Name + "=" + $_.Value) }
Write-Output "--- nvidia-smi BEFORE ---"
nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,power.draw,memory.used --format=csv

Set-Location $repo
$sw = [System.Diagnostics.Stopwatch]::StartNew()
$cmd = "`"$nsys`" profile --stats=true --show-output=true --force-overwrite=true -o `"$rep`" `"$exe`" > `"$log`" 2>&1"
& cmd /c $cmd
$rc = $LASTEXITCODE
$sw.Stop()
Write-Output ("exit_code = " + $rc)
Write-Output ("elapsed_sec = " + [math]::Round($sw.Elapsed.TotalSeconds,2))
Write-Output "--- nvidia-smi AFTER ---"
nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,power.draw,memory.used --format=csv
Write-Output ""
Write-Output "--- target stdout captured? ---"
$s4 = Select-String -Path $log -Pattern '\[S4\]'
if ($null -eq $s4) { Write-Output "  NO [S4] lines in log" }
else { $s4 | ForEach-Object { Write-Output ("  " + $_.Line) } }
Write-Output ""
Write-Output ("stats_log_sha256 = " + (Get-FileHash $log -Algorithm SHA256).Hash)
Get-ChildItem $NSD | Select-Object Name,Length | Format-Table -AutoSize
Write-Output "=== end ==="

# G-10k T2c follow-up : Nsight Systems capture of prof_arena_e2e (B=2 / fast path)
# src is UNCHANGED. DOLLAMA_PROFILE is NOT set (no sync perturbation).
$ErrorActionPreference = 'Continue'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$repo = 'E:\Develop\Projects\dollama'
$OUT  = 'E:\Develop\logs\g10k-t2c'
$NSD  = Join-Path $OUT 'nsys'
New-Item -ItemType Directory -Force -Path $NSD | Out-Null

$nsys = 'C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.1.3\target-windows-x64\nsys.exe'
$exe  = Join-Path $repo 'build\src\prof_arena_e2e.exe'
$rep  = Join-Path $NSD  'g10k_t2c_nsys'
$log  = Join-Path $OUT  't2c_nsys_stats.log'
$smi  = Join-Path $OUT  't2c_nsys_gpu_sample.csv'

$env:PROF_IMAGES = '2'
$env:PROF_STEPS  = '20'
$env:PROF_G      = '7.5'
$env:PROF_FAST   = '1'
Remove-Item Env:DOLLAMA_PROFILE   -ErrorAction SilentlyContinue
Remove-Item Env:DOLLAMA_POOL      -ErrorAction SilentlyContinue
Remove-Item Env:DOLLAMA_ARENA_RELEASE -ErrorAction SilentlyContinue
Remove-Item Env:PROF_SAMPLE_MS    -ErrorAction SilentlyContinue

Write-Output "=== T2c nsys meta ==="
Write-Output ("start_local = " + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss K'))
Write-Output ("nsys = " + $nsys)
Write-Output ("nsys_exists = " + (Test-Path $nsys))
Write-Output ("exe = " + $exe)
Write-Output ("exe_sha256 = " + (Get-FileHash $exe -Algorithm SHA256).Hash)
& $nsys --version
Write-Output ""
Write-Output "--- env actually set (PROF_* / DOLLAMA_*) ---"
Get-ChildItem Env: | Where-Object { $_.Name -like 'PROF_*' -or $_.Name -like 'DOLLAMA_*' } |
    Sort-Object Name | ForEach-Object { Write-Output ("  {0}={1}" -f $_.Name, $_.Value) }
Write-Output "--- deliberately unset ---"
foreach ($n in @('DOLLAMA_PROFILE','DOLLAMA_POOL','DOLLAMA_ARENA_RELEASE','PROF_SAMPLE_MS'))
{
    $v = [System.Environment]::GetEnvironmentVariable($n)
    Write-Output ("  {0} = {1}" -f $n, $(if ($null -eq $v) { '<unset>' } else { "SET:'" + $v + "' (UNEXPECTED)" }))
}
Write-Output ""
Write-Output "--- nvidia-smi BEFORE ---"
nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,power.draw,memory.used --format=csv

if (Test-Path $smi) { Remove-Item $smi -Force }
$smiProc = Start-Process -FilePath 'nvidia-smi' -PassThru -WindowStyle Hidden `
    -ArgumentList @('--query-gpu=timestamp,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu,memory.used',
                    '--format=csv','-lms','2000','-f',$smi)

Set-Location $repo
Write-Output ""
Write-Output "--- launching nsys profile (stdout+stderr merged) ---"
$sw = [System.Diagnostics.Stopwatch]::StartNew()
$cmd = "`"$nsys`" profile --stats=true --force-overwrite=true -o `"$rep`" `"$exe`" > `"$log`" 2>&1"
& cmd /c $cmd
$rc = $LASTEXITCODE
$sw.Stop()

Start-Sleep -Milliseconds 500
try { Stop-Process -Id $smiProc.Id -Force -ErrorAction Stop } catch { Write-Output ("smi stop: " + $_.Exception.Message) }

Write-Output ("exit_code = " + $rc)
Write-Output ("elapsed_sec = " + [math]::Round($sw.Elapsed.TotalSeconds,2))
Write-Output ("end_local = " + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss K'))
Write-Output ""
Write-Output "--- nvidia-smi AFTER ---"
nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,power.draw,memory.used --format=csv
Write-Output ""
Write-Output "--- artifacts ---"
Get-ChildItem $NSD | Select-Object Name,Length,LastWriteTime | Format-Table -AutoSize
if (Test-Path $log)
{
    Write-Output ("stats_log_bytes = " + (Get-Item $log).Length)
    Write-Output ("stats_log_sha256 = " + (Get-FileHash $log -Algorithm SHA256).Hash)
}
Write-Output "=== end ==="

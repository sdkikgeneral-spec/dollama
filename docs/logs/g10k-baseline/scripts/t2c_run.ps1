# G-10k T2c : DB2_BENCH (profile ON) single-process run on the conv2d-UNCHANGED tree.
# Collection only. No judgement, no pass/fail.
$ErrorActionPreference = 'Continue'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$OUT   = 'E:\Develop\logs\g10k-t2c'
$EXED  = Join-Path $OUT 'exe'
$exe   = Join-Path $EXED 'test_diffusion_batch2.exe'
$log   = Join-Path $OUT 't2c_db2bench.log'
$smi   = Join-Path $OUT 't2c_gpu_sample.csv'

# --- env pin (must match T7 exactly) ---
$env:DB2_BENCH       = '1'
$env:DB2_BENCH_STEPS = '20'
$env:DB2_BENCH_ITERS = '1'
$env:DOLLAMA_PROFILE = '1'
# DB2_BENCH_G is deliberately NOT set -> harness default 7.5
# (src/tests/test_diffusion_batch2.cu:559 `float bg = 7.5f;`, :562 only overridden if env present)
Remove-Item Env:DB2_BENCH_G       -ErrorAction SilentlyContinue
# seed: harness default (hard-coded 1234) -> DB2_* seed override does not exist
Remove-Item Env:DB2_STEPS         -ErrorAction SilentlyContinue
Remove-Item Env:DB2_UNCOND_ZERO   -ErrorAction SilentlyContinue
Remove-Item Env:DOLLAMA_POOL      -ErrorAction SilentlyContinue
Remove-Item Env:DOLLAMA_EPILOGUE  -ErrorAction SilentlyContinue
Remove-Item Env:DOLLAMA_FAST      -ErrorAction SilentlyContinue
Remove-Item Env:DOLLAMA_CONV_BATCH -ErrorAction SilentlyContinue
Remove-Item Env:DOLLAMA_GEMM      -ErrorAction SilentlyContinue
Remove-Item Env:DOLLAMA_ARENA_RELEASE -ErrorAction SilentlyContinue
Remove-Item Env:DOLLAMA_G8K_DUMP  -ErrorAction SilentlyContinue

Write-Output "=== T2c run meta ==="
Write-Output ("start_local = " + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss K'))
Write-Output ("exe = " + $exe)
Write-Output ("exe_sha256 = " + (Get-FileHash $exe -Algorithm SHA256).Hash)
Write-Output ("cwd = " + (Get-Location).Path)
Write-Output ""
Write-Output "--- env actually set in this process (DB2_* / DOLLAMA_*) ---"
Get-ChildItem Env: | Where-Object { $_.Name -like 'DB2_*' -or $_.Name -like 'DOLLAMA_*' } |
    Sort-Object Name | ForEach-Object { Write-Output ("  {0}={1}" -f $_.Name, $_.Value) }
Write-Output "--- env deliberately NOT set (verified absent) ---"
foreach ($n in @('DB2_BENCH_G','DB2_STEPS','DB2_UNCOND_ZERO','DOLLAMA_POOL','DOLLAMA_EPILOGUE','DOLLAMA_FAST','DOLLAMA_CONV_BATCH','DOLLAMA_GEMM','DOLLAMA_ARENA_RELEASE','DOLLAMA_G8K_DUMP'))
{
    $v = [System.Environment]::GetEnvironmentVariable($n)
    Write-Output ("  {0} = {1}" -f $n, $(if ($null -eq $v) { '<unset>' } else { "SET:'" + $v + "' (UNEXPECTED)" }))
}
Write-Output ""
Write-Output "--- nvidia-smi BEFORE ---"
nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,clocks.mem,power.draw,power.limit,utilization.gpu,memory.used --format=csv

# --- background GPU sampling (query only; negligible GPU load) ---
if (Test-Path $smi) { Remove-Item $smi -Force }
$smiProc = Start-Process -FilePath 'nvidia-smi' -PassThru -WindowStyle Hidden `
    -ArgumentList @('--query-gpu=timestamp,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu,memory.used',
                    '--format=csv','-lms','2000','-f',$smi)

Write-Output ""
Write-Output "--- launching (stdout+stderr merged via cmd 2>&1) ---"
$sw = [System.Diagnostics.Stopwatch]::StartNew()
# cmd /c gives a faithful `> file 2>&1` merge of the native process' two streams.
& cmd /c "`"$exe`" > `"$log`" 2>&1"
$rc = $LASTEXITCODE
$sw.Stop()

Start-Sleep -Milliseconds 500
try { Stop-Process -Id $smiProc.Id -Force -ErrorAction Stop } catch { Write-Output ("smi stop: " + $_.Exception.Message) }

Write-Output ("exit_code = " + $rc)
Write-Output ("elapsed_sec = " + [math]::Round($sw.Elapsed.TotalSeconds,2))
Write-Output ("end_local = " + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss K'))
Write-Output ""
Write-Output "--- nvidia-smi AFTER (immediately post-run) ---"
nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,clocks.mem,power.draw,power.limit,utilization.gpu,memory.used --format=csv
Write-Output ""
Write-Output ("log_bytes = " + (Get-Item $log).Length)
Write-Output ("log_sha256 = " + (Get-FileHash $log -Algorithm SHA256).Hash)
Write-Output ("smi_csv_lines = " + (Get-Content $smi | Measure-Object -Line).Lines)
Write-Output ""
Write-Output "--- grep: reserve shortage ---"
$rs = Select-String -Path $log -Pattern 'reserve shortage' -SimpleMatch
if ($null -eq $rs) { Write-Output "  0 hits" } else { $rs | ForEach-Object { Write-Output ("  " + $_.Line) } }
Write-Output "=== end ==="

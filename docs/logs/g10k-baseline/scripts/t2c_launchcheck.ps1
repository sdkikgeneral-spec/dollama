# Launchability check of the SAVED (persistent-path) copies. Run AFTER all measurements,
# so it cannot contaminate any timing. PROF_IMAGES=0 -> no image is generated.
$ErrorActionPreference = 'Continue'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$EXED = 'E:\Develop\logs\g10k-t2c\exe'
$log  = 'E:\Develop\logs\g10k-t2c\t2c_launchcheck.log'
$env:PROF_IMAGES = '0'
$env:PROF_STEPS  = '1'
$env:PROF_FAST   = '1'
Write-Output "=== launch check of saved exe copies ==="
Write-Output ("start_local = " + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss K'))
Write-Output ("target = " + (Join-Path $EXED 'prof_arena_e2e.exe'))
Write-Output ("sha256 = " + (Get-FileHash (Join-Path $EXED 'prof_arena_e2e.exe') -Algorithm SHA256).Hash)
Write-Output "env: PROF_IMAGES=0 PROF_STEPS=1 PROF_FAST=1 (no image generated)"
$sw = [System.Diagnostics.Stopwatch]::StartNew()
& cmd /c "`"$EXED\prof_arena_e2e.exe`" > `"$log`" 2>&1"
$rc = $LASTEXITCODE
$sw.Stop()
Write-Output ("exit_code = " + $rc + "   elapsed_sec = " + [math]::Round($sw.Elapsed.TotalSeconds,2))
Write-Output "--- output ---"
Get-Content $log
Write-Output ""
Write-Output "NOTE: test_diffusion_batch2.exe launchability from the saved path is already proven"
Write-Output "      by the T2c measurement run itself (t2c_run_meta.log: exe = ...\exe\test_diffusion_batch2.exe, exit 0)."
Write-Output "=== end ==="

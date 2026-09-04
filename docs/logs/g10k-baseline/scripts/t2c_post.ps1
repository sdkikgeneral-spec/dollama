$ErrorActionPreference = 'Continue'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OUT = 'E:\Develop\logs\g10k-t2c'
$NSD = Join-Path $OUT 'nsys'
$S   = Join-Path $OUT 'scripts'

py (Join-Path $S 't2c_target_stdout.py') (Join-Path $NSD 'g10k_t2c_nsys.sqlite')  (Join-Path $OUT 't2c_nsys_target_stdout.log')
py (Join-Path $S 't2c_target_stdout.py') (Join-Path $NSD 'g10k_t2c_nsys2.sqlite') (Join-Path $OUT 't2c_nsys2_target_stdout.log')

py (Join-Path $S 't2c_nsys_analyze.py')  *> (Join-Path $OUT 't2c_nsys_timeline.log')
py (Join-Path $S 't2c_nsys_analyze2.py') (Join-Path $NSD 'g10k_t2c_nsys.sqlite')  *> (Join-Path $OUT 't2c_nsys_gen_split.log')
py (Join-Path $S 't2c_nsys_analyze2.py') (Join-Path $NSD 'g10k_t2c_nsys2.sqlite') *> (Join-Path $OUT 't2c_nsys2_gen_split.log')

& 'C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.1.3\target-windows-x64\nsys.exe' stats `
    --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum --report cuda_api_sum `
    --format csv --output (Join-Path $NSD 'stats2') --force-overwrite=true (Join-Path $NSD 'g10k_t2c_nsys2.sqlite') *> (Join-Path $OUT 't2c_nsys2_stats_csv.log')

Write-Output "=== produced ==="
Get-ChildItem $OUT -File | Select-Object Name,Length | Format-Table -AutoSize
Get-ChildItem $NSD -File | Select-Object Name,Length | Format-Table -AutoSize
Write-Output "=== end ==="

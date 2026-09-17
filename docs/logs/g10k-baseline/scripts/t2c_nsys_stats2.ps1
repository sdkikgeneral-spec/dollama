# Re-run nsys stats on the ALREADY CAPTURED .sqlite (post-processing only; no GPU work).
# The first pass lost cuda_gpu_kern_sum: nsys hit
#   [libprotobuf ERROR] String field 'Agent.StatsReportExecutionInfo.output' contains invalid UTF-8
# and that report's table never reached stdout. CSV files bypass that path.
$ErrorActionPreference = 'Continue'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$nsys = 'C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.1.3\target-windows-x64\nsys.exe'
$NSD  = 'E:\Develop\logs\g10k-t2c\nsys'
$sq   = Join-Path $NSD 'g10k_t2c_nsys.sqlite'

Write-Output "=== nsys profile --help : output-related flags ==="
$h = & $nsys profile --help 2>&1
$h | Select-String -Pattern 'output' | ForEach-Object { Write-Output ("  " + $_.Line) }

Write-Output ""
Write-Output "=== nsys stats (CSV) on existing sqlite ==="
Write-Output ("sqlite = " + $sq + "  bytes=" + (Get-Item $sq).Length)
& $nsys stats --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum --report cuda_api_sum `
    --format csv --output (Join-Path $NSD 'stats') --force-overwrite=true $sq 2>&1 |
    ForEach-Object { Write-Output $_ }

Write-Output ""
Write-Output "=== generated csv ==="
Get-ChildItem $NSD -Filter '*.csv' | Select-Object Name,Length | Format-Table -AutoSize
Write-Output "=== end ==="

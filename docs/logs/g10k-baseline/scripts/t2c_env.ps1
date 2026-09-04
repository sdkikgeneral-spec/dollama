# G-10k T2c : environment snapshot (ASCII only)
$ErrorActionPreference = 'Continue'
$OUT = 'E:\Develop\logs\g10k-t2c'
New-Item -ItemType Directory -Force -Path $OUT | Out-Null
$repo = 'E:\Develop\Projects\dollama'

Write-Output "=== T2c env snapshot ==="
Write-Output ("date_local = " + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss K'))
Write-Output ("date_utc   = " + (Get-Date).ToUniversalTime().ToString('yyyy-MM-dd HH:mm:ss') + " UTC")
Write-Output ("hostname   = " + $env:COMPUTERNAME)
Write-Output ("os         = " + (Get-CimInstance Win32_OperatingSystem).Caption + " " + (Get-CimInstance Win32_OperatingSystem).Version)
Write-Output ("pwsh       = " + $PSVersionTable.PSVersion.ToString())

Write-Output ""
Write-Output "=== git (research machine) ==="
Set-Location $repo
git rev-parse HEAD
git branch --show-current
Write-Output "--- porcelain start ---"
git status --porcelain
Write-Output "--- porcelain end ---"
git log --oneline -3

Write-Output ""
Write-Output "=== nvidia-smi (full) ==="
nvidia-smi
Write-Output ""
Write-Output "=== nvidia-smi query (idle) ==="
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,temperature.gpu,clocks.sm,clocks.mem,power.draw,power.limit,utilization.gpu --format=csv

Write-Output ""
Write-Output "=== nvcc ==="
nvcc --version

Write-Output ""
Write-Output "=== meson compile (expect: no work to do) ==="
py -m mesonbuild.mesonmain compile -C build

Write-Output ""
Write-Output "=== exe sha256 (post compile-check) ==="
foreach ($n in @('test_diffusion_batch2.exe','prof_arena_e2e.exe','dollama.exe','test_conv2d.exe'))
{
    $p = Join-Path $repo ('build\src\' + $n)
    if (Test-Path $p)
    {
        $h = (Get-FileHash $p -Algorithm SHA256).Hash
        $f = Get-Item $p
        Write-Output ("{0}  size={1}  mtime={2}  sha256={3}" -f $n, $f.Length, $f.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'), $h)
    }
}

Write-Output ""
Write-Output "=== nsys presence ==="
$nsysRoot = 'C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.1.3'
if (Test-Path $nsysRoot)
{
    Get-ChildItem -Path $nsysRoot -Recurse -Filter 'nsys*.exe' -ErrorAction SilentlyContinue | ForEach-Object { Write-Output $_.FullName }
}
else
{
    Write-Output "NOT FOUND: $nsysRoot"
    Get-ChildItem 'C:\Program Files\NVIDIA Corporation' -ErrorAction SilentlyContinue | ForEach-Object { Write-Output ("sibling: " + $_.Name) }
}
Write-Output "=== end ==="

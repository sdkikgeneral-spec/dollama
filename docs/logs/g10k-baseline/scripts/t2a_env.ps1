# T2a: 環境スナップショット採取 (研究機 KIK-WIN-RTX58)
# 必ず pwsh (PowerShell 7) で実行すること。powershell 5.1 は BOM 無し UTF-8 を
# CP932 と解釈して日本語コメント/文字列が壊れる。
$ErrorActionPreference = 'Continue'
$repo = 'E:\Develop\Projects\dollama'
$out  = 'E:\Develop\logs\g10k-t2a'
New-Item -ItemType Directory -Force -Path $out | Out-Null
$log = Join-Path $out 'env_snapshot.log'

$meson = 'C:\Users\sdkik\AppData\Local\Python\pythoncore-3.14-64\Scripts\meson.exe'

$lines = @()
$lines += '=== T2a 環境スナップショット (研究機) ==='
$lines += ('採取日時 (ローカル): ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss K'))
$lines += ('採取日時 (UTC)     : ' + (Get-Date).ToUniversalTime().ToString('yyyy-MM-dd HH:mm:ss') + ' UTC')
$lines += ('hostname           : ' + (hostname))
$lines += ('OS                 : ' + (Get-CimInstance Win32_OperatingSystem).Caption + ' ' + [Environment]::OSVersion.Version.ToString())
$lines += ('PowerShell         : ' + $PSVersionTable.PSVersion.ToString())
$lines += ''
$lines += '=== git (研究機) ==='
Push-Location $repo
$lines += ('branch    : ' + (git branch --show-current))
$lines += ('HEAD      : ' + (git rev-parse HEAD))
$porcelain = @(git status --porcelain)
if ($porcelain.Count -eq 0) {
    $lines += 'status --porcelain: (出力 0 行 = clean)'
} else {
    $lines += 'status --porcelain:'
    $lines += $porcelain
}
Pop-Location
$lines += ''
$lines += '=== nvcc --version ==='
$lines += (& nvcc --version 2>&1 | Out-String).TrimEnd()
$lines += ''
$lines += '=== meson --version ==='
if (Test-Path $meson) {
    $lines += ('meson path: ' + $meson)
    $lines += ('meson version: ' + (& $meson --version 2>&1 | Out-String).Trim())
} else {
    $lines += ('meson.exe が見つからない: ' + $meson)
}
$lines += ''
$lines += '=== nvidia-smi (フル出力) ==='
$lines += (& nvidia-smi 2>&1 | Out-String).TrimEnd()
$lines += ''
$lines += '=== nvidia-smi --query (要約) ==='
$lines += (& nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,power.draw,power.limit,temperature.gpu,clocks.current.sm --format=csv 2>&1 | Out-String).TrimEnd()
$lines += ''
$lines += '=== 実行に使う exe の同一性 (T2b との突合用) ==='
$lines += '★ 下の LastWriteTime は「本 T2a 走行より前に既に存在していた」ビルド成果物の時刻。'
$lines += '★ 本タスクでは meson compile を一切実行していない。'
$targets = @('prof_arena_e2e.exe','test_conv2d.exe','test_diffusion_batch2.exe','test_unet_fast.exe','dollama.exe')
foreach ($t in $targets) {
    $p = Join-Path $repo ('build\src\' + $t)
    if (Test-Path $p) {
        $fi = Get-Item $p
        $h  = (Get-FileHash -Algorithm SHA256 $p).Hash
        $lines += ('{0,-26} size={1,-9} mtime={2}  sha256={3}' -f $t, $fi.Length, $fi.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'), $h)
    } else {
        $lines += ('{0,-26} (存在しない)' -f $t)
    }
}

$lines | Out-File -FilePath $log -Encoding utf8
Write-Output ('WROTE ' + $log)
Get-FileHash -Algorithm SHA256 $log | ForEach-Object { Write-Output ('SHA256 ' + $_.Hash) }

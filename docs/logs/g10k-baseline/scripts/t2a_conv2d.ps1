# T2a: 生成物ファイル (バイト列) の基準採取 — test_conv2d の DOLLAMA_G8K_DUMP
#   launch_conv2d の出力そのものを .bin へ書き出す (N=1 / N=2 の 7 形状)。
#   G-10k (conv true batch2) が触るカーネルの before バイト列基準。
# 必ず pwsh で実行すること。
param([int]$RunIndex = 1)

$ErrorActionPreference = 'Continue'
$repo = 'E:\Develop\Projects\dollama'
$out  = 'E:\Develop\logs\g10k-t2a'
$bin  = Join-Path $out ('bin_run{0}' -f $RunIndex)
$exe  = Join-Path $repo 'build\src\test_conv2d.exe'

New-Item -ItemType Directory -Force -Path $bin | Out-Null
Get-ChildItem $bin -Filter *.bin -ErrorAction SilentlyContinue | Remove-Item -Force

foreach ($k in @('DOLLAMA_PROFILE','DOLLAMA_POOL','DOLLAMA_ARENA_RELEASE','DOLLAMA_ARENA_RESERVE_MB','DOLLAMA_EPILOGUE','DOLLAMA_BATCH2','DOLLAMA_FAST')) {
    if (Test-Path ('env:' + $k)) { Remove-Item ('env:' + $k) }
}
$env:DOLLAMA_G8K_DUMP = Join-Path $bin 'conv'

$tag = 'conv2d_run{0}' -f $RunIndex
$so  = Join-Path $out ($tag + '.stdout.tmp')
$se  = Join-Path $out ($tag + '.stderr.tmp')
$log = Join-Path $out ($tag + '.log')

$t0 = Get-Date
$p = Start-Process -FilePath $exe -NoNewWindow -Wait -PassThru -RedirectStandardOutput $so -RedirectStandardError $se
$t1 = Get-Date

# --- 生成された .bin の sha256 を採る (これが「生成物ファイルのバイト列」基準) ---
$hashLines = @()
Get-ChildItem $bin -Filter *.bin | Sort-Object Name | ForEach-Object {
    $h = (Get-FileHash -Algorithm SHA256 $_.FullName).Hash
    $hashLines += ('{0,-24} {1,12} bytes  sha256={2}' -f $_.Name, $_.Length, $h)
}

$hdr = @()
$hdr += ('=== T2a conv2d バイト列基準 走行 #{0} (test_conv2d) ===' -f $RunIndex)
$hdr += ('開始 : ' + $t0.ToString('yyyy-MM-dd HH:mm:ss'))
$hdr += ('終了 : ' + $t1.ToString('yyyy-MM-dd HH:mm:ss'))
$hdr += ('経過秒: ' + [math]::Round(($t1 - $t0).TotalSeconds, 3))
$hdr += ('exit code: ' + $p.ExitCode)
$hdr += ('exe  : ' + $exe)
$hdr += ('exe sha256 : ' + (Get-FileHash -Algorithm SHA256 $exe).Hash)
$hdr += ('env  : DOLLAMA_G8K_DUMP=' + $env:DOLLAMA_G8K_DUMP)
$hdr += '       DOLLAMA_PROFILE=(未設定) DOLLAMA_POOL=(未設定=既定 ON)'
$hdr += 'seed : 各形状ごとに test_conv2d.cu:696-703 でハードコード (3101..3107)'
$hdr += ''
$hdr += ('=== 生成物 .bin の sha256 ({0} 件) ===' -f $hashLines.Count)
$hdr += $hashLines
$hdr += ''
$hdr += '--- stdout ---'
$hdr += (Get-Content $so -Raw -ErrorAction SilentlyContinue)
$hdr += '--- stderr ---'
$errTxt = (Get-Content $se -Raw -ErrorAction SilentlyContinue)
if ([string]::IsNullOrWhiteSpace($errTxt)) { $hdr += '(空)' } else { $hdr += $errTxt }

$hdr | Out-File -FilePath $log -Encoding utf8
Remove-Item $so, $se -ErrorAction SilentlyContinue

Write-Output ('WROTE ' + $log + '  exit=' + $p.ExitCode + '  elapsed=' + [math]::Round(($t1-$t0).TotalSeconds,1) + 's')
Write-Output ('dump dir: ' + $bin)
$hashLines | ForEach-Object { Write-Output $_ }
Write-Output '--- 異常行 ---'
$bad = Select-String -Path $log -Pattern 'DIFF|FAIL|reserve shortage|exception|ERROR|SKIP'
if ($null -eq $bad) { Write-Output '(なし)' } else { $bad | ForEach-Object { Write-Output $_.Line } }

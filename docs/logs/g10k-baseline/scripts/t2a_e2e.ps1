# T2a: e2e 基準採取 — prof_arena_e2e (generate_txt2img・固定 seed 1234・batch2 ON)
# 必ず pwsh で実行すること。
param([int]$RunIndex = 1)

$ErrorActionPreference = 'Continue'
$repo = 'E:\Develop\Projects\dollama'
$out  = 'E:\Develop\logs\g10k-t2a'
$exe  = Join-Path $repo 'build\src\prof_arena_e2e.exe'

# --- 被験構成の env を明示的に組む。DOLLAMA_PROFILE は「未設定」が条件なので必ず消す ---
foreach ($k in @('DOLLAMA_PROFILE','DOLLAMA_POOL','DOLLAMA_ARENA_RELEASE','DOLLAMA_ARENA_RESERVE_MB','DOLLAMA_EPILOGUE','DOLLAMA_BATCH2','DOLLAMA_FAST','DOLLAMA_G8K_DUMP')) {
    if (Test-Path ('env:' + $k)) { Remove-Item ('env:' + $k) }
}
$env:PROF_IMAGES    = '2'
$env:PROF_STEPS     = '20'
$env:PROF_G         = '7.5'
$env:PROF_FAST      = '1'
$env:PROF_SAMPLE_MS = '5'

$tag = 'e2e_run{0}' -f $RunIndex
$so  = Join-Path $out ($tag + '.stdout.tmp')
$se  = Join-Path $out ($tag + '.stderr.tmp')
$log = Join-Path $out ($tag + '.log')

$smiBefore = (& nvidia-smi --query-gpu=temperature.gpu,clocks.current.sm,memory.used,power.draw --format=csv,noheader 2>&1 | Out-String).Trim()
$t0 = Get-Date
$p = Start-Process -FilePath $exe -NoNewWindow -Wait -PassThru -RedirectStandardOutput $so -RedirectStandardError $se
$t1 = Get-Date
$smiAfter = (& nvidia-smi --query-gpu=temperature.gpu,clocks.current.sm,memory.used,power.draw --format=csv,noheader 2>&1 | Out-String).Trim()

$hdr = @()
$hdr += ('=== T2a e2e 基準走行 #{0} (prof_arena_e2e) ===' -f $RunIndex)
$hdr += ('開始 : ' + $t0.ToString('yyyy-MM-dd HH:mm:ss'))
$hdr += ('終了 : ' + $t1.ToString('yyyy-MM-dd HH:mm:ss'))
$hdr += ('経過秒: ' + [math]::Round(($t1 - $t0).TotalSeconds, 3))
$hdr += ('exit code: ' + $p.ExitCode)
$hdr += ('exe  : ' + $exe)
$hdr += ('exe sha256 : ' + (Get-FileHash -Algorithm SHA256 $exe).Hash)
$hdr += 'env  : PROF_IMAGES=2 PROF_STEPS=20 PROF_G=7.5 PROF_FAST=1 PROF_SAMPLE_MS=5'
$hdr += '       DOLLAMA_PROFILE=(未設定) DOLLAMA_POOL=(未設定=既定 ON) DOLLAMA_ARENA_*=(未設定)'
$hdr += 'seed : 1234 (prof_arena_e2e.cu:233 でハードコード)'
$hdr += ('nvidia-smi 前 (temp,sm_clk,mem_used,power): ' + $smiBefore)
$hdr += ('nvidia-smi 後 (temp,sm_clk,mem_used,power): ' + $smiAfter)
$hdr += ''
$hdr += '--- stdout ---'
$hdr += (Get-Content $so -Raw -ErrorAction SilentlyContinue)
$hdr += '--- stderr ---'
$errTxt = (Get-Content $se -Raw -ErrorAction SilentlyContinue)
if ([string]::IsNullOrWhiteSpace($errTxt)) { $hdr += '(空)' } else { $hdr += $errTxt }

$hdr | Out-File -FilePath $log -Encoding utf8
Remove-Item $so, $se -ErrorAction SilentlyContinue

Write-Output ('WROTE ' + $log + '  exit=' + $p.ExitCode + '  elapsed=' + [math]::Round(($t1-$t0).TotalSeconds,1) + 's')
Write-Output '--- rgb_hash / PEAK / 異常行の抜粋 ---'
Select-String -Path $log -Pattern 'rgb_hash|PEAK_USED|reserve shortage|exception|ERROR|SKIP' | ForEach-Object { Write-Output $_.Line }

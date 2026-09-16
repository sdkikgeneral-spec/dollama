# G-10k T7: 任意の exe を 1 プロセス実走し、生ログをヘッダ付きで退避する (T6 の t6_run.ps1 と同型・出力先のみ g10k-e2e)。
#   使い方: pwsh -File t7_run.ps1 -Label <ラベル> -Exe <exe 相対または絶対パス> [-Cwd <作業ディレクトリ>] [-EnvSet 'K=V;K2=V2'] [-Note <文>]
#   DOLLAMA_* / PROF_* / DB2_* は先に全て未設定にしてから -EnvSet の分だけ設定する (プロセス単位固定の
#   getenv キャッシュ型キルスイッチが混ざらないよう、既定走行は「全 <unset>」を明示的に保証する)。
#   -EnvSet は ';' 区切りの単一文字列。ログのヘッダ "--- env ---" 節が実際に効いた env の一次証拠。
param(
    [Parameter(Mandatory = $true)][string]$Label,
    [Parameter(Mandatory = $true)][string]$Exe,
    [string]$Cwd = 'E:\Develop\Projects\dollama\build',
    [string]$EnvSet = '',
    [string]$Note = ''
)
$ErrorActionPreference = 'Continue'
$repoRoot = 'E:\Develop\Projects\dollama'
Set-Location $repoRoot

$keys = 'DOLLAMA_CONV_BATCH','DOLLAMA_GEMM','DOLLAMA_POOL','DOLLAMA_PROFILE','DOLLAMA_EPILOGUE','DOLLAMA_G8K_DUMP',
        'DOLLAMA_ARENA_RESERVE_MB','DOLLAMA_ARENA_RELEASE',
        'PROF_IMAGES','PROF_STEPS','PROF_G','PROF_FAST','PROF_SAMPLE_MS',
        'DB2_BENCH','DB2_BENCH_STEPS','DB2_BENCH_ITERS','DB2_BENCH_G','DB2_STEPS','DB2_UNCOND_ZERO'
foreach ($k in $keys) { Remove-Item "Env:$k" -ErrorAction SilentlyContinue }
foreach ($kv in ($EnvSet -split ';' | Where-Object { $_ -ne '' })) {
    $i = $kv.IndexOf('=')
    [Environment]::SetEnvironmentVariable($kv.Substring(0, $i), $kv.Substring($i + 1))
}

$log = "docs\logs\g10k-e2e\t7_$Label.log"

$hdr = @()
$hdr += "===== G-10k T7 実走ヘッダ ($Label) ====="
if ($Note -ne '') { $hdr += "note         = $Note" }
$hdr += 'start(local) = ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss K')
$hdr += 'hostname     = ' + $env:COMPUTERNAME
$hdr += 'HEAD         = ' + (git rev-parse HEAD)
$hdr += 'porcelain    = [' + ((git status --porcelain) -join '|') + ']  (空 = clean)'
$hdr += "exe          = $Exe"
$hdr += "cwd          = $Cwd"
$hdr += 'exe sha256   = ' + (Get-FileHash $Exe -Algorithm SHA256).Hash
foreach ($f in 'src\kernels\conv2d.cu','src\tests\test_diffusion_batch2.cu') {
    $hdr += ('{0,-36} sha256 = {1}' -f $f, (Get-FileHash (Join-Path $repoRoot $f) -Algorithm SHA256).Hash)
}
$hdr += 'SAC VerifiedAndReputablePolicyState = 未確認 (裏取りせず)'
$hdr += 'nvidia-smi (pre) = ' + ((nvidia-smi --query-gpu=memory.used,temperature.gpu,clocks.sm,power.draw --format=csv,noheader) -join ' ')
$hdr += '--- env ---'
foreach ($k in $keys) {
    $v = [Environment]::GetEnvironmentVariable($k)
    if ($null -eq $v) { $hdr += "  $k = <unset>" } else { $hdr += "  $k = $v" }
}
$hdr += '===== stdout+stderr (2>&1) ====='
$hdr | Out-File -FilePath $log -Encoding utf8

Push-Location $Cwd
& $Exe 2>&1 | Out-File -FilePath (Join-Path $repoRoot $log) -Encoding utf8 -Append
$code = $LASTEXITCODE
Pop-Location
"end(local)   = " + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss K') | Out-File -FilePath $log -Encoding utf8 -Append
'nvidia-smi (post) = ' + ((nvidia-smi --query-gpu=memory.used,temperature.gpu,clocks.sm,power.draw --format=csv,noheader) -join ' ') | Out-File -FilePath $log -Encoding utf8 -Append
"exit=$code" | Out-File -FilePath $log -Encoding utf8 -Append
Write-Host "log=$log exit=$code"
exit $code

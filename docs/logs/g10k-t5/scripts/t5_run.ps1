# G-10k T5: 任意の exe を 1 プロセス実走し、生ログをヘッダ付きで退避する (T4 の t4_run_conv2d.ps1 と同型)。
#   使い方: pwsh -File t5_run.ps1 -Label <ラベル> -Exe <exe 相対パス> [-EnvSet 'K=V;K2=V2'] [-Note <文>]
#   DOLLAMA_* / PROF_* / DB2_* は先に全て未設定にしてから -EnvSet の分だけ設定する (プロセス単位固定の
#   getenv キャッシュ型キルスイッチが混ざらないよう、既定走行は「全 <unset>」を明示的に保証する)。
#   ★-EnvSet は ';' 区切りの単一文字列。初版の [string[]] は `pwsh -File` 越しに複数要素が落ちて
#     全 <unset> の走行になった (DISCARDED_arena_e2e_pool0_envset_not_applied.log) ため単一文字列に変更。
#     ログのヘッダ "--- env ---" 節が実際に効いた env の一次証拠なので、走行後は必ずそこを読むこと。
param(
    [Parameter(Mandatory = $true)][string]$Label,
    [Parameter(Mandatory = $true)][string]$Exe,
    [string]$EnvSet = '',
    [string]$Note = ''
)
$ErrorActionPreference = 'Continue'
Set-Location E:\Develop\Projects\dollama

$keys = 'DOLLAMA_CONV_BATCH','DOLLAMA_GEMM','DOLLAMA_POOL','DOLLAMA_PROFILE','DOLLAMA_EPILOGUE','DOLLAMA_G8K_DUMP',
        'DOLLAMA_ARENA_RESERVE_MB','DOLLAMA_ARENA_RELEASE',
        'PROF_IMAGES','PROF_STEPS','PROF_G','PROF_FAST','PROF_SAMPLE_MS',
        'DB2_BENCH','DB2_BENCH_STEPS','DB2_BENCH_ITERS','DB2_BENCH_G','DB2_STEPS','DB2_UNCOND_ZERO'
foreach ($k in $keys) { Remove-Item "Env:$k" -ErrorAction SilentlyContinue }
foreach ($kv in ($EnvSet -split ';' | Where-Object { $_ -ne '' })) {
    $i = $kv.IndexOf('=')
    [Environment]::SetEnvironmentVariable($kv.Substring(0, $i), $kv.Substring($i + 1))
}

$log = "docs\logs\g10k-t5\t5_$Label.log"

$hdr = @()
$hdr += "===== G-10k T5 実走ヘッダ ($Label) ====="
if ($Note -ne '') { $hdr += "note         = $Note" }
$hdr += 'start(local) = ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss K')
$hdr += 'hostname     = ' + $env:COMPUTERNAME
$hdr += 'HEAD         = ' + (git rev-parse HEAD)
$hdr += 'porcelain    = [' + ((git status --porcelain) -join '|') + ']  (空 = clean)'
$hdr += "exe          = $Exe"
$hdr += 'exe sha256   = ' + (Get-FileHash $Exe -Algorithm SHA256).Hash
foreach ($f in 'src\kernels\conv2d.cu','src\kernels\gemm.cu','src\kernels\gemm.cuh','src\kernels\device_arena.cu',
               'src\infer\diffusion.cu','src\infer\unet.cu',
               'src\tests\test_conv2d.cu','src\tests\test_diffusion_batch2.cu','src\tests\test_unet_fast.cu','src\tests\prof_arena_e2e.cu') {
    $hdr += ('{0,-36} sha256 = {1}' -f $f, (Get-FileHash $f -Algorithm SHA256).Hash)
}
$hdr += 'SAC VerifiedAndReputablePolicyState = ' + (Get-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy' -Name VerifiedAndReputablePolicyState -ErrorAction SilentlyContinue).VerifiedAndReputablePolicyState
$hdr += 'nvidia-smi (pre) = ' + ((nvidia-smi --query-gpu=memory.used,temperature.gpu,clocks.sm,power.draw --format=csv,noheader) -join ' ')
$hdr += '--- env ---'
foreach ($k in $keys) {
    $v = [Environment]::GetEnvironmentVariable($k)
    if ($null -eq $v) { $hdr += "  $k = <unset>" } else { $hdr += "  $k = $v" }
}
$hdr += '===== stdout+stderr (2>&1) ====='
$hdr | Out-File -FilePath $log -Encoding utf8

& $Exe 2>&1 | Out-File -FilePath $log -Encoding utf8 -Append
$code = $LASTEXITCODE
"end(local)   = " + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss K') | Out-File -FilePath $log -Encoding utf8 -Append
'nvidia-smi (post) = ' + ((nvidia-smi --query-gpu=memory.used,temperature.gpu,clocks.sm,power.draw --format=csv,noheader) -join ' ') | Out-File -FilePath $log -Encoding utf8 -Append
"exit=$code" | Out-File -FilePath $log -Encoding utf8 -Append
Write-Host "log=$log exit=$code"
exit $code

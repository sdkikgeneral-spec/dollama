# G-10k T4: test_conv2d.exe を 1 プロセス実走し、生ログをヘッダ付きで退避する。
#   使い方: pwsh -File t4_run_conv2d.ps1 -Label <ラベル> [-ConvBatchOff] [-Note <文>]
#   -ConvBatchOff を付けると DOLLAMA_CONV_BATCH=0 (キルスイッチ経路) で走らせる。
#   それ以外の DOLLAMA_* は未設定を明示的に保証する。
param(
    [Parameter(Mandatory = $true)][string]$Label,
    [switch]$ConvBatchOff,
    [string]$Note = ''
)
$ErrorActionPreference = 'Continue'
Set-Location E:\Develop\Projects\dollama

Remove-Item Env:DOLLAMA_CONV_BATCH -ErrorAction SilentlyContinue
Remove-Item Env:DOLLAMA_GEMM       -ErrorAction SilentlyContinue
Remove-Item Env:DOLLAMA_POOL       -ErrorAction SilentlyContinue
Remove-Item Env:DOLLAMA_PROFILE    -ErrorAction SilentlyContinue
Remove-Item Env:DOLLAMA_EPILOGUE   -ErrorAction SilentlyContinue
Remove-Item Env:DOLLAMA_G8K_DUMP   -ErrorAction SilentlyContinue
if ($ConvBatchOff) { $env:DOLLAMA_CONV_BATCH = '0' }

$log = "docs\logs\g10k-t4\t4_$Label.log"
$exe = 'build\src\test_conv2d.exe'

$hdr = @()
$hdr += "===== G-10k T4 実走ヘッダ ($Label) ====="
if ($Note -ne '') { $hdr += "note         = $Note" }
$hdr += 'start(local) = ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss K')
$hdr += 'hostname     = ' + $env:COMPUTERNAME
$hdr += 'HEAD         = ' + (git rev-parse HEAD)
$hdr += 'porcelain    = [' + ((git status --porcelain) -join '|') + ']  (空 = clean)'
$hdr += "exe          = $exe"
$hdr += 'exe sha256   = ' + (Get-FileHash $exe -Algorithm SHA256).Hash
$hdr += 'conv2d.cu sha256      = ' + (Get-FileHash 'src\kernels\conv2d.cu' -Algorithm SHA256).Hash
$hdr += 'test_conv2d.cu sha256 = ' + (Get-FileHash 'src\tests\test_conv2d.cu' -Algorithm SHA256).Hash
$hdr += 'gemm.cu sha256        = ' + (Get-FileHash 'src\kernels\gemm.cu' -Algorithm SHA256).Hash
$hdr += 'gemm.cuh sha256       = ' + (Get-FileHash 'src\kernels\gemm.cuh' -Algorithm SHA256).Hash
$hdr += 'SAC VerifiedAndReputablePolicyState = ' + (Get-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy' -Name VerifiedAndReputablePolicyState -ErrorAction SilentlyContinue).VerifiedAndReputablePolicyState
$hdr += '--- env ---'
foreach ($k in 'DOLLAMA_CONV_BATCH','DOLLAMA_GEMM','DOLLAMA_POOL','DOLLAMA_PROFILE','DOLLAMA_EPILOGUE','DOLLAMA_G8K_DUMP') {
    $v = [Environment]::GetEnvironmentVariable($k)
    if ($null -eq $v) { $hdr += "  $k = <unset>" } else { $hdr += "  $k = $v" }
}
$hdr += '===== stdout+stderr (2>&1) ====='
$hdr | Out-File -FilePath $log -Encoding utf8

& $exe 2>&1 | Out-File -FilePath $log -Encoding utf8 -Append
$code = $LASTEXITCODE
"end(local)   = " + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss K') | Out-File -FilePath $log -Encoding utf8 -Append
"exit=$code" | Out-File -FilePath $log -Encoding utf8 -Append
Write-Host "log=$log exit=$code"
exit $code

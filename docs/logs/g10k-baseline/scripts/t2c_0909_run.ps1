# G-10k T2c: DB2_BENCH (profile ON) 1 プロセス実走。採取のみ・判定なし。
$ErrorActionPreference = 'Continue'
Set-Location E:\Develop\Projects\dollama

# --- env の完全 pin (既定値に依存させない) ---
$env:DB2_BENCH        = '1'
$env:DB2_BENCH_STEPS  = '20'
$env:DB2_BENCH_ITERS  = '1'
$env:DOLLAMA_PROFILE  = '1'
# 以下は「未設定」であることを明示的に保証する
Remove-Item Env:DB2_BENCH_G          -ErrorAction SilentlyContinue
Remove-Item Env:DOLLAMA_CONV_BATCH   -ErrorAction SilentlyContinue
Remove-Item Env:DOLLAMA_POOL         -ErrorAction SilentlyContinue
Remove-Item Env:DOLLAMA_EPILOGUE     -ErrorAction SilentlyContinue
Remove-Item Env:DOLLAMA_ARENA_RESERVE_MB -ErrorAction SilentlyContinue
Remove-Item Env:DOLLAMA_GEMM         -ErrorAction SilentlyContinue

$log = 'E:\Develop\logs\g10k-t2c\t2c_db2bench_profile.log'

$hdr = @()
$hdr += '===== G-10k T2c 実走ヘッダ (採取のみ・判定なし) ====='
$hdr += 'start(local) = ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss K')
$hdr += 'hostname     = ' + $env:COMPUTERNAME
$hdr += 'HEAD         = ' + (git rev-parse HEAD)
$hdr += 'porcelain    = [' + ((git status --porcelain) -join '|') + ']  (空 = clean)'
$hdr += 'exe          = build\src\test_diffusion_batch2.exe'
$hdr += 'exe sha256   = ' + (Get-FileHash 'build\src\test_diffusion_batch2.exe' -Algorithm SHA256).Hash
$hdr += 'cwd          = ' + (Get-Location).Path
$hdr += '--- env (set) ---'
$hdr += '  DB2_BENCH=1  DB2_BENCH_STEPS=20  DB2_BENCH_ITERS=1  DOLLAMA_PROFILE=1'
$hdr += '--- env (explicitly UNSET; defaults apply) ---'
foreach ($n in 'DB2_BENCH_G','DOLLAMA_CONV_BATCH','DOLLAMA_POOL','DOLLAMA_EPILOGUE','DOLLAMA_ARENA_RESERVE_MB','DOLLAMA_GEMM') {
  $v = [Environment]::GetEnvironmentVariable($n)
  $hdr += ('  {0} = {1}' -f $n, $(if ($null -eq $v) { '<unset>' } else { "SET:$v" }))
}
$hdr += '  (seed は harness ハードコード 1234 / DB2_BENCH_G 未設定 = 既定 guidance 7.5)'
$hdr += '--- nvidia-smi (pre-run, 1 行) ---'
$hdr += (nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw,memory.used --format=csv,noheader)
$hdr += '===================================================='
$hdr += ''
$hdr | Set-Content -Path $log -Encoding utf8

$sw = [System.Diagnostics.Stopwatch]::StartNew()
& 'E:\Develop\Projects\dollama\build\src\test_diffusion_batch2.exe' *>&1 | Tee-Object -FilePath $log -Append
$code = $LASTEXITCODE
$sw.Stop()

$ftr = @()
$ftr += ''
$ftr += '===== G-10k T2c 実走フッタ ====='
$ftr += 'exit code    = ' + $code
$ftr += 'wall elapsed = ' + [math]::Round($sw.Elapsed.TotalSeconds,2) + ' s  (プロセス全体・harness の e2e 秒とは別)'
$ftr += 'end(local)   = ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss K')
$ftr += '--- nvidia-smi (post-run, 1 行) ---'
$ftr += (nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw,memory.used --format=csv,noheader)
$ftr += '================================'
$ftr | Add-Content -Path $log -Encoding utf8
exit $code

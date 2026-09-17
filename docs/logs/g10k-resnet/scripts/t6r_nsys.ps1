# G-10k T6r: prof_conv_breakdown を nsys (--capture-range=cudaProfilerApi) で 5 形状 × batched/seq 実走し、
#   cuda_gpu_kern_sum (カーネル名別集計) を CSV へ落とす。src コミットなし・計測 exe は作業ツリー限定。
#   使い方: pwsh -File docs\logs\g10k-resnet\scripts\t6r_nsys.ps1 [-Iters 50]
param([int]$Iters = 50)
$ErrorActionPreference = 'Continue'
Set-Location E:\Develop\Projects\dollama
$nsys = 'C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.1.3\target-windows-x64\nsys.exe'
$exe  = 'build\src\prof_conv_breakdown.exe'
$dir  = 'docs\logs\g10k-resnet\t6r'
New-Item -ItemType Directory -Force $dir | Out-Null
foreach ($k in 'DOLLAMA_CONV_BATCH','DOLLAMA_GEMM','DOLLAMA_POOL','DOLLAMA_PROFILE') { Remove-Item "Env:$k" -ErrorAction SilentlyContinue }

$hdr = @()
$hdr += '===== G-10k T6r 実走ヘッダ ====='
$hdr += 'start(local) = ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss K')
$hdr += 'hostname     = ' + $env:COMPUTERNAME
$hdr += 'HEAD         = ' + (git rev-parse HEAD)
$hdr += 'porcelain    = [' + ((git status --porcelain) -join '|') + ']'
$hdr += 'exe sha256   = ' + (Get-FileHash $exe -Algorithm SHA256).Hash
foreach ($f in 'src\kernels\conv2d.cu','src\kernels\gemm.cu','src\tests\prof_conv_breakdown.cu') {
    $hdr += ('{0,-36} sha256 = {1}' -f $f, (Get-FileHash $f -Algorithm SHA256).Hash)
}
$hdr += 'SAC VerifiedAndReputablePolicyState = ' + (Get-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy' -Name VerifiedAndReputablePolicyState -ErrorAction SilentlyContinue).VerifiedAndReputablePolicyState
$hdr += 'nvidia-smi (pre) = ' + ((nvidia-smi --query-gpu=memory.used,temperature.gpu,clocks.sm,power.draw --format=csv,noheader) -join ' ')
$hdr += "iters = $Iters"
$hdr += '===== runs ====='
$hdr | Out-File "$dir\summary.log" -Encoding utf8

foreach ($shape in 'rep_320_128','rep_640_64','rep_1280_32','G4_band_640to320_128','unet_c320_64') {
    foreach ($mode in 'batched','seq') {
        $tag = "${shape}_${mode}"
        & $nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop --trace=cuda -o "$dir\$tag" --force-overwrite true $exe $shape $mode $Iters 2>&1 | Out-File "$dir\$tag.stdout.log" -Encoding utf8
        "[$tag] exit=$LASTEXITCODE" | Out-File "$dir\summary.log" -Encoding utf8 -Append
        Get-Content "$dir\$tag.stdout.log" | Select-String 'prof_conv_breakdown' | Out-File "$dir\summary.log" -Encoding utf8 -Append
        & $nsys stats --report cuda_gpu_kern_sum --format csv --force-export true -o "$dir\$tag" "$dir\$tag.nsys-rep" 2>&1 | Out-Null
    }
}
'end(local)   = ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss K') | Out-File "$dir\summary.log" -Encoding utf8 -Append
'nvidia-smi (post) = ' + ((nvidia-smi --query-gpu=memory.used,temperature.gpu,clocks.sm,power.draw --format=csv,noheader) -join ' ') | Out-File "$dir\summary.log" -Encoding utf8 -Append
Get-Content "$dir\summary.log"

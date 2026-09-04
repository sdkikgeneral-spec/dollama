# G-10k T2c : stage build artifacts to a PERSISTENT path (repo-external, no session id)
# The T2c run itself will be launched FROM this path, so "the exe used" == "the exe saved".
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$repo   = 'E:\Develop\Projects\dollama'
$srcdir = Join-Path $repo 'build\src'
$dst    = 'E:\Develop\logs\g10k-t2c\exe'
New-Item -ItemType Directory -Force -Path $dst | Out-Null

$exes = @('test_diffusion_batch2.exe','prof_arena_e2e.exe')
foreach ($e in $exes)
{
    Copy-Item (Join-Path $srcdir $e) (Join-Path $dst $e) -Force
}
# test_diffusion_batch2 / prof_arena_e2e link OpenVINO (src/meson.build:1-7 deps += openvino_dep),
# so the side-by-side DLLs must travel with the exe for it to start from another directory.
Copy-Item (Join-Path $srcdir '*.dll') $dst -Force

Write-Output "=== staged to $dst ==="
foreach ($e in $exes)
{
    $a = (Get-FileHash (Join-Path $srcdir $e) -Algorithm SHA256).Hash
    $b = (Get-FileHash (Join-Path $dst    $e) -Algorithm SHA256).Hash
    $same = if ($a -eq $b) { 'IDENTICAL' } else { 'MISMATCH' }
    Write-Output ("{0}`n  build/src : {1}`n  staged    : {2}`n  -> {3}" -f $e, $a, $b, $same)
}
Write-Output ""
Write-Output "=== staged dir listing ==="
Get-ChildItem $dst | Select-Object Name,Length,LastWriteTime | Format-Table -AutoSize
Write-Output ("dll_count = " + (Get-ChildItem $dst -Filter *.dll).Count)
Write-Output ("total_bytes = " + ((Get-ChildItem $dst | Measure-Object -Property Length -Sum).Sum))

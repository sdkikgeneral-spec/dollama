$ErrorActionPreference = 'Continue'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OUT = 'E:\Develop\logs\g10k-t2c'
$NSD = Join-Path $OUT 'nsys'
Write-Output "=== research-machine sha256 (files to be collected) ==="
Get-ChildItem $OUT -File | Sort-Object Name | ForEach-Object {
    Write-Output ("{0}  {1}  {2}" -f (Get-FileHash $_.FullName -Algorithm SHA256).Hash, $_.Length, $_.Name)
}
Get-ChildItem $NSD -Filter '*.csv' | Sort-Object Name | ForEach-Object {
    Write-Output ("{0}  {1}  nsys/{2}" -f (Get-FileHash $_.FullName -Algorithm SHA256).Hash, $_.Length, $_.Name)
}
Write-Output ""
Write-Output "=== large artifacts kept on the research machine (NOT collected) ==="
Get-ChildItem $NSD -Include '*.nsys-rep','*.sqlite' -File | Sort-Object Name | ForEach-Object {
    Write-Output ("{0}  {1} bytes  {2}" -f (Get-FileHash $_.FullName -Algorithm SHA256).Hash, $_.Length, $_.FullName)
}
Write-Output ""
Write-Output "=== saved exe (persistent path) ==="
Get-ChildItem (Join-Path $OUT 'exe') -Filter '*.exe' | Sort-Object Name | ForEach-Object {
    Write-Output ("{0}  {1} bytes  {2}" -f (Get-FileHash $_.FullName -Algorithm SHA256).Hash, $_.Length, $_.FullName)
}
Write-Output ""
Write-Output "=== git re-check (after all runs) ==="
Set-Location 'E:\Develop\Projects\dollama'
git rev-parse HEAD
Write-Output "--- porcelain start ---"
git status --porcelain
Write-Output "--- porcelain end ---"
Write-Output ""
Write-Output "=== reserve shortage sweep over every collected log ==="
$hit = Select-String -Path (Join-Path $OUT '*.log') -Pattern 'reserve shortage' -SimpleMatch
if ($null -eq $hit) { Write-Output "  0 hits across all logs" } else { $hit | ForEach-Object { Write-Output ("  " + $_.Filename + ": " + $_.Line) } }
Write-Output "=== end ==="

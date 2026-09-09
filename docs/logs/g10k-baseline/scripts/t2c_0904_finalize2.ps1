$ErrorActionPreference = 'Continue'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OUT = 'E:\Develop\logs\g10k-t2c'
$NSD = Join-Path $OUT 'nsys'
Write-Output "=== large artifacts kept on the research machine (NOT collected into the repo) ==="
Get-ChildItem $NSD -File | Where-Object { $_.Extension -in '.sqlite' -or $_.Name -like '*.nsys-rep' } |
    Sort-Object Name | ForEach-Object {
        Write-Output ("  {0}   {1} bytes   mtime={2}" -f $_.FullName, $_.Length, $_.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'))
    }
Write-Output ""
Write-Output "=== reserve shortage: literal-hit sweep (excluding the grep header line) ==="
$hit = Select-String -Path (Join-Path $OUT '*.log') -Pattern 'reserve shortage:' -SimpleMatch
if ($null -eq $hit) { Write-Output "  0 hits ('[ALLOC] reserve shortage:' never printed)" }
else { $hit | ForEach-Object { Write-Output ("  " + $_.Filename + ": " + $_.Line) } }
Write-Output ""
Write-Output "=== [ALLOC] reserve lines actually present ==="
Select-String -Path (Join-Path $OUT '*.log') -Pattern '\[ALLOC\] reserve' | ForEach-Object { Write-Output ("  " + $_.Filename + ": " + $_.Line) }
Write-Output ""
Write-Output "=== stderr check: any exception / SAC block / WinError ==="
$e = Select-String -Path (Join-Path $OUT '*.log') -Pattern 'WinError 4551|Smart App|exception|FAIL:|0xC0000' 
if ($null -eq $e) { Write-Output "  0 hits" } else { $e | ForEach-Object { Write-Output ("  " + $_.Filename + ": " + $_.Line) } }
Write-Output "=== end ==="

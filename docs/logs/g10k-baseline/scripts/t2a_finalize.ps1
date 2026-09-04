# T2a: 基準値の集約 + 回収前の sha256 採取
$ErrorActionPreference = 'Continue'
$out = 'E:\Develop\logs\g10k-t2a'
$bin1 = Join-Path $out 'bin_run1'
$bin2 = Join-Path $out 'bin_run2'

# --- bin_run2 は決定性確認用の使い捨て。基準ブロブは bin_run1 のみ残す ---
if (Test-Path $bin2) { Remove-Item $bin2 -Recurse -Force }

$L = @()
$L += '=== T2a 基準 sha256 まとめ (研究機 KIK-WIN-RTX58) ==='
$L += ('採取日時: ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss K'))
$L += ''
$L += '--- [基準A] e2e 生成物の内容ダイジェスト (prof_arena_e2e / generate_txt2img) ---'
$L += '採取条件: exe=build/src/prof_arena_e2e.exe'
$L += '          env  PROF_IMAGES=2 PROF_STEPS=20 PROF_G=7.5 PROF_FAST=1 PROF_SAMPLE_MS=5'
$L += '               DOLLAMA_PROFILE=(未設定) DOLLAMA_POOL=(未設定=既定 ON)'
$L += '          seed 1234 (prof_arena_e2e.cu:233)  解像度 1024x1024  steps 20  guidance 7.5'
$L += '          構成 PROF_FAST=1 -> FastConfig{attn_fast,batch2,epilogue}=true (prof_arena_e2e.cu:212-215)'
$L += '★ 注意: prof_arena_e2e は画像ファイルを書き出さない。下の値は RGB バッファ全体の'
$L += '   FNV-1a 64bit (prof_arena_e2e.cu:85 fnv1a / :247 出力) であって sha256 ではない。'
foreach ($f in @('e2e_run1.log','e2e_run2.log')) {
    $p = Join-Path $out $f
    Select-String -Path $p -Pattern 'rgb_hash' | ForEach-Object {
        $L += ('  {0}: {1}' -f $f, $_.Line.Trim())
    }
}
$L += ''
$L += '--- [基準B] 生成物ファイルのバイト列 (test_conv2d / launch_conv2d 出力 .bin) ---'
$L += '採取条件: exe=build/src/test_conv2d.exe'
$L += '          env  DOLLAMA_G8K_DUMP=<dir>\conv  DOLLAMA_PROFILE=(未設定) DOLLAMA_POOL=(未設定)'
$L += '          seed 形状ごとにハードコード 3101..3107 (test_conv2d.cu:696-703)'
$L += ('          基準ブロブの保管先 (研究機): ' + $bin1)
Get-ChildItem $bin1 -Filter *.bin | Sort-Object Name | ForEach-Object {
    $h = (Get-FileHash -Algorithm SHA256 $_.FullName).Hash
    $L += ('  {0,-24} {1,12} bytes  sha256={2}' -f $_.Name, $_.Length, $h)
}
$L += ''
$L += '--- 2 回走行の一致 (決定性の確認。無改変の証明ではない) ---'
$L += '  基準A: e2e_run1 / e2e_run2 の 計 4 画像すべて rgb_hash 一致 (上記)'
$L += '  基準B: conv2d_run1 / conv2d_run2 の 7 ファイルすべて sha256 一致'
$L += '         (run2 のブロブは確認後に削除済み。ログ conv2d_run2.log に値が残っている)'

$sum = Join-Path $out 'baseline_sha256.txt'
$L | Out-File -FilePath $sum -Encoding utf8

Write-Output '=== 回収対象ファイルの sha256 (研究機側) ==='
Get-ChildItem $out -File | Sort-Object Name | ForEach-Object {
    $h = (Get-FileHash -Algorithm SHA256 $_.FullName).Hash
    Write-Output ('{0}  {1}  {2}' -f $h, $_.Length, $_.Name)
}

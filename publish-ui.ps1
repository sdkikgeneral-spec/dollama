# dollama UI (Blazor Server) 配布スクリプト (run-ui.ps1 と対)
#
# 使い方:
#   .\publish-ui.ps1                          # 既定: 自己完結 (win-x64・単一ファイル)。.NET 不要の exe を作る
#   .\publish-ui.ps1 -FrameworkDependent      # フレームワーク依存 (実行先に .NET 10 ランタイムが要る・軽量)
#   .\publish-ui.ps1 -Runtime linux-x64       # 別 RID 向け (自己完結は RID 必須)
#   .\publish-ui.ps1 -Output C:\dist\dollama-ui  # 出力先を指定
#
# 注: これは Blazor *Server* なので、publish 物は「Web サーバー (Kestrel) を内蔵した常駐 exe」。
#     ブラウザはこの exe にだけ繋ぎ、C++ 生成サーバー (dollama --http) は別プロセスのまま。
#     C++ サーバー URL は環境変数 Dollama__BaseUrl で上書きする。

[CmdletBinding()]
param(
    # 自己完結 (.NET ランタイム同梱・単一ファイル化)。既定 true。
    [switch]$SelfContained = $true,
    # フレームワーク依存に切り替える (-SelfContained を打ち消す)。実行先に .NET 10 が必要。
    [switch]$FrameworkDependent,
    # ターゲット RID。自己完結時は必須 (既定 win-x64)。
    [string]$Runtime = "win-x64",
    # 出力先フォルダ。未指定なら ui/bin/Release/net10.0/<rid>/publish。
    [string]$Output = ""
)

$ErrorActionPreference = "Stop"

# このスクリプトの場所を基準に ui/ を解決 (どこから呼んでも動く)
$uiDir = Join-Path $PSScriptRoot "ui"
$csproj = Join-Path $uiDir "Dollama.Ui.csproj"
if (-not (Test-Path $csproj))
{
    Write-Error "ui/Dollama.Ui.csproj が見つかりません ($uiDir)。リポジトリルートに置いてください。"
}

# dotnet の存在確認
if (-not (Get-Command dotnet -ErrorAction SilentlyContinue))
{
    Write-Error "dotnet が見つかりません。.NET 10 SDK をインストールしてください。"
}

# -FrameworkDependent が指定されたら自己完結を打ち消す
$selfContained = $SelfContained -and (-not $FrameworkDependent)

# 出力先の既定 (RID を含めて run 物と衝突しないパス)
if ($Output -eq "")
{
    $Output = Join-Path $uiDir "bin\Release\net10.0\$Runtime\publish"
}

# dotnet publish の引数を組み立てる
$args = @(
    "publish", $csproj,
    "-c", "Release",
    "-r", $Runtime,
    "-o", $Output
)
if ($selfContained)
{
    # 自己完結 + 単一ファイル (.NET ランタイム同梱・配布が exe 1 つで済む)
    $args += "--self-contained", "true"
    $args += "-p:PublishSingleFile=true"
}
else
{
    # フレームワーク依存 (軽量・実行先に .NET 10 ランタイムが要る)
    $args += "--self-contained", "false"
}

$mode = if ($selfContained) { "自己完結 (単一ファイル)" } else { "フレームワーク依存" }
Write-Host "[publish-ui] 構成     : Release" -ForegroundColor Cyan
Write-Host "[publish-ui] RID      : $Runtime" -ForegroundColor Cyan
Write-Host "[publish-ui] モード   : $mode" -ForegroundColor Cyan
Write-Host "[publish-ui] 出力先   : $Output" -ForegroundColor Cyan
Write-Host "[publish-ui] dotnet $($args -join ' ')" -ForegroundColor DarkGray

# 発行を実行
dotnet @args

# 完了案内
$exe = Join-Path $Output "Dollama.Ui.exe"
Write-Host ""
Write-Host "[publish-ui] 発行が完了しました。" -ForegroundColor Green
Write-Host "  出力フォルダ : $Output" -ForegroundColor Green
Write-Host ""
Write-Host "起動手順:" -ForegroundColor Cyan
Write-Host "  1. $exe を起動 (Blazor Server = Web サーバー内蔵の常駐 exe)" -ForegroundColor Gray
Write-Host "  2. ブラウザで http://localhost:5074 を開く" -ForegroundColor Gray
Write-Host "  3. C++ 生成サーバー (dollama --http) は別プロセス。URL は環境変数で上書き:" -ForegroundColor Gray
Write-Host "       `$env:Dollama__BaseUrl = 'http://127.0.0.1:8080'   # C++ サーバー URL" -ForegroundColor Gray
Write-Host "       `$env:ASPNETCORE_URLS  = 'http://0.0.0.0:5074'     # UI 待受 (LAN 公開時)" -ForegroundColor Gray
Write-Host "     未起動でも UI は開く (接続灯が赤・テレメトリは動く)。" -ForegroundColor Gray

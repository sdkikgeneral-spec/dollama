# dollama UI (Blazor Server) 起動スクリプト
#
# 使い方:
#   .\run-ui.ps1                       # 既定で起動 (UI: http://localhost:5074, C++: http://127.0.0.1:8080)
#   .\run-ui.ps1 -Urls http://0.0.0.0:5074   # 別 PC のブラウザから見たいとき (LAN 公開)
#   .\run-ui.ps1 -BaseUrl http://127.0.0.1:9000  # C++ 生成サーバーのポートを変えたとき
#   .\run-ui.ps1 -Release              # Release ビルドで起動
#
# 注: C++ 生成サーバー (dollama --http) は別プロセス。未起動でも UI は開くが接続灯は赤。

[CmdletBinding()]
param(
    # UI (Blazor) の待受アドレス
    [string]$Urls = "http://localhost:5074",
    # 叩きに行く C++ 生成サーバーの URL (appsettings の Dollama:BaseUrl を上書き)
    [string]$BaseUrl = "",
    # Release 構成で起動する
    [switch]$Release,
    # 起動後にブラウザを自動で開かない
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"

# このスクリプトの場所を基準に ui/ を解決 (どこから呼んでも動く)
$uiDir = Join-Path $PSScriptRoot "ui"
if (-not (Test-Path (Join-Path $uiDir "Dollama.Ui.csproj")))
{
    Write-Error "ui/Dollama.Ui.csproj が見つかりません ($uiDir)。リポジトリルートに置いてください。"
}

# dotnet の存在確認
if (-not (Get-Command dotnet -ErrorAction SilentlyContinue))
{
    Write-Error "dotnet が見つかりません。.NET 10 SDK をインストールしてください。"
}

$config = if ($Release) { "Release" } else { "Debug" }

# 環境変数で UI の待受 URL と (指定時) C++ サーバー URL を渡す
$env:ASPNETCORE_URLS = $Urls
if ($BaseUrl -ne "")
{
    $env:Dollama__BaseUrl = $BaseUrl
}

# Development を明示する。--no-launch-profile だと未設定→既定 Production になり、
# 未 publish では静的 Web アセット (blazor.web.js / スコープ CSS) が解決できず
# FileNotFoundException でページが壊れる。Development なら build 出力から自動で有効。
if (-not $env:ASPNETCORE_ENVIRONMENT)
{
    $env:ASPNETCORE_ENVIRONMENT = "Development"
}

Write-Host "[run-ui] UI       : $Urls" -ForegroundColor Cyan
$shownBase = if ($BaseUrl -ne "") { $BaseUrl } else { "(appsettings 既定: http://127.0.0.1:8080)" }
Write-Host "[run-ui] C++ API  : $shownBase" -ForegroundColor Cyan
Write-Host "[run-ui] config   : $config" -ForegroundColor Cyan
Write-Host "[run-ui] 停止は Ctrl+C" -ForegroundColor DarkGray

# 起動後にブラウザを自動で開く (dotnet run はブロックするので、
# サーバーが応答するまで待ってから開く別ジョブを先に仕掛けておく)。
if (-not $NoBrowser)
{
    # 0.0.0.0 / + などの待受専用ホストはブラウザでは localhost に読み替える
    $openUrl = ($Urls -split ';')[0] -replace '://(0\.0\.0\.0|\+|\[::\])', '://localhost'
    Start-Job -ScriptBlock {
        param($u)
        for ($i = 0; $i -lt 60; $i++)
        {
            try
            {
                $r = Invoke-WebRequest -Uri $u -UseBasicParsing -TimeoutSec 2
                if ($r.StatusCode -eq 200) { Start-Process $u; break }
            }
            catch { Start-Sleep -Milliseconds 500 }
        }
    } -ArgumentList $openUrl | Out-Null
    Write-Host "[run-ui] 起動後 $openUrl をブラウザで開きます (-NoBrowser で無効化)" -ForegroundColor DarkGray
}

# launchSettings の applicationUrl より ASPNETCORE_URLS を優先させるため --no-launch-profile
dotnet run --project $uiDir -c $config --no-launch-profile

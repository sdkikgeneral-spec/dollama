<#
.SYNOPSIS
    dollama Windows 用ライブラリインストーラー (ブートストラップ)。

.DESCRIPTION
    まっさらな Windows 機で、dollama のビルド/実行に必要なライブラリ一式を
    winget + pip で一括導入する。最後に存在チェックと推奨 meson setup を提示する。

    - Windows 限定。winget + pip を使う。
    - CUDA は任意: NVIDIA dGPU 検出時のみ CUDA Toolkit を導入。
      無ければスキップし torch=CPU・meson は -Dwith_cuda=false を案内する。
    - 冪等: 導入済みはスキップ、pip 再実行は安全、失敗しても続行し最後に集計する。

.PARAMETER SkipCuda
    NVIDIA があっても CUDA Toolkit を導入しない (torch=CPU・-Dwith_cuda=false 案内)。

.PARAMETER SkipSdk
    重量級 SDK (VS Build Tools / CUDA / OpenVINO) を入れず、Python + Meson のみ導入する。

.PARAMETER PythonExe
    使用する Python 実行ファイルのパス。未指定なら py -3.12 を使う。

.PARAMETER DryRun
    実際には実行せず、組み立てた全コマンドを表示する (副作用ゼロ)。

.PARAMETER CheckOnly
    導入は行わず、存在チェックとサマリ・推奨 meson コマンドのみ出力する。

.PARAMETER Force
    requirements.txt が存在しても再生成する。

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File install_windows.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File install_windows.ps1 -DryRun
#>

[CmdletBinding()]
param(
    [switch]$SkipCuda,
    [switch]$SkipSdk,
    [string]$PythonExe,
    [switch]$DryRun,
    [switch]$CheckOnly,
    [switch]$Force
)

# --------------------------------------------------------------------------
# 共通設定・ロギング
# --------------------------------------------------------------------------
$ErrorActionPreference = 'Stop'   # 既定は厳格。個別コマンドは try/catch で握って続行する。

$Script:RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Script:Failures = New-Object System.Collections.Generic.List[string]

function Write-Info  ([string]$m) { Write-Host "[情報] $m"  -ForegroundColor Cyan }
function Write-Ok    ([string]$m) { Write-Host "[OK]   $m"   -ForegroundColor Green }
function Write-Warn  ([string]$m) { Write-Host "[警告] $m"   -ForegroundColor Yellow }
function Write-Err   ([string]$m) { Write-Host "[エラー] $m" -ForegroundColor Red }
function Write-Step  ([string]$m) { Write-Host ""; Write-Host "==== $m ====" -ForegroundColor Magenta }

function Add-Failure ([string]$m)
{
    $Script:Failures.Add($m)
    Write-Err $m
}

# コマンドを実行 (DryRun のときは表示のみ)。失敗しても例外を投げず $false を返す。
function Invoke-Action
{
    param(
        [string]$Label,
        [scriptblock]$Action,
        [string[]]$DisplayCmd
    )

    if ($DryRun)
    {
        Write-Host "  [dry-run] $Label" -ForegroundColor DarkGray
        if ($DisplayCmd)
        {
            foreach ($line in $DisplayCmd)
            {
                Write-Host "    > $line" -ForegroundColor DarkGray
            }
        }
        return $true
    }

    Write-Info $Label
    try
    {
        & $Action
        return $true
    }
    catch
    {
        Add-Failure "$Label : $($_.Exception.Message)"
        return $false
    }
}

# コマンドの存在確認
function Test-CommandExists ([string]$name)
{
    return [bool](Get-Command $name -ErrorAction SilentlyContinue)
}

# --------------------------------------------------------------------------
# 1. 前提ガード
# --------------------------------------------------------------------------
function Test-Prerequisites
{
    Write-Step "前提チェック"

    # Windows 判定
    $isWin = $true
    if (Test-Path variable:IsWindows)
    {
        # PowerShell 6+ では自動変数 $IsWindows が存在する
        $isWin = $IsWindows
    }
    if (-not $isWin)
    {
        Add-Failure "このインストーラーは Windows 専用です。"
        return $false
    }
    Write-Ok "OS: Windows"

    # PowerShell バージョン
    $psv = $PSVersionTable.PSVersion
    Write-Ok "PowerShell バージョン: $psv"
    if ($psv.Major -lt 5)
    {
        Add-Failure "PowerShell 5.0 以上が必要です (現在 $psv)。"
        return $false
    }

    # winget 有無
    if (-not (Test-CommandExists 'winget'))
    {
        Write-Err "winget が見つかりません。"
        Write-Warn "Microsoft Store から 'アプリ インストーラー (App Installer)' を導入してください:"
        Write-Warn "  https://apps.microsoft.com/detail/9nblggh4nns1"
        Add-Failure "winget 未導入のため中断。"
        return $false
    }
    $wingetVer = (& winget --version) 2>$null
    Write-Ok "winget: $wingetVer"

    # 管理者権限の確認 (machine-scope の重量級 SDK 用)
    $isAdmin = $false
    try
    {
        $id = [System.Security.Principal.WindowsIdentity]::GetCurrent()
        $pr = New-Object System.Security.Principal.WindowsPrincipal($id)
        $isAdmin = $pr.IsInRole([System.Security.Principal.WindowsBuiltInRole]::Administrator)
    }
    catch { $isAdmin = $false }

    $Script:IsAdmin = $isAdmin
    if ($isAdmin)
    {
        Write-Ok "管理者権限: あり (machine-scope インストール可)"
    }
    else
    {
        Write-Warn "管理者権限: なし。可能な範囲で --scope user を優先します。"
        Write-Warn "VS Build Tools / CUDA / OpenVINO は machine-scope が要る場合があります。"
        Write-Warn "失敗したら管理者 PowerShell で本スクリプトを再実行してください。"
    }

    return $true
}

# --------------------------------------------------------------------------
# 2. GPU 検出
# --------------------------------------------------------------------------
function Get-HasNvidia
{
    Write-Step "GPU 検出"
    $hasNvidia = $false
    try
    {
        $vcs = Get-CimInstance Win32_VideoController -ErrorAction Stop
        foreach ($vc in $vcs)
        {
            Write-Info "ビデオコントローラー: $($vc.Name)"
            if ($vc.Name -match 'NVIDIA')
            {
                $hasNvidia = $true
            }
        }
    }
    catch
    {
        Write-Warn "GPU 情報の取得に失敗: $($_.Exception.Message)"
    }

    if ($hasNvidia)
    {
        Write-Ok "NVIDIA GPU を検出 (CUDA 候補)"
    }
    else
    {
        Write-Warn "NVIDIA GPU 未検出 (CUDA はスキップ・torch は CPU 版)"
    }
    return $hasNvidia
}

# --------------------------------------------------------------------------
# winget インストールヘルパー (冪等)
# --------------------------------------------------------------------------
function Test-WingetInstalled ([string]$id)
{
    if ($DryRun) { return $false }   # dry-run では「未導入」前提で組み立て表示
    try
    {
        $out = & winget list --id $id --exact --accept-source-agreements 2>$null
        return ($LASTEXITCODE -eq 0 -and ($out -match [regex]::Escape($id)))
    }
    catch { return $false }
}

function Install-WingetPackage
{
    param(
        [string]$Id,
        [string]$Label,
        [string[]]$ExtraArgs
    )

    if (Test-WingetInstalled $Id)
    {
        Write-Ok "$Label は導入済み (winget id: $Id) — スキップ"
        return $true
    }

    $args = @(
        'install', '--id', $Id, '--exact',
        '--accept-package-agreements', '--accept-source-agreements'
    )
    # 管理者でなければ user スコープを優先 (対応していないパッケージは winget 側で無視/失敗)
    if (-not $Script:IsAdmin)
    {
        $args += @('--scope', 'user')
    }
    if ($ExtraArgs)
    {
        $args += $ExtraArgs
    }

    $display = "winget $($args -join ' ')"
    return Invoke-Action -Label "$Label を導入 (winget)" -DisplayCmd @($display) -Action {
        & winget @args
        if ($LASTEXITCODE -ne 0)
        {
            throw "winget が終了コード $LASTEXITCODE を返しました ($Id)"
        }
    }
}

# --------------------------------------------------------------------------
# 3. 重量級 SDK
# --------------------------------------------------------------------------
function Install-BuildTools
{
    # VS Build Tools 2022 + C++ workload
    # winget id は `winget search` で確認済み: Microsoft.VisualStudio.2022.BuildTools
    $override = '--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended'
    Install-WingetPackage `
        -Id 'Microsoft.VisualStudio.2022.BuildTools' `
        -Label 'Visual Studio 2022 Build Tools (C++ ワークロード)' `
        -ExtraArgs @('--override', $override) | Out-Null
}

function Install-Cuda
{
    # CUDA Toolkit。sm_120 (Blackwell/RTX5080) は 12.8 以上が必須。
    # winget id は `winget search CUDA` で確認済み: Nvidia.CUDA (最新は 12.8+ を満たす)
    Install-WingetPackage `
        -Id 'Nvidia.CUDA' `
        -Label 'NVIDIA CUDA Toolkit' | Out-Null
}

function Install-OpenVino
{
    # OpenVINO C++ SDK。
    # winget search で id を確認済み: Intel.OpenVINOToolkit.2026.2.0 (winget 配布あり)
    # winget に id があるため公式アーカイブ DL フォールバックは使わない。
    $ovId = 'Intel.OpenVINOToolkit.2026.2.0'
    Install-WingetPackage -Id $ovId -Label 'Intel OpenVINO Toolkit (C++ SDK)' | Out-Null
}

function Install-HeavySdks
{
    Write-Step "重量級 SDK (VS Build Tools / CUDA / OpenVINO)"

    if ($SkipSdk)
    {
        Write-Warn "-SkipSdk 指定: 重量級 SDK をすべてスキップします。"
        return
    }

    Install-BuildTools

    if ($Script:HasNvidia -and -not $SkipCuda)
    {
        Install-Cuda
    }
    elseif ($SkipCuda)
    {
        Write-Warn "-SkipCuda 指定: CUDA Toolkit をスキップ。"
    }
    else
    {
        Write-Warn "NVIDIA 未検出: CUDA Toolkit をスキップ。"
    }

    Install-OpenVino
}

# --------------------------------------------------------------------------
# 4. Python + pip
# --------------------------------------------------------------------------

# 使用する Python 起動コマンドを配列で返す (例: @('py','-3.12') または @('C:\...\python.exe'))
function Resolve-PythonLauncher
{
    if ($PythonExe)
    {
        if (-not $DryRun -and -not (Test-Path $PythonExe))
        {
            Write-Warn "-PythonExe のパスが見つかりません: $PythonExe"
        }
        return ,@($PythonExe)
    }

    # py -3.12 を優先
    if (Test-CommandExists 'py')
    {
        if ($DryRun)
        {
            return ,@('py', '-3.12')
        }
        $ok = $false
        try
        {
            & py -3.12 --version 1>$null 2>$null
            $ok = ($LASTEXITCODE -eq 0)
        }
        catch { $ok = $false }
        if ($ok)
        {
            return ,@('py', '-3.12')
        }
        Write-Warn "py -3.12 が利用できません。Python 3.12 の導入を試みます。"
    }
    else
    {
        Write-Warn "py ランチャーが見つかりません。Python 3.12 の導入を試みます。"
    }

    # Python 3.12 を winget で導入 (CheckOnly では導入しない)
    if (-not $CheckOnly)
    {
        Install-WingetPackage -Id 'Python.Python.3.12' -Label 'Python 3.12' | Out-Null
    }
    return ,@('py', '-3.12')
}

function Ensure-Requirements
{
    $reqPath = Join-Path $Script:RootDir 'requirements.txt'
    if ((Test-Path $reqPath) -and -not $Force)
    {
        Write-Ok "requirements.txt は既存 — そのまま使用 (再生成は -Force)"
        return $reqPath
    }

    if ($DryRun)
    {
        Write-Host "  [dry-run] requirements.txt を生成 ($reqPath)" -ForegroundColor DarkGray
        return $reqPath
    }

    Write-Info "requirements.txt を生成: $reqPath"
    $content = @'
# dollama Python 依存 (Windows / pip)
# torch は GPU 有無で index が変わるため install_windows.ps1 が別途導入する。
#   NVIDIA 有: pip install torch --index-url https://download.pytorch.org/whl/cu128
#   NVIDIA 無: pip install torch   (CPU 版・既定 index)
numpy
safetensors
certifi
pillow
openvino
openvino-tokenizers
transformers
diffusers
accelerate
optimum[openvino]
onnxruntime
huggingface_hub
meson
ninja
'@
    Set-Content -Path $reqPath -Value $content -Encoding UTF8
    return $reqPath
}

function Install-PythonDeps
{
    Write-Step "Python + pip 依存"

    $py = Resolve-PythonLauncher
    $Script:PyLauncher = $py
    $pyDisplay = ($py -join ' ')
    Write-Info "使用する Python: $pyDisplay"

    $reqPath = Ensure-Requirements

    # pip 自身を更新
    Invoke-Action -Label "pip を更新" `
        -DisplayCmd @("$pyDisplay -m pip install -U pip") `
        -Action {
            & $py[0] @($py[1..($py.Length-1)] + @('-m','pip','install','-U','pip'))
            if ($LASTEXITCODE -ne 0) { throw "pip 更新失敗 (exit $LASTEXITCODE)" }
        } | Out-Null

    # torch を index 分岐で別行導入
    if ($Script:HasNvidia -and -not $SkipCuda)
    {
        $torchIdx = 'https://download.pytorch.org/whl/cu128'
        Invoke-Action -Label "torch (CUDA cu128) を導入" `
            -DisplayCmd @("$pyDisplay -m pip install -U torch --index-url $torchIdx") `
            -Action {
                & $py[0] @($py[1..($py.Length-1)] + @('-m','pip','install','-U','torch','--index-url',$torchIdx))
                if ($LASTEXITCODE -ne 0) { throw "torch(cu128) 導入失敗 (exit $LASTEXITCODE)" }
            } | Out-Null
    }
    else
    {
        Invoke-Action -Label "torch (CPU 版) を導入" `
            -DisplayCmd @("$pyDisplay -m pip install -U torch") `
            -Action {
                & $py[0] @($py[1..($py.Length-1)] + @('-m','pip','install','-U','torch'))
                if ($LASTEXITCODE -ne 0) { throw "torch(CPU) 導入失敗 (exit $LASTEXITCODE)" }
            } | Out-Null
    }

    # 残りを requirements.txt から導入
    Invoke-Action -Label "requirements.txt の依存を導入" `
        -DisplayCmd @("$pyDisplay -m pip install -U -r `"$reqPath`"") `
        -Action {
            & $py[0] @($py[1..($py.Length-1)] + @('-m','pip','install','-U','-r',$reqPath))
            if ($LASTEXITCODE -ne 0) { throw "requirements 導入失敗 (exit $LASTEXITCODE)" }
        } | Out-Null
}

# --------------------------------------------------------------------------
# 5. 存在チェック + サマリ + 推奨 meson setup
# --------------------------------------------------------------------------

# CUDA Toolkit のインストールパスを探す
function Find-CudaRoot
{
    if (Test-CommandExists 'nvcc')
    {
        $nvccPath = (Get-Command nvcc).Source
        # ...\CUDA\vXX.Y\bin\nvcc.exe -> ...\CUDA\vXX.Y
        $root = Split-Path -Parent (Split-Path -Parent $nvccPath)
        return $root
    }
    $base = "$env:ProgramFiles\NVIDIA GPU Computing Toolkit\CUDA"
    if (Test-Path $base)
    {
        $versions = Get-ChildItem $base -Directory -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending
        if ($versions -and $versions.Count -gt 0)
        {
            return $versions[0].FullName
        }
    }
    return ''
}

# OpenVINO ランタイムのルートを探す
function Find-OpenVinoRoot
{
    if ($env:OpenVINO_DIR)
    {
        return $env:OpenVINO_DIR
    }
    $bases = @(
        "${env:ProgramFiles(x86)}\Intel",
        "$env:ProgramFiles\Intel"
    )
    foreach ($b in $bases)
    {
        if (Test-Path $b)
        {
            $cand = Get-ChildItem $b -Directory -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -match 'openvino' } |
                Sort-Object Name -Descending
            if ($cand -and $cand.Count -gt 0)
            {
                return $cand[0].FullName
            }
        }
    }
    return ''
}

function Invoke-CheckSummary
{
    Write-Step "存在チェック + サマリ"

    $py = $Script:PyLauncher
    if (-not $py)
    {
        $py = Resolve-PythonLauncher
    }
    $pyDisplay = ($py -join ' ')

    $rows = New-Object System.Collections.Generic.List[object]
    function Add-Row ([string]$name, [bool]$ok, [string]$detail)
    {
        $rows.Add([pscustomobject]@{ 項目 = $name; 状態 = $(if ($ok) { '[OK]' } else { '[NG]' }); 詳細 = $detail })
    }

    # Python
    $pyVer = ''
    $pyOk = $false
    try
    {
        $pyVer = (& $py[0] @($py[1..($py.Length-1)] + @('--version')) 2>&1 | Out-String).Trim()
        $pyOk = ($LASTEXITCODE -eq 0)
    }
    catch { $pyOk = $false }
    Add-Row 'Python 3.12' $pyOk $pyVer

    # meson
    $mesonOk = Test-CommandExists 'meson'
    $mesonVer = ''
    if ($mesonOk) { try { $mesonVer = (& meson --version 2>&1 | Out-String).Trim() } catch {} }
    Add-Row 'meson' $mesonOk $mesonVer

    # ninja
    $ninjaOk = Test-CommandExists 'ninja'
    $ninjaVer = ''
    if ($ninjaOk) { try { $ninjaVer = (& ninja --version 2>&1 | Out-String).Trim() } catch {} }
    Add-Row 'ninja' $ninjaOk $ninjaVer

    # cl.exe (MSVC)
    $clOk = Test-CommandExists 'cl'
    Add-Row 'MSVC (cl.exe)' $clOk $(if ($clOk) { (Get-Command cl).Source } else { 'PATH 上に無し (VS Dev Prompt で利用可)' })

    # nvcc (NVIDIA & not SkipCuda のとき意味を持つ)
    $cudaRoot = Find-CudaRoot
    if ($Script:HasNvidia -and -not $SkipCuda)
    {
        $nvccOk = Test-CommandExists 'nvcc'
        $nvccVer = ''
        if ($nvccOk) { try { $nvccVer = ((& nvcc --version 2>&1 | Out-String) -split "`n" | Where-Object { $_ -match 'release' } | Select-Object -First 1).Trim() } catch {} }
        Add-Row 'nvcc (CUDA)' $nvccOk $(if ($nvccVer) { $nvccVer } else { $cudaRoot })
    }
    else
    {
        Add-Row 'nvcc (CUDA)' $true 'スキップ (NVIDIA 無し or -SkipCuda)'
    }

    # OpenVINO ルート
    $ovRoot = Find-OpenVinoRoot
    Add-Row 'OpenVINO ルート' ([bool]$ovRoot) $(if ($ovRoot) { $ovRoot } else { '未検出' })

    # import openvino,torch + torch.cuda.is_available()
    $impOk = $false
    $impDetail = ''
    if ($pyOk)
    {
        try
        {
            $impDetail = (& $py[0] @($py[1..($py.Length-1)] + @('-c','import openvino,torch;print("ov",openvino.__version__,"torch",torch.__version__,"cuda",torch.cuda.is_available())')) 2>&1 | Out-String).Trim()
            $impOk = ($LASTEXITCODE -eq 0)
        }
        catch { $impOk = $false; $impDetail = $_.Exception.Message }
    }
    Add-Row 'import openvino,torch' $impOk $impDetail

    $rows | Format-Table -AutoSize | Out-String | Write-Host

    # 推奨 meson setup の組み立て
    Write-Step "推奨 meson setup コマンド"

    $opts = New-Object System.Collections.Generic.List[string]
    if ($Script:HasNvidia -and -not $SkipCuda)
    {
        if ($cudaRoot)
        {
            $cudaFwd = ($cudaRoot -replace '\\','/')
            $opts.Add("-Dgpu_sdk_root='$cudaFwd'")
        }
    }
    else
    {
        $opts.Add('-Dwith_cuda=false')
    }

    if ($ovRoot)
    {
        $ovFwd = ($ovRoot -replace '\\','/')
        $opts.Add("-Dnpu_sdk_root='$ovFwd'")
    }
    else
    {
        $opts.Add('-Dwith_openvino=false  # OpenVINO 未検出時。SDK 導入後は外す')
    }

    $cmd = "meson setup build " + ($opts -join ' ')
    Write-Host ""
    Write-Host "  $cmd" -ForegroundColor Green
    Write-Host "  meson compile -C build" -ForegroundColor Green
    Write-Host ""
    Write-Warn "MSVC を使う場合は 'x64 Native Tools Command Prompt for VS 2022' か"
    Write-Warn "VS Dev PowerShell から meson を実行してください (cl.exe が PATH に入ります)。"
}

# --------------------------------------------------------------------------
# メイン
# --------------------------------------------------------------------------
function Main
{
    Write-Host ""
    Write-Host "########################################" -ForegroundColor Magenta
    Write-Host "#  dollama Windows インストーラー       #" -ForegroundColor Magenta
    Write-Host "########################################" -ForegroundColor Magenta
    if ($DryRun)    { Write-Warn "DryRun モード: 実行せずコマンドを表示のみ (副作用ゼロ)" }
    if ($CheckOnly) { Write-Warn "CheckOnly モード: 存在チェックのみ" }

    $Script:IsAdmin = $false
    $Script:HasNvidia = $false
    $Script:PyLauncher = $null

    if (-not (Test-Prerequisites))
    {
        Write-Err "前提チェックに失敗しました。中断します。"
        return 1
    }

    $Script:HasNvidia = Get-HasNvidia

    if ($CheckOnly)
    {
        Invoke-CheckSummary
        return 0
    }

    Install-HeavySdks
    Install-PythonDeps
    Invoke-CheckSummary

    # 失敗集計 (all-or-nothing にしない)
    Write-Step "結果サマリ"
    if ($Script:Failures.Count -eq 0)
    {
        Write-Ok "すべての処理が完了しました。"
        return 0
    }
    else
    {
        Write-Warn "$($Script:Failures.Count) 件の失敗がありました:"
        foreach ($f in $Script:Failures)
        {
            Write-Host "  - $f" -ForegroundColor Red
        }
        Write-Warn "上記を解消のうえ再実行してください (本スクリプトは冪等です)。"
        return 2
    }
}

exit (Main)

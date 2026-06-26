# dollama UI (Blazor Server)

dollama の C++ 生成サーバー (OpenAI Images 互換 HTTP API) を操作するための Web UI。
**研究コアではなく「配管」**なので、C++ の単一バイナリ思想とは切り離し、
**別プロセス・別デプロイの Blazor Server (.NET 10)** として実装している。

```
ブラウザ ──▶ [Blazor Server (.NET) このプロジェクト] ──HTTP──▶ [C++ 生成サーバー dollama --http]
```

ブラウザは Blazor サーバーとだけ通信し、C++ サーバーを叩くのは .NET 側 (サーバー間通信)。
そのため **CORS は不要**で、**C++ 側は一切改修していない**。

## 前提

- .NET 10 SDK (`dotnet --list-sdks` で `10.x` があること)
- (本生成を見るなら) C++ 生成サーバーを別途起動。未起動でも UI は開く (接続灯が赤・テレメトリは動く)

## 起動

リポジトリルートの起動スクリプトを使うのが簡単 ([run-ui.ps1](../run-ui.ps1)):

```powershell
.\run-ui.ps1                               # 既定 (UI: http://localhost:5074, C++: http://127.0.0.1:8080)
.\run-ui.ps1 -Urls http://0.0.0.0:5074     # LAN の別 PC から見たいとき
.\run-ui.ps1 -BaseUrl http://127.0.0.1:9000  # C++ サーバーのポートを変えたとき
.\run-ui.ps1 -Release                      # Release ビルドで起動
.\run-ui.ps1 -NoBrowser                    # 起動後にブラウザを自動で開かない
```

スクリプトは `ASPNETCORE_ENVIRONMENT=Development` を明示し (静的アセットを有効化)、
サーバーが応答したら既定ブラウザを自動で開く。

直接叩く場合:

```powershell
cd ui
dotnet run
```

C++ 生成サーバー側 (別ターミナル):

```powershell
# 例 (リポジトリの C++ バイナリ)
.\build\dollama.exe --http --port 8080
```

## 配布 (publish)

別 PC へ配るときはリポジトリルートの配布スクリプトを使う ([publish-ui.ps1](../publish-ui.ps1)):

```powershell
.\publish-ui.ps1                          # 既定: 自己完結 (win-x64・単一ファイル)。.NET 不要の exe を作る
.\publish-ui.ps1 -FrameworkDependent      # フレームワーク依存 (実行先に .NET 10 ランタイムが要る・軽量)
.\publish-ui.ps1 -Runtime linux-x64       # 別 RID 向け (自己完結は RID 必須)
.\publish-ui.ps1 -Output C:\dist\dollama-ui  # 出力先を指定
```

`dotnet publish ui/Dollama.Ui.csproj -c Release` をラップする。既定は **自己完結 + 単一ファイル**
(`--self-contained true -p:PublishSingleFile=true`) で、実行先に .NET ランタイムが無くても動く。
完了後に出力フォルダのパスと起動手順を表示する。出力先の既定は
`ui/bin/Release/net10.0/<rid>/publish`。

**配布物の性質と起動:**

- これは Blazor **Server** なので、publish 物は「**Web サーバー (Kestrel) を内蔵した常駐 exe**」。
  ブラウザはこの `Dollama.Ui.exe` にだけ繋ぎ、画面のロジックはサーバー側 (.NET) で動く。
- 起動後、ブラウザで `http://localhost:5074` を開く。
- C++ 生成サーバー (`dollama --http`) は**別プロセス**のまま。URL は環境変数で上書きする:
  - `Dollama__BaseUrl` … C++ サーバー URL (既定 `http://127.0.0.1:8080`)
  - `ASPNETCORE_URLS` … UI の待受アドレス (LAN 公開なら `http://0.0.0.0:5074`)
- C++ サーバーが未起動でも UI は開く (接続灯が赤・テレメトリは動く)。

```powershell
$env:Dollama__BaseUrl = "http://127.0.0.1:8080"
$env:ASPNETCORE_URLS  = "http://0.0.0.0:5074"
.\Dollama.Ui.exe
```

## 機能

| 機能 | 説明 |
|---|---|
| 2 カラム・レイアウト | 左ペイン (タグパレット + プリセット一覧) + 中央 (生成パネル)。テレメは上部バーへ移動 |
| タグパレット | **左ペイン**に常設。①キュレーション済み**カテゴリ木** (静的 JSON・クリックで追加) ②**お気に入り** (編集可・永続)。追加先トグル (◉プロンプト ○ネガティブ) で行き先を切替 |
| お気に入りタグ | ★お気に入りはクリックで追加・× で解除・`+` で入力タグを登録。`favorites.json` に永続化 |
| タグ・チップ入力 | プロンプト/ネガを danbooru タグとして 1 つずつ追加 (Enter / カンマ確定)・× で削除・空 Backspace で末尾削除。重複は無視・小文字寄せ |
| 種別別プリセット | タグ群を **kind (prompt / negative) 別**に名前付き保存・ロード・削除。保存は各入力欄の保存バー、一覧は**左ペインに集約** |
| サムネ付きカード選択 | プリセットを左ペインに**サムネイル付きコンパクトカード**で表示。カードクリックでロード・× で削除。サムネは**直近の生成画像を自動取得**し、保存時に 128px 上限へ縮小して紐付ける (サムネ無しはプレースホルダ) |
| 生成 | チップ群を `", "` 結合して `POST /v1/images/generations` へ。返った base64 PNG を表示 |
| 接続インジケータ | `GET /health` で C++ サーバー接続状態を上部バーに表示 (緑/赤) |
| HW テレメトリ | CPU / NPU / iGPU / RTX5080 の稼働を SignalR で push。**上部バーに横並びミニメーター**で表示 (**現状スタブ値**) |
| タグの日本語表示 + EN⇔日本語トグル | パレット・中央チップのタグを**既定で日本語表示**。上部バーの `[日本語\|EN]` トグルで一括切替 (CascadingValue 伝播)。**内部に保持するタグ値・C++ へ送る prompt は英語の danbooru タグのまま不変**で表示専用。辞書に無いタグは英語フォールバック |

### タグパレットのデータ形式

キュレーション済みのカテゴリ木は **静的 JSON** `ui/wwwroot/tag-palette.json` (prompt-engineer 管轄)。
UI 側は起動時に 1 回読むだけ (内容に依存しない・不在/壊れ JSON でも空で安全動作)。形式:

```json
[
  { "category": "人数", "tags": ["1girl", "1boy", "solo"] },
  { "category": "表情", "tags": ["smile", "blush"] }
]
```

お気に入りは `ui/data/favorites.json` (gitignore 済み・**個人データ**)。形式は文字列配列:

```json
[ "1girl", "long hair", "smile" ]
```

追加時に前後空白除去 + 小文字寄せで正規化し、空・重複は無視。書き込みは一時ファイル経由の
**アトミック置換** (`File.Move(tmp, path, overwrite:true)`) で、並行時にも半端な破損を残さない。

### タグの日本語ラベル辞書

タグの日本語表示は **静的 JSON** `ui/wwwroot/tag-labels.ja.json` (prompt-engineer 管轄) を引く。
形式は `{ 英語タグ: 日本語ラベル }` の辞書:

```json
{
  "long hair": "ロングヘア",
  "smile": "笑顔"
}
```

UI 側 (`TagLabels` サービス) は起動時に 1 回読むだけ (内容に依存しない・不在/壊れ JSON でも
空辞書で安全動作 = 全件英語フォールバック)。`Display(tag, lang)` は `lang=="ja"` かつ辞書ヒット時のみ
日本語ラベルを返し、それ以外 (EN 表示・辞書外・null) は英語タグをそのまま返す。**この写像は表示専用**で、
保持するタグ値・生成 prompt は常に英語の danbooru タグのまま (生成品質に影響しない)。

### プリセットの保存場所

`ui/data/presets.json` (gitignore 済み・個人データ)。形式:

```json
[
  { "name": "base-girl", "kind": "prompt",   "tags": ["1girl", "long hair", "smile"], "thumbnail": "prompt_base-girl.png" },
  { "name": "neg-common", "kind": "negative", "tags": ["lowres", "bad anatomy"] }
]
```

同一 kind 内で `name` は一意 (prompt と negative で同名は別物として共存可)。

`thumbnail` は **nullable** (省略可)。サムネ無しの旧 `presets.json` もそのまま読める (後方互換)。
サムネイル本体は別ファイル PNG として **`ui/data/thumbs/`** 配下に保存する
(`{kind}_{name}.png`・日本語名は保持・不正文字は `_` 置換しパストラバーサル防止)。
このディレクトリも gitignore 済み (`ui/data/` 配下)。`/thumb` で静的公開している。

### サムネイルの縮小 (依存)

縮小には [SixLabors.ImageSharp](https://github.com/SixLabors/ImageSharp) (クロスプラットフォーム・
`System.Drawing` 不使用) を使い、128px 上限・アスペクト比維持で PNG 保存する。
ライセンスは **Six Labors Split License** (オープンソース利用は Apache-2.0 相当で利用可・本プロジェクトはこの範囲)。

## プロジェクト構成

```
ui/
├─ Program.cs                    DI 配線 (HttpClient / SignalR / PresetStore / TagPaletteCatalog / FavoriteTagStore / TagLabels / テレメトリ常駐)
├─ appsettings.json              Dollama:BaseUrl = C++ サーバー URL
├─ wwwroot/
│  ├─ app.css                    ダークテーマ (2 カラム・左ペイン・上部バー・ミニメーター)
│  └─ tag-palette.json           キュレーション済みタグ木 (静的・prompt-engineer 管轄・[{category,tags[]}])
│  └─ tag-labels.ja.json         タグ日本語ラベル辞書 (静的・prompt-engineer 管轄・{英語タグ:日本語ラベル})
├─ Components/
│  ├─ Pages/Generate.razor       メイン画面 (2 カラム + 上部バー・ミニテレメ)
│  └─ Shared/
│     ├─ TagInput.razor          チップ入力 (再利用)
│     ├─ TagPresetField.razor    ラベル + チップ入力 + 保存バー (スリム化・一覧は左ペインへ)
│     ├─ TagPalette.razor        左ペイン: カテゴリ木 + お気に入り (タグをクリックで追加)
│     └─ PresetSidebar.razor     左ペイン: プリセット一覧 (サムネ付きコンパクトカード)
├─ Services/
│  ├─ DollamaClient.cs           C++ API ラッパ (型付き HttpClient)
│  ├─ Dtos.cs                    生成リクエスト/レスポンス DTO (snake_case で C++ に一致)
│  ├─ TagPreset.cs               プリセットのモデル (name / kind / tags / thumbnail)
│  ├─ TagCategory.cs             タグパレットの 1 カテゴリ (category / tags)
│  ├─ TagAdd.cs                  パレット → 親へのタグ追加通知 (tag / target)
│  ├─ PresetStore.cs             presets.json + thumbs の読み書き (スレッドセーフ singleton)
│  ├─ TagPaletteCatalog.cs       tag-palette.json を起動時に 1 回読む singleton (不在/壊れで空)
│  └─ FavoriteTagStore.cs        favorites.json の読み書き (スレッドセーフ・アトミック書込 singleton)
│  └─ TagLabels.cs               tag-labels.ja.json を起動時に 1 回読む singleton (不在/壊れで空=全件英語)
└─ Telemetry/
   ├─ TelemetrySample.cs         push する 1 フレーム分のデータ
   ├─ TelemetryHub.cs            SignalR ハブ (/hubs/telemetry)
   ├─ TelemetryBroadcaster.cs    周期 push する常駐サービス (現状スタブ波形)
   └─ GenerationActivity.cs      生成中フラグ (Broadcaster と画面で共有)
```

## C++ サーバーとの連携 (API 契約)

`src/server/api.cpp` の実装に合わせている。`Dollama:BaseUrl` (既定 `http://127.0.0.1:8080`) を叩く。

- `POST /v1/images/generations`
  - req: `{ "prompt"(必須), "negative_prompt"?, "steps"?, "size"?:"WxH", "response_format":"b64_json" }`
  - res: `{ "created", "data": [ { "b64_json" } ] }`
- `GET /health` → `{ "status": "ok" }`

## テレメトリの現状と差し替え点

C++ 側に `/telemetry` エンドポイントが**まだ無い**ため、初版は
[TelemetryBroadcaster.cs](Telemetry/TelemetryBroadcaster.cs) が
時刻ベースの擬似波形 (生成中は GPU が上昇) を push している。表示は上部バーの横並び
ミニメーター。

実 HW 計測に差し替えるときは、Broadcaster のループを
「`DollamaClient` で C++ の `/telemetry` をポーリング → 実測値を中継」に置き換える。
SignalR の push 構造とクライアント (Generate.razor) 側はそのまま再利用できる。

## トラブルシュート (実装メモ)

- **「ページを読み込めませんでした」/ 表示が固まる**
  接続チェック・SignalR 購読はプリレンダリング中ではなく `OnAfterRenderAsync(firstRender)` で行う。
  `OnInitializedAsync` でやると `HealthAsync` (最大 2 秒) と自己 SignalR 接続がプリレンダーを止め、
  最初のバイトが返らなくなる。
- **`FileNotFoundException: blazor.web.js / *.styles.css`**
  Production 環境かつ未 publish だと静的 Web アセットが解決できない。
  `ASPNETCORE_ENVIRONMENT=Development` で起動する (run-ui.ps1 は自動設定)。
- **パレットからタグを足しても入力欄に反映されない**
  タグ列は親 (Generate.razor) を真実源とし、追加時は**新しいリストを再代入**して
  子 (`TagInput`) の再描画をトリガする。リストをその場で `Add` するだけだと参照が
  変わらず再描画が走らないことがある。

## スコープ外 / 今後

- 実 HW テレメトリ (C++ `/telemetry`)、img2img (`/v1/images/edits` は C++ 側 501)、
  複数枚生成 (n>1) は未対応。
- 本物の txt2img は C++ 側 2-6b (CLIP-G OV 化 + CFG) で結線済み。重み配置が前提。
```

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

リポジトリルートの起動スクリプトを使うのが簡単 ([run-ui.ps1](run-ui.ps1)):

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

## 機能

| 機能 | 説明 |
|---|---|
| タグ・チップ入力 | プロンプト/ネガを danbooru タグとして 1 つずつ追加 (Enter / カンマ確定)・× で削除・空 Backspace で末尾削除。重複は無視・小文字寄せ |
| 種別別プリセット | タグ群を **kind (prompt / negative) 別**に名前付き保存・ロード・削除。各入力欄に専用バー |
| サムネ付きカード選択 | プリセットを **サムネイル付きカード一覧**で表示。カード本体クリックでロード・× で削除。サムネは**直近の生成画像を自動取得**し、保存時に 128px 上限へ縮小して紐付ける (サムネ無しはプレースホルダ) |
| 生成 | チップ群を `", "` 結合して `POST /v1/images/generations` へ。返った base64 PNG を表示 |
| 接続インジケータ | `GET /health` で C++ サーバー接続状態を表示 (緑/赤) |
| HW テレメトリ | CPU / NPU / iGPU / RTX5080 の稼働を SignalR で push 表示 (**現状スタブ値**) |

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
├─ Program.cs                    DI 配線 (HttpClient / SignalR / PresetStore / テレメトリ常駐)
├─ appsettings.json              Dollama:BaseUrl = C++ サーバー URL
├─ Components/
│  ├─ Pages/Generate.razor       メイン画面 (フォーム + 画像 + テレメトリ)
│  └─ Shared/
│     ├─ TagInput.razor          チップ入力 (再利用)
│     └─ TagPresetField.razor    ラベル + プリセット・カード一覧 (サムネ) + チップ入力 (再利用)
├─ Services/
│  ├─ DollamaClient.cs           C++ API ラッパ (型付き HttpClient)
│  ├─ Dtos.cs                    生成リクエスト/レスポンス DTO (snake_case で C++ に一致)
│  ├─ TagPreset.cs               プリセットのモデル (name / kind / tags)
│  └─ PresetStore.cs             presets.json の読み書き (スレッドセーフ singleton)
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
[TelemetryBroadcaster.cs](ui/Telemetry/TelemetryBroadcaster.cs) が
時刻ベースの擬似波形 (生成中は GPU が上昇) を push している。

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

## スコープ外 / 今後

- 現状は**簡素タイプ**。リッチ UI (ノードグラフ / カードダッシュボード) は将来の検討事項。
  チップ入力・プリセットはそのまま部品として持ち上げられる作りにしてある。
- 実 HW テレメトリ (C++ `/telemetry`)、img2img (`/v1/images/edits` は C++ 側 501)、
  よく使うタグのクリック追加パレット、複数枚生成 (n>1) は未対応。
- 本物の txt2img は C++ 側 2-6b (CLIP-G OV 化 + CFG) 待ち。それまでは StubGenerator か
  golden 埋め込み経由のデモになる。

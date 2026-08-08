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
| 追加先の可視化 | パレットのタグが**今どこに入るか**を面で示す。追加先のフィールドは入力欄の左に 3px の accent 縁 (`box-shadow: inset` — border だとレイアウトが 3px ずれる)、追加先がお気に入りのときは**パレット全体**がうっすら accent 地になる。フォーカスリングとは両立する (両方同時に出る) |
| お気に入りタグ | ★お気に入りはクリックで追加・× で解除・`+` で入力タグを登録。`favorites.json` に永続化 |
| タグ・チップ入力 | プロンプト/ネガを danbooru タグとして 1 つずつ追加 (**Enter / カンマ / 入力欄から離れる (blur) / 生成ボタンを押した瞬間**の 4 経路で確定)・× で削除・空 Backspace で末尾削除。重複は無視・小文字寄せ。IME 変換確定の Enter (`key=="Process"`) は無視して未変換ローマ字のタグ化を防ぐ |
| 種別別プリセット | タグ群を **kind (prompt / negative) 別**に名前付き保存・ロード・削除。保存は各入力欄の保存バー、一覧は**左ペインに集約**。**保存バーはタグ 0 個のとき出ない** (保存するものが無いので)。**同名保存は上書き**になり「〜を**上書き保存**しました」と明示する (文言は `Services/PresetSaveMessage.cs`) |
| サムネ付きカード選択 | プリセットを左ペインに**サムネイル付きコンパクトカード**で表示。カードクリックでロード・× で削除。サムネは**直近の生成画像を自動取得**し、保存時に 128px 上限へ縮小して紐付ける (サムネ無しはプレースホルダ) |
| 生成 | チップ群を `", "` 結合して `POST /v1/images/generations` へ。返った base64 PNG を表示。**Ctrl+Enter (Mac は Cmd+Enter)** でも生成できる (チップ入力にフォーカスがあっても効く) |
| 生成ボタンの活性と理由表示 | 押せないときは **disabled + 直下に理由テキスト**を出す (「効いているのに何も起きない」をなくす)。条件は `Services/GenerateGate.cs` に集約 |
| 生成中プレビュー (前回画像 + 経過秒) | 生成中も**前回画像を破棄せず減光 (opacity 0.35) して残し**、その上にスピナーと**経過秒 `mm:ss`** をオーバーレイ表示。下書き → 本番の見比べができ、失敗しても直前の絵が消えない。刻みは `PeriodicTimer` (1 秒)・表示整形は `Services/ElapsedFormat.cs` |
| PNG 保存 | 画像表示中は右ペイン右上に「**PNG 保存**」。`<a download href="data:image/png;base64,…">` だけで保存でき **JS interop 不要**。ファイル名は `dollama_{yyyyMMdd_HHmmss}_{size}.png` (`Services/DownloadName.cs`) |
| 下書き(高速プレビュー) | 生成パネルの 2 ボタン目。**本番と同じ重み・同じステップ数**のまま**解像度だけ 768²**に落として下書きを高速生成し、タグの当たり付けを速くする。右ペイン左上に「下書き 768²／本番 1024²」のモードバッジを表示。サイズ選択 (`_size`) 自体は不変で送信 size のみ落とす (512² は崩れるので不採用・幅>768 のみ縮小)。C++ サーバーは無改修 (`size` を変えて投げるだけ) |
| LoRA 選択チップ | カタログ (`wwwroot/loras.json`) の LoRA をトグルチップで選択し、強度スライダ (0.0–1.5) を添える。**スライダの数値表示はドラッグ中に追従する** (`@oninput`)。未選択なら生成リクエストに `loras` キー自体を出さない |
| 接続インジケータ | `GET /health` で C++ サーバー接続状態を上部バーに表示 (緑/赤)。**未接続のときだけ「再接続」ボタン**が出て `GET /health` を 1 回だけ叩き直す (定期ポーリングはしない・従来はページリロードしか回復手段がなかった) |
| HW テレメトリ | CPU / NPU / iGPU / RTX5080 の稼働を SignalR で push。**上部バーに横並びミニメーター**で表示 (**現状スタブ値**)。バーは**デバイス別の色** |
| 生成中のテレメトリ強調 | 生成中は 4 本のバーが**それぞれのデバイス色で淡く光り**、ブランド横に accent 塗りの「**生成中**」pill が出る (従来は title 属性のホバーのみ)。真実源はテレメトリの `Generating` フラグなので、**C++ サーバーが無くても**生成を試みている間は光る |
| テレメトリの役割ラベル | 各バーの右に**担当している役割** ("Qwen2 LLM" / "CLIP enc" / "VAE enc" / "SDXL UNet") を出し、「**NPU が CLIP を、RTX5080 が UNet を**やっている」が一目で分かるようにする。**ウィンドウ幅 ≥1500px のときだけ**表示 (`.tm-role` の既定は `display: none` = 通常フローに入らないので**バーの位置・幅は動かない**)。狭い幅では従来どおりホバーの `title` が受け皿。出し分けは **CSS だけ**で完結し razor に状態を持たない |
| タグの日本語表示 + EN⇔日本語トグル | パレット・中央チップのタグを**既定で日本語表示**。上部バーの `[日本語\|EN]` トグルで一括切替 (CascadingValue 伝播)。**内部に保持するタグ値・C++ へ送る prompt は英語の danbooru タグのまま不変**で表示専用。辞書に無いタグは英語フォールバック |

### 下書き(高速プレビュー)モード

生成パネルには「生成」(本番) と「下書き(高速プレビュー)」の 2 ボタンがある。下書きは
**本番と同じ SDXL 重み・同じステップ数**のまま、**解像度だけ 768² に下げて**高速生成し、
タグ・構図の当たり付けを速くするためのもの。落とすのは送信 `size` だけで、画面のサイズ選択
(`_size`) は書き換えない (下書きの後に「生成」を押せば選んだ解像度で出る)。

解像度決定は [`Services/DraftPreview.cs`](Services/DraftPreview.cs) の `ResolveDraftSize` に集約:

- 判定軸は **幅 W のみ**。**W > 768 → `768x768`**、**W <= 768 → そのまま据え置き** (512² は崩れるため
  下書きでも下げない・不採用)、**パース不能/空/null → `768x768`** (安全側)。
- `int.TryParse` で例外を投げない。`"WxH"` の `x`/`X` 区切りに対応。

生成成功後、右ペイン左上に「**下書き 768²**」「**本番 1024²**」のモードバッジを出す (未生成時は非表示)。
C++ サーバーは無改修で、既存 `POST /v1/images/generations` に変えた `size` を投げるだけ。

### 生成ボタンの活性条件・理由表示・未確定タグの確定

「入力したのに生成されない」「押しても何も起きない」という無言失敗を出さないための配線
(設計は `docs/ui-brushup-plan.md` §5 P2-1 / P2-8 / P2-9)。

**押せる条件** ([`Services/GenerateGate.cs`](Services/GenerateGate.cs) に集約):

```text
CanGenerate = 生成中でない && C++ サーバーに接続済み && (確定タグ > 0 || 未確定テキストが非空)
```

押せないときは 2 ボタンを disabled にし、**直下に理由を 1 つだけ**出す (優先順):

| 状態 | 理由テキスト |
|---|---|
| 生成中 | (出さない — ボタン文言が「生成中…」) |
| 未接続 | 「C++ サーバーに接続していません」+ 上部バーに再接続ボタン |
| タグ 0 かつ未確定テキストも空 | 「プロンプトにタグを 1 つ以上追加してください」 |

**未確定テキスト (draft) の確定**: 「1girl」と打って Enter せずに生成ボタンを押しても、
生成直前に自動で確定してから送る。`Generate` が `@ref` 経由で
`TagPresetField.CommitPendingAsync()` → `TagInput.CommitPendingAsync()` をプロンプト・
ネガティブの**両方 await** してからタグ列を読む。blur 確定も併設しているが、blur と click は
Blazor Server では**別の SignalR ラウンドトリップ**で順序が保証されないため主経路にはしない。

**そのため「未確定テキストあり・確定タグ 0」でもボタンは押せる** (押した瞬間に確定 → 生成)。
ここを disabled にすると確定の機会が永久に来ない。確定規則そのものは
[`Services/TagCommit.cs`](Services/TagCommit.cs) (`Normalize` / `Split` / `Merge`) に切り出してテスト済み。

**IME**: 変換中/変換確定の Enter は `key == "Process"` (keyCode 229) として来るのでキー処理ごと
無視する (チップ入力・お気に入り入力の 2 箇所)。JS interop は入れていない (配管を増やさない方針)。

**Ctrl+Enter**: `section.gen` で keydown を拾うのでチップ入力にフォーカスがあっても発火し、
可否判定はボタンと同じ `GenerateGate` を通る。

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
│                                 色・寸法は :root のトークン経由 (直書き禁止・36 トークン)。
│                                 ボタンは .btn + .btn-primary/.btn-ghost/.btn-icon の 4 クラス
│                                 (置き場所で決まる padding 等は器側の規則が持つ)
│  └─ tag-palette.json           キュレーション済みタグ木 (静的・prompt-engineer 管轄・[{category,tags[]}])
│  └─ tag-labels.ja.json         タグ日本語ラベル辞書 (静的・prompt-engineer 管轄・{英語タグ:日本語ラベル})
├─ Components/
│  ├─ Pages/Generate.razor       メイン画面 (2 カラム + 上部バー・ミニテレメ)
│  └─ Shared/
│     ├─ TagInput.razor          チップ入力 (再利用・未確定テキストの確定を public メソッドで公開)
│     ├─ TagPresetField.razor    ラベル + チップ入力 + 保存バー (一覧は左ペインへ・確定を 2 段中継・保存バーはタグ 0 で非表示・追加先なら IsTarget で左縁)
│     ├─ TagPalette.razor        左ペイン: カテゴリ木 + お気に入り (タグをクリックで追加)
│     └─ PresetSidebar.razor     左ペイン: プリセット一覧 (サムネ付きコンパクトカード)
├─ Services/
│  ├─ DollamaClient.cs           C++ API ラッパ (型付き HttpClient)
│  ├─ Dtos.cs                    生成リクエスト/レスポンス DTO (snake_case で C++ に一致)
│  ├─ DraftPreview.cs            下書き(高速プレビュー)の解像度決定 (幅>768 のみ 768² へ・純ロジック・テスト対象)
│  ├─ TagCommit.cs               チップ入力の未確定テキスト → タグ列の確定規則 (正規化/カンマ展開/重複排除・常に新リストを返す・純ロジック・テスト対象)
│  ├─ GenerateGate.cs            生成ボタンの活性条件と理由テキスト (P2-1/P2-8/P2-9 の唯一の合流点・純ロジック・テスト対象)
│  ├─ ElapsedFormat.cs           生成中オーバーレイの経過秒 TimeSpan → mm:ss (負値/端数/頭打ち・カルチャ非依存・純ロジック・テスト対象)
│  ├─ DownloadName.cs            「PNG 保存」のファイル名 dollama_{日時}_{size}.png (size は許可リストで正規化・純ロジック・テスト対象)
│  ├─ PresetSaveMessage.cs       プリセット保存後の一行メッセージ (同名上書き × サムネ有無の 4 通り・純ロジック・テスト対象)
│  ├─ TagPreset.cs               プリセットのモデル (name / kind / tags / thumbnail)
│  ├─ TagCategory.cs             タグパレットの 1 カテゴリ (category / tags)
│  ├─ TagAdd.cs                  パレット → 親へのタグ追加通知 (tag / target)
│  ├─ DeviceStyle.cs             テレメトリのデバイス名 → CSS クラス (未知/null は既定へフォールバック・純ロジック・テスト対象)
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
- **生成ボタンが押せない**
  理由がボタン直下に出る (未接続 / プロンプト空)。未接続なら上部バーの「再接続」を押す
  (C++ サーバー `dollama --http` が起動しているか確認)。**未確定テキストが残っているだけなら
  ボタンは押せる** — 押した瞬間に確定してから生成する仕様。

## スコープ外 / 今後

- 実 HW テレメトリ (C++ `/telemetry`)、img2img (`/v1/images/edits` は C++ 側 501)、
  複数枚生成 (n>1) は未対応。
- 本物の txt2img は C++ 側 2-6b (CLIP-G OV 化 + CFG) で結線済み。重み配置が前提。
```

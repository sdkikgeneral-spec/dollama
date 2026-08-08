# dollama UI ブラッシュアップ計画 (設計ドキュメント)

> 対象: `ui/` (Blazor Server)。**P1-1〜P1-5 / P1-7 実装済 (2026-08-08 / `feat/ui-p1-tokens`)**、
> **P2-1 / P2-5 / P2-8 / P2-9 実装済 (2026-08-08 / `feat/ui-p2-gate` = P2 バッチ A)**、
> **P2-2 / P2-3 / P2-4 実装済 (2026-08-08 / `feat/ui-p2-preview` = P2 バッチ B)**。P1-6 と残り P2 以降は計画。
> 残りは本 doc の実装バックログ (P1-6 → P2 → P3) に沿って csharp-ui-implementer が着手する。
> レビュー日: 2026-07-14 / レビュー対象コミット: `feat/fast-mode-g0b-g3k` ブランチ先端 (3f16d6c 時点の ui ツリー)。

---

## 1. 目的とスコープ

### 目的

- **使い勝手**: 生成フロー (タグ入力 → サイズ/ステップ → 生成 → プレビュー) の摩擦を減らす。
  特に「入力したのに生成されない」「今なにが起きているか分からない」系の無言失敗をなくす。
- **配色・コントラスト**: WCAG AA (テキスト 4.5:1 / UI 非テキスト 3:1) を基準に、境界・フォーカス・
  placeholder の視認性を底上げする。LM Studio 風ダークの雰囲気は維持。
- **dollama らしさ**: トップバーの HW テレメトリ (CPU/NPU/iGPU/RTX5080) を「全 HW が協調して
  動いている感」がひと目で伝わる見せ方へ強化する。
- **一貫性**: 散在するボタン/チップ/入力スタイルと直書き色値をトークン化し、以後の変更を
  `:root` 差し替えだけで効く構造にする。

### 非目標 (やらないこと)

- **機能追加・breadth 拡大はしない** ([[project-output-quality-over-features]])。ノードエディタ、
  ギャラリー管理、複雑なワークフロー等 ComfyUI 的機能は対象外。
- C++ 側 (`src/`) は無改修。UI ↔ C++ の API 契約 (`POST /v1/images/generations`, `GET /health`) も不変。
- 全面リライトはしない。**トークン差し替え + 局所改善で 8 割**を取る。各項目は独立にマージ可能・可逆。
- ライトテーマは**追加しない** (判断根拠は §4.1 末尾)。

---

## 2. 現状サマリ

### 構造

- **3 ペイン grid**: 左 320px (タグパレット `TagPalette` + プリセット一覧 `PresetSidebar`) /
  中央 minmax(340px, 400px) (プロンプト・ネガティブの `TagPresetField`、サイズ/ステップ、
  `LoraChips`、生成 2 ボタン) / 右 1fr (画像プレビュー専任)。1100px 以下で 1 カラム縦積み。
- **トップバー**: ブランド → テレメトリ・ミニメーター (SignalR push・4 デバイスのバー+%) →
  タグ表示言語トグル [日本語|EN] → 接続インジケータ。
- **スタイル集約**: 実質すべて `ui/wwwroot/app.css` (約 930 行)。CSS 変数は `:root` に 9 個
  (`--bg/--panel/--panel-2/--border/--text/--muted/--accent/--ok/--ng`)。

### 現行配色トークン

| トークン | 値 | 主用途 |
|---|---|---|
| `--bg` | `#14161a` | ページ背景・チップボタン背景 |
| `--panel` | `#1c1f26` | ペイン・トップバー |
| `--panel-2` | `#23272f` | 入力・カテゴリ・セカンダリボタン |
| `--border` | `#2c313b` | ほぼ全境界線 |
| `--text` | `#e6e9ef` | 本文 |
| `--muted` | `#8b92a0` | ラベル・補足 |
| `--accent` | `#6ea8fe` | 主ボタン・選択・hover・fill |
| `--ok` / `--ng` | `#3fb950` / `#f85149` | 接続 OK / NG・エラー |

### 良い点 (壊さない)

- **状態管理が堅い**: タグ列・追加先・LoRA 選択の真実源が `Generate.razor` に一元化され、
  新リスト再代入で子へ確実に伝播する流儀が徹底されている。プリレンダリング対策
  (`OnAfterRenderAsync`) も正しい。
- **本文コントラストは優秀**: `--text` on `--panel` ≈ **13.6:1**。`--accent` on `--panel` ≈ **6.8:1**、
  accent ボタン上の暗文字 (#0b1220) ≈ **7.8:1** で、主要な文字はすべて AA/AAA 圏。
- **下書き/本番の分離**が視覚 (塗り vs アウトライン) と文言の両方で表現済み。モードバッジ
  (「下書き 768²」) で直近プレビューの条件が分かる。
- チップ UI (Enter/カンマ確定・空 Backspace で末尾削除) は operate 感がよい。
- テレメトリのミニメーターという「dollama らしさ」の核が既に存在する。

---

## 3. 課題一覧

深刻度: **高** = 無言失敗・視認不能級 / **中** = 迷い・引っかかり / **低** = 磨き。

| # | 分類 | 課題 | 現状 | 深刻度 | 対象ファイル |
|---|---|---|---|---|---|
| 1 | UX | **未確定テキストのまま生成すると無視される**。「1girl」と打って Enter せず生成ボタンを押すと、`_draft` は `Tags` に入らずプロンプト空 → #2 と合わさり完全な無言失敗 **→ P2-1 で解消 (2026-08-08・生成前に親が `CommitPendingAsync()` を await + blur 確定を併設)** | `TagInput` は Enter/カンマでのみ確定。blur・生成時の確定なし | **高** | `TagInput.razor`, `Generate.razor` |
| 2 | UX | **プロンプト空だと生成ボタンが「効いているのに何も起きない」**。`GenerateAsync` 冒頭で `_promptTags.Count == 0` なら silent return。ボタンは有効に見える **→ P2-1 で解消 (2026-08-08・`GenerateGate` で disabled + 理由テキスト)** | disabled は `_busy` のみ。空時の理由表示なし | **高** | `Generate.razor` |
| 3 | 配色 | **チップ入力にフォーカス表示がゼロ**。`.chip-entry:focus { outline: none; }` のみで、外枠 `.chips` も光らない。キーボード利用時に現在地が分からない **→ P1-2 で解消 (2026-08-08)** | `textarea/select` は border 色替えのみ、`.chip-entry` は完全無表示 | **高** | `app.css` |
| 4 | 配色 | **入力・カード境界がほぼ見えない**。`--border #2c313b` vs `--panel-2 #23272f` ≈ **1.15:1**、vs `--panel` ≈ **1.26:1** (UI 非テキストの目安 3:1 に遠い)。入力欄とパネルの区別が輝度でつかない **→ P1-1/P1-4 で部分解消 (2026-08-08・境界 2 段化で vs panel 1.26→1.46 / 操作要素 vs panel 2.10。3:1 まで上げると縞模様化するため意図的に手前で止めている)** | 全境界が単一の `--border` | **高** | `app.css` |
| 5 | UX | **IME 変換確定の Enter でタグが誤確定しうる**。日本語 UI なのに `OnKeyDown` が composing を見ない (変換確定 Enter で未変換ローマ字や変換途中文字列がタグ化する恐れ) **→ P2-5 で対処 (2026-08-08・`key=="Process"` を無視・JS interop は不採用)** | `e.Key == "Enter"` のみ判定 | 中 | `TagInput.razor`, `TagPalette.razor` |
| 6 | UX | **生成開始で前回画像を即破棄** (`_imageData = null`)。下書き→本番の比較ができず、画面が毎回プレースホルダへフラッシュする **→ P2 で解消 (2026-08-08・破棄をやめ `preview-dim` で減光保持)** | busy 中はスピナーのみ | 中 | `Generate.razor`, `app.css` |
| 7 | UX | **生成中フィードバックがスピナーのみ**。1024² 本番は 20 秒前後かかるのに経過時間も目安もない (進捗 API は C++ に無いので進捗バーは対象外) **→ P2 で解消 (2026-08-08・`PeriodicTimer` 1s 刻みの経過秒オーバーレイ)** | `.spinner` 単体 | 中 | `Generate.razor`, `app.css` |
| 8 | UX | **生成画像の保存導線がない**。右クリック→名前を付けて保存頼み。data URI なのでファイル名も不定 **→ P2 で解消 (2026-08-08・右上「PNG 保存」+ `dollama_{日時}_{size}.png`)** | ダウンロード/コピー UI なし | 中 | `Generate.razor` |
| 9 | UX | **追加先フォーカス連動が暗黙的すぎる**。お気に入り入力欄に一度触れると以降のパレットクリックが全部 favorites 行きになるが、変化はパレット上部トグルの小さなハイライトだけ。追加先のフィールド側には何も出ない | トグル label の `.on` (accent 枠) のみ | 中 | `app.css` (+ 軽微に `TagPresetField.razor`) |
| 10 | UX | **LoRA 強度スライダの数値表示がドラッグに追従しない**。`@onchange` (離した時) なので操作中は古い値が見え続ける **→ P2 で解消 (2026-08-08・`@oninput` 化 + parse/表示の InvariantCulture 固定)** | `@onchange` バインド | 中 | `LoraChips.razor` |
| 11 | UX | **テレメトリの「生成中」が視覚に出ない**。`Generating` フラグは title 属性 (ホバー) のみ。全 HW 協調の見せ場が眠っている | バー幅の変化だけ | 中 | `Generate.razor`, `app.css` |
| 12 | 配色 | **placeholder が未スタイル**。ブラウザ既定色 (UA 依存・半透明) でダーク背景ではコントラスト不定。例示文言 (「例: 1girl と入力して Enter」) が導線なのに読みにくい **→ P1-3 で解消 (2026-08-08)** | `::placeholder` 宣言なし | 中 | `app.css` |
| 13 | 配色 | **accent 青一色に役割が集中**。主ボタン・選択チップ・hover・保存完了メッセージ・テレメトリ%・言語トグル on がすべて `--accent`。「操作できる」「選択中」「情報」の区別が色で付かない | 単一 `--accent` | 中 | `app.css` |
| 14 | 一貫性 | **色の直書きが散在**。`rgba(110,168,254,…)` (チップ 2 箇所)、`#0d1117`/`#0b1220`/`#0b0f17` (accent 上の暗文字が 3 種微妙に違う)、`#ffb4ae` (エラー淡色 2 箇所)、`#9d7bff` (バーのグラデ)。accent を差し替えるとチップだけ旧色で残る **→ P1-1 で解消 (2026-08-08・`:root` 外の直書き色ゼロを `AppCssTokenTests` で機械保証)** | トークン未経由 | 中 | `app.css` |
| 15 | 一貫性 | **ボタン系が 5 系統の重複定義**。`.preset-btn` / `.fav-plus` / `.lang-toggle button` / `.generate.secondary` / `.ps-del` がほぼ同じ見た目を別々に宣言。hover 作法も border 色替え/文字色替えが混在 **→ P1 で部分解消 (2026-08-08・境界とフォーカスの作法は統一。ボタン統合は P3-1 に残)** | 共通クラスなし | 中 | `app.css` |
| 16 | レイアウト | **タイポ階層がフラット**。11/12/13/14/15px がアドホックに散在し、ペイン見出し (`palette-head` 13px) と本文の差が 1px。視線が「どこから読むか」を掴めない | スケール定義なし | 中 | `app.css` |
| 17 | レイアウト | **1100px 縦積みで生成ボタンとプレビューが分断**。縦積み順が 左→中央→右 のため、生成ボタンを押した後に画面最下部までスクロールしないと結果が見えない | `@media (max-width:1100px)` 単純 1fr | 中 | `app.css` |
| 18 | 配色 | **`--muted` の小サイズ使用が余裕薄**。#8b92a0 on `--panel` ≈ **5.3:1**、on `--panel-2` ≈ **4.8:1** で AA は満たすが、11px 箇所 (`tm-item`, `palette-empty`, `lora-val`) はぎりぎり感 **→ P1-1 で解消 (2026-08-08・#9aa2b1 で on panel 6.42:1 / on panel-2 5.83:1)** | 単一 muted | 低 | `app.css` |
| 19 | UX | **プリセット同名保存が確認なし上書き**。`PresetStore.Save` は同 kind 同名を黙って置換する仕様 (store 側は正しい)。UI で一言も出ない | 保存メッセージは成功文言のみ | 低 | `TagPresetField.razor` |
| 20 | UX | **保存バーが常時表示でノイズ**。タグ 0 個でもプリセット名入力+保存ボタンが両フィールドに出続ける | 常時表示 | 低 | `TagPresetField.razor`, `app.css` |
| 21 | UX | **キーボードで生成できない**。タグ確定後に毎回マウスでボタンへ移動する。Ctrl+Enter 等のショートカットなし **→ P2-8 で解消 (2026-08-08・`section.gen` で keydown を拾い Ctrl/Cmd+Enter)** | なし | 低 | `Generate.razor`, `TagInput.razor` |
| 22 | 一貫性 | **再接続モーダル/エラーバーが英語+ライト配色**。`ReconnectModal` は "Rejoining the server..."、`MainLayout.razor.css` の `#blazor-error-ui` は `lightyellow` でダークテーマ内で異物 **→ P1-7 で解消 (2026-08-08)** | テンプレート既定のまま | 低 | `ReconnectModal.razor(.css)`, `MainLayout.razor(.css)` |
| 23 | 一貫性 | **角丸・チップの微差**。radius 4/6/8/10/999px が混在 (それ自体は可) だが体系がなく、`fav-chip` と `tag-chip` は同じ「タグチップ」なのに padding/font-size が別定義 | 個別指定 | 低 | `app.css` |
| 24 | レイアウト | **中央列の上限 400px が窮屈**。長いタグ列 (15 個超) でチップが縦に積み上がる一方、右ペイン 1fr は正方形画像の左右に大きな余白が残る | `minmax(340px,400px)` 固定 | 低 | `app.css` |

**計 24 件** (高 4 / 中 13 / 低 7)。

### コントラスト比の算定根拠 (概算・WCAG 相対輝度)

| ペア | 比 | 判定 |
|---|---|---|
| `--text` #e6e9ef / `--panel` #1c1f26 | ≈ 13.6:1 | AAA ✅ |
| `--muted` #8b92a0 / `--panel` | ≈ 5.3:1 | AA ✅ (小サイズは余裕薄) |
| `--muted` / `--panel-2` #23272f | ≈ 4.8:1 | AA ✅ ぎりぎり |
| `--accent` #6ea8fe / `--panel` | ≈ 6.8:1 | AA ✅ |
| #0b1220 (ボタン文字) / `--accent` (ボタン地) | ≈ 7.8:1 | AA ✅ |
| `--ok` #3fb950 / `--panel` | ≈ 6.5:1 | AA ✅ |
| `--ng` #f85149 / `--panel` | ≈ 4.9:1 | AA ✅ (テキスト用は淡色 #ffb4ae 併用が正解) |
| `--border` #2c313b / `--panel-2` | ≈ **1.15:1** | UI 3:1 ✗ |
| `--border` / `--panel` | ≈ **1.26:1** | UI 3:1 ✗ |

→ **文字は健全・「枠とフォーカス」が弱点**という診断。

---

## 4. 提案デザイン方針

### 4.1 配色トークン改訂案

方針: **ダーク単一継続** (LM Studio 風維持)。ベース色相は現行のまま、(a) 境界 2 段化、
(b) フォーカスリング新設、(c) 直書き色のトークン化、(d) テレメトリ用デバイス色 4 色の追加。
`:root` 差し替え + 直書き箇所の `var()` 置換だけで完結し、レイアウトへの影響ゼロ。

| トークン | 現行 | 改訂案 | 根拠 / 想定コントラスト |
|---|---|---|---|
| `--bg` | `#14161a` | **据え置き** | 問題なし |
| `--panel` | `#1c1f26` | **据え置き** | 問題なし |
| `--panel-2` | `#23272f` | **据え置き** | 問題なし |
| `--border` | `#2c313b` | `#343b47` | 非操作境界。vs panel ≈ 1.6:1 (装飾線としては十分・全枠を 3:1 にすると縞模様化して逆効果) |
| `--border-strong` | — (新設) | `#4a5261` | **操作可能要素** (入力・ボタン・カード) の境界専用。vs panel-2 ≈ 2.3:1 + hover/focus で補完 |
| `--text` | `#e6e9ef` | **据え置き** | 13.6:1 |
| `--muted` | `#8b92a0` | `#9aa2b1` | 11–12px 箇所の余裕確保。on panel ≈ 6.3:1 / on panel-2 ≈ 5.7:1 (AA 余裕) |
| `--accent` | `#6ea8fe` | **据え置き** | 6.8:1・ブランド色として維持 |
| `--on-accent` | — (新設) | `#0b1220` | accent 塗り上の文字。現在 3 種散在 (#0d1117/#0b1220/#0b0f17) を 1 本化。≈ 7.8:1 |
| `--accent-weak` | — (新設) | `rgba(110,168,254,0.15)` | 選択チップ地。`rgba(110,168,254,…)` 直書き置換 (accent 連動は P2 の color-mix 化で完成) |
| `--accent-border` | — (新設) | `rgba(110,168,254,0.4)` | 選択チップ枠。同上 |
| `--ok` | `#3fb950` | **据え置き** | 6.5:1 |
| `--ng` | `#f85149` | **据え置き** | 枠・ドット用。4.9:1 |
| `--ng-soft` | — (新設) | `#ffb4ae` | エラー本文用の淡色。#ffb4ae 直書き 2 箇所を統合。on エラー地 ≈ 9:1 超 |
| `--ng-weak` | — (新設) | `rgba(248, 81, 73, 0.12)` | **PL 裁定で追加** (`.error` 背景の直書き `rgba(248,81,73,0.12)` が本表から漏れていた。置換してトークン経由率 100% を例外なしで成立させるため) |
| `--focus-ring` | — (新設) | `rgba(110,168,254,0.45)` | `box-shadow: 0 0 0 2px var(--focus-ring)` の全フォーカス統一リング。リング自体の視認 3:1 相当 |
| `--overlay` | — (**P2-2 で追加**) | `rgba(0, 0, 0, 0.45)` | 生成中オーバーレイの暗幕 (`.preview-overlay`)。減光 (0.35) した前回画像の上でも経過秒が読めるようにする。明るい画像でも `--text` 比 ≈ 5.9:1 |
| `--dev-cpu` | — (新設) | `#6ea8fe` | テレメトリ・デバイス色 (青=CPU)。on panel ≈ 6.8:1 |
| `--dev-npu` | — (新設) | `#a78bfa` | 紫=NPU (現グラデ #9d7bff の系譜)。on panel ≈ 5.8:1 |
| `--dev-igpu` | — (新設) | `#5fd4c8` | ティール=iGPU。on panel ≈ 9:1 |
| `--dev-gpu` | — (新設) | `#f0a860` | オレンジ=RTX5080 (主役の熱量)。on panel ≈ 8:1 |

**ライト/ダーク判断**: **ダーク単一継続**。理由: ① 参照体験 (LM Studio) がダーク前提でブランド一貫、
② 生成画像 (アニメ塗り) の見えはダーク地が有利、③ ライト対応は全トークンの二重管理 + 検証コスト増で
「絵の質・使い勝手第一」の投資判断に合わない。将来必要になっても本改訂でトークン経由率を
100% にしておけば `:root` 追加だけで足せる (可逆)。

### 4.2 タイポ / スペーシングのスケール

現行のアドホット値を最小限のスケールに畳む。**px 値自体はほぼ現行踏襲**で、トークン名を通すのが主目的。

```css
/* タイポ (4 段) */
--fs-xs: 11px;   /* テレメトリ・補足 (使用箇所を最小化) */
--fs-sm: 12px;   /* ラベル・チップ・ボタン小 */
--fs-md: 14px;   /* 本文 (html 既定) */
--fs-lg: 16px;   /* ブランド・ペイン見出し */

/* スペーシング (4px グリッド) */
--sp-1: 4px;  --sp-2: 8px;  --sp-3: 12px;  --sp-4: 16px;  --sp-6: 24px;

/* 角丸 (3 段 + pill) */
--r-sm: 6px;   /* 入力・小ボタン */
--r-md: 10px;  /* ペイン・カード (現 8/10 を 10 に寄せる) */
--r-pill: 999px;
```

- ペイン見出し (`palette-head` 等) は `--fs-sm` + `letter-spacing: 0.06em` + `--muted` の
  「小さく薄く広く」型セクションヘッダへ (LM Studio 流儀)。本文と 1px 差で並ぶ現状より階層が立つ。
- 13px / 15px の中間値は最寄りへ寄せる (13→12 or 14、15→14 or 16)。視覚差は軽微で棚卸し効果が大きい。

### 4.3 コンポーネント別の改善方針

**トップバー**
- 接続インジケータに **未接続時の再試行ボタン** (「再接続」小ボタン) を添える。現在は未接続だと
  ページリロードしか回復手段がない。
- 未接続時は生成 2 ボタンを disabled + tooltip「C++ サーバー未接続」にし、押してからエラーで知るのをやめる。

**テレメトリ (§4.4 に詳細)**

**タグパレット (左)**
- 追加先トグルのラジオ丸ポチを隠し、**セグメントコントロール見た目** (言語トグルと同型・`.on` は
  accent 塗り) に統一。現在「◉ラジオ + accent 枠」の二重表現で言語トグルと作法が違う。
- パレットのタグボタン hover は現行 (accent 枠) 維持。favorites ターゲット中はパレット全体に
  うっすら地色 (例: `--accent-weak` の縁) を敷き「今クリックするとお気に入りに入る」を面で示す (#9)。

**生成コントロール (中央)**
- **追加先ハイライト**: `_target` に一致するフィールドの `.chips` に accent の左縁 (3px) を付け、
  「パレットのタグはここに入る」を入力欄側でも示す (#9)。
- **保存バーの整理**: タグ 0 個のときは保存バーを非表示 (または「保存…」の折りたたみ) にして
  ノイズを減らす (#20)。同名保存時はメッセージを「〜を上書き保存しました」に変える (#19)。
- **生成ボタン**: プロンプト空なら disabled + ボタン下に muted で「プロンプトにタグを 1 つ以上
  追加してください」。未確定 draft がある場合は生成時に自動確定してから送る (#1, #2)。
- **ステップスライダ**: 現行値表示は良い。既定 20 の位置に datalist 目盛り (1 行) を足す程度で十分。
- **LoRA**: スライダを `@oninput` (即時) バインドへ (#10)。LoRA 行はタイトル行「LoRA」を
  他フィールドの `field > span` と同スタイルに揃える。

**プレビュー (右)**
- **前回画像を保持**: busy 中は前回画像を `opacity: 0.35` で残し、中央にスピナー + 経過秒
  (`00:12`) をオーバーレイ (#6, #7)。完了時に差し替え。
- **アクションバー**: 画像表示中のみ右上に「PNG 保存」ボタン (a[download] + data URI で JS 追加不要)。
  ファイル名は `dollama_{yyyyMMdd_HHmmss}_{size}.png` (#8)。
- モードバッジは現行維持。バッジに steps を追記 (「本番 1024² / 20 steps」) すると再現に効く (#24 の軽減にも)。

**プリセット (左下)**
- カード hover 時に **タグ列のプレビューを title でなくポップ表示は不要** (breadth 抑制)。
  現行 title (名前のみ) を `名前 — タグ列先頭 5 個` に変えるだけで十分。

### 4.4 「全 HW 協調」の可視化強化 (dollama らしさ)

テレメトリは dollama の看板。ただし派手にはしない — **色と 1 つのアニメーションだけで語る**。

1. **デバイス固有色** (P1・CSS のみ): バーの fill を全デバイス同一グラデ → デバイス別単色へ
   (`--dev-cpu/npu/igpu/gpu`)。`tm-item` に `data-dev` 属性 (または device 名の CSS クラス) を付けて
   fill と `tm-pct` を各色に。**4 色が並んで動く = 4 HW が協調して動いている**が一瞥で伝わる。
   **実装済 (2026-08-08)**: デバイス名 → クラス名 (`tm-cpu`/`tm-npu`/`tm-igpu`/`tm-gpu`) の写像は
   razor の三項演算子ではなく純クラス `Services/DeviceStyle.cs` に切り出した (`DraftPreview` と同じ流儀・
   `ui.Tests/DeviceStyleTests.cs` で検証)。未知デバイスは `""` を返し既定グラデへフォールバックする。
   なお既定グラデに残っていた直書き `#9d7bff` は `--dev-npu` (`#a78bfa`) に置換した — **意図的な微変更**
   (トークン経由率 100% を優先し、ほぼ同系の紫へ寄せた)。
2. **生成中の強調** (P1–P2): `_sample.Generating == true` のとき `telemetry-mini` にクラス `generating`
   を付け、(a) 各バーに `box-shadow: 0 0 6px <デバイス色 40%>` の淡いグロー、(b) ブランド横に
   小さな「生成中」pill (accent 塗り) を出す。title 頼みをやめる (#11)。
3. **役割ラベルの露出** (P2): 現在 `Role` ("CLIP enc" / "SDXL UNet" 等) は title のみ。ホバー時に
   バー下へ 10px で表示、または幅に余裕がある時だけ常時表示 (`min-width` メディアクエリ)。
   「NPU が CLIP を、RTX5080 が UNet をやっている」というパイプライン理解が UI から得られる。
4. **やらないこと**: パイプライン DAG 図・履歴グラフ・スパークライン等は breadth。テレメトリは
   現状スタブ波形なので、実測が入るまで見せ方への過剰投資はしない。

---

## 5. 実装バックログ

工数目安: S = 〜1h / M = 半日 / L = 1 日超。各項目は独立にマージ可能な単位に分割済み。

### P1 — トークンと CSS だけで効く (app.css 中心・挙動無変更・完全可逆)

| ID | 課題# | 変更内容 | 対象 | 工数 | リスク・可逆性 | 依存 |
|---|---|---|---|---|---|---|
| P1-1 ✅ 2026-08-08 | 4, 18, 14 | `:root` トークン改訂 (§4.1 の表どおり): `--border` 明度上げ・`--border-strong`/`--on-accent`/`--accent-weak`/`--accent-border`/`--ng-soft`/`--focus-ring`/`--dev-*` 新設・`--muted` 更新。既存の直書き色 (`rgba(110,168,254,…)`, `#0d1117`, `#0b1220`, `#0b0f17`, `#ffb4ae`, `#9d7bff`) を `var()` へ置換 | `app.css` | S | ほぼゼロ。git revert 1 発で戻る | なし |
| P1-2 ✅ 2026-08-08 | 3 | フォーカス統一リング: `textarea:focus, select:focus, input:focus-visible, button:focus-visible, .chips:focus-within` に `box-shadow: 0 0 0 2px var(--focus-ring)` + border-color accent。`.chip-entry:focus { outline:none }` は残すが親 `.chips:focus-within` で必ず光らせる | `app.css` | S | ゼロ | P1-1 |
| P1-3 ✅ 2026-08-08 | 12 | `::placeholder { color: var(--muted); opacity: 1; }` を明示 | `app.css` | S | ゼロ | P1-1 |
| P1-4 ✅ 2026-08-08 | 4, 15 | 操作可能要素 (`textarea/select/.preset-name/.fav-entry/.chips/.preset-btn/.fav-plus/.ps-card/.palette-tag/.lora-chip`) の境界を `--border-strong` へ。非操作 (ペイン枠・区切り) は `--border` のまま | `app.css` | S | 低。見た目のみ | P1-1 |
| P1-5 ✅ 2026-08-08 | — (§4.4-1) | テレメトリのデバイス別色: `tm-fill`/`tm-pct` をデバイス色に。razor 側は `tm-item` にデバイス名クラス (`tm-cpu` 等) を 1 属性足すだけ | `app.css`, `Generate.razor` (1 行), `Services/DeviceStyle.cs` | S | 低 | P1-1 |
| P1-6 🔲 | 16, 23 | タイポ/スペーシング/角丸トークン (§4.2) を `:root` に追加し、既存宣言を段階置換。ペイン見出しをセクションヘッダ型へ | `app.css` | M | 低。px 実値ほぼ不変 | P1-1 |
| P1-7 ✅ 2026-08-08 | 22 | `#blazor-error-ui` (`MainLayout.razor.css`) をダーク配色へ (app.css 側の同名定義と重複しているので css 側を整理)。`ReconnectModal` の文言を日本語化 + ダーク配色 | `MainLayout.razor(.css)`, `ReconnectModal.razor(.css)` | S | 低 | なし |

> **P1 実装メモ (2026-08-08)**: 重複整理の実体は「app.css 側が死にコードだった」。scoped CSS は
> `#blazor-error-ui[b-xxxx]` へ書き換わり裸のセレクタより詳細度で必ず勝つため、実際に効いていたのは
> `MainLayout.razor.css` の `lightyellow` 側。app.css 側のブロックを削除し scoped 側をダーク化して一本化した。
> P1-4 の `--border-strong` は 13 箇所 (上記 + `.generate.secondary`/`.ps-del`/`.palette-target label`/`.lang-toggle` 外枠)、
> `--border` 据え置きは 10 箇所 (`.topbar`/`.gen`/`.canvas`/`.palette,.preset-sidebar`/`.preview`/`.ps-noimg`/
> `.gen-mode`/`.palette-cat`/`.lang-toggle` 内側区切り/`.spinner`)。規準は「クリック/フォーカスできる要素の外周 = strong」。

### P2 — 挙動を伴う UX 改善 (razor/cs 変更・各項目独立)

| ID | 課題# | 変更内容 | 対象 | 工数 | リスク・可逆性 | 依存 |
|---|---|---|---|---|---|---|
| P2-1 ✅ 2026-08-08 (`feat/ui-p2-gate`) | 1, 2 | **無言失敗の根絶**: ① `TagInput` に「未確定 draft を確定して返す」公開メソッド or blur 確定を追加 ② `Generate` は生成前に確定を要求 ③ プロンプト空なら生成ボタン disabled + 理由テキスト。**テスト**: draft 確定ロジックを ui.Tests で (正規化・カンマ展開・重複) | `TagInput.razor`, `TagPresetField.razor`, `Generate.razor`, `Services/TagCommit.cs`, `Services/GenerateGate.cs` | M | 中 (バインド伝播に注意・既存の新リスト再代入流儀を踏襲) | なし |
| P2-2 ✅ 2026-08-08 (`feat/ui-p2-preview`) | 6, 7 | busy 中は前回画像を減光保持 + スピナー/経過秒オーバーレイ。経過秒は **`PeriodicTimer`** で 1s 刻み `StateHasChanged` (表示整形は純クラス `ElapsedFormat`) | `Generate.razor`, `app.css`, `Services/ElapsedFormat.cs` | M | 低 | なし |
| P2-3 ✅ 2026-08-08 (`feat/ui-p2-preview`) | 8 | プレビュー右上に「PNG 保存」(`<a download="dollama_….png" href="data:image/png;base64,…">`)。JS interop 不要 (ファイル名は純クラス `DownloadName`) | `Generate.razor`, `app.css`, `Services/DownloadName.cs` | S | 低 | なし |
| P2-4 ✅ 2026-08-08 (`feat/ui-p2-preview`) | 10 | LoRA スライダを `@oninput` 即時反映へ (`@onchange` → `@oninput`)。SignalR 往復頻度が上がるが値は軽量。**併せて parse/表示を InvariantCulture 固定** (潜在バグの同時修正) | `LoraChips.razor` | S | 低 | なし |
| P2-5 ✅ 2026-08-08 (`feat/ui-p2-gate`) | 5 | IME ガード: `e.Key == "Process"` を無視 + 可能なら `KeyboardEventArgs.IsComposing` 相当の判定 (Blazor 標準に無ければ `@onkeydown` の代わりに keypress 系 or 小さな JS interop を検討。**JS を足すなら最小 1 関数**) | `TagInput.razor`, `TagPalette.razor` | M | 中 (ブラウザ差・要手動確認) | P2-1 |
| P2-6 | 9 | 追加先の可視化: `_target` 一致フィールドの `.chips` に accent 左縁 + favorites ターゲット中のパレット縁色 | `Generate.razor`, `TagPresetField.razor`, `TagPalette.razor`, `app.css` | M | 低 | P1-1 |
| P2-7 | 19, 20 | 保存バー整理: タグ 0 で非表示・同名時の文言を「上書き保存」へ (`PresetStore.All(kind)` で既存名照合)。**テスト**: 上書き判定ロジック | `TagPresetField.razor` | S | 低 | なし |
| P2-8 ✅ 2026-08-08 (`feat/ui-p2-gate`) | 21 | Ctrl+Enter で生成 (チップ入力フォーカス中でも発火)。`@onkeydown` を `.gen` セクションで拾う | `Generate.razor`, `TagInput.razor` | S | 低 | P2-1 |
| P2-9 ✅ 2026-08-08 (`feat/ui-p2-gate`) | — (§4.3) | 未接続時: 生成ボタン disabled + 接続インジケータに「再接続」ボタン (HealthAsync 再試行) | `Generate.razor`, `app.css` | S | 低 | なし |
| P2-10 | 11 (§4.4-2) | 生成中のテレメトリ強調 (`generating` クラス + グロー + 「生成中」pill) | `Generate.razor`, `app.css` | S | 低 | P1-5 |

> **P2 バッチ A 実装メモ (2026-08-08 / `feat/ui-p2-gate` / P2-1・P2-9・P2-8・P2-5)**
>
> - **確定方式は「公開メソッドを親から呼ぶ」が主・blur が従**。`Generate.GenerateAsync` が
>   `@ref` 経由で `TagPresetField.CommitPendingAsync()` → `TagInput.CommitPendingAsync()` を
>   **両フィールド分 await** してからタグ列を読む。blur 単独に頼らないのは、Blazor Server では
>   blur と click が**別の SignalR ラウンドトリップ**で飛び順序が保証されないため。
>   blur (`@onblur`) は Tab 移動時の取りこぼし救済として併設する。
> - **draft の文字列は親へ渡さない**。親が draft を持つと真実源の二重管理になるので、
>   `TagInput` は「空かどうか」だけ `DraftEmptyChanged` (真偽) で通知し、親はそれを
>   ボタンの活性判定にのみ使う。**これがないと「draft に文字があるのにタグ 0 なので disabled」→
>   確定の機会が永久に来ない**という別の無言失敗になる (実装上いちばん踏みやすい罠)。
> - **`Services/GenerateGate.cs` が P2-1 / P2-8 / P2-9 の唯一の合流点**。
>   `Evaluate(busy, connected, promptTagCount, draftEmpty) → (CanGenerate, Reason?)` で
>   2 ボタンの disabled・理由テキスト・Ctrl+Enter が同じ判定を通る。入力が真偽 4 つなので
>   **16 組合せを xUnit で全数検査**でき、razor に条件式が散らない。
> - **確定規則は `Services/TagCommit.cs`** (`Normalize` / `Split` / `Merge`) に集約。名前は
>   生成モードの `DraftPreview` と紛らわしくないよう `TagDraft` ではなく `TagCommit`。
>   ついでに `TagInput` の確定・削除・Backspace を**新リスト再代入**へ是正した
>   (旧 `AddOne` は `Tags.Add` でその場 Mutate = 本プロジェクトの伝播流儀違反だった)。
> - **P2-5 は JS interop を不採用**とした (PL 裁定)。Blazor の `KeyboardEventArgs` に
>   `IsComposing` が無く完全解には JS が要るが、① このチップ入力に入るのは danbooru タグ =
>   英数字で IME を使う場面が乏しい ② `wwwroot/*.js` + `IJSRuntime` + Dispose 管理 +
>   テスト不能領域が増え「配管を増やさない」方針に反する ③ 実害である「未変換ローマ字のタグ化」は
>   Windows の Chrome/Edge/Firefox が変換中/確定の Enter を `key="Process"` (keyCode 229) で
>   送るため `Process` 無視だけで防げる。実装は `TagInput.OnKeyDown` と
>   `TagPalette.OnFavKeyDown` の 2 箇所で即 return。
>   **残存リスク**: composition 終了と同一 keydown で `Enter` が来るブラウザでは 1 回だけ
>   確定が漏れうる。手動確認で再現したら実装で粘らず PL へ上げる (JS を勝手に足さない)。
> - **P2-9 は定期ポーリングを足さない**。未接続判定は既存 `_connected` の再利用で、再判定経路は
>   「初回描画」「生成の成否」「再接続ボタン押下時の `HealthAsync()` 1 回」の 3 つだけ
>   (`_reconnecting` で二度押し防止)。
> - テスト: `TagCommitTests` 20 件 + `GenerateGateTests` 24 件を追加し **91 → 135 件**。
>   bUnit は入れない方針を維持 (依存追加ゼロ)。

> **P2 バッチ B 実装メモ (2026-08-08 / `feat/ui-p2-preview` / P2-2・P2-3・P2-4)**
>
> - **`_imageData = null` を消した影響は 4 点に閉じる**。① **サムネ**: `_imageBytes` は
>   `TagPresetField.CurrentImage` に渡るプリセット保存のサムネ元なので、破棄をやめると
>   「生成失敗後に保存 → 直近の成功画像がサムネになる」。**画面に見えている画像がサムネになる方が
>   直感的**なので許容する (`docs/ui-preset-thumbnail-spec.md` に明記)。② **モードバッジ**:
>   `_lastMode` は成功時のみ更新の現状維持 — バッジと表示画像が常に同じ生成を指す一貫性を壊さない。
>   ③ **下書き/本番**: `_size` 非破壊・`sendSize` ローカル上書き・`_runningDraft` の文言出し分けは不変
>   (`DraftPreview` も無改修)。④ **エラー**: `_error` の生成開始時クリアは現状維持。
> - **タイマーは fire-and-forget にしない**。`GenerateAsync` 内で `CancellationTokenSource` を作り、
>   `finally` で「参照を外す → `Cancel()` → `await tickTask` → `Dispose()`」の順に確実に畳む。
>   `DisposeAsync` からも `Cancel()` するが、この順序なので破棄済み cts が見えることはない。
>   停止経路は **完了 / エラー / 離脱の 3 つ**。刻みは加算ではなく開始時刻との差分
>   (`DateTimeOffset.UtcNow - startedAt`) で取り、刻みがずれても表示が遅れない。
> - **`System.Timers.Timer` を使わない理由**: コールバックが別スレッドで来るので毎回
>   `InvokeAsync` が要るうえ、Dispose と発火のレースを自前で塞ぐ必要がある。
>   `PeriodicTimer` + `await` ならループが 1 本の Task に閉じ、キャンセルで必ず終わる。
> - **`.preview-save` はフォーカスリングの統一ブロックへ追記**した。`<a download>` なので
>   `button:focus-visible` に当たらないが、別ブロックを作ると「`var(--focus-ring)` を使う宣言は
>   1 つ」という `AppCssTokenTests` の機械検査 (P1-2 の回帰止め) に引っかかるため。
> - **P2-4 は `@oninput` 化とセットで `float.TryParse` / `ToString` を `InvariantCulture` へ固定**した。
>   `input[type=range]` の値は常に `"0.85"` 形式で来るのに parse が現在カルチャ依存だったため、
>   小数点がカンマのロケール (de-DE 等) では**強度がまったく動かない**潜在バグだった。
>   発火頻度が上がる変更と同時に入れるのは範囲拡大ではなく回帰予防。
> - テスト: `ElapsedFormatTests` 31 件 + `DownloadNameTests` 38 件を追加し **135 → 204 件**。
>   タイマー本体は自動テスト対象外 (制御フローだけを使い捨てハーネスで確認)、
>   検証するのは切り出した純ロジック 2 つだけ。

### P3 — 構造に触れる・効果はあるが急がない

| ID | 課題# | 変更内容 | 対象 | 工数 | リスク・可逆性 | 依存 |
|---|---|---|---|---|---|---|
| P3-1 | 15, 23 | ボタン共通化: `.btn` / `.btn-primary` / `.btn-ghost` / `.btn-icon` の 4 クラスへ既存 5 系統を統合。razor 側のクラス名張り替えを伴うため回帰確認範囲が広い | `app.css`, 各 razor | M | 中 (見た目回帰は目視のみ) | P1-6 |
| P3-2 | 17 | レスポンシブ再構成: 縦積み時の order を 中央→右→左 に変更 (生成→結果→パレットの導線) + 生成ボタン行を `position: sticky; bottom: 0` に | `app.css` | M | 中 (ブレークポイント目視必須) | なし |
| P3-3 | — (§4.4-3) | テレメトリ役割ラベル露出 (ホバー or 広幅時常時) | `Generate.razor`, `app.css` | S | 低 | P1-5 |
| P3-4 | 24 | 中央列上限を 400→460px + 右ペイン余白調整。長タグ列での窮屈さ緩和 | `app.css` | S | 低 | なし |
| P3-5 | — | 直近生成のミニ履歴 (右ペイン下に直近 4 枚のサムネ・クリックで再表示・**セッション内メモリのみ・永続化しない**)。breadth 境界ぎりぎりなので、P1/P2 完了後にユーザー判断を仰いでから | `Generate.razor`, `app.css` | L | 中 (メモリ保持量・保留可) | P2-2 |

---

## 6. 段階適用の順序

1. ✅ **P1-1 (トークン改訂) が筆頭** (2026-08-08 完了)。`:root` 差し替え + 直書き置換だけで境界・muted・チップ・
   エラー色が全画面に効く。挙動無変更・1 コミット・revert 容易。
2. ✅ **P1-2〜P1-4 (フォーカス/placeholder/境界)** (2026-08-08 完了) — 高深刻度 #3/#4 をコードロジックに触れず解消。
3. ✅ **P1-5 (テレメトリのデバイス色)** (2026-08-08 完了) / 🔲 P2-10 (生成中強調) は P2 で — dollama らしさの即効改善。
   見た目の変化が最も分かりやすく、ユーザーの体感リターンが早い。
   ※ P1-7 (エラー UI ダーク化・再接続モーダル日本語化) も同時に完了させた (依存なし・独立)。
4. ✅ **P2-1 (無言失敗の根絶)** (2026-08-08 完了・`feat/ui-p2-gate`) — 挙動変更の第一弾。
   UX 深刻度は最高だが razor ロジックに触れるため P1 の安定後に。ui.Tests を同時に追加。
   ※ 同じ「生成ボタンの活性」に触れる P2-9 (未接続 disabled + 再接続) / P2-8 (Ctrl+Enter) と、
   同じ keydown を触る P2-5 (IME ガード) を**同一バッチ (P2 バッチ A)** でまとめて実装した
   — 判定を `GenerateGate` 1 箇所に集約するには同時に入れるのが安全なため。
5. ✅ **P2-2/P2-3 (プレビュー保持・保存)** (2026-08-08 完了・`feat/ui-p2-preview` = P2 バッチ B) →
   残りの P2 を小さく順次。※ 同じプレビュー節に触る P2-3 と、独立だが 1 行で済む P2-4
   (LoRA スライダ即時反映) を同一バッチでまとめた。残りは P2-6 / P2-7 / P2-10。
6. **P1-6/P3-1 (タイポトークン→ボタン統合)** はリファクタ性格なので、機能系 P2 が落ち着いた
   タイミングで**まとめて後日**。P1 実装時も P1-6 は意図的に見送り、P3-1 と一緒に扱う方針を維持する
   (タイポ/スペーシングを先に畳んでもボタン 5 系統の重複が残ると二度手間になるため)。
   P3-2 (レスポンシブ) と P3-5 (履歴) は最後・P3-5 は着手前にユーザー確認。

各段階で `dotnet build ui/Dollama.Ui.csproj` + `dotnet test ui.Tests` 緑を確認してからマージする。

---

## 7. 検証方法

### 自動 (ui.Tests / xUnit)

- **P1 (実装済・依存追加ゼロ / bUnit 不使用)** — 2 ファイル追加。CSS/razor をテキストとして読み、
  壊れると気づきにくい性質だけを機械検査する。
  - `ui.Tests/AppCssTokenTests.cs` (28 件): ① `:root` に既存 7 + 新設 13 のトークンが定義済で値が空でない
    ② **コントラスト比を hex 完全一致ではなく WCAG 相対輝度で計算**して下限を守る (`--text`/`--panel` ≥ 7.0、
    `--muted` ≥ 4.5、`--on-accent`/`--accent` ≥ 4.5、`--ng-soft` ≥ 4.5、`--dev-*` 4 色 ≥ 4.5、
    `--border-strong`/`--panel` ≥ 2.0、`--border`/`--panel` ≥ 1.4)。値一致は単なる変更検知器になるため採らない
    ③ 半透明トークン 4 種が `rgba()` としてパースでき α が (0,1) ④ **`:root` 外に直書き色ゼロ** (hex / rgba とも・許可リストなし)
    ⑤ `app.css` + scoped CSS 2 本の `var(--x)` が全て `:root` 定義済 (かつ scoped に `:root` を書いていない)
    ⑥ フォーカス統一リングが 1 ブロックで 5 セレクタ全部を覆い `.chip-entry` の二重リング打ち消しがある
    ⑦ `::placeholder` が `color: var(--muted)` + `opacity: 1` ⑧ `.tm-cpu/npu/igpu/gpu` が各 `--dev-*` を参照し
    `Generate.razor` が `DeviceStyle.CssClass` を呼ぶ ⑨ P1-7 の回帰止め (app.css に `#blazor-error-ui` が無い・
    scoped が `lightyellow`/`color-scheme: light only` でない・`ReconnectModal` の Blazor JS 契約 id/class が全部残っている)。
  - `ui.Tests/DeviceStyleTests.cs` (17 件): `DeviceStyle.CssClass` の 4 デバイス正常系・大小文字非依存・
    未知/null/空は `""` (既定グラデへフォールバック)。
- **P2-1 / P2-8 / P2-9 (実装済 2026-08-08 / 依存追加ゼロ・bUnit 不使用)** — 2 ファイル追加。
  razor から純クラスへ切り出した「確定規則」と「ゲート判定」を検証する。
  - `ui.Tests/TagCommitTests.cs` (20 件): `Normalize` (trim/lowercase・null 安全・語中の空白は保持)、
    `Split` (カンマ展開・空除去・入力内重複を先勝ちで除去・順序保持)、`Merge` (既存列への連結・
    **常に新インスタンス**を返し `current` を Mutate しない・重複判定は正規化後だが既存表記は温存・
    Count 差で「増えたか」が判る)。
  - `ui.Tests/GenerateGateTests.cs` (24 件): `Evaluate` の **16 組合せ全数** (busy × connected ×
    タグ 0/1 × draft 空) + 理由の優先順位 (busy > 未接続 > プロンプト空) + 「draft のみでも
    enabled」の単独固定 + タグ数の境界 (負値は 0 扱い) + 理由文言が空でなく互いに異なること。
- **P2-2 / P2-3 (実装済 2026-08-08 / 依存追加ゼロ・bUnit 不使用)** — 2 ファイル追加。
  razor に残るのはタイマーと markup だけで、テストするのは切り出した純ロジック 2 つ。
  - `ui.Tests/ElapsedFormatTests.cs` (31 件): `Mmss` の通常系 (0/1/59/60/3599 秒)・60 分超も
    分として伸びる (3661 → `61:01`)・**100 分以上は `99:59` で頭打ち** (`TimeSpan.MaxValue` でも
    落ちない)・負値は `00:00` (`TimeSpan.MinValue` 含む)・端数切り捨て (12.9 秒 → `00:12`)・
    形式の不変条件 (常に 5 文字・数字とコロンのみ・秒欄は 0–59)・カルチャ非依存・単調非減少。
  - `ui.Tests/DownloadNameTests.cs` (38 件): `ForPng` が `dollama_{yyyyMMdd_HHmmss}_{size}.png` に
    なること・size の正規化 (`1024X1024` / 空白入り)・読めない size は `unknown`・
    **パス区切りと `Path.GetInvalidFileNameChars()` が絶対に混入しない** (`../../etc/passwd` 等 12 種)・
    出力は常に `[A-Za-z0-9_.]+`・日時のゼロ埋めとカルチャ非依存 (仏暦/ヒジュラ暦ロケールでも不変)・
    秒違いで名前が変わる・例外を投げない。
- **P2-2 のタイマー**は自動テスト対象外 (razor + レンダラが要る)。停止 3 経路
  (完了 / エラー / 離脱) は制御フローだけを抜き出した使い捨てハーネスで確認する。
- **P2-7**: 同名上書き判定を PresetStore 経由で検証 (既存テスト拡張)。
- CSS/razor の見た目はテスト対象外 — ビルド緑 + 手動確認で担保。

### 手動 (実装 PR ごとのチェックリスト)

1. **コントラスト実測**: 改訂トークンを DevTools のカラーピッカー (コントラスト表示) で確認し、
   §3 の表の想定値 (muted ≥ 4.5:1、focus リング視認、accent ボタン ≥ 4.5:1) を数値で記録する。
2. **フォーカス一巡**: Tab キーだけで トップバー → パレット → プロンプト → ネガティブ →
   サイズ/ステップ → LoRA → 生成ボタン を一巡し、全停留点でリングが見えること。
3. **無言失敗シナリオ** (P2 バッチ A 後・ブラウザと IME はエージェントで検証不能なのでユーザーが実施):
   ① 「1girl」と打って Enter せず生成ボタン → 確定されてチップ化され生成される
   ② タグ 0 かつ draft 空 → ボタン disabled + 理由テキスト
   ③ **draft に文字あり・タグ 0 → ボタン enabled** (ここが罠の核心・disabled だと確定不能)
   ④ 未接続 → ボタン disabled +「C++ サーバーに接続していません」+ 再接続ボタン
   ⑤ IME で「少女」を変換確定 → タグ化されない (`key=="Process"` ガード)
   ⑥ Ctrl+Enter (チップ入力にフォーカスがある状態) → 生成が走る。
4. **ブレークポイント・スクショ観点**: 1440px (3 ペイン) / 1100px 境界 / 900px (縦積み) /
   768px の 4 点で、(a) テレメトリの折返し・省略 (b) 追加先トグル 3 ラベルの非折返し
   (c) 縦積み時の生成→プレビュー導線 (P3-2 後) をスクショで残す。
5. **テレメトリ**: スタブ波形で 4 色バーが個別に動くこと、生成中 (`Activity.Enter` 中) に
   強調表示が出て終了で消えること。
6. **回帰**: プリセット保存/ロード/削除、お気に入り追加/削除、LoRA 選択→生成リクエストの
   `loras` キー有無 (未選択で送らない)、下書き/本番のサイズ切替 — 従来挙動が不変であること。

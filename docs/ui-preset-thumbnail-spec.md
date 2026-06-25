# UI プリセット・サムネイル 設計案

UI で作成したタグプリセットに**サムネイル画像**を持たせ、選択 UI で一覧表示できるようにする提案。
ステータス: **未着手・設計案 (要レビュー承認)**。実装はプランモード承認 → `project-leader` 経由で行う。

## 背景 / 現状

- プリセットは「タグ列だけ」のデータ。[ui/Services/TagPreset.cs](../ui/Services/TagPreset.cs) は `name` / `kind` / `tags` のみで画像情報を持たない。
- 永続化は [ui/Services/PresetStore.cs](../ui/Services/PresetStore.cs) が `ui/data/presets.json` に読み書き (スレッドセーフ singleton)。
- 選択 UI は [ui/Components/Shared/TagPresetField.razor](../ui/Components/Shared/TagPresetField.razor) の `<select>` ドロップダウン。
  - **制約**: HTML の `<select>`/`<option>` の中に画像は描画できない。サムネ表示するなら選択 UI をカード/グリッドへ作り替える必要がある。
- 生成画像は [ui/Components/Pages/Generate.razor](../ui/Components/Pages/Generate.razor) の `_imageData` (base64 PNG) として画面に表示済み。

## ねらい

プリセットを「タグの集合」ではなく「見た目の見本付きの構成」として選べるようにする。
生成 → 気に入った → そのタグ構成をサムネ付きで保存 → 次回はサムネを見て選ぶ、という動線。

## 設計方針 (推奨)

3 点の変更で成立する。

### 1. モデルにサムネ参照を追加

`TagPreset` に縮小 PNG への参照を追加する。

```csharp
public sealed class TagPreset
{
    public string Name { get; set; } = "";
    public string Kind { get; set; } = "";
    public List<string> Tags { get; set; } = new();

    // 追加: サムネ画像ファイル名 (ui/data/thumbs/ 配下・相対)。無い場合は null。
    [JsonPropertyName("thumbnail")]
    public string? Thumbnail { get; set; }
}
```

### 2. 保存時にサムネを取り込む

**推奨: 直近の生成画像を自動でサムネにする** (追加操作ゼロ)。

- `Generate.razor` の `_imageData` (base64 PNG) を `TagPresetField` 経由で保存時に渡す。
- `PresetStore.Save` 側で受け取った PNG を **128px 程度に縮小**して `ui/data/thumbs/{kind}_{name}.png` に書き出し、`Thumbnail` にファイル名を入れる。
- 縮小は `System.Drawing` ではなく **ImageSharp** 等のクロスプラットフォームなライブラリを使う (Linux/コンテナ配慮)。導入を避けたい場合は初版では「縮小なし・原寸 PNG を保存」でも可 (容量とのトレードオフ)。
- 代替案: ユーザーがファイルを手動選択 (`<InputFile>`)。自動取り込みと併存可能だが初版では自動のみで十分。

### 3. 選択 UI をサムネ・グリッド化

`<select>` を廃し、同一 kind のプリセットを**カード一覧** (サムネ + 名前) で表示する。

- カードクリックで「ロード」(現 `Load()` 相当)。
- 各カードに削除ボタン (現 `DeleteSelected()` 相当)。
- サムネが無いプリセットはプレースホルダ画像/アイコンを表示。
- 保存名入力 + 保存ボタンは現状を踏襲。

サムネ配信は `ui/data/thumbs/` を静的ファイルとして公開するか、`/thumb/{kind}/{name}` の最小エンドポイントで返す
(`data/` は wwwroot 外なので、静的公開するなら `app.UseStaticFiles` でマッピングを追加)。

## 保存形式の選択肢

| 方式 | 長所 | 短所 |
|---|---|---|
| **別ファイル PNG (推奨)** `ui/data/thumbs/*.png` をパス参照 | `presets.json` が軽いまま・画像配信が素直 | ファイル管理 (削除時に PNG も消す) が要る |
| base64 を `presets.json` に直接埋め込み | 単一ファイルで完結・配信不要 | JSON が肥大 (1 枚数十 KB〜)・全件ロード時に重い |

→ **別ファイル PNG + 128px 縮小** を推奨。

## 既存プリセットとの互換

- `Thumbnail` は nullable。既存 `presets.json` (サムネ無し) はそのまま読め、UI ではプレースホルダ表示。
- 削除時は対応する `thumbs/*.png` も併せて削除する (`PresetStore.Delete`)。

## 決定が必要な点 (レビュー時に確認)

1. サムネ取得元: **①直近の生成画像を自動** (推奨) / ②手動ファイル選択 / ③両方。
2. 保存形式: **①別ファイル PNG** (推奨) / ②base64 埋め込み。
3. 縮小ライブラリ導入の可否 (ImageSharp 追加 OK か / 初版は原寸でいくか)。

## 影響範囲 (実装時の触る場所)

- [ui/Services/TagPreset.cs](../ui/Services/TagPreset.cs) — `Thumbnail` フィールド追加。
- [ui/Services/PresetStore.cs](../ui/Services/PresetStore.cs) — 保存時のサムネ書き出し・削除時の PNG 削除・(必要なら) 縮小。
- [ui/Components/Shared/TagPresetField.razor](../ui/Components/Shared/TagPresetField.razor) — 選択 UI をカード一覧化・保存時に現在の生成画像を受け取る。
- [ui/Components/Pages/Generate.razor](../ui/Components/Pages/Generate.razor) — 現在の `_imageData` を `TagPresetField` に渡す結線。
- `ui/Program.cs` — `ui/data/thumbs/` の静的公開 or サムネ配信エンドポイント。
- [ui/README.md](../ui/README.md) — 機能表とプリセット形式の追記。
- スコープ外: **C++ 生成サーバー側は無改修** (UI 内で完結)。

## 備考

- これは研究コアではなく「配管」(UI) の範囲。C++ 単一バイナリ思想とは切り離された Blazor 側の機能追加。
- 将来のリッチ UI (ui/README.md「スコープ外 / 今後」) と整合する方向 (カード・ダッシュボード化の一歩)。

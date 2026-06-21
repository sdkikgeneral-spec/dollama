# character_bible データ構造仕様 — dollama C++ 実装

## スコープ (確定)

**dollama は 2D イラスト/漫画の「キャラクター描画」生成に特化する。**

- 背景は生成対象外 (Grok / Gemini / Stable Diffusion + CLIP Studio Paint で合成)
- 出力は **切り抜き済み透過 PNG (キャラのみ)**
- マッティング (α 抽出) も HW 協調パイプラインの一段とする
- 背景を捨てることで CLIP 77 トークンを丸ごとキャラ記述に使え、`canonical_tags` の
  支配力が上がる → キャラ一貫性が原理的に向上する

## 設計方針

| 原則 | 内容 |
|---|---|
| 二層分離 | **同一性層** (不変) と **シーン層** (都度) を別構造体に |
| 名前は索引 | `name` は拡散モデルへの入力ではなく台帳の主キー |
| 年齢はコンパイル元 | `age` は生で渡さず body/proportion タグへ変換 |
| 背景非依存 | キャラ層は背景情報を持たない。背景はシーン層が「抜きやすい単色」を指定するのみ |
| STL のみ | 外部ライブラリなし。`std::string` / `std::vector` / `std::unordered_map` |
| 三案を内包 | 案A=`CharacterBible` 台帳 / 案B=`embedding_slot` / 案C=`identity_features` |

## ファイル (実装予定)

```
src/core/character.hpp        — CharacterIdentity / SceneSpec / OutputSpec / CharacterBible
src/tests/test_character.cpp  — 台帳の put/find・プロンプト合成・年齢コンパイルのテスト
```

## 1. 同一性層 `CharacterIdentity` (コマ間で不変)

```cpp
#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <unordered_map>

namespace dollama {

// 性別 (記録用メタ。外見は canonical_tags で明示する)
enum class Sex
{
    Female,
    Male,
    Other,
};

// ------------------------------------------------------------
// CharacterIdentity: キャラクター同一性層
// 一度決めたらコマ間で「不変」に保つべき設定。
// name は拡散モデルへ渡す文字列ではなく、台帳を引く「主キー」。
// 見た目を決めるのは canonical_tags + embedding。age は外見を決めない。
// ------------------------------------------------------------
struct CharacterIdentity
{
    // --- 索引 (人間・システムが扱うハンドル) ---
    std::string name;                        // 主キー (例 "Aria")
    std::vector<std::string> aliases;        // 別名・検索キー

    // --- 設定メタ (記録専用・拡散には渡さない) ---
    // 重要: age は「真の設定年齢」であり外見年齢ではない。タグへ自動展開しない。
    //   外見年齢 (老け顔/童顔) と体型は canonical_tags に明示する。
    //   これにより 老け顔の男子高校生 (age:17 + [mature]) や
    //   ロリババア (age:800 + [loli]) を年齢を偽らずに表現できる。
    int  age = 0;                            // 真の設定年齢 (メタ・LLM 文脈・PNG 用)
    Sex  sex = Sex::Female;

    // --- 正準外見 (実際に拡散を固定する本体) ---
    // 外見年齢・体型もここに明示する ("mature","loli","tall","petite"...)。
    std::vector<std::string> canonical_tags; // 固定外見 ("silver hair","red eyes"...)
    std::vector<std::string> forbidden_tags; // 絶対に出さない (ネガティブへ注入)

    // --- 正準解剖 (一度決めたら不変。L3 指数検査の照合基準) ---
    // 重要: 「5本」を正解として焼き込まない。4本指カートゥーン・3本指・
    //   非人間も許容する。最初に決めた本数を以降の全コマで固定する。
    //   検査は「宣言値と一致するか」を見る (絶対値5ではない)。
    //   DIGITS_UNCOUNTABLE (=0): 肉球/鉤爪/ミトン等。指数カウント検査を無効化し
    //   タグ + embedding + seed で固定する。
    int  digits_per_hand = 5;                // 片手の指の本数 (5=人間, 4=カートゥーン, 0=非カウント)

    // --- 案B: NPU 常駐埋め込みへのポインタ (ID ハンドル。埋め込み実体は外部) ---
    int32_t  embedding_slot = -1;            // NPU 埋め込みテーブルの slot (-1=未登録)

    // 注: 案C の同一性重心 (identity_features) は learned 層なので本構造体に持たない。
    //   生成→学習→フィードバックで蓄積される状態は別構造 CharacterMemory (別タスク, §11)。

    // --- 同一性の最終保険 ---
    uint64_t seed = 0;                       // 0 = ランダム許可

    // --- 管理メタ ---
    uint32_t schema_version = 1;             // 台帳スキーマ版
};
```

## 2. シーン層 `SceneSpec` (コマごとに可変)

背景タグは廃止。代わりに「綺麗に抜くための」固定背景指定のみ持つ。

```cpp
// ------------------------------------------------------------
// SceneSpec: シーン層 (1コマぶんの可変設定)
// 背景は生成しない。抜きやすい単色背景を指定するのみ。
// ------------------------------------------------------------
struct SceneSpec
{
    std::vector<std::string> pose_tags;        // "sitting","waving"...
    std::vector<std::string> expression_tags;  // "smile","blush"...
    std::string composition;                   // "upper body","full body"...
    uint64_t    scene_seed = 0;                // 構図用 seed (同一性 seed とは別)
};
```

## 3. 出力層 `OutputSpec` (切り抜き=合成前提)

```cpp
// 切り抜き方式 (CLIP Studio 合成前提)
enum class MattingMode
{
    None,        // 切り抜かない (デバッグ用)
    Segment,     // 道B: 単色生成 → anime-segmentation で α 抽出 (第一候補)
    Native,      // 道A: LayerDiffuse で RGBA 直接生成 (将来研究)
};

// ------------------------------------------------------------
// OutputSpec: 出力・合成設定
// 背景は生成しない。キャラのみを透過 PNG で出す。
// matting_device は probe 比較で決定するため未確定 (空文字 = 自動)。
//
// 背景色について: 切り抜きは「意味的セグメンテーション」(isnet-anime 等) で
//   行うため、クロマキー (緑/マゼンタ) のような特定キー色は不要・不採用。
//   キー色が効くのは YCbCr クロマサブサンプリング前提の映像クロマキーの話で、
//   RGB 意味セグメンテーションには無関係。
//   実際に効くのは (1) 縁のスピル: 無彩色 (白/グレー) が縁が綺麗、
//   (2) danbooru タグ頻度: "simple background"/"white background" が最頻出で
//   生成品質が高い。よってデフォルトは "simple background"。
//   白髪・白服キャラで白背景がコントラスト不足になる場合に備え可変フィールドとし、
//   最終的な既定色は probe で決定する。
// ------------------------------------------------------------
// 色モード (方式A: プロンプトタグ注入)
//   AI 絵の均質な塗りの違和感を避けたいとき、白黒/線画で出して CLIP Studio で
//   人が塗る運用を可能にする。方式A は danbooru タグを positive に注入するだけで、
//   新モデルや後処理は持たない。
//   注: 方式B (線画抽出 HW ステージ: Anime2Sketch 等) はバックログ (§9 参照、
//       matting と並ぶ出力段候補)。
enum class ColorMode
{
    Color,       // 通常 (タグ注入なし)
    Greyscale,   // monochrome, greyscale
    Lineart,     // lineart, monochrome, sketch
};

struct OutputSpec
{
    MattingMode matting           = MattingMode::Segment;
    ColorMode   color_mode        = ColorMode::Color;    // 方式A: 色モード (タグ注入)
    std::string isolation_tag     = "simple background"; // 抜きやすい背景 (可変・probe 判断)
    std::string matting_device    = "";                  // "iGPU"/"NPU"/"CPU" — probe で決定
    bool        emit_alpha        = true;                // RGBA PNG で出力
    bool        quality_negatives = true;                // L1: 品質ネガティブを negative へ注入
    bool        refine_hands      = true;                // L2: 手検出→インペイント段 (Phase 2 実装)
};
```

## 4. 台帳本体 `CharacterBible`

```cpp
// ------------------------------------------------------------
// CharacterBible: 全キャラクターの台帳
// LLM (将来 自作タグ生成 LM) が name をキーに引いて同一性層を解決する。
// ------------------------------------------------------------
class CharacterBible
{
public:
    void put(const CharacterIdentity& c)
    {
        index_[c.name] = c;
    }

    const CharacterIdentity* find(const std::string& name) const
    {
        auto it = index_.find(name);
        return it == index_.end() ? nullptr : &it->second;
    }

    size_t size() const noexcept
    {
        return index_.size();
    }

private:
    std::unordered_map<std::string, CharacterIdentity> index_;
};

} // namespace dollama
```

## 5. 年齢は外見に変換しない (設計判断)

**`age` をタグへ自動コンパイルしない。** 当初は `age → body/proportion タグ` を
検討したが、外見年齢と設定年齢を1つの数字に潰すと表現力を失うため撤回した。

問題例:
- 老け顔の男子高校生: `age:17` を自動展開すると常に童顔・細身になり作れない
- ロリババア: `age:800` を自動展開すると mature/adult になりロリ絵が出せない。
  回避に `age:10` と偽れば設定メタ (LLM 文脈・PNG・物語) が壊れる

解決: **設定年齢 (数字) と外見 (タグ) を直交させる。**

| 軸 | 持ち方 | 例 |
|---|---|---|
| 設定年齢 | `CharacterIdentity.age` (メタ専用) | 17 / 800 |
| 外見年齢・体型 | `canonical_tags` に明示 | `mature` / `loli` / `tall` |

これで老け顔男子高校生 (`age:17` + `[mature, old]`)、ロリババア (`age:800` + `[loli]`)
を**年齢を偽らず**表現できる。danbooru ネイティブのタグ (`aged up`/`aged down`/
`mature male`/`loli`/`old`) をそのまま `canonical_tags` に書く運用とする。

→ `compile_age_tags()` はプロンプト合成パイプラインに**存在しない**。

## 6. プロンプト合成 (同一性層 → シーン層の優先順位)

```
positive = canonical_tags          // 不変外見が最優先 (CLIP 前方トークンが効く)
         + color_mode_tags(output.color_mode)  // 方式A 色モード (canonical 直後・scene 前)
         + scene.pose_tags
         + scene.expression_tags
         + [scene.composition]      // 空文字なら追加しない
         + [output.isolation_tag]   // "simple background" で抜きやすく (空文字なら追加しない)

negative = forbidden_tags          // このキャラに矛盾する要素を排除
         + (output.quality_negatives ? default_quality_negatives() : {})  // L1 品質

seed     = identity.seed (!=0) ? identity.seed : scene.scene_seed
```

`canonical_tags` を **先頭** に置くのがブレ止めの肝 (CLIP は前方トークンの寄与が強い)。
体型・外見年齢も canonical_tags に含まれるため、別途の age 展開は不要。

**色モードタグの注入位置 (方式A)**: `output.color_mode` に応じた画風タグを
`canonical_tags` の**直後・scene タグの前**に挿入する。同一性 (#1 canonical) を
最優先に保ちつつ、画風タグは前方トークンに置いて画像全体へ効かせる狙い。
`Color` は何も足さない (既定・既存挙動と同一)。タグは `color_mode_tags()` が返す:

| color_mode | 注入タグ |
|---|---|
| `Color` | (なし) |
| `Greyscale` | `monochrome`, `greyscale` |
| `Lineart` | `lineart`, `monochrome`, `sketch` |

品質ネガティブは**本数非依存の崩れ**だけを叩く (指の本数は §10 の解剖固定で扱う)。
`extra fingers`/`fewer digits` のような「本数を仮定する語」は入れない。

```cpp
constexpr int DIGITS_UNCOUNTABLE = 0;  // 肉球/鉤爪/ミトン: 指数カウント検査を無効化

// L1: 普遍的な品質ネガティブ (キャラ非依存・本数非依存)。
// per-character の forbidden_tags とは別物として扱う。
inline std::vector<std::string> default_quality_negatives()
{
    return {
        "bad hands", "malformed hands", "mutated hands",
        "fused fingers", "bad anatomy", "deformed",
        "lowres", "worst quality", "jpeg artifacts",
    };
}
```

## 7. PNG メタ往復スキーマ (`tEXt` / `iTXt` 焼き込み)

前のコマの PNG を読み戻して次コマへキャラ設定を引き継ぐ往復に使う。
`identity_features` はサイズが大きいため PNG には焼かず、キャラ DB 側に保持し
メタには `embedding_slot` の参照のみ載せる。

```json
{
  "dollama_bible_version": 1,
  "identity": {
    "name": "Aria",
    "age": 17,
    "sex": "female",
    "canonical_tags": ["silver hair", "red eyes", "long hair", "twin tails"],
    "forbidden_tags": ["short hair", "blue eyes"],
    "digits_per_hand": 5,
    "embedding_slot": 26,
    "seed": 998244353
  },
  "scene": {
    "pose_tags": ["sitting"],
    "expression_tags": ["smile"],
    "composition": "upper body",
    "scene_seed": 41
  },
  "output": {
    "matting": "Segment",
    "color_mode": "color",
    "isolation_tag": "simple background",
    "emit_alpha": true
  }
}
```

## 8. ブレ止めの三点と三案の対応

| ブレ止め | 担当 | 由来案 | 層 |
|---|---|---|---|
| タグ固定 | `canonical_tags` (先頭固定) + `forbidden_tags` | 案A 台帳 | authored |
| 埋め込み固定 | `embedding_slot` (NPU 常駐へのハンドル) | 案B | authored (実体は外部) |
| 同一性照合ループ | `CharacterMemory.identity_centroid` (cosine 照合 → 再生成) | 案C | learned (§11) |
| 最終保険 | `seed` | — | authored |

## 9. 未確定事項 (probe / 後続タスク)

- マッティングを乗せる HW (`matting_device`): iGPU / NPU / CPU を probe 比較して決定
- 既定の `isolation_tag` (背景色): `simple background` / `white background` / `grey background`
  を切り抜き品質 (縁スピル・白系キャラのコントラスト) で probe 比較して決定
- `identity_features` の次元・抽出元: WD14 (1536dim) か CLIP pooler (768dim) か
- `embedding_slot` の NPU 埋め込みテーブル構造 (案B の常駐方式)
- 道A (LayerDiffuse 透過生成) の自作カーネル化可否
- 方式B (線画抽出 HW ステージ: Anime2Sketch 等) はバックログ。matting と並ぶ出力段候補 (現状はタグ注入の方式A のみ実装)
- 手検出モデル (§10 L2/L3) の選定と HW (NPU/CPU)、指数カウント手法
- 将来: 背景・合成のプラグイン化。透過PNG境界を交換物として、背景生成は差し替え可能な
  外部バックエンド (Grok/Gemini/SD 等) に委譲し自動合成する任意パス。コアの芯 (ローカルHWで
  キャラ) は不変、手動 CLIP Studio 合成と排他でなく選択肢。宿主は Phase2+ の HTTP サーバ
  (src/server/api.cpp, cpp-httplib)。合成品質 (ライティング/パース馴染ませ) は別課題。character.hpp 非影響。

## 10. 手指問題 (ハード要件)

「指の本数が崩れない」ことを**必達要件**とする。ただしネガティブプロンプトだけでは
保証できないため、3層で担保する。**重要: 「5本」を正解として焼き込まない。**
指の本数はキャラ同一性属性 (`digits_per_hand`) で、最初に決めたら全コマで固定する。

| 層 | 手法 | 実装フェーズ |
|---|---|---|
| L1 予防 | `default_quality_negatives()` を常時注入 (本数非依存の崩れのみ) | 本タスク (器のみ) |
| L2 修復 | 手を検出 → その領域だけ高解像度インペイント再生成 (adetailer 方式)。`refine_hands` で有効化 | Phase 2 |
| L3 検査 | 生成画像の指数を `digits_per_hand` と照合。不一致なら L2 か全再生成。`DIGITS_UNCOUNTABLE` の手は検査スキップ | Phase 2 (案C ループ拡張) |

HW 協調: L2/L3 の手検出は小型モデル → NPU/CPU。インペイント本体は RTX5080。
L3 は WD14 フィードバックループ (案C) の拡張として実装する。

本タスク (character.hpp) の責務は **L1 の器** (`default_quality_negatives()` +
`quality_negatives` フラグ) と **L2/L3 の意思表示** (`refine_hands` フラグ +
`digits_per_hand` フィールド) まで。検出・インペイント・指数カウントの実体は Phase 2。

## 11. 学習層 `CharacterMemory` (別タスク・Phase 2/3)

手・ポーズは静的設定では固定できず、**生成→学習→フィードバック→生成**のループで
精度を上げる。authored 層 (`CharacterIdentity`、人間/LLM が宣言・静的) とは
ライフサイクルが違うため、蓄積状態は別構造 `CharacterMemory` に分離する。
**本タスク (character.hpp) では実装しない。** 学習方式を決める専用タスクで設計する。

ループ概要 (案C の発展形):
```
生成 → QA 検査 (指数 vs digits_per_hand / ポーズ破綻 / 同一性 cosine)
  → 合格: GenerationRecord 追記 + identity_centroid をオンライン更新 (記憶)
  → 次回の seed / pose タグをそれへバイアス (フィードバック)
  → 合格画像が貯まったら Phase3 でバッチ fine-tune (LoRA / 埋め込み / リワード)
↑___________________________________________________________________|
```

`CharacterMemory` 想定フィールド (確定は別タスク):
- `std::vector<float> identity_centroid;`  // 案C 重心 (オンライン更新)
- `std::vector<GenerationRecord> accepted;`  // {seed, pose_tags, score}
- `uint32_t generation_count; uint32_t accept_count;`

**QA 検査の設計指針: 「数・位相」のみ照合し「角度・比率」は見ない (バックログ採用案)**
生成画像を NPU で部位/骨格検出し、検査軸を**ポーズ不変量に絞る**:
- 採用 (頑健): 指数 vs `digits_per_hand` (既存 L3) / 四肢の本数・有無 / 重複・欠損
  (腕3本・頭2つ・目3つ) / **左右の「本数」対称** (左手5・右手4 はおかしい)。
- 不採用 (脆い): 関節角の可動域・プロポーション・ポーズの自然さ・角度の左右対称。
  2D イラスト/漫画はパース・デフォルメ・アオリで容易に誤検出するため soft flag 止まり。
理由: 「数」と「有無」はアングルが変わっても不変。L3 指数検査をカウント全体へ一般化した形。
照合基準は `CharacterIdentity` の宣言値 (例: `digits_per_hand`、必要なら `expected_limbs`)。
`DIGITS_UNCOUNTABLE` と同様、検出できない部位は「検査スキップ」で逃がす。
HW: NPU は拡散 3.8s 中ほぼ遊休 → WD14 と同じ「生成中に裏で並列採点」で実質ゼロコスト。
一次フィルタとして WD14 の danbooru 異常タグ (`extra_digits`/`extra_arms`/`missing_limb`)
を無料で流用し (精度は要検証)、厳密カウントを NPU 検出で詰める二段構成も可。
ポーズ推定モデルは実写訓練が主で 2D 精度に難 → アニメ系 (DWPose 等) を当て、ハードゲート
にせず異常度の高いコマのみ拾う。蒸留スコアラ (下記) の teacher ラベル生成側にもなれる。

学習の2層 (混同しない):
- **記憶層** (retrieval/bandit): 訓練なし。合格 seed/pose/重心の蓄積とバイアス。早期に実装可。
- **本訓練層** (fine-tune): キャラ LoRA / 埋め込み (案B) / 手品質リワード。RTX5080 単機では
  生成と競合するため、合格画像をバッチで offline 学習。Phase3。過大広告しない。

**蒸留オプション (Phase2/3 候補)**: L3 の手・ポーズ品質スコアラは、ループが蓄積した
合格/不合格ラベル (+ 必要なら大型評価器) を teacher として、小型・固定形状の分類ネットへ
**蒸留**できる。固定形状の小分類ネットは NPU の得意分野 (probe2: 512dim MLP=0.88ms) で、
生成の裏で待ち時間ゼロに採点できる。記憶層(データ生成)と蒸留(小モデルへ圧縮)は相補的。
重い SDXL fine-tune に行く前段として有効。teacher 信号=蓄積データが前提なので時期は後。
プロジェクト哲学 (大モデル→自作小モデルを NPU/CPU で) とも一致。

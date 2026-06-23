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

// 切り抜き方式 (CLIP Studio 合成前提)
enum class MattingMode
{
    None,        // 切り抜かない (デバッグ用)
    Segment,     // 道B: 単色生成 → anime-segmentation で α 抽出 (第一候補)
    Native,      // 道A: LayerDiffuse で RGBA 直接生成 (将来研究)
};

// 色モード (方式A: プロンプトタグ注入)
// AI 絵の均質な塗りの違和感を避けたいとき、白黒/線画で出して
// CLIP Studio で人が塗る運用を可能にする。新モデルや後処理は無し。
// 注: 方式B (線画抽出 HW ステージ: Anime2Sketch 等) はバックログ
//   (matting と並ぶ出力段候補)。
enum class ColorMode
{
    Color,       // 通常 (タグ注入なし)
    Greyscale,   // monochrome, greyscale
    Lineart,     // lineart, monochrome, sketch
};

// color_mode に応じて positive へ注入する danbooru タグを返す。
// Color は空 (何も足さない)。テストから直接参照できるよう独立ヘルパにする。
inline std::vector<std::string> color_mode_tags(ColorMode mode)
{
    switch (mode)
    {
    case ColorMode::Color:
        return {};
    case ColorMode::Greyscale:
        return {"monochrome", "greyscale"};
    case ColorMode::Lineart:
        return {"lineart", "monochrome", "sketch"};
    }
    return {};  // 到達しない (安全側: 無注入)
}

// ------------------------------------------------------------
// CharacterIdentity: キャラクター同一性層 (authored)
// 一度決めたらコマ間で「不変」に保つべき設定。
// name は拡散モデルへ渡す文字列ではなく、台帳を引く「主キー」。
// 見た目を決めるのは canonical_tags + embedding。age は外見を決めない。
// 注: 案C の同一性重心 (identity_features) は learned 層 (CharacterMemory) へ
//   移動済みのため、本構造体には持たない (spec §11, 別タスク)。
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
    //   DIGITS_UNCOUNTABLE (=0): 肉球/鉤爪/ミトン等。指数カウント検査を無効化する。
    int  digits_per_hand = 5;                // 片手の指の本数 (5=人間, 4=カートゥーン, 0=非カウント)

    // --- 案B: NPU 常駐埋め込みへのポインタ (ID ハンドル。埋め込み実体は外部) ---
    int32_t  embedding_slot = -1;            // NPU 埋め込みテーブルの slot (-1=未登録)

    // --- 同一性の最終保険 ---
    uint64_t seed = 0;                       // 0 = ランダム許可

    // --- 管理メタ ---
    uint32_t schema_version = 1;             // 台帳スキーマ版
};

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

// ------------------------------------------------------------
// OutputSpec: 出力・合成設定
// 背景は生成しない。キャラのみを透過 PNG で出す。
// matting_device は M-5 probe で "iGPU" 確定 (ISNet 1024²: iGPU 99.96ms 最速・NPU 142.96 / CPU 204.20 / RTX5080-OV 220.47)。
// 既定は "" のまま (= 自動。M-6 PipelineGenerator 結線で確定値を消費する)。
// ------------------------------------------------------------
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

// ------------------------------------------------------------
// CharacterBible: 全キャラクターの台帳
// LLM (将来 BitNet) が name をキーに引いて同一性層を解決する。
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

// 肉球/鉤爪/ミトン: 指数カウント検査を無効化する番兵値
constexpr int DIGITS_UNCOUNTABLE = 0;

// L1: 普遍的な品質ネガティブ (キャラ非依存・本数非依存)。
// per-character の forbidden_tags とは別物として扱う。
// 指の本数を仮定する語 (extra fingers / fewer digits 等) は入れない。
inline std::vector<std::string> default_quality_negatives()
{
    return {
        "bad hands", "malformed hands", "mutated hands",
        "fused fingers", "bad anatomy", "deformed",
        "lowres", "worst quality", "jpeg artifacts",
    };
}

// ------------------------------------------------------------
// PromptParts: プロンプト合成結果
// positive / negative タグ列と確定 seed を保持する。
// ------------------------------------------------------------
struct PromptParts
{
    std::vector<std::string> positive;
    std::vector<std::string> negative;
    uint64_t seed;
};

// ------------------------------------------------------------
// compose_prompt: 同一性層 → シーン層の優先順位でプロンプトを合成する。
// canonical_tags を先頭に置くのがブレ止めの肝 (CLIP は前方トークンが効く)。
// age は外見へ自動変換しない (compile_age_tags は存在しない, spec §5)。
// composition / isolation_tag は空文字なら positive に追加しない。
// 色モードタグ (方式A) は canonical_tags の直後・scene タグの前に注入する
//   (同一性 #1 を維持しつつ、画風タグを前方に置いて全体に効かせる)。
// ------------------------------------------------------------
inline PromptParts compose_prompt(const CharacterIdentity& identity,
                                  const SceneSpec& scene,
                                  const OutputSpec& output)
{
    PromptParts parts;

    // positive = canonical_tags + [color_mode_tags] + pose_tags + expression_tags
    //          + [composition] + [isolation_tag]
    parts.positive.insert(parts.positive.end(),
                          identity.canonical_tags.begin(),
                          identity.canonical_tags.end());

    // 色モードタグ注入: canonical 直後・scene タグの前 (Color は空なので無注入)
    const auto color_tags = color_mode_tags(output.color_mode);
    parts.positive.insert(parts.positive.end(),
                          color_tags.begin(), color_tags.end());

    parts.positive.insert(parts.positive.end(),
                          scene.pose_tags.begin(), scene.pose_tags.end());
    parts.positive.insert(parts.positive.end(),
                          scene.expression_tags.begin(), scene.expression_tags.end());

    // composition は空文字なら追加しない
    if (!scene.composition.empty())
    {
        parts.positive.push_back(scene.composition);
    }
    // isolation_tag は空文字なら追加しない
    if (!output.isolation_tag.empty())
    {
        parts.positive.push_back(output.isolation_tag);
    }

    // negative = forbidden_tags + (quality_negatives ? default_quality_negatives() : {})
    parts.negative.insert(parts.negative.end(),
                          identity.forbidden_tags.begin(),
                          identity.forbidden_tags.end());
    if (output.quality_negatives)
    {
        const auto qn = default_quality_negatives();
        parts.negative.insert(parts.negative.end(), qn.begin(), qn.end());
    }

    // seed = identity.seed (!=0) ? identity.seed : scene.scene_seed
    parts.seed = identity.seed != 0 ? identity.seed : scene.scene_seed;

    return parts;
}

} // namespace dollama

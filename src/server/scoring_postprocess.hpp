// 採点後処理 — 生成された RGB8 画像を ScorerNet (IScorer) で「安全に採点」し、
// 異常軸を soft に収集する純 cpp グルー (Phase 4 Model B / B-5-2)。
//
// ----------------------------------------------------------------
// 設計意図 (matting_postprocess.hpp と対称):
//   生成器 (Txt2ImgGenerator / PipelineGenerator) の最終段に、IScorer 越しの
//   品質採点を差し込むための中間層。本ヘッダは IScorer (純 cpp interface) 越しに
//   採点を呼ぶため CUDA/OV 非依存に保たれ、CUDA TU (.cu) からも OV TU (.cpp) からも
//   include できる (実体 = QualityScorer/OpenVINO は scorer_runner.cpp に隔離)。
//
//   フォールバック方針 (matting と同じ「生成を絶対に失敗させない」):
//     - scorer == nullptr      → 未採点 (scored=false)。品質採点なしで生成は成功させる。
//     - score() が例外を投げた  → 未採点 (scored=false・catch して握りつぶす)。
//   いずれのフォールバックでも例外を呼び出し側へ伝播させない。
//
//   collect_anomalous_axes は ScorerResult (sigmoid 済み確率) を引数で受けるだけの
//   純 C++ なので、OV 無効ビルドの開発機でもテスト実走できる
//   (matting.hpp::compose_rgba が OV ガード外でテスト実走できるのと同方針)。
// ----------------------------------------------------------------
#pragma once

#include <cstdint>
#include <vector>

#include "infer/quality_gate.hpp"    // AnomalyAxis
#include "server/scorer_runner.hpp"  // IScorer / ScorerResult (再利用)

namespace dollama
{

// ------------------------------------------------------------
// ScoringOutcome: 採点結果 (採点したか否かと結果)
//   scored == true  のとき result が有効。
//   scored == false は未採点 (scorer 無効 or 採点失敗)。result は全 0。
// ------------------------------------------------------------
struct ScoringOutcome
{
    bool         scored = false; // true のとき result が有効
    ScorerResult result = {};    // scored==false は全 0 (未採点)
};

// ------------------------------------------------------------
// score_image_safe: RGB8 生成画像を IScorer で安全に採点する。
//   scorer : 有効なら IScorer::score で採点。nullptr なら未採点。
//   rgb    : [w*h*3] RGB8 (行優先)。
//   w,h    : 解像度。
// matting の encode_png_maybe_transparent と同じく「生成を絶対に失敗させない」方針:
//   - scorer == nullptr → {scored=false} (品質採点なし)。
//   - score() が例外を投げた → {scored=false} (catch して握りつぶす)。
// サイズ不一致 (rgb.size() != w*h*3 等) は QualityScorer 側が std::invalid_argument を
// 投げるので、ここで別途検証はせず本 catch がまとめて握る (未採点へ倒す)。
// 例外は呼び出し側へ伝播させない。
// ------------------------------------------------------------
inline ScoringOutcome score_image_safe(IScorer* scorer,
                                       const std::vector<uint8_t>& rgb,
                                       int w, int h)
{
    // scorer 無効 → 未採点 (品質採点なしで生成は成功させる)。
    if (scorer == nullptr)
    {
        return {};
    }

    try
    {
        return {true, scorer->score(rgb, w, h)};
    }
    catch (...)
    {
        // サイズ不一致 (std::invalid_argument) を含む採点失敗はすべてここで握り、
        // 未採点へ倒す (例外は呼び出し側へ伝播させない)。
        return {};
    }
}

// ------------------------------------------------------------
// AxisFlag: 閾値を超えた 1 軸ぶんの異常フラグ (soft)
//   axis  : 異常軸 (AnomalyAxis)
//   score : その軸の sigmoid 確率 [0,1]
// ------------------------------------------------------------
struct AxisFlag
{
    AnomalyAxis axis;
    float       score; // sigmoid 確率 [0,1]
};

// ------------------------------------------------------------
// collect_anomalous_axes: ScorerResult の 8 軸から閾値以上の軸を soft に収集する。
//   r         : ScorerResult (sigmoid 済み確率)。axis[0..7] が AnomalyAxis 順。
//   threshold : この値以上 (境界含む) の軸を拾う。
// 返り値は将来の FB ループ (B-5) の入力かつログ用 (§11「異常度の高いコマのみ拾う」)。
//   - 棄却しない・並べ替えない (axis[i] を i=0..7 の昇順のまま収集)。
//   - r.quality は flag 対象外。B-3b で quality head 凍結中ゆえ採点に使わないため。
//   - threshold 既定 0.5 は調整可能なノブで load-bearing でない。quality_gate は 0.4 だが
//     ScorerNet の軸とは別物 (WD14 スコア vs ScorerNet logit→sigmoid) なので合わせない。
// ------------------------------------------------------------
inline std::vector<AxisFlag> collect_anomalous_axes(const ScorerResult& r,
                                                    float threshold = 0.5f)
{
    std::vector<AxisFlag> flags;
    for (int i = 0; i < 8; ++i)
    {
        // 閾値以上 (境界含む) を拾う。並べ替えず enum 順のまま収集する。
        if (r.axis[i] >= threshold)
        {
            flags.push_back({static_cast<AnomalyAxis>(i), r.axis[i]});
        }
    }
    return flags;
}

// ------------------------------------------------------------
// axis_name: AnomalyAxis を表示用文字列へ変換する (純 cpp・ログ用)。
//   2 生成器 (Txt2Img/Pipeline) の採点ログで同じ switch を二重に持たないための置き場。
//   enum 名は infer/quality_gate.hpp の AnomalyAxis と厳密一致させること。
// ------------------------------------------------------------
inline const char* axis_name(AnomalyAxis axis)
{
    switch (axis)
    {
    case AnomalyAxis::Hands:         return "Hands";
    case AnomalyAxis::Limbs:         return "Limbs";
    case AnomalyAxis::Head:          return "Head";
    case AnomalyAxis::Eyes:          return "Eyes";
    case AnomalyAxis::Ears:          return "Ears";
    case AnomalyAxis::Mouth:         return "Mouth";
    case AnomalyAxis::Digits:        return "Digits";
    case AnomalyAxis::GlobalAnatomy: return "GlobalAnatomy";
    }
    return "Unknown";
}

} // namespace dollama

// 採点後処理グルー (scoring_postprocess.hpp) 単体テスト — Phase 4 Model B / B-5-2
//
// 純 cpp (OV/CUDA 非依存) ゆえ開発機でも SKIP なし実走する。fake IScorer をヘッダ内に
// 定義し、score_image_safe のフォールバック契約と collect_anomalous_axes の閾値挙動を検証する。
//
// 全テスト通過時は "[test_scoring_postprocess] ALL PASSED" を出力して return 0。
// 失敗時は std::cerr に出力して return 1。

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "server/scoring_postprocess.hpp" // ScoringOutcome / score_image_safe / collect_anomalous_axes

namespace
{

int g_fail = 0;

void check(bool cond, const std::string& msg)
{
    if (!cond)
    {
        std::cerr << "[test_scoring_postprocess] FAIL: " << msg << std::endl;
        ++g_fail;
    }
}

using dollama::IScorer;
using dollama::ScorerResult;

// 既知の ScorerResult を返す fake (成功経路の検証用)。
struct FakeScorer : IScorer
{
    ScorerResult fixed; // score() が常に返す固定値

    explicit FakeScorer(const ScorerResult& r) : fixed(r) {}

    ScorerResult score(const std::vector<uint8_t>& /*rgb*/, int /*w*/, int /*h*/) override
    {
        return fixed;
    }
};

// 必ず例外を投げる fake (例外フォールバックの検証用)。
struct ThrowingScorer : IScorer
{
    ScorerResult score(const std::vector<uint8_t>& /*rgb*/, int /*w*/, int /*h*/) override
    {
        throw std::invalid_argument("ThrowingScorer: 採点失敗 (テスト用例外)");
    }
};

} // namespace

int main()
{
    using namespace dollama;

    const int            w = 8;
    const int            h = 8;
    std::vector<uint8_t> rgb(static_cast<size_t>(w) * h * 3, 128); // ダミー RGB8

    // ------------------------------------------------------------
    // 1. null フォールバック: scorer == nullptr → 未採点。
    // ------------------------------------------------------------
    {
        ScoringOutcome out = score_image_safe(nullptr, rgb, w, h);
        check(out.scored == false, "nullptr scorer は scored=false であるべき");
        // result は全 0 (ScoringOutcome の既定値)。
        check(out.result.quality == 0.0f, "未採点の result.quality は 0 であるべき");
        for (int a = 0; a < 8; ++a)
        {
            check(out.result.axis[a] == 0.0f,
                  "未採点の result.axis[" + std::to_string(a) + "] は 0 であるべき");
        }
    }

    // ------------------------------------------------------------
    // 2. 例外フォールバック: score() が throw → 未採点・result 全 0。
    // ------------------------------------------------------------
    {
        ThrowingScorer  ts;
        ScoringOutcome  out = score_image_safe(&ts, rgb, w, h);
        check(out.scored == false, "throw する scorer は scored=false であるべき");
        check(out.result.quality == 0.0f, "例外時の result.quality は 0 であるべき");
        for (int a = 0; a < 8; ++a)
        {
            check(out.result.axis[a] == 0.0f,
                  "例外時の result.axis[" + std::to_string(a) + "] は 0 であるべき");
        }
    }

    // ------------------------------------------------------------
    // 3. 成功経路: 既知の ScorerResult を返す fake → scored=true かつ result 一致。
    // ------------------------------------------------------------
    {
        ScorerResult known;
        known.quality = 0.73f;
        for (int a = 0; a < 8; ++a)
        {
            known.axis[a] = 0.1f * static_cast<float>(a + 1); // 0.1, 0.2, ... 0.8
        }

        FakeScorer     fs(known);
        ScoringOutcome out = score_image_safe(&fs, rgb, w, h);

        check(out.scored == true, "有効 scorer は scored=true であるべき");
        const float eps = 1e-6f;
        check(std::fabs(out.result.quality - known.quality) < eps,
              "成功経路で result.quality が一致すべき");
        for (int a = 0; a < 8; ++a)
        {
            check(std::fabs(out.result.axis[a] - known.axis[a]) < eps,
                  "成功経路で result.axis[" + std::to_string(a) + "] が一致すべき");
        }
    }

    // ------------------------------------------------------------
    // 4. flag 閾値: 閾値の上/下/同値を混ぜた ScorerResult で collect_anomalous_axes を検証。
    //    threshold は >= (境界含む) で拾い、enum 順のまま並ぶこと。
    // ------------------------------------------------------------
    {
        const float threshold = 0.5f;

        ScorerResult r;
        r.quality = 0.0f;
        r.axis[0] = 0.9f;  // Hands         : 上回り → 拾う
        r.axis[1] = 0.4f;  // Limbs         : 下回り → 拾わない
        r.axis[2] = 0.5f;  // Head          : 同値   → 拾う (境界含む)
        r.axis[3] = 0.49f; // Eyes          : 直下   → 拾わない
        r.axis[4] = 0.7f;  // Ears          : 上回り → 拾う
        r.axis[5] = 0.0f;  // Mouth         : 下回り → 拾わない
        r.axis[6] = 0.51f; // Digits        : 上回り → 拾う
        r.axis[7] = 0.5f;  // GlobalAnatomy : 同値   → 拾う (境界含む)

        std::vector<AxisFlag> flags = collect_anomalous_axes(r, threshold);

        // 期待: Hands, Head, Ears, Digits, GlobalAnatomy の 5 軸が enum 順に並ぶ。
        check(flags.size() == 5, "閾値 0.5 で 5 軸が拾われるべき");

        if (flags.size() == 5)
        {
            check(flags[0].axis == AnomalyAxis::Hands, "flag[0] は Hands であるべき");
            check(flags[1].axis == AnomalyAxis::Head, "flag[1] は Head であるべき");
            check(flags[2].axis == AnomalyAxis::Ears, "flag[2] は Ears であるべき");
            check(flags[3].axis == AnomalyAxis::Digits, "flag[3] は Digits であるべき");
            check(flags[4].axis == AnomalyAxis::GlobalAnatomy,
                  "flag[4] は GlobalAnatomy であるべき");

            // score が対応軸の値で写っていること。
            const float eps = 1e-6f;
            check(std::fabs(flags[0].score - 0.9f) < eps, "flag[0].score は 0.9 であるべき");
            check(std::fabs(flags[2].score - 0.7f) < eps, "flag[2].score は 0.7 であるべき");
        }

        // 全軸が閾値を下回るケース → 空集合。
        ScorerResult low;
        low.quality = 0.0f;
        for (int a = 0; a < 8; ++a)
        {
            low.axis[a] = 0.1f;
        }
        std::vector<AxisFlag> none = collect_anomalous_axes(low, threshold);
        check(none.empty(), "全軸が閾値未満なら空であるべき");
    }

    // ------------------------------------------------------------
    // 5. quality 非 flag: quality を閾値超にしても collect_anomalous_axes に影響しない。
    //    quality は flag 対象外 (B-3b で head 凍結中ゆえ採点に使わない)。
    // ------------------------------------------------------------
    {
        ScorerResult r;
        r.quality = 0.99f; // 高い quality だが flag には寄与しないはず
        for (int a = 0; a < 8; ++a)
        {
            r.axis[a] = 0.0f; // 全軸下回り
        }
        std::vector<AxisFlag> flags = collect_anomalous_axes(r, 0.5f);
        check(flags.empty(), "quality が高くても軸が下回れば空であるべき");
    }

    // ------------------------------------------------------------
    // 6. axis_name: AnomalyAxis 8 軸が想定文字列へ写ること (純 cpp・ログ用)。
    // ------------------------------------------------------------
    {
        check(std::string(axis_name(AnomalyAxis::Hands)) == "Hands", "axis_name Hands");
        check(std::string(axis_name(AnomalyAxis::Limbs)) == "Limbs", "axis_name Limbs");
        check(std::string(axis_name(AnomalyAxis::Head)) == "Head", "axis_name Head");
        check(std::string(axis_name(AnomalyAxis::Eyes)) == "Eyes", "axis_name Eyes");
        check(std::string(axis_name(AnomalyAxis::Ears)) == "Ears", "axis_name Ears");
        check(std::string(axis_name(AnomalyAxis::Mouth)) == "Mouth", "axis_name Mouth");
        check(std::string(axis_name(AnomalyAxis::Digits)) == "Digits", "axis_name Digits");
        check(std::string(axis_name(AnomalyAxis::GlobalAnatomy)) == "GlobalAnatomy",
              "axis_name GlobalAnatomy");
    }

    if (g_fail == 0)
    {
        std::cout << "[test_scoring_postprocess] ALL PASSED" << std::endl;
        return 0;
    }
    std::cerr << "[test_scoring_postprocess] " << g_fail << " checks FAILED" << std::endl;
    return 1;
}

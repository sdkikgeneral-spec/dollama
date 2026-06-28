// IScorer / make_scorer (ScorerNet 品質スコアラ・純 cpp 結線) 契約テスト — B-5-1
//
// 開発機は OV C++ 無 + SAC (Smart App Control) で新規 exe の実走が取れないため、
// 本テストのバーは「実走緑」ではなく「build 緑 + nullptr 契約検証」とする
// (実走は OV+CUDA 研究機マージ時)。make_scorer の nullptr 契約を主に検証する:
//
//   1. make_scorer("", device) → nullptr (空 xml 契約。OV 有無に関わらず成立)。
//   2. stub ビルド (HAVE_OPENVINO 未定義) では任意 xml でも nullptr。
//   3. logits_to_result 再利用の健全性 (sigmoid 変換が ScorerResult を正しく埋める軽い sanity)。
//
// scorer_runner(_stub).cpp を test exe にリンクすることで make_scorer の実体を解決する
// (test_matting の matter_runner(_stub).cpp 切替に倣う)。
//
// 全テスト通過時は "[test_scorer_runner] ALL PASSED" を出力して return 0。
// 失敗時は std::cerr に出力して return 1。

#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "server/scorer_runner.hpp" // IScorer / make_scorer / ScorerResult (再利用)

namespace
{

int g_fail = 0;

void check(bool cond, const std::string& msg)
{
    if (!cond)
    {
        std::cerr << "[test_scorer_runner] FAIL: " << msg << std::endl;
        ++g_fail;
    }
}

} // namespace

int main()
{
    using namespace dollama;

    // ------------------------------------------------------------
    // 1. 空 xml は OV 有無に関わらず nullptr (空 xml 契約)。
    //    OV 実体側でも model_xml.empty() を QualityScorer 構築前に弾く。
    // ------------------------------------------------------------
    {
        std::unique_ptr<IScorer> s = make_scorer("", "CPU");
        check(s == nullptr, "make_scorer(\"\", \"CPU\") は nullptr であるべき");

        std::unique_ptr<IScorer> s2 = make_scorer("", "NPU");
        check(s2 == nullptr, "make_scorer(\"\", \"NPU\") は nullptr であるべき");
    }

    // ------------------------------------------------------------
    // 2. ロード不能 / 不在の xml も nullptr に倒れる (例外を投げない)。
    //    - OV 有効ビルド: read_model 失敗 → catch(...) → nullptr。
    //    - OV 無効 (stub) ビルド: 任意 xml でも常に nullptr。
    //    いずれのビルドでも結果は nullptr で一致するため、ビルド種別に依らず検証できる。
    // ------------------------------------------------------------
    {
        std::unique_ptr<IScorer> s =
            make_scorer("E:/__definitely_missing__/scorer_net.xml", "CPU");
        check(s == nullptr, "不在 xml の make_scorer は nullptr であるべき (例外なし)");
    }

    // ------------------------------------------------------------
    // 3. logits_to_result (quality_scorer.hpp・OV 非依存) 再利用の健全性。
    //    IScorer::score の実体が呼ぶ変換ヘルパ。sigmoid(0)=0.5 を軸ごとに確認する。
    // ------------------------------------------------------------
    {
        std::vector<float> logits9(9, 0.0f); // 全 0 → sigmoid すべて 0.5
        ScorerResult r = logits_to_result(logits9);

        const float eps = 1e-6f;
        check(std::fabs(r.quality - 0.5f) < eps, "logits 0 で quality は 0.5 であるべき");
        for (int a = 0; a < 8; ++a)
        {
            check(std::fabs(r.axis[a] - 0.5f) < eps,
                  "logits 0 で axis[" + std::to_string(a) + "] は 0.5 であるべき");
        }

        // 大きい正/負 logit が [0,1] 端へ漸近すること (単調性の軽い確認)。
        std::vector<float> logits_pos(9, 10.0f);
        ScorerResult rp = logits_to_result(logits_pos);
        check(rp.quality > 0.99f, "大きい正 logit で quality は ~1 に漸近すべき");

        std::vector<float> logits_neg(9, -10.0f);
        ScorerResult rn = logits_to_result(logits_neg);
        check(rn.axis[0] < 0.01f, "大きい負 logit で axis[0] は ~0 に漸近すべき");
    }

    if (g_fail == 0)
    {
        std::cout << "[test_scorer_runner] ALL PASSED" << std::endl;
        return 0;
    }
    std::cerr << "[test_scorer_runner] " << g_fail << " checks FAILED" << std::endl;
    return 1;
}

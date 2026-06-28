// make_scorer のスタブ実装 (純 cpp・OpenVINO 無効ビルド用) — B-5-1
//
// OpenVINO 無効ビルド (openvino_dep 未解決) では quality_scorer.hpp の QualityScorer
// (OV 依存) をコンパイルできないため、scorer_runner.cpp の代わりにこの cpp スタブを
// リンクする。make_scorer は常に nullptr を返し、呼び出し側 (将来の生成器結線) が
// 「品質採点なし」へフォールバックする。これにより OV 無効でも採点結線を含む
// test / exe がリンク・実行できる。
#include "server/scorer_runner.hpp"

namespace dollama
{

std::unique_ptr<IScorer> make_scorer(const std::string& model_xml,
                                     const std::string& device)
{
    // OV 無効: 品質採点は使えない。常に nullptr (→ 採点なしフォールバック)。
    (void)model_xml;
    (void)device;
    return nullptr;
}

} // namespace dollama

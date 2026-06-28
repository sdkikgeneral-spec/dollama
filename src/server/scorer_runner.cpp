// IScorer の OpenVINO 実体 — QualityScorer (ScorerNet) をラップして make_scorer を実装する。
//   (Phase 4 Model B / B-5-1)
//
// 依存隔離:
//   本 .cpp は OV 依存を内部に閉じ込める。OV 有効ビルド (openvino_dep.found()) のとき
//   test / exe にリンクされる。OV 無効ビルドでは scorer_runner_stub.cpp が代わりに
//   リンクされ make_scorer は常に nullptr を返す (両ビルドで scorer_runner.hpp の
//   宣言は満たされる)。
//
// HAVE_OPENVINO の二重ガード:
//   meson は OV 有効時にこの .cpp をリンクするが、念のため #ifdef HAVE_OPENVINO で
//   実体を囲い、未定義時 (誤って OV 無で本 .cpp がリンクされた場合) でも nullptr 返しで
//   コンパイル・リンクが通るようにする。
#include "server/scorer_runner.hpp"

#ifdef HAVE_OPENVINO

#include "infer/quality_scorer.hpp" // QualityScorer (OV 依存・HAVE_OPENVINO ガード越し)

namespace dollama
{

namespace
{

// IScorer の OV 実体。QualityScorer を値で保持し score() で ScorerResult を返す。
class ScorerRunner : public IScorer
{
public:
    ScorerRunner(const std::string& model_xml, const std::string& device)
        : scorer_(model_xml, device)
    {
    }

    ScorerResult score(const std::vector<uint8_t>& rgb, int w, int h) override
    {
        // QualityScorer::infer は内部で 512 へ前処理し logits[9] を返す。
        //   matter のような mask リサイズ後処理は不要 (出力は固定 9 要素のスコア)。
        std::vector<float> logits = scorer_.infer(rgb, w, h);
        return logits_to_result(logits); // sigmoid 済み ScorerResult へ変換
    }

private:
    QualityScorer scorer_;
};

} // namespace

std::unique_ptr<IScorer> make_scorer(const std::string& model_xml,
                                     const std::string& device)
{
    // モデル xml 未指定なら構築しない (nullptr 契約)。
    //   QualityScorer 構築前に弾く (空 xml で read_model("") を投げさせない)。
    if (model_xml.empty())
    {
        return nullptr;
    }
    try
    {
        return std::make_unique<ScorerRunner>(model_xml, device);
    }
    catch (...)
    {
        // モデル不在 / ロード失敗 / デバイス未対応 等は nullptr に倒す。
        return nullptr;
    }
}

} // namespace dollama

#else // HAVE_OPENVINO 未定義

namespace dollama
{

// OV 無効でこの .cpp がリンクされた場合でもリンクが通るよう nullptr を返す。
std::unique_ptr<IScorer> make_scorer(const std::string& model_xml,
                                     const std::string& device)
{
    (void)model_xml;
    (void)device;
    return nullptr;
}

} // namespace dollama

#endif // HAVE_OPENVINO

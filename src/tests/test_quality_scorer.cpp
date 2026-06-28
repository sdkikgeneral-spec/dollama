// QualityScorer (ScorerNet 品質スコアラ) 単体テスト + 推論レイテンシ計測 (Phase 4 Model B / B-3e)
// HAVE_OPENVINO 未定義時は実 QualityScorer 推論を [SKIP] するが、OV 非依存テスト
// (logits_to_result の sigmoid 変換・軸対応・サイズガード) は常に実行する。
// モデルファイル / golden が存在しない場合も実 QualityScorer 推論は [SKIP]。
// 全テスト通過時は "[test_quality_scorer] ALL PASSED" を出力して return 0。
// 失敗時は std::cerr に出力して return 1。
//
// golden (src/tests/data/scorer/golden_scorernet.safetensors):
//   input  [1,3,512,512] F32 = 前処理済み NCHW [0,1]  → QualityScorer::infer_nchw に渡す
//   logits [1,9]         F32 = OV IR(FP32) CPU の期待出力 → max abs err で突合
// golden は同じ FP32 IR の出力を正とするため、C++ QualityScorer は高精度一致するはず。
//
// パス埋め込み (cwd 非依存): meson の cpp_args で
//   -DSCORER_MODEL_XML="..." / -DSCORER_GOLDEN_PATH="..." を渡す。

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "infer/quality_scorer.hpp" // logits_to_result は OV 非依存 (常に使える)
#include "io/safetensors.hpp"

#ifndef SCORER_MODEL_XML
#define SCORER_MODEL_XML "models/scorer-net/model_ov_fp32.xml"
#endif
#ifndef SCORER_GOLDEN_PATH
#define SCORER_GOLDEN_PATH "src/tests/data/scorer/golden_scorernet.safetensors"
#endif

namespace fs = std::filesystem;

namespace dollama
{

// 参照用の sigmoid (テストの手計算突合に使う)。
static double ref_sigmoid(double x)
{
    return 1.0 / (1.0 + std::exp(-x));
}

// ----------------------------------------------------------------
// OV 非依存テスト①: logits_to_result の sigmoid 値が手計算と一致
//   既知 logits を流し、quality / 各 axis が ref_sigmoid と一致することを確認する。
// ----------------------------------------------------------------
static bool test_logits_to_result_values()
{
    // index0=quality, index1..8=8軸。値は範囲が広めに散るよう選ぶ。
    std::vector<float> logits = {0.0f, 2.0f, -2.0f, 5.0f, -5.0f, 1.0f, -1.0f, 0.5f, -0.5f};

    ScorerResult r = logits_to_result(logits);

    const double tol = 1e-6;

    // quality = sigmoid(logits[0])
    if (std::fabs(r.quality - ref_sigmoid(logits[0])) > tol)
    {
        std::cerr << "[test_logits_to_result_values] quality 不一致: " << r.quality
                  << " vs " << ref_sigmoid(logits[0]) << "\n";
        return false;
    }
    // axis[a] = sigmoid(logits[1+a])
    for (int a = 0; a < 8; ++a)
    {
        double expect = ref_sigmoid(logits[1 + a]);
        if (std::fabs(r.axis[a] - expect) > tol)
        {
            std::cerr << "[test_logits_to_result_values] axis[" << a << "] 不一致: "
                      << r.axis[a] << " vs " << expect << "\n";
            return false;
        }
    }

    std::cout << "[test_logits_to_result_values] PASSED  (sigmoid 手計算一致)\n";
    return true;
}

// ----------------------------------------------------------------
// OV 非依存テスト②: 全 axis と quality が [0,1] に収まる
// ----------------------------------------------------------------
static bool test_result_range()
{
    // 極端値も含めて [0,1] に収まることを確認する。
    std::vector<float> logits = {100.0f, -100.0f, 50.0f, -50.0f, 0.0f, 7.0f, -7.0f, 3.3f, -3.3f};
    ScorerResult r = logits_to_result(logits);

    if (r.quality < 0.0f || r.quality > 1.0f)
    {
        std::cerr << "[test_result_range] quality 値域外: " << r.quality << "\n";
        return false;
    }
    for (int a = 0; a < 8; ++a)
    {
        if (r.axis[a] < 0.0f || r.axis[a] > 1.0f)
        {
            std::cerr << "[test_result_range] axis[" << a << "] 値域外: " << r.axis[a] << "\n";
            return false;
        }
    }

    std::cout << "[test_result_range] PASSED  (quality/axis すべて [0,1])\n";
    return true;
}

// ----------------------------------------------------------------
// OV 非依存テスト③: index → 軸対応が正しい
//   logits[1] が axis[0]=Hands に、logits[8] が axis[7]=GlobalAnatomy に来ること。
//   各軸を一意な値にして、対応がズレていないことを検証する。
// ----------------------------------------------------------------
static bool test_axis_mapping()
{
    // index1..8 に単調に異なる logit を入れ、sigmoid 後も単調で対応が追えるようにする。
    std::vector<float> logits = {0.0f, -4.0f, -3.0f, -2.0f, -1.0f, 1.0f, 2.0f, 3.0f, 4.0f};
    ScorerResult r = logits_to_result(logits);

    const double tol = 1e-6;
    // AnomalyAxis 順 (Hands/Limbs/Head/Eyes/Ears/Mouth/Digits/GlobalAnatomy) = axis[0..7]
    for (int a = 0; a < 8; ++a)
    {
        double expect = ref_sigmoid(logits[1 + a]);
        if (std::fabs(r.axis[a] - expect) > tol)
        {
            std::cerr << "[test_axis_mapping] axis[" << a << "] が logits[" << (1 + a)
                      << "] に対応していない\n";
            return false;
        }
    }

    // 具体的に: axis[0] (Hands) は logits[1]=-4.0 由来で小さく、axis[7] (GlobalAnatomy) は
    // logits[8]=4.0 由来で大きいはず (対応が逆順なら破綻する)。
    if (!(r.axis[0] < 0.1f && r.axis[7] > 0.9f))
    {
        std::cerr << "[test_axis_mapping] index→軸対応が逆: axis[0]=" << r.axis[0]
                  << " axis[7]=" << r.axis[7] << "\n";
        return false;
    }

    // AnomalyAxis enum 順が 0..7 であること (quality_gate.hpp と同期している前提)。
    static_assert(static_cast<int>(AnomalyAxis::Hands) == 0, "Hands は軸 0");
    static_assert(static_cast<int>(AnomalyAxis::GlobalAnatomy) == 7, "GlobalAnatomy は軸 7");

    std::cout << "[test_axis_mapping] PASSED  (Hands=axis[0] .. GlobalAnatomy=axis[7])\n";
    return true;
}

// ----------------------------------------------------------------
// OV 非依存テスト④: size != 9 で invalid_argument
// ----------------------------------------------------------------
static bool test_logits_size_guard()
{
    bool ok = true;

    for (size_t bad_size : {static_cast<size_t>(0), static_cast<size_t>(8), static_cast<size_t>(10)})
    {
        std::vector<float> bad(bad_size, 0.0f);
        bool threw = false;
        try
        {
            logits_to_result(bad);
        }
        catch (const std::invalid_argument&)
        {
            threw = true;
        }
        catch (const std::exception& e)
        {
            std::cerr << "[test_logits_size_guard] 想定外の例外 (size=" << bad_size
                      << "): " << e.what() << "\n";
            ok = false;
        }
        if (!threw)
        {
            std::cerr << "[test_logits_size_guard] size=" << bad_size
                      << " で例外が飛ばなかった\n";
            ok = false;
        }
    }

    if (ok)
    {
        std::cout << "[test_logits_size_guard] PASSED  (size!=9 で invalid_argument)\n";
    }
    return ok;
}

#ifdef HAVE_OPENVINO

using Clock    = std::chrono::steady_clock;
using MsDouble = std::chrono::duration<double, std::milli>;

// golden safetensors から input [1,3,512,512] と logits [1,9] を float 配列で取り出す。
static bool load_golden(std::vector<float>& input, std::vector<float>& logits)
{
    SafeTensors st(SCORER_GOLDEN_PATH);

    if (!st.contains("input") || !st.contains("logits"))
    {
        std::cerr << "[load_golden] golden に input/logits テンソルがない\n";
        return false;
    }
    if (st.dtype("input") != StDtype::F32 || st.dtype("logits") != StDtype::F32)
    {
        std::cerr << "[load_golden] golden の dtype が F32 でない\n";
        return false;
    }

    size_t nb = 0;
    const uint8_t* pin = st.tensor_bytes("input", nb);
    if (nb != QualityScorer::INPUT_NUMEL * sizeof(float))
    {
        std::cerr << "[load_golden] input バイト数不正: " << nb << "\n";
        return false;
    }
    input.resize(QualityScorer::INPUT_NUMEL);
    std::memcpy(input.data(), pin, nb);

    const uint8_t* plg = st.tensor_bytes("logits", nb);
    if (nb != static_cast<size_t>(QualityScorer::N_OUT) * sizeof(float))
    {
        std::cerr << "[load_golden] logits バイト数不正: " << nb << "\n";
        return false;
    }
    logits.resize(QualityScorer::N_OUT);
    std::memcpy(logits.data(), plg, nb);
    return true;
}

// max abs err: 最大絶対誤差
static double compute_max_abs_err(const std::vector<float>& a, const std::vector<float>& b)
{
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        double d = std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        if (d > m)
        {
            m = d;
        }
    }
    return m;
}

// ----------------------------------------------------------------
// テスト: golden 突合 (max abs err)
// golden input を QualityScorer に通し、出力 logits を golden logits と突合する。
// 同じ FP32 IR の出力が正なので高精度一致するはず (B-3c 実測 ~1e-5)。
// ----------------------------------------------------------------
static bool test_golden_logits(QualityScorer& scorer,
                               const std::vector<float>& gin,
                               const std::vector<float>& glogits)
{
    std::vector<float> out = scorer.infer_nchw(gin);

    if (out.size() != glogits.size())
    {
        std::cerr << "[test_golden_logits] 出力サイズ不一致: got=" << out.size()
                  << " expected=" << glogits.size() << "\n";
        return false;
    }

    const double max_abs_err = compute_max_abs_err(out, glogits);
    const double err_thr      = 1e-3; // 同一 FP32 IR の再現ゆえ厳しめ (実測 ~1e-5 想定)

    std::cout << "[test_golden_logits] max_abs_err=" << max_abs_err
              << " (thr<=" << err_thr << ")\n";

    if (max_abs_err > err_thr)
    {
        std::cerr << "[test_golden_logits] max_abs_err が閾値超過\n";
        return false;
    }

    std::cout << "[test_golden_logits] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// テスト: 出力長 9
// ----------------------------------------------------------------
static bool test_output_length(QualityScorer& scorer, const std::vector<float>& gin)
{
    std::vector<float> out = scorer.infer_nchw(gin);
    if (out.size() != static_cast<size_t>(QualityScorer::N_OUT))
    {
        std::cerr << "[test_output_length] 出力長が " << QualityScorer::N_OUT
                  << " でない: " << out.size() << "\n";
        return false;
    }
    std::cout << "[test_output_length] PASSED  (N_OUT=" << out.size() << ")\n";
    return true;
}

// ----------------------------------------------------------------
// テスト: 決定性 (同一入力 2 回で完全一致)
// ----------------------------------------------------------------
static bool test_determinism(QualityScorer& scorer, const std::vector<float>& gin)
{
    std::vector<float> a = scorer.infer_nchw(gin);
    std::vector<float> b = scorer.infer_nchw(gin);
    if (a.size() != b.size())
    {
        std::cerr << "[test_determinism] サイズ不一致\n";
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i)
    {
        if (a[i] != b[i])
        {
            std::cerr << "[test_determinism] 出力不一致: idx=" << i
                      << " " << a[i] << " vs " << b[i] << "\n";
            return false;
        }
    }
    std::cout << "[test_determinism] PASSED  (2 回完全一致)\n";
    return true;
}

// ----------------------------------------------------------------
// テスト: 入力サイズガード (不正サイズで invalid_argument)
// ----------------------------------------------------------------
static bool test_input_size_guard(QualityScorer& scorer)
{
    std::vector<float> bad(QualityScorer::INPUT_NUMEL - 1, 0.0f);
    bool threw = false;
    try
    {
        scorer.infer_nchw(bad);
    }
    catch (const std::invalid_argument&)
    {
        threw = true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[test_input_size_guard] 想定外の例外: " << e.what() << "\n";
        return false;
    }
    if (!threw)
    {
        std::cerr << "[test_input_size_guard] 例外が飛ばなかった\n";
        return false;
    }
    std::cout << "[test_input_size_guard] PASSED  (invalid_argument を捕捉)\n";
    return true;
}

// ----------------------------------------------------------------
// ベンチ: 推論レイテンシ (中央値)。warmup 3 + 計測 N=7。
// B-3c 実測中央値 (512²): CPU 11.7ms / NPU 8.32ms / iGPU 6.28ms。テストは CPU。
// ----------------------------------------------------------------
static bool bench_infer(QualityScorer& scorer, const std::vector<float>& gin)
{
    for (int i = 0; i < 3; ++i)
    {
        scorer.infer_nchw(gin);
    }

    constexpr int kN = 7;
    std::vector<double> times;
    times.reserve(kN);
    for (int i = 0; i < kN; ++i)
    {
        auto t0 = Clock::now();
        scorer.infer_nchw(gin);
        times.push_back(MsDouble(Clock::now() - t0).count());
    }
    std::sort(times.begin(), times.end());
    double median = times[kN / 2];

    std::cout << "[bench_infer]\n";
    std::cout << "  BENCH: scorer_infer median=" << median << " ms"
              << "  (min=" << times.front()
              << " ms, max=" << times.back() << " ms, N=" << kN << ")\n";
    return true;
}

#endif // HAVE_OPENVINO

} // namespace dollama

int main()
{
    // OV 非依存テスト (logits_to_result の sigmoid 変換・軸対応・サイズガード) は常に実行する。
    bool ok = true;
    ok = dollama::test_logits_to_result_values() && ok;
    ok = dollama::test_result_range()            && ok;
    ok = dollama::test_axis_mapping()            && ok;
    ok = dollama::test_logits_size_guard()       && ok;

#ifndef HAVE_OPENVINO
    std::cout << "[SKIP] HAVE_OPENVINO が未定義です。"
                 "OpenVINO なしでビルドされているため実 QualityScorer 推論テストをスキップします。\n";
    if (!ok)
    {
        std::cerr << "[test_quality_scorer] FAILED\n";
        return 1;
    }
    std::cout << "[test_quality_scorer] ALL PASSED\n";
    return 0;
#else
    // モデル / golden の存在確認
    const std::string xml_path    = SCORER_MODEL_XML;
    const std::string golden_path = SCORER_GOLDEN_PATH;
    if (!fs::exists(xml_path))
    {
        std::cout << "[SKIP] モデルファイルが見つかりません: " << xml_path << "\n";
        if (!ok)
        {
            std::cerr << "[test_quality_scorer] FAILED\n";
            return 1;
        }
        std::cout << "[test_quality_scorer] ALL PASSED\n";
        return 0;
    }
    if (!fs::exists(golden_path))
    {
        std::cout << "[SKIP] golden が見つかりません: " << golden_path << "\n";
        if (!ok)
        {
            std::cerr << "[test_quality_scorer] FAILED\n";
            return 1;
        }
        std::cout << "[test_quality_scorer] ALL PASSED\n";
        return 0;
    }

    // golden ロード
    std::vector<float> gin;
    std::vector<float> glogits;
    if (!dollama::load_golden(gin, glogits))
    {
        std::cerr << "[test_quality_scorer] golden ロード失敗\n";
        return 1;
    }

    // QualityScorer を CPU でコンパイル (golden の正は FP32 IR の CPU 出力)
    dollama::QualityScorer scorer(xml_path, "CPU");

    ok = dollama::test_golden_logits(scorer, gin, glogits) && ok;
    ok = dollama::test_output_length(scorer, gin)          && ok;
    ok = dollama::test_determinism(scorer, gin)            && ok;
    ok = dollama::test_input_size_guard(scorer)            && ok;
    ok = dollama::bench_infer(scorer, gin)                 && ok;

    if (!ok)
    {
        std::cerr << "[test_quality_scorer] FAILED\n";
        return 1;
    }
    std::cout << "[test_quality_scorer] ALL PASSED\n";
    return 0;
#endif // HAVE_OPENVINO
}

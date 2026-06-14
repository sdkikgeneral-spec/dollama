// ClipEncoder 単体テスト + 推論レイテンシ計測
// HAVE_OPENVINO 未定義時は [SKIP] を出力して終了する。
// モデルファイルが存在しない場合も [SKIP] で終了する。
// 全テスト通過時は "[test_clip] ALL PASSED" を出力して return 0。
// 失敗時は std::cerr に出力して return 1。
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <vector>

#ifdef HAVE_OPENVINO
#include "infer/clip.hpp"
#endif

namespace fs = std::filesystem;

// モデルファイルの候補パス (実行ディレクトリに応じて探索)
// meson test: build/ から実行 → ../models/...
// 直接実行: プロジェクトルートから → models/...
static std::string find_model_xml()
{
    const std::vector<std::string> candidates = {
        "../models/clip-l-text-encoder/model_ov.xml",
        "models/clip-l-text-encoder/model_ov.xml",
        "../../models/clip-l-text-encoder/model_ov.xml",
    };
    for (const auto& p : candidates)
    {
        if (fs::exists(p))
        {
            return p;
        }
    }
    return candidates[0]; // 見つからない場合は先頭を返す (SKIP メッセージ用)
}

namespace dollama {

#ifdef HAVE_OPENVINO

using Clock = std::chrono::steady_clock;
using MsDouble = std::chrono::duration<double, std::milli>;

// ----------------------------------------------------------------
// テスト1: 出力サイズ確認
// BOS(49406) + anime(2368) + EOS(49407) + 残り 0 のトークン列を入力し、
// 出力ベクタのサイズが SEQ_LEN * HIDDEN = 59136 であることを確認する。
// ----------------------------------------------------------------
static bool test_output_shape(ClipEncoder& enc)
{
    std::array<int32_t, ClipEncoder::SEQ_LEN> ids{};
    ids[0] = 49406; // BOS
    ids[1] = 2368;  // "anime"
    ids[2] = 49407; // EOS
    // 残りは 0 (PAD)

    const auto out = enc.infer(ids);
    const size_t expected = static_cast<size_t>(ClipEncoder::SEQ_LEN)
                          * static_cast<size_t>(ClipEncoder::HIDDEN);

    if (out.size() != expected)
    {
        std::cerr << "[test_output_shape] サイズ不正: got=" << out.size()
                  << " expected=" << expected << "\n";
        return false;
    }

    std::cout << "[test_output_shape] PASSED  (size=" << out.size() << ")\n";
    return true;
}

// ----------------------------------------------------------------
// テスト2: 出力 L2 norm 確認
// 同上の入力で L2 norm が 0 < norm < 10000.0f であることを確認する。
// ----------------------------------------------------------------
static bool test_output_l2_norm(ClipEncoder& enc)
{
    std::array<int32_t, ClipEncoder::SEQ_LEN> ids{};
    ids[0] = 49406;
    ids[1] = 2368;
    ids[2] = 49407;

    const auto out = enc.infer(ids);

    float sq_sum = 0.0f;
    for (float v : out)
    {
        sq_sum += v * v;
    }
    const float norm = std::sqrt(sq_sum);

    if (norm <= 0.0f || norm >= 10000.0f)
    {
        std::cerr << "[test_output_l2_norm] L2 norm が範囲外: norm=" << norm << "\n";
        return false;
    }

    std::cout << "[test_output_l2_norm] PASSED  (L2 norm=" << norm << ")\n";
    return true;
}

// ----------------------------------------------------------------
// テスト3: 全ゼロ入力で例外が起きないこと・サイズが正しいことを確認する
// ----------------------------------------------------------------
static bool test_zero_input(ClipEncoder& enc)
{
    std::array<int32_t, ClipEncoder::SEQ_LEN> ids{};
    // すべて 0 (PAD トークンのみ)

    std::vector<float> out;
    try
    {
        out = enc.infer(ids);
    }
    catch (const std::exception& e)
    {
        std::cerr << "[test_zero_input] 例外が発生: " << e.what() << "\n";
        return false;
    }

    const size_t expected = static_cast<size_t>(ClipEncoder::SEQ_LEN)
                          * static_cast<size_t>(ClipEncoder::HIDDEN);
    if (out.size() != expected)
    {
        std::cerr << "[test_zero_input] サイズ不正: got=" << out.size()
                  << " expected=" << expected << "\n";
        return false;
    }

    std::cout << "[test_zero_input] PASSED  (size=" << out.size() << ")\n";
    return true;
}

// ----------------------------------------------------------------
// ベンチマーク: 推論レイテンシ (中央値)
// ウォームアップ 5 回 + 計測 100 回
// ----------------------------------------------------------------
static bool bench_infer_latency(ClipEncoder& enc)
{
    std::array<int32_t, ClipEncoder::SEQ_LEN> ids{};
    ids[0] = 49406;
    ids[1] = 2368;
    ids[2] = 49407;

    // ウォームアップ
    for (int i = 0; i < 5; ++i)
    {
        enc.infer(ids);
    }

    // 計測 100 回
    constexpr int kN = 100;
    std::vector<double> times;
    times.reserve(kN);

    for (int i = 0; i < kN; ++i)
    {
        auto t0 = Clock::now();
        enc.infer(ids);
        double ms = MsDouble(Clock::now() - t0).count();
        times.push_back(ms);
    }

    // 中央値を算出
    std::sort(times.begin(), times.end());
    double median = times[kN / 2];

    std::cout << "[bench_infer_latency]\n";
    std::cout << "  BENCH: clip_infer median=" << median << " ms"
              << "  (min=" << times.front()
              << " ms, max=" << times.back() << " ms, N=" << kN << ")\n";
    return true;
}

#endif // HAVE_OPENVINO

} // namespace dollama

int main()
{
#ifndef HAVE_OPENVINO
    std::cout << "[SKIP] HAVE_OPENVINO が未定義です。"
                 "OpenVINO なしでビルドされているためテストをスキップします。\n";
    return 0;
#else
    // モデルファイル存在確認
    const std::string xml_path = find_model_xml();
    if (!fs::exists(xml_path))
    {
        std::cout << "[SKIP] モデルファイルが見つかりません: " << xml_path << "\n";
        return 0;
    }

    // ClipEncoder を NPU でコンパイル
    dollama::ClipEncoder enc(xml_path, "NPU");

    bool ok = true;
    ok = dollama::test_output_shape(enc)    && ok;
    ok = dollama::test_output_l2_norm(enc)  && ok;
    ok = dollama::test_zero_input(enc)      && ok;
    ok = dollama::bench_infer_latency(enc)  && ok;

    if (!ok)
    {
        std::cerr << "[test_clip] FAILED\n";
        return 1;
    }

    std::cout << "[test_clip] ALL PASSED\n";
    return 0;
#endif // HAVE_OPENVINO
}

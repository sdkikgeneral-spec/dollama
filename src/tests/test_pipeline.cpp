// Pipeline (Phase 1 マルチスレッド骨格) 統合テスト + ベンチ
// HAVE_OPENVINO 未定義時は [SKIP]。CLIP/WD14 両モデル不在時も [SKIP]。
// 検証:
//   (a) N フレーム実行で各フレーム非空タグ
//   (b) 全体がタイムアウト内に完了 (デッドロックなし)
//   (c) クリーン join (run() が返る)
// ベンチ: 定常スループット frames/s と 1 フレームレイテンシ中央値 (PASS/FAIL 外)。
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <future>
#include <iostream>
#include <string>
#include <vector>

#ifdef HAVE_OPENVINO
#include "pipeline.hpp"
#endif

namespace fs = std::filesystem;

// モデル .xml の候補パス探索 (test_clip/test_wd14 と同方式)
static std::string find_xml(const std::string& rel)
{
    const std::vector<std::string> candidates = {
        "../models/" + rel,
        "models/" + rel,
        "../../models/" + rel,
    };
    for (const auto& p : candidates)
    {
        if (fs::exists(p))
        {
            return p;
        }
    }
    return "";
}

#ifdef HAVE_OPENVINO

namespace dollama {

using Clock    = std::chrono::steady_clock;
using MsDouble = std::chrono::duration<double, std::milli>;

// ----------------------------------------------------------------
// テスト: N フレーム実行 — 非空タグ・デッドロックなし・クリーン join
// run() を別スレッドで起動し、ウォッチドッグでタイムアウト監視する。
// ----------------------------------------------------------------
static bool test_run_frames(Pipeline& pipe, int frames)
{
    // run() を非同期で起動し、デッドロック検出のためタイムアウト待ちする
    auto fut = std::async(std::launch::async,
                          [&pipe, frames]
                          {
                              return pipe.run(frames);
                          });

    // ウォッチドッグ: frames * 1s + 余裕 10s (CLIP 8ms + WD14 ~105ms/frame で十分)
    const auto wd_timeout =
        std::chrono::seconds(10 + frames);
    if (fut.wait_for(wd_timeout) != std::future_status::ready)
    {
        // run() が返らない = デッドロック。future を放置するとブロックするため、
        // ここで明示的に異常終了させる (テストプロセスを fail で落とす)。
        std::cerr << "[test_run_frames] run() が " << wd_timeout.count()
                  << "s 以内に返らなかった (デッドロックの可能性)\n";
        std::cerr << "[test_run_frames] pipeline-debugger 投入が必要\n";
        // future の取得をブロックしないため、ここでは return false のみ。
        // (jthread はプロセス終了時に強制 join されるが、CI 上は fail で十分)
        std::abort();
    }

    std::vector<std::string> tags = fut.get(); // クリーン join 済み

    // (c) クリーン join: run() が返った
    if (static_cast<int>(tags.size()) != frames)
    {
        std::cerr << "[test_run_frames] フレーム数不一致: got=" << tags.size()
                  << " expected=" << frames << "\n";
        return false;
    }

    // (a) 各フレーム非空タグ
    for (size_t i = 0; i < tags.size(); ++i)
    {
        if (tags[i].empty())
        {
            std::cerr << "[test_run_frames] frame " << i << " のタグが空\n";
            return false;
        }
    }

    std::cout << "[test_run_frames] PASSED  (frames=" << frames
              << ", 全フレーム非空)\n";
    for (size_t i = 0; i < tags.size(); ++i)
    {
        std::cout << "    frame " << i << ": " << tags[i] << "\n";
    }
    return true;
}

// ----------------------------------------------------------------
// ベンチ: 定常スループット (frames/s) と 1 フレームレイテンシ中央値
// ----------------------------------------------------------------
static bool bench_pipeline(Pipeline& pipe)
{
    // ウォームアップ (NPU/CPU コンパイル後の初回ロードを吸収)
    pipe.run(2);

    // スループット計測: まとまった frames を一括で流す
    constexpr int kFrames = 10;
    auto t0 = Clock::now();
    auto tags = pipe.run(kFrames);
    double total_ms = MsDouble(Clock::now() - t0).count();

    if (static_cast<int>(tags.size()) != kFrames)
    {
        std::cerr << "[bench_pipeline] フレーム欠落: " << tags.size()
                  << "/" << kFrames << "\n";
        return false;
    }

    double fps          = kFrames / (total_ms / 1000.0);
    double per_frame_ms = total_ms / kFrames;

    // 1 フレームレイテンシ中央値: 単発 run(1) を複数回計測
    constexpr int kN = 11;
    std::vector<double> lat;
    lat.reserve(kN);
    for (int i = 0; i < kN; ++i)
    {
        auto s = Clock::now();
        pipe.run(1);
        lat.push_back(MsDouble(Clock::now() - s).count());
    }
    std::sort(lat.begin(), lat.end());
    double median = lat[kN / 2];

    std::cout << "[bench_pipeline]\n";
    std::cout << "  BENCH: pipeline_throughput=" << fps << " frames/s"
              << "  (per_frame=" << per_frame_ms << " ms, N=" << kFrames << ")\n";
    std::cout << "  BENCH: pipeline_frame_latency median=" << median << " ms"
              << "  (min=" << lat.front() << " ms, max=" << lat.back()
              << " ms, N=" << kN << ")\n";
    return true;
}

} // namespace dollama

#endif // HAVE_OPENVINO

int main()
{
#ifndef HAVE_OPENVINO
    std::cout << "[SKIP] HAVE_OPENVINO が未定義です。テストをスキップします。\n";
    return 0;
#else
    const std::string clip_xml = find_xml("clip-l-text-encoder/model_ov.xml");
    const std::string wd14_xml = find_xml("wd14-swinv2-tagger-v3/model_ov.xml");

    if (clip_xml.empty() || wd14_xml.empty())
    {
        std::cout << "[SKIP] CLIP/WD14 モデルが見つかりません "
                     "(clip='" << clip_xml << "', wd14='" << wd14_xml << "')\n";
        return 0;
    }

    dollama::Pipeline pipe(clip_xml, wd14_xml);

    bool ok = true;
    ok = dollama::test_run_frames(pipe, 5) && ok;
    ok = dollama::bench_pipeline(pipe)     && ok;

    if (!ok)
    {
        std::cerr << "[test_pipeline] FAILED\n";
        return 1;
    }
    std::cout << "[test_pipeline] ALL PASSED\n";
    return 0;
#endif // HAVE_OPENVINO
}

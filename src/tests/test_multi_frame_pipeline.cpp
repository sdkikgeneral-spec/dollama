// MultiFramePipeline (Phase 4-(1) W1) 正当性テスト + 並列ベンチ
//
// OV/CUDA 非依存 (各段は sleep_for スタブ callable で駆動) ゆえ全環境で SKIP なし実走。
//
// 正当性 (PASS/FAIL):
//   (a) 順序 + 全件     : 各フレームに index を埋め込み、results が入力順・件数 N に一致
//   (b) feedback drain  : stage C で一部 flagged=true → stats.flagged_count が期待値
//   (c) 例外時 abort    : ある段で例外 → stats.aborted==true・run() が返る (デッドロックなし)
//                         test_pipeline.cpp L52-74 の async + wait_for + abort 方式を踏襲。
//
// 並列ベンチ (DOLLAMA_BENCH=1 のときのみ・PASS/FAIL 外):
//   - 実デバイス比率の縮尺スタブ (LM 40ms / CLIP 0.8ms / SDXL 380ms / WD14 10ms)
//     DOLLAMA_BENCH_FULL=1 で実寸 (LM 400 / CLIP 8 / SDXL 3800 / WD14 100ms)
//   - GPU バウンド確認 / queue_bclip_to_bsdxl 待ち≈0 (GPU 飢餓検出) / LM レイテンシ隠蔽
//   - QueueDepth ∈ {2,4,8} スイープ (非型テンプレ param ゆえ明示インスタンス化)

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "core/multi_frame_pipeline.hpp"

namespace dollama {

using Clock    = std::chrono::steady_clock;
using MsDouble = std::chrono::duration<double, std::milli>;

// 縮尺スタブのパイプライン型を簡潔にするエイリアス
//   PromptT=std::string / CondT,ImageT=std::vector<float> / FeedbackT=std::string
template<size_t QD>
using TestPipeline =
    MultiFramePipeline<std::string, std::vector<float>, std::vector<float>, std::string, QD>;

// ----------------------------------------------------------------
// 縮尺スタブのステージ callable を組み立てる (index を float で素通しさせ、
// 順序・件数の検証ができるようにする)。sleep_ms<0 なら sleep しない。
// ----------------------------------------------------------------
template<size_t QD>
static TestPipeline<QD> make_stub_pipeline(double lm_ms, double clip_ms,
                                           double sdxl_ms, double wd14_ms,
                                           int flag_mod)
{
    auto sleep_ms = [](double ms)
    {
        if (ms > 0.0)
        {
            std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(ms));
        }
    };

    // A: FrameSpec.user_text (= index 文字列) をそのまま prompt にする
    auto stage_a =
        [sleep_ms, lm_ms](const FrameSpec& fs) -> std::string
        {
            sleep_ms(lm_ms);
            return fs.user_text;
        };

    // B-CLIP: prompt 中の index を float 1 要素にして流す
    auto stage_b_clip =
        [sleep_ms, clip_ms](std::string prompt) -> std::vector<float>
        {
            sleep_ms(clip_ms);
            return std::vector<float>{ static_cast<float>(std::stoi(prompt)) };
        };

    // B-SDXL: conditioning を素通し (index 保持)
    auto stage_b_sdxl =
        [sleep_ms, sdxl_ms](std::vector<float> cond) -> std::vector<float>
        {
            sleep_ms(sdxl_ms);
            return cond;
        };

    // C: image[0] の index を結果文字列に。flag_mod>0 のとき index%flag_mod==0 を flagged
    using Out = typename TestPipeline<QD>::StageCOutput;
    auto stage_c =
        [sleep_ms, wd14_ms, flag_mod](std::vector<float> img) -> Out
        {
            sleep_ms(wd14_ms);
            int idx = static_cast<int>(img[0]);
            Out out;
            out.feedback = std::to_string(idx);
            out.flagged  = (flag_mod > 0) && (idx % flag_mod == 0);
            return out;
        };

    return TestPipeline<QD>(std::move(stage_a), std::move(stage_b_clip),
                            std::move(stage_b_sdxl), std::move(stage_c));
}

// index 文字列を載せた N フレームを作る
static std::vector<FrameSpec> make_frames(int n)
{
    std::vector<FrameSpec> frames;
    frames.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i)
    {
        frames.push_back(FrameSpec{ std::to_string(i) });
    }
    return frames;
}

// ----------------------------------------------------------------
// (a) 順序 + 全件: results が入力順 (0..N-1) で件数 N
// (b) feedback drain + flagged 集計: flag_mod の倍数件が flagged
// ----------------------------------------------------------------
static bool test_order_and_flagged()
{
    constexpr int kN       = 12;
    constexpr int kFlagMod = 3; // 0,3,6,9 → 4 件 flagged

    auto pipe = make_stub_pipeline<2>(-1, -1, -1, -1, kFlagMod);
    auto rr   = pipe.run(make_frames(kN));

    if (rr.stats.aborted)
    {
        std::cerr << "[order_flagged] 予期しない abort\n";
        return false;
    }

    // (a) 件数
    if (static_cast<int>(rr.results.size()) != kN)
    {
        std::cerr << "[order_flagged] 件数不一致 got=" << rr.results.size()
                  << " expected=" << kN << "\n";
        return false;
    }

    // (a) 順序: results[i] == std::to_string(i)
    for (int i = 0; i < kN; ++i)
    {
        if (rr.results[static_cast<size_t>(i)] != std::to_string(i))
        {
            std::cerr << "[order_flagged] 順序不一致 results[" << i << "]="
                      << rr.results[static_cast<size_t>(i)]
                      << " expected=" << i << "\n";
            return false;
        }
    }

    // (b) flagged 件数 = [0,N) で i%kFlagMod==0 の数
    int expected_flagged = 0;
    for (int i = 0; i < kN; ++i)
    {
        if (i % kFlagMod == 0)
        {
            ++expected_flagged;
        }
    }
    if (rr.stats.flagged_count != expected_flagged)
    {
        std::cerr << "[order_flagged] flagged 不一致 got=" << rr.stats.flagged_count
                  << " expected=" << expected_flagged << "\n";
        return false;
    }

    std::cout << "[order_flagged] PASSED (N=" << kN << " 順序一致, flagged="
              << rr.stats.flagged_count << ")\n";
    return true;
}

// ----------------------------------------------------------------
// (c) 例外時 abort + デッドロックなし
// 任意の段 (ここでは B-SDXL) で例外を投げ、stats.aborted==true かつ
// run() が返ることを確認する。test_pipeline.cpp L52-74 の
// async(launch::async) + wait_for + タイムアウト時 std::abort() を踏襲。
// ----------------------------------------------------------------
static bool test_exception_abort()
{
    constexpr int kN = 4;

    auto sleep_none = [](double) {};
    (void)sleep_none;

    // B-SDXL が必ず例外を投げるパイプライン
    using Out = typename TestPipeline<2>::StageCOutput;
    TestPipeline<2> pipe(
        [](const FrameSpec& fs) -> std::string
        {
            return fs.user_text;
        },
        [](std::string prompt) -> std::vector<float>
        {
            return std::vector<float>{ static_cast<float>(std::stoi(prompt)) };
        },
        [](std::vector<float>) -> std::vector<float>
        {
            // GPU 段の失敗を模擬: 例外を投げる
            throw std::runtime_error("stub B-SDXL 故障 (テスト用)");
        },
        [](std::vector<float> img) -> Out
        {
            Out out;
            out.feedback = std::to_string(static_cast<int>(img[0]));
            return out;
        });

    auto frames = make_frames(kN);

    // run() を非同期起動し、ウォッチドッグでデッドロックを監視する。
    auto fut = std::async(std::launch::async,
                          [&pipe, &frames]
                          {
                              return pipe.run(frames);
                          });

    // kQueueTimeout=5s が各段でほぼ並行に発火するため、十分な余裕を取る。
    const auto wd_timeout = std::chrono::seconds(40);
    if (fut.wait_for(wd_timeout) != std::future_status::ready)
    {
        std::cerr << "[exception_abort] run() が " << wd_timeout.count()
                  << "s 以内に返らなかった (デッドロックの可能性)\n";
        // future を放置するとブロックするため明示的に異常終了させる
        std::abort();
    }

    auto rr = fut.get(); // クリーン join 済み

    if (!rr.stats.aborted)
    {
        std::cerr << "[exception_abort] aborted が false (例外が伝播していない)\n";
        return false;
    }

    std::cout << "[exception_abort] PASSED (aborted=true, run() 復帰, frames_processed="
              << rr.stats.frames_processed << ")\n";
    return true;
}

// ----------------------------------------------------------------
// ベンチ補助: 指定 QueueDepth + 縮尺で N フレーム流し stats を返す
// ----------------------------------------------------------------
template<size_t QD>
static PipelineStats bench_depth(int n, double lm, double clip,
                                 double sdxl, double wd14)
{
    auto pipe = make_stub_pipeline<QD>(lm, clip, sdxl, wd14, 0);
    return pipe.run(make_frames(n)).stats;
}

// ----------------------------------------------------------------
// 並列ベンチ (DOLLAMA_BENCH=1 のときのみ)
//   - GPU バウンド確認 / queue_bclip_to_bsdxl 待ち≈0 / LM レイテンシ隠蔽
//   - QueueDepth {2,4,8} スイープの throughput 曲線
// ----------------------------------------------------------------
static void run_bench()
{
    const bool full = []
    {
        const char* f = std::getenv("DOLLAMA_BENCH_FULL");
        return f != nullptr && f[0] == '1';
    }();

    // 実デバイス比率の縮尺スタブ (full=実寸×10)
    const double lm   = full ? 400.0  : 40.0;
    const double clip = full ? 8.0    : 0.8;
    const double sdxl = full ? 3800.0 : 380.0;
    const double wd14 = full ? 100.0  : 10.0;

    constexpr int kN = 8;

    std::cerr << "\n[mfp-bench] ====== 並列ベンチ (N=" << kN
              << ", scale=" << (full ? "FULL(実寸)" : "縮尺")
              << " | LM=" << lm << " CLIP=" << clip
              << " SDXL=" << sdxl << " WD14=" << wd14 << " ms) ======\n";

    // 主診断 (QueueDepth=2)。run() 自身が DOLLAMA_BENCH=1 で dump_stats する。
    auto s = bench_depth<2>(kN, lm, clip, sdxl, wd14);

    const double ideal_gpu_fps = 1000.0 / sdxl;          // SDXL 律速なら理論上限
    const double per_frame_ms  = (s.throughput_fps > 0.0)
                                     ? 1000.0 / s.throughput_fps : 0.0;
    const double sum_stage_ms  = lm + clip + sdxl + wd14; // 直列だった場合の 1 フレーム合計

    std::cerr << "[mfp-bench] --- 診断 (QueueDepth=2) ---\n";
    std::cerr << "  throughput            = " << s.throughput_fps << " frames/s"
              << " (理論 GPU 上限 " << ideal_gpu_fps << " fps)\n";
    std::cerr << "  per_frame (1/throughput)= " << per_frame_ms << " ms"
              << "  vs SDXL " << sdxl << " ms"
              << "  → GPU バウンド " << (per_frame_ms <= sdxl * 1.5 ? "YES" : "NO")
              << "\n";
    std::cerr << "  queue_bclip_to_bsdxl wait median = "
              << s.queue_bclip_to_bsdxl.median_ms << " ms"
              << "  → GPU 飢餓 "
              << (s.queue_bclip_to_bsdxl.median_ms <= sdxl * 0.1 ? "なし(≈0)" : "あり")
              << "\n";
    std::cerr << "  LM レイテンシ隠蔽: stage_a median=" << s.stage_a.median_ms
              << " ms / 直列合計=" << sum_stage_ms << " ms / per_frame="
              << per_frame_ms << " ms → LM は "
              << (per_frame_ms < sum_stage_ms * 0.95 ? "SDXL の裏に隠蔽" : "未隠蔽")
              << "\n";
    std::cerr << "  frame_latency median  = " << s.frame_latency_median_ms << " ms"
              << " (端から端)\n";

    // QueueDepth スイープ (2/4/8 を明示インスタンス化・ループ不可)
    auto s2 = bench_depth<2>(kN, lm, clip, sdxl, wd14);
    auto s4 = bench_depth<4>(kN, lm, clip, sdxl, wd14);
    auto s8 = bench_depth<8>(kN, lm, clip, sdxl, wd14);

    std::cerr << "[mfp-bench] --- QueueDepth スイープ (throughput 曲線) ---\n";
    std::cerr << "  QueueDepth=2 : " << s2.throughput_fps << " frames/s\n";
    std::cerr << "  QueueDepth=4 : " << s4.throughput_fps << " frames/s\n";
    std::cerr << "  QueueDepth=8 : " << s8.throughput_fps << " frames/s\n";

    const double spread =
        std::max({s2.throughput_fps, s4.throughput_fps, s8.throughput_fps}) -
        std::min({s2.throughput_fps, s4.throughput_fps, s8.throughput_fps});
    const double rel = (s2.throughput_fps > 0.0) ? spread / s2.throughput_fps : 0.0;
    std::cerr << "  結論: " << (rel <= 0.05
                  ? "平坦 → look-ahead 2 段で GPU 飢餓なし (深さ増の利得なし)"
                  : "深い側で伸びる → 2 段では先読み不足")
              << "  (相対ばらつき " << (rel * 100.0) << "%)\n";
    std::cerr << "[mfp-bench] ============================================\n";
}

} // namespace dollama

int main()
{
    bool ok = true;
    ok = dollama::test_order_and_flagged() && ok;
    ok = dollama::test_exception_abort()   && ok;

    if (const char* bench = std::getenv("DOLLAMA_BENCH");
        bench != nullptr && bench[0] == '1')
    {
        dollama::run_bench();
    }

    if (!ok)
    {
        std::cerr << "[test_multi_frame_pipeline] FAILED\n";
        return 1;
    }
    std::cout << "[test_multi_frame_pipeline] ALL PASSED\n";
    return 0;
}

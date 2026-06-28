#pragma once

// dollama マルチフレーム・パイプライン orchestrator (Phase 4-(1) W1)
//
// src/pipeline.hpp (Phase 1) の 4 段マルチスレッド骨格を、ステージ処理を
// std::function callable で外から注入する形に一般化したもの。
//
//   A(P-core)      : 入力 FrameSpec      -> prompt
//   B-CLIP(E-core) : prompt              -> conditioning
//   B-SDXL(GPU)    : conditioning        -> image
//   C(E-core)      : image               -> 結果収集 + feedback (A へ返す)
//
// W1 では実部品 (BitNet / TextConditioner / IDiffusionRunner / WD14) には
// 一切依存しない。各段は std::function で外部注入し、ステージ間データ型は
// テンプレートパラメータ化する。実結線は W2 で行う。
// よって本ヘッダの依存は std + core/queue.hpp + core/affinity.hpp のみ。
//
// 設計参照: src/pipeline.hpp (無改変・9.13 frames/s ベースライン保全)
//   - SPSCQueue<T, Capacity> (capacity 2/4・push_wait/pop_wait + timeout)
//   - set_current_thread_affinity (各ワーカーが起動直後に自己ピン留め)
//   - jthread + stop_token + abort_flag
//   - feedback キューの唯一 consumer 契約 (A が drain しないと詰まる)

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "core/queue.hpp"
#include "core/affinity.hpp"

namespace dollama {

// ----------------------------------------------------------------
// アフィニティマスク (src/pipeline.hpp の kPCoreMask/kECoreMask と同値)
// Intel Core Ultra 9 285 (probe11 実測) 固有の値。
// pipeline.hpp の定数と衝突しないよう別名で持つ (両ヘッダ同時 include 時の
// 再定義回避)。将来は core/cpu_topology.hpp で動的取得して置き換える。
// ----------------------------------------------------------------
constexpr uint64_t kMfpPCoreMask = 0x00C03C03ULL; // P-core 8 論理コア (LM/SDXL)
constexpr uint64_t kMfpECoreMask = 0x003FC3FCULL; // E-core 16 論理コア (CLIP/WD14)

// ----------------------------------------------------------------
// FrameSpec: パイプライン入力 1 フレームの仕様
// W1 段階では user_text のみ。W2 で CharacterIdentity 相当の identity データ
// (不透明データ) をここに載せて同一性条件付き生成へ拡張する。
// ----------------------------------------------------------------
struct FrameSpec
{
    std::string user_text;
    // TODO(W2): identity (CharacterIdentity 相当) をここに追加。
    //   ステージ A callable が user_text + identity から prompt を組む。
};

// ----------------------------------------------------------------
// StageTiming: 1 段 (または 1 キュー) の処理/待ち時間統計 (ミリ秒)
// ----------------------------------------------------------------
struct StageTiming
{
    double median_ms = 0.0;
    double min_ms    = 0.0;
    double max_ms    = 0.0;
    int    count     = 0;
};

// ----------------------------------------------------------------
// PipelineStats: run() の計測結果
//   - 各段処理時間 (中央値/min/max)
//   - 各キューの待ち時間 (consumer が pop で待った時間)
//   - 定常スループット (frames/s)
//   - 単発レイテンシ中央値 (フレーム入口 A → 出口 C の端から端まで)
// ----------------------------------------------------------------
struct PipelineStats
{
    StageTiming stage_a;              // A: FrameSpec -> prompt
    StageTiming stage_b_clip;         // B-CLIP: prompt -> conditioning
    StageTiming stage_b_sdxl;         // B-SDXL: conditioning -> image
    StageTiming stage_c;              // C: image -> 結果 + feedback

    StageTiming queue_a_to_bclip;     // B-CLIP が prompt を待った時間
    StageTiming queue_bclip_to_bsdxl; // B-SDXL が conditioning を待った時間
    StageTiming queue_bsdxl_to_c;     // C が image を待った時間
    StageTiming queue_feedback;       // A が feedback を待った時間

    double throughput_fps          = 0.0; // 定常スループット (全フレーム / 総壁時間)
    double frame_latency_median_ms = 0.0; // 単発レイテンシ中央値
    int    frames_processed        = 0;   // C が完走したフレーム数
    int    flagged_count           = 0;   // A が feedback drain 中に数えた flagged 件数
    bool   aborted                 = false;
};

// ----------------------------------------------------------------
// MultiFramePipeline<PromptT, CondT, ImageT, FeedbackT, QueueDepth>
//   - PromptT   : A の出力 (例: prompt 文字列)
//   - CondT     : B-CLIP の出力 (例: TextConditioning)
//   - ImageT    : B-SDXL の出力 (例: rgb フレーム)
//   - FeedbackT : C の収集結果 + feedback データ (例: WD14 タグ文字列)
//   - QueueDepth: 段間データキューの look-ahead 深さ (既定 2・2 の冪のみ)
//
// ステージ callable は std::function で外部注入する:
//   StageA     : FrameSpec      -> PromptT
//   StageBClip : PromptT        -> CondT
//   StageBSdxl : CondT          -> ImageT
//   StageC     : ImageT         -> StageCOutput { FeedbackT feedback; bool flagged; }
//
// C の出力 (StageCOutput) は results へ収集されると同時に feedback キュー
// 経由で A へ返る。A は feedback を drain し flagged 件数を集計するのみ
// (再プロンプト戦略はスコープ外・W2 以降)。
// ----------------------------------------------------------------
// 非型テンプレ QueueDepth: 段間データキューの look-ahead 深さ (= 先読み段数)。
//   既定 2 で従来挙動と完全非回帰。SPSCQueue は 2 の冪 capacity のみ許可するため
//   2/4/8/... のみ受け付ける (static_assert で強制)。feedback キューは現状の
//   2:4 比を維持し QueueDepth*2 とする。深さスイープで GPU 飢餓の有無を測る軸。
template<typename PromptT, typename CondT, typename ImageT, typename FeedbackT,
         size_t QueueDepth = 2>
class MultiFramePipeline
{
    // QueueDepth は SPSCQueue の制約 (2 以上の 2 の冪) を満たす必要がある。
    static_assert((QueueDepth & (QueueDepth - 1)) == 0 && QueueDepth >= 2,
                  "QueueDepth は 2 以上の 2 の冪");

public:
    // C 段の出力 (= feedback キューを流れるメッセージ型)
    struct StageCOutput
    {
        FeedbackT feedback;        // results に収集 + A へ返すデータ
        bool      flagged = false; // 異常検知フラグ (A が件数を集計)
    };

    // ステージ callable シグネチャ (公開 API・W2 がこれに結線する)
    using StageA     = std::function<PromptT(const FrameSpec&)>;
    using StageBClip = std::function<CondT(PromptT)>;
    using StageBSdxl = std::function<ImageT(CondT)>;
    using StageC     = std::function<StageCOutput(ImageT)>;

    // run() の戻り値: 収集結果 (frames と同順) + 計測統計
    struct RunResult
    {
        std::vector<FeedbackT> results; // C が収集した順序付き結果
        PipelineStats          stats;
    };

    // キュー満杯/空時のタイムアウト (src/pipeline.hpp kQueueTimeout と同値)
    static constexpr std::chrono::milliseconds kQueueTimeout{5000};

    // コンストラクタ: 4 段の callable を受け取る (空 callable は不可)
    MultiFramePipeline(StageA a, StageBClip b_clip, StageBSdxl b_sdxl, StageC c)
        : stage_a_(std::move(a))
        , stage_b_clip_(std::move(b_clip))
        , stage_b_sdxl_(std::move(b_sdxl))
        , stage_c_(std::move(c))
    {
    }

    // ------------------------------------------------------------
    // run: frames を流し、各フレームの結果 + 統計を返す。
    // 4 スレッド (A/B-CLIP/B-SDXL/C) を立て、各段を順に通す。
    // タイムアウト・例外時は abort_flag を立てて全スレッド join し、
    // cerr にログを出して握り潰さない (src/pipeline.hpp と同方針)。
    // CPU アフィニティは各ワーカーが起動直後に自分自身へ設定する
    // (A/B-SDXL = P-core、B-CLIP/C = E-core)。失敗は警告ログのみで続行。
    // ------------------------------------------------------------
    RunResult run(const std::vector<FrameSpec>& frames)
    {
        using Clock    = std::chrono::steady_clock;
        using MsDouble = std::chrono::duration<double, std::milli>;

        const int n = static_cast<int>(frames.size());

        // 段間キュー (capacity は QueueDepth でパラメータ化:
        //   データ段=QueueDepth、feedback=QueueDepth*2 で現状 2:4 比を維持)。
        SPSCQueue<PromptT, QueueDepth>          a_to_bclip;
        SPSCQueue<CondT, QueueDepth>            bclip_to_bsdxl;
        SPSCQueue<ImageT, QueueDepth>           bsdxl_to_c;
        SPSCQueue<StageCOutput, QueueDepth * 2> c_to_a; // feedback

        std::atomic<bool> abort_flag{false};

        // 結果収集 (C スレッドのみが書き込む・join 後に読む)
        std::vector<FeedbackT> results;
        results.reserve(static_cast<size_t>(n));

        // 各段の処理時間 (各ベクタは単一スレッドのみが push・join 後に集計)
        std::vector<double> a_times, bclip_times, bsdxl_times, c_times;
        // 各キューの pop 待ち時間 (consumer 側で計測)
        std::vector<double> w_a_to_bclip, w_bclip_to_bsdxl, w_bsdxl_to_c, w_feedback;
        // 端から端までのフレームレイテンシ (C スレッドが push)
        std::vector<double> latencies;
        a_times.reserve(n); bclip_times.reserve(n); bsdxl_times.reserve(n); c_times.reserve(n);
        w_a_to_bclip.reserve(n); w_bclip_to_bsdxl.reserve(n);
        w_bsdxl_to_c.reserve(n); w_feedback.reserve(n);
        latencies.reserve(n);

        // フレーム f がパイプラインに入った時刻 (A が書き・C が読む)。
        // 順序は全 SPSC FIFO で保たれるため C の f 番目出力は A の f 番目入力に対応し、
        // release/acquire 連鎖で start_times[f] の可視性が C まで伝播する。
        std::vector<Clock::time_point> start_times(static_cast<size_t>(n));

        int flagged_count = 0; // A スレッドのみが更新

        // 全体の壁時計 (スループット算出用)
        auto run_t0 = Clock::now();

        // ========================================================
        // A スレッド (producer, P-core)
        // FrameSpec -> prompt を frames 回 push しつつ feedback を drain する。
        // A は c_to_a の唯一 consumer。drain しないと feedback キューが詰まる。
        // ========================================================
        std::jthread a_thread(
            [&](std::stop_token st)
            {
                if (!set_current_thread_affinity(kMfpPCoreMask))
                {
                    std::cerr << "[mfp] 警告: A スレッドのアフィニティ設定に失敗\n";
                }

                try
                {
                    int fed_back = 0;

                    for (int f = 0; f < n; ++f)
                    {
                        if (st.stop_requested() || abort_flag.load())
                        {
                            return;
                        }

                        start_times[f] = Clock::now();

                        auto t0 = Clock::now();
                        PromptT prompt = stage_a_(frames[f]);
                        a_times.push_back(MsDouble(Clock::now() - t0).count());

                        if (!a_to_bclip.push_wait(prompt, kQueueTimeout))
                        {
                            std::cerr << "[mfp] A: a_to_bclip push timeout (frame "
                                      << f << ")\n";
                            abort_flag.store(true);
                            return;
                        }

                        // 溜まった feedback をノンブロッキングで吸い出す
                        StageCOutput fb;
                        while (c_to_a.pop(fb))
                        {
                            ++fed_back;
                            if (fb.flagged)
                            {
                                ++flagged_count;
                            }
                        }
                    }

                    // 残りの feedback が n 件揃うまで drain する
                    // TODO(W2): feedback タグで再プロンプトする
                    while (fed_back < n)
                    {
                        if (st.stop_requested() || abort_flag.load())
                        {
                            return;
                        }

                        StageCOutput fb;
                        auto ws = Clock::now();
                        bool ok = c_to_a.pop_wait(fb, kQueueTimeout);
                        w_feedback.push_back(MsDouble(Clock::now() - ws).count());
                        if (!ok)
                        {
                            std::cerr << "[mfp] A: c_to_a pop timeout (feedback "
                                      << fed_back << "/" << n << ")\n";
                            abort_flag.store(true);
                            return;
                        }
                        ++fed_back;
                        if (fb.flagged)
                        {
                            ++flagged_count;
                        }
                    }
                }
                catch (const std::exception& e)
                {
                    std::cerr << "[mfp] A: 例外 " << e.what() << "\n";
                    abort_flag.store(true);
                }
            });

        // ========================================================
        // B-CLIP スレッド (E-core)
        // prompt を pop -> stage_b_clip -> conditioning を bclip_to_bsdxl へ。
        // ========================================================
        std::jthread bclip_thread(
            [&](std::stop_token st)
            {
                if (!set_current_thread_affinity(kMfpECoreMask))
                {
                    std::cerr << "[mfp] 警告: B-CLIP スレッドのアフィニティ設定に失敗\n";
                }

                try
                {
                    for (int f = 0; f < n; ++f)
                    {
                        if (st.stop_requested() || abort_flag.load())
                        {
                            return;
                        }

                        PromptT prompt;
                        auto ws = Clock::now();
                        bool ok = a_to_bclip.pop_wait(prompt, kQueueTimeout);
                        w_a_to_bclip.push_back(MsDouble(Clock::now() - ws).count());
                        if (!ok)
                        {
                            std::cerr << "[mfp] B-CLIP: a_to_bclip pop timeout (frame "
                                      << f << ")\n";
                            abort_flag.store(true);
                            return;
                        }

                        auto t0 = Clock::now();
                        CondT cond = stage_b_clip_(std::move(prompt));
                        bclip_times.push_back(MsDouble(Clock::now() - t0).count());

                        if (!bclip_to_bsdxl.push_wait(cond, kQueueTimeout))
                        {
                            std::cerr << "[mfp] B-CLIP: bclip_to_bsdxl push timeout (frame "
                                      << f << ")\n";
                            abort_flag.store(true);
                            return;
                        }
                    }
                }
                catch (const std::exception& e)
                {
                    std::cerr << "[mfp] B-CLIP: 例外 " << e.what() << "\n";
                    abort_flag.store(true);
                }
            });

        // ========================================================
        // B-SDXL スレッド (P-core)
        // conditioning を pop -> stage_b_sdxl -> image を bsdxl_to_c へ。
        // ========================================================
        std::jthread bsdxl_thread(
            [&](std::stop_token st)
            {
                if (!set_current_thread_affinity(kMfpPCoreMask))
                {
                    std::cerr << "[mfp] 警告: B-SDXL スレッドのアフィニティ設定に失敗\n";
                }

                try
                {
                    for (int f = 0; f < n; ++f)
                    {
                        if (st.stop_requested() || abort_flag.load())
                        {
                            return;
                        }

                        CondT cond;
                        auto ws = Clock::now();
                        bool ok = bclip_to_bsdxl.pop_wait(cond, kQueueTimeout);
                        w_bclip_to_bsdxl.push_back(MsDouble(Clock::now() - ws).count());
                        if (!ok)
                        {
                            std::cerr << "[mfp] B-SDXL: bclip_to_bsdxl pop timeout (frame "
                                      << f << ")\n";
                            abort_flag.store(true);
                            return;
                        }

                        auto t0 = Clock::now();
                        ImageT img = stage_b_sdxl_(std::move(cond));
                        bsdxl_times.push_back(MsDouble(Clock::now() - t0).count());

                        if (!bsdxl_to_c.push_wait(img, kQueueTimeout))
                        {
                            std::cerr << "[mfp] B-SDXL: bsdxl_to_c push timeout (frame "
                                      << f << ")\n";
                            abort_flag.store(true);
                            return;
                        }
                    }
                }
                catch (const std::exception& e)
                {
                    std::cerr << "[mfp] B-SDXL: 例外 " << e.what() << "\n";
                    abort_flag.store(true);
                }
            });

        // ========================================================
        // C スレッド (E-core)
        // image を pop -> stage_c -> results へ収集 + feedback を c_to_a へ。
        // ========================================================
        std::jthread c_thread(
            [&](std::stop_token st)
            {
                if (!set_current_thread_affinity(kMfpECoreMask))
                {
                    std::cerr << "[mfp] 警告: C スレッドのアフィニティ設定に失敗\n";
                }

                try
                {
                    for (int f = 0; f < n; ++f)
                    {
                        if (st.stop_requested() || abort_flag.load())
                        {
                            return;
                        }

                        ImageT img;
                        auto ws = Clock::now();
                        bool ok = bsdxl_to_c.pop_wait(img, kQueueTimeout);
                        w_bsdxl_to_c.push_back(MsDouble(Clock::now() - ws).count());
                        if (!ok)
                        {
                            std::cerr << "[mfp] C: bsdxl_to_c pop timeout (frame "
                                      << f << ")\n";
                            abort_flag.store(true);
                            return;
                        }

                        auto t0 = Clock::now();
                        StageCOutput out = stage_c_(std::move(img));
                        c_times.push_back(MsDouble(Clock::now() - t0).count());

                        // 端から端までのレイテンシ (C の f 番目出力 = A の f 番目入力)
                        latencies.push_back(
                            MsDouble(Clock::now() - start_times[f]).count());

                        results.push_back(out.feedback);

                        // feedback を A へ (満杯でも results 収集は済んでいる)
                        if (!c_to_a.push_wait(out, kQueueTimeout))
                        {
                            std::cerr << "[mfp] C: c_to_a push timeout (frame "
                                      << f << ")\n";
                            abort_flag.store(true);
                            return;
                        }
                    }
                }
                catch (const std::exception& e)
                {
                    std::cerr << "[mfp] C: 例外 " << e.what() << "\n";
                    abort_flag.store(true);
                }
            });

        // jthread デストラクタが request_stop + join するが、明示 join で
        // 壁時計を正しく閉じる。abort 時も全スレッドが抜けて join できる。
        a_thread.join();
        bclip_thread.join();
        bsdxl_thread.join();
        c_thread.join();

        double total_ms = MsDouble(Clock::now() - run_t0).count();

        // ----------------------------------------------------------------
        // 統計集計
        // ----------------------------------------------------------------
        RunResult rr;
        rr.results = std::move(results);

        PipelineStats& s = rr.stats;
        s.stage_a              = summarize(a_times);
        s.stage_b_clip         = summarize(bclip_times);
        s.stage_b_sdxl         = summarize(bsdxl_times);
        s.stage_c              = summarize(c_times);
        s.queue_a_to_bclip     = summarize(w_a_to_bclip);
        s.queue_bclip_to_bsdxl = summarize(w_bclip_to_bsdxl);
        s.queue_bsdxl_to_c     = summarize(w_bsdxl_to_c);
        s.queue_feedback       = summarize(w_feedback);
        s.frame_latency_median_ms = summarize(latencies).median_ms;
        s.frames_processed     = static_cast<int>(rr.results.size());
        s.flagged_count        = flagged_count;
        s.aborted              = abort_flag.load();
        s.throughput_fps       =
            (total_ms > 0.0) ? (s.frames_processed / (total_ms / 1000.0)) : 0.0;

        if (s.aborted)
        {
            std::cerr << "[mfp] 異常終了 (タイムアウトまたは例外)。"
                         "収集済みフレーム数=" << s.frames_processed << "\n";
        }

        // DOLLAMA_BENCH=1 で詳細を cerr へ (prof_bitnet / 既存 BENCH の流儀)
        if (const char* bench = std::getenv("DOLLAMA_BENCH");
            bench != nullptr && bench[0] == '1')
        {
            dump_stats(s);
        }

        return rr;
    }

private:
    // 処理/待ち時間ベクタ -> StageTiming (中央値/min/max)
    static StageTiming summarize(std::vector<double>& v)
    {
        StageTiming t{};
        t.count = static_cast<int>(v.size());
        if (v.empty())
        {
            return t;
        }
        std::sort(v.begin(), v.end());
        t.min_ms    = v.front();
        t.max_ms    = v.back();
        t.median_ms = v[v.size() / 2];
        return t;
    }

    // DOLLAMA_BENCH=1 時の詳細ダンプ
    static void dump_stats(const PipelineStats& s)
    {
        auto line = [](const char* name, const StageTiming& t)
        {
            std::cerr << "  BENCH: " << name
                      << " median=" << t.median_ms << " ms"
                      << " (min=" << t.min_ms << " max=" << t.max_ms
                      << " N=" << t.count << ")\n";
        };
        std::cerr << "[mfp] BENCH ----------------------------------------\n";
        line("stage_a       ", s.stage_a);
        line("stage_b_clip  ", s.stage_b_clip);
        line("stage_b_sdxl  ", s.stage_b_sdxl);
        line("stage_c       ", s.stage_c);
        line("q_a_to_bclip  ", s.queue_a_to_bclip);
        line("q_bclip_bsdxl ", s.queue_bclip_to_bsdxl);
        line("q_bsdxl_to_c  ", s.queue_bsdxl_to_c);
        line("q_feedback    ", s.queue_feedback);
        std::cerr << "  BENCH: throughput=" << s.throughput_fps << " frames/s"
                  << " (frames=" << s.frames_processed << ")\n";
        std::cerr << "  BENCH: frame_latency median=" << s.frame_latency_median_ms
                  << " ms\n";
        std::cerr << "  BENCH: flagged=" << s.flagged_count
                  << " aborted=" << (s.aborted ? "true" : "false") << "\n";
        std::cerr << "[mfp] BENCH ----------------------------------------\n";
    }

    StageA     stage_a_;
    StageBClip stage_b_clip_;
    StageBSdxl stage_b_sdxl_;
    StageC     stage_c_;
};

} // namespace dollama

#pragma once

// dollama 推論パイプライン骨格 (Phase 1)
// stub(LLM) → CLIP(NPU) → stub(SDXL) → WD14(CPU) → feedback
// を std::jthread でマルチスレッド実行する。
//
// 設計: docs/pipeline-spec.md (スレッド/キュー/型/capacity/エラーハンドリング)
//       docs/cpu-topology.md (アフィニティマスク)
//
// SDXL / LLM は本フェーズではスタブ。CLIP は NPU、WD14 は CPU で実推論する。
// OpenVINO 必須のため #ifdef HAVE_OPENVINO でガードする。

#ifdef HAVE_OPENVINO

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// OpenVINO ヘッダを先に展開してから windows.h を含む affinity.hpp を入れる。
// (windows.h のマクロが OpenVINO ヘッダを壊すのを避けるため順序が重要)
#include "infer/clip.hpp"
#include "infer/wd14.hpp"
#include "core/queue.hpp"
#include "core/affinity.hpp"

namespace dollama {

// ----------------------------------------------------------------
// アフィニティマスク (名前付き定数・一箇所集約)
// このマシン (Intel Core Ultra 9 285 / probe11 実測) 固有の値。
// 将来は GetLogicalProcessorInformationEx で動的取得して置き換える。
// ----------------------------------------------------------------
constexpr uint64_t kPCoreMask = 0x00C03C03ULL; // P-core 8 論理コア
constexpr uint64_t kECoreMask = 0x003FC3FCULL; // E-core 16 論理コア

// ----------------------------------------------------------------
// Pipeline: Phase 1 マルチスレッドパイプライン
// ----------------------------------------------------------------
class Pipeline
{
public:
    // キュー間データ型・capacity は pipeline-spec.md に準拠
    using Tokens    = std::array<int32_t, ClipEncoder::SEQ_LEN>; // [77] token ids
    using Embedding = std::vector<float>;                        // [77*768] embedding
    using Image     = std::vector<float>;                        // ダミー画像 448*448*3
    using TagString = std::string;                              // feedback タグ列

    // キュー満杯/空時のタイムアウト (この時間内に進まなければ異常とみなす)
    static constexpr std::chrono::milliseconds kQueueTimeout{5000};

    // コンストラクタ: CLIP / WD14 のモデルをロードしてデバイスにコンパイルする
    // clip_xml: CLIP-L OV IR (.xml)、wd14_xml: WD14 SwinV2 OV IR (.xml)
    Pipeline(const std::string& clip_xml, const std::string& wd14_xml)
        : clip_(clip_xml, "NPU")
        , wd14_(wd14_xml, "CPU")
    {
    }

    // ------------------------------------------------------------
    // run: frames フレームを流し、各フレームのタグ文字列を集めて返す。
    // 内部で 4 スレッド (llm/clip/sdxl/wd14) を立て、stop_token で停止する。
    // タイムアウト・例外時はシャットダウンフラグを立てて全スレッド join し、
    // cerr にログを出して握り潰さない。
    // ------------------------------------------------------------
    std::vector<TagString> run(int frames)
    {
        // 各段のキュー (pipeline-spec.md の型/capacity)
        SPSCQueue<Tokens, 2>    llm_to_clip;
        SPSCQueue<Embedding, 2> clip_to_sdxl;
        SPSCQueue<Image, 2>     sdxl_to_wd14; // 画像は大きい。capacity の概念値は1だがキューは2の冪のみ許可するため2
        SPSCQueue<TagString, 4> wd14_to_llm;

        // 異常停止フラグ (タイムアウト・例外で立てる)
        std::atomic<bool> abort_flag{false};

        // 収集したタグ文字列 (wd14 スレッドのみが書き込む)
        std::vector<TagString> results;
        results.reserve(static_cast<size_t>(frames));

        // ========================================================
        // llm スレッド (stub, P-core)
        // 固定トークン列を frames 回 llm_to_clip へ push する。
        // TODO(Phase2): replace with BitNet b1.58
        // ========================================================
        std::jthread llm_thread(
            [&](std::stop_token st)
            {
                // 固定トークン列: BOS(49406) + anime(2368) + EOS(49407) + PAD(0)
                Tokens base{};
                base[0] = 49406;
                base[1] = 2368;
                base[2] = 49407;

                int fed_back = 0; // 受信したフィードバック数

                // 生産フェーズ: frames 回 push しつつ feedback を随時 drain する。
                // llm は wd14_to_llm の唯一の consumer なので、ここで吸い出さないと
                // feedback キューが満杯になりパイプラインが詰まる (SPSC 単一 consumer 契約)。
                for (int f = 0; f < frames; ++f)
                {
                    if (st.stop_requested() || abort_flag.load())
                    {
                        return;
                    }

                    // フレーム番号を 4 トークン目に埋めて決定的に変化させる
                    Tokens t = base;
                    t[3] = static_cast<int32_t>(f + 1);

                    if (!llm_to_clip.push_wait(t, kQueueTimeout))
                    {
                        std::cerr << "[pipeline] llm: llm_to_clip push timeout "
                                     "(frame " << f << ")\n";
                        abort_flag.store(true);
                        return;
                    }

                    // 溜まっているフィードバックをノンブロッキングで吸い出す
                    TagString fb;
                    while (wd14_to_llm.pop(fb))
                    {
                        ++fed_back;
                    }
                }

                // 排出フェーズ: 残りのフィードバックが frames 件揃うまで drain する。
                // TODO(Phase2): フィードバックタグで再プロンプトする
                while (fed_back < frames)
                {
                    if (st.stop_requested() || abort_flag.load())
                    {
                        return;
                    }

                    TagString fb;
                    if (!wd14_to_llm.pop_wait(fb, kQueueTimeout))
                    {
                        std::cerr << "[pipeline] llm: wd14_to_llm pop timeout "
                                     "(feedback " << fed_back << "/" << frames << ")\n";
                        abort_flag.store(true);
                        return;
                    }
                    ++fed_back;
                }
            });

        // ========================================================
        // clip スレッド (NPU, E-core)
        // トークンを pop → ClipEncoder.infer → embedding を clip_to_sdxl へ。
        // ========================================================
        std::jthread clip_thread(
            [&](std::stop_token st)
            {
                try
                {
                    for (int f = 0; f < frames; ++f)
                    {
                        if (st.stop_requested() || abort_flag.load())
                        {
                            return;
                        }

                        Tokens t;
                        if (!llm_to_clip.pop_wait(t, kQueueTimeout))
                        {
                            std::cerr << "[pipeline] clip: llm_to_clip pop timeout "
                                         "(frame " << f << ")\n";
                            abort_flag.store(true);
                            return;
                        }

                        Embedding emb = clip_.infer(t);

                        if (!clip_to_sdxl.push_wait(std::move(emb), kQueueTimeout))
                        {
                            std::cerr << "[pipeline] clip: clip_to_sdxl push timeout "
                                         "(frame " << f << ")\n";
                            abort_flag.store(true);
                            return;
                        }
                    }
                }
                catch (const std::exception& e)
                {
                    std::cerr << "[pipeline] clip: 例外 " << e.what() << "\n";
                    abort_flag.store(true);
                }
            });

        // ========================================================
        // sdxl スレッド (stub, P-core)
        // embedding を pop → ダミー画像 (448*448*3, 0-255) を決定的に生成 →
        // sdxl_to_wd14 へ。
        // TODO(Phase2): replace with CUDA UNet
        // ========================================================
        std::jthread sdxl_thread(
            [&](std::stop_token st)
            {
                try
                {
                    for (int f = 0; f < frames; ++f)
                    {
                        if (st.stop_requested() || abort_flag.load())
                        {
                            return;
                        }

                        Embedding emb;
                        if (!clip_to_sdxl.pop_wait(emb, kQueueTimeout))
                        {
                            std::cerr << "[pipeline] sdxl: clip_to_sdxl pop timeout "
                                         "(frame " << f << ")\n";
                            abort_flag.store(true);
                            return;
                        }

                        // embedding から決定的にダミー画像を生成する。
                        // 先頭数要素の総和を 0-255 にマップして一様画像を作る。
                        float acc = 0.0f;
                        const size_t take = emb.size() < 64 ? emb.size() : 64;
                        for (size_t i = 0; i < take; ++i)
                        {
                            acc += emb[i];
                        }
                        // 0-255 レンジへ折り返す (決定的)
                        float base = std::fabs(acc);
                        base = base - std::floor(base / 256.0f) * 256.0f;

                        Image img(Wd14Tagger::INPUT_NUMEL,
                                  static_cast<float>(base));

                        if (!sdxl_to_wd14.push_wait(std::move(img), kQueueTimeout))
                        {
                            std::cerr << "[pipeline] sdxl: sdxl_to_wd14 push timeout "
                                         "(frame " << f << ")\n";
                            abort_flag.store(true);
                            return;
                        }
                    }
                }
                catch (const std::exception& e)
                {
                    std::cerr << "[pipeline] sdxl: 例外 " << e.what() << "\n";
                    abort_flag.store(true);
                }
            });

        // ========================================================
        // wd14 スレッド (CPU, E-core)
        // 画像を pop → Wd14Tagger.infer → top_k からタグ文字列を作り
        // wd14_to_llm へ (feedback)。同時に results へ収集する。
        // ========================================================
        std::jthread wd14_thread(
            [&](std::stop_token st)
            {
                try
                {
                    for (int f = 0; f < frames; ++f)
                    {
                        if (st.stop_requested() || abort_flag.load())
                        {
                            return;
                        }

                        Image img;
                        if (!sdxl_to_wd14.pop_wait(img, kQueueTimeout))
                        {
                            std::cerr << "[pipeline] wd14: sdxl_to_wd14 pop timeout "
                                         "(frame " << f << ")\n";
                            abort_flag.store(true);
                            return;
                        }

                        std::vector<float> scores = wd14_.infer(img);
                        std::vector<size_t> top   = Wd14Tagger::top_k(scores, 5);

                        // top-k インデックスとスコアからタグ文字列を組む
                        // (CSV 非依存。タグ名解決は Phase2 で CharacterBible 連携)
                        TagString tags;
                        for (size_t r = 0; r < top.size(); ++r)
                        {
                            if (r != 0)
                            {
                                tags += ", ";
                            }
                            tags += "tag" + std::to_string(top[r]);
                        }

                        results.push_back(tags);

                        // feedback キューへ (満杯でも results 収集は済んでいる)
                        if (!wd14_to_llm.push_wait(tags, kQueueTimeout))
                        {
                            std::cerr << "[pipeline] wd14: wd14_to_llm push timeout "
                                         "(frame " << f << ")\n";
                            abort_flag.store(true);
                            return;
                        }
                    }
                }
                catch (const std::exception& e)
                {
                    std::cerr << "[pipeline] wd14: 例外 " << e.what() << "\n";
                    abort_flag.store(true);
                }
            });

        // CPU アフィニティを設定する (probe11 実測マスク)。
        // llm/sdxl は P-core、clip/wd14 は E-core に固定する。
        // 失敗 (mask=0 / API 失敗 / 非対応 OS) は性能最適化が効かないだけで
        // 機能には影響しないため、警告ログのみ出して続行する。
        if (!set_thread_affinity(llm_thread, kPCoreMask))
        {
            std::cerr << "[pipeline] 警告: llm_thread のアフィニティ設定に失敗\n";
        }
        if (!set_thread_affinity(clip_thread, kECoreMask))
        {
            std::cerr << "[pipeline] 警告: clip_thread のアフィニティ設定に失敗\n";
        }
        if (!set_thread_affinity(sdxl_thread, kPCoreMask))
        {
            std::cerr << "[pipeline] 警告: sdxl_thread のアフィニティ設定に失敗\n";
        }
        if (!set_thread_affinity(wd14_thread, kECoreMask))
        {
            std::cerr << "[pipeline] 警告: wd14_thread のアフィニティ設定に失敗\n";
        }

        // jthread のデストラクタが request_stop + join を行う。
        // 全段が frames 回処理し終えるか、abort_flag が立つと各スレッドが抜ける。
        llm_thread.join();
        clip_thread.join();
        sdxl_thread.join();
        wd14_thread.join();

        if (abort_flag.load())
        {
            std::cerr << "[pipeline] 異常終了 (タイムアウトまたは例外)。"
                         "収集済みフレーム数=" << results.size() << "\n";
        }

        return results;
    }

private:
    ClipEncoder clip_;
    Wd14Tagger  wd14_;
};

} // namespace dollama

#endif // HAVE_OPENVINO

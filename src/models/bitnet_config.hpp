#pragma once
// BitNet dense アーキ次元の単一情報源 (3 ファイル共有)。
//
// 施策 D「容量増のコード化」: アーキ次元を **ビルド時 config** で切替可能にする。
//   default (現行・無改変) = 32,976,896 params (≈ 33M)。
//   d80m (容量増)          = 79,908,864 params (≈ 80M)。
//
//   models/bitnet.hpp (参照 forward) / infer/bitnet.hpp (CPU dense 推論) /
//   infer/bitnet_gpu.cuh (GPU 推論) の 3 箇所が **必ず同一の次元** を見るように、
//   ここで一元定義する。各ファイルの constexpr はこの値を引くだけにする。
//
// ────────────────────────────────────────────────────────────────
// config 切替 (コンパイル時マクロ DOLLAMA_BITNET_ARCH):
//   未定義 / 0 / "default" → 32.98M (default ビルド・既存 golden / 本番重み非回帰)
//   1 / "d80m"             → 79.91M (容量増・本訓練/GPU 実走/golden 差し替えは別タスク)
//
//   据え置き (変えると tokenizer / 全 golden / カーネル head_dim 前提が連鎖崩壊する):
//     VOCAB_SIZE=4999 / D_MODEL=512 / N_HEADS=8 / HEAD_DIM=64 / MAX_SEQ_LEN=64
//   伸ばすのは N_LAYERS と FFN_DIM のみ。
//
//   nvcc (.cu) でも cl (.cpp) でも同一の値が見えるよう、純粋な整数マクロ /
//   constexpr のみで構成する (STL / heavy include に依存しない)。
// ────────────────────────────────────────────────────────────────

#include <cstddef>

// DOLLAMA_BITNET_ARCH を整数フラグに正規化する。
//   - 未定義 → 0 (default)。
//   - 文字列 "d80m" 等は meson 側で整数 (=1) に変換して渡す前提だが、
//     保険として >=1 の任意整数を d80m として扱う。
#ifndef DOLLAMA_BITNET_ARCH
#define DOLLAMA_BITNET_ARCH 0
#endif

namespace dollama
{
namespace bitnet_arch
{

// ── 全 config 共通 (据え置き) ────────────────────────────────────
constexpr int    VOCAB_SIZE  = 4999;  // specials 5 + tags 4994 (dataset-spec §3.2)
constexpr int    D_MODEL     = 512;
constexpr int    N_HEADS     = 8;
constexpr int    HEAD_DIM    = D_MODEL / N_HEADS;  // 64
constexpr int    MAX_SEQ_LEN = 64;
constexpr double ROPE_BASE   = 10000.0;
constexpr double RMS_EPS     = 1e-5;

// ── config 依存 (伸ばすのは N_LAYERS / FFN_DIM のみ) ──────────────
#if DOLLAMA_BITNET_ARCH >= 1
// d80m: 79,908,864 params (≈ 80M)。
constexpr int N_LAYERS = 16;
constexpr int FFN_DIM  = 2464;
#else
// default: 32,976,896 params (≈ 33M)。
constexpr int N_LAYERS = 8;
constexpr int FFN_DIM  = 1792;
#endif

// 任意次元から dense パラメータ要素数を計算する free 関数 (embed tied)。
//   1 バイナリで複数 config の数値を検証するために使う (test が両 config を一度に
//   突合できるよう、class の constexpr 既定とは独立に値を取れる形)。
//   レイアウトは models/bitnet.hpp / infer/bitnet.hpp と完全一致:
//     embed (tied)       : vocab * d_model
//     per layer attn q/k/v/o : 4 * d_model * d_model
//     per layer ffn g/u/d    : 3 * ffn_dim * d_model
//     per layer rmsnorm ×2   : 2 * d_model
//     final rmsnorm          : d_model
constexpr size_t param_count_for(int vocab, int d_model, int n_layers,
                                 int /*n_heads*/, int ffn_dim)
{
    const size_t embed      = static_cast<size_t>(vocab) * d_model;
    const size_t per_attn   = 4ull * d_model * d_model;
    const size_t per_ffn    = 3ull * static_cast<size_t>(ffn_dim) * d_model;
    const size_t per_norm   = 2ull * d_model;
    const size_t per_layer  = per_attn + per_ffn + per_norm;
    const size_t final_norm = d_model;
    return embed + per_layer * static_cast<size_t>(n_layers) + final_norm;
}

// 現在ビルドされている config のパラメータ数 (constexpr で確定)。
constexpr size_t param_count()
{
    return param_count_for(VOCAB_SIZE, D_MODEL, N_LAYERS, N_HEADS, FFN_DIM);
}

} // namespace bitnet_arch
} // namespace dollama

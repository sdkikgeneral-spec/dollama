// broadcast bias add — ホストラッパー宣言 (Phase 2 マイルストーン 2-5 / UNet)
// 対象: RTX5080 (Blackwell / sm_120) / CUDA Runtime API のみ
#pragma once

#include <cuda_fp16.h>

namespace dollama
{

// ----------------------------------------------------------------
// broadcast bias add: 2 用途のブロードキャスト加算
// ----------------------------------------------------------------
// 既存 elementwise add は同形状 (out[i]=a[i]+b[i]) のみ。ここでは形状の異なる
// bias を「ブロードキャストして」加算する 2 ケースを提供する。
// 共通規約: FP16 in/out / 内部加算は FP32 蓄積。in-place 安全 (out==in 可。
// 各スレッドは自分の 1 要素のみ読み書きするため依存なし)。
// すべてデバイスポインタを受け取る (内部で cudaMalloc しない)。
// 起動後に CUDA_CHECK_KERNEL() でエラー検査を行う。
//
// ----------------------------------------------------------------
// (a) row-vector bias: Linear 出力 [tokens, C] に bias [C] を加算
// ----------------------------------------------------------------
//   各行に同じ bias を足す: out[t, c] = in[t, c] + bias[c]。
//   レイアウト row-major [tokens, C]。out[t*C + c] = in[t*C + c] + bias[c]。
//   Linear (GEMM) の後段で使う。
//
//   d_in   : 入力 (FP16, [tokens, C])
//   d_bias : バイアス (FP16, [C])
//   d_out  : 出力 (FP16, [tokens, C])。d_in と同一でも可。
//   tokens : 行数
//   C      : 列数 (= bias の長さ)
void launch_bias_add_rowvec(const __half* d_in,
                            const __half* d_bias,
                            __half*       d_out,
                            int           tokens,
                            int           C);

// ----------------------------------------------------------------
// (b) per-channel scalar bias: 特徴 [N, C, H, W] に per-channel [C] を加算
// ----------------------------------------------------------------
//   各チャネル平面 (H*W 要素) に同じスカラを足す:
//     out[n, c, h, w] = in[n, c, h, w] + bias[c]。
//   レイアウト row-major NCHW。out[((n*C+c)*H+h)*W+w] = in[...] + bias[c]。
//   ResnetBlock で time embedding を空間にブロードキャスト加算するのに使う
//   (time emb を Linear で [C] に射影 → 各チャネル平面へ broadcast)。
//
//   d_in   : 入力 (FP16, [N, C, H, W])
//   d_bias : per-channel バイアス (FP16, [C])
//   d_out  : 出力 (FP16, [N, C, H, W])。d_in と同一でも可。
//   N,C,H,W: 論理次元
void launch_bias_add_channel(const __half* d_in,
                             const __half* d_bias,
                             __half*       d_out,
                             int           N,
                             int           C,
                             int           H,
                             int           W);

} // namespace dollama

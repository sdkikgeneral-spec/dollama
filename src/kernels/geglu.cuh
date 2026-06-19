// GEGLU — ホストラッパー宣言 (Phase 2 マイルストーン 2-5 / UNet FeedForward)
// 対象: RTX5080 (Blackwell / sm_120) / CUDA Runtime API のみ
#pragma once

#include <cuda_fp16.h>

namespace dollama
{

// ----------------------------------------------------------------
// GEGLU: diffusers FeedForward の GEGLU 活性化
// ----------------------------------------------------------------
// diffusers の GEGLU は Linear で [tokens, d] → [tokens, 2*d] に射影した後、
// 最終次元を前半 x1[:d] と後半 x2[d:] に等分し、out = x1 * gelu(x2) を返す。
// ここではその「分割 + gelu + 要素積」部分のみ担当する (Linear/proj は GEMM 側)。
//   入力レイアウト: row-major [tokens, 2*d]。row t の要素アドレスは in[t*(2*d) + j]。
//     x1[t,c] = in[t*(2*d) + c]        (c = 0..d-1)
//     x2[t,c] = in[t*(2*d) + d + c]    (c = 0..d-1)
//   出力レイアウト: row-major [tokens, d]。out[t*d + c] = x1[t,c] * gelu(x2[t,c])。
//
// gelu:
//   erf 版 (activation.cu の gelu_erf_fp16 と一致):
//     gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))   (1/sqrt(2) = 0.70710678f)
//   diffusers の GEGLU デフォルトも erf 版。
//
// dtype:
//   入出力は __half (FP16)。内部計算は必ず FP32 (gelu の erf を FP16 で取ると破綻)。
//
// in-place 不可:
//   出力は入力と次元が異なる ([tokens,2*d] → [tokens,d]) ため別バッファが必要。
//
// 起動:
//   1D グリッド (1 スレッド = 出力 1 要素 = tokens*d 個)。threads=256。
//   すべてデバイスポインタを受け取る (内部で cudaMalloc しない)。
//   起動後に CUDA_CHECK_KERNEL() でエラー検査を行う。
//
// 引数:
//   d_in   : 入力デバイスポインタ (FP16, [tokens, 2*d])
//   d_out  : 出力デバイスポインタ (FP16, [tokens, d])。d_in との別バッファ必須。
//   tokens : 行数
//   d      : 出力の最終次元 (入力の最終次元の半分)
void launch_geglu(const __half* d_in, __half* d_out, int tokens, int d);

} // namespace dollama

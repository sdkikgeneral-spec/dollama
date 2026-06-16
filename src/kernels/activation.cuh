// 要素ごと活性化関数 — ホストラッパー宣言 (Phase 2 マイルストーン 2-2-2)
// 対象: RTX5080 (Blackwell / sm_120) / CUDA Runtime API のみ
#pragma once

#include <cuda_fp16.h>

namespace dollama
{

// ----------------------------------------------------------------
// 要素ごと活性化関数 (SiLU / GeLU)
// ----------------------------------------------------------------
// dtype:
//   入出力は __half (FP16)。内部計算は必ず FP32 で行う
//   (__half2float でデコード → FP32 で計算 → __float2half でエンコード)。
//   FP16 のまま exp/erf を取ると精度が破綻するため、後続カーネルでもこの方針を踏襲する。
//
// in-place 安全:
//   d_in == d_out を許容する。各スレッドは out[i] = f(in[i]) のみで
//   他要素を参照しないため、入出力が同一バッファでもレースは発生しない。
//
// 起動:
//   1D グリッド、threads=256、blocks=ceil_div(n, 256)。
//   すべてデバイスポインタを受け取る (launch_vector_add / launch_gemm_fp16 と同規約、
//   内部で cudaMalloc しない)。起動後に CUDA_CHECK_KERNEL() でエラー検査を行う。
//
// 引数:
//   d_in  : 入力デバイスポインタ (FP16, 長さ n)
//   d_out : 出力デバイスポインタ (FP16, 長さ n)。d_in と同一でも可。
//   n     : 要素数 (n <= 0 は no-op)

// SiLU (Swish): f(x) = x * sigmoid(x) = x / (1 + exp(-x))
void launch_silu(const __half* d_in, __half* d_out, int n);

// GeLU 正確版 (erf): f(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
//   SDXL / diffusers の GEGLU デフォルトはこの erf 版。主経路として使う。
void launch_gelu(const __half* d_in, __half* d_out, int n);

// GeLU tanh 近似版: f(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//   互換用に併設。主経路は erf 版 (launch_gelu)。
void launch_gelu_tanh(const __half* d_in, __half* d_out, int n);

} // namespace dollama

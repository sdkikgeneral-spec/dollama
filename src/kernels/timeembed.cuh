// sinusoidal time embedding — ホストラッパー宣言 (Phase 2 マイルストーン 2-5 / UNet)
// 対象: RTX5080 (Blackwell / sm_120) / CUDA Runtime API のみ
#pragma once

#include <cuda_fp16.h>

namespace dollama
{

// ----------------------------------------------------------------
// sinusoidal time embedding: diffusers get_timestep_embedding 相当
// ----------------------------------------------------------------
// scalar timestep (float) → [dim] の正弦波埋め込み。SDXL は dim=320。
// diffusers の実装 (flip_sin_to_cos=True, downscale_freq_shift=0, scale=1) に厳密一致:
//
//   half     = dim / 2
//   exponent = -log(max_period) * arange(half) / half
//   freqs    = exp(exponent)                       (= max_period^(-i/half), i=0..half-1)
//   args     = timestep * freqs                    ([half])
//   emb      = concat([cos(args), sin(args)])      (flip_sin_to_cos=True → cos が先)
//
// 出力レイアウト [dim]:
//   out[i]        = cos(args[i])   (i = 0..half-1)
//   out[half + i] = sin(args[i])   (i = 0..half-1)
//
// dim が奇数の場合は diffusers が末尾を 1 要素ゼロパッドするが、SDXL は dim=320 (偶数)
// なのでここでは half = dim/2 とし、奇数 dim は out[dim-1] を生成しない (= 末尾未書き込み)
// 想定では使わない。安全のため奇数 dim でも out[2*half..dim-1] は触らない。
//
// dtype:
//   出力は __half (FP16)。内部計算は必ず FP32 (exp/cos/sin は FP32)。
//
// 起動:
//   1D グリッド、1 スレッド = 出力 1 要素 (half 個の freq を half スレッドが担当し、
//   各スレッドが cos と sin の 2 要素を書く)。dim は SDXL で 320 と小さいため 1 ブロックで足りる。
//   すべてデバイスポインタを受け取る (内部で cudaMalloc しない)。
//   起動後に CUDA_CHECK_KERNEL() でエラー検査を行う。
//
// 引数:
//   d_out      : 出力デバイスポインタ (FP16, [dim])
//   timestep   : スカラ timestep (host 側 float。カーネル引数として値渡し)
//   dim        : 埋め込み次元 (SDXL 320)
//   max_period : 周波数の最大周期 (diffusers 既定 10000)
void launch_timestep_embedding(__half* d_out,
                               float   timestep,
                               int     dim,
                               float   max_period = 10000.0f);

} // namespace dollama

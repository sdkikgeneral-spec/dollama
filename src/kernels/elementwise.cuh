// Elementwise — ホストラッパー宣言 (Phase 2 マイルストーン 2-4 / VAE 準備)
// 対象: RTX5080 (Blackwell / sm_120) / CUDA Runtime API のみ
#pragma once

#include <cuda_fp16.h>

// ----------------------------------------------------------------
// VAE decoder (および将来の UNet) で使う素朴な elementwise カーネル群。
// vae_decode.cu に埋めず独立 TU にして再利用可能にする。
// 共通規約: FP16 in/out / 内部蓄積は FP32 / レイアウトは row-major NCHW。
// すべてデバイスポインタを受け取る (内部で cudaMalloc しない)。
// 起動後に CUDA_CHECK_KERNEL() でエラー検査を行う。
// ----------------------------------------------------------------

namespace dollama
{

// ----------------------------------------------------------------
// residual add: out[i] = a[i] + b[i]
//   FP16 in/out / 内部加算は FP32 蓄積 (a,b を FP32 化して加算 → __float2half)。
//   1D グリッド (n 要素を線形に処理)。形状の解釈は呼び側に委ねる (要素数 n のみ)。
//   in-place 安全: out == a または out == b でも正しく動く (各スレッドが
//   自分の 1 要素しか読み書きしないため依存が無い)。
//   ResnetBlock の skip 接続 (hidden + residual) で使う。
// ----------------------------------------------------------------
void launch_add(const __half* d_a, const __half* d_b, __half* d_out, int n);

// ----------------------------------------------------------------
// nearest-neighbor upsample 2x: [N,C,H,W] -> [N,C,2H,2W]
//   各出力画素は対応する入力画素のコピー: out[n,c,ho,wo] = in[n,c,ho/2,wo/2]。
//   diffusers の Upsample2D (scale_factor=2, mode="nearest") 相当。
//   レイアウトは row-major NCHW (conv2d.cuh と同じ要素アドレス規約):
//     in[((n*C + c)*H + hi)*W + wi]
//     out[((n*C + c)*(2H) + ho)*(2W) + wo]
//   in-place 不可 (出力サイズが入力の 4 倍で別バッファが必要)。
// ----------------------------------------------------------------
void launch_upsample_nearest2x(const __half* d_in, __half* d_out, int N, int C, int H, int W);

} // namespace dollama

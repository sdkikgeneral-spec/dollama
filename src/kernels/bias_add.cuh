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

// ----------------------------------------------------------------
// (c) G-4k S2 P1: conv bias + per-(n,c) time-bias の後段融合
// ----------------------------------------------------------------
//   resnet_block conv1 の epilogue 用。conv を d_bias=nullptr で回した出力
//   d_conv[N,Cout,H,W] に対し、GEMM 経路の 2 パス
//     段1: conv_bias_add_rows 相当  t = f2h( h2f(conv) + h2f(convbias[c]) )
//     段2: bias_add_channel 相当  out = f2h( h2f(t) + h2f(tproj[n*Cout+c]) )
//   の丸め列を 1 カーネルで 1:1 再現する (中間 t を必ず half に落とす 2 段丸め)。
//   ※ bit 一致するのは conv が GEMM 経路 (bias を後付け丸め) のときのみ。
//     direct 経路 (bias を FP32 蓄積へ畳む単一丸め) の shape に当ててはならない
//     (呼び側は conv2d_uses_gemm_bias_path でガードすること)。
//
//   d_conv    : conv 出力 (FP16, [B, Cout, H, W])。d_out と同一でも可 (in-place 安全)。
//   d_convbias: conv バイアス (FP16, [Cout])
//   d_tproj   : time embedding 射影 (FP16, [B, Cout]。per-(n,c) で異なる)
//   d_out     : 出力 (FP16, [B, Cout, H, W])
void launch_conv_bias_biasch(const __half* d_conv,
                             const __half* d_convbias,
                             const __half* d_tproj,
                             __half*       d_out,
                             int           B,
                             int           Cout,
                             int           H,
                             int           W);

// ----------------------------------------------------------------
// (d) G-4k S2 P2: conv bias + residual add の後段融合
// ----------------------------------------------------------------
//   resnet_block conv2 の epilogue 用。丸め列は (c) と同型:
//     段1: conv_bias_add_rows 相当  t = f2h( h2f(conv) + h2f(convbias[c]) )
//     段2: add_fp16 相当          out = f2h( h2f(t) + h2f(residual[idx]) )
//   GEMM 経路限定の注意も (c) と同じ (conv2d_uses_gemm_bias_path でガード)。
//
//   d_conv    : conv 出力 (FP16, [B, Cout, H, W])。d_out と同一でも可 (in-place 安全)。
//   d_convbias: conv バイアス (FP16, [Cout])
//   d_residual: 残差 (FP16, [B, Cout, H, W]。skip 接続の x または conv_shortcut(x))
//   d_out     : 出力 (FP16, [B, Cout, H, W])
void launch_conv_bias_residual(const __half* d_conv,
                               const __half* d_convbias,
                               const __half* d_residual,
                               __half*       d_out,
                               int           B,
                               int           Cout,
                               int           H,
                               int           W);

} // namespace dollama

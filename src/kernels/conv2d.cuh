// Conv2d — ホストラッパー宣言 (Phase 2 マイルストーン 2-2-4)
// 対象: RTX5080 (Blackwell / sm_120) / CUDA Runtime API のみ
#pragma once

#include <cuda_fp16.h>

// ----------------------------------------------------------------
// 最適化メモ (後続で性能チューニングする際の指針):
//   - 1x1 conv (KH=KW=1, pad=0, stride=1) は実質 GEMM 相当
//     (out[n,co,h,w] = Σ_ci weight[co,ci] * in[n,ci,h,w])。SDXL の projection や
//     ResBlock の skip 1x1 はここに該当し、本来は gemm.cu のタイリング GEMM に
//     落とした方が速い。本段は direct で正しさを優先する。
//   - 3x3 conv は im2col + GEMM、または shared memory への入力タイル展開
//     (ハロー込みタイリング) へ昇格する余地がある。weight をレジスタ/shared に
//     常駐させ、入力タイルを再利用すればグローバルメモリ読み出しを大幅削減できる。
//   - 本段 (2-2-4) は 1 スレッド = 出力 1 画素の direct 実装。各スレッドが
//     Cin*KH*KW を FP32 蓄積する単純版で、まず数値正当性とベンチ基準を確立する。
//     ボトルネック化したら 1x1=GEMM / 3x3=im2col・タイリングへ差し替える。
// ----------------------------------------------------------------

namespace dollama
{

// ----------------------------------------------------------------
// direct 2D convolution (NCHW, FP16 in/weight/out, FP32蓄積)
//   in:[N,Cin,H,W] weight:[Cout,Cin,KH,KW] bias:[Cout]またはnullptr out:[N,Cout,Hout,Wout]
//   Hout=(H+2*pad_h-dilation_h*(KH-1)-1)/stride_h+1
//   Wout=(W+2*pad_w-dilation_w*(KW-1)-1)/stride_w+1
// レイアウトはすべて row-major。要素アドレス:
//   in[((n*Cin + ci)*H + hi)*W + wi]
//   weight[((co*Cin + ci)*KH + kh)*KW + kw]
//   out[((n*Cout + co)*Hout + ho)*Wout + wo]
// 範囲外 (パディング域) の入力はゼロとして扱う (ゼロパディング)。
// bias != nullptr のとき出力にチャネルごとの bias[co] を加算する。
// すべてデバイスポインタを受け取る (内部で cudaMalloc しない)。
// 起動後に CUDA_CHECK_KERNEL() でエラー検査を行う。
// ----------------------------------------------------------------
void launch_conv2d(const __half* d_in, const __half* d_weight, const __half* d_bias,
                   __half* d_out, int N, int Cin, int H, int W, int Cout, int KH, int KW,
                   int stride_h, int stride_w, int pad_h, int pad_w, int dilation_h, int dilation_w);

} // namespace dollama

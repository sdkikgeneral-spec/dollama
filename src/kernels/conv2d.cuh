// Conv2d — ホストラッパー宣言 (Phase 2 マイルストーン 2-2-4 / 2-6 S3)
// 対象: RTX5080 (Blackwell / sm_120) / CUDA Runtime API のみ
#pragma once

#include <cuda_fp16.h>

// ----------------------------------------------------------------
// 実装メモ (S3 で direct → GEMM 経路を追加):
//   - 1x1 conv (KH=KW=1, pad=0, stride=1) は実質 GEMM。weight[Cout,Cin] @ in[Cin,HW]
//     を wmma Tensor Core GEMM (launch_gemm_fp16, transB=false) に落とす。
//   - 3x3 等 (KH/KW>1) は im2col + wmma GEMM。in を [Cin*KH*KW, Hout*Wout] に展開し
//     weight[Cout, Cin*KH*KW] と GEMM。im2col バッファが大きい場合は Hout を帯分割。
//   - direct conv (1 スレッド = 出力 1 画素, FP32 蓄積) は温存し、N>1 / 小チャネル /
//     小解像度 のフォールバックとして launch_conv2d 内で形状により選択する。
//   - 数値は全経路で FP32 蓄積 (GEMM の accumulator も FP32)。bias は GEMM が
//     非対応のため GEMM 後に行 (= co) ごとブロードキャスト加算する。
// ----------------------------------------------------------------

namespace dollama
{

// ----------------------------------------------------------------
// 2D convolution (NCHW, FP16 in/weight/out, FP32蓄積)
//   in:[N,Cin,H,W] weight:[Cout,Cin,KH,KW] bias:[Cout]またはnullptr out:[N,Cout,Hout,Wout]
//   Hout=(H+2*pad_h-dilation_h*(KH-1)-1)/stride_h+1
//   Wout=(W+2*pad_w-dilation_w*(KW-1)-1)/stride_w+1
// レイアウトはすべて row-major。要素アドレス:
//   in[((n*Cin + ci)*H + hi)*W + wi]
//   weight[((co*Cin + ci)*KH + kh)*KW + kw]
//   out[((n*Cout + co)*Hout + ho)*Wout + wo]
// 範囲外 (パディング域) の入力はゼロとして扱う (ゼロパディング)。
// bias != nullptr のとき出力にチャネルごとの bias[co] を加算する。
// 形状により 1x1=GEMM / 一般=im2col+GEMM / direct を内部で自動選択する (数値は同一)。
// im2col 経路は内部で一時バッファを cudaMalloc/cudaFree する (帯分割で VRAM を抑制)。
// それ以外はすべてデバイスポインタを受け取る (内部で in/weight/out は確保しない)。
// 起動後に CUDA_CHECK_KERNEL() でエラー検査を行う。
// ----------------------------------------------------------------
void launch_conv2d(const __half* d_in, const __half* d_weight, const __half* d_bias,
                   __half* d_out, int N, int Cin, int H, int W, int Cout, int KH, int KW,
                   int stride_h, int stride_w, int pad_h, int pad_w, int dilation_h, int dilation_w);

// ----------------------------------------------------------------
// direct conv (1 スレッド = 出力 1 画素, FP32 蓄積) を明示的に起動する。
//   launch_conv2d が形状により選ぶ GEMM 経路をバイパスし、必ず direct で計算する。
//   GEMM 経路 (im2col+wmma / 帯分割) の数値検証用に、信頼できる参照として test から
//   呼べるよう公開する (大形状は CPU 6 重ループが非現実的なため GPU 同士で突合する)。
//   Hout/Wout は呼び出し側が conv_out_dim と同一式で計算して渡すこと。
// ----------------------------------------------------------------
void launch_conv2d_direct(const __half* d_in, const __half* d_weight, const __half* d_bias,
                          __half* d_out, int N, int Cin, int H, int W, int Cout,
                          int KH, int KW, int Hout, int Wout,
                          int stride_h, int stride_w, int pad_h, int pad_w,
                          int dilation_h, int dilation_w);

} // namespace dollama

// LayerNorm — ホストラッパー宣言 (Phase 2 マイルストーン 2-5 / UNet Transformer2D)
// 対象: RTX5080 (Blackwell / sm_120) / CUDA Runtime API のみ
#pragma once

#include <cuda_fp16.h>

namespace dollama
{

// ----------------------------------------------------------------
// LayerNorm: 最終次元 (feature / channel) で正規化 → affine
// ----------------------------------------------------------------
// GroupNorm とは正規化軸が異なる。LayerNorm は「1 行 (= 最終次元 C 要素) ごと」に
// mean / var を取る。Transformer2D / Attention 前後の norm で使う。
//   入力レイアウト: row-major [rows, C]。row r の要素アドレスは in[r*C + c]。
//   SDXL では rows = batch*tokens、C = 1280 / 640 などの埋め込み次元。
//
// 正規化 (PyTorch nn.LayerNorm 仕様、normalized_shape = (C,)):
//   mean_r = (1/C) Σ_c x[r,c]
//   var_r  = (1/C) Σ_c (x[r,c] - mean_r)^2    (biased, 1/C で割る)
//   out[r,c] = (x[r,c] - mean_r) / sqrt(var_r + eps) * gamma[c] + beta[c]
//
// dtype:
//   入出力は __half (FP16)。内部計算は必ず FP32。部分和 (sum, sum-sq) は float 蓄積し、
//   var = E[x^2] - E[x]^2、var = max(var, 0.0f) でクランプ (桁落ち対策。GroupNorm と同方針)。
//
// gamma / beta:
//   FP16, 長さ C。チャネル (最終次元) ごと。
//   nullptr を許容する: gamma==nullptr のとき scale=1、beta==nullptr のとき bias=0
//   (elementwise affine を持たない LayerNorm にも対応)。
//
// in-place 安全:
//   d_in == d_out を許容する。各ブロックは担当行を読み切って mean/var を確定してから
//   書き戻すため、入出力が同一バッファでもレースは発生しない。
//
// 起動:
//   1 行 = 1 ブロックの 1 パス FP32 リダクション。グリッド = rows ブロック。
//   ブロック内でグリッドストライドして部分和を集め、shared memory + warp shuffle の
//   ツリーリダクションで mean/var を求め、同ブロックが正規化 + affine を書き戻す。
//   すべてデバイスポインタを受け取る (内部で cudaMalloc しない)。
//   起動後に CUDA_CHECK_KERNEL() でエラー検査を行う。
//
// 引数:
//   d_in    : 入力デバイスポインタ (FP16, [rows, C])
//   d_gamma : affine scale (FP16, [C]) または nullptr
//   d_beta  : affine bias  (FP16, [C]) または nullptr
//   d_out   : 出力デバイスポインタ (FP16, [rows, C])。d_in と同一でも可。
//   rows    : 行数 (batch*tokens 等)
//   C       : 最終次元 (正規化軸の長さ)
//   eps     : 分散の数値安定化項 (既定 1e-5)
void launch_layer_norm(const __half* d_in,
                       const __half* d_gamma,
                       const __half* d_beta,
                       __half*       d_out,
                       int           rows,
                       int           C,
                       float         eps = 1e-5f);

} // namespace dollama

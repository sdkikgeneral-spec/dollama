// Attention — ホストラッパー宣言 (Phase 2 マイルストーン 2-2-5)
// 対象: RTX5080 (Blackwell / sm_120) / CUDA Runtime API のみ
#pragma once

#include <cuda_fp16.h>

// ----------------------------------------------------------------
// scaled dot-product attention (self / cross 共通)
//   O = softmax(scale * Q·Kᵀ) · V
//
// レイアウト (すべて row-major / contiguous):
//   Q: [B, H, Sq, Dh]
//   K: [B, H, Sk, Dh]
//   V: [B, H, Sk, Dh]
//   O: [B, H, Sq, Dh]
//   要素アドレス:
//     Q[((b*H + h)*Sq + i)*Dh + d]
//     K[((b*H + h)*Sk + j)*Dh + d]
//     V[((b*H + h)*Sk + j)*Dh + d]
//     O[((b*H + h)*Sq + i)*Dh + d]
//
// self-attention と cross-attention は同一カーネルで扱う。両者の違いは Q の
// 系列長 Sq と K/V の系列長 Sk が一致するか否か (self: Sq==Sk, cross: Sq!=Sk)
// だけで、Sq と Sk を別パラメータにすることで両対応する。
//
// FP16 in/out / 内部の積和・softmax は必ず FP32 蓄積 (GEMM/GroupNorm/Conv と同規約)。
// softmax は数値安定のため max 減算する 2 パス方式 (max → exp/sum → 正規化)。
//
// scale はデフォルト 1/sqrt(Dh) を呼び側が渡す (引数化)。
// ----------------------------------------------------------------

// ----------------------------------------------------------------
// 最適化メモ (後続で性能チューニングする際の指針):
//   - 本段 (2-2-5) は 1 ブロック = 1 つの (b, h, query 行 i) を担当し、scores[Sk] を
//     動的 shared memory (FP32) に置く「正しさ優先版」。Sk が大きい場合 (SDXL
//     self-attn の最大 Sk ~4096 でも 4096*4byte = 16KB) でも shared に収まる。
//   - 極端に Sk が大きく shared に収まらなくなったら、scores を materialize せず
//     online-softmax (flash-attention) へ昇格する。running max / running sum を
//     保持しながら K/V をタイル走査し、O のアキュムレータを逐次補正する方式。
//   - Q·Kᵀ と P·V はいずれも GEMM なので、Tensor Core (wmma/mma) を使う GEMM
//     カーネルへ委譲する余地もある。flash 化と GEMM 委譲は本段の数値正当性と
//     ベンチ基準を確立してから実測ベースで判断する。
// ----------------------------------------------------------------

namespace dollama
{

// scaled dot-product attention (self/cross 共通)
//   Q:[B,H,Sq,Dh] K:[B,H,Sk,Dh] V:[B,H,Sk,Dh] O:[B,H,Sq,Dh] (row-major)
//   O = softmax(scale * Q·Kᵀ) · V。FP16 in/out, FP32 蓄積。
// すべてデバイスポインタを受け取る (内部で cudaMalloc しない)。
// 起動後に CUDA_CHECK_KERNEL() でエラー検査を行う。
void launch_attention(const __half* d_q, const __half* d_k, const __half* d_v,
                      __half* d_out, int B, int H, int Sq, int Sk, int Dh, float scale);

// ----------------------------------------------------------------
// launch_attention_fast — FlashAttention-2 級の高速バリアント (FAST モード G-3k)
// ----------------------------------------------------------------
// 既存 launch_attention (1 block = 1 warp) は占有率が低く SM が空くのが SDXL UNet
// attention の単一最大律速 (実測 40.3% / 4.621s)。本バリアントは同一の数学
// (O = softmax(scale·Q·Kᵀ)·V, FP32 蓄積) を保ったまま以下で GPU を飽和させる:
//   1. multi-warp block 化 (1 block = N warp)。各 warp が自分の 16 行 query タイルを
//      担当し、K/V タイルは block 内 N warp で共有ロード → K/V の DRAM 再ロードを
//      N 分の 1 に償却し占有率を N 倍に。
//   2. cp.async のダブルバッファで K/V ロードと wmma をパイプライン化 (sm_120)。
//   3. QK^T / P·V を Tensor Core (wmma) で計算 (既存 wmma 経路を踏襲)。
// 数学は同一だが reduction 順が変わるため golden とは微差 (SSIM ゲート内)。
//
// 対応範囲: Dh が 16 の倍数 (UNet self/cross の Dh=64/80 が主役)。それ以外や
//   確保可能な shared を超える Dh は既存 launch_attention へフォールバックする
//   (呼び側は常に本関数を呼べばよい)。引数・レイアウトは launch_attention と同一。
void launch_attention_fast(const __half* d_q, const __half* d_k, const __half* d_v,
                           __half* d_out, int B, int H, int Sq, int Sk, int Dh, float scale);

} // namespace dollama

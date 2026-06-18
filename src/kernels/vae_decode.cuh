// SDXL AutoencoderKL decoder — ホストラッパー宣言 (Phase 2 マイルストーン 2-4)
// 対象: RTX5080 (Blackwell / sm_120) / CUDA Runtime API のみ
#pragma once

#include <cuda_fp16.h>
#include <string>

#include "io/safetensors.hpp"

namespace dollama
{

// ----------------------------------------------------------------
// SDXL VAE decoder
// ----------------------------------------------------------------
// latent [1,4,128,128] (= scaling_factor 0.13025 で割り済みのスケール後 latent)
// を image [1,3,1024,1024] へデコードする。
//
// 構造 (diffusers AutoencoderKL):
//   post_quant_conv (1x1, 4->4)
//   conv_in         (3x3 pad1, 4->512)
//   mid_block: Resnet(512) -> Attn(512 spatial self-attn) -> Resnet(512)
//   up_blocks[0]: Resnet x3 (512) -> Upsample (nearest2x + 3x3, 512)
//   up_blocks[1]: Resnet x3 (512) -> Upsample (512)
//   up_blocks[2]: Resnet x3 (512->256) -> Upsample (256)
//   up_blocks[3]: Resnet x3 (256->128)  (upsample なし)
//   conv_norm_out (GroupNorm32) -> SiLU -> conv_out (3x3 pad1, 128->3)
//
// dtype: 重みは FP16 (内部の積和・正規化は各カーネルが FP32 蓄積)。中間活性は
//   post_quant_conv 〜 up_block[1] が FP16、up_block[2] 以降は FP32 で保持する。
//   理由: up_block[2]/[3] の残差ブロック内は活性が FP16 範囲 (±65504) を超え (実測
//   up3 full-res で ±100000 超)、FP16 中間だと Inf 化 → GroupNorm で NaN 伝播するため。
//   FP32 中間化で final SSIM=0.99999 を達成 (詳細は vae_decode.cu の数値方針メモ)。
//
// 重みは SafeTensors (vae_weights.safetensors) からロードする。decoder.* と
// post_quant_conv.* のキーを使う。launch_vae_decode 内で必要な重みを GPU へ
// 一括アップロードし、中間バッファを cudaMalloc する。
//
// 引数:
//   weights      : ロード済み SafeTensors (decoder.* / post_quant_conv.* を含む)
//   d_latent     : 入力 latent デバイスポインタ (FP16, [1,4,128,128])
//   d_image_out  : 出力画像デバイスポインタ (FP16, [1,3,1024,1024])。事前確保必須。
//
// すべてデバイスポインタを受け取る (latent / image は呼び側が確保)。
// 中間バッファは内部で cudaMalloc / cudaFree する。起動後に CUDA_CHECK_KERNEL()。
// ----------------------------------------------------------------
void launch_vae_decode(const SafeTensors& weights,
                       const __half*      d_latent,
                       __half*            d_image_out);

} // namespace dollama

// SDXL UNet2DConditionModel forward — ホストラッパー宣言 (Phase 2 マイルストーン 2-5)
// 対象: RTX5080 (Blackwell / sm_120) / CUDA Runtime API のみ
#pragma once

#include <cuda_fp16.h>

#include "io/safetensors.hpp"

namespace dollama
{

// ----------------------------------------------------------------
// SDXL UNet2DConditionModel 1 step forward
// ----------------------------------------------------------------
// noise 予測 1 step を実行する。VAE decoder (vae_decode.cu) と同じ
// 「CUDA オーケストレーション + safetensors 重みロード + FP16 in/out」方針。
// 中間活性は FP16 (各カーネルは内部 FP32 蓄積)。S1 報告で UNet 全中間が FP16
// 範囲に収まることを確認済み。
//
// 構造 (diffusers UNet2DConditionModel, SDXL):
//   conv_in (3x3 pad1, 4->320)
//   time embed:   sinusoidal(320) -> linear_1(1280,320)+SiLU -> linear_2(1280,1280)
//   add embed:    concat(text_embeds[1280], time_ids 各成分 sinusoidal(256) -> [1536])
//                 -> linear_1(1280,2816)+SiLU -> linear_2(1280,1280)。temb に加算。
//   down_blocks[0] = DownBlock2D       (resnet x2 + downsample)
//   down_blocks[1] = CrossAttnDownBlock (resnet+Transformer2D[L=2] x2 + downsample)
//   down_blocks[2] = CrossAttnDownBlock (resnet+Transformer2D[L=10] x2, downsample なし)
//   mid_block      = UNetMidBlock2DCrossAttn (resnet, Transformer2D[L=10], resnet)
//   up_blocks[0]   = CrossAttnUpBlock  (resnet+Transformer2D[L=10] x3 + upsample)
//   up_blocks[1]   = CrossAttnUpBlock  (resnet+Transformer2D[L=2] x3 + upsample)
//   up_blocks[2]   = UpBlock2D         (resnet x3, upsample なし)
//   conv_norm_out (GroupNorm32) -> SiLU -> conv_out (3x3 pad1, 320->4)
//
// 解像度: 128 -> 64 -> 32 -> 64 -> 128。block_out_channels=(320,640,1280)。
// attention head_dim=64。
//
// 引数:
//   weights                : ロード済み SafeTensors (UNet 全重み, FP16)
//   d_latent               : 入力 latent (FP16, [1,4,128,128])。スケジューラでスケール済み。
//   timestep               : スカラ timestep (host float)
//   d_encoder_hidden_states: cross-attn の K/V 源 (FP16, [1,77,2048])
//   d_text_embeds          : pooled text embeds (FP16, [1,1280])
//   d_time_ids             : SDXL added time ids (FP16, [6])
//   d_noise_pred_out       : 出力 noise 予測 (FP16, [1,4,128,128])。事前確保必須。
//
// 中間バッファは内部で cudaMalloc / cudaFree する。起動後に CUDA_CHECK_KERNEL()。
// ----------------------------------------------------------------
// 段ごとのゴールデン突合フック登録 (テスト専用)。
//   launch_unet 内の各段出力 (conv_in_out / down_block_*_out / mid_block_out /
//   up_block_*_out / time_embedding_out / add_embedding_out / noise_pred など) で
//   hook(name, デバイスポインタ, 要素数) が呼ばれる。nullptr で解除。
using UnetStageHook = void (*)(const char* name, const __half* d_buf, size_t n);
void unet_set_stage_hook(UnetStageHook hook);

void launch_unet(const SafeTensors& weights,
                 const __half*      d_latent,
                 float              timestep,
                 const __half*      d_encoder_hidden_states,
                 const __half*      d_text_embeds,
                 const __half*      d_time_ids,
                 __half*            d_noise_pred_out);

} // namespace dollama

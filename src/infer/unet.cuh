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

// 後方互換: 呼び出しごとに全重みをデバイスへ転送する版 (test_unet が使用)。
//   attn_fast: FAST モード (G-3k)。true で self/cross attention を launch_attention_fast
//              へ切り替える。既定 false = 現行 (default) 挙動を 1 命令も変えず通す。
//   epilogue : FAST モード (G-4k S1a)。true で resnet の norm1/norm2 と conv_norm_out の
//              GroupNorm を multi-block 版 (launch_group_norm_mb) へ切り替える。
//              既定 false = 現行 (default) 挙動を 1 命令も変えず通す。
void launch_unet(const SafeTensors& weights,
                 const __half*      d_latent,
                 float              timestep,
                 const __half*      d_encoder_hidden_states,
                 const __half*      d_text_embeds,
                 const __half*      d_time_ids,
                 __half*            d_noise_pred_out,
                 bool               attn_fast = false,
                 bool               epilogue = false);

// ----------------------------------------------------------------
// デバイス常駐重みハンドル (S1 最適化)
// ----------------------------------------------------------------
// 上記 launch_unet(const SafeTensors&, ...) は呼び出しごとに全重み (5.1GB) を
// cudaMalloc + H2D 転送するため、拡散ループ (20step) では同じ重みを 20 回
// 再転送してしまう。これを避けるため、重みを一度だけデバイスへ常駐させて全 step で
// 使い回すハンドル API を提供する。
//
//   handle = unet_weights_create(weights);   // 全重みをデバイスへ常駐 (初回 launch で upload)
//   for step: launch_unet(handle, ...);      // 重み転送ゼロ・キャッシュ済みを使い回す
//   unet_weights_destroy(handle);            // 全デバイス重みを解放
//
// ハンドルは不透明ポインタ (内部実装 = DeviceWeights) で公開し、ヘッダ露出を最小化する。
// 注意: ハンドルは create に渡した SafeTensors の生存期間内でのみ有効
//       (内部で SafeTensors& を保持し、初回 launch 時に各重みを遅延 upload するため)。
using UnetWeightsHandle = void*;

// 重み常駐ハンドルを生成する。weights の生存期間より短く使うこと。
UnetWeightsHandle unet_weights_create(const SafeTensors& weights);

// ハンドルが保持する全デバイス重みを解放する。nullptr は無視。
void unet_weights_destroy(UnetWeightsHandle handle);

// 常駐重みハンドルを使う launch_unet。重み転送/再 malloc は発生しない (2step 目以降)。
//   attn_fast: FAST モード (G-3k)。既定 false = 現行 (default) 挙動。
//   epilogue : FAST モード (G-4k S1a)。既定 false = 現行 (default) 挙動。
void launch_unet(UnetWeightsHandle  handle,
                 const __half*      d_latent,
                 float              timestep,
                 const __half*      d_encoder_hidden_states,
                 const __half*      d_text_embeds,
                 const __half*      d_time_ids,
                 __half*            d_noise_pred_out,
                 bool               attn_fast = false,
                 bool               epilogue = false);

// ----------------------------------------------------------------
// バッチ (B>1) forward (G-2k S2)。CFG cond/uncond を B=2 に束ねて 1 forward で回す
// 下地。常駐重みハンドルを使う。各テンソルは b-major で連続:
//   d_latent [B,4,128,128] / d_encoder_hidden_states [B,77,2048] /
//   d_text_embeds [B,1280] / d_time_ids [6] (全 b 共有) / d_noise_pred_out [B,4,128,128]。
// B=1 で呼べば launch_unet(handle,...) と各カーネル起動列・数値が完全同一 (default 無改変)。
//   attn_fast: FAST モード (G-3k)。既定 false = 現行 (default) 挙動。
//   epilogue : FAST モード (G-4k S1a)。既定 false = 現行 (default) 挙動。
void launch_unet_batched(UnetWeightsHandle  handle,
                         int                B,
                         const __half*      d_latent,
                         float              timestep,
                         const __half*      d_encoder_hidden_states,
                         const __half*      d_text_embeds,
                         const __half*      d_time_ids,
                         __half*            d_noise_pred_out,
                         bool               attn_fast = false,
                         bool               epilogue = false);

} // namespace dollama

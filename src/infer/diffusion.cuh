// SDXL 拡散ループ結線モジュール — ホストラッパー宣言 (Phase 2 マイルストーン 2-6a / ST-1+ST-2)
// 対象: RTX5080 (Blackwell / sm_120) / CUDA Runtime API のみ
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <cuda_fp16.h>

#include "io/safetensors.hpp"
#include "infer/unet.cuh"
#include "kernels/vae_decode.cuh"

namespace dollama
{

// ----------------------------------------------------------------
// DiffusionPipeline — UNet × Nstep + Euler scheduler + VAE decode を結線し、
//   golden 埋め込み (encoder_hidden_states / text_embeds / time_ids) を入力に
//   noise → latent → 画像 [1024×1024 RGB uint8] を生成する。
// ----------------------------------------------------------------
// 方針 (2-6a):
//   - CFG なし (guidance_scale = 1.0 のみ。>1 は今回 TODO スタブ)。
//   - テキストエンコードは結線しない。golden 埋め込みをそのまま全 step 使い回す。
//   - UNet 重み / VAE 重み / golden 埋め込みは 1 回だけロードして保持 (再ロード厳禁、5GB)。
//   - 拡散ループ本体は host float (scheduler は host)。step ごとの latent
//     D2H/H2D (~256KB) は無視可能なので素直に往復する。
//
// 拡散ループ (epsilon 予測, EulerDiscreteScheduler):
//   sched.set_timesteps(steps)
//   latent_host = randn(seed) * sigmas[0]     // init_noise_sigma = sigmas[0]
//   for i in 0..steps-1:
//       scaled = scale_model_input(latent_host, i)        // host
//       H2D  d_latent <- scaled (FP16)
//       launch_unet(...)  → d_noise_pred (FP16)
//       D2H  noise_host(FP32) <- d_noise_pred
//       latent_host = step(noise_host, i, latent_host)    // host
//   latent_for_vae = latent_host / 0.13025
//   launch_vae_decode → d_image (FP16, [-1,1] 値域)
//   rgb = clamp(image*0.5+0.5, 0, 1) * 255  (uint8, [H,W,3] row-major)
//
// VAE 出力の画像化規約: VAE decoder の出力 (= decode().sample) は値域 ~[-1,1]。
//   dollma_dump_vae_golden.py と同じく (x*0.5+0.5) を掛けて [0,1] へ写し、clamp 後 ×255。
// ----------------------------------------------------------------
class DiffusionPipeline
{
public:
    // 重み 2 つ (UNet / VAE) と golden 埋め込み safetensors のパスを受けてロード・保持する。
    //   unet_weights_path : UNet 全重み (FP16, 5.1GB)
    //   vae_weights_path  : VAE decoder 重み (FP16)
    //   embeds_path       : golden 埋め込み (unet_io.safetensors)。以下のキーを使う:
    //       input_encoder_hidden_states_f16 [1,77,2048]
    //       input_text_embeds_f16           [1,1280]
    //       input_time_ids_f16              [6]
    DiffusionPipeline(const std::string& unet_weights_path,
                      const std::string& vae_weights_path,
                      const std::string& embeds_path);

    ~DiffusionPipeline();

    DiffusionPipeline(const DiffusionPipeline&)            = delete;
    DiffusionPipeline& operator=(const DiffusionPipeline&) = delete;

    // 拡散ループを実行し、1024×1024 RGB uint8 画像を生成する。
    //   steps          : 推論ステップ数 (例 20)
    //   seed           : 初期ノイズ randn のシード
    //   guidance_scale : CFG スケール。今回は 1.0 のみ対応 (>1 は std::runtime_error)。
    //   rgb_out        : [H*W*3] の uint8 RGB ([y,x,c] = row-major HWC) で返す。
    //   w, h           : 出力解像度 (常に 1024)。
    void generate(int                   steps,
                  uint64_t              seed,
                  float                 guidance_scale,
                  std::vector<uint8_t>& rgb_out,
                  int&                  w,
                  int&                  h);

    // guidance_scale=1.0 の簡略オーバーロード。
    void generate(int                   steps,
                  uint64_t              seed,
                  std::vector<uint8_t>& rgb_out,
                  int&                  w,
                  int&                  h);

private:
    SafeTensors unet_weights_;
    SafeTensors vae_weights_;

    // S1: UNet 全重みをデバイス常駐させたハンドル。構築時に 1 度だけ upload し、
    //     全 step で使い回す (重み再転送/再 malloc をゼロにする)。
    UnetWeightsHandle unet_weights_handle_ = nullptr;

    // S3-D: VAE decoder 全重み (~92MB) を構築時に 1 度だけ upload し常駐させたハンドル。
    //       毎回の VAE 重みアップロード (生成あたり固定 ~3.89s) を排除する。
    VaeWeightsHandle vae_weights_handle_ = nullptr;

    // golden 埋め込み (全 step 使い回す)。デバイス常駐。
    __half* d_encoder_hidden_states_ = nullptr;  // [1,77,2048]
    __half* d_text_embeds_           = nullptr;  // [1,1280]
    __half* d_time_ids_              = nullptr;  // [6]
};

} // namespace dollama

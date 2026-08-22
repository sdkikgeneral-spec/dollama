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
#include "server/fast_config.hpp" // FAST モードのフラグ枠 (G-0b・純 cpp・保持するだけ)

namespace dollama
{

// ----------------------------------------------------------------
// DiffusionPipeline — UNet × Nstep + Euler scheduler + VAE decode を結線し、
//   golden 埋め込み (encoder_hidden_states / text_embeds / time_ids) を入力に
//   noise → latent → 画像 [1024×1024 RGB uint8] を生成する。
// ----------------------------------------------------------------
// 方針 (2-6a):
//   - CFG なし (guidance_scale = 1.0 のみ。>1 は generate では TODO スタブ)。
//   - テキストエンコードは結線しない。golden 埋め込みをそのまま全 step 使い回す。
//   - UNet 重み / VAE 重み / golden 埋め込みは 1 回だけロードして保持 (再ロード厳禁、5GB)。
//   - 拡散ループ本体は host float (scheduler は host)。step ごとの latent
//     D2H/H2D (~256KB) は無視可能なので素直に往復する。
//
// 方針 (2-6b Stage E):
//   - generate_txt2img で CFG (classifier-free guidance) を実装。外部の cond/uncond
//     埋め込みを受け、各 step で UNet を 2 回回して合成する。CFG なし経路とは独立。
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
    //   fast_cfg          : FAST モードフラグ (G-0b)。既定 (全 off) は現行挙動。
    //       この Pkg では保持するだけで拡散経路に fast 分岐を一切足さない (byte-for-byte 無改変)。
    DiffusionPipeline(const std::string& unet_weights_path,
                      const std::string& vae_weights_path,
                      const std::string& embeds_path,
                      const FastConfig&  fast_cfg = FastConfig{});

    ~DiffusionPipeline();

    DiffusionPipeline(const DiffusionPipeline&)            = delete;
    DiffusionPipeline& operator=(const DiffusionPipeline&) = delete;

    // 拡散ループを実行し、1024×1024 RGB uint8 画像を生成する。
    //   steps          : 推論ステップ数 (例 20)
    //   seed           : 初期ノイズ randn のシード
    //   guidance_scale : CFG スケール。generate では 1.0 のみ対応 (>1 は std::runtime_error)。
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

    // ----------------------------------------------------------------
    // 2-6b Stage E: CFG (classifier-free guidance) 付き txt2img。
    //   外部 (Stage D の TextConditioner) が構築した cond / uncond の埋め込みを
    //   host float (FP32) で受け、内部で FP16 へ変換しデバイスへ常駐させる。
    //   コンストラクタが常駐させた golden 埋め込み (d_encoder_hidden_states_ 等) は
    //   一切使わない。CFG なし経路 (generate) とは独立。
    //
    //   各 step で UNet を cond / uncond の 2 回回し、
    //     noise = uncond + guidance_scale * (cond - uncond)
    //   を host で合成してから Euler step に渡す (noise_pred は 65536 要素と小さい)。
    //
    //   cond_ehs / uncond_ehs : encoder_hidden_states [77*2048] row-major (FP32)
    //   cond_text_embeds / uncond_text_embeds : text_embeds [1280] (FP32)
    //   time_ids : [6] (FP32, cond/uncond 共通)
    //   guidance_scale : CFG スケール (例 7.5)
    //   rgb_out / w / h : generate と同じ HWC uint8 出力。
    // ----------------------------------------------------------------
    // ランタイム LoRA (L-2)。生成前に常駐 UNet 重みへ焼き込み、生成後に復元する。
    //   apply_lora_file: LoRA safetensors (kohya 命名) をロードし、
    //     load_lora_modules (base = unet_weights_) → unet_apply_loras で
    //     W += strength*(alpha/rank)*(B@A) をデバイス上でマージする。
    //     ファイル不正・写像不能は std::runtime_error (適用途中失敗は呼び出し側が
    //     clear_loras してから再試行する)。
    //   clear_loras: patch 済み key を base host バイトから再 upload し bit-exact 復元。
    // ----------------------------------------------------------------
    void apply_lora_file(const std::string& path, float strength);
    void clear_loras();

    void generate_txt2img(int                   steps,
                          uint64_t              seed,
                          float                 guidance_scale,
                          const float*          cond_ehs,
                          const float*          cond_text_embeds,
                          const float*          uncond_ehs,
                          const float*          uncond_text_embeds,
                          const float*          time_ids,
                          std::vector<uint8_t>& rgb_out,
                          int&                  w,
                          int&                  h);

private:
    // G-8k T2 (F2/F4): デバイス資源の後始末。デストラクタと、コンストラクタ末尾の
    //   reserve 失敗時 catch の 2 箇所から呼ぶ (破棄順を 1 箇所に持つため)。
    //   例外は外へ出さない・冪等 (破棄したメンバは nullptr に落とす)。
    void destroy_resources() noexcept;

    SafeTensors unet_weights_;
    SafeTensors vae_weights_;

    // G-0b: FAST モードフラグ。構築時に受けて保持するだけ (この Pkg では未参照)。
    //   後続 Pkg (G-1k/G-2k/G-3k/G-4k/G-5k) が本メンバを参照して fast 経路を分岐する。
    FastConfig fast_cfg_;

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

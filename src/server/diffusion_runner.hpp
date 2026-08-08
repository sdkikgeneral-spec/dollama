// IDiffusionRunner — SDXL 拡散ループ (DiffusionPipeline) を純 cpp 層へ繋ぐ抽象 interface。
// (Phase 2 マイルストーン 2-6b Stage F / ウェーブ1)
//
// ----------------------------------------------------------------
// 設計意図 (重要):
//   pipeline_generator.hpp は DiffusionPipeline (= __half / CUDA 型を含む diffusion.cuh)
//   をヘッダに直接抱えるため #ifdef HAVE_CUDA でガードされ、.cu からしか include できない。
//   一方、本 interface は CUDA 非対応の翻訳単位 (cpp TU、例: api.cpp / HTTP サーバ層) からも
//   常に参照できる必要があるため、以下を厳守する:
//     - CUDA 型 (__half 等) も OpenVINO 型も一切シグネチャに露出させない。
//     - CUDA / OV ヘッダを include しない (std と cstdint 等の標準ヘッダのみ)。
//     - #ifdef HAVE_CUDA でガードしない (interface 宣言は常に可視)。
//   これにより cpp TU は IDiffusionRunner 越しに拡散を呼べ、実体 (CUDA 依存) は
//   .cu 側 (make_diffusion_runner の実装) に隔離される。
// ----------------------------------------------------------------
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "server/fast_config.hpp" // FAST モードのフラグ枠 (G-0b・DiffusionPipeline へ運ぶだけ)

namespace dollama
{

// ランタイム LoRA (L-2) のファイル単位リクエスト DTO (純 cpp・CUDA 型なし)。
//   path     : LoRA safetensors の実ファイルパス (name→path 解決は backend 層の責務)。
//   strength : 適用強度 (scale = strength * alpha / rank)。
struct LoraFileRequest
{
    std::string path;
    float       strength = 1.0f;
};

// 拡散ループの実行 interface。実体は .cu 側で DiffusionPipeline をラップして実装する。
struct IDiffusionRunner
{
    virtual ~IDiffusionRunner() = default;

    // ランタイム LoRA (L-2): 常駐 UNet 重みへ生成前にマージする / 復元する。
    //   default no-op (stub / 非対応 runner は無改修で無効)。実体 (DiffusionRunner) が
    //   override し、適用途中の失敗時は自ら復元してから例外を投げる。
    virtual void apply_loras(const std::vector<LoraFileRequest>& reqs) { (void)reqs; }
    virtual void clear_loras() {}

    // CFG (classifier-free guidance) 付き txt2img を 1 枚生成する。
    //   diffusion.cuh の DiffusionPipeline::generate_txt2img と 1:1 対応 (同じ引数規約)。
    //
    //   steps               : 推論ステップ数 (例 20)
    //   seed                : 初期ノイズ randn のシード
    //   guidance_scale      : CFG スケール (例 7.5)
    //   cond_ehs            : cond   encoder_hidden_states [77*2048] row-major (FP32)
    //   cond_text_embeds    : cond   text_embeds [1280] (FP32)
    //   uncond_ehs          : uncond encoder_hidden_states [77*2048] row-major (FP32)
    //   uncond_text_embeds  : uncond text_embeds [1280] (FP32)
    //   time_ids            : [6] (FP32, cond/uncond 共通)
    //   rgb_out             : [H*W*3] の uint8 RGB ([y,x,c] = row-major HWC) で返す。
    //   w, h                : 出力解像度 (常に 1024)。
    virtual void run_txt2img(
        int steps, uint64_t seed, float guidance_scale,
        const float* cond_ehs, const float* cond_text_embeds,
        const float* uncond_ehs, const float* uncond_text_embeds,
        const float* time_ids,
        std::vector<uint8_t>& rgb_out, int& w, int& h) = 0;
};

// 重み 3 ファイルを受けて DiffusionRunner を構築する。
//
// nullptr 契約:
//   いずれかの重み (unet_weights / vae_weights) が不在・ロード不能の場合は
//   nullptr を返す (例外は投げない)。呼び出し側はこれを検知してスタブ生成等へ
//   フォールバックできる (main.cpp の DI と同方針)。
//
// embeds_path について:
//   DiffusionPipeline のコンストラクタが golden 埋め込み (unet_io.safetensors) を
//   要求するため引数として受けて DiffusionPipeline へ渡す。ただし txt2img
//   (run_txt2img → generate_txt2img) では golden 埋め込みは一切使わず、外部から
//   渡される cond/uncond 埋め込みのみを使う。したがって embeds_path はコンストラクタ
//   要件を満たすためだけに必要であり、生成結果には影響しない。
//
// fast_cfg について (G-0b):
//   FAST モードフラグを DiffusionPipeline のメンバとして運ぶだけ。既定 (全 off) は現行挙動。
//   この Pkg では fast 分岐を一切足さないため、既存呼び出し (3 引数) は既定で無改変。
std::unique_ptr<IDiffusionRunner> make_diffusion_runner(
    const std::string& unet_weights, const std::string& vae_weights,
    const std::string& embeds_path, const FastConfig& fast_cfg = FastConfig{});

} // namespace dollama

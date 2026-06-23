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

namespace dollama
{

// 拡散ループの実行 interface。実体は .cu 側で DiffusionPipeline をラップして実装する。
struct IDiffusionRunner
{
    virtual ~IDiffusionRunner() = default;

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
std::unique_ptr<IDiffusionRunner> make_diffusion_runner(
    const std::string& unet_weights, const std::string& vae_weights,
    const std::string& embeds_path);

} // namespace dollama

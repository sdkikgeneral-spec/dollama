// Txt2ImgGenerator — 本物の text→image を IImageGenerator として HTTP サーバ層へ繋ぐ。
//   (Phase 2-6b Stage F / ウェーブ2)
//
// 役割:
//   prompt / negative_prompt を SDXL dual encoder (TextConditioner) で
//   cond/uncond 埋め込みに変換し、CFG 付き拡散ループ (IDiffusionRunner) へ渡して
//   1024x1024 PNG を生成する。pipeline_generator.hpp が golden 埋め込み使い回し
//   (CFG なし) だったのに対し、本クラスは実プロンプトを反映する本 txt2img 経路。
//
// ----------------------------------------------------------------
// 依存隔離 (重要):
//   - TextConditioner (text_conditioner.hpp) は OpenVINO 依存のため、本ヘッダ全体を
//     #ifdef HAVE_OPENVINO でガードする。HAVE_OPENVINO が無い TU から include しても
//     空になりコンパイルが壊れない。
//   - 拡散ループ (CUDA / __half) は純 cpp interface IDiffusionRunner 越しに呼ぶため、
//     本ヘッダに CUDA 型は一切露出しない。よって CUDA ヘッダ非依存の OV 翻訳単位
//     (例: server/api.cpp を OV ビルドで) からも include できる。
//   - runner の実体 (CUDA) は make_diffusion_runner (.cu / _stub.cpp) に隔離される。
//   - M-6: マッティング (α 抽出) は純 cpp interface IMatter 越し (set_matter で注入)。
//     最終 PNG エンコードを encode_png_maybe_transparent に委譲し、matter が有れば
//     透過 PNG、無ければ不透明 PNG を返す。
//
//   従って本ヘッダを include してよいのは HAVE_OPENVINO 定義済みの cpp TU
//   (= MSVC でコンパイルされる .cpp、CUDA 非露出)。
// ----------------------------------------------------------------
#pragma once

#ifdef HAVE_OPENVINO

#include <cstdint>
#include <ctime>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "infer/text_conditioner.hpp"
#include "server/diffusion_runner.hpp"
#include "server/generator.hpp"
#include "server/matter_runner.hpp"
#include "server/matting_postprocess.hpp"
#include "server/png.hpp"

namespace dollama
{

// TextConditioner + IDiffusionRunner を結線し、prompt → PNG を生成する。
class Txt2ImgGenerator : public IImageGenerator
{
public:
    // CFG (classifier-free guidance) スケール。SDXL の一般的な既定値。
    static constexpr float kGuidanceScale = 7.5f;

    // 各モデルパス・デバイス・重みパスを受けて TextConditioner と IDiffusionRunner を構築する。
    //   tokenizer_l/g_xml : SDXL トークナイザ (CLIP-L / bigG) の OV xml。
    //   encoder_l/g_xml   : SDXL text encoder (CLIP-L / bigG) の OV xml。
    //   tokenizers_dll    : openvino_tokenizers.dll の絶対パス。
    //   unet/vae_weights  : 拡散ループ用 safetensors 重み。
    //   embeds_path       : DiffusionPipeline 構築要件用の golden 埋め込み
    //                       (txt2img では使われないがコンストラクタ要件のため必要)。
    //   device_l/g        : CLIP-L / bigG エンコーダの実行デバイス (既定 NPU)。
    //
    //   runner (= make_diffusion_runner) が nullptr の場合 (重み不在 / CUDA 無効) は
    //   構築失敗として std::runtime_error を投げる。呼び出し側 (main の DI) はこれを
    //   捕捉して PipelineGenerator / StubGenerator へフォールバックできる。
    Txt2ImgGenerator(const std::string& tokenizer_l_xml,
                     const std::string& tokenizer_g_xml,
                     const std::string& encoder_l_xml,
                     const std::string& encoder_g_xml,
                     const std::string& tokenizers_dll,
                     const std::string& unet_weights,
                     const std::string& vae_weights,
                     const std::string& embeds_path,
                     const std::string& device_l = "NPU",
                     const std::string& device_g = "NPU")
        : tc_(tokenizer_l_xml, tokenizer_g_xml, encoder_l_xml, encoder_g_xml,
              tokenizers_dll, device_l, device_g),
          runner_(make_diffusion_runner(unet_weights, vae_weights, embeds_path))
    {
        if (!runner_)
        {
            // 重み不在 / ロード失敗 / CUDA 無効ビルド。呼び出し側がフォールバック判断できる。
            throw std::runtime_error(
                "Txt2ImgGenerator: DiffusionRunner の構築に失敗 "
                "(重み不在 / CUDA 無効ビルド)");
        }
    }

    Txt2ImgGenerator(const Txt2ImgGenerator&)            = delete;
    Txt2ImgGenerator& operator=(const Txt2ImgGenerator&) = delete;

    // M-6: マッティング器を後付け注入する (build_image_generator が DI 後に 1 回呼ぶ)。
    void set_matter(std::unique_ptr<IMatter> m) override
    {
        matter_ = std::move(m);
    }

    // 1 リクエスト分を本 txt2img で生成し PNG バイト列を返す。
    GenResult generate(const GenRequest& req) override
    {
        // --- 解像度: 当面 1024x1024 固定。1024 以外が来たら reject する ---
        //   (pipeline_generator.hpp と同方針。clamp で黙って差し替えず明示的に弾く。)
        if ((req.width != 0 && req.width != 1024) ||
            (req.height != 0 && req.height != 1024))
        {
            throw std::runtime_error(
                "Txt2ImgGenerator: width/height は 1024 のみ対応 (SDXL 1024x1024 固定)");
        }

        // --- ステップ数: req.steps をそのまま (1 未満は 20) ---
        const int steps = (req.steps > 0) ? req.steps : 20;

        // --- seed: GenRequest に seed フィールドが無いため内部で決める ---
        //   pipeline_generator.hpp と同方針 (本番なので時刻ベースで毎回変える)。
        const uint64_t seed = static_cast<uint64_t>(std::time(nullptr));

        // --- テキスト条件付け (prompt / negative_prompt → cond/uncond 埋め込み) ---
        const TextConditioning tc = tc_.encode(req.prompt, req.negative_prompt);

        // --- 拡散ループ (CFG) 実行 ---
        std::vector<uint8_t> rgb;
        int w = 0, h = 0;
        runner_->run_txt2img(
            steps, seed, kGuidanceScale,
            tc.cond.encoder_hidden_states.data(),
            tc.cond.text_embeds.data(),
            tc.uncond.encoder_hidden_states.data(),
            tc.uncond.text_embeds.data(),
            tc.time_ids.data(),
            rgb, w, h);

        // --- HWC uint8 RGB → PNG (M-6: matting ON なら透過 PNG・OFF/無なら不透明) ---
        GenResult out;
        out.png_bytes = encode_png_maybe_transparent(
            req.matting ? matter_.get() : nullptr, rgb, w, h);
        return out;
    }

    // モデル識別子 (GET /v1/models 用)。
    std::string model_id() const override
    {
        return "sdxl-1.0";
    }

private:
    TextConditioner                  tc_;
    std::unique_ptr<IDiffusionRunner> runner_;
    std::unique_ptr<IMatter>          matter_; // M-6: set_matter で注入 (既定 nullptr = 不透明)
};

} // namespace dollama

#endif // HAVE_OPENVINO

// PipelineGenerator — DiffusionPipeline (CUDA 拡散ループ) を IImageGenerator として
// HTTP サーバ層へ繋ぐ薄いアダプタ (Phase 2 マイルストーン 2-6a / ST-3+ST-4)。
//
// HTTP サーバ層 (api.cpp) は IImageGenerator 越しに呼ぶため、main.cpp の DI を
// StubGenerator → PipelineGenerator に差し替える (後続 ST-5) だけで本 txt2img へ移行できる。
//
// ----------------------------------------------------------------
// CUDA 隔離設計 (重要):
//   DiffusionPipeline (diffusion.cuh) は __half / CUDA 型を含むため、CUDA 非対応の
//   翻訳単位 (cpp TU、例: api.cpp / test_http.cpp) からこのヘッダを include すると
//   ビルドが壊れる。そこで本ヘッダ全体を #ifdef HAVE_CUDA でガードし、HAVE_CUDA が
//   定義された TU (= nvcc でコンパイルされる .cu) からのみ実体が見えるようにする。
//
//   従ってこのヘッダを include してよいのは .cu (nvcc 経由) のみ。
//   - 実体保持: PipelineGenerator は DiffusionPipeline を値で持つ (コピー禁止クラスだが
//     PipelineGenerator もコピー禁止にして移動不可で運用。HTTP からは生存期間中 1 個保持)。
//   - 別 .cu に実体を分離せずヘッダオンリーにした理由: DiffusionPipeline の宣言は
//     diffusion.cuh にあり、その実体は diffusion.cu (テスト exe にリンク) にある。
//     PipelineGenerator 自身はメンバ関数が薄い (generate のグルーのみ) ため、ヘッダ
//     インラインで十分。include は .cu 限定なので ODR / CUDA 漏れの心配はない。
//   - M-6: マッティング (α 抽出) は純 cpp interface IMatter 越し (set_matter で注入)。
//     matting_postprocess.hpp / matter_runner.hpp は純 cpp (CUDA/OV 非依存) なので
//     CUDA TU から安全に include でき、factory シグネチャは不変。
// ----------------------------------------------------------------
#pragma once

#ifdef HAVE_CUDA

#include <cstdint>
#include <ctime>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "infer/diffusion.cuh"
#include "server/generator.hpp"
#include "server/matter_runner.hpp"
#include "server/matting_postprocess.hpp"
#include "server/png.hpp"

namespace dollama
{

// DiffusionPipeline を IImageGenerator 実装としてラップする。
class PipelineGenerator : public IImageGenerator
{
public:
    // unet/vae 重みと golden 埋め込みのパスを受けて DiffusionPipeline を構築・保持する。
    //   重み (5GB) は DiffusionPipeline 内で 1 回だけロードされる (再ロード厳禁)。
    //   seed を固定したいテスト等のため、決定論シードを任意で渡せる (省略時は時刻ベース)。
    PipelineGenerator(const std::string& unet_weights_path,
                      const std::string& vae_weights_path,
                      const std::string& embeds_path,
                      bool               deterministic_seed = false,
                      uint64_t           fixed_seed         = 1234ULL)
        : pipe_(unet_weights_path, vae_weights_path, embeds_path),
          deterministic_seed_(deterministic_seed),
          fixed_seed_(fixed_seed)
    {
    }

    PipelineGenerator(const PipelineGenerator&)            = delete;
    PipelineGenerator& operator=(const PipelineGenerator&) = delete;

    // M-6: マッティング器を後付け注入する (build_image_generator が DI 後に 1 回呼ぶ)。
    void set_matter(std::unique_ptr<IMatter> m) override
    {
        matter_ = std::move(m);
    }

    // 1 リクエスト分を拡散ループで生成し PNG バイト列を返す。
    GenResult generate(const GenRequest& req) override
    {
        // --- 解像度: 当面 1024×1024 固定。1024 以外が来たら reject する ---
        // 理由: DiffusionPipeline は latent[1,4,128,128]→image[1024×1024] 固定で結線
        //   されており、任意解像度に未対応。clamp で黙って差し替えると利用者が指定した
        //   サイズと出力が食い違い気づきにくいため、明示的に runtime_error で弾く
        //   (サーバ層が 400/500 に変換する)。
        if ((req.width != 0 && req.width != 1024) ||
            (req.height != 0 && req.height != 1024))
        {
            throw std::runtime_error(
                "PipelineGenerator: width/height は 1024 のみ対応 (SDXL 1024x1024 固定)");
        }

        // --- ステップ数: req.steps をそのまま渡す (1 未満は 1 にクランプ) ---
        const int steps = (req.steps > 0) ? req.steps : 20;

        // --- seed: GenRequest に seed フィールドが無いため内部で決める ---
        //   deterministic_seed_ が真なら fixed_seed_ (テストの再現用)、
        //   そうでなければ時刻ベースで毎回変える (本番の多様性確保)。
        const uint64_t seed = deterministic_seed_
            ? fixed_seed_
            : static_cast<uint64_t>(std::time(nullptr));

        // TODO(2-6b): req.prompt / req.negative_prompt はこの段では未使用。
        //   テキストエンコードが未結線で、DiffusionPipeline は golden 埋め込みを
        //   全 step 使い回すため。CLIP-G dual encoder 結線時に prompt→embeds を
        //   生成して DiffusionPipeline へ差し込み、ここで req.prompt を反映する。
        (void)req.prompt;
        (void)req.negative_prompt;

        // 拡散ループ実行 (HWC uint8 RGB を受け取る)。
        std::vector<uint8_t> rgb;
        int w = 0, h = 0;
        pipe_.generate(steps, seed, rgb, w, h);

        // HWC uint8 RGB → PNG (M-6: matting ON なら透過 PNG・OFF/無なら不透明)。
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
    DiffusionPipeline pipe_;
    bool              deterministic_seed_;
    uint64_t          fixed_seed_;
    std::unique_ptr<IMatter> matter_; // M-6: set_matter で注入 (既定 nullptr = 不透明)
};

} // namespace dollama

#endif // HAVE_CUDA

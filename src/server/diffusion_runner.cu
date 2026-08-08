// DiffusionRunner — DiffusionPipeline (CUDA 拡散ループ) を純 cpp interface
//   IDiffusionRunner として外部 (cpp TU) へ繋ぐアダプタ (Phase 2-6b Stage F / ウェーブ1)。
//
// ----------------------------------------------------------------
// CUDA 隔離設計 (重要):
//   DiffusionPipeline (diffusion.cuh) は __half / CUDA 型を含むため、CUDA 非対応の
//   翻訳単位 (cpp TU、例: api.cpp / test_*.cpp) から diffusion.cuh を include すると
//   ビルドが壊れる。そこで diffusion.cuh の include は本 .cu (nvcc 経由) からのみ行い、
//   外部には純 cpp 宣言 diffusion_runner.hpp (IDiffusionRunner) だけを見せる。
//   これにより cpp 側へ CUDA 型が一切漏れない (pipeline_generator_factory.cu と同方針)。
//
//   重み不在時は make_diffusion_runner が nullptr を返し、呼び出し側がフォールバック
//   できる (pipeline_generator_factory.cu の file_readable / nullptr 返しに厳密に倣う)。
// ----------------------------------------------------------------
#include "server/diffusion_runner.hpp"

#include <fstream>
#include <iostream>

#ifdef HAVE_CUDA
#include "infer/diffusion.cuh"  // CUDA 型 (DiffusionPipeline)。本 .cu からのみ include する
#endif

namespace dollama
{

#ifdef HAVE_CUDA

// DiffusionPipeline を IDiffusionRunner 実装としてラップする。
//   pipeline_generator.hpp と同方針で DiffusionPipeline を値で所有 (コピー禁止クラス・
//   本クラスもコピー禁止 + 移動不可運用。生存期間中 1 個保持)。
class DiffusionRunner : public IDiffusionRunner
{
public:
    // unet/vae 重みと golden 埋め込みのパスを受けて DiffusionPipeline を構築・保持する。
    //   重み (5GB) は DiffusionPipeline 内で 1 回だけロードされる (再ロード厳禁)。
    //   fast_cfg (G-0b): FAST フラグを DiffusionPipeline へ運ぶだけ (既定 = 現行挙動)。
    DiffusionRunner(const std::string& unet_weights_path,
                    const std::string& vae_weights_path,
                    const std::string& embeds_path,
                    const FastConfig&  fast_cfg)
        : pipe_(unet_weights_path, vae_weights_path, embeds_path, fast_cfg)
    {
    }

    DiffusionRunner(const DiffusionRunner&)            = delete;
    DiffusionRunner& operator=(const DiffusionRunner&) = delete;

    // 受けた引数を DiffusionPipeline::generate_txt2img へ 1:1 で委譲する。
    //   引数順は diffusion.cuh の generate_txt2img と厳密に一致させる
    //   (steps, seed, guidance_scale, cond_ehs, cond_text_embeds,
    //    uncond_ehs, uncond_text_embeds, time_ids, rgb_out, w, h)。
    void run_txt2img(int                   steps,
                     uint64_t              seed,
                     float                 guidance_scale,
                     const float*          cond_ehs,
                     const float*          cond_text_embeds,
                     const float*          uncond_ehs,
                     const float*          uncond_text_embeds,
                     const float*          time_ids,
                     std::vector<uint8_t>& rgb_out,
                     int&                  w,
                     int&                  h) override
    {
        pipe_.generate_txt2img(steps,
                               seed,
                               guidance_scale,
                               cond_ehs,
                               cond_text_embeds,
                               uncond_ehs,
                               uncond_text_embeds,
                               time_ids,
                               rgb_out,
                               w,
                               h);
    }

    // ランタイム LoRA (L-2): ファイルリストを DiffusionPipeline へ順に適用する。
    //   適用途中で失敗した場合は部分適用が残らないよう復元してから再 throw する
    //   (呼び出し側 = HTTP 層はエラー変換だけに専念できる)。
    void apply_loras(const std::vector<LoraFileRequest>& reqs) override
    {
        try
        {
            for (const LoraFileRequest& r : reqs)
            {
                pipe_.apply_lora_file(r.path, r.strength);
            }
        }
        catch (...)
        {
            pipe_.clear_loras();
            throw;
        }
    }

    // patch 済み常駐重みを base へ bit-exact 復元する。
    void clear_loras() override
    {
        pipe_.clear_loras();
    }

private:
    DiffusionPipeline pipe_;
};

#endif // HAVE_CUDA

namespace
{

// ファイルが読み出し可能か (存在し開けるか) を判定する。
//   (pipeline_generator_factory.cu の file_readable と同一)
bool file_readable(const std::string& path)
{
    if (path.empty())
    {
        return false;
    }
    std::ifstream f(path, std::ios::binary);
    return f.good();
}

} // namespace

std::unique_ptr<IDiffusionRunner> make_diffusion_runner(
    const std::string& unet_weights,
    const std::string& vae_weights,
    const std::string& embeds_path,
    const FastConfig&  fast_cfg)
{
#ifdef HAVE_CUDA
    // 重み/golden の存在チェック。1 つでも欠ければ nullptr (→ 呼び出し側フォールバック)。
    if (!file_readable(unet_weights) ||
        !file_readable(vae_weights) ||
        !file_readable(embeds_path))
    {
        std::cerr << "[diffusion_runner] 重み/golden が不足しています — フォールバック\n";
        std::cerr << "  unet  = " << unet_weights
                  << (file_readable(unet_weights) ? " [OK]" : " [MISSING]") << "\n";
        std::cerr << "  vae   = " << vae_weights
                  << (file_readable(vae_weights) ? " [OK]" : " [MISSING]") << "\n";
        std::cerr << "  embed = " << embeds_path
                  << (file_readable(embeds_path) ? " [OK]" : " [MISSING]") << "\n";
        return nullptr;
    }

    // 重み (5GB) のロードは DiffusionRunner の構築時に 1 回だけ走る。
    try
    {
        return std::make_unique<DiffusionRunner>(
            unet_weights, vae_weights, embeds_path, fast_cfg);
    }
    catch (const std::exception& e)
    {
        // 重みロード失敗等。呼び出し側がフォールバックできるよう nullptr を返す。
        std::cerr << "[diffusion_runner] DiffusionRunner 構築に失敗 — フォールバック: "
                  << e.what() << "\n";
        return nullptr;
    }
#else
    // CUDA 無効ビルド: DiffusionRunner は存在しない。常に nullptr。
    // (実際にはこの .cu は CUDA 有効時のみリンクされるが、念のためガードを置く。)
    (void)unet_weights;
    (void)vae_weights;
    (void)embeds_path;
    (void)fast_cfg;
    return nullptr;
#endif
}

} // namespace dollama

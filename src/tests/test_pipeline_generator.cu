// PipelineGenerator (DiffusionPipeline → IImageGenerator アダプタ) テスト
// (Phase 2 マイルストーン 2-6a / ST-3+ST-4)
//
// CUDA 隔離: PipelineGenerator (= diffusion.cuh / __half 依存) を include するため
//   本テストは .cu (nvcc 経由) としてビルドする。cpp TU には CUDA を漏らさない。
//
// HAVE_CUDA 未定義時は [SKIP] return 0。golden/重み不在時も [SKIP] return 0。
//
// 検証 (重み/golden が揃うとき, steps=2, ~20-30s 想定):
//   (a) png_bytes 先頭が PNG シグネチャ \x89PNG\r\n\x1a\n
//   (b) read_png_size で 1024×1024
//   (c) png_bytes が空でない
//   (d) model_id() == "sdxl-1.0"
//
// 重みパス UNET_WEIGHTS_PATH / VAE_WEIGHTS_PATH / UNET_IO_PATH は meson cuda_args で
// -D 埋め込み (cwd 非依存)。

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#ifdef HAVE_CUDA
#include "server/generator.hpp"
#include "server/pipeline_generator.hpp"
#include "server/png.hpp"
#endif

namespace dollama
{

#ifdef HAVE_CUDA

// PNG シグネチャ検査。
static bool has_png_signature(const std::vector<uint8_t>& png)
{
    static const uint8_t sig[8] = {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A};
    if (png.size() < 8)
    {
        return false;
    }
    for (int i = 0; i < 8; ++i)
    {
        if (png[i] != sig[i])
        {
            return false;
        }
    }
    return true;
}

static int run_test()
{
    const std::string unet_w = UNET_WEIGHTS_PATH;
    const std::string vae_w  = VAE_WEIGHTS_PATH;
    const std::string embeds = UNET_IO_PATH;

    std::cout << "[test_pipeline_generator] unet_weights=" << unet_w << "\n";
    std::cout << "[test_pipeline_generator] vae_weights="  << vae_w  << "\n";
    std::cout << "[test_pipeline_generator] embeds="       << embeds << "\n";

    // 重み/golden 不在時は [SKIP] (リポジトリには重みを含めない。dump スクリプトで再生成)。
    {
        std::ifstream fu(unet_w, std::ios::binary);
        std::ifstream fv(vae_w, std::ios::binary);
        std::ifstream fe(embeds, std::ios::binary);
        if (!fu.good() || !fv.good() || !fe.good())
        {
            std::cout << "[test_pipeline_generator] [SKIP] golden/weights not found\n";
            return 0;
        }
    }

    bool ok = true;

    // 決定論シードでパイプライン生成器を構築 (再現性のためテストは固定 seed)。
    PipelineGenerator gen(unet_w, vae_w, embeds, /*deterministic_seed=*/true,
                          /*fixed_seed=*/1234ULL);

    // (d) model_id 検査 (重い生成の前に確認)。
    if (gen.model_id() != "sdxl-1.0")
    {
        std::cerr << "[test_pipeline_generator] FAIL: model_id()=" << gen.model_id()
                  << " != sdxl-1.0\n";
        ok = false;
    }
    else
    {
        std::cout << "[test_pipeline_generator] model_id() == sdxl-1.0 OK\n";
    }

    // steps=2 で生成 (~20-30s)。
    std::cout << "[test_pipeline_generator] --- generate (steps=2) ---\n";
    GenRequest req;
    req.prompt          = "1girl, solo";  // この段では未使用 (golden 埋め込み使い回し)
    req.negative_prompt = "";
    req.steps           = 2;
    req.width           = 1024;
    req.height          = 1024;

    GenResult res = gen.generate(req);

    // (c) png_bytes が空でない。
    if (res.png_bytes.empty())
    {
        std::cerr << "[test_pipeline_generator] FAIL: png_bytes is empty\n";
        ok = false;
    }
    else
    {
        std::cout << "[test_pipeline_generator] png_bytes size=" << res.png_bytes.size()
                  << " bytes\n";
    }

    // (a) PNG シグネチャ。
    if (!has_png_signature(res.png_bytes))
    {
        std::cerr << "[test_pipeline_generator] FAIL: PNG signature mismatch\n";
        ok = false;
    }
    else
    {
        std::cout << "[test_pipeline_generator] PNG signature OK\n";
    }

    // (b) read_png_size で 1024×1024。
    int pw = 0, ph = 0;
    if (!read_png_size(res.png_bytes, pw, ph))
    {
        std::cerr << "[test_pipeline_generator] FAIL: read_png_size failed\n";
        ok = false;
    }
    else if (pw != 1024 || ph != 1024)
    {
        std::cerr << "[test_pipeline_generator] FAIL: PNG size " << pw << "x" << ph
                  << " != 1024x1024\n";
        ok = false;
    }
    else
    {
        std::cout << "[test_pipeline_generator] PNG size 1024x1024 OK\n";
    }

    // 異常解像度 reject の検査 (生成は走らせない・例外確認のみ)。
    {
        GenRequest bad;
        bad.steps  = 2;
        bad.width  = 512;
        bad.height = 512;
        bool threw = false;
        try
        {
            gen.generate(bad);
        }
        catch (const std::runtime_error&)
        {
            threw = true;
        }
        if (!threw)
        {
            std::cerr << "[test_pipeline_generator] FAIL: width/height!=1024 was not rejected\n";
            ok = false;
        }
        else
        {
            std::cout << "[test_pipeline_generator] non-1024 resolution rejected OK\n";
        }
    }

    if (ok)
    {
        std::cout << "[test_pipeline_generator] PASSED\n";
    }
    return ok ? 0 : 1;
}

#endif // HAVE_CUDA

} // namespace dollama

int main()
{
#ifndef HAVE_CUDA
    std::cout << "[test_pipeline_generator] [SKIP] HAVE_CUDA undefined\n";
    return 0;
#else
    try
    {
        const int rc = dollama::run_test();
        if (rc == 0) { std::cout << "[test_pipeline_generator] ALL PASSED\n"; }
        else         { std::cerr << "[test_pipeline_generator] FAILED\n"; }
        return rc;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[test_pipeline_generator] exception: " << e.what() << "\n";
        return 1;
    }
#endif
}

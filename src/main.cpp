#include <iostream>
#include <string>

#ifdef HAVE_OPENVINO
#include <filesystem>
#include <openvino/openvino.hpp>
#include "pipeline.hpp"
#endif

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef HAVE_HTTP
#include <cstdlib>
#include <memory>
#include "server/api.hpp"
#include "server/generator.hpp"
#include "server/stub_generator.hpp"
// 純 cpp 宣言のみ (CUDA 非依存)。実体は CUDA 有効時 pipeline_generator_factory.cu、
// 無効時 pipeline_generator_factory_stub.cpp が提供する。main.cpp に CUDA は漏れない。
#include "server/pipeline_generator_factory.hpp"
#endif

#ifdef HAVE_OPENVINO
namespace
{
// モデル .xml の候補パスを探索する (実行ディレクトリ差を吸収)
std::string find_model_xml(const std::string& rel)
{
    namespace fs = std::filesystem;
    const std::string candidates[] = {
        "../models/" + rel,
        "models/" + rel,
        "../../models/" + rel,
    };
    for (const auto& p : candidates)
    {
        if (fs::exists(p))
        {
            return p;
        }
    }
    return ""; // 見つからなければ空文字
}
} // namespace
#endif

namespace
{

// デバイスチェック (従来のデフォルト挙動) を実行する
int run_device_check()
{
    std::cout << "dollama v0.1.0 — device check\n";
    std::cout << "================================\n";

    // ---- OpenVINO (NPU / iGPU) ----
#ifdef HAVE_OPENVINO
    ov::Core core;
    auto devices = core.get_available_devices();
    std::cout << "\n[OpenVINO] available devices:\n";
    for (const auto& d : devices)
    {
        auto name = core.get_property(d, ov::device::full_name);
        std::cout << "  " << d << " — " << name << "\n";
    }
    // 期待値: CPU, GPU.0 (Intel Xe iGPU), GPU.1 (RTX5080), NPU
#else
    std::cout << "\n[OpenVINO] not compiled\n";
#endif

    // ---- CUDA (RTX5080) ----
#ifdef HAVE_CUDA
    int n = 0;
    // [Bug-4] cudaGetDeviceCount の戻り値をチェックする
    if (cudaGetDeviceCount(&n) != cudaSuccess)
    {
        std::cerr << "[CUDA] cudaGetDeviceCount failed — CUDA 初期化エラー\n";
        return 1;
    }
    std::cout << "\n[CUDA] devices: " << n << "\n";
    for (int i = 0; i < n; ++i)
    {
        cudaDeviceProp p;
        // [Bug-3] cudaGetDeviceProperties の戻り値をチェックする
        if (cudaGetDeviceProperties(&p, i) != cudaSuccess)
        {
            std::cerr << "[CUDA] cudaGetDeviceProperties failed for device " << i << "\n";
            continue;
        }
        std::cout << "  [" << i << "] " << p.name
                  << "  VRAM " << p.totalGlobalMem / 1024 / 1024 / 1024 << " GB"
                  << "  sm_" << p.major << p.minor << "\n";
    }
    // 期待値: RTX5080 / sm_120 / ~16 GB
#else
    std::cout << "\n[CUDA] not compiled\n";
#endif

    // ---- Phase 1 パイプライン (stub→CLIP(NPU)→queue→WD14(CPU)) ----
#ifdef HAVE_OPENVINO
    std::cout << "\n[pipeline] Phase 1 パイプライン縦通し\n";
    const std::string clip_xml = find_model_xml("clip-l-text-encoder/model_ov.xml");
    const std::string wd14_xml = find_model_xml("wd14-swinv2-tagger-v3/model_ov.xml");

    if (clip_xml.empty() || wd14_xml.empty())
    {
        std::cout << "  [SKIP] CLIP/WD14 モデルが見つかりません "
                     "(clip='" << clip_xml << "', wd14='" << wd14_xml << "')\n";
    }
    else
    {
        try
        {
            dollama::Pipeline pipe(clip_xml, wd14_xml);
            const int frames = 3;
            auto tags = pipe.run(frames);
            std::cout << "  処理フレーム数: " << tags.size() << " / " << frames << "\n";
            for (size_t i = 0; i < tags.size(); ++i)
            {
                std::cout << "  frame " << i << ": " << tags[i] << "\n";
            }
        }
        catch (const std::exception& e)
        {
            std::cerr << "  [pipeline] 例外: " << e.what() << "\n";
        }
    }
#else
    std::cout << "\n[pipeline] OpenVINO 無効のためスキップ\n";
#endif

    std::cout << "\nOK\n";
    return 0;
}

#ifdef HAVE_HTTP
// 環境変数で上書き可能な重みパスを解決する。
//   env が空でなければそれを使い、無ければソースツリーの既定 data パスを使う。
//   本番でどこから重みを読むかは未定のため、当面は test data を既定にしておく
//   (ファイル不在なら factory が nullptr → StubGenerator にフォールバックする)。
std::string resolve_path(const char* env_name, const std::string& fallback)
{
    // MSVC は std::getenv に C4996 (非推奨) を出すが、ここは読み取り専用で
    // スレッド前 (起動時 1 回) のため安全。局所的に警告を抑止する。
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
    if (const char* v = std::getenv(env_name))
    {
        if (v[0] != '\0')
        {
            return std::string(v);
        }
    }
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
    return fallback;
}
#endif

} // namespace

// エントリポイント。
//   引数なし          : 従来のデバイスチェック (既存挙動を維持)。
//   --http [--port N] : OpenAI Images 互換 HTTP サーバーを起動。
//                       重み/golden が揃えば PipelineGenerator、無ければ StubGenerator。
int main(int argc, char** argv)
{
#ifdef HAVE_HTTP
    bool http_mode = false;
    int port = 8080;
    int steps = 20;
    int width = 1024;
    int height = 1024;

    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        auto next_int = [&](int def) -> int
        {
            if (i + 1 < argc)
            {
                try
                {
                    return std::stoi(argv[++i]);
                }
                catch (...)
                {
                    return def;
                }
            }
            return def;
        };
        if (a == "--http")
        {
            http_mode = true;
        }
        else if (a == "--port")
        {
            port = next_int(port);
        }
        else if (a == "--steps")
        {
            steps = next_int(steps);
        }
        else if (a == "--width")
        {
            width = next_int(width);
        }
        else if (a == "--height")
        {
            height = next_int(height);
        }
    }

    if (http_mode)
    {
        // 重み/golden パスを解決 (env 変数で上書き可。既定は test data パス)。
        // 既定は src/tests/data。本番の重み配置先が決まったら DEFAULT を差し替える。
        const std::string unet_w =
            resolve_path("DOLLAMA_UNET_WEIGHTS", "src/tests/data/unet_weights.safetensors");
        const std::string vae_w =
            resolve_path("DOLLAMA_VAE_WEIGHTS", "src/tests/data/vae_weights.safetensors");
        const std::string embeds =
            resolve_path("DOLLAMA_EMBEDS", "src/tests/data/unet_io.safetensors");

        // ファクトリで PipelineGenerator を試みる。重み不在 / CUDA 無効なら nullptr。
        //   本番なので deterministic=false (毎回異なる画像)。
        std::unique_ptr<dollama::IImageGenerator> gen =
            dollama::make_pipeline_generator(unet_w, vae_w, embeds, /*deterministic=*/false);

        if (gen)
        {
            std::cout << "dollama HTTP server (pipeline generator)\n";
            std::cout << "  weights: unet='" << unet_w << "' vae='" << vae_w
                      << "' embeds='" << embeds << "'\n";
        }
        else
        {
            // フォールバック: 重みが無い / CUDA 無効でも HTTP は起動し続ける (回帰防止)。
            gen = std::make_unique<dollama::StubGenerator>();
            std::cout << "dollama HTTP server (stub generator — 重み未解決のためフォールバック)\n";
        }

        std::cout << "  defaults: steps=" << steps
                  << " size=" << width << "x" << height << "\n";
        return dollama::start_server(*gen, "127.0.0.1", port);
    }
#else
    (void)argc;
    (void)argv;
#endif

    return run_device_check();
}

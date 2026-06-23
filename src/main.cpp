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
#include <memory>
#include <vector>
#include "server/api.hpp"
#include "server/generator.hpp"
// 生成器 DI (3 段フォールバック) と PNG 書き出しヘルパ。--http / CLI 生成で共有。
// find_model_xml / resolve_path もこのヘッダに集約済 (dollama 名前空間)。
#include "server/cli_generate.hpp"
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
    const std::string clip_xml = dollama::find_model_xml("clip-l-text-encoder/model_ov.xml");
    const std::string wd14_xml = dollama::find_model_xml("wd14-swinv2-tagger-v3/model_ov.xml");

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

} // namespace

// エントリポイント。
//   引数なし                       : 従来のデバイスチェック (既存挙動を維持)。
//   --http [--port N]              : OpenAI Images 互換 HTTP サーバーを起動。
//   --prompt <str> [--out <path>]  : HTTP を起動せず CLI で 1 枚生成し PNG 保存。
//                                    重み/golden が揃えば本 txt2img、無ければ
//                                    Pipeline→Stub フォールバックで必ず PNG が出る。
int main(int argc, char** argv)
{
#ifdef HAVE_HTTP
    bool http_mode = false;
    int port = 8080;
    int steps = 20;
    int width = 1024;
    int height = 1024;
    std::string prompt;            // 指定かつ --http 無し → CLI 生成モード
    std::string negative;          // 既定 ""
    std::string out_path = "out.png";

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
        auto next_str = [&](const std::string& def) -> std::string
        {
            if (i + 1 < argc)
            {
                return std::string(argv[++i]);
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
        else if (a == "--prompt")
        {
            prompt = next_str(prompt);
        }
        else if (a == "--negative")
        {
            negative = next_str(negative);
        }
        else if (a == "--out")
        {
            out_path = next_str(out_path);
        }
    }

    // モード分岐: --http 最優先 → 次に --prompt あれば CLI 生成 → どちらも無ければ device check。
    if (http_mode)
    {
        // 生成器を 3 段フォールバックで構築 (--http / CLI 共有のヘルパ)。
        std::unique_ptr<dollama::IImageGenerator> gen =
            dollama::build_image_generator(std::cout);

        std::cout << "  defaults: steps=" << steps
                  << " size=" << width << "x" << height << "\n";
        return dollama::start_server(*gen, "127.0.0.1", port);
    }

    if (!prompt.empty())
    {
        // CLI 生成モード: HTTP を起動せず 1 枚生成して PNG 保存。
        std::cout << "dollama — CLI 生成モード\n";
        std::cout << "  prompt='" << prompt << "' negative='" << negative << "'\n";

        std::unique_ptr<dollama::IImageGenerator> gen =
            dollama::build_image_generator(std::cout);

        dollama::GenRequest req{prompt, negative, 1, steps, width, height};
        try
        {
            dollama::GenResult r = gen->generate(req);
            if (!dollama::write_png_file(out_path, r.png_bytes))
            {
                std::cerr << "[generate] PNG 書き出しに失敗: " << out_path << "\n";
                return 1;
            }
            std::cout << "[generate] wrote " << out_path << " (" << r.png_bytes.size()
                      << " bytes, " << width << "x" << height << ", steps=" << steps << ")\n";
            return 0;
        }
        catch (const std::exception& e)
        {
            std::cerr << "[generate] 失敗: " << e.what() << "\n";
            return 1;
        }
    }
#else
    (void)argc;
    (void)argv;
#endif

    return run_device_check();
}

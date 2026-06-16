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

// デバイス情報を表示して環境チェック
int main()
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

#include <iostream>
#include <string>

#ifdef HAVE_OPENVINO
#include <openvino/openvino.hpp>
#endif

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
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

    std::cout << "\nOK\n";
    return 0;
}

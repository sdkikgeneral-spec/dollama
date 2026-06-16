// CUDA 疎通テスト + vector_add ベンチ (Phase 2 マイルストーン 2-0 / 2-1)
// HAVE_CUDA 未定義時は [SKIP] で return 0。
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

#ifdef HAVE_CUDA
#include "kernels/utils.cuh"
#endif

namespace dollama
{

#ifdef HAVE_CUDA

// 基本疎通: ホスト配列 2 本 → cudaMalloc → H2D → launch_vector_add → D2H → 期待値一致
static bool test_vector_add()
{
    const int n = 1 << 20; // 約 100 万要素
    std::vector<float> h_a(n), h_b(n), h_out(n);
    for (int i = 0; i < n; ++i)
    {
        h_a[i] = static_cast<float>(i) * 0.5f;
        h_b[i] = static_cast<float>(i) * 0.25f + 1.0f;
    }

    float* d_a   = nullptr;
    float* d_b   = nullptr;
    float* d_out = nullptr;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    launch_vector_add(d_a, d_b, d_out, n);

    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));

    // 数値検証
    for (int i = 0; i < n; ++i)
    {
        const float expected = h_a[i] + h_b[i];
        if (std::fabs(h_out[i] - expected) > 1e-3f)
        {
            std::cerr << "[test_vector_add] mismatch at " << i
                      << ": got " << h_out[i] << " expected " << expected << "\n";
            return false;
        }
    }

    std::cout << "[test_vector_add] PASSED (n=" << n << ")\n";
    return true;
}

// ceil_div ヘルパーの境界検証
static bool test_ceil_div()
{
    if (ceil_div(0, 256) != 0) { std::cerr << "[test_ceil_div] 0/256 failed\n"; return false; }
    if (ceil_div(1, 256) != 1) { std::cerr << "[test_ceil_div] 1/256 failed\n"; return false; }
    if (ceil_div(256, 256) != 1) { std::cerr << "[test_ceil_div] 256/256 failed\n"; return false; }
    if (ceil_div(257, 256) != 2) { std::cerr << "[test_ceil_div] 257/256 failed\n"; return false; }
    std::cout << "[test_ceil_div] PASSED\n";
    return true;
}

// ベンチ: N=1<<24 の H2D + kernel + D2H を 100 回、中央値 ms / 実効 GB/s を出力 (assert 外)
static void bench_vector_add()
{
    const int n = 1 << 24; // 約 1670 万要素
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);

    std::vector<float> h_a(n, 1.0f), h_b(n, 2.0f), h_out(n);

    float* d_a   = nullptr;
    float* d_b   = nullptr;
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    const int warmup = 5;
    const int iters  = 100;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    auto run_once = [&]() -> float
    {
        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));
        launch_vector_add(d_a, d_b, d_out, n);
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    };

    for (int i = 0; i < warmup; ++i)
    {
        (void)run_once();
    }

    std::vector<float> times;
    times.reserve(iters);
    for (int i = 0; i < iters; ++i)
    {
        times.push_back(run_once());
    }

    std::sort(times.begin(), times.end());
    const float median_ms = times[times.size() / 2];

    // 実効帯域: H2D 2本 + D2H 1本 = 3*bytes が PCIe を往復する総転送量
    const double total_bytes = 3.0 * static_cast<double>(bytes);
    const double gbps = total_bytes / (static_cast<double>(median_ms) * 1e-3) / 1e9;

    std::cout << "[bench_vector_add] N=" << n
              << " median=" << median_ms << " ms"
              << " (min=" << times.front() << " max=" << times.back() << ")"
              << " effective=" << gbps << " GB/s (H2D x2 + D2H, N=" << iters << ")\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));
}

#endif // HAVE_CUDA

} // namespace dollama

int main()
{
#ifndef HAVE_CUDA
    std::cout << "[test_cuda_smoke] [SKIP] HAVE_CUDA 未定義\n";
    return 0;
#else
    bool ok = true;
    ok = dollama::test_ceil_div() && ok;
    ok = dollama::test_vector_add() && ok;

    if (!ok)
    {
        std::cerr << "[test_cuda_smoke] FAILED\n";
        return 1;
    }

    dollama::bench_vector_add();

    std::cout << "[test_cuda_smoke] ALL PASSED\n";
    return 0;
#endif
}

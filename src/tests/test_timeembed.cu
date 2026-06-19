// sinusoidal time embedding 単体テスト + ベンチ (Phase 2 マイルストーン 2-5)
// HAVE_CUDA 未定義時は [SKIP] で return 0。
//
// CPU 参照は diffusers get_timestep_embedding と同一式 (flip_sin_to_cos=True,
// downscale_freq_shift=0)。順序確認: out[0..half-1]=cos(args)、out[half..2half-1]=sin(args)。
// 出力 FP16 だが式は単純なので固定 tol。
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#ifdef HAVE_CUDA
#include <cuda_fp16.h>
#include "kernels/timeembed.cuh"
#include "kernels/utils.cuh"
#endif

namespace dollama
{

#ifdef HAVE_CUDA

static constexpr float ATOL = 2e-3f;
static constexpr float RTOL = 2e-3f;

// CPU 参照 (diffusers get_timestep_embedding 厳密版)。
//   half     = dim/2
//   freqs[i] = exp(-log(max_period) * i / half)
//   args     = timestep * freqs[i]
//   out[i]        = cos(args)   (cos が先: flip_sin_to_cos=True)
//   out[half + i] = sin(args)
static std::vector<float> cpu_time_embedding(float timestep, int dim, float max_period)
{
    const int half = dim / 2;
    const float log_mp = std::log(max_period);
    std::vector<float> out(dim, 0.0f);
    for (int i = 0; i < half; ++i)
    {
        const float exponent = -log_mp * static_cast<float>(i) / static_cast<float>(half);
        const float freq = std::exp(exponent);
        const float arg  = timestep * freq;
        out[i]        = std::cos(arg);
        out[half + i] = std::sin(arg);
    }
    return out;
}

static std::vector<float> run_gpu_te(float timestep, int dim, float max_period)
{
    __half* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, static_cast<size_t>(dim) * sizeof(__half)));
    // diffusers の偶数 dim を想定。万一末尾未書き込みでも検出できるよう 0 初期化。
    CUDA_CHECK(cudaMemset(d_out, 0, static_cast<size_t>(dim) * sizeof(__half)));

    launch_timestep_embedding(d_out, timestep, dim, max_period);

    std::vector<__half> h_out(dim);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, static_cast<size_t>(dim) * sizeof(__half),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_out));

    std::vector<float> out(dim);
    for (int i = 0; i < dim; ++i)
    {
        out[i] = __half2float(h_out[i]);
    }
    return out;
}

static bool compare(const std::vector<float>& got,
                    const std::vector<float>& ref,
                    const char* name)
{
    float max_rel = 0.0f;
    for (size_t i = 0; i < got.size(); ++i)
    {
        const float diff = std::fabs(got[i] - ref[i]);
        const float lim  = ATOL + RTOL * std::fabs(ref[i]);
        if (diff > lim)
        {
            std::cerr << "[" << name << "] mismatch at " << i
                      << ": got " << got[i] << " ref " << ref[i]
                      << " diff " << diff << " lim " << lim << "\n";
            return false;
        }
        if (std::fabs(ref[i]) > 1e-3f)
        {
            max_rel = std::max(max_rel, diff / std::fabs(ref[i]));
        }
    }
    std::cout << "[" << name << "] PASSED (max_rel=" << max_rel << ")\n";
    return true;
}

// ----------------------------------------------------------------
// 1. test_timeembed_order: 順序確認 (cos が先頭 half)。
//   i=0 では freq=1, args=timestep。out[0]=cos(t), out[half]=sin(t)。
// ----------------------------------------------------------------
static bool test_timeembed_order()
{
    const int dim = 320;
    const float t = 2.5f;
    std::vector<float> got = run_gpu_te(t, dim, 10000.0f);

    const int half = dim / 2;
    // i=0: freq=exp(0)=1 → arg = t。
    if (std::fabs(got[0] - std::cos(t)) > 3e-3f)
    {
        std::cerr << "[test_timeembed_order] out[0] != cos(t): got " << got[0]
                  << " expect " << std::cos(t) << "\n";
        return false;
    }
    if (std::fabs(got[half] - std::sin(t)) > 3e-3f)
    {
        std::cerr << "[test_timeembed_order] out[half] != sin(t): got " << got[half]
                  << " expect " << std::sin(t) << "\n";
        return false;
    }
    std::cout << "[test_timeembed_order] PASSED (cos が先頭 half)\n";
    return true;
}

// ----------------------------------------------------------------
// 2. test_timeembed_sdxl: SDXL dim=320、複数 timestep を CPU 参照と突合。
// ----------------------------------------------------------------
static bool test_timeembed_sdxl()
{
    const int dim = 320;
    const std::vector<float> ts = {0.0f, 1.0f, 100.0f, 500.0f, 999.0f};
    for (float t : ts)
    {
        std::vector<float> got = run_gpu_te(t, dim, 10000.0f);
        std::vector<float> ref = cpu_time_embedding(t, dim, 10000.0f);
        std::string name = "test_timeembed_sdxl[t=" + std::to_string(static_cast<int>(t)) + "]";
        if (!compare(got, ref, name.c_str()))
        {
            return false;
        }
    }
    return true;
}

// ----------------------------------------------------------------
// 3. test_timeembed_dim_variants: 別 dim (256, 1280) でも一致。
// ----------------------------------------------------------------
static bool test_timeembed_dim_variants()
{
    const std::vector<int> dims = {128, 256, 1280};
    for (int dim : dims)
    {
        const float t = 321.0f;
        std::vector<float> got = run_gpu_te(t, dim, 10000.0f);
        std::vector<float> ref = cpu_time_embedding(t, dim, 10000.0f);
        std::string name = "test_timeembed_dim[" + std::to_string(dim) + "]";
        if (!compare(got, ref, name.c_str()))
        {
            return false;
        }
    }
    return true;
}

// ----------------------------------------------------------------
// ベンチ: 単発カーネルなので生レイテンシ (ms) で報告。
// ----------------------------------------------------------------
static void bench_timeembed()
{
    const int dim = 320;
    __half* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, static_cast<size_t>(dim) * sizeof(__half)));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    auto run_once = [&]() -> float
    {
        CUDA_CHECK(cudaEventRecord(start));
        launch_timestep_embedding(d_out, 500.0f, dim, 10000.0f);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    };

    const int warmup = 5;
    const int iters  = 50;
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

    std::cout << "[bench_timeembed] dim=" << dim
              << " median=" << times[times.size() / 2] << " ms"
              << " (min=" << times.front() << " max=" << times.back() << ") (N=" << iters << ")\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_out));
}

#endif // HAVE_CUDA

} // namespace dollama

int main()
{
#ifndef HAVE_CUDA
    std::cout << "[test_timeembed] [SKIP] HAVE_CUDA 未定義\n";
    return 0;
#else
    bool ok = true;
    ok = dollama::test_timeembed_order()        && ok;
    ok = dollama::test_timeembed_sdxl()         && ok;
    ok = dollama::test_timeembed_dim_variants() && ok;

    if (!ok)
    {
        std::cerr << "[test_timeembed] FAILED\n";
        return 1;
    }

    dollama::bench_timeembed();

    std::cout << "[test_timeembed] ALL PASSED\n";
    return 0;
#endif
}

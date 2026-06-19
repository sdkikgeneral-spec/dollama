// GEGLU 単体テスト + ベンチ (Phase 2 マイルストーン 2-5)
// HAVE_CUDA 未定義時は [SKIP] で return 0。
//
// 入力ビット一致: 乱数を FP32 生成 → FP16 へ丸め、その FP16 をデコードした値を
// CPU 参照の入力にも使う。tol は要素ごと活性化なので固定 (test_activation.cu と同方針)。
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#ifdef HAVE_CUDA
#include <cuda_fp16.h>
#include "kernels/geglu.cuh"
#include "kernels/utils.cuh"
#endif

namespace dollama
{

#ifdef HAVE_CUDA

// GEGLU は x1 * gelu(x2) の積で誤差が乗る。固定 tol だが activation よりやや緩める。
static constexpr float ATOL = 3e-3f;
static constexpr float RTOL = 3e-3f;

struct HalfBuffer
{
    std::vector<__half> h;
    std::vector<float>  ref;
};

static HalfBuffer make_half(int n, unsigned seed, float lo = -6.0f, float hi = 6.0f)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    HalfBuffer b;
    b.h.resize(n);
    b.ref.resize(n);
    for (int i = 0; i < n; ++i)
    {
        const float v = dist(rng);
        const __half hv = __float2half(v);
        b.h[i]   = hv;
        b.ref[i] = __half2float(hv);
    }
    return b;
}

static float cpu_gelu_erf(float x)
{
    return 0.5f * x * (1.0f + std::erf(x * 0.70710678f));
}

// CPU 参照: 入力 [tokens, 2d] → 出力 [tokens, d] = x1 * gelu(x2)。
static std::vector<float> cpu_geglu(const std::vector<float>& in, int tokens, int d)
{
    std::vector<float> out(static_cast<size_t>(tokens) * d);
    for (int t = 0; t < tokens; ++t)
    {
        const long row_base = static_cast<long>(t) * (2 * d);
        for (int c = 0; c < d; ++c)
        {
            const float x1 = in[row_base + c];
            const float x2 = in[row_base + d + c];
            out[static_cast<long>(t) * d + c] = x1 * cpu_gelu_erf(x2);
        }
    }
    return out;
}

static std::vector<float> run_gpu_geglu(const std::vector<__half>& in, int tokens, int d)
{
    const size_t n_in  = in.size();
    const size_t n_out = static_cast<size_t>(tokens) * d;

    __half* d_in  = nullptr;
    __half* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, n_in * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_in, in.data(), n_in * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_out, n_out * sizeof(__half)));

    launch_geglu(d_in, d_out, tokens, d);

    std::vector<__half> h_out(n_out);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, n_out * sizeof(__half), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    std::vector<float> out(n_out);
    for (size_t i = 0; i < n_out; ++i)
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
// 1. test_geglu_known: tokens=1, d=2。手計算で out = x1*gelu(x2)。
//   in = [x1_0, x1_1, x2_0, x2_1] = [1, -1, 1, 0]。
//   out[0] = 1 * gelu(1) ≈ 0.8413, out[1] = -1 * gelu(0) = 0。
// ----------------------------------------------------------------
static bool test_geglu_known()
{
    const int tokens = 1, d = 2;
    std::vector<float> xf = {1.0f, -1.0f, 1.0f, 0.0f};
    std::vector<__half> xh(xf.size());
    std::vector<float>  xref(xf.size());
    for (size_t i = 0; i < xf.size(); ++i)
    {
        xh[i]   = __float2half(xf[i]);
        xref[i] = __half2float(xh[i]);
    }

    std::vector<float> got = run_gpu_geglu(xh, tokens, d);
    std::vector<float> ref = cpu_geglu(xref, tokens, d);

    // out[0] = gelu(1) ≈ 0.8413, out[1] = -1*gelu(0) = 0。
    if (std::fabs(got[0] - 0.8413447f) > 3e-3f)
    {
        std::cerr << "[test_geglu_known] out[0] !~ 0.8413: " << got[0] << "\n";
        return false;
    }
    if (std::fabs(got[1]) > 3e-3f)
    {
        std::cerr << "[test_geglu_known] out[1] !~ 0: " << got[1] << "\n";
        return false;
    }
    return compare(got, ref, "test_geglu_known");
}

// ----------------------------------------------------------------
// 2. test_geglu_random: SDXL 代表 shape (tokens=4096, d=2560 → in 5120)。
// ----------------------------------------------------------------
static bool test_geglu_random()
{
    const int tokens = 4096, d = 2560;
    HalfBuffer b = make_half(tokens * (2 * d), 1234);
    std::vector<float> got = run_gpu_geglu(b.h, tokens, d);
    std::vector<float> ref = cpu_geglu(b.ref, tokens, d);
    return compare(got, ref, "test_geglu_random");
}

// ----------------------------------------------------------------
// 3. test_geglu_odd: 端数 tokens/d でインデックス分解が正しいか。
// ----------------------------------------------------------------
static bool test_geglu_odd()
{
    const int tokens = 37, d = 53;
    HalfBuffer b = make_half(tokens * (2 * d), 9999);
    std::vector<float> got = run_gpu_geglu(b.h, tokens, d);
    std::vector<float> ref = cpu_geglu(b.ref, tokens, d);
    return compare(got, ref, "test_geglu_odd");
}

// ----------------------------------------------------------------
// ベンチ: メモリ律速 → 有効帯域 GB/s。read in (tokens*2d) + write out (tokens*d)。
// ----------------------------------------------------------------
static void bench_one(int tokens, int d, const char* label)
{
    const long n_in  = static_cast<long>(tokens) * (2 * d);
    const long n_out = static_cast<long>(tokens) * d;
    HalfBuffer b = make_half(static_cast<int>(n_in), 777);

    __half* d_in  = nullptr;
    __half* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, static_cast<size_t>(n_in) * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_out, static_cast<size_t>(n_out) * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_in, b.h.data(), static_cast<size_t>(n_in) * sizeof(__half),
                          cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    auto run_once = [&]() -> float
    {
        CUDA_CHECK(cudaEventRecord(start));
        launch_geglu(d_in, d_out, tokens, d);
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
    const float median_ms = times[times.size() / 2];

    const double bytes = static_cast<double>(n_in + n_out) * sizeof(__half);
    const double gbps  = bytes / (static_cast<double>(median_ms) * 1e-3) / 1e9;

    std::cout << "[bench_geglu] " << label
              << " tokens=" << tokens << " d=" << d
              << " median=" << median_ms << " ms"
              << " (min=" << times.front() << " max=" << times.back() << ")"
              << " BW=" << gbps << " GB/s (N=" << iters << ")\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
}

static void bench_geglu()
{
    bench_one(4096, 2560, "unet_4096x2560");
    bench_one(1024, 1280, "unet_1024x1280");
}

#endif // HAVE_CUDA

} // namespace dollama

int main()
{
#ifndef HAVE_CUDA
    std::cout << "[test_geglu] [SKIP] HAVE_CUDA 未定義\n";
    return 0;
#else
    bool ok = true;
    ok = dollama::test_geglu_known()  && ok;
    ok = dollama::test_geglu_random() && ok;
    ok = dollama::test_geglu_odd()    && ok;

    if (!ok)
    {
        std::cerr << "[test_geglu] FAILED\n";
        return 1;
    }

    dollama::bench_geglu();

    std::cout << "[test_geglu] ALL PASSED\n";
    return 0;
#endif
}

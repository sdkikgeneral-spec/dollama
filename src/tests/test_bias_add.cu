// broadcast bias add 単体テスト + ベンチ (Phase 2 マイルストーン 2-5)
// HAVE_CUDA 未定義時は [SKIP] で return 0。
//
// 2 用途 (rowvec / channel) を CPU 参照と突合。加算のみなので固定 tol。
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#ifdef HAVE_CUDA
#include <cuda_fp16.h>
#include "kernels/bias_add.cuh"
#include "kernels/utils.cuh"
#endif

namespace dollama
{

#ifdef HAVE_CUDA

static constexpr float ATOL = 2e-3f;
static constexpr float RTOL = 2e-3f;

struct HalfBuffer
{
    std::vector<__half> h;
    std::vector<float>  ref;
};

static HalfBuffer make_half(int n, unsigned seed, float lo = -4.0f, float hi = 4.0f)
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

// ---- (a) rowvec ----

static std::vector<float> cpu_rowvec(const std::vector<float>& in,
                                     const std::vector<float>& bias,
                                     int tokens, int C)
{
    std::vector<float> out(in.size());
    for (int t = 0; t < tokens; ++t)
    {
        for (int c = 0; c < C; ++c)
        {
            const long idx = static_cast<long>(t) * C + c;
            out[idx] = in[idx] + bias[c];
        }
    }
    return out;
}

static std::vector<float> run_gpu_rowvec(const std::vector<__half>& in,
                                         const std::vector<__half>& bias,
                                         int tokens, int C, bool inplace)
{
    const size_t n = in.size();
    __half* d_in   = nullptr;
    __half* d_bias = nullptr;
    __half* d_out  = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_in, in.data(), n * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_bias, bias.size() * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_bias, bias.data(), bias.size() * sizeof(__half),
                          cudaMemcpyHostToDevice));
    if (inplace)
    {
        d_out = d_in;
    }
    else
    {
        CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(__half)));
    }

    launch_bias_add_rowvec(d_in, d_bias, d_out, tokens, C);

    std::vector<__half> h_out(n);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, n * sizeof(__half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_bias));
    if (!inplace)
    {
        CUDA_CHECK(cudaFree(d_out));
    }

    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i)
    {
        out[i] = __half2float(h_out[i]);
    }
    return out;
}

// ---- (b) channel ----

static std::vector<float> cpu_channel(const std::vector<float>& in,
                                      const std::vector<float>& bias,
                                      int N, int C, int H, int W)
{
    const int HW = H * W;
    std::vector<float> out(in.size());
    for (int n = 0; n < N; ++n)
    {
        for (int c = 0; c < C; ++c)
        {
            for (int p = 0; p < HW; ++p)
            {
                const long idx = (static_cast<long>(n) * C + c) * HW + p;
                out[idx] = in[idx] + bias[c];
            }
        }
    }
    return out;
}

static std::vector<float> run_gpu_channel(const std::vector<__half>& in,
                                          const std::vector<__half>& bias,
                                          int N, int C, int H, int W, bool inplace)
{
    const size_t n = in.size();
    __half* d_in   = nullptr;
    __half* d_bias = nullptr;
    __half* d_out  = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_in, in.data(), n * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_bias, bias.size() * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_bias, bias.data(), bias.size() * sizeof(__half),
                          cudaMemcpyHostToDevice));
    if (inplace)
    {
        d_out = d_in;
    }
    else
    {
        CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(__half)));
    }

    launch_bias_add_channel(d_in, d_bias, d_out, N, C, H, W);

    std::vector<__half> h_out(n);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, n * sizeof(__half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_bias));
    if (!inplace)
    {
        CUDA_CHECK(cudaFree(d_out));
    }

    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i)
    {
        out[i] = __half2float(h_out[i]);
    }
    return out;
}

// ----------------------------------------------------------------
// 1. test_rowvec_known: tokens=2, C=3。各行に同じ bias を足す。
// ----------------------------------------------------------------
static bool test_rowvec_known()
{
    const int tokens = 2, C = 3;
    std::vector<float> xf = {1.0f, 2.0f, 3.0f, 10.0f, 20.0f, 30.0f};
    std::vector<float> bf = {0.5f, -1.0f, 2.0f};
    std::vector<__half> xh(xf.size()), bh(bf.size());
    std::vector<float>  xref(xf.size()), bref(bf.size());
    for (size_t i = 0; i < xf.size(); ++i) { xh[i] = __float2half(xf[i]); xref[i] = __half2float(xh[i]); }
    for (size_t i = 0; i < bf.size(); ++i) { bh[i] = __float2half(bf[i]); bref[i] = __half2float(bh[i]); }

    std::vector<float> got = run_gpu_rowvec(xh, bh, tokens, C, false);
    std::vector<float> ref = cpu_rowvec(xref, bref, tokens, C);
    return compare(got, ref, "test_rowvec_known");
}

// ----------------------------------------------------------------
// 2. test_rowvec_random: SDXL Linear 出力相当 (tokens=4096, C=1280)。in-place も。
// ----------------------------------------------------------------
static bool test_rowvec_random()
{
    const int tokens = 4096, C = 1280;
    HalfBuffer b  = make_half(tokens * C, 2025);
    HalfBuffer bi = make_half(C, 31337, -2.0f, 2.0f);

    std::vector<float> got = run_gpu_rowvec(b.h, bi.h, tokens, C, false);
    std::vector<float> ref = cpu_rowvec(b.ref, bi.ref, tokens, C);
    if (!compare(got, ref, "test_rowvec_random"))
    {
        return false;
    }

    std::vector<float> ip = run_gpu_rowvec(b.h, bi.h, tokens, C, true);
    if (got != ip)
    {
        std::cerr << "[test_rowvec_random] in-place != out-of-place\n";
        return false;
    }
    return true;
}

// ----------------------------------------------------------------
// 3. test_channel_known: N=1, C=2, H=2, W=2。各チャネル平面に同じスカラ。
// ----------------------------------------------------------------
static bool test_channel_known()
{
    const int N = 1, C = 2, H = 2, W = 2;
    std::vector<float> xf = {
        1.0f, 2.0f, 3.0f, 4.0f,   // ch0
        5.0f, 6.0f, 7.0f, 8.0f    // ch1
    };
    std::vector<float> bf = {10.0f, -10.0f};
    std::vector<__half> xh(xf.size()), bh(bf.size());
    std::vector<float>  xref(xf.size()), bref(bf.size());
    for (size_t i = 0; i < xf.size(); ++i) { xh[i] = __float2half(xf[i]); xref[i] = __half2float(xh[i]); }
    for (size_t i = 0; i < bf.size(); ++i) { bh[i] = __float2half(bf[i]); bref[i] = __half2float(bh[i]); }

    std::vector<float> got = run_gpu_channel(xh, bh, N, C, H, W, false);
    std::vector<float> ref = cpu_channel(xref, bref, N, C, H, W);

    // ch0 全要素 +10、ch1 全要素 -10。
    if (std::fabs(got[0] - 11.0f) > 2e-3f || std::fabs(got[4] - (-5.0f)) > 2e-3f)
    {
        std::cerr << "[test_channel_known] broadcast 値が不正: got[0]=" << got[0]
                  << " got[4]=" << got[4] << "\n";
        return false;
    }
    return compare(got, ref, "test_channel_known");
}

// ----------------------------------------------------------------
// 4. test_channel_random: ResnetBlock 相当 (N=2, C=640, H=W=32)。in-place も。
// ----------------------------------------------------------------
static bool test_channel_random()
{
    const int N = 2, C = 640, H = 32, W = 32;
    HalfBuffer b  = make_half(N * C * H * W, 5678);
    HalfBuffer bi = make_half(C, 24680, -2.0f, 2.0f);

    std::vector<float> got = run_gpu_channel(b.h, bi.h, N, C, H, W, false);
    std::vector<float> ref = cpu_channel(b.ref, bi.ref, N, C, H, W);
    if (!compare(got, ref, "test_channel_random"))
    {
        return false;
    }

    std::vector<float> ip = run_gpu_channel(b.h, bi.h, N, C, H, W, true);
    if (got != ip)
    {
        std::cerr << "[test_channel_random] in-place != out-of-place\n";
        return false;
    }
    return true;
}

// ----------------------------------------------------------------
// ベンチ: メモリ律速 → 有効帯域 GB/s (read in + write out)。
// ----------------------------------------------------------------
static void bench_rowvec(int tokens, int C, const char* label)
{
    const int total = tokens * C;
    HalfBuffer b  = make_half(total, 777);
    HalfBuffer bi = make_half(C, 778);

    __half* d_in   = nullptr;
    __half* d_bias = nullptr;
    __half* d_out  = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, static_cast<size_t>(total) * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_bias, static_cast<size_t>(C) * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_out, static_cast<size_t>(total) * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_in, b.h.data(), static_cast<size_t>(total) * sizeof(__half),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, bi.h.data(), static_cast<size_t>(C) * sizeof(__half),
                          cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    auto run_once = [&]() -> float
    {
        CUDA_CHECK(cudaEventRecord(start));
        launch_bias_add_rowvec(d_in, d_bias, d_out, tokens, C);
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

    const double bytes = 2.0 * static_cast<double>(total) * sizeof(__half);
    const double gbps  = bytes / (static_cast<double>(median_ms) * 1e-3) / 1e9;

    std::cout << "[bench_bias_add] rowvec " << label
              << " tokens=" << tokens << " C=" << C
              << " median=" << median_ms << " ms"
              << " (min=" << times.front() << " max=" << times.back() << ")"
              << " BW=" << gbps << " GB/s (N=" << iters << ")\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_bias));
    CUDA_CHECK(cudaFree(d_out));
}

static void bench_bias_add()
{
    bench_rowvec(4096, 1280, "unet_4096x1280");
}

#endif // HAVE_CUDA

} // namespace dollama

int main()
{
#ifndef HAVE_CUDA
    std::cout << "[test_bias_add] [SKIP] HAVE_CUDA 未定義\n";
    return 0;
#else
    bool ok = true;
    ok = dollama::test_rowvec_known()   && ok;
    ok = dollama::test_rowvec_random()  && ok;
    ok = dollama::test_channel_known()  && ok;
    ok = dollama::test_channel_random() && ok;

    if (!ok)
    {
        std::cerr << "[test_bias_add] FAILED\n";
        return 1;
    }

    dollama::bench_bias_add();

    std::cout << "[test_bias_add] ALL PASSED\n";
    return 0;
#endif
}

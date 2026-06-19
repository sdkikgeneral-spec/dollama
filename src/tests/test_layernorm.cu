// LayerNorm 単体テスト + ベンチ (Phase 2 マイルストーン 2-5)
// HAVE_CUDA 未定義時は [SKIP] で return 0。
//
// 入力ビット一致: 乱数を FP32 生成 → FP16 へ丸め、その FP16 をデコードした値を
// CPU 参照の入力にも使う (カーネルの数値誤差だけを測る)。test_groupnorm.cu の枠を流用。
//
// tol: C 要素 (1280 など) のリダクションが入るため、要素ごと固定 tol では厳しい。
// groupnorm の K スケーリングと同じ考え方で C に応じて atol/rtol を緩める。
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#ifdef HAVE_CUDA
#include <cuda_fp16.h>
#include "kernels/layernorm.cuh"
#include "kernels/utils.cuh"
#endif

namespace dollama
{

#ifdef HAVE_CUDA

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

// C (正規化軸長) に応じた許容誤差。
static float atol_for(int C)
{
    return 3e-3f + 1e-6f * static_cast<float>(C);
}
static float rtol_for(int C)
{
    return 4e-3f + 5e-7f * static_cast<float>(C);
}

static bool compare(const std::vector<float>& got,
                    const std::vector<float>& ref,
                    int C,
                    const char* name)
{
    const float atol = atol_for(C);
    const float rtol = rtol_for(C);
    float max_rel = 0.0f;
    for (size_t i = 0; i < got.size(); ++i)
    {
        const float diff = std::fabs(got[i] - ref[i]);
        const float lim  = atol + rtol * std::fabs(ref[i]);
        if (diff > lim)
        {
            std::cerr << "[" << name << "] mismatch at " << i
                      << ": got " << got[i] << " ref " << ref[i]
                      << " diff " << diff << " lim " << lim
                      << " (C=" << C << " atol=" << atol << " rtol=" << rtol << ")\n";
            return false;
        }
        if (std::fabs(ref[i]) > 1e-3f)
        {
            max_rel = std::max(max_rel, diff / std::fabs(ref[i]));
        }
    }
    std::cout << "[" << name << "] PASSED (max_rel=" << max_rel
              << " atol=" << atol << " rtol=" << rtol << " C=" << C << ")\n";
    return true;
}

// CPU 参照 (FP32, 1 パス sum/sum-sq = カーネルと同一式)。
// gamma/beta が空ベクトルなら affine なし (scale=1, bias=0)。
static std::vector<float> cpu_layer_norm(const std::vector<float>& in,
                                         const std::vector<float>& gamma,
                                         const std::vector<float>& beta,
                                         int rows, int C, float eps)
{
    const bool has_g = !gamma.empty();
    const bool has_b = !beta.empty();
    std::vector<float> out(in.size());

    for (int r = 0; r < rows; ++r)
    {
        const long base = static_cast<long>(r) * C;
        float sum = 0.0f;
        float sq  = 0.0f;
        for (int c = 0; c < C; ++c)
        {
            const float x = in[base + c];
            sum += x;
            sq  += x * x;
        }
        const float inv_c = 1.0f / static_cast<float>(C);
        const float mean  = sum * inv_c;
        float var = sq * inv_c - mean * mean;
        var = std::max(var, 0.0f);
        const float inv_std = 1.0f / std::sqrt(var + eps);

        for (int c = 0; c < C; ++c)
        {
            const float gm = has_g ? gamma[c] : 1.0f;
            const float bt = has_b ? beta[c]  : 0.0f;
            out[base + c] = (in[base + c] - mean) * inv_std * gm + bt;
        }
    }
    return out;
}

// device 経由で LayerNorm を実行し FP16 結果をデコードして返す。
// gamma_h / beta_h が空なら nullptr を渡す (affine なし)。
static std::vector<float> run_gpu_ln(const std::vector<__half>& in,
                                     const std::vector<__half>& gamma_h,
                                     const std::vector<__half>& beta_h,
                                     int rows, int C, float eps, bool inplace)
{
    const size_t n = in.size();

    __half* d_in    = nullptr;
    __half* d_out   = nullptr;
    __half* d_gamma = nullptr;
    __half* d_beta  = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_in, in.data(), n * sizeof(__half), cudaMemcpyHostToDevice));

    if (!gamma_h.empty())
    {
        CUDA_CHECK(cudaMalloc(&d_gamma, gamma_h.size() * sizeof(__half)));
        CUDA_CHECK(cudaMemcpy(d_gamma, gamma_h.data(), gamma_h.size() * sizeof(__half),
                              cudaMemcpyHostToDevice));
    }
    if (!beta_h.empty())
    {
        CUDA_CHECK(cudaMalloc(&d_beta, beta_h.size() * sizeof(__half)));
        CUDA_CHECK(cudaMemcpy(d_beta, beta_h.data(), beta_h.size() * sizeof(__half),
                              cudaMemcpyHostToDevice));
    }

    if (inplace)
    {
        d_out = d_in;
    }
    else
    {
        CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(__half)));
    }

    launch_layer_norm(d_in, d_gamma, d_beta, d_out, rows, C, eps);

    std::vector<__half> h_out(n);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, n * sizeof(__half), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_in));
    if (!inplace)
    {
        CUDA_CHECK(cudaFree(d_out));
    }
    if (d_gamma)
    {
        CUDA_CHECK(cudaFree(d_gamma));
    }
    if (d_beta)
    {
        CUDA_CHECK(cudaFree(d_beta));
    }

    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i)
    {
        out[i] = __half2float(h_out[i]);
    }
    return out;
}

static std::vector<__half> ones_half(int c)
{
    return std::vector<__half>(c, __float2half(1.0f));
}
static std::vector<__half> zeros_half(int c)
{
    return std::vector<__half>(c, __float2half(0.0f));
}

// ----------------------------------------------------------------
// 1. test_layernorm_known: rows=2, C=4。gamma=1,beta=0 で各行平均≈0・分散≈1。
// ----------------------------------------------------------------
static bool test_layernorm_known()
{
    const int rows = 2, C = 4;
    const float eps = 1e-5f;
    std::vector<float> xf = {
        1.0f, 2.0f, 3.0f, 4.0f,    // row0
        -2.0f, 0.0f, 2.0f, 4.0f    // row1
    };
    std::vector<__half> xh(xf.size());
    std::vector<float>  xref(xf.size());
    for (size_t i = 0; i < xf.size(); ++i)
    {
        xh[i]   = __float2half(xf[i]);
        xref[i] = __half2float(xh[i]);
    }

    auto gamma = ones_half(C);
    auto beta  = zeros_half(C);
    std::vector<float> gamma_f(C, 1.0f);
    std::vector<float> beta_f(C, 0.0f);

    std::vector<float> got = run_gpu_ln(xh, gamma, beta, rows, C, eps, false);
    std::vector<float> ref = cpu_layer_norm(xref, gamma_f, beta_f, rows, C, eps);

    // 各行平均≈0・分散≈1 を直接確認。
    for (int r = 0; r < rows; ++r)
    {
        const int base = r * C;
        float m = 0.0f;
        for (int c = 0; c < C; ++c)
        {
            m += got[base + c];
        }
        m /= C;
        float v = 0.0f;
        for (int c = 0; c < C; ++c)
        {
            const float d = got[base + c] - m;
            v += d * d;
        }
        v /= C;
        if (std::fabs(m) > 5e-3f)
        {
            std::cerr << "[test_layernorm_known] row " << r << " mean !~ 0: " << m << "\n";
            return false;
        }
        if (std::fabs(v - 1.0f) > 2e-2f)
        {
            std::cerr << "[test_layernorm_known] row " << r << " var !~ 1: " << v << "\n";
            return false;
        }
    }

    return compare(got, ref, C, "test_layernorm_known");
}

// ----------------------------------------------------------------
// 2. test_layernorm_random: SDXL 代表 shape (rows=4096, C=1280)。affine あり。
// ----------------------------------------------------------------
static bool test_layernorm_random()
{
    const int rows = 4096, C = 1280;
    const float eps = 1e-5f;
    HalfBuffer b = make_half(rows * C, 2025);

    HalfBuffer gb = make_half(C, 24680, 0.2f, 2.0f);
    HalfBuffer bb = make_half(C, 11111, -1.5f, 1.5f);

    std::vector<float> got = run_gpu_ln(b.h, gb.h, bb.h, rows, C, eps, false);
    std::vector<float> ref = cpu_layer_norm(b.ref, gb.ref, bb.ref, rows, C, eps);
    return compare(got, ref, C, "test_layernorm_random");
}

// ----------------------------------------------------------------
// 3. test_layernorm_no_affine: gamma/beta = nullptr (affine なし)。
// ----------------------------------------------------------------
static bool test_layernorm_no_affine()
{
    const int rows = 333, C = 640;
    const float eps = 1e-5f;
    HalfBuffer b = make_half(rows * C, 13579);

    std::vector<__half> empty;
    std::vector<float>  empty_f;
    std::vector<float> got = run_gpu_ln(b.h, empty, empty, rows, C, eps, false);
    std::vector<float> ref = cpu_layer_norm(b.ref, empty_f, empty_f, rows, C, eps);
    return compare(got, ref, C, "test_layernorm_no_affine");
}

// ----------------------------------------------------------------
// 4. test_layernorm_inplace: d_in==d_out が out-of-place と一致。
// ----------------------------------------------------------------
static bool test_layernorm_inplace()
{
    const int rows = 512, C = 768;
    const float eps = 1e-5f;
    HalfBuffer b = make_half(rows * C, 31415);
    HalfBuffer gb = make_half(C, 27182, 0.5f, 1.5f);
    HalfBuffer bb = make_half(C, 16180, -0.5f, 0.5f);

    std::vector<float> oop = run_gpu_ln(b.h, gb.h, bb.h, rows, C, eps, false);
    std::vector<float> ip  = run_gpu_ln(b.h, gb.h, bb.h, rows, C, eps, true);

    if (oop != ip)
    {
        std::cerr << "[test_layernorm_inplace] in-place != out-of-place\n";
        return false;
    }
    std::vector<float> ref = cpu_layer_norm(b.ref, gb.ref, bb.ref, rows, C, eps);
    return compare(ip, ref, C, "test_layernorm_inplace");
}

// ----------------------------------------------------------------
// ベンチ: LayerNorm はメモリ律速 → 有効帯域 GB/s で報告。
// ----------------------------------------------------------------
static void bench_one(int rows, int C, const char* label)
{
    const int total = rows * C;
    HalfBuffer b = make_half(total, 777);
    auto gamma = ones_half(C);
    auto beta  = zeros_half(C);
    const float eps = 1e-5f;

    __half* d_in    = nullptr;
    __half* d_out   = nullptr;
    __half* d_gamma = nullptr;
    __half* d_beta  = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, static_cast<size_t>(total) * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_out, static_cast<size_t>(total) * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_gamma, static_cast<size_t>(C) * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_beta, static_cast<size_t>(C) * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_in, b.h.data(), static_cast<size_t>(total) * sizeof(__half),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, gamma.data(), static_cast<size_t>(C) * sizeof(__half),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, beta.data(), static_cast<size_t>(C) * sizeof(__half),
                          cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    auto run_once = [&]() -> float
    {
        CUDA_CHECK(cudaEventRecord(start));
        launch_layer_norm(d_in, d_gamma, d_beta, d_out, rows, C, eps);
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

    std::cout << "[bench_layernorm] " << label
              << " rows=" << rows << " C=" << C
              << " median=" << median_ms << " ms"
              << " (min=" << times.front() << " max=" << times.back() << ")"
              << " BW=" << gbps << " GB/s (N=" << iters << ")\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_beta));
}

static void bench_layernorm()
{
    // SDXL Transformer2D 代表 shape。
    bench_one(4096, 1280, "unet_4096x1280");
    bench_one(1024, 640, "unet_1024x640");
}

#endif // HAVE_CUDA

} // namespace dollama

int main()
{
#ifndef HAVE_CUDA
    std::cout << "[test_layernorm] [SKIP] HAVE_CUDA 未定義\n";
    return 0;
#else
    bool ok = true;
    ok = dollama::test_layernorm_known()     && ok;
    ok = dollama::test_layernorm_random()    && ok;
    ok = dollama::test_layernorm_no_affine() && ok;
    ok = dollama::test_layernorm_inplace()   && ok;

    if (!ok)
    {
        std::cerr << "[test_layernorm] FAILED\n";
        return 1;
    }

    dollama::bench_layernorm();

    std::cout << "[test_layernorm] ALL PASSED\n";
    return 0;
#endif
}

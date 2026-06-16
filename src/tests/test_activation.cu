// 要素ごと活性化関数 単体テスト + ベンチ (Phase 2 マイルストーン 2-2-2)
// HAVE_CUDA 未定義時は [SKIP] で return 0。
//
// 入力ビット一致: 乱数を FP32 生成 → FP16 へ丸め、その FP16 をデコードした値を
// CPU 参照の入力にも使う (カーネルの数値誤差だけを測る)。
// tol は固定 (atol=2e-3, rtol=2e-3)。要素ごとで K 蓄積はないためスケーリング不要。
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#ifdef HAVE_CUDA
#include <cuda_fp16.h>
#include "kernels/activation.cuh"
#include "kernels/utils.cuh"
#endif

namespace dollama
{

#ifdef HAVE_CUDA

// 固定許容誤差。要素ごとなので K 蓄積はない。
static constexpr float ATOL = 2e-3f;
static constexpr float RTOL = 2e-3f;

// ----------------------------------------------------------------
// テスト用ヘルパー
// ----------------------------------------------------------------

// FP32 乱数 → FP16 丸め。デコード値 (FP32) も同時に返し CPU 参照と入力ビットを一致させる。
struct HalfBuffer
{
    std::vector<__half> h;   // デバイスへ送る FP16
    std::vector<float>  ref; // CPU 参照に使う「FP16 をデコードした値」
};

static HalfBuffer make_half(int n, unsigned seed, float lo = -8.0f, float hi = 8.0f)
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
        b.ref[i] = __half2float(hv); // 丸め後の真値
    }
    return b;
}

// 活性化の種別。
enum class Act
{
    SiLU,
    GeluErf,
    GeluTanh
};

// device 経由で活性化を実行し FP16 結果をデコードして返す。
// inplace=true のときは d_in == d_out で起動する。
static std::vector<float> run_gpu_act(const std::vector<__half>& in, Act act, bool inplace)
{
    const size_t n = in.size();

    __half* d_in  = nullptr;
    __half* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_in, in.data(), n * sizeof(__half), cudaMemcpyHostToDevice));

    if (inplace)
    {
        d_out = d_in;
    }
    else
    {
        CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(__half)));
    }

    const int ni = static_cast<int>(n);
    switch (act)
    {
    case Act::SiLU:
        launch_silu(d_in, d_out, ni);
        break;
    case Act::GeluErf:
        launch_gelu(d_in, d_out, ni);
        break;
    case Act::GeluTanh:
        launch_gelu_tanh(d_in, d_out, ni);
        break;
    }

    std::vector<__half> h_out(n);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, n * sizeof(__half), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_in));
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

// CPU 参照 (FP32)。
static float cpu_silu(float x)
{
    return x / (1.0f + std::exp(-x));
}

static float cpu_gelu_erf(float x)
{
    return 0.5f * x * (1.0f + std::erf(x * 0.70710678f));
}

static float cpu_gelu_tanh(float x)
{
    const float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + std::tanh(inner));
}

static std::vector<float> cpu_apply(const std::vector<float>& in, Act act)
{
    std::vector<float> out(in.size());
    for (size_t i = 0; i < in.size(); ++i)
    {
        switch (act)
        {
        case Act::SiLU:
            out[i] = cpu_silu(in[i]);
            break;
        case Act::GeluErf:
            out[i] = cpu_gelu_erf(in[i]);
            break;
        case Act::GeluTanh:
            out[i] = cpu_gelu_tanh(in[i]);
            break;
        }
    }
    return out;
}

// 固定許容誤差で比較。
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
// 1. test_silu_known: x=0→0、大正→≈x、大負→≈0、±20 の極値
// ----------------------------------------------------------------
static bool test_silu_known()
{
    std::vector<float> xs = {0.0f, 1.0f, -1.0f, 5.0f, -5.0f, 20.0f, -20.0f};
    std::vector<__half> in(xs.size());
    std::vector<float>  ref(xs.size());
    for (size_t i = 0; i < xs.size(); ++i)
    {
        const __half hv = __float2half(xs[i]);
        in[i]  = hv;
        ref[i] = cpu_silu(__half2float(hv));
    }

    std::vector<float> got = run_gpu_act(in, Act::SiLU, false);

    // 既知性質も確認: x=0→0、大正→≈x、大負→≈0。
    if (std::fabs(got[0]) > 1e-3f)
    {
        std::cerr << "[test_silu_known] silu(0) != 0: " << got[0] << "\n";
        return false;
    }
    // x=20 → ≈20 (sigmoid(20)≈1)。FP16 で 20 は厳密表現可。
    if (std::fabs(got[5] - 20.0f) > 5e-2f)
    {
        std::cerr << "[test_silu_known] silu(20) !~ 20: " << got[5] << "\n";
        return false;
    }
    // x=-20 → ≈0。
    if (std::fabs(got[6]) > 5e-3f)
    {
        std::cerr << "[test_silu_known] silu(-20) !~ 0: " << got[6] << "\n";
        return false;
    }

    return compare(got, ref, "test_silu_known");
}

// ----------------------------------------------------------------
// 2. test_silu_random: 端数長 n=100003、CPU FP32 SiLU 参照
// ----------------------------------------------------------------
static bool test_silu_random()
{
    const int n = 100003;
    HalfBuffer b = make_half(n, 1234);
    std::vector<float> got = run_gpu_act(b.h, Act::SiLU, false);
    std::vector<float> ref = cpu_apply(b.ref, Act::SiLU);
    return compare(got, ref, "test_silu_random");
}

// ----------------------------------------------------------------
// 3. test_gelu_known: x=0→0、x=1→≈0.8413 (erf 既知値)
// ----------------------------------------------------------------
static bool test_gelu_known()
{
    std::vector<float> xs = {0.0f, 1.0f, -1.0f, 3.0f, -3.0f};
    std::vector<__half> in(xs.size());
    std::vector<float>  ref(xs.size());
    for (size_t i = 0; i < xs.size(); ++i)
    {
        const __half hv = __float2half(xs[i]);
        in[i]  = hv;
        ref[i] = cpu_gelu_erf(__half2float(hv));
    }

    std::vector<float> got = run_gpu_act(in, Act::GeluErf, false);

    if (std::fabs(got[0]) > 1e-3f)
    {
        std::cerr << "[test_gelu_known] gelu(0) != 0: " << got[0] << "\n";
        return false;
    }
    // gelu(1) = 0.5*1*(1+erf(1/sqrt2)) ≈ 0.84134。
    if (std::fabs(got[1] - 0.8413447f) > 2e-3f)
    {
        std::cerr << "[test_gelu_known] gelu(1) !~ 0.8413: " << got[1] << "\n";
        return false;
    }

    return compare(got, ref, "test_gelu_known");
}

// ----------------------------------------------------------------
// 4. test_gelu_random: CPU erf 版参照
// ----------------------------------------------------------------
static bool test_gelu_random()
{
    const int n = 100003;
    HalfBuffer b = make_half(n, 5678);
    std::vector<float> got = run_gpu_act(b.h, Act::GeluErf, false);
    std::vector<float> ref = cpu_apply(b.ref, Act::GeluErf);
    return compare(got, ref, "test_gelu_random");
}

// ----------------------------------------------------------------
// 5. test_gelu_tanh_random: CPU tanh 式参照 (erf 版とは別の参照式)
// ----------------------------------------------------------------
static bool test_gelu_tanh_random()
{
    const int n = 100003;
    HalfBuffer b = make_half(n, 9012);
    std::vector<float> got = run_gpu_act(b.h, Act::GeluTanh, false);
    std::vector<float> ref = cpu_apply(b.ref, Act::GeluTanh);
    return compare(got, ref, "test_gelu_tanh_random");
}

// ----------------------------------------------------------------
// 6. test_inplace: d_in==d_out で 1〜2 を再実行、out-of-place と同一結果
// ----------------------------------------------------------------
static bool test_inplace()
{
    // SiLU known 入力で in-place == out-of-place。
    {
        std::vector<float> xs = {0.0f, 1.0f, -1.0f, 5.0f, -5.0f, 20.0f, -20.0f};
        std::vector<__half> in(xs.size());
        for (size_t i = 0; i < xs.size(); ++i)
        {
            in[i] = __float2half(xs[i]);
        }
        std::vector<float> oop = run_gpu_act(in, Act::SiLU, false);
        std::vector<float> ip  = run_gpu_act(in, Act::SiLU, true);
        if (oop != ip)
        {
            std::cerr << "[test_inplace] silu known: in-place != out-of-place\n";
            return false;
        }
    }
    // SiLU random で in-place == out-of-place。
    {
        const int n = 100003;
        HalfBuffer b = make_half(n, 4242);
        std::vector<float> oop = run_gpu_act(b.h, Act::SiLU, false);
        std::vector<float> ip  = run_gpu_act(b.h, Act::SiLU, true);
        if (oop != ip)
        {
            std::cerr << "[test_inplace] silu random: in-place != out-of-place\n";
            return false;
        }
        // 参照とも一致することを確認 (in-place 経路の正しさ)。
        std::vector<float> ref = cpu_apply(b.ref, Act::SiLU);
        if (!compare(ip, ref, "test_inplace"))
        {
            return false;
        }
    }
    return true;
}

// ----------------------------------------------------------------
// ベンチ: 要素ごとはメモリ律速 → 有効帯域 GB/s で報告
// 有効帯域 = (read n + write n) * sizeof(__half) / ms = 2*n*2byte / ms
// ----------------------------------------------------------------
static void bench_one(int n, Act act, const char* label)
{
    HalfBuffer b = make_half(n, 777);

    __half* d_in  = nullptr;
    __half* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, static_cast<size_t>(n) * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_out, static_cast<size_t>(n) * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_in, b.h.data(), static_cast<size_t>(n) * sizeof(__half),
                          cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    auto run_once = [&]() -> float
    {
        CUDA_CHECK(cudaEventRecord(start));
        switch (act)
        {
        case Act::SiLU:
            launch_silu(d_in, d_out, n);
            break;
        case Act::GeluErf:
            launch_gelu(d_in, d_out, n);
            break;
        case Act::GeluTanh:
            launch_gelu_tanh(d_in, d_out, n);
            break;
        }
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

    // 有効帯域: read + write = 2 * n 要素 * 2 byte。
    const double bytes = 2.0 * static_cast<double>(n) * sizeof(__half);
    const double gbps  = bytes / (static_cast<double>(median_ms) * 1e-3) / 1e9;

    std::cout << "[bench_activation] " << label
              << " n=" << n
              << " median=" << median_ms << " ms"
              << " (min=" << times.front() << " max=" << times.back() << ")"
              << " BW=" << gbps << " GB/s (N=" << iters << ")\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
}

static void bench_activation()
{
    const int unet_fm = 1280 * 32 * 32; // UNet feature map 相当
    const int ffn     = 4096 * 5120;    // FFN 中間
    bench_one(unet_fm, Act::SiLU, "silu_unet_fm");
    bench_one(unet_fm, Act::GeluErf, "gelu_unet_fm");
    bench_one(ffn, Act::SiLU, "silu_ffn");
    bench_one(ffn, Act::GeluErf, "gelu_ffn");
}

#endif // HAVE_CUDA

} // namespace dollama

int main()
{
#ifndef HAVE_CUDA
    std::cout << "[test_activation] [SKIP] HAVE_CUDA 未定義\n";
    return 0;
#else
    bool ok = true;
    ok = dollama::test_silu_known()       && ok;
    ok = dollama::test_silu_random()      && ok;
    ok = dollama::test_gelu_known()       && ok;
    ok = dollama::test_gelu_random()      && ok;
    ok = dollama::test_gelu_tanh_random() && ok;
    ok = dollama::test_inplace()          && ok;

    if (!ok)
    {
        std::cerr << "[test_activation] FAILED\n";
        return 1;
    }

    dollama::bench_activation();

    std::cout << "[test_activation] ALL PASSED\n";
    return 0;
#endif
}

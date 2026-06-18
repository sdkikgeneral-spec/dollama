// Elementwise 単体テスト + ベンチ (Phase 2 マイルストーン 2-4 / VAE 準備)
// HAVE_CUDA 未定義時は [SKIP] で return 0。
//
// 入力ビット一致: 乱数を FP32 生成 → FP16 へ丸め、その FP16 をデコードした値を
// CPU 参照の入力にも使う (カーネルの数値誤差だけを測る)。test_conv2d.cu の枠を流用。
//
// add: CPU FP32 参照と tol 比較 (既知値 + 乱数 + in-place ケース)。
// upsample: 既知パターンで複製位置を厳密検証 (FP16 コピーなので bit 一致) + 乱数で形状確認。
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#ifdef HAVE_CUDA
#include <cuda_fp16.h>
#include "kernels/elementwise.cuh"
#include "kernels/utils.cuh"
#endif

namespace dollama
{

#ifdef HAVE_CUDA

// ----------------------------------------------------------------
// テスト用ヘルパー
// ----------------------------------------------------------------

// FP32 乱数 → FP16 丸め。デコード値 (FP32) も同時に返し CPU 参照と入力ビットを一致させる。
struct HalfBuffer
{
    std::vector<__half> h;   // デバイスへ送る FP16
    std::vector<float>  ref; // CPU 参照に使う「FP16 をデコードした値」
};

static HalfBuffer make_half(int n, unsigned seed, float lo = -1.0f, float hi = 1.0f)
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

// add 用 tol。1 回の FP32 加算 → FP16 丸めなので誤差は小さい。
static bool compare_add(const std::vector<float>& got,
                        const std::vector<float>& ref,
                        const char* name)
{
    const float atol = 2e-3f;
    const float rtol = 2e-3f;
    float max_rel = 0.0f;
    for (size_t i = 0; i < got.size(); ++i)
    {
        const float diff = std::fabs(got[i] - ref[i]);
        const float lim  = atol + rtol * std::fabs(ref[i]);
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
// add: device 実行ヘルパー。out_buf を別途指定できるよう d_out を引数で受ける版を
// 持たず、ここでは alias フラグで in-place を切り替える。
//   mode 0: out 別バッファ / mode 1: out==a / mode 2: out==b
// ----------------------------------------------------------------
static std::vector<float> run_gpu_add(const std::vector<__half>& a,
                                      const std::vector<__half>& b,
                                      int alias_mode)
{
    const int n = static_cast<int>(a.size());
    __half* d_a   = nullptr;
    __half* d_b   = nullptr;
    __half* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, a.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_b, b.size() * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_a, a.data(), a.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), b.size() * sizeof(__half), cudaMemcpyHostToDevice));

    __half* out_ptr = nullptr;
    if (alias_mode == 1)
    {
        out_ptr = d_a; // in-place: out==a
    }
    else if (alias_mode == 2)
    {
        out_ptr = d_b; // in-place: out==b
    }
    else
    {
        CUDA_CHECK(cudaMalloc(&d_out, a.size() * sizeof(__half)));
        out_ptr = d_out;
    }

    launch_add(d_a, d_b, out_ptr, n);

    std::vector<__half> h_out(n);
    CUDA_CHECK(cudaMemcpy(h_out.data(), out_ptr, n * sizeof(__half), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    if (d_out != nullptr)
    {
        CUDA_CHECK(cudaFree(d_out));
    }

    std::vector<float> out(n);
    for (int i = 0; i < n; ++i)
    {
        out[i] = __half2float(h_out[i]);
    }
    return out;
}

// ----------------------------------------------------------------
// 1. add: 既知値 (手計算近傍) + 乱数 + in-place 3 モード。
// ----------------------------------------------------------------
static bool test_add()
{
    bool ok = true;

    // 既知値: a = [1, -2, 0.5, 3], b = [0.25, 2, -0.5, -3] → out = [1.25, 0, 0, 0]
    {
        std::vector<float> av = {1.0f, -2.0f, 0.5f, 3.0f};
        std::vector<float> bv = {0.25f, 2.0f, -0.5f, -3.0f};
        std::vector<__half> ah(4), bh(4);
        std::vector<float> ref(4);
        for (int i = 0; i < 4; ++i)
        {
            ah[i] = __float2half(av[i]);
            bh[i] = __float2half(bv[i]);
            ref[i] = __half2float(ah[i]) + __half2float(bh[i]);
        }
        std::vector<float> got = run_gpu_add(ah, bh, 0);
        ok = compare_add(got, ref, "add_known") && ok;
    }

    // 乱数 (別バッファ)。
    {
        const int n = 4096 + 17; // 端数を含めて境界も踏む
        HalfBuffer a = make_half(n, 11);
        HalfBuffer b = make_half(n, 22);
        std::vector<float> ref(n);
        for (int i = 0; i < n; ++i)
        {
            ref[i] = a.ref[i] + b.ref[i];
        }
        std::vector<float> got = run_gpu_add(a.h, b.h, 0);
        ok = compare_add(got, ref, "add_rand") && ok;
    }

    // in-place: out==a。
    {
        const int n = 1000;
        HalfBuffer a = make_half(n, 33);
        HalfBuffer b = make_half(n, 44);
        std::vector<float> ref(n);
        for (int i = 0; i < n; ++i)
        {
            ref[i] = a.ref[i] + b.ref[i];
        }
        std::vector<float> got = run_gpu_add(a.h, b.h, 1);
        ok = compare_add(got, ref, "add_inplace_a") && ok;
    }

    // in-place: out==b。
    {
        const int n = 1000;
        HalfBuffer a = make_half(n, 55);
        HalfBuffer b = make_half(n, 66);
        std::vector<float> ref(n);
        for (int i = 0; i < n; ++i)
        {
            ref[i] = a.ref[i] + b.ref[i];
        }
        std::vector<float> got = run_gpu_add(a.h, b.h, 2);
        ok = compare_add(got, ref, "add_inplace_b") && ok;
    }

    return ok;
}

// ----------------------------------------------------------------
// upsample: device 実行ヘルパー。
// ----------------------------------------------------------------
static std::vector<__half> run_gpu_upsample(const std::vector<__half>& in,
                                            int N, int C, int H, int W)
{
    const size_t out_n = static_cast<size_t>(N) * C * (2 * H) * (2 * W);
    __half* d_in  = nullptr;
    __half* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, in.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_out, out_n * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_in, in.data(), in.size() * sizeof(__half), cudaMemcpyHostToDevice));

    launch_upsample_nearest2x(d_in, d_out, N, C, H, W);

    std::vector<__half> h_out(out_n);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, out_n * sizeof(__half), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return h_out;
}

// ----------------------------------------------------------------
// 2. upsample: 2x2 → 4x4 既知パターンで各画素の複製位置を厳密検証。
//    入力 (1,1,2,2) = [[10, 20],[30, 40]] (FP16 で正確に表せる値)。
//    期待 4x4: 各入力画素が 2x2 ブロックに複製される (nearest)。
//      10 10 20 20
//      10 10 20 20
//      30 30 40 40
//      30 30 40 40
//    FP16 コピーなので bit 一致を要求する。
// ----------------------------------------------------------------
static bool test_upsample_known()
{
    const int N = 1, C = 1, H = 2, W = 2;
    std::vector<__half> in(4);
    in[0] = __float2half(10.0f);
    in[1] = __float2half(20.0f);
    in[2] = __float2half(30.0f);
    in[3] = __float2half(40.0f);

    std::vector<__half> got = run_gpu_upsample(in, N, C, H, W);

    // 期待値を CPU で nearest 規則 (ho/2, wo/2) から生成。
    const int Ho = 4, Wo = 4;
    std::vector<float> expected(16);
    for (int ho = 0; ho < Ho; ++ho)
    {
        for (int wo = 0; wo < Wo; ++wo)
        {
            const int hi = ho / 2;
            const int wi = wo / 2;
            expected[ho * Wo + wo] = __half2float(in[hi * W + wi]);
        }
    }

    for (int idx = 0; idx < 16; ++idx)
    {
        const float g = __half2float(got[idx]);
        if (g != expected[idx]) // FP16 コピーなので完全一致
        {
            std::cerr << "[upsample_known] mismatch at " << idx
                      << " (ho=" << idx / Wo << " wo=" << idx % Wo << ")"
                      << ": got " << g << " expected " << expected[idx] << "\n";
            return false;
        }
    }
    std::cout << "[upsample_known] PASSED (bit-exact 2x2->4x4)\n";
    return true;
}

// ----------------------------------------------------------------
// 3. upsample: 乱数で形状 + 複製位置を多チャネル/多バッチで厳密検証 (bit 一致)。
// ----------------------------------------------------------------
static bool test_upsample_rand()
{
    const int N = 2, C = 3, H = 5, W = 7; // 奇数サイズも踏む
    HalfBuffer in = make_half(N * C * H * W, 12345);
    std::vector<__half> got = run_gpu_upsample(in.h, N, C, H, W);

    const int Ho = 2 * H, Wo = 2 * W;
    for (int n = 0; n < N; ++n)
    {
        for (int c = 0; c < C; ++c)
        {
            for (int ho = 0; ho < Ho; ++ho)
            {
                for (int wo = 0; wo < Wo; ++wo)
                {
                    const int hi = ho / 2;
                    const int wi = wo / 2;
                    const long in_idx  = (static_cast<long>(n) * C + c) * H * W
                                         + static_cast<long>(hi) * W + wi;
                    const long out_idx = (static_cast<long>(n) * C + c) * Ho * Wo
                                         + static_cast<long>(ho) * Wo + wo;
                    const float g = __half2float(got[out_idx]);
                    const float e = __half2float(in.h[in_idx]);
                    if (g != e)
                    {
                        std::cerr << "[upsample_rand] mismatch n=" << n << " c=" << c
                                  << " ho=" << ho << " wo=" << wo
                                  << ": got " << g << " expected " << e << "\n";
                        return false;
                    }
                }
            }
        }
    }
    std::cout << "[upsample_rand] PASSED (bit-exact " << N << "x" << C << "x" << H << "x" << W
              << " -> 2x)\n";
    return true;
}

// ----------------------------------------------------------------
// ベンチ: いずれもメモリ律速 → 有効帯域 (GB/s) を報告。
// add:      bytes = (2 read + 1 write) * n * 2byte
// upsample: bytes = (in + out) * 2byte = (n_in + 4*n_in) * 2byte
// warmup3 / iters100 / cudaEvent 中央値。
// ----------------------------------------------------------------
static void bench_add(int n, const char* label)
{
    HalfBuffer a = make_half(n, 7777);
    HalfBuffer b = make_half(n, 8888);
    __half* d_a = nullptr;
    __half* d_b = nullptr;
    __half* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, static_cast<size_t>(n) * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_b, static_cast<size_t>(n) * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_out, static_cast<size_t>(n) * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_a, a.h.data(), static_cast<size_t>(n) * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.h.data(), static_cast<size_t>(n) * sizeof(__half), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    auto run_once = [&]() -> float
    {
        CUDA_CHECK(cudaEventRecord(start));
        launch_add(d_a, d_b, d_out, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    };
    for (int i = 0; i < 3; ++i) { (void)run_once(); }
    std::vector<float> times;
    for (int i = 0; i < 100; ++i) { times.push_back(run_once()); }
    std::sort(times.begin(), times.end());
    const float med = times[times.size() / 2];
    const double bytes = 3.0 * static_cast<double>(n) * sizeof(__half);
    std::cout << "[bench_add] " << label << " n=" << n
              << " median=" << med << " ms BW=" << (bytes / (med * 1e-3) / 1e9) << " GB/s\n";
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));
}

static void bench_upsample(int N, int C, int H, int W, const char* label)
{
    const int in_n = N * C * H * W;
    const size_t out_n = static_cast<size_t>(N) * C * (2 * H) * (2 * W);
    HalfBuffer in = make_half(in_n, 7777);
    __half* d_in = nullptr;
    __half* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, static_cast<size_t>(in_n) * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_out, out_n * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_in, in.h.data(), static_cast<size_t>(in_n) * sizeof(__half), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    auto run_once = [&]() -> float
    {
        CUDA_CHECK(cudaEventRecord(start));
        launch_upsample_nearest2x(d_in, d_out, N, C, H, W);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    };
    for (int i = 0; i < 3; ++i) { (void)run_once(); }
    std::vector<float> times;
    for (int i = 0; i < 100; ++i) { times.push_back(run_once()); }
    std::sort(times.begin(), times.end());
    const float med = times[times.size() / 2];
    const double bytes = (static_cast<double>(in_n) + static_cast<double>(out_n)) * sizeof(__half);
    std::cout << "[bench_upsample] " << label
              << " " << N << "x" << C << "x" << H << "x" << W << " -> 2x"
              << " median=" << med << " ms BW=" << (bytes / (med * 1e-3) / 1e9) << " GB/s\n";
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
}

#endif // HAVE_CUDA

} // namespace dollama

int main()
{
#ifndef HAVE_CUDA
    std::cout << "[test_elementwise] [SKIP] HAVE_CUDA 未定義\n";
    return 0;
#else
    bool ok = true;
    ok = dollama::test_add()             && ok;
    ok = dollama::test_upsample_known()  && ok;
    ok = dollama::test_upsample_rand()   && ok;

    if (!ok)
    {
        std::cerr << "[test_elementwise] FAILED\n";
        return 1;
    }

    // ベンチ: VAE 代表サイズ。
    dollama::bench_add(1 * 512 * 128 * 128, "vae_c512_128");
    dollama::bench_upsample(1, 512, 128, 128, "vae_c512_128");
    dollama::bench_upsample(1, 128, 512, 512, "vae_c128_512");

    std::cout << "[test_elementwise] ALL PASSED\n";
    return 0;
#endif
}

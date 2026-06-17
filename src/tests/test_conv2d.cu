// Conv2d 単体テスト + ベンチ (Phase 2 マイルストーン 2-2-4)
// HAVE_CUDA 未定義時は [SKIP] で return 0。
//
// 入力ビット一致: 乱数を FP32 生成 → FP16 へ丸め、その FP16 をデコードした値を
// CPU 参照の入力にも使う (カーネルの数値誤差だけを測る)。test_groupnorm.cu の枠を流用。
//
// tol: 蓄積長 K = Cin*KH*KW に応じて atol/rtol を緩める (gemm / groupnorm と同じ
// 考え方)。FP16 入出力相応。K が大きいときは絶対誤差を併用 (atol が効く)。
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#ifdef HAVE_CUDA
#include <cuda_fp16.h>
#include "kernels/conv2d.cuh"
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
        b.ref[i] = __half2float(hv); // 丸め後の真値
    }
    return b;
}

// 出力サイズ計算 (カーネルと同一式)。
static int out_dim(int in, int pad, int dilation, int k, int stride)
{
    return (in + 2 * pad - dilation * (k - 1) - 1) / stride + 1;
}

// K (蓄積長 = Cin*KH*KW) に応じた許容誤差。
static float atol_for(int K)
{
    return 3e-3f + 2e-5f * static_cast<float>(K);
}
static float rtol_for(int K)
{
    return 5e-3f + 1e-5f * static_cast<float>(K);
}

// K スケール tol で比較。max_rel を表示。
static bool compare(const std::vector<float>& got,
                    const std::vector<float>& ref,
                    int K,
                    const char* name)
{
    const float atol = atol_for(K);
    const float rtol = rtol_for(K);
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
                      << " (K=" << K << " atol=" << atol << " rtol=" << rtol << ")\n";
            return false;
        }
        if (std::fabs(ref[i]) > 1e-3f)
        {
            max_rel = std::max(max_rel, diff / std::fabs(ref[i]));
        }
    }
    std::cout << "[" << name << "] PASSED (max_rel=" << max_rel
              << " atol=" << atol << " rtol=" << rtol << " K=" << K << ")\n";
    return true;
}

// CPU 参照 (FP32, direct conv = カーネルと同一式)。
// in/weight/bias は「FP16 をデコードした FP32 値」を渡すこと。bias 空なら加算なし。
static std::vector<float> cpu_conv2d(const std::vector<float>& in,
                                     const std::vector<float>& weight,
                                     const std::vector<float>& bias,
                                     int N, int Cin, int H, int W,
                                     int Cout, int KH, int KW,
                                     int stride_h, int stride_w,
                                     int pad_h, int pad_w,
                                     int dilation_h, int dilation_w)
{
    const int Hout = out_dim(H, pad_h, dilation_h, KH, stride_h);
    const int Wout = out_dim(W, pad_w, dilation_w, KW, stride_w);
    std::vector<float> out(static_cast<size_t>(N) * Cout * Hout * Wout, 0.0f);
    const bool has_bias = !bias.empty();

    for (int n = 0; n < N; ++n)
    {
        for (int co = 0; co < Cout; ++co)
        {
            for (int ho = 0; ho < Hout; ++ho)
            {
                for (int wo = 0; wo < Wout; ++wo)
                {
                    const int hi0 = ho * stride_h - pad_h;
                    const int wi0 = wo * stride_w - pad_w;
                    float acc = 0.0f;
                    for (int ci = 0; ci < Cin; ++ci)
                    {
                        const long in_ch_base = (static_cast<long>(n) * Cin + ci) * H * W;
                        const long w_ch_base  = (static_cast<long>(co) * Cin + ci) * KH * KW;
                        for (int kh = 0; kh < KH; ++kh)
                        {
                            const int hi = hi0 + kh * dilation_h;
                            if (hi < 0 || hi >= H)
                            {
                                continue;
                            }
                            for (int kw = 0; kw < KW; ++kw)
                            {
                                const int wi = wi0 + kw * dilation_w;
                                if (wi < 0 || wi >= W)
                                {
                                    continue;
                                }
                                const float x = in[in_ch_base + static_cast<long>(hi) * W + wi];
                                const float w = weight[w_ch_base + kh * KW + kw];
                                acc += x * w;
                            }
                        }
                    }
                    if (has_bias)
                    {
                        acc += bias[co];
                    }
                    const long oidx =
                        ((static_cast<long>(n) * Cout + co) * Hout + ho) * Wout + wo;
                    out[oidx] = acc;
                }
            }
        }
    }
    return out;
}

// device 経由で conv2d を実行し FP16 結果をデコードして返す。
// bias_h が空のときは nullptr で起動する。
static std::vector<float> run_gpu_conv(const std::vector<__half>& in,
                                       const std::vector<__half>& weight,
                                       const std::vector<__half>& bias_h,
                                       int N, int Cin, int H, int W,
                                       int Cout, int KH, int KW,
                                       int stride_h, int stride_w,
                                       int pad_h, int pad_w,
                                       int dilation_h, int dilation_w)
{
    const int Hout = out_dim(H, pad_h, dilation_h, KH, stride_h);
    const int Wout = out_dim(W, pad_w, dilation_w, KW, stride_w);
    const size_t out_n = static_cast<size_t>(N) * Cout * Hout * Wout;

    __half* d_in     = nullptr;
    __half* d_weight = nullptr;
    __half* d_bias   = nullptr;
    __half* d_out    = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, in.size() * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_in, in.data(), in.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_weight, weight.size() * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_weight, weight.data(), weight.size() * sizeof(__half),
                          cudaMemcpyHostToDevice));
    if (!bias_h.empty())
    {
        CUDA_CHECK(cudaMalloc(&d_bias, bias_h.size() * sizeof(__half)));
        CUDA_CHECK(cudaMemcpy(d_bias, bias_h.data(), bias_h.size() * sizeof(__half),
                              cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMalloc(&d_out, out_n * sizeof(__half)));

    launch_conv2d(d_in, d_weight, d_bias, d_out, N, Cin, H, W, Cout, KH, KW,
                  stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w);

    std::vector<__half> h_out(out_n);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, out_n * sizeof(__half), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_weight));
    if (d_bias != nullptr)
    {
        CUDA_CHECK(cudaFree(d_bias));
    }
    CUDA_CHECK(cudaFree(d_out));

    std::vector<float> out(out_n);
    for (size_t i = 0; i < out_n; ++i)
    {
        out[i] = __half2float(h_out[i]);
    }
    return out;
}

// 1 ケースを GPU 実行・CPU 参照と比較する共通ルーチン。
static bool run_case(const char* name,
                     int N, int Cin, int H, int W,
                     int Cout, int KH, int KW,
                     int stride_h, int stride_w,
                     int pad_h, int pad_w,
                     int dilation_h, int dilation_w,
                     bool with_bias,
                     unsigned seed)
{
    const int in_n = N * Cin * H * W;
    const int w_n  = Cout * Cin * KH * KW;

    HalfBuffer in = make_half(in_n, seed);
    HalfBuffer w  = make_half(w_n, seed + 1);

    std::vector<__half> bias_h;
    std::vector<float>  bias_ref;
    if (with_bias)
    {
        HalfBuffer bb = make_half(Cout, seed + 2, -0.5f, 0.5f);
        bias_h   = bb.h;
        bias_ref = bb.ref;
    }

    std::vector<float> got = run_gpu_conv(in.h, w.h, bias_h, N, Cin, H, W, Cout, KH, KW,
                                          stride_h, stride_w, pad_h, pad_w,
                                          dilation_h, dilation_w);
    std::vector<float> ref = cpu_conv2d(in.ref, w.ref, bias_ref, N, Cin, H, W, Cout, KH, KW,
                                        stride_h, stride_w, pad_h, pad_w,
                                        dilation_h, dilation_w);

    const int K = Cin * KH * KW; // 蓄積長
    return compare(got, ref, K, name);
}

// ----------------------------------------------------------------
// 1. 1x1 stride1 pad0 (実質 GEMM 相当)
// ----------------------------------------------------------------
static bool test_conv_1x1()
{
    bool ok = true;
    ok = run_case("conv_1x1[odd]", 1, 8, 15, 15, 6, 1, 1, 1, 1, 0, 0, 1, 1, false, 101) && ok;
    ok = run_case("conv_1x1[even]", 1, 8, 16, 16, 6, 1, 1, 1, 1, 0, 0, 1, 1, false, 102) && ok;
    return ok;
}

// ----------------------------------------------------------------
// 2. 3x3 stride1 pad1 (same conv)。奇数/偶数サイズ。
// ----------------------------------------------------------------
static bool test_conv_3x3_s1p1()
{
    bool ok = true;
    ok = run_case("conv_3x3_s1p1[odd]", 1, 4, 17, 17, 4, 3, 3, 1, 1, 1, 1, 1, 1, false, 201) && ok;
    ok = run_case("conv_3x3_s1p1[even]", 1, 4, 16, 16, 4, 3, 3, 1, 1, 1, 1, 1, 1, false, 202) && ok;
    return ok;
}

// ----------------------------------------------------------------
// 3. 3x3 stride2 pad1 (ダウンサンプル)。奇数/偶数サイズ。
// ----------------------------------------------------------------
static bool test_conv_3x3_s2p1()
{
    bool ok = true;
    ok = run_case("conv_3x3_s2p1[odd]", 1, 4, 17, 17, 8, 3, 3, 2, 2, 1, 1, 1, 1, false, 301) && ok;
    ok = run_case("conv_3x3_s2p1[even]", 1, 4, 16, 16, 8, 3, 3, 2, 2, 1, 1, 1, 1, false, 302) && ok;
    return ok;
}

// ----------------------------------------------------------------
// 4. bias 有/無 の比較 (同一 shape)。
// ----------------------------------------------------------------
static bool test_conv_bias()
{
    bool ok = true;
    ok = run_case("conv_bias[off]", 1, 6, 12, 12, 5, 3, 3, 1, 1, 1, 1, 1, 1, false, 401) && ok;
    ok = run_case("conv_bias[on]", 1, 6, 12, 12, 5, 3, 3, 1, 1, 1, 1, 1, 1, true, 401) && ok;
    return ok;
}

// ----------------------------------------------------------------
// 5. N=2・Cin>1・Cout>1 (バッチ + 多チャネル)。
// ----------------------------------------------------------------
static bool test_conv_batch_multi()
{
    return run_case("conv_batch_multi", 2, 5, 14, 18, 7, 3, 3, 1, 1, 1, 1, 1, 1, true, 501);
}

// ----------------------------------------------------------------
// 6. dilation=2 の 3x3 (受容野拡張)。
// ----------------------------------------------------------------
static bool test_conv_dilation()
{
    bool ok = true;
    ok = run_case("conv_dil2[pad2]", 1, 4, 20, 20, 4, 3, 3, 1, 1, 2, 2, 2, 2, false, 601) && ok;
    ok = run_case("conv_dil2[pad0]", 1, 4, 20, 20, 4, 3, 3, 1, 1, 0, 0, 2, 2, true, 602) && ok;
    return ok;
}

// ----------------------------------------------------------------
// ベンチ: conv は積和律速 → GFLOPS と有効帯域 (GB/s) を報告。
// FLOPs ≈ 2 * N*Cout*Hout*Wout * Cin*KH*KW (積 + 和)。
// bytes ≈ (in + weight + out) * 2byte (おおまかな読み書き総量)。
// warmup3 / iters100 / cudaEvent 中央値。
// ----------------------------------------------------------------
static void bench_one(int N, int Cin, int H, int W, int Cout, int KH, int KW,
                      int stride_h, int stride_w, int pad_h, int pad_w,
                      int dilation_h, int dilation_w, const char* label)
{
    const int Hout = out_dim(H, pad_h, dilation_h, KH, stride_h);
    const int Wout = out_dim(W, pad_w, dilation_w, KW, stride_w);
    const int in_n  = N * Cin * H * W;
    const int w_n   = Cout * Cin * KH * KW;
    const size_t out_n = static_cast<size_t>(N) * Cout * Hout * Wout;

    HalfBuffer in = make_half(in_n, 7777);
    HalfBuffer w  = make_half(w_n, 8888);
    HalfBuffer bb = make_half(Cout, 9999, -0.5f, 0.5f);

    __half* d_in     = nullptr;
    __half* d_weight = nullptr;
    __half* d_bias   = nullptr;
    __half* d_out    = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, static_cast<size_t>(in_n) * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_weight, static_cast<size_t>(w_n) * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_bias, static_cast<size_t>(Cout) * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_out, out_n * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_in, in.h.data(), static_cast<size_t>(in_n) * sizeof(__half),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, w.h.data(), static_cast<size_t>(w_n) * sizeof(__half),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, bb.h.data(), static_cast<size_t>(Cout) * sizeof(__half),
                          cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    auto run_once = [&]() -> float
    {
        CUDA_CHECK(cudaEventRecord(start));
        launch_conv2d(d_in, d_weight, d_bias, d_out, N, Cin, H, W, Cout, KH, KW,
                      stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    };

    const int warmup = 3;
    const int iters  = 100;
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

    const double flops = 2.0 * static_cast<double>(out_n)
                         * static_cast<double>(Cin) * KH * KW;
    const double gflops = flops / (static_cast<double>(median_ms) * 1e-3) / 1e9;
    const double bytes = (static_cast<double>(in_n) + static_cast<double>(w_n)
                          + static_cast<double>(out_n)) * sizeof(__half);
    const double gbps  = bytes / (static_cast<double>(median_ms) * 1e-3) / 1e9;

    std::cout << "[bench_conv2d] " << label
              << " N=" << N << " Cin=" << Cin << " H=" << H << " W=" << W
              << " Cout=" << Cout << " K=" << KH << "x" << KW
              << " s=" << stride_h << " p=" << pad_h
              << " -> " << Hout << "x" << Wout
              << " median=" << median_ms << " ms"
              << " (min=" << times.front() << " max=" << times.back() << ")"
              << " " << gflops << " GFLOPS"
              << " BW=" << gbps << " GB/s (N=" << iters << ")\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_weight));
    CUDA_CHECK(cudaFree(d_bias));
    CUDA_CHECK(cudaFree(d_out));
}

static void bench_conv2d()
{
    // UNet 系: C320 64x64 3x3 same conv。
    bench_one(1, 320, 64, 64, 320, 3, 3, 1, 1, 1, 1, 1, 1, "unet_c320_64");
    // VAE 系: C128 512x512 3x3 same conv。
    bench_one(1, 128, 512, 512, 128, 3, 3, 1, 1, 1, 1, 1, 1, "vae_c128_512");
}

#endif // HAVE_CUDA

} // namespace dollama

int main()
{
#ifndef HAVE_CUDA
    std::cout << "[test_conv2d] [SKIP] HAVE_CUDA 未定義\n";
    return 0;
#else
    bool ok = true;
    ok = dollama::test_conv_1x1()         && ok;
    ok = dollama::test_conv_3x3_s1p1()    && ok;
    ok = dollama::test_conv_3x3_s2p1()    && ok;
    ok = dollama::test_conv_bias()        && ok;
    ok = dollama::test_conv_batch_multi() && ok;
    ok = dollama::test_conv_dilation()    && ok;

    if (!ok)
    {
        std::cerr << "[test_conv2d] FAILED\n";
        return 1;
    }

    dollama::bench_conv2d();

    std::cout << "[test_conv2d] ALL PASSED\n";
    return 0;
#endif
}

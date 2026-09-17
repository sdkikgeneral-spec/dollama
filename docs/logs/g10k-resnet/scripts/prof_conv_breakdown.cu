// G-10k T6r: conv2d の batched (N=2) / seq (N=1 x2) per-call 内訳採取用の計測専用 exe。
//   ★コミットしない (作業ツリー限定・§8 ケース B' の「src コミットなし」)。
//   conv2d.cu には一切触らず、nsys の --capture-range=cudaProfilerApi で「計測区間の
//   カーネルだけ」を採る。内訳 (im2col / GEMM / bias / band scatter) はカーネル名で
//   nsys stats (cuda_gpu_kern_sum) から集計する。
//   使い方: prof_conv_breakdown <shape> <batched|seq> [iters]
//     shape = rep_320_128 | rep_640_64 | rep_1280_32 | G4_band_640to320_128 | unet_c320_64
//   同時に cuda event の per-call 中央値も印字する (bench_batch_vs_persample と同一の型・
//   nsys 集計値との整合確認用)。
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include "kernels/conv2d.cuh"
#include "kernels/utils.cuh"

using dollama::launch_conv2d;  // 宣言は conv2d.cuh (namespace dollama) 側

struct Shape
{
    const char* name;
    int Cin, H, W, Cout, KH, KW;
};

static const Shape kShapes[] = {
    {"unet_c320_64",         320,  64,  64,  320, 3, 3},
    {"rep_320_128",          320, 128, 128,  320, 3, 3},
    {"rep_640_64",           640,  64,  64,  640, 3, 3},
    {"rep_1280_32",         1280,  32,  32, 1280, 3, 3},
    {"G4_band_640to320_128", 640, 128, 128,  320, 3, 3},
};

static std::vector<__half> make_half(size_t n, unsigned seed, float lo = -1.0f, float hi = 1.0f)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<__half> v(n);
    for (size_t i = 0; i < n; ++i)
    {
        v[i] = __float2half(dist(rng));
    }
    return v;
}

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::fprintf(stderr, "usage: prof_conv_breakdown <shape> <batched|seq> [iters]\n");
        return 2;
    }
    const Shape* s = nullptr;
    for (const Shape& c : kShapes)
    {
        if (std::strcmp(c.name, argv[1]) == 0)
        {
            s = &c;
        }
    }
    if (s == nullptr)
    {
        std::fprintf(stderr, "unknown shape\n");
        return 2;
    }
    const bool batched = (std::strcmp(argv[2], "batched") == 0);
    const int iters = (argc >= 4) ? std::atoi(argv[3]) : 50;

    const int Hout = s->H, Wout = s->W; // 3x3 s1 p1 → same
    const int per_in  = s->Cin * s->H * s->W;
    const int per_out = s->Cout * Hout * Wout;
    const int w_n     = s->Cout * s->Cin * s->KH * s->KW;

    std::vector<__half> in = make_half(static_cast<size_t>(2) * per_in, 5551);
    std::vector<__half> w  = make_half(w_n, 5552);
    std::vector<__half> bb = make_half(s->Cout, 5553, -0.5f, 0.5f);

    __half *d_in = nullptr, *d_weight = nullptr, *d_bias = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, static_cast<size_t>(2) * per_in * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_weight, static_cast<size_t>(w_n) * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_bias, static_cast<size_t>(s->Cout) * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_out, static_cast<size_t>(2) * per_out * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_in, in.data(), static_cast<size_t>(2) * per_in * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, w.data(), static_cast<size_t>(w_n) * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, bb.data(), static_cast<size_t>(s->Cout) * sizeof(__half), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    auto run_once = [&]()
    {
        if (batched)
        {
            launch_conv2d(d_in, d_weight, d_bias, d_out, 2, s->Cin, s->H, s->W, s->Cout,
                          s->KH, s->KW, 1, 1, 1, 1, 1, 1);
        }
        else
        {
            launch_conv2d(d_in, d_weight, d_bias, d_out, 1, s->Cin, s->H, s->W, s->Cout,
                          s->KH, s->KW, 1, 1, 1, 1, 1, 1);
            launch_conv2d(d_in + per_in, d_weight, d_bias, d_out + per_out, 1, s->Cin, s->H, s->W,
                          s->Cout, s->KH, s->KW, 1, 1, 1, 1, 1, 1);
        }
    };
    auto time_ms = [&]() -> float
    {
        CUDA_CHECK(cudaEventRecord(start));
        run_once();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    };

    for (int i = 0; i < 3; ++i)
    {
        (void)time_ms();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaProfilerStart());
    std::vector<float> t;
    t.reserve(iters);
    for (int i = 0; i < iters; ++i)
    {
        t.push_back(time_ms());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaProfilerStop());
    std::sort(t.begin(), t.end());
    const double flops_per_call = 2.0 * 2.0 * s->Cout * static_cast<double>(Hout) * Wout
                                * static_cast<double>(s->Cin) * s->KH * s->KW; // N=2 item 分
    std::printf("[prof_conv_breakdown] shape=%s mode=%s iters=%d per-call median=%.6f ms min=%.6f ms"
                " gemm_flops_per_call(N=2)=%.6e\n",
                s->name, batched ? "batched" : "seq", iters, t[t.size() / 2], t[0], flops_per_call);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_weight));
    CUDA_CHECK(cudaFree(d_bias));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}

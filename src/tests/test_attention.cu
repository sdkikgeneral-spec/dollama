// Attention 単体テスト + ベンチ (Phase 2 マイルストーン 2-2-5)
// HAVE_CUDA 未定義時は [SKIP] で return 0。
//
// 入力ビット一致: 乱数を FP32 生成 → FP16 へ丸め、その FP16 をデコードした値を
// CPU 参照の入力にも使う (カーネルの数値誤差だけを測る)。test_conv2d.cu の枠を流用。
//
// tol: 蓄積長 (内積 Dh + softmax 加重和 Sk) に応じて atol/rtol を緩める。FP16 入出力
// 相応で、Sk/Dh が大きいときは絶対誤差を併用 (atol が効く)。softmax は確率の重み
// 平均なので出力レンジは入力 V と同程度に収まり、絶対誤差で十分捕捉できる。
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#ifdef HAVE_CUDA
#include <cuda_fp16.h>
#include "kernels/attention.cuh"
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

// 蓄積規模 K (= Dh + Sk) に応じた許容誤差。
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

// CPU 参照 (FP32, scaled dot-product attention = カーネルと同一式)。
// q/k/v は「FP16 をデコードした FP32 値」を渡すこと。
//   O = softmax(scale * Q·Kᵀ) · V。softmax は max 減算の 2 パス。
static std::vector<float> cpu_attention(const std::vector<float>& q,
                                        const std::vector<float>& k,
                                        const std::vector<float>& v,
                                        int B, int H, int Sq, int Sk, int Dh,
                                        float scale)
{
    std::vector<float> out(static_cast<size_t>(B) * H * Sq * Dh, 0.0f);

    for (int b = 0; b < B; ++b)
    {
        for (int h = 0; h < H; ++h)
        {
            const long q_head = (static_cast<long>(b) * H + h) * Sq * Dh;
            const long k_head = (static_cast<long>(b) * H + h) * Sk * Dh;
            const long v_head = (static_cast<long>(b) * H + h) * Sk * Dh;
            const long o_head = (static_cast<long>(b) * H + h) * Sq * Dh;

            for (int i = 0; i < Sq; ++i)
            {
                const long q_row = q_head + static_cast<long>(i) * Dh;

                // scores[j] = scale * Σ_d Q[i,d]*K[j,d]
                std::vector<float> scores(Sk);
                float row_max = -INFINITY;
                for (int j = 0; j < Sk; ++j)
                {
                    const long k_row = k_head + static_cast<long>(j) * Dh;
                    float dot = 0.0f;
                    for (int d = 0; d < Dh; ++d)
                    {
                        dot += q[q_row + d] * k[k_row + d];
                    }
                    scores[j] = scale * dot;
                    row_max = std::max(row_max, scores[j]);
                }

                // softmax (max 減算)
                float sum = 0.0f;
                for (int j = 0; j < Sk; ++j)
                {
                    scores[j] = std::exp(scores[j] - row_max);
                    sum += scores[j];
                }
                const float inv_sum = 1.0f / sum;

                // O[i,d] = Σ_j P[j]*V[j,d]
                for (int d = 0; d < Dh; ++d)
                {
                    float acc = 0.0f;
                    for (int j = 0; j < Sk; ++j)
                    {
                        const float p = scores[j] * inv_sum;
                        acc += p * v[v_head + static_cast<long>(j) * Dh + d];
                    }
                    out[o_head + static_cast<long>(i) * Dh + d] = acc;
                }
            }
        }
    }
    return out;
}

// device 経由で attention を実行し FP16 結果をデコードして返す。
static std::vector<float> run_gpu_attention(const std::vector<__half>& q,
                                            const std::vector<__half>& k,
                                            const std::vector<__half>& v,
                                            int B, int H, int Sq, int Sk, int Dh,
                                            float scale)
{
    const size_t out_n = static_cast<size_t>(B) * H * Sq * Dh;

    __half* d_q   = nullptr;
    __half* d_k   = nullptr;
    __half* d_v   = nullptr;
    __half* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_q, q.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_k, k.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_v, v.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_out, out_n * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_q, q.data(), q.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, k.data(), k.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, v.data(), v.size() * sizeof(__half), cudaMemcpyHostToDevice));

    launch_attention(d_q, d_k, d_v, d_out, B, H, Sq, Sk, Dh, scale);

    std::vector<__half> h_out(out_n);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, out_n * sizeof(__half), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_out));

    std::vector<float> out(out_n);
    for (size_t i = 0; i < out_n; ++i)
    {
        out[i] = __half2float(h_out[i]);
    }
    return out;
}

// 1 ケースを GPU 実行・CPU 参照と比較する共通ルーチン。
// scale < 0 のときはデフォルト 1/sqrt(Dh) を使う。
static bool run_case(const char* name,
                     int B, int H, int Sq, int Sk, int Dh,
                     float scale,
                     unsigned seed)
{
    const float s = (scale < 0.0f)
                        ? (1.0f / std::sqrt(static_cast<float>(Dh)))
                        : scale;

    const int q_n = B * H * Sq * Dh;
    const int k_n = B * H * Sk * Dh;
    const int v_n = B * H * Sk * Dh;

    HalfBuffer q = make_half(q_n, seed);
    HalfBuffer k = make_half(k_n, seed + 1);
    HalfBuffer v = make_half(v_n, seed + 2);

    std::vector<float> got = run_gpu_attention(q.h, k.h, v.h, B, H, Sq, Sk, Dh, s);
    std::vector<float> ref = cpu_attention(q.ref, k.ref, v.ref, B, H, Sq, Sk, Dh, s);

    const int K = Dh + Sk; // 蓄積規模 (内積長 + 加重和長)
    return compare(got, ref, K, name);
}

// ----------------------------------------------------------------
// 1. 小 self-attn (B1 H1 Sq=Sk=4 Dh=8)。手計算近傍の最小ケース。
// ----------------------------------------------------------------
static bool test_attn_small()
{
    return run_case("attn_small_self", 1, 1, 4, 4, 8, -1.0f, 1001);
}

// ----------------------------------------------------------------
// 2. self-attn 多ヘッダ多バッチ (B2 H8 Sq=Sk=64 Dh=40)。
// ----------------------------------------------------------------
static bool test_attn_multi()
{
    return run_case("attn_multi_self", 2, 8, 64, 64, 40, -1.0f, 2001);
}

// ----------------------------------------------------------------
// 3. cross-attn (Sq=256 Sk=77 Dh=64, SDXL 相当)。
// ----------------------------------------------------------------
static bool test_attn_cross()
{
    return run_case("attn_cross_sdxl", 1, 8, 256, 77, 64, -1.0f, 3001);
}

// ----------------------------------------------------------------
// 4. Sq != Sk の非対称ケース (上記とは別の比率)。
// ----------------------------------------------------------------
static bool test_attn_asym()
{
    bool ok = true;
    ok = run_case("attn_asym[Sq>Sk]", 1, 4, 100, 33, 32, -1.0f, 4001) && ok;
    ok = run_case("attn_asym[Sq<Sk]", 1, 4, 33, 100, 32, -1.0f, 4002) && ok;
    return ok;
}

// ----------------------------------------------------------------
// 5. scale が結果に効くこと (scale=1 と 1/sqrt(Dh) で別結果になる)。
//    両者を CPU 参照と一致させた上で、互いに異なることも確認する。
// ----------------------------------------------------------------
static bool test_attn_scale()
{
    bool ok = true;

    const int B = 1, H = 2, Sq = 16, Sk = 16, Dh = 32;
    const float s_one  = 1.0f;
    const float s_sqrt = 1.0f / std::sqrt(static_cast<float>(Dh));

    // scale=1 と scale=1/sqrt(Dh) それぞれが CPU 参照と一致すること。
    ok = run_case("attn_scale[=1]", B, H, Sq, Sk, Dh, s_one, 5001) && ok;
    ok = run_case("attn_scale[=1/sqrtDh]", B, H, Sq, Sk, Dh, s_sqrt, 5001) && ok;

    // 2 つの scale で GPU 出力が実際に異なることを確認 (同一なら scale 未反映)。
    const int q_n = B * H * Sq * Dh;
    const int k_n = B * H * Sk * Dh;
    const int v_n = B * H * Sk * Dh;
    HalfBuffer q = make_half(q_n, 5001);
    HalfBuffer k = make_half(k_n, 5002);
    HalfBuffer v = make_half(v_n, 5003);

    std::vector<float> o1 = run_gpu_attention(q.h, k.h, v.h, B, H, Sq, Sk, Dh, s_one);
    std::vector<float> o2 = run_gpu_attention(q.h, k.h, v.h, B, H, Sq, Sk, Dh, s_sqrt);

    float max_abs = 0.0f;
    for (size_t i = 0; i < o1.size(); ++i)
    {
        max_abs = std::max(max_abs, std::fabs(o1[i] - o2[i]));
    }
    if (max_abs < 1e-3f)
    {
        std::cerr << "[attn_scale[differs]] FAILED: scale=1 と 1/sqrtDh の出力差が小さすぎる ("
                  << max_abs << ")\n";
        ok = false;
    }
    else
    {
        std::cout << "[attn_scale[differs]] PASSED (max_abs_diff=" << max_abs << ")\n";
    }
    return ok;
}

// ----------------------------------------------------------------
// 6. 大 Sk (VAE mid_block 相当)。spatial self-attention は [1,512,128,128] を
//    [1, H=1, S=128*128=16384, Dh=512] に reshape した self-attn になる。
//    scores[Sk] が Sk*4byte = 64KB+ になり、デフォルト動的 shared 上限 48KB を
//    超えるため cudaFuncSetAttribute の opt-in 経路を踏む。
//
//    サイズ選定の判断:
//      - shared 要件は scores(Sk*4) で決まる (Dh は内積長で shared を食わない)。
//        Sk > 12288 で 48KB 超 → opt-in が必要。本ケースは Sk=16384 で 64KB を
//        要求し、本番 VAE mid と同じ shared 経路を厳密に踏む。
//      - 一方 CPU 参照は Sq*Sk*Dh の計算量なので、Sq を 16384 のままにすると
//        16384*16384*512 ≈ 1.4e11 で非現実的。各 query 行は独立に処理されるので、
//        Sq を 8 に絞っても「64KB 超の shared を踏む」性質は完全に保たれる
//        (kernel は 1 ブロック = 1 query 行で Sk 全体を shared に置くため)。
//        Sq=8 なら CPU 参照は 8*16384*512 ≈ 6.7e7 で現実的時間に収まる。
//      - Dh は本番 VAE mid と同じ 512 を維持 (内積長の数値挙動を保つ)。
// ----------------------------------------------------------------
static bool test_attn_large_sk()
{
    bool ok = true;
    // Sk=16384 (= 128*128) で scores = 64KB。Sq=8 に絞って CPU 参照を現実的に。
    ok = run_case("attn_vae_mid_largeSk", 1, 1, 8, 16384, 512, -1.0f, 6001) && ok;
    // 48KB 境界直上の追加ケース (Sk=12289 → 48.004KB)。opt-in 分岐の下限を踏む。
    ok = run_case("attn_largeSk_boundary", 1, 1, 4, 12289, 64, -1.0f, 6101) && ok;
    return ok;
}

// ----------------------------------------------------------------
// ベンチ: attention は 2 つの GEMM (Q·Kᵀ と P·V) 律速。
// FLOPs = 2 * (2*B*H*Sq*Sk*Dh) (QKᵀ と PV の積和ぶん)。
// warmup3 / iters100 / cudaEvent 中央値。
// ----------------------------------------------------------------
static void bench_one(int B, int H, int Sq, int Sk, int Dh, const char* label, int iters = 100)
{
    const float scale = 1.0f / std::sqrt(static_cast<float>(Dh));
    const int q_n = B * H * Sq * Dh;
    const int k_n = B * H * Sk * Dh;
    const int v_n = B * H * Sk * Dh;
    const size_t out_n = static_cast<size_t>(B) * H * Sq * Dh;

    HalfBuffer q = make_half(q_n, 7777);
    HalfBuffer k = make_half(k_n, 8888);
    HalfBuffer v = make_half(v_n, 9999);

    __half* d_q   = nullptr;
    __half* d_k   = nullptr;
    __half* d_v   = nullptr;
    __half* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_q, static_cast<size_t>(q_n) * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_k, static_cast<size_t>(k_n) * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_v, static_cast<size_t>(v_n) * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_out, out_n * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_q, q.h.data(), static_cast<size_t>(q_n) * sizeof(__half),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, k.h.data(), static_cast<size_t>(k_n) * sizeof(__half),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, v.h.data(), static_cast<size_t>(v_n) * sizeof(__half),
                          cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    auto run_once = [&]() -> float
    {
        CUDA_CHECK(cudaEventRecord(start));
        launch_attention(d_q, d_k, d_v, d_out, B, H, Sq, Sk, Dh, scale);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    };

    const int warmup = 3;
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

    // QKᵀ: 2*B*H*Sq*Sk*Dh, PV: 2*B*H*Sq*Sk*Dh。合計で 2 倍。
    const double flops = 2.0 * 2.0 * static_cast<double>(B) * H
                         * static_cast<double>(Sq) * Sk * Dh;
    const double gflops = flops / (static_cast<double>(median_ms) * 1e-3) / 1e9;

    std::cout << "[bench_attention] " << label
              << " B=" << B << " H=" << H << " Sq=" << Sq << " Sk=" << Sk << " Dh=" << Dh
              << " median=" << median_ms << " ms"
              << " (min=" << times.front() << " max=" << times.back() << ")"
              << " " << gflops << " GFLOPS (N=" << iters << ")\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_out));
}

static void bench_attention()
{
    // SDXL 代表 self-attn: B1 H8 Sq=Sk=1024 Dh=80。
    bench_one(1, 8, 1024, 1024, 80, "sdxl_self_1024");
    // SDXL 代表 cross-attn: B1 H8 Sq=1024 Sk=77 Dh=80。
    bench_one(1, 8, 1024, 77, 80, "sdxl_cross_1024x77");
    // VAE mid 代表 self-attn: B1 H1 Sq=Sk=16384 Dh=512 (64KB shared opt-in 経路)。
    // VAE mid は 1 反復 ~0.7s と重いので iters を 5 に絞る (テスト全体の時間短縮)。
    bench_one(1, 1, 16384, 16384, 512, "vae_mid_16384", 5);
}

#endif // HAVE_CUDA

} // namespace dollama

int main()
{
#ifndef HAVE_CUDA
    std::cout << "[test_attention] [SKIP] HAVE_CUDA 未定義\n";
    return 0;
#else
    bool ok = true;
    ok = dollama::test_attn_small()    && ok;
    ok = dollama::test_attn_multi()    && ok;
    ok = dollama::test_attn_cross()    && ok;
    ok = dollama::test_attn_asym()     && ok;
    ok = dollama::test_attn_scale()    && ok;
    ok = dollama::test_attn_large_sk() && ok;

    if (!ok)
    {
        std::cerr << "[test_attention] FAILED\n";
        return 1;
    }

    dollama::bench_attention();

    std::cout << "[test_attention] ALL PASSED\n";
    return 0;
#endif
}

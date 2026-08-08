// Attention 高速バリアント (launch_attention_fast) の単体テスト + ベンチ (FAST モード G-3k)
//
// 目的:
//   1. 正確性 — 小〜中規模 shape で launch_attention_fast を CPU 参照 (FP32) と突合。
//   2. 精度ゲート — SDXL 実 shape (self Sq=Sk=4096 / cross Sq≠Sk, B=1/B=2) で
//      launch_attention (既存 golden 経路) の出力と要素毎 SSIM を計算し、SSIM>=0.9999
//      を確定する。FP32 蓄積維持のため数学的には同一で、reduction 順の差だけが出る。
//   3. 速度 — 既存カーネル比の実測 ms をログ (4.621s ベースの改善帯を確認)。
//
// HAVE_CUDA 未定義時は [SKIP] で return 0。
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
// FP32 乱数 → FP16 丸め。デコード値も返し CPU 参照と入力ビットを一致させる。
// ----------------------------------------------------------------
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
        const float  v  = dist(rng);
        const __half hv = __float2half(v);
        b.h[i]   = hv;
        b.ref[i] = __half2float(hv);
    }
    return b;
}

// 蓄積規模 K (= Dh + Sk) に応じた許容誤差 (test_attention.cu と同じ物差し)。
static float atol_for(int K)
{
    return 3e-3f + 2e-5f * static_cast<float>(K);
}
static float rtol_for(int K)
{
    return 5e-3f + 1e-5f * static_cast<float>(K);
}

// CPU 参照 (FP32, scaled dot-product attention)。
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
                const long         q_row = q_head + static_cast<long>(i) * Dh;
                std::vector<float> scores(Sk);
                float              row_max = -INFINITY;
                for (int j = 0; j < Sk; ++j)
                {
                    const long k_row = k_head + static_cast<long>(j) * Dh;
                    float      dot   = 0.0f;
                    for (int d = 0; d < Dh; ++d)
                    {
                        dot += q[q_row + d] * k[k_row + d];
                    }
                    scores[j] = scale * dot;
                    row_max   = std::max(row_max, scores[j]);
                }
                float sum = 0.0f;
                for (int j = 0; j < Sk; ++j)
                {
                    scores[j] = std::exp(scores[j] - row_max);
                    sum += scores[j];
                }
                const float inv_sum = 1.0f / sum;
                for (int d = 0; d < Dh; ++d)
                {
                    float acc = 0.0f;
                    for (int j = 0; j < Sk; ++j)
                    {
                        acc += (scores[j] * inv_sum) * v[v_head + static_cast<long>(j) * Dh + d];
                    }
                    out[o_head + static_cast<long>(i) * Dh + d] = acc;
                }
            }
        }
    }
    return out;
}

// device 経由で attention を実行し FP16 結果をデコードして返す。
//   use_fast=true なら launch_attention_fast、false なら launch_attention。
static std::vector<float> run_gpu(const std::vector<__half>& q,
                                  const std::vector<__half>& k,
                                  const std::vector<__half>& v,
                                  int B, int H, int Sq, int Sk, int Dh, float scale,
                                  bool use_fast)
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

    if (use_fast)
    {
        launch_attention_fast(d_q, d_k, d_v, d_out, B, H, Sq, Sk, Dh, scale);
    }
    else
    {
        launch_attention(d_q, d_k, d_v, d_out, B, H, Sq, Sk, Dh, scale);
    }

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

// 要素毎の大域 SSIM。ref (基準 = 既存カーネル出力) のダイナミックレンジで正規化。
static double ssim(const std::vector<float>& a, const std::vector<float>& ref)
{
    const size_t n = a.size();
    double mu_a = 0.0, mu_b = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        mu_a += a[i];
        mu_b += ref[i];
    }
    mu_a /= static_cast<double>(n);
    mu_b /= static_cast<double>(n);

    double var_a = 0.0, var_b = 0.0, cov = 0.0;
    float  mn = ref.empty() ? 0.0f : ref[0];
    float  mx = ref.empty() ? 0.0f : ref[0];
    for (size_t i = 0; i < n; ++i)
    {
        const double da = a[i] - mu_a;
        const double db = ref[i] - mu_b;
        var_a += da * da;
        var_b += db * db;
        cov += da * db;
        mn = std::min(mn, ref[i]);
        mx = std::max(mx, ref[i]);
    }
    const double denom = (n > 1) ? static_cast<double>(n - 1) : 1.0;
    var_a /= denom;
    var_b /= denom;
    cov /= denom;

    double L = static_cast<double>(mx) - static_cast<double>(mn);
    if (L < 1e-6)
    {
        L = 1.0;
    }
    const double c1 = (0.01 * L) * (0.01 * L);
    const double c2 = (0.03 * L) * (0.03 * L);
    return ((2.0 * mu_a * mu_b + c1) * (2.0 * cov + c2))
           / ((mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2));
}

// ----------------------------------------------------------------
// 正確性: launch_attention_fast を CPU 参照と突合 (絶対/相対 tol)。
// ----------------------------------------------------------------
static bool correctness_case(const char* name,
                             int B, int H, int Sq, int Sk, int Dh,
                             unsigned seed)
{
    const float scale = 1.0f / std::sqrt(static_cast<float>(Dh));
    const int   q_n   = B * H * Sq * Dh;
    const int   k_n   = B * H * Sk * Dh;
    const int   v_n   = B * H * Sk * Dh;

    HalfBuffer q = make_half(q_n, seed);
    HalfBuffer k = make_half(k_n, seed + 1);
    HalfBuffer v = make_half(v_n, seed + 2);

    std::vector<float> got = run_gpu(q.h, k.h, v.h, B, H, Sq, Sk, Dh, scale, /*use_fast=*/true);
    std::vector<float> ref = cpu_attention(q.ref, k.ref, v.ref, B, H, Sq, Sk, Dh, scale);

    const int   K    = Dh + Sk;
    const float atol = atol_for(K);
    const float rtol = rtol_for(K);
    float       max_rel = 0.0f;
    for (size_t i = 0; i < got.size(); ++i)
    {
        const float diff = std::fabs(got[i] - ref[i]);
        const float lim  = atol + rtol * std::fabs(ref[i]);
        if (diff > lim)
        {
            std::cerr << "[" << name << "] mismatch at " << i << ": got " << got[i]
                      << " ref " << ref[i] << " diff " << diff << " lim " << lim
                      << " (K=" << K << ")\n";
            return false;
        }
        if (std::fabs(ref[i]) > 1e-3f)
        {
            max_rel = std::max(max_rel, diff / std::fabs(ref[i]));
        }
    }
    std::cout << "[" << name << "] PASSED (vs CPU: max_rel=" << max_rel
              << " atol=" << atol << " rtol=" << rtol << ")\n";
    return true;
}

// ----------------------------------------------------------------
// 精度ゲート: SDXL 実 shape で fast を既存 launch_attention と突合し SSIM>=0.9999。
// ----------------------------------------------------------------
static bool ssim_gate_case(const char* name,
                           int B, int H, int Sq, int Sk, int Dh,
                           unsigned seed)
{
    const float scale = 1.0f / std::sqrt(static_cast<float>(Dh));
    const int   q_n   = B * H * Sq * Dh;
    const int   k_n   = B * H * Sk * Dh;
    const int   v_n   = B * H * Sk * Dh;

    HalfBuffer q = make_half(q_n, seed);
    HalfBuffer k = make_half(k_n, seed + 1);
    HalfBuffer v = make_half(v_n, seed + 2);

    std::vector<float> base = run_gpu(q.h, k.h, v.h, B, H, Sq, Sk, Dh, scale, /*use_fast=*/false);
    std::vector<float> fast = run_gpu(q.h, k.h, v.h, B, H, Sq, Sk, Dh, scale, /*use_fast=*/true);

    const double s = ssim(fast, base);

    // 参考: fast vs base の最大絶対差も出す。
    float max_abs = 0.0f;
    for (size_t i = 0; i < fast.size(); ++i)
    {
        max_abs = std::max(max_abs, std::fabs(fast[i] - base[i]));
    }

    const bool ok = (s >= 0.9999);
    std::cout << "[" << name << "] SSIM=" << s << " (max_abs=" << max_abs
              << ") gate>=0.9999 " << (ok ? "PASSED" : "FAILED") << "\n";
    return ok;
}

// ----------------------------------------------------------------
// ベンチ: 既存 vs fast の中央値 ms を比較。warmup3 / N 中央値 / cudaEvent。
// ----------------------------------------------------------------
static void bench_pair(int B, int H, int Sq, int Sk, int Dh, const char* label, int iters = 20)
{
    const float  scale = 1.0f / std::sqrt(static_cast<float>(Dh));
    const int    q_n   = B * H * Sq * Dh;
    const int    k_n   = B * H * Sk * Dh;
    const int    v_n   = B * H * Sk * Dh;
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
    CUDA_CHECK(cudaMemcpy(d_q, q.h.data(), static_cast<size_t>(q_n) * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, k.h.data(), static_cast<size_t>(k_n) * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, v.h.data(), static_cast<size_t>(v_n) * sizeof(__half), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    auto run_once = [&](bool use_fast) -> float
    {
        CUDA_CHECK(cudaEventRecord(start));
        if (use_fast)
        {
            launch_attention_fast(d_q, d_k, d_v, d_out, B, H, Sq, Sk, Dh, scale);
        }
        else
        {
            launch_attention(d_q, d_k, d_v, d_out, B, H, Sq, Sk, Dh, scale);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    };

    auto median_of = [&](bool use_fast) -> float
    {
        for (int i = 0; i < 3; ++i)
        {
            (void)run_once(use_fast);
        }
        std::vector<float> t;
        t.reserve(iters);
        for (int i = 0; i < iters; ++i)
        {
            t.push_back(run_once(use_fast));
        }
        std::sort(t.begin(), t.end());
        return t[t.size() / 2];
    };

    const float base_ms = median_of(false);
    const float fast_ms = median_of(true);

    std::cout << "[bench_fast] " << label << " B=" << B << " H=" << H << " Sq=" << Sq
              << " Sk=" << Sk << " Dh=" << Dh << " | baseline=" << base_ms << " ms"
              << " fast=" << fast_ms << " ms"
              << " speedup=" << (fast_ms > 0.0f ? base_ms / fast_ms : 0.0f) << "x (N="
              << iters << ")\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_out));
}

#endif // HAVE_CUDA

} // namespace dollama

int main()
{
#ifndef HAVE_CUDA
    std::cout << "[test_attention_fast] [SKIP] HAVE_CUDA 未定義\n";
    return 0;
#else
    using namespace dollama;
    bool ok = true;

    // ---- 1. 正確性 (fast vs CPU 参照) ----
    // Dh=64 (UNet self/cross の主役)。self / cross / B=2 を網羅。
    ok = correctness_case("fast_self_small", 1, 2, 64, 64, 64, 1001) && ok;
    ok = correctness_case("fast_cross_small", 1, 2, 256, 77, 64, 1101) && ok;
    ok = correctness_case("fast_self_B2", 2, 2, 128, 128, 64, 1201) && ok;
    ok = correctness_case("fast_cross_B2", 2, 2, 96, 77, 64, 1301) && ok;
    // Dh=80 (SDXL self-attn) も正確性を確認。
    ok = correctness_case("fast_self_dh80", 1, 2, 64, 64, 80, 1401) && ok;
    // Dh が 16 の倍数でない (=フォールバック) 経路も CPU 参照と一致すること。
    ok = correctness_case("fast_fallback_dh40", 2, 4, 64, 40, 40, 1501) && ok;

    // ---- 2. 精度ゲート (fast vs 既存 launch_attention, SSIM>=0.9999) ----
    // SDXL 実 shape: self Sq=Sk=4096 / cross Sq=4096,Sk=77。B=1 と B=2 の両方。
    // baseline (1 warp/block) を回すため H=1 に絞って実行時間を抑える (律速の性質は不変)。
    ok = ssim_gate_case("gate_self4096_B1", 1, 1, 4096, 4096, 64, 2001) && ok;
    ok = ssim_gate_case("gate_self4096_B2", 2, 1, 4096, 4096, 64, 2101) && ok;
    ok = ssim_gate_case("gate_cross4096_B1", 1, 1, 4096, 77, 64, 2201) && ok;
    ok = ssim_gate_case("gate_cross4096_B2", 2, 1, 4096, 77, 64, 2301) && ok;

    if (!ok)
    {
        std::cerr << "[test_attention_fast] FAILED\n";
        return 1;
    }

    // ---- 3. ベンチ (既存 vs fast) ----
    bench_pair(1, 1, 4096, 4096, 64, "sdxl_self_4096");
    bench_pair(1, 8, 1024, 1024, 64, "sdxl_self_1024_h8");
    bench_pair(1, 8, 4096, 77, 64, "sdxl_cross_4096x77_h8");

    std::cout << "[test_attention_fast] ALL PASSED\n";
    return 0;
#endif
}

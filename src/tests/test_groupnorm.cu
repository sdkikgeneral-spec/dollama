// GroupNorm 単体テスト + ベンチ (Phase 2 マイルストーン 2-2-3)
// HAVE_CUDA 未定義時は [SKIP] で return 0。
//
// 入力ビット一致: 乱数を FP32 生成 → FP16 へ丸め、その FP16 をデコードした値を
// CPU 参照の入力にも使う (カーネルの数値誤差だけを測る)。test_activation.cu の枠を流用。
//
// tol: グループ内 K = cpg*H*W (数万) のリダクションが入るため、要素ごと activation の
// 固定 2e-3 では厳しい。gemm の K スケーリングの考え方に倣い、K に応じて atol/rtol を
// 緩める。FP16 入出力相応 (出力は __half へ丸めるため最低 1 ULP 級の誤差が乗る)。
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

#ifdef HAVE_CUDA
#include <cuda_fp16.h>
#include "kernels/groupnorm.cuh"
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
        b.ref[i] = __half2float(hv); // 丸め後の真値
    }
    return b;
}

// K (グループ要素数) に応じた許容誤差。
// atol は FP16 出力丸め分のフロア + K に比例した蓄積分。rtol も K で少し緩める。
// gemm の K スケーリングと同じ考え方 (リダクション長が長いほど誤差が乗る)。
static float atol_for(int K)
{
    return 3e-3f + 1e-6f * static_cast<float>(K);
}
static float rtol_for(int K)
{
    return 4e-3f + 5e-7f * static_cast<float>(K);
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

// CPU 参照 (FP32, 1 パス sum/sum-sq 方式 = カーネルと同一式)。
// in/gamma/beta は「FP16 をデコードした FP32 値」を渡すこと。
static std::vector<float> cpu_group_norm(const std::vector<float>& in,
                                         const std::vector<float>& gamma,
                                         const std::vector<float>& beta,
                                         int N, int C, int H, int W,
                                         int num_groups, float eps)
{
    const int HW  = H * W;
    const int cpg = C / num_groups;
    const int K   = cpg * HW;
    std::vector<float> out(in.size());

    for (int n = 0; n < N; ++n)
    {
        for (int g = 0; g < num_groups; ++g)
        {
            const int c_begin = g * cpg;
            const long base   = (static_cast<long>(n) * C + c_begin) * HW;

            // 1 パス sum / sum-sq。
            float sum = 0.0f;
            float sq  = 0.0f;
            for (int i = 0; i < K; ++i)
            {
                const float x = in[base + i];
                sum += x;
                sq  += x * x;
            }
            const float inv_k = 1.0f / static_cast<float>(K);
            const float mean  = sum * inv_k;
            float var = sq * inv_k - mean * mean;
            var = std::max(var, 0.0f);
            const float inv_std = 1.0f / std::sqrt(var + eps);

            for (int i = 0; i < K; ++i)
            {
                const int c = c_begin + (i / HW);
                const float x = in[base + i];
                out[base + i] = (x - mean) * inv_std * gamma[c] + beta[c];
            }
        }
    }
    return out;
}

// device 経由で GroupNorm を実行し FP16 結果をそのまま (raw ビット) 返す。
// inplace=true のときは d_in == d_out で起動する。
// gamma_h / beta_h は FP16。mb=true で multi-block 版 (G-4k S1a) を起動する。
static std::vector<__half> run_gpu_gn_raw(const std::vector<__half>& in,
                                          const std::vector<__half>& gamma_h,
                                          const std::vector<__half>& beta_h,
                                          int N, int C, int H, int W,
                                          int num_groups, float eps, bool inplace,
                                          bool mb = false)
{
    const size_t n = in.size();

    __half* d_in    = nullptr;
    __half* d_out   = nullptr;
    __half* d_gamma = nullptr;
    __half* d_beta  = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_in, in.data(), n * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_gamma, gamma_h.size() * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_gamma, gamma_h.data(), gamma_h.size() * sizeof(__half),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_beta, beta_h.size() * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_beta, beta_h.data(), beta_h.size() * sizeof(__half),
                          cudaMemcpyHostToDevice));

    if (inplace)
    {
        d_out = d_in;
    }
    else
    {
        CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(__half)));
    }

    if (mb)
    {
        launch_group_norm_mb(d_in, d_gamma, d_beta, d_out, N, C, H, W, num_groups, eps);
    }
    else
    {
        launch_group_norm(d_in, d_gamma, d_beta, d_out, N, C, H, W, num_groups, eps);
    }

    std::vector<__half> h_out(n);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, n * sizeof(__half), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_in));
    if (!inplace)
    {
        CUDA_CHECK(cudaFree(d_out));
    }
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_beta));

    return h_out;
}

// run_gpu_gn_raw の FP32 デコード版 (既存テストの物差し)。
static std::vector<float> run_gpu_gn(const std::vector<__half>& in,
                                     const std::vector<__half>& gamma_h,
                                     const std::vector<__half>& beta_h,
                                     int N, int C, int H, int W,
                                     int num_groups, float eps, bool inplace,
                                     bool mb = false)
{
    std::vector<__half> h_out = run_gpu_gn_raw(in, gamma_h, beta_h,
                                               N, C, H, W, num_groups, eps, inplace, mb);
    std::vector<float> out(h_out.size());
    for (size_t i = 0; i < h_out.size(); ++i)
    {
        out[i] = __half2float(h_out[i]);
    }
    return out;
}

// gamma=1 / beta=0 の affine 恒等を FP16 で作るヘルパー。
static std::vector<__half> ones_half(int c)
{
    return std::vector<__half>(c, __float2half(1.0f));
}
static std::vector<__half> zeros_half(int c)
{
    return std::vector<__half>(c, __float2half(0.0f));
}

// ----------------------------------------------------------------
// 1. test_groupnorm_known: N=1,C=4,H=2,W=2,G=2、CPU 手計算で mean/var/出力一致。
//    gamma=1,beta=0 で各グループ平均≈0・分散≈1。
// ----------------------------------------------------------------
static bool test_groupnorm_known()
{
    const int N = 1, C = 4, H = 2, W = 2, G = 2;
    const float eps = 1e-6f;
    // グループ0 = ch0,ch1 (8要素)、グループ1 = ch2,ch3 (8要素)。
    // 既知の値を入れる。
    std::vector<float> xf = {
        // ch0
        1.0f, 2.0f, 3.0f, 4.0f,
        // ch1
        5.0f, 6.0f, 7.0f, 8.0f,
        // ch2
        -1.0f, -2.0f, -3.0f, -4.0f,
        // ch3
        2.0f, 2.0f, 2.0f, 2.0f
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

    std::vector<float> got = run_gpu_gn(xh, gamma, beta, N, C, H, W, G, eps, false);
    std::vector<float> ref = cpu_group_norm(xref, gamma_f, beta_f, N, C, H, W, G, eps);

    const int K = (C / G) * H * W; // 8

    // 各グループ平均≈0・分散≈1 を直接確認 (gamma=1,beta=0)。
    for (int g = 0; g < G; ++g)
    {
        const int base = g * K;
        float m = 0.0f;
        for (int i = 0; i < K; ++i)
        {
            m += got[base + i];
        }
        m /= K;
        float v = 0.0f;
        for (int i = 0; i < K; ++i)
        {
            const float d = got[base + i] - m;
            v += d * d;
        }
        v /= K;
        if (std::fabs(m) > 5e-3f)
        {
            std::cerr << "[test_groupnorm_known] group " << g << " mean !~ 0: " << m << "\n";
            return false;
        }
        if (std::fabs(v - 1.0f) > 2e-2f)
        {
            std::cerr << "[test_groupnorm_known] group " << g << " var !~ 1: " << v << "\n";
            return false;
        }
    }

    return compare(got, ref, K, "test_groupnorm_known");
}

// ----------------------------------------------------------------
// 2. test_groupnorm_random: SDXL 代表 shape、CPU 1 パス FP32 参照と tol 比較。
// ----------------------------------------------------------------
static bool test_groupnorm_random()
{
    const int N = 2, C = 320, H = 32, W = 32, G = 32;
    const float eps = 1e-6f;
    const int total = N * C * H * W;
    HalfBuffer b = make_half(total, 2025);

    // gamma=1,beta=0 (純正規化のみ検証)。
    auto gamma = ones_half(C);
    auto beta  = zeros_half(C);
    std::vector<float> gamma_f(C, 1.0f);
    std::vector<float> beta_f(C, 0.0f);

    std::vector<float> got = run_gpu_gn(b.h, gamma, beta, N, C, H, W, G, eps, false);
    std::vector<float> ref = cpu_group_norm(b.ref, gamma_f, beta_f, N, C, H, W, G, eps);

    const int K = (C / G) * H * W;
    return compare(got, ref, K, "test_groupnorm_random");
}

// ----------------------------------------------------------------
// 3. test_groupnorm_affine: gamma/beta 非自明値。
// ----------------------------------------------------------------
static bool test_groupnorm_affine()
{
    const int N = 1, C = 128, H = 16, W = 16, G = 32;
    const float eps = 1e-5f;
    const int total = N * C * H * W;
    HalfBuffer b = make_half(total, 13579);

    // チャネルごとに変化する gamma/beta。
    HalfBuffer gb = make_half(C, 24680, 0.2f, 2.0f);   // gamma
    HalfBuffer bb = make_half(C, 11111, -1.5f, 1.5f);  // beta

    std::vector<float> got = run_gpu_gn(b.h, gb.h, bb.h, N, C, H, W, G, eps, false);
    std::vector<float> ref = cpu_group_norm(b.ref, gb.ref, bb.ref, N, C, H, W, G, eps);

    const int K = (C / G) * H * W;
    return compare(got, ref, K, "test_groupnorm_affine");
}

// ----------------------------------------------------------------
// 4. test_groupnorm_inplace: d_in==d_out が out-of-place と一致。
// ----------------------------------------------------------------
static bool test_groupnorm_inplace()
{
    const int N = 2, C = 64, H = 32, W = 32, G = 16;
    const float eps = 1e-6f;
    const int total = N * C * H * W;
    HalfBuffer b = make_half(total, 31415);
    HalfBuffer gb = make_half(C, 27182, 0.5f, 1.5f);
    HalfBuffer bb = make_half(C, 16180, -0.5f, 0.5f);

    std::vector<float> oop = run_gpu_gn(b.h, gb.h, bb.h, N, C, H, W, G, eps, false);
    std::vector<float> ip  = run_gpu_gn(b.h, gb.h, bb.h, N, C, H, W, G, eps, true);

    if (oop != ip)
    {
        std::cerr << "[test_groupnorm_inplace] in-place != out-of-place\n";
        return false;
    }

    // 参照とも一致することを確認 (in-place 経路の正しさ)。
    std::vector<float> ref = cpu_group_norm(b.ref, gb.ref, bb.ref, N, C, H, W, G, eps);
    const int K = (C / G) * H * W;
    return compare(ip, ref, K, "test_groupnorm_inplace");
}

// ----------------------------------------------------------------
// 5. test_groupnorm_edges: num_groups エッジ。
//    num_groups==C (InstanceNorm 相当) / num_groups==1 (LayerNorm 相当)。
// ----------------------------------------------------------------
static bool test_groupnorm_edges()
{
    const int N = 2, C = 32, H = 8, W = 8;
    const float eps = 1e-6f;
    const int total = N * C * H * W;
    HalfBuffer b = make_half(total, 99999);
    HalfBuffer gb = make_half(C, 88888, 0.3f, 1.7f);
    HalfBuffer bb = make_half(C, 77777, -1.0f, 1.0f);

    // num_groups == C: 各チャネル単独で正規化 = InstanceNorm 相当。
    {
        const int G = C;
        std::vector<float> got = run_gpu_gn(b.h, gb.h, bb.h, N, C, H, W, G, eps, false);
        std::vector<float> ref = cpu_group_norm(b.ref, gb.ref, bb.ref, N, C, H, W, G, eps);
        const int K = (C / G) * H * W; // = H*W
        if (!compare(got, ref, K, "test_groupnorm_edges[G==C]"))
        {
            return false;
        }
    }

    // num_groups == 1: 全チャネルまとめて正規化 = LayerNorm 相当。
    {
        const int G = 1;
        std::vector<float> got = run_gpu_gn(b.h, gb.h, bb.h, N, C, H, W, G, eps, false);
        std::vector<float> ref = cpu_group_norm(b.ref, gb.ref, bb.ref, N, C, H, W, G, eps);
        const int K = (C / G) * H * W; // = C*H*W
        if (!compare(got, ref, K, "test_groupnorm_edges[G==1]"))
        {
            return false;
        }
    }

    return true;
}

// ----------------------------------------------------------------
// 6. test_groupnorm_mb_parity (G-4k S1a ゲート1):
//    multi-block 版 vs 1-block 版のパリティ。蓄積順が異なるため FP16 で微差が乗る。
//    ゲート: MAE < 1e-3 / max abs < 4e-3 (FP16 tol)。SDXL 実形状 + B=2。
//    加えて CPU FP32 参照とも compare (mb 単体の正しさ)。
// ----------------------------------------------------------------
static bool test_groupnorm_mb_parity_one(int N, int C, int H, int W, int G, unsigned seed)
{
    const float eps = 1e-5f;
    const int total = N * C * H * W;
    HalfBuffer b  = make_half(total, seed);
    HalfBuffer gb = make_half(C, seed + 1, 0.2f, 2.0f);
    HalfBuffer bb = make_half(C, seed + 2, -1.5f, 1.5f);

    std::vector<float> ref1b = run_gpu_gn(b.h, gb.h, bb.h, N, C, H, W, G, eps, false, false);
    std::vector<float> gotmb = run_gpu_gn(b.h, gb.h, bb.h, N, C, H, W, G, eps, false, true);

    double sum_abs = 0.0, max_abs = 0.0;
    for (size_t i = 0; i < gotmb.size(); ++i)
    {
        const double d = std::fabs((double)gotmb[i] - (double)ref1b[i]);
        sum_abs += d;
        if (d > max_abs)
        {
            max_abs = d;
        }
    }
    const double mae = sum_abs / static_cast<double>(gotmb.size());
    std::cout << "[test_groupnorm_mb_parity] N=" << N << " C=" << C << " H=" << H
              << " W=" << W << " G=" << G << " MAE=" << mae << " max_abs=" << max_abs << "\n";
    if (mae >= 1e-3 || max_abs >= 4e-3)
    {
        std::cerr << "[test_groupnorm_mb_parity] FAIL: MAE " << mae
                  << " (gate <1e-3) / max_abs " << max_abs << " (gate <4e-3)\n";
        return false;
    }

    // CPU FP32 参照との突合 (mb 単体の正しさ)。
    std::vector<float> ref = cpu_group_norm(b.ref, gb.ref, bb.ref, N, C, H, W, G, eps);
    const int K = (C / G) * H * W;
    return compare(gotmb, ref, K, "test_groupnorm_mb_parity[cpu_ref]");
}

static bool test_groupnorm_mb_parity()
{
    bool ok = true;
    // SDXL UNet 実形状 (resnet norm1/norm2 と conv_norm_out が使う形状)。
    ok = test_groupnorm_mb_parity_one(1, 320,  128, 128, 32, 41001) && ok;
    ok = test_groupnorm_mb_parity_one(1, 640,  64,  64,  32, 41002) && ok;
    ok = test_groupnorm_mb_parity_one(1, 1280, 32,  32,  32, 41003) && ok;
    // CFG batch2 (B=2)。
    ok = test_groupnorm_mb_parity_one(2, 320,  128, 128, 32, 41004) && ok;
    // skip concat 後の大チャネル (up_blocks resnet 入力: C=2560)。
    ok = test_groupnorm_mb_parity_one(1, 2560, 32,  32,  32, 41005) && ok;
    return ok;
}

// ----------------------------------------------------------------
// 7. test_groupnorm_mb_bitexact (G-4k S1a ゲート2):
//    multi-block 版を同一入力で複数回走らせ、出力 FP16 が完全ビット一致すること
//    (atomic 不使用 = 決定的 2 段集約の証明)。memcmp で raw 比較する。
// ----------------------------------------------------------------
static bool test_groupnorm_mb_bitexact()
{
    const int N = 2, C = 320, H = 128, W = 128, G = 32;
    const float eps = 1e-5f;
    const int total = N * C * H * W;
    HalfBuffer b  = make_half(total, 51501);
    HalfBuffer gb = make_half(C, 51502, 0.2f, 2.0f);
    HalfBuffer bb = make_half(C, 51503, -1.5f, 1.5f);

    std::vector<__half> first =
        run_gpu_gn_raw(b.h, gb.h, bb.h, N, C, H, W, G, eps, false, true);
    const int runs = 3;
    for (int r = 1; r < runs; ++r)
    {
        std::vector<__half> again =
            run_gpu_gn_raw(b.h, gb.h, bb.h, N, C, H, W, G, eps, false, true);
        if (std::memcmp(first.data(), again.data(), first.size() * sizeof(__half)) != 0)
        {
            std::cerr << "[test_groupnorm_mb_bitexact] FAIL: run " << r
                      << " is not bit-identical to run 0\n";
            return false;
        }
    }
    std::cout << "[test_groupnorm_mb_bitexact] PASSED (" << runs
              << " runs bit-identical, memcmp over " << total << " halfs)\n";
    return true;
}

// ----------------------------------------------------------------
// ベンチ: GroupNorm はメモリ律速 → 有効帯域 GB/s で報告。
// 有効帯域 ≈ 2 * N*C*H*W * 2byte / ms (read 入力 + write 出力。gamma/beta は無視)。
// ※カーネルは入力を 2 回読む (sum パス + 書き戻しパス) が、PL 指定の帯域定義に従い
//   read 1 + write 1 として計上する。
// warmup5 / iters50 / cudaEvent 中央値。
// ----------------------------------------------------------------
static double bench_one(int N, int C, int H, int W, int G, const char* label, bool mb = false)
{
    const int total = N * C * H * W;
    HalfBuffer b = make_half(total, 777);
    auto gamma = ones_half(C);
    auto beta  = zeros_half(C);
    const float eps = 1e-6f;

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
        if (mb)
        {
            launch_group_norm_mb(d_in, d_gamma, d_beta, d_out, N, C, H, W, G, eps);
        }
        else
        {
            launch_group_norm(d_in, d_gamma, d_beta, d_out, N, C, H, W, G, eps);
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

    // 有効帯域: read + write = 2 * total 要素 * 2 byte。
    const double bytes = 2.0 * static_cast<double>(total) * sizeof(__half);
    const double gbps  = bytes / (static_cast<double>(median_ms) * 1e-3) / 1e9;

    std::cout << "[bench_groupnorm] " << label << (mb ? " [mb]" : " [1b]")
              << " N=" << N << " C=" << C << " H=" << H << " W=" << W << " G=" << G
              << " median=" << median_ms << " ms"
              << " (min=" << times.front() << " max=" << times.back() << ")"
              << " BW=" << gbps << " GB/s (N=" << iters << ")\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_beta));
    return gbps;
}

static void bench_groupnorm()
{
    // VAE 系大 feature map。
    bench_one(1, 128, 512, 512, 32, "vae_fm");
    // UNet 系。
    bench_one(1, 1280, 32, 32, 32, "unet_1280_32");
    bench_one(1, 640, 64, 64, 32, "unet_640_64");
}

// ----------------------------------------------------------------
// 8. bench_groupnorm_mb (G-4k S1a ゲート3): multi-block 版の帯域。
//    SDXL 実形状で 1-block 版 (before) と比較。ゲート = 最大形状 (C=320, 128x128) で
//    multi-block ≥ 300 GB/s (現行 73 GB/s から ≥4x)。小形状は起動レイテンシ支配に
//    なり得るため参考出力のみ。
// ----------------------------------------------------------------
static bool bench_groupnorm_mb()
{
    // 参考: 同形状での 1-block 版 (before)。
    bench_one(1, 320,  128, 128, 32, "unet_320_128");
    const double mb_big = bench_one(1, 320, 128, 128, 32, "unet_320_128", true);
    bench_one(1, 640,  64,  64,  32, "unet_640_64",  true);
    bench_one(1, 1280, 32,  32,  32, "unet_1280_32", true);
    bench_one(2, 320,  128, 128, 32, "unet_320_128_B2", true);

    if (mb_big < 300.0)
    {
        std::cerr << "[bench_groupnorm_mb] FAIL: multi-block BW " << mb_big
                  << " GB/s < 300 GB/s (gate)\n";
        return false;
    }
    std::cout << "[bench_groupnorm_mb] PASSED (unet_320_128 mb BW=" << mb_big
              << " GB/s >= 300)\n";
    return true;
}

#endif // HAVE_CUDA

} // namespace dollama

int main()
{
#ifndef HAVE_CUDA
    std::cout << "[test_groupnorm] [SKIP] HAVE_CUDA 未定義\n";
    return 0;
#else
    bool ok = true;
    ok = dollama::test_groupnorm_known()   && ok;
    ok = dollama::test_groupnorm_random()  && ok;
    ok = dollama::test_groupnorm_affine()  && ok;
    ok = dollama::test_groupnorm_inplace() && ok;
    ok = dollama::test_groupnorm_edges()   && ok;
    // G-4k S1a: multi-block 版 (epilogue 経路)。
    ok = dollama::test_groupnorm_mb_parity()   && ok;
    ok = dollama::test_groupnorm_mb_bitexact() && ok;

    if (!ok)
    {
        std::cerr << "[test_groupnorm] FAILED\n";
        return 1;
    }

    dollama::bench_groupnorm();
    if (!dollama::bench_groupnorm_mb())
    {
        std::cerr << "[test_groupnorm] FAILED (mb bench gate)\n";
        return 1;
    }

    std::cout << "[test_groupnorm] ALL PASSED\n";
    return 0;
#endif
}

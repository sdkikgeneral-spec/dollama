// dense FP16 GEMM 単体テスト + ベンチ (Phase 2 マイルストーン 2-2-1)
// HAVE_CUDA 未定義時は [SKIP] で return 0。
//
// 入力ビット一致: 乱数を FP32 生成 → FP16 へ丸め、その FP16 をデコードした値を
// CPU 参照の入力にも使う (カーネルの数値誤差だけを測る)。
// tol は FP16 相応: atol + rtol*|ref|、rtol は K に応じて緩める。
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

#ifdef HAVE_CUDA
#include <cuda_fp16.h>
#include "kernels/gemm.cuh"
#include "kernels/utils.cuh"
#endif

namespace dollama
{

#ifdef HAVE_CUDA

// ----------------------------------------------------------------
// テスト用ヘルパー
// ----------------------------------------------------------------

// FP32 乱数 → FP16 丸め。デコード値 (FP32) も同時に返すことで CPU 参照と入力ビットを一致させる。
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

// CPU 参照 GEMM (FP32 蓄積、row-major)。
// transB に対応 (transA は本テストでは false 固定)。
static std::vector<float> cpu_gemm(const std::vector<float>& A,
                                   const std::vector<float>& B,
                                   const std::vector<float>& C_in,
                                   int M, int N, int K,
                                   float alpha, float beta,
                                   bool transB)
{
    std::vector<float> out(static_cast<size_t>(M) * N);
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k)
            {
                const float a = A[static_cast<size_t>(i) * K + k];
                const float b = transB ? B[static_cast<size_t>(j) * K + k]
                                       : B[static_cast<size_t>(k) * N + j];
                acc += a * b;
            }
            const float c_old = (beta != 0.0f) ? C_in[static_cast<size_t>(i) * N + j] : 0.0f;
            out[static_cast<size_t>(i) * N + j] = alpha * acc + beta * c_old;
        }
    }
    return out;
}

// device 経由で GEMM を実行し FP16 結果をデコードして返す。
static std::vector<float> run_gpu_gemm(const std::vector<__half>& A,
                                       const std::vector<__half>& B,
                                       const std::vector<__half>& C_in,
                                       int M, int N, int K,
                                       float alpha, float beta,
                                       bool transA, bool transB)
{
    const size_t a_n = static_cast<size_t>(M) * K;
    const size_t b_n = static_cast<size_t>(K) * N;
    const size_t c_n = static_cast<size_t>(M) * N;

    __half* d_A = nullptr;
    __half* d_B = nullptr;
    __half* d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, a_n * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_B, b_n * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_C, c_n * sizeof(__half)));

    CUDA_CHECK(cudaMemcpy(d_A, A.data(), a_n * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), b_n * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, C_in.data(), c_n * sizeof(__half), cudaMemcpyHostToDevice));

    launch_gemm_fp16(d_A, d_B, d_C, M, N, K, alpha, beta, transA, transB);

    std::vector<__half> h_out(c_n);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_C, c_n * sizeof(__half), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    std::vector<float> out(c_n);
    for (size_t i = 0; i < c_n; ++i)
    {
        out[i] = __half2float(h_out[i]);
    }
    return out;
}

// FP16 相応の許容誤差で比較。rtol は K に応じて緩める (蓄積項数に比例して誤差が増えるため)。
static bool compare(const std::vector<float>& got,
                    const std::vector<float>& ref,
                    int K, const char* name)
{
    // FP16 の相対精度は ~2^-10 ≈ 1e-3。蓄積誤差を K のルートで見積もって緩める。
    const float rtol = 1e-2f * std::sqrt(static_cast<float>(K));
    const float atol = 1e-2f * std::sqrt(static_cast<float>(K));
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
// 1. test_identity: B=単位行列で C==A
// ----------------------------------------------------------------
static bool test_identity()
{
    const int M = 16, N = 16, K = 16;
    HalfBuffer A = make_half(M * K, 1);

    // 単位行列 (K x N、K==N)。
    std::vector<__half> B(static_cast<size_t>(K) * N, __float2half(0.0f));
    for (int k = 0; k < K; ++k)
    {
        B[static_cast<size_t>(k) * N + k] = __float2half(1.0f);
    }
    std::vector<__half> C(static_cast<size_t>(M) * N, __float2half(0.0f));

    std::vector<float> got = run_gpu_gemm(A.h, B, C, M, N, K, 1.0f, 0.0f, false, false);

    // 期待値: A をそのままデコードした値。
    return compare(got, A.ref, K, "test_identity");
}

// ----------------------------------------------------------------
// 2. test_small_square: M=N=K=8、CPU 素朴三重ループ参照
// ----------------------------------------------------------------
static bool test_small_square()
{
    const int M = 8, N = 8, K = 8;
    HalfBuffer A = make_half(M * K, 11);
    HalfBuffer B = make_half(K * N, 22);
    std::vector<__half> C(static_cast<size_t>(M) * N, __float2half(0.0f));
    std::vector<float>  C_ref(static_cast<size_t>(M) * N, 0.0f);

    std::vector<float> got = run_gpu_gemm(A.h, B.h, C, M, N, K, 1.0f, 0.0f, false, false);
    std::vector<float> ref = cpu_gemm(A.ref, B.ref, C_ref, M, N, K, 1.0f, 0.0f, false);
    return compare(got, ref, K, "test_small_square");
}

// ----------------------------------------------------------------
// 3. test_rectangular: M=32,N=64,K=48 (添字バグ検出)
// ----------------------------------------------------------------
static bool test_rectangular()
{
    const int M = 32, N = 64, K = 48;
    HalfBuffer A = make_half(M * K, 101);
    HalfBuffer B = make_half(K * N, 202);
    std::vector<__half> C(static_cast<size_t>(M) * N, __float2half(0.0f));
    std::vector<float>  C_ref(static_cast<size_t>(M) * N, 0.0f);

    std::vector<float> got = run_gpu_gemm(A.h, B.h, C, M, N, K, 1.0f, 0.0f, false, false);
    std::vector<float> ref = cpu_gemm(A.ref, B.ref, C_ref, M, N, K, 1.0f, 0.0f, false);
    return compare(got, ref, K, "test_rectangular");
}

// ----------------------------------------------------------------
// 4. test_transB: transB=true、SDXL Linear 代表形状 (x[M,K] @ W^T[K,N], W=[N,K])
// ----------------------------------------------------------------
static bool test_transB()
{
    const int M = 77, N = 320, K = 768; // SDXL 系の代表 Linear (seq=77, in=768, out=320)
    HalfBuffer A = make_half(M * K, 7);
    HalfBuffer B = make_half(N * K, 8); // W は [N, K] row-major
    std::vector<__half> C(static_cast<size_t>(M) * N, __float2half(0.0f));
    std::vector<float>  C_ref(static_cast<size_t>(M) * N, 0.0f);

    std::vector<float> got = run_gpu_gemm(A.h, B.h, C, M, N, K, 1.0f, 0.0f, false, true);
    std::vector<float> ref = cpu_gemm(A.ref, B.ref, C_ref, M, N, K, 1.0f, 0.0f, true);
    return compare(got, ref, K, "test_transB");
}

// ----------------------------------------------------------------
// 5. test_alpha_beta: alpha≠1, beta≠0, C 初期値あり
// ----------------------------------------------------------------
static bool test_alpha_beta()
{
    const int M = 24, N = 24, K = 40;
    const float alpha = 0.5f;
    const float beta  = 2.0f;
    HalfBuffer A = make_half(M * K, 31);
    HalfBuffer B = make_half(K * N, 32);
    HalfBuffer C = make_half(M * N, 33);

    std::vector<float> got = run_gpu_gemm(A.h, B.h, C.h, M, N, K, alpha, beta, false, false);
    std::vector<float> ref = cpu_gemm(A.ref, B.ref, C.ref, M, N, K, alpha, beta, false);
    return compare(got, ref, K, "test_alpha_beta");
}

// ----------------------------------------------------------------
// 6. bench_gemm: FLOPs=2*M*N*K、warmup5 → N回 cudaEvent 中央値 → GFLOPS
// ----------------------------------------------------------------
static void bench_one(int M, int N, int K, bool transB, const char* label)
{
    const size_t a_n = static_cast<size_t>(M) * K;
    const size_t b_n = static_cast<size_t>(K) * N;
    const size_t c_n = static_cast<size_t>(M) * N;

    HalfBuffer A = make_half(static_cast<int>(a_n), 501);
    HalfBuffer B = make_half(static_cast<int>(b_n), 502);

    __half* d_A = nullptr;
    __half* d_B = nullptr;
    __half* d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, a_n * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_B, b_n * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_C, c_n * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_A, A.h.data(), a_n * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.h.data(), b_n * sizeof(__half), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    auto run_once = [&]() -> float
    {
        CUDA_CHECK(cudaEventRecord(start));
        launch_gemm_fp16(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f, false, transB);
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

    const double flops  = 2.0 * static_cast<double>(M) * N * K;
    const double gflops = flops / (static_cast<double>(median_ms) * 1e-3) / 1e9;

    std::cout << "[bench_gemm] " << label
              << " M=" << M << " N=" << N << " K=" << K
              << (transB ? " transB" : "")
              << " median=" << median_ms << " ms"
              << " (min=" << times.front() << " max=" << times.back() << ")"
              << " GFLOPS=" << gflops << " (N=" << iters << ")\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

static void bench_gemm()
{
    // 正方 1024^3
    bench_one(1024, 1024, 1024, false, "square");
    // SDXL 代表縦長 Linear (transB)
    bench_one(4096, 1280, 1280, true, "sdxl_linear_transB");
}

// ================================================================
// G-10k T3: batched GEMM ゲート [H1]〜[H7] + floor (docs/g10k-plan.md §7b)
//
// ★[H6] (分岐カウンタ) が無いと [H1]〜[H5] は全部空撃ちしうる —
//   ラッパが内部でただ直列ループしているだけでも [H1]〜[H5] は全緑になるため、
//   「cuBLAS strided batched 枝を実際に通った」ことを担保するのは [H6] だけである。
//
// ハーネスは既存と同じ非 abort 型 (ok = gate() && ok) で書く。assert や早期 return で
// 打ち切らないこと — 打ち切ると DOLLAMA_GEMM=wmma 走行でケース 1 で止まり、
// 期待値表 (B) のもう 1 行 (cuBLAS 不適格形状) のカウンタが print されず目視判定が成立しない。
// ================================================================

// 直前の launch_gemm_fp16_batched 呼び出しで進んだ分岐カウンタの差分。
// 走行前に stats を取り、後で差を見る (src/tests/test_conv2d.cu:645 と同じ型)。
static GemmBatchedStats g_last_delta;

static void print_delta(const char* label)
{
    std::cout << "[" << label << "] counters delta:"
              << " wrapper_calls=" << g_last_delta.wrapper_calls
              << " cublas_batched_calls=" << g_last_delta.cublas_batched_calls
              << " fallback_loops=" << g_last_delta.fallback_loops
              << " fallback_items=" << g_last_delta.fallback_items << "\n";
}

// batched ラッパを 1 回呼び、C をホストへ戻す。分岐カウンタ差分を g_last_delta に残す。
static std::vector<__half> run_batched(const std::vector<__half>& A,
                                       const std::vector<__half>& B,
                                       const std::vector<__half>& C_in,
                                       const GemmBatchedDesc&     desc)
{
    __half* dA = nullptr;
    __half* dB = nullptr;
    __half* dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, A.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dB, B.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dC, C_in.size() * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(dA, A.data(), A.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), B.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC, C_in.data(), C_in.size() * sizeof(__half), cudaMemcpyHostToDevice));

    const GemmBatchedStats s0 = gemm_batched_stats();
    launch_gemm_fp16_batched(dA, dB, dC, desc);
    CUDA_CHECK(cudaDeviceSynchronize());
    const GemmBatchedStats s1 = gemm_batched_stats();
    g_last_delta.wrapper_calls        = s1.wrapper_calls        - s0.wrapper_calls;
    g_last_delta.cublas_batched_calls = s1.cublas_batched_calls - s0.cublas_batched_calls;
    g_last_delta.fallback_loops       = s1.fallback_loops       - s0.fallback_loops;
    g_last_delta.fallback_items       = s1.fallback_items       - s0.fallback_items;

    std::vector<__half> out(C_in.size());
    CUDA_CHECK(cudaMemcpy(out.data(), dC, out.size() * sizeof(__half), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return out;
}

// 参照 (seq): 同じレイアウトに対して既存 launch_gemm_fp16 を item ごとに N 回呼ぶ。
static std::vector<__half> run_seq(const std::vector<__half>& A,
                                   const std::vector<__half>& B,
                                   const std::vector<__half>& C_in,
                                   const GemmBatchedDesc&     desc)
{
    __half* dA = nullptr;
    __half* dB = nullptr;
    __half* dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, A.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dB, B.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dC, C_in.size() * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(dA, A.data(), A.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), B.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC, C_in.data(), C_in.size() * sizeof(__half), cudaMemcpyHostToDevice));

    for (int n = 0; n < desc.batch; ++n)
    {
        launch_gemm_fp16(dA + desc.stride_a * n,
                         dB + desc.stride_b * n,
                         dC + desc.stride_c * n,
                         desc.M, desc.N, desc.K,
                         desc.alpha, desc.beta,
                         false, desc.trans_b);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__half> out(C_in.size());
    CUDA_CHECK(cudaMemcpy(out.data(), dC, out.size() * sizeof(__half), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return out;
}

// __half バッファの [off, off+count) 区間が bit 一致するか。
static bool half_range_equal(const std::vector<__half>& x, size_t off_x,
                             const std::vector<__half>& y, size_t off_y,
                             size_t count)
{
    return std::memcmp(x.data() + off_x, y.data() + off_y, count * sizeof(__half)) == 0;
}

// 数値レポート ([H5])。E / theta / MAE / max_abs / exact 率を出す。
struct NumReport
{
    double E       = 0.0;
    double theta   = 0.0;
    double mae     = 0.0;
    double max_abs = 0.0;
    double exact   = 0.0; // got と seq が bit 一致した要素の割合
    double denom   = 0.0; // max_j |seq_j| (batch 全 item)
    double S       = 0.0; // 分離上界 (max_i Sum_k |A_ik|) * (max_(k,j) |B_kj|)
};

// |v| の最大 (B の分離上界の第 2 因子に使う)。
static double abs_max_of(const std::vector<float>& v)
{
    double m = 0.0;
    for (size_t i = 0; i < v.size(); ++i)
    {
        m = std::max(m, std::fabs(static_cast<double>(v[i])));
    }
    return m;
}

// max_i Sum_k |A_ik| (全 item を走る。stride_a=0 なら同じ行を見るだけ)。
// コストは O(M*K*batch) の一巡。★ floor の三重ループとは別ループ (2026-09-10 決裁 5・D5-2)。
static double a_rowsum_max_of(const std::vector<float>& Aref, const GemmBatchedDesc& desc)
{
    double best = 0.0;
    for (int n = 0; n < desc.batch; ++n)
    {
        const size_t base = static_cast<size_t>(desc.stride_a) * static_cast<size_t>(n);
        for (int i = 0; i < desc.M; ++i)
        {
            double s = 0.0;
            for (int k = 0; k < desc.K; ++k)
            {
                s += std::fabs(static_cast<double>(Aref[base + static_cast<size_t>(i) * desc.K + k]));
            }
            best = std::max(best, s);
        }
    }
    return best;
}

// got / seq の差分メトリクスと theta を組み立てる。
//   S = (max_i Sum_k |A_ik|) * (max_(k,j) |B_kj|)  … 分離上界 (呼び出し側で算出して渡す)
//   theta = 2^-10 + (2*K*2^-24*S) / max_j |seq_j|
static NumReport numeric_report(const std::vector<__half>& got,
                                const std::vector<__half>& seq,
                                double                     a_rowsum_max,
                                double                     b_absmax,
                                const GemmBatchedDesc&     desc)
{
    NumReport r;

    // E のスコープは batch 全 item の全出力要素 (M*N*batch 個) の平坦な添字。
    double sum_abs = 0.0;
    size_t exact_n = 0;
    for (size_t i = 0; i < got.size(); ++i)
    {
        const double g = static_cast<double>(__half2float(got[i]));
        const double s = static_cast<double>(__half2float(seq[i]));
        const double d = std::fabs(g - s);
        sum_abs += d;
        r.max_abs = std::max(r.max_abs, d);
        r.denom   = std::max(r.denom, std::fabs(s));
        unsigned short gb = 0, sb = 0;
        std::memcpy(&gb, &got[i], sizeof(unsigned short));
        std::memcpy(&sb, &seq[i], sizeof(unsigned short));
        if (gb == sb)
        {
            ++exact_n;
        }
    }
    r.mae   = sum_abs / static_cast<double>(got.size());
    r.exact = static_cast<double>(exact_n) / static_cast<double>(got.size());
    r.E     = (r.denom > 0.0) ? (r.max_abs / r.denom) : 0.0;

    r.S = a_rowsum_max * b_absmax;

    // theta = 2^-10 + (2*K*2^-24*S) / max_j |seq_j|
    const double u32 = 1.0 / 16777216.0; // 2^-24
    const double u16 = 1.0 / 1024.0;     // 2^-10
    r.theta = u16 + (2.0 * static_cast<double>(desc.K) * u32 * r.S)
                        / ((r.denom > 0.0) ? r.denom : 1.0);
    return r;
}

// [H5] の判定 + 報告義務 (E / theta / E/theta / MAE / max_abs / exact 率を必ず print)。
static bool gate_h5(const char* label, const NumReport& r)
{
    const double ratio = (r.theta > 0.0) ? (r.E / r.theta) : 0.0;
    std::cout << "[H5:" << label << "] E=" << r.E
              << " theta=" << r.theta
              << " E/theta=" << ratio
              << " MAE=" << r.mae
              << " max_abs=" << r.max_abs
              << " exact=" << (r.exact * 100.0) << "%"
              << " denom=" << r.denom
              << " S=" << r.S << "\n";
    if (r.denom <= 0.0)
    {
        std::cerr << "[H5:" << label << "] FAILED (denom == 0: seq が全ゼロ)\n";
        return false;
    }
    if (r.E > r.theta)
    {
        std::cerr << "[H5:" << label << "] FAILED (E > theta)\n";
        return false;
    }
    if (r.E > r.theta * 0.5)
    {
        // 緑でも PL へ報告する tripwire (保守的な上限に張り付くのは写像バグの徴候)。
        std::cout << "[H5:" << label << "] WARN: E > theta/2 (report to PL)\n";
    }
    std::cout << "[H5:" << label << "] PASSED\n";
    return true;
}

// floor 用: 指定行だけ CPU 参照を回す純追加ヘルパ (cpu_gemm 本体は無改変)。
static std::vector<float> cpu_gemm_rows(const std::vector<float>& A, size_t a_off,
                                        const std::vector<float>& B, size_t b_off,
                                        const std::vector<float>& C_in, size_t c_off,
                                        int M, int N, int K,
                                        float alpha, float beta, bool transB,
                                        const std::vector<int>& rows)
{
    (void)M;
    std::vector<float> out(rows.size() * static_cast<size_t>(N));
    for (size_t ri = 0; ri < rows.size(); ++ri)
    {
        const int i = rows[ri];
        for (int j = 0; j < N; ++j)
        {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k)
            {
                const float a = A[a_off + static_cast<size_t>(i) * K + k];
                const float b = transB ? B[b_off + static_cast<size_t>(j) * K + k]
                                       : B[b_off + static_cast<size_t>(k) * N + j];
                acc += a * b;
            }
            const float c_old = (beta != 0.0f) ? C_in[c_off + static_cast<size_t>(i) * N + j] : 0.0f;
            out[ri * static_cast<size_t>(N) + j] = alpha * acc + beta * c_old;
        }
    }
    return out;
}

// __half 出力から指定行を取り出して FP32 デコード。
static std::vector<float> extract_rows(const std::vector<__half>& C, size_t c_off,
                                       int N, const std::vector<int>& rows)
{
    std::vector<float> out(rows.size() * static_cast<size_t>(N));
    for (size_t ri = 0; ri < rows.size(); ++ri)
    {
        for (int j = 0; j < N; ++j)
        {
            out[ri * static_cast<size_t>(N) + j] =
                __half2float(C[c_off + static_cast<size_t>(rows[ri]) * N + j]);
        }
    }
    return out;
}

// __half 全体を FP32 デコード。
static std::vector<float> decode_all(const std::vector<__half>& C, size_t off, size_t count)
{
    std::vector<float> out(count);
    for (size_t i = 0; i < count; ++i)
    {
        out[i] = __half2float(C[off + i]);
    }
    return out;
}

// ----------------------------------------------------------------
// ケース 1 + 2: conv 実使用形態 (M=Cout=320 / N=Ncol=16384 (128^2) / K=Cin*3*3=2880 / batch=2)。
//   A = 重み [M,K] を全 n で共有 (stride_a = 0) / B = im2col 列 [K,N] は n ごと / C も n ごと。
//   ★形状は縮小禁止 (docs/g10k-plan.md §7b・2026-09-10 決裁 5)。
//   通すゲート: [H1] [H2] [H3] [H5] [H6] + floor (ケース 2 = 置換ケースは [H2])。
// ----------------------------------------------------------------
static bool case1_conv_form()
{
    const int M = 320;     // Cout
    const int N = 16384;   // Ncol = 128*128
    const int K = 2880;    // Cin*KH*KW = 320*3*3
    const int batch = 2;

    const size_t kn = static_cast<size_t>(K) * N;
    const size_t mn = static_cast<size_t>(M) * N;

    GemmBatchedDesc desc;
    desc.batch    = batch;
    desc.M        = M;
    desc.N        = N;
    desc.K        = K;
    desc.stride_a = 0; // 重み共有 (conv の実使用形態)
    desc.stride_b = static_cast<long long>(kn);
    desc.stride_c = static_cast<long long>(mn);
    desc.alpha    = 1.0f;
    desc.beta     = 0.0f;
    desc.trans_b  = false;

    HalfBuffer A  = make_half(M * K, 9001);
    HalfBuffer BX = make_half(static_cast<int>(kn), 9002);
    HalfBuffer BY = make_half(static_cast<int>(kn), 9003);

    std::vector<__half> Bcat(kn * batch);
    std::vector<__half> Czero(mn * batch, __float2half(0.0f));

    bool ok = true;

    // ---- [H1] 一様性: 全 n に同一の B → 出力の n 断片が memcmp 一致 ----
    {
        std::memcpy(Bcat.data(),      BX.h.data(), kn * sizeof(__half));
        std::memcpy(Bcat.data() + kn, BX.h.data(), kn * sizeof(__half));
        const std::vector<__half> got_u = run_batched(A.h, Bcat, Czero, desc);
        const bool h1 = half_range_equal(got_u, 0, got_u, mn, mn);
        std::cout << "[H1:case1] uniform-B item0 vs item1 memcmp: "
                  << (h1 ? "BIT-EXACT PASSED" : "MISMATCH FAILED") << "\n";
        ok = h1 && ok;
    }

    // ---- 本走行 (X, Y) ----
    std::memcpy(Bcat.data(),      BX.h.data(), kn * sizeof(__half));
    std::memcpy(Bcat.data() + kn, BY.h.data(), kn * sizeof(__half));
    const std::vector<__half> got_xy = run_batched(A.h, Bcat, Czero, desc);

    // ---- [H6] 分岐カウンタ (期待値表 (A) = 既定経路: cuBLAS batched +1 / フォールバック +0) ----
    print_delta("H6:case1");
    {
        const bool h6 = (g_last_delta.wrapper_calls == 1)
                        && (g_last_delta.cublas_batched_calls == 1)
                        && (g_last_delta.fallback_loops == 0)
                        && (g_last_delta.fallback_items == 0);
        std::cout << "[H6:case1] expected (A) wrapper=1 cublas_batched=1 fallback_loops=0"
                  << " fallback_items=0 -> " << (h6 ? "PASSED" : "FAILED") << "\n";
        ok = h6 && ok;
    }

    // ---- [H3] 決定性: 同一設定 3 runs が memcmp 一致 ----
    {
        const std::vector<__half> r2 = run_batched(A.h, Bcat, Czero, desc);
        const std::vector<__half> r3 = run_batched(A.h, Bcat, Czero, desc);
        const bool h3 = half_range_equal(got_xy, 0, r2, 0, mn * batch)
                        && half_range_equal(got_xy, 0, r3, 0, mn * batch);
        std::cout << "[H3:case1] 3 runs memcmp: "
                  << (h3 ? "BIT-EXACT PASSED" : "MISMATCH FAILED") << "\n";
        ok = h3 && ok;
    }

    // ---- [H5] 数値近接: batched vs N 回直列 ----
    std::vector<__half> seq_xy = run_seq(A.h, Bcat, Czero, desc);
    {
        // A は stride 0 なので A 走査は 1 item ぶんを 2 回見るだけ。
        // B の絶対値最大は X / Y の両方 (= 実際にアップロードした全 item) を見る。
        const double arm  = a_rowsum_max_of(A.ref, desc);
        const double bmax = std::max(abs_max_of(BX.ref), abs_max_of(BY.ref));
        const NumReport r = numeric_report(got_xy, seq_xy, arm, bmax, desc);
        ok = gate_h5("case1", r) && ok;
    }

    // ---- floor: batched 出力 vs cpu_gemm を既存 compare(..., K, ...) で判定 ----
    //   走査範囲は事前固定の 4 行 i in {0, 1, M/2, M-1} (M=320 -> 0/1/160/319)・列は全 N。
    //   n=0 / n=1 の両 item に適用する。★実測を見て行集合を変えることは不可。
    {
        const std::vector<int> rows = {0, 1, M / 2, M - 1};
        std::cout << "[floor:case1] rows = {" << rows[0] << ", " << rows[1]
                  << ", " << rows[2] << ", " << rows[3] << "} x all N=" << N << "\n";
        const std::vector<float> dummy_c;
        const std::vector<float> ref0 =
            cpu_gemm_rows(A.ref, 0, BX.ref, 0, dummy_c, 0, M, N, K, 1.0f, 0.0f, false, rows);
        const std::vector<float> got0 = extract_rows(got_xy, 0, N, rows);
        ok = compare(got0, ref0, K, "floor:case1:n0") && ok;

        const std::vector<float> ref1 =
            cpu_gemm_rows(A.ref, 0, BY.ref, 0, dummy_c, 0, M, N, K, 1.0f, 0.0f, false, rows);
        const std::vector<float> got1 = extract_rows(got_xy, mn, N, rows);
        ok = compare(got1, ref1, K, "floor:case1:n1") && ok;
    }

    // seq はもう不要 (メモリを早めに返す)。
    seq_xy.clear();
    seq_xy.shrink_to_fit();

    // ---- ケース 2 = [H2] 置換: (X,Y) と (Y,X) が入れ替えで memcmp 一致 ----
    {
        std::memcpy(Bcat.data(),      BY.h.data(), kn * sizeof(__half));
        std::memcpy(Bcat.data() + kn, BX.h.data(), kn * sizeof(__half));
        const std::vector<__half> got_yx = run_batched(A.h, Bcat, Czero, desc);
        const bool sw0 = half_range_equal(got_xy, 0,  got_yx, mn, mn);
        const bool sw1 = half_range_equal(got_xy, mn, got_yx, 0,  mn);
        std::cout << "[H2:case2] swap memcmp: item0<->item1' " << (sw0 ? "OK" : "NG")
                  << " / item1<->item0' " << (sw1 ? "OK" : "NG") << " -> "
                  << ((sw0 && sw1) ? "BIT-EXACT PASSED" : "MISMATCH FAILED") << "\n";
        ok = (sw0 && sw1) && ok;
    }

    return ok;
}

// ----------------------------------------------------------------
// ケース 3: cuBLAS 不適格形状 (M < 16 = use_cublas が false)。
//   通すゲート: [H4] (batched vs N 回直列が memcmp 一致) [H6] + floor。
// ----------------------------------------------------------------
static bool case3_non_cublas()
{
    const int M = 8, N = 32, K = 32, batch = 2;
    const size_t mk = static_cast<size_t>(M) * K;
    const size_t kn = static_cast<size_t>(K) * N;
    const size_t mn = static_cast<size_t>(M) * N;

    GemmBatchedDesc desc;
    desc.batch    = batch;
    desc.M        = M;
    desc.N        = N;
    desc.K        = K;
    desc.stride_a = static_cast<long long>(mk);
    desc.stride_b = static_cast<long long>(kn);
    desc.stride_c = static_cast<long long>(mn);

    HalfBuffer A = make_half(static_cast<int>(mk * batch), 9101);
    HalfBuffer B = make_half(static_cast<int>(kn * batch), 9102);
    std::vector<__half> Czero(mn * batch, __float2half(0.0f));

    bool ok = true;

    const std::vector<__half> got = run_batched(A.h, B.h, Czero, desc);

    // ---- [H6] 期待値表 (A): cuBLAS batched +0 / フォールバック +1 (item 数 +batch) ----
    print_delta("H6:case3");
    {
        const bool h6 = (g_last_delta.wrapper_calls == 1)
                        && (g_last_delta.cublas_batched_calls == 0)
                        && (g_last_delta.fallback_loops == 1)
                        && (g_last_delta.fallback_items == static_cast<uint64_t>(batch));
        std::cout << "[H6:case3] expected (A) wrapper=1 cublas_batched=0 fallback_loops=1"
                  << " fallback_items=" << batch << " -> " << (h6 ? "PASSED" : "FAILED") << "\n";
        ok = h6 && ok;
    }

    // ---- [H4] batched vs N 回直列が memcmp 完全一致 ----
    {
        const std::vector<__half> seq = run_seq(A.h, B.h, Czero, desc);
        const bool h4 = half_range_equal(got, 0, seq, 0, mn * batch);
        std::cout << "[H4:case3] batched vs seq memcmp: "
                  << (h4 ? "BIT-EXACT PASSED" : "MISMATCH FAILED") << "\n";
        ok = h4 && ok;
    }

    // ---- floor (小形状なので全走査) ----
    {
        const std::vector<float> dummy_c;
        for (int n = 0; n < batch; ++n)
        {
            const std::vector<int> rows_all = [&]
            {
                std::vector<int> r(M);
                for (int i = 0; i < M; ++i)
                {
                    r[i] = i;
                }
                return r;
            }();
            const std::vector<float> ref = cpu_gemm_rows(A.ref, mk * n, B.ref, kn * n,
                                                         dummy_c, 0, M, N, K,
                                                         1.0f, 0.0f, false, rows_all);
            const std::vector<float> g = decode_all(got, mn * n, mn);
            ok = compare(g, ref, K, (n == 0) ? "floor:case3:n0" : "floor:case3:n1") && ok;
        }
    }

    return ok;
}

// ----------------------------------------------------------------
// ケース 4: transB=true の batched 1 ケース (SDXL Linear 形態 = 写像の一般性確認)。
//   A も n ごと (stride_a != 0) にして、A 側 stride 写像も通す。
//   通すゲート: [H1] [H3] [H5] + floor。
// ----------------------------------------------------------------
static bool case4_transb()
{
    const int M = 256, N = 320, K = 768, batch = 2;
    const size_t mk = static_cast<size_t>(M) * K;
    const size_t nk = static_cast<size_t>(N) * K; // B は [N,K] row-major (要素数は K*N と同じ)
    const size_t mn = static_cast<size_t>(M) * N;

    GemmBatchedDesc desc;
    desc.batch    = batch;
    desc.M        = M;
    desc.N        = N;
    desc.K        = K;
    desc.stride_a = static_cast<long long>(mk);
    desc.stride_b = static_cast<long long>(nk);
    desc.stride_c = static_cast<long long>(mn);
    desc.trans_b  = true;

    HalfBuffer A = make_half(static_cast<int>(mk * batch), 9201);
    HalfBuffer B = make_half(static_cast<int>(nk * batch), 9202);
    std::vector<__half> Czero(mn * batch, __float2half(0.0f));

    bool ok = true;

    // ---- [H1] 一様性: A も B も全 n 同一にして n 断片が memcmp 一致することを見る ----
    {
        std::vector<__half> Au(mk * batch);
        std::vector<__half> Bu(nk * batch);
        for (int n = 0; n < batch; ++n)
        {
            std::memcpy(Au.data() + mk * n, A.h.data(), mk * sizeof(__half));
            std::memcpy(Bu.data() + nk * n, B.h.data(), nk * sizeof(__half));
        }
        const std::vector<__half> got_u = run_batched(Au, Bu, Czero, desc);
        const bool h1 = half_range_equal(got_u, 0, got_u, mn, mn);
        std::cout << "[H1:case4] uniform item0 vs item1 memcmp: "
                  << (h1 ? "BIT-EXACT PASSED" : "MISMATCH FAILED") << "\n";
        ok = h1 && ok;
    }

    const std::vector<__half> got = run_batched(A.h, B.h, Czero, desc);
    // ここの print_delta は characterization (参考) であって hard ゲートではない。
    // [H6] の hard 判定は §7b 表 (A) の 2 行 = case1 (conv 実使用形態) / case3 (cuBLAS 不適格形状)
    // のみで、expected (A) -> PASSED を出すのもその 2 本だけ
    // (docs/logs/g10k-t3/t3_default_run.log:37 = case1 / :19 = case3)。
    // case4 の値は「[H6] が case4 でも合否を判定した」と読んではならない。
    //
    // 実数値 (t3_default_run.log:24) は §7b 表 (A)「conv 実使用形態」行と同型
    // (cuBLAS batched +1 / フォールバック +0):
    //   wrapper_calls=1 cublas_batched_calls=1 fallback_loops=0 fallback_items=0
    // = 既定経路 (cuBLAS 有効) で cuBLAS batched 枝を通ったことを示す。
    // ただし表 (A)/(B) に case4/case5 の行は無く、この「同型」は hard 判定ではない。
    // (DOLLAMA_GEMM=wmma 走行では表 (B) どおり反転し、cuBLAS batched +0 / フォールバック +1 になる)
    print_delta("H6:case4(参考)");

    // ---- [H3] 決定性 ----
    {
        const std::vector<__half> r2 = run_batched(A.h, B.h, Czero, desc);
        const std::vector<__half> r3 = run_batched(A.h, B.h, Czero, desc);
        const bool h3 = half_range_equal(got, 0, r2, 0, mn * batch)
                        && half_range_equal(got, 0, r3, 0, mn * batch);
        std::cout << "[H3:case4] 3 runs memcmp: "
                  << (h3 ? "BIT-EXACT PASSED" : "MISMATCH FAILED") << "\n";
        ok = h3 && ok;
    }

    // ---- [H5] ----
    {
        const std::vector<__half> seq = run_seq(A.h, B.h, Czero, desc);
        const NumReport r = numeric_report(got, seq,
                                           a_rowsum_max_of(A.ref, desc),
                                           abs_max_of(B.ref), desc);
        ok = gate_h5("case4", r) && ok;
    }

    // ---- floor (全走査) ----
    {
        const std::vector<float> dummy_c;
        std::vector<int> rows_all(M);
        for (int i = 0; i < M; ++i)
        {
            rows_all[i] = i;
        }
        for (int n = 0; n < batch; ++n)
        {
            const std::vector<float> ref = cpu_gemm_rows(A.ref, mk * n, B.ref, nk * n,
                                                         dummy_c, 0, M, N, K,
                                                         1.0f, 0.0f, true, rows_all);
            const std::vector<float> g = decode_all(got, mn * n, mn);
            ok = compare(g, ref, K, (n == 0) ? "floor:case4:n0" : "floor:case4:n1") && ok;
        }
    }

    return ok;
}

// ----------------------------------------------------------------
// ケース 5: beta != 0 の 1 ケース (C を「読む」形。[H7] の重なり要件と対)。
//   通すゲート: [H5] + floor。
// ----------------------------------------------------------------
static bool case5_beta()
{
    const int M = 64, N = 64, K = 64, batch = 2;
    const float alpha = 0.5f;
    const float beta  = 2.0f;
    const size_t mk = static_cast<size_t>(M) * K;
    const size_t kn = static_cast<size_t>(K) * N;
    const size_t mn = static_cast<size_t>(M) * N;

    GemmBatchedDesc desc;
    desc.batch    = batch;
    desc.M        = M;
    desc.N        = N;
    desc.K        = K;
    desc.stride_a = static_cast<long long>(mk);
    desc.stride_b = static_cast<long long>(kn);
    desc.stride_c = static_cast<long long>(mn);
    desc.alpha    = alpha;
    desc.beta     = beta;

    HalfBuffer A = make_half(static_cast<int>(mk * batch), 9301);
    HalfBuffer B = make_half(static_cast<int>(kn * batch), 9302);
    HalfBuffer C = make_half(static_cast<int>(mn * batch), 9303);

    bool ok = true;

    const std::vector<__half> got = run_batched(A.h, B.h, C.h, desc);
    // case4 と同じく characterization (参考) であって hard ゲートではない。
    // [H6] の hard 判定は §7b 表 (A) の case1 / case3 の 2 行のみ
    // (docs/logs/g10k-t3/t3_default_run.log:37 / :19)。
    //
    // 実数値 (t3_default_run.log:30) は §7b 表 (A)「conv 実使用形態」行と同型
    // (cuBLAS batched +1 / フォールバック +0):
    //   wrapper_calls=1 cublas_batched_calls=1 fallback_loops=0 fallback_items=0
    // = 既定経路 (cuBLAS 有効) で cuBLAS batched 枝を通ったことを示す。
    // ただし表 (A)/(B) に case4/case5 の行は無く、この「同型」は hard 判定ではない。
    // (DOLLAMA_GEMM=wmma 走行では表 (B) どおり反転し、cuBLAS batched +0 / フォールバック +1 になる)
    print_delta("H6:case5(参考)");

    {
        const std::vector<__half> seq = run_seq(A.h, B.h, C.h, desc);
        const NumReport r = numeric_report(got, seq,
                                           a_rowsum_max_of(A.ref, desc),
                                           abs_max_of(B.ref), desc);
        ok = gate_h5("case5", r) && ok;
    }

    {
        std::vector<int> rows_all(M);
        for (int i = 0; i < M; ++i)
        {
            rows_all[i] = i;
        }
        for (int n = 0; n < batch; ++n)
        {
            const std::vector<float> ref = cpu_gemm_rows(A.ref, mk * n, B.ref, kn * n,
                                                         C.ref, mn * n, M, N, K,
                                                         alpha, beta, false, rows_all);
            const std::vector<float> g = decode_all(got, mn * n, mn);
            ok = compare(g, ref, K, (n == 0) ? "floor:case5:n0" : "floor:case5:n1") && ok;
        }
    }

    return ok;
}

// ----------------------------------------------------------------
// [H7] 引数検証 (host-only・GPU 不要)。
//   validator が重なり / 退化した記述子に false (Ok 以外) を返すことを単体で確認する。
//   ★ validator は純関数でポインタを参照外ししないので、ホスト配列のアドレスで足りる。
// ----------------------------------------------------------------
static bool gate_h7()
{
    const int M = 8, N = 8, K = 8, batch = 2;
    const size_t mk = static_cast<size_t>(M) * K;
    const size_t kn = static_cast<size_t>(K) * N;
    const size_t mn = static_cast<size_t>(M) * N;

    // A / B / C を互いに離して置いた 1 本のホストバッファ。
    std::vector<__half> buf((mk + kn + mn) * batch * 2, __float2half(0.0f));
    const __half* pA = buf.data();
    const __half* pB = buf.data() + mk * batch;
    __half*       pC = buf.data() + (mk + kn) * batch;

    GemmBatchedDesc base;
    base.batch    = batch;
    base.M        = M;
    base.N        = N;
    base.K        = K;
    base.stride_a = static_cast<long long>(mk);
    base.stride_b = static_cast<long long>(kn);
    base.stride_c = static_cast<long long>(mn);

    struct Sub
    {
        const char*           label;
        GemmBatchedDesc       d;
        const __half*         a;
        const __half*         b;
        __half*               c;
        GemmBatchedValidation want;
    };

    std::vector<Sub> subs;

    subs.push_back({"valid", base, pA, pB, pC, GemmBatchedValidation::Ok});

    // A の stride 0 (conv の重み共有形態) は合法として明示的に許可される。
    {
        GemmBatchedDesc d = base;
        d.stride_a = 0;
        subs.push_back({"stride_a=0 (weight share)", d, pA, pB, pC, GemmBatchedValidation::Ok});
    }
    // A と B が重なるのは許可 (禁止するのは C との重なりのみ)。
    {
        GemmBatchedDesc d = base;
        d.stride_a = 0;
        subs.push_back({"A overlaps B", d, pB, pB, pC, GemmBatchedValidation::Ok});
    }
    {
        GemmBatchedDesc d = base;
        d.batch = 0;
        subs.push_back({"batch=0", d, pA, pB, pC, GemmBatchedValidation::BadBatch});
    }
    {
        GemmBatchedDesc d = base;
        d.M = 0;
        subs.push_back({"M=0", d, pA, pB, pC, GemmBatchedValidation::BadDims});
    }
    {
        GemmBatchedDesc d = base;
        d.K = -1;
        subs.push_back({"K=-1", d, pA, pB, pC, GemmBatchedValidation::BadDims});
    }
    {
        subs.push_back({"A=null", base, nullptr, pB, pC, GemmBatchedValidation::NullPointer});
    }
    {
        subs.push_back({"C=null", base, pA, pB, nullptr, GemmBatchedValidation::NullPointer});
    }
    {
        // C の item 同士が重なる (stride_c < M*N)。
        GemmBatchedDesc d = base;
        d.stride_c = static_cast<long long>(mn) - 1;
        subs.push_back({"C items overlap (stride_c<M*N)", d, pA, pB, pC,
                        GemmBatchedValidation::OverlapC});
    }
    {
        // C の item 同士が完全に重なる。
        GemmBatchedDesc d = base;
        d.stride_c = 0;
        subs.push_back({"C items overlap (stride_c=0)", d, pA, pB, pC,
                        GemmBatchedValidation::OverlapC});
    }
    {
        // C が A の区間と重なる。
        subs.push_back({"C overlaps A", base, pA, pB, buf.data(),
                        GemmBatchedValidation::OverlapCA});
    }
    {
        // C が B の区間と重なる。
        subs.push_back({"C overlaps B", base, pA, pB, buf.data() + mk * batch,
                        GemmBatchedValidation::OverlapCB});
    }

    bool ok = true;
    for (const Sub& s : subs)
    {
        const GemmBatchedValidation got = gemm_batched_validate(s.d, s.a, s.b, s.c);
        const bool hit = (got == s.want);
        std::cout << "[H7] " << s.label
                  << ": got=" << gemm_batched_validation_str(got)
                  << " want=" << gemm_batched_validation_str(s.want)
                  << " -> " << (hit ? "PASSED" : "FAILED") << "\n";
        ok = hit && ok;
    }
    return ok;
}

static bool test_gemm_batched_gates()
{
    bool ok = true;
    ok = gate_h7()        && ok; // host-only を先に (GPU 不要)
    ok = case3_non_cublas() && ok;
    ok = case4_transb()     && ok;
    ok = case5_beta()       && ok;
    ok = case1_conv_form()  && ok; // 最大形状は最後に
    std::cout << "[batched_gates] " << (ok ? "ALL PASSED" : "FAILED") << "\n";
    return ok;
}

#endif // HAVE_CUDA

} // namespace dollama

int main()
{
#ifndef HAVE_CUDA
    std::cout << "[test_gemm] [SKIP] HAVE_CUDA 未定義\n";
    return 0;
#else
    bool ok = true;
    ok = dollama::test_identity()     && ok;
    ok = dollama::test_small_square() && ok;
    ok = dollama::test_rectangular()  && ok;
    ok = dollama::test_transB()       && ok;
    ok = dollama::test_alpha_beta()   && ok;

    // G-10k T3: batched GEMM ゲート [H1]〜[H7] + floor。
    ok = dollama::test_gemm_batched_gates() && ok;

    if (!ok)
    {
        std::cerr << "[test_gemm] FAILED\n";
        return 1;
    }

    dollama::bench_gemm();

    std::cout << "[test_gemm] ALL PASSED\n";
    return 0;
#endif
}

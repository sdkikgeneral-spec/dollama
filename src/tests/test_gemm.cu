// dense FP16 GEMM 単体テスト + ベンチ (Phase 2 マイルストーン 2-2-1)
// HAVE_CUDA 未定義時は [SKIP] で return 0。
//
// 入力ビット一致: 乱数を FP32 生成 → FP16 へ丸め、その FP16 をデコードした値を
// CPU 参照の入力にも使う (カーネルの数値誤差だけを測る)。
// tol は FP16 相応: atol + rtol*|ref|、rtol は K に応じて緩める。
#include <algorithm>
#include <cmath>
#include <cstdint>
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

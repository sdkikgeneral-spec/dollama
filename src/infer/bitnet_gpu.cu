// BitNet dense (FP32) GPU 推論カーネル群 — Phase 4 #6-GPU / T1。
//
// CPU 版 src/infer/bitnet.hpp::BitNetDenseInfer と数値一致させる目的の
// RTX5080 (Blackwell / sm_120) forward 本体。CUDA Runtime API のみ。
//
// ──────────────────────────────────────────────────────────────────
// 最重要制約: TF32 厳禁・純 FP32 で通す
//   golden 許容 (max_abs_err < 1e-3 / corr ≥ 0.99999) は厳しく、lm_head
//   (K=512・4999 出力) の積和を TF32 (仮数10bit) で割るリスクがある。
//   よって本 TU 内に LM ローカルの cuBLAS ハンドルを持ち、CUBLAS_COMPUTE_32F
//   固定の純 FP32 GEMM ラッパ (lm_gemm_f32) を用意する。共有 src/kernels/gemm.cu
//   (env DOLLAMA_VAE_GEMM 依存で TF32 既定) は一切呼ばず・無改変のまま隔離する。
//
// ──────────────────────────────────────────────────────────────────
// CPU 版との数値一致方針:
//   CPU 版は linear / lm_head / RMSNorm / softmax / attention dot を全て
//   double 蓄積している。GPU 側もこれに合わせ、リダクションを伴う自前カーネル
//   (RMSNorm / causal SDPA) はカーネル内で double 蓄積する (seq≤64・D=512 と
//   小規模ゆえ FP64 演算でも速度は問題にならない)。
//   Linear / lm_head の巨大積和は cuBLAS CUBLAS_COMPUTE_32F に委譲する
//   (内部はブロック分割の FP32 蓄積で、素朴な単精度逐次ループより誤差が小さい)。
//   RoPE / SwiGLU の要素ごとの超越関数も double で評価して CPU と桁を揃える。
//
// 重みレイアウト [out,in] row-major: y[o] = Σ_i w[o*in+i]*x[i] (transB=true 相当)。
//
// ヘッダ内/.cu の __global__ は static (LNK2005 回避)。

#include "infer/bitnet_gpu.cuh"
#include "kernels/utils.cuh"

#include <cublas_v2.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace dollama
{


// ====================================================================
// cuBLAS エラーチェック
// ====================================================================
#define LM_CUBLAS_CHECK(call)                                               \
    do                                                                      \
    {                                                                       \
        cublasStatus_t st = (call);                                         \
        if (st != CUBLAS_STATUS_SUCCESS)                                    \
        {                                                                   \
            throw std::runtime_error(                                       \
                std::string("LM cuBLAS error ") + std::to_string((int)st)   \
                + " at " __FILE__ ":" + std::to_string(__LINE__));          \
        }                                                                   \
    } while (0)

// ====================================================================
// LM ローカル純 FP32 GEMM ラッパ (方針A)
//   C[M,N] = op(A)[M,K] @ op(B)[K,N] (alpha=1, beta=0)。すべて FP32 device ptr。
//   compute type は CUBLAS_COMPUTE_32F 固定 (TF32 厳禁)・algo は CUBLAS_GEMM_DEFAULT。
//
//   row-major → cuBLAS の col-major 変換 (gemm.cu の gemm_cublas_f32 と同手順):
//     row-major C[M,N] = col-major C^T[N,M]。col-major で C^T = op(B)^T @ op(A)^T を
//     計算する。A は logical row-major [M,K] (ld=K) = col-major [K,M] = A^T なので
//     opA'=N・ldA'=K で固定。B 側:
//       transB=false: B row-major [K,N] (ld=N) = col-major [N,K] = op(B)^T → opB'=N, ldB'=N
//       transB=true : B row-major [N,K] (ld=K)。欲しい op(B)=[K,N] へ転置 → opB'=T, ldB'=K
//   本 LM では Linear は重み [out,in] が B・活性 [S,in] が A・transB=true で
//     y[S,out] = x[S,in] @ W[out,in]^T を表現する。
// ====================================================================
static void lm_gemm_f32(cublasHandle_t h,
                        const float*   d_A,   // [M,K] row-major (活性 [S,in])
                        const float*   d_B,   // 重み (row-major [out,in] 等)
                        float*         d_C,   // [M,N] row-major (出力 [S,out])
                        int            M,
                        int            N,
                        int            K,
                        bool           transB)
{
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    const cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
    const int ldB = transB ? K : N; // transB=true: B[N,K] ld=K / false: B[K,N] ld=N
    const int ldA = K;              // A row-major [M,K] = col-major [K,M] ld=K
    const int ldC = N;              // C row-major [M,N] = col-major [N,M] ld=N

    // C^T[N,M] = op(B)^T @ op(A)^T。純 FP32 (CUBLAS_COMPUTE_32F)・default algo。
    LM_CUBLAS_CHECK(cublasGemmEx(h,
                                 opB, CUBLAS_OP_N,
                                 N, M, K,
                                 &alpha,
                                 d_B, CUDA_R_32F, ldB,
                                 d_A, CUDA_R_32F, ldA,
                                 &beta,
                                 d_C, CUDA_R_32F, ldC,
                                 CUBLAS_COMPUTE_32F,
                                 CUBLAS_GEMM_DEFAULT));
}

// ====================================================================
// LM 専用 FP32 static __global__ カーネル群
//   リダクションを伴うものは double 蓄積で CPU 版 (double) と桁を揃える。
// ====================================================================

// ── embedding gather ────────────────────────────────────────────────
// token id 列 d_tokens[S] から embed[id*D] を hidden h[S,D] へ展開する。
//   1 thread = 1 要素 (s,d)。
static __global__ void embed_gather_kernel(const float* __restrict__ embed,
                                           const int*   __restrict__ tokens,
                                           float*       __restrict__ h,
                                           int S, int D)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= S * D)
    {
        return;
    }
    const int s = idx / D;
    const int d = idx % D;
    const int tok = tokens[s];
    h[idx] = embed[static_cast<long long>(tok) * D + d];
}

// ── RMSNorm (eps 1e-5・double 蓄積) ─────────────────────────────────
// out[s,d] = x[s,d] / sqrt(mean_d(x^2) + eps) * weight[d]。
//   1 block = 1 行 (位置 s)。block 内で x^2 の総和を double で並列リダクション。
static __global__ void rmsnorm_kernel(const float* __restrict__ x,
                                      const float* __restrict__ weight,
                                      float*       __restrict__ out,
                                      int S, int D, double eps)
{
    const int s = blockIdx.x;
    if (s >= S)
    {
        return;
    }
    const float* xr = x + static_cast<long long>(s) * D;
    float*       or_ = out + static_cast<long long>(s) * D;

    extern __shared__ double sdata[];
    const int tid = threadIdx.x;

    // 各 thread が担当要素の x^2 を double で部分和。
    double local = 0.0;
    for (int d = tid; d < D; d += blockDim.x)
    {
        const double v = static_cast<double>(xr[d]);
        local += v * v;
    }
    sdata[tid] = local;
    __syncthreads();

    // block 内 double リダクション。
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    const double ms  = sdata[0] / static_cast<double>(D);
    const double inv = 1.0 / sqrt(ms + eps);

    for (int d = tid; d < D; d += blockDim.x)
    {
        or_[d] = static_cast<float>(static_cast<double>(xr[d]) * inv
                                    * static_cast<double>(weight[d]));
    }
}

// ── RoPE (GPT-NeoX ペアリング・double 評価) ─────────────────────────
// Q/K [S, D_MODEL] の各 head ([HEAD_DIM]) に位置 s の回転を適用する (in-place)。
//   (i, i+half) ペアを回す。1 thread = 1 ペア (s, head, i)。
static __global__ void rope_kernel(float* __restrict__ vec,
                                   int S, int n_heads, int head_dim,
                                   double rope_base)
{
    const int half = head_dim >> 1;
    const int total = S * n_heads * half;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total)
    {
        return;
    }
    const int i   = idx % half;
    const int tmp = idx / half;
    const int hd  = tmp % n_heads;
    const int s   = tmp / n_heads;

    const int D = n_heads * head_dim;
    float* base = vec + static_cast<long long>(s) * D + hd * head_dim;

    const double freq = 1.0 / pow(rope_base,
        static_cast<double>(2 * i) / static_cast<double>(head_dim));
    const double angle = static_cast<double>(s) * freq;
    const double cs = cos(angle);
    const double sn = sin(angle);
    const double a = static_cast<double>(base[i]);
    const double b = static_cast<double>(base[i + half]);
    base[i]        = static_cast<float>(a * cs - b * sn);
    base[i + half] = static_cast<float>(a * sn + b * cs);
}

// ── causal scaled-dot-product attention (double softmax) ────────────
// Q/K/V [S, D_MODEL] (head 連結) から attn_out[S, D_MODEL] を計算する。
//   scale = 1/sqrt(head_dim)・softmax は max 減算の数値安定版・j<=s の causal。
//   1 block = (head hd, query 位置 s)。block 内 thread が key j と dim d を分担。
//   CPU 版 (attention_block 391-436) と同手順・double 蓄積。seq≤64。
static __global__ void causal_sdpa_kernel(const float* __restrict__ Q,
                                          const float* __restrict__ K,
                                          const float* __restrict__ V,
                                          float*       __restrict__ attn_out,
                                          int S, int n_heads, int head_dim)
{
    const int hd = blockIdx.x;   // head
    const int s  = blockIdx.y;   // query 位置
    if (hd >= n_heads || s >= S)
    {
        return;
    }
    const int D   = n_heads * head_dim;
    const int off = hd * head_dim;
    const double scale = 1.0 / sqrt(static_cast<double>(head_dim));

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    // shared: scores[j] (double, j=0..s) と head_dim ぶんの出力アキュムレータ。
    extern __shared__ double smem[];
    double* scores = smem;                 // [S]
    double* outacc = smem + S;             // [head_dim]

    const float* qrow = Q + static_cast<long long>(s) * D + off;

    // 1) score[j] = scale * dot(q_s, k_j)  (j<=s)。各 thread が j を分担し double dot。
    for (int j = tid; j <= s; j += nthreads)
    {
        const float* krow = K + static_cast<long long>(j) * D + off;
        double dot = 0.0;
        for (int d = 0; d < head_dim; ++d)
        {
            dot += static_cast<double>(qrow[d]) * static_cast<double>(krow[d]);
        }
        scores[j] = dot * scale;
    }
    __syncthreads();

    // 2) max 減算 softmax (thread 0 が逐次・CPU と同じ走査順)。S≤64 なので軽い。
    if (tid == 0)
    {
        double maxv = -1e300;
        for (int j = 0; j <= s; ++j)
        {
            if (scores[j] > maxv)
            {
                maxv = scores[j];
            }
        }
        double sum = 0.0;
        for (int j = 0; j <= s; ++j)
        {
            const double e = exp(scores[j] - maxv);
            scores[j] = e;
            sum += e;
        }
        const double inv = 1.0 / sum;
        for (int j = 0; j <= s; ++j)
        {
            scores[j] *= inv;
        }
    }
    // 出力アキュムレータを 0 初期化。
    for (int d = tid; d < head_dim; d += nthreads)
    {
        outacc[d] = 0.0;
    }
    __syncthreads();

    // 3) out[d] = Σ_j w[j] * V[j,d] (double 蓄積)。各 thread が d を分担。
    for (int d = tid; d < head_dim; d += nthreads)
    {
        double acc = 0.0;
        for (int j = 0; j <= s; ++j)
        {
            const float* vrow = V + static_cast<long long>(j) * D + off;
            acc += scores[j] * static_cast<double>(vrow[d]);
        }
        outacc[d] = acc;
    }
    __syncthreads();

    float* outv = attn_out + static_cast<long long>(s) * D + off;
    for (int d = tid; d < head_dim; d += nthreads)
    {
        outv[d] = static_cast<float>(outacc[d]);
    }
}

// ── SwiGLU: inter[f] = silu(g[f]) * u[f] (double 評価) ───────────────
// g, u は [S, FFN_DIM]。silu(x) = x / (1+exp(-x))。1 thread = 1 要素。
static __global__ void swiglu_kernel(const float* __restrict__ g,
                                     const float* __restrict__ u,
                                     float*       __restrict__ inter,
                                     int total)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total)
    {
        return;
    }
    const double gv   = static_cast<double>(g[idx]);
    const double silu = gv / (1.0 + exp(-gv));
    inter[idx] = static_cast<float>(silu * static_cast<double>(u[idx]));
}

// ── residual add: h[i] += y[i] ──────────────────────────────────────
static __global__ void residual_add_kernel(float* __restrict__ h,
                                           const float* __restrict__ y,
                                           int total)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total)
    {
        h[idx] += y[idx];
    }
}

// ====================================================================
// device 経路の本体 (T2 の class BitNetGpuInfer から呼ばれる内部関数)
// ====================================================================

// ── kernel launch ヘルパ ────────────────────────────────────────────
static constexpr int THREADS = 256;

static void launch_attention(cublasHandle_t handle,
                             const BitNetGpuLayerDev& L,
                             BitNetGpuScratch& sc, int S)
{
    using namespace bitnet_gpu_cfg;
    const int D = D_MODEL;
    const int SD = S * D;

    // 1) pre-norm: d_x <- RMSNorm(d_h, attn_norm)。1 block = 1 行。
    rmsnorm_kernel<<<S, THREADS, THREADS * sizeof(double)>>>(
        sc.d_h, L.attn_norm, sc.d_x, S, D, RMS_EPS);
    CUDA_CHECK_KERNEL();

    // 2) Q/K/V = x @ W^T  (W [out=D, in=D] row-major・transB=true)。
    lm_gemm_f32(handle, sc.d_x, L.wq, sc.d_Q, S, D, D, /*transB=*/true);
    lm_gemm_f32(handle, sc.d_x, L.wk, sc.d_K, S, D, D, /*transB=*/true);
    lm_gemm_f32(handle, sc.d_x, L.wv, sc.d_V, S, D, D, /*transB=*/true);

    // 3) RoPE を Q/K に適用 (per head)。
    {
        const int total = S * N_HEADS * (HEAD_DIM >> 1);
        const int blocks = ceil_div(total, THREADS);
        rope_kernel<<<blocks, THREADS>>>(sc.d_Q, S, N_HEADS, HEAD_DIM, ROPE_BASE);
        CUDA_CHECK_KERNEL();
        rope_kernel<<<blocks, THREADS>>>(sc.d_K, S, N_HEADS, HEAD_DIM, ROPE_BASE);
        CUDA_CHECK_KERNEL();
    }

    // 4) causal SDPA。1 block = (head, query 位置)。
    {
        const dim3 grid(N_HEADS, S);
        const int sdpa_threads = HEAD_DIM; // 64
        const size_t shmem = (static_cast<size_t>(S) + HEAD_DIM) * sizeof(double);
        causal_sdpa_kernel<<<grid, sdpa_threads, shmem>>>(
            sc.d_Q, sc.d_K, sc.d_V, sc.d_attn, S, N_HEADS, HEAD_DIM);
        CUDA_CHECK_KERNEL();
    }

    // 5) 出力 projection: proj = attn @ Wo^T、residual d_h += proj。
    lm_gemm_f32(handle, sc.d_attn, L.wo, sc.d_proj, S, D, D, /*transB=*/true);
    {
        const int blocks = ceil_div(SD, THREADS);
        residual_add_kernel<<<blocks, THREADS>>>(sc.d_h, sc.d_proj, SD);
        CUDA_CHECK_KERNEL();
    }
}

static void launch_ffn(cublasHandle_t handle,
                       const BitNetGpuLayerDev& L,
                       BitNetGpuScratch& sc, int S)
{
    using namespace bitnet_gpu_cfg;
    const int D = D_MODEL;
    const int F = FFN_DIM;
    const int SD = S * D;
    const int SF = S * F;

    // pre-norm: d_x <- RMSNorm(d_h, ffn_norm)。
    rmsnorm_kernel<<<S, THREADS, THREADS * sizeof(double)>>>(
        sc.d_h, L.ffn_norm, sc.d_x, S, D, RMS_EPS);
    CUDA_CHECK_KERNEL();

    // g = x @ Wgate^T [S,F], u = x @ Wup^T [S,F] (W [out=F, in=D]・transB=true)。
    lm_gemm_f32(handle, sc.d_x, L.w_gate, sc.d_g, S, F, D, /*transB=*/true);
    lm_gemm_f32(handle, sc.d_x, L.w_up,   sc.d_u, S, F, D, /*transB=*/true);

    // inter = silu(g) * u。
    {
        const int blocks = ceil_div(SF, THREADS);
        swiglu_kernel<<<blocks, THREADS>>>(sc.d_g, sc.d_u, sc.d_inter, SF);
        CUDA_CHECK_KERNEL();
    }

    // y = inter @ Wdown^T [S,D] (W [out=D, in=F]・transB=true)、residual d_h += y。
    lm_gemm_f32(handle, sc.d_inter, L.w_down, sc.d_proj, S, D, F, /*transB=*/true);
    {
        const int blocks = ceil_div(SD, THREADS);
        residual_add_kernel<<<blocks, THREADS>>>(sc.d_h, sc.d_proj, SD);
        CUDA_CHECK_KERNEL();
    }
}

// ── forward device 本体 ─────────────────────────────────────────────
// 入力 host tokens [S] → 出力 host logits [S * VOCAB_SIZE]。
//   重み (w) と作業バッファ (sc) は T2 が確保済みで渡す。host↔device コピーは
//   この関数内で行う (T2 のクラスが API 境界でさらに隠蔽する)。
//   cublasHandle は T2 が LM ローカルに生成して渡す (純 FP32 を担保)。
void bitnet_gpu_forward(cublasHandle_t handle,
                        const BitNetGpuWeightsDev& w,
                        BitNetGpuScratch& sc,
                        const int* host_tokens, int S,
                        float* host_logits_out)
{
    using namespace bitnet_gpu_cfg;
    const int D = D_MODEL;

    // 1) tokens H2D → embedding gather で d_h を初期化。
    CUDA_CHECK(cudaMemcpy(sc.d_tokens, host_tokens,
                          static_cast<size_t>(S) * sizeof(int),
                          cudaMemcpyHostToDevice));
    {
        const int total = S * D;
        const int blocks = ceil_div(total, THREADS);
        embed_gather_kernel<<<blocks, THREADS>>>(w.embed, sc.d_tokens, sc.d_h, S, D);
        CUDA_CHECK_KERNEL();
    }

    // 2) 各層 attention + ffn。
    for (int li = 0; li < N_LAYERS; ++li)
    {
        launch_attention(handle, w.layers[static_cast<size_t>(li)], sc, S);
        launch_ffn(handle, w.layers[static_cast<size_t>(li)], sc, S);
    }

    // 3) 最終 RMSNorm (d_x へ)。
    rmsnorm_kernel<<<S, THREADS, THREADS * sizeof(double)>>>(
        sc.d_h, w.final_norm, sc.d_x, S, D, RMS_EPS);
    CUDA_CHECK_KERNEL();

    // 4) tied lm_head: logits[S,VOCAB] = normed @ embed^T
    //    (embed [out=VOCAB, in=D] row-major・transB=true)。純 FP32。
    lm_gemm_f32(handle, sc.d_x, w.embed, sc.d_logits, S, VOCAB_SIZE, D, /*transB=*/true);

    // 5) logits D2H。
    CUDA_CHECK(cudaMemcpy(host_logits_out, sc.d_logits,
                          static_cast<size_t>(S) * VOCAB_SIZE * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace dollama

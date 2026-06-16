// dense FP16 GEMM カーネル実装 (Phase 2 マイルストーン 2-2-1)
// 対象: RTX5080 (Blackwell / sm_120) / CUDA Runtime API のみ
//
// 設計判断 (2-2-2 以降でも再利用):
//   - レイアウトは row-major 固定。呼び出し側は常に row-major で考える。
//   - accumulator は必ず FP32 (FP16 蓄積は K が大きいと誤差が爆発するため)。
//   - transB=true 経路 (SDXL Linear `x @ W^T`) を本実装の主役として用意。
//   - 実装は naive (1-thread-1-output) + shared-memory タイリングの 2 段構え。
//     既定はタイリング版を使う。
#include "kernels/gemm.cuh"
#include "kernels/utils.cuh"

#include <cuda_fp16.h>

namespace dollama
{

// タイル一辺 (16x16 = 256 threads/block)。共有メモリ 2*16*16*2byte = 1KB。
static constexpr int TILE = 16;

// ----------------------------------------------------------------
// op(A), op(B) の要素アクセスを transpose フラグで吸収するデバイス補助。
// row-major 前提。
//   transA=false: A は [M,K], A[i*K + k]
//   transA=true : A は [K,M], A[k*M + i]  (op(A)[i,k] = A[k,i])
//   transB=false: B は [K,N], B[k*N + j]
//   transB=true : B は [N,K], B[j*K + k]  (op(B)[k,j] = B[j,k])
// ----------------------------------------------------------------
__device__ __forceinline__ float load_a(const __half* A, int i, int k, int M, int K, bool transA)
{
    const int idx = transA ? (k * M + i) : (i * K + k);
    return __half2float(A[idx]);
}

__device__ __forceinline__ float load_b(const __half* B, int k, int j, int K, int N, bool transB)
{
    const int idx = transB ? (j * K + k) : (k * N + j);
    return __half2float(B[idx]);
}

// ----------------------------------------------------------------
// shared-memory タイリング版 GEMM。
// 各ブロックが C の [TILE x TILE] サブブロックを担当する。
// ----------------------------------------------------------------
__global__ void gemm_fp16_tiled(const __half* A,
                                const __half* B,
                                __half*       C,
                                int           M,
                                int           N,
                                int           K,
                                float         alpha,
                                float         beta,
                                bool          transA,
                                bool          transB)
{
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    const int row = blockIdx.y * TILE + threadIdx.y; // C の行 i
    const int col = blockIdx.x * TILE + threadIdx.x; // C の列 j

    float acc = 0.0f;

    // K 方向をタイル幅で分割して進める。
    const int num_tiles = ceil_div(K, TILE);
    for (int t = 0; t < num_tiles; ++t)
    {
        const int a_k = t * TILE + threadIdx.x; // sA に読む A の k
        const int b_k = t * TILE + threadIdx.y; // sB に読む B の k

        // 範囲外は 0 で埋める (端のタイルでの領域外読み出し防止)。
        sA[threadIdx.y][threadIdx.x] =
            (row < M && a_k < K) ? load_a(A, row, a_k, M, K, transA) : 0.0f;
        sB[threadIdx.y][threadIdx.x] =
            (b_k < K && col < N) ? load_b(B, b_k, col, K, N, transB) : 0.0f;

        __syncthreads();

        // FP32 で蓄積。
#pragma unroll
        for (int kk = 0; kk < TILE; ++kk)
        {
            acc += sA[threadIdx.y][kk] * sB[kk][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N)
    {
        const int c_idx = row * N + col;
        const float c_old = (beta != 0.0f) ? __half2float(C[c_idx]) : 0.0f;
        C[c_idx] = __float2half(alpha * acc + beta * c_old);
    }
}

// ----------------------------------------------------------------
// naive 版 (1-thread-1-output)。正しさの基準・小行列のフォールバック用に残す。
// ----------------------------------------------------------------
__global__ void gemm_fp16_naive(const __half* A,
                                const __half* B,
                                __half*       C,
                                int           M,
                                int           N,
                                int           K,
                                float         alpha,
                                float         beta,
                                bool          transA,
                                bool          transB)
{
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N)
    {
        return;
    }

    float acc = 0.0f;
    for (int k = 0; k < K; ++k)
    {
        acc += load_a(A, row, k, M, K, transA) * load_b(B, k, col, K, N, transB);
    }

    const int c_idx = row * N + col;
    const float c_old = (beta != 0.0f) ? __half2float(C[c_idx]) : 0.0f;
    C[c_idx] = __float2half(alpha * acc + beta * c_old);
}

// ----------------------------------------------------------------
// ホストラッパー。デバイスポインタを受け取りタイリングカーネルを起動する。
// ----------------------------------------------------------------
void launch_gemm_fp16(const __half* d_A,
                      const __half* d_B,
                      __half*       d_C,
                      int           M,
                      int           N,
                      int           K,
                      float         alpha,
                      float         beta,
                      bool          transA,
                      bool          transB)
{
    const dim3 block(TILE, TILE);
    const dim3 grid(ceil_div(N, TILE), ceil_div(M, TILE));
    gemm_fp16_tiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K, alpha, beta, transA, transB);
    CUDA_CHECK_KERNEL();
}

} // namespace dollama

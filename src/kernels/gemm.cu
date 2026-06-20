// dense FP16 GEMM カーネル実装 (Phase 2 マイルストーン 2-2-1 / 2-6 S2)
// 対象: RTX5080 (Blackwell / sm_120) / CUDA Runtime API のみ
//
// 設計判断 (2-2-2 以降でも再利用):
//   - レイアウトは row-major 固定。呼び出し側は常に row-major で考える。
//   - accumulator は必ず FP32 (FP16 蓄積は K が大きいと誤差が爆発するため)。
//   - transB=true 経路 (SDXL Linear `x @ W^T`) を本実装の主役として用意。
//   - 実装は 3 段構え:
//       (a) naive (1-thread-1-output)  … 正しさの基準・小行列フォールバック
//       (b) shared-memory タイリング   … 自作版 (Tensor Core 不可環境のフォールバック)
//       (c) Tensor Core (wmma) m16n16k16, FP32 蓄積  … 既定の高速経路 (S2 で追加)
//     自作版 (b)(naive(a)) は哲学として残し、(c) を別カーネルとして追加、launch
//     側で形状により自動選択する。
//
// S2 (Tensor Core 化) のメモ:
//   - wmma fragment は matrix_a/matrix_b が __half、accumulator が float。
//   - transB の col_major トリック:
//       B は [N,K] row-major (B[j*K+k])。これを「shape [K,N]・leading dim K の
//       col_major 行列」とみなすと、その (k,j) 要素は j*K+k = B[j,k] であり、
//       ちょうど op(B)=B^T (= [K,N]) になる。よって matrix_b を col_major・ldb=K で
//       load_matrix_sync すれば transB が自然に表現できる (転置コピー不要)。
//     transB=false は B[k*N+j] = 標準の row_major・ldb=N。
//   - 端数 (M/N/K が 16 非倍数): タイル境界をまたぐ部分は共有メモリへガード付き
//     ロードして 0 パディングしてから wmma で読む。ストアも row/col ガード。
//     SDXL は seq=77 (cross-attn) 等の 16 非倍数があるため必須。
#include "kernels/gemm.cuh"
#include "kernels/utils.cuh"

#include <cuda_fp16.h>
#include <mma.h>

namespace dollama
{

// タイル一辺 (16x16 = 256 threads/block)。共有メモリ 2*16*16*4byte = 2KB。
static constexpr int TILE = 16;

// wmma タイル形状 (Blackwell sm_120 の FP16 サポート形状)。
static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16;
static constexpr int WARP_SIZE = 32;

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
// Tensor Core (wmma) 版 GEMM。transA=false 専用 (transA は launch でフォールバック)。
//   - 1 warp = C の 1 個の 16x16 タイルを担当する。
//   - block は (BWARP_X * BWARP_Y) 個の warp を持ち、複数タイルを並列処理する。
//   - K 方向を WMMA_K=16 ずつ進め、accumulator fragment に FP32 蓄積する。
//   - 端数タイル (タイルが M/N/K 境界をまたぐ) は共有メモリに 0 パディングして
//     から fragment へロードする。境界をまたがない完全タイルは A/B を直接
//     load_matrix_sync する (高速経路)。
//
// block 構成: blockDim = (WARP_SIZE * BWARP_X, BWARP_Y)。
//   warp 内 32 lane が 1 タイルを協調処理。
//   warp 座標 = (threadIdx.x / WARP_SIZE + blockIdx.x * BWARP_X,
//               threadIdx.y           + blockIdx.y * BWARP_Y)
// ----------------------------------------------------------------
static constexpr int BWARP_X = 4; // ブロック内 warp 列数
static constexpr int BWARP_Y = 4; // ブロック内 warp 行数

__global__ void gemm_fp16_wmma(const __half* A,
                               const __half* B,
                               __half*       C,
                               int           M,
                               int           N,
                               int           K,
                               float         alpha,
                               float         beta,
                               bool          transB)
{
    using namespace nvcuda::wmma;

    // この warp が担当する C タイルの先頭 (row, col)。
    const int warp_x = blockIdx.x * BWARP_X + (threadIdx.x / WARP_SIZE);
    const int warp_y = blockIdx.y * BWARP_Y + threadIdx.y;
    const int tile_row = warp_y * WMMA_M; // C の行先頭 i
    const int tile_col = warp_x * WMMA_N; // C の列先頭 j

    // 端数パディング用の共有メモリ。各 warp に A/B の 16x16 タイル領域を割り当てる。
    // レイアウト: sA は row-major (lda=WMMA_K), sB は col_major (ldb=WMMA_K)。
    //   warp 線形 index = threadIdx.y * BWARP_X + threadIdx.x / WARP_SIZE
    constexpr int NWARP = BWARP_X * BWARP_Y;
    __shared__ __half sA[NWARP][WMMA_M * WMMA_K];
    __shared__ __half sB[NWARP][WMMA_K * WMMA_N];
    const int warp_lin = threadIdx.y * BWARP_X + (threadIdx.x / WARP_SIZE);
    const int lane = threadIdx.x % WARP_SIZE;

    // タイルが完全に範囲内かどうか (行/列方向)。K 端数は別途判定。
    const bool full_row = (tile_row + WMMA_M) <= M;
    const bool full_col = (tile_col + WMMA_N) <= N;
    // このタイルが少しでも出力範囲にかかるか。
    const bool active = (tile_row < M) && (tile_col < N);

    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    fill_fragment(acc_frag, 0.0f);

    const int num_k = (K + WMMA_K - 1) / WMMA_K;
    for (int kt = 0; kt < num_k; ++kt)
    {
        const int k0 = kt * WMMA_K;
        const bool full_k = (k0 + WMMA_K) <= K;

        fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> a_frag;
        fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, col_major> b_frag_t; // transB 用
        fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, row_major> b_frag_n; // 非 transB 用

        if (active && full_row && full_col && full_k)
        {
            // 完全タイル: グローバルメモリから直接ロード (高速経路)。
            //   A: [M,K] row-major、先頭 A[tile_row*K + k0]、lda=K。
            load_matrix_sync(a_frag, A + tile_row * K + k0, K);
            if (transB)
            {
                // B: [N,K] row-major を col_major [K,N]・ldb=K とみなす。
                //   タイル先頭 (k=k0, j=tile_col) の col_major 要素は B[tile_col*K + k0]。
                load_matrix_sync(b_frag_t, B + tile_col * K + k0, K);
            }
            else
            {
                // B: [K,N] row-major、先頭 B[k0*N + tile_col]、ldb=N。
                load_matrix_sync(b_frag_n, B + k0 * N + tile_col, N);
            }
        }
        else if (active)
        {
            // 端数タイル: 共有メモリへガード付きロードして 0 パディング。
            // 32 lane で 16x16=256 要素を 8 要素/lane でカバーする。
            __half* psA = sA[warp_lin];
            __half* psB = sB[warp_lin];
#pragma unroll
            for (int e = lane; e < WMMA_M * WMMA_K; e += WARP_SIZE)
            {
                const int r = e / WMMA_K; // タイル内行 (0..15)
                const int c = e % WMMA_K; // タイル内 k (0..15)
                const int gr = tile_row + r;
                const int gk = k0 + c;
                psA[r * WMMA_K + c] =
                    (gr < M && gk < K) ? A[gr * K + gk] : __float2half(0.0f);
            }
#pragma unroll
            for (int e = lane; e < WMMA_K * WMMA_N; e += WARP_SIZE)
            {
                // sB は col_major (ldb=WMMA_K): 要素 (k, j) = psB[j*WMMA_K + k]。
                const int kk = e % WMMA_K; // タイル内 k (0..15)
                const int jj = e / WMMA_K; // タイル内列 (0..15)
                const int gk = k0 + kk;
                const int gj = tile_col + jj;
                __half v = __float2half(0.0f);
                if (gk < K && gj < N)
                {
                    v = transB ? B[gj * K + gk] : B[gk * N + gj];
                }
                psB[jj * WMMA_K + kk] = v;
            }
            __syncwarp();
            load_matrix_sync(a_frag, psA, WMMA_K);          // row-major lda=16
            load_matrix_sync(b_frag_t, psB, WMMA_K);        // col_major ldb=16
        }

        if (active)
        {
            if (full_row && full_col && full_k && !transB)
            {
                mma_sync(acc_frag, a_frag, b_frag_n, acc_frag);
            }
            else
            {
                mma_sync(acc_frag, a_frag, b_frag_t, acc_frag);
            }
        }
    }

    if (!active)
    {
        return;
    }

    // 出力: alpha/beta を適用して FP16 でストア。
    // 端数なら共有メモリ経由でガード付きストア、完全タイルなら一時バッファ→直書き。
    // accumulator fragment を一旦 float でストアしてから個別に処理する。
    __shared__ float sC[NWARP][WMMA_M * WMMA_N];
    float* psC = sC[warp_lin];
    store_matrix_sync(psC, acc_frag, WMMA_N, mem_row_major);
    __syncwarp();

#pragma unroll
    for (int e = lane; e < WMMA_M * WMMA_N; e += WARP_SIZE)
    {
        const int r = e / WMMA_N;
        const int c = e % WMMA_N;
        const int gr = tile_row + r;
        const int gc = tile_col + c;
        if (gr < M && gc < N)
        {
            const int idx = gr * N + gc;
            const float c_old = (beta != 0.0f) ? __half2float(C[idx]) : 0.0f;
            C[idx] = __float2half(alpha * psC[e] + beta * c_old);
        }
    }
}

// ----------------------------------------------------------------
// Tensor Core 経路を使うか判定する。
//   - transA=true は wmma 未対応 (col_major matrix_a が必要・現状不要) → タイリング。
//   - K が極端に小さい (< WMMA_K) と端数オーバヘッドが勝つため小行列はタイリング。
//   - それ以外 (SDXL の Linear: M=4096, N/K=320..2560 等) は wmma を使う。
// ----------------------------------------------------------------
static bool use_wmma(int M, int N, int K, bool transA)
{
    if (transA)
    {
        return false;
    }
    // 小さすぎる行列は wmma の起動・端数処理コストが割に合わない。
    if (M < WMMA_M || N < WMMA_N || K < WMMA_K)
    {
        return false;
    }
    return true;
}

// ----------------------------------------------------------------
// ホストラッパー。デバイスポインタを受け取り、形状に応じて
// Tensor Core (wmma) 版 / shared-mem タイリング版を選択して起動する。
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
    if (use_wmma(M, N, K, transA))
    {
        // 1 warp = 1 タイル (16x16)。block は BWARP_X*BWARP_Y warp。
        const dim3 block(WARP_SIZE * BWARP_X, BWARP_Y);
        const dim3 grid(ceil_div(N, WMMA_N * BWARP_X), ceil_div(M, WMMA_M * BWARP_Y));
        gemm_fp16_wmma<<<grid, block>>>(d_A, d_B, d_C, M, N, K, alpha, beta, transB);
        CUDA_CHECK_KERNEL();
        return;
    }

    // フォールバック: 自作 shared-mem タイリング版。
    const dim3 block(TILE, TILE);
    const dim3 grid(ceil_div(N, TILE), ceil_div(M, TILE));
    gemm_fp16_tiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K, alpha, beta, transA, transB);
    CUDA_CHECK_KERNEL();
}

} // namespace dollama

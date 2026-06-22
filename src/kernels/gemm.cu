// dense FP16 GEMM カーネル実装 (Phase 2 マイルストーン 2-2-1 / 2-6 S2 / 2-6 S3-B)
// 対象: RTX5080 (Blackwell / sm_120) / CUDA Runtime API のみ
//
// 設計判断 (2-2-2 以降でも再利用):
//   - レイアウトは row-major 固定。呼び出し側は常に row-major で考える。
//   - accumulator は必ず FP32 (FP16 蓄積は K が大きいと誤差が爆発するため)。
//   - transB=true 経路 (SDXL Linear `x @ W^T`) を本実装の主役として用意。
//   - 実装は 4 段構え:
//       (a) naive (1-thread-1-output)  … 正しさの基準・小行列フォールバック
//       (b) shared-memory タイリング   … 自作版 (Tensor Core 不可環境のフォールバック)
//       (c) Tensor Core (wmma) m16n16k16, FP32 蓄積  … 自作の高速経路 (S2 で追加)
//       (d) cuBLAS GemmEx (FP16 in / FP32 acc)        … 既定の高速経路 (S3-B で追加)
//     自作版 (a)(b)(c) は哲学として残し、(d) を既定経路として追加。
//     env DOLLAMA_GEMM=wmma で (d) を無効化し自作 wmma (c) に戻せる。
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
//
// S3-B (cuBLAS 委譲) のメモ:
//   - SDXL transformer の Linear (M=1024..4096, N/K=640..10240) は自作 wmma だと
//     1 warp = 1 出力タイル・ブロック内 A/B 再利用なしでメモリ帯域律速になり、
//     Tensor Core ピークの ~5% (25 TFLOPS @ M4096) 止まり。cuBLAS GemmEx に委譲する
//     と同形状で大幅に伸びる。CLAUDE.md 方針 (配管/重実装は定番に委ね、自作は HW を
//     叩く研究コアに限定。cuBLAS フォールバック許容) に沿う判断。自作版は残置。
#include "kernels/gemm.cuh"
#include "kernels/utils.cuh"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <cstdlib>
#include <cstring>

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

        // wmma load_matrix_sync は leading dim が 16byte (= __half 8 要素) 境界で
        // 整列している必要がある。A の lda=K、非 transB の B の ldb=N が 8 の倍数で
        // ないと領域外/誤読する。整列していなければ高速経路を諦め、共有メモリへ
        // 0 パディングして読む安全経路 (else if) に落とす (端数タイルと同じ扱い)。
        const bool aligned = ((K & 7) == 0) && (transB || ((N & 7) == 0));

        if (active && full_row && full_col && full_k && aligned)
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
            if (full_row && full_col && full_k && aligned && !transB)
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
// cuBLAS バックエンド (S3-B)。
//   自作 wmma GEMM は転置吸収・端数対応の研究コアとして残すが、SDXL transformer の
//   Linear (M=1024..4096, N/K=640..10240) では占有率/メモリ再利用が頭打ちで Tensor
//   Core ピークの ~5% 程度に留まる。CLAUDE.md の「自作は HW を叩く研究コアに限定し、
//   配管/重実装は定番に委ねる」方針に沿い、巨大 GEMM のみ cuBLAS (FP16 in / FP32 acc
//   = CUBLAS_COMPUTE_32F) に委譲する。env DOLLAMA_GEMM=wmma で常に自作 wmma へ戻せる。
//
// レイアウト変換 (row-major → cuBLAS の col-major):
//   欲しいのは row-major C[M,N] = alpha * op(A)[M,K] @ op(B)[K,N] + beta*C。
//   row-major X は col-major X^T と同じバッファ。col-major で C^T[N,M] を計算すると
//   C^T = op(B)^T @ op(A)^T。cuBLAS GemmEx を
//       (opB', opA'=N, m'=N, n'=M, k'=K, B, ldB', A, ldA'=K, C, ldC'=N)
//   と呼べばよい。A は logical row-major [M,K] (ld=K) = col-major [K,M] = A^T なので
//   opA'=N・ldA'=K で固定。B 側:
//     transB=false: B row-major [K,N] (ld=N) = col-major [N,K] = op(B)^T → opB'=N, ldB'=N
//     transB=true : B row-major [N,K] (ld=K) = col-major [K,N]、欲しいのは [N,K] なので
//                   転置 → opB'=T, ldB'=K
//   transA=true は本経路では非対応 (呼び出し側で自作タイリングへフォールバック)。
// ----------------------------------------------------------------

#define CUBLAS_CHECK(call)                                                  \
    do                                                                      \
    {                                                                       \
        cublasStatus_t st = (call);                                         \
        if (st != CUBLAS_STATUS_SUCCESS)                                    \
        {                                                                   \
            throw std::runtime_error(                                       \
                std::string("cuBLAS error ") + std::to_string((int)st)      \
                + " at " __FILE__ ":" + std::to_string(__LINE__));          \
        }                                                                   \
    } while (0)

// プロセス内で 1 個だけ持つ cuBLAS ハンドル (遅延生成・破棄はプロセス終了に委ねる)。
// SDXL は単一 stream・単一スレッドから GEMM を呼ぶ前提。
static cublasHandle_t g_cublas_handle = nullptr;

static cublasHandle_t cublas_handle()
{
    if (g_cublas_handle == nullptr)
    {
        CUBLAS_CHECK(cublasCreate(&g_cublas_handle));
        // FP16 入力でも Tensor Core 経路を許可する (既定 MATH でも可だが明示)。
        CUBLAS_CHECK(cublasSetMathMode(g_cublas_handle, CUBLAS_DEFAULT_MATH));
    }
    return g_cublas_handle;
}

// env DOLLAMA_GEMM=wmma で cuBLAS を無効化し自作 wmma へ強制フォールバック (研究/比較用)。
static bool cublas_disabled()
{
    static int cached = -1;
    if (cached < 0)
    {
        const char* e = std::getenv("DOLLAMA_GEMM");
        cached = (e != nullptr && std::strcmp(e, "wmma") == 0) ? 1 : 0;
    }
    return cached != 0;
}

// cuBLAS GemmEx を使うか判定する。
//   - transA=true は本経路非対応 → 自作へ。
//   - env で無効化されていれば使わない。
//   - 小さすぎる行列は cuBLAS のオーバヘッドが勝つため自作 wmma に任せる。
static bool use_cublas(int M, int N, int K, bool transA)
{
    if (transA || cublas_disabled())
    {
        return false;
    }
    if (M < WMMA_M || N < WMMA_N || K < WMMA_K)
    {
        return false;
    }
    return true;
}

static void gemm_cublas(const __half* d_A,
                        const __half* d_B,
                        __half*       d_C,
                        int           M,
                        int           N,
                        int           K,
                        float         alpha,
                        float         beta,
                        bool          transB)
{
    cublasHandle_t h = cublas_handle();

    const cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
    const int ldB = transB ? K : N; // transB=true: B[N,K] ld=K / false: B[K,N] ld=N
    const int ldA = K;              // A row-major [M,K] = col-major [K,M] ld=K
    const int ldC = N;              // C row-major [M,N] = col-major [N,M] ld=N

    // C^T[N,M] = op(B)^T @ op(A)^T。FP16 in / FP32 acc。
    CUBLAS_CHECK(cublasGemmEx(h,
                              opB, CUBLAS_OP_N,
                              N, M, K,
                              &alpha,
                              d_B, CUDA_R_16F, ldB,
                              d_A, CUDA_R_16F, ldA,
                              &beta,
                              d_C, CUDA_R_16F, ldC,
                              CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

// ----------------------------------------------------------------
// Tensor Core 経路 (自作 wmma) を使うか判定する。
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
// cuBLAS / Tensor Core (自作 wmma) / shared-mem タイリング版を選択して起動する。
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
    // 既定: cuBLAS GemmEx (FP16 in / FP32 acc)。
    if (use_cublas(M, N, K, transA))
    {
        gemm_cublas(d_A, d_B, d_C, M, N, K, alpha, beta, transB);
        return;
    }

    // 研究/比較用フォールバック: 自作 wmma。
    if (use_wmma(M, N, K, transA))
    {
        // 1 warp = 1 タイル (16x16)。block は BWARP_X*BWARP_Y warp。
        const dim3 block(WARP_SIZE * BWARP_X, BWARP_Y);
        const dim3 grid(ceil_div(N, WMMA_N * BWARP_X), ceil_div(M, WMMA_M * BWARP_Y));
        gemm_fp16_wmma<<<grid, block>>>(d_A, d_B, d_C, M, N, K, alpha, beta, transB);
        CUDA_CHECK_KERNEL();
        return;
    }

    // 最終フォールバック: 自作 shared-mem タイリング版 (transA / 小行列)。
    const dim3 block(TILE, TILE);
    const dim3 grid(ceil_div(N, TILE), ceil_div(M, TILE));
    gemm_fp16_tiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K, alpha, beta, transA, transB);
    CUDA_CHECK_KERNEL();
}

} // namespace dollama

// LayerNorm カーネル実装 (Phase 2 マイルストーン 2-5 / UNet Transformer2D)
// 対象: RTX5080 (Blackwell / sm_120) / CUDA Runtime API のみ
//
// 設計判断 (groupnorm.cu と同方針):
//   - 入出力 FP16 / 内部計算 FP32。部分和 (sum, sum-sq) は必ず float 蓄積。
//     var = E[x^2] - E[x]^2 を 1 パスで求め、var = max(var, 0.0f) でクランプ。
//   - リダクション戦略: 1 行 = 1 ブロック。グリッド = rows ブロック。
//     ブロック内でグリッドストライドして部分和を集め、shared memory + warp shuffle
//     (__shfl_down_sync) のツリーリダクションで mean/var を確定し、同ブロックが
//     正規化 + affine を書き戻す。reduction ヘルパーは本 TU 内に閉じる。
//   - GroupNorm と異なり正規化軸は最終次元 C (1 行) のみ。空間方向は混ぜない。
//   - gamma/beta は nullptr 許容 (affine なし LayerNorm 対応)。
//   - in-place 安全: 行の全要素を読み切って mean/var を確定してから書くため、
//     d_in == d_out でもレースは発生しない。
#include "kernels/layernorm.cuh"
#include "kernels/utils.cuh"

#include <cuda_fp16.h>

namespace dollama
{

// 1 行 = 1 ブロックのスレッド数。32 の倍数 (warp 境界揃え)。
static constexpr int LN_THREADS = 256;

// ----------------------------------------------------------------
// warp 内ツリーリダクション (和)。__shfl_down_sync で 32 レーンを集約。
// 戻り値はレーン 0 にのみ全和が入る。
// ----------------------------------------------------------------
__device__ inline float ln_warp_reduce_sum(float v)
{
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

// ----------------------------------------------------------------
// ブロック全体のリダクション (和)。結果は全スレッドで共有してから返す。
// smem は少なくとも (blockDim.x / warpSize) 要素必要。
// ----------------------------------------------------------------
__device__ inline float ln_block_reduce_sum(float v, float* smem)
{
    const int lane   = threadIdx.x % warpSize;
    const int wid    = threadIdx.x / warpSize;
    const int nwarps = (blockDim.x + warpSize - 1) / warpSize;

    v = ln_warp_reduce_sum(v);

    if (lane == 0)
    {
        smem[wid] = v;
    }
    __syncthreads();

    if (wid == 0)
    {
        float w = (lane < nwarps) ? smem[lane] : 0.0f;
        w = ln_warp_reduce_sum(w);
        if (lane == 0)
        {
            smem[0] = w;
        }
    }
    __syncthreads();

    const float total = smem[0];
    __syncthreads();
    return total;
}

// ----------------------------------------------------------------
// LayerNorm カーネル: 1 ブロック = 1 行 (最終次元 C 要素) を担当。
//   C 要素について 1 パスで sum / sum-sq を集め、mean / var を求めてから
//   正規化 + affine を書き戻す。gamma/beta は nullptr 許容。
// ----------------------------------------------------------------
__global__ void layer_norm_fp16(const __half* in,
                                const __half* gamma,
                                const __half* beta,
                                __half*       out,
                                int           rows,
                                int           C,
                                float         eps)
{
    const int row = blockIdx.x;
    if (row >= rows)
    {
        return;
    }

    const long base = static_cast<long>(row) * C;

    __shared__ float smem[LN_THREADS / 32];

    // ---- 1 パスで部分和を集める (グリッドストライド) ----
    float local_sum = 0.0f;
    float local_sq  = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x)
    {
        const float x = __half2float(in[base + c]);
        local_sum += x;
        local_sq  += x * x;
    }

    // ---- mean / var をブロック全体で確定 ----
    const float sum    = ln_block_reduce_sum(local_sum, smem);
    const float sum_sq = ln_block_reduce_sum(local_sq, smem);

    const float inv_c = 1.0f / static_cast<float>(C);
    const float mean  = sum * inv_c;
    // var = E[x^2] - E[x]^2。桁落ちで負になり得るので 0 クランプ。
    float var = sum_sq * inv_c - mean * mean;
    var = fmaxf(var, 0.0f);
    const float inv_std = rsqrtf(var + eps);

    // ---- 正規化 + affine を書き戻す ----
    for (int c = threadIdx.x; c < C; c += blockDim.x)
    {
        const float x  = __half2float(in[base + c]);
        const float gm = (gamma != nullptr) ? __half2float(gamma[c]) : 1.0f;
        const float bt = (beta  != nullptr) ? __half2float(beta[c])  : 0.0f;
        const float y  = (x - mean) * inv_std * gm + bt;
        out[base + c] = __float2half(y);
    }
}

// ----------------------------------------------------------------
// ホストラッパー
// ----------------------------------------------------------------
void launch_layer_norm(const __half* d_in,
                       const __half* d_gamma,
                       const __half* d_beta,
                       __half*       d_out,
                       int           rows,
                       int           C,
                       float         eps)
{
    if (rows <= 0 || C <= 0)
    {
        return;
    }
    // グリッド = rows ブロック (1 行 = 1 ブロック)。
    layer_norm_fp16<<<rows, LN_THREADS>>>(d_in, d_gamma, d_beta, d_out, rows, C, eps);
    CUDA_CHECK_KERNEL();
}

} // namespace dollama

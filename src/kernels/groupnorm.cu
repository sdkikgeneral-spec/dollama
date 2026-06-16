// GroupNorm カーネル実装 (Phase 2 マイルストーン 2-2-3)
// 対象: RTX5080 (Blackwell / sm_120) / CUDA Runtime API のみ
//
// 設計判断 (後続カーネルでも参照する):
//   - 入出力 FP16 / 内部計算 FP32。部分和 (sum, sum-sq) は必ず float 蓄積。
//     var = E[x^2] - E[x]^2 を 1 パスで求め、var = max(var, 0.0f) でクランプ。
//     (2 パスや Welford は使わない。PL 指定の 1 パス sum/sum-sq 方式。)
//   - リダクション戦略: 1 グループ = 1 ブロック。グリッド = N*num_groups ブロック。
//     ブロック内でグリッドストライドして部分和を集め、shared memory + warp shuffle
//     (__shfl_down_sync) のツリーリダクションで mean/var を確定し、同ブロックが
//     正規化 + affine を書き戻す。reduction ヘルパーは本 TU 内に閉じる
//     (utils.cuh は変更しない)。
//   - in-place 安全: グループの全要素を読み切って mean/var を確定してから書くため、
//     d_in == d_out でもレースは発生しない。
//   - SiLU 融合はしない (PL 指定。GroupNorm 単体のみ)。GroupNorm→SiLU は ResNet で
//     連続しメモリ律速なので、2-4/2-5 で実測ボトルネック化したら
//     launch_group_norm_silu として融合を検討する。
//   - 占有率: 1 グループ = 1 ブロックの単純版で正しさを優先。大 feature map で
//     SM 占有率不足が顕在化したら multi-block reduction (グループを複数ブロックで
//     分担し partial sum を atomic/2 段で集約) へ昇格する。
#include "kernels/groupnorm.cuh"
#include "kernels/utils.cuh"

#include <cuda_fp16.h>

namespace dollama
{

// 1 グループ = 1 ブロックのスレッド数。32 の倍数 (warp 境界揃え)。
static constexpr int GN_THREADS = 256;

// ----------------------------------------------------------------
// warp 内ツリーリダクション (和)。__shfl_down_sync で 32 レーンを集約。
// 戻り値はレーン 0 にのみ全和が入る (他レーンは部分和)。
// ----------------------------------------------------------------
__device__ inline float warp_reduce_sum(float v)
{
    // フルマスク。GN_THREADS は 32 の倍数なので分岐発散はない。
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

// ----------------------------------------------------------------
// ブロック全体のリダクション (和)。
// 1) 各 warp 内を warp_reduce_sum で集約
// 2) 各 warp の代表値 (レーン0) を shared memory に格納
// 3) 先頭 warp が shared memory 上の代表値を再度集約
// 結果は全スレッドで共有 (shared memory 経由) してから返す。
// smem は少なくとも (blockDim.x / warpSize) 要素必要。
// ----------------------------------------------------------------
__device__ inline float block_reduce_sum(float v, float* smem)
{
    const int lane = threadIdx.x % warpSize;
    const int wid  = threadIdx.x / warpSize;
    const int nwarps = (blockDim.x + warpSize - 1) / warpSize;

    v = warp_reduce_sum(v);

    if (lane == 0)
    {
        smem[wid] = v;
    }
    __syncthreads();

    // 先頭 warp が各 warp の代表値を集約。
    float total = 0.0f;
    if (wid == 0)
    {
        float w = (lane < nwarps) ? smem[lane] : 0.0f;
        w = warp_reduce_sum(w);
        if (lane == 0)
        {
            smem[0] = w;
        }
    }
    __syncthreads();

    total = smem[0];
    __syncthreads();
    return total;
}

// ----------------------------------------------------------------
// GroupNorm カーネル: 1 ブロック = サンプル n・グループ g を担当。
//   group_size = cpg * H * W 要素について 1 パスで sum / sum-sq を集め、
//   mean / var を求めてから正規化 + affine を書き戻す。
// ----------------------------------------------------------------
__global__ void group_norm_fp16(const __half* in,
                                const __half* gamma,
                                const __half* beta,
                                __half*       out,
                                int           C,
                                int           HW,        // H*W
                                int           cpg,       // C / num_groups
                                int           num_groups,
                                float         eps)
{
    // blockIdx.x = n * num_groups + g
    const int block_id = blockIdx.x;
    const int g        = block_id % num_groups;
    const int n        = block_id / num_groups;

    const int c_begin   = g * cpg;             // このグループの先頭チャネル
    const int group_size = cpg * HW;           // グループ内の総要素数 K

    // このグループの先頭要素オフセット ((n*C + c_begin)*HW)。
    const long base = (static_cast<long>(n) * C + c_begin) * static_cast<long>(HW);

    // shared memory: block_reduce_sum 用 (warp 数ぶん)。
    // sum と sum-sq を別々にリダクションするため 2 回使い回す。
    __shared__ float smem[GN_THREADS / 32];

    // ---- 1 パスで部分和を集める (グリッドストライド) ----
    float local_sum = 0.0f;
    float local_sq  = 0.0f;
    for (int i = threadIdx.x; i < group_size; i += blockDim.x)
    {
        const float x = __half2float(in[base + i]);
        local_sum += x;
        local_sq  += x * x;
    }

    // ---- mean / var をブロック全体で確定 ----
    const float sum    = block_reduce_sum(local_sum, smem);
    const float sum_sq = block_reduce_sum(local_sq, smem);

    const float inv_k = 1.0f / static_cast<float>(group_size);
    const float mean  = sum * inv_k;
    // var = E[x^2] - E[x]^2。桁落ちで負になり得るので 0 クランプ。
    float var = sum_sq * inv_k - mean * mean;
    var = fmaxf(var, 0.0f);
    const float inv_std = rsqrtf(var + eps);

    // ---- 正規化 + affine を書き戻す ----
    // i はグループ内インデックス。チャネルは c_begin + (i / HW)。
    for (int i = threadIdx.x; i < group_size; i += blockDim.x)
    {
        const int   c = c_begin + (i / HW);
        const float x = __half2float(in[base + i]);
        const float gm = __half2float(gamma[c]);
        const float bt = __half2float(beta[c]);
        const float y = (x - mean) * inv_std * gm + bt;
        out[base + i] = __float2half(y);
    }
}

// ----------------------------------------------------------------
// ホストラッパー
// ----------------------------------------------------------------
void launch_group_norm(const __half* d_in,
                       const __half* d_gamma,
                       const __half* d_beta,
                       __half*       d_out,
                       int           N,
                       int           C,
                       int           H,
                       int           W,
                       int           num_groups,
                       float         eps)
{
    if (N <= 0 || C <= 0 || H <= 0 || W <= 0 || num_groups <= 0)
    {
        return;
    }
    // C はグループ数で割り切れること (PyTorch GroupNorm 前提)。
    assert(C % num_groups == 0);

    const int cpg = C / num_groups;
    const int HW  = H * W;

    // グリッド = N * num_groups ブロック (1 グループ = 1 ブロック)。
    const int blocks = N * num_groups;
    group_norm_fp16<<<blocks, GN_THREADS>>>(d_in, d_gamma, d_beta, d_out,
                                            C, HW, cpg, num_groups, eps);
    CUDA_CHECK_KERNEL();
}

} // namespace dollama

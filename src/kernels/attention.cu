// Attention カーネル実装 (Phase 2 マイルストーン 2-2-5)
// 対象: RTX5080 (Blackwell / sm_120) / CUDA Runtime API のみ
//
// 設計判断 (後続カーネルでも参照する):
//   - 入出力 FP16 / 内部計算 (内積・softmax・PV 積和) は必ず FP32 蓄積。最後に
//     __float2half で書き戻す (GEMM / GroupNorm / Conv と同規約)。
//   - 1 ブロック = 1 つの (b, h, query 行 i) を担当。グリッド = B*H*Sq ブロック。
//     スレッドは Sk と Dh を協調して処理する。
//       1) scores[Sk] (FP32) を動的 shared memory に確保。各 j について
//          score[j] = scale * Σ_d Q[i,d]*K[j,d] を FP32 蓄積 (スレッドで Sk を分担)。
//       2) softmax を FP32 2 パスで: shared 上の max を block reduce → exp して
//          sum を block reduce → 正規化。数値安定のため max 減算必須。
//       3) O[i,d] = Σ_j P[j] * V[j,d] を FP32 蓄積 → __float2half で書き戻す。
//   - self / cross は同一カーネル。違いは Sq と Sk (および Dh) のみで、Sq と Sk を
//     別パラメータにして両対応する。
//   - reduction ヘルパー (warp/block の sum/max) は本 TU 内に閉じる
//     (utils.cuh は変更しない)。groupnorm.cu の warp_reduce_sum/block_reduce_sum と
//     同じ shuffle ツリーパターンに max 版を加えたもの。
//   - 占有率: scores を shared に materialize する単純版で正しさを優先。Sk が
//     極端に大きく shared に収まらない場合は online-softmax (flash) へ昇格する
//     (attention.cuh の最適化メモ参照)。
//   - 大 Sk 対応 (VAE mid_block): scores[Sk] がデフォルト 48KB を超える場合
//     (Sk > 12288)、launch ラッパーで cudaFuncSetAttribute による動的 shared の
//     opt-in を行う。VAE mid は Sq=Sk=16384 → 64KB+ で 48KB を超えるため必須。
#include "kernels/attention.cuh"
#include "kernels/utils.cuh"

#include <cuda_fp16.h>

namespace dollama
{

// 1 ブロック (1 query 行) のスレッド数。32 の倍数 (warp 境界揃え)。
static constexpr int ATTN_THREADS = 256;

// ----------------------------------------------------------------
// warp 内ツリーリダクション (和)。__shfl_down_sync で 32 レーンを集約。
// 戻り値はレーン 0 にのみ全和が入る (他レーンは部分和)。
// ----------------------------------------------------------------
__device__ inline float warp_reduce_sum(float v)
{
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

// ----------------------------------------------------------------
// warp 内ツリーリダクション (最大)。__shfl_down_sync で 32 レーンを集約。
// 戻り値はレーン 0 にのみ全最大が入る (他レーンは部分最大)。
// ----------------------------------------------------------------
__device__ inline float warp_reduce_max(float v)
{
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    {
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, offset));
    }
    return v;
}

// ----------------------------------------------------------------
// ブロック全体のリダクション (和)。結果は全スレッドで共有してから返す。
// smem は少なくとも (blockDim.x / warpSize) 要素必要。
// ----------------------------------------------------------------
__device__ inline float block_reduce_sum(float v, float* smem)
{
    const int lane   = threadIdx.x % warpSize;
    const int wid    = threadIdx.x / warpSize;
    const int nwarps = (blockDim.x + warpSize - 1) / warpSize;

    v = warp_reduce_sum(v);

    if (lane == 0)
    {
        smem[wid] = v;
    }
    __syncthreads();

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

    const float total = smem[0];
    __syncthreads();
    return total;
}

// ----------------------------------------------------------------
// ブロック全体のリダクション (最大)。結果は全スレッドで共有してから返す。
// smem は少なくとも (blockDim.x / warpSize) 要素必要。
// 空集合での -inf 初期値が問題にならないよう、呼び側は各スレッドが最低 1 要素を
// 担当する (担当が無いスレッドは -inf を渡す)。
// ----------------------------------------------------------------
__device__ inline float block_reduce_max(float v, float* smem)
{
    const int lane   = threadIdx.x % warpSize;
    const int wid    = threadIdx.x / warpSize;
    const int nwarps = (blockDim.x + warpSize - 1) / warpSize;

    v = warp_reduce_max(v);

    if (lane == 0)
    {
        smem[wid] = v;
    }
    __syncthreads();

    if (wid == 0)
    {
        float w = (lane < nwarps) ? smem[lane] : -INFINITY;
        w = warp_reduce_max(w);
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
// attention カーネル: 1 ブロック = (b, h, query 行 i) を担当。
//   動的 shared memory の先頭に scores[Sk] (FP32) を置き、後ろに block reduce 用の
//   warp スクラッチ (ATTN_THREADS/32 要素) を置く。
// ----------------------------------------------------------------
__global__ void attention_fp16(const __half* q,
                               const __half* k,
                               const __half* v,
                               __half*       out,
                               int           B,
                               int           H,
                               int           Sq,
                               int           Sk,
                               int           Dh,
                               float         scale)
{
    // blockIdx.x = (b*H + h)*Sq + i
    const long block_id = blockIdx.x;
    const int  i        = static_cast<int>(block_id % Sq);
    const long bh       = block_id / Sq;          // b*H + h
    // (b, h) は系列オフセット計算にしか使わないので bh のまま扱う。

    // 各テンソルの (b, h) ヘッド先頭オフセット。
    const long q_head = bh * Sq * Dh;
    const long k_head = bh * Sk * Dh;
    const long v_head = bh * Sk * Dh;
    const long o_head = bh * Sq * Dh;

    // この query 行 Q[i, :] の先頭。
    const long q_row = q_head + static_cast<long>(i) * Dh;

    // 動的 shared memory レイアウト: [ scores(Sk float) | reduce scratch(nwarps float) ]
    extern __shared__ float smem[];
    float* scores  = smem;          // Sk 要素
    float* scratch = smem + Sk;     // block reduce 用 (nwarps 要素)

    // ---- 1) scores[j] = scale * Σ_d Q[i,d] * K[j,d] を FP32 蓄積 ----
    // スレッドで Sk を分担。各 score は Dh の内積。
    for (int j = threadIdx.x; j < Sk; j += blockDim.x)
    {
        const long k_row = k_head + static_cast<long>(j) * Dh;
        float dot = 0.0f;
        for (int d = 0; d < Dh; ++d)
        {
            const float qd = __half2float(q[q_row + d]);
            const float kd = __half2float(k[k_row + d]);
            dot += qd * kd;
        }
        scores[j] = scale * dot;
    }
    __syncthreads();

    // ---- 2) softmax (FP32 2 パス, max 減算で数値安定) ----
    // 2a) 行最大を block reduce で求める。担当が無いスレッドは -inf。
    float local_max = -INFINITY;
    for (int j = threadIdx.x; j < Sk; j += blockDim.x)
    {
        local_max = fmaxf(local_max, scores[j]);
    }
    const float row_max = block_reduce_max(local_max, scratch);

    // 2b) exp(score - max) を書き戻しつつ和を block reduce。
    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < Sk; j += blockDim.x)
    {
        const float e = __expf(scores[j] - row_max);
        scores[j] = e;
        local_sum += e;
    }
    const float row_sum = block_reduce_sum(local_sum, scratch);
    // row_sum は Sk>=1 なら最低でも exp(0)=1 を含むので 0 にはならない。
    const float inv_sum = 1.0f / row_sum;

    // ---- 3) O[i,d] = Σ_j P[j] * V[j,d] を FP32 蓄積 ----
    // スレッドで Dh を分担。各出力は Sk について P[j]*V[j,d] を蓄積。
    // P[j] = scores[j] * inv_sum (正規化を内積内で掛ける)。
    for (int d = threadIdx.x; d < Dh; d += blockDim.x)
    {
        float acc = 0.0f;
        for (int j = 0; j < Sk; ++j)
        {
            const float p  = scores[j] * inv_sum;
            const float vd = __half2float(v[v_head + static_cast<long>(j) * Dh + d]);
            acc += p * vd;
        }
        out[o_head + static_cast<long>(i) * Dh + d] = __float2half(acc);
    }
}

// ----------------------------------------------------------------
// ホストラッパー
// ----------------------------------------------------------------
void launch_attention(const __half* d_q, const __half* d_k, const __half* d_v,
                      __half* d_out, int B, int H, int Sq, int Sk, int Dh, float scale)
{
    // 不正・空入力ガード。負やゼロの次元は何もしない。
    if (B <= 0 || H <= 0 || Sq <= 0 || Sk <= 0 || Dh <= 0)
    {
        return;
    }

    // グリッド = B*H*Sq ブロック (1 ブロック = 1 query 行)。
    const long blocks_l = static_cast<long>(B) * H * Sq;
    assert(blocks_l > 0);
    const int blocks = static_cast<int>(blocks_l);

    // 動的 shared memory: scores(Sk float) + block reduce scratch(nwarps float)。
    const int nwarps = (ATTN_THREADS + 31) / 32;
    const size_t shmem = (static_cast<size_t>(Sk) + nwarps) * sizeof(float);

    // Blackwell (sm_120) の動的 shared memory opt-in:
    //   起動時に静的に確保できる動的 shared はデフォルト 48KB が上限。VAE mid_block の
    //   spatial self-attention (Sq=Sk=16384) では scores[Sk] が 16384*4=64KB に達し
    //   48KB を超えるため、cudaFuncSetAttribute で MaxDynamicSharedMemorySize を
    //   引き上げる必要がある (sm_120 は ~227KB/SM まで許可)。opt-in は冪等なので
    //   毎回呼んでよいが、48KB 以下のときは不要 (既存の小 Sk ケースの挙動を変えない)。
    //   確保上限を超える要求は cudaFuncSetAttribute が cudaErrorInvalidValue を返し、
    //   CUDA_CHECK が必要バイト数を含む例外を投げて止まる (online-softmax 昇格の判断材料)。
    constexpr size_t kDefaultShmemLimit = 48u * 1024u; // 48KB (デフォルト動的 shared 上限)
    if (shmem > kDefaultShmemLimit)
    {
        CUDA_CHECK(cudaFuncSetAttribute(attention_fp16,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(shmem)));
    }

    attention_fp16<<<blocks, ATTN_THREADS, shmem>>>(d_q, d_k, d_v, d_out,
                                                    B, H, Sq, Sk, Dh, scale);
    CUDA_CHECK_KERNEL();
}

} // namespace dollama

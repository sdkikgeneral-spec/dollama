// Attention カーネル実装 (Phase 2 マイルストーン 2-2-5 / S2 flash 化)
// 対象: RTX5080 (Blackwell / sm_120) / CUDA Runtime API のみ
//
// 設計判断 (後続カーネルでも参照する):
//   - 入出力 FP16 / 内部計算 (内積・softmax・PV 積和) は必ず FP32 蓄積。最後に
//     __float2half で書き戻す (GEMM / GroupNorm / Conv と同規約)。
//   - 経路は 2 本:
//       (A) naive (attention_fp16): 1 ブロック = 1 つの (b,h,query 行 i)。scores[Sk]
//           を動的 shared に materialize し 2 パス softmax。Dh が大きい VAE mid_block
//           (Dh=512) 用フォールバック。
//       (B) flash (attention_flash_fp16): online softmax。scores を materialize せず
//           K/V を Bk 行ずつタイル走査。UNet self/cross (Dh=64) の主役。S2 で追加。
//   - self / cross は同一カーネルで扱う。違いは Sq と Sk (および Dh) のみ。
//   - reduction ヘルパー (warp/block の sum/max) は本 TU 内に閉じる (utils.cuh 不変)。
//   - 経路選択は launch_attention で Dh により行う (Dh<=128 は flash、それ超は naive)。
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
// naive attention カーネル: 1 ブロック = (b, h, query 行 i) を担当。
//   動的 shared memory の先頭に scores[Sk] (FP32) を置き、後ろに block reduce 用の
//   warp スクラッチ (ATTN_THREADS/32 要素) を置く。
//   Dh が大きい VAE mid_block (Dh=512) 用フォールバック経路。
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

// ================================================================
// flash-attention 経路 (online softmax, scores を materialize しない)
// ================================================================
// S2 最適化: naive 版は scores[Sk] を shared に展開し 2 パス softmax を行うため
//   Sk が大きい UNet self-attn (Sq=Sk=4096/1024) で shared 帯域と再ロードが律速。
//   flash 版は K/V を Bk 行ずつタイル走査し、各 query 行が running max m と running
//   sum l と acc[Dh] を保持して逐次補正 (online softmax)。scores を一切書き戻さず
//   K/V を 1 回だけ shared に読む。FP32 蓄積は naive と同一規約。
//
// ブロック割り当て:
//   1 ブロック = 1 つの (b, h) の query 行ブロック (FLASH_BQ 行)。
//   blockDim.x = FLASH_BQ スレッド。スレッド t は query 行 i = row_base + t を担当。
//   各スレッドは自分の Q[i,:] を読み (Dh<=128 なのでローカル配列)、acc[Dh] / m / l を
//   ローカルに保持する。K/V タイル (Bk 行 x Dh) を shared にブロック協調でロードし、
//   ブロック内 FLASH_BQ スレッドで再利用する (K/V の DRAM 読み出しを 1 回に削減)。
//
// 対応範囲: Dh <= FLASH_MAX_DH (=128)。UNet の Dh=64 を確実にカバーする。
//   VAE mid_block の Dh=512 はローカル acc[512]+q[512] が大きすぎ (レジスタ溢れ +
//   占有率破壊) ため flash 経路に乗せず、既存 naive (attention_fp16) にフォールバック。
// ----------------------------------------------------------------

// flash の query 行ブロック幅 (= blockDim.x)。warp 境界 (32) の倍数。
static constexpr int FLASH_BQ = 64;
// flash の K/V タイル行数。shared に Bk*Dh*2 (K と V) の FP16 を置く。
static constexpr int FLASH_BK = 64;
// flash 経路が扱える Dh 上限 (ローカル acc[Dh] / Q[Dh] のサイズ制約)。
static constexpr int FLASH_MAX_DH = 128;

__global__ void attention_flash_fp16(const __half* __restrict__ q,
                                     const __half* __restrict__ k,
                                     const __half* __restrict__ v,
                                     __half* __restrict__       out,
                                     int                        B,
                                     int                        H,
                                     int                        Sq,
                                     int                        Sk,
                                     int                        Dh,
                                     float                      scale)
{
    // blockIdx.y = (b*H + h)、blockIdx.x = query 行ブロック番号。
    const long bh       = blockIdx.y;
    const int  row_base = static_cast<int>(blockIdx.x) * FLASH_BQ;
    const int  i        = row_base + static_cast<int>(threadIdx.x);

    const long q_head = bh * Sq * Dh;
    const long k_head = bh * Sk * Dh;
    const long v_head = bh * Sk * Dh;
    const long o_head = bh * Sq * Dh;

    // この行が有効か (末尾ブロックでは row_base+t が Sq を超える場合がある)。
    const bool active = (i < Sq);

    // ローカル状態: Q 行・acc・online softmax の m / l。
    float q_reg[FLASH_MAX_DH];
    float acc[FLASH_MAX_DH];
    for (int d = 0; d < Dh; ++d)
    {
        acc[d] = 0.0f;
    }
    float m_run = -INFINITY; // running max
    float l_run = 0.0f;      // running sum (補正済み)

    if (active)
    {
        const long q_row = q_head + static_cast<long>(i) * Dh;
        for (int d = 0; d < Dh; ++d)
        {
            q_reg[d] = __half2float(q[q_row + d]);
        }
    }

    // K/V タイル用 shared。[ K_tile(FLASH_BK*Dh) | V_tile(FLASH_BK*Dh) ] (FP16)。
    extern __shared__ __half ksmem[];
    __half* k_tile = ksmem;
    __half* v_tile = ksmem + static_cast<long>(FLASH_BK) * Dh;

    // K/V を Bk 行ずつタイル走査。
    for (int j0 = 0; j0 < Sk; j0 += FLASH_BK)
    {
        const int tile_rows = min(FLASH_BK, Sk - j0);

        // ブロック協調で K/V タイルを shared にロード (全 FLASH_BQ スレッドで分担)。
        const int tile_elems = tile_rows * Dh;
        for (int e = threadIdx.x; e < tile_elems; e += blockDim.x)
        {
            const int  r   = e / Dh;
            const int  d   = e - r * Dh;
            const long src = static_cast<long>(j0 + r) * Dh + d;
            k_tile[e] = k[k_head + src];
            v_tile[e] = v[v_head + src];
        }
        __syncthreads();

        if (active)
        {
            // このタイル内の各 K 行についてスコアを計算し online softmax で acc/l を補正。
            for (int r = 0; r < tile_rows; ++r)
            {
                const __half* k_ptr = k_tile + static_cast<long>(r) * Dh;
                float dot = 0.0f;
                for (int d = 0; d < Dh; ++d)
                {
                    dot += q_reg[d] * __half2float(k_ptr[d]);
                }
                const float s = scale * dot;

                // online softmax: 新しい running max と補正係数。
                const float m_new = fmaxf(m_run, s);
                const float corr  = __expf(m_run - m_new); // 既存 acc/l のスケール
                const float p     = __expf(s - m_new);     // この行の確率重み (未正規化)

                // 既存 acc を corr でスケールし、p*V を加算。
                const __half* v_ptr = v_tile + static_cast<long>(r) * Dh;
                for (int d = 0; d < Dh; ++d)
                {
                    acc[d] = acc[d] * corr + p * __half2float(v_ptr[d]);
                }
                l_run = l_run * corr + p;
                m_run = m_new;
            }
        }
        __syncthreads(); // 次タイルのロード前に shared 再利用を保護
    }

    // 正規化して書き戻す。
    if (active)
    {
        const float inv_l = (l_run > 0.0f) ? (1.0f / l_run) : 0.0f;
        const long  o_row = o_head + static_cast<long>(i) * Dh;
        for (int d = 0; d < Dh; ++d)
        {
            out[o_row + d] = __float2half(acc[d] * inv_l);
        }
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

    // ---- 経路選択 ----
    // Dh <= FLASH_MAX_DH (=128): flash 経路 (online softmax)。UNet self/cross の主役。
    // Dh > FLASH_MAX_DH (VAE mid Dh=512 等): naive 経路にフォールバック。
    if (Dh <= FLASH_MAX_DH)
    {
        // グリッド: x = query 行ブロック数, y = B*H。各ブロック FLASH_BQ スレッド。
        const int  qblocks = (Sq + FLASH_BQ - 1) / FLASH_BQ;
        const dim3 grid(static_cast<unsigned>(qblocks),
                        static_cast<unsigned>(static_cast<long>(B) * H), 1);

        // shared: K/V タイル (FLASH_BK*Dh 要素ずつ) を FP16 で 2 本。
        const size_t shmem = static_cast<size_t>(FLASH_BK) * Dh * 2 * sizeof(__half);

        attention_flash_fp16<<<grid, FLASH_BQ, shmem>>>(d_q, d_k, d_v, d_out,
                                                        B, H, Sq, Sk, Dh, scale);
        CUDA_CHECK_KERNEL();
        return;
    }

    // ---- naive 経路 (Dh > 128, VAE mid_block) ----
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

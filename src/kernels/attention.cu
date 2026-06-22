// Attention カーネル実装 (Phase 2 マイルストーン 2-2-5 / S2 flash 化 / S3-C wmma 化)
// 対象: RTX5080 (Blackwell / sm_120) / CUDA Runtime API のみ
//
// 設計判断 (後続カーネルでも参照する):
//   - 入出力 FP16 / 内部計算 (内積・softmax・PV 積和) は必ず FP32 蓄積。最後に
//     __float2half で書き戻す (GEMM / GroupNorm / Conv と同規約)。
//   - 経路は 3 本:
//       (A) naive (attention_fp16): 1 ブロック = 1 つの (b,h,query 行 i)。scores[Sk]
//           を動的 shared に materialize し 2 パス softmax。Dh が大きい VAE mid_block
//           (Dh=512) 用フォールバック。
//       (B) flash (attention_flash_fp16): online softmax。scores を materialize せず
//           K/V を Bk 行ずつタイル走査。1 スレッド = 1 query 行。Dh<=128 の汎用版。
//       (C) flash + wmma (attention_flash_wmma_fp16): online softmax を warp-tile
//           (16x16) で書き直し、QK^T と P·V を Tensor Core (wmma) で計算する。
//           Dh が 16 の倍数 (UNet self/cross の Dh=64) のときの主役。S3-C で追加。
//   - self / cross は同一カーネルで扱う。違いは Sq と Sk (および Dh) のみ。
//   - reduction ヘルパー (warp/block の sum/max) は本 TU 内に閉じる (utils.cuh 不変)。
//   - 経路選択は launch_attention で Dh により行う:
//       Dh が 16 の倍数 かつ wmma flash の shared が SM 上限内 → flash+wmma (C)
//         (S3-A: VAE mid_block Dh=512 もここに乗せる。shared 81.6KB を opt-in 確保)
//       それ以外で Dh<=128       → flash (B)
//       上記いずれにも乗らない    → naive (A) フォールバック
#include "kernels/attention.cuh"
#include "kernels/utils.cuh"

#include <cuda_fp16.h>
#include <mma.h>

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

// ================================================================
// flash-attention + wmma 経路 (S3-C: Tensor Core 化)
// ================================================================
// 設計 (PL レビュー確定):
//   - per-thread flash (B) は 1 スレッド = 1 query 行で q_reg[Dh]/acc[Dh] をスレッド
//     ローカル保持するため wmma (warp 単位 16x16 タイル協調) と非互換。そこで flash を
//     warp-tile online-softmax として書き直す。
//   - 1 ブロック = 1 warp (32 スレッド) = 1 つの (b, h) の query 行 16 行ブロック (WM_BQ)。
//     gridDim.x = ceil(Sq/16)、gridDim.y = B*H。
//   - QK^T: Q[16,Dh] x K^T[Dh,16] を wmma で計算 (Dh を 16 ずつ K 次元に分割し蓄積)。
//     accumulator は FP32 fragment。結果スコア S[16,16] を shared に FP32 で書く。
//   - online softmax の m/l 補正は FP32 のまま (SSIM 条件)。S に scale を掛け、行 max を
//     warp で求め、P=exp(S-m_new) を計算。acc[16][Dh] (shared FP32) を corr で行スケール。
//   - P·V: P[16,16] (FP16 にダウンキャスト) x V[16,Dh] を wmma で計算し、その FP32
//     accumulator (16xDh) を shared acc に加算。
//   - cross の Sk=77 は 16 の倍数でない → 末尾タイルは K/V/P を 0 パディング。スコアの
//     パディング列は -INF にして exp→0 にし、softmax に混入させない (マスク処理)。
//   - 末尾 query 行タイル (Sq が 16 の倍数でない) も行マスクで書き戻しを抑止。
//
// 対応範囲: Dh が 16 の倍数。UNet self/cross の Dh=64 が主役。S3-A で VAE
//   mid_block の Dh=512 もここで処理する (acc/K/V/P を動的 shared に持つため Dh
//   非依存。上限は確保できる shared バイト数のみ。launch 側で判定)。
// ----------------------------------------------------------------

using namespace nvcuda;

// wmma タイル寸法 (16x16x16 固定)。
static constexpr int WM_M   = 16; // query 行タイル
static constexpr int WM_N   = 16; // K 行タイル (= スコア列)
static constexpr int WM_K   = 16; // 内積分割幅 (Dh / 16 回)
static constexpr int WM_BQ  = 16; // 1 ブロックが担当する query 行数 (= WM_M)
static constexpr int WM_BK  = 16; // K/V タイル行数 (= WM_N)
// (旧) wmma flash の Dh 上限。S3-A で shared バイト数判定に置き換え未使用化。
//   定義は履歴と FLASH_MAX_DH との対比のため残す。
static constexpr int WM_MAX_DH = 128;

__global__ void attention_flash_wmma_fp16(const __half* __restrict__ q,
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
    // blockIdx.y = (b*H + h)、blockIdx.x = query 行タイル番号 (16 行)。
    const long bh       = blockIdx.y;
    const int  row_base = static_cast<int>(blockIdx.x) * WM_BQ;

    const long q_head = bh * Sq * Dh;
    const long k_head = bh * Sk * Dh;
    const long v_head = bh * Sk * Dh;
    const long o_head = bh * Sq * Dh;

    const int tid  = static_cast<int>(threadIdx.x); // 0..31 (1 warp)
    const int lane = tid;                            // warp 内レーン

    // この query 行タイルの有効行数 (末尾タイルでは <16)。
    const int q_rows = min(WM_BQ, Sq - row_base);

    // ---- shared レイアウト ----
    // Q_tile : [WM_BQ][Dh]      (FP16)  K^T 入力用に row-major (Q[m,d])
    // K_tile : [WM_BK][Dh]      (FP16)  K[n,d] row-major
    // V_tile : [WM_BK][Dh]      (FP16)  V[n,d] row-major
    // P_tile : [WM_BQ][WM_BK]   (FP16)  ダウンキャスト済み確率
    // S_tile : [WM_BQ][WM_BK]   (FP32)  QK^T スコア (一時)
    // acc    : [WM_BQ][Dh]      (FP32)  出力アキュムレータ
    // m_row  : [WM_BQ]          (FP32)  running max
    // l_row  : [WM_BQ]          (FP32)  running sum
    extern __shared__ unsigned char smem_raw[];
    __half* q_tile = reinterpret_cast<__half*>(smem_raw);
    __half* k_tile = q_tile + WM_BQ * Dh;
    __half* v_tile = k_tile + WM_BK * Dh;
    __half* p_tile = v_tile + WM_BK * Dh;
    // FP32 領域は __half 領域の直後 (16B 境界に揃う: 上の __half 数は全て偶数)。
    float*  s_tile = reinterpret_cast<float*>(p_tile + WM_BQ * WM_BK);
    float*  acc    = s_tile + WM_BQ * WM_BK;
    float*  m_row  = acc + WM_BQ * Dh;
    float*  l_row  = m_row + WM_BQ;

    // ---- 初期化 ----
    // acc を 0、m を -INF、l を 0 に。warp 32 レーンで分担。
    for (int e = lane; e < WM_BQ * Dh; e += 32)
    {
        acc[e] = 0.0f;
    }
    if (lane < WM_BQ)
    {
        m_row[lane] = -INFINITY;
        l_row[lane] = 0.0f;
    }

    // Q タイルを shared にロード (有効行のみ。無効行は 0 パディング)。
    for (int e = lane; e < WM_BQ * Dh; e += 32)
    {
        const int m = e / Dh;
        const int d = e - m * Dh;
        const int qi = row_base + m;
        q_tile[e] = (qi < Sq) ? q[q_head + static_cast<long>(qi) * Dh + d]
                              : __float2half(0.0f);
    }
    __syncwarp();

    // K 次元の分割数 (Dh / 16)。
    const int kchunks = Dh / WM_K;

    // ---- K/V を WM_BK 行ずつタイル走査 (online softmax) ----
    for (int j0 = 0; j0 < Sk; j0 += WM_BK)
    {
        const int tile_rows = min(WM_BK, Sk - j0);

        // K/V タイルを shared にロード (パディング行は 0)。
        for (int e = lane; e < WM_BK * Dh; e += 32)
        {
            const int n = e / Dh;
            const int d = e - n * Dh;
            const int kj = j0 + n;
            if (kj < Sk)
            {
                k_tile[e] = k[k_head + static_cast<long>(kj) * Dh + d];
                v_tile[e] = v[v_head + static_cast<long>(kj) * Dh + d];
            }
            else
            {
                k_tile[e] = __float2half(0.0f);
                v_tile[e] = __float2half(0.0f);
            }
        }
        __syncwarp();

        // ---- QK^T を wmma で。S[16,16] = Q[16,Dh] * K^T[Dh,16] ----
        // K は row-major [n][d] なので、wmma の b 行列を col_major で K^T として読む:
        //   K^T[d, n] = K[n, d]。col_major fragment は「列が連続」を仮定するので、
        //   K[n][d] (n が行) を col_major で渡すと b[d][n]=K[n][d] と解釈され狙い通り。
        wmma::fragment<wmma::accumulator, WM_M, WM_N, WM_K, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);
        for (int kc = 0; kc < kchunks; ++kc)
        {
            wmma::fragment<wmma::matrix_a, WM_M, WM_N, WM_K, __half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WM_M, WM_N, WM_K, __half, wmma::col_major> b_frag;
            // a = Q[0..15][kc*16 .. kc*16+15], leading dim = Dh。
            wmma::load_matrix_sync(a_frag, q_tile + kc * WM_K, Dh);
            // b = K^T。K_tile は [n][d]、col_major で leading dim = Dh、起点は列 d=kc*16。
            wmma::load_matrix_sync(b_frag, k_tile + kc * WM_K, Dh);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        // S_tile に FP32 で書く (row-major)。
        wmma::store_matrix_sync(s_tile, c_frag, WM_BK, wmma::mem_row_major);
        __syncwarp();

        // ---- online softmax 補正 (FP32, 各 query 行ごと) ----
        // 1 行を 1 レーンが担当 (WM_BQ=16 <= 32)。パディング列/行はマスク。
        if (lane < WM_BQ)
        {
            const int m = lane;
            const bool row_valid = (m < q_rows);

            // この行の scaled スコア最大 (パディング列は除外)。
            float tile_max = -INFINITY;
            for (int n = 0; n < tile_rows; ++n)
            {
                const float s = scale * s_tile[m * WM_BK + n];
                s_tile[m * WM_BK + n] = s; // scale 済みを書き戻し (後段で再利用)
                tile_max = fmaxf(tile_max, s);
            }
            // パディング列を -INF にしておく (P=0 になる)。
            for (int n = tile_rows; n < WM_BK; ++n)
            {
                s_tile[m * WM_BK + n] = -INFINITY;
            }

            const float m_old = m_row[m];
            const float m_new = fmaxf(m_old, tile_max);
            const float corr  = (m_old == -INFINITY) ? 0.0f : __expf(m_old - m_new);

            // 既存 acc をこの行について corr でスケール。
            for (int d = 0; d < Dh; ++d)
            {
                acc[m * Dh + d] *= corr;
            }

            // P[m][n] = exp(s - m_new) を p_tile に FP16 で。l を更新。
            float l_new = l_row[m] * corr;
            for (int n = 0; n < WM_BK; ++n)
            {
                const float s = s_tile[m * WM_BK + n];
                const float p = (s == -INFINITY) ? 0.0f : __expf(s - m_new);
                p_tile[m * WM_BK + n] = __float2half(row_valid ? p : 0.0f);
                l_new += (row_valid ? p : 0.0f);
            }
            m_row[m] = row_valid ? m_new : m_old;
            l_row[m] = l_new;
        }
        __syncwarp();

        // ---- P·V を wmma で。acc[16,Dh] += P[16,16] * V[16,Dh] ----
        // Dh を 16 列ずつ出力タイルに分けて計算 (Dh/16 = kchunks タイル)。
        // P[m][n] row_major、V[n][d] row_major → そのまま a (row_major), b (row_major)。
        for (int dc = 0; dc < kchunks; ++dc)
        {
            wmma::fragment<wmma::accumulator, WM_M, WM_N, WM_K, float> o_frag;
            wmma::fill_fragment(o_frag, 0.0f);

            wmma::fragment<wmma::matrix_a, WM_M, WM_N, WM_K, __half, wmma::row_major> pa;
            wmma::fragment<wmma::matrix_b, WM_M, WM_N, WM_K, __half, wmma::row_major> vb;
            // a = P[0..15][0..15], leading dim = WM_BK。
            wmma::load_matrix_sync(pa, p_tile, WM_BK);
            // b = V[0..15][dc*16 .. dc*16+15], leading dim = Dh。
            wmma::load_matrix_sync(vb, v_tile + dc * WM_K, Dh);
            wmma::mma_sync(o_frag, pa, vb, o_frag);

            // o_frag (16x16) を一時 shared (s_tile を再利用) に書いて acc に加算。
            wmma::store_matrix_sync(s_tile, o_frag, WM_BK, wmma::mem_row_major);
            __syncwarp();
            for (int e = lane; e < WM_M * WM_K; e += 32)
            {
                const int m = e / WM_K;
                const int d = e - m * WM_K;
                acc[m * Dh + dc * WM_K + d] += s_tile[m * WM_BK + d];
            }
            __syncwarp();
        }
        __syncwarp();
    }

    // ---- 正規化して書き戻す (有効行のみ) ----
    for (int e = lane; e < WM_BQ * Dh; e += 32)
    {
        const int m  = e / Dh;
        const int d  = e - m * Dh;
        const int qi = row_base + m;
        if (qi < Sq)
        {
            const float l    = l_row[m];
            const float invl = (l > 0.0f) ? (1.0f / l) : 0.0f;
            out[o_head + static_cast<long>(qi) * Dh + d] = __float2half(acc[m * Dh + d] * invl);
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
    // (C) flash+wmma: Dh が 16 の倍数 かつ wmma flash の shared が SM 上限に収まる。
    //     UNet self/cross の Dh=64 主役。S3-A で VAE mid_block の Dh=512 もここに乗せる
    //     (kernel は Dh を動的 shared から取るため Dh 汎用。WM_MAX_DH 制約は撤廃し、
    //      実際に確保できる shared バイト数だけで判定する)。
    // (B) flash:      その他 Dh<=128。
    // (A) naive:      上記いずれにも乗らない Dh (フォールバック)。
    if ((Dh % WM_K) == 0)
    {
        // wmma flash の動的 shared サイズ:
        //   __half: Q(WM_BQ*Dh) + K(WM_BK*Dh) + V(WM_BK*Dh) + P(WM_BQ*WM_BK)
        //   float : S(WM_BQ*WM_BK) + acc(WM_BQ*Dh) + m(WM_BQ) + l(WM_BQ)
        const size_t half_elems =
            static_cast<size_t>(WM_BQ) * Dh + static_cast<size_t>(WM_BK) * Dh * 2
            + static_cast<size_t>(WM_BQ) * WM_BK;
        const size_t float_elems =
            static_cast<size_t>(WM_BQ) * WM_BK + static_cast<size_t>(WM_BQ) * Dh
            + 2 * static_cast<size_t>(WM_BQ);
        const size_t shmem = half_elems * sizeof(__half) + float_elems * sizeof(float);

        // sm_120 の SM あたり shared 上限 (~227KB) を超える Dh は wmma flash に乗せられない
        // ため naive へフォールバックさせる。現行モデルの最大 Dh=512 は 81.6KB で収まる。
        constexpr size_t kMaxOptinShmem = 224u * 1024u;
        if (shmem <= kMaxOptinShmem)
        {
            // グリッド: x = query 行タイル数 (16 行), y = B*H。各ブロック 1 warp (32 スレッド)。
            const int  qtiles = (Sq + WM_BQ - 1) / WM_BQ;
            const dim3 grid(static_cast<unsigned>(qtiles),
                            static_cast<unsigned>(static_cast<long>(B) * H), 1);

            // 動的 shared が既定 48KB を超える Dh (=128 で 49KB / VAE mid Dh=512 で 81.6KB)
            // は cudaFuncSetAttribute で MaxDynamicSharedMemorySize を引き上げる (opt-in は
            // 冪等)。48KB 以下のときは呼ばない (UNet Dh=64 の既存挙動を不変に保つ)。
            constexpr size_t kDefaultShmemLimit = 48u * 1024u;
            if (shmem > kDefaultShmemLimit)
            {
                CUDA_CHECK(cudaFuncSetAttribute(
                    attention_flash_wmma_fp16,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(shmem)));
            }

            attention_flash_wmma_fp16<<<grid, 32, shmem>>>(d_q, d_k, d_v, d_out,
                                                           B, H, Sq, Sk, Dh, scale);
            CUDA_CHECK_KERNEL();
            return;
        }
        // shmem 超過時は下の flash / naive へフォールバック。
    }

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

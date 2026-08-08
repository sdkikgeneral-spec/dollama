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


// ================================================================
// G-4k S1a: multi-block GroupNorm (epilogue=on 経路専用)
// ================================================================
// 上の 1-block 版は grid = N*num_groups (B=1 で 32 block) しか立たず、RTX5080 の
// 84 SM の大半が遊休 → 実測 73 GB/s (理論 ~960 GB/s の 8%) の構造要因だった
// (冒頭コメント 18-20 行の宿題)。ここでは 1 グループを複数ブロックで分担する
// 「2 段の決定的集約」へ昇格する:
//   カーネル1 (partial):   グループを bpg 個のブロックで分担し、各ブロックが
//                          block_reduce_sum で (sum, sum-sq) の部分和を中間バッファへ書く。
//   カーネル2 (finalize):  1 グループ = 1 ブロックが bpg 個の部分和を固定順で集約し
//                          mean / inv_std を確定してバッファへ書く。
//   カーネル3 (normalize): 全要素 grid-stride の純 map で正規化 + affine を書き戻す。
//
// 決定性 (非交渉ゲート): atomicAdd は浮動小数の加算順が run-to-run で変わり
// テストが非決定になるため使わない。上記 3 カーネルはいずれも grid/block 形状と
// リダクション木が入力形状のみで決まる固定順なので、同一入力 → 完全ビット一致。
//
// default との関係: multi-block は 1-block 版と蓄積順が異なるため FP32 蓄積でも
// 数値が微妙にずれる (FP16 出力で高々 1 ULP 級)。よって default (epilogue=off) は
// 従来の launch_group_norm を byte-for-byte 維持し、本経路は epilogue フラグの
// 裏に隔離する (呼び分けは src/infer/unet.cu)。
//
// 中間バッファの確保方針 (G-8k との整合):
//   必要量は NG*bpg*2 + NG*2 float。UNet 実形状の上界は NG = B*32 ≤ 64、
//   bpg ≤ GN_MB_MAX_BPG = 128 なので 64*128*2 + 64*2 = 16,512 float ≈ 65KB と極小。
//   step ループ内の cudaMalloc/cudaFree (G-8k で撲滅対象の既知問題) を新たに
//   増やさないため、本 TU 内の grow-only 静的常駐バッファとする (初回のみ確保、
//   以降は使い回し。プロセス終了までデバイス常駐 = 65KB は VRAM 誤差)。
//   Scratch プールを使わない理由: Scratch は FP16 プールで float 部分和と型が
//   合わず、段スコープ生成のため毎回 cudaMalloc/cudaFree が発生してしまう。
//   注意: 拡散オーケストレーションは単一スレッド前提 (プロジェクト全体の規約)。
// ----------------------------------------------------------------

// multi-block 版のスレッド数 (1-block 版と同じ 32 の倍数)。
static constexpr int GN_MB_THREADS = 256;
// 1 グループあたりのブロック数上限 (バッファ上界と finalize の集約幅を抑える)。
static constexpr int GN_MB_MAX_BPG = 128;
// 1 スレッドが partial パスで担う目安要素数 (bpg の算出に使う)。
static constexpr int GN_MB_ELEMS_PER_THREAD = 8;

// grow-only 静的常駐バッファ (partial sums + stats)。上記コメント参照。
static float* g_mb_buf        = nullptr;
static size_t g_mb_buf_floats = 0;

static float* mb_buffer(size_t floats)
{
    if (floats > g_mb_buf_floats)
    {
        if (g_mb_buf != nullptr)
        {
            CUDA_CHECK(cudaFree(g_mb_buf));
            g_mb_buf = nullptr;
        }
        CUDA_CHECK(cudaMalloc(&g_mb_buf, floats * sizeof(float)));
        g_mb_buf_floats = floats;
    }
    return g_mb_buf;
}

// ----------------------------------------------------------------
// カーネル1: 部分和。blockIdx.x = gid * bpg + blk (gid = n*num_groups + g)。
// 各ブロックはグループ内をインターリーブ (stride = bpg*blockDim) で舐める
// (連続アドレスを warp が読む coalesced 配置)。
// ----------------------------------------------------------------
__global__ void group_norm_mb_partial(const __half* in,
                                      float*        partial,
                                      int           C,
                                      int           HW,
                                      int           cpg,
                                      int           num_groups,
                                      int           bpg)
{
    const int gid = blockIdx.x / bpg;   // n * num_groups + g
    const int blk = blockIdx.x % bpg;
    const int g   = gid % num_groups;
    const int n   = gid / num_groups;

    const int  c_begin    = g * cpg;
    const int  group_size = cpg * HW;
    const long base = (static_cast<long>(n) * C + c_begin) * static_cast<long>(HW);

    __shared__ float smem[GN_MB_THREADS / 32];

    float local_sum = 0.0f;
    float local_sq  = 0.0f;
    const int stride = bpg * blockDim.x;
    for (int i = blk * blockDim.x + threadIdx.x; i < group_size; i += stride)
    {
        const float x = __half2float(in[base + i]);
        local_sum += x;
        local_sq  += x * x;
    }

    const float sum    = block_reduce_sum(local_sum, smem);
    const float sum_sq = block_reduce_sum(local_sq, smem);

    if (threadIdx.x == 0)
    {
        const long slot = (static_cast<long>(gid) * bpg + blk) << 1;
        partial[slot]     = sum;
        partial[slot + 1] = sum_sq;
    }
}

// ----------------------------------------------------------------
// カーネル2: 集約。1 グループ = 1 ブロックが bpg 個の部分和を固定順で読み、
// mean / inv_std を確定して stats[gid*2] へ書く。
// ----------------------------------------------------------------
__global__ void group_norm_mb_finalize(const float* partial,
                                       float*       stats,
                                       int          bpg,
                                       int          group_size,
                                       float        eps)
{
    const int gid = blockIdx.x;

    __shared__ float smem[GN_MB_THREADS / 32];

    float local_sum = 0.0f;
    float local_sq  = 0.0f;
    for (int j = threadIdx.x; j < bpg; j += blockDim.x)
    {
        const long slot = (static_cast<long>(gid) * bpg + j) << 1;
        local_sum += partial[slot];
        local_sq  += partial[slot + 1];
    }

    const float sum    = block_reduce_sum(local_sum, smem);
    const float sum_sq = block_reduce_sum(local_sq, smem);

    if (threadIdx.x == 0)
    {
        const float inv_k = 1.0f / static_cast<float>(group_size);
        const float mean  = sum * inv_k;
        float var = sum_sq * inv_k - mean * mean;
        var = fmaxf(var, 0.0f);
        stats[gid << 1]       = mean;
        stats[(gid << 1) + 1] = rsqrtf(var + eps);
    }
}

// ----------------------------------------------------------------
// カーネル3: 正規化 + affine。全要素 grid-stride の純 map (リダクションなし =
// 自明に決定的)。stats は確定済みなので d_in == d_out の in-place も安全。
// ----------------------------------------------------------------
__global__ void group_norm_mb_normalize(const __half* in,
                                        const __half* gamma,
                                        const __half* beta,
                                        const float*  stats,
                                        __half*       out,
                                        int           C,
                                        int           HW,
                                        int           cpg,
                                        int           num_groups,
                                        long          total)
{
    const long chw = static_cast<long>(C) * HW;
    for (long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
         idx += static_cast<long>(gridDim.x) * blockDim.x)
    {
        const int n   = static_cast<int>(idx / chw);
        const int c   = static_cast<int>((idx - static_cast<long>(n) * chw) / HW);
        const int gid = n * num_groups + c / cpg;

        const float mean    = stats[gid << 1];
        const float inv_std = stats[(gid << 1) + 1];
        const float x  = __half2float(in[idx]);
        const float gm = __half2float(gamma[c]);
        const float bt = __half2float(beta[c]);
        out[idx] = __float2half((x - mean) * inv_std * gm + bt);
    }
}

// ----------------------------------------------------------------
// ホストラッパー (multi-block 版)。引数仕様は launch_group_norm と同一。
// ----------------------------------------------------------------
void launch_group_norm_mb(const __half* d_in,
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
    assert(C % num_groups == 0);

    const int cpg        = C / num_groups;
    const int HW         = H * W;
    const int group_size = cpg * HW;
    const int NG         = N * num_groups;

    // 1 グループあたりのブロック数: 1 スレッド ~8 要素を目安に立て、上限でクランプ。
    int bpg = ceil_div(group_size, GN_MB_THREADS * GN_MB_ELEMS_PER_THREAD);
    if (bpg > GN_MB_MAX_BPG)
    {
        bpg = GN_MB_MAX_BPG;
    }
    if (bpg < 1)
    {
        bpg = 1;
    }

    // 常駐バッファ: [NG*bpg*2] partial + [NG*2] stats。
    const size_t part_floats = (static_cast<size_t>(NG) * bpg) << 1;
    float* buf     = mb_buffer(part_floats + (static_cast<size_t>(NG) << 1));
    float* d_part  = buf;
    float* d_stats = buf + part_floats;

    group_norm_mb_partial<<<NG * bpg, GN_MB_THREADS>>>(d_in, d_part, C, HW, cpg,
                                                       num_groups, bpg);
    CUDA_CHECK_KERNEL();

    group_norm_mb_finalize<<<NG, GN_MB_THREADS>>>(d_part, d_stats, bpg, group_size, eps);
    CUDA_CHECK_KERNEL();

    const long total = static_cast<long>(N) * C * HW;
    const long bl    = (total + GN_MB_THREADS - 1) / GN_MB_THREADS;
    const int blocks = static_cast<int>(bl < 65535 ? bl : 65535);
    group_norm_mb_normalize<<<blocks, GN_MB_THREADS>>>(d_in, d_gamma, d_beta, d_stats,
                                                       d_out, C, HW, cpg, num_groups, total);
    CUDA_CHECK_KERNEL();
}

// ================================================================
// G-4k S1b: multi-block GroupNorm + SiLU 融合 (epilogue=on 経路専用)
// ================================================================
// resnet_block の GN→SiLU は「mb 版 normalize (write) → launch_silu (read+write)」の
// 2 パスで、SiLU ぶんの往復 (read 1 + write 1) が丸ごと余分だった。normalize の
// 末尾 map に SiLU を畳んで 1 パス化する。partial / finalize は mb 版を共用。
//
// bit-exact パリティ (非交渉ゲート):
//   既存 launch_silu (activation.cu silu_fp16) は __half 入力を float へ戻して
//   x / (1 + __expf(-x)) を計算し __half で書く。融合版もこの丸め順を厳密に踏む:
//   GN 結果 y を __float2half で half へ丸め → その half を float に戻して
//   同式・同精度 (__expf) の SiLU → __float2half で書く。これで
//   「launch_group_norm_mb → launch_silu」の 2 パスと完全ビット一致する。
// 決定性: SiLU は要素ごとの純 map (reduction なし) なので、mb 版の
//   run-to-run 完全ビット一致はそのまま維持される。
// ----------------------------------------------------------------
__global__ void group_norm_mb_normalize_silu(const __half* in,
                                             const __half* gamma,
                                             const __half* beta,
                                             const float*  stats,
                                             __half*       out,
                                             int           C,
                                             int           HW,
                                             int           cpg,
                                             int           num_groups,
                                             long          total)
{
    const long chw = static_cast<long>(C) * HW;
    for (long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
         idx += static_cast<long>(gridDim.x) * blockDim.x)
    {
        const int n   = static_cast<int>(idx / chw);
        const int c   = static_cast<int>((idx - static_cast<long>(n) * chw) / HW);
        const int gid = n * num_groups + c / cpg;

        const float mean    = stats[gid << 1];
        const float inv_std = stats[(gid << 1) + 1];
        const float x  = __half2float(in[idx]);
        const float gm = __half2float(gamma[c]);
        const float bt = __half2float(beta[c]);

        // GN 結果を一旦 __half へ丸める (launch_silu が half 入力を読むのと等価)。
        const __half hy = __float2half((x - mean) * inv_std * gm + bt);

        // silu_fp16 (activation.cu) と同式・同精度: v / (1 + __expf(-v))。
        const float v   = __half2float(hy);
        const float sig = 1.0f / (1.0f + __expf(-v));
        out[idx] = __float2half(v * sig);
    }
}

// ----------------------------------------------------------------
// ホストラッパー (GN+SiLU 融合版)。引数仕様は launch_group_norm_mb と完全同一。
// partial / finalize / バッファは mb 版を共用し、normalize のみ SiLU 融合版に差し替え。
// ----------------------------------------------------------------
void launch_group_norm_silu(const __half* d_in,
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
    assert(C % num_groups == 0);

    const int cpg        = C / num_groups;
    const int HW         = H * W;
    const int group_size = cpg * HW;
    const int NG         = N * num_groups;

    // bpg 算出は mb 版と同一 (stats の蓄積順が一致 = GN 部の bit-exact 前提)。
    int bpg = ceil_div(group_size, GN_MB_THREADS * GN_MB_ELEMS_PER_THREAD);
    if (bpg > GN_MB_MAX_BPG)
    {
        bpg = GN_MB_MAX_BPG;
    }
    if (bpg < 1)
    {
        bpg = 1;
    }

    const size_t part_floats = (static_cast<size_t>(NG) * bpg) << 1;
    float* buf     = mb_buffer(part_floats + (static_cast<size_t>(NG) << 1));
    float* d_part  = buf;
    float* d_stats = buf + part_floats;

    group_norm_mb_partial<<<NG * bpg, GN_MB_THREADS>>>(d_in, d_part, C, HW, cpg,
                                                       num_groups, bpg);
    CUDA_CHECK_KERNEL();

    group_norm_mb_finalize<<<NG, GN_MB_THREADS>>>(d_part, d_stats, bpg, group_size, eps);
    CUDA_CHECK_KERNEL();

    const long total = static_cast<long>(N) * C * HW;
    const long bl    = (total + GN_MB_THREADS - 1) / GN_MB_THREADS;
    const int blocks = static_cast<int>(bl < 65535 ? bl : 65535);
    group_norm_mb_normalize_silu<<<blocks, GN_MB_THREADS>>>(d_in, d_gamma, d_beta, d_stats,
                                                            d_out, C, HW, cpg, num_groups,
                                                            total);
    CUDA_CHECK_KERNEL();
}

} // namespace dollama

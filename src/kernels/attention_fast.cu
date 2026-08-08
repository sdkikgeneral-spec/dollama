// Attention 高速バリアント実装 (FAST モード G-3k / FlashAttention-2 級)
// 対象: RTX5080 (Blackwell / sm_120) / CUDA Runtime API のみ
//
// 位置づけ:
//   既存 launch_attention (kernels/attention.cu) は 1 block = 1 warp (32 スレッド) で
//   占有率が低く、SDXL UNet attention の単一最大律速 (実測 40.3% / 4.621s) になっている。
//   本 TU はその代替となる launch_attention_fast を「別ファイル」で追加する。既存
//   attention.cu は一切改変しない (default 経路を byte-for-byte 温存 = 回帰アンカー保存)。
//
// 高速化の芯 (数学は既存と同一 = O = softmax(scale·Q·Kᵀ)·V, FP32 蓄積):
//   1. multi-warp block: 1 block = N warp。各 warp が自分の 16 行 query タイルを担当し、
//      K/V タイルは block 内 N warp で共有ロード。K/V の DRAM 再ロードを N 分の 1 に
//      償却し、占有率 (稼働 warp 数) を N 倍にして SM を埋める。
//   2. cp.async のダブルバッファ: K/V タイルのロードと wmma 計算をパイプライン化
//      (sm_120 の非同期コピー)。現タイルを計算する裏で次タイルをプリフェッチ。
//   3. QK^T / P·V を Tensor Core (wmma 16x16x16) で計算 (既存 wmma 経路を踏襲)。
//
// reduction 順が既存と変わるため golden とは微差になるが、FP32 蓄積を維持するので
// SSIM ゲート (>=0.9999) 内に収まる。
//
// 対応範囲: Dh が 16 の倍数 (UNet self/cross の Dh=64/80 が主役。cp.async は 16 バイト
//   = 8 half 単位なので Dh は 8 の倍数でもある必要があり、16 の倍数がこれを含意する)。
//   それ以外の Dh、または確保可能な shared を超える大 Dh は既存 launch_attention へ
//   フォールバックする (呼び側は常に launch_attention_fast を呼べばよい)。
#include "kernels/attention.cuh"
#include "kernels/utils.cuh"

#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_pipeline.h>
#include <initializer_list>

namespace dollama
{

using namespace nvcuda;

namespace
{
// wmma タイル寸法 (16x16x16 固定)。
constexpr int WF_M = 16; // query 行タイル
constexpr int WF_N = 16; // K 行タイル (= スコア列)
constexpr int WF_K = 16; // 内積分割幅 (Dh / 16 回)
// 1 warp が担当する query 行数 (= WF_M)。
constexpr int WF_WARP_ROWS = 16;
// K/V タイル行数 (= WF_N)。
constexpr int WF_KV_ROWS = 16;
} // namespace

// ----------------------------------------------------------------
// multi-warp + cp.async ダブルバッファ + wmma の flash-attention カーネル。
//   1 block = nwarps warp。blockDim.x = nwarps*32。
//   1 block が担当する query 行 = nwarps*16 行 (warp w が [w*16, w*16+16))。
//   grid.x = ceil(Sq / (nwarps*16)), grid.y = B*H。
//   nwarps は launch 側が shared 制約から動的に決めて blockDim で渡す (kernel は
//   blockDim.x/32 から読む)。動的 shared のレイアウトも Dh と nwarps から実行時算出。
// ----------------------------------------------------------------
__global__ void attention_flash_wmma_fast_fp16(const __half* __restrict__ q,
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
    const int nwarps   = blockDim.x / 32;
    const int warp     = threadIdx.x / 32;
    const int lane     = threadIdx.x % 32;
    const int tid      = static_cast<int>(threadIdx.x);
    const int nthreads = static_cast<int>(blockDim.x);

    const long bh             = blockIdx.y;
    const int  block_rows     = nwarps * WF_WARP_ROWS;
    const int  block_row_base = static_cast<int>(blockIdx.x) * block_rows;
    const int  warp_row_base  = block_row_base + warp * WF_WARP_ROWS;

    const long q_head = bh * Sq * Dh;
    const long k_head = bh * Sk * Dh;
    const long v_head = bh * Sk * Dh;
    const long o_head = bh * Sq * Dh;

    // ---- 動的 shared レイアウト ----
    // half : q_tile[block_rows*Dh]
    //        | kbuf[2][WF_KV_ROWS*Dh] | vbuf[2][WF_KV_ROWS*Dh]  (ダブルバッファ)
    //        | p_tile[nwarps*WF_M*WF_N]                          (warp ごとの確率)
    // float: s_tile[nwarps*WF_M*WF_N] (warp ごとの一時)
    //        | acc[block_rows*Dh]     (出力アキュムレータ)
    //        | m_row[block_rows] | l_row[block_rows]             (online softmax)
    // half 領域の要素数はすべて 16B の倍数になるよう Dh を 16 の倍数に制約しているため、
    // 続く float 領域も自然に 16B 境界に揃う。
    extern __shared__ unsigned char smem_raw[];
    __half* q_tile = reinterpret_cast<__half*>(smem_raw);
    __half* kbuf0  = q_tile + static_cast<long>(block_rows) * Dh;
    __half* kbuf1  = kbuf0 + static_cast<long>(WF_KV_ROWS) * Dh;
    __half* vbuf0  = kbuf1 + static_cast<long>(WF_KV_ROWS) * Dh;
    __half* vbuf1  = vbuf0 + static_cast<long>(WF_KV_ROWS) * Dh;
    __half* p_tile = vbuf1 + static_cast<long>(WF_KV_ROWS) * Dh;
    float*  s_tile = reinterpret_cast<float*>(p_tile + static_cast<long>(nwarps) * WF_M * WF_N);
    float*  acc    = s_tile + static_cast<long>(nwarps) * WF_M * WF_N;
    float*  m_row  = acc + static_cast<long>(block_rows) * Dh;
    float*  l_row  = m_row + block_rows;

    __half* kbuf[2] = { kbuf0, kbuf1 };
    __half* vbuf[2] = { vbuf0, vbuf1 };

    // ---- 初期化 (block 全スレッドで分担) ----
    for (int e = tid; e < block_rows * Dh; e += nthreads)
    {
        acc[e] = 0.0f;
    }
    for (int r = tid; r < block_rows; r += nthreads)
    {
        m_row[r] = -INFINITY;
        l_row[r] = 0.0f;
    }
    // Q タイルロード。範囲外行 (Sq を超える末尾) は最終有効行にクランプ (有限値を保証)。
    // 出力側で qi<Sq のマスクにより書き戻さないので値は問わない。
    for (int e = tid; e < block_rows * Dh; e += nthreads)
    {
        const int r  = e / Dh;
        const int d  = e - r * Dh;
        int       qi = block_row_base + r;
        qi = (qi < Sq) ? qi : (Sq - 1);
        q_tile[e] = q[q_head + static_cast<long>(qi) * Dh + d];
    }
    __syncthreads();

    const int kchunks = Dh / WF_K;
    const int ntiles  = (Sk + WF_KV_ROWS - 1) / WF_KV_ROWS;

    // K/V タイルを cp.async でロードするヘルパ (16 バイト = 8 half 単位)。
    // 範囲外行 (kj>=Sk) は最終有効行 (Sk-1) にクランプして OOB 読みを避ける。
    // その行のスコアは後段の列マスクで -INF (P=0) にするため寄与しない。
    auto prefetch_tile = [&](__half* kd, __half* vd, int base_j)
    {
        const int total = WF_KV_ROWS * Dh;
        for (int e = tid * 8; e < total; e += nthreads * 8)
        {
            const int r  = e / Dh;
            const int d  = e - r * Dh;
            int       kj = base_j + r;
            kj = (kj < Sk) ? kj : (Sk - 1);
            __pipeline_memcpy_async(kd + e, k + k_head + static_cast<long>(kj) * Dh + d, 16);
            __pipeline_memcpy_async(vd + e, v + v_head + static_cast<long>(kj) * Dh + d, 16);
        }
    };

    // タイル 0 をプリフェッチ (commit)。
    prefetch_tile(kbuf[0], vbuf[0], 0);
    __pipeline_commit();

    // この warp が担当する有効 query 行数 (末尾 warp/block では <16 or <=0)。
    const int q_rows = min(WF_WARP_ROWS, Sq - warp_row_base);

    // warp ローカルの shared ビュー。
    __half* wq   = q_tile + static_cast<long>(warp) * WF_WARP_ROWS * Dh;
    __half* wp   = p_tile + static_cast<long>(warp) * WF_M * WF_N;
    float*  ws   = s_tile + static_cast<long>(warp) * WF_M * WF_N;
    float*  wacc = acc + static_cast<long>(warp) * WF_WARP_ROWS * Dh;
    float*  wm   = m_row + warp * WF_WARP_ROWS;
    float*  wl   = l_row + warp * WF_WARP_ROWS;

    // ---- K/V を WF_KV_ROWS 行ずつタイル走査 (online softmax) ----
    for (int t = 0; t < ntiles; ++t)
    {
        const int cur       = t & 1;
        const int j0        = t * WF_KV_ROWS;
        const int tile_rows = min(WF_KV_ROWS, Sk - j0);

        // 次タイルをプリフェッチ (存在すれば)。現タイルの完了を待ってから計算に入る。
        if (t + 1 < ntiles)
        {
            prefetch_tile(kbuf[(t + 1) & 1], vbuf[(t + 1) & 1], (t + 1) * WF_KV_ROWS);
            __pipeline_commit();
            __pipeline_wait_prior(1); // 直近 1 バッチ (次タイル) 以外 = 現タイルの完了待ち
        }
        else
        {
            __pipeline_wait_prior(0); // 残り全バッチ (現タイル) の完了待ち
        }
        __syncthreads(); // 現タイルバッファが全 warp から可視であることを保証

        if (q_rows > 0)
        {
            // ---- QK^T: ws[16,16] = wq[16,Dh] * K^T[Dh,16] ----
            // K は row-major [n][d]。col_major fragment を ld=Dh で読むと b[d][n]=K[n][d]=K^T。
            wmma::fragment<wmma::accumulator, WF_M, WF_N, WF_K, float> c_frag;
            wmma::fill_fragment(c_frag, 0.0f);
            for (int kc = 0; kc < kchunks; ++kc)
            {
                wmma::fragment<wmma::matrix_a, WF_M, WF_N, WF_K, __half, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, WF_M, WF_N, WF_K, __half, wmma::col_major> b_frag;
                wmma::load_matrix_sync(a_frag, wq + kc * WF_K, Dh);
                wmma::load_matrix_sync(b_frag, kbuf[cur] + kc * WF_K, Dh);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            wmma::store_matrix_sync(ws, c_frag, WF_N, wmma::mem_row_major);
            __syncwarp();

            // ---- online softmax 補正 (FP32, 1 行 = 1 レーン) ----
            if (lane < WF_WARP_ROWS)
            {
                const int  m         = lane;
                const bool row_valid = (m < q_rows);

                // この行の scaled スコア最大 (パディング列は除外)。
                float tile_max = -INFINITY;
                for (int n = 0; n < tile_rows; ++n)
                {
                    const float s = scale * ws[m * WF_N + n];
                    ws[m * WF_N + n] = s; // scale 済みを書き戻し
                    tile_max = fmaxf(tile_max, s);
                }
                for (int n = tile_rows; n < WF_N; ++n)
                {
                    ws[m * WF_N + n] = -INFINITY; // パディング列 (P=0)
                }

                const float m_old = wm[m];
                const float m_new = fmaxf(m_old, tile_max);
                const float corr  = (m_old == -INFINITY) ? 0.0f : __expf(m_old - m_new);

                // 既存 acc をこの行について corr でスケール。
                for (int d = 0; d < Dh; ++d)
                {
                    wacc[m * Dh + d] *= corr;
                }

                // P[m][n] = exp(s - m_new) を FP16 で。l を更新。
                float l_new = wl[m] * corr;
                for (int n = 0; n < WF_N; ++n)
                {
                    const float s = ws[m * WF_N + n];
                    const float p = (s == -INFINITY) ? 0.0f : __expf(s - m_new);
                    wp[m * WF_N + n] = __float2half(row_valid ? p : 0.0f);
                    l_new += (row_valid ? p : 0.0f);
                }
                wm[m] = row_valid ? m_new : m_old;
                wl[m] = l_new;
            }
            __syncwarp();

            // ---- P·V: wacc[16,Dh] += wp[16,16] * V[16,Dh] ----
            // Dh を 16 列ずつ出力タイルに分けて計算。P/V ともに row_major。
            for (int dc = 0; dc < kchunks; ++dc)
            {
                wmma::fragment<wmma::accumulator, WF_M, WF_N, WF_K, float> o_frag;
                wmma::fill_fragment(o_frag, 0.0f);
                wmma::fragment<wmma::matrix_a, WF_M, WF_N, WF_K, __half, wmma::row_major> pa;
                wmma::fragment<wmma::matrix_b, WF_M, WF_N, WF_K, __half, wmma::row_major> vb;
                wmma::load_matrix_sync(pa, wp, WF_N);
                wmma::load_matrix_sync(vb, vbuf[cur] + dc * WF_K, Dh);
                wmma::mma_sync(o_frag, pa, vb, o_frag);

                // o_frag (16x16) を warp ローカル ws に書いて wacc に加算。
                wmma::store_matrix_sync(ws, o_frag, WF_N, wmma::mem_row_major);
                __syncwarp();
                for (int e = lane; e < WF_M * WF_K; e += 32)
                {
                    const int m = e / WF_K;
                    const int d = e - m * WF_K;
                    wacc[m * Dh + dc * WF_K + d] += ws[m * WF_N + d];
                }
                __syncwarp();
            }
        }
        __syncthreads(); // 現タイルバッファ再利用 (t+2) 前に全 warp の計算完了を保証
    }

    // ---- 正規化して書き戻す (有効行のみ) ----
    for (int e = tid; e < block_rows * Dh; e += nthreads)
    {
        const int m  = e / Dh;
        const int d  = e - m * Dh;
        const int qi = block_row_base + m;
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
void launch_attention_fast(const __half* d_q, const __half* d_k, const __half* d_v,
                           __half* d_out, int B, int H, int Sq, int Sk, int Dh, float scale)
{
    // 不正・空入力ガード。
    if (B <= 0 || H <= 0 || Sq <= 0 || Sk <= 0 || Dh <= 0)
    {
        return;
    }

    // Dh が 16 の倍数でなければ fast 経路の前提 (wmma 16x16 / cp.async 16B) を満たさない
    // ため、既存 launch_attention にフォールバック (数値・経路とも実績のある default)。
    if ((Dh % WF_K) != 0)
    {
        launch_attention(d_q, d_k, d_v, d_out, B, H, Sq, Sk, Dh, scale);
        return;
    }

    // nwarps を {4,2,1} から shared 制約内で最大選択。多 warp ほど占有率が上がり K/V
    // 再ロードの償却も進むが、shared (特に acc[block_rows*Dh] の FP32) が増える。
    constexpr size_t kMaxOptinShmem = 224u * 1024u; // sm_120 の SM あたり動的 shared 上限帯
    int    nwarps = 0;
    size_t shmem  = 0;
    for (int cand : { 4, 2, 1 })
    {
        const int    block_rows = cand * WF_WARP_ROWS;
        const size_t half_elems =
            static_cast<size_t>(block_rows) * Dh          // q_tile
            + 4u * static_cast<size_t>(WF_KV_ROWS) * Dh    // kbuf x2 + vbuf x2
            + static_cast<size_t>(cand) * WF_M * WF_N;     // p_tile
        const size_t float_elems =
            static_cast<size_t>(cand) * WF_M * WF_N        // s_tile
            + static_cast<size_t>(block_rows) * Dh         // acc
            + 2u * static_cast<size_t>(block_rows);        // m_row + l_row
        const size_t sh = half_elems * sizeof(__half) + float_elems * sizeof(float);
        if (sh <= kMaxOptinShmem)
        {
            nwarps = cand;
            shmem  = sh;
            break;
        }
    }

    // どの nwarps でも shared に乗らない大 Dh は既存経路へフォールバック。
    if (nwarps == 0)
    {
        launch_attention(d_q, d_k, d_v, d_out, B, H, Sq, Sk, Dh, scale);
        return;
    }

    const int  block_rows = nwarps * WF_WARP_ROWS;
    const int  qtiles     = (Sq + block_rows - 1) / block_rows;
    const dim3 grid(static_cast<unsigned>(qtiles),
                    static_cast<unsigned>(static_cast<long>(B) * H), 1);
    const int  threads = nwarps * 32;

    // 動的 shared が既定 48KB を超えるときのみ opt-in (冪等)。
    constexpr size_t kDefaultShmemLimit = 48u * 1024u;
    if (shmem > kDefaultShmemLimit)
    {
        CUDA_CHECK(cudaFuncSetAttribute(attention_flash_wmma_fast_fp16,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(shmem)));
    }

    attention_flash_wmma_fast_fp16<<<grid, threads, shmem>>>(d_q, d_k, d_v, d_out,
                                                             B, H, Sq, Sk, Dh, scale);
    CUDA_CHECK_KERNEL();
}

} // namespace dollama

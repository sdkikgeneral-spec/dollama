// Conv2d カーネル実装 (Phase 2 マイルストーン 2-2-4 / 2-6 S3)
// 対象: RTX5080 (Blackwell / sm_120) / CUDA Runtime API のみ
//
// 設計判断 (後続カーネルでも参照する):
//   - 入出力・重み・bias は FP16 / 内部の積和は必ず FP32 蓄積。最後に
//     __float2half で書き戻す (GEMM / GroupNorm と同規約)。
//   - 経路は 3 段構え (S3 で GEMM 2 経路を追加):
//       (a) direct conv (1 スレッド = 出力 1 画素, FP32 蓄積)
//           … 自作版 (哲学として温存)・小行列/dilation>1/N>1 のフォールバック。
//       (b) 1x1 conv → wmma Tensor Core GEMM (launch_gemm_fp16, transB=false)
//           out[co,hw] = Σ_ci W[co,ci] * in[ci,hw]。weight[Cout,Cin] × in[Cin,HW]。
//       (c) 3x3 等 (KH/KW>1) → im2col + wmma GEMM
//           in を [Cin*KH*KW, Hout*Wout] へ展開 → weight[Cout, Cin*KH*KW] と GEMM。
//     direct (a) は消さず launch_conv2d 内で形状により選択する。
//
// S3 (conv の Tensor Core 化) のメモ:
//   - GEMM は M=Cout, N=Hout*Wout, K=Cin(*KH*KW), transB=false。
//     A=weight[Cout,K] row-major, B=col/in[K,N] row-major, C=out[Cout,N] row-major。
//   - GEMM は bias を扱えない (alpha/beta のみ) ので、conv の bias は GEMM 後に
//     行 (= co) ごとブロードキャスト加算する専用カーネルで足す。
//   - N>1 (バッチ) は per-n オフセットの batch ループで 1 枚ずつ GEMM 経路を回す
//     (G-2k S1: CFG cond/uncond の B=2 束ね対応)。in を +n*Cin*H*W、out を
//     +n*Cout*Hout*Wout でずらし、N==1 GEMM ヘルパ (im2col→GEMM→bias) を N 回呼ぶ。
//     各サンプルは独立ゆえ K-loop 順・蓄積順は N==1 と不変 → ビット一致が狙える。
//     SDXL 既定は UNet/VAE とも N=1 なのでループは 1 回で従来経路そのまま。GEMM の
//     下限 (Cout/HW/K>=16) を満たさない N>1 は従来どおり direct に落とす。
//   - im2col 中間バッファ VRAM: Cin*KH*KW * Hout*Wout * 2byte。VAE C128 512²(3x3) で
//     約 600MB。重み 5.1GB 常駐 + 中間群があるため、上限 (IM2COL_TILE_BYTES) を超える
//     場合は Hout を行帯 (タイル) に分割し、帯ごとに im2col→GEMM して OOM を避ける。
//
// 帯分割の正当性に関する重要な注意 (2-6 S2 数値バグ修正):
//   - GEMM は出力 C を ldc=N (= 渡した N) 固定で書く。帯分割で N=帯幅 (rows*Wout) に
//     なるとき、全体出力テンソル d_out[Cout, Hout*Wout] の途中 (d_out + ho_base*Wout)
//     を直接 C に渡すと、GEMM は行 stride を「帯幅」とみなして書くため、co>=1 の
//     チャネル行が全体テンソル上で別チャネルの領域へずれて書かれ、出力が完全に
//     破壊される (UNet up_block_2 concat Cin=960 / VAE C128 512² で発症)。
//   - 対策: 帯 GEMM は「帯幅を leading dim とする連続バッファ d_out_band」へ書き、
//     その後 scatter_band_to_out で全体テンソルの正しい stride (= Hout*Wout) へ
//     チャネル行ごとに散布コピーする。帯が 1 つ (= 分割なし) のときは帯幅が
//     Hout*Wout に一致するので、d_out へ直接 GEMM してコピーを省く。
//
// G-10k T4 (conv2d 真の batch2) のメモ:
//   - N>1 枝 (launch_conv2d 末尾) を「im2col の n 次元対応 + strided batched GEMM
//     (launch_gemm_fp16_batched)」へ置き換えた。N==1 分岐・direct・N==1 用ヘルパ
//     (launch_conv2d_1x1_gemm / launch_conv2d_im2col_gemm) は 1 バイトも変えていない。
//   - G-10k T7/T7b (docs/g10k-plan.md §8 ケース B) の実走で秒中立・陰性クローズと判定した
//     ため revert はせず、既定を反転して opt-in DOLLAMA_CONV_BATCH=1 に降格した
//     (getenv キャッシュ型・プロセス単位固定)。未設定/空/"1"以外は
//     G-2k S1 の per-n 直列ループ (旧コード) がそのまま実行される。旧ループは新設の
//     N 対応関数を経由せず、rows_cap の N 分割も通らない (構造保存)。
//   - VRAM: IM2COL_TILE_BYTES (256MB) は「N 込みの合計上限」として扱い、rows_cap を
//     N で割る。col バッファの同時生存量は N によらず 256MB 以下に据え置かれる。
//   - bias: 出力 [N, Cout, HW] では row = idx / Ncols の 1 段分解 (conv_bias_add_rows)
//     が使えないため、co = (idx / Ncols) % Cout の 2 段分解を持つ別カーネル
//     (conv_bias_add_rows_batched) を用意した。要素ごとの演算 (h2f+h2f -> f2h) は同一。
//   - 1x1 経路に帯分割は無い (N=HW の単一 GEMM で C を全体テンソルへ直接書く)。
//     N 対応でも 1x1 に band 経路を持ち込まない。
//   - GemmBatchedDesc へ渡す stride はすべて非負 (stride_a=0 の重み共有 / stride_b・
//     stride_c は正)。負 stride は T4 スコープ外。
#include "kernels/conv2d.cuh"
#include "kernels/device_arena.cuh"
#include "kernels/gemm.cuh"
#include "kernels/utils.cuh"

#include <cuda_fp16.h>

#include <cstdlib>
#include <cstring>

namespace dollama
{

// 1D グリッドのスレッド数 (32 の倍数)。
static constexpr int CONV_THREADS = 256;

// im2col 中間バッファの 1 回あたり上限バイト数 (これを超えたら Hout を帯分割する)。
// 256MB。VAE C128 512² の 600MB を 3 帯程度に割って VRAM ピークを抑える。
static constexpr long IM2COL_TILE_BYTES = 256L * 1024 * 1024;

// 1D グリッドのブロック数上限 (gridDim.x の安全上限)。
static constexpr long GRID_BLOCKS_CAP = 65535;

// ----------------------------------------------------------------
// direct conv2d カーネル: 1 スレッド = 出力 1 画素。
//   global index idx を (n, co, ho, wo) に分解し、Cin*KH*KW を FP32 蓄積する。
//   total を超えるスレッドはグリッドストライドで複数画素を担当する。
// ----------------------------------------------------------------
__global__ void conv2d_fp16(const __half* in,
                            const __half* weight,
                            const __half* bias,    // nullptr 可
                            __half*       out,
                            int           N,
                            int           Cin,
                            int           H,
                            int           W,
                            int           Cout,
                            int           KH,
                            int           KW,
                            int           Hout,
                            int           Wout,
                            int           stride_h,
                            int           stride_w,
                            int           pad_h,
                            int           pad_w,
                            int           dilation_h,
                            int           dilation_w)
{
    const long total = static_cast<long>(N) * Cout * Hout * Wout;

    for (long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<long>(gridDim.x) * blockDim.x)
    {
        // idx を (n, co, ho, wo) へ分解 (row-major [N,Cout,Hout,Wout])。
        const int wo = static_cast<int>(idx % Wout);
        long t       = idx / Wout;
        const int ho = static_cast<int>(t % Hout);
        t /= Hout;
        const int co = static_cast<int>(t % Cout);
        const int n  = static_cast<int>(t / Cout);

        // 入力の左上参照位置 (kh=kw=0 のとき)。
        const int hi0 = ho * stride_h - pad_h;
        const int wi0 = wo * stride_w - pad_w;

        float acc = 0.0f;

        // Cin × KH × KW の積和 (FP32 蓄積)。
        for (int ci = 0; ci < Cin; ++ci)
        {
            // in の (n, ci) チャネル先頭オフセット。
            const long in_ch_base = (static_cast<long>(n) * Cin + ci) * H * W;
            // weight の (co, ci) フィルタ先頭オフセット。
            const long w_ch_base  = (static_cast<long>(co) * Cin + ci) * KH * KW;

            for (int kh = 0; kh < KH; ++kh)
            {
                const int hi = hi0 + kh * dilation_h;
                if (hi < 0 || hi >= H)
                {
                    continue; // 範囲外 = ゼロパディング
                }
                for (int kw = 0; kw < KW; ++kw)
                {
                    const int wi = wi0 + kw * dilation_w;
                    if (wi < 0 || wi >= W)
                    {
                        continue; // 範囲外 = ゼロパディング
                    }
                    const float x = __half2float(in[in_ch_base + static_cast<long>(hi) * W + wi]);
                    const float w = __half2float(weight[w_ch_base + kh * KW + kw]);
                    acc += x * w;
                }
            }
        }

        if (bias != nullptr)
        {
            acc += __half2float(bias[co]);
        }

        out[idx] = __float2half(acc);
    }
}

// ----------------------------------------------------------------
// 行ごとブロードキャスト bias 加算: out[co, j] += bias[co]
//   out は [Cout, N_cols] row-major。GEMM (bias 非対応) の後段で使う。
//   グリッドストライドで全要素を走査する。
// ----------------------------------------------------------------
__global__ void conv_bias_add_rows(__half* out, const __half* bias, int Cout, long Ncols)
{
    const long total = static_cast<long>(Cout) * Ncols;
    for (long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<long>(gridDim.x) * blockDim.x)
    {
        const int co = static_cast<int>(idx / Ncols);
        const float v = __half2float(out[idx]) + __half2float(bias[co]);
        out[idx] = __float2half(v);
    }
}

// ----------------------------------------------------------------
// 帯 GEMM 出力の散布コピー: src[Cout, band_cols] (row-major, 連続) を
//   全体出力 dst[Cout, Hout*Wout] の該当帯 (ho_base*Wout から band_cols 列) へ
//   チャネル行ごとの正しい stride (= Hout*Wout) で書き戻す。
//     dst[co * dst_row_stride + band_col_off + j] = src[co * band_cols + j]
//   1 スレッド = src の 1 要素。グリッドストライド。
//   GEMM は ldc=band_cols 固定で連続に書くため、全体テンソルの行 stride
//   (= Hout*Wout) との食い違いをここで吸収する (帯分割数値バグの修正点)。
// ----------------------------------------------------------------
__global__ void scatter_band_to_out(const __half* src,
                                    __half*       dst,
                                    int           Cout,
                                    long          band_cols,      // この帯の列数 (= rows*Wout)
                                    long          dst_row_stride, // = Hout*Wout
                                    long          band_col_off)   // = ho_base*Wout
{
    const long total = static_cast<long>(Cout) * band_cols;
    for (long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<long>(gridDim.x) * blockDim.x)
    {
        const long co = idx / band_cols;
        const long j  = idx % band_cols;
        dst[co * dst_row_stride + band_col_off + j] = src[idx];
    }
}

// ----------------------------------------------------------------
// im2col カーネル (1 バッチ分・行帯対応)。
//   入力 in_n: [Cin, H, W] row-major (バッチ n のスライス先頭ポインタ)。
//   出力 col : [Cin*KH*KW, tile_rows*Wout] row-major。
//     row = (ci*KH + kh)*KW + kw、col 列 = (ho_local)*Wout + wo。
//     ho = ho_base + ho_local。範囲外入力はゼロパディング。
//   1 スレッド = col の 1 要素を埋める。グリッドストライド。
// ----------------------------------------------------------------
__global__ void im2col_fp16(const __half* in_n,
                            __half*       col,
                            int           Cin,
                            int           H,
                            int           W,
                            int           KH,
                            int           KW,
                            int           Hout,
                            int           Wout,
                            int           ho_base,
                            int           tile_rows,
                            int           stride_h,
                            int           stride_w,
                            int           pad_h,
                            int           pad_w,
                            int           dilation_h,
                            int           dilation_w)
{
    const int K = Cin * KH * KW;        // col の行数
    const long Ncol = static_cast<long>(tile_rows) * Wout; // col の列数
    const long total = static_cast<long>(K) * Ncol;

    for (long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<long>(gridDim.x) * blockDim.x)
    {
        // idx を (row, col_idx) に分解。
        const long col_idx = idx % Ncol;
        const int  krow    = static_cast<int>(idx / Ncol);

        // col_idx を (ho_local, wo) に分解。
        const int wo        = static_cast<int>(col_idx % Wout);
        const int ho_local  = static_cast<int>(col_idx / Wout);
        const int ho        = ho_base + ho_local;

        // krow を (ci, kh, kw) に分解。
        const int kw = krow % KW;
        int        tk = krow / KW;
        const int  kh = tk % KH;
        const int  ci = tk / KH;

        const int hi = ho * stride_h - pad_h + kh * dilation_h;
        const int wi = wo * stride_w - pad_w + kw * dilation_w;

        __half v = __float2half(0.0f);
        if (hi >= 0 && hi < H && wi >= 0 && wi < W)
        {
            v = in_n[(static_cast<long>(ci) * H + hi) * W + wi];
        }
        col[idx] = v;
    }
}

// ----------------------------------------------------------------
// 出力サイズ計算 (PyTorch nn.Conv2d と同一式)。
// ----------------------------------------------------------------
static inline int conv_out_dim(int in, int pad, int dilation, int k, int stride)
{
    return (in + 2 * pad - dilation * (k - 1) - 1) / stride + 1;
}

// 1D グリッドのブロック数を total から計算し上限でクランプする。
static inline int grid_blocks_for(long total)
{
    long bl = (total + CONV_THREADS - 1) / CONV_THREADS;
    return static_cast<int>(bl < GRID_BLOCKS_CAP ? bl : GRID_BLOCKS_CAP);
}

// ----------------------------------------------------------------
// direct conv 起動 (フォールバック / 全形状対応)。test からの強制呼び出しにも使える。
// ----------------------------------------------------------------
void launch_conv2d_direct(const __half* d_in, const __half* d_weight, const __half* d_bias,
                          __half* d_out, int N, int Cin, int H, int W, int Cout,
                          int KH, int KW, int Hout, int Wout,
                          int stride_h, int stride_w, int pad_h, int pad_w,
                          int dilation_h, int dilation_w)
{
    const long total = static_cast<long>(N) * Cout * Hout * Wout;
    const int blocks = grid_blocks_for(total);

    conv2d_fp16<<<blocks, CONV_THREADS>>>(d_in, d_weight, d_bias, d_out,
                                          N, Cin, H, W, Cout, KH, KW, Hout, Wout,
                                          stride_h, stride_w, pad_h, pad_w,
                                          dilation_h, dilation_w);
    CUDA_CHECK_KERNEL();
}

// ----------------------------------------------------------------
// GEMM 経路を使うか判定する。
//   - dilation>1 / pad の特殊形状でも im2col は正しく扱えるが、テスト網羅と
//     リスク低減のため N==1 かつ十分大きい形状に限定し、それ以外は direct。
//   - wmma が効く下限 (Cout>=16, Hout*Wout>=16, K>=16) を満たすこと。
// ----------------------------------------------------------------
static bool use_gemm_path(int N, int Cin, int Cout, int KH, int KW, int Hout, int Wout)
{
    if (N != 1)
    {
        return false; // バッチは direct (SDXL は N=1。テスト N=2 もここで direct)
    }
    const long Ncol = static_cast<long>(Hout) * Wout;
    const long K    = static_cast<long>(Cin) * KH * KW;
    // wmma が起動・端数処理コストを上回る下限。
    if (Cout < 16 || Ncol < 16 || K < 16)
    {
        return false;
    }
    return true;
}

// ----------------------------------------------------------------
// 1x1 conv → GEMM 経路。N==1 前提。
//   out[Cout, HW] = weight[Cout, Cin] @ in[Cin, HW] (transB=false)。
//   in/out のメモリレイアウトは [Cin,H,W]/[Cout,H,W] = そのまま [Cin,HW]/[Cout,HW]。
//   N=HW は単一 GEMM の C にそのまま全体テンソルへ書く (ldc=N=HW=全体行 stride) ので
//   帯分割問題は起きない (HW が int に収まる SDXL 形状前提)。
// ----------------------------------------------------------------
static void launch_conv2d_1x1_gemm(const __half* d_in, const __half* d_weight,
                                   const __half* d_bias, __half* d_out,
                                   int Cin, int H, int W, int Cout)
{
    const int M = Cout;
    const long N = static_cast<long>(H) * W; // HW
    const int K = Cin;

    // GEMM の N は int 引数。SDXL の HW は最大 128*128=16384 で int に収まる。
    launch_gemm_fp16(d_weight, d_in, d_out, M, static_cast<int>(N), K,
                     1.0f, 0.0f, false, false);

    if (d_bias != nullptr)
    {
        const int blocks = grid_blocks_for(static_cast<long>(Cout) * N);
        conv_bias_add_rows<<<blocks, CONV_THREADS>>>(d_out, d_bias, Cout, N);
        CUDA_CHECK_KERNEL();
    }
}

// ----------------------------------------------------------------
// 一般 (KH/KW>1 等) conv → im2col + GEMM 経路。N==1 前提。
//   col[K, Ncol_tile] = im2col(in)、out_tile[Cout, Ncol_tile] = weight[Cout,K] @ col。
//   im2col バッファが上限を超える場合は Hout を帯分割して帯ごとに処理する。
//   帯分割時は GEMM 出力を「帯幅を leading dim とする連続バッファ」へ書き、その後
//   全体テンソルへチャネル stride を合わせて散布コピーする (数値バグの根本対策)。
// ----------------------------------------------------------------
static void launch_conv2d_im2col_gemm(const __half* d_in, const __half* d_weight,
                                      const __half* d_bias, __half* d_out,
                                      int Cin, int H, int W, int Cout,
                                      int KH, int KW, int Hout, int Wout,
                                      int stride_h, int stride_w, int pad_h, int pad_w,
                                      int dilation_h, int dilation_w)
{
    const int K = Cin * KH * KW; // col 行数 = GEMM の K
    const long full_ncol = static_cast<long>(Hout) * Wout; // 全体出力の行 stride
    // 1 行 (ho=1) あたりの col バイト数。これで帯の高さを決める。
    const long bytes_per_row = static_cast<long>(K) * Wout * sizeof(__half);
    int tile_rows = Hout;
    if (bytes_per_row > 0)
    {
        long rows_cap = IM2COL_TILE_BYTES / bytes_per_row;
        if (rows_cap < 1)
        {
            rows_cap = 1; // 1 行でも上限超過する巨大形状でも最低 1 行ずつ進める
        }
        if (rows_cap < tile_rows)
        {
            tile_rows = static_cast<int>(rows_cap);
        }
    }

    // 帯分割が発生するか (= tile_rows が Hout 未満)。
    const bool banded = (tile_rows < Hout);

    // G-8k S1b: 中間バッファ (d_col / d_out_band) の cudaMalloc/cudaFree を
    // デバイスアリーナの bump 確保へ置換する (配管のみ・数値は完全不変)。
    //   - RAII: 関数スコープ脱出 (早期 return / 例外) で必ず rewind される。
    //   - DOLLAMA_POOL=0 なら素の cudaMalloc/cudaFree に完全フォールバック (旧経路)。
    // 数値的な正当性: d_col は帯ごとに im2col が K*Ncol 要素を全書き、d_out_band は
    // GEMM が beta=0 で Cout*Ncol 要素を全書きするため、再利用領域の残留値は読まれない。
    DeviceArenaScope arena(DeviceArenaId::UNet);

    // 帯 col バッファ確保 (最大帯サイズ分)。
    const long col_elems = static_cast<long>(K) * tile_rows * Wout;
    __half* d_col = arena.alloc<__half>(static_cast<size_t>(col_elems));

    // 帯分割時のみ、GEMM 出力先の連続バッファ (帯幅を leading dim とする) を確保。
    // 分割なしのときは d_out へ直接書ける (帯幅 = 全体行 stride のため)。
    __half* d_out_band = nullptr;
    if (banded)
    {
        const long band_out_elems = static_cast<long>(Cout) * tile_rows * Wout;
        d_out_band = arena.alloc<__half>(static_cast<size_t>(band_out_elems));
    }

    for (int ho_base = 0; ho_base < Hout; ho_base += tile_rows)
    {
        const int rows = (ho_base + tile_rows <= Hout) ? tile_rows : (Hout - ho_base);
        const long Ncol = static_cast<long>(rows) * Wout; // この帯の GEMM N

        // im2col (この帯)。
        const int blocks = grid_blocks_for(static_cast<long>(K) * Ncol);
        im2col_fp16<<<blocks, CONV_THREADS>>>(d_in, d_col, Cin, H, W, KH, KW,
                                              Hout, Wout, ho_base, rows,
                                              stride_h, stride_w, pad_h, pad_w,
                                              dilation_h, dilation_w);
        CUDA_CHECK_KERNEL();

        // GEMM: out_band[Cout, Ncol] = weight[Cout, K] @ col[K, Ncol]。
        // 分割なし → 連続な d_out へ直接 (ldc=Ncol=full_ncol で整合)。
        // 分割あり → 帯連続バッファ d_out_band (ldc=Ncol=帯幅) へ書き、後で散布。
        __half* d_gemm_out = banded ? d_out_band : d_out;
        launch_gemm_fp16(d_weight, d_col, d_gemm_out, Cout, static_cast<int>(Ncol), K,
                         1.0f, 0.0f, false, false);

        if (banded)
        {
            // 帯バッファ (ldc=Ncol) を全体テンソル (行 stride=full_ncol) の正しい
            // 列オフセット (ho_base*Wout) へチャネルごとに散布コピーする。
            const int sblocks = grid_blocks_for(static_cast<long>(Cout) * Ncol);
            scatter_band_to_out<<<sblocks, CONV_THREADS>>>(
                d_out_band, d_out, Cout, Ncol, full_ncol,
                static_cast<long>(ho_base) * Wout);
            CUDA_CHECK_KERNEL();
        }
    }

    if (d_bias != nullptr)
    {
        const int blocks = grid_blocks_for(static_cast<long>(Cout) * full_ncol);
        conv_bias_add_rows<<<blocks, CONV_THREADS>>>(d_out, d_bias, Cout, full_ncol);
        CUDA_CHECK_KERNEL();
    }

    // d_col / d_out_band の解放は arena のデストラクタ (rewind) が行う。
}

// ================================================================
// G-10k T4: N>1 用の真 batch 経路 (ここから launch_conv2d の直前までは純追加)。
//   上の N==1 用ヘルパ (launch_conv2d_1x1_gemm / launch_conv2d_im2col_gemm) と
//   direct は 1 バイトも変更していない。
// ================================================================

// ----------------------------------------------------------------
// opt-in スイッチ: DOLLAMA_CONV_BATCH=1 のときだけ N>1 の真 batch 経路を有効化し、
//   それ以外 (未設定/空/"1"以外) は G-2k S1 の per-n 直列ループ (旧コード) を
//   そのまま実行する。G-10k T7/T7b (docs/g10k-plan.md §8 ケース B) の実走で
//   削減率 +1.9〜3.4% 悪化 = 秒中立・陰性クローズと判定したため、revert はせず
//   既定を反転して opt-in に降格した。
//   作法は gemm.cu の cublas_disabled() / device_arena.cu の
//   device_arena_pool_enabled() と同型 (getenv は初回のみ・以後キャッシュ =
//   プロセス単位固定。同一プロセス内で切り替えることはできない)。
//   未設定 / 空 / "1" 以外 は既定 (旧経路)。"1" のときだけ新経路 ON。
// ----------------------------------------------------------------
static bool conv_batch_enabled()
{
    static int cached = -1;
    if (cached < 0)
    {
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
        const char* v = std::getenv("DOLLAMA_CONV_BATCH");
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
        cached = (v != nullptr && std::strcmp(v, "1") == 0) ? 1 : 0;
    }
    return cached == 1;
}

// ----------------------------------------------------------------
// N 次元対応 im2col (行帯対応)。grid.y = n (バッチ index)。
//   入力 in : [N, Cin, H, W] row-major。item n の先頭は in + n*in_batch_stride。
//   出力 col: [N][Cin*KH*KW, tile_rows*Wout]。item n の先頭は col + n*col_batch_stride
//             (= K*Ncol・item 内は N==1 版と同一の [K, Ncol] row-major)。
//   item 内の要素写像 (row / col 列 / ゼロパディング) は im2col_fp16 と完全同一。
//   同一データを 2 item に入れれば col の両 item はビット一致する ([G2a] の前提)。
// ----------------------------------------------------------------
__global__ void im2col_fp16_batched(const __half* in,
                                    __half*       col,
                                    long long     in_batch_stride,
                                    long long     col_batch_stride,
                                    int           Cin,
                                    int           H,
                                    int           W,
                                    int           KH,
                                    int           KW,
                                    int           Hout,
                                    int           Wout,
                                    int           ho_base,
                                    int           tile_rows,
                                    int           stride_h,
                                    int           stride_w,
                                    int           pad_h,
                                    int           pad_w,
                                    int           dilation_h,
                                    int           dilation_w)
{
    const __half* in_n  = in  + static_cast<long long>(blockIdx.y) * in_batch_stride;
    __half*       col_n = col + static_cast<long long>(blockIdx.y) * col_batch_stride;

    const int K = Cin * KH * KW;        // col の行数
    const long Ncol = static_cast<long>(tile_rows) * Wout; // col の列数
    const long total = static_cast<long>(K) * Ncol;

    for (long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<long>(gridDim.x) * blockDim.x)
    {
        // idx を (row, col_idx) に分解。
        const long col_idx = idx % Ncol;
        const int  krow    = static_cast<int>(idx / Ncol);

        // col_idx を (ho_local, wo) に分解。
        const int wo        = static_cast<int>(col_idx % Wout);
        const int ho_local  = static_cast<int>(col_idx / Wout);
        const int ho        = ho_base + ho_local;

        // krow を (ci, kh, kw) に分解。
        const int kw = krow % KW;
        int        tk = krow / KW;
        const int  kh = tk % KH;
        const int  ci = tk / KH;

        const int hi = ho * stride_h - pad_h + kh * dilation_h;
        const int wi = wo * stride_w - pad_w + kw * dilation_w;

        __half v = __float2half(0.0f);
        if (hi >= 0 && hi < H && wi >= 0 && wi < W)
        {
            v = in_n[(static_cast<long>(ci) * H + hi) * W + wi];
        }
        col_n[idx] = v;
    }
}

// ----------------------------------------------------------------
// N 次元対応の行ごとブロードキャスト bias 加算: out[n, co, j] += bias[co]
//   out は [N, Cout, Ncols] row-major。
//   ★conv_bias_add_rows (N==1 用) は co = idx / Ncols の 1 段分解を前提にしており、
//     Ncols に N*HW を渡すと bias index が壊れる。ここでは
//     co = (idx / Ncols) % Cout の 2 段分解で [N, Cout, Ncols] を正しく分解する。
//   要素ごとの演算 (h2f(out) + h2f(bias) -> f2h) は conv_bias_add_rows と同一。
// ----------------------------------------------------------------
__global__ void conv_bias_add_rows_batched(__half* out, const __half* bias,
                                           int N, int Cout, long long Ncols)
{
    const long long total = static_cast<long long>(N) * Cout * Ncols;
    for (long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<long long>(gridDim.x) * blockDim.x)
    {
        const int co = static_cast<int>((idx / Ncols) % Cout);
        const float v = __half2float(out[idx]) + __half2float(bias[co]);
        out[idx] = __float2half(v);
    }
}

// ----------------------------------------------------------------
// N 次元対応の帯 GEMM 出力散布コピー。grid.y = n。
//   src: [N][Cout, band_cols] (batched GEMM の C・item stride = Cout*band_cols)
//   dst: [N][Cout, dst_row_stride] (全体出力・item stride = Cout*dst_row_stride)
//     dst_n[co * dst_row_stride + band_col_off + j] = src_n[co * band_cols + j]
//   item 内の写像は scatter_band_to_out と同一。
// ----------------------------------------------------------------
__global__ void scatter_band_to_out_batched(const __half* src,
                                            __half*       dst,
                                            int           Cout,
                                            long long     band_cols,      // = rows*Wout
                                            long long     dst_row_stride, // = Hout*Wout
                                            long long     band_col_off)   // = ho_base*Wout
{
    const __half* src_n = src + static_cast<long long>(blockIdx.y) * Cout * band_cols;
    __half*       dst_n = dst + static_cast<long long>(blockIdx.y) * Cout * dst_row_stride;

    const long long total = static_cast<long long>(Cout) * band_cols;
    for (long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<long long>(gridDim.x) * blockDim.x)
    {
        const long long co = idx / band_cols;
        const long long j  = idx % band_cols;
        dst_n[co * dst_row_stride + band_col_off + j] = src_n[idx];
    }
}

// ----------------------------------------------------------------
// 1x1 conv → batched GEMM 経路 (N>1)。
//   item n: out_n[Cout, HW] = weight[Cout, Cin] @ in_n[Cin, HW] (transB=false)。
//   GemmBatchedDesc 写像: A=weight (stride_a=0 = 全 item 共有) /
//   B=in (stride_b=Cin*HW) / C=out (stride_c=Cout*HW)。いずれも非負。
//   帯分割は無い (N==1 版と同じく HW の単一 GEMM で全体テンソルへ直接書く)。
// ----------------------------------------------------------------
static void launch_conv2d_1x1_gemm_batched(const __half* d_in, const __half* d_weight,
                                           const __half* d_bias, __half* d_out,
                                           int N, int Cin, int H, int W, int Cout)
{
    const long long HW = static_cast<long long>(H) * W;

    GemmBatchedDesc desc;
    desc.batch    = N;
    desc.M        = Cout;
    desc.N        = static_cast<int>(HW); // SDXL の HW は最大 128*128=16384 で int に収まる
    desc.K        = Cin;
    desc.stride_a = 0;                                 // 重み共有
    desc.stride_b = static_cast<long long>(Cin) * HW;  // in の item stride
    desc.stride_c = static_cast<long long>(Cout) * HW; // out の item stride
    desc.alpha    = 1.0f;
    desc.beta     = 0.0f;
    desc.trans_b  = false;
    launch_gemm_fp16_batched(d_weight, d_in, d_out, desc);

    if (d_bias != nullptr)
    {
        const int blocks =
            grid_blocks_for(static_cast<long>(static_cast<long long>(N) * Cout * HW));
        conv_bias_add_rows_batched<<<blocks, CONV_THREADS>>>(d_out, d_bias, N, Cout, HW);
        CUDA_CHECK_KERNEL();
    }
}

// ----------------------------------------------------------------
// 一般 conv → N 次元対応 im2col + batched GEMM 経路 (N>1)。
//   col[N][K, Ncol_tile] = im2col(in)、out_n[Cout, Ncol_tile] = weight[Cout,K] @ col_n。
//   IM2COL_TILE_BYTES は N 込みの合計上限: rows_cap = 256MB / (bytes_per_row * N)。
//   帯分割時は N==1 版と同じく帯連続バッファ (item stride = Cout*Ncol) へ書き、
//   scatter_band_to_out_batched で全体テンソルへ散布する。
//   GemmBatchedDesc 写像: A=weight (stride_a=0) / B=col (stride_b=K*Ncol) /
//   C=帯バッファ (stride_c=Cout*Ncol) または d_out (stride_c=Cout*full_ncol・
//   分割なしのとき Ncol==full_ncol なので同値)。いずれも非負。
// ----------------------------------------------------------------
static void launch_conv2d_im2col_gemm_batched(const __half* d_in, const __half* d_weight,
                                              const __half* d_bias, __half* d_out,
                                              int N, int Cin, int H, int W, int Cout,
                                              int KH, int KW, int Hout, int Wout,
                                              int stride_h, int stride_w, int pad_h, int pad_w,
                                              int dilation_h, int dilation_w)
{
    const int K = Cin * KH * KW; // col 行数 = GEMM の K
    const long long full_ncol = static_cast<long long>(Hout) * Wout; // 全体出力の行 stride
    // 1 行 (ho=1) あたりの col バイト数 (全 N item 分)。これで帯の高さを決める。
    const long long bytes_per_row_all =
        static_cast<long long>(K) * Wout * sizeof(__half) * N;
    int tile_rows = Hout;
    if (bytes_per_row_all > 0)
    {
        long long rows_cap = static_cast<long long>(IM2COL_TILE_BYTES) / bytes_per_row_all;
        if (rows_cap < 1)
        {
            rows_cap = 1; // 1 行でも上限超過する巨大形状でも最低 1 行ずつ進める
        }
        if (rows_cap < tile_rows)
        {
            tile_rows = static_cast<int>(rows_cap);
        }
    }

    // 帯分割が発生するか (= tile_rows が Hout 未満)。
    const bool banded = (tile_rows < Hout);

    // 中間バッファはデバイスアリーナの bump 確保 (G-8k と同じ配管)。
    // 数値的な正当性: d_col は帯ごとに im2col が全 item の K*Ncol 要素を全書き、
    // d_out_band は GEMM が beta=0 で全 item の Cout*Ncol 要素を全書きするため、
    // 再利用領域の残留値は読まれない。
    DeviceArenaScope arena(DeviceArenaId::UNet);

    // 帯 col バッファ確保 (最大帯サイズ × N item 分・合計 <= IM2COL_TILE_BYTES)。
    const long long col_elems = static_cast<long long>(N) * K * tile_rows * Wout;
    __half* d_col = arena.alloc<__half>(static_cast<size_t>(col_elems));

    // 帯分割時のみ、GEMM 出力先の連続バッファ (item stride = Cout*帯幅) を確保。
    __half* d_out_band = nullptr;
    if (banded)
    {
        const long long band_out_elems = static_cast<long long>(N) * Cout * tile_rows * Wout;
        d_out_band = arena.alloc<__half>(static_cast<size_t>(band_out_elems));
    }

    const long long in_stride  = static_cast<long long>(Cin) * H * W;   // in の item stride
    const long long out_stride = static_cast<long long>(Cout) * full_ncol; // out の item stride

    for (int ho_base = 0; ho_base < Hout; ho_base += tile_rows)
    {
        const int rows = (ho_base + tile_rows <= Hout) ? tile_rows : (Hout - ho_base);
        const long long Ncol = static_cast<long long>(rows) * Wout; // この帯の GEMM N

        // im2col (この帯・全 item)。grid.y = N。
        const int blocks = grid_blocks_for(static_cast<long>(K) * static_cast<long>(Ncol));
        const dim3 grid(static_cast<unsigned>(blocks), static_cast<unsigned>(N));
        im2col_fp16_batched<<<grid, CONV_THREADS>>>(d_in, d_col, in_stride,
                                                    static_cast<long long>(K) * Ncol,
                                                    Cin, H, W, KH, KW,
                                                    Hout, Wout, ho_base, rows,
                                                    stride_h, stride_w, pad_h, pad_w,
                                                    dilation_h, dilation_w);
        CUDA_CHECK_KERNEL();

        // batched GEMM: out_n[Cout, Ncol] = weight[Cout, K] @ col_n[K, Ncol]。
        GemmBatchedDesc desc;
        desc.batch    = N;
        desc.M        = Cout;
        desc.N        = static_cast<int>(Ncol);
        desc.K        = K;
        desc.stride_a = 0;                                  // 重み共有
        desc.stride_b = static_cast<long long>(K) * Ncol;   // col の item stride
        desc.stride_c = banded ? static_cast<long long>(Cout) * Ncol : out_stride;
        desc.alpha    = 1.0f;
        desc.beta     = 0.0f;
        desc.trans_b  = false;
        __half* d_gemm_out = banded ? d_out_band : d_out;
        launch_gemm_fp16_batched(d_weight, d_col, d_gemm_out, desc);

        if (banded)
        {
            // 帯バッファ (item stride=Cout*Ncol) を全体テンソル (行 stride=full_ncol・
            // item stride=Cout*full_ncol) の列オフセット (ho_base*Wout) へ散布コピー。
            const int sblocks = grid_blocks_for(static_cast<long>(Cout) * static_cast<long>(Ncol));
            const dim3 sgrid(static_cast<unsigned>(sblocks), static_cast<unsigned>(N));
            scatter_band_to_out_batched<<<sgrid, CONV_THREADS>>>(
                d_out_band, d_out, Cout, Ncol, full_ncol,
                static_cast<long long>(ho_base) * Wout);
            CUDA_CHECK_KERNEL();
        }
    }

    if (d_bias != nullptr)
    {
        const int blocks =
            grid_blocks_for(static_cast<long>(static_cast<long long>(N) * Cout * full_ncol));
        conv_bias_add_rows_batched<<<blocks, CONV_THREADS>>>(d_out, d_bias, N, Cout, full_ncol);
        CUDA_CHECK_KERNEL();
    }

    // d_col / d_out_band の解放は arena のデストラクタ (rewind) が行う。
}

// ----------------------------------------------------------------
// ホストラッパー: 形状に応じて 1x1=GEMM / 3x3 等=im2col+GEMM / direct を選択。
// ----------------------------------------------------------------
void launch_conv2d(const __half* d_in, const __half* d_weight, const __half* d_bias,
                   __half* d_out, int N, int Cin, int H, int W, int Cout, int KH, int KW,
                   int stride_h, int stride_w, int pad_h, int pad_w, int dilation_h, int dilation_w)
{
    // 不正・空入力ガード。負やゼロの次元・ストライド・拡張は何もしない。
    if (N <= 0 || Cin <= 0 || H <= 0 || W <= 0 || Cout <= 0 || KH <= 0 || KW <= 0)
    {
        return;
    }
    if (stride_h <= 0 || stride_w <= 0 || dilation_h <= 0 || dilation_w <= 0)
    {
        return;
    }
    assert(pad_h >= 0 && pad_w >= 0);

    // 出力サイズ (PyTorch nn.Conv2d と同一式)。
    const int Hout = conv_out_dim(H, pad_h, dilation_h, KH, stride_h);
    const int Wout = conv_out_dim(W, pad_w, dilation_w, KW, stride_w);
    if (Hout <= 0 || Wout <= 0)
    {
        return; // カーネルが入力より大きい等で出力が空
    }

    assert(static_cast<long>(N) * Cout * Hout * Wout > 0);

    // 1x1 conv (pad0 / stride1 / dilation1) は実質 GEMM。最も低リスクで効く。
    const bool is_1x1 = (KH == 1 && KW == 1 && pad_h == 0 && pad_w == 0
                         && stride_h == 1 && stride_w == 1
                         && dilation_h == 1 && dilation_w == 1);

    if (is_1x1 && use_gemm_path(N, Cin, Cout, KH, KW, Hout, Wout))
    {
        launch_conv2d_1x1_gemm(d_in, d_weight, d_bias, d_out, Cin, H, W, Cout);
        return;
    }

    // 一般 conv (3x3 same / 3x3 stride2 downsample 等) は im2col + GEMM。
    if (use_gemm_path(N, Cin, Cout, KH, KW, Hout, Wout))
    {
        launch_conv2d_im2col_gemm(d_in, d_weight, d_bias, d_out, Cin, H, W, Cout,
                                  KH, KW, Hout, Wout, stride_h, stride_w,
                                  pad_h, pad_w, dilation_h, dilation_w);
        return;
    }

    // ----------------------------------------------------------------
    // N>1 (CFG batch 等) で GEMM 経路が使える形状なら、per-n オフセットの
    // バッチループで N==1 GEMM ヘルパを N 回呼ぶ (G-2k S1)。
    //   入力 d_in + n*Cin*H*W / 出力 d_out + n*Cout*Hout*Wout を n ごとにずらす。
    //   各サンプルは独立で K-loop 順・蓄積順は N==1 と完全同一 → ビット一致が狙える。
    //   use_gemm_path(1,...) を渡すことで N==1 の形状下限判定をそのまま再利用する
    //   (上の N==1 既存分岐は 1 バイトも変更していない。ここは N>1 の追加枝のみ)。
    // ----------------------------------------------------------------
    if (N > 1 && use_gemm_path(1, Cin, Cout, KH, KW, Hout, Wout))
    {
        // ------------------------------------------------------------
        // G-10k T4: 真の batch 経路 (opt-in DOLLAMA_CONV_BATCH=1)。im2col の n 次元対応 +
        // strided batched GEMM で N item を 1 発で回す。数値は per-n 直列と bit 一致を
        // 狙わない (batched のタイル選択差 = FP16 tol 内・G-2k S2 と同種)。
        // T7/T7b で秒中立・陰性クローズと判定したため既定は旧経路 (opt-in 降格)。
        // DOLLAMA_CONV_BATCH=1 以外はこのブロックに入らず、下の
        // G-2k S1 per-n 直列ループ (旧コード・無改変) がそのまま実行される。
        //   ★構造保存: 旧ループは新設の N 対応関数を経由せず、rows_cap の N 分割も
        //     通らない (N==1 ヘルパをそのまま N 回呼ぶ)。
        // ------------------------------------------------------------
        if (conv_batch_enabled())
        {
            if (is_1x1)
            {
                launch_conv2d_1x1_gemm_batched(d_in, d_weight, d_bias, d_out,
                                               N, Cin, H, W, Cout);
            }
            else
            {
                launch_conv2d_im2col_gemm_batched(d_in, d_weight, d_bias, d_out,
                                                  N, Cin, H, W, Cout,
                                                  KH, KW, Hout, Wout, stride_h, stride_w,
                                                  pad_h, pad_w, dilation_h, dilation_w);
            }
            return;
        }

        // ---- ここから下は G-2k S1 の per-n 直列ループ (既定の旧経路・無改変。
        //      DOLLAMA_CONV_BATCH が "1" 以外のとき = 未設定/空/=0/=true 等すべて) ----
        const long in_stride  = static_cast<long>(Cin) * H * W;
        const long out_stride = static_cast<long>(Cout) * Hout * Wout;
        for (int n = 0; n < N; ++n)
        {
            const __half* in_n  = d_in + static_cast<long>(n) * in_stride;
            __half*       out_n = d_out + static_cast<long>(n) * out_stride;
            if (is_1x1)
            {
                launch_conv2d_1x1_gemm(in_n, d_weight, d_bias, out_n, Cin, H, W, Cout);
            }
            else
            {
                launch_conv2d_im2col_gemm(in_n, d_weight, d_bias, out_n, Cin, H, W, Cout,
                                          KH, KW, Hout, Wout, stride_h, stride_w,
                                          pad_h, pad_w, dilation_h, dilation_w);
            }
        }
        return;
    }

    // フォールバック: 自作 direct conv (小行列 / N>1 で GEMM 下限割れ / 小チャネルなど)。
    launch_conv2d_direct(d_in, d_weight, d_bias, d_out, N, Cin, H, W, Cout, KH, KW,
                         Hout, Wout, stride_h, stride_w, pad_h, pad_w,
                         dilation_h, dilation_w);
}

// ----------------------------------------------------------------
// G-4k S2: GEMM 経路 (bias 2 段丸め) 保証判定 (宣言コメントは conv2d.cuh 参照)。
//   launch_conv2d の経路選択ロジックを 1:1 でなぞる (本体は一切変更しない純追加)。
//   ここが false の shape に bias 後段融合を当てると direct 経路の単一丸めと食い違い、
//   静かに bit が崩れる — 呼び側 (resnet_block epilogue) の必須ガード。
// ----------------------------------------------------------------
bool conv2d_uses_gemm_bias_path(int N, int Cin, int H, int W, int Cout, int KH, int KW,
                                int stride_h, int stride_w, int pad_h, int pad_w,
                                int dilation_h, int dilation_w)
{
    // launch_conv2d の early return と同じ不正入力は「GEMM 経路ではない」扱い。
    if (N <= 0 || Cin <= 0 || H <= 0 || W <= 0 || Cout <= 0 || KH <= 0 || KW <= 0)
    {
        return false;
    }
    if (stride_h <= 0 || stride_w <= 0 || dilation_h <= 0 || dilation_w <= 0)
    {
        return false;
    }
    if (pad_h < 0 || pad_w < 0)
    {
        return false;
    }
    const int Hout = conv_out_dim(H, pad_h, dilation_h, KH, stride_h);
    const int Wout = conv_out_dim(W, pad_w, dilation_w, KW, stride_w);
    if (Hout <= 0 || Wout <= 0)
    {
        return false;
    }
    // N==1 も N>1 per-n ループも判定は use_gemm_path(1, ...) に帰着する。
    return use_gemm_path(1, Cin, Cout, KH, KW, Hout, Wout);
}


// ================================================================
// FP32 im2col + GEMM 経路 (S3-E)。
//   VAE up2/up3 は FP32 中間が必須 (FP16 だと Inf->GroupNorm NaN 伝播)。そのため
//   FP16 im2col 経路 (launch_conv2d) を使えず、活性 float / 重み FP16 / 出力 float の
//   FP32 経路を別に用意する。アルゴリズムは上の FP16 版と同一・dtype のみ float。
//   GEMM は cuBLAS FP32 (TF32 既定) に委譲する (launch_gemm_f32)。GEMM の A は
//   FP32 行列が必要なので、conv 重み (FP16) を内部スクラッチへ FP32 化してから渡す。
// ================================================================

// FP32 im2col (FP16 版 im2col_fp16 と同一・dtype のみ float)。
__global__ void im2col_f32(const float* in_n,
                           float*       col,
                           int          Cin,
                           int          H,
                           int          W,
                           int          KH,
                           int          KW,
                           int          Hout,
                           int          Wout,
                           int          ho_base,
                           int          tile_rows,
                           int          stride_h,
                           int          stride_w,
                           int          pad_h,
                           int          pad_w,
                           int          dilation_h,
                           int          dilation_w)
{
    const int K = Cin * KH * KW;
    const long Ncol = static_cast<long>(tile_rows) * Wout;
    const long total = static_cast<long>(K) * Ncol;

    for (long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<long>(gridDim.x) * blockDim.x)
    {
        const long col_idx = idx % Ncol;
        const int  krow    = static_cast<int>(idx / Ncol);
        const int wo       = static_cast<int>(col_idx % Wout);
        const int ho_local = static_cast<int>(col_idx / Wout);
        const int ho       = ho_base + ho_local;
        const int kw = krow % KW;
        int       tk = krow / KW;
        const int kh = tk % KH;
        const int ci = tk / KH;

        const int hi = ho * stride_h - pad_h + kh * dilation_h;
        const int wi = wo * stride_w - pad_w + kw * dilation_w;

        float v = 0.0f;
        if (hi >= 0 && hi < H && wi >= 0 && wi < W)
        {
            v = in_n[(static_cast<long>(ci) * H + hi) * W + wi];
        }
        col[idx] = v;
    }
}

// FP32 行ごとブロードキャスト bias 加算 (bias は FP16)。
__global__ void conv_bias_add_rows_f32(float* out, const __half* bias, int Cout, long Ncols)
{
    const long total = static_cast<long>(Cout) * Ncols;
    for (long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<long>(gridDim.x) * blockDim.x)
    {
        const int co = static_cast<int>(idx / Ncols);
        out[idx] = out[idx] + __half2float(bias[co]);
    }
}

// FP32 帯出力散布コピー (FP16 版 scatter_band_to_out と同一・dtype のみ float)。
__global__ void scatter_band_to_out_f32(const float* src,
                                        float*       dst,
                                        int          Cout,
                                        long         band_cols,
                                        long         dst_row_stride,
                                        long         band_col_off)
{
    const long total = static_cast<long>(Cout) * band_cols;
    for (long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<long>(gridDim.x) * blockDim.x)
    {
        const long co = idx / band_cols;
        const long j  = idx % band_cols;
        dst[co * dst_row_stride + band_col_off + j] = src[idx];
    }
}

// FP16 重み -> FP32 変換 (GEMM の A 行列を FP32 に揃えるため)。
__global__ void weight_h2f(const __half* in, float* out, long n)
{
    for (long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n;
         i += static_cast<long>(gridDim.x) * blockDim.x)
    {
        out[i] = __half2float(in[i]);
    }
}

// FP32 1x1 conv -> GEMM 経路。N==1 前提 (FP16 版 launch_conv2d_1x1_gemm の FP32 版)。
static void launch_conv2d_f32_1x1_gemm(const float* d_in, const float* d_weight_f32,
                                       const __half* d_bias, float* d_out,
                                       int Cin, int H, int W, int Cout)
{
    const int M = Cout;
    const long N = static_cast<long>(H) * W;
    const int K = Cin;

    launch_gemm_f32(d_weight_f32, d_in, d_out, M, static_cast<int>(N), K,
                    1.0f, 0.0f, false, false);

    if (d_bias != nullptr)
    {
        const int blocks = grid_blocks_for(static_cast<long>(Cout) * N);
        conv_bias_add_rows_f32<<<blocks, CONV_THREADS>>>(d_out, d_bias, Cout, N);
        CUDA_CHECK_KERNEL();
    }
}

// FP32 一般 conv (3x3 等) -> im2col + GEMM 経路。N==1 前提。
// FP16 版 launch_conv2d_im2col_gemm の FP32 版 (帯分割・散布も同一)。
static void launch_conv2d_f32_im2col_gemm(const float* d_in, const float* d_weight_f32,
                                          const __half* d_bias, float* d_out,
                                          int Cin, int H, int W, int Cout,
                                          int KH, int KW, int Hout, int Wout,
                                          int stride_h, int stride_w, int pad_h, int pad_w,
                                          int dilation_h, int dilation_w)
{
    const int K = Cin * KH * KW;
    const long full_ncol = static_cast<long>(Hout) * Wout;
    const long bytes_per_row = static_cast<long>(K) * Wout * sizeof(float);
    int tile_rows = Hout;
    if (bytes_per_row > 0)
    {
        long rows_cap = IM2COL_TILE_BYTES / bytes_per_row;
        if (rows_cap < 1)
        {
            rows_cap = 1;
        }
        if (rows_cap < tile_rows)
        {
            tile_rows = static_cast<int>(rows_cap);
        }
    }

    const bool banded = (tile_rows < Hout);

    // G-8k S3: FP16 版 (launch_conv2d_im2col_gemm, S1b) と同型に、中間バッファの
    // cudaMalloc/cudaFree をデバイスアリーナの bump 確保へ置換する (配管のみ・数値は不変)。
    //   - RAII: 関数スコープ脱出 (早期 return / 例外) で必ず rewind される。
    //   - 呼び出し元 launch_conv2d_f32_gemm の d_weight_f32 スコープの内側 = LIFO 合法。
    // 数値的な正当性: d_col は帯ごとに im2col が K*Ncol 要素を全書き、d_out_band は
    // GEMM が beta=0 で Cout*Ncol 要素を全書きするため、再利用領域の残留値は読まれない。
    DeviceArenaScope arena(DeviceArenaId::UNet);

    const long col_elems = static_cast<long>(K) * tile_rows * Wout;
    float* d_col = arena.alloc<float>(static_cast<size_t>(col_elems));

    float* d_out_band = nullptr;
    if (banded)
    {
        const long band_out_elems = static_cast<long>(Cout) * tile_rows * Wout;
        d_out_band = arena.alloc<float>(static_cast<size_t>(band_out_elems));
    }

    for (int ho_base = 0; ho_base < Hout; ho_base += tile_rows)
    {
        const int rows = (ho_base + tile_rows <= Hout) ? tile_rows : (Hout - ho_base);
        const long Ncol = static_cast<long>(rows) * Wout;

        const int blocks = grid_blocks_for(static_cast<long>(K) * Ncol);
        im2col_f32<<<blocks, CONV_THREADS>>>(d_in, d_col, Cin, H, W, KH, KW,
                                             Hout, Wout, ho_base, rows,
                                             stride_h, stride_w, pad_h, pad_w,
                                             dilation_h, dilation_w);
        CUDA_CHECK_KERNEL();

        float* d_gemm_out = banded ? d_out_band : d_out;
        launch_gemm_f32(d_weight_f32, d_col, d_gemm_out, Cout, static_cast<int>(Ncol), K,
                        1.0f, 0.0f, false, false);

        if (banded)
        {
            const int sblocks = grid_blocks_for(static_cast<long>(Cout) * Ncol);
            scatter_band_to_out_f32<<<sblocks, CONV_THREADS>>>(
                d_out_band, d_out, Cout, Ncol, full_ncol,
                static_cast<long>(ho_base) * Wout);
            CUDA_CHECK_KERNEL();
        }
    }

    if (d_bias != nullptr)
    {
        const int blocks = grid_blocks_for(static_cast<long>(Cout) * full_ncol);
        conv_bias_add_rows_f32<<<blocks, CONV_THREADS>>>(d_out, d_bias, Cout, full_ncol);
        CUDA_CHECK_KERNEL();
    }
}

// ----------------------------------------------------------------
// FP32 conv ホストラッパー (S3-E)。活性 float / 重み・bias FP16 / 出力 float。
//   形状に応じて 1x1=GEMM / 3x3 等=im2col+GEMM を選択する (N==1 前提・VAE 専用)。
//   重み (FP16) は内部で FP32 スクラッチへ変換してから GEMM の A に渡す。
//   GEMM 経路の下限割れ (極小形状) は VAE up2/up3 では起きないため考慮しない。
// ----------------------------------------------------------------
void launch_conv2d_f32_gemm(const float* d_in, const __half* d_weight, const __half* d_bias,
                            float* d_out, int N, int Cin, int H, int W, int Cout, int KH, int KW,
                            int stride_h, int stride_w, int pad_h, int pad_w,
                            int dilation_h, int dilation_w)
{
    if (N <= 0 || Cin <= 0 || H <= 0 || W <= 0 || Cout <= 0 || KH <= 0 || KW <= 0)
    {
        return;
    }
    if (stride_h <= 0 || stride_w <= 0 || dilation_h <= 0 || dilation_w <= 0)
    {
        return;
    }
    assert(pad_h >= 0 && pad_w >= 0);
    assert(N == 1); // VAE up2/up3 は N=1

    const int Hout = conv_out_dim(H, pad_h, dilation_h, KH, stride_h);
    const int Wout = conv_out_dim(W, pad_w, dilation_w, KW, stride_w);
    if (Hout <= 0 || Wout <= 0)
    {
        return;
    }

    // 重み (FP16) を FP32 へ変換 (A 行列 = weight[Cout, Cin*KH*KW])。
    // G-8k S3: 重み FP32 スクラッチもアリーナから切る。**キャッシュ化はしない**
    // (S3 スコープ外・数値不変を最優先。毎回 weight_h2f で作り直す挙動は据え置き)。
    const long wcount = static_cast<long>(Cout) * Cin * KH * KW;
    DeviceArenaScope warena(DeviceArenaId::UNet);
    float* d_weight_f32 = warena.alloc<float>(static_cast<size_t>(wcount));
    {
        const int blocks = grid_blocks_for(wcount);
        weight_h2f<<<blocks, CONV_THREADS>>>(d_weight, d_weight_f32, wcount);
        CUDA_CHECK_KERNEL();
    }

    const bool is_1x1 = (KH == 1 && KW == 1 && pad_h == 0 && pad_w == 0
                         && stride_h == 1 && stride_w == 1
                         && dilation_h == 1 && dilation_w == 1);

    if (is_1x1)
    {
        launch_conv2d_f32_1x1_gemm(d_in, d_weight_f32, d_bias, d_out, Cin, H, W, Cout);
    }
    else
    {
        launch_conv2d_f32_im2col_gemm(d_in, d_weight_f32, d_bias, d_out, Cin, H, W, Cout,
                                      KH, KW, Hout, Wout, stride_h, stride_w,
                                      pad_h, pad_w, dilation_h, dilation_w);
    }
}

} // namespace dollama

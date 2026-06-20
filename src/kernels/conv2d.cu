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
//   - N>1 (バッチ) は batch ループで 1 枚ずつ GEMM (im2col も 1 枚ずつ)。SDXL は
//     UNet/VAE とも N=1 なのでループは 1 回。テストの N=2 ケースは direct に落とす
//     (バッチ + 多チャネル + dilation など direct を確実に通す形状のため)。
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
#include "kernels/conv2d.cuh"
#include "kernels/gemm.cuh"
#include "kernels/utils.cuh"

#include <cuda_fp16.h>

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

    // 帯 col バッファ確保 (最大帯サイズ分)。
    const long col_elems = static_cast<long>(K) * tile_rows * Wout;
    __half* d_col = nullptr;
    CUDA_CHECK(cudaMalloc(&d_col, static_cast<size_t>(col_elems) * sizeof(__half)));

    // 帯分割時のみ、GEMM 出力先の連続バッファ (帯幅を leading dim とする) を確保。
    // 分割なしのときは d_out へ直接書ける (帯幅 = 全体行 stride のため)。
    __half* d_out_band = nullptr;
    if (banded)
    {
        const long band_out_elems = static_cast<long>(Cout) * tile_rows * Wout;
        CUDA_CHECK(cudaMalloc(&d_out_band, static_cast<size_t>(band_out_elems) * sizeof(__half)));
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

    CUDA_CHECK(cudaFree(d_col));
    if (d_out_band != nullptr)
    {
        CUDA_CHECK(cudaFree(d_out_band));
    }
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

    // フォールバック: 自作 direct conv (小行列 / N>1 / 小チャネルなど)。
    launch_conv2d_direct(d_in, d_weight, d_bias, d_out, N, Cin, H, W, Cout, KH, KW,
                         Hout, Wout, stride_h, stride_w, pad_h, pad_w,
                         dilation_h, dilation_w);
}

} // namespace dollama

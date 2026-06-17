// Conv2d カーネル実装 (Phase 2 マイルストーン 2-2-4)
// 対象: RTX5080 (Blackwell / sm_120) / CUDA Runtime API のみ
//
// 設計判断 (後続カーネルでも参照する):
//   - 入出力・重み・bias は FP16 / 内部の積和は必ず FP32 蓄積。最後に
//     __float2half で書き戻す (GEMM / GroupNorm と同規約)。
//   - 1 スレッド = 出力 1 画素 (n, co, ho, wo)。total = N*Cout*Hout*Wout を
//     256 threads/block で 1D グリッドストライド走査する。各スレッドが
//     Cin*KH*KW の積和を回す direct 実装。
//   - 範囲外 (パディング域) の入力アクセスはゼロとして扱う (ゼロパディング)。
//     hi/wi が [0,H)/[0,W) の外なら加算をスキップするだけ。
//   - 高速化 (1x1=GEMM / 3x3=im2col・タイリング) は本段では行わない。conv2d.cuh の
//     最適化メモを参照。本段は数値正当性とベンチ基準の確立が目的。
#include "kernels/conv2d.cuh"
#include "kernels/utils.cuh"

#include <cuda_fp16.h>

namespace dollama
{

// 1D グリッドのスレッド数 (32 の倍数)。
static constexpr int CONV_THREADS = 256;

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
// ホストラッパー
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
    const int Hout = (H + 2 * pad_h - dilation_h * (KH - 1) - 1) / stride_h + 1;
    const int Wout = (W + 2 * pad_w - dilation_w * (KW - 1) - 1) / stride_w + 1;
    if (Hout <= 0 || Wout <= 0)
    {
        return; // カーネルが入力より大きい等で出力が空
    }

    const long total = static_cast<long>(N) * Cout * Hout * Wout;
    assert(total > 0);

    // 1D グリッドストライド。ブロック数は total から算出しつつ上限でクランプ。
    long blocks_l = (total + CONV_THREADS - 1) / CONV_THREADS;
    const long blocks_cap = 65535;
    const int blocks = static_cast<int>(blocks_l < blocks_cap ? blocks_l : blocks_cap);

    conv2d_fp16<<<blocks, CONV_THREADS>>>(d_in, d_weight, d_bias, d_out,
                                          N, Cin, H, W, Cout, KH, KW, Hout, Wout,
                                          stride_h, stride_w, pad_h, pad_w,
                                          dilation_h, dilation_w);
    CUDA_CHECK_KERNEL();
}

} // namespace dollama

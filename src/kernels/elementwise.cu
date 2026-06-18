// Elementwise カーネル実装 (Phase 2 マイルストーン 2-4 / VAE 準備)
// 対象: RTX5080 (Blackwell / sm_120) / CUDA Runtime API のみ
//
// 設計判断:
//   - 入出力 FP16 / 内部の加算は FP32 蓄積 (GEMM / Conv / Attention と同規約)。
//   - add は 1D グリッド (1 スレッド = 出力 1 要素)。各スレッドが自分の 1 要素しか
//     読み書きしないので out==a / out==b の in-place が安全。
//   - upsample は 1D グリッド (1 スレッド = 出力 1 画素)。出力インデックスを
//     n,c,ho,wo に分解し、入力 (ho/2, wo/2) を引いてコピーするだけ (積和なし)。
#include "kernels/elementwise.cuh"
#include "kernels/utils.cuh"

#include <cuda_fp16.h>

namespace dollama
{

// 1D カーネルの 1 ブロックあたりスレッド数。
static constexpr int EW_THREADS = 256;

// ----------------------------------------------------------------
// residual add: out[i] = a[i] + b[i] (FP32 蓄積)。
// ----------------------------------------------------------------
__global__ void add_fp16(const __half* a, const __half* b, __half* out, int n)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        const float av = __half2float(a[idx]);
        const float bv = __half2float(b[idx]);
        out[idx] = __float2half(av + bv);
    }
}

// ----------------------------------------------------------------
// nearest-neighbor upsample 2x: out[n,c,ho,wo] = in[n,c,ho>>1,wo>>1]。
// 1 スレッド = 出力 1 画素。出力線形 idx を (n,c,ho,wo) に分解する。
// 添字はすべて非負なので ×2/÷2 はビットシフト (<<1 / >>1) で等価
// (符号付き除算の 0 方向丸め補正コードも出ない)。
// ----------------------------------------------------------------
__global__ void upsample_nearest2x_fp16(const __half* in,
                                        __half*       out,
                                        int           N,
                                        int           C,
                                        int           H,
                                        int           W)
{
    const int Ho  = H << 1;
    const int Wo  = W << 1;
    const long total = static_cast<long>(N) * C * Ho * Wo;

    const long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total)
    {
        return;
    }

    // 出力線形 idx を NCHW (Ho×Wo) に分解。
    const int wo = static_cast<int>(idx % Wo);
    long t       = idx / Wo;
    const int ho = static_cast<int>(t % Ho);
    t            = t / Ho;
    const int c  = static_cast<int>(t % C);
    const int n  = static_cast<int>(t / C);

    // 対応する入力画素 (nearest: ho>>1, wo>>1)。
    const int hi = ho >> 1;
    const int wi = wo >> 1;
    const long in_idx = (static_cast<long>(n) * C + c) * H * W
                        + static_cast<long>(hi) * W + wi;

    out[idx] = in[in_idx];
}

// ----------------------------------------------------------------
// ホストラッパー: residual add
// ----------------------------------------------------------------
void launch_add(const __half* d_a, const __half* d_b, __half* d_out, int n)
{
    if (n <= 0)
    {
        return;
    }
    const int blocks = ceil_div(n, EW_THREADS);
    add_fp16<<<blocks, EW_THREADS>>>(d_a, d_b, d_out, n);
    CUDA_CHECK_KERNEL();
}

// ----------------------------------------------------------------
// ホストラッパー: nearest-neighbor upsample 2x
// ----------------------------------------------------------------
void launch_upsample_nearest2x(const __half* d_in, __half* d_out, int N, int C, int H, int W)
{
    if (N <= 0 || C <= 0 || H <= 0 || W <= 0)
    {
        return;
    }
    const long total = static_cast<long>(N) * C * (H << 1) * (W << 1);
    assert(total > 0);
    const long blocks_l = (total + EW_THREADS - 1) / EW_THREADS;
    // グリッド x は 2^31-1 までなので VAE 最大 (1×128×1024×1024 = 1.3e8) で十分収まる。
    const int blocks = static_cast<int>(blocks_l);
    upsample_nearest2x_fp16<<<blocks, EW_THREADS>>>(d_in, d_out, N, C, H, W);
    CUDA_CHECK_KERNEL();
}

} // namespace dollama

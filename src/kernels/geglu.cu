// GEGLU カーネル実装 (Phase 2 マイルストーン 2-5 / UNet FeedForward)
// 対象: RTX5080 (Blackwell / sm_120) / CUDA Runtime API のみ
//
// 設計判断:
//   - 入出力 FP16 / 内部計算 FP32。gelu の erf は必ず FP32 で取る (activation.cu と同じ式)。
//   - 1D グリッド (1 スレッド = 出力 1 要素)。出力線形 idx を (t, c) に分解し、
//     入力の前半 x1 = in[t*(2d)+c] と後半 x2 = in[t*(2d)+d+c] を引いて
//     out = x1 * gelu(x2) を書く。
//   - in-place 不可 (入力 [tokens,2d] と出力 [tokens,d] で次元が異なる)。
#include "kernels/geglu.cuh"
#include "kernels/utils.cuh"

#include <cuda_fp16.h>

namespace dollama
{

// 1D 起動のスレッド数 (activation と揃える)。
static constexpr int GEGLU_THREADS = 256;

// ----------------------------------------------------------------
// GEGLU: out[t,c] = x1[t,c] * gelu_erf(x2[t,c])
//   x1 = in[t*(2d) + c], x2 = in[t*(2d) + d + c]
//   gelu_erf(x) = 0.5 * x * (1 + erf(x * 0.70710678f))
// ----------------------------------------------------------------
__global__ void geglu_fp16(const __half* in, __half* out, int tokens, int d)
{
    const long total = static_cast<long>(tokens) * d;
    const long idx   = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total)
    {
        return;
    }

    // 出力線形 idx を (t, c) に分解 (出力は [tokens, d])。
    const int c = static_cast<int>(idx % d);
    const int t = static_cast<int>(idx / d);

    // 入力 [tokens, 2d] の前半 x1 / 後半 x2。
    const long row_base = static_cast<long>(t) * (2 * d);
    const float x1 = __half2float(in[row_base + c]);
    const float x2 = __half2float(in[row_base + d + c]);

    // erf 版 gelu (FP32)。
    const float g = 0.5f * x2 * (1.0f + erff(x2 * 0.70710678f));

    out[idx] = __float2half(x1 * g);
}

// ----------------------------------------------------------------
// ホストラッパー
// ----------------------------------------------------------------
void launch_geglu(const __half* d_in, __half* d_out, int tokens, int d)
{
    if (tokens <= 0 || d <= 0)
    {
        return;
    }
    const long total    = static_cast<long>(tokens) * d;
    const long blocks_l  = (total + GEGLU_THREADS - 1) / GEGLU_THREADS;
    // SDXL 最大 (tokens=4096 × d=5120 ≈ 2.1e7) でも 2^31 グリッドに十分収まる。
    const int blocks = static_cast<int>(blocks_l);
    geglu_fp16<<<blocks, GEGLU_THREADS>>>(d_in, d_out, tokens, d);
    CUDA_CHECK_KERNEL();
}

} // namespace dollama

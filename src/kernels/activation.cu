// 要素ごと活性化関数 カーネル実装 (Phase 2 マイルストーン 2-2-2)
// 対象: RTX5080 (Blackwell / sm_120) / CUDA Runtime API のみ
//
// 設計判断 (後続カーネルでも再利用):
//   - 入出力 FP16 / 内部計算 FP32。exp/erf を FP16 のまま取ると精度が破綻するため、
//     必ず __half2float でデコードして FP32 で計算する。
//   - GeLU の主経路は erf 版 (SDXL/diffusers GEGLU のデフォルト)。tanh 近似は併設のみ。
//   - in-place 安全: 各スレッドは out[i] = f(in[i]) のみで他要素を参照しないため、
//     d_in == d_out でもレースは発生しない。
//   - 1D グリッド threads=256 (launch_vector_add と同規約)。
#include "kernels/activation.cuh"
#include "kernels/utils.cuh"

#include <cuda_fp16.h>

namespace dollama
{

// 1D 起動のスレッド数 (vector_add と揃える)。
static constexpr int THREADS = 256;

// ----------------------------------------------------------------
// SiLU: f(x) = x * sigmoid(x) = x / (1 + exp(-x))
// ----------------------------------------------------------------
__global__ void silu_fp16(const __half* in, __half* out, int n)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
    {
        return;
    }
    const float x   = __half2float(in[idx]);
    const float sig = 1.0f / (1.0f + __expf(-x));
    out[idx] = __float2half(x * sig);
}

// ----------------------------------------------------------------
// GeLU 正確版 (erf): f(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
//   1/sqrt(2) = 0.70710678f
// ----------------------------------------------------------------
__global__ void gelu_erf_fp16(const __half* in, __half* out, int n)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
    {
        return;
    }
    const float x = __half2float(in[idx]);
    const float y = 0.5f * x * (1.0f + erff(x * 0.70710678f));
    out[idx] = __float2half(y);
}

// ----------------------------------------------------------------
// GeLU tanh 近似版:
//   f(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//   sqrt(2/pi) = 0.7978845608f
// ----------------------------------------------------------------
__global__ void gelu_tanh_fp16(const __half* in, __half* out, int n)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
    {
        return;
    }
    const float x     = __half2float(in[idx]);
    const float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
    const float y     = 0.5f * x * (1.0f + tanhf(inner));
    out[idx] = __float2half(y);
}

// ----------------------------------------------------------------
// ホストラッパー
// ----------------------------------------------------------------
void launch_silu(const __half* d_in, __half* d_out, int n)
{
    if (n <= 0)
    {
        return;
    }
    const int blocks = ceil_div(n, THREADS);
    silu_fp16<<<blocks, THREADS>>>(d_in, d_out, n);
    CUDA_CHECK_KERNEL();
}

void launch_gelu(const __half* d_in, __half* d_out, int n)
{
    if (n <= 0)
    {
        return;
    }
    const int blocks = ceil_div(n, THREADS);
    gelu_erf_fp16<<<blocks, THREADS>>>(d_in, d_out, n);
    CUDA_CHECK_KERNEL();
}

void launch_gelu_tanh(const __half* d_in, __half* d_out, int n)
{
    if (n <= 0)
    {
        return;
    }
    const int blocks = ceil_div(n, THREADS);
    gelu_tanh_fp16<<<blocks, THREADS>>>(d_in, d_out, n);
    CUDA_CHECK_KERNEL();
}

} // namespace dollama

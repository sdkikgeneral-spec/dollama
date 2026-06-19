// broadcast bias add カーネル実装 (Phase 2 マイルストーン 2-5 / UNet)
// 対象: RTX5080 (Blackwell / sm_120) / CUDA Runtime API のみ
//
// 設計判断:
//   - 入出力 FP16 / 内部加算は FP32 蓄積 (elementwise.cu と同規約)。
//   - 1D グリッド (1 スレッド = 出力 1 要素)。出力線形 idx から「どの bias 要素を
//     引くか」のチャネル index を求めて加算するだけ (積和なし)。
//   - in-place 安全: 各スレッドが自分の 1 要素のみ読み書きするため out==in が安全。
//   (a) rowvec: bias index = idx % C。
//   (b) channel: bias index = (idx / (H*W)) % C。
#include "kernels/bias_add.cuh"
#include "kernels/utils.cuh"

#include <cuda_fp16.h>

namespace dollama
{

// 1D カーネルの 1 ブロックあたりスレッド数。
static constexpr int BA_THREADS = 256;

// ----------------------------------------------------------------
// (a) row-vector bias: out[t,c] = in[t,c] + bias[c]
//   出力線形 idx の列 c = idx % C を bias index に使う。
// ----------------------------------------------------------------
__global__ void bias_add_rowvec_fp16(const __half* in,
                                     const __half* bias,
                                     __half*       out,
                                     long          total,
                                     int           C)
{
    const long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total)
    {
        return;
    }
    const int   c = static_cast<int>(idx % C);
    const float v = __half2float(in[idx]) + __half2float(bias[c]);
    out[idx] = __float2half(v);
}

// ----------------------------------------------------------------
// (b) per-channel scalar bias: out[n,c,h,w] = in[n,c,h,w] + bias[c]
//   出力線形 idx から c = (idx / HW) % C を求める。
// ----------------------------------------------------------------
__global__ void bias_add_channel_fp16(const __half* in,
                                      const __half* bias,
                                      __half*       out,
                                      long          total,
                                      int           C,
                                      int           HW)
{
    const long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total)
    {
        return;
    }
    const int   c = static_cast<int>((idx / HW) % C);
    const float v = __half2float(in[idx]) + __half2float(bias[c]);
    out[idx] = __float2half(v);
}

// ----------------------------------------------------------------
// ホストラッパー: (a) row-vector bias
// ----------------------------------------------------------------
void launch_bias_add_rowvec(const __half* d_in,
                            const __half* d_bias,
                            __half*       d_out,
                            int           tokens,
                            int           C)
{
    if (tokens <= 0 || C <= 0)
    {
        return;
    }
    const long total    = static_cast<long>(tokens) * C;
    const long blocks_l = (total + BA_THREADS - 1) / BA_THREADS;
    const int  blocks   = static_cast<int>(blocks_l);
    bias_add_rowvec_fp16<<<blocks, BA_THREADS>>>(d_in, d_bias, d_out, total, C);
    CUDA_CHECK_KERNEL();
}

// ----------------------------------------------------------------
// ホストラッパー: (b) per-channel scalar bias
// ----------------------------------------------------------------
void launch_bias_add_channel(const __half* d_in,
                             const __half* d_bias,
                             __half*       d_out,
                             int           N,
                             int           C,
                             int           H,
                             int           W)
{
    if (N <= 0 || C <= 0 || H <= 0 || W <= 0)
    {
        return;
    }
    const int  HW       = H * W;
    const long total    = static_cast<long>(N) * C * HW;
    const long blocks_l = (total + BA_THREADS - 1) / BA_THREADS;
    const int  blocks   = static_cast<int>(blocks_l);
    bias_add_channel_fp16<<<blocks, BA_THREADS>>>(d_in, d_bias, d_out, total, C, HW);
    CUDA_CHECK_KERNEL();
}

} // namespace dollama

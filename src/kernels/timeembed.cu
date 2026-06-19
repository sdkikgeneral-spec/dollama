// sinusoidal time embedding カーネル実装 (Phase 2 マイルストーン 2-5 / UNet)
// 対象: RTX5080 (Blackwell / sm_120) / CUDA Runtime API のみ
//
// 設計判断:
//   - diffusers get_timestep_embedding に厳密一致 (flip_sin_to_cos=True →
//     cos が先頭 half、sin が後半 half)。downscale_freq_shift=0 なので分母は half。
//   - 出力 FP16 / 内部計算 FP32。exp/cos/sin は FP32 で取る (FP16 直接は精度破綻)。
//   - 1 スレッド = freq index i を担当し、out[i]=cos(args), out[half+i]=sin(args) の
//     2 要素を書く。half スレッドで全出力を埋める。
//   - timestep はスカラなのでカーネル引数に値渡しする (device バッファ不要)。
#include "kernels/timeembed.cuh"
#include "kernels/utils.cuh"

#include <cuda_fp16.h>

namespace dollama
{

// 1D 起動のスレッド数。SDXL dim=320 → half=160 なので 1 ブロックで収まる。
static constexpr int TE_THREADS = 256;

// ----------------------------------------------------------------
// time embedding カーネル:
//   freqs[i] = exp(-log(max_period) * i / half)
//   args     = timestep * freqs[i]
//   out[i]        = cos(args)
//   out[half + i] = sin(args)
// 1 スレッド = freq index i (i = 0..half-1)。
// ----------------------------------------------------------------
__global__ void timestep_embedding_fp16(__half* out,
                                        float   timestep,
                                        int     half,
                                        float   log_max_period)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= half)
    {
        return;
    }

    // exponent = -log(max_period) * i / half。
    const float exponent = -log_max_period * static_cast<float>(i) / static_cast<float>(half);
    const float freq     = __expf(exponent);
    const float arg      = timestep * freq;

    // flip_sin_to_cos=True: cos が先 (0..half-1)、sin が後 (half..2*half-1)。
    out[i]        = __float2half(cosf(arg));
    out[half + i] = __float2half(sinf(arg));
}

// ----------------------------------------------------------------
// ホストラッパー
// ----------------------------------------------------------------
void launch_timestep_embedding(__half* d_out,
                               float   timestep,
                               int     dim,
                               float   max_period)
{
    if (dim <= 0)
    {
        return;
    }
    const int   half           = dim / 2;
    const float log_max_period = logf(max_period);

    const int blocks = ceil_div(half, TE_THREADS);
    timestep_embedding_fp16<<<blocks, TE_THREADS>>>(d_out, timestep, half, log_max_period);
    CUDA_CHECK_KERNEL();
}

} // namespace dollama

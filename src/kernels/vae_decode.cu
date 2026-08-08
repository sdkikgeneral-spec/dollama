// SDXL AutoencoderKL decoder — 実装 (Phase 2 マイルストーン 2-4)
// 対象: RTX5080 (Blackwell / sm_120) / CUDA Runtime API のみ
//
// 既存プリミティブ (conv2d / group_norm / silu / attention / gemm / add /
// upsample_nearest2x) を組み上げて latent[1,4,128,128] -> image[1,3,1024,1024]
// を生成する。post_quant_conv 〜 up_block[1] までは FP16 中間 (各カーネル内部 FP32
// 蓄積)、up_block[2] 以降は FP32 中間で計算する (理由は下記)。
//
// 数値方針メモ (実測で確定):
//   - post_quant_conv 〜 up_block[1] は活性が FP16 範囲 (±65504) に収まり、各中間段
//     ゴールデン (post_quant_conv_out / conv_in_out / mid_block_out / up_block_0_out /
//     up_block_1_out) と一致する。ここは FP16 中間で十分。
//   - up_block[2] / up_block[3] は残差ブロック内で conv2 出力や conv_shortcut 出力が
//     個別に FP16 範囲を超え (実測 up3 full-res で ±100000 超)、残差加算で打ち消されて
//     最終 ±18000 程度に収まる。中間を FP16 で持つと打ち消し前に Inf 化し、後段
//     GroupNorm の分散計算で NaN がグループ全体へ伝播する (実測: FP16 中間だと
//     up3.r0 で min=-65504/bad>0 → up3.r1 全 NaN → final 全 NaN)。
//   - よって up_block[2] 以降のデータ経路を FP32 で保持する (FP32 版カーネルを本 TU に
//     用意)。重み/gamma/beta/bias は safetensors の FP16 のまま read-only で使う。
//     この構成で final SSIM=0.99999 / MAE=5.5e-4 を達成 (FP16 のみでは到達不可)。
//
// VRAM 設計:
//   段間の受け渡しは 2 本の「キャリーバッファ」(各 268M 要素 = 512MB) を
//   ping-pong する。各段内のスクラッチは局所 Scratch がスコープ脱出時に解放し、
//   ピーク VRAM を抑える。最大解像度は up2 出力 256ch x 1024x1024 = 268M 要素。

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "kernels/vae_decode.cuh"
#include "kernels/conv2d.cuh"
#include "kernels/groupnorm.cuh"
#include "kernels/activation.cuh"
#include "kernels/attention.cuh"
#include "kernels/gemm.cuh"
#include "kernels/elementwise.cuh"
#include "kernels/utils.cuh"

namespace dollama
{

// ----------------------------------------------------------------
// デバッグ用: 環境変数 VAE_DEBUG がセットされていれば、指定デバイスバッファの
// 統計 (min/max/NaN個数) と左上 crop 先頭値を表示する。配線切り分け専用。
// ----------------------------------------------------------------
static bool vae_debug_enabled()
{
    static int e = -1;
    if (e < 0)
    {
        e = std::getenv("VAE_DEBUG") != nullptr ? 1 : 0;
    }
    return e == 1;
}

static void dbg_stat(const char* name, const __half* d_buf, size_t n)
{
    if (!vae_debug_enabled())
    {
        return;
    }
    cudaDeviceSynchronize();
    std::vector<__half> h(n);
    cudaMemcpy(h.data(), d_buf, n * sizeof(__half), cudaMemcpyDeviceToHost);
    float mn = 1e30f, mx = -1e30f;
    long bad = 0;
    for (size_t i = 0; i < n; ++i)
    {
        float v = __half2float(h[i]);
        if (std::isnan(v) || std::isinf(v)) { ++bad; continue; }
        if (v < mn) mn = v;
        if (v > mx) mx = v;
    }
    std::printf("[VAE_DEBUG] %-24s n=%zu min=%g max=%g bad=%ld head=%g\n",
                name, n, mn, mx, bad, __half2float(h[0]));
    std::fflush(stdout);
}

static void dbg_stat_f32(const char* name, const float* d_buf, size_t n)
{
    if (!vae_debug_enabled())
    {
        return;
    }
    cudaDeviceSynchronize();
    std::vector<float> h(n);
    cudaMemcpy(h.data(), d_buf, n * sizeof(float), cudaMemcpyDeviceToHost);
    float mn = 1e30f, mx = -1e30f;
    long bad = 0;
    for (size_t i = 0; i < n; ++i)
    {
        float v = h[i];
        if (std::isnan(v) || std::isinf(v)) { ++bad; continue; }
        if (v < mn) mn = v;
        if (v > mx) mx = v;
    }
    std::printf("[VAE_DEBUG] %-24s n=%zu min=%g max=%g bad=%ld head=%g\n",
                name, n, mn, mx, bad, h[0]);
    std::fflush(stdout);
}


// ----------------------------------------------------------------
// 転置カーネル: [C, S] (row-major, S=H*W) <-> [S, C]
// AttnBlock で group_norm 後の [1,C,H,W] を系列×チャネル [S,C] に並べ替える。
// chw_to_sc: in[c*S + s] -> out[s*C + c]
// sc_to_chw: in[s*C + c] -> out[c*S + s]
// static: ヘッダオンリー規約に合わせ内部リンケージ。
// ----------------------------------------------------------------
static __global__ void k_chw_to_sc(const __half* in, __half* out, int C, int S)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < C * S)
    {
        const int c = idx / S;
        const int s = idx - c * S;
        out[s * C + c] = in[idx];
    }
}

static __global__ void k_sc_to_chw(const __half* in, __half* out, int C, int S)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < C * S)
    {
        const int s = idx / C;
        const int c = idx - s * C;
        out[c * S + s] = in[idx];
    }
}

static void launch_chw_to_sc(const __half* d_in, __half* d_out, int C, int S)
{
    const int threads = 256;
    const int blocks  = ceil_div(C * S, threads);
    k_chw_to_sc<<<blocks, threads>>>(d_in, d_out, C, S);
    CUDA_CHECK_KERNEL();
}

static void launch_sc_to_chw(const __half* d_in, __half* d_out, int C, int S)
{
    const int threads = 256;
    const int blocks  = ceil_div(C * S, threads);
    k_sc_to_chw<<<blocks, threads>>>(d_in, d_out, C, S);
    CUDA_CHECK_KERNEL();
}

// ----------------------------------------------------------------
// bias 加算カーネル: x[s*C + c] += bias[c]  ([S, C] レイアウト)
// gemm (transB) には bias 引数が無いため Linear の bias をここで足す。
// ----------------------------------------------------------------
static __global__ void k_add_bias_sc(__half* x, const __half* bias, int S, int C)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < S * C)
    {
        const int c = idx - (idx / C) * C;
        x[idx] = __float2half(__half2float(x[idx]) + __half2float(bias[c]));
    }
}

static void launch_add_bias_sc(__half* d_x, const __half* d_bias, int S, int C)
{
    const int threads = 256;
    const int blocks  = ceil_div(S * C, threads);
    k_add_bias_sc<<<blocks, threads>>>(d_x, d_bias, S, C);
    CUDA_CHECK_KERNEL();
}

// ----------------------------------------------------------------
// ODR 注意 (T1b / C0 修正): 以下は本 TU (translation unit) ローカルの実装型で、
// 必ず匿名 namespace の中に置くこと。外に出すと ODR 違反で実行時に落ちる。
//
//   機序: unet.cu と vae_decode.cu は同名の DeviceWeights / Scratch を
//   それぞれ独自レイアウトで定義している。namespace dollama 直下 (外部リンケージ)
//   に置くと、暗黙 inline のデストラクタが COMDAT 弱シンボルとして出力され、
//   リンカが同名 dtor のうち「片方だけ」を選んで両 TU の全 call site に適用する。
//   どちらが勝つかはリンク単位ごとに変わるため、あるバイナリでは 0xC0000005
//   (アクセス違反)、別のバイナリでは単なるメモリリークとして現れる。
//
//   発火点: L-2 ランタイム LoRA で unet 側 DeviceWeights にだけ 4 メンバ目
//   std::set<std::string> patched_ を足したことでレイアウトが決定的にずれ、
//   ~DiffusionPipeline (VAE ハンドルと UNet ハンドルが同居して破棄される経路) で
//   AV が顕在化した。Scratch も ptrs_ が両方先頭なので現状は無症状なだけの同型の地雷。
//
//   よって「他 TU から参照されていないから外でもよい」ではなく、参照されていない
//   からこそ匿名 namespace が正解。善意で外に出すと再発する。
// ----------------------------------------------------------------
namespace
{
// ----------------------------------------------------------------
// 重み管理: SafeTensors から FP16 テンソルを GPU へアップロードし、
// デバイスポインタを名前で引けるようにする小ヘルパ。
// vae_weights は全テンソル FP16 なので raw bytes をそのまま転送する。
// ----------------------------------------------------------------
class DeviceWeights
{
public:
    explicit DeviceWeights(const SafeTensors& st) : st_(st) {}

    ~DeviceWeights()
    {
        for (__half* p : ptrs_)
        {
            cudaFree(p);
        }
    }

    // 名前から FP16 デバイスポインタを得る (初回でアップロード・以後キャッシュ)。
    const __half* get(const std::string& name)
    {
        auto it = cache_.find(name);
        if (it != cache_.end())
        {
            return it->second;
        }
        if (st_.dtype(name) != StDtype::F16)
        {
            throw std::runtime_error("vae_decode: weight '" + name + "' is not FP16");
        }
        size_t nbytes = 0;
        const uint8_t* src = st_.tensor_bytes(name, nbytes);
        __half* d_ptr = nullptr;
        CUDA_CHECK(cudaMalloc(&d_ptr, nbytes));
        CUDA_CHECK(cudaMemcpy(d_ptr, src, nbytes, cudaMemcpyHostToDevice));
        ptrs_.push_back(d_ptr);
        cache_[name] = d_ptr;
        return d_ptr;
    }

    // S3-D: 常駐ハンドル生成時に全 FP16 重みを先読みして upload しておく。
    // こうすると VAE decode 本体の計測区間では重み転送が一切発生せず、
    // 生成あたり ~3.89s の固定 cudaMalloc+H2D コストが初回 create に寄せられる。
    // F16 以外のテンソル (もしあれば) は VAE では使わないのでスキップする。
    void upload_all()
    {
        for (const std::string& name : st_.names())
        {
            if (st_.dtype(name) == StDtype::F16)
            {
                (void)get(name);
            }
        }
    }

private:
    const SafeTensors&             st_;
    std::map<std::string, __half*> cache_;
    std::vector<__half*>           ptrs_;
};

// ----------------------------------------------------------------
// 局所スクラッチプール: 段内で確保し、スコープ脱出 (デストラクタ) で全解放する。
// ----------------------------------------------------------------
class Scratch
{
public:
    // n 要素 (FP16) を確保したデバイスバッファを返す。
    __half* alloc(size_t n)
    {
        __half* p = nullptr;
        CUDA_CHECK(cudaMalloc(&p, n * sizeof(__half)));
        ptrs_.push_back(p);
        return p;
    }

    ~Scratch()
    {
        for (__half* p : ptrs_)
        {
            cudaFree(p);
        }
    }

private:
    std::vector<__half*> ptrs_;
};

// FP32 版スクラッチプール (up2 以降の FP32 経路用)。
class Scratch32
{
public:
    float* alloc(size_t n)
    {
        float* p = nullptr;
        CUDA_CHECK(cudaMalloc(&p, n * sizeof(float)));
        ptrs_.push_back(p);
        return p;
    }

    ~Scratch32()
    {
        for (float* p : ptrs_)
        {
            cudaFree(p);
        }
    }

private:
    std::vector<float*> ptrs_;
};
} // namespace (匿名): TU ローカル型 DeviceWeights / Scratch / Scratch32 ここまで

// ----------------------------------------------------------------
// FP32 中間バリアント (up_block の高振幅段専用)
// ----------------------------------------------------------------
// 数値方針: up_block 段の残差ブロック内では conv2 出力や conv_shortcut 出力が
// 個別には FP16 範囲 (65504) を超え、残差加算で打ち消されて最終 18000 程度に
// 収まる。中間を FP16 で持つと打ち消し前に Inf 化し、後段 GroupNorm の分散計算で
// NaN がグループ全体へ伝播する (実測: up3.r0 で min=-65504/bad>0 -> up3.r1 全 NaN)。
// そこで up2 以降の活性バッファを FP32 で保持する。重み/gamma/beta/bias は
// safetensors の FP16 のまま (read only)、データ経路だけ FP32 にする。
// 各カーネルは FP16 版と同一アルゴリズム (内部蓄積も FP32) で、入出力 dtype だけ
// float へ変えたもの。

// FP16 -> FP32 変換 (キャリーバッファをまたぐ境界で使用)。
static __global__ void k_h2f(const __half* in, float* out, long n)
{
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long)gridDim.x * blockDim.x)
    {
        out[i] = __half2float(in[i]);
    }
}

// FP32 -> FP16 変換 (最終 image など FP16 出力へ戻す境界で使用)。
static __global__ void k_f2h(const float* in, __half* out, long n)
{
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long)gridDim.x * blockDim.x)
    {
        out[i] = __float2half(in[i]);
    }
}

static void launch_h2f(const __half* d_in, float* d_out, long n)
{
    const int threads = 256;
    long b = (n + threads - 1) / threads;
    const int blocks = (int)(b < 65535 ? b : 65535);
    k_h2f<<<blocks, threads>>>(d_in, d_out, n);
    CUDA_CHECK_KERNEL();
}

static void launch_f2h(const float* d_in, __half* d_out, long n)
{
    const int threads = 256;
    long b = (n + threads - 1) / threads;
    const int blocks = (int)(b < 65535 ? b : 65535);
    k_f2h<<<blocks, threads>>>(d_in, d_out, n);
    CUDA_CHECK_KERNEL();
}

// FP32 elementwise add: out[i] = a[i] + b[i]
static __global__ void k_add_f32(const float* a, const float* b, float* out, long n)
{
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long)gridDim.x * blockDim.x)
    {
        out[i] = a[i] + b[i];
    }
}

static void launch_add_f32(const float* d_a, const float* d_b, float* d_out, long n)
{
    const int threads = 256;
    long bl = (n + threads - 1) / threads;
    const int blocks = (int)(bl < 65535 ? bl : 65535);
    k_add_f32<<<blocks, threads>>>(d_a, d_b, d_out, n);
    CUDA_CHECK_KERNEL();
}

// FP32 SiLU: f(x) = x / (1 + exp(-x))
static __global__ void k_silu_f32(const float* in, float* out, long n)
{
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long)gridDim.x * blockDim.x)
    {
        const float x = in[i];
        out[i] = x / (1.0f + __expf(-x));
    }
}

static void launch_silu_f32(const float* d_in, float* d_out, long n)
{
    const int threads = 256;
    long bl = (n + threads - 1) / threads;
    const int blocks = (int)(bl < 65535 ? bl : 65535);
    k_silu_f32<<<blocks, threads>>>(d_in, d_out, n);
    CUDA_CHECK_KERNEL();
}

// FP32 conv2d (データ float / 重み bias FP16, 内部 FP32 蓄積)。
// conv2d.cu の conv2d_fp16 と同一アルゴリズム。
static __global__ void k_conv2d_f32(const float* in, const __half* weight, const __half* bias,
                                    float* out, int N, int Cin, int H, int W, int Cout,
                                    int KH, int KW, int Hout, int Wout,
                                    int stride_h, int stride_w, int pad_h, int pad_w,
                                    int dilation_h, int dilation_w)
{
    const long total = (long)N * Cout * Hout * Wout;
    for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += (long)gridDim.x * blockDim.x)
    {
        const int wo = (int)(idx % Wout);
        long t = idx / Wout;
        const int ho = (int)(t % Hout);
        t /= Hout;
        const int co = (int)(t % Cout);
        const int n  = (int)(t / Cout);
        const int hi0 = ho * stride_h - pad_h;
        const int wi0 = wo * stride_w - pad_w;
        float acc = 0.0f;
        for (int ci = 0; ci < Cin; ++ci)
        {
            const long in_ch_base = ((long)n * Cin + ci) * H * W;
            const long w_ch_base  = ((long)co * Cin + ci) * KH * KW;
            for (int kh = 0; kh < KH; ++kh)
            {
                const int hi = hi0 + kh * dilation_h;
                if (hi < 0 || hi >= H) { continue; }
                for (int kw = 0; kw < KW; ++kw)
                {
                    const int wi = wi0 + kw * dilation_w;
                    if (wi < 0 || wi >= W) { continue; }
                    const float x = in[in_ch_base + (long)hi * W + wi];
                    const float wv = __half2float(weight[w_ch_base + kh * KW + kw]);
                    acc += x * wv;
                }
            }
        }
        if (bias != nullptr) { acc += __half2float(bias[co]); }
        out[idx] = acc;
    }
}

static void launch_conv2d_f32(const float* d_in, const __half* d_weight, const __half* d_bias,
                              float* d_out, int N, int Cin, int H, int W, int Cout,
                              int KH, int KW, int stride_h, int stride_w,
                              int pad_h, int pad_w, int dilation_h, int dilation_w)
{
    const int Hout = (H + 2 * pad_h - dilation_h * (KH - 1) - 1) / stride_h + 1;
    const int Wout = (W + 2 * pad_w - dilation_w * (KW - 1) - 1) / stride_w + 1;
    const long total = (long)N * Cout * Hout * Wout;
    long bl = (total + 255) / 256;
    const int blocks = (int)(bl < 65535 ? bl : 65535);
    k_conv2d_f32<<<blocks, 256>>>(d_in, d_weight, d_bias, d_out, N, Cin, H, W, Cout,
                                  KH, KW, Hout, Wout, stride_h, stride_w, pad_h, pad_w,
                                  dilation_h, dilation_w);
    CUDA_CHECK_KERNEL();
}

// FP32 GroupNorm (データ float / gamma beta FP16, 内部 FP32)。
// groupnorm.cu の group_norm_fp16 と同一 (1 グループ=1 ブロック 1 パス sum/sum-sq)。
static __global__ void k_group_norm_f32(const float* in, const __half* gamma, const __half* beta,
                                        float* out, int C, int HW, int cpg, int num_groups, float eps)
{
    const int block_id = blockIdx.x;
    const int g = block_id % num_groups;
    const int n = block_id / num_groups;
    const int c_begin = g * cpg;
    const int group_size = cpg * HW;
    const long base = ((long)n * C + c_begin) * (long)HW;

    __shared__ float smem[256 / 32];

    float local_sum = 0.0f, local_sq = 0.0f;
    for (int i = threadIdx.x; i < group_size; i += blockDim.x)
    {
        const float x = in[base + i];
        local_sum += x;
        local_sq  += x * x;
    }
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;
    const int nwarps = (blockDim.x + 31) >> 5;
    // warp 和 (インライン)。
    float v = local_sum;
    for (int o = 16; o > 0; o >>= 1) { v += __shfl_down_sync(0xffffffffu, v, o); }
    if (lane == 0) { smem[wid] = v; }
    __syncthreads();
    float sum = 0.0f;
    if (wid == 0)
    {
        float wv = (lane < nwarps) ? smem[lane] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) { wv += __shfl_down_sync(0xffffffffu, wv, o); }
        if (lane == 0) { smem[0] = wv; }
    }
    __syncthreads();
    sum = smem[0];
    __syncthreads();

    float v2 = local_sq;
    for (int o = 16; o > 0; o >>= 1) { v2 += __shfl_down_sync(0xffffffffu, v2, o); }
    if (lane == 0) { smem[wid] = v2; }
    __syncthreads();
    float sum_sq = 0.0f;
    if (wid == 0)
    {
        float wv = (lane < nwarps) ? smem[lane] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) { wv += __shfl_down_sync(0xffffffffu, wv, o); }
        if (lane == 0) { smem[0] = wv; }
    }
    __syncthreads();
    sum_sq = smem[0];
    __syncthreads();

    const float inv_k = 1.0f / (float)group_size;
    const float mean = sum * inv_k;
    float var = sum_sq * inv_k - mean * mean;
    var = fmaxf(var, 0.0f);
    const float inv_std = rsqrtf(var + eps);
    for (int i = threadIdx.x; i < group_size; i += blockDim.x)
    {
        const int c = c_begin + (i / HW);
        const float x = in[base + i];
        const float gm = __half2float(gamma[c]);
        const float bt = __half2float(beta[c]);
        out[base + i] = (x - mean) * inv_std * gm + bt;
    }
}

static void launch_group_norm_f32(const float* d_in, const __half* d_gamma, const __half* d_beta,
                                  float* d_out, int N, int C, int H, int W, int num_groups, float eps)
{
    const int cpg = C / num_groups;
    const int HW  = H * W;
    const int blocks = N * num_groups;
    k_group_norm_f32<<<blocks, 256>>>(d_in, d_gamma, d_beta, d_out, C, HW, cpg, num_groups, eps);
    CUDA_CHECK_KERNEL();
}

// FP32 nearest 2x upsample: [N,C,H,W] -> [N,C,2H,2W]
static __global__ void k_upsample2x_f32(const float* in, float* out, int N, int C, int H, int W)
{
    const int H2 = H << 1, W2 = W << 1;
    const long total = (long)N * C * H2 * W2;
    for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += (long)gridDim.x * blockDim.x)
    {
        const int wo = (int)(idx % W2);
        long t = idx / W2;
        const int ho = (int)(t % H2);
        t /= H2;
        const int c = (int)(t % C);
        const int n = (int)(t / C);
        const int hi = ho >> 1, wi = wo >> 1;
        out[idx] = in[(((long)n * C + c) * H + hi) * W + wi];
    }
}

static void launch_upsample2x_f32(const float* d_in, float* d_out, int N, int C, int H, int W)
{
    const long total = (long)N * C * (H << 1) * (W << 1);
    long bl = (total + 255) / 256;
    const int blocks = (int)(bl < 65535 ? bl : 65535);
    k_upsample2x_f32<<<blocks, 256>>>(d_in, d_out, N, C, H, W);
    CUDA_CHECK_KERNEL();
}

// FP32 ResnetBlock2D (FP16 版 resnet_block と同一構造)。
static void resnet_block_f32(DeviceWeights& w, const std::string& prefix,
                             const float* d_x, int Cin, int Cout, int H, int W,
                             float* d_out, float* d_tmp_a, float* d_tmp_b)
{
    const float eps = 1e-6f;
    const int groups = 32;
    const long HW = (long)H * W;

    launch_group_norm_f32(d_x, w.get(prefix + "norm1.weight"), w.get(prefix + "norm1.bias"),
                          d_tmp_a, 1, Cin, H, W, groups, eps);
    launch_silu_f32(d_tmp_a, d_tmp_a, (long)Cin * HW);
    launch_conv2d_f32_gemm(d_tmp_a, w.get(prefix + "conv1.weight"), w.get(prefix + "conv1.bias"),
                      d_tmp_b, 1, Cin, H, W, Cout, 3, 3, 1, 1, 1, 1, 1, 1);

    launch_group_norm_f32(d_tmp_b, w.get(prefix + "norm2.weight"), w.get(prefix + "norm2.bias"),
                          d_tmp_a, 1, Cout, H, W, groups, eps);
    launch_silu_f32(d_tmp_a, d_tmp_a, (long)Cout * HW);
    launch_conv2d_f32_gemm(d_tmp_a, w.get(prefix + "conv2.weight"), w.get(prefix + "conv2.bias"),
                      d_out, 1, Cout, H, W, Cout, 3, 3, 1, 1, 1, 1, 1, 1);

    if (Cin == Cout)
    {
        launch_add_f32(d_out, d_x, d_out, (long)Cout * HW);
    }
    else
    {
        launch_conv2d_f32_gemm(d_x, w.get(prefix + "conv_shortcut.weight"),
                          w.get(prefix + "conv_shortcut.bias"),
                          d_tmp_b, 1, Cin, H, W, Cout, 1, 1, 1, 1, 0, 0, 1, 1);
        launch_add_f32(d_out, d_tmp_b, d_out, (long)Cout * HW);
    }
}

// ----------------------------------------------------------------
// ResnetBlock2D (diffusers 準拠)
//   h = conv1(silu(norm1(x)))         norm:GroupNorm32, conv:3x3 pad1
//   h = conv2(silu(norm2(h)))
//   skip = (Cin==Cout) ? x : conv_shortcut(x)   (1x1)
//   out = h + skip
// 解像度は不変。out は x と別バッファ。
//   d_tmp_a / d_tmp_b : スクラッチ ([1,max(Cin,Cout),H,W] を満たすサイズ)。
// ----------------------------------------------------------------
static void resnet_block(DeviceWeights& w, const std::string& prefix,
                         const __half* d_x, int Cin, int Cout, int H, int W,
                         __half* d_out, __half* d_tmp_a, __half* d_tmp_b)
{
    const float eps    = 1e-6f;
    const int   groups = 32;
    const int   HW     = H * W;

    // --- norm1 -> silu -> conv1 (Cin -> Cout) ---
    launch_group_norm(d_x, w.get(prefix + "norm1.weight"), w.get(prefix + "norm1.bias"),
                      d_tmp_a, 1, Cin, H, W, groups, eps);
    launch_silu(d_tmp_a, d_tmp_a, Cin * HW);
    launch_conv2d(d_tmp_a, w.get(prefix + "conv1.weight"), w.get(prefix + "conv1.bias"),
                  d_tmp_b, 1, Cin, H, W, Cout, 3, 3, 1, 1, 1, 1, 1, 1);

    // --- norm2 -> silu -> conv2 (Cout -> Cout) ---
    launch_group_norm(d_tmp_b, w.get(prefix + "norm2.weight"), w.get(prefix + "norm2.bias"),
                      d_tmp_a, 1, Cout, H, W, groups, eps);
    launch_silu(d_tmp_a, d_tmp_a, Cout * HW);
    launch_conv2d(d_tmp_a, w.get(prefix + "conv2.weight"), w.get(prefix + "conv2.bias"),
                  d_out, 1, Cout, H, W, Cout, 3, 3, 1, 1, 1, 1, 1, 1);

    // --- skip ---
    if (Cin == Cout)
    {
        launch_add(d_out, d_x, d_out, Cout * HW);
    }
    else
    {
        // skip = conv_shortcut(x) (1x1)。d_tmp_b にスキップを作って加算。
        launch_conv2d(d_x, w.get(prefix + "conv_shortcut.weight"),
                      w.get(prefix + "conv_shortcut.bias"),
                      d_tmp_b, 1, Cin, H, W, Cout, 1, 1, 1, 1, 0, 0, 1, 1);
        launch_add(d_out, d_tmp_b, d_out, Cout * HW);
    }
}

// ----------------------------------------------------------------
// mid_block の AttnBlock (spatial self-attention, C=512)
//   residual = x ; h = group_norm(x) ; [C,HW]->[HW,C] 転置
//   q/k/v = Linear(h) ; attn = softmax(q k^T / sqrt(C)) v ; o = Linear(attn)
//   [HW,C]->[C,HW] 転置 ; x = o + residual (in-place)
// ----------------------------------------------------------------
static void attn_block(DeviceWeights& w, const std::string& prefix,
                       __half* d_x, int C, int H, int W,
                       __half* d_chw_norm, __half* d_sc_h,
                       __half* d_sc_q, __half* d_sc_k, __half* d_sc_v, __half* d_sc_o)
{
    const float eps   = 1e-6f;
    const int   S     = H * W;
    const float scale = 1.0f / sqrtf(static_cast<float>(C));

    // h = group_norm(x) (CHW) -> [S,C]
    launch_group_norm(d_x, w.get(prefix + "group_norm.weight"), w.get(prefix + "group_norm.bias"),
                      d_chw_norm, 1, C, H, W, 32, eps);
    launch_chw_to_sc(d_chw_norm, d_sc_h, C, S);

    // q/k/v = [S,C] @ W[C,C]^T + b  (gemm transB → bias)
    launch_gemm_fp16(d_sc_h, w.get(prefix + "to_q.weight"), d_sc_q, S, C, C, 1.0f, 0.0f, false, true);
    launch_add_bias_sc(d_sc_q, w.get(prefix + "to_q.bias"), S, C);
    launch_gemm_fp16(d_sc_h, w.get(prefix + "to_k.weight"), d_sc_k, S, C, C, 1.0f, 0.0f, false, true);
    launch_add_bias_sc(d_sc_k, w.get(prefix + "to_k.bias"), S, C);
    launch_gemm_fp16(d_sc_h, w.get(prefix + "to_v.weight"), d_sc_v, S, C, C, 1.0f, 0.0f, false, true);
    launch_add_bias_sc(d_sc_v, w.get(prefix + "to_v.bias"), S, C);

    // attention: B=1,H=1,Sq=Sk=S,Dh=C
    launch_attention(d_sc_q, d_sc_k, d_sc_v, d_sc_o, 1, 1, S, S, C, scale);

    // o = Linear(attn) (to_out.0)。d_sc_h を再利用。
    launch_gemm_fp16(d_sc_o, w.get(prefix + "to_out.0.weight"), d_sc_h, S, C, C, 1.0f, 0.0f, false, true);
    launch_add_bias_sc(d_sc_h, w.get(prefix + "to_out.0.bias"), S, C);

    // [S,C] -> [C,S] を d_chw_norm に戻し、x = o + residual
    launch_sc_to_chw(d_sc_h, d_chw_norm, C, S);
    launch_add(d_x, d_chw_norm, d_x, C * S);
}

// ----------------------------------------------------------------
// VAE decoder 本体
// ----------------------------------------------------------------
// 本体ロジック (重みは DeviceWeights& を呼び出し側から受け取る)。
// 後方互換版 launch_vae_decode と常駐ハンドル版 launch_vae_decode が共有する。
static void launch_vae_decode_impl(DeviceWeights& w,
                                   const __half*  d_latent,
                                   __half*        d_image_out)
{
    const int   groups = 32;
    const float eps    = 1e-6f;

    // 段間キャリーバッファ 2 本を ping-pong。最大解像度 256ch 1024x1024 = 268M 要素。
    const size_t CARRY = static_cast<size_t>(256) * 1024 * 1024;
    __half* d_cur  = nullptr;  // 現在の活性
    __half* d_next = nullptr;  // 次段の出力先
    CUDA_CHECK(cudaMalloc(&d_cur,  CARRY * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_next, CARRY * sizeof(__half)));

    // ============ post_quant_conv (1x1, 4->4) + conv_in (3x3 pad1, 4->512), 128x128 ============
    {
        Scratch sc;
        __half* d_pqc = sc.alloc(static_cast<size_t>(4) * 128 * 128);
        launch_conv2d(d_latent, w.get("post_quant_conv.weight"), w.get("post_quant_conv.bias"),
                      d_pqc, 1, 4, 128, 128, 4, 1, 1, 1, 1, 0, 0, 1, 1);
        dbg_stat("post_quant_conv_out", d_pqc, (size_t)4*128*128);
        // conv_in -> d_cur (512ch 128x128)
        launch_conv2d(d_pqc, w.get("decoder.conv_in.weight"), w.get("decoder.conv_in.bias"),
                      d_cur, 1, 4, 128, 128, 512, 3, 3, 1, 1, 1, 1, 1, 1);
        dbg_stat("conv_in_out", d_cur, (size_t)512*128*128);
    }

    // ============ mid_block: Resnet -> Attn -> Resnet (512ch 128x128) ============
    {
        const int H = 128, W = 128, C = 512;
        const size_t chw = static_cast<size_t>(C) * H * W;
        const size_t S   = static_cast<size_t>(H) * W;
        Scratch sc;
        __half* d_tmp_a = sc.alloc(chw);
        __half* d_tmp_b = sc.alloc(chw);
        // resnet0: 512->512 (d_cur -> d_next)
        resnet_block(w, "decoder.mid_block.resnets.0.", d_cur, C, C, H, W, d_next, d_tmp_a, d_tmp_b);
        // attention (in-place on d_next)
        __half* d_chw_norm = sc.alloc(chw);
        __half* d_sc_h = sc.alloc(S * C);
        __half* d_sc_q = sc.alloc(S * C);
        __half* d_sc_k = sc.alloc(S * C);
        __half* d_sc_v = sc.alloc(S * C);
        __half* d_sc_o = sc.alloc(S * C);
        attn_block(w, "decoder.mid_block.attentions.0.", d_next, C, H, W,
                   d_chw_norm, d_sc_h, d_sc_q, d_sc_k, d_sc_v, d_sc_o);
        // resnet1: 512->512 (d_next -> d_cur)
        resnet_block(w, "decoder.mid_block.resnets.1.", d_next, C, C, H, W, d_cur, d_tmp_a, d_tmp_b);
        dbg_stat("mid_block_out", d_cur, (size_t)512*128*128);
    }
    // d_cur = mid_block_out (512ch 128x128)

    // ============ up_blocks[0]: Resnet x3 (512) -> Upsample, 128->256 ============
    {
        const int H = 128, W = 128, C = 512;
        const size_t chw = static_cast<size_t>(C) * H * W;
        Scratch sc;
        __half* d_a = sc.alloc(chw);
        __half* d_b = sc.alloc(chw);
        __half* d_r = sc.alloc(chw);
        resnet_block(w, "decoder.up_blocks.0.resnets.0.", d_cur, C, C, H, W, d_r, d_a, d_b);
        resnet_block(w, "decoder.up_blocks.0.resnets.1.", d_r, C, C, H, W, d_cur, d_a, d_b);
        resnet_block(w, "decoder.up_blocks.0.resnets.2.", d_cur, C, C, H, W, d_r, d_a, d_b);
        // upsample: nearest2x (128->256) -> 3x3 conv (512->512) -> d_next
        const int H2 = H << 1, W2 = W << 1;
        __half* d_up = sc.alloc(static_cast<size_t>(C) * H2 * W2);
        launch_upsample_nearest2x(d_r, d_up, 1, C, H, W);
        launch_conv2d(d_up, w.get("decoder.up_blocks.0.upsamplers.0.conv.weight"),
                      w.get("decoder.up_blocks.0.upsamplers.0.conv.bias"),
                      d_next, 1, C, H2, W2, C, 3, 3, 1, 1, 1, 1, 1, 1);
    }
    std::swap(d_cur, d_next);  // d_cur = up0 出力 (512ch 256x256)
    dbg_stat("up_block_0_out", d_cur, (size_t)512*256*256);

    // ============ up_blocks[1]: Resnet x3 (512) -> Upsample, 256->512 ============
    {
        const int H = 256, W = 256, C = 512;
        const size_t chw = static_cast<size_t>(C) * H * W;
        Scratch sc;
        __half* d_a = sc.alloc(chw);
        __half* d_b = sc.alloc(chw);
        __half* d_r = sc.alloc(chw);
        resnet_block(w, "decoder.up_blocks.1.resnets.0.", d_cur, C, C, H, W, d_r, d_a, d_b);
        resnet_block(w, "decoder.up_blocks.1.resnets.1.", d_r, C, C, H, W, d_cur, d_a, d_b);
        resnet_block(w, "decoder.up_blocks.1.resnets.2.", d_cur, C, C, H, W, d_r, d_a, d_b);
        const int H2 = H << 1, W2 = W << 1;
        __half* d_up = sc.alloc(static_cast<size_t>(C) * H2 * W2);
        launch_upsample_nearest2x(d_r, d_up, 1, C, H, W);
        launch_conv2d(d_up, w.get("decoder.up_blocks.1.upsamplers.0.conv.weight"),
                      w.get("decoder.up_blocks.1.upsamplers.0.conv.bias"),
                      d_next, 1, C, H2, W2, C, 3, 3, 1, 1, 1, 1, 1, 1);
    }
    std::swap(d_cur, d_next);  // d_cur = up1 出力 (512ch 512x512)
    dbg_stat("up_block_1_out", d_cur, (size_t)512*512*512);

    // ============ ここから FP32 経路 ============
    // up2/up3 段の残差ブロック内は FP16 範囲を超えて Inf 化するため、up1 出力を
    // FP32 化し、以降 (up2 -> up3 -> conv_norm_out -> conv_out) を FP32 で計算する。
    // FP32 キャリーバッファ 2 本 (各 256ch x 1024x1024 = 268M 要素 = 1GB) を ping-pong。
    const size_t FCARRY = static_cast<size_t>(256) * 1024 * 1024;
    float* f_cur  = nullptr;
    float* f_next = nullptr;
    CUDA_CHECK(cudaMalloc(&f_cur,  FCARRY * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&f_next, FCARRY * sizeof(float)));
    // up1 出力 (FP16 d_cur, 512ch 512x512) を f_cur へ FP32 化。
    launch_h2f(d_cur, f_cur, static_cast<long>(512) * 512 * 512);

    // ============ up_blocks[2]: Resnet x3 (512->256) -> Upsample, 512->1024 (FP32) ============
    {
        const int H = 512, W = 512;
        const int Cin = 512, Cout = 256;
        const size_t chw_in  = static_cast<size_t>(Cin)  * H * W;
        const size_t chw_out = static_cast<size_t>(Cout) * H * W;
        Scratch32 sc;
        float* d_a  = sc.alloc(chw_in);   // max(Cin,Cout)=512
        float* d_b  = sc.alloc(chw_in);
        float* d_r  = sc.alloc(chw_out);
        float* d_r2 = sc.alloc(chw_out);
        resnet_block_f32(w, "decoder.up_blocks.2.resnets.0.", f_cur, Cin, Cout, H, W, d_r, d_a, d_b);
        resnet_block_f32(w, "decoder.up_blocks.2.resnets.1.", d_r, Cout, Cout, H, W, d_r2, d_a, d_b);
        resnet_block_f32(w, "decoder.up_blocks.2.resnets.2.", d_r2, Cout, Cout, H, W, d_r, d_a, d_b);
        const int H2 = H << 1, W2 = W << 1;
        float* d_up = sc.alloc(static_cast<size_t>(Cout) * H2 * W2);
        launch_upsample2x_f32(d_r, d_up, 1, Cout, H, W);
        launch_conv2d_f32_gemm(d_up, w.get("decoder.up_blocks.2.upsamplers.0.conv.weight"),
                          w.get("decoder.up_blocks.2.upsamplers.0.conv.bias"),
                          f_next, 1, Cout, H2, W2, Cout, 3, 3, 1, 1, 1, 1, 1, 1);
    }
    std::swap(f_cur, f_next);  // f_cur = up2 出力 (256ch 1024x1024)
    dbg_stat_f32("up_block_2_out", f_cur, (size_t)256*1024*1024);

    // ============ up_blocks[3]: Resnet x3 (256->128), upsample なし, 1024x1024 (FP32) ============
    {
        const int H = 1024, W = 1024;
        const int Cin = 256, Cout = 128;
        const size_t chw_in  = static_cast<size_t>(Cin)  * H * W;
        const size_t chw_out = static_cast<size_t>(Cout) * H * W;
        Scratch32 sc;
        float* d_a  = sc.alloc(chw_in);   // 256ch
        float* d_b  = sc.alloc(chw_in);
        float* d_r2 = sc.alloc(chw_out); // 128ch
        resnet_block_f32(w, "decoder.up_blocks.3.resnets.0.", f_cur, Cin, Cout, H, W, f_next, d_a, d_b);
        resnet_block_f32(w, "decoder.up_blocks.3.resnets.1.", f_next, Cout, Cout, H, W, d_r2, d_a, d_b);
        resnet_block_f32(w, "decoder.up_blocks.3.resnets.2.", d_r2, Cout, Cout, H, W, f_next, d_a, d_b);
    }
    std::swap(f_cur, f_next);  // f_cur = up3 出力 (128ch 1024x1024)
    dbg_stat_f32("up_block_3_out", f_cur, (size_t)128*1024*1024);

    // ============ conv_norm_out (GroupNorm32) -> SiLU -> conv_out (3x3 pad1, 128->3) (FP32) ============
    {
        const int H = 1024, W = 1024, C = 128;
        const size_t chw = static_cast<size_t>(C) * H * W;
        Scratch32 sc;
        float* d_n = sc.alloc(chw);
        launch_group_norm_f32(f_cur, w.get("decoder.conv_norm_out.weight"),
                              w.get("decoder.conv_norm_out.bias"),
                              d_n, 1, C, H, W, groups, eps);
        launch_silu_f32(d_n, d_n, (long)C * H * W);
        dbg_stat_f32("conv_norm_out_out", d_n, (size_t)128*1024*1024);
        // conv_out: 128 -> 3。FP32 で計算し、出力を直接 FP16 image へ書く。
        float* d_img32 = sc.alloc(static_cast<size_t>(3) * H * W);
        launch_conv2d_f32_gemm(d_n, w.get("decoder.conv_out.weight"), w.get("decoder.conv_out.bias"),
                          d_img32, 1, C, H, W, 3, 3, 3, 1, 1, 1, 1, 1, 1);
        launch_f2h(d_img32, d_image_out, static_cast<long>(3) * H * W);
        dbg_stat("final_image", d_image_out, (size_t)3*1024*1024);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(f_cur));
    CUDA_CHECK(cudaFree(f_next));
    CUDA_CHECK(cudaFree(d_cur));
    CUDA_CHECK(cudaFree(d_next));
}

// ----------------------------------------------------------------
// 後方互換版: SafeTensors を直接受け取り、毎回ローカルに全重みを
// upload して decode する (test_vae_decode が使用)。
// ----------------------------------------------------------------
void launch_vae_decode(const SafeTensors& weights,
                       const __half*      d_latent,
                       __half*            d_image_out)
{
    DeviceWeights w(weights);
    launch_vae_decode_impl(w, d_latent, d_image_out);
}

// ----------------------------------------------------------------
// 常駐重みハンドル API (S3-D)。
//   DeviceWeights を 1 個ヒープに確保して不透明ポインタとして返す。create 時に
//   upload_all() で全 FP16 重みをデバイスへ常駐させるため、以降の launch では
//   再 malloc / 再 H2D が一切発生しない (生成あたり ~3.89s の固定コストを消す)。
// ----------------------------------------------------------------
VaeWeightsHandle vae_weights_create(const SafeTensors& weights)
{
    DeviceWeights* w = new DeviceWeights(weights);
    // 全重みをデバイスへ常駐させる (一度きり)。以降の get() は即返し。
    w->upload_all();
    return static_cast<VaeWeightsHandle>(w);
}

void vae_weights_destroy(VaeWeightsHandle handle)
{
    if (handle == nullptr)
    {
        return;
    }
    delete static_cast<DeviceWeights*>(handle);
}

void launch_vae_decode(VaeWeightsHandle handle,
                       const __half*    d_latent,
                       __half*          d_image_out)
{
    DeviceWeights& w = *static_cast<DeviceWeights*>(handle);
    launch_vae_decode_impl(w, d_latent, d_image_out);
}

} // namespace dollama

// SDXL 拡散ループ結線モジュール — 実装 (Phase 2 マイルストーン 2-6a / ST-1+ST-2)
// 対象: RTX5080 (Blackwell / sm_120) / CUDA Runtime API のみ
//
// UNet × Nstep + EulerDiscreteScheduler + VAE decode を結線し、golden 埋め込みを
// 入力に noise → latent → 画像 (1024×1024 RGB uint8) を生成する。
// scheduler は host float、UNet/VAE は FP16 デバイス。中間 latent の D2H/H2D は
// 1 step あたり ~256KB で無視可能なので素直に往復する。

#include "infer/diffusion.cuh"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "infer/unet.cuh"
#include "infer/scheduler.hpp"
#include "kernels/vae_decode.cuh"
#include "kernels/utils.cuh"
#include "io/safetensors.hpp"

namespace dollama
{

namespace
{

// SDXL scaling_factor。decode 前に latent をこの値で割る (golden 生成と同一)。
constexpr float kScalingFactor = 0.13025f;

// 形状定数。
constexpr int    kLatentC = 4;
constexpr int    kLatentH = 128;
constexpr int    kLatentW = 128;
constexpr size_t kLatentN = static_cast<size_t>(kLatentC) * kLatentH * kLatentW;

constexpr int    kImgC = 3;
constexpr int    kImgH = 1024;
constexpr int    kImgW = 1024;
constexpr size_t kImgN = static_cast<size_t>(kImgC) * kImgH * kImgW;

constexpr size_t kEhsN  = static_cast<size_t>(77) * 2048;  // encoder_hidden_states
constexpr size_t kTxtN  = 1280;                            // text_embeds
constexpr size_t kTidsN = 6;                               // time_ids

// ----------------------------------------------------------------
// SafeTensors から FP16 テンソルを std::vector<__half> に展開 (要素数検査付き)。
// ----------------------------------------------------------------
std::vector<__half> load_f16(const SafeTensors& st, const std::string& name, size_t expect)
{
    if (st.dtype(name) != StDtype::F16)
    {
        throw std::runtime_error("diffusion load_f16: '" + name + "' is not F16");
    }
    size_t nbytes = 0;
    const uint8_t* p = st.tensor_bytes(name, nbytes);
    const size_t n = nbytes / sizeof(__half);
    if (n != expect)
    {
        throw std::runtime_error("diffusion load_f16: '" + name + "' size mismatch");
    }
    std::vector<__half> out(n);
    std::memcpy(out.data(), p, nbytes);
    return out;
}

// ----------------------------------------------------------------
// xorshift128+ 系の簡易 PRNG + Box-Muller で標準正規乱数を生成する。
// host 完結・決定論的 (seed 固定で再現)。CUDA に依存しない。
// ----------------------------------------------------------------
class Randn
{
public:
    explicit Randn(uint64_t seed)
    {
        // splitmix64 で 2 状態を初期化。
        s0_ = splitmix(seed);
        s1_ = splitmix(seed + 0x9E3779B97F4A7C15ULL);
        has_spare_ = false;
        spare_     = 0.0;
    }

    // 標準正規 N(0,1) を 1 個返す。
    double next()
    {
        if (has_spare_)
        {
            has_spare_ = false;
            return spare_;
        }
        // Box-Muller。u1 は (0,1] を保証 (log(0) 回避)。
        double u1, u2;
        do
        {
            u1 = uniform();
        } while (u1 <= 1e-12);
        u2 = uniform();
        const double mag = std::sqrt(-2.0 * std::log(u1));
        const double z0  = mag * std::cos(2.0 * 3.14159265358979323846 * u2);
        const double z1  = mag * std::sin(2.0 * 3.14159265358979323846 * u2);
        spare_     = z1;
        has_spare_ = true;
        return z0;
    }

private:
    uint64_t s0_, s1_;
    bool     has_spare_;
    double   spare_;

    static uint64_t splitmix(uint64_t x)
    {
        x += 0x9E3779B97F4A7C15ULL;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
        return x ^ (x >> 31);
    }

    // xorshift128+ で [0,1) の一様乱数。
    double uniform()
    {
        uint64_t x       = s0_;
        const uint64_t y = s1_;
        s0_ = y;
        x ^= x << 23;
        s1_ = x ^ y ^ (x >> 17) ^ (y >> 26);
        const uint64_t r = s1_ + y;
        // 上位 53bit を [0,1) double へ。
        return static_cast<double>(r >> 11) * (1.0 / 9007199254740992.0);
    }
};

} // namespace

// ----------------------------------------------------------------
// コンストラクタ: 重み 2 つと golden 埋め込みをロードし、埋め込みをデバイス常駐させる。
// ----------------------------------------------------------------
DiffusionPipeline::DiffusionPipeline(const std::string& unet_weights_path,
                                     const std::string& vae_weights_path,
                                     const std::string& embeds_path)
    : unet_weights_(unet_weights_path)
    , vae_weights_(vae_weights_path)
{
    // golden 埋め込みを host にロード (全 step 使い回すためデバイス常駐させる)。
    SafeTensors embeds(embeds_path);
    std::vector<__half> h_ehs  = load_f16(embeds, "input_encoder_hidden_states_f16", kEhsN);
    std::vector<__half> h_txt  = load_f16(embeds, "input_text_embeds_f16",           kTxtN);
    std::vector<__half> h_tids = load_f16(embeds, "input_time_ids_f16",              kTidsN);

    CUDA_CHECK(cudaMalloc(&d_encoder_hidden_states_, kEhsN  * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_text_embeds_,           kTxtN  * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_time_ids_,              kTidsN * sizeof(__half)));

    CUDA_CHECK(cudaMemcpy(d_encoder_hidden_states_, h_ehs.data(),
                          kEhsN * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_text_embeds_, h_txt.data(),
                          kTxtN * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_time_ids_, h_tids.data(),
                          kTidsN * sizeof(__half), cudaMemcpyHostToDevice));
}

DiffusionPipeline::~DiffusionPipeline()
{
    // デストラクタでは例外を投げない (CUDA_CHECK は使わず素の free)。
    if (d_encoder_hidden_states_ != nullptr) { cudaFree(d_encoder_hidden_states_); }
    if (d_text_embeds_ != nullptr)           { cudaFree(d_text_embeds_); }
    if (d_time_ids_ != nullptr)              { cudaFree(d_time_ids_); }
}

// ----------------------------------------------------------------
// guidance_scale=1.0 の簡略オーバーロード。
// ----------------------------------------------------------------
void DiffusionPipeline::generate(int                   steps,
                                 uint64_t              seed,
                                 std::vector<uint8_t>& rgb_out,
                                 int&                  w,
                                 int&                  h)
{
    generate(steps, seed, 1.0f, rgb_out, w, h);
}

// ----------------------------------------------------------------
// 拡散ループ本体。
// ----------------------------------------------------------------
void DiffusionPipeline::generate(int                   steps,
                                 uint64_t              seed,
                                 float                 guidance_scale,
                                 std::vector<uint8_t>& rgb_out,
                                 int&                  w,
                                 int&                  h)
{
    if (steps <= 0)
    {
        throw std::runtime_error("DiffusionPipeline::generate: steps must be > 0");
    }
    // CFG > 1 は今回 TODO スタブ (テキストエンコード未結線・negative 埋め込みもない)。
    if (std::fabs(guidance_scale - 1.0f) > 1e-6f)
    {
        throw std::runtime_error(
            "DiffusionPipeline::generate: guidance_scale != 1.0 is not supported yet (TODO)");
    }

    // --- scheduler 構築 ---
    EulerDiscreteScheduler sched;
    sched.set_timesteps(steps);
    const std::vector<float>& timesteps = sched.timesteps();
    const std::vector<float>& sigmas    = sched.sigmas();

    // --- 初期ノイズ: randn(seed) * sigmas[0] (init_noise_sigma) ---
    std::vector<float> latent_host(kLatentN);
    {
        Randn rng(seed);
        const double init_sigma = static_cast<double>(sigmas[0]);
        for (size_t k = 0; k < kLatentN; ++k)
        {
            latent_host[k] = static_cast<float>(rng.next() * init_sigma);
        }
    }

    // --- デバイスバッファ確保 ---
    __half* d_latent     = nullptr;  // UNet 入力 (scale_model_input 済み)
    __half* d_noise_pred = nullptr;  // UNet 出力
    __half* d_image      = nullptr;  // VAE 出力
    CUDA_CHECK(cudaMalloc(&d_latent,     kLatentN * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_noise_pred, kLatentN * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_image,      kImgN    * sizeof(__half)));

    std::vector<float>  scaled_host(kLatentN);   // scale_model_input 済み (host)
    std::vector<__half> h_latent_f16(kLatentN);  // H2D 用 FP16 バッファ
    std::vector<float>  noise_host(kLatentN);    // D2H noise_pred (FP32)
    std::vector<__half> h_np_f16(kLatentN);      // D2H 受け FP16
    std::vector<float>  latent_next(kLatentN);   // step 出力

    // --- 拡散ループ ---
    for (int i = 0; i < steps; ++i)
    {
        // scale_model_input: scaled = latent / sqrt(sigma^2 + 1) (host, in-place)
        scaled_host = latent_host;
        sched.scale_model_input(scaled_host.data(), kLatentN, i);

        // H2D: FP16 変換して d_latent へ
        for (size_t k = 0; k < kLatentN; ++k)
        {
            h_latent_f16[k] = __float2half(scaled_host[k]);
        }
        CUDA_CHECK(cudaMemcpy(d_latent, h_latent_f16.data(),
                              kLatentN * sizeof(__half), cudaMemcpyHostToDevice));

        // UNet 1 step (埋め込みは全 step 使い回し)。
        launch_unet(unet_weights_,
                    d_latent,
                    timesteps[i],
                    d_encoder_hidden_states_,
                    d_text_embeds_,
                    d_time_ids_,
                    d_noise_pred);

        // D2H: noise_pred を FP32 へ
        CUDA_CHECK(cudaMemcpy(h_np_f16.data(), d_noise_pred,
                              kLatentN * sizeof(__half), cudaMemcpyDeviceToHost));
        for (size_t k = 0; k < kLatentN; ++k)
        {
            noise_host[k] = __half2float(h_np_f16[k]);
        }

        // Euler step: latent_next = step(noise, i, latent_host) (host)
        sched.step(noise_host.data(), i, latent_host.data(), latent_next.data(), kLatentN);
        latent_host.swap(latent_next);
    }

    // --- VAE decode 用に latent を scaling_factor で割り FP16 で H2D ---
    for (size_t k = 0; k < kLatentN; ++k)
    {
        h_latent_f16[k] = __float2half(latent_host[k] / kScalingFactor);
    }
    CUDA_CHECK(cudaMemcpy(d_latent, h_latent_f16.data(),
                          kLatentN * sizeof(__half), cudaMemcpyHostToDevice));

    launch_vae_decode(vae_weights_, d_latent, d_image);

    // --- D2H: image (FP16, [-1,1] 値域) ---
    std::vector<__half> h_image(kImgN);
    CUDA_CHECK(cudaMemcpy(h_image.data(), d_image,
                          kImgN * sizeof(__half), cudaMemcpyDeviceToHost));

    // --- 画像化: VAE 出力は [-1,1] 系。dump_vae_golden.py と同じく
    //     (x*0.5+0.5) で [0,1] へ写し、clamp 後 ×255 で uint8。
    //     出力は NCHW [3,1024,1024] → HWC [1024,1024,3] row-major へ並べ替え。
    w = kImgW;
    h = kImgH;
    rgb_out.assign(kImgN, 0);
    for (int c = 0; c < kImgC; ++c)
    {
        const size_t cbase = static_cast<size_t>(c) * kImgH * kImgW;
        for (int y = 0; y < kImgH; ++y)
        {
            for (int x = 0; x < kImgW; ++x)
            {
                const float v   = __half2float(h_image[cbase + static_cast<size_t>(y) * kImgW + x]);
                float       n01 = v * 0.5f + 0.5f;
                if (n01 < 0.0f) { n01 = 0.0f; }
                if (n01 > 1.0f) { n01 = 1.0f; }
                const int   q   = static_cast<int>(n01 * 255.0f + 0.5f);
                const size_t dst = (static_cast<size_t>(y) * kImgW + x) * 3 + c;
                rgb_out[dst] = static_cast<uint8_t>(q < 0 ? 0 : (q > 255 ? 255 : q));
            }
        }
    }

    CUDA_CHECK(cudaFree(d_latent));
    CUDA_CHECK(cudaFree(d_noise_pred));
    CUDA_CHECK(cudaFree(d_image));
}

} // namespace dollama

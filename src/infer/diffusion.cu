// SDXL 拡散ループ結線モジュール — 実装 (Phase 2 マイルストーン 2-6a / ST-1+ST-2)
// 対象: RTX5080 (Blackwell / sm_120) / CUDA Runtime API のみ
//
// UNet × Nstep + EulerDiscreteScheduler + VAE decode を結線し、golden 埋め込みを
// 入力に noise → latent → 画像 (1024×1024 RGB uint8) を生成する。
// scheduler は host float、UNet/VAE は FP16 デバイス。中間 latent の D2H/H2D は
// 1 step あたり ~256KB で無視可能なので素直に往復する。
//
// 2-6b Stage E: generate_txt2img で CFG (classifier-free guidance) を追加。
//   各 step で UNet を cond / uncond の 2 回回し host で合成する。

#include "infer/diffusion.cuh"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "infer/unet.cuh"
#include "infer/profile.cuh"
#include "infer/scheduler.hpp"
#include "kernels/vae_decode.cuh"
#include "kernels/device_arena.cuh"
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

// ----------------------------------------------------------------
// VAE 出力 (FP16, [-1,1] 値域, NCHW [3,1024,1024]) を HWC uint8 RGB へ変換する。
// (x*0.5+0.5) で [0,1] に写し clamp 後 ×255。generate / generate_txt2img 共通。
// ----------------------------------------------------------------
void image_f16_to_rgb_u8(const std::vector<__half>& h_image, std::vector<uint8_t>& rgb_out)
{
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
}

} // namespace

// ----------------------------------------------------------------
// コンストラクタ: 重み 2 つと golden 埋め込みをロードし、埋め込みをデバイス常駐させる。
// ----------------------------------------------------------------
DiffusionPipeline::DiffusionPipeline(const std::string& unet_weights_path,
                                     const std::string& vae_weights_path,
                                     const std::string& embeds_path,
                                     const FastConfig&  fast_cfg)
    : unet_weights_(unet_weights_path)
    , vae_weights_(vae_weights_path)
    , fast_cfg_(fast_cfg) // FAST フラグを保持 (G-3k で attention の分岐に使用)
{
    // G-3k フラグ結線: fast の下で attention 高速化 (attn_fast) を有効化する単一箇所。
    // fp8 は resolve_fast_config で fast を含意済み。fast=false (default) では何も立たない。
    if (fast_cfg_.fast)
    {
        fast_cfg_.attn_fast = true;
        // G-2k フラグ結線: fast の下で CFG batch=2 (batch2) も有効化する。
        // default (fast=false) 経路ではこの if に入らないため batch2 は false のまま。
        fast_cfg_.batch2 = true;
        // G-4k フラグ結線: fast の下で epilogue 融合 (epilogue) も有効化する。
        // default (fast=false) 経路ではこの if に入らないため epilogue は false のまま。
        fast_cfg_.epilogue = true;
    }
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

    // S1: UNet 全重み (5.1GB) を 1 度だけデバイスへ常駐させる。以降の全 step は
    //     このハンドルを使い回し、重み転送/再 malloc を発生させない。
    unet_weights_handle_ = unet_weights_create(unet_weights_);

    // S3-D: VAE decoder 全重み (~92MB) を 1 度だけデバイスへ常駐させる。
    //       以降の全生成は launch_vae_decode(handle, ...) で重み転送ゼロ。
    vae_weights_handle_ = vae_weights_create(vae_weights_);
}

DiffusionPipeline::~DiffusionPipeline()
{
    // デストラクタでは例外を投げない (CUDA_CHECK は使わず素の free)。
    if (d_encoder_hidden_states_ != nullptr) { cudaFree(d_encoder_hidden_states_); }
    if (d_text_embeds_ != nullptr)           { cudaFree(d_text_embeds_); }
    if (d_time_ids_ != nullptr)              { cudaFree(d_time_ids_); }
    if (unet_weights_handle_ != nullptr)     { unet_weights_destroy(unet_weights_handle_); }
    if (vae_weights_handle_ != nullptr)      { vae_weights_destroy(vae_weights_handle_); }
}

// ----------------------------------------------------------------
// ランタイム LoRA (L-2)。常駐 UNet 重みへの適用時マージ / bit-exact 復元。
// ----------------------------------------------------------------
void DiffusionPipeline::apply_lora_file(const std::string& path, float strength)
{
    // LoRA safetensors をロードし、base (= unet_weights_) へ写像する (lora.hpp)。
    SafeTensors    lora_st(path);
    LoraLoadReport rep;
    std::vector<LoraModule> mods = load_lora_modules(unet_weights_, lora_st, strength, &rep);
    if (rep.te_skipped > 0)
    {
        // te1/te2 (テキストエンコーダ) LoRA はスコープ外 skip
        std::cout << "[lora] skipped " << rep.te_skipped
                  << " te1/te2 keys (UNet merge only)\n";
    }
    if (rep.incomplete > 0 || rep.other_keys > 0)
    {
        // down/up 片欠けの不完全 module / 解釈不能 key の skip 件数
        std::cout << "[lora] skipped incomplete modules=" << rep.incomplete
                  << " / unknown keys=" << rep.other_keys << "\n";
    }
    unet_apply_loras(unet_weights_handle_, mods.data(), static_cast<int>(mods.size()));
    std::cout << "[lora] applied: " << path << " (strength=" << strength
              << ", modules=" << rep.modules << ")\n";
}

void DiffusionPipeline::clear_loras()
{
    unet_clear_loras(unet_weights_handle_);
}

// ----------------------------------------------------------------
// G-8k S3: 画像 1 枚を出し終えた境界でアリーナのチャンクを返すかどうか。
//   既定 OFF。env DOLLAMA_ARENA_RELEASE=1 のときだけ有効。
//   位置づけ: S4 の VRAM 主ゲートが落ちたときの救済策 (VRAM を返す代わりに次画像の
//   初回チャンク確保が復活する) を、同一走行で A/B 比較できるようにするためのスイッチ。
//   **これは device_arena.cuh が禁じている「定期 trim / step 間 trim」ではない**:
//   呼ばれるのは画像境界 (step ループの完全な外側) だけで、step ループ内の
//   cudaMalloc/cudaFree 0 という目的は一切損なわない。
//   env 読みの作法は fast_config.hpp の fast_env_true に倣い初回のみ getenv (以後キャッシュ)。
// ----------------------------------------------------------------
static bool arena_release_enabled()
{
    static int e = -1;
    if (e < 0)
    {
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
        const char* v = std::getenv("DOLLAMA_ARENA_RELEASE");
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
        e = (v != nullptr && std::strcmp(v, "1") == 0) ? 1 : 0;
    }
    return e == 1;
}

// 画像境界でのアリーナ解放 (既定 OFF)。
static void maybe_release_arenas()
{
    if (!arena_release_enabled())
    {
        return;
    }
    device_arena_release(DeviceArenaId::UNet);
    device_arena_release(DeviceArenaId::UNetPersist);
}

// ----------------------------------------------------------------
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
// 拡散ループ本体 (CFG なし・golden 埋め込み使い回し)。
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
    // CFG > 1 は generate では非対応 (CFG は generate_txt2img を使う)。
    if (std::fabs(guidance_scale - 1.0f) > 1e-6f)
    {
        throw std::runtime_error(
            "DiffusionPipeline::generate: guidance_scale != 1.0 is not supported "
            "(use generate_txt2img for CFG)");
    }

    // --- S0 プロファイル: 総時間計測開始 + カウンタリセット (DOLLAMA_PROFILE 時のみ) ---
    const bool prof = profile_enabled();
    if (prof) { profile_counters().reset(); }
    std::chrono::high_resolution_clock::time_point prof_t_total;
    if (prof)
    {
        cudaDeviceSynchronize();
        prof_t_total = std::chrono::high_resolution_clock::now();
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
        std::chrono::high_resolution_clock::time_point prof_h0;
        if (prof) { prof_h0 = std::chrono::high_resolution_clock::now(); }
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

        if (prof)
        {
            const auto h1 = std::chrono::high_resolution_clock::now();
            profile_counters().host_roundtrip_sec += std::chrono::duration<double>(h1 - prof_h0).count();
        }
        // UNet 1 step (埋め込みは全 step 使い回し)。
        launch_unet(unet_weights_handle_,
                    d_latent,
                    timesteps[i],
                    d_encoder_hidden_states_,
                    d_text_embeds_,
                    d_time_ids_,
                    d_noise_pred,
                    fast_cfg_.attn_fast,
                    fast_cfg_.epilogue);

        std::chrono::high_resolution_clock::time_point prof_h2;
        if (prof) { prof_h2 = std::chrono::high_resolution_clock::now(); }
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
        if (prof)
        {
            const auto h3 = std::chrono::high_resolution_clock::now();
            profile_counters().host_roundtrip_sec += std::chrono::duration<double>(h3 - prof_h2).count();
        }
    }

    // --- VAE decode 用に latent を scaling_factor で割り FP16 で H2D ---
    for (size_t k = 0; k < kLatentN; ++k)
    {
        h_latent_f16[k] = __float2half(latent_host[k] / kScalingFactor);
    }
    CUDA_CHECK(cudaMemcpy(d_latent, h_latent_f16.data(),
                          kLatentN * sizeof(__half), cudaMemcpyHostToDevice));

    {
        ScopedSyncTimer vt(prof ? &profile_counters().vae_sec : nullptr, prof);
        launch_vae_decode(vae_weights_handle_, d_latent, d_image);
        vt.stop();
    }

    // --- D2H: image (FP16, [-1,1] 値域) ---
    std::vector<__half> h_image(kImgN);
    CUDA_CHECK(cudaMemcpy(h_image.data(), d_image,
                          kImgN * sizeof(__half), cudaMemcpyDeviceToHost));

    // --- 画像化: VAE 出力は [-1,1] 系。(x*0.5+0.5)→clamp→×255、NCHW→HWC。
    w = kImgW;
    h = kImgH;
    image_f16_to_rgb_u8(h_image, rgb_out);

    // --- S0 プロファイル: 総時間確定 + 内訳テーブル出力 (DOLLAMA_PROFILE 時のみ) ---
    if (prof)
    {
        cudaDeviceSynchronize();
        const auto prof_end = std::chrono::high_resolution_clock::now();
        ProfileCounters& pc = profile_counters();
        pc.total_sec = std::chrono::duration<double>(prof_end - prof_t_total).count();

        const double tot   = pc.total_sec > 0.0 ? pc.total_sec : 1e-9;
        const double upl   = pc.weight_upload_sec;
        const double unet  = pc.unet_total_sec;
        const double pure  = unet - upl;  // UNet 純カーネル = step 全体 - 重み転送
        const double gbyte = (double)pc.weight_upload_bytes / (1024.0 * 1024.0 * 1024.0);
        auto pct = [&](double s) { return 100.0 * s / tot; };

        std::printf("\n");
        std::printf("==================== DOLLAMA_PROFILE (steps=%d) ====================\n",
                    pc.unet_steps);
        std::printf("  %-34s %9.3f s  %6.2f%%\n", "weight upload+malloc (UNet, total)", upl, pct(upl));
        std::printf("      uploads=%llu  bytes=%.3f GB  (avg %.2f ms/upload)\n",
                    (unsigned long long)pc.weight_upload_count, gbyte,
                    pc.weight_upload_count ? 1000.0 * upl / (double)pc.weight_upload_count : 0.0);
        std::printf("  %-34s %9.3f s  %6.2f%%\n", "UNet step total (all steps)", unet, pct(unet));
        std::printf("  %-34s %9.3f s  %6.2f%%\n", "  -> UNet pure kernels (total-upl)", pure, pct(pure));
        std::printf("      [group, wall incl. upload]\n");
        std::printf("      %-30s %9.3f s  %6.2f%%\n", "embed",    pc.unet_embed_sec,   pct(pc.unet_embed_sec));
        std::printf("      %-30s %9.3f s  %6.2f%%\n", "down",     pc.unet_down_sec,    pct(pc.unet_down_sec));
        std::printf("      %-30s %9.3f s  %6.2f%%\n", "mid",      pc.unet_mid_sec,     pct(pc.unet_mid_sec));
        std::printf("      %-30s %9.3f s  %6.2f%%\n", "up",       pc.unet_up_sec,      pct(pc.unet_up_sec));
        std::printf("      %-30s %9.3f s  %6.2f%%\n", "conv_out", pc.unet_convout_sec, pct(pc.unet_convout_sec));
        std::printf("      [kernel category, orthogonal to groups]\n");
        std::printf("      %-30s %9.3f s  %6.2f%%\n", "resnet (conv/groupnorm)",
                    pc.cat_resnet_sec, pct(pc.cat_resnet_sec));
        std::printf("      %-30s %9.3f s  %6.2f%%\n", "transformer (attn/gemm)",
                    pc.cat_transformer_sec, pct(pc.cat_transformer_sec));
        std::printf("      %-30s %9.3f s  %6.2f%%  (subset of transformer)\n",
                    "  -> attention only", pc.cat_attention_sec, pct(pc.cat_attention_sec));
        std::printf("  %-34s %9.3f s  %6.2f%%\n", "VAE decode", pc.vae_sec, pct(pc.vae_sec));
        std::printf("  %-34s %9.3f s  %6.2f%%\n", "host roundtrip (scale/H2D/D2H/sched)",
                    pc.host_roundtrip_sec, pct(pc.host_roundtrip_sec));
        std::printf("  %-34s %9.3f s  %6.2f%%\n", "TOTAL", pc.total_sec, 100.0);
        std::printf("  (per-step UNet: %.3f s  | per-step pure: %.3f s)\n",
                    pc.unet_steps ? unet / pc.unet_steps : 0.0,
                    pc.unet_steps ? pure / pc.unet_steps : 0.0);
        std::printf("====================================================================\n\n");
        std::fflush(stdout);
    }

    CUDA_CHECK(cudaFree(d_latent));
    CUDA_CHECK(cudaFree(d_noise_pred));
    CUDA_CHECK(cudaFree(d_image));

    // G-8k S3: 画像境界での明示解放 (既定 OFF / DOLLAMA_ARENA_RELEASE=1 のみ)。
    maybe_release_arenas();
}

// ----------------------------------------------------------------
// 2-6b Stage E: CFG (classifier-free guidance) 付き txt2img 拡散ループ。
//   各 step で UNet を cond / uncond の 2 回回し、
//     noise = uncond + guidance_scale * (cond - uncond)
//   を host で合成して Euler step に渡す。
// ----------------------------------------------------------------
void DiffusionPipeline::generate_txt2img(int                   steps,
                                         uint64_t              seed,
                                         float                 guidance_scale,
                                         const float*          cond_ehs,
                                         const float*          cond_text_embeds,
                                         const float*          uncond_ehs,
                                         const float*          uncond_text_embeds,
                                         const float*          time_ids,
                                         std::vector<uint8_t>& rgb_out,
                                         int&                  w,
                                         int&                  h)
{
    if (steps <= 0)
    {
        throw std::runtime_error("DiffusionPipeline::generate_txt2img: steps must be > 0");
    }
    if (cond_ehs == nullptr || cond_text_embeds == nullptr
        || uncond_ehs == nullptr || uncond_text_embeds == nullptr || time_ids == nullptr)
    {
        throw std::runtime_error("DiffusionPipeline::generate_txt2img: null embedding pointer");
    }

    // G-2k S3b: batch2 経路スイッチ。fast/DOLLAMA_BATCH2 で立つ (default=false)。
    const bool use_batch2 = fast_cfg_.batch2;

    // --- 外部 cond / uncond 埋め込みを FP16 へ変換してデバイス常駐させる ---
    //     コンストラクタ常駐の golden 埋め込みは使わない (本 txt2img 経路)。
    __half* d_cond_ehs    = nullptr;  // [77*2048]
    __half* d_cond_txt    = nullptr;  // [1280]
    __half* d_uncond_ehs  = nullptr;  // [77*2048]
    __half* d_uncond_txt  = nullptr;  // [1280]
    __half* d_time_ids    = nullptr;  // [6] (cond/uncond 共通)
    CUDA_CHECK(cudaMalloc(&d_cond_ehs,   kEhsN  * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_cond_txt,   kTxtN  * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_uncond_ehs, kEhsN  * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_uncond_txt, kTxtN  * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_time_ids,   kTidsN * sizeof(__half)));

    {
        std::vector<__half> tmp_ce(kEhsN), tmp_ue(kEhsN);
        std::vector<__half> tmp_ct(kTxtN), tmp_ut(kTxtN);
        std::vector<__half> tmp_ti(kTidsN);
        for (size_t k = 0; k < kEhsN; ++k)
        {
            tmp_ce[k] = __float2half(cond_ehs[k]);
            tmp_ue[k] = __float2half(uncond_ehs[k]);
        }
        for (size_t k = 0; k < kTxtN; ++k)
        {
            tmp_ct[k] = __float2half(cond_text_embeds[k]);
            tmp_ut[k] = __float2half(uncond_text_embeds[k]);
        }
        for (size_t k = 0; k < kTidsN; ++k)
        {
            tmp_ti[k] = __float2half(time_ids[k]);
        }
        CUDA_CHECK(cudaMemcpy(d_cond_ehs,   tmp_ce.data(), kEhsN  * sizeof(__half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_uncond_ehs, tmp_ue.data(), kEhsN  * sizeof(__half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cond_txt,   tmp_ct.data(), kTxtN  * sizeof(__half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_uncond_txt, tmp_ut.data(), kTxtN  * sizeof(__half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_time_ids,   tmp_ti.data(), kTidsN * sizeof(__half), cudaMemcpyHostToDevice));
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
    __half* d_latent      = nullptr;  // UNet 入力 (scale_model_input 済み)
    __half* d_noise_cond  = nullptr;  // UNet 出力 (cond)
    __half* d_noise_unc   = nullptr;  // UNet 出力 (uncond)
    __half* d_image       = nullptr;  // VAE 出力
    CUDA_CHECK(cudaMalloc(&d_latent,     kLatentN * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_noise_cond, kLatentN * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_noise_unc,  kLatentN * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_image,      kImgN    * sizeof(__half)));

    // --- G-2k S3b: batch2 用の連続バッファ ([2,...]) を確保し埋め込みを束ねる ---
    //     slice0=cond / slice1=uncond の順 (S2 テストと厳密一致)。既存の cond/uncond
    //     デバイスバッファから DtoD で写すため、上の H2D コードは無改変で共用できる。
    __half* d_ehs2    = nullptr;  // [2,77,2048]
    __half* d_txt2    = nullptr;  // [2,1280]
    __half* d_latent2 = nullptr;  // [2,4,128,128]
    __half* d_noise2  = nullptr;  // [2,4,128,128] UNet 出力
    if (use_batch2)
    {
        CUDA_CHECK(cudaMalloc(&d_ehs2,    2 * kEhsN    * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_txt2,    2 * kTxtN    * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_latent2, 2 * kLatentN * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_noise2,  2 * kLatentN * sizeof(__half)));
        // b=0=cond / b=1=uncond の順で束ねる。
        CUDA_CHECK(cudaMemcpy(d_ehs2,         d_cond_ehs,   kEhsN * sizeof(__half), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_ehs2 + kEhsN, d_uncond_ehs, kEhsN * sizeof(__half), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_txt2,         d_cond_txt,   kTxtN * sizeof(__half), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_txt2 + kTxtN, d_uncond_txt, kTxtN * sizeof(__half), cudaMemcpyDeviceToDevice));
    }

    std::vector<float>  scaled_host(kLatentN);
    std::vector<__half> h_latent_f16(kLatentN);
    std::vector<float>  noise_cond(kLatentN);    // D2H cond (FP32)
    std::vector<float>  noise_unc(kLatentN);     // D2H uncond (FP32)
    std::vector<__half> h_nc_f16(kLatentN);
    std::vector<__half> h_nu_f16(kLatentN);
    std::vector<float>  noise_cfg(kLatentN);     // CFG 合成後の noise
    std::vector<float>  latent_next(kLatentN);

    // --- CFG 拡散ループ ---
    for (int i = 0; i < steps; ++i)
    {
        // scale_model_input (host, in-place) → cond/uncond 共通の d_latent へ H2D。
        scaled_host = latent_host;
        sched.scale_model_input(scaled_host.data(), kLatentN, i);
        for (size_t k = 0; k < kLatentN; ++k)
        {
            h_latent_f16[k] = __float2half(scaled_host[k]);
        }
        CUDA_CHECK(cudaMemcpy(d_latent, h_latent_f16.data(),
                              kLatentN * sizeof(__half), cudaMemcpyHostToDevice));

        if (use_batch2)
        {
            // G-2k S3b: cond/uncond を B=2 に束ね 1 forward で回す。
            // 共通の d_latent を [2,...] の slice0/slice1 に複製する。
            CUDA_CHECK(cudaMemcpy(d_latent2,            d_latent,
                                  kLatentN * sizeof(__half), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_latent2 + kLatentN, d_latent,
                                  kLatentN * sizeof(__half), cudaMemcpyDeviceToDevice));
            launch_unet_batched(unet_weights_handle_,
                                2,
                                d_latent2,
                                timesteps[i],
                                d_ehs2,
                                d_txt2,
                                d_time_ids,
                                d_noise2,
                                fast_cfg_.attn_fast,
                                fast_cfg_.epilogue);
            // 出力スライスを既存 cond/uncond バッファへ写し以降のロジックを共用。
            CUDA_CHECK(cudaMemcpy(d_noise_cond, d_noise2,
                                  kLatentN * sizeof(__half), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_noise_unc,  d_noise2 + kLatentN,
                                  kLatentN * sizeof(__half), cudaMemcpyDeviceToDevice));
        }
        else
        {
            // UNet を cond 埋め込みで 1 回。
            launch_unet(unet_weights_handle_,
                        d_latent,
                        timesteps[i],
                        d_cond_ehs,
                        d_cond_txt,
                        d_time_ids,
                        d_noise_cond,
                        fast_cfg_.attn_fast,
                        fast_cfg_.epilogue);

            // UNet を uncond 埋め込みで 1 回 (同じ d_latent)。
            launch_unet(unet_weights_handle_,
                        d_latent,
                        timesteps[i],
                        d_uncond_ehs,
                        d_uncond_txt,
                        d_time_ids,
                        d_noise_unc,
                        fast_cfg_.attn_fast,
                        fast_cfg_.epilogue);
        }

        // D2H: cond / uncond noise_pred を FP32 へ。
        CUDA_CHECK(cudaMemcpy(h_nc_f16.data(), d_noise_cond,
                              kLatentN * sizeof(__half), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_nu_f16.data(), d_noise_unc,
                              kLatentN * sizeof(__half), cudaMemcpyDeviceToHost));
        for (size_t k = 0; k < kLatentN; ++k)
        {
            noise_cond[k] = __half2float(h_nc_f16[k]);
            noise_unc[k]  = __half2float(h_nu_f16[k]);
        }

        // CFG 合成 (host): noise = uncond + scale * (cond - uncond)。
        //   noise_pred は 65536 要素と小さいので host 合成で十分。
        for (size_t k = 0; k < kLatentN; ++k)
        {
            noise_cfg[k] = noise_unc[k] + guidance_scale * (noise_cond[k] - noise_unc[k]);
        }

        // Euler step: latent_next = step(noise_cfg, i, latent_host) (host)。
        sched.step(noise_cfg.data(), i, latent_host.data(), latent_next.data(), kLatentN);
        latent_host.swap(latent_next);
    }

    // --- VAE decode 用に latent を scaling_factor で割り FP16 で H2D ---
    for (size_t k = 0; k < kLatentN; ++k)
    {
        h_latent_f16[k] = __float2half(latent_host[k] / kScalingFactor);
    }
    CUDA_CHECK(cudaMemcpy(d_latent, h_latent_f16.data(),
                          kLatentN * sizeof(__half), cudaMemcpyHostToDevice));

    launch_vae_decode(vae_weights_handle_, d_latent, d_image);

    // --- D2H: image (FP16, [-1,1] 値域) ---
    std::vector<__half> h_image(kImgN);
    CUDA_CHECK(cudaMemcpy(h_image.data(), d_image,
                          kImgN * sizeof(__half), cudaMemcpyDeviceToHost));

    // --- 画像化 ---
    w = kImgW;
    h = kImgH;
    image_f16_to_rgb_u8(h_image, rgb_out);

    // --- 後始末 ---
    CUDA_CHECK(cudaFree(d_latent));
    CUDA_CHECK(cudaFree(d_noise_cond));
    CUDA_CHECK(cudaFree(d_noise_unc));
    CUDA_CHECK(cudaFree(d_image));
    CUDA_CHECK(cudaFree(d_cond_ehs));
    CUDA_CHECK(cudaFree(d_cond_txt));
    CUDA_CHECK(cudaFree(d_uncond_ehs));
    CUDA_CHECK(cudaFree(d_uncond_txt));
    CUDA_CHECK(cudaFree(d_time_ids));

    // G-2k S3b: batch2 で確保した連続バッファのみ解放 (スライスは二重 free しない)。
    if (use_batch2)
    {
        CUDA_CHECK(cudaFree(d_ehs2));
        CUDA_CHECK(cudaFree(d_txt2));
        CUDA_CHECK(cudaFree(d_latent2));
        CUDA_CHECK(cudaFree(d_noise2));
    }

    // G-8k S3: 画像境界での明示解放 (既定 OFF / DOLLAMA_ARENA_RELEASE=1 のみ)。
    // 出荷経路 (txt2img) も同一走行で A/B できるよう generate と同じ位置に置く。
    maybe_release_arenas();
}

} // namespace dollama

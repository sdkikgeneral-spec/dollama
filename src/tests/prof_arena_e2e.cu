// G-8k S4 — e2e アリーナ A/B 計測ハーネス (計測専用・meson test 未登録)
//
// 目的:
//   同一プロセスで 1024x1024 / 20step の CFG 生成を N 枚 (既定 3) 連続実行し、
//     (a) プロセス GPU peak (cudaMemGetInfo の used = total - free の最大値)
//     (b) 画像ごとの device_arena 統計デルタ (cudaMalloc / cudaFree / chunk_alloc)
//     (c) e2e 秒 (characterization)
//   を出す。走行間で変えるのは DOLLAMA_POOL / DOLLAMA_ARENA_RELEASE のみ。
//
// 被験コード (diffusion.cu / unet.cu / vae_decode.cu / device_arena.cu) は
// 一切改変しない。本ファイルは観測側だけを足す。
//
// 環境変数:
//   PROF_IMAGES  生成枚数 (既定 3)
//   PROF_STEPS   step 数 (既定 20)
//   PROF_G       guidance (既定 7.5)
//   PROF_FAST    1 なら出荷 --fast 相当 (attn_fast+batch2+epilogue・既定 1)
//   PROF_SAMPLE_MS サンプラ周期 ms (既定 5)
//   DOLLAMA_PROFILE=1 で破棄時の [ALLOC] 行 (unet.cu) も出る。

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef HAVE_CUDA
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "infer/diffusion.cuh"
#include "kernels/device_arena.cuh"
#include "kernels/utils.cuh"
#endif

namespace dollama
{

#ifdef HAVE_CUDA

static int env_int(const char* k, int dflt)
{
    if (const char* e = std::getenv(k))
    {
        return std::atoi(e);
    }
    return dflt;
}

static double env_dbl(const char* k, double dflt)
{
    if (const char* e = std::getenv(k))
    {
        return std::atof(e);
    }
    return dflt;
}

static std::vector<float> load_f16_as_f32(const SafeTensors& st, const std::string& name,
                                          size_t expect)
{
    size_t nbytes = 0;
    const uint8_t* p = st.tensor_bytes(name, nbytes);
    const size_t   n = nbytes / sizeof(__half);
    if (n != expect)
    {
        throw std::runtime_error("size mismatch: " + name);
    }
    std::vector<__half> tmp(n);
    std::memcpy(tmp.data(), p, nbytes);
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i)
    {
        out[i] = __half2float(tmp[i]);
    }
    return out;
}

// 出力の同一性確認用 64bit FNV-1a。
static uint64_t fnv1a(const std::vector<uint8_t>& v)
{
    uint64_t h = 1469598103934665603ULL;
    for (uint8_t b : v)
    {
        h = (h ^ b) * 1099511628211ULL;
    }
    return h;
}

// ---- プロセス GPU peak サンプラ (cudaMemGetInfo の used 最大) ----
static std::atomic<bool>     g_sampler_run{false};
static std::atomic<uint64_t> g_peak_used{0};
static std::atomic<uint64_t> g_total_mem{0};

static uint64_t used_now()
{
    size_t freeb = 0, totalb = 0;
    if (cudaMemGetInfo(&freeb, &totalb) != cudaSuccess)
    {
        return 0;
    }
    g_total_mem.store(totalb);
    return static_cast<uint64_t>(totalb - freeb);
}

static void bump_peak()
{
    const uint64_t u = used_now();
    uint64_t prev = g_peak_used.load();
    while (u > prev && !g_peak_used.compare_exchange_weak(prev, u))
    {
        // compare_exchange_weak が prev を実測値で更新するので body は空でよい。
    }
}

static void sampler_main(int period_ms)
{
    while (g_sampler_run.load())
    {
        bump_peak();
        std::this_thread::sleep_for(std::chrono::milliseconds(period_ms));
    }
    bump_peak();
}

static void print_stats(const char* tag, DeviceArenaId id,
                        const DeviceArenaStats& a, const DeviceArenaStats& b)
{
    std::cout << "[S4] " << tag << " arena=" << device_arena_name(id)
              << " d_cudaMalloc=" << (b.cuda_malloc_calls - a.cuda_malloc_calls)
              << " d_cudaFree="   << (b.cuda_free_calls   - a.cuda_free_calls)
              << " d_chunkAlloc=" << (b.chunk_alloc_calls - a.chunk_alloc_calls)
              << " d_alloc="      << (b.alloc_calls       - a.alloc_calls)
              << " | cap=" << (b.total_capacity >> 20) << "MiB"
              << " chunks=" << b.live_chunks
              << " live_peak=" << (b.peak_request_bytes >> 20) << "MiB"
              << " cursor_peak=" << (b.peak_bytes_in_use >> 20) << "MiB"
              << " cum_cudaMalloc=" << b.cuda_malloc_calls
              << " cum_cudaFree="   << b.cuda_free_calls
              << " cum_chunkAlloc=" << b.chunk_alloc_calls
              << "\n";
}

static int run_prof()
{
    std::cout << std::unitbuf;
    const std::string unet_w = UNET_WEIGHTS_PATH;
    const std::string vae_w  = VAE_WEIGHTS_PATH;
    const std::string iopath = UNET_IO_PATH;
    {
        std::ifstream fu(unet_w, std::ios::binary);
        std::ifstream fv(vae_w, std::ios::binary);
        std::ifstream fe(iopath, std::ios::binary);
        if (!fu.good() || !fv.good() || !fe.good())
        {
            std::cout << "[S4] [SKIP] weights not found\n";
            return 0;
        }
    }

    const int    images    = env_int("PROF_IMAGES", 3);
    const int    steps     = env_int("PROF_STEPS", 20);
    const float  g         = static_cast<float>(env_dbl("PROF_G", 7.5));
    const bool   fast      = env_int("PROF_FAST", 1) != 0;
    const int    sample_ms = env_int("PROF_SAMPLE_MS", 5);
    const char*  pool      = std::getenv("DOLLAMA_POOL");
    const char*  rel       = std::getenv("DOLLAMA_ARENA_RELEASE");

    std::cout << "[S4] config images=" << images << " steps=" << steps << " g=" << g
              << " fast=" << (fast ? 1 : 0)
              << " DOLLAMA_POOL=" << (pool ? pool : "(unset)")
              << " DOLLAMA_ARENA_RELEASE=" << (rel ? rel : "(unset)")
              << " pool_enabled=" << (device_arena_pool_enabled() ? 1 : 0)
              << " sample_ms=" << sample_ms << "\n";

    // CUDA コンテキスト初期化 → 以降 used のベースラインを取る。
    CUDA_CHECK(cudaFree(0));
    const uint64_t used_ctx = used_now();
    std::cout << "[S4] used_after_ctx=" << (used_ctx >> 20) << "MB"
              << " total=" << (g_total_mem.load() >> 20) << "MB\n";

    g_sampler_run.store(true);
    std::thread sampler(sampler_main, sample_ms);

    constexpr size_t kEhsN  = static_cast<size_t>(77) * 2048;
    constexpr size_t kTxtN  = 1280;
    constexpr size_t kTidsN = 6;
    std::vector<float> cond_ehs, cond_txt, time_ids;
    {
        SafeTensors io(iopath);
        cond_ehs = load_f16_as_f32(io, "input_encoder_hidden_states_f16", kEhsN);
        cond_txt = load_f16_as_f32(io, "input_text_embeds_f16",           kTxtN);
        time_ids = load_f16_as_f32(io, "input_time_ids_f16",              kTidsN);
    }
    std::vector<float> uncond_ehs(cond_ehs);
    std::vector<float> uncond_txt(cond_txt);
    for (float& v : uncond_ehs)
    {
        v *= 0.5f;
    }
    for (float& v : uncond_txt)
    {
        v *= 0.5f;
    }

    {
        FastConfig cfg;
        cfg.attn_fast = fast;
        cfg.batch2    = fast;
        cfg.epilogue  = fast;
        DiffusionPipeline pipe(unet_w, vae_w, iopath, cfg);

        const uint64_t used_load = used_now();
        bump_peak();
        std::cout << "[S4] used_after_weight_load=" << (used_load >> 20) << "MB"
                  << " free=" << ((g_total_mem.load() - used_load) >> 20) << "MB\n";

        for (int i = 1; i <= images; ++i)
        {
            const DeviceArenaStats a0 = device_arena_stats(DeviceArenaId::UNet);
            const DeviceArenaStats p0 = device_arena_stats(DeviceArenaId::UNetPersist);
            const uint64_t peak_before = g_peak_used.load();

            std::vector<uint8_t> rgb;
            int w = 0, h = 0;
            CUDA_CHECK(cudaDeviceSynchronize());
            const auto t0 = std::chrono::steady_clock::now();
            pipe.generate_txt2img(steps, 1234ULL, g,
                                  cond_ehs.data(), cond_txt.data(),
                                  uncond_ehs.data(), uncond_txt.data(),
                                  time_ids.data(), rgb, w, h);
            CUDA_CHECK(cudaDeviceSynchronize());
            const auto t1 = std::chrono::steady_clock::now();
            const double sec = std::chrono::duration<double>(t1 - t0).count();

            const DeviceArenaStats a1 = device_arena_stats(DeviceArenaId::UNet);
            const DeviceArenaStats p1 = device_arena_stats(DeviceArenaId::UNetPersist);
            const uint64_t used_after = used_now();
            bump_peak();

            std::cout << "[S4] image=" << i << " sec=" << sec
                      << " rgb_hash=0x" << std::hex << fnv1a(rgb) << std::dec
                      << " " << w << "x" << h << "\n";
            print_stats("image", DeviceArenaId::UNet,        a0, a1);
            print_stats("image", DeviceArenaId::UNetPersist, p0, p1);
            std::cout << "[S4] image=" << i
                      << " peak_used_total=" << (g_peak_used.load() >> 20) << "MB"
                      << " peak_used_delta_this_image="
                      << ((g_peak_used.load() > peak_before)
                              ? ((g_peak_used.load() - peak_before) >> 20) : 0) << "MB"
                      << " used_after_image=" << (used_after >> 20) << "MB\n";
        }

        std::cout << "[S4] PEAK_USED=" << (g_peak_used.load() >> 20) << "MB"
                  << " (total=" << (g_total_mem.load() >> 20) << "MB)\n";
    } // ~DiffusionPipeline → unet_weights_destroy → [ALLOC] 行 (DOLLAMA_PROFILE=1)

    g_sampler_run.store(false);
    sampler.join();

    const uint64_t used_end = used_now();
    std::cout << "[S4] used_after_destroy=" << (used_end >> 20) << "MB\n";
    std::cout << "[S4] FINAL_PEAK_USED=" << (g_peak_used.load() >> 20) << "MB\n";
    return 0;
}

#endif // HAVE_CUDA

} // namespace dollama

int main()
{
#ifndef HAVE_CUDA
    std::cout << "[S4] [SKIP] HAVE_CUDA undefined\n";
    return 0;
#else
    try
    {
        return dollama::run_prof();
    }
    catch (const std::exception& e)
    {
        std::cerr << "[S4] exception: " << e.what() << "\n";
        return 1;
    }
#endif
}

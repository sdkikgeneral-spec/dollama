// SDXL UNet FAST モード (G-3k-wire) e2e ゲート。
//   attn_fast フラグを launch_unet 経由で attention の call site まで貫通させた結線を検証する。
//
//   (1) default (attn_fast=false): 既存 golden (noise_pred_f32) と一致 = 無改変を確認。
//       → launch_attention (従来経路) をそのまま通す。
//   (2) fast    (attn_fast=true) : default 出力と突合し SSIM >= 0.9999。
//       launch_attention_fast は数学同一 (SSIM=1.0 実測) ゆえ実質ビット一致になる想定。
//   (2b) epilogue (G-4k S1a)     : attn_fast+epilogue=on を default と突合し SSIM >= 0.9999
//       (multi-block GroupNorm は蓄積順が変わるため FP16 で微差・ビット一致は狙わない)。
//       default (epilogue=off) は (1) がそのまま無改変ゲート (既定引数 false)。
//   (3) 速度   : default / fast / fast+epilogue の UNet 1step レイテンシを cudaEvent で計測しログ。
//
// G-3kf (warm 化): 従来は後方互換 launch_unet(const SafeTensors&, ...) を使い、毎 call で
//   4.9GB を H2D 再アップロード (cold) していた。これは attention 比率を 40%→20% に希釈し、
//   速度ゲートが実態 (UNet forward 1.19x / attention 1.43x) より低く出る原因だった。
//   本テストは unet_weights_create で全重みを 1 度だけデバイス常駐させ、以降 launch_unet(handle, ...)
//   の warm 経路で計測・突合する (重み再転送ゼロ)。正しさ検証 (default vs golden / fast vs default)
//   は warm 経路でも数値同一のまま維持する。ハンドルは weights の生存期間内で使い、destroy する。
//
// HAVE_CUDA 未定義時 / golden 不在時は [SKIP] で return 0。
// ゴールデンパス UNET_WEIGHTS_PATH / UNET_IO_PATH は meson cuda_args で -D 埋め込み。

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#ifdef HAVE_CUDA
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "infer/unet.cuh"
#include "kernels/device_arena.cuh"
#include "kernels/utils.cuh"
#include "io/safetensors.hpp"
#endif

namespace dollama
{

#ifdef HAVE_CUDA

static std::vector<float> load_f32(const SafeTensors& st, const std::string& name)
{
    if (st.dtype(name) != StDtype::F32)
    {
        throw std::runtime_error("load_f32: '" + name + "' is not F32");
    }
    size_t nbytes = 0;
    const uint8_t* p = st.tensor_bytes(name, nbytes);
    const size_t   n = nbytes / sizeof(float);
    std::vector<float> out(n);
    std::memcpy(out.data(), p, nbytes);
    return out;
}

static std::vector<__half> load_f16(const SafeTensors& st, const std::string& name)
{
    if (st.dtype(name) != StDtype::F16)
    {
        throw std::runtime_error("load_f16: '" + name + "' is not F16");
    }
    size_t nbytes = 0;
    const uint8_t* p = st.tensor_bytes(name, nbytes);
    const size_t   n = nbytes / sizeof(__half);
    std::vector<__half> out(n);
    std::memcpy(out.data(), p, nbytes);
    return out;
}

// 簡易 SSIM (一様窓, NCHW)。test_unet と同方針・同係数。
static double ssim_uniform(const std::vector<float>& a, const std::vector<float>& b,
                           int C, int H, int W)
{
    float lo = a[0], hi = a[0];
    for (float v : a) { lo = std::min(lo, v); hi = std::max(hi, v); }
    for (float v : b) { lo = std::min(lo, v); hi = std::max(hi, v); }
    const double L  = static_cast<double>(hi - lo);
    const double C1 = (0.01 * L) * (0.01 * L);
    const double C2 = (0.03 * L) * (0.03 * L);
    const int    win = 7, half = win >> 1;
    const double inv = 1.0 / static_cast<double>(win * win);
    double ssim_sum = 0.0;
    long   cnt = 0;
    for (int c = 0; c < C; ++c)
    {
        const long base = static_cast<long>(c) * H * W;
        for (int y = half; y < H - half; y += half)
        {
            for (int x = half; x < W - half; x += half)
            {
                double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
                for (int dy = -half; dy <= half; ++dy)
                {
                    for (int dx = -half; dx <= half; ++dx)
                    {
                        const long   idx = base + static_cast<long>(y + dy) * W + (x + dx);
                        const double va = a[idx], vb = b[idx];
                        sa += va; sb += vb; saa += va * va; sbb += vb * vb; sab += va * vb;
                    }
                }
                const double mu_a = sa * inv, mu_b = sb * inv;
                const double va = saa * inv - mu_a * mu_a;
                const double vb = sbb * inv - mu_b * mu_b;
                const double cov = sab * inv - mu_a * mu_b;
                ssim_sum += ((2 * mu_a * mu_b + C1) * (2 * cov + C2)) /
                            ((mu_a * mu_a + mu_b * mu_b + C1) * (va + vb + C2));
                ++cnt;
            }
        }
    }
    return cnt > 0 ? ssim_sum / static_cast<double>(cnt) : 0.0;
}

// noise_pred (device FP16) を host float へ回収。bad = Inf/NaN 数。
static std::vector<float> gather_np(const __half* d_np, size_t n, long& bad)
{
    std::vector<__half> h(n);
    CUDA_CHECK(cudaMemcpy(h.data(), d_np, n * sizeof(__half), cudaMemcpyDeviceToHost));
    std::vector<float> out(n);
    bad = 0;
    for (size_t i = 0; i < n; ++i)
    {
        const float v = __half2float(h[i]);
        if (std::isnan(v) || std::isinf(v)) { ++bad; }
        out[i] = v;
    }
    return out;
}

// ----------------------------------------------------------------
// G-8k S2: unet.cu の Scratch / 常駐バッファ (temb / skip / d_cur) をアリーナ化した
//   ことに対する回帰ゲート。**配管のみ**の変更なので出力は完全不変であるべきで、
//   ハードゲートは SSIM ではなく memcmp のビット一致とする。
//
//   S1b と同じ思想で「単なる 3 連走」では不足とみなす:
//     run 間にアリーナ領域を 0xFF / 0x00 で汚染してから rewind し、次の run が
//     同じ領域を再利用するよう仕向ける。ここで出力が変われば「未初期化領域を
//     読んでいた既存バグ」の発覚であり、ゼロ埋め等で通してはならない (即報告)。
//
//   さらに forward あたりの実 cudaMalloc / cudaFree が 0 に収束することも見る
//   (定常状態の最終ゲートは S4 で e2e 実バイナリで取る)。
// ----------------------------------------------------------------

// アリーナ領域を bump 順にたどって汚染し、rewind で戻す (次の確保が同じ領域を再利用)。
static void poison_arena(DeviceArenaId id, size_t bytes, int pattern)
{
    if (bytes == 0)
    {
        return;
    }
    DeviceArenaScope sc(id);
    const size_t     block = (size_t)16 << 20;
    size_t           done  = 0;
    while (done < bytes)
    {
        const size_t n = (bytes - done < block) ? (bytes - done) : block;
        void*        p = sc.alloc_bytes(n);
        if (p == nullptr)
        {
            break;
        }
        CUDA_CHECK(cudaMemset(p, pattern, n));
        done += n;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

// noise_pred の生バイト列を回収 (FP16 のまま = memcmp 用)。
static std::vector<__half> gather_raw(const __half* d_np, size_t n)
{
    std::vector<__half> h(n);
    CUDA_CHECK(cudaMemcpy(h.data(), d_np, n * sizeof(__half), cudaMemcpyDeviceToHost));
    return h;
}

// バイト列をファイルへダンプ (DOLLAMA_G8K_DUMP 接頭辞が設定されているときのみ)。
//   DOLLAMA_POOL=0 との突合はプロセス単位のキルスイッチのため外部 cmp で行う。
static void dump_bytes(const char* label, const void* data, size_t bytes)
{
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
    const char* prefix = std::getenv("DOLLAMA_G8K_DUMP");
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
    if (prefix == nullptr || prefix[0] == '\0')
    {
        return;
    }
    const std::string path = std::string(prefix) + "_" + label + ".bin";
    std::ofstream     ofs(path.c_str(), std::ios::binary);
    if (!ofs)
    {
        std::cerr << "[g8k] ダンプ失敗: " << path << "\n";
        return;
    }
    ofs.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
}

static int run_test()
{
    const std::string wpath  = UNET_WEIGHTS_PATH;
    const std::string iopath = UNET_IO_PATH;

    std::cout << "[test_unet_fast] weights=" << wpath << "\n";
    std::cout << "[test_unet_fast] io=" << iopath << "\n";

    {
        std::ifstream fw(wpath, std::ios::binary), fio(iopath, std::ios::binary);
        if (!fw.good() || !fio.good())
        {
            std::cout << "[test_unet_fast] [SKIP] golden data not found "
                         "(unet_weights/unet_io.safetensors)\n";
            return 0;
        }
    }

    SafeTensors weights(wpath);
    SafeTensors io(iopath);

    // G-3kf: 全 UNet 重みを 1 度だけデバイス常駐させる (warm)。以降の全 launch_unet(handle,...)
    //        は重み再転送ゼロで走る。ハンドルは weights の生存期間内でのみ有効。
    UnetWeightsHandle uh = unet_weights_create(weights);

    // 入力。latent はスケジューラでスケール済み (step0)。
    std::vector<__half> h_latent = load_f16(io, "input_sample_scaled_step0_f16");
    std::vector<__half> h_ehs    = load_f16(io, "input_encoder_hidden_states_f16");
    std::vector<__half> h_txt    = load_f16(io, "input_text_embeds_f16");
    std::vector<__half> h_tids   = load_f16(io, "input_time_ids_f16");
    std::vector<float>  h_ts     = load_f32(io, "input_timestep_f32");
    const float         timestep = h_ts[0];

    const size_t latent_n = (size_t)4 * 128 * 128;
    const size_t ehs_n    = (size_t)77 * 2048;

    __half *d_latent = nullptr, *d_ehs = nullptr, *d_txt = nullptr, *d_tids = nullptr;
    __half *d_np_def = nullptr, *d_np_fast = nullptr;
    CUDA_CHECK(cudaMalloc(&d_latent,  latent_n * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_ehs,     ehs_n    * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_txt,     1280     * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_tids,    6        * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_np_def,  latent_n * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_np_fast, latent_n * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_latent, h_latent.data(), latent_n * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ehs,    h_ehs.data(),    ehs_n    * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_txt,    h_txt.data(),    1280     * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tids,   h_tids.data(),   6        * sizeof(__half), cudaMemcpyHostToDevice));

    bool ok = true;

    // --- (1) default (attn_fast=false) を golden と突合 (warm ハンドル経路) ---
    launch_unet(uh, d_latent, timestep, d_ehs, d_txt, d_tids, d_np_def, /*attn_fast=*/false);
    long              bad_def = 0;
    std::vector<float> got_def = gather_np(d_np_def, latent_n, bad_def);

    std::vector<float> golden = load_f32(io, "noise_pred_f32");
    double sum_abs = 0, max_abs = 0;
    for (size_t i = 0; i < latent_n; ++i)
    {
        const double d = std::fabs((double)got_def[i] - golden[i]);
        sum_abs += d;
        max_abs = std::max(max_abs, d);
    }
    const double mae_def  = sum_abs / static_cast<double>(latent_n);
    const double ssim_def = ssim_uniform(got_def, golden, 4, 128, 128);
    std::cout << "[test_unet_fast] default vs golden: MAE=" << mae_def
              << " max_abs=" << max_abs << " bad=" << bad_def
              << " SSIM=" << ssim_def << "\n";
    if (bad_def != 0)   { std::cerr << "[test_unet_fast] FAIL: default Inf/NaN\n"; ok = false; }
    if (mae_def > 5e-3) { std::cerr << "[test_unet_fast] FAIL: default MAE " << mae_def << " > 5e-3\n"; ok = false; }
    if (ssim_def < 0.99){ std::cerr << "[test_unet_fast] FAIL: default SSIM " << ssim_def << " < 0.99\n"; ok = false; }

    // --- (2) fast (attn_fast=true) を default と突合 (warm ハンドル経路) ---
    launch_unet(uh, d_latent, timestep, d_ehs, d_txt, d_tids, d_np_fast, /*attn_fast=*/true);
    long              bad_fast = 0;
    std::vector<float> got_fast = gather_np(d_np_fast, latent_n, bad_fast);

    double sum_abs2 = 0, max_abs2 = 0;
    for (size_t i = 0; i < latent_n; ++i)
    {
        const double d = std::fabs((double)got_fast[i] - (double)got_def[i]);
        sum_abs2 += d;
        max_abs2 = std::max(max_abs2, d);
    }
    const double mae_fd   = sum_abs2 / static_cast<double>(latent_n);
    const double ssim_fd  = ssim_uniform(got_fast, got_def, 4, 128, 128);
    // fast も golden とも突合しておく (参考ログ)。
    const double ssim_fg  = ssim_uniform(got_fast, golden, 4, 128, 128);
    std::cout << "[test_unet_fast] fast vs default: MAE=" << mae_fd
              << " max_abs=" << max_abs2 << " bad=" << bad_fast
              << " SSIM=" << ssim_fd << "  (fast vs golden SSIM=" << ssim_fg << ")\n";
    if (bad_fast != 0)     { std::cerr << "[test_unet_fast] FAIL: fast Inf/NaN\n"; ok = false; }
    if (ssim_fd < 0.9999)  { std::cerr << "[test_unet_fast] FAIL: fast vs default SSIM "
                                       << ssim_fd << " < 0.9999\n"; ok = false; }

    // --- (2b) epilogue=on (G-4k S1a): attn_fast+epilogue を default と突合 ---
    //     multi-block GN は 1-block 版と蓄積順が異なるため FP16 で微差が乗る (ビット
    //     一致は狙わない)。ゲート = noise_pred SSIM >= 0.9999。
    __half* d_np_ep = nullptr;
    CUDA_CHECK(cudaMalloc(&d_np_ep, latent_n * sizeof(__half)));
    launch_unet(uh, d_latent, timestep, d_ehs, d_txt, d_tids, d_np_ep,
                /*attn_fast=*/true, /*epilogue=*/true);
    long               bad_ep = 0;
    std::vector<float> got_ep = gather_np(d_np_ep, latent_n, bad_ep);

    double sum_abs3 = 0, max_abs3 = 0;
    for (size_t i = 0; i < latent_n; ++i)
    {
        const double d = std::fabs((double)got_ep[i] - (double)got_def[i]);
        sum_abs3 += d;
        max_abs3 = std::max(max_abs3, d);
    }
    const double mae_ep  = sum_abs3 / static_cast<double>(latent_n);
    const double ssim_ep = ssim_uniform(got_ep, got_def, 4, 128, 128);
    const double ssim_eg = ssim_uniform(got_ep, golden, 4, 128, 128);
    std::cout << "[test_unet_fast] epilogue vs default: MAE=" << mae_ep
              << " max_abs=" << max_abs3 << " bad=" << bad_ep
              << " SSIM=" << ssim_ep << "  (epilogue vs golden SSIM=" << ssim_eg << ")\n";
    if (bad_ep != 0)      { std::cerr << "[test_unet_fast] FAIL: epilogue Inf/NaN\n"; ok = false; }
    if (ssim_ep < 0.9999) { std::cerr << "[test_unet_fast] FAIL: epilogue vs default SSIM "
                                      << ssim_ep << " < 0.9999\n"; ok = false; }

    // --- (2c) G-8k S2: アリーナ再利用に対する memcmp ビット一致ゲート ---
    //     default / fast の両方で、汚染 → rewind → 再走 を 3 回まわしてビット一致を取る。
    {
        __half* d_np_g8k = nullptr;
        CUDA_CHECK(cudaMalloc(&d_np_g8k, latent_n * sizeof(__half)));

        struct Cfg { const char* label; bool attn_fast; bool epilogue; };
        const Cfg cfgs[3] = {
            {"default",  false, false},
            {"fast",     true,  false},
            {"epilogue", true,  true },
        };

        for (int ci = 0; ci < 3; ++ci)
        {
            const Cfg& cfg = cfgs[ci];
            std::vector<std::vector<__half>> runs(3);
            const int patterns[3] = {-1, 0xFF, 0x00};  // -1 = 汚染なし
            uint64_t  last_malloc = 0, last_free = 0;
            size_t    cap_unet = 0, cap_persist = 0;

            for (int r = 0; r < 3; ++r)
            {
                if (patterns[r] >= 0)
                {
                    // 直前 run のピーク使用量ぶんを、両アリーナとも bump 順になぞって汚す。
                    const DeviceArenaStats su = device_arena_stats(DeviceArenaId::UNet);
                    const DeviceArenaStats sp = device_arena_stats(DeviceArenaId::UNetPersist);
                    poison_arena(DeviceArenaId::UNet,        su.peak_bytes_in_use, patterns[r]);
                    poison_arena(DeviceArenaId::UNetPersist, sp.peak_bytes_in_use, patterns[r]);
                }
                // 出力バッファも毎回汚す (書き残しがあれば検出できる)。
                CUDA_CHECK(cudaMemset(d_np_g8k, 0xCD, latent_n * sizeof(__half)));

                const DeviceArenaStats u0 = device_arena_stats(DeviceArenaId::UNet);
                const DeviceArenaStats p0 = device_arena_stats(DeviceArenaId::UNetPersist);
                launch_unet(uh, d_latent, timestep, d_ehs, d_txt, d_tids, d_np_g8k,
                            cfg.attn_fast, cfg.epilogue);
                CUDA_CHECK(cudaDeviceSynchronize());
                const DeviceArenaStats u1 = device_arena_stats(DeviceArenaId::UNet);
                const DeviceArenaStats p1 = device_arena_stats(DeviceArenaId::UNetPersist);

                last_malloc = (u1.cuda_malloc_calls - u0.cuda_malloc_calls)
                              + (p1.cuda_malloc_calls - p0.cuda_malloc_calls);
                last_free   = (u1.cuda_free_calls - u0.cuda_free_calls)
                              + (p1.cuda_free_calls - p0.cuda_free_calls);
                cap_unet    = u1.total_capacity;
                cap_persist = p1.total_capacity;
                if (r == 0)
                {
                    // 汚染前 = 実運用そのままの VRAM フットプリント (汚染は 16MiB 粒度で
                    // チャンクを跨ぐため容量を水増しする。ゲート値と分けて報告する)。
                    std::cout << "[g8k_s2] " << cfg.label << " footprint(pre-poison): unet cap="
                              << (u1.total_capacity >> 20) << "MiB peak_in_use="
                              << (u1.peak_bytes_in_use >> 20) << "MiB / persist cap="
                              << (p1.total_capacity >> 20) << "MiB peak_in_use="
                              << (p1.peak_bytes_in_use >> 20) << "MiB\n";
                }

                runs[r] = gather_raw(d_np_g8k, latent_n);
            }

            const size_t nbytes = latent_n * sizeof(__half);
            const bool eq01 = (std::memcmp(runs[0].data(), runs[1].data(), nbytes) == 0);
            const bool eq02 = (std::memcmp(runs[0].data(), runs[2].data(), nbytes) == 0);
            std::cout << "[g8k_s2] " << cfg.label
                      << " run0==run1(0xFF poisoned): " << (eq01 ? "BIT-EXACT" : "DIFF")
                      << " / run0==run2(0x00 poisoned): " << (eq02 ? "BIT-EXACT" : "DIFF")
                      << " | pool=" << (device_arena_pool_enabled() ? "on" : "off")
                      << " last-run cudaMalloc +" << last_malloc
                      << " cudaFree +" << last_free
                      << " cap(unet/persist)=" << (cap_unet >> 20) << "/"
                      << (cap_persist >> 20) << "MiB\n";
            if (!eq01 || !eq02)
            {
                // 未初期化領域の読み出し疑い。ゼロ埋めで通してはならない (即エスカレーション)。
                std::cerr << "[g8k_s2] FAIL: arena reuse changed the output (" << cfg.label << ")\n";
                ok = false;
            }
            // 定常状態 (3 run 目) では forward あたりの実 cudaMalloc/cudaFree が 0 であること。
            if (device_arena_pool_enabled() && (last_malloc != 0 || last_free != 0))
            {
                std::cerr << "[g8k_s2] FAIL: steady-state alloc not zero (" << cfg.label
                          << ") malloc=" << last_malloc << " free=" << last_free << "\n";
                ok = false;
            }
            // default 経路の出力は golden 相当のアンカー。POOL=0 との外部 cmp 用にダンプ。
            dump_bytes(cfg.label, runs[0].data(), nbytes);
        }

        CUDA_CHECK(cudaFree(d_np_g8k));
    }

    // --- (3) 速度: default / fast の 1step レイテンシ (cudaEvent 中央値, warmup2/iters5) ---
    //     warm ハンドル経路 (重み再転送なし) で計測するため attention の律速比率が実態のまま
    //     反映され、fast の速度差 (UNet forward ~1.19x / attention ~1.43x 帯) が正しく出る。
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    auto bench = [&](bool attn_fast, bool epilogue, __half* d_out) -> float
    {
        auto run_once = [&]() -> float
        {
            CUDA_CHECK(cudaEventRecord(start));
            launch_unet(uh, d_latent, timestep, d_ehs, d_txt, d_tids, d_out,
                        attn_fast, epilogue);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            return ms;
        };
        const int warmup = 2, iters = 5;
        for (int i = 0; i < warmup; ++i) { (void)run_once(); }
        std::vector<float> t;
        t.reserve(iters);
        for (int i = 0; i < iters; ++i) { t.push_back(run_once()); }
        std::sort(t.begin(), t.end());
        return t[t.size() / 2];
    };
    const float ms_def  = bench(false, false, d_np_def);
    const float ms_fast = bench(true,  false, d_np_fast);
    const float ms_ep   = bench(true,  true,  d_np_ep);
    std::cout << "[test_unet_fast] 1step latency median (warm): default=" << ms_def
              << " ms  fast=" << ms_fast << " ms  fast+epilogue=" << ms_ep << " ms";
    if (ms_fast > 0.0f)
    {
        std::cout << "  (fast x" << (ms_def / ms_fast)
                  << " / +epilogue x" << (ms_def / ms_ep) << ")";
    }
    std::cout << "\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_latent));
    CUDA_CHECK(cudaFree(d_ehs));
    CUDA_CHECK(cudaFree(d_txt));
    CUDA_CHECK(cudaFree(d_tids));
    CUDA_CHECK(cudaFree(d_np_def));
    CUDA_CHECK(cudaFree(d_np_fast));
    CUDA_CHECK(cudaFree(d_np_ep));
    // 常駐重みハンドルを解放 (weights の生存期間内)。
    unet_weights_destroy(uh);
    return ok ? 0 : 1;
}

#endif // HAVE_CUDA

} // namespace dollama

int main()
{
#ifndef HAVE_CUDA
    std::cout << "[test_unet_fast] [SKIP] HAVE_CUDA undefined\n";
    return 0;
#else
    try
    {
        const int rc = dollama::run_test();
        if (rc == 0) { std::cout << "[test_unet_fast] ALL PASSED\n"; }
        else         { std::cerr << "[test_unet_fast] FAILED\n"; }
        return rc;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[test_unet_fast] exception: " << e.what() << "\n";
        return 1;
    }
#endif
}

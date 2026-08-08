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
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "infer/unet.cuh"
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

// SDXL UNet2DConditionModel ゴールデン突合テスト + 1step レイテンシ計測
// (Phase 2 マイルストーン 2-5)
// HAVE_CUDA 未定義時は [SKIP] で return 0。golden 不在時も [SKIP]。
//
// 段ごとの突合 (unet_set_stage_hook):
//   time_embedding / add_embedding / conv_in / 各 down_block / mid / 各 up_block /
//   conv_norm_out / noise_pred を golden (<name>_f32) と多段粒度で突合する。
//   各段で MAE / max_abs / 相関 (Pearson) を出し、しきい値超過で FAIL。
//   ズレたら直前段で原因特定できる (VAE で実証済みの動線)。
// 最終 noise_pred は SSIM/MAE で本線突合。
//
// ゴールデンパス UNET_WEIGHTS_PATH / UNET_IO_PATH は meson cuda_args で -D 埋め込み。

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
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
    const size_t n = nbytes / sizeof(float);
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
    const size_t n = nbytes / sizeof(__half);
    std::vector<__half> out(n);
    std::memcpy(out.data(), p, nbytes);
    return out;
}

// ----------------------------------------------------------------
// 段ごと突合: グローバルに golden を抱え、フックで都度比較する。
// ----------------------------------------------------------------
static const SafeTensors* g_io   = nullptr;
static bool               g_fail = false;

// 段ごとのしきい値: UNet 中間は FP16 相応。深い段ほど累積誤差が乗るので、
// 緩めの絶対 + 相関で判定する (相関は配線の正しさに敏感、絶対誤差は FP16 量子化に鈍感)。
struct StageTol
{
    double mae;   // 平均絶対誤差の上限
    double corr;  // 相関係数の下限
};

static void compare_stage(const char* name, const __half* d_buf, size_t n)
{
    const std::string key = std::string(name) + "_f32";
    if (g_io == nullptr || !g_io->contains(key))
    {
        return;
    }
    std::vector<float> golden = load_f32(*g_io, key);
    if (golden.size() != n)
    {
        std::cerr << "[test_unet]   " << name << " size mismatch got=" << n
                  << " golden=" << golden.size() << "\n";
        g_fail = true;
        return;
    }
    cudaDeviceSynchronize();
    std::vector<__half> h(n);
    cudaMemcpy(h.data(), d_buf, n * sizeof(__half), cudaMemcpyDeviceToHost);

    double sum_abs = 0, max_abs = 0;
    double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
    long   bad = 0;
    for (size_t i = 0; i < n; ++i)
    {
        const float v = __half2float(h[i]);
        if (std::isnan(v) || std::isinf(v)) { ++bad; }
        const double a = v, b = golden[i];
        const double d = std::fabs(a - b);
        sum_abs += d;
        max_abs = std::max(max_abs, d);
        sa += a; sb += b; saa += a * a; sbb += b * b; sab += a * b;
    }
    const double inv  = 1.0 / static_cast<double>(n);
    const double mae  = sum_abs * inv;
    const double cova = sab * inv - (sa * inv) * (sb * inv);
    const double va   = saa * inv - (sa * inv) * (sa * inv);
    const double vb   = sbb * inv - (sb * inv) * (sb * inv);
    const double corr = (va > 0 && vb > 0) ? cova / std::sqrt(va * vb) : 1.0;

    // 段別しきい値 (累積誤差を考慮)。
    static const std::map<std::string, StageTol> tols = {
        {"time_embedding_out",       {1e-3, 0.9999}},
        {"add_embedding_out",        {1e-3, 0.9999}},
        {"conv_in_out",              {5e-4, 0.9999}},
        {"down_block_0_resnet0_out", {1e-3, 0.9999}},
        {"down_block_0_out",         {2e-3, 0.9999}},
        {"down_block_1_resnet0_out", {2e-3, 0.9999}},
        {"down_block_1_attn0_out",   {4e-3, 0.9999}},
        {"down_block_1_out",         {6e-3, 0.9999}},
        {"down_block_2_resnet0_out", {6e-3, 0.9999}},
        {"down_block_2_attn0_out",   {1e-2, 0.9999}},
        {"down_block_2_out",         {1.2e-2, 0.9999}},
        {"mid_block_resnet0_out",    {1.2e-2, 0.9999}},
        {"mid_block_attn0_out",      {2.5e-2, 0.9999}},
        {"mid_block_out",            {2.5e-2, 0.9999}},
        {"up_block_0_resnet0_out",   {2.5e-2, 0.9999}},
        {"up_block_0_attn0_out",     {3e-2, 0.9999}},
        {"up_block_0_out",           {3e-2, 0.9999}},
        {"up_block_1_resnet0_out",   {1.5e-2, 0.9999}},
        {"up_block_1_attn0_out",     {1.5e-2, 0.9999}},
        {"up_block_1_out",           {1e-2, 0.9999}},
        {"up_block_2_resnet0_out",   {5e-3, 0.9999}},
        {"up_block_2_out",           {5e-3, 0.9999}},
        {"conv_norm_out_out",        {2e-3, 0.9999}},
        {"noise_pred",               {1e-3, 0.9999}},
    };
    StageTol tol{1.0, 0.0};
    auto it = tols.find(name);
    if (it != tols.end()) { tol = it->second; }

    const bool ok = (bad == 0) && (mae <= tol.mae) && (corr >= tol.corr);
    std::cout << "[test_unet]   " << (ok ? "OK  " : "FAIL")
              << " " << name
              << "  MAE=" << mae << " max=" << max_abs
              << " corr=" << corr << " bad=" << bad
              << "  (tol mae<=" << tol.mae << " corr>=" << tol.corr << ")\n";
    if (!ok) { g_fail = true; }
}

// ----------------------------------------------------------------
// 簡易 SSIM (一様窓, NCHW)。test_vae_decode と同方針。noise_pred 用。
// ----------------------------------------------------------------
static double ssim_uniform(const std::vector<float>& a, const std::vector<float>& b,
                           int C, int H, int W)
{
    float lo = a[0], hi = a[0];
    for (float v : a) { lo = std::min(lo, v); hi = std::max(hi, v); }
    for (float v : b) { lo = std::min(lo, v); hi = std::max(hi, v); }
    const double L  = static_cast<double>(hi - lo);
    const double C1 = (0.01 * L) * (0.01 * L);
    const double C2 = (0.03 * L) * (0.03 * L);
    const int win = 7, half = win >> 1;
    const double inv = 1.0 / static_cast<double>(win * win);
    double ssim_sum = 0.0; long cnt = 0;
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
                        const long idx = base + static_cast<long>(y + dy) * W + (x + dx);
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

static int run_test()
{
    const std::string wpath  = UNET_WEIGHTS_PATH;
    const std::string iopath = UNET_IO_PATH;

    std::cout << "[test_unet] weights=" << wpath << "\n";
    std::cout << "[test_unet] io=" << iopath << "\n";

    {
        std::ifstream fw(wpath, std::ios::binary), fio(iopath, std::ios::binary);
        if (!fw.good() || !fio.good())
        {
            std::cout << "[test_unet] [SKIP] golden data not found "
                         "(unet_weights/unet_io.safetensors)\n";
            return 0;
        }
    }

    SafeTensors weights(wpath);
    SafeTensors io(iopath);
    g_io = &io;

    // 入力。latent はスケジューラでスケール済み (step0)。
    std::vector<__half> h_latent  = load_f16(io, "input_sample_scaled_step0_f16");
    std::vector<__half> h_ehs     = load_f16(io, "input_encoder_hidden_states_f16");
    std::vector<__half> h_txt     = load_f16(io, "input_text_embeds_f16");
    std::vector<__half> h_tids    = load_f16(io, "input_time_ids_f16");
    std::vector<float>  h_ts      = load_f32(io, "input_timestep_f32");
    const float timestep = h_ts[0];

    const size_t latent_n = (size_t)4 * 128 * 128;
    const size_t ehs_n    = (size_t)77 * 2048;
    if (h_latent.size() != latent_n) { std::cerr << "latent size\n"; return 1; }
    if (h_ehs.size()    != ehs_n)    { std::cerr << "ehs size\n";    return 1; }
    if (h_txt.size()    != 1280)     { std::cerr << "txt size\n";    return 1; }
    if (h_tids.size()   != 6)        { std::cerr << "tids size\n";   return 1; }

    std::cout << "[test_unet] timestep=" << timestep << "\n";

    __half *d_latent = nullptr, *d_ehs = nullptr, *d_txt = nullptr, *d_tids = nullptr, *d_np = nullptr;
    CUDA_CHECK(cudaMalloc(&d_latent, latent_n * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_ehs,    ehs_n    * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_txt,    1280     * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_tids,   6        * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_np,     latent_n * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_latent, h_latent.data(), latent_n * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ehs,    h_ehs.data(),    ehs_n    * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_txt,    h_txt.data(),    1280     * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tids,   h_tids.data(),   6        * sizeof(__half), cudaMemcpyHostToDevice));

    {
        size_t freeb = 0, totalb = 0;
        CUDA_CHECK(cudaMemGetInfo(&freeb, &totalb));
        std::cout << "[test_unet] VRAM free=" << (freeb >> 20) << "MB / total="
                  << (totalb >> 20) << "MB (before forward)\n";
    }

    // --- 段ごと突合付きで 1 回実行 ---
    std::cout << "[test_unet] --- staged golden compare ---\n";
    unet_set_stage_hook(compare_stage);
    launch_unet(weights, d_latent, timestep, d_ehs, d_txt, d_tids, d_np);
    unet_set_stage_hook(nullptr);

    // --- 最終 noise_pred を SSIM/MAE で本線突合 ---
    std::vector<float> golden = load_f32(io, "noise_pred_f32");
    std::vector<__half> h_np(latent_n);
    CUDA_CHECK(cudaMemcpy(h_np.data(), d_np, latent_n * sizeof(__half), cudaMemcpyDeviceToHost));
    std::vector<float> got(latent_n);
    double sum_abs = 0, max_abs = 0; long bad = 0;
    for (size_t i = 0; i < latent_n; ++i)
    {
        const float v = __half2float(h_np[i]);
        if (std::isnan(v) || std::isinf(v)) { ++bad; }
        got[i] = v;
        const double d = std::fabs((double)v - golden[i]);
        sum_abs += d; max_abs = std::max(max_abs, d);
    }
    const double mae  = sum_abs / static_cast<double>(latent_n);
    const double ssim = ssim_uniform(got, golden, 4, 128, 128);
    std::cout << "[test_unet] noise_pred MAE=" << mae << " max_abs=" << max_abs
              << " bad=" << bad << " SSIM=" << ssim << "\n";

    bool ok = !g_fail;
    if (bad != 0)            { std::cerr << "[test_unet] FAIL: Inf/NaN in noise_pred\n"; ok = false; }
    if (mae > 5e-3)          { std::cerr << "[test_unet] FAIL: noise_pred MAE " << mae << " > 5e-3\n"; ok = false; }
    if (ssim < 0.99)         { std::cerr << "[test_unet] FAIL: noise_pred SSIM " << ssim << " < 0.99\n"; ok = false; }

    // --- 1step レイテンシ計測 (cudaEvent 中央値, warmup3/iters7) ---
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    auto run_once = [&]() -> float
    {
        CUDA_CHECK(cudaEventRecord(start));
        launch_unet(weights, d_latent, timestep, d_ehs, d_txt, d_tids, d_np);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    };
    const int warmup = 2;  // 上の突合実行 + ここで計 3 回ウォーム
    const int iters  = 7;
    for (int i = 0; i < warmup; ++i) { (void)run_once(); }
    std::vector<float> times; times.reserve(iters);
    for (int i = 0; i < iters; ++i) { times.push_back(run_once()); }
    std::sort(times.begin(), times.end());
    std::cout << "[test_unet] 1step latency median=" << times[times.size() / 2] << " ms"
              << " (min=" << times.front() << " max=" << times.back()
              << ", N=" << iters << ")\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_latent));
    CUDA_CHECK(cudaFree(d_ehs));
    CUDA_CHECK(cudaFree(d_txt));
    CUDA_CHECK(cudaFree(d_tids));
    CUDA_CHECK(cudaFree(d_np));
    g_io = nullptr;
    return ok ? 0 : 1;
}

#endif // HAVE_CUDA

} // namespace dollama

int main()
{
#ifndef HAVE_CUDA
    std::cout << "[test_unet] [SKIP] HAVE_CUDA undefined\n";
    return 0;
#else
    try
    {
        const int rc = dollama::run_test();
        if (rc == 0) { std::cout << "[test_unet] ALL PASSED\n"; }
        else         { std::cerr << "[test_unet] FAILED\n"; }
        return rc;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[test_unet] exception: " << e.what() << "\n";
        return 1;
    }
#endif
}

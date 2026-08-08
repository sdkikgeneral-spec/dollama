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

// ----------------------------------------------------------------
// B=2 バッチ forward の per-sample パリティ突合 (G-2k S2 フォローアップ)。
// launch_unet_batched(handle, 2, ...) 1 回の出力を、launch_unet(handle,...) を
// 2 サンプル個別に呼んで連結したリファレンスと突合する。
// 主眼: cross-attn to_k/to_v が M=B*77=154 で wmma M タイル 16 を跨ぐ B=2 の
//       サンプル行相対位置ずれを実測で塞ぐ (sample1 = 2 番目のバッチ行が要)。
// noise_pred の MAE=0 (ビット一致) が第一目標。届かねば FP16 tol 内許容。
// ----------------------------------------------------------------
static int run_batch_parity_test()
{
    const std::string wpath  = UNET_WEIGHTS_PATH;
    const std::string iopath = UNET_IO_PATH;

    std::cout << "[test_unet] === batch parity (B=2 per-sample) ===\n";

    {
        std::ifstream fw(wpath, std::ios::binary), fio(iopath, std::ios::binary);
        if (!fw.good() || !fio.good())
        {
            std::cout << "[test_unet] [SKIP] golden data not found (batch parity)\n";
            return 0;
        }
    }

    SafeTensors weights(wpath);
    SafeTensors io(iopath);

    // sample0 = 既存 golden 入力をそのまま流用。
    std::vector<__half> h_latent0 = load_f16(io, "input_sample_scaled_step0_f16");
    std::vector<__half> h_ehs0    = load_f16(io, "input_encoder_hidden_states_f16");
    std::vector<__half> h_txt0    = load_f16(io, "input_text_embeds_f16");
    std::vector<__half> h_tids    = load_f16(io, "input_time_ids_f16");
    std::vector<float>  h_ts      = load_f32(io, "input_timestep_f32");
    const float timestep = h_ts[0];

    const size_t latent_n = (size_t)4 * 128 * 128;
    const size_t ehs_n    = (size_t)77 * 2048;
    if (h_latent0.size() != latent_n) { std::cerr << "latent size\n"; return 1; }
    if (h_ehs0.size()    != ehs_n)    { std::cerr << "ehs size\n";    return 1; }
    if (h_txt0.size()    != 1280)     { std::cerr << "txt size\n";    return 1; }
    if (h_tids.size()    != 6)        { std::cerr << "tids size\n";   return 1; }

    // sample1 = sample0 を決定的に微小摂動したもの (乱数不使用・再現可能)。
    // index 依存の微小オフセットを FP16 精度で加える。摂動により 2 番目のバッチ行が
    // 1 番目と異なる値を持ち、タイル内相対位置の取り違えがあれば MAE が跳ねる。
    std::vector<__half> h_latent1(latent_n);
    for (size_t i = 0; i < latent_n; ++i)
    {
        const float v   = __half2float(h_latent0[i]);
        const float off = 0.01f * std::sin(0.001f * static_cast<float>(i));
        h_latent1[i]    = __float2half(v + off);
    }
    std::vector<__half> h_ehs1(ehs_n);
    for (size_t i = 0; i < ehs_n; ++i)
    {
        const float v   = __half2float(h_ehs0[i]);
        const float off = 0.005f * std::cos(0.0007f * static_cast<float>(i));
        h_ehs1[i]       = __float2half(v + off);
    }
    std::vector<__half> h_txt1(1280);
    for (size_t i = 0; i < 1280; ++i)
    {
        const float v   = __half2float(h_txt0[i]);
        const float off = 0.003f * static_cast<float>((int)(i % 7) - 3);
        h_txt1[i]       = __float2half(v + off);
    }

    // 常駐ハンドルを 1 個作り、参照/バッチ両経路で共有する。
    UnetWeightsHandle handle = unet_weights_create(weights);

    // --- 参照経路: sample0 / sample1 を B=1 で個別に 2 回呼んで連結 ---
    __half *d_latent = nullptr, *d_ehs = nullptr, *d_txt = nullptr, *d_tids = nullptr, *d_np = nullptr;
    CUDA_CHECK(cudaMalloc(&d_latent, latent_n * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_ehs,    ehs_n    * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_txt,    1280     * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_tids,   6        * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_np,     latent_n * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_tids, h_tids.data(), 6 * sizeof(__half), cudaMemcpyHostToDevice));

    std::vector<__half> ref(2 * latent_n);

    // sample0
    CUDA_CHECK(cudaMemcpy(d_latent, h_latent0.data(), latent_n * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ehs,    h_ehs0.data(),    ehs_n    * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_txt,    h_txt0.data(),    1280     * sizeof(__half), cudaMemcpyHostToDevice));
    launch_unet(handle, d_latent, timestep, d_ehs, d_txt, d_tids, d_np, /*attn_fast=*/false);
    CUDA_CHECK(cudaMemcpy(ref.data(), d_np, latent_n * sizeof(__half), cudaMemcpyDeviceToHost));

    // sample1
    CUDA_CHECK(cudaMemcpy(d_latent, h_latent1.data(), latent_n * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ehs,    h_ehs1.data(),    ehs_n    * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_txt,    h_txt1.data(),    1280     * sizeof(__half), cudaMemcpyHostToDevice));
    launch_unet(handle, d_latent, timestep, d_ehs, d_txt, d_tids, d_np, /*attn_fast=*/false);
    CUDA_CHECK(cudaMemcpy(ref.data() + latent_n, d_np, latent_n * sizeof(__half), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_latent));
    CUDA_CHECK(cudaFree(d_ehs));
    CUDA_CHECK(cudaFree(d_txt));
    CUDA_CHECK(cudaFree(d_np));

    // --- バッチ経路: b-major 連続に詰めて launch_unet_batched(handle, 2, ...) を 1 回 ---
    const int B = 2;
    __half *db_latent = nullptr, *db_ehs = nullptr, *db_txt = nullptr, *db_np = nullptr;
    CUDA_CHECK(cudaMalloc(&db_latent, B * latent_n * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&db_ehs,    B * ehs_n    * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&db_txt,    B * 1280     * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&db_np,     B * latent_n * sizeof(__half)));

    // latent [B,4,128,128] b-major
    CUDA_CHECK(cudaMemcpy(db_latent,            h_latent0.data(), latent_n * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db_latent + latent_n, h_latent1.data(), latent_n * sizeof(__half), cudaMemcpyHostToDevice));
    // ehs [B,77,2048] b-major
    CUDA_CHECK(cudaMemcpy(db_ehs,         h_ehs0.data(), ehs_n * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db_ehs + ehs_n, h_ehs1.data(), ehs_n * sizeof(__half), cudaMemcpyHostToDevice));
    // txt [B,1280] b-major
    CUDA_CHECK(cudaMemcpy(db_txt,        h_txt0.data(), 1280 * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db_txt + 1280, h_txt1.data(), 1280 * sizeof(__half), cudaMemcpyHostToDevice));

    {
        size_t freeb = 0, totalb = 0;
        CUDA_CHECK(cudaMemGetInfo(&freeb, &totalb));
        std::cout << "[test_unet] VRAM free=" << (freeb >> 20) << "MB / total="
                  << (totalb >> 20) << "MB (before batched forward)\n";
    }

    // time_ids は 6 要素で B 共有 (複製しない・d_tids をそのまま渡す)。
    launch_unet_batched(handle, B, db_latent, timestep, db_ehs, db_txt, d_tids, db_np, /*attn_fast=*/false);

    std::vector<__half> got(B * latent_n);
    CUDA_CHECK(cudaMemcpy(got.data(), db_np, B * latent_n * sizeof(__half), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_tids));
    CUDA_CHECK(cudaFree(db_latent));
    CUDA_CHECK(cudaFree(db_ehs));
    CUDA_CHECK(cudaFree(db_txt));
    CUDA_CHECK(cudaFree(db_np));
    unet_weights_destroy(handle);

    // --- 突合: 両サンプルを個別に比較。sample1 (2 番目のバッチ行) を主眼にログ ---
    bool ok = true;
    for (int b = 0; b < B; ++b)
    {
        const size_t off = static_cast<size_t>(b) * latent_n;
        double sum_abs = 0, max_abs = 0;
        size_t exact = 0;
        long   bad = 0;
        for (size_t i = 0; i < latent_n; ++i)
        {
            const float g = __half2float(got[off + i]);
            const float r = __half2float(ref[off + i]);
            if (std::isnan(g) || std::isinf(g)) { ++bad; }
            const double d = std::fabs((double)g - (double)r);
            sum_abs += d;
            max_abs = std::max(max_abs, d);
            if (got[off + i] == ref[off + i]) { ++exact; }
        }
        const double mae = sum_abs / static_cast<double>(latent_n);
        const bool bit_exact = (max_abs == 0.0);
        const char* tag = (b == 0)
            ? "sample0"
            : "sample1 (2nd batch row = wmma M-tile 相対位置ずれ検証の主眼)";
        std::cout << "[test_unet]   " << tag
                  << "  MAE=" << mae << " max_abs=" << max_abs
                  << " exact=" << exact << "/" << latent_n
                  << " bad=" << bad
                  << (bit_exact ? "  (BIT-EXACT)" : "  (not bit-exact)") << "\n";

        // ビット一致が第一目標。届かなくても既存 noise_pred と同じ FP16 tol 内なら緑。
        if (bad != 0)   { std::cerr << "[test_unet] FAIL: Inf/NaN in batch sample " << b << "\n"; ok = false; }
        if (mae > 5e-3) { std::cerr << "[test_unet] FAIL: batch sample " << b
                                    << " MAE " << mae << " > 5e-3\n"; ok = false; }
    }

    std::cout << "[test_unet] batch parity " << (ok ? "OK" : "FAIL") << "\n";
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

        // G-2k S2 フォローアップ: B=2 per-sample パリティ突合を続けて実行。
        const int rc_batch = dollama::run_batch_parity_test();
        if (rc_batch == 0) { std::cout << "[test_unet] BATCH PARITY PASSED\n"; }
        else               { std::cerr << "[test_unet] BATCH PARITY FAILED\n"; }

        return (rc == 0 && rc_batch == 0) ? 0 : 1;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[test_unet] exception: " << e.what() << "\n";
        return 1;
    }
#endif
}

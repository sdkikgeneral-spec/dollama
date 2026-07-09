// FAST モード G-2k S3c: CFG batch2 on/off パリティゲート (guidance=1.0 判定 + sweep 記録)。
//   generate_txt2img (CFG ループ全体) を batch2=off / batch2=on で走らせ、最終 RGB を突合する。
//
//   batch2 は「cond/uncond を逐次 2 forward」を「B=2 で 1 forward」へ束ねるだけの数学的同一変換。
//   ただし束ねると linear の GEMM が M=tokens → M=B*tokens に変わり cuBLAS のタイル選択=蓄積順が
//   変わるため、FP16 蓄積で ~1 ULP/要素の差が出る (S2 で per-sample MAE 6.4e-5 と実証済み・ビット
//   一致には非到達)。この微小差は CFG 合成 noise = uncond + g*(cond-uncond) で **guidance 倍に増幅**
//   され step ごとに累積 → 最終 RGB の発散量は guidance に単調 (実測 g=1:SSIM0.9995 / g=3:0.9988 /
//   g=7.5:0.9966・平均差 <=0.3/255 で絵は実質同一)。ノイズ床 (off↔off 同一構成) は完全 bit-exact。
//
//   → **ゲート設計 (ユーザー決裁 2026-07-09)**: CFG 増幅を排して batch2 単独の数学的等価性を判定する
//     ため **guidance=1.0 の最終 RGB SSIM >= 0.999 をハードゲート**にする (計画書 `--fast` の RGB 基準)。
//     g=1.0 は noise = cond で CFG 増幅ゼロゆえ batch2 の tiling 差そのものだけが出る。realistic な
//     g=3/7.5 は characterization として出力のみ (増幅の単調性を可視化・regression 監視の soft ゲート)。
//
//   attn_fast は両側 false 固定で batch2 単独の差分だけを分離する (batch2=true でも fast=false ゆえ
//   コンストラクタは attn_fast を立てない)。
//
//   VRAM 二重常駐回避: off パイプライン (UNet 5.1GB + VAE) で全 guidance を回してから破棄し、on を構築
//   する (重みの再ロードも off/on 各 1 回に抑える・generate_txt2img は guidance を引数で受ける)。
//
// HAVE_CUDA 未定義時 / 重み不在時は [SKIP] で return 0。
// ゴールデンパス UNET_WEIGHTS_PATH / VAE_WEIGHTS_PATH / UNET_IO_PATH は
// meson cuda_args で -D 埋め込み (cwd 非依存)。

#include <algorithm>
#include <chrono>
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
#include "infer/diffusion.cuh"
#include "kernels/utils.cuh"
#include "io/safetensors.hpp"
#endif

namespace dollama
{

#ifdef HAVE_CUDA

// SafeTensors の FP16 テンソルを host float に展開する (要素数検査付き)。
static std::vector<float> load_f16_as_f32(const SafeTensors& st, const std::string& name,
                                          size_t expect)
{
    if (st.dtype(name) != StDtype::F16)
    {
        throw std::runtime_error("load_f16_as_f32: '" + name + "' is not F16");
    }
    size_t nbytes = 0;
    const uint8_t* p = st.tensor_bytes(name, nbytes);
    const size_t   n = nbytes / sizeof(__half);
    if (n != expect)
    {
        throw std::runtime_error("load_f16_as_f32: '" + name + "' size mismatch");
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

// HWC uint8 RGB を planar CHW float へ並べ替える (SSIM をチャンネル面ごとに取るため)。
static std::vector<float> hwc_u8_to_chw_f32(const std::vector<uint8_t>& rgb, int H, int W)
{
    std::vector<float> out(static_cast<size_t>(3) * H * W);
    for (int y = 0; y < H; ++y)
    {
        for (int x = 0; x < W; ++x)
        {
            const size_t hwc = (static_cast<size_t>(y) * W + x) * 3;
            for (int c = 0; c < 3; ++c)
            {
                out[(static_cast<size_t>(c) * H + y) * W + x] =
                    static_cast<float>(rgb[hwc + c]);
            }
        }
    }
    return out;
}

// 簡易 SSIM (一様窓, CHW)。test_unet_fast と同方針・同係数。
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

// uint8 RGB 2 枚の MAE と最大絶対差を返す。
struct DiffStat { double mae; int max_abs; };
static DiffStat rgb_diff(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b)
{
    double sum_abs = 0.0;
    int    max_abs = 0;
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i)
    {
        const int d = std::abs(static_cast<int>(a[i]) - static_cast<int>(b[i]));
        sum_abs += d;
        max_abs = std::max(max_abs, d);
    }
    return { n > 0 ? sum_abs / static_cast<double>(n) : 0.0, max_abs };
}

// on/off の 1024x1024 RGB を突合し MAE/max/SSIM を返す (SSIM は CHW planar float 上で計算)。
static double compare_rgb(const std::vector<uint8_t>& off, const std::vector<uint8_t>& on,
                          double& mae_out, int& max_out)
{
    const DiffStat d = rgb_diff(off, on);
    mae_out = d.mae;
    max_out = d.max_abs;
    const std::vector<float> a = hwc_u8_to_chw_f32(off, 1024, 1024);
    const std::vector<float> b = hwc_u8_to_chw_f32(on, 1024, 1024);
    return ssim_uniform(a, b, 3, 1024, 1024);
}

static int run_test()
{
    const std::string unet_w = UNET_WEIGHTS_PATH;
    const std::string vae_w  = VAE_WEIGHTS_PATH;
    const std::string iopath = UNET_IO_PATH;

    std::cout << "[test_diffusion_batch2] unet_weights=" << unet_w << "\n";
    std::cout << "[test_diffusion_batch2] vae_weights=" << vae_w << "\n";
    std::cout << "[test_diffusion_batch2] io=" << iopath << "\n";

    {
        std::ifstream fu(unet_w, std::ios::binary);
        std::ifstream fv(vae_w, std::ios::binary);
        std::ifstream fe(iopath, std::ios::binary);
        if (!fu.good() || !fv.good() || !fe.good())
        {
            std::cout << "[test_diffusion_batch2] [SKIP] golden/weights not found "
                         "(unet_weights / vae_weights / unet_io safetensors)\n";
            return 0;
        }
    }

    // --- CFG 埋め込みを golden から用意する ---
    //   cond   = golden 埋め込み (input_encoder_hidden_states / text_embeds)。
    //   uncond = cond のスケール変形 (実 CFG に近い有限差・cond != uncond で 2 スライスを別々に走らせる)。
    //   time_ids は cond/uncond 共通 (golden の input_time_ids)。
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
    // uncond: 既定は cond のスケール変形。DB2_UNCOND_ZERO=1 で zeros (診断用)。
    std::vector<float> uncond_ehs(cond_ehs);
    std::vector<float> uncond_txt(cond_txt);
    if (std::getenv("DB2_UNCOND_ZERO"))
    {
        std::fill(uncond_ehs.begin(), uncond_ehs.end(), 0.0f);
        std::fill(uncond_txt.begin(), uncond_txt.end(), 0.0f);
    }
    else
    {
        for (float& v : uncond_ehs) { v *= 0.5f; }
        for (float& v : uncond_txt) { v *= 0.5f; }
    }

    int            steps = 4;        // 束ねの同一性は step 数に依らない (短縮)。
    const uint64_t seed  = 1234ULL;
    if (const char* e = std::getenv("DB2_STEPS")) { steps = std::atoi(e); }

    // guidance sweep: [0]=1.0 (CFG 増幅ゼロ=ハードゲート) / [1]=3.0 / [2]=7.5 (characterization)。
    const float guids[3] = { 1.0f, 3.0f, 7.5f };
    const int   nG        = 3;
    std::cout << "[test_diffusion_batch2] steps=" << steps
              << " guidance sweep={1.0, 3.0, 7.5} (gate=g1.0)\n";

    // 1 パイプラインで guidance を振る generate ラッパ。
    auto gen = [&](DiffusionPipeline& pipe, float g, std::vector<uint8_t>& rgb, int& w, int& h)
    {
        pipe.generate_txt2img(steps, seed, g,
                              cond_ehs.data(), cond_txt.data(),
                              uncond_ehs.data(), uncond_txt.data(),
                              time_ids.data(), rgb, w, h);
    };

    // --- (1) batch2 = off (default 相当)。全 guidance を回し、g=1.0 は determinism 制御用に 2 回 ---
    std::vector<uint8_t> off[3];
    std::vector<uint8_t> off_g1_again;   // ノイズ床制御 (off@1.0 の 2 回目)
    int w = 0, h = 0;
    {
        FastConfig cfg_off;  // 全 false (batch2=false / attn_fast=false)
        DiffusionPipeline pipe_off(unet_w, vae_w, iopath, cfg_off);
        for (int k = 0; k < nG; ++k)
        {
            int ww = 0, hh = 0;
            gen(pipe_off, guids[k], off[k], ww, hh);
            if (k == 0) { w = ww; h = hh; }
        }
        int ww = 0, hh = 0;
        gen(pipe_off, guids[0], off_g1_again, ww, hh);  // 同一構成 same seed = bit-exact 期待
    } // off パイプライン破棄 → VRAM 解放してから on を構築。

    {
        size_t freeb = 0, totalb = 0;
        CUDA_CHECK(cudaMemGetInfo(&freeb, &totalb));
        std::cout << "[test_diffusion_batch2] VRAM free=" << (freeb >> 20)
                  << "MB (after off pipeline destroyed)\n";
    }

    // --- (2) batch2 = on。全 guidance を回す (attn_fast は off 側と同値 = false 固定) ---
    std::vector<uint8_t> on[3];
    int w2 = 0, h2 = 0;
    {
        FastConfig cfg_on;
        cfg_on.batch2 = true;
        DiffusionPipeline pipe_on(unet_w, vae_w, iopath, cfg_on);
        for (int k = 0; k < nG; ++k)
        {
            int ww = 0, hh = 0;
            gen(pipe_on, guids[k], on[k], ww, hh);
            if (k == 0) { w2 = ww; h2 = hh; }
        }
    }

    bool ok = true;

    // --- 形状一致検査 ---
    if (w != 1024 || h != 1024 || w2 != 1024 || h2 != 1024)
    {
        std::cerr << "[test_diffusion_batch2] FAIL: resolution off=" << w << "x" << h
                  << " on=" << w2 << "x" << h2 << " != 1024x1024\n";
        ok = false;
    }
    for (int k = 0; k < nG && ok; ++k)
    {
        if (off[k].size() != on[k].size()
            || off[k].size() != static_cast<size_t>(1024) * 1024 * 3)
        {
            std::cerr << "[test_diffusion_batch2] FAIL: rgb size g=" << guids[k]
                      << " off=" << off[k].size() << " on=" << on[k].size() << "\n";
            ok = false;
        }
    }

    if (!ok)
    {
        std::cerr << "[test_diffusion_batch2] FAILED (shape)\n";
        return 1;
    }

    // --- ノイズ床制御: off@g=1.0 の 2 回が bit-exact (determinism) であること ---
    {
        const DiffStat d = rgb_diff(off[0], off_g1_again);
        std::cout << "[test_diffusion_batch2] noise floor (off@g=1.0 x2): MAE=" << d.mae
                  << " max_abs=" << d.max_abs << "\n";
        if (d.mae != 0.0 || d.max_abs != 0)
        {
            std::cerr << "[test_diffusion_batch2] FAIL: noise floor not bit-exact "
                         "(default 経路が非決定的 = 発散源が batch2 に切り分けられない)\n";
            ok = false;
        }
    }

    // --- characterization sweep: on vs off @ 各 guidance ---
    double ssim_g1 = -1.0;
    for (int k = 0; k < nG; ++k)
    {
        double mae = 0.0;
        int    mx  = 0;
        const double ssim = compare_rgb(off[k], on[k], mae, mx);
        const char* tag = (k == 0) ? " [GATE]" : "";
        std::cout << "[test_diffusion_batch2] sweep g=" << guids[k]
                  << " on-vs-off: MAE=" << mae << " max_abs=" << mx
                  << " SSIM=" << ssim << tag << "\n";
        if (mae == 0.0)
        {
            std::cout << "[test_diffusion_batch2]   bit-exact @ g=" << guids[k] << "\n";
        }
        if (k == 0) { ssim_g1 = ssim; }
    }

    // --- ハードゲート: g=1.0 の最終 RGB SSIM >= 0.999 (CFG 増幅を排した batch2 数学的等価性) ---
    if (ssim_g1 < 0.999)
    {
        std::cerr << "[test_diffusion_batch2] FAIL: gate SSIM(g=1.0) " << ssim_g1
                  << " < 0.999\n";
        ok = false;
    }

    if (ok)
    {
        std::cout << "[test_diffusion_batch2] parity PASSED (gate g=1.0 SSIM=" << ssim_g1
                  << " >= 0.999)\n";
    }

    // --- DB2_BENCH: batch2 の e2e 速度 (G-2k のヘッドライン ~2×) を warm 実測する ---
    //   generate_txt2img 全体 (CFG UNet ループ + VAE decode) の wall-clock を off/on で計測。
    //   VAE decode は両者共通 (~1.2s) ゆえ speedup は UNet CFG ループ差で希釈されるが、
    //   これが正典 CFG e2e ベースライン (計画書 line52) = 出荷判定 (diffusers 3.8s 比) の分母。
    //   meson test では非設定 → parity のみで高速。手動 DB2_BENCH=1 で計測モードに入る。
    if (ok && std::getenv("DB2_BENCH"))
    {
        int bsteps = 20;                          // 出荷相当 20step。
        int iters  = 2;                           // 本計測 (min を採用)。
        const float bg = 7.5f;                    // 出荷相当 guidance。
        if (const char* e = std::getenv("DB2_BENCH_STEPS")) { bsteps = std::atoi(e); }
        if (const char* e = std::getenv("DB2_BENCH_ITERS")) { iters  = std::atoi(e); }
        std::cout << "[test_diffusion_batch2] BENCH steps=" << bsteps << " guidance=" << bg
                  << " iters=" << iters << " (warmup1・min 採用・VAE 共通込み e2e)\n";

        // 1 パイプラインを warmup 1 回 + iters 回計測し最小 ms を返す。
        auto bench_pipe = [&](bool attn_fast, bool batch2) -> double
        {
            FastConfig cfg;
            cfg.attn_fast = attn_fast;
            cfg.batch2    = batch2;
            DiffusionPipeline pipe(unet_w, vae_w, iopath, cfg);
            std::vector<uint8_t> rgb;
            int ww = 0, hh = 0;
            auto one = [&]()
            {
                pipe.generate_txt2img(bsteps, seed, bg,
                                      cond_ehs.data(), cond_txt.data(),
                                      uncond_ehs.data(), uncond_txt.data(),
                                      time_ids.data(), rgb, ww, hh);
            };
            one();                                      // warmup (lazy init を排す)
            double best = 1e30;
            for (int it = 0; it < iters; ++it)
            {
                CUDA_CHECK(cudaDeviceSynchronize());
                const auto t0 = std::chrono::steady_clock::now();
                one();
                CUDA_CHECK(cudaDeviceSynchronize());
                const auto t1 = std::chrono::steady_clock::now();
                const double ms =
                    std::chrono::duration<double, std::milli>(t1 - t0).count();
                best = std::min(best, ms);
            }
            return best;
        };

        const double def_ms  = bench_pipe(false, false);  // default (逐次 2 forward)
        const double b2_ms   = bench_pipe(false, true);   // batch2 単独 (attn_fast off)
        const double fast_ms = bench_pipe(true,  true);   // --fast 全部 (attn_fast + batch2)
        std::cout << "[test_diffusion_batch2] BENCH e2e (min ms): default=" << def_ms
                  << " batch2=" << b2_ms << " fast(attn+batch2)=" << fast_ms << "\n";
        std::cout << "[test_diffusion_batch2] BENCH speedup vs default: batch2 x"
                  << (b2_ms  > 0.0 ? def_ms / b2_ms  : 0.0) << "  fast x"
                  << (fast_ms > 0.0 ? def_ms / fast_ms : 0.0) << "\n";
    }

    return ok ? 0 : 1;
}

#endif // HAVE_CUDA

} // namespace dollama

int main()
{
#ifndef HAVE_CUDA
    std::cout << "[test_diffusion_batch2] [SKIP] HAVE_CUDA undefined\n";
    return 0;
#else
    try
    {
        const int rc = dollama::run_test();
        if (rc == 0) { std::cout << "[test_diffusion_batch2] ALL PASSED\n"; }
        else         { std::cerr << "[test_diffusion_batch2] FAILED\n"; }
        return rc;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[test_diffusion_batch2] exception: " << e.what() << "\n";
        return 1;
    }
#endif
}

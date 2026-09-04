// FAST モード G-2k S3c / G-4k S3: CFG batch2 + epilogue 融合のパリティゲート (guidance=1.0 判定)。
//   generate_txt2img (CFG ループ全体) を構成違いで走らせ、最終 RGB を突合する。
//
//   batch2 は「cond/uncond を逐次 2 forward」を「B=2 で 1 forward」へ束ねるだけの数学的同一変換。
//   ただし束ねると linear の GEMM が M=tokens → M=B*tokens に変わり cuBLAS のタイル選択=蓄積順が
//   変わるため、FP16 蓄積で ~1 ULP/要素の差が出る (S2 で per-sample MAE 6.4e-5 と実証済み・ビット
//   一致には非到達)。この微小差は CFG 合成 noise = uncond + g*(cond-uncond) で **guidance 倍に増幅**
//   され step ごとに累積 → 最終 RGB の発散量は guidance に単調 (実測 g=1:SSIM0.9995 / g=3:0.9988 /
//   g=7.5:0.9966・平均差 <=0.3/255 で絵は実質同一)。ノイズ床 (off↔off 同一構成) は完全 bit-exact。
//
//   epilogue (G-4k) も同型の性質を持つ: multi-block GroupNorm は reduction 順が変わるため 1step
//   noise_pred で MAE 6.6e-5 (batch2 の 6.4e-5 と同オーダー) を出し、同じく CFG で guidance 倍に
//   増幅されて 20step 累積する。よって **epilogue も guidance=1.0 で判定する**。
//
//   → **ゲート設計 (ユーザー決裁 2026-07-09 / G-4k S3 で epilogue へ拡張)**: CFG 増幅を排して
//     被験変数の数学的等価性だけを判定するため **guidance=1.0 の最終 RGB SSIM >= 0.999 をハード
//     ゲート**にする (計画書 `--fast` の RGB 基準)。g=1.0 は noise = cond で CFG 増幅ゼロゆえ
//     被験変数の tiling/reduction 差そのものだけが出る。realistic な g=3/7.5 は characterization
//     として出力のみ (増幅の単調性を可視化・regression 監視の soft ゲート)。
//
//   ハードゲート 4 本 (いずれも steps=4 の parity 節 = meson test で自動実行):
//     [1] ノイズ床    : off@g=1.0 二連走が bit-exact (default 経路の determinism)
//     [2] batch2 ゲート: batch2 vs off @ g=1.0 SSIM >= 0.999 (attn_fast は両辺 false)
//     [3] epi determinism: fast+epi@g=1.0 二連走が bit-exact
//     [4] epi ゲート  : fast+epi vs fast @ g=1.0 SSIM >= 0.999
//
//   G-10k T2b で 1 本追加 (条件付き・DOLLAMA_PROFILE=1 のときだけ判定):
//     [5] reset ゲート: generate_txt2img の profile カウンタが呼び出しごとにリセットされる。
//         (a) 主検査 = 番兵: 生成直前に 1e6 s を注入し、生成後に消えていること。
//         (b) 副検査 = 積み上がり比: 同一パイプラインの 1 回目 (g=1.0) 直後と、同一構成の
//             再走 (g1_again) 直後の cat_resnet_sec がほぼ等しいこと。リセットが無ければ
//             後者は前者の ~4 倍 (間に g=3.0 / g=7.5 の 2 走が挟まる) に積み上がる。
//         ★(b) 単独では検出力が位置依存になる (負のコントロールの実測でプロセス 4 本目の
//           構成が閾値を下回った)。経緯は run_test 内 ResnetProbe のコメント。
//         cat_resnet_sec は profile_enabled() の
//         ときしか加算されない (src/infer/unet.cu:541-542) ので、DOLLAMA_PROFILE
//         未設定の meson test 既定実行では [SKIP] を印字するだけで判定しない。
//         ★被験は generate_txt2img 側に新設したリセットである。generate (別メソッド) は
//           元からリセットを持つので、そちらを 2 回呼んでも被験を一度も通らない。
//
//   [4] の比較対象を **fast+epi vs fast** にしたのは、両辺に attn_fast+batch2 が等しく載るので
//   **差分が epilogue のみに分離される**ため (fast+epi vs default だと batch2 の CFG 増幅が混入し、
//   epilogue が完全に正しくてもゲートが落ちる = 被験変数を測っていない)。`batch2+epi vs batch2` は
//   純度が上がるが出荷構成 (`--fast` = attn+batch2+epi) から遠のくので採らない (attn_fast は両辺で相殺)。
//
//   [3] は **determinism 検査であって「無改変証明」ではない**。二連走は同一バイナリ内の再実行ゆえ、
//   epilogue を壊しても壊れた同じコードが 2 回走って bit 一致する。GN multi-block は 2 段決定的集約
//   (atomic 禁止) で設計上決定的だが、その設計上の主張をゲートで裏取りするのが目的。default 経路の
//   真の無改変担保は test_unet_fast の **default vs 保存済み golden** (SSIM 0.999996) 側にある。
//
//   VRAM 二重常駐回避: 各パイプライン (UNet 5.1GB + VAE) は scope を抜けて破棄してから次を構築する
//   (重みの再ロードも構成ごと 1 回に抑える・generate_txt2img は guidance を引数で受ける)。
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
#include "infer/profile.cuh"
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
    // クラッシュ時にも進捗ログが残るよう毎出力で flush する (バッファ喪失で診断不能になるのを防ぐ)。
    std::cout << std::unitbuf;
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

    // G-10k T2b [GATE5] 用のプローブ: 既存の g1_again 二連走に相乗りして
    //   「1 回目直後」と「再走直後」の cat_resnet_sec を採る (追加の生成は行わない)。
    //
    // ★検出力について (負のコントロール実測で判明した弱点と、その手当て):
    //   「積み上がり比 again/first」だけを見る設計は、プロセス後半の構成では検出力が落ちる。
    //   リセットが無いとカウンタは構成をまたいで通算で積み上がるので、n 走目の構成では
    //   again/first が (n+3)/n に縮むためである。実測 (T2b の負のコントロール・リセット無し):
    //   1 本目の off = 3.97 で赤くなる一方、4 本目の fast+epi は 1.24 で閾値 1.5 を下回り
    //   **単独では検出できなかった**。
    //   → そこで各プローブの生成直前に「実測値としてあり得ない値」を注入し、
    //     generate_txt2img がそれを消していることを直接検査する。これは比にも
    //     プロセス内の位置にも依存しない。リセットが効いていれば毒は生成の冒頭で消え、
    //     印字される内訳にも残らない。
    constexpr double kResetSentinel = 1.0e6;  // 秒。実測 (0.1〜3s 級) とは桁が違う番兵。
    struct ResnetProbe
    {
        double first = -1.0;  // k=0 (g=1.0) の generate_txt2img 直後
        double again = -1.0;  // g1_again (同一構成・同一 g) の generate_txt2img 直後
        int    gens  = 0;     // first と again の間に挟まった generate_txt2img 回数を含む総数
        bool   used  = false;
    };

    // 構成を FastConfig で直接指定してパイプラインを 1 本構築し、全 guidance を回す。
    //   fast=false のまま attn_fast/batch2/epilogue を直に立てる (コンストラクタの fast 含意は
    //   fast=true のときだけ働くので、ここで立てた値がそのまま使われる = 構成を厳密に制御できる)。
    auto run_config = [&](bool attn_fast, bool batch2, bool epilogue,
                          std::vector<uint8_t>* rgb_out, std::vector<uint8_t>* g1_again,
                          int& w_out, int& h_out, ResnetProbe* probe = nullptr)
    {
        FastConfig cfg;
        cfg.attn_fast = attn_fast;
        cfg.batch2    = batch2;
        cfg.epilogue  = epilogue;
        DiffusionPipeline pipe(unet_w, vae_w, iopath, cfg);
        int ngen = 0;
        // 番兵の注入。profile 無効時はカウンタが誰にも読まれず reset もされないので入れない。
        auto poison = [&]()
        {
            if (probe != nullptr && profile_enabled())
            {
                profile_counters().cat_resnet_sec += kResetSentinel;
            }
        };
        for (int k = 0; k < nG; ++k)
        {
            int ww = 0, hh = 0;
            if (k == 0) { poison(); }
            gen(pipe, guids[k], rgb_out[k], ww, hh);
            ++ngen;
            if (k == 0) { w_out = ww; h_out = hh; }
            // 1 回目の直後の累積値を採る (dump 側でリセットが効いていれば「1 回分」)。
            if (k == 0 && probe != nullptr) { probe->first = profile_counters().cat_resnet_sec; }
        }
        if (g1_again != nullptr)
        {
            int ww = 0, hh = 0;
            poison();
            gen(pipe, guids[0], *g1_again, ww, hh);  // 同一構成 same seed = bit-exact 期待
            ++ngen;
            if (probe != nullptr)
            {
                probe->again = profile_counters().cat_resnet_sec;
                probe->gens  = ngen;
                probe->used  = true;
            }
        }
    }; // scope を抜けて pipe 破棄 → VRAM 解放してから次構成を構築。

    // --- (1) default 相当 (全 false)。g=1.0 は determinism 制御用に 2 回走らせる ---
    std::vector<uint8_t> off[3];
    std::vector<uint8_t> off_g1_again;   // ノイズ床制御 (off@1.0 の 2 回目)
    ResnetProbe          off_probe;      // G-10k T2b [GATE5]
    int w = 0, h = 0;
    run_config(false, false, false, off, &off_g1_again, w, h, &off_probe);

    // --- (2) batch2 単独 (attn_fast は off 側と同値 = false 固定) ---
    std::vector<uint8_t> on[3];
    int w2 = 0, h2 = 0;
    run_config(false, true, false, on, nullptr, w2, h2);

    // --- (3) fast 相当 = attn_fast + batch2 (epilogue なし)。epi ゲート [4] の基準辺 ---
    std::vector<uint8_t> fst[3];
    int w3 = 0, h3 = 0;
    run_config(true, true, false, fst, nullptr, w3, h3);

    // --- (4) fast + epilogue = 出荷正典 (`--fast`)。g=1.0 は determinism 用に 2 回 ---
    std::vector<uint8_t> epi[3];
    std::vector<uint8_t> epi_g1_again;
    ResnetProbe          epi_probe;      // G-10k T2b [GATE5]
    int w4 = 0, h4 = 0;
    run_config(true, true, true, epi, &epi_g1_again, w4, h4, &epi_probe);

    {
        size_t freeb = 0, totalb = 0;
        CUDA_CHECK(cudaMemGetInfo(&freeb, &totalb));
        std::cout << "[test_diffusion_batch2] VRAM free=" << (freeb >> 20)
                  << "MB (after all parity pipelines destroyed)\n";
    }

    bool ok = true;

    // --- 形状一致検査 ---
    if (w != 1024 || h != 1024 || w2 != 1024 || h2 != 1024
        || w3 != 1024 || h3 != 1024 || w4 != 1024 || h4 != 1024)
    {
        std::cerr << "[test_diffusion_batch2] FAIL: resolution off=" << w << "x" << h
                  << " batch2=" << w2 << "x" << h2
                  << " fast=" << w3 << "x" << h3
                  << " fast+epi=" << w4 << "x" << h4 << " != 1024x1024\n";
        ok = false;
    }
    constexpr size_t kRgbN = static_cast<size_t>(1024) * 1024 * 3;
    for (int k = 0; k < nG && ok; ++k)
    {
        if (off[k].size() != kRgbN || on[k].size() != kRgbN
            || fst[k].size() != kRgbN || epi[k].size() != kRgbN)
        {
            std::cerr << "[test_diffusion_batch2] FAIL: rgb size g=" << guids[k]
                      << " off=" << off[k].size() << " batch2=" << on[k].size()
                      << " fast=" << fst[k].size() << " fast+epi=" << epi[k].size() << "\n";
            ok = false;
        }
    }

    if (!ok)
    {
        std::cerr << "[test_diffusion_batch2] FAILED (shape)\n";
        return 1;
    }

    // --- [1] ノイズ床制御: off@g=1.0 の 2 回が bit-exact (default 経路の determinism) ---
    {
        const DiffStat d = rgb_diff(off[0], off_g1_again);
        std::cout << "[test_diffusion_batch2] [GATE1] noise floor (off@g=1.0 x2): MAE=" << d.mae
                  << " max_abs=" << d.max_abs << "\n";
        if (d.mae != 0.0 || d.max_abs != 0)
        {
            std::cerr << "[test_diffusion_batch2] FAIL: noise floor not bit-exact "
                         "(default 経路が非決定的 = 発散源が被験変数に切り分けられない)\n";
            ok = false;
        }
    }

    // --- [2] batch2 ゲート: batch2 vs off。g=1.0 がハード・g=3/7.5 は characterization ---
    double ssim_g1 = -1.0;
    for (int k = 0; k < nG; ++k)
    {
        double mae = 0.0;
        int    mx  = 0;
        const double ssim = compare_rgb(off[k], on[k], mae, mx);
        const char* tag = (k == 0) ? " [GATE2 >=0.999]" : " (characterization)";
        std::cout << "[test_diffusion_batch2] sweep g=" << guids[k]
                  << " batch2-vs-off: MAE=" << mae << " max_abs=" << mx
                  << " SSIM=" << ssim << tag << "\n";
        if (mae == 0.0)
        {
            std::cout << "[test_diffusion_batch2]   bit-exact @ g=" << guids[k] << "\n";
        }
        if (k == 0) { ssim_g1 = ssim; }
    }
    if (ssim_g1 < 0.999)
    {
        std::cerr << "[test_diffusion_batch2] FAIL: [GATE2] batch2 SSIM(g=1.0) " << ssim_g1
                  << " < 0.999\n";
        ok = false;
    }

    // --- [3] epi determinism: fast+epi@g=1.0 の 2 回が bit-exact ---
    //   注意: これは **determinism 検査であって無改変証明ではない** (二連走は同一バイナリ内の
    //   再実行ゆえ、epilogue を壊しても壊れた同じコードが 2 回走って bit 一致する)。GN multi-block
    //   の 2 段決定的集約 (atomic 禁止) という設計上の主張をゲートで裏取りするのが目的。
    {
        const DiffStat d = rgb_diff(epi[0], epi_g1_again);
        std::cout << "[test_diffusion_batch2] [GATE3] epi determinism (fast+epi@g=1.0 x2): MAE="
                  << d.mae << " max_abs=" << d.max_abs << "\n";
        if (d.mae != 0.0 || d.max_abs != 0)
        {
            std::cerr << "[test_diffusion_batch2] FAIL: fast+epi 経路が非決定的 "
                         "(GN multi-block の決定的集約が崩れている疑い)\n";
            ok = false;
        }
    }

    // --- [4] epi ゲート: fast+epi vs fast (両辺 attn_fast+batch2 = 差分は epilogue のみ) ---
    double ssim_epi_g1 = -1.0;
    for (int k = 0; k < nG; ++k)
    {
        double mae = 0.0;
        int    mx  = 0;
        const double ssim = compare_rgb(fst[k], epi[k], mae, mx);
        const char* tag = (k == 0) ? " [GATE4 >=0.999]" : " (characterization)";
        std::cout << "[test_diffusion_batch2] sweep g=" << guids[k]
                  << " (fast+epi)-vs-fast: MAE=" << mae << " max_abs=" << mx
                  << " SSIM=" << ssim << tag << "\n";
        if (mae == 0.0)
        {
            std::cout << "[test_diffusion_batch2]   bit-exact @ g=" << guids[k] << "\n";
        }
        if (k == 0) { ssim_epi_g1 = ssim; }
    }
    if (ssim_epi_g1 < 0.999)
    {
        std::cerr << "[test_diffusion_batch2] FAIL: [GATE4] epilogue SSIM(g=1.0) "
                  << ssim_epi_g1 << " < 0.999\n";
        ok = false;
    }

    // --- [5] G-10k T2b: generate_txt2img の profile カウンタ reset ゲート ---
    //   被験 = generate_txt2img (src/infer/diffusion.cu) に新設した profile_counters().reset()。
    //   検査は 2 本立て:
    //     (a) 番兵: 生成直前に注入した kResetSentinel (1e6 s) が生成後に残っていないこと。
    //         これが主検査。比にもプロセス内の位置にも依存しないので、構成順を入れ替えても
    //         検出力が変わらない (上の ResnetProbe のコメントに実測の弱点を記録した)。
    //     (b) 積み上がり比: 同一パイプライン内の 1 回目直後と再走直後がほぼ等しいこと。
    //         リセットが無ければ 4 走ぶん (g=1.0 / 3.0 / 7.5 / g1_again) 積み上がる。
    //         閾値 1.5 は「1 走ぶんの走行間ばらつき」より十分上・「積み上がり」より下。
    //   cat_resnet_sec は profile_enabled() のときだけ加算される (src/infer/unet.cu:541-542) ので、
    //   DOLLAMA_PROFILE 未設定 (= meson test の既定) では判定せず [SKIP] を出す。
    {
        const bool prof_on = profile_enabled();
        if (!prof_on)
        {
            std::cout << "[test_diffusion_batch2] [GATE5] SKIP (DOLLAMA_PROFILE unset: "
                         "cat_resnet_sec is only accumulated while profiling is enabled)\n";
        }
        else
        {
            auto check_reset = [&](const char* tag, const ResnetProbe& p)
            {
                if (!p.used || p.first <= 0.0 || p.again <= 0.0)
                {
                    std::cerr << "[test_diffusion_batch2] FAIL: [GATE5] " << tag
                              << " cat_resnet_sec not accumulating (first=" << p.first
                              << " again=" << p.again << " used=" << (p.used ? 1 : 0)
                              << "): the probe never reached the code under test\n";
                    ok = false;
                    return;
                }
                const double ratio        = p.again / p.first;
                const bool   sentinel_ok  = (p.first < kResetSentinel) && (p.again < kResetSentinel);
                std::cout << "[test_diffusion_batch2] [GATE5] " << tag
                          << " cat_resnet_sec: run1=" << p.first << "s  rerun=" << p.again
                          << "s  ratio=" << ratio << "  (gens in pipeline=" << p.gens
                          << ", sentinel " << (sentinel_ok ? "cleared" : "SURVIVED")
                          << ", required: sentinel cleared and ratio<1.5)\n";
                if (!sentinel_ok)
                {
                    std::cerr << "[test_diffusion_batch2] FAIL: [GATE5] " << tag
                              << " sentinel (1e6 s) survived the call: generate_txt2img"
                                 " does not reset the profile counters\n";
                    ok = false;
                }
                if (!(ratio < 1.5))
                {
                    std::cerr << "[test_diffusion_batch2] FAIL: [GATE5] " << tag
                              << " ratio " << ratio
                              << " >= 1.5: generate_txt2img does not reset the profile"
                                 " counters (values pile up across calls)\n";
                    ok = false;
                }
            };
            check_reset("off(default)", off_probe);
            check_reset("fast+epi", epi_probe);
        }
    }

    if (ok)
    {
        std::cout << "[test_diffusion_batch2] parity PASSED (GATE2 batch2 g=1.0 SSIM="
                  << ssim_g1 << " / GATE4 epilogue g=1.0 SSIM=" << ssim_epi_g1
                  << " >= 0.999・GATE1/GATE3 determinism bit-exact)\n";
    }

    // --- DB2_BENCH: batch2 / epilogue の e2e 速度を warm 実測 (速度計測の本務に純化) ---
    //   generate_txt2img 全体 (CFG UNet ループ + VAE decode) の wall-clock を計測する。
    //   VAE decode は共通 (~1.2s) ゆえ speedup は UNet CFG ループ差で希釈されるが、
    //   これが正典 CFG e2e ベースライン (計画書 line52) = 出荷判定 (diffusers 3.8s 比) の分母。
    //   3 構成を min 実測: default / attn+batch2 (合成構成) / fast+epilogue (= 出荷正典 `--fast`)。
    //   meson test では非設定 → parity のみで高速。手動 DB2_BENCH=1 で計測モードに入る。
    //
    //   **ハードゲートはここには無い** (G-4k S3 で parity 節 [1]〜[4] へ移設済み)。理由:
    //   bench は出荷相当 guidance=7.5 で回すため CFG 増幅が乗り、被験変数 (epilogue) 単独の
    //   等価性を測れない (batch2 単独ですら g=7.5 では SSIM 0.9966 まで落ちると G-2k で実測済み)。
    //   ここの SSIM 2 本は characterization (合否なし) = 出荷構成の実発散量の記録。
    //   速度にもハード合否を課さない (resnet <= 0.95s は G-4k スコープ外確定・ログ監視のみ)。
    if (ok && std::getenv("DB2_BENCH"))
    {
        int bsteps = 20;                          // 出荷相当 20step。
        int iters  = 2;                           // 本計測 (min を採用)。
        float bg   = 7.5f;                        // 出荷相当 guidance (DB2_BENCH_G で可変)。
        if (const char* e = std::getenv("DB2_BENCH_STEPS")) { bsteps = std::atoi(e); }
        if (const char* e = std::getenv("DB2_BENCH_ITERS")) { iters  = std::atoi(e); }
        if (const char* e = std::getenv("DB2_BENCH_G"))     { bg     = static_cast<float>(std::atof(e)); }
        std::cout << "[test_diffusion_batch2] BENCH steps=" << bsteps << " guidance=" << bg
                  << " iters=" << iters << " (warmup1・min 採用・VAE 共通込み e2e)\n";

        // 1 パイプラインを warmup 1 回 + iters 回計測し最小 ms を返す。
        // 最終 RGB を out_rgb へ回収する (パリティ突合用・決定的ゆえ最終走の出力で足りる)。
        auto bench_pipe = [&](bool attn_fast, bool batch2, bool epilogue,
                              std::vector<uint8_t>& out_rgb) -> double
        {
            FastConfig cfg;
            cfg.attn_fast = attn_fast;
            cfg.batch2    = batch2;
            cfg.epilogue  = epilogue;
            DiffusionPipeline pipe(unet_w, vae_w, iopath, cfg);
            int ww = 0, hh = 0;
            auto one = [&]()
            {
                pipe.generate_txt2img(bsteps, seed, bg,
                                      cond_ehs.data(), cond_txt.data(),
                                      uncond_ehs.data(), uncond_txt.data(),
                                      time_ids.data(), out_rgb, ww, hh);
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

        // fast(attn+batch2) は S3 以降 **CLI から到達できない合成構成** (`--fast` は epilogue を
        // 含意するようになったため)。epilogue の寄与を分離して見るための計測専用の中間点。
        std::vector<uint8_t> def_rgb, fast_rgb, epi_rgb;
        const double def_ms  = bench_pipe(false, false, false, def_rgb);   // default (逐次 2 forward)
        const double fast_ms = bench_pipe(true,  true,  false, fast_rgb);  // attn+batch2 (合成・非 CLI)
        const double epi_ms  = bench_pipe(true,  true,  true,  epi_rgb);   // fast+epilogue (= 出荷 `--fast`)

        std::cout << "[test_diffusion_batch2] BENCH e2e (min ms): default=" << def_ms
                  << " attn+batch2(composite / not CLI-reachable)=" << fast_ms
                  << " fast+epilogue(shipping `--fast`)=" << epi_ms << "\n";
        std::cout << "[test_diffusion_batch2] BENCH speedup vs default: attn+batch2 x"
                  << (fast_ms > 0.0 ? def_ms / fast_ms : 0.0)
                  << "  fast+epi(shipping) x"
                  << (epi_ms  > 0.0 ? def_ms / epi_ms  : 0.0)
                  << "  (速度はハード合否を課さない・監視のみ)\n";

        // --- characterization: fast+epi vs attn+batch2 (差分は epilogue のみ・g=bg の実発散量) ---
        {
            double mae = 0.0;
            int    mx  = 0;
            const double ssim = compare_rgb(fast_rgb, epi_rgb, mae, mx);
            std::cout << "[test_diffusion_batch2] BENCH (fast+epi) vs attn+batch2 @g=" << bg
                      << ": MAE=" << mae << " max_abs=" << mx << " SSIM=" << ssim
                      << " (characterization・合否なし / ハード判定は parity 節 GATE4 @g=1.0)\n";
        }

        // --- characterization: fast+epi vs default (出荷構成の対 default 総発散量・CFG 増幅込み) ---
        {
            double mae = 0.0;
            int    mx  = 0;
            const double ssim = compare_rgb(def_rgb, epi_rgb, mae, mx);
            std::cout << "[test_diffusion_batch2] BENCH (fast+epi) vs default @g=" << bg
                      << ": MAE=" << mae << " max_abs=" << mx << " SSIM=" << ssim
                      << " (characterization・合否なし / batch2+epilogue の CFG 増幅込み総和)\n";
        }
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

// SDXL 拡散ループ結線 (DiffusionPipeline) テスト + フルベンチ
// (Phase 2 マイルストーン 2-6a / ST-1+ST-2)
//
// HAVE_CUDA 未定義時は [SKIP] で return 0。golden/重み不在時も [SKIP]。
//
// 数値 smoke (緑/赤 CI 判定):
//   steps=2 で generate() を実行し
//     (a) 出力 RGB に NaN/Inf 由来の異常がない (uint8 なので範囲外は構造上出ないが
//         パイプライン途中の NaN は最終的に飽和してしまうため、全黒/全白に張り付いて
//         いないことを (c) 分散で間接検査する)
//     (b) 全ピクセルが [0,255] (uint8 なので自明だが念のため検査)
//     (c) 出力が非定数 (分散 > 小閾値)
//   を確認する (~20-30s 想定)。
//
// フルベンチ (計測のみ・合否にしない):
//   steps=20 を 1 回、総時間を cudaEvent で計測し probe10 3.80s と並べてログ。
//   環境変数 DOLLAMA_BENCH があるときのみ実行 (デフォルト CI はスキップ)。
//
// ゴールデンパス UNET_WEIGHTS_PATH / VAE_WEIGHTS_PATH / UNET_IO_PATH は
// meson cuda_args で -D 埋め込み (cwd 非依存)。

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#include "infer/diffusion.cuh"
#include "kernels/utils.cuh"
#endif

namespace dollama
{

#ifdef HAVE_CUDA

// uint8 RGB 出力の統計を取る。
struct RgbStats
{
    double mean   = 0.0;
    double var    = 0.0;
    int    mn     = 256;
    int    mx     = -1;
    long   out_of_range = 0;  // [0,255] 外 (構造上 0 のはずだが念のため)
};

static RgbStats rgb_stats(const std::vector<uint8_t>& rgb)
{
    RgbStats s;
    double sum = 0.0, sum2 = 0.0;
    for (uint8_t b : rgb)
    {
        const int v = static_cast<int>(b);
        sum  += v;
        sum2 += static_cast<double>(v) * v;
        s.mn = std::min(s.mn, v);
        s.mx = std::max(s.mx, v);
        if (v < 0 || v > 255) { ++s.out_of_range; }
    }
    const double n = static_cast<double>(rgb.size());
    s.mean = sum / n;
    s.var  = sum2 / n - s.mean * s.mean;
    return s;
}

static int run_test()
{
    const std::string unet_w  = UNET_WEIGHTS_PATH;
    const std::string vae_w   = VAE_WEIGHTS_PATH;
    const std::string embeds  = UNET_IO_PATH;

    std::cout << "[test_diffusion] unet_weights=" << unet_w << "\n";
    std::cout << "[test_diffusion] vae_weights=" << vae_w << "\n";
    std::cout << "[test_diffusion] embeds=" << embeds << "\n";

    // ゴールデン/重み不在時は [SKIP] (CLIP/VAE/UNet テストと同様。GitHub 100MB 制限のため
    // 重みはリポジトリに含めない。scripts の dump スクリプトで再生成する)。
    {
        std::ifstream fu(unet_w, std::ios::binary);
        std::ifstream fv(vae_w, std::ios::binary);
        std::ifstream fe(embeds, std::ios::binary);
        if (!fu.good() || !fv.good() || !fe.good())
        {
            std::cout << "[test_diffusion] [SKIP] golden/weights not found "
                         "(unet_weights / vae_weights / unet_io safetensors)\n";
            return 0;
        }
    }

    // パイプライン構築 (重み 2 つ + 埋め込みを 1 回だけロード)。
    // unique_ptr で保持する (レビュー是正②: 後段の vae_scaling_factor 検証で
    // 複数の DiffusionPipeline を構築するため、UNet 5.1GB の多重常駐を避けるべく
    // ここで一旦 reset() して手放せるようにする)。
    auto pipe = std::make_unique<DiffusionPipeline>(unet_w, vae_w, embeds);

    {
        size_t freeb = 0, totalb = 0;
        CUDA_CHECK(cudaMemGetInfo(&freeb, &totalb));
        std::cout << "[test_diffusion] VRAM free=" << (freeb >> 20) << "MB / total="
                  << (totalb >> 20) << "MB (after weight load)\n";
    }

    bool ok = true;

    // ============ 数値 smoke (steps=2) ============
    std::cout << "[test_diffusion] --- smoke (steps=2) ---\n";
    std::vector<uint8_t> rgb;
    int w = 0, h = 0;
    pipe->generate(/*steps=*/2, /*seed=*/1234ULL, rgb, w, h);

    if (w != 1024 || h != 1024)
    {
        std::cerr << "[test_diffusion] FAIL: resolution " << w << "x" << h << " != 1024x1024\n";
        ok = false;
    }
    if (rgb.size() != static_cast<size_t>(1024) * 1024 * 3)
    {
        std::cerr << "[test_diffusion] FAIL: rgb size " << rgb.size() << " != 1024*1024*3\n";
        ok = false;
    }

    const RgbStats st = rgb_stats(rgb);
    std::cout << "[test_diffusion] rgb mean=" << st.mean << " var=" << st.var
              << " min=" << st.mn << " max=" << st.mx
              << " out_of_range=" << st.out_of_range << "\n";

    // (b) 全ピクセル [0,255]
    if (st.out_of_range != 0)
    {
        std::cerr << "[test_diffusion] FAIL: " << st.out_of_range << " pixels out of [0,255]\n";
        ok = false;
    }
    // (a)+(c) 非定数 (NaN/Inf 飽和で全黒/全白に張り付くと var≈0)。閾値は十分小さく。
    if (st.var < 1.0)
    {
        std::cerr << "[test_diffusion] FAIL: output nearly constant (var=" << st.var
                  << " < 1.0) - likely NaN/Inf saturation\n";
        ok = false;
    }
    if (ok)
    {
        std::cout << "[test_diffusion] smoke PASSED\n";
    }

    // ============ E-0: vae_scaling_factor 検証 (fallback 発火 + 実除算) ============
    // レビュー指摘②: DiffusionPipeline ctor に追加した vae_scaling_factor の検証
    // (<=0 / 非有限値でフォールバック + [warn]) と、非既定値での VAE decode 前除算の
    // 実行経路は、計測専用 exe (prof_e0_vae_scaling, meson test 未登録) でしか通って
    // いなかった。meson test に登録される本テストへ以下を追加する:
    //   (1) 不正値 (0 以下 / 非有限) を渡すと既定 kDefaultVaeScalingFactor へフォール
    //       バックし、既定 3 引数 ctor (= 上の smoke と同一 seed/steps) と bit-exact に
    //       なることを確認する (= フォールバックが実際に発火した直接証拠)。
    //   (2) 有効な非既定値 (0.18215f = illustrious-xl preset.json の実値) を渡すと、
    //       既定値のときと出力が異なることを確認する (= 非既定値が実際に VAE decode 前
    //       除算に使われている直接証拠。bit-exact なら configuration が無視されている
    //       ことになるので FAIL とする)。
    // 各ケースは順にスコープ内で構築・破棄する (UNet 5.1GB を同時多重常駐させない)。
    // smoke 用の `pipe` はここで reset() して手放す (16GB 板で 2 インスタンス分の
    // 常駐 (UNet 5.1GB×2 + 共有アリーナ予約 ~6GB) は VRAM を圧迫しうるため)。
    pipe.reset();
    std::cout << "[test_diffusion] --- vae_scaling_factor fallback/実除算検証 ---\n";
    {
        // (1) 不正値 → フォールバックし、smoke の出力 (既定値) と bit-exact になるべき。
        const float invalid_values[] = {
            -1.0f, 0.0f, std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::quiet_NaN(),
        };
        for (float bad : invalid_values)
        {
            std::vector<uint8_t> rgb_fb;
            int wf = 0, hf = 0;
            {
                DiffusionPipeline pipe_fb(unet_w, vae_w, embeds, FastConfig{}, bad);
                pipe_fb.generate(/*steps=*/2, /*seed=*/1234ULL, rgb_fb, wf, hf);
            }
            if (rgb_fb != rgb)
            {
                // 急所 (nvcc/cudafe の既知不具合・diffusion.cu 側と同じ理由で ASCII のみに
                // している): 末尾が日本語の複数演算子チェーンで "missing closing quote" /
                // 実行時にリテラル "\n" が混入する再現性のある不具合を実機で確認した。
                std::cerr << "[test_diffusion] FAIL: vae_scaling_factor=" << bad
                          << " (invalid) did not fall back to the default (smoke) output"
                          << std::endl;
                ok = false;
            }
            else
            {
                std::cout << "[test_diffusion] vae_scaling_factor=" << bad
                          << " -> fallback bit-exact OK" << std::endl;
            }
        }

        // (2) 有効な非既定値 → 実際に除算へ反映され、既定値の出力と異なるべき。
        {
            std::vector<uint8_t> rgb_nd;
            int wn = 0, hn = 0;
            {
                DiffusionPipeline pipe_nd(unet_w, vae_w, embeds, FastConfig{}, 0.18215f);
                pipe_nd.generate(/*steps=*/2, /*seed=*/1234ULL, rgb_nd, wn, hn);
            }
            if (rgb_nd == rgb)
            {
                std::cerr << "[test_diffusion] FAIL: vae_scaling_factor=0.18215 (non-default) "
                             "is bit-exact with the default output -- not actually used in the division"
                          << std::endl;
                ok = false;
            }
            else
            {
                std::cout << "[test_diffusion] vae_scaling_factor=0.18215 -> differs from default OK "
                             "(reflected in the actual division)" << std::endl;
            }
        }
    }

    // ============ フルベンチ (steps=20) — DOLLAMA_BENCH 時のみ ============
    if (std::getenv("DOLLAMA_BENCH") != nullptr)
    {
        std::cout << "[test_diffusion] --- full bench (steps=20) ---\n";
        // vae_scaling_factor 検証で reset() 済みのため再構築する (既定 3 引数 ctor)。
        pipe = std::make_unique<DiffusionPipeline>(unet_w, vae_w, embeds);
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        std::vector<uint8_t> rgb20;
        int w20 = 0, h20 = 0;

        CUDA_CHECK(cudaEventRecord(start));
        pipe->generate(/*steps=*/20, /*seed=*/1234ULL, rgb20, w20, h20);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        const double sec = ms / 1000.0;
        const RgbStats st20 = rgb_stats(rgb20);

        std::cout << "[test_diffusion] 20step total=" << sec << " s"
                  << "  (probe10=3.80s, ratio=" << (sec / 3.80) << "x slower)\n";
        std::cout << "[test_diffusion] 20step rgb mean=" << st20.mean
                  << " var=" << st20.var << " min=" << st20.mn << " max=" << st20.mx << "\n";

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    else
    {
        std::cout << "[test_diffusion] (full bench skipped - set DOLLAMA_BENCH=1 to run 20step)\n";
    }

    return ok ? 0 : 1;
}

#endif // HAVE_CUDA

} // namespace dollama

int main()
{
#ifndef HAVE_CUDA
    std::cout << "[test_diffusion] [SKIP] HAVE_CUDA undefined\n";
    return 0;
#else
    try
    {
        const int rc = dollama::run_test();
        if (rc == 0) { std::cout << "[test_diffusion] ALL PASSED\n"; }
        else         { std::cerr << "[test_diffusion] FAILED\n"; }
        return rc;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[test_diffusion] exception: " << e.what() << "\n";
        return 1;
    }
#endif
}

// SDXL VAE decoder ゴールデン突合テスト + レイテンシ計測 (Phase 2 マイルストーン 2-4)
// HAVE_CUDA 未定義時は [SKIP] で return 0。
//
// 進め方は段階導入だが、本テストは完成版 launch_vae_decode を 1 回走らせて
// 最終 image をゴールデン (final_image) と SSIM 突合する。中間段の突合は
// 開発時に内部フックで確認済み (報告参照)。ここでは:
//   - final_image SSIM >= 0.99 (本線ゴール)
//   - final_image の平均絶対誤差 / Inf・NaN 無し
//   - VAE decode 1 回のレイテンシ (cudaEvent 中央値)
// を検証・計測する。
//
// ゴールデンパス VAE_WEIGHTS_PATH / VAE_IO_PATH は meson cuda_args で -D 埋め込み (cwd 非依存)。

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#ifdef HAVE_CUDA
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "kernels/vae_decode.cuh"
#include "kernels/device_arena.cuh"
#include "kernels/utils.cuh"
#include "io/safetensors.hpp"
#endif

namespace dollama
{

#ifdef HAVE_CUDA

// ----------------------------------------------------------------
// SafeTensors から F32 テンソルを読み出して std::vector<float> に展開。
// ----------------------------------------------------------------
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

// SafeTensors から F16 テンソルを読み出して std::vector<__half> に展開。
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
// SSIM (簡易): 11x11 平均窓 (一様窓) で各画素の局所統計を取り、
//   SSIM = ((2*mu_x*mu_y + C1)(2*sigma_xy + C2)) /
//          ((mu_x^2+mu_y^2 + C1)(sigma_x^2+sigma_y^2 + C2))
// を全窓平均する。C1=(0.01*L)^2, C2=(0.03*L)^2、動的レンジ L は
// 両画像の値域から推定 (max-min)。チャネルごとに計算して平均。
// (ガウシアン窓ではなく一様窓の素朴版。配線の正しさ検証には十分。)
// 画像は [1,3,1024,1024] NCHW row-major。
// ----------------------------------------------------------------
static double ssim_uniform(const std::vector<float>& a, const std::vector<float>& b,
                           int C, int H, int W)
{
    // 動的レンジ L
    float lo = a[0], hi = a[0];
    for (float v : a)
    {
        lo = std::min(lo, v);
        hi = std::max(hi, v);
    }
    for (float v : b)
    {
        lo = std::min(lo, v);
        hi = std::max(hi, v);
    }
    const double L  = static_cast<double>(hi - lo);
    const double C1 = (0.01 * L) * (0.01 * L);
    const double C2 = (0.03 * L) * (0.03 * L);

    const int win   = 11;
    const int half  = win >> 1;
    const double inv = 1.0 / static_cast<double>(win * win);

    double ssim_sum = 0.0;
    long   win_cnt  = 0;

    for (int c = 0; c < C; ++c)
    {
        const long base = static_cast<long>(c) * H * W;
        // 窓は stride=half (重なりつつ間引き) で走らせ計算量を抑える。
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
                        const double va = a[idx];
                        const double vb = b[idx];
                        sa  += va;
                        sb  += vb;
                        saa += va * va;
                        sbb += vb * vb;
                        sab += va * vb;
                    }
                }
                const double mu_a = sa * inv;
                const double mu_b = sb * inv;
                const double va   = saa * inv - mu_a * mu_a;
                const double vb   = sbb * inv - mu_b * mu_b;
                const double cov  = sab * inv - mu_a * mu_b;
                const double s = ((2 * mu_a * mu_b + C1) * (2 * cov + C2)) /
                                 ((mu_a * mu_a + mu_b * mu_b + C1) * (va + vb + C2));
                ssim_sum += s;
                ++win_cnt;
            }
        }
    }
    return win_cnt > 0 ? ssim_sum / static_cast<double>(win_cnt) : 0.0;
}

static int run_test()
{
    const std::string wpath  = VAE_WEIGHTS_PATH;
    const std::string iopath = VAE_IO_PATH;

    std::cout << "[test_vae_decode] weights=" << wpath << "\n";
    std::cout << "[test_vae_decode] io=" << iopath << "\n";

    // ゴールデンデータ (vae_weights/vae_io.safetensors, 計 ~400MB) は GitHub の
    // 100MB 制限のためリポジトリに含めない (models/ と同様。.gitignore 済み)。
    // scripts/dollma_dump_vae_golden.py で再生成する。不在時は CLIP/WD14 と同じく [SKIP]。
    {
        std::ifstream fw(wpath, std::ios::binary), fio(iopath, std::ios::binary);
        if (!fw.good() || !fio.good())
        {
            std::cout << "[test_vae_decode] [SKIP] golden data not found "
                         "(run scripts/dollma_dump_vae_golden.py to generate)\n";
            return 0;
        }
    }

    SafeTensors weights(wpath);
    SafeTensors io(iopath);

    // 入力 latent (post-scale) を FP16 で取得。
    std::vector<__half> h_latent = load_f16(io, "input_latent_f16");
    const size_t latent_n = static_cast<size_t>(4) * 128 * 128;
    if (h_latent.size() != latent_n)
    {
        std::cerr << "[test_vae_decode] latent size mismatch: " << h_latent.size() << "\n";
        return 1;
    }

    // ゴールデン最終画像 (F32, [1,3,1024,1024])
    std::vector<float> golden = load_f32(io, "final_image_f32");
    const int C = 3, H = 1024, W = 1024;
    const size_t img_n = static_cast<size_t>(C) * H * W;
    if (golden.size() != img_n)
    {
        std::cerr << "[test_vae_decode] golden size mismatch: " << golden.size() << "\n";
        return 1;
    }

    // デバイスバッファ確保
    __half* d_latent = nullptr;
    __half* d_image  = nullptr;
    CUDA_CHECK(cudaMalloc(&d_latent, latent_n * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_image,  img_n * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_latent, h_latent.data(), latent_n * sizeof(__half),
                          cudaMemcpyHostToDevice));

    // 空き VRAM を確認
    {
        size_t freeb = 0, totalb = 0;
        CUDA_CHECK(cudaMemGetInfo(&freeb, &totalb));
        std::cout << "[test_vae_decode] VRAM free=" << (freeb >> 20) << "MB / total="
                  << (totalb >> 20) << "MB (before decode)\n";
    }

    // --- ウォームアップ + 正当性 ---
    launch_vae_decode(weights, d_latent, d_image);

    std::vector<__half> h_image(img_n);
    CUDA_CHECK(cudaMemcpy(h_image.data(), d_image, img_n * sizeof(__half),
                          cudaMemcpyDeviceToHost));

    // ------------------------------------------------------------
    // G-8k S3: アリーナ化した VAE 経路の bit-exact 検査。
    //   (a) 2 回目の decode が 1 回目と memcmp 一致すること (アリーナ再利用の決定性)。
    //   (b) env DOLLAMA_ARENA_RELEASE=1 のときは 1 回目と 2 回目の間でアリーナを
    //       release し、チャンクを返して再確保しても出力が変わらないことを見る。
    //   (c) env DOLLAMA_VAE_DUMP=<path> で 1 回目の FP16 出力を生バイトで落とす。
    //       DOLLAMA_POOL=0 / 既定 / RELEASE=1 の 3 プロセスを外から cmp して
    //       BIT-EXACT を確認するための出口 (env は初回キャッシュされるため、
    //       1 プロセス内でプール経路を混ぜられない)。
    // ------------------------------------------------------------
    {
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
        const char* rel  = std::getenv("DOLLAMA_ARENA_RELEASE");
        const char* dump = std::getenv("DOLLAMA_VAE_DUMP");
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
        const bool do_release = (rel != nullptr && std::strcmp(rel, "1") == 0);
        if (do_release)
        {
            device_arena_release(DeviceArenaId::UNet);
            device_arena_release(DeviceArenaId::UNetPersist);
        }

        // 再利用領域の残留値に依存していないことを炙り出すため、2 回目の前に
        // 出力バッファを毒値で潰しておく (全書きされない場所があれば差分になる)。
        CUDA_CHECK(cudaMemset(d_image, 0x5A, img_n * sizeof(__half)));

        // G-8k の完了条件そのもの: 定常状態の decode で実 cudaMalloc / cudaFree が
        // 0 回になること。チャンク列は「入りきらない要求が来たら挿入」で育つため、
        // 1 回目の decode だけでは育ち切らない (実測: 2 回目でも +1 本生える)。
        // 数回まで回して 0 回に収束することを見る。
        // プール経路のときだけ判定する (POOL=0 は素の cudaMalloc/cudaFree へ戻す
        // 経路なので 0 にならないのが正しく、release を挟む場合も再確保が走る)。
        int converged_at = -1;
        for (int it = 0; it < 6; ++it)
        {
            device_arena_reset_counters(DeviceArenaId::UNet);
            launch_vae_decode(weights, d_latent, d_image);
            const DeviceArenaStats sd = device_arena_stats(DeviceArenaId::UNet);
            std::cout << "[test_vae_decode] decode#" << (it + 2)
                      << ": cudaMalloc=" << sd.cuda_malloc_calls
                      << " cudaFree=" << sd.cuda_free_calls
                      << " alloc=" << sd.alloc_calls
                      << " cap=" << (sd.total_capacity >> 20) << "MiB\n";
            if (sd.cuda_malloc_calls == 0 && sd.cuda_free_calls == 0)
            {
                converged_at = it;
                break;
            }
        }
        std::cout << "[test_vae_decode] steady state reached at decode#"
                  << (converged_at >= 0 ? converged_at + 2 : -1) << "\n";
        if (device_arena_pool_enabled() && !do_release && converged_at < 0)
        {
            std::cerr << "[test_vae_decode] FAIL: decode still calls"
                         " cudaMalloc/cudaFree after 6 iterations\n";
            return 1;
        }

        std::vector<__half> h_image2(img_n);
        CUDA_CHECK(cudaMemcpy(h_image2.data(), d_image, img_n * sizeof(__half),
                              cudaMemcpyDeviceToHost));
        const bool same = (std::memcmp(h_image.data(), h_image2.data(),
                                       img_n * sizeof(__half)) == 0);
        std::cout << "[test_vae_decode] arena bit-exact (2nd decode"
                  << (do_release ? " after release" : "") << "): "
                  << (same ? "BIT-EXACT" : "MISMATCH") << "\n";
        if (!same)
        {
            std::cerr << "[test_vae_decode] FAIL: 2nd decode != 1st decode"
                         " (arena reuse changed the numerics)\n";
            return 1;
        }

        {
            const DeviceArenaStats su = device_arena_stats(DeviceArenaId::UNet);
            std::cout << "[test_vae_decode] [ALLOC] arena="
                      << device_arena_name(DeviceArenaId::UNet)
                      << " cap=" << (su.total_capacity >> 20) << "MiB"
                      << " live_peak=" << (su.peak_request_bytes >> 20) << "MiB"
                      << " cursor_peak=" << (su.peak_bytes_in_use >> 20) << "MiB"
                      << " chunks=" << su.live_chunks
                      << " cudaMalloc=" << su.cuda_malloc_calls
                      << " cudaFree=" << su.cuda_free_calls
                      << " alloc=" << su.alloc_calls << "\n";
        }

        if (dump != nullptr && dump[0] != '\0')
        {
            std::ofstream of(dump, std::ios::binary);
            of.write(reinterpret_cast<const char*>(h_image.data()),
                     (std::streamsize)(img_n * sizeof(__half)));
            std::cout << "[test_vae_decode] dumped FP16 image to " << dump << "\n";
        }
    }

    // FP16 -> FP32 デコード + Inf/NaN 検査
    std::vector<float> got(img_n);
    double sum_abs = 0.0;
    double max_abs = 0.0;
    long   bad     = 0;
    float  gmin = 1e30f, gmax = -1e30f;
    for (size_t i = 0; i < img_n; ++i)
    {
        const float v = __half2float(h_image[i]);
        if (std::isnan(v) || std::isinf(v))
        {
            ++bad;
        }
        got[i] = v;
        gmin = std::min(gmin, v);
        gmax = std::max(gmax, v);
        const double d = std::fabs(static_cast<double>(v) - golden[i]);
        sum_abs += d;
        max_abs = std::max(max_abs, d);
    }
    const double mae = sum_abs / static_cast<double>(img_n);

    std::cout << "[test_vae_decode] image range=[" << gmin << ", " << gmax << "] "
              << "golden range=[" ;
    {
        float lo = golden[0], hi = golden[0];
        for (float v : golden) { lo = std::min(lo, v); hi = std::max(hi, v); }
        std::cout << lo << ", " << hi << "]\n";
    }
    std::cout << "[test_vae_decode] MAE=" << mae << " max_abs=" << max_abs
              << " bad(Inf/NaN)=" << bad << "\n";

    const double ssim = ssim_uniform(got, golden, C, H, W);
    std::cout << "[test_vae_decode] SSIM=" << ssim << " (target >= 0.99)\n";

    bool ok = true;
    if (bad != 0)
    {
        std::cerr << "[test_vae_decode] FAIL: Inf/NaN detected\n";
        ok = false;
    }
    if (ssim < 0.99)
    {
        std::cerr << "[test_vae_decode] FAIL: SSIM " << ssim << " < 0.99\n";
        ok = false;
    }

    // --- レイテンシ計測 (cudaEvent 中央値, warmup3/iters7) ---
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    auto run_once = [&]() -> float
    {
        CUDA_CHECK(cudaEventRecord(start));
        launch_vae_decode(weights, d_latent, d_image);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    };
    const int warmup = 2;  // 上の正当性実行 + ここで計 3 回ウォーム
    const int iters  = 7;
    for (int i = 0; i < warmup; ++i)
    {
        (void)run_once();
    }
    std::vector<float> times;
    times.reserve(iters);
    for (int i = 0; i < iters; ++i)
    {
        times.push_back(run_once());
    }
    std::sort(times.begin(), times.end());
    const float median_ms = times[times.size() / 2];
    std::cout << "[test_vae_decode] decode latency median=" << median_ms << " ms"
              << " (min=" << times.front() << " max=" << times.back()
              << ", N=" << iters << ")\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_latent));
    CUDA_CHECK(cudaFree(d_image));

    return ok ? 0 : 1;
}

#endif // HAVE_CUDA

} // namespace dollama

int main()
{
#ifndef HAVE_CUDA
    std::cout << "[test_vae_decode] [SKIP] HAVE_CUDA undefined\n";
    return 0;
#else
    try
    {
        const int rc = dollama::run_test();
        if (rc == 0)
        {
            std::cout << "[test_vae_decode] ALL PASSED\n";
        }
        else
        {
            std::cerr << "[test_vae_decode] FAILED\n";
        }
        return rc;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[test_vae_decode] exception: " << e.what() << "\n";
        return 1;
    }
#endif
}

// Conv2d 単体テスト + ベンチ (Phase 2 マイルストーン 2-2-4 / 2-6 S3 / 2-6 S2 数値バグ修正)
// HAVE_CUDA 未定義時は [SKIP] で return 0。
//
// 入力ビット一致: 乱数を FP32 生成 → FP16 へ丸め、その FP16 をデコードした値を
// CPU 参照の入力にも使う (カーネルの数値誤差だけを測る)。test_groupnorm.cu の枠を流用。
//
// tol: 蓄積長 K = Cin*KH*KW に応じて atol/rtol を緩める (gemm / groupnorm と同じ
// 考え方)。FP16 入出力相応。K が大きいときは絶対誤差を併用 (atol が効く)。
//
// S3: 1x1=GEMM / 3x3=im2col+GEMM の新経路は Cout/HW/K が下限以上のときだけ起動するため、
// direct に落ちない「大きい形状」のケースを別途用意して GEMM 経路の正当性も検証する。
//
// 大形状の検証方針 (S2 数値バグ修正):
//   - CPU 参照 (cpu_conv2d) は 6 重ループ総当たりで、大形状 (up_block_2 concat 30億 MAC /
//     VAE 512² 39億 MAC) では数分かかり CI 非現実的。よって大形状は GPU の direct conv
//     (launch_conv2d_direct、ゴールデン緑で信頼できる) を参照に GPU 同士で突合する。
//     direct は GEMM 経路 (im2col+wmma / 帯分割) をバイパスするため、両者一致が
//     数値正当性の証明になる。CPU 参照は小形状ゴールデン専用に残す。
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#ifdef HAVE_CUDA
#include <cuda_fp16.h>
#include "kernels/conv2d.cuh"
#include "kernels/device_arena.cuh"
#include "kernels/gemm.cuh"   // G-10k T4: batched GEMM 分岐カウンタ (経路の実証に使う)
#include "kernels/utils.cuh"
#endif

namespace dollama
{

#ifdef HAVE_CUDA

// ----------------------------------------------------------------
// テスト用ヘルパー
// ----------------------------------------------------------------

// FP32 乱数 → FP16 丸め。デコード値 (FP32) も同時に返し CPU 参照と入力ビットを一致させる。
struct HalfBuffer
{
    std::vector<__half> h;   // デバイスへ送る FP16
    std::vector<float>  ref; // CPU 参照に使う「FP16 をデコードした値」
};

static HalfBuffer make_half(int n, unsigned seed, float lo = -1.0f, float hi = 1.0f)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    HalfBuffer b;
    b.h.resize(n);
    b.ref.resize(n);
    for (int i = 0; i < n; ++i)
    {
        const float v = dist(rng);
        const __half hv = __float2half(v);
        b.h[i]   = hv;
        b.ref[i] = __half2float(hv); // 丸め後の真値
    }
    return b;
}

// 出力サイズ計算 (カーネルと同一式)。
static int out_dim(int in, int pad, int dilation, int k, int stride)
{
    return (in + 2 * pad - dilation * (k - 1) - 1) / stride + 1;
}

// K (蓄積長 = Cin*KH*KW) に応じた許容誤差。
static float atol_for(int K)
{
    return 3e-3f + 2e-5f * static_cast<float>(K);
}
static float rtol_for(int K)
{
    return 5e-3f + 1e-5f * static_cast<float>(K);
}

// K スケール tol で比較。max_rel を表示。
static bool compare(const std::vector<float>& got,
                    const std::vector<float>& ref,
                    int K,
                    const char* name)
{
    const float atol = atol_for(K);
    const float rtol = rtol_for(K);
    float max_rel = 0.0f;
    double sad = 0.0; float max_abs = 0.0f;
    for (size_t i = 0; i < got.size(); ++i)
    {
        const float d0 = std::fabs(got[i] - ref[i]);
        sad += d0; if (d0 > max_abs) max_abs = d0;
    }
    const double mae = sad / static_cast<double>(got.size());
    std::cout << "[" << name << "] MAE=" << mae << " max_abs=" << max_abs << "\n";
    for (size_t i = 0; i < got.size(); ++i)
    {
        const float diff = std::fabs(got[i] - ref[i]);
        const float lim  = atol + rtol * std::fabs(ref[i]);
        if (diff > lim)
        {
            std::cerr << "[" << name << "] mismatch at " << i
                      << ": got " << got[i] << " ref " << ref[i]
                      << " diff " << diff << " lim " << lim
                      << " (K=" << K << " atol=" << atol << " rtol=" << rtol << ")\n";
            return false;
        }
        if (std::fabs(ref[i]) > 1e-3f)
        {
            max_rel = std::max(max_rel, diff / std::fabs(ref[i]));
        }
    }
    std::cout << "[" << name << "] PASSED (max_rel=" << max_rel
              << " atol=" << atol << " rtol=" << rtol << " K=" << K << ")\n";
    return true;
}

// CPU 参照 (FP32, direct conv = カーネルと同一式)。小形状ゴールデン専用。
// in/weight/bias は「FP16 をデコードした FP32 値」を渡すこと。bias 空なら加算なし。
//
// ループ順序メモ (2-6 S2): 出力 (n,co,ho,wo) を外側、入力チャネル/カーネルを内側に
// 置き、最内は wo を連続走査する。in/weight/out のいずれもチャネル先頭オフセットを
// ループ外で 1 回だけ計算して内側で加算アクセスし、L1/L2 キャッシュを当てやすくする。
// 参照は「明らかに正しい」ことが命なのでブロッキング等の凝った最適化は入れない
// (素直な 6 重ループのまま、アクセス順だけ素直にして小形状で十分速い)。
static std::vector<float> cpu_conv2d(const std::vector<float>& in,
                                     const std::vector<float>& weight,
                                     const std::vector<float>& bias,
                                     int N, int Cin, int H, int W,
                                     int Cout, int KH, int KW,
                                     int stride_h, int stride_w,
                                     int pad_h, int pad_w,
                                     int dilation_h, int dilation_w)
{
    const int Hout = out_dim(H, pad_h, dilation_h, KH, stride_h);
    const int Wout = out_dim(W, pad_w, dilation_w, KW, stride_w);
    std::vector<float> out(static_cast<size_t>(N) * Cout * Hout * Wout, 0.0f);
    const bool has_bias = !bias.empty();

    for (int n = 0; n < N; ++n)
    {
        for (int co = 0; co < Cout; ++co)
        {
            for (int ho = 0; ho < Hout; ++ho)
            {
                const int hi0 = ho * stride_h - pad_h;
                // 出力行先頭 (wo=0)。最内 wo は out[orow + wo] と連続アクセス。
                const long orow =
                    ((static_cast<long>(n) * Cout + co) * Hout + ho) * Wout;
                for (int wo = 0; wo < Wout; ++wo)
                {
                    const int wi0 = wo * stride_w - pad_w;
                    float acc = 0.0f;
                    for (int ci = 0; ci < Cin; ++ci)
                    {
                        const long in_ch_base = (static_cast<long>(n) * Cin + ci) * H * W;
                        const long w_ch_base  = (static_cast<long>(co) * Cin + ci) * KH * KW;
                        for (int kh = 0; kh < KH; ++kh)
                        {
                            const int hi = hi0 + kh * dilation_h;
                            if (hi < 0 || hi >= H)
                            {
                                continue;
                            }
                            const long in_row = in_ch_base + static_cast<long>(hi) * W;
                            const long w_row  = w_ch_base + static_cast<long>(kh) * KW;
                            for (int kw = 0; kw < KW; ++kw)
                            {
                                const int wi = wi0 + kw * dilation_w;
                                if (wi < 0 || wi >= W)
                                {
                                    continue;
                                }
                                acc += in[in_row + wi] * weight[w_row + kw];
                            }
                        }
                    }
                    if (has_bias)
                    {
                        acc += bias[co];
                    }
                    out[orow + wo] = acc;
                }
            }
        }
    }
    return out;
}

// device 経由で conv2d を実行し FP16 結果を「ビット列のまま」返す (G-10k T4 で切り出し)。
// memcmp 系のゲート ([G1] / [G2a] / [G2b] / [G3] / [G5]) はデコード前のこの列で突合する。
// force_direct=true なら GEMM 経路をバイパスして必ず direct conv で計算する
// (大形状の GPU-vs-direct 突合の「参照」側に使う)。
// bias_h が空のときは nullptr で起動する。
static std::vector<__half> run_gpu_conv_raw(const std::vector<__half>& in,
                                            const std::vector<__half>& weight,
                                            const std::vector<__half>& bias_h,
                                            int N, int Cin, int H, int W,
                                            int Cout, int KH, int KW,
                                            int stride_h, int stride_w,
                                            int pad_h, int pad_w,
                                            int dilation_h, int dilation_w,
                                            bool force_direct,
                                            int out_fill = -1)
{
    // out_fill >= 0 のときは出力バッファをそのバイト値で汚してから conv を呼ぶ
    // (G-10k T4 [G3] poison 用。書き残しがあれば検出できる。既定 -1 = 汚さない)。
    const int Hout = out_dim(H, pad_h, dilation_h, KH, stride_h);
    const int Wout = out_dim(W, pad_w, dilation_w, KW, stride_w);
    const size_t out_n = static_cast<size_t>(N) * Cout * Hout * Wout;

    __half* d_in     = nullptr;
    __half* d_weight = nullptr;
    __half* d_bias   = nullptr;
    __half* d_out    = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, in.size() * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_in, in.data(), in.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_weight, weight.size() * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_weight, weight.data(), weight.size() * sizeof(__half),
                          cudaMemcpyHostToDevice));
    if (!bias_h.empty())
    {
        CUDA_CHECK(cudaMalloc(&d_bias, bias_h.size() * sizeof(__half)));
        CUDA_CHECK(cudaMemcpy(d_bias, bias_h.data(), bias_h.size() * sizeof(__half),
                              cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMalloc(&d_out, out_n * sizeof(__half)));
    if (out_fill >= 0)
    {
        CUDA_CHECK(cudaMemset(d_out, out_fill, out_n * sizeof(__half)));
    }

    if (force_direct)
    {
        launch_conv2d_direct(d_in, d_weight, d_bias, d_out, N, Cin, H, W, Cout, KH, KW,
                             Hout, Wout, stride_h, stride_w, pad_h, pad_w,
                             dilation_h, dilation_w);
    }
    else
    {
        launch_conv2d(d_in, d_weight, d_bias, d_out, N, Cin, H, W, Cout, KH, KW,
                      stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w);
    }

    std::vector<__half> h_out(out_n);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, out_n * sizeof(__half), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_weight));
    if (d_bias != nullptr)
    {
        CUDA_CHECK(cudaFree(d_bias));
    }
    CUDA_CHECK(cudaFree(d_out));
    return h_out;
}

// FP16 ビット列をデコードして float 列にする。
static std::vector<float> decode_half(const std::vector<__half>& h)
{
    std::vector<float> out(h.size());
    for (size_t i = 0; i < h.size(); ++i)
    {
        out[i] = __half2float(h[i]);
    }
    return out;
}

// device 経由で conv2d を実行し FP16 結果をデコードして返す (従来の入口・挙動不変)。
static std::vector<float> run_gpu_conv(const std::vector<__half>& in,
                                       const std::vector<__half>& weight,
                                       const std::vector<__half>& bias_h,
                                       int N, int Cin, int H, int W,
                                       int Cout, int KH, int KW,
                                       int stride_h, int stride_w,
                                       int pad_h, int pad_w,
                                       int dilation_h, int dilation_w,
                                       bool force_direct = false)
{
    return decode_half(run_gpu_conv_raw(in, weight, bias_h, N, Cin, H, W, Cout, KH, KW,
                                        stride_h, stride_w, pad_h, pad_w,
                                        dilation_h, dilation_w, force_direct));
}

// 1 ケースを GPU 実行・CPU 参照と比較する共通ルーチン (小形状ゴールデン用)。
static bool run_case(const char* name,
                     int N, int Cin, int H, int W,
                     int Cout, int KH, int KW,
                     int stride_h, int stride_w,
                     int pad_h, int pad_w,
                     int dilation_h, int dilation_w,
                     bool with_bias,
                     unsigned seed)
{
    const int in_n = N * Cin * H * W;
    const int w_n  = Cout * Cin * KH * KW;

    HalfBuffer in = make_half(in_n, seed);
    HalfBuffer w  = make_half(w_n, seed + 1);

    std::vector<__half> bias_h;
    std::vector<float>  bias_ref;
    if (with_bias)
    {
        HalfBuffer bb = make_half(Cout, seed + 2, -0.5f, 0.5f);
        bias_h   = bb.h;
        bias_ref = bb.ref;
    }

    std::vector<float> got = run_gpu_conv(in.h, w.h, bias_h, N, Cin, H, W, Cout, KH, KW,
                                          stride_h, stride_w, pad_h, pad_w,
                                          dilation_h, dilation_w);
    std::vector<float> ref = cpu_conv2d(in.ref, w.ref, bias_ref, N, Cin, H, W, Cout, KH, KW,
                                        stride_h, stride_w, pad_h, pad_w,
                                        dilation_h, dilation_w);

    const int K = Cin * KH * KW; // 蓄積長
    return compare(got, ref, K, name);
}

// 大形状用: GEMM 経路 (launch_conv2d) 出力 vs direct 強制出力を GPU 同士で突合する。
// CPU 参照を踏まないため up_block_2 concat / VAE 512² 帯分割も数秒で検証できる。
static bool run_case_vs_direct(const char* name,
                               int N, int Cin, int H, int W,
                               int Cout, int KH, int KW,
                               int stride_h, int stride_w,
                               int pad_h, int pad_w,
                               int dilation_h, int dilation_w,
                               bool with_bias,
                               unsigned seed)
{
    const size_t in_n = static_cast<size_t>(N) * Cin * H * W;
    const size_t w_n  = static_cast<size_t>(Cout) * Cin * KH * KW;

    HalfBuffer in = make_half(static_cast<int>(in_n), seed);
    HalfBuffer w  = make_half(static_cast<int>(w_n), seed + 1);

    std::vector<__half> bias_h;
    if (with_bias)
    {
        HalfBuffer bb = make_half(Cout, seed + 2, -0.5f, 0.5f);
        bias_h = bb.h;
    }

    // GEMM 経路 (被験) と direct 経路 (参照) を同一入力で実行。
    std::vector<float> got = run_gpu_conv(in.h, w.h, bias_h, N, Cin, H, W, Cout, KH, KW,
                                          stride_h, stride_w, pad_h, pad_w,
                                          dilation_h, dilation_w, /*force_direct=*/false);
    std::vector<float> ref = run_gpu_conv(in.h, w.h, bias_h, N, Cin, H, W, Cout, KH, KW,
                                          stride_h, stride_w, pad_h, pad_w,
                                          dilation_h, dilation_w, /*force_direct=*/true);

    const int K = Cin * KH * KW;
    // GEMM 経路と direct はともに FP32 蓄積。残差は丸め順序の違いのみなので
    // FP16 相応 tol (compare の atol/rtol) で十分一致するはず。
    return compare(got, ref, K, name);
}

// ----------------------------------------------------------------
// 1. 1x1 stride1 pad0 (実質 GEMM 相当)
// ----------------------------------------------------------------
static bool test_conv_1x1()
{
    bool ok = true;
    ok = run_case("conv_1x1[odd]", 1, 8, 15, 15, 6, 1, 1, 1, 1, 0, 0, 1, 1, false, 101) && ok;
    ok = run_case("conv_1x1[even]", 1, 8, 16, 16, 6, 1, 1, 1, 1, 0, 0, 1, 1, false, 102) && ok;
    return ok;
}

// ----------------------------------------------------------------
// 2. 3x3 stride1 pad1 (same conv)。奇数/偶数サイズ。
// ----------------------------------------------------------------
static bool test_conv_3x3_s1p1()
{
    bool ok = true;
    ok = run_case("conv_3x3_s1p1[odd]", 1, 4, 17, 17, 4, 3, 3, 1, 1, 1, 1, 1, 1, false, 201) && ok;
    ok = run_case("conv_3x3_s1p1[even]", 1, 4, 16, 16, 4, 3, 3, 1, 1, 1, 1, 1, 1, false, 202) && ok;
    return ok;
}

// ----------------------------------------------------------------
// 3. 3x3 stride2 pad1 (ダウンサンプル)。奇数/偶数サイズ。
// ----------------------------------------------------------------
static bool test_conv_3x3_s2p1()
{
    bool ok = true;
    ok = run_case("conv_3x3_s2p1[odd]", 1, 4, 17, 17, 8, 3, 3, 2, 2, 1, 1, 1, 1, false, 301) && ok;
    ok = run_case("conv_3x3_s2p1[even]", 1, 4, 16, 16, 8, 3, 3, 2, 2, 1, 1, 1, 1, false, 302) && ok;
    return ok;
}

// ----------------------------------------------------------------
// 4. bias 有/無 の比較 (同一 shape)。
// ----------------------------------------------------------------
static bool test_conv_bias()
{
    bool ok = true;
    ok = run_case("conv_bias[off]", 1, 6, 12, 12, 5, 3, 3, 1, 1, 1, 1, 1, 1, false, 401) && ok;
    ok = run_case("conv_bias[on]", 1, 6, 12, 12, 5, 3, 3, 1, 1, 1, 1, 1, 1, true, 401) && ok;
    return ok;
}

// ----------------------------------------------------------------
// 5. N=2・Cin>1・Cout>1 (バッチ + 多チャネル)。
// ----------------------------------------------------------------
static bool test_conv_batch_multi()
{
    return run_case("conv_batch_multi", 2, 5, 14, 18, 7, 3, 3, 1, 1, 1, 1, 1, 1, true, 501);
}

// ----------------------------------------------------------------
// 6. dilation=2 の 3x3 (受容野拡張)。
// ----------------------------------------------------------------
static bool test_conv_dilation()
{
    bool ok = true;
    ok = run_case("conv_dil2[pad2]", 1, 4, 20, 20, 4, 3, 3, 1, 1, 2, 2, 2, 2, false, 601) && ok;
    ok = run_case("conv_dil2[pad0]", 1, 4, 20, 20, 4, 3, 3, 1, 1, 0, 0, 2, 2, true, 602) && ok;
    return ok;
}

// ----------------------------------------------------------------
// 7. GEMM 経路 (S3) の正当性。Cout/HW/K が下限 (16) 以上で direct に落ちない形状。
//    1x1=GEMM / 3x3 same=im2col+GEMM / 3x3 stride2=im2col+GEMM / shortcut 1x1+bias を網羅。
//    端数 (16 非倍数の Cout/HW) も含めて wmma の境界ガードを叩く。
// ----------------------------------------------------------------
static bool test_conv_gemm_path()
{
    bool ok = true;
    // 1x1 GEMM (bias なし) — N=HW=20x20=400, Cout=32, K=Cin=24。
    ok = run_case("gemm_1x1", 1, 24, 20, 20, 32, 1, 1, 1, 1, 0, 0, 1, 1, false, 701) && ok;
    // 1x1 GEMM + bias (ResBlock shortcut 相当) — 端数 Cout=20。
    ok = run_case("gemm_1x1_bias", 1, 24, 17, 19, 20, 1, 1, 1, 1, 0, 0, 1, 1, true, 702) && ok;
    // 3x3 same (im2col+GEMM) + bias — Cin=20, Cout=24, 18x18。K=Cin*9=180。
    ok = run_case("gemm_3x3_same", 1, 20, 18, 18, 24, 3, 3, 1, 1, 1, 1, 1, 1, true, 703) && ok;
    // 3x3 stride2 downsample (im2col+GEMM) — 33x33 -> 17x17。Cout=32。
    ok = run_case("gemm_3x3_s2", 1, 16, 33, 33, 32, 3, 3, 2, 2, 1, 1, 1, 1, true, 704) && ok;
    // UNet 相当 (小さめ): Cin=Cout=64, 32x32 3x3 same。
    ok = run_case("gemm_unet_like", 1, 64, 32, 32, 64, 3, 3, 1, 1, 1, 1, 1, 1, true, 705) && ok;
    return ok;
}

// ----------------------------------------------------------------
// 8. 実モデル相当の「大形状」GEMM 経路 (S2 conv 数値バグ再現)。GPU-vs-direct 突合。
//    前任は大 K かつ帯分割を踏まず破綻を見逃した。ここで叩く (すべて GPU 同士・数秒):
//      - down_block_0 相当 (帯分割なし対照): Cin=320 Cout=320 128² → K=2880 col=188MB<256MB
//      - up_block_2 単帯 (帯分割なし): Cin=640 Cout=320 128² → K=5760 col=188MB<256MB
//      - up_block_2 concat (帯分割発動): Cin=960 Cout=320 128² → K=8640 col=283MB>256MB ★旧破綻
//      - VAE 帯分割発動: Cin=128 Cout=128 512² → K=1152 Ncol=262144 col=604MB>256MB ★旧破綻
//    direct conv を参照に GEMM 経路 (im2col+wmma / 帯分割散布) の数値一致を確認する。
// ----------------------------------------------------------------
static bool test_conv_gemm_large()
{
    bool ok = true;
    ok = run_case_vs_direct("large_down0_320_128", 1, 320, 128, 128, 320, 3, 3, 1, 1, 1, 1, 1, 1,
                            false, 801) && ok;
    ok = run_case_vs_direct("large_up2_640to320_128", 1, 640, 128, 128, 320, 3, 3, 1, 1, 1, 1, 1, 1,
                            true, 802) && ok;
    ok = run_case_vs_direct("large_up2_concat_960to320_128", 1, 960, 128, 128, 320, 3, 3, 1, 1, 1, 1,
                            1, 1, true, 803) && ok;
    ok = run_case_vs_direct("large_vae_c128_512_band", 1, 128, 512, 512, 128, 3, 3, 1, 1, 1, 1, 1, 1,
                            true, 804) && ok;
    return ok;
}

// ================================================================
// G-10k T4 ヘルパ群 (docs/g10k-plan.md §7 の 7 ゲート用)。
//
// ★2 プロセス体制について (§9-4 / §9-7):
//   conv2d.cu の opt-in スイッチ DOLLAMA_CONV_BATCH は getenv キャッシュ型 = プロセス単位固定
//   なので、[G1] (既定 = per-n 直列の旧経路) と [G2a]〜[G4] (DOLLAMA_CONV_BATCH=1 の新経路) は
//   同一プロセスでは両方取れない。本 exe は env の値を読んで「このプロセスがどちらの経路か」を
//   知り、各ゲートの期待値を切り替える (meson test 自動枠 = 素の test_conv2d が既定経路 /
//   conv2d_batch_on(=1) が新経路)。
//   ★これは「同一プロセス内で切り替えた」のではない。記録にもそう書かないこと。
//   env の値を読むこと自体は分岐の証拠にならないので、各ゲートは必ず
//   「実際に通った枝」の計器 (gemm_batched_stats の差分・アリーナ alloc 回数の差分) と
//   突き合わせる = env 表示と実経路が食い違えば赤になる (opt-in 迂回の負のコントロールは
//   ここで捕まる)。
// ================================================================

// このプロセスが per-n 直列の旧経路 (既定 = DOLLAMA_CONV_BATCH が未設定/空/"1"以外) か。
static bool t4_conv_batch_off()
{
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
    const char* v = std::getenv("DOLLAMA_CONV_BATCH");
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
    return !(v != nullptr && std::strcmp(v, "1") == 0);
}

// FP16 ビット列の範囲 memcmp。
static bool t4_half_range_equal(const std::vector<__half>& a, size_t off_a,
                                const std::vector<__half>& b, size_t off_b, size_t n)
{
    if (off_a + n > a.size() || off_b + n > b.size())
    {
        return false;
    }
    return std::memcmp(a.data() + off_a, b.data() + off_b, n * sizeof(__half)) == 0;
}

// 経路計器の差分 (gemm batched 分岐カウンタ + アリーナ alloc 回数)。
struct T4PathDelta
{
    uint64_t wrapper  = 0; // launch_gemm_fp16_batched 呼び出し回数
    uint64_t cublas   = 0; // cuBLAS strided batched 発行回数
    uint64_t fallback = 0; // フォールバック直列ループ回数
    uint64_t arena    = 0; // device_arena alloc() 回数 (im2col col / 帯バッファ)
};

struct T4PathSnapshot
{
    GemmBatchedStats g;
    DeviceArenaStats a;
};

static T4PathSnapshot t4_snapshot()
{
    T4PathSnapshot s;
    s.g = gemm_batched_stats();
    s.a = device_arena_stats(DeviceArenaId::UNet);
    return s;
}

static T4PathDelta t4_delta(const T4PathSnapshot& s0, const T4PathSnapshot& s1)
{
    T4PathDelta d;
    d.wrapper  = s1.g.wrapper_calls        - s0.g.wrapper_calls;
    d.cublas   = s1.g.cublas_batched_calls - s0.g.cublas_batched_calls;
    d.fallback = s1.g.fallback_loops       - s0.g.fallback_loops;
    d.arena    = s1.a.alloc_calls          - s0.a.alloc_calls;
    return d;
}

static void t4_print_delta(const char* tag, const T4PathDelta& d)
{
    std::cout << "[" << tag << "] path counters: gemm_batched wrapper +" << d.wrapper
              << " cublas_batched +" << d.cublas << " fallback_loops +" << d.fallback
              << " | arena alloc +" << d.arena << "\n";
}

// 2 サンプル (X, Y) を連結した N=2 入力を作る。
static std::vector<__half> t4_concat(const std::vector<__half>& x, const std::vector<__half>& y)
{
    std::vector<__half> cat;
    cat.reserve(x.size() + y.size());
    cat.insert(cat.end(), x.begin(), x.end());
    cat.insert(cat.end(), y.begin(), y.end());
    return cat;
}

// MAE / max_abs / max_rel / exact 率を print する (合否なし・characterization)。
static void t4_print_numeric(const char* tag, const std::vector<float>& got,
                             const std::vector<float>& ref)
{
    double sad = 0.0;
    float  max_abs = 0.0f;
    float  max_rel = 0.0f;
    size_t exact = 0;
    for (size_t i = 0; i < got.size(); ++i)
    {
        const float d = std::fabs(got[i] - ref[i]);
        sad += d;
        if (d > max_abs) max_abs = d;
        if (got[i] == ref[i]) ++exact;
        if (std::fabs(ref[i]) > 1e-3f)
        {
            max_rel = std::max(max_rel, d / std::fabs(ref[i]));
        }
    }
    const double mae = sad / static_cast<double>(got.size());
    std::cout << "[" << tag << "] MAE=" << mae << " max_abs=" << max_abs
              << " max_rel=" << max_rel
              << " exact=" << exact << "/" << got.size()
              << " (" << (100.0 * static_cast<double>(exact) / static_cast<double>(got.size()))
              << "%)"
              << (max_abs == 0.0f ? " BIT-EXACT" : " not bit-exact")
              << " | characterization: G-2k S2 per-sample MAE 6.4e-05 と同オーダーか (合否なし)\n";
}

// ----------------------------------------------------------------
// 9. N>1 バッチ GEMM 経路 (G-2k S1)。CFG cond/uncond の B=2 束ねの下地。
//    launch_conv2d の N=2 出力を、per-sample に N=1 で 2 回呼んだ出力
//    (= 既存 N==1 GEMM 経路そのまま) と突合する。Cout/HW/K は GEMM 下限 (16) 以上にする。
//
//    G-10k T4 で [G1] へ昇格 (docs/g10k-plan.md §7・§8 ケース B で opt-in 降格後も維持):
//      - 既定のプロセス (= per-n 直列の旧経路。DOLLAMA_CONV_BATCH が未設定/空/"1"以外) では、
//        N=2 出力と per-sample 参照の **memcmp 一致を hard 合否**にする (各サンプルは独立で
//        K-loop 順・蓄積順が N==1 と完全同一 = 既に観測済みの性質の固定)。
//        加えて「旧経路は batched GEMM ラッパを 1 度も呼ばない」ことを計器差分で hard 化する
//        (旧経路が新経路へ迂回していれば wrapper 差分が非 0 になり赤)。
//      - DOLLAMA_CONV_BATCH=1 のプロセス (= 新経路) では bit 一致は要求しない (batched の
//        タイル選択差 = FP16 tol 内が設計・§5)。bit 一致の有無は print のみ (characterization)。
//      - どちらのプロセスでも既存 compare(..., K, ...) は floor として維持する
//        (緩和ではなく分岐。超えたら緩めず BLOCK)。
// ----------------------------------------------------------------
static bool run_case_batch_vs_persample(const char* name,
                                        int Cin, int H, int W,
                                        int Cout, int KH, int KW,
                                        int stride_h, int stride_w,
                                        int pad_h, int pad_w,
                                        int dilation_h, int dilation_w,
                                        bool with_bias,
                                        unsigned seed)
{
    const int Hout = out_dim(H, pad_h, dilation_h, KH, stride_h);
    const int Wout = out_dim(W, pad_w, dilation_w, KW, stride_w);
    const int per_in  = Cin * H * W;          // 1 サンプルの入力要素数
    const int per_out = Cout * Hout * Wout;   // 1 サンプルの出力要素数
    const int w_n     = Cout * Cin * KH * KW;

    // 2 サンプル分の入力 (別データ)。weight/bias は共有。
    HalfBuffer in0 = make_half(per_in, seed);
    HalfBuffer in1 = make_half(per_in, seed + 10);
    HalfBuffer w   = make_half(w_n, seed + 1);

    std::vector<__half> bias_h;
    if (with_bias)
    {
        HalfBuffer bb = make_half(Cout, seed + 2, -0.5f, 0.5f);
        bias_h = bb.h;
    }

    // N=2 バッチ入力 (in0 ++ in1)。
    const std::vector<__half> in_batch = t4_concat(in0.h, in1.h);

    // 被験: launch_conv2d に N=2 で通す。経路計器の差分も採る。
    const T4PathSnapshot s0 = t4_snapshot();
    const std::vector<__half> got_h = run_gpu_conv_raw(in_batch, w.h, bias_h, 2, Cin, H, W,
                                                       Cout, KH, KW,
                                                       stride_h, stride_w, pad_h, pad_w,
                                                       dilation_h, dilation_w,
                                                       /*force_direct=*/false);
    const T4PathDelta d = t4_delta(s0, t4_snapshot());

    // 参照: 各サンプルを N=1 で個別に通し (= 既存 N==1 GEMM 経路)、連結する。
    const std::vector<__half> ref0_h = run_gpu_conv_raw(in0.h, w.h, bias_h, 1, Cin, H, W,
                                                        Cout, KH, KW,
                                                        stride_h, stride_w, pad_h, pad_w,
                                                        dilation_h, dilation_w,
                                                        /*force_direct=*/false);
    const std::vector<__half> ref1_h = run_gpu_conv_raw(in1.h, w.h, bias_h, 1, Cin, H, W,
                                                        Cout, KH, KW,
                                                        stride_h, stride_w, pad_h, pad_w,
                                                        dilation_h, dilation_w,
                                                        /*force_direct=*/false);
    const std::vector<__half> ref_h = t4_concat(ref0_h, ref1_h);

    const std::vector<float> got = decode_half(got_h);
    const std::vector<float> ref = decode_half(ref_h);

    // ビット一致 (memcmp) の明示チェック + 数値 characterization。
    const bool bit_exact = t4_half_range_equal(got_h, 0, ref_h, 0, static_cast<size_t>(2) * per_out);
    t4_print_numeric(name, got, ref);
    t4_print_delta(name, d);

    bool ok = true;
    const int K = Cin * KH * KW;
    if (t4_conv_batch_off())
    {
        // [G1] hard: 既定 (旧経路) は per-sample と memcmp 一致、かつ batched ラッパ非経由。
        const bool g1_bits = bit_exact;
        const bool g1_path = (d.wrapper == 0) && (d.cublas == 0) && (d.fallback == 0);
        std::cout << "[G1:" << name << "] default (per-n serial path) process: memcmp vs per-sample "
                  << (g1_bits ? "BIT-EXACT" : "MISMATCH")
                  << " / batched wrapper untouched (expected +0): "
                  << (g1_path ? "yes" : "NO")
                  << " -> " << ((g1_bits && g1_path) ? "PASSED" : "FAILED") << "\n";
        ok = g1_bits && g1_path && ok;
    }
    else
    {
        // DOLLAMA_CONV_BATCH=1 プロセス (新経路): [G1] は本プロセスでは検査しない (別プロセスで採る)。
        // bit 一致の有無は上の print (characterization) のみ。
        std::cout << "[G1:" << name << "] n/a in this process (batched path;"
                  << " [G1] is taken in a separate default (DOLLAMA_CONV_BATCH unset) process)\n";
    }

    // floor: 既存 compare(..., K, ...) は両プロセスで維持 (超えたら緩めず BLOCK)。
    ok = compare(got, ref, K, name) && ok;
    return ok;
}

// ----------------------------------------------------------------
// 9b. N=2 バッチ GEMM 突合ケース群。1x1 / 3x3 same / 3x3 stride2 / bias 有無。
//     G-10k T4: この 5 ケースが [G1] の対象 (既定=DOLLAMA_CONV_BATCH 未設定のプロセスで hard memcmp)。
// ----------------------------------------------------------------
static bool test_conv_batch_gemm()
{
    bool ok = true;
    std::cout << "[test_conv_batch_gemm] process mode: "
              << (t4_conv_batch_off() ? "default (per-n serial path; [G1] hard)"
                                      : "DOLLAMA_CONV_BATCH=1 (batched path; [G1] n/a here)")
              << "\n";
    // 1x1 GEMM バッチ — Cin=24 Cout=32 20x20。
    ok = run_case_batch_vs_persample("batch2_1x1", 24, 20, 20, 32, 1, 1, 1, 1, 0, 0, 1, 1,
                                     false, 901) && ok;
    // 1x1 GEMM + bias バッチ — 端数 Cout=20。
    ok = run_case_batch_vs_persample("batch2_1x1_bias", 24, 17, 19, 20, 1, 1, 1, 1, 0, 0, 1, 1,
                                     true, 902) && ok;
    // 3x3 same (im2col+GEMM) + bias バッチ — Cin=20 Cout=24 18x18。
    ok = run_case_batch_vs_persample("batch2_3x3_same", 20, 18, 18, 24, 3, 3, 1, 1, 1, 1, 1, 1,
                                     true, 903) && ok;
    // 3x3 stride2 downsample バッチ — 33x33 -> 17x17 Cout=32。
    ok = run_case_batch_vs_persample("batch2_3x3_s2", 16, 33, 33, 32, 3, 3, 2, 2, 1, 1, 1, 1,
                                     true, 904) && ok;
    // UNet 相当 (CFG B=2 の代表): Cin=Cout=320 64x64 3x3 same。
    ok = run_case_batch_vs_persample("batch2_unet_c320_64", 320, 64, 64, 320, 3, 3, 1, 1, 1, 1,
                                     1, 1, true, 905) && ok;
    return ok;
}

// ----------------------------------------------------------------
// 10. G-10k T4: 新経路 (真 batch2) のゲート [G2a] / [G2b] / [G2] / [G3] / [G4] / [G5]。
//
//   検査する性質 (docs/g10k-plan.md §7・実装方法は台帳が指定しない):
//     [G2a] 同一データを sample0/1 に入れたとき出力の前半/後半が memcmp 一致 (hard)。
//           stride / オフセット誤り・サンプル間の漏れを tol 無しで捕まえる。
//           ★全サンプルで一様にずれる bias バグは捕まえない (受け皿は [G2] floor)。
//     [G2b] (X,Y) と (Y,X) の出力が入れ替えで memcmp 一致 (hard)。非対称バグ。
//     [G2]  batched vs per-sample 参照を既存 compare(..., K, ...) で判定 (floor・超えたら BLOCK)。
//           MAE / max_abs / max_rel / exact 率を print (characterization)。
//     [G3]  同一設定 3 runs が memcmp 一致 (hard)。方式は 9d `test_g8k_arena_bitexact` と同型:
//           run1 / run2 の前にアリーナを 0xFF / 0x00 で汚染 (poison_arena) してから rewind し、
//           次の run が同じ領域 (col + 帯バッファ) を再利用するよう仕向ける。出力バッファも
//           0xCD で汚す。poison 量は「新経路の実使用量を覆う上限」を形状から先験的に決め
//           (col <= min(N*K*Hout*Wout*2B, IM2COL_TILE_BYTES) + 帯 <= N*Cout*Hout*Wout*2B
//           + 整列余裕 1MiB)、実走の arena peak_request がその中に収まることを同時に hard 化する
//           (収まらなければ汚染が実使用域を覆っておらず、[G3] は空撃ちになるため)。
//           ★poison 無しの素の 3 連走では「前走の書き残しをそのまま読む」種類のバグを
//             捕まえられない (T4 初版の [G3] がそうだった・台帳 §7「[G3] への poison 追加」)。
//     [G4]  rows_cap を N で割った結果、帯分割が発動する形状で [G2a]/[G2b]/[G3] を通す (hard)。
//           帯分割が「実際に発動した」ことは計器で示す: アリーナ alloc が 1 call あたり
//           +2 (col + 帯バッファ) / batched GEMM 発行が 1 call あたり >= 2 (帯の数)。
//     [G5]  GEMM 下限割れの N=2 形状が direct 経路のまま通る (hard)。
//           direct 強制呼びと memcmp 一致 + batched ラッパ非経由 + アリーナ alloc 0。
//
//   ★空撃ち防止: [G2a]/[G2b]/[G3] は「ラッパが内部で直列ループしているだけ」でも緑になる。
//     そのため DOLLAMA_CONV_BATCH=1 プロセスでは各 launch_conv2d(N=2) 呼び出しで
//     「cuBLAS strided batched が実際に >= 1 回発行された (fallback 0)」ことを計器差分で hard 化する
//     (T3 の [H6] と同じ役割)。DOLLAMA_GEMM=wmma を付けた走行ではこの検査は設計上 red になる
//     (全ケースがフォールバック枝に落ちるため・§9-7)。T4 は wmma 走行を要求しない。
//   ★既定 (DOLLAMA_CONV_BATCH 未設定) のプロセスでは、同じ性質 ([G2a]/[G2b]/[G3]/floor) は
//     per-n 直列でも自明に成立するので同様に通し、経路計器は「wrapper +0」を期待する。
//
//   [G4] の形状選定 (実装者が rows_cap の実値から選んだ・N=2):
//     rows_cap(N) = IM2COL_TILE_BYTES / (K*Wout*2*N)。
//     - Cin=640 -> Cout=320 128^2 3x3 (up_block_2 の resnet 2/3 の conv1 相当):
//         K=5760, Wout=128 -> bytes_per_row=1474560
//         N=1: rows_cap=182 >= Hout=128 -> 帯分割なし (test 8 の "up_block_2 単帯" と同じ)
//         N=2: rows_cap=91  <  Hout=128 -> 帯分割 2 本 (91 行 + 37 行)  ★N 分割で初めて発動
//       = 「rows_cap を N で割った結果として発動する」形状そのもの。これを [G4] の正典にする。
//     - 補助: Cin=960 -> Cout=320 128^2 (up_block_2 concat・N=1 でも帯分割):
//         K=8640 -> bytes_per_row=2211840。N=1: rows_cap=121 (2 帯) / N=2: rows_cap=60 (3 帯 = 60+60+8)
//       端数の小さい最終帯 (8 行) を含む 3 帯を踏む。
//     代表形状 3 つ (320/128^2・640/64^2・1280/32^2) は N=2 でも rows_cap=182 >= Hout で帯分割なし。
// ----------------------------------------------------------------

// 9d (G-8k S1b) のアリーナ汚染ヘルパ (定義は後方)。[G3] の poison に共用する。
static void poison_arena(size_t bytes, int pattern);

struct T4Shape
{
    const char* name;
    int Cin, H, W, Cout, KH, KW, stride, pad;
    bool with_bias;
    unsigned seed;
    bool expect_banded;  // 新経路で帯分割が発動する形状か (計器の期待値に使う)
    bool is_1x1;         // 1x1 経路 (アリーナ alloc 0 が期待値)
};

static bool run_case_t4_batched(const T4Shape& sh)
{
    const int Hout = out_dim(sh.H, sh.pad, 1, sh.KH, sh.stride);
    const int Wout = out_dim(sh.W, sh.pad, 1, sh.KW, sh.stride);
    const size_t per_in  = static_cast<size_t>(sh.Cin) * sh.H * sh.W;
    const size_t per_out = static_cast<size_t>(sh.Cout) * Hout * Wout;
    const size_t w_n     = static_cast<size_t>(sh.Cout) * sh.Cin * sh.KH * sh.KW;
    const int K = sh.Cin * sh.KH * sh.KW;
    const bool off = t4_conv_batch_off();

    HalfBuffer X = make_half(static_cast<int>(per_in), sh.seed);
    HalfBuffer Y = make_half(static_cast<int>(per_in), sh.seed + 10);
    HalfBuffer w = make_half(static_cast<int>(w_n), sh.seed + 1);
    std::vector<__half> bias_h;
    if (sh.with_bias)
    {
        HalfBuffer bb = make_half(sh.Cout, sh.seed + 2, -0.5f, 0.5f);
        bias_h = bb.h;
    }

    auto run2 = [&](const std::vector<__half>& in2, T4PathDelta* d,
                    int out_fill = -1) -> std::vector<__half>
    {
        const T4PathSnapshot s0 = t4_snapshot();
        std::vector<__half> o = run_gpu_conv_raw(in2, w.h, bias_h, 2, sh.Cin, sh.H, sh.W,
                                                 sh.Cout, sh.KH, sh.KW,
                                                 sh.stride, sh.stride, sh.pad, sh.pad, 1, 1,
                                                 /*force_direct=*/false, out_fill);
        if (d != nullptr)
        {
            *d = t4_delta(s0, t4_snapshot());
        }
        return o;
    };
    auto run1 = [&](const std::vector<__half>& in1) -> std::vector<__half>
    {
        return run_gpu_conv_raw(in1, w.h, bias_h, 1, sh.Cin, sh.H, sh.W,
                                sh.Cout, sh.KH, sh.KW,
                                sh.stride, sh.stride, sh.pad, sh.pad, 1, 1,
                                /*force_direct=*/false);
    };

    // 経路計器の期待値 (1 回の launch_conv2d(N=2) あたり)。
    //   =1  : wrapper >= 1 かつ wrapper == cublas (全発行が cuBLAS batched) かつ fallback 0。
    //         帯分割形状なら wrapper >= 2 (帯の数)。アリーナ alloc は 1x1=0 / 非帯=1 / 帯=2。
    //   既定: wrapper == 0 (旧経路は batched ラッパを経由しない)。
    auto path_ok = [&](const char* tag, const T4PathDelta& d) -> bool
    {
        t4_print_delta(tag, d);
        bool ok = true;
        if (off)
        {
            ok = (d.wrapper == 0) && (d.cublas == 0) && (d.fallback == 0);
            std::cout << "[" << tag << "] path check (default: expected wrapper +0) -> "
                      << (ok ? "PASSED" : "FAILED") << "\n";
            return ok;
        }
        const uint64_t min_calls = sh.expect_banded ? 2 : 1;
        const uint64_t exp_arena = sh.is_1x1 ? 0 : (sh.expect_banded ? 2 : 1);
        ok = (d.wrapper >= min_calls) && (d.cublas == d.wrapper) && (d.fallback == 0)
             && (d.arena == exp_arena);
        std::cout << "[" << tag << "] path check (DOLLAMA_CONV_BATCH=1: expected wrapper>=" << min_calls
                  << " cublas==wrapper fallback==0 arena==" << exp_arena << ") -> "
                  << (ok ? "PASSED" : "FAILED") << "\n";
        return ok;
    };

    bool ok = true;
    std::string tag;

    // ---- [G2a] 同一データ 2 本 -> 前半/後半 memcmp ----
    {
        T4PathDelta d;
        const std::vector<__half> o = run2(t4_concat(X.h, X.h), &d);
        const bool g2a = t4_half_range_equal(o, 0, o, per_out, per_out);
        tag = std::string("G2a:") + sh.name;
        std::cout << "[" << tag << "] uniform (X,X): sample0 vs sample1 memcmp "
                  << (g2a ? "BIT-EXACT PASSED" : "MISMATCH FAILED") << "\n";
        ok = g2a && ok;
        ok = path_ok(tag.c_str(), d) && ok;
    }

    // ---- 本走行 (X,Y) + [G2b] 置換 (Y,X) + [G3] 3 runs ----
    T4PathDelta dxy;
    const std::vector<__half> o_xy = run2(t4_concat(X.h, Y.h), &dxy);
    {
        const std::vector<__half> o_yx = run2(t4_concat(Y.h, X.h), nullptr);
        const bool sw0 = t4_half_range_equal(o_xy, 0,       o_yx, per_out, per_out);
        const bool sw1 = t4_half_range_equal(o_xy, per_out, o_yx, 0,       per_out);
        tag = std::string("G2b:") + sh.name;
        std::cout << "[" << tag << "] swap (X,Y) vs (Y,X): "
                  << (sw0 && sw1 ? "BIT-EXACT PASSED" : "MISMATCH FAILED")
                  << " (s0==s1' " << (sw0 ? "yes" : "no") << " / s1==s0' " << (sw1 ? "yes" : "no")
                  << ")\n";
        ok = sw0 && sw1 && ok;
        ok = path_ok(tag.c_str(), dxy) && ok;
    }
    {
        // poison 量 = この形状の中間バッファの先験的上限 (実測から決めない):
        //   col  <= min(N*K*Hout*Wout*2B, IM2COL_TILE_BYTES=256MiB)  (新経路・旧経路とも)
        //   帯   <= N*Cout*Hout*Wout*2B                              (tile_rows <= Hout)
        //   + 1MiB (alloc 2 本の個別整列の余裕)
        // 1x1 経路はアリーナ alloc 0 なので poison は素通り (害なし)。
        const size_t kTileCap = static_cast<size_t>(256) << 20;
        const size_t col_bound  = std::min(static_cast<size_t>(2) * K * Hout * Wout * sizeof(__half),
                                           kTileCap);
        const size_t band_bound = static_cast<size_t>(2) * per_out * sizeof(__half);
        const size_t poison_bytes = col_bound + band_bound + (static_cast<size_t>(1) << 20);

        // run1: 0xFF 汚染 (FP16 では NaN) / run2: 0x00 汚染。出力バッファは 0xCD で汚す。
        // 各 run の前にアリーナ計数を reset し、その run の peak_request が poison に
        // 収まっているか (= 汚染が実使用域を覆っているか) を採る。
        const int patterns[2] = {0xFF, 0x00};
        std::vector<__half> runs[2];
        size_t peaks[2] = {0, 0};
        for (int r = 0; r < 2; ++r)
        {
            poison_arena(poison_bytes, patterns[r]);
            device_arena_reset_counters(DeviceArenaId::UNet);
            runs[r] = run2(t4_concat(X.h, Y.h), nullptr, /*out_fill=*/0xCD);
            peaks[r] = device_arena_stats(DeviceArenaId::UNet).peak_request_bytes;
        }
        const bool eq01 = t4_half_range_equal(o_xy, 0, runs[0], 0, 2 * per_out);
        const bool eq02 = t4_half_range_equal(o_xy, 0, runs[1], 0, 2 * per_out);
        const bool covered = (peaks[0] <= poison_bytes) && (peaks[1] <= poison_bytes);
        const bool g3 = eq01 && eq02 && covered;
        tag = std::string("G3:") + sh.name;
        std::cout << "[" << tag << "] 3 runs memcmp: run0==run1(0xFF poisoned): "
                  << (eq01 ? "BIT-EXACT" : "DIFF")
                  << " / run0==run2(0x00 poisoned): " << (eq02 ? "BIT-EXACT" : "DIFF")
                  << " | poison=" << (poison_bytes >> 20) << "MiB"
                  << " (col_bound " << (col_bound >> 20) << " + band_bound " << (band_bound >> 20)
                  << " + 1) covers peak_request " << (peaks[0] >> 20) << "/" << (peaks[1] >> 20)
                  << "MiB: " << (covered ? "yes" : "NO")
                  << " -> " << (g3 ? "PASSED" : "FAILED") << "\n";
        ok = g3 && ok;
    }

    // ---- [G2] floor: batched vs per-sample (N=1 x2) を既存 compare で判定 ----
    {
        const std::vector<__half> ref_h = t4_concat(run1(X.h), run1(Y.h));
        const std::vector<float> got = decode_half(o_xy);
        const std::vector<float> ref = decode_half(ref_h);
        tag = std::string("G2:") + sh.name;
        t4_print_numeric(tag.c_str(), got, ref);
        ok = compare(got, ref, K, tag.c_str()) && ok;
    }

    return ok;
}

// [G4] 帯分割形状の VRAM characterization: 1 回の launch_conv2d(N=2) が要求した
// アリーナ同時生存量のピークを print (IM2COL_TILE_BYTES=256MiB が N 込み上限であることの観測)。
static void t4_print_band_peak(const T4Shape& sh)
{
    const int Hout = out_dim(sh.H, sh.pad, 1, sh.KH, sh.stride);
    const int Wout = out_dim(sh.W, sh.pad, 1, sh.KW, sh.stride);
    const size_t per_in  = static_cast<size_t>(sh.Cin) * sh.H * sh.W;
    const size_t per_out = static_cast<size_t>(sh.Cout) * Hout * Wout;
    const size_t w_n     = static_cast<size_t>(sh.Cout) * sh.Cin * sh.KH * sh.KW;

    HalfBuffer X = make_half(static_cast<int>(2 * per_in), sh.seed + 100);
    HalfBuffer w = make_half(static_cast<int>(w_n), sh.seed + 101);

    __half *d_in = nullptr, *d_w = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, 2 * per_in * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_w, w_n * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_out, 2 * per_out * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_in, X.h.data(), 2 * per_in * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, w.h.data(), w_n * sizeof(__half), cudaMemcpyHostToDevice));

    device_arena_reset_counters(DeviceArenaId::UNet);
    launch_conv2d(d_in, d_w, nullptr, d_out, 2, sh.Cin, sh.H, sh.W, sh.Cout, sh.KH, sh.KW,
                  sh.stride, sh.stride, sh.pad, sh.pad, 1, 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    const DeviceArenaStats st = device_arena_stats(DeviceArenaId::UNet);
    std::cout << "[G4:" << sh.name << "] arena peak_request during one launch_conv2d(N=2) = "
              << (st.peak_request_bytes >> 20) << " MiB"
              << " (col total cap = 256 MiB incl. N; band buffer is extra)"
              << " alloc=" << st.alloc_calls << " | characterization only\n";

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_out));
}

static bool test_g10k_t4_gates()
{
    bool ok = true;
    std::cout << "[test_g10k_t4_gates] process mode: "
              << (t4_conv_batch_off() ? "default (per-n serial path)" : "DOLLAMA_CONV_BATCH=1 (batched path)")
              << "\n";

    // 小形状 (端数あり): 1x1+bias (17x19, Cout=20) / 3x3 stride2 (33->17)。
    const T4Shape small_1x1 = {"small_1x1_bias_17x19", 24, 17, 19, 20, 1, 1, 1, 0, true, 1901, false, true};
    const T4Shape small_s2  = {"small_3x3_s2_33to17",  16, 33, 33, 32, 3, 3, 2, 1, true, 1902, false, false};
    ok = run_case_t4_batched(small_1x1) && ok;
    ok = run_case_t4_batched(small_s2)  && ok;

    // 1x1 の UNet 代表 (ResBlock skip 相当 320->640, 32^2)。
    const T4Shape unet_1x1 = {"unet_1x1_320to640_32", 320, 32, 32, 640, 1, 1, 1, 0, true, 1903, false, true};
    ok = run_case_t4_batched(unet_1x1) && ok;

    // 代表形状 3 つ (§5): 320/128^2・640/64^2・1280/32^2 (3x3 same + bias)。N=2 でも帯分割なし。
    const T4Shape rep_320  = {"rep_320_128",  320, 128, 128, 320,  3, 3, 1, 1, true, 1911, false, false};
    const T4Shape rep_640  = {"rep_640_64",   640,  64,  64, 640,  3, 3, 1, 1, true, 1912, false, false};
    const T4Shape rep_1280 = {"rep_1280_32", 1280,  32,  32, 1280, 3, 3, 1, 1, true, 1913, false, false};
    ok = run_case_t4_batched(rep_320)  && ok;
    ok = run_case_t4_batched(rep_640)  && ok;
    ok = run_case_t4_batched(rep_1280) && ok;

    // [G4] 帯分割形状 (正典): 640->320 128^2 = N=2 で初めて帯分割 (91+37 行)。
    const T4Shape g4_band = {"G4_band_640to320_128", 640, 128, 128, 320, 3, 3, 1, 1, true, 1921, true, false};
    ok = run_case_t4_batched(g4_band) && ok;
    // [G4] 補助: 960->320 128^2 = N=2 で 3 帯 (60+60+8 行)。
    const T4Shape g4_band3 = {"G4_band_960to320_128", 960, 128, 128, 320, 3, 3, 1, 1, true, 1922, true, false};
    ok = run_case_t4_batched(g4_band3) && ok;
    if (!t4_conv_batch_off())
    {
        t4_print_band_peak(g4_band);
        t4_print_band_peak(g4_band3);
    }

    // ---- [G5] GEMM 下限割れの N=2 形状 (test 5 と同一: Cin=5 14x18 Cout=7 3x3) は direct のまま ----
    {
        const int N = 2, Cin = 5, H = 14, W = 18, Cout = 7, KH = 3, KW = 3;
        const int Hout = out_dim(H, 1, 1, KH, 1);
        const int Wout = out_dim(W, 1, 1, KW, 1);
        HalfBuffer in = make_half(N * Cin * H * W, 501);
        HalfBuffer w  = make_half(Cout * Cin * KH * KW, 502);
        HalfBuffer bb = make_half(Cout, 503, -0.5f, 0.5f);

        const T4PathSnapshot s0 = t4_snapshot();
        const std::vector<__half> got = run_gpu_conv_raw(in.h, w.h, bb.h, N, Cin, H, W, Cout, KH, KW,
                                                         1, 1, 1, 1, 1, 1, /*force_direct=*/false);
        const T4PathDelta d = t4_delta(s0, t4_snapshot());
        const std::vector<__half> ref = run_gpu_conv_raw(in.h, w.h, bb.h, N, Cin, H, W, Cout, KH, KW,
                                                         1, 1, 1, 1, 1, 1, /*force_direct=*/true);
        const bool bits = t4_half_range_equal(got, 0, ref, 0, static_cast<size_t>(N) * Cout * Hout * Wout);
        const bool path = (d.wrapper == 0) && (d.arena == 0);
        t4_print_delta("G5:direct_N2_Cout7", d);
        std::cout << "[G5:direct_N2_Cout7] launch_conv2d vs forced direct memcmp "
                  << (bits ? "BIT-EXACT" : "MISMATCH")
                  << " / batched wrapper +0 and arena alloc +0: " << (path ? "yes" : "NO")
                  << " -> " << ((bits && path) ? "PASSED" : "FAILED") << "\n";
        ok = bits && path && ok;
    }

    if (!ok)
    {
        std::cerr << "[test_g10k_t4_gates] FAILED\n";
    }
    return ok;
}

// ----------------------------------------------------------------
// 9d. G-8k S1b: im2col 中間バッファのアリーナ化 bit-exact ゲート。
//
//   配管 (cudaMalloc → bump アリーナ) だけの変更なので、出力は完全不変であるべき。
//   ここでは 3 つの独立した性質をハードゲートにする:
//     (1) 同一設定 3 runs の自己一致 (memcmp)。単発比較では「たまたま同じ」を排除できない。
//     (2) **アリーナ残留値の非依存性**: run の間にアリーナ領域を 0xFF / 0x00 で汚染して
//         から rewind し、次の run が同じ領域を再利用するよう仕向ける。ここで出力が
//         変われば「未初期化領域を読んでいた既存バグ」の発覚であり、ゼロ埋めで
//         通してはならない (即報告)。
//     (3) DOLLAMA_POOL=0 (素の cudaMalloc/cudaFree) との突合。プロセス単位の
//         キルスイッチのため、出力バイト列をファイルへダンプし外部で cmp する
//         (環境変数 DOLLAMA_G8K_DUMP=<接頭辞> が設定されているときだけ書く)。
//
//   形状は G-4k S2 と同じ 6 形状 (B1/B2 × 320ch/128² 640ch/64² 1280ch/32²) +
//   帯分割 (d_out_band) を踏む VAE 相当 1 形状。
// ----------------------------------------------------------------

// アリーナ領域を汚染してから rewind する (次の確保が同じ領域を再利用する)。
static void poison_arena(size_t bytes, int pattern)
{
    if (bytes == 0)
    {
        return;
    }
    DeviceArenaScope sc(DeviceArenaId::UNet);
    void* p = sc.alloc_bytes(bytes);
    if (p != nullptr)
    {
        CUDA_CHECK(cudaMemset(p, pattern, bytes));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

// バイト列をファイルへダンプ (DOLLAMA_G8K_DUMP 接頭辞が設定されているときのみ)。
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
    std::string path = std::string(prefix) + "_" + label + ".bin";
    std::ofstream ofs(path.c_str(), std::ios::binary);
    if (!ofs)
    {
        std::cerr << "[g8k] ダンプ失敗: " << path << "\n";
        return;
    }
    ofs.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
}

static bool run_case_g8k_arena(const char* label, int N, int Cin, int H, int W, int Cout,
                               unsigned seed)
{
    const int KH = 3, KW = 3, stride = 1, pad = 1, dil = 1;
    const int Hout = out_dim(H, pad, dil, KH, stride);
    const int Wout = out_dim(W, pad, dil, KW, stride);

    const size_t in_n  = static_cast<size_t>(N) * Cin * H * W;
    const size_t w_n   = static_cast<size_t>(Cout) * Cin * KH * KW;
    const size_t out_n = static_cast<size_t>(N) * Cout * Hout * Wout;

    HalfBuffer in = make_half(static_cast<int>(in_n), seed);
    HalfBuffer w  = make_half(static_cast<int>(w_n), seed + 1, -0.2f, 0.2f);
    HalfBuffer bb = make_half(Cout, seed + 2, -0.5f, 0.5f);

    __half* d_in     = nullptr;
    __half* d_weight = nullptr;
    __half* d_bias   = nullptr;
    __half* d_out    = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, in_n * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_weight, w_n * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_bias, static_cast<size_t>(Cout) * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_out, out_n * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_in, in.h.data(), in_n * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, w.h.data(), w_n * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, bb.h.data(), static_cast<size_t>(Cout) * sizeof(__half),
                          cudaMemcpyHostToDevice));

    // 汚染サイズ = この形状が使う im2col バッファ相当 (実際に同じ領域が再利用される)。
    // 帯分割の有無に関わらず、col バッファは最大 256MiB (IM2COL_TILE_BYTES) 以下。
    size_t poison_bytes = static_cast<size_t>(Cin) * KH * KW
                          * static_cast<size_t>(Hout) * Wout * sizeof(__half);
    const size_t poison_cap = (size_t)256 << 20;
    if (poison_bytes > poison_cap)
    {
        poison_bytes = poison_cap;
    }

    const DeviceArenaStats st0 = device_arena_stats(DeviceArenaId::UNet);

    std::vector<std::vector<__half>> runs(3);
    const int patterns[3] = {-1, 0xFF, 0x00}; // -1 = 汚染なし
    for (int r = 0; r < 3; ++r)
    {
        if (patterns[r] >= 0)
        {
            poison_arena(poison_bytes, patterns[r]);
        }
        // 出力バッファも毎回汚しておく (書き残しがあれば検出できる)。
        CUDA_CHECK(cudaMemset(d_out, 0xCD, out_n * sizeof(__half)));
        launch_conv2d(d_in, d_weight, d_bias, d_out, N, Cin, H, W, Cout, KH, KW,
                      stride, stride, pad, pad, dil, dil);
        CUDA_CHECK(cudaDeviceSynchronize());
        runs[r].resize(out_n);
        CUDA_CHECK(cudaMemcpy(runs[r].data(), d_out, out_n * sizeof(__half),
                              cudaMemcpyDeviceToHost));
    }

    const DeviceArenaStats st1 = device_arena_stats(DeviceArenaId::UNet);

    const size_t nbytes = out_n * sizeof(__half);
    const bool eq01 = (std::memcmp(runs[0].data(), runs[1].data(), nbytes) == 0);
    const bool eq02 = (std::memcmp(runs[0].data(), runs[2].data(), nbytes) == 0);
    const bool ok   = eq01 && eq02;

    std::cout << "[g8k_arena] " << label
              << " N=" << N << " C=" << Cin << " " << H << "x" << W
              << " run0==run1(0xFF poisoned): " << (eq01 ? "BIT-EXACT" : "DIFF")
              << " / run0==run2(0x00 poisoned): " << (eq02 ? "BIT-EXACT" : "DIFF")
              << " | pool=" << (device_arena_pool_enabled() ? "on" : "off")
              << " cudaMalloc +" << (st1.cuda_malloc_calls - st0.cuda_malloc_calls)
              << " cudaFree +" << (st1.cuda_free_calls - st0.cuda_free_calls)
              << " alloc +" << (st1.alloc_calls - st0.alloc_calls)
              << " chunks=" << st1.live_chunks
              << " cap=" << (st1.total_capacity >> 20) << "MiB\n";

    dump_bytes(label, runs[0].data(), nbytes);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_weight));
    CUDA_CHECK(cudaFree(d_bias));
    CUDA_CHECK(cudaFree(d_out));
    return ok;
}

static bool test_g8k_arena_bitexact()
{
    bool ok = true;
    // G-4k S2 と同じ 6 形状 (resnet 代表)。
    ok = run_case_g8k_arena("b1_320_128",  1, 320, 128, 128, 320, 3101) && ok;
    ok = run_case_g8k_arena("b2_320_128",  2, 320, 128, 128, 320, 3102) && ok;
    ok = run_case_g8k_arena("b1_640_64",   1, 640,  64,  64, 640, 3103) && ok;
    ok = run_case_g8k_arena("b2_640_64",   2, 640,  64,  64, 640, 3104) && ok;
    ok = run_case_g8k_arena("b1_1280_32",  1, 1280, 32,  32, 1280, 3105) && ok;
    ok = run_case_g8k_arena("b2_1280_32",  2, 1280, 32,  32, 1280, 3106) && ok;
    // 帯分割 (d_out_band + scatter) を踏む VAE 相当形状。
    ok = run_case_g8k_arena("b1_128_512",  1, 128, 512, 512, 128, 3107) && ok;

    const DeviceArenaStats st = device_arena_stats(DeviceArenaId::UNet);
    std::cout << "[g8k_arena] 累計: cudaMalloc=" << st.cuda_malloc_calls
              << " cudaFree=" << st.cuda_free_calls
              << " alloc=" << st.alloc_calls
              << " chunk_alloc=" << st.chunk_alloc_calls
              << " chunks=" << st.live_chunks
              << " capacity=" << (st.total_capacity >> 20) << "MiB"
              << " peak_in_use=" << (st.peak_bytes_in_use >> 20) << "MiB\n";
    if (!ok)
    {
        // 未初期化領域の読み出し疑い。ゼロ埋めで通してはならない (即エスカレーション)。
        std::cerr << "[g8k_arena] FAILED: arena reuse changed the output\n";
    }
    return ok;
}

// ----------------------------------------------------------------
// 9c. warm ベンチ: N=2 バッチ GEMM (1 forward) vs N=1 を 2 回逐次呼び。
//    per-call ms (中央値) を報告。CFG B=2 束ねの launch オーバーヘッド低減を観測する。
//    ★既定プロセス (DOLLAMA_CONV_BATCH 未設定) では batched 経路自体が per-n 直列に
//      フォールバックするため、この比は 1.0 付近になり T6 としての意味を持たない。
//      T6 の意味を持つ計測は `conv2d_batch_on` (DOLLAMA_CONV_BATCH=1) 側のみ。
// ----------------------------------------------------------------
static void bench_batch_vs_persample(int Cin, int H, int W, int Cout, int KH, int KW,
                                     int stride_h, int stride_w, int pad_h, int pad_w,
                                     const char* label)
{
    const int Hout = out_dim(H, pad_h, 1, KH, stride_h);
    const int Wout = out_dim(W, pad_w, 1, KW, stride_w);
    const int per_in  = Cin * H * W;
    const int per_out = Cout * Hout * Wout;
    const int w_n     = Cout * Cin * KH * KW;

    HalfBuffer in = make_half(2 * per_in, 5551);
    HalfBuffer w  = make_half(w_n, 5552);
    HalfBuffer bb = make_half(Cout, 5553, -0.5f, 0.5f);

    __half *d_in = nullptr, *d_weight = nullptr, *d_bias = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, static_cast<size_t>(2) * per_in * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_weight, static_cast<size_t>(w_n) * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_bias, static_cast<size_t>(Cout) * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_out, static_cast<size_t>(2) * per_out * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_in, in.h.data(), static_cast<size_t>(2) * per_in * sizeof(__half),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, w.h.data(), static_cast<size_t>(w_n) * sizeof(__half),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, bb.h.data(), static_cast<size_t>(Cout) * sizeof(__half),
                          cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    auto time_ms = [&](bool batched) -> float
    {
        CUDA_CHECK(cudaEventRecord(start));
        if (batched)
        {
            launch_conv2d(d_in, d_weight, d_bias, d_out, 2, Cin, H, W, Cout, KH, KW,
                          stride_h, stride_w, pad_h, pad_w, 1, 1);
        }
        else
        {
            launch_conv2d(d_in, d_weight, d_bias, d_out, 1, Cin, H, W, Cout, KH, KW,
                          stride_h, stride_w, pad_h, pad_w, 1, 1);
            launch_conv2d(d_in + per_in, d_weight, d_bias, d_out + per_out,
                          1, Cin, H, W, Cout, KH, KW,
                          stride_h, stride_w, pad_h, pad_w, 1, 1);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    };

    const int warmup = 3, iters = 50;
    for (int i = 0; i < warmup; ++i) { (void)time_ms(true); (void)time_ms(false); }
    std::vector<float> tb, ts;
    tb.reserve(iters); ts.reserve(iters);
    for (int i = 0; i < iters; ++i) { tb.push_back(time_ms(true)); ts.push_back(time_ms(false)); }
    std::sort(tb.begin(), tb.end());
    std::sort(ts.begin(), ts.end());
    std::cout << "[bench_batch] " << label
              << " N=2 batch median=" << tb[tb.size() / 2] << " ms"
              << " | N=1x2 seq median=" << ts[ts.size() / 2] << " ms\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_weight));
    CUDA_CHECK(cudaFree(d_bias));
    CUDA_CHECK(cudaFree(d_out));
}

// ----------------------------------------------------------------
// ベンチ: conv は積和律速 → GFLOPS と有効帯域 (GB/s) を報告。
// FLOPs ≈ 2 * N*Cout*Hout*Wout * Cin*KH*KW (積 + 和)。
// bytes ≈ (in + weight + out) * 2byte (おおまかな読み書き総量)。
// warmup3 / iters100 / cudaEvent 中央値。
// ----------------------------------------------------------------
static void bench_one(int N, int Cin, int H, int W, int Cout, int KH, int KW,
                      int stride_h, int stride_w, int pad_h, int pad_w,
                      int dilation_h, int dilation_w, const char* label)
{
    const int Hout = out_dim(H, pad_h, dilation_h, KH, stride_h);
    const int Wout = out_dim(W, pad_w, dilation_w, KW, stride_w);
    const int in_n  = N * Cin * H * W;
    const int w_n   = Cout * Cin * KH * KW;
    const size_t out_n = static_cast<size_t>(N) * Cout * Hout * Wout;

    HalfBuffer in = make_half(in_n, 7777);
    HalfBuffer w  = make_half(w_n, 8888);
    HalfBuffer bb = make_half(Cout, 9999, -0.5f, 0.5f);

    __half* d_in     = nullptr;
    __half* d_weight = nullptr;
    __half* d_bias   = nullptr;
    __half* d_out    = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, static_cast<size_t>(in_n) * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_weight, static_cast<size_t>(w_n) * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_bias, static_cast<size_t>(Cout) * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_out, out_n * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_in, in.h.data(), static_cast<size_t>(in_n) * sizeof(__half),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, w.h.data(), static_cast<size_t>(w_n) * sizeof(__half),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, bb.h.data(), static_cast<size_t>(Cout) * sizeof(__half),
                          cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    auto run_once = [&]() -> float
    {
        CUDA_CHECK(cudaEventRecord(start));
        launch_conv2d(d_in, d_weight, d_bias, d_out, N, Cin, H, W, Cout, KH, KW,
                      stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    };

    const int warmup = 3;
    const int iters  = 100;
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

    const double flops = 2.0 * static_cast<double>(out_n)
                         * static_cast<double>(Cin) * KH * KW;
    const double gflops = flops / (static_cast<double>(median_ms) * 1e-3) / 1e9;
    const double bytes = (static_cast<double>(in_n) + static_cast<double>(w_n)
                          + static_cast<double>(out_n)) * sizeof(__half);
    const double gbps  = bytes / (static_cast<double>(median_ms) * 1e-3) / 1e9;

    std::cout << "[bench_conv2d] " << label
              << " N=" << N << " Cin=" << Cin << " H=" << H << " W=" << W
              << " Cout=" << Cout << " K=" << KH << "x" << KW
              << " s=" << stride_h << " p=" << pad_h
              << " -> " << Hout << "x" << Wout
              << " median=" << median_ms << " ms"
              << " (min=" << times.front() << " max=" << times.back() << ")"
              << " " << gflops << " GFLOPS"
              << " BW=" << gbps << " GB/s (N=" << iters << ")\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_weight));
    CUDA_CHECK(cudaFree(d_bias));
    CUDA_CHECK(cudaFree(d_out));
}

static void bench_conv2d()
{
    // UNet 系: C320 64x64 3x3 same conv (im2col+GEMM 経路)。
    bench_one(1, 320, 64, 64, 320, 3, 3, 1, 1, 1, 1, 1, 1, "unet_c320_64");
    // VAE 系: C128 512x512 3x3 same conv (im2col+GEMM 経路・帯分割が効く)。
    bench_one(1, 128, 512, 512, 128, 3, 3, 1, 1, 1, 1, 1, 1, "vae_c128_512");
    // 1x1 GEMM 経路の代表: UNet ResBlock skip 相当 (Cin=320 -> Cout=640, 32x32)。
    bench_one(1, 320, 32, 32, 640, 1, 1, 1, 1, 0, 0, 1, 1, "unet_1x1_320to640");
    // VAE 1x1: Cin=512 -> Cout=512, 64x64。
    bench_one(1, 512, 64, 64, 512, 1, 1, 1, 1, 0, 0, 1, 1, "vae_1x1_512_64");
    // G-2k S1: CFG B=2 束ねの効果観測 (UNet C320 64x64 3x3 same)。
    bench_batch_vs_persample(320, 64, 64, 320, 3, 3, 1, 1, 1, 1, "unet_c320_64");
    // G-10k T4 (T6 の計器): 代表形状 3 つ (§5) + [G4] の帯分割形状。
    //   T6 の合否 = 全形状で batched median / seq median <= 0.95 (判定は T6 側・ここは計器のみ)。
    //   ★per-call ms であり e2e 秒には翻訳できない。
    bench_batch_vs_persample(320, 128, 128, 320, 3, 3, 1, 1, 1, 1, "rep_320_128");
    bench_batch_vs_persample(640, 64, 64, 640, 3, 3, 1, 1, 1, 1, "rep_640_64");
    bench_batch_vs_persample(1280, 32, 32, 1280, 3, 3, 1, 1, 1, 1, "rep_1280_32");
    bench_batch_vs_persample(640, 128, 128, 320, 3, 3, 1, 1, 1, 1, "G4_band_640to320_128");
}

#endif // HAVE_CUDA

} // namespace dollama

int main()
{
#ifndef HAVE_CUDA
    std::cout << "[test_conv2d] [SKIP] HAVE_CUDA 未定義\n";
    return 0;
#else
    bool ok = true;
    ok = dollama::test_conv_1x1()         && ok;
    ok = dollama::test_conv_3x3_s1p1()    && ok;
    ok = dollama::test_conv_3x3_s2p1()    && ok;
    ok = dollama::test_conv_bias()        && ok;
    ok = dollama::test_conv_batch_multi() && ok;
    ok = dollama::test_conv_dilation()    && ok;
    ok = dollama::test_conv_gemm_path()   && ok;
    ok = dollama::test_conv_gemm_large()  && ok;
    ok = dollama::test_conv_batch_gemm()  && ok;
    ok = dollama::test_g8k_arena_bitexact() && ok;
    ok = dollama::test_g10k_t4_gates()    && ok;

    if (!ok)
    {
        std::cerr << "[test_conv2d] FAILED\n";
        return 1;
    }

    dollama::bench_conv2d();

    std::cout << "[test_conv2d] ALL PASSED\n";
    return 0;
#endif
}

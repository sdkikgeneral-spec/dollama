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
#include <iostream>
#include <random>
#include <vector>

#ifdef HAVE_CUDA
#include <cuda_fp16.h>
#include "kernels/conv2d.cuh"
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

// device 経由で conv2d を実行し FP16 結果をデコードして返す。
// force_direct=true なら GEMM 経路をバイパスして必ず direct conv で計算する
// (大形状の GPU-vs-direct 突合の「参照」側に使う)。
// bias_h が空のときは nullptr で起動する。
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

    std::vector<float> out(out_n);
    for (size_t i = 0; i < out_n; ++i)
    {
        out[i] = __half2float(h_out[i]);
    }
    return out;
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

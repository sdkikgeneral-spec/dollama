// 計測専用ドライバ (本番 bitnet.hpp は無改変・public API 経由)。
//
// Tier 1 速度最適化の効果測定:
//   double 参照経路 forward() と float32+AVX2 高速パス forward_fast() を
//   seq8 / seq32 / seq63 で end-to-end 計時し、中央値と倍率 (double 比) を出す。
//
// 本番 bitnet.hpp の積和ロジックには一切触らず、public API
//   - BitNetDenseInfer::forward(tokens)             … double 参照経路
//   - BitNetDenseInfer::forward_fast(tokens, false) … float32+AVX2 高速パス
// だけを呼んで計時する (区間分解はしない・end-to-end のみ)。
//
// WEIGHTS_PATH は meson の -D 埋め込み (既定は data/bitnet/bitnet_dense_fp32.safetensors)。
// 重み不在時は健全に skip メッセージを出して終了する (CI でビルド緑維持)。

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "infer/bitnet.hpp"

#ifndef WEIGHTS_PATH
#define WEIGHTS_PATH "E:/Projects/dollama/data/bitnet/bitnet_dense_fp32.safetensors"
#endif

using namespace dollama;
using Clock = std::chrono::steady_clock;
using std::chrono::duration;

static double ms_since(Clock::time_point a, Clock::time_point b)
{
    return duration<double, std::milli>(b - a).count();
}

static double median(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

static bool file_exists(const std::string& path)
{
    if (path.empty())
    {
        return false;
    }
    std::ifstream f(path, std::ios::binary);
    return static_cast<bool>(f);
}

int main()
{
    const std::string wpath = WEIGHTS_PATH;
    if (!file_exists(wpath))
    {
        std::cout << "[prof_bitnet] SKIP: weights 不在 (" << wpath << ")\n";
        return 0;  // 健全に終了 (ビルド緑維持)
    }

    std::cout << "[prof_bitnet] weights: " << wpath << "\n";
    BitNetDenseInfer model(wpath);

    constexpr int V = BitNetDenseInfer::VOCAB_SIZE;
    const int seqs[3] = {8, 32, 63};
    const int WARMUP = 3;
    const int ITERS = 15;

    std::cout << "seq,forward_double_ms,forward_fast_ms,speedup\n";

    for (int si = 0; si < 3; ++si)
    {
        const int S = seqs[si];
        std::vector<int> tokens(static_cast<size_t>(S));
        for (int i = 0; i < S; ++i)
        {
            tokens[static_cast<size_t>(i)] = (i * 37 + 5) % V;  // 適当な合法 token id
        }

        // warmup
        for (int w = 0; w < WARMUP; ++w)
        {
            volatile float a = model.forward(tokens)[0];
            volatile float b = model.forward_fast(tokens, false)[0];
            (void)a; (void)b;
        }

        // double 参照経路。
        std::vector<double> dv;
        dv.reserve(static_cast<size_t>(ITERS));
        for (int it = 0; it < ITERS; ++it)
        {
            auto t0 = Clock::now();
            volatile float sink = model.forward(tokens)[0];
            auto t1 = Clock::now();
            (void)sink;
            dv.push_back(ms_since(t0, t1));
        }

        // float32 + AVX2 高速パス。
        std::vector<double> fv;
        fv.reserve(static_cast<size_t>(ITERS));
        for (int it = 0; it < ITERS; ++it)
        {
            auto t0 = Clock::now();
            volatile float sink = model.forward_fast(tokens, false)[0];
            auto t1 = Clock::now();
            (void)sink;
            fv.push_back(ms_since(t0, t1));
        }

        const double md = median(dv);
        const double mf = median(fv);
        std::cout << S << ","
                  << md << ","
                  << mf << ","
                  << (mf > 0.0 ? md / mf : 0.0) << "\n";
    }

    return 0;
}

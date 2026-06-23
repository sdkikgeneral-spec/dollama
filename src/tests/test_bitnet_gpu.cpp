// BitNetGpuInfer 単体テスト — Phase 4 #6-GPU / T2。
//
// 対象: src/infer/bitnet_gpu.cuh (class BitNetGpuInfer) — device 常駐 + GPU forward。
//
// 突合戦略 (CPU 版を golden 代わりに使う・PyTorch golden は不要):
//   同一の synthetic 重み (bitnet_dense_fp32.safetensors) で CPU 版 BitNetDenseInfer と
//   GPU 版 BitNetGpuInfer を構築し、複数の token 列で forward の logits を突合する
//   (max_abs_err < 1e-3 かつ corr >= 0.99999)。さらに generate の生成 id 列が
//   CPU と完全一致することを検証する。identity 重みがあれば
//   generate_with_identity も CPU 版と完全一致を確認する。
//
// 重み (data/bitnet/*.safetensors) / vocab.json は gitignore 対象 (再生成可)。
// いずれか不在 (or 例外) なら [SKIP] で 0 終了 (CI でビルド緑維持・GPU 不在環境も同様)。
//
// ファイルは .cpp 拡張子だが BitNetGpuInfer が cudaMalloc 等を呼ぶため
// meson 側で nvcc (cuda 言語) ではなく cpp としてビルドされると CUDA ランタイム
// シンボルが解決できない。test_bitnet_gpu は infer/bitnet_gpu.cu を同時にリンクし、
// cuda_dep (cudart/cublas) を依存に取る (meson.build 参照)。

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "infer/bitnet.hpp"       // CPU 版 (golden 代わり)
#include "infer/bitnet_gpu.cuh"   // GPU 版 (被テスト)
#include "io/tokenizer.hpp"

#ifndef WEIGHTS_PATH
#define WEIGHTS_PATH ""
#endif
#ifndef IDENTITY_WEIGHTS_PATH
#define IDENTITY_WEIGHTS_PATH ""
#endif
#ifndef VOCAB_PATH
#define VOCAB_PATH ""
#endif

namespace dollama
{

static bool file_exists(const std::string& path)
{
    if (path.empty())
    {
        return false;
    }
    std::ifstream f(path, std::ios::binary);
    return static_cast<bool>(f);
}

// CPU/GPU の logits を突合し max_abs_err / corr を判定する。
static bool compare_logits(const std::vector<float>& cpu,
                           const std::vector<float>& gpu,
                           int seq_len)
{
    if (cpu.size() != gpu.size())
    {
        std::cerr << "[test_bitnet_gpu] FAIL seq=" << seq_len
                  << " size mismatch cpu=" << cpu.size()
                  << " gpu=" << gpu.size() << "\n";
        return false;
    }

    double max_abs = 0.0;
    double sx = 0.0, sy = 0.0, sxx = 0.0, syy = 0.0, sxy = 0.0;
    const size_t n = cpu.size();
    for (size_t i = 0; i < n; ++i)
    {
        const double a = static_cast<double>(gpu[i]);
        const double b = static_cast<double>(cpu[i]);
        const double d = std::fabs(a - b);
        if (d > max_abs)
        {
            max_abs = d;
        }
        sx += a;
        sy += b;
        sxx += a * a;
        syy += b * b;
        sxy += a * b;
    }
    const double nn = static_cast<double>(n);
    const double cov = sxy - sx * sy / nn;
    const double vx  = sxx - sx * sx / nn;
    const double vy  = syy - sy * sy / nn;
    const double corr = cov / std::sqrt(vx * vy);

    std::cout << "[test_bitnet_gpu] seq=" << seq_len
              << " max_abs_err=" << max_abs << " corr=" << corr << "\n";

    if (max_abs >= 1e-3)
    {
        std::cerr << "[test_bitnet_gpu] FAIL seq=" << seq_len
                  << " max_abs_err " << max_abs << " >= 1e-3\n";
        return false;
    }
    if (!(corr >= 0.99999))
    {
        std::cerr << "[test_bitnet_gpu] FAIL seq=" << seq_len
                  << " corr " << corr << " < 0.99999\n";
        return false;
    }
    return true;
}

// 範囲内の決定的な token 列を生成する (seq 長 sl)。
static std::vector<int> make_tokens(int sl)
{
    std::vector<int> ids(static_cast<size_t>(sl));
    // 単純な決定的疑似乱数 (LCG) で VOCAB_SIZE 内に収める。
    uint32_t s = 0x1234567u + static_cast<uint32_t>(sl) * 2654435761u;
    for (int i = 0; i < sl; ++i)
    {
        s = s * 1664525u + 1013904223u;
        ids[static_cast<size_t>(i)] =
            static_cast<int>(s % static_cast<uint32_t>(BitNetDenseInfer::VOCAB_SIZE));
    }
    return ids;
}

// ── forward logits 突合 (CPU vs GPU) ───────────────────────────────
static bool test_forward_match()
{
    if (!file_exists(WEIGHTS_PATH))
    {
        std::cout << "[test_forward_match] SKIP (weights 不在)\n";
        return true;
    }

    BitNetDenseInfer cpu(WEIGHTS_PATH);
    BitNetGpuInfer   gpu(WEIGHTS_PATH);

    const int seq_lens[3] = {8, 32, 63};
    for (int sl : seq_lens)
    {
        const std::vector<int> ids = make_tokens(sl);
        const std::vector<float> lc = cpu.forward(ids);
        const std::vector<float> lg = gpu.forward(ids);
        if (!compare_logits(lc, lg, sl))
        {
            return false;
        }
    }
    std::cout << "[test_forward_match] PASSED\n";
    return true;
}

// ── generate 完全一致 (CPU vs GPU) ─────────────────────────────────
static bool ids_equal(const std::vector<int>& a, const std::vector<int>& b)
{
    if (a.size() != b.size())
    {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i)
    {
        if (a[i] != b[i])
        {
            return false;
        }
    }
    return true;
}

static void dump_ids(const char* label, const std::vector<int>& v)
{
    std::cerr << "  " << label << ":";
    for (int id : v)
    {
        std::cerr << " " << id;
    }
    std::cerr << "\n";
}

static bool test_generate_match()
{
    if (!file_exists(WEIGHTS_PATH))
    {
        std::cout << "[test_generate_match] SKIP (weights 不在)\n";
        return true;
    }

    BitNetDenseInfer cpu(WEIGHTS_PATH);
    BitNetGpuInfer   gpu(WEIGHTS_PATH);

    // 複数の短い prompt から greedy 生成し CPU と完全一致を確認する。
    const std::vector<std::vector<int>> prompts = {
        {TOK_BOS, 5, 6, 7, TOK_SEP},
        {TOK_BOS, 10, 20, 30, 40, TOK_SEP},
        {TOK_BOS, 100, TOK_SEP},
    };
    for (size_t k = 0; k < prompts.size(); ++k)
    {
        const std::vector<int> gc = cpu.generate(prompts[k]);
        const std::vector<int> gg = gpu.generate(prompts[k]);
        if (!ids_equal(gc, gg))
        {
            std::cerr << "[test_generate_match] FAIL case " << k
                      << " (cpu len=" << gc.size() << " gpu len=" << gg.size() << ")\n";
            dump_ids("cpu", gc);
            dump_ids("gpu", gg);
            return false;
        }
        std::cout << "[test_generate_match] case " << k
                  << " matched (" << gg.size() << " tokens)\n";
    }
    std::cout << "[test_generate_match] PASSED\n";
    return true;
}

// ── generate_with_identity 完全一致 (identity 重みがあれば) ─────────
static bool test_identity_match()
{
    if (!file_exists(IDENTITY_WEIGHTS_PATH) || !file_exists(VOCAB_PATH))
    {
        std::cout << "[test_identity_match] SKIP (identity weights/vocab 不在)\n";
        return true;
    }

    BitNetDenseInfer cpu(IDENTITY_WEIGHTS_PATH);
    BitNetGpuInfer   gpu(IDENTITY_WEIGHTS_PATH);
    Tokenizer        tok(VOCAB_PATH);

    struct Case
    {
        std::vector<std::string> identity_tags;
        std::string              scene_text;
    };
    const Case cases[2] = {
        { {"1girl", "short hair", "black hair", "red eyes", "animal ears"},
          "Please draw a girl: short hair, black hair, red eyes, animal ears." },
        { {"1girl", "breasts", "brown hair", "medium hair", "pink eyes"},
          "a girl having breasts, brown hair, medium hair, pink eyes." },
    };

    for (int k = 0; k < 2; ++k)
    {
        // prompt 組み立ても CPU/GPU で一致するはず (純 host ロジック)。
        const std::vector<int> pc =
            cpu.build_identity_prompt(tok, cases[k].identity_tags, cases[k].scene_text);
        const std::vector<int> pg =
            gpu.build_identity_prompt(tok, cases[k].identity_tags, cases[k].scene_text);
        if (!ids_equal(pc, pg))
        {
            std::cerr << "[test_identity_match] FAIL case " << k << " prompt mismatch\n";
            dump_ids("cpu", pc);
            dump_ids("gpu", pg);
            return false;
        }

        const std::vector<int> gc =
            cpu.generate_with_identity(tok, cases[k].identity_tags, cases[k].scene_text);
        const std::vector<int> gg =
            gpu.generate_with_identity(tok, cases[k].identity_tags, cases[k].scene_text);
        if (!ids_equal(gc, gg))
        {
            std::cerr << "[test_identity_match] FAIL case " << k
                      << " gen mismatch (cpu len=" << gc.size()
                      << " gpu len=" << gg.size() << ")\n";
            dump_ids("cpu", gc);
            dump_ids("gpu", gg);
            return false;
        }
        std::cout << "[test_identity_match] case " << k
                  << " prompt+gen matched (" << gg.size() << " tokens)\n";
    }
    std::cout << "[test_identity_match] PASSED\n";
    return true;
}

// ── 速度計測 (env DOLLAMA_BENCH=1 のときのみ・pass/fail には影響しない) ──
// CPU 版 BitNetDenseInfer と GPU 版 BitNetGpuInfer の forward レイテンシ中央値を
// seq 8/32/63 で比較する。forward 内で cudaDeviceSynchronize 済みなので
// wall-clock で GPU の実時間を計測できる。
static double median_ms(std::vector<double>& v)
{
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n == 0 ? 0.0
                  : (n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]));
}

static void run_bench()
{
    if (!file_exists(WEIGHTS_PATH))
    {
        std::cout << "[bench] SKIP (weights 不在)\n";
        return;
    }
    using clk = std::chrono::steady_clock;

    BitNetDenseInfer cpu(WEIGHTS_PATH);
    BitNetGpuInfer   gpu(WEIGHTS_PATH);

    const int seq_lens[3] = {8, 32, 63};
    const int warmup = 3;
    const int iters  = 30;

    std::cout << "[bench] forward latency median (warmup=" << warmup
              << " iters=" << iters << ")\n";
    std::cout << "[bench]   seq |   CPU ms |   GPU ms | speedup\n";
    for (int sl : seq_lens)
    {
        const std::vector<int> ids = make_tokens(sl);

        for (int w = 0; w < warmup; ++w)
        {
            (void)cpu.forward(ids);
            (void)gpu.forward(ids);
        }

        std::vector<double> tc, tg;
        tc.reserve(iters);
        tg.reserve(iters);
        for (int it = 0; it < iters; ++it)
        {
            auto a0 = clk::now();
            (void)cpu.forward(ids);
            auto a1 = clk::now();
            (void)gpu.forward(ids);
            auto a2 = clk::now();
            tc.push_back(std::chrono::duration<double, std::milli>(a1 - a0).count());
            tg.push_back(std::chrono::duration<double, std::milli>(a2 - a1).count());
        }
        const double mc = median_ms(tc);
        const double mg = median_ms(tg);
        std::printf("[bench]  %4d | %8.2f | %8.2f | %6.2fx\n",
                    sl, mc, mg, mc / mg);
    }
}

} // namespace dollama

int main()
{
    bool ok = true;
    try
    {
        ok = dollama::test_forward_match() && ok;
        ok = dollama::test_generate_match() && ok;
        ok = dollama::test_identity_match() && ok;
        if (std::getenv("DOLLAMA_BENCH"))
        {
            dollama::run_bench();
        }
    }
    catch (const std::exception& e)
    {
        // GPU 不在 / cudaMalloc 失敗等は [SKIP] 扱い (CI ビルド緑維持)。
        std::cout << "[test_bitnet_gpu] SKIP (exception: " << e.what() << ")\n";
        return 0;
    }

    if (!ok)
    {
        std::cerr << "[test_bitnet_gpu] FAILED\n";
        return 1;
    }
    std::cout << "[test_bitnet_gpu] ALL PASSED\n";
    return 0;
}

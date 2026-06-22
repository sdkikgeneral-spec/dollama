// BitNetDenseInfer 単体テスト — Phase 4 dense #6 (C++ dense 推論の数値突合)
//
// 対象: src/infer/bitnet.hpp (dense FP32 forward + greedy デコード)。
// golden は PyTorch (scripts/train_bitnet.py --dump-golden) が出力した safetensors:
//   - logits_golden.safetensors : input_ids_s{8,32,63} / logits_s{8,32,63}
//   - gen_golden.safetensors    : prompt_ids_c{0..4} / gen_ids_c{0..4}
//   - 重み: bitnet_dense_fp32.safetensors (embed/layers.*/final_norm, 全 F32)
//
// パスは meson の -D 埋め込み (WEIGHTS_PATH / LOGITS_GOLDEN_PATH / GEN_GOLDEN_PATH)。
// golden / 重みは gitignore 対象なので、いずれか不在なら該当テストを [SKIP] する
// (VAE/UNet golden テストの skip 規約に倣う・CI で golden 不在でもビルド緑)。
//
// golden 生成手順:
//   python scripts/train_bitnet.py --dump-golden
//   → data/bitnet/bitnet_dense_fp32.safetensors と data/bitnet/golden/*.safetensors を作る。
//
// 受け入れ条件:
//   1. 各 input_ids を C++ dense forward に通し、全 logits を golden と突合:
//        max abs error < 1e-3 かつ相関 >= 0.99999。
//   2. prompt から greedy 生成した id 列が gen_ids と完全一致 (5 ケース全て)。

#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "infer/bitnet.hpp"
#include "io/safetensors.hpp"

#ifndef WEIGHTS_PATH
#define WEIGHTS_PATH ""
#endif
#ifndef LOGITS_GOLDEN_PATH
#define LOGITS_GOLDEN_PATH ""
#endif
#ifndef GEN_GOLDEN_PATH
#define GEN_GOLDEN_PATH ""
#endif

namespace dollama
{

// ファイルが存在するか (ifstream で開けるか)。
static bool file_exists(const std::string& path)
{
    if (path.empty())
    {
        return false;
    }
    std::ifstream f(path, std::ios::binary);
    return static_cast<bool>(f);
}

// safetensors の I32 テンソルを int ベクタで読む。
static std::vector<int> read_i32(const SafeTensors& st, const std::string& name)
{
    if (st.dtype(name) != StDtype::I32)
    {
        throw std::runtime_error("test_bitnet_infer: '" + name + "' must be I32");
    }
    size_t nbytes = 0;
    const uint8_t* p = st.tensor_bytes(name, nbytes);
    const size_t n = nbytes / 4;
    std::vector<int> out(n);
    const int32_t* ip = reinterpret_cast<const int32_t*>(p);
    for (size_t i = 0; i < n; ++i)
    {
        out[i] = static_cast<int>(ip[i]);
    }
    return out;
}

// safetensors の F32 テンソルを float ベクタで読む。
static std::vector<float> read_f32(const SafeTensors& st, const std::string& name)
{
    if (st.dtype(name) != StDtype::F32)
    {
        throw std::runtime_error("test_bitnet_infer: '" + name + "' must be F32");
    }
    size_t nbytes = 0;
    const uint8_t* p = st.tensor_bytes(name, nbytes);
    const size_t n = nbytes / 4;
    std::vector<float> out(n);
    const float* fp = reinterpret_cast<const float*>(p);
    for (size_t i = 0; i < n; ++i)
    {
        out[i] = fp[i];
    }
    return out;
}

// ── ロジット突合 ───────────────────────────────────────────────────
// 各 seq_len ケースで C++ forward の全 logits を golden と比較する。
static bool test_logits_match()
{
    if (!file_exists(WEIGHTS_PATH) || !file_exists(LOGITS_GOLDEN_PATH))
    {
        std::cout << "[test_logits_match] SKIP (weights/golden 不在)\n";
        return true;
    }

    BitNetDenseInfer model(WEIGHTS_PATH);
    SafeTensors gold(LOGITS_GOLDEN_PATH);

    const int seq_lens[3] = {8, 32, 63};
    for (int sl : seq_lens)
    {
        const std::string suffix = "s" + std::to_string(sl);
        const std::vector<int> ids = read_i32(gold, "input_ids_" + suffix);
        const std::vector<float> g  = read_f32(gold, "logits_" + suffix);

        if (static_cast<int>(ids.size()) != sl)
        {
            std::cerr << "[test_logits_match] FAIL: input_ids_" << suffix
                      << " size " << ids.size() << " != " << sl << "\n";
            return false;
        }

        std::vector<float> logits = model.forward(ids);
        if (logits.size() != g.size())
        {
            std::cerr << "[test_logits_match] FAIL: logits size "
                      << logits.size() << " != golden " << g.size() << "\n";
            return false;
        }

        // max abs error と Pearson 相関を計算 (相関は double 蓄積)。
        double max_abs = 0.0;
        double sx = 0.0, sy = 0.0, sxx = 0.0, syy = 0.0, sxy = 0.0;
        const size_t n = g.size();
        for (size_t i = 0; i < n; ++i)
        {
            const double a = static_cast<double>(logits[i]);
            const double b = static_cast<double>(g[i]);
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

        std::cout << "[test_logits_match] seq=" << sl
                  << " max_abs_err=" << max_abs
                  << " corr=" << corr << "\n";

        if (max_abs >= 1e-3)
        {
            std::cerr << "[test_logits_match] FAIL: seq=" << sl
                      << " max_abs_err " << max_abs << " >= 1e-3\n";
            return false;
        }
        if (!(corr >= 0.99999))
        {
            std::cerr << "[test_logits_match] FAIL: seq=" << sl
                      << " corr " << corr << " < 0.99999\n";
            return false;
        }
    }

    std::cout << "[test_logits_match] PASSED\n";
    return true;
}

// ── greedy デコード突合 ────────────────────────────────────────────
// prompt_ids から生成した id 列が gen_ids と完全一致するか (5 ケース)。
static bool test_greedy_decode()
{
    if (!file_exists(WEIGHTS_PATH) || !file_exists(GEN_GOLDEN_PATH))
    {
        std::cout << "[test_greedy_decode] SKIP (weights/golden 不在)\n";
        return true;
    }

    BitNetDenseInfer model(WEIGHTS_PATH);
    SafeTensors gold(GEN_GOLDEN_PATH);

    for (int ci = 0; ci < 5; ++ci)
    {
        const std::string c = "c" + std::to_string(ci);
        const std::vector<int> prompt = read_i32(gold, "prompt_ids_" + c);
        const std::vector<int> expect = read_i32(gold, "gen_ids_" + c);

        const std::vector<int> got = model.generate(prompt);

        bool ok = (got.size() == expect.size());
        if (ok)
        {
            for (size_t i = 0; i < got.size(); ++i)
            {
                if (got[i] != expect[i])
                {
                    ok = false;
                    break;
                }
            }
        }
        if (!ok)
        {
            std::cerr << "[test_greedy_decode] FAIL case " << ci
                      << " (len got=" << got.size()
                      << " expect=" << expect.size() << ")\n  got   :";
            for (int id : got)
            {
                std::cerr << " " << id;
            }
            std::cerr << "\n  expect:";
            for (int id : expect)
            {
                std::cerr << " " << id;
            }
            std::cerr << "\n";
            return false;
        }
        std::cout << "[test_greedy_decode] case " << ci << " matched ("
                  << got.size() << " tokens)\n";
    }

    std::cout << "[test_greedy_decode] PASSED\n";
    return true;
}

// ── forward 速度の参考計測 (assert 外・情報) ───────────────────────
static bool info_forward_speed()
{
    if (!file_exists(WEIGHTS_PATH))
    {
        std::cout << "[info_forward_speed] SKIP (weights 不在)\n";
        return true;
    }
    BitNetDenseInfer model(WEIGHTS_PATH);
    std::vector<int> ids = {1, 9, 13, 14, 18, 3, 5, 6};  // seq=8

    // warmup
    model.forward(ids);

    const int N = 5;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; ++i)
    {
        volatile float sink = model.forward(ids)[0];
        (void)sink;
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / N;
    std::cout << "[info_forward_speed] seq=8 forward ~" << ms << " ms ("
              << (ms / ids.size()) << " ms/token-pos)\n";
    return true;
}

} // namespace dollama

int main()
{
    bool ok = true;
    try
    {
        ok = dollama::test_logits_match() && ok;
        ok = dollama::test_greedy_decode() && ok;
        ok = dollama::info_forward_speed() && ok;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[test_bitnet_infer] EXCEPTION: " << e.what() << "\n";
        return 1;
    }

    if (!ok)
    {
        std::cerr << "[test_bitnet_infer] FAILED\n";
        return 1;
    }
    std::cout << "[test_bitnet_infer] ALL PASSED\n";
    return 0;
}

// BitNet b1.58 モデル定義 単体テスト + forward ベンチ (純 C++ / CUDA 不要)
// testing.md 形式: 各検証は if (!cond) { cerr; return false; }、
// main で ok = test_xxx() && ok 集約、成功は PASSED / 最後に ALL PASSED。
//
// 検証項目 (4-2 完了条件):
//   - パラメータ数: 30M ≤ params ≤ 100M (実数を標準出力)
//   - ternary 量子化: 既知ベクトルで α と {-1,0,+1} が手計算一致 (全正/全負/0近傍/全0)
//   - int8 absmax 量子化: スケール・round 一致 (全0 ゼロ除算回避・飽和境界)
//   - RMSNorm: 既知入力で手計算一致 (全0 入力でゼロ除算しない)
//   - 決定的 forward: 形状 [seq_len, 4999]・2回実行で完全一致・NaN/Inf なし
//   - 値の健全性: logits 有限・softmax 和1・極端発散なし
//   - 形状/語彙整合: VOCAB_SIZE==4999・embed tied の確認
//   - bench_forward_latency: 代表系列長 (~32 token) で中央値 ms/forward

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "models/bitnet.hpp"

namespace dollama
{

// 浮動小数の近接判定 (絶対許容差)。
static bool approx(double a, double b, double tol)
{
    return std::fabs(a - b) <= tol;
}

// ----------------------------------------------------------------
// パラメータ数: 30M ≤ params ≤ 100M
// ----------------------------------------------------------------
static bool test_param_count()
{
    const size_t p = BitNet::param_count();
    std::cout << "[test_param_count] params = " << p
              << " (" << (static_cast<double>(p) / 1.0e6) << " M)\n";
    std::cout << "[test_param_count] arch: D_MODEL=" << BitNet::D_MODEL
              << " N_LAYERS=" << BitNet::N_LAYERS
              << " N_HEADS=" << BitNet::N_HEADS
              << " HEAD_DIM=" << BitNet::HEAD_DIM
              << " FFN_DIM=" << BitNet::FFN_DIM
              << " MAX_SEQ_LEN=" << BitNet::MAX_SEQ_LEN
              << " VOCAB_SIZE=" << BitNet::VOCAB_SIZE << "\n";

    if (p < 30000000ull)
    {
        std::cerr << "[test_param_count] params が 30M 未満: " << p << "\n";
        return false;
    }
    if (p > 100000000ull)
    {
        std::cerr << "[test_param_count] params が 100M 超過: " << p << "\n";
        return false;
    }
    std::cout << "[test_param_count] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// ternary 重み量子化: α = mean(|W|)、W_q = clamp(round(W/α), -1, 1)
// ----------------------------------------------------------------
static bool test_ternary_quant()
{
    // (1) 全正・既知値: W = {1, 2, 3, 4} → α = 2.5
    //     round(1/2.5)=round(0.4)=0、round(2/2.5)=round(0.8)=1、
    //     round(3/2.5)=round(1.2)=1(clamp済)、round(4/2.5)=round(1.6)=2→clamp 1
    {
        std::vector<float> w = {1.0f, 2.0f, 3.0f, 4.0f};
        TernaryQuant q = quantize_weight_ternary(w.data(), w.size());
        if (!approx(q.scale, 2.5, 1e-6))
        {
            std::cerr << "[test_ternary_quant] (1) scale 不一致: " << q.scale << "\n";
            return false;
        }
        const std::vector<int8_t> want = {0, 1, 1, 1};
        if (q.q != want)
        {
            std::cerr << "[test_ternary_quant] (1) q 不一致\n";
            return false;
        }
    }

    // (2) 全負・対称: W = {-1, -2, -3, -4} → α = 2.5、q = {0,-1,-1,-1}
    {
        std::vector<float> w = {-1.0f, -2.0f, -3.0f, -4.0f};
        TernaryQuant q = quantize_weight_ternary(w.data(), w.size());
        if (!approx(q.scale, 2.5, 1e-6))
        {
            std::cerr << "[test_ternary_quant] (2) scale 不一致: " << q.scale << "\n";
            return false;
        }
        const std::vector<int8_t> want = {0, -1, -1, -1};
        if (q.q != want)
        {
            std::cerr << "[test_ternary_quant] (2) q 不一致\n";
            return false;
        }
    }

    // (3) 0 近傍: W = {0.1, -0.1, 0.0, 0.2} → α = 0.4/4 = 0.1
    //     round(0.1/0.1)=1、round(-0.1/0.1)=-1、round(0)=0、round(0.2/0.1)=2→clamp 1
    {
        std::vector<float> w = {0.1f, -0.1f, 0.0f, 0.2f};
        TernaryQuant q = quantize_weight_ternary(w.data(), w.size());
        if (!approx(q.scale, 0.1, 1e-6))
        {
            std::cerr << "[test_ternary_quant] (3) scale 不一致: " << q.scale << "\n";
            return false;
        }
        const std::vector<int8_t> want = {1, -1, 0, 1};
        if (q.q != want)
        {
            std::cerr << "[test_ternary_quant] (3) q 不一致\n";
            return false;
        }
    }

    // (4) 全 0: α = 0 → ゼロ除算せず scale 0・q 全 0
    {
        std::vector<float> w = {0.0f, 0.0f, 0.0f};
        TernaryQuant q = quantize_weight_ternary(w.data(), w.size());
        if (q.scale != 0.0f)
        {
            std::cerr << "[test_ternary_quant] (4) 全0 で scale が 0 でない: " << q.scale << "\n";
            return false;
        }
        const std::vector<int8_t> want = {0, 0, 0};
        if (q.q != want)
        {
            std::cerr << "[test_ternary_quant] (4) 全0 で q が 0 でない\n";
            return false;
        }
    }

    std::cout << "[test_ternary_quant] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// int8 absmax 量子化: scale = absmax/127、q = clamp(round(x/scale), -127, 127)
// ----------------------------------------------------------------
static bool test_int8_quant()
{
    // (1) 既知値: x = {1, -2, 4, 0.5}、absmax = 4 → scale = 4/127
    //     round(1/(4/127)) = round(31.75) = 32
    //     round(-2/(4/127)) = round(-63.5) = -64 (round half away from zero)
    //     round(4/(4/127)) = 127
    //     round(0.5/(4/127)) = round(15.875) = 16
    {
        std::vector<float> x = {1.0f, -2.0f, 4.0f, 0.5f};
        Int8Quant q = quantize_activation_int8(x.data(), x.size());
        if (!approx(q.scale, 4.0 / 127.0, 1e-9))
        {
            std::cerr << "[test_int8_quant] (1) scale 不一致: " << q.scale << "\n";
            return false;
        }
        // std::round は 0 から遠ざかる方向に丸める (.5 → away from zero)。
        const std::vector<int8_t> want = {32, -64, 127, 16};
        if (q.q != want)
        {
            std::cerr << "[test_int8_quant] (1) q 不一致: ";
            for (auto v : q.q) std::cerr << static_cast<int>(v) << " ";
            std::cerr << "\n";
            return false;
        }
    }

    // (2) 飽和境界: 最大要素は必ず ±127 にマップされる。
    {
        std::vector<float> x = {-7.0f, 3.5f, 7.0f};
        Int8Quant q = quantize_activation_int8(x.data(), x.size());
        if (q.q.front() != -127)
        {
            std::cerr << "[test_int8_quant] (2) 負側最大が -127 でない: "
                      << static_cast<int>(q.q.front()) << "\n";
            return false;
        }
        if (q.q.back() != 127)
        {
            std::cerr << "[test_int8_quant] (2) 正側最大が 127 でない: "
                      << static_cast<int>(q.q.back()) << "\n";
            return false;
        }
    }

    // (3) 全 0: absmax = 0 → ゼロ除算せず scale 0・q 全 0
    {
        std::vector<float> x = {0.0f, 0.0f};
        Int8Quant q = quantize_activation_int8(x.data(), x.size());
        if (q.scale != 0.0f)
        {
            std::cerr << "[test_int8_quant] (3) 全0 で scale が 0 でない: " << q.scale << "\n";
            return false;
        }
        const std::vector<int8_t> want = {0, 0};
        if (q.q != want)
        {
            std::cerr << "[test_int8_quant] (3) 全0 で q が 0 でない\n";
            return false;
        }
    }

    std::cout << "[test_int8_quant] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// RMSNorm: out[i] = x[i] / sqrt(mean(x^2) + eps) * weight[i]
// ----------------------------------------------------------------
static bool test_rms_norm()
{
    // (1) 既知値: x = {3, 4}、weight = {1, 1}、eps ~ 0
    //     mean(x^2) = (9+16)/2 = 12.5、rms = sqrt(12.5) = 3.5355339...
    //     out = {3/3.5355, 4/3.5355} = {0.848528, 1.131371}
    {
        std::vector<float> x = {3.0f, 4.0f};
        std::vector<float> w = {1.0f, 1.0f};
        std::vector<float> out(2);
        rms_norm(x.data(), w.data(), out.data(), 2, 1e-12);
        const double rms = std::sqrt(12.5);
        if (!approx(out[0], 3.0 / rms, 1e-5) || !approx(out[1], 4.0 / rms, 1e-5))
        {
            std::cerr << "[test_rms_norm] (1) 不一致: " << out[0] << " " << out[1] << "\n";
            return false;
        }
    }

    // (2) weight でスケール: x = {1, 1, 1, 1}、weight = {2, 2, 2, 2}
    //     mean(x^2)=1、rms=1、out = {2, 2, 2, 2}
    {
        std::vector<float> x = {1.0f, 1.0f, 1.0f, 1.0f};
        std::vector<float> w = {2.0f, 2.0f, 2.0f, 2.0f};
        std::vector<float> out(4);
        rms_norm(x.data(), w.data(), out.data(), 4, 1e-12);
        for (int i = 0; i < 4; ++i)
        {
            if (!approx(out[i], 2.0, 1e-5))
            {
                std::cerr << "[test_rms_norm] (2) 不一致 i=" << i << ": " << out[i] << "\n";
                return false;
            }
        }
    }

    // (3) 全 0 入力: eps でゼロ除算しない (out は全 0・有限)。
    {
        std::vector<float> x = {0.0f, 0.0f, 0.0f};
        std::vector<float> w = {1.0f, 1.0f, 1.0f};
        std::vector<float> out(3);
        rms_norm(x.data(), w.data(), out.data(), 3, 1e-5);
        for (int i = 0; i < 3; ++i)
        {
            if (!std::isfinite(out[i]))
            {
                std::cerr << "[test_rms_norm] (3) 全0 で非有限: " << out[i] << "\n";
                return false;
            }
            if (out[i] != 0.0f)
            {
                std::cerr << "[test_rms_norm] (3) 全0 で 0 でない: " << out[i] << "\n";
                return false;
            }
        }
    }

    std::cout << "[test_rms_norm] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// 形状/語彙整合: VOCAB_SIZE==4999、embed tied (embed と lm_head が同一ポインタ)
// ----------------------------------------------------------------
static bool test_vocab_and_tied()
{
    if (BitNet::VOCAB_SIZE != 4999)
    {
        std::cerr << "[test_vocab_and_tied] VOCAB_SIZE が 4999 でない: "
                  << BitNet::VOCAB_SIZE << "\n";
        return false;
    }

    BitNet model;
    model.init_random(12345);
    if (model.embed_weight() != model.lm_head_weight())
    {
        std::cerr << "[test_vocab_and_tied] embed tied でない (ポインタ不一致)\n";
        return false;
    }
    std::cout << "[test_vocab_and_tied] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// 決定的 forward: 形状 [seq_len, 4999]・2回実行で完全一致・NaN/Inf なし
// ----------------------------------------------------------------
static bool test_forward_deterministic()
{
    BitNet model;
    model.init_random(42);

    const std::vector<int> tokens = {1, 5, 100, 4998, 7, 5};  // <bos>+tags, seq=6
    const int S = static_cast<int>(tokens.size());

    std::vector<float> a = model.forward(tokens);
    std::vector<float> b = model.forward(tokens);

    const size_t want_n = static_cast<size_t>(S) * BitNet::VOCAB_SIZE;
    if (a.size() != want_n)
    {
        std::cerr << "[test_forward_deterministic] 出力サイズ不一致: got "
                  << a.size() << " want " << want_n << "\n";
        return false;
    }

    // 2 回実行で完全一致 (ビット一致でなく値一致)。
    if (a != b)
    {
        std::cerr << "[test_forward_deterministic] 2 回実行で不一致\n";
        return false;
    }

    // NaN / Inf なし。
    for (float v : a)
    {
        if (!std::isfinite(v))
        {
            std::cerr << "[test_forward_deterministic] 非有限 logit: " << v << "\n";
            return false;
        }
    }

    std::cout << "[test_forward_deterministic] 形状 [" << S << ", "
              << BitNet::VOCAB_SIZE << "] OK / 決定的 OK / 有限 OK\n";
    std::cout << "[test_forward_deterministic] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// 値の健全性: 各位置の logits を softmax したとき和が 1・極端発散なし
// ----------------------------------------------------------------
static bool test_logit_sanity()
{
    BitNet model;
    model.init_random(7);

    const std::vector<int> tokens = {1, 5, 6, 7, 8};
    const int S = static_cast<int>(tokens.size());
    const int V = BitNet::VOCAB_SIZE;

    std::vector<float> logits = model.forward(tokens);

    for (int s = 0; s < S; ++s)
    {
        const float* row = &logits[static_cast<size_t>(s) * V];
        // max を引いて安定 softmax。
        float maxv = row[0];
        for (int v = 1; v < V; ++v)
        {
            if (row[v] > maxv) maxv = row[v];
        }
        // 極端発散チェック (|logit| が異常に大きくない)。
        if (std::fabs(static_cast<double>(maxv)) > 1.0e4)
        {
            std::cerr << "[test_logit_sanity] logit 発散 s=" << s
                      << " max=" << maxv << "\n";
            return false;
        }
        double sum = 0.0;
        for (int v = 0; v < V; ++v)
        {
            sum += std::exp(static_cast<double>(row[v]) - static_cast<double>(maxv));
        }
        // softmax の和 (= sum/sum) は 1。ここでは sum 自体が有限かつ正であることを確認し、
        // 正規化後の総和が 1 に十分近いことを別途検算する。
        if (!(sum > 0.0) || !std::isfinite(sum))
        {
            std::cerr << "[test_logit_sanity] softmax 分母が不正 s=" << s
                      << " sum=" << sum << "\n";
            return false;
        }
        double norm_sum = 0.0;
        for (int v = 0; v < V; ++v)
        {
            norm_sum += std::exp(static_cast<double>(row[v]) - static_cast<double>(maxv)) / sum;
        }
        if (!approx(norm_sum, 1.0, 1e-6))
        {
            std::cerr << "[test_logit_sanity] softmax 和が 1 でない s=" << s
                      << " sum=" << norm_sum << "\n";
            return false;
        }
    }

    std::cout << "[test_logit_sanity] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// ベンチ: 代表系列長 (~32 token) でホスト参照 forward を warmup 後 N 回、中央値 ms
// ----------------------------------------------------------------
static bool bench_forward_latency()
{
    BitNet model;
    model.init_random(2024);

    // 代表的な text + tags 系列 (32 token)。
    std::vector<int> tokens;
    tokens.reserve(32);
    for (int i = 0; i < 32; ++i)
    {
        tokens.push_back(5 + (i * 137) % (BitNet::VOCAB_SIZE - 5));
    }

    // warmup (naive 参照は 1 回 ~19s と重いため最小限)
    for (int i = 0; i < 1; ++i)
    {
        volatile float sink = model.forward(tokens)[0];
        (void)sink;
    }

    const int N = 5;  // 中央値が取れる最小限 (1 回 ~19s のため抑制)
    std::vector<double> ms;
    ms.reserve(N);
    for (int i = 0; i < N; ++i)
    {
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<float> out = model.forward(tokens);
        const auto t1 = std::chrono::steady_clock::now();
        volatile float sink = out[0];
        (void)sink;
        ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(ms.begin(), ms.end());
    const double median = ms[ms.size() / 2];

    std::cout << "[bench_forward_latency] seq_len=" << tokens.size()
              << " N=" << N
              << " median=" << median << " ms/forward"
              << " (min=" << ms.front() << " max=" << ms.back() << ")\n";
    std::cout << "[bench_forward_latency] PASSED\n";
    return true;
}

} // namespace dollama

int main()
{
    bool ok = true;
    try
    {
        ok = dollama::test_param_count()          && ok;
        ok = dollama::test_ternary_quant()        && ok;
        ok = dollama::test_int8_quant()           && ok;
        ok = dollama::test_rms_norm()             && ok;
        ok = dollama::test_vocab_and_tied()       && ok;
        ok = dollama::test_forward_deterministic() && ok;
        ok = dollama::test_logit_sanity()         && ok;
        ok = dollama::bench_forward_latency()     && ok;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[test_bitnet] 例外: " << e.what() << "\n";
        return 1;
    }

    if (!ok)
    {
        std::cerr << "[test_bitnet] FAILED\n";
        return 1;
    }
    std::cout << "[test_bitnet] ALL PASSED\n";
    return 0;
}

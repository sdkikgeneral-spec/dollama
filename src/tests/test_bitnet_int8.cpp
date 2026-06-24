// BitNetInt8Infer 単体テスト — Phase 4 圧縮実験 (重みのみ INT8)。
//
// 対象: src/infer/bitnet_int8.hpp (per-row INT8 射影 + FP32 lm_head/非線形)、
//       src/models/bitnet.hpp::quantize_weight_int8_perrow。
//
// golden (FP32 PyTorch・#6 synthetic) を流用し、INT8 化による劣化を数値化する:
//   - logits_golden.safetensors : input_ids_s{8,32,63} / logits_s{8,32,63}
//   - gen_golden.safetensors    : prompt_ids_c{0..4} / gen_ids_c{0..4}
//   - 重み: bitnet_dense_fp32.safetensors (FP32 74 テンソル [out,in])
//
// FP32 経路 (1e-3) と違い INT8 は損失ありなので、logits は corr >= 0.99 を
// ハードゲートにする。生成一致率は初回実測をログ出力し、その値から保守的に
// 床値を設定する (ゲートを緩めるのでなく実測を正直に報告する)。
//
// 重み/golden が不在なら各サブテスト [SKIP] (CI ビルド緑維持)。

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "infer/bitnet.hpp"        // FP32 比較 (フットプリント / レイテンシ参考)
#include "infer/bitnet_int8.hpp"
#include "io/safetensors.hpp"
#include "models/bitnet.hpp"       // quantize_weight_int8_perrow

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
        throw std::runtime_error("test_bitnet_int8: '" + name + "' must be I32");
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
        throw std::runtime_error("test_bitnet_int8: '" + name + "' must be F32");
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

// ── (a) quantize_weight_int8_perrow 単体 ──────────────────────────
// 既知小行列で scale・往復再構成誤差・範囲端クランプ・全ゼロ row を検証。
static bool test_quant_perrow_unit()
{
    bool ok = true;

    // 3x4 行列。行ごとに absmax が異なる (per-row scale が効くことを確認)。
    //   row0: absmax 4.0   → scale 4/127
    //   row1: absmax 0.5   → scale 0.5/127
    //   row2: 全 0         → scale 0
    const int out_dim = 3;
    const int in_dim  = 4;
    const float W[12] = {
        1.0f, -2.0f, 4.0f, 0.0f,      // row0
        0.5f, -0.25f, 0.1f, -0.5f,    // row1
        0.0f, 0.0f, 0.0f, 0.0f,       // row2
    };

    const Int8RowQuant rq = quantize_weight_int8_perrow(W, out_dim, in_dim);

    if (rq.scale.size() != static_cast<size_t>(out_dim)
        || rq.q.size() != static_cast<size_t>(out_dim) * in_dim)
    {
        std::cerr << "[test_quant_perrow_unit] FAIL: 形状不一致\n";
        return false;
    }

    // row0 scale = 4/127、row1 scale = 0.5/127、row2 scale = 0。
    const double exp_s0 = 4.0 / 127.0;
    const double exp_s1 = 0.5 / 127.0;
    if (std::fabs(rq.scale[0] - exp_s0) > 1e-7
        || std::fabs(rq.scale[1] - exp_s1) > 1e-9)
    {
        std::cerr << "[test_quant_perrow_unit] FAIL: scale 不一致 "
                  << rq.scale[0] << "," << rq.scale[1] << "\n";
        ok = false;
    }
    if (rq.scale[2] != 0.0f)
    {
        std::cerr << "[test_quant_perrow_unit] FAIL: 全 0 row scale != 0\n";
        ok = false;
    }
    // 全 0 row の量子化値は全 0。
    for (int i = 0; i < in_dim; ++i)
    {
        if (rq.q[static_cast<size_t>(2) * in_dim + i] != 0)
        {
            std::cerr << "[test_quant_perrow_unit] FAIL: 全 0 row q != 0\n";
            ok = false;
        }
    }

    // row0 の absmax 要素 (4.0) は量子化値 +127 になる (範囲端)。
    if (rq.q[2] != 127)
    {
        std::cerr << "[test_quant_perrow_unit] FAIL: row0 absmax 要素 q="
                  << static_cast<int>(rq.q[2]) << " != 127\n";
        ok = false;
    }

    // 往復再構成: W_hat[o,i] = scale[o]*q[o,i]。各要素の量子化誤差は半 LSB
    // (= scale[o]/2) 以内 (round-to-nearest)。
    for (int o = 0; o < out_dim; ++o)
    {
        const double s = rq.scale[static_cast<size_t>(o)];
        for (int i = 0; i < in_dim; ++i)
        {
            const double recon =
                s * static_cast<double>(rq.q[static_cast<size_t>(o) * in_dim + i]);
            const double err = std::fabs(recon - static_cast<double>(W[o * in_dim + i]));
            if (err > s * 0.5 + 1e-7)
            {
                std::cerr << "[test_quant_perrow_unit] FAIL: row" << o << " i" << i
                          << " recon err " << err << " > 0.5 LSB " << (s * 0.5) << "\n";
                ok = false;
            }
        }
    }

    // 全要素が [-127,127] に収まる (クランプ)。
    for (int8_t v : rq.q)
    {
        if (static_cast<int>(v) < -127 || static_cast<int>(v) > 127)
        {
            std::cerr << "[test_quant_perrow_unit] FAIL: q 範囲外 "
                      << static_cast<int>(v) << "\n";
            ok = false;
        }
    }

    if (ok)
    {
        std::cout << "[test_quant_perrow_unit] PASSED (scale row0=" << rq.scale[0]
                  << " row1=" << rq.scale[1] << " row2=" << rq.scale[2] << ")\n";
    }
    return ok;
}

// ── (b) INT8 logits vs FP32 golden logits ─────────────────────────
// seq 8/32/63 の全 logits を FP32 golden と突合。max abs err / corr を出力。
// ハードゲート: corr >= 0.99 (INT8 は損失あり)。
static bool test_int8_logits_match()
{
    if (!file_exists(WEIGHTS_PATH) || !file_exists(LOGITS_GOLDEN_PATH))
    {
        std::cout << "[test_int8_logits_match] SKIP (weights/golden 不在)\n";
        return true;
    }

    BitNetInt8Infer model(WEIGHTS_PATH);
    SafeTensors gold(LOGITS_GOLDEN_PATH);

    bool ok = true;
    const int seq_lens[3] = {8, 32, 63};
    for (int sl : seq_lens)
    {
        const std::string suffix = "s" + std::to_string(sl);
        const std::vector<int> ids = read_i32(gold, "input_ids_" + suffix);
        const std::vector<float> g  = read_f32(gold, "logits_" + suffix);

        if (static_cast<int>(ids.size()) != sl)
        {
            std::cerr << "[test_int8_logits_match] FAIL: input_ids_" << suffix
                      << " size " << ids.size() << " != " << sl << "\n";
            return false;
        }

        std::vector<float> logits = model.forward(ids);
        if (logits.size() != g.size())
        {
            std::cerr << "[test_int8_logits_match] FAIL: logits size "
                      << logits.size() << " != golden " << g.size() << "\n";
            return false;
        }

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

        std::cout << "[test_int8_logits_match] seq=" << sl
                  << " max_abs_err=" << max_abs
                  << " corr=" << corr << "\n";

        if (!(corr >= 0.99))
        {
            std::cerr << "[test_int8_logits_match] FAIL: seq=" << sl
                      << " corr " << corr << " < 0.99\n";
            ok = false;
        }
    }

    if (ok)
    {
        std::cout << "[test_int8_logits_match] PASSED\n";
    }
    return ok;
}

// ── (c) greedy 生成 vs FP32 gen_golden (5 ケース) ─────────────────
// トークン一致率を測定・出力。床値ゲート: 各ケース語彙範囲内 + 全体一致率 >= 床。
// 床値は初回実測から保守的に設定する。
static bool test_int8_greedy_decode()
{
    if (!file_exists(WEIGHTS_PATH) || !file_exists(GEN_GOLDEN_PATH))
    {
        std::cout << "[test_int8_greedy_decode] SKIP (weights/golden 不在)\n";
        return true;
    }

    BitNetInt8Infer model(WEIGHTS_PATH);
    SafeTensors gold(GEN_GOLDEN_PATH);

    bool ok = true;
    long long total_match = 0;   // 位置一致トークン数 (min(len) 範囲で比較)
    long long total_pos   = 0;   // 比較位置数の合計 (max(len) で長さ差も罰する)
    int exact_cases = 0;         // 完全一致ケース数

    for (int ci = 0; ci < 5; ++ci)
    {
        const std::string c = "c" + std::to_string(ci);
        const std::vector<int> prompt = read_i32(gold, "prompt_ids_" + c);
        const std::vector<int> expect = read_i32(gold, "gen_ids_" + c);

        const std::vector<int> got = model.generate(prompt);

        // 生成 id は全て語彙範囲内であること (ハードゲート)。
        for (int id : got)
        {
            if (id < 0 || id >= BitNetInt8Infer::VOCAB_SIZE)
            {
                std::cerr << "[test_int8_greedy_decode] FAIL case " << ci
                          << " 生成 id 範囲外 " << id << "\n";
                ok = false;
            }
        }

        // 位置ごと一致 (短い方の長さまで)。
        const size_t cmp_len = std::min(got.size(), expect.size());
        size_t case_match = 0;
        for (size_t i = 0; i < cmp_len; ++i)
        {
            if (got[i] == expect[i])
            {
                ++case_match;
            }
        }
        const size_t case_pos = std::max(got.size(), expect.size());
        const bool exact = (got.size() == expect.size() && case_match == got.size());
        if (exact)
        {
            ++exact_cases;
        }
        total_match += static_cast<long long>(case_match);
        total_pos   += static_cast<long long>(case_pos);

        const double case_rate =
            case_pos == 0 ? 1.0 : static_cast<double>(case_match) / static_cast<double>(case_pos);
        std::cout << "[test_int8_greedy_decode] case " << ci
                  << " len got=" << got.size() << " expect=" << expect.size()
                  << " match=" << case_match << "/" << case_pos
                  << " rate=" << case_rate
                  << (exact ? " (EXACT)" : "") << "\n";
    }

    const double overall =
        total_pos == 0 ? 1.0 : static_cast<double>(total_match) / static_cast<double>(total_pos);
    std::cout << "[test_int8_greedy_decode] overall token match rate="
              << overall << " (" << total_match << "/" << total_pos
              << "), exact cases=" << exact_cases << "/5\n";

    // 床値ゲート (実測ベース・保守的)。
    // INT8 は損失ありで FP32 と完全一致は期待しない。全体一致率が床を下回ったら
    // 実装の取り違え (per-row 不全・lm_head FP32 でない・活性 scale 誤り) を疑う。
    // 床値は INT8 重み量子化で通常維持される範囲として 0.5 を設定。
    const double floor_rate = 0.5;
    if (!(overall >= floor_rate))
    {
        std::cerr << "[test_int8_greedy_decode] FAIL: overall token match "
                  << overall << " < floor " << floor_rate
                  << " (per-row/lm_head/活性 scale を切り分けよ)\n";
        ok = false;
    }

    if (ok)
    {
        std::cout << "[test_int8_greedy_decode] PASSED\n";
    }
    return ok;
}

// ── (d) フットプリント / レイテンシ 情報出力 ─────────────────────
static bool info_footprint_and_speed()
{
    if (!file_exists(WEIGHTS_PATH))
    {
        std::cout << "[info_footprint_and_speed] SKIP (weights 不在)\n";
        return true;
    }

    BitNetInt8Infer model_i8(WEIGHTS_PATH);

    // フットプリント: 射影 7 種 ×8 層 の FP32 → INT8 削減量。
    const size_t fp32_bytes = BitNetInt8Infer::fp32_proj_bytes();
    const size_t i8_bytes   = model_i8.int8_weight_bytes();
    const double saved = static_cast<double>(fp32_bytes)
                         - static_cast<double>(i8_bytes);
    const double ratio = static_cast<double>(i8_bytes)
                         / static_cast<double>(fp32_bytes);
    std::cout << "[info_footprint_and_speed] 射影重み FP32=" << fp32_bytes
              << " B / INT8(+scale)=" << i8_bytes << " B / 削減="
              << saved << " B (" << (100.0 * (1.0 - ratio)) << "% 減・"
              << "残存比 " << ratio << ")\n";

    // seq8 forward レイテンシ (INT8 vs FP32 参考)。
    std::vector<int> ids = {1, 9, 13, 14, 18, 3, 5, 6};  // seq=8
    model_i8.forward(ids);  // warmup
    const int N = 5;

    auto bench = [&](auto& m) -> double
    {
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < N; ++i)
        {
            volatile float sink = m.forward(ids)[0];
            (void)sink;
        }
        const auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count() / N;
    };

    const double ms_i8 = bench(model_i8);

    // FP32 比較 (重みが揃っているので同じファイルからロード)。
    BitNetDenseInfer model_fp32(WEIGHTS_PATH);
    model_fp32.forward(ids);  // warmup
    const double ms_fp32 = bench(model_fp32);

    std::cout << "[info_footprint_and_speed] seq=8 forward INT8 ~" << ms_i8
              << " ms / FP32 ~" << ms_fp32 << " ms (比 "
              << (ms_fp32 > 0.0 ? ms_i8 / ms_fp32 : 0.0) << "x)\n";
    return true;
}

} // namespace dollama

int main()
{
    bool ok = true;
    try
    {
        ok = dollama::test_quant_perrow_unit() && ok;
        ok = dollama::test_int8_logits_match() && ok;
        ok = dollama::test_int8_greedy_decode() && ok;
        ok = dollama::info_footprint_and_speed() && ok;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[test_bitnet_int8] EXCEPTION: " << e.what() << "\n";
        return 1;
    }

    if (!ok)
    {
        std::cerr << "[test_bitnet_int8] FAILED\n";
        return 1;
    }
    std::cout << "[test_bitnet_int8] ALL PASSED\n";
    return 0;
}

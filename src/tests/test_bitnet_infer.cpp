// BitNetDenseInfer 単体テスト — Phase 4 dense #6 + Phase 4 A (A3 同一性条件付き)。
//
// 対象: src/infer/bitnet.hpp (dense FP32 forward + greedy デコード +
//       generate_with_identity)。
//
// #6 (synthetic) golden — PyTorch (scripts/train_bitnet.py --dump-golden):
//   - logits_golden.safetensors : input_ids_s{8,32,63} / logits_s{8,32,63}
//   - gen_golden.safetensors    : prompt_ids_c{0..4} / gen_ids_c{0..4}
//   - 重み: bitnet_dense_fp32.safetensors (embed/layers.*/final_norm, 全 F32)
//
// A3 (identity) golden — PyTorch (scripts/train_bitnet.py --dump-golden-identity):
//   - logits_golden_identity.safetensors : identity_cond 2-<sep> 系列の logits/input_ids
//   - gen_golden_identity.safetensors    : prompt_ids_c{0..4} / gen_ids_c{0..4}
//   - 重み: bitnet_dense_identity_fp32.safetensors (#6 と同一 74 テンソル [out,in])
//
// パスは meson の -D 埋め込み。golden / 重みは gitignore 対象なので、いずれか不在
// なら該当テストを [SKIP] する (VAE/UNet golden テストの skip 規約に倣う・CI で
// golden 不在でもビルド緑)。
//
// 受け入れ条件:
//   1. synthetic / identity の各 input_ids を C++ dense forward に通し全 logits を
//      golden と突合: max abs error < 1e-3 かつ相関 >= 0.99999。
//   2. synthetic / identity の prompt から greedy 生成した id 列が gen_ids と
//      完全一致 (各 5 ケース)。
//   3. A3 identity: prompt は identity_tags + scene_text から C++ 側で組み立て
//      (generate_with_identity)、その生成も gen_ids と完全一致。
//   4. CharacterBible.find->canonical_tags を identity_tags に流す結線疎通 (定性)。
//   5. Tier 1 高速パス健全性: forward() (double, full) と forward_fast(last_only=false)
//      (float32 + AVX2) の全 logits を突合 (max_abs < 1e-3 かつ corr >= 0.99999)。

#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "core/character.hpp"
#include "infer/bitnet.hpp"
#include "io/safetensors.hpp"
#include "io/tokenizer.hpp"

#ifndef WEIGHTS_PATH
#define WEIGHTS_PATH ""
#endif
#ifndef LOGITS_GOLDEN_PATH
#define LOGITS_GOLDEN_PATH ""
#endif
#ifndef GEN_GOLDEN_PATH
#define GEN_GOLDEN_PATH ""
#endif
// A3 identity 版パス。
#ifndef IDENTITY_WEIGHTS_PATH
#define IDENTITY_WEIGHTS_PATH ""
#endif
#ifndef IDENTITY_LOGITS_GOLDEN_PATH
#define IDENTITY_LOGITS_GOLDEN_PATH ""
#endif
#ifndef IDENTITY_GEN_GOLDEN_PATH
#define IDENTITY_GEN_GOLDEN_PATH ""
#endif
#ifndef VOCAB_PATH
#define VOCAB_PATH ""
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

// ── ロジット突合 (共通ヘルパ) ──────────────────────────────────────
// 指定の重み / logits golden で seq 8/32/63 の全 logits を突合する。
// label はログ用 (synthetic / identity)。
static bool logits_match_with(const std::string& weights,
                              const std::string& logits_golden,
                              const char* label)
{
    if (!file_exists(weights) || !file_exists(logits_golden))
    {
        std::cout << "[test_logits_match:" << label << "] SKIP (weights/golden 不在)\n";
        return true;
    }

    BitNetDenseInfer model(weights);
    SafeTensors gold(logits_golden);

    const int seq_lens[3] = {8, 32, 63};
    for (int sl : seq_lens)
    {
        const std::string suffix = "s" + std::to_string(sl);
        const std::vector<int> ids = read_i32(gold, "input_ids_" + suffix);
        const std::vector<float> g  = read_f32(gold, "logits_" + suffix);

        if (static_cast<int>(ids.size()) != sl)
        {
            std::cerr << "[test_logits_match:" << label << "] FAIL: input_ids_"
                      << suffix << " size " << ids.size() << " != " << sl << "\n";
            return false;
        }

        std::vector<float> logits = model.forward(ids);
        if (logits.size() != g.size())
        {
            std::cerr << "[test_logits_match:" << label << "] FAIL: logits size "
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

        std::cout << "[test_logits_match:" << label << "] seq=" << sl
                  << " max_abs_err=" << max_abs
                  << " corr=" << corr << "\n";

        if (max_abs >= 1e-3)
        {
            std::cerr << "[test_logits_match:" << label << "] FAIL: seq=" << sl
                      << " max_abs_err " << max_abs << " >= 1e-3\n";
            return false;
        }
        if (!(corr >= 0.99999))
        {
            std::cerr << "[test_logits_match:" << label << "] FAIL: seq=" << sl
                      << " corr " << corr << " < 0.99999\n";
            return false;
        }
    }

    std::cout << "[test_logits_match:" << label << "] PASSED\n";
    return true;
}

// ── greedy デコード突合 (共通ヘルパ) ───────────────────────────────
// gen golden の prompt_ids から生成した id 列が gen_ids と完全一致するか (5 ケース)。
static bool greedy_decode_with(const std::string& weights,
                               const std::string& gen_golden,
                               const char* label)
{
    if (!file_exists(weights) || !file_exists(gen_golden))
    {
        std::cout << "[test_greedy_decode:" << label << "] SKIP (weights/golden 不在)\n";
        return true;
    }

    BitNetDenseInfer model(weights);
    SafeTensors gold(gen_golden);

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
            std::cerr << "[test_greedy_decode:" << label << "] FAIL case " << ci
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
        std::cout << "[test_greedy_decode:" << label << "] case " << ci
                  << " matched (" << got.size() << " tokens)\n";
    }

    std::cout << "[test_greedy_decode:" << label << "] PASSED\n";
    return true;
}

// ── #6 synthetic ロジット突合 ──────────────────────────────────────
static bool test_logits_match()
{
    return logits_match_with(WEIGHTS_PATH, LOGITS_GOLDEN_PATH, "synthetic");
}

// ── #6 synthetic greedy デコード突合 ───────────────────────────────
static bool test_greedy_decode()
{
    return greedy_decode_with(WEIGHTS_PATH, GEN_GOLDEN_PATH, "synthetic");
}

// ── A3 identity ロジット突合 ───────────────────────────────────────
// identity 重み + identity logits golden (2-<sep> 系列) を突合する。
static bool test_identity_logits_match()
{
    return logits_match_with(IDENTITY_WEIGHTS_PATH, IDENTITY_LOGITS_GOLDEN_PATH,
                             "identity");
}

// ── A3 identity greedy デコード突合 (golden prompt 直行) ───────────
// gen_golden_identity の prompt_ids (= identity prefix + scene greedy + <sep>) を
// そのまま generate() に流し、gen_ids と完全一致するか。
static bool test_identity_greedy_decode()
{
    return greedy_decode_with(IDENTITY_WEIGHTS_PATH, IDENTITY_GEN_GOLDEN_PATH,
                              "identity");
}

// ── Tier 1 高速パス健全性: forward (double) vs forward_fast (float32 + AVX2) ──
// 同一入力 (logits golden の input_ids_s{8,32,63}) で参照経路 forward() と高速パス
// forward_fast(last_only=false) の全 logits を突合する。double→float32 蓄積に
// 変えても数値が崩れていないこと (max_abs < 1e-3 かつ corr >= 0.99999) を固定する。
// weights/golden 不在時は [SKIP]。
static bool test_forward_fast_matches_double()
{
    if (!file_exists(WEIGHTS_PATH) || !file_exists(LOGITS_GOLDEN_PATH))
    {
        std::cout << "[test_forward_fast_matches_double] SKIP (weights/golden 不在)\n";
        return true;
    }

    BitNetDenseInfer model(WEIGHTS_PATH);
    SafeTensors gold(LOGITS_GOLDEN_PATH);

    const int seq_lens[3] = {8, 32, 63};
    for (int sl : seq_lens)
    {
        const std::string suffix = "s" + std::to_string(sl);
        const std::vector<int> ids = read_i32(gold, "input_ids_" + suffix);

        const std::vector<float> ref  = model.forward(ids);
        const std::vector<float> fast = model.forward_fast(ids, /*last_only=*/false);

        if (ref.size() != fast.size())
        {
            std::cerr << "[test_forward_fast_matches_double] FAIL: seq=" << sl
                      << " size ref " << ref.size() << " != fast " << fast.size()
                      << "\n";
            return false;
        }

        double max_abs = 0.0;
        double sx = 0.0, sy = 0.0, sxx = 0.0, syy = 0.0, sxy = 0.0;
        const size_t n = ref.size();
        for (size_t i = 0; i < n; ++i)
        {
            const double a = static_cast<double>(ref[i]);
            const double b = static_cast<double>(fast[i]);
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

        std::cout << "[test_forward_fast_matches_double] seq=" << sl
                  << " max_abs=" << max_abs << " corr=" << corr << "\n";

        if (max_abs >= 1e-3)
        {
            std::cerr << "[test_forward_fast_matches_double] FAIL: seq=" << sl
                      << " max_abs " << max_abs << " >= 1e-3\n";
            return false;
        }
        if (!(corr >= 0.99999))
        {
            std::cerr << "[test_forward_fast_matches_double] FAIL: seq=" << sl
                      << " corr " << corr << " < 0.99999\n";
            return false;
        }
    }

    std::cout << "[test_forward_fast_matches_double] PASSED\n";
    return true;
}

// ── A3 identity 生成 (C++ 側で prompt 組み立て) ────────────────────
// manifest_identity.json の各ケースの identity_tags / scene_text から
// generate_with_identity で prompt を C++ 内で組み立てて生成し、
// gen_golden_identity の gen_ids と完全一致するか (条件付け機構の end-to-end)。
//
// manifest の text は CP932 環境で文字化けしうるため、scene_text は manifest に
// 依存せず C++ ハードコードする。identity_tags も同様 (語彙内タグのみで <unk> なし)。
// 期待値は gen_golden_identity の gen_ids (= golden が prompt から生成した id 列)。
// 併せて build_identity_prompt が golden の prompt_ids と完全一致することも検証する
// (=条件付け機構 = tag_to_id + scene greedy + 2-<sep> が PyTorch 側と一致)。
struct IdentityCase
{
    std::vector<std::string> identity_tags;
    std::string              scene_text;
};

static bool test_identity_generate_from_tags()
{
    if (!file_exists(IDENTITY_WEIGHTS_PATH) || !file_exists(IDENTITY_GEN_GOLDEN_PATH)
        || !file_exists(VOCAB_PATH))
    {
        std::cout << "[test_identity_generate_from_tags] SKIP (weights/golden/vocab 不在)\n";
        return true;
    }

    BitNetDenseInfer model(IDENTITY_WEIGHTS_PATH);
    SafeTensors gold(IDENTITY_GEN_GOLDEN_PATH);
    Tokenizer tok(VOCAB_PATH);

    // manifest_identity.json の 5 ケースの identity_tags / scene_text。
    // scene_text は ASCII ケース (1/4) を採用 (CP932 文字化け回避)。
    // 日本語 scene_text のケース (0/2/3) は scene が語彙外でほぼ空になり、
    // prompt 組み立てが golden と一致することは golden 直行テストでカバー済みのため、
    // ここでは ASCII ケースで end-to-end (tag_to_id + scene greedy + 2-<sep>) を検証する。
    const IdentityCase cases[2] = {
        // case 1
        { {"1girl", "short hair", "black hair", "red eyes", "animal ears",
           "pink hair", "multicolored hair"},
          "Please draw a girl: short hair, black hair, red eyes, animal ears." },
        // case 4
        { {"1girl", "breasts", "brown hair", "medium hair", "pink eyes"},
          "a girl having breasts, brown hair, medium hair, pink eyes." },
    };
    const int case_idx[2] = {1, 4};

    for (int k = 0; k < 2; ++k)
    {
        const int ci = case_idx[k];
        const std::string c = "c" + std::to_string(ci);
        const std::vector<int> gold_prompt = read_i32(gold, "prompt_ids_" + c);
        const std::vector<int> expect_gen  = read_i32(gold, "gen_ids_" + c);

        // 1) prompt 組み立てが golden の prompt_ids と一致するか。
        const std::vector<int> prompt = model.build_identity_prompt(
            tok, cases[k].identity_tags, cases[k].scene_text);
        bool prompt_ok = (prompt.size() == gold_prompt.size());
        if (prompt_ok)
        {
            for (size_t i = 0; i < prompt.size(); ++i)
            {
                if (prompt[i] != gold_prompt[i])
                {
                    prompt_ok = false;
                    break;
                }
            }
        }
        if (!prompt_ok)
        {
            std::cerr << "[test_identity_generate_from_tags] FAIL case " << ci
                      << " prompt mismatch\n  got   :";
            for (int id : prompt)
            {
                std::cerr << " " << id;
            }
            std::cerr << "\n  golden:";
            for (int id : gold_prompt)
            {
                std::cerr << " " << id;
            }
            std::cerr << "\n";
            return false;
        }

        // 2) generate_with_identity の生成が gen_ids と一致するか。
        const std::vector<int> got = model.generate_with_identity(
            tok, cases[k].identity_tags, cases[k].scene_text);
        bool ok = (got.size() == expect_gen.size());
        if (ok)
        {
            for (size_t i = 0; i < got.size(); ++i)
            {
                if (got[i] != expect_gen[i])
                {
                    ok = false;
                    break;
                }
            }
        }
        if (!ok)
        {
            std::cerr << "[test_identity_generate_from_tags] FAIL case " << ci
                      << " gen mismatch (got=" << got.size()
                      << " expect=" << expect_gen.size() << ")\n  got   :";
            for (int id : got)
            {
                std::cerr << " " << id;
            }
            std::cerr << "\n  expect:";
            for (int id : expect_gen)
            {
                std::cerr << " " << id;
            }
            std::cerr << "\n";
            return false;
        }
        std::cout << "[test_identity_generate_from_tags] case " << ci
                  << " prompt+gen matched (" << got.size() << " tokens)\n";
    }

    std::cout << "[test_identity_generate_from_tags] PASSED\n";
    return true;
}

// ── CharacterBible 結線疎通 (定性) ─────────────────────────────────
// CharacterBible に 1 体登録 → find → canonical_tags を identity_tags として
// generate_with_identity に渡せることを示す薄いグルー。
// ハードゲートにはしない (生成 id の値そのものは問わない)。NaN/例外なく
// 何かしらの id 列が返ること・prompt 組み立てが期待構造を持つことだけ確認する。
static bool test_character_bible_wiring()
{
    if (!file_exists(IDENTITY_WEIGHTS_PATH) || !file_exists(VOCAB_PATH))
    {
        std::cout << "[test_character_bible_wiring] SKIP (weights/vocab 不在)\n";
        return true;
    }

    BitNetDenseInfer model(IDENTITY_WEIGHTS_PATH);
    Tokenizer tok(VOCAB_PATH);

    // 台帳に 1 体登録 (canonical_tags は語彙内タグで構成)。
    CharacterBible bible;
    CharacterIdentity aria;
    aria.name = "Aria";
    aria.canonical_tags = {"1girl", "long hair", "brown hair", "twintails"};
    bible.put(aria);

    const CharacterIdentity* found = bible.find("Aria");
    if (found == nullptr)
    {
        std::cerr << "[test_character_bible_wiring] FAIL: find('Aria') == null\n";
        return false;
    }

    // canonical_tags を identity_tags として渡して生成。
    const std::string scene = "smile, looking at viewer.";
    const std::vector<int> prompt =
        model.build_identity_prompt(tok, found->canonical_tags, scene);
    const std::vector<int> gen =
        model.generate_with_identity(tok, found->canonical_tags, scene);

    // prompt 構造の最低限の健全性: <bos> 始まり・末尾 <sep>・<sep> が 2 個。
    if (prompt.empty() || prompt.front() != TOK_BOS || prompt.back() != TOK_SEP)
    {
        std::cerr << "[test_character_bible_wiring] FAIL: prompt 構造異常\n";
        return false;
    }
    int n_sep = 0;
    for (int id : prompt)
    {
        if (id == TOK_SEP)
        {
            ++n_sep;
        }
    }
    if (n_sep != 2)
    {
        std::cerr << "[test_character_bible_wiring] FAIL: <sep> 個数 " << n_sep
                  << " != 2\n";
        return false;
    }

    // 生成 id が全て語彙範囲内であること。
    for (int id : gen)
    {
        if (id < 0 || id >= BitNetDenseInfer::VOCAB_SIZE)
        {
            std::cerr << "[test_character_bible_wiring] FAIL: 生成 id 範囲外 "
                      << id << "\n";
            return false;
        }
    }

    // 生成タグを可読化して情報出力 (定性確認)。
    const std::vector<std::string> gen_tags = tok.decode(gen);
    std::cout << "[test_character_bible_wiring] Aria gen " << gen.size()
              << " tokens: ";
    for (size_t i = 0; i < gen_tags.size() && i < 16; ++i)
    {
        std::cout << "'" << gen_tags[i] << "' ";
    }
    std::cout << "\n[test_character_bible_wiring] PASSED\n";
    return true;
}

// ── forward 速度の参考計測 (assert 外・情報) ───────────────────────
// double 参照経路 forward() と Tier 1 高速パス forward_fast() の両方を計時して倍率を出す。
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
    model.forward_fast(ids, false);

    const int N = 5;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; ++i)
    {
        volatile float sink = model.forward(ids)[0];
        (void)sink;
    }
    auto t1 = std::chrono::steady_clock::now();
    const double ms_d = std::chrono::duration<double, std::milli>(t1 - t0).count() / N;

    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; ++i)
    {
        volatile float sink = model.forward_fast(ids, false)[0];
        (void)sink;
    }
    t1 = std::chrono::steady_clock::now();
    const double ms_f = std::chrono::duration<double, std::milli>(t1 - t0).count() / N;

    std::cout << "[info_forward_speed] seq=8 forward(double) ~" << ms_d
              << " ms / forward_fast(float) ~" << ms_f << " ms ("
              << (ms_f > 0.0 ? ms_d / ms_f : 0.0) << "x)\n";
    return true;
}

} // namespace dollama

int main()
{
    bool ok = true;
    try
    {
        // #6 synthetic (非回帰)
        ok = dollama::test_logits_match() && ok;
        ok = dollama::test_greedy_decode() && ok;
        // A3 identity
        ok = dollama::test_identity_logits_match() && ok;
        ok = dollama::test_identity_greedy_decode() && ok;
        ok = dollama::test_identity_generate_from_tags() && ok;
        ok = dollama::test_character_bible_wiring() && ok;
        // Tier 1 高速パス健全性 (double vs float32+AVX2)
        ok = dollama::test_forward_fast_matches_double() && ok;
        // 参考計測
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

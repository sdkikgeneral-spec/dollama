// タグトークナイザ 単体テスト + ベンチ
// testing.md 形式: 各検証は if (!cond) { cerr; return false; }、
// main で ok = test_xxx() && ok 集約、成功は PASSED / 最後に ALL PASSED。
//
// fixture:
//   VOCAB_PATH      = data/bitnet/vocab.json (リポジトリ追跡済・必須)
//   PAIRS_TRAIN_PATH / PAIRS_VAL_PATH = data/bitnet/pairs.*.jsonl (不在時 [SKIP])
// パスは meson が -D 埋め込み (cwd 非依存)。
//
// 既知 id (vocab.json 頻度降順連番):
//   <pad>=0 <bos>=1 <eos>=2 <sep>=3 <unk>=4
//   1girl=5 solo=6 "long hair"=7 breasts=8 "looking at viewer"=9 blush=10
//   smile=11 "open mouth"=12 "short hair"=13 "blue eyes"=14
//   "simple background"=15 shirt=16 "large breasts"=17 skirt=18 "blonde hair"=19
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "io/tokenizer.hpp"

#ifndef VOCAB_PATH
#define VOCAB_PATH "data/bitnet/vocab.json"
#endif
#ifndef PAIRS_TRAIN_PATH
#define PAIRS_TRAIN_PATH ""
#endif
#ifndef PAIRS_VAL_PATH
#define PAIRS_VAL_PATH ""
#endif

namespace dollama {

static const std::string g_vocab       = VOCAB_PATH;
static const std::string g_pairs_train = PAIRS_TRAIN_PATH;
static const std::string g_pairs_val   = PAIRS_VAL_PATH;

// ----------------------------------------------------------------
// ロード + 検証: 総語彙 4999・tag 数 4994・specials id 0..4 固定
// ----------------------------------------------------------------
static bool test_load()
{
    Tokenizer tok(g_vocab);
    if (tok.vocab_size() != 4999)
    {
        std::cerr << "[test_load] vocab_size != 4999: " << tok.vocab_size() << "\n";
        return false;
    }
    if (tok.tag_count() != 4994)
    {
        std::cerr << "[test_load] tag_count != 4994: " << tok.tag_count() << "\n";
        return false;
    }
    // specials 表記
    const char* sp[5] = {"<pad>", "<bos>", "<eos>", "<sep>", "<unk>"};
    for (int id = 0; id < 5; ++id)
    {
        if (tok.id_to_token(id) != sp[id])
        {
            std::cerr << "[test_load] id_to_token(" << id << ") = '"
                      << tok.id_to_token(id) << "' expected '" << sp[id] << "'\n";
            return false;
        }
    }
    // 既知タグ id (頻度降順連番)
    struct Pair { const char* tag; int id; };
    Pair known[] = {
        {"1girl", 5}, {"solo", 6}, {"long hair", 7}, {"breasts", 8},
        {"shirt", 16}, {"blonde hair", 19},
    };
    for (const Pair& p : known)
    {
        if (tok.tag_to_id(p.tag) != p.id)
        {
            std::cerr << "[test_load] tag '" << p.tag << "' -> "
                      << tok.tag_to_id(p.tag) << " expected " << p.id << "\n";
            return false;
        }
        if (tok.id_to_token(p.id) != p.tag)
        {
            std::cerr << "[test_load] id " << p.id << " -> '"
                      << tok.id_to_token(p.id) << "' expected '" << p.tag << "'\n";
            return false;
        }
    }
    std::cout << "[test_load] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// 不正 vocab で例外: specials 数不足 / id 非連番 / 総数不一致
// ----------------------------------------------------------------
static bool write_tmp(const std::string& path, const std::string& body)
{
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f)
    {
        return false;
    }
    f << body;
    return static_cast<bool>(f);
}

static bool expect_throws(const std::string& path, const std::string& tag)
{
    bool threw = false;
    try
    {
        Tokenizer tok(path);
        (void)tok;
    }
    catch (const std::runtime_error&)
    {
        threw = true;
    }
    if (!threw)
    {
        std::cerr << "[test_invalid] expected throw for case: " << tag << "\n";
    }
    return threw;
}

static bool test_invalid_vocab()
{
    const std::string tmp = "test_tokenizer_invalid.json";

    // specials が 4 件しかない
    if (!write_tmp(tmp,
        R"({"version":1,"separator":"space",)"
        R"("specials":["<pad>","<bos>","<eos>","<sep>"],)"
        R"("tags":[{"id":4,"tag":"a","category":0,"freq":1}]})"))
    {
        std::cerr << "[test_invalid] cannot write tmp\n";
        return false;
    }
    bool ok = expect_throws(tmp, "specials!=5");

    // specials は正しいが id が連番でない (5 でなく 6 始まり)
    ok = write_tmp(tmp,
        R"({"version":1,"separator":"space",)"
        R"("specials":["<pad>","<bos>","<eos>","<sep>","<unk>"],)"
        R"("tags":[{"id":6,"tag":"a","category":0,"freq":1}]})") && ok;
    ok = expect_throws(tmp, "id_not_contiguous") && ok;

    // specials 名が違う
    ok = write_tmp(tmp,
        R"({"version":1,"separator":"space",)"
        R"("specials":["<PAD>","<bos>","<eos>","<sep>","<unk>"],)"
        R"("tags":[{"id":5,"tag":"a","category":0,"freq":1}]})") && ok;
    ok = expect_throws(tmp, "specials_name_mismatch") && ok;

    // 総語彙が 4999 でない (specials 5 + tag 1 = 6)
    ok = write_tmp(tmp,
        R"({"version":1,"separator":"space",)"
        R"("specials":["<pad>","<bos>","<eos>","<sep>","<unk>"],)"
        R"("tags":[{"id":5,"tag":"a","category":0,"freq":1}]})") && ok;
    ok = expect_throws(tmp, "total!=4999") && ok;

    std::remove(tmp.c_str());
    if (ok)
    {
        std::cout << "[test_invalid_vocab] PASSED\n";
    }
    return ok;
}

// ----------------------------------------------------------------
// encode(tags) -> decode 往復: フレーミング・UNK 0・完全一致復元
// ----------------------------------------------------------------
static bool test_encode_decode_roundtrip()
{
    Tokenizer tok(g_vocab);
    std::vector<std::string> tags = {"1girl", "solo", "long hair", "blonde hair", "shirt"};
    std::vector<int> ids = tok.encode(tags);

    // フレーミング <bos> ... <eos>
    if (ids.front() != TOK_BOS || ids.back() != TOK_EOS)
    {
        std::cerr << "[test_roundtrip] framing missing\n";
        return false;
    }
    // UNK 0
    int unk = 0;
    for (int id : ids)
    {
        if (id == TOK_UNK)
        {
            ++unk;
        }
    }
    if (unk != 0)
    {
        std::cerr << "[test_roundtrip] UNK count != 0: " << unk << "\n";
        return false;
    }
    // decode で完全一致復元 (構造トークンは消える)
    std::vector<std::string> back = tok.decode(ids);
    if (back != tags)
    {
        std::cerr << "[test_roundtrip] decode mismatch. got:";
        for (const auto& s : back) std::cerr << " '" << s << "'";
        std::cerr << "\n";
        return false;
    }
    std::cout << "[test_encode_decode_roundtrip] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// 未知タグ -> <unk>(4)
// ----------------------------------------------------------------
static bool test_unknown_tag()
{
    Tokenizer tok(g_vocab);
    if (tok.tag_to_id("this_is_not_a_real_tag_zzz") != TOK_UNK)
    {
        std::cerr << "[test_unknown] unknown tag not mapped to UNK\n";
        return false;
    }
    std::vector<std::string> tags = {"1girl", "nonexistent_xyz", "solo"};
    std::vector<int> ids = tok.encode(tags);
    int unk = 0;
    for (int id : ids)
    {
        if (id == TOK_UNK) ++unk;
    }
    if (unk != 1)
    {
        std::cerr << "[test_unknown] expected exactly 1 UNK, got " << unk << "\n";
        return false;
    }
    std::cout << "[test_unknown_tag] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// 正規化: long_hair -> "long hair" (id 7) / 顔文字 ^_^ >_< の _ 保持
// ----------------------------------------------------------------
static bool test_normalize()
{
    Tokenizer tok(g_vocab);
    // 英数字に挟まれた _ をスペース化 -> id 7 が引ける
    if (tok.tag_to_id("long_hair") != 7)
    {
        std::cerr << "[test_normalize] 'long_hair' -> " << tok.tag_to_id("long_hair")
                  << " expected 7\n";
        return false;
    }
    // 静的正規化関数の直接検証
    if (Tokenizer::normalize("long_hair") != "long hair")
    {
        std::cerr << "[test_normalize] normalize(long_hair) != 'long hair'\n";
        return false;
    }
    if (Tokenizer::normalize("looking_at_viewer") != "looking at viewer")
    {
        std::cerr << "[test_normalize] normalize(looking_at_viewer) wrong\n";
        return false;
    }
    // 顔文字: 記号隣接の _ は保持
    if (Tokenizer::normalize("^_^") != "^_^")
    {
        std::cerr << "[test_normalize] '^_^' modified: '" << Tokenizer::normalize("^_^") << "'\n";
        return false;
    }
    if (Tokenizer::normalize(">_<") != ">_<")
    {
        std::cerr << "[test_normalize] '>_<' modified: '" << Tokenizer::normalize(">_<") << "'\n";
        return false;
    }
    std::cout << "[test_normalize] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// encode_text greedy 最長一致: タグ回収・接続語スキップ・分割しない
// ----------------------------------------------------------------
static bool decode_contains(const std::vector<std::string>& v, const std::string& t)
{
    for (const auto& s : v)
    {
        if (s == t) return true;
    }
    return false;
}

static bool test_encode_text()
{
    Tokenizer tok(g_vocab);
    // pairs.train の代表行: "several girls with shirt, blonde hair, bow, twintails."
    std::vector<int> ids = tok.encode_text(
        "several girls with shirt, blonde hair, bow, twintails.");
    if (ids.front() != TOK_BOS || ids.back() != TOK_EOS)
    {
        std::cerr << "[test_encode_text] framing missing\n";
        return false;
    }
    std::vector<std::string> dec = tok.decode(ids);
    // shirt / blonde hair を回収する (接続語 several/girls/with はスキップ)
    if (!decode_contains(dec, "shirt"))
    {
        std::cerr << "[test_encode_text] 'shirt' not recovered\n";
        return false;
    }
    if (!decode_contains(dec, "blonde hair"))
    {
        std::cerr << "[test_encode_text] 'blonde hair' not recovered\n";
        return false;
    }
    // "blonde" / "hair" 単体が分割されて入っていないこと (最長一致)
    if (decode_contains(dec, "blonde") || decode_contains(dec, "hair"))
    {
        std::cerr << "[test_encode_text] 'blonde hair' was split\n";
        return false;
    }

    // "long hair" が分割されないこと
    std::vector<int> ids2 = tok.encode_text("she has long hair");
    std::vector<std::string> dec2 = tok.decode(ids2);
    if (!decode_contains(dec2, "long hair"))
    {
        std::cerr << "[test_encode_text] 'long hair' not recovered as one token\n";
        return false;
    }
    if (decode_contains(dec2, "long") || decode_contains(dec2, "hair"))
    {
        std::cerr << "[test_encode_text] 'long hair' was split\n";
        return false;
    }
    std::cout << "[test_encode_text] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// フレーミング配置 + <pad>/<sep> の扱い
// ----------------------------------------------------------------
static bool test_framing()
{
    Tokenizer tok(g_vocab);
    std::vector<std::string> tags = {"solo"};
    std::vector<int> ids = tok.encode(tags);
    // [<bos>, solo(6), <eos>]
    if (ids.size() != 3 || ids[0] != TOK_BOS || ids[1] != 6 || ids[2] != TOK_EOS)
    {
        std::cerr << "[test_framing] unexpected ids:";
        for (int id : ids) std::cerr << " " << id;
        std::cerr << "\n";
        return false;
    }
    // decode は <bos>/<eos>/<pad>/<sep> をスキップ
    std::vector<int> withspecials = {TOK_PAD, TOK_BOS, 6, TOK_SEP, 5, TOK_EOS, TOK_PAD};
    std::vector<std::string> dec = tok.decode(withspecials);
    std::vector<std::string> want = {"solo", "1girl"};
    if (dec != want)
    {
        std::cerr << "[test_framing] decode with specials wrong:";
        for (const auto& s : dec) std::cerr << " '" << s << "'";
        std::cerr << "\n";
        return false;
    }
    std::cout << "[test_framing] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// 境界: 空入力・空タグ列・長さ打ち切り
// ----------------------------------------------------------------
static bool test_boundaries()
{
    Tokenizer tok(g_vocab);

    // 空タグ列 -> [<bos>, <eos>]
    std::vector<int> e0 = tok.encode({});
    if (e0.size() != 2 || e0[0] != TOK_BOS || e0[1] != TOK_EOS)
    {
        std::cerr << "[test_boundaries] empty tags wrong size " << e0.size() << "\n";
        return false;
    }

    // 空テキスト -> [<bos>, <eos>]
    std::vector<int> t0 = tok.encode_text("");
    if (t0.size() != 2 || t0[0] != TOK_BOS || t0[1] != TOK_EOS)
    {
        std::cerr << "[test_boundaries] empty text wrong\n";
        return false;
    }

    // 接続語のみ (語彙にない) テキスト -> [<bos>, <eos>]
    std::vector<int> t1 = tok.encode_text("the and with of");
    if (t1.size() != 2 || t1.front() != TOK_BOS || t1.back() != TOK_EOS)
    {
        std::cerr << "[test_boundaries] no-vocab text wrong size " << t1.size() << "\n";
        return false;
    }

    // 長さ打ち切り: max_len=4 で多数タグ -> 末尾は必ず <eos>、長さ <= 4
    std::vector<std::string> many = {"1girl", "solo", "long hair", "breasts", "shirt", "skirt"};
    std::vector<int> trunc = tok.encode(many, 4);
    if (static_cast<int>(trunc.size()) > 4)
    {
        std::cerr << "[test_boundaries] truncate exceeded max_len: " << trunc.size() << "\n";
        return false;
    }
    if (trunc.front() != TOK_BOS || trunc.back() != TOK_EOS)
    {
        std::cerr << "[test_boundaries] truncate framing wrong\n";
        return false;
    }

    // max_len=0 -> 空
    if (!tok.encode(many, 0).empty())
    {
        std::cerr << "[test_boundaries] max_len=0 not empty\n";
        return false;
    }
    std::cout << "[test_boundaries] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// 実 pairs 全 target タグで UNK 0 (fixture 不在時 [SKIP])
//   jsonl から各行の "tags":[...] を抽出して encode、UNK が 0 件を assert。
// ----------------------------------------------------------------
// jsonl 1 行から "tags":[ "a", "b", ... ] の文字列要素を取り出す簡易抽出。
static std::vector<std::string> extract_tags(const std::string& line)
{
    std::vector<std::string> out;
    size_t key = line.find("\"tags\"");
    if (key == std::string::npos)
    {
        return out;
    }
    size_t lb = line.find('[', key);
    if (lb == std::string::npos)
    {
        return out;
    }
    size_t rb = line.find(']', lb);
    if (rb == std::string::npos)
    {
        return out;
    }
    size_t i = lb + 1;
    while (i < rb)
    {
        size_t q1 = line.find('"', i);
        if (q1 == std::string::npos || q1 >= rb)
        {
            break;
        }
        // 終端クォート (エスケープ非考慮: タグ名にエスケープは出ない)
        size_t q2 = line.find('"', q1 + 1);
        if (q2 == std::string::npos || q2 > rb)
        {
            break;
        }
        out.push_back(line.substr(q1 + 1, q2 - q1 - 1));
        i = q2 + 1;
    }
    return out;
}

static bool run_pairs_file(const Tokenizer& tok, const std::string& path,
                           long& total_tags, long& unk_count, long& lines)
{
    std::ifstream f(path);
    if (!f)
    {
        return false;  // 不在
    }
    std::string line;
    while (std::getline(f, line))
    {
        if (line.empty())
        {
            continue;
        }
        ++lines;
        std::vector<std::string> tags = extract_tags(line);
        std::vector<int> ids = tok.encode(tags, 1024);  // 打ち切りで UNK を消さない
        for (int id : ids)
        {
            if (id == TOK_UNK)
            {
                ++unk_count;
            }
        }
        total_tags += static_cast<long>(tags.size());
    }
    return true;
}

static bool test_real_pairs()
{
    if (g_pairs_train.empty() && g_pairs_val.empty())
    {
        std::cout << "[test_real_pairs] [SKIP] no pairs path embedded\n";
        return true;
    }
    Tokenizer tok(g_vocab);
    long total_tags = 0, unk_count = 0, lines = 0;
    bool any = false;
    if (!g_pairs_train.empty())
    {
        any = run_pairs_file(tok, g_pairs_train, total_tags, unk_count, lines) || any;
    }
    if (!g_pairs_val.empty())
    {
        any = run_pairs_file(tok, g_pairs_val, total_tags, unk_count, lines) || any;
    }
    if (!any)
    {
        std::cout << "[test_real_pairs] [SKIP] pairs files not found\n";
        return true;
    }
    std::cout << "[test_real_pairs] lines=" << lines << " total_tags=" << total_tags
              << " UNK=" << unk_count << "\n";
    if (unk_count != 0)
    {
        std::cerr << "[test_real_pairs] UNK count != 0: " << unk_count << "\n";
        return false;
    }
    std::cout << "[test_real_pairs] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// ベンチ: encode(tags) / decode / encode_text の中央値 ns/op
// ----------------------------------------------------------------
static double median(std::vector<double>& v)
{
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static void bench()
{
    Tokenizer tok(g_vocab);
    std::vector<std::string> tags = {
        "1girl", "solo", "long hair", "blonde hair", "shirt",
        "skirt", "blue eyes", "smile", "large breasts", "blush"
    };
    const std::string text =
        "several girls with shirt, blonde hair, bow, twintails, blue eyes, smile.";

    const int reps = 200;
    const int inner = 1000;

    // encode
    {
        std::vector<double> samples;
        std::vector<int> sink;
        for (int r = 0; r < 5; ++r) sink = tok.encode(tags);  // warmup
        for (int r = 0; r < reps; ++r)
        {
            auto t0 = std::chrono::steady_clock::now();
            for (int k = 0; k < inner; ++k)
            {
                sink = tok.encode(tags);
            }
            auto t1 = std::chrono::steady_clock::now();
            double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / inner;
            samples.push_back(ns);
        }
        std::cout << "[bench] encode(tags x" << tags.size() << ") = "
                  << median(samples) << " ns/op (sink=" << sink.size() << ")\n";
    }

    // decode
    {
        std::vector<int> ids = tok.encode(tags);
        std::vector<double> samples;
        std::vector<std::string> sink;
        for (int r = 0; r < 5; ++r) sink = tok.decode(ids);
        for (int r = 0; r < reps; ++r)
        {
            auto t0 = std::chrono::steady_clock::now();
            for (int k = 0; k < inner; ++k)
            {
                sink = tok.decode(ids);
            }
            auto t1 = std::chrono::steady_clock::now();
            double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / inner;
            samples.push_back(ns);
        }
        std::cout << "[bench] decode = " << median(samples)
                  << " ns/op (sink=" << sink.size() << ")\n";
    }

    // encode_text
    {
        std::vector<double> samples;
        std::vector<int> sink;
        for (int r = 0; r < 5; ++r) sink = tok.encode_text(text);
        for (int r = 0; r < reps; ++r)
        {
            auto t0 = std::chrono::steady_clock::now();
            for (int k = 0; k < inner; ++k)
            {
                sink = tok.encode_text(text);
            }
            auto t1 = std::chrono::steady_clock::now();
            double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / inner;
            samples.push_back(ns);
        }
        std::cout << "[bench] encode_text = " << median(samples)
                  << " ns/op (sink=" << sink.size() << ")\n";
    }
}

} // namespace dollama

int main()
{
    bool ok = true;
    try
    {
        ok = dollama::test_load() && ok;
        ok = dollama::test_invalid_vocab() && ok;
        ok = dollama::test_encode_decode_roundtrip() && ok;
        ok = dollama::test_unknown_tag() && ok;
        ok = dollama::test_normalize() && ok;
        ok = dollama::test_encode_text() && ok;
        ok = dollama::test_framing() && ok;
        ok = dollama::test_boundaries() && ok;
        ok = dollama::test_real_pairs() && ok;
        dollama::bench();
    }
    catch (const std::exception& e)
    {
        std::cerr << "[test_tokenizer] EXCEPTION: " << e.what() << "\n";
        return 1;
    }

    if (!ok)
    {
        std::cerr << "[test_tokenizer] FAILED\n";
        return 1;
    }
    std::cout << "[test_tokenizer] ALL PASSED\n";
    return 0;
}

#pragma once
// dollama タグトークナイザ (ヘッダオンリー・純ホスト C++)
//
// 実体は vocab.json 駆動の「タグ単位完全一致トークナイザ」。
// サブワード BPE ではない。dataset-spec.md §3.2 / §6 / §10 / §11 準拠。
//
//   フォーマット (vocab.json):
//     {
//       "version": 1,
//       "separator": "space",
//       "specials": ["<pad>", "<bos>", "<eos>", "<sep>", "<unk>"],
//       "tags": [ {"id":5, "tag":"1girl", "category":0, "freq":...}, ... ]
//     }
//
//   特殊トークン (id 0..4 固定予約):
//     <pad>=0, <bos>=1, <eos>=2, <sep>=3, <unk>=4
//   タグ id は specials の続き (5 から) で連番 (tags[i].id == 5 + i)。
//   総語彙 = specials 5 + tags 4994 = 4999 (bitnet.hpp VOCAB_SIZE と一致)。
//
//   正規化 (§6):
//     danbooru の `_` 区切りを受けても引けるよう、「英数字に挟まれた `_`」のみ
//     半角スペースへ置換する (long_hair -> long hair)。顔文字系 (^_^, >_<) の
//     記号に隣接する `_` は保持 (置換しない)。内部はスペース区切り正準形で
//     完全一致引き。
//
// CUDA / OpenVINO 非依存。STL のみで自己完結する。

#include <cctype>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace dollama {

// トークナイザの特殊トークン id (id 0..4 固定予約)。
enum TokenId : int
{
    TOK_PAD = 0,
    TOK_BOS = 1,
    TOK_EOS = 2,
    TOK_SEP = 3,
    TOK_UNK = 4,
};

class Tokenizer
{
public:
    // 既定の長さ打ち切り (bitnet.hpp MAX_SEQ_LEN と整合)。
    static constexpr int DEFAULT_MAX_SEQ_LEN = 64;

    // vocab.json をロードして検証する。違反は std::runtime_error。
    explicit Tokenizer(const std::string& vocab_path)
    {
        std::string json = read_file(vocab_path);
        parse_vocab(json);
        validate();
        build_index();
    }

    // 登録タグ数 (specials を含まない)。
    size_t tag_count() const
    {
        return tags_.size();
    }

    // 総語彙サイズ (specials 5 + tags)。
    size_t vocab_size() const
    {
        return specials_.size() + tags_.size();
    }

    // id -> 表記文字列。specials は "<pad>" 等、タグは正準形文字列。範囲外は "<unk>"。
    std::string id_to_token(int id) const
    {
        if (id >= 0 && static_cast<size_t>(id) < specials_.size())
        {
            return specials_[static_cast<size_t>(id)];
        }
        size_t ti = static_cast<size_t>(id) - specials_.size();
        if (id >= 0 && ti < tags_.size())
        {
            return tags_[ti];
        }
        return specials_.empty() ? std::string("<unk>") : specials_[TOK_UNK];
    }

    // 単一タグ文字列 -> id。正規化して完全一致引き。未知なら <unk>(4)。
    int tag_to_id(const std::string& tag) const
    {
        std::string norm = normalize(tag);
        auto it = tag_to_id_.find(norm);
        if (it == tag_to_id_.end())
        {
            return TOK_UNK;
        }
        return it->second;
    }

    // ------------------------------------------------------------------
    // encode(tags[]) : タグ列 -> id 列。<bos> ... <eos> フレーミング。
    //   未知タグは <unk>(4)。max_len で全体 (bos/eos 含む) を打ち切る。
    // ------------------------------------------------------------------
    std::vector<int> encode(const std::vector<std::string>& tags,
                            int max_len = DEFAULT_MAX_SEQ_LEN) const
    {
        std::vector<int> out;
        if (max_len <= 0)
        {
            return out;
        }
        out.reserve(tags.size() + 2);
        out.push_back(TOK_BOS);
        for (const std::string& t : tags)
        {
            // <eos> 用に 1 枠残す。
            if (static_cast<int>(out.size()) >= max_len - 1)
            {
                break;
            }
            out.push_back(tag_to_id(t));
        }
        // <eos> を入れる余地があれば付与。なければ末尾を <eos> に詰める。
        if (static_cast<int>(out.size()) < max_len)
        {
            out.push_back(TOK_EOS);
        }
        else
        {
            out[out.size() - 1] = TOK_EOS;
        }
        return out;
    }

    // ------------------------------------------------------------------
    // encode_text(text) : 自然文 -> id 列 (greedy 最長一致セグメント)。
    //   語彙にマッチする最長タグ区間をトークン化し、非語彙区間はスキップ。
    //   例: "long hair" を "long"+"hair" に割らず "long hair" 1 トークン。
    //   <bos> ... <eos> フレーミング。max_len で打ち切る。
    // ------------------------------------------------------------------
    std::vector<int> encode_text(const std::string& text,
                                 int max_len = DEFAULT_MAX_SEQ_LEN) const
    {
        std::vector<int> out;
        if (max_len <= 0)
        {
            return out;
        }
        out.push_back(TOK_BOS);

        // テキストを正規化 (_ -> space) してから語境界で走査する。
        std::string norm = normalize(text);

        // 単語トークン (英数字連結) の開始位置一覧を作る。
        // 各単語境界から「最長何単語まで連結するとタグにマッチするか」を貪欲に探す。
        std::vector<std::pair<size_t, size_t>> words;  // [begin, end)
        tokenize_words(norm, words);

        size_t wi = 0;
        while (wi < words.size())
        {
            if (static_cast<int>(out.size()) >= max_len - 1)
            {
                break;
            }
            // wi から最長一致を探す: 連結語数 j を最大から試す。
            int matched_id = TOK_UNK;
            size_t matched_words = 0;
            // 連結上限は max_tag_words_ (語彙中の最長タグの単語数)。
            size_t max_j = words.size() - wi;
            if (max_j > max_tag_words_)
            {
                max_j = max_tag_words_;
            }
            for (size_t j = max_j; j >= 1; --j)
            {
                size_t b = words[wi].first;
                size_t e = words[wi + j - 1].second;
                std::string cand = norm.substr(b, e - b);
                auto it = tag_to_id_.find(cand);
                if (it != tag_to_id_.end())
                {
                    matched_id = it->second;
                    matched_words = j;
                    break;
                }
            }
            if (matched_words > 0)
            {
                out.push_back(matched_id);
                wi += matched_words;
            }
            else
            {
                // 非語彙単語はスキップ (v1 方針: 接続語等を捨てる)。
                ++wi;
            }
        }

        if (static_cast<int>(out.size()) < max_len)
        {
            out.push_back(TOK_EOS);
        }
        else
        {
            out[out.size() - 1] = TOK_EOS;
        }
        return out;
    }

    // ------------------------------------------------------------------
    // decode(ids[]) : id 列 -> タグ文字列列。
    //   <bos>/<eos>/<pad>/<sep> はスキップ (構造トークンなので出力しない)。
    //   <unk> は "<unk>" 文字列で復元 (情報欠損を可視化)。タグは正準形。
    // ------------------------------------------------------------------
    std::vector<std::string> decode(const std::vector<int>& ids) const
    {
        std::vector<std::string> out;
        out.reserve(ids.size());
        for (int id : ids)
        {
            if (id == TOK_BOS || id == TOK_EOS || id == TOK_PAD || id == TOK_SEP)
            {
                continue;  // 構造トークンは出力しない
            }
            if (id == TOK_UNK)
            {
                out.push_back(specials_[TOK_UNK]);
                continue;
            }
            out.push_back(id_to_token(id));
        }
        return out;
    }

    // ------------------------------------------------------------------
    // 正規化 (§6): 英数字に挟まれた `_` のみ ` ` に置換。
    //   それ以外の `_` (記号隣接の顔文字 ^_^ / >_< 等) は保持。
    // ------------------------------------------------------------------
    static std::string normalize(const std::string& s)
    {
        std::string out;
        out.reserve(s.size());
        for (size_t i = 0; i < s.size(); ++i)
        {
            char c = s[i];
            if (c == '_')
            {
                bool prev_alnum = (i > 0) && is_alnum(s[i - 1]);
                bool next_alnum = (i + 1 < s.size()) && is_alnum(s[i + 1]);
                if (prev_alnum && next_alnum)
                {
                    out.push_back(' ');
                    continue;
                }
            }
            out.push_back(c);
        }
        return out;
    }

private:
    std::vector<std::string>             specials_;       // index = id (0..4)
    std::vector<std::string>             tags_;           // index = id - 5
    std::unordered_map<std::string, int> tag_to_id_;      // 正準形 -> id
    int                                  version_ = 0;
    std::string                          separator_;
    size_t                               max_tag_words_ = 1;  // 語彙中の最長タグ単語数

    static bool is_alnum(char c)
    {
        unsigned char uc = static_cast<unsigned char>(c);
        return std::isalnum(uc) != 0;
    }

    // 正規化済み文字列を「英数字連結の単語」へ分割 ([begin,end) 一覧)。
    // 区切りは空白・カンマ・ピリオド等の非英数字。タグ照合の単位語を作る。
    static void tokenize_words(const std::string& s,
                               std::vector<std::pair<size_t, size_t>>& words)
    {
        size_t i = 0;
        while (i < s.size())
        {
            // 非英数字 (区切り) をスキップ。
            while (i < s.size() && !is_alnum(s[i]))
            {
                ++i;
            }
            if (i >= s.size())
            {
                break;
            }
            size_t b = i;
            while (i < s.size() && is_alnum(s[i]))
            {
                ++i;
            }
            words.emplace_back(b, i);
        }
    }

    static std::string read_file(const std::string& path)
    {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f)
        {
            throw std::runtime_error("tokenizer: cannot open vocab file: " + path);
        }
        std::streamoff sz = f.tellg();
        if (sz < 0)
        {
            throw std::runtime_error("tokenizer: cannot stat vocab file: " + path);
        }
        f.seekg(0, std::ios::beg);
        std::string s;
        s.resize(static_cast<size_t>(sz));
        if (sz > 0 && !f.read(s.data(), sz))
        {
            throw std::runtime_error("tokenizer: vocab read failed: " + path);
        }
        return s;
    }

    // ------------------------------------------------------------------
    // vocab.json パース (safetensors.hpp の最小 JSON パーサ流儀を踏襲)。
    // 必要キー: version (int) / separator (str) / specials ([str]) /
    //           tags ([{id,tag,category,freq}])。未知キー (_build 等) は無視。
    // ------------------------------------------------------------------
    void parse_vocab(const std::string& s)
    {
        size_t i = 0;
        skip_ws(s, i);
        expect(s, i, '{');
        skip_ws(s, i);
        if (peek(s, i) == '}')
        {
            return;
        }
        while (true)
        {
            skip_ws(s, i);
            std::string key = parse_string(s, i);
            skip_ws(s, i);
            expect(s, i, ':');
            skip_ws(s, i);

            if (key == "version")
            {
                version_ = static_cast<int>(parse_int(s, i));
            }
            else if (key == "separator")
            {
                separator_ = parse_string(s, i);
            }
            else if (key == "specials")
            {
                specials_ = parse_string_array(s, i);
            }
            else if (key == "tags")
            {
                parse_tags(s, i);
            }
            else
            {
                skip_value(s, i);
            }

            skip_ws(s, i);
            char c = peek(s, i);
            if (c == ',')
            {
                ++i;
                continue;
            }
            if (c == '}')
            {
                ++i;
                break;
            }
            throw std::runtime_error("tokenizer: vocab JSON syntax error (expected ',' or '}')");
        }
    }

    // "tags": [ {id,tag,category,freq}, ... ] を読む。
    // tag 文字列を tags_ に id 順 (= 配列順) で詰める。同時に id 連番を検証。
    void parse_tags(const std::string& s, size_t& i)
    {
        expect(s, i, '[');
        skip_ws(s, i);
        if (peek(s, i) == ']')
        {
            ++i;
            return;
        }
        size_t index = 0;
        while (true)
        {
            skip_ws(s, i);
            int         obj_id = -1;
            std::string obj_tag;
            bool        has_id = false, has_tag = false;

            expect(s, i, '{');
            skip_ws(s, i);
            if (peek(s, i) == '}')
            {
                ++i;
                throw std::runtime_error("tokenizer: empty tag object");
            }
            while (true)
            {
                skip_ws(s, i);
                std::string field = parse_string(s, i);
                skip_ws(s, i);
                expect(s, i, ':');
                skip_ws(s, i);
                if (field == "id")
                {
                    obj_id = static_cast<int>(parse_int(s, i));
                    has_id = true;
                }
                else if (field == "tag")
                {
                    obj_tag = parse_string(s, i);
                    has_tag = true;
                }
                else
                {
                    skip_value(s, i);  // category / freq / 未知
                }
                skip_ws(s, i);
                char c = peek(s, i);
                if (c == ',')
                {
                    ++i;
                    continue;
                }
                if (c == '}')
                {
                    ++i;
                    break;
                }
                throw std::runtime_error("tokenizer: tag object syntax error");
            }

            if (!has_id || !has_tag)
            {
                throw std::runtime_error("tokenizer: tag object missing id/tag");
            }
            // id 連番検証 (specials 数 + index)。specials は tags より先に読まれる
            // 前提だが、順序非依存にするため検証は validate() で行う。
            // ここでは配列順で詰め、id も保持して後で突合する。
            int expect_id = static_cast<int>(specials_.size() + index);
            // specials がまだ読まれていない場合 (順序逆) は size 0 になるため、
            // 厳密な連番検証は validate() に委ねる。ここでは緩く格納する。
            (void)expect_id;
            tags_.push_back(obj_tag);
            tag_ids_.push_back(obj_id);

            ++index;
            skip_ws(s, i);
            char c = peek(s, i);
            if (c == ',')
            {
                ++i;
                continue;
            }
            if (c == ']')
            {
                ++i;
                break;
            }
            throw std::runtime_error("tokenizer: tags array syntax error");
        }
    }

    std::vector<int> tag_ids_;  // 検証用 (各 tag の宣言 id)

    // ------------------------------------------------------------------
    // ロード時検証 (dataset-spec §3.2):
    //   specials 5 件 (id 0..4 = <pad>/<bos>/<eos>/<sep>/<unk>)、
    //   tags[i].id == 5 + i 連番、総語彙 4999。
    // ------------------------------------------------------------------
    void validate()
    {
        static const char* kExpectSpecials[5] = {
            "<pad>", "<bos>", "<eos>", "<sep>", "<unk>"
        };
        if (specials_.size() != 5)
        {
            throw std::runtime_error(
                "tokenizer: specials must be 5 entries, got " +
                std::to_string(specials_.size()));
        }
        for (size_t k = 0; k < 5; ++k)
        {
            if (specials_[k] != kExpectSpecials[k])
            {
                throw std::runtime_error(
                    "tokenizer: specials[" + std::to_string(k) + "] = '" +
                    specials_[k] + "' expected '" + kExpectSpecials[k] + "'");
            }
        }
        if (tags_.size() != tag_ids_.size())
        {
            throw std::runtime_error("tokenizer: internal tag/id count mismatch");
        }
        for (size_t k = 0; k < tags_.size(); ++k)
        {
            int expect_id = static_cast<int>(5 + k);
            if (tag_ids_[k] != expect_id)
            {
                throw std::runtime_error(
                    "tokenizer: tags[" + std::to_string(k) + "].id = " +
                    std::to_string(tag_ids_[k]) + " expected " +
                    std::to_string(expect_id) + " (id must be 5+i contiguous)");
            }
        }
        size_t total = specials_.size() + tags_.size();
        if (total != 4999)
        {
            throw std::runtime_error(
                "tokenizer: total vocab must be 4999, got " + std::to_string(total));
        }
    }

    // タグ文字列 -> id の逆引き索引を作る。最長タグ単語数も求める。
    void build_index()
    {
        tag_to_id_.reserve(tags_.size() * 2);
        for (size_t k = 0; k < tags_.size(); ++k)
        {
            // vocab の tag は正準形 (スペース区切り) だが、念のため正規化を通す。
            std::string norm = normalize(tags_[k]);
            int id = static_cast<int>(5 + k);
            tag_to_id_[norm] = id;

            // 単語数 (空白区切り) を数えて max_tag_words_ を更新。
            size_t wc = count_words(norm);
            if (wc > max_tag_words_)
            {
                max_tag_words_ = wc;
            }
        }
    }

    static size_t count_words(const std::string& s)
    {
        size_t n = 0;
        size_t i = 0;
        while (i < s.size())
        {
            while (i < s.size() && !is_alnum(s[i]))
            {
                ++i;
            }
            if (i >= s.size())
            {
                break;
            }
            ++n;
            while (i < s.size() && is_alnum(s[i]))
            {
                ++i;
            }
        }
        return n;
    }

    // ------------------------------------------------------------------
    // 最小 JSON 字句ヘルパ (safetensors.hpp 流儀)。
    // ------------------------------------------------------------------
    static void skip_ws(const std::string& s, size_t& i)
    {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r'))
        {
            ++i;
        }
    }

    static char peek(const std::string& s, size_t i)
    {
        if (i >= s.size())
        {
            throw std::runtime_error("tokenizer: JSON ended prematurely");
        }
        return s[i];
    }

    static void expect(const std::string& s, size_t& i, char c)
    {
        if (peek(s, i) != c)
        {
            throw std::runtime_error(std::string("tokenizer: JSON expected '") + c + "'");
        }
        ++i;
    }

    static std::string parse_string(const std::string& s, size_t& i)
    {
        expect(s, i, '"');
        std::string out;
        while (i < s.size())
        {
            char c = s[i++];
            if (c == '"')
            {
                return out;
            }
            if (c == '\\')
            {
                if (i >= s.size())
                {
                    break;
                }
                char e = s[i++];
                switch (e)
                {
                case '"':
                    out.push_back('"');
                    break;
                case '\\':
                    out.push_back('\\');
                    break;
                case '/':
                    out.push_back('/');
                    break;
                case 'n':
                    out.push_back('\n');
                    break;
                case 't':
                    out.push_back('\t');
                    break;
                case 'r':
                    out.push_back('\r');
                    break;
                case 'b':
                    out.push_back('\b');
                    break;
                case 'f':
                    out.push_back('\f');
                    break;
                case 'u':
                    out += parse_unicode_escape(s, i);
                    break;
                default:
                    out.push_back(e);
                    break;
                }
            }
            else
            {
                out.push_back(c);
            }
        }
        throw std::runtime_error("tokenizer: unterminated string");
    }

    // \uXXXX を UTF-8 へ変換 (サロゲートペア対応)。i は 'u' の次を指す。
    static std::string parse_unicode_escape(const std::string& s, size_t& i)
    {
        uint32_t cp = parse_hex4(s, i);
        // 上位サロゲート: 後続の \uXXXX と結合。
        if (cp >= 0xD800 && cp <= 0xDBFF)
        {
            if (i + 1 < s.size() && s[i] == '\\' && s[i + 1] == 'u')
            {
                i += 2;
                uint32_t lo = parse_hex4(s, i);
                if (lo >= 0xDC00 && lo <= 0xDFFF)
                {
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                }
            }
        }
        return utf8_encode(cp);
    }

    static uint32_t parse_hex4(const std::string& s, size_t& i)
    {
        uint32_t v = 0;
        for (int k = 0; k < 4; ++k)
        {
            char c = peek(s, i++);
            v <<= 4;
            if (c >= '0' && c <= '9')
            {
                v |= static_cast<uint32_t>(c - '0');
            }
            else if (c >= 'a' && c <= 'f')
            {
                v |= static_cast<uint32_t>(c - 'a' + 10);
            }
            else if (c >= 'A' && c <= 'F')
            {
                v |= static_cast<uint32_t>(c - 'A' + 10);
            }
            else
            {
                throw std::runtime_error("tokenizer: invalid \\u hex digit");
            }
        }
        return v;
    }

    static std::string utf8_encode(uint32_t cp)
    {
        std::string out;
        if (cp <= 0x7F)
        {
            out.push_back(static_cast<char>(cp));
        }
        else if (cp <= 0x7FF)
        {
            out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
        else if (cp <= 0xFFFF)
        {
            out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
        else
        {
            out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
        return out;
    }

    // 整数 (符号付き) を読む。
    static long long parse_int(const std::string& s, size_t& i)
    {
        size_t start = i;
        if (i < s.size() && (s[i] == '-' || s[i] == '+'))
        {
            ++i;
        }
        while (i < s.size() && s[i] >= '0' && s[i] <= '9')
        {
            ++i;
        }
        if (i == start || (i == start + 1 && !std::isdigit(static_cast<unsigned char>(s[start]))))
        {
            throw std::runtime_error("tokenizer: expected integer");
        }
        return std::stoll(s.substr(start, i - start));
    }

    static std::vector<std::string> parse_string_array(const std::string& s, size_t& i)
    {
        std::vector<std::string> out;
        expect(s, i, '[');
        skip_ws(s, i);
        if (peek(s, i) == ']')
        {
            ++i;
            return out;
        }
        while (true)
        {
            skip_ws(s, i);
            out.push_back(parse_string(s, i));
            skip_ws(s, i);
            char c = peek(s, i);
            if (c == ',')
            {
                ++i;
                continue;
            }
            if (c == ']')
            {
                ++i;
                break;
            }
            throw std::runtime_error("tokenizer: string array syntax error");
        }
        return out;
    }

    static void skip_value(const std::string& s, size_t& i)
    {
        skip_ws(s, i);
        char c = peek(s, i);
        if (c == '"')
        {
            parse_string(s, i);
        }
        else if (c == '{')
        {
            skip_container(s, i, '{', '}');
        }
        else if (c == '[')
        {
            skip_container(s, i, '[', ']');
        }
        else
        {
            while (i < s.size() && s[i] != ',' && s[i] != '}' && s[i] != ']')
            {
                ++i;
            }
        }
    }

    static void skip_container(const std::string& s, size_t& i, char open, char close)
    {
        expect(s, i, open);
        int depth = 1;
        while (i < s.size() && depth > 0)
        {
            char c = s[i];
            if (c == '"')
            {
                parse_string(s, i);
                continue;
            }
            if (c == open)
            {
                ++depth;
            }
            else if (c == close)
            {
                --depth;
            }
            ++i;
        }
        if (depth != 0)
        {
            throw std::runtime_error("tokenizer: unterminated container");
        }
    }
};

} // namespace dollama

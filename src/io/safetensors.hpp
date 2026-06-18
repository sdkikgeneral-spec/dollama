#pragma once
// safetensors 重みローダー (ヘッダオンリー)
//
// フォーマット (https://github.com/huggingface/safetensors の README フォーマット節):
//   [0..8)        : リトルエンディアン uint64 = JSON ヘッダのバイト長 N
//   [8..8+N)      : UTF-8 JSON ヘッダ
//   [8+N..EOF)    : テンソルデータ本体 (raw bytes)
// 各テンソルは JSON 内の "data_offsets":[begin,end) で
// 「データ本体先頭からの相対 raw bytes 範囲」を指す。
// "__metadata__" キーはテンソルではないので無視する。
//
// 注意: ここに置く JSON パーサは **safetensors ヘッダ専用スコープ**に閉じる。
//       safetensors ヘッダは構造が固定 (オブジェクトの値は文字列か配列か
//       ネストオブジェクトのみ) なので最小実装で足りる。汎用 JSON が必要に
//       なったら Phase 3 で導入する nlohmann/json に一本化すること。
//       ここで JSON 機能を拡張しない。

#include <cstdint>
#include <cstddef>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace dollama {

// safetensors がサポートする dtype。
enum class StDtype
{
    F32,
    F16,
    BF16,
    I64,
    I32,
    I8,
    U8,
};

// dtype の 1 要素あたりバイト数。
inline size_t st_itemsize(StDtype dt)
{
    switch (dt)
    {
    case StDtype::F32:
        return 4;
    case StDtype::F16:
        return 2;
    case StDtype::BF16:
        return 2;
    case StDtype::I64:
        return 8;
    case StDtype::I32:
        return 4;
    case StDtype::I8:
        return 1;
    case StDtype::U8:
        return 1;
    }
    throw std::runtime_error("st_itemsize: unknown dtype");
}

// dtype 文字列 → enum。未知なら例外。
inline StDtype st_dtype_from_string(const std::string& s)
{
    if (s == "F32")  return StDtype::F32;
    if (s == "F16")  return StDtype::F16;
    if (s == "BF16") return StDtype::BF16;
    if (s == "I64")  return StDtype::I64;
    if (s == "I32")  return StDtype::I32;
    if (s == "I8")   return StDtype::I8;
    if (s == "U8")   return StDtype::U8;
    throw std::runtime_error("safetensors: unknown dtype string '" + s + "'");
}

// dtype 文字列表現 (デバッグ・突合用)。
inline const char* st_dtype_name(StDtype dt)
{
    switch (dt)
    {
    case StDtype::F32:
        return "F32";
    case StDtype::F16:
        return "F16";
    case StDtype::BF16:
        return "BF16";
    case StDtype::I64:
        return "I64";
    case StDtype::I32:
        return "I32";
    case StDtype::I8:
        return "I8";
    case StDtype::U8:
        return "U8";
    }
    return "?";
}

// 1 テンソルのメタ情報。
struct StTensorInfo
{
    StDtype             dtype = StDtype::F32;
    std::vector<size_t> shape;
    size_t              begin = 0;  // データ本体先頭からの相対バイト
    size_t              end   = 0;  // 同上 (排他)
};

class SafeTensors
{
public:
    // ファイルを全読みしてヘッダを解析する。
    explicit SafeTensors(const std::string& path)
    {
        load_file(path);
        parse_header();
    }

    // 登録テンソル名一覧 (JSON 出現順は保持しない。map のキー順 = 辞書順)。
    std::vector<std::string> names() const
    {
        std::vector<std::string> out;
        out.reserve(infos_.size());
        for (const auto& kv : infos_)
        {
            out.push_back(kv.first);
        }
        return out;
    }

    bool contains(const std::string& name) const
    {
        return infos_.find(name) != infos_.end();
    }

    StDtype dtype(const std::string& name) const
    {
        return info(name).dtype;
    }

    const std::vector<size_t>& shape(const std::string& name) const
    {
        return info(name).shape;
    }

    // テンソルの生バイト列 (const ポインタ + バイト数)。
    // ポインタは SafeTensors の生存期間中だけ有効。
    const uint8_t* tensor_bytes(const std::string& name, size_t& out_nbytes) const
    {
        const StTensorInfo& ti = info(name);
        out_nbytes = ti.end - ti.begin;
        return data_.data() + data_offset_ + ti.begin;
    }

private:
    std::vector<uint8_t>             data_;         // ファイル全体
    size_t                           data_offset_ = 0;  // データ本体の先頭オフセット (= 8 + N)
    std::map<std::string, StTensorInfo> infos_;

    const StTensorInfo& info(const std::string& name) const
    {
        auto it = infos_.find(name);
        if (it == infos_.end())
        {
            throw std::runtime_error("safetensors: tensor '" + name + "' not found");
        }
        return it->second;
    }

    void load_file(const std::string& path)
    {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f)
        {
            throw std::runtime_error("safetensors: cannot open file: " + path);
        }
        std::streamoff sz = f.tellg();
        if (sz < 8)
        {
            throw std::runtime_error("safetensors: file too small (<8 bytes)");
        }
        f.seekg(0, std::ios::beg);
        data_.resize(static_cast<size_t>(sz));
        if (!f.read(reinterpret_cast<char*>(data_.data()), sz))
        {
            throw std::runtime_error("safetensors: file read failed: " + path);
        }
    }

    void parse_header()
    {
        // 先頭 8 バイト LE = ヘッダ長 N
        uint64_t n = 0;
        for (int i = 0; i < 8; ++i)
        {
            n |= static_cast<uint64_t>(data_[i]) << (8 * i);
        }
        if (n > data_.size() - 8)
        {
            throw std::runtime_error("safetensors: header length exceeds file size");
        }
        data_offset_ = 8 + static_cast<size_t>(n);

        std::string json(reinterpret_cast<const char*>(data_.data() + 8),
                         static_cast<size_t>(n));

        const size_t body_len = data_.size() - data_offset_;
        parse_json_header(json, body_len);
    }

    // ------------------------------------------------------------------
    // 最小 JSON パーサ (safetensors ヘッダ専用)
    // トップレベルオブジェクト { "name": {dtype,shape,data_offsets}, ... }
    // ------------------------------------------------------------------
    void parse_json_header(const std::string& s, size_t body_len)
    {
        size_t i = 0;
        skip_ws(s, i);
        expect(s, i, '{');
        skip_ws(s, i);
        if (peek(s, i) == '}')
        {
            return;  // 空ヘッダ
        }
        while (true)
        {
            skip_ws(s, i);
            std::string key = parse_string(s, i);
            skip_ws(s, i);
            expect(s, i, ':');
            skip_ws(s, i);

            if (key == "__metadata__")
            {
                // メタデータオブジェクトはスキップ (テンソルではない)
                skip_value(s, i);
            }
            else
            {
                StTensorInfo ti = parse_tensor_object(s, i);
                validate(key, ti, body_len);
                infos_[key] = std::move(ti);
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
            throw std::runtime_error("safetensors: JSON syntax error (expected ',' or '}')");
        }
    }

    // テンソルオブジェクト {dtype, shape, data_offsets} を読む
    StTensorInfo parse_tensor_object(const std::string& s, size_t& i)
    {
        StTensorInfo ti;
        bool has_dtype = false, has_shape = false, has_offsets = false;

        expect(s, i, '{');
        skip_ws(s, i);
        if (peek(s, i) == '}')
        {
            ++i;
            throw std::runtime_error("safetensors: empty tensor object");
        }
        while (true)
        {
            skip_ws(s, i);
            std::string field = parse_string(s, i);
            skip_ws(s, i);
            expect(s, i, ':');
            skip_ws(s, i);

            if (field == "dtype")
            {
                ti.dtype  = st_dtype_from_string(parse_string(s, i));
                has_dtype = true;
            }
            else if (field == "shape")
            {
                ti.shape  = parse_uint_array(s, i);
                has_shape = true;
            }
            else if (field == "data_offsets")
            {
                std::vector<size_t> off = parse_uint_array(s, i);
                if (off.size() != 2)
                {
                    throw std::runtime_error("safetensors: data_offsets must have 2 elements");
                }
                ti.begin    = off[0];
                ti.end      = off[1];
                has_offsets = true;
            }
            else
            {
                // 未知フィールドは無視
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
            throw std::runtime_error("safetensors: tensor object syntax error");
        }

        if (!has_dtype || !has_shape || !has_offsets)
        {
            throw std::runtime_error("safetensors: missing dtype/shape/data_offsets");
        }
        return ti;
    }

    // dtype・offset・shape の整合性検証
    void validate(const std::string& name, const StTensorInfo& ti, size_t body_len)
    {
        if (ti.end < ti.begin)
        {
            throw std::runtime_error("safetensors: '" + name + "' data_offsets reversed");
        }
        if (ti.end > body_len)
        {
            throw std::runtime_error("safetensors: '" + name + "' offset out of data body range");
        }
        // shape 要素数の積 × itemsize がバイト長と一致するか
        size_t numel = 1;
        for (size_t d : ti.shape)
        {
            numel *= d;
        }
        size_t expect_bytes = numel * st_itemsize(ti.dtype);
        size_t actual_bytes = ti.end - ti.begin;
        if (expect_bytes != actual_bytes)
        {
            throw std::runtime_error(
                "safetensors: '" + name + "' shape*itemsize=" +
                std::to_string(expect_bytes) + " vs byte length " +
                std::to_string(actual_bytes) + " mismatch");
        }
    }

    // ------------------------------------------------------------------
    // 低レベル字句ヘルパ
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
            throw std::runtime_error("safetensors: JSON ended prematurely");
        }
        return s[i];
    }

    static void expect(const std::string& s, size_t& i, char c)
    {
        if (peek(s, i) != c)
        {
            throw std::runtime_error(std::string("safetensors: JSON expected '") + c + "'");
        }
        ++i;
    }

    // "..." を読む (基本的なエスケープのみ対応)
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
                default:
                    out.push_back(e);  // それ以外はそのまま (ヘッダでは出現しない想定)
                    break;
                }
            }
            else
            {
                out.push_back(c);
            }
        }
        throw std::runtime_error("safetensors: unterminated string");
    }

    // 非負整数の配列 [a, b, ...] を読む
    static std::vector<size_t> parse_uint_array(const std::string& s, size_t& i)
    {
        std::vector<size_t> out;
        expect(s, i, '[');
        skip_ws(s, i);
        if (peek(s, i) == ']')
        {
            ++i;
            return out;  // 空配列 (スカラー shape)
        }
        while (true)
        {
            skip_ws(s, i);
            out.push_back(parse_uint(s, i));
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
            throw std::runtime_error("safetensors: array syntax error");
        }
        return out;
    }

    static size_t parse_uint(const std::string& s, size_t& i)
    {
        size_t start = i;
        while (i < s.size() && s[i] >= '0' && s[i] <= '9')
        {
            ++i;
        }
        if (i == start)
        {
            throw std::runtime_error("safetensors: expected integer");
        }
        return static_cast<size_t>(std::stoull(s.substr(start, i - start)));
    }

    // 任意の値を読み飛ばす (__metadata__ や未知フィールド用)
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
            // 数値・true/false/null など
            while (i < s.size() && s[i] != ',' && s[i] != '}' && s[i] != ']')
            {
                ++i;
            }
        }
    }

    // 入れ子のコンテナを括弧対応を見ながら読み飛ばす (文字列内の括弧は無視)
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
            throw std::runtime_error("safetensors: unterminated container");
        }
    }
};

} // namespace dollama

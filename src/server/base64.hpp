// Base64 エンコード / デコード (配管小物)
//
// 自作価値が薄い配管だが、cpp-httplib 付属の base64_encode は detail 名前空間で
// API 安定性が保証されないこと・デコード (テスト往復検証用) が無いことから、
// 数十行の自己完結実装を置く。RFC 4648 標準アルファベット (パディング '=')。
// CUDA / OpenVINO 依存ゼロの純 C++。
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace dollama
{

inline std::string base64_encode(const std::vector<uint8_t>& data)
{
    static const char* tbl =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((data.size() + 2) / 3) * 4);

    size_t i = 0;
    const size_t n = data.size();
    while (i + 3 <= n)
    {
        const uint32_t v = (static_cast<uint32_t>(data[i]) << 16) |
                           (static_cast<uint32_t>(data[i + 1]) << 8) |
                           static_cast<uint32_t>(data[i + 2]);
        out.push_back(tbl[(v >> 18) & 0x3F]);
        out.push_back(tbl[(v >> 12) & 0x3F]);
        out.push_back(tbl[(v >> 6) & 0x3F]);
        out.push_back(tbl[v & 0x3F]);
        i += 3;
    }

    const size_t rem = n - i;
    if (rem == 1)
    {
        const uint32_t v = static_cast<uint32_t>(data[i]) << 16;
        out.push_back(tbl[(v >> 18) & 0x3F]);
        out.push_back(tbl[(v >> 12) & 0x3F]);
        out.push_back('=');
        out.push_back('=');
    }
    else if (rem == 2)
    {
        const uint32_t v = (static_cast<uint32_t>(data[i]) << 16) |
                           (static_cast<uint32_t>(data[i + 1]) << 8);
        out.push_back(tbl[(v >> 18) & 0x3F]);
        out.push_back(tbl[(v >> 12) & 0x3F]);
        out.push_back(tbl[(v >> 6) & 0x3F]);
        out.push_back('=');
    }
    return out;
}

inline std::string base64_encode(const std::string& data)
{
    return base64_encode(std::vector<uint8_t>(data.begin(), data.end()));
}

// デコード (主にテスト往復検証用)。不正文字は無視する寛容実装。
inline std::vector<uint8_t> base64_decode(const std::string& s)
{
    auto dec = [](char c) -> int
    {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+') return 62;
        if (c == '/') return 63;
        return -1; // '=' や空白など
    };

    std::vector<uint8_t> out;
    uint32_t buf = 0;
    int bits = 0;
    for (char c : s)
    {
        if (c == '=')
        {
            break;
        }
        const int d = dec(c);
        if (d < 0)
        {
            continue; // 改行などスキップ
        }
        buf = (buf << 6) | static_cast<uint32_t>(d);
        bits += 6;
        if (bits >= 8)
        {
            bits -= 8;
            out.push_back(static_cast<uint8_t>((buf >> bits) & 0xFF));
        }
    }
    return out;
}

} // namespace dollama

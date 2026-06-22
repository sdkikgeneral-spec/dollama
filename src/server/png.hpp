// 最小 PNG エンコーダ (RGB8 / RGBA8, zlib stored block)
//
// 目的: 生成画像 (RGB8 / RGBA8 ピクセル列) を PNG バイト列に変換する。
// 圧縮は行わず zlib の「非圧縮ブロック (stored / BTYPE=00)」で IDAT を構築する。
// 圧縮率は犠牲になるが、依存ライブラリゼロ・実装が小さく検証しやすい。
// CUDA / OpenVINO 依存は一切ない純 C++。
//
// PNG 構造: シグネチャ + IHDR + IDAT + IEND。
//   - IHDR: width, height, bit_depth=8, color_type=2 (RGB) または 6 (RGBA)
//   - 各スキャンラインの先頭に filter byte 0x00 を付ける (フィルタなし)
//   - IDAT は zlib ストリーム (2 バイトヘッダ + deflate stored blocks + Adler-32)
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace dollama
{

namespace detail
{

// CRC-32 (PNG チャンク用, IEEE 多項式 0xEDB88320)
inline uint32_t crc32_update(uint32_t crc, const uint8_t* data, size_t len)
{
    static uint32_t table[256];
    static bool init = false;
    if (!init)
    {
        for (uint32_t n = 0; n < 256; ++n)
        {
            uint32_t c = n;
            for (int k = 0; k < 8; ++k)
            {
                c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            }
            table[n] = c;
        }
        init = true;
    }
    crc ^= 0xFFFFFFFFu;
    for (size_t i = 0; i < len; ++i)
    {
        crc = table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    }
    return crc ^ 0xFFFFFFFFu;
}

// Adler-32 (zlib ストリーム末尾のチェックサム)
inline uint32_t adler32(const uint8_t* data, size_t len)
{
    uint32_t a = 1, b = 0;
    const uint32_t MOD = 65521;
    for (size_t i = 0; i < len; ++i)
    {
        a = (a + data[i]) % MOD;
        b = (b + a) % MOD;
    }
    return (b << 16) | a;
}

// ビッグエンディアン 32bit を末尾へ追記
inline void put_be32(std::vector<uint8_t>& out, uint32_t v)
{
    out.push_back(static_cast<uint8_t>((v >> 24) & 0xFF));
    out.push_back(static_cast<uint8_t>((v >> 16) & 0xFF));
    out.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
    out.push_back(static_cast<uint8_t>(v & 0xFF));
}

// PNG チャンク (length + type + data + CRC) を追記
inline void put_chunk(std::vector<uint8_t>& out, const char type[4],
                      const std::vector<uint8_t>& data)
{
    put_be32(out, static_cast<uint32_t>(data.size()));
    const size_t crc_begin = out.size();
    out.push_back(static_cast<uint8_t>(type[0]));
    out.push_back(static_cast<uint8_t>(type[1]));
    out.push_back(static_cast<uint8_t>(type[2]));
    out.push_back(static_cast<uint8_t>(type[3]));
    out.insert(out.end(), data.begin(), data.end());
    const uint32_t crc = crc32_update(0, out.data() + crc_begin, out.size() - crc_begin);
    put_be32(out, crc);
}

// raw バイト列を zlib ストリーム (stored blocks) に包む
inline std::vector<uint8_t> zlib_store(const std::vector<uint8_t>& raw)
{
    std::vector<uint8_t> z;
    // zlib ヘッダ: CMF=0x78 (deflate, 32K window), FLG=0x01 (FCHECK 調整済み, no dict, fastest)
    z.push_back(0x78);
    z.push_back(0x01);

    // deflate stored ブロック群。各ブロック最大 65535 バイト。
    size_t off = 0;
    const size_t n = raw.size();
    if (n == 0)
    {
        // 空でも最終空ブロックを 1 つ出す
        z.push_back(0x01); // BFINAL=1, BTYPE=00
        z.push_back(0x00);
        z.push_back(0x00);
        z.push_back(0xFF);
        z.push_back(0xFF);
    }
    while (off < n)
    {
        const size_t block = (n - off > 65535) ? 65535 : (n - off);
        const bool final = (off + block >= n);
        z.push_back(final ? 0x01 : 0x00); // BFINAL, BTYPE=00 (stored)
        const uint16_t len = static_cast<uint16_t>(block);
        const uint16_t nlen = static_cast<uint16_t>(~len);
        z.push_back(static_cast<uint8_t>(len & 0xFF));
        z.push_back(static_cast<uint8_t>((len >> 8) & 0xFF));
        z.push_back(static_cast<uint8_t>(nlen & 0xFF));
        z.push_back(static_cast<uint8_t>((nlen >> 8) & 0xFF));
        z.insert(z.end(), raw.begin() + off, raw.begin() + off + block);
        off += block;
    }

    // Adler-32 (元データに対して, ビッグエンディアン)
    const uint32_t ad = adler32(raw.data(), raw.size());
    put_be32(z, ad);
    return z;
}

} // namespace detail

// RGB8 ピクセル列を PNG バイト列にエンコードする。
//   rgb: サイズ width*height*3 の RGB8 (行優先)。
//   戻り値: PNG ファイルバイト列。
inline std::vector<uint8_t> encode_png_rgb8(const std::vector<uint8_t>& rgb,
                                            int width, int height)
{
    if (width <= 0 || height <= 0)
    {
        throw std::runtime_error("encode_png_rgb8: 幅/高さが不正");
    }
    const size_t expect = static_cast<size_t>(width) * height * 3;
    if (rgb.size() != expect)
    {
        throw std::runtime_error("encode_png_rgb8: ピクセル数が width*height*3 と不一致");
    }

    std::vector<uint8_t> out;

    // PNG シグネチャ 89 50 4E 47 0D 0A 1A 0A
    const uint8_t sig[8] = {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A};
    out.insert(out.end(), sig, sig + 8);

    // IHDR
    std::vector<uint8_t> ihdr;
    detail::put_be32(ihdr, static_cast<uint32_t>(width));
    detail::put_be32(ihdr, static_cast<uint32_t>(height));
    ihdr.push_back(8);    // bit depth
    ihdr.push_back(2);    // color type 2 = truecolor RGB
    ihdr.push_back(0);    // compression method
    ihdr.push_back(0);    // filter method
    ihdr.push_back(0);    // interlace method
    detail::put_chunk(out, "IHDR", ihdr);

    // raw スキャンライン (各行頭に filter byte 0x00)
    std::vector<uint8_t> raw;
    raw.reserve(static_cast<size_t>(height) * (1 + static_cast<size_t>(width) * 3));
    const size_t row_bytes = static_cast<size_t>(width) * 3;
    for (int y = 0; y < height; ++y)
    {
        raw.push_back(0x00); // filter type: none
        const uint8_t* row = rgb.data() + static_cast<size_t>(y) * row_bytes;
        raw.insert(raw.end(), row, row + row_bytes);
    }

    // IDAT (zlib stored)
    detail::put_chunk(out, "IDAT", detail::zlib_store(raw));

    // IEND
    detail::put_chunk(out, "IEND", {});

    return out;
}

// RGBA8 ピクセル列を PNG バイト列にエンコードする (透過 PNG / マッティング出力用)。
//   rgba: サイズ width*height*4 の RGBA8 (行優先、各画素 R,G,B,A の 4 バイト)。
//   戻り値: PNG ファイルバイト列 (color_type=6 truecolor with alpha)。
// encode_png_rgb8 と同じ仕組み (CRC/Adler/zlib stored block/フィルタなし) を
// 4 チャンネルへ拡張しただけ。RGB 経路は非破壊で別関数のまま残す。
inline std::vector<uint8_t> encode_png_rgba8(const std::vector<uint8_t>& rgba,
                                             int width, int height)
{
    if (width <= 0 || height <= 0)
    {
        throw std::runtime_error("encode_png_rgba8: 幅/高さが不正");
    }
    const size_t expect = static_cast<size_t>(width) * height * 4;
    if (rgba.size() != expect)
    {
        throw std::runtime_error("encode_png_rgba8: ピクセル数が width*height*4 と不一致");
    }

    std::vector<uint8_t> out;

    // PNG シグネチャ 89 50 4E 47 0D 0A 1A 0A
    const uint8_t sig[8] = {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A};
    out.insert(out.end(), sig, sig + 8);

    // IHDR
    std::vector<uint8_t> ihdr;
    detail::put_be32(ihdr, static_cast<uint32_t>(width));
    detail::put_be32(ihdr, static_cast<uint32_t>(height));
    ihdr.push_back(8);    // bit depth
    ihdr.push_back(6);    // color type 6 = truecolor + alpha (RGBA)
    ihdr.push_back(0);    // compression method
    ihdr.push_back(0);    // filter method
    ihdr.push_back(0);    // interlace method
    detail::put_chunk(out, "IHDR", ihdr);

    // raw スキャンライン (各行頭に filter byte 0x00)
    std::vector<uint8_t> raw;
    raw.reserve(static_cast<size_t>(height) * (1 + static_cast<size_t>(width) * 4));
    const size_t row_bytes = static_cast<size_t>(width) * 4;
    for (int y = 0; y < height; ++y)
    {
        raw.push_back(0x00); // filter type: none
        const uint8_t* row = rgba.data() + static_cast<size_t>(y) * row_bytes;
        raw.insert(raw.end(), row, row + row_bytes);
    }

    // IDAT (zlib stored)
    detail::put_chunk(out, "IDAT", detail::zlib_store(raw));

    // IEND
    detail::put_chunk(out, "IEND", {});

    return out;
}

// PNG ヘッダから幅・高さを取り出す (テスト検証用、最小実装)。
// シグネチャと IHDR が先頭にある前提。失敗時 false。
inline bool read_png_size(const std::vector<uint8_t>& png, int& width, int& height)
{
    if (png.size() < 24)
    {
        return false;
    }
    const uint8_t sig[8] = {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A};
    for (int i = 0; i < 8; ++i)
    {
        if (png[i] != sig[i])
        {
            return false;
        }
    }
    // png[8..11] = IHDR length, png[12..15] = "IHDR", png[16..19]=width, png[20..23]=height
    auto be32 = [&](size_t o) -> uint32_t
    {
        return (static_cast<uint32_t>(png[o]) << 24) |
               (static_cast<uint32_t>(png[o + 1]) << 16) |
               (static_cast<uint32_t>(png[o + 2]) << 8) |
               static_cast<uint32_t>(png[o + 3]);
    };
    width  = static_cast<int>(be32(16));
    height = static_cast<int>(be32(20));
    return true;
}

} // namespace dollama

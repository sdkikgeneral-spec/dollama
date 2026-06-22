// PNG エンコーダ (RGB8 / RGBA8) 単体テスト
// testing.md 形式: 各検証は if (!cond) { cerr; return false; }、
// main で ok = test_xxx() && ok 集約、成功は PASSED / 最後に ALL PASSED。
//
// 検証範囲 (M-3: RGBA PNG エンコーダ追加):
//   1. RGBA8 の構造 (シグネチャ・IHDR width/height/bit_depth/color_type=6・末尾 IEND)
//   2. α 往復 (既知 RGBA パターン → encode → stored IDAT 展開 → 画素一致)
//   3. 不正引数 (寸法・バイト数不一致) で例外
//   4. 既存 RGB8 経路の無回帰 (color_type=2・寸法・画素往復)
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "server/png.hpp"

namespace dollama
{

// ----------------------------------------------------------------
// テスト用ヘルパ: PNG からチャンクを線形走査し、type が一致する最初の
// チャンクのデータ範囲 [begin, end) を返す。見つからなければ false。
// ----------------------------------------------------------------
static bool find_chunk(const std::vector<uint8_t>& png, const char type[4],
                       size_t& data_begin, size_t& data_len)
{
    if (png.size() < 8)
    {
        return false;
    }
    size_t pos = 8; // シグネチャの後ろから
    while (pos + 8 <= png.size())
    {
        const uint32_t len = (uint32_t(png[pos]) << 24) | (uint32_t(png[pos + 1]) << 16) |
                             (uint32_t(png[pos + 2]) << 8) | uint32_t(png[pos + 3]);
        const uint8_t* t = &png[pos + 4];
        if (pos + 12 + len > png.size())
        {
            return false; // 境界外
        }
        if (t[0] == uint8_t(type[0]) && t[1] == uint8_t(type[1]) &&
            t[2] == uint8_t(type[2]) && t[3] == uint8_t(type[3]))
        {
            data_begin = pos + 8;
            data_len   = len;
            return true;
        }
        pos += 12 + len; // length(4)+type(4)+data(len)+crc(4)
    }
    return false;
}

// ----------------------------------------------------------------
// テスト用ヘルパ: zlib_store が吐いた stored block 形式の IDAT を展開して
// raw スキャンライン (filter byte 込み) を復元する。
// (zlib ヘッダ 2B + [BFINAL/BTYPE, LEN, NLEN, data]* + Adler 4B)。
// ----------------------------------------------------------------
static bool inflate_stored(const std::vector<uint8_t>& z, std::vector<uint8_t>& raw)
{
    raw.clear();
    if (z.size() < 2 + 4)
    {
        return false;
    }
    size_t pos = 2;               // zlib ヘッダ CMF/FLG をスキップ
    const size_t end = z.size() - 4; // 末尾 Adler-32 の手前まで
    bool saw_final = false;
    while (pos < end)
    {
        const uint8_t hdr = z[pos++];
        const bool bfinal = (hdr & 0x01) != 0;
        const uint8_t btype = (hdr >> 1) & 0x03;
        if (btype != 0x00)
        {
            return false; // stored 以外は非対応 (本エンコーダは stored のみ)
        }
        if (pos + 4 > end)
        {
            return false;
        }
        const uint16_t len  = uint16_t(z[pos]) | (uint16_t(z[pos + 1]) << 8);
        const uint16_t nlen = uint16_t(z[pos + 2]) | (uint16_t(z[pos + 3]) << 8);
        pos += 4;
        if (uint16_t(~len) != nlen)
        {
            return false; // LEN/NLEN 整合性
        }
        if (pos + len > end)
        {
            return false;
        }
        raw.insert(raw.end(), z.begin() + pos, z.begin() + pos + len);
        pos += len;
        if (bfinal)
        {
            saw_final = true;
            break;
        }
    }
    return saw_final;
}

// ----------------------------------------------------------------
// 1. RGBA8 構造: シグネチャ・IHDR (width/height/bit_depth=8/color_type=6)・末尾 IEND
// ----------------------------------------------------------------
static bool test_rgba_structure()
{
    const int W = 5, H = 3;
    std::vector<uint8_t> rgba(static_cast<size_t>(W) * H * 4, 0x40);
    std::vector<uint8_t> png = encode_png_rgba8(rgba, W, H);

    // シグネチャ
    const uint8_t sig[8] = {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A};
    if (png.size() < 8)
    {
        std::cerr << "[test_rgba_structure] 短すぎ\n";
        return false;
    }
    for (int i = 0; i < 8; ++i)
    {
        if (png[i] != sig[i])
        {
            std::cerr << "[test_rgba_structure] シグネチャ不正\n";
            return false;
        }
    }

    // read_png_size で幅高さ (IHDR 非破壊)
    int w = 0, h = 0;
    if (!read_png_size(png, w, h) || w != W || h != H)
    {
        std::cerr << "[test_rgba_structure] 寸法不正: " << w << "x" << h << "\n";
        return false;
    }

    // IHDR を直接見て bit_depth=8・color_type=6 を確認
    size_t ib = 0, il = 0;
    if (!find_chunk(png, "IHDR", ib, il) || il != 13)
    {
        std::cerr << "[test_rgba_structure] IHDR 不在/長さ不正\n";
        return false;
    }
    // IHDR data: width(4) height(4) bit_depth(1) color_type(1) ...
    if (png[ib + 8] != 8)
    {
        std::cerr << "[test_rgba_structure] bit_depth != 8\n";
        return false;
    }
    if (png[ib + 9] != 6)
    {
        std::cerr << "[test_rgba_structure] color_type != 6 (RGBA)\n";
        return false;
    }

    // 末尾 IEND チャンク (type(4)+crc(4))
    const uint8_t* tail = png.data() + png.size() - 8;
    if (!(tail[0] == 'I' && tail[1] == 'E' && tail[2] == 'N' && tail[3] == 'D'))
    {
        std::cerr << "[test_rgba_structure] 末尾が IEND でない\n";
        return false;
    }

    std::cout << "[test_rgba_structure] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// 2. α 往復: 既知 RGBA パターン → encode → IDAT 展開 → 画素・α 完全一致
// ----------------------------------------------------------------
static bool test_rgba_roundtrip()
{
    const int W = 4, H = 2;
    std::vector<uint8_t> rgba(static_cast<size_t>(W) * H * 4);
    // 各画素にユニークな R/G/B と、位置で変化する α (0,85,170,255,... の階調) を入れる
    for (int y = 0; y < H; ++y)
    {
        for (int x = 0; x < W; ++x)
        {
            const size_t idx = (static_cast<size_t>(y) * W + x) * 4;
            const uint8_t base = static_cast<uint8_t>((y * W + x) * 13 + 1);
            rgba[idx + 0] = base;                                   // R
            rgba[idx + 1] = static_cast<uint8_t>(base + 50);        // G
            rgba[idx + 2] = static_cast<uint8_t>(base + 100);       // B
            rgba[idx + 3] = static_cast<uint8_t>((y * W + x) * 36); // A: 0,36,72,...,252
        }
    }

    std::vector<uint8_t> png = encode_png_rgba8(rgba, W, H);

    // IDAT を取り出して stored block を展開
    size_t db = 0, dl = 0;
    if (!find_chunk(png, "IDAT", db, dl))
    {
        std::cerr << "[test_rgba_roundtrip] IDAT 不在\n";
        return false;
    }
    std::vector<uint8_t> idat(png.begin() + db, png.begin() + db + dl);
    std::vector<uint8_t> raw;
    if (!inflate_stored(idat, raw))
    {
        std::cerr << "[test_rgba_roundtrip] IDAT 展開失敗\n";
        return false;
    }

    // raw は各行 [filter(1) + W*4] が H 行
    const size_t row_bytes = static_cast<size_t>(W) * 4;
    const size_t expect = static_cast<size_t>(H) * (1 + row_bytes);
    if (raw.size() != expect)
    {
        std::cerr << "[test_rgba_roundtrip] 展開サイズ不正: " << raw.size()
                  << " != " << expect << "\n";
        return false;
    }

    for (int y = 0; y < H; ++y)
    {
        const size_t rpos = static_cast<size_t>(y) * (1 + row_bytes);
        if (raw[rpos] != 0x00) // filter type none
        {
            std::cerr << "[test_rgba_roundtrip] filter byte != 0\n";
            return false;
        }
        for (size_t k = 0; k < row_bytes; ++k)
        {
            const uint8_t got = raw[rpos + 1 + k];
            const uint8_t exp = rgba[static_cast<size_t>(y) * row_bytes + k];
            if (got != exp)
            {
                std::cerr << "[test_rgba_roundtrip] 画素不一致 y=" << y
                          << " k=" << k << " got=" << int(got)
                          << " exp=" << int(exp) << "\n";
                return false;
            }
        }
    }

    std::cout << "[test_rgba_roundtrip] PASSED (α 含む全画素一致)\n";
    return true;
}

// ----------------------------------------------------------------
// 3. 不正引数: 寸法不正・バイト数不一致で例外
// ----------------------------------------------------------------
static bool test_rgba_invalid_args()
{
    // 幅 0
    {
        bool threw = false;
        try
        {
            encode_png_rgba8(std::vector<uint8_t>(), 0, 2);
        }
        catch (const std::runtime_error&)
        {
            threw = true;
        }
        if (!threw)
        {
            std::cerr << "[test_rgba_invalid_args] 幅0で例外なし\n";
            return false;
        }
    }
    // バイト数不一致 (4ch 期待に 3ch 分しか渡さない)
    {
        bool threw = false;
        try
        {
            std::vector<uint8_t> wrong(static_cast<size_t>(2) * 2 * 3, 0);
            encode_png_rgba8(wrong, 2, 2);
        }
        catch (const std::runtime_error&)
        {
            threw = true;
        }
        if (!threw)
        {
            std::cerr << "[test_rgba_invalid_args] バイト数不一致で例外なし\n";
            return false;
        }
    }
    std::cout << "[test_rgba_invalid_args] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// 4. 既存 RGB8 経路の無回帰: color_type=2・寸法・画素往復
// ----------------------------------------------------------------
static bool test_rgb_no_regression()
{
    const int W = 3, H = 2;
    std::vector<uint8_t> rgb(static_cast<size_t>(W) * H * 3);
    for (size_t i = 0; i < rgb.size(); ++i)
    {
        rgb[i] = static_cast<uint8_t>(i * 11 + 5);
    }
    std::vector<uint8_t> png = encode_png_rgb8(rgb, W, H);

    int w = 0, h = 0;
    if (!read_png_size(png, w, h) || w != W || h != H)
    {
        std::cerr << "[test_rgb_no_regression] 寸法不正\n";
        return false;
    }

    size_t ib = 0, il = 0;
    if (!find_chunk(png, "IHDR", ib, il) || il != 13 || png[ib + 9] != 2)
    {
        std::cerr << "[test_rgb_no_regression] color_type != 2\n";
        return false;
    }

    // IDAT 往復で画素一致
    size_t db = 0, dl = 0;
    if (!find_chunk(png, "IDAT", db, dl))
    {
        std::cerr << "[test_rgb_no_regression] IDAT 不在\n";
        return false;
    }
    std::vector<uint8_t> idat(png.begin() + db, png.begin() + db + dl);
    std::vector<uint8_t> raw;
    if (!inflate_stored(idat, raw))
    {
        std::cerr << "[test_rgb_no_regression] IDAT 展開失敗\n";
        return false;
    }
    const size_t row_bytes = static_cast<size_t>(W) * 3;
    for (int y = 0; y < H; ++y)
    {
        const size_t rpos = static_cast<size_t>(y) * (1 + row_bytes);
        for (size_t k = 0; k < row_bytes; ++k)
        {
            if (raw[rpos + 1 + k] != rgb[static_cast<size_t>(y) * row_bytes + k])
            {
                std::cerr << "[test_rgb_no_regression] 画素不一致\n";
                return false;
            }
        }
    }

    std::cout << "[test_rgb_no_regression] PASSED\n";
    return true;
}

} // namespace dollama

int main()
{
    bool ok = true;
    ok = dollama::test_rgba_structure()     && ok;
    ok = dollama::test_rgba_roundtrip()      && ok;
    ok = dollama::test_rgba_invalid_args()   && ok;
    ok = dollama::test_rgb_no_regression()   && ok;

    if (!ok)
    {
        std::cerr << "[test_png] FAILED\n";
        return 1;
    }
    std::cout << "[test_png] ALL PASSED\n";
    return 0;
}

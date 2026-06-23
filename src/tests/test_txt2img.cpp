// Txt2ImgGenerator (本物の text→image) エンドツーエンドテスト
//   (Phase 2-6b Stage F / ウェーブ2)
//
// 検証内容 (実アセットが全て揃う環境のみ実走):
//   実プロンプトで 20step の txt2img を 1 回走らせ、生成 PNG の健全性を確認する。
//     - GenResult.png_bytes が非空 + PNG マジック (89 50 4E 47) を持つ。
//     - PNG ヘッダの幅/高さが 1024x1024。
//     - 生成画素 (RGB uint8) に NaN/Inf 相当の異常がない (uint8 なので有限は自明だが、
//       PNG をデコードして実画素を取り出すところまで往復で健全性を確認する)。
//     - 画素分散 var>0 (全画素が同一でない = 一様塗りつぶしでない)。
//
// HAVE_OPENVINO か HAVE_CUDA いずれか未定義、またはアセット (OV tokenizer/encoder L/G +
// unet/vae 重み + embeds + tokenizers.dll) 不在なら [SKIP] return 0
// (test_clip / test_matting / test_text_conditioner の SKIP 規約に倣う)。
//
// パスはコンパイル時に -D 埋め込み (テスト実行 cwd = build のため cwd 非依存)。
// デバイスは test_text_conditioner と同じく NPU 第一・失敗時 CPU フォールバック。

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#if defined(HAVE_OPENVINO) && defined(HAVE_CUDA)
#include "server/txt2img_generator.hpp"
#endif

namespace fs = std::filesystem;

// コンパイル時に埋め込まれる定数 (meson の cpp_args で -D)。
#ifndef TOKENIZER_L_XML
#define TOKENIZER_L_XML ""
#endif
#ifndef TOKENIZER_G_XML
#define TOKENIZER_G_XML ""
#endif
#ifndef ENCODER_L_XML
#define ENCODER_L_XML ""
#endif
#ifndef ENCODER_G_XML
#define ENCODER_G_XML ""
#endif
#ifndef TOKENIZERS_DLL
#define TOKENIZERS_DLL ""
#endif
#ifndef UNET_WEIGHTS_PATH
#define UNET_WEIGHTS_PATH ""
#endif
#ifndef VAE_WEIGHTS_PATH
#define VAE_WEIGHTS_PATH ""
#endif
#ifndef EMBEDS_PATH
#define EMBEDS_PATH ""
#endif

#if defined(HAVE_OPENVINO) && defined(HAVE_CUDA)

namespace
{

// PNG の IDAT (zlib stored block) をデコードして RGB8 画素列を復元する。
//   本テストの PNG は png.hpp の encode_png_rgb8 が生成したもの (フィルタなし・
//   zlib stored block のみ) と分かっているため、最小デコーダで十分。
//   失敗時は false。
bool decode_png_rgb8_stored(const std::vector<uint8_t>& png,
                            std::vector<uint8_t>&       rgb_out,
                            int& width, int& height)
{
    if (png.size() < 8)
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

    auto be32 = [&](size_t o) -> uint32_t
    {
        return (static_cast<uint32_t>(png[o]) << 24) |
               (static_cast<uint32_t>(png[o + 1]) << 16) |
               (static_cast<uint32_t>(png[o + 2]) << 8) |
               static_cast<uint32_t>(png[o + 3]);
    };

    // チャンクを走査して IHDR と IDAT を集める。
    int      bit_depth   = 0;
    int      color_type  = -1;
    std::vector<uint8_t> idat;
    size_t   off = 8;
    while (off + 8 <= png.size())
    {
        const uint32_t len = be32(off);
        if (off + 12 + len > png.size())
        {
            return false;
        }
        const char* type = reinterpret_cast<const char*>(&png[off + 4]);
        const size_t data_off = off + 8;
        if (type[0] == 'I' && type[1] == 'H' && type[2] == 'D' && type[3] == 'R')
        {
            if (len < 13)
            {
                return false;
            }
            width      = static_cast<int>(be32(data_off));
            height     = static_cast<int>(be32(data_off + 4));
            bit_depth  = png[data_off + 8];
            color_type = png[data_off + 9];
        }
        else if (type[0] == 'I' && type[1] == 'D' && type[2] == 'A' && type[3] == 'T')
        {
            idat.insert(idat.end(), png.begin() + data_off, png.begin() + data_off + len);
        }
        else if (type[0] == 'I' && type[1] == 'E' && type[2] == 'N' && type[3] == 'D')
        {
            break;
        }
        off = data_off + len + 4; // データ末尾 + CRC4
    }

    if (width <= 0 || height <= 0 || bit_depth != 8 || color_type != 2)
    {
        return false; // RGB8 truecolor のみ想定
    }
    // zlib: 2byte ヘッダ + deflate stored blocks + 4byte adler32。
    if (idat.size() < 2 + 4)
    {
        return false;
    }

    std::vector<uint8_t> raw; // フィルタ byte 込みのスキャンライン群
    size_t p = 2;             // zlib ヘッダをスキップ
    while (p + 5 <= idat.size())
    {
        const uint8_t bfinal_btype = idat[p];
        const bool    bfinal       = (bfinal_btype & 0x01) != 0;
        const int     btype        = (bfinal_btype >> 1) & 0x03;
        if (btype != 0)
        {
            return false; // stored ブロック以外は非対応
        }
        const uint16_t blen =
            static_cast<uint16_t>(idat[p + 1]) |
            (static_cast<uint16_t>(idat[p + 2]) << 8);
        p += 5; // BFINAL/BTYPE(1) + LEN(2) + NLEN(2)
        if (p + blen > idat.size())
        {
            return false;
        }
        raw.insert(raw.end(), idat.begin() + p, idat.begin() + p + blen);
        p += blen;
        if (bfinal)
        {
            break;
        }
    }

    // フィルタ byte (各行頭 0x00 = filter none) を剥がす。
    const size_t row_bytes = static_cast<size_t>(width) * 3;
    const size_t expect    = static_cast<size_t>(height) * (1 + row_bytes);
    if (raw.size() != expect)
    {
        return false;
    }
    rgb_out.resize(static_cast<size_t>(width) * height * 3);
    for (int y = 0; y < height; ++y)
    {
        const size_t src = static_cast<size_t>(y) * (1 + row_bytes);
        if (raw[src] != 0x00)
        {
            return false; // filter none 以外は非対応
        }
        std::copy(raw.begin() + src + 1,
                  raw.begin() + src + 1 + row_bytes,
                  rgb_out.begin() + static_cast<size_t>(y) * row_bytes);
    }
    return true;
}

} // namespace

#endif // HAVE_OPENVINO && HAVE_CUDA

int main()
{
#if !defined(HAVE_OPENVINO) || !defined(HAVE_CUDA)
    std::cout << "[SKIP] HAVE_OPENVINO / HAVE_CUDA いずれか未定義のためスキップします。\n";
    return 0;
#else
    const std::string tok_l_xml = TOKENIZER_L_XML;
    const std::string tok_g_xml = TOKENIZER_G_XML;
    const std::string enc_l_xml = ENCODER_L_XML;
    const std::string enc_g_xml = ENCODER_G_XML;
    const std::string tok_dll   = TOKENIZERS_DLL;
    const std::string unet_w    = UNET_WEIGHTS_PATH;
    const std::string vae_w     = VAE_WEIGHTS_PATH;
    const std::string embeds    = EMBEDS_PATH;

    // 必須アセットの存在確認 (いずれか欠けたら [SKIP])。
    const std::vector<std::pair<std::string, std::string>> required = {
        {"tokenizer-l", tok_l_xml}, {"tokenizer-g", tok_g_xml},
        {"encoder-l", enc_l_xml}, {"encoder-g", enc_g_xml},
        {"tokenizers.dll", tok_dll},
        {"unet-weights", unet_w}, {"vae-weights", vae_w}, {"embeds", embeds},
    };
    for (const auto& kv : required)
    {
        if (kv.second.empty() || !fs::exists(kv.second))
        {
            std::cout << "[SKIP] アセットが見つかりません (" << kv.first
                      << "): " << kv.second << "\n";
            return 0;
        }
    }

    try
    {
        // Txt2ImgGenerator を構築 (NPU 第一・失敗時 CPU フォールバック)。
        std::string device_l = "NPU";
        std::string device_g = "NPU";
        std::unique_ptr<dollama::Txt2ImgGenerator> gen;
        try
        {
            gen = std::make_unique<dollama::Txt2ImgGenerator>(
                tok_l_xml, tok_g_xml, enc_l_xml, enc_g_xml, tok_dll,
                unet_w, vae_w, embeds, device_l, device_g);
        }
        catch (const std::exception& e)
        {
            std::cout << "[warn] NPU での構築に失敗 (" << e.what()
                      << ") → CPU フォールバックを試します。\n";
            device_l = "CPU";
            device_g = "CPU";
            gen = std::make_unique<dollama::Txt2ImgGenerator>(
                tok_l_xml, tok_g_xml, enc_l_xml, enc_g_xml, tok_dll,
                unet_w, vae_w, embeds, device_l, device_g);
        }

        std::cout << "[info] 使用デバイス: CLIP-L=" << device_l
                  << " / bigG=" << device_g << " — 20step txt2img を実行します。\n";

        dollama::GenRequest req;
        req.prompt          = "1girl, solo, long hair, looking at viewer";
        req.negative_prompt = "lowres, bad anatomy, worst quality";
        req.steps           = 20;
        req.width           = 1024;
        req.height          = 1024;

        const dollama::GenResult res = gen->generate(req);

        bool ok = true;

        // (1) PNG バイト列が非空 + PNG マジック。
        if (res.png_bytes.size() < 8 ||
            res.png_bytes[0] != 0x89 || res.png_bytes[1] != 0x50 ||
            res.png_bytes[2] != 0x4E || res.png_bytes[3] != 0x47)
        {
            std::cerr << "[FAIL] PNG マジックが不正 (size="
                      << res.png_bytes.size() << ")\n";
            ok = false;
        }
        else
        {
            std::cout << "[PASS] PNG マジック OK (size="
                      << res.png_bytes.size() << " bytes)\n";
        }

        // (2) PNG をデコードして画素往復・サイズ・分散を確認。
        std::vector<uint8_t> rgb;
        int w = 0, h = 0;
        if (!decode_png_rgb8_stored(res.png_bytes, rgb, w, h))
        {
            std::cerr << "[FAIL] PNG デコードに失敗\n";
            ok = false;
        }
        else
        {
            if (w != 1024 || h != 1024)
            {
                std::cerr << "[FAIL] サイズが 1024x1024 でない: " << w << "x" << h << "\n";
                ok = false;
            }
            else
            {
                std::cout << "[PASS] サイズ 1024x1024\n";
            }

            // 画素分散 var>0 (全画素同一でない)。uint8 なので有限性は自明。
            double sum = 0.0, sum2 = 0.0;
            const size_t n = rgb.size();
            for (size_t i = 0; i < n; ++i)
            {
                const double v = static_cast<double>(rgb[i]);
                sum  += v;
                sum2 += v * v;
            }
            const double mean = sum / static_cast<double>(n);
            const double var  = sum2 / static_cast<double>(n) - mean * mean;
            if (var > 0.0)
            {
                std::cout << "[PASS] 画素分散 var=" << var
                          << " > 0 (一様塗りつぶしでない)\n";
            }
            else
            {
                std::cerr << "[FAIL] 画素分散 var=" << var
                          << " (全画素が同一)\n";
                ok = false;
            }
        }

        if (!ok)
        {
            std::cerr << "[test_txt2img] FAILED\n";
            return 1;
        }
        std::cout << "[test_txt2img] ALL PASSED\n";
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[test_txt2img] 例外: " << e.what() << "\n";
        return 1;
    }
#endif // HAVE_OPENVINO && HAVE_CUDA
}

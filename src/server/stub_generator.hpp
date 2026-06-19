// ダミー画像生成器 (2-6 未完のため interface 越しのスタブ)
//
// prompt のハッシュから決定的に色を決め、width×height の単色 (+簡単なグラデーション)
// 画像を生成して PNG バイト列にする。CUDA / OpenVINO 依存ゼロの純 C++。
// 2-6 完了時に PipelineGenerator へ差し替える前提。
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "server/generator.hpp"
#include "server/png.hpp"

namespace dollama
{

class StubGenerator : public IImageGenerator
{
public:
    GenResult generate(const GenRequest& req) override
    {
        // prompt + negative_prompt から決定的な色を導く (FNV-1a 32bit)
        const uint32_t h = fnv1a(req.prompt) ^ (fnv1a(req.negative_prompt) * 16777619u);
        const uint8_t base_r = static_cast<uint8_t>((h >> 16) & 0xFF);
        const uint8_t base_g = static_cast<uint8_t>((h >> 8) & 0xFF);
        const uint8_t base_b = static_cast<uint8_t>(h & 0xFF);

        const int w = req.width;
        const int hgt = req.height;
        std::vector<uint8_t> rgb(static_cast<size_t>(w) * hgt * 3);

        // 左上→右下にかけて base 色を弱く変化させる確認しやすいダミー画像
        for (int y = 0; y < hgt; ++y)
        {
            const int gy = (hgt > 1) ? (y * 64 / hgt) : 0;
            for (int x = 0; x < w; ++x)
            {
                const int gx = (w > 1) ? (x * 64 / w) : 0;
                const size_t idx = (static_cast<size_t>(y) * w + x) * 3;
                rgb[idx + 0] = static_cast<uint8_t>((base_r + gx) & 0xFF);
                rgb[idx + 1] = static_cast<uint8_t>((base_g + gy) & 0xFF);
                rgb[idx + 2] = base_b;
            }
        }

        GenResult out;
        out.png_bytes = encode_png_rgb8(rgb, w, hgt);
        return out;
    }

    std::string model_id() const override
    {
        return "sdxl-1.0";
    }

private:
    static uint32_t fnv1a(const std::string& s)
    {
        uint32_t h = 2166136261u;
        for (unsigned char c : s)
        {
            h ^= c;
            h *= 16777619u;
        }
        return h;
    }
};

} // namespace dollama

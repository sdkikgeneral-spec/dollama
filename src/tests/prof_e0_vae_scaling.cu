// E-0: VAE scaling_factor A/B 実走プローブ (計測専用 exe・meson test 非登録)。
// 対象: RTX5080 (Blackwell / sm_120) / CUDA Runtime API のみ
//
// 目的:
//   DiffusionPipeline::generate (golden 埋め込み・guidance_scale=1.0 固定) を
//   直接叩き、CLIP テキストエンコード (NPU/OV) を経由せずに VAE scaling_factor の
//   差だけを純粋に見る。preset の prompt/TextConditioner とは無関係な軸のため、
//   NPU 側の環境事情 (OV ドライバ/ブロブ不整合など) に一切影響されない。
//
// 使い方:
//   prof_e0_vae_scaling.exe <unet_weights> <vae_weights> <embeds_io> <seed>
//                            <vae_scaling_factor|0=既定> <out.png> [<steps=20>]
//
//   vae_scaling_factor に 0 を渡すと DiffusionPipeline の既定引数
//   (kDefaultVaeScalingFactor = 0.13025f) をそのまま使う (コンストラクタの
//   オーバーロード解決ではなく明示的に既定定数を渡す形。既定経路 sha256 一致の
//   確認に使う)。
//
// 標準出力に sha256 (自前 SHA-256 実装。追加依存を増やさないため) と
// RGB のピクセル統計 (mean/var/min/max) を出す。

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "infer/diffusion.cuh"
#include "kernels/utils.cuh"
#include "server/png.hpp"

namespace
{

// ----------------------------------------------------------------
// 自前 SHA-256 (FIPS 180-4)。追加ライブラリ依存を増やさないための最小実装。
// 検証専用プローブなのでパフォーマンスは問わない。
// ----------------------------------------------------------------
class Sha256
{
public:
    Sha256() { reset(); }

    void update(const uint8_t* data, size_t len)
    {
        total_len_ += len;
        size_t i = 0;
        if (buf_len_ > 0)
        {
            while (i < len && buf_len_ < 64)
            {
                buf_[buf_len_++] = data[i++];
            }
            if (buf_len_ == 64)
            {
                process(buf_);
                buf_len_ = 0;
            }
        }
        while (i + 64 <= len)
        {
            process(data + i);
            i += 64;
        }
        while (i < len)
        {
            buf_[buf_len_++] = data[i++];
        }
    }

    std::string hexdigest()
    {
        const uint64_t bit_len = total_len_ * 8ULL;
        uint8_t pad = 0x80;
        update(&pad, 1);
        uint8_t zero = 0x00;
        while (buf_len_ != 56)
        {
            update(&zero, 1);
        }
        uint8_t lenbytes[8];
        for (int i = 0; i < 8; ++i)
        {
            lenbytes[i] = static_cast<uint8_t>(bit_len >> (56 - 8 * i));
        }
        update(lenbytes, 8);

        char out[65];
        for (int i = 0; i < 8; ++i)
        {
            std::snprintf(out + i * 8, 9, "%08x", h_[i]);
        }
        return std::string(out, 64);
    }

private:
    uint32_t h_[8];
    uint8_t  buf_[64];
    size_t   buf_len_ = 0;
    uint64_t total_len_ = 0;

    void reset()
    {
        static const uint32_t init[8] = {
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
        std::memcpy(h_, init, sizeof(h_));
        buf_len_   = 0;
        total_len_ = 0;
    }

    static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

    void process(const uint8_t* chunk)
    {
        static const uint32_t k[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
            0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
            0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
            0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
            0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
            0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
            0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
            0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
        {
            w[i] = (static_cast<uint32_t>(chunk[i * 4]) << 24) |
                   (static_cast<uint32_t>(chunk[i * 4 + 1]) << 16) |
                   (static_cast<uint32_t>(chunk[i * 4 + 2]) << 8) |
                   (static_cast<uint32_t>(chunk[i * 4 + 3]));
        }
        for (int i = 16; i < 64; ++i)
        {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3];
        uint32_t e = h_[4], f = h_[5], g = h_[6], hh = h_[7];
        for (int i = 0; i < 64; ++i)
        {
            const uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t ch = (e & f) ^ ((~e) & g);
            const uint32_t t1 = hh + s1 + ch + k[i] + w[i];
            const uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = s0 + maj;
            hh = g; g = f; f = e; e = d + t1;
            d = c; c = b; b = a; a = t1 + t2;
        }
        h_[0] += a; h_[1] += b; h_[2] += c; h_[3] += d;
        h_[4] += e; h_[5] += f; h_[6] += g; h_[7] += hh;
    }
};

std::string sha256_hex(const std::vector<uint8_t>& data)
{
    Sha256 s;
    s.update(data.data(), data.size());
    return s.hexdigest();
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 7)
    {
        std::cerr << "usage: prof_e0_vae_scaling <unet_weights> <vae_weights> <embeds_io> "
                     "<seed> <vae_scaling_factor|0=default> <out.png> [<steps=20>]\n";
        return 1;
    }
    const std::string unet_w  = argv[1];
    const std::string vae_w   = argv[2];
    const std::string embeds  = argv[3];
    const uint64_t seed       = std::strtoull(argv[4], nullptr, 10);
    const float    sf_arg     = std::strtof(argv[5], nullptr);
    const std::string out_png = argv[6];
    const int steps           = (argc >= 8) ? std::atoi(argv[7]) : 20;

    try
    {
        std::unique_ptr<dollama::DiffusionPipeline> pipe;
        if (sf_arg > 0.0f)
        {
            pipe = std::make_unique<dollama::DiffusionPipeline>(
                unet_w, vae_w, embeds, dollama::FastConfig{}, sf_arg);
        }
        else
        {
            // 引数 0 = 既定 (ctor のデフォルト引数をそのまま使う経路)。
            pipe = std::make_unique<dollama::DiffusionPipeline>(unet_w, vae_w, embeds);
        }

        std::vector<uint8_t> rgb;
        int w = 0, h = 0;
        pipe->generate(steps, seed, rgb, w, h);

        const std::vector<uint8_t> png = dollama::encode_png_rgb8(rgb, w, h);
        std::ofstream ofs(out_png, std::ios::binary);
        ofs.write(reinterpret_cast<const char*>(png.data()),
                  static_cast<std::streamsize>(png.size()));
        ofs.close();

        double sum = 0.0, sum2 = 0.0;
        int mn = 256, mx = -1;
        for (uint8_t b : rgb)
        {
            const int v = static_cast<int>(b);
            sum += v; sum2 += static_cast<double>(v) * v;
            mn = std::min(mn, v); mx = std::max(mx, v);
        }
        const double n = static_cast<double>(rgb.size());
        const double mean = sum / n;
        const double var  = sum2 / n - mean * mean;

        std::cout << "[prof_e0] out=" << out_png << " w=" << w << " h=" << h
                  << " sha256=" << sha256_hex(rgb)
                  << " sha256_png=" << sha256_hex(png)
                  << " mean=" << mean << " var=" << var
                  << " min=" << mn << " max=" << mx << "\n";
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[prof_e0] FAIL: " << e.what() << "\n";
        return 1;
    }
}

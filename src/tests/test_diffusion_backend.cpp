// 拡散 backend プラグイン枠 単体テスト — diffusion_backend.hpp / backend_image_generator.hpp
//
// 純 cpp (OV/CUDA 非依存) ゆえ開発機でも SKIP なし常時実走する。
//   1. fake backend (固定 RGB を返す IDiffusionBackend 実装) を BackendImageGenerator に
//      注入し、PNG 化 (color_type 検証)・set_matter/set_scorer に no-op 相当を注入して
//      落ちないことを検証する。
//   2. BackendInfo の値 (sd35: latent=16 / needs_t5=true 等) を検証する。
//   3. registry: make_backend("sd35") が非 null で generate() が throw すること、
//      未知名が nullptr であることを検証する。
//
// 全テスト通過時は "[test_diffusion_backend] ALL PASSED" を出力して return 0。
// 失敗時は std::cerr に出力して return 1。

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "server/backend_image_generator.hpp" // BackendImageGenerator
#include "server/diffusion_backend.hpp"        // IDiffusionBackend / make_backend / BackendConfig
#include "server/matter_runner.hpp"            // IMatter (no-op 注入用)
#include "server/scorer_runner.hpp"            // IScorer (no-op 注入用)
#include "server/sd35_backend.hpp"             // SD35Backend (info 値検証用)

namespace
{

int g_fail = 0;

void check(bool cond, const std::string& msg)
{
    if (!cond)
    {
        std::cerr << "[test_diffusion_backend] FAIL: " << msg << std::endl;
        ++g_fail;
    }
}

using dollama::BackendInfo;
using dollama::IDiffusionBackend;
using dollama::IMatter;
using dollama::IScorer;
using dollama::ScorerResult;

// 固定色 RGB を返す fake backend (成功経路の検証用)。
struct FakeBackend : IDiffusionBackend
{
    void generate(const std::string& /*prompt*/, const std::string& /*negative*/,
                  int /*steps*/, uint64_t /*seed*/, float /*cfg*/, int w, int h,
                  std::vector<uint8_t>& rgb_out, int& w_out, int& h_out) override
    {
        // 要求解像度で固定色 (R=10,G=20,B=30) を返す。
        const int rw = (w > 0) ? w : 1024;
        const int rh = (h > 0) ? h : 1024;
        rgb_out.assign(static_cast<size_t>(rw) * rh * 3, 0);
        for (size_t i = 0; i + 2 < rgb_out.size(); i += 3)
        {
            rgb_out[i + 0] = 10;
            rgb_out[i + 1] = 20;
            rgb_out[i + 2] = 30;
        }
        w_out = rw;
        h_out = rh;
    }

    BackendInfo info() const override
    {
        return {"fake", 4, 1024, false};
    }

    std::string model_id() const override
    {
        return "fake-1.0";
    }
};

// no-op マッター (matte が w*h と不一致の空マスクを返す → 不透明 PNG へフォールバック)。
//   BackendImageGenerator が matter 注入下でも落ちないことの検証用。
struct NoopMatter : IMatter
{
    std::vector<float> matte(const std::vector<uint8_t>& /*rgb*/, int /*w*/, int /*h*/) override
    {
        return {}; // 空 → サイズ不一致 → encode_png_maybe_transparent が不透明 PNG に倒す
    }
};

// no-op スコアラ (固定 ScorerResult を返す)。採点ログ経路が落ちないことの検証用。
struct NoopScorer : IScorer
{
    ScorerResult score(const std::vector<uint8_t>& /*rgb*/, int /*w*/, int /*h*/) override
    {
        ScorerResult r;
        r.quality = 0.5f;
        for (int a = 0; a < 8; ++a)
        {
            r.axis[a] = 0.1f;
        }
        return r;
    }
};

// PNG シグネチャ (8 バイト) の検証。
bool has_png_signature(const std::vector<uint8_t>& b)
{
    static const uint8_t sig[8] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    if (b.size() < 8)
    {
        return false;
    }
    for (int i = 0; i < 8; ++i)
    {
        if (b[i] != sig[i])
        {
            return false;
        }
    }
    return true;
}

// IHDR の color_type を読む (PNG: シグネチャ8 + IHDR長4 + "IHDR"4 + 幅4 + 高4 + bitdepth1 + colortype1)。
//   color_type オフセット = 8 + 4 + 4 + 4 + 4 + 1 = 25。
int png_color_type(const std::vector<uint8_t>& b)
{
    const size_t off = 25;
    if (b.size() <= off)
    {
        return -1;
    }
    return static_cast<int>(b[off]);
}

} // namespace

int main()
{
    using namespace dollama;

    // ------------------------------------------------------------
    // 1. fake backend を BackendImageGenerator に注入 → generate → PNG。
    //    matter/scorer 未注入なら不透明 PNG (color_type=2)。
    // ------------------------------------------------------------
    {
        auto backend = std::make_unique<FakeBackend>();
        BackendImageGenerator gen(std::move(backend));

        check(gen.model_id() == "fake-1.0", "model_id が backend へ委譲されるべき");

        GenRequest req;
        req.prompt   = "test prompt";
        req.width    = 1024;
        req.height   = 1024;
        req.matting  = true; // matter 未注入なので不透明にフォールバックするはず

        GenResult res = gen.generate(req);
        check(has_png_signature(res.png_bytes), "生成結果は PNG シグネチャを持つべき");
        check(png_color_type(res.png_bytes) == 2,
              "matter 未注入なら不透明 RGB PNG (color_type=2) であるべき");
    }

    // ------------------------------------------------------------
    // 2. matter/scorer を no-op 注入しても generate が落ちないこと。
    //    NoopMatter は空マスク → 不透明 PNG へフォールバック (color_type=2)。
    // ------------------------------------------------------------
    {
        auto backend = std::make_unique<FakeBackend>();
        BackendImageGenerator gen(std::move(backend));
        gen.set_matter(std::make_unique<NoopMatter>());
        gen.set_scorer(std::make_unique<NoopScorer>());

        GenRequest req;
        req.prompt  = "with matter and scorer";
        req.width   = 1024;
        req.height  = 1024;
        req.matting = true;

        GenResult res = gen.generate(req);
        check(has_png_signature(res.png_bytes),
              "matter/scorer 注入下でも PNG が返るべき");
        check(png_color_type(res.png_bytes) == 2,
              "空マスクの matter は不透明 PNG へフォールバックすべき");
    }

    // ------------------------------------------------------------
    // 3. 非 1024 解像度は reject される。
    // ------------------------------------------------------------
    {
        auto backend = std::make_unique<FakeBackend>();
        BackendImageGenerator gen(std::move(backend));

        GenRequest req;
        req.prompt = "bad size";
        req.width  = 512;
        req.height = 512;

        bool threw = false;
        try
        {
            gen.generate(req);
        }
        catch (const std::exception&)
        {
            threw = true;
        }
        check(threw, "非 1024 解像度は reject (throw) されるべき");
    }

    // ------------------------------------------------------------
    // 4. BackendInfo 値検証: SD35Backend (直接) の静的メタ。
    // ------------------------------------------------------------
    {
        SD35Backend sd35;
        BackendInfo info = sd35.info();
        check(info.name == "sd35", "sd35 info.name");
        check(info.latent_channels == 16, "sd35 latent_channels == 16");
        check(info.native_resolution == 1024, "sd35 native_resolution == 1024");
        check(info.needs_t5 == true, "sd35 needs_t5 == true");
        check(sd35.model_id() == "sd35", "sd35 model_id");
    }

    // ------------------------------------------------------------
    // 5. registry: make_backend("sd35") が非 null で generate() が throw すること。
    // ------------------------------------------------------------
    {
        BackendConfig cfg;
        cfg.backend_name = "sd35";
        std::unique_ptr<IDiffusionBackend> b = make_backend(cfg);
        check(b != nullptr, "make_backend(\"sd35\") は非 null であるべき");

        if (b)
        {
            check(b->info().latent_channels == 16, "registry 経由 sd35 latent==16");

            bool threw = false;
            try
            {
                std::vector<uint8_t> rgb;
                int w = 0, h = 0;
                b->generate("p", "n", 20, 1, 7.5f, 1024, 1024, rgb, w, h);
            }
            catch (const std::exception&)
            {
                threw = true;
            }
            check(threw, "sd35 backend の generate() は throw (未実装) すべき");
        }
    }

    // ------------------------------------------------------------
    // 6. registry: 未知の backend 名は nullptr。
    // ------------------------------------------------------------
    {
        BackendConfig cfg;
        cfg.backend_name = "does-not-exist";
        std::unique_ptr<IDiffusionBackend> b = make_backend(cfg);
        check(b == nullptr, "未知 backend 名は nullptr であるべき");
    }

    if (g_fail == 0)
    {
        std::cout << "[test_diffusion_backend] ALL PASSED" << std::endl;
        return 0;
    }
    std::cerr << "[test_diffusion_backend] " << g_fail << " checks FAILED" << std::endl;
    return 1;
}

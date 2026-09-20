// 2-6d: preset 解決 (src/server/preset.hpp) 単体テスト
// 2-6e: PresetPrefix (read_preset_prefix) + BackendImageGenerator への自動付与を追加。
//
// 純 cpp (std::filesystem + nlohmann/json) ゆえ開発機でも SKIP なし常時実走する。
// テスト用の偽 checkpoint 一式を一時ディレクトリに作り、is_valid_preset_name /
// resolve_preset_paths の挙動を検証する。
//
// 検証項目:
//   1. 空名 → nullopt
//   2. 4 ファイル揃い → 4 パスと dir が期待どおり
//   3. 4 のうち 1 つ欠損 → nullopt
//   4. path traversal / 不正文字を含む名前 → is_valid_preset_name false かつ nullopt
//   5. roots 2 つのうち 1 番目が部分・2 番目が完全 → 2 番目を採用 /
//      両方完全 → 1 番目を採用
//   6. unet_weights.safetensors がディレクトリ → nullopt
//   7. 0 バイトファイル → nullopt
//   8. read_preset_prefix: 正常 / 片方欠落 / 不在 / 壊れ json / 両方空 /
//      prompt_prefix が数値 (無視) / 非 object (配列)
//   9. BackendImageGenerator への自動付与: ON 連結 / OFF 素通し / prefix なし素通し /
//      片方空 join
//
// 全テスト通過時は "[test_preset] ALL PASSED" を出力して return 0。
// 失敗時は std::cerr に出力して return 1。

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "server/backend_image_generator.hpp" // BackendImageGenerator (⑨)
#include "server/diffusion_backend.hpp"        // IDiffusionBackend (⑨)
#include "server/generator.hpp"                // GenRequest (⑨)
#include "server/preset.hpp"
#include "server/preset_json.hpp" // read_preset_prefix (⑧)

namespace
{

int g_fail = 0;

void check(bool cond, const std::string& msg)
{
    if (!cond)
    {
        std::cerr << "[test_preset] FAIL: " << msg << std::endl;
        ++g_fail;
    }
}

namespace fs = std::filesystem;

// 1 バイト以上のファイルを作る (親ディレクトリも作成)。
void touch(const std::string& path)
{
    fs::path p(path);
    fs::create_directories(p.parent_path());
    std::ofstream ofs(p, std::ios::binary);
    ofs << "x";
}

// 0 バイトファイルを作る (部分 DL 事故の再現用)。
void touch_empty(const std::string& path)
{
    fs::path p(path);
    fs::create_directories(p.parent_path());
    std::ofstream ofs(p, std::ios::binary); // 何も書かない
}

// 任意内容のファイルを作る (preset.json 用)。
void write_text(const std::string& path, const std::string& content)
{
    fs::path p(path);
    fs::create_directories(p.parent_path());
    std::ofstream ofs(p, std::ios::binary);
    ofs << content;
}

// name の 4 ファイルを root 配下に揃えて作る (dir = root + "presets/" + name + "/")。
void make_full_preset(const std::string& root, const std::string& name)
{
    const std::string dir = root + "presets/" + name + "/";
    touch(dir + "unet_weights.safetensors");
    touch(dir + "vae_weights.safetensors");
    touch(dir + "text-encoder-l/model_ov.xml");
    touch(dir + "text-encoder-g/model_ov.xml");
}

// ⑨: prompt/negative を記録するだけの fake backend (test_diffusion_backend.cpp の
//   FakeBackend を写経し、BackendImageGenerator が backend へ渡す最終文字列を捕捉する)。
struct RecordingFakeBackend : dollama::IDiffusionBackend
{
    std::string seen_prompt;
    std::string seen_negative;

    void generate(const std::string& prompt, const std::string& negative,
                  int /*steps*/, uint64_t /*seed*/, float /*cfg*/, int w, int h,
                  std::vector<uint8_t>& rgb_out, int& w_out, int& h_out) override
    {
        seen_prompt   = prompt;
        seen_negative = negative;
        const int rw = (w > 0) ? w : 1024;
        const int rh = (h > 0) ? h : 1024;
        rgb_out.assign(static_cast<size_t>(rw) * rh * 3, 0);
        w_out = rw;
        h_out = rh;
    }

    dollama::BackendInfo info() const override
    {
        return {"fake", 4, 1024, false};
    }

    std::string model_id() const override
    {
        return "fake-1.0";
    }
};

} // namespace

int main()
{
    using dollama::is_valid_preset_name;
    using dollama::PresetPaths;
    using dollama::resolve_preset_paths;

    // ⓪ E-0: BackendConfig の vae_scaling_factor 既定値は preset.hpp の
    //   kServerDefaultVaeScalingFactor (0.13025f) と一致すべき (既定無改変ゲート)。
    //   diffusion.cuh 側 (infer/CUDA 層) の kDefaultVaeScalingFactor との一致は
    //   server/diffusion_runner.cu の static_assert (両ヘッダを唯一同時 include する TU・
    //   レビュー是正③) がコンパイル時に保証する。
    {
        dollama::BackendConfig cfg;
        check(std::abs(cfg.vae_scaling_factor - dollama::kServerDefaultVaeScalingFactor) < 1e-9f,
              "BackendConfig::vae_scaling_factor の既定値は kServerDefaultVaeScalingFactor と一致すべき");
        check(std::abs(dollama::kServerDefaultVaeScalingFactor - 0.13025f) < 1e-9f,
              "kServerDefaultVaeScalingFactor は 0.13025f であるべき (既定無改変)");
    }

    // 一時ディレクトリ (プロセス固有名で衝突回避)。
    const std::string tmp_root =
        (fs::temp_directory_path() /
         ("dollama_test_preset_" + std::to_string(
              static_cast<long long>(reinterpret_cast<intptr_t>(&g_fail))) + "/"))
            .string();
    fs::remove_all(tmp_root);
    fs::create_directories(tmp_root);

    // 本体は例外時も一時ディレクトリを片付けられるよう try/catch で囲む。
    try
    {
    const std::string root1 = tmp_root + "root1/";
    const std::string root2 = tmp_root + "root2/";
    const std::vector<std::string> roots = {root1, root2};

    // ① 空名 → nullopt
    {
        auto r = resolve_preset_paths("", roots);
        check(!r.has_value(), "空名は nullopt であるべき");
    }

    // ② 4 ファイル揃い → 4 パスと dir が期待どおり
    make_full_preset(root1, "ok-preset_1");
    {
        auto r = resolve_preset_paths("ok-preset_1", roots);
        check(r.has_value(), "4 ファイル揃いは解決できるべき");
        if (r)
        {
            const std::string dir = root1 + "presets/ok-preset_1/";
            check(r->dir == dir, "dir が期待値と一致すべき: " + r->dir);
            check(r->unet == dir + "unet_weights.safetensors", "unet パスが期待値と一致すべき");
            check(r->vae == dir + "vae_weights.safetensors", "vae パスが期待値と一致すべき");
            check(r->enc_l == dir + "text-encoder-l/model_ov.xml",
                  "enc_l パスが期待値と一致すべき");
            check(r->enc_g == dir + "text-encoder-g/model_ov.xml",
                  "enc_g パスが期待値と一致すべき");
        }
    }

    // ③ 4 のうち 1 つ欠損 → nullopt
    {
        const std::string dir = root1 + "presets/partial-preset/";
        touch(dir + "unet_weights.safetensors");
        touch(dir + "vae_weights.safetensors");
        touch(dir + "text-encoder-l/model_ov.xml");
        // text-encoder-g/model_ov.xml は作らない (欠損)。
        auto r = resolve_preset_paths("partial-preset", roots);
        check(!r.has_value(), "1 ファイル欠損は nullopt であるべき");
    }

    // ④ path traversal / 不正文字を含む名前 → is_valid_preset_name false かつ nullopt
    {
        const std::vector<std::string> bad_names = {"../x", "Bad", "a b", "a/b"};
        for (const auto& n : bad_names)
        {
            check(!is_valid_preset_name(n), "不正名は is_valid_preset_name false であるべき: " + n);
            auto r = resolve_preset_paths(n, roots);
            check(!r.has_value(), "不正名は resolve_preset_paths nullopt であるべき: " + n);
        }
    }

    // ⑤-a: roots 2 つのうち 1 番目が部分・2 番目が完全 → 2 番目を採用
    {
        const std::string dir1 = root1 + "presets/mixed-preset/";
        touch(dir1 + "unet_weights.safetensors"); // 部分のみ
        make_full_preset(root2, "mixed-preset");  // 2 番目は完全

        auto r = resolve_preset_paths("mixed-preset", roots);
        check(r.has_value(), "2 番目 root で完全なら解決できるべき");
        if (r)
        {
            const std::string expected_dir = root2 + "presets/mixed-preset/";
            check(r->dir == expected_dir, "1番目部分・2番目完全 → 2番目を採用すべき: " + r->dir);
        }
    }

    // ⑤-b: 両方完全 → 1 番目を採用
    {
        make_full_preset(root1, "both-preset");
        make_full_preset(root2, "both-preset");

        auto r = resolve_preset_paths("both-preset", roots);
        check(r.has_value(), "両方完全なら解決できるべき");
        if (r)
        {
            const std::string expected_dir = root1 + "presets/both-preset/";
            check(r->dir == expected_dir, "両方完全 → 1番目を採用すべき: " + r->dir);
        }
    }

    // ⑥ unet_weights.safetensors がディレクトリ → nullopt
    {
        const std::string dir = root1 + "presets/dir-as-unet/";
        fs::create_directories(fs::path(dir + "unet_weights.safetensors")); // ディレクトリとして作る
        touch(dir + "vae_weights.safetensors");
        touch(dir + "text-encoder-l/model_ov.xml");
        touch(dir + "text-encoder-g/model_ov.xml");
        auto r = resolve_preset_paths("dir-as-unet", roots);
        check(!r.has_value(), "unet がディレクトリなら nullopt であるべき");
    }

    // ⑦ 0 バイトファイル (部分 DL 事故) → nullopt
    {
        const std::string dir = root1 + "presets/zero-byte/";
        touch_empty(dir + "unet_weights.safetensors"); // 0 バイト
        touch(dir + "vae_weights.safetensors");
        touch(dir + "text-encoder-l/model_ov.xml");
        touch(dir + "text-encoder-g/model_ov.xml");
        auto r = resolve_preset_paths("zero-byte", roots);
        check(!r.has_value(), "0 バイトファイルは nullopt であるべき");
    }

    // ⑧ read_preset_prefix: 正常 / 片方欠落 / 不在 / 壊れ json
    {
        using dollama::read_preset_prefix;

        // ⑧-a 正常 (両方非空)
        {
            const std::string dir = root1 + "presets/prefix-ok/";
            write_text(dir + "preset.json",
                       R"({"prompt_prefix": "masterpiece, best quality", )"
                       R"("negative_prefix": "worst quality, lowres"})");
            auto p = read_preset_prefix(dir);
            check(p.has_value(), "正常な preset.json は値を返すべき");
            if (p)
            {
                check(p->prompt == "masterpiece, best quality", "prompt_prefix が期待値と一致すべき");
                check(p->negative == "worst quality, lowres", "negative_prefix が期待値と一致すべき");
            }
        }

        // ⑧-b 片方欠落 (prompt_prefix のみ)
        {
            const std::string dir = root1 + "presets/prefix-partial/";
            write_text(dir + "preset.json", R"({"prompt_prefix": "masterpiece"})");
            auto p = read_preset_prefix(dir);
            check(p.has_value(), "片方欠落でも非空フィールドがあれば値を返すべき");
            if (p)
            {
                check(p->prompt == "masterpiece", "prompt_prefix が期待値と一致すべき");
                check(p->negative.empty(), "negative_prefix 欠落は空文字であるべき");
            }
        }

        // ⑧-c 不在 (preset.json を作らない) → nullopt
        {
            const std::string dir = root1 + "presets/prefix-missing/";
            fs::create_directories(dir);
            auto p = read_preset_prefix(dir);
            check(!p.has_value(), "preset.json 不在は nullopt であるべき");
        }

        // ⑧-d 壊れ json → nullopt
        {
            const std::string dir = root1 + "presets/prefix-broken/";
            write_text(dir + "preset.json", "{not valid json");
            auto p = read_preset_prefix(dir);
            check(!p.has_value(), "壊れた json は nullopt であるべき");
        }

        // ⑧-e 両方空 (フィールドは存在するが値が空文字) → nullopt
        {
            const std::string dir = root1 + "presets/prefix-both-empty/";
            write_text(dir + "preset.json",
                       R"({"prompt_prefix": "", "negative_prefix": ""})");
            auto p = read_preset_prefix(dir);
            check(!p.has_value(), "両方空文字は nullopt であるべき");
        }

        // ⑧-f prompt_prefix が数値 (型不一致) → その軸は無視・negative_prefix は生きる
        {
            const std::string dir = root1 + "presets/prefix-wrong-type/";
            write_text(dir + "preset.json",
                       R"({"prompt_prefix": 123, "negative_prefix": "worst quality"})");
            auto p = read_preset_prefix(dir);
            check(p.has_value(), "片方が型不一致でももう片方が有効なら値を返すべき");
            if (p)
            {
                check(p->prompt.empty(), "型不一致の prompt_prefix は無視 (空文字) であるべき");
                check(p->negative == "worst quality", "negative_prefix は期待値と一致すべき");
            }
        }

        // ⑧-g 非 object (配列) → nullopt
        {
            const std::string dir = root1 + "presets/prefix-not-object/";
            write_text(dir + "preset.json", R"(["masterpiece", "worst quality"])");
            auto p = read_preset_prefix(dir);
            check(!p.has_value(), "非 object (配列) の json は nullopt であるべき");
        }

        // ⑧-h E-0: vae_scaling_factor 正常値
        {
            const std::string dir = root1 + "presets/vaesf-ok/";
            write_text(dir + "preset.json", R"({"vae_scaling_factor": 0.18215})");
            auto p = read_preset_prefix(dir);
            check(p.has_value(), "vae_scaling_factor のみでも値を返すべき (prompt/negative は空でも可)");
            if (p)
            {
                check(p->vae_scaling_factor.has_value(), "正常値は vae_scaling_factor に格納されるべき");
                if (p->vae_scaling_factor)
                {
                    check(std::abs(*p->vae_scaling_factor - 0.18215f) < 1e-6f,
                          "vae_scaling_factor が期待値と一致すべき");
                }
            }
        }

        // ⑧-i E-0: vae_scaling_factor キー不在 (他フィールドは正常) → 素通し・nullopt にはならない
        {
            const std::string dir = root1 + "presets/vaesf-missing/";
            write_text(dir + "preset.json", R"({"prompt_prefix": "masterpiece"})");
            auto p = read_preset_prefix(dir);
            check(p.has_value(), "prompt_prefix があれば値を返すべき");
            if (p)
            {
                check(!p->vae_scaling_factor.has_value(),
                      "vae_scaling_factor キー不在は nullopt であるべき (フォールバックは呼び出し側)");
            }
        }

        // ⑧-j E-0: vae_scaling_factor が非数値 (文字列) → 無視され nullopt・既定へフォールバック
        {
            const std::string dir = root1 + "presets/vaesf-nonnumeric/";
            write_text(dir + "preset.json",
                       R"({"prompt_prefix": "masterpiece", "vae_scaling_factor": "oops"})");
            auto p = read_preset_prefix(dir);
            check(p.has_value(), "他フィールドが有効なら値を返すべき");
            if (p)
            {
                check(!p->vae_scaling_factor.has_value(),
                      "非数値の vae_scaling_factor は無視 (nullopt) であるべき");
            }
        }

        // ⑧-k E-0: vae_scaling_factor が 0 以下 → 無視され nullopt・既定へフォールバック
        {
            const std::string dir = root1 + "presets/vaesf-nonpositive/";
            write_text(dir + "preset.json",
                       R"({"prompt_prefix": "masterpiece", "vae_scaling_factor": 0})");
            auto p = read_preset_prefix(dir);
            check(p.has_value(), "他フィールドが有効なら値を返すべき");
            if (p)
            {
                check(!p->vae_scaling_factor.has_value(),
                      "0 以下の vae_scaling_factor は無視 (nullopt) であるべき");
            }
        }

        // ⑧-l E-0: vae_scaling_factor が負値 → 無視され nullopt
        {
            const std::string dir = root1 + "presets/vaesf-negative/";
            write_text(dir + "preset.json", R"({"vae_scaling_factor": -0.5})");
            auto p = read_preset_prefix(dir);
            // prompt/negative とも空・vae_scaling_factor も無視されるため付帯情報なし。
            check(!p.has_value(), "負値のみ・他フィールド無しは全体で nullopt であるべき");
        }
    }

    // ⑨ BackendImageGenerator への自動付与: ON 連結 / OFF 素通し / prefix なし素通し / 片方空 join
    {
        using dollama::BackendImageGenerator;
        using dollama::GenRequest;
        using dollama::PresetPrefix;

        // ⑨-a ON: prefix + user 両方非空 → "prefix, user" に連結される
        {
            auto* backend_raw = new RecordingFakeBackend();
            BackendImageGenerator gen(std::unique_ptr<RecordingFakeBackend>(backend_raw),
                                       PresetPrefix{"masterpiece, best quality", "worst quality"});
            GenRequest req;
            req.prompt          = "1girl, solo";
            req.negative_prompt = "bad anatomy";
            req.preset_prefix   = true;
            gen.generate(req);
            check(backend_raw->seen_prompt == "masterpiece, best quality, 1girl, solo",
                  "ON: prompt は prefix, user に連結されるべき: " + backend_raw->seen_prompt);
            check(backend_raw->seen_negative == "worst quality, bad anatomy",
                  "ON: negative は prefix, user に連結されるべき: " + backend_raw->seen_negative);
        }

        // ⑨-b OFF: req.preset_prefix=false なら prefix があっても user そのまま
        {
            auto* backend_raw = new RecordingFakeBackend();
            BackendImageGenerator gen(std::unique_ptr<RecordingFakeBackend>(backend_raw),
                                       PresetPrefix{"masterpiece, best quality", "worst quality"});
            GenRequest req;
            req.prompt          = "1girl, solo";
            req.negative_prompt = "bad anatomy";
            req.preset_prefix   = false;
            gen.generate(req);
            check(backend_raw->seen_prompt == "1girl, solo",
                  "OFF: prompt は user そのままであるべき: " + backend_raw->seen_prompt);
            check(backend_raw->seen_negative == "bad anatomy",
                  "OFF: negative は user そのままであるべき: " + backend_raw->seen_negative);
        }

        // ⑨-c prefix なし (nullopt): ON でも user そのまま
        {
            auto* backend_raw = new RecordingFakeBackend();
            BackendImageGenerator gen{std::unique_ptr<RecordingFakeBackend>(backend_raw)};
            GenRequest req;
            req.prompt          = "1girl, solo";
            req.negative_prompt = "bad anatomy";
            req.preset_prefix   = true;
            gen.generate(req);
            check(backend_raw->seen_prompt == "1girl, solo",
                  "prefix なし: prompt は user そのままであるべき: " + backend_raw->seen_prompt);
            check(backend_raw->seen_negative == "bad anatomy",
                  "prefix なし: negative は user そのままであるべき: " + backend_raw->seen_negative);
        }

        // ⑨-d 片方空 join: prefix.negative が空 / user.prompt が空
        {
            auto* backend_raw = new RecordingFakeBackend();
            BackendImageGenerator gen(std::unique_ptr<RecordingFakeBackend>(backend_raw),
                                       PresetPrefix{"masterpiece", ""});
            GenRequest req;
            req.prompt          = ""; // user prompt 空 → prefix のみ
            req.negative_prompt = "bad anatomy"; // prefix.negative 空 → user のみ
            req.preset_prefix   = true;
            gen.generate(req);
            check(backend_raw->seen_prompt == "masterpiece",
                  "片方空: user 空なら prefix のみであるべき: " + backend_raw->seen_prompt);
            check(backend_raw->seen_negative == "bad anatomy",
                  "片方空: prefix 空なら user のみであるべき (先頭カンマ無し): " + backend_raw->seen_negative);
        }
    }
    }
    catch (const std::exception& e)
    {
        std::cerr << "[test_preset] FAIL: 例外送出: " << e.what() << std::endl;
        ++g_fail;
    }

    // 後始末 (例外時もここへ到達する)。
    fs::remove_all(tmp_root);

    if (g_fail == 0)
    {
        std::cout << "[test_preset] ALL PASSED" << std::endl;
        return 0;
    }
    std::cerr << "[test_preset] " << g_fail << " FAILURES" << std::endl;
    return 1;
}

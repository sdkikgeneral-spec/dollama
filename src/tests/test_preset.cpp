// 2-6d: preset 解決 (src/server/preset.hpp) 単体テスト
//
// 純 cpp (std::filesystem のみ) ゆえ開発機でも SKIP なし常時実走する。
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
//
// 全テスト通過時は "[test_preset] ALL PASSED" を出力して return 0。
// 失敗時は std::cerr に出力して return 1。

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "server/preset.hpp"

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

// name の 4 ファイルを root 配下に揃えて作る (dir = root + "presets/" + name + "/")。
void make_full_preset(const std::string& root, const std::string& name)
{
    const std::string dir = root + "presets/" + name + "/";
    touch(dir + "unet_weights.safetensors");
    touch(dir + "vae_weights.safetensors");
    touch(dir + "text-encoder-l/model_ov.xml");
    touch(dir + "text-encoder-g/model_ov.xml");
}

} // namespace

int main()
{
    using dollama::is_valid_preset_name;
    using dollama::PresetPaths;
    using dollama::resolve_preset_paths;

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

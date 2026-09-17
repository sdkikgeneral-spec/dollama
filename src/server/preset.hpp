// 2-6d: アニメ特化 SDXL checkpoint プリセット解決 (純 cpp・ヘッダオンリー)。
//
// 目的:
//   models/presets/<name>/ 配下に配置された checkpoint 一式 (unet/vae/text-encoder-l/g)
//   を、既定の探索 root 群 (find_model_xml と同順) から探し当てる。SDXLBackend 自体は
//   無改修 (BackendConfig のパスフィールドを差し替えるだけ)。
//
//   4 ファイルが「全部」揃った root のみ採用する (部分的に揃った dir は次の root へ
//   フォールバック)。名前は path traversal 対策で英数字・'-'・'_' のみ許可する。
//
//   本ヘッダは CUDA / OpenVINO を一切 include しない (std::filesystem のみ)。
#pragma once

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace dollama
{

// preset 名の妥当性検証: 非空 かつ 全文字が [a-z0-9_-]。
//   path traversal ("../x") やパス区切り ("a/b") を含む名前は false になる。
inline bool is_valid_preset_name(const std::string& name)
{
    if (name.empty())
    {
        return false;
    }
    for (unsigned char c : name)
    {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') ||
                        c == '_' || c == '-';
        if (!ok)
        {
            return false;
        }
    }
    return true;
}

// preset 解決結果 (4 パス + dir)。
struct PresetPaths
{
    std::string unet;
    std::string vae;
    std::string enc_l;
    std::string enc_g;
    std::string dir;
};

// 既定の探索 root 群 (find_model_xml (cli_generate.hpp) と同順)。
inline const std::vector<std::string>& default_model_roots()
{
    static const std::vector<std::string> roots = {
        "../models/",
        "models/",
        "../../models/",
    };
    return roots;
}

// path が「実体のある通常ファイル」であることを検証する (std::error_code 版で例外を
//   出さない)。ディレクトリ・アクセス拒否・存在しない は「無い」扱い。0 バイト (部分
//   ダウンロード事故) も「無い」扱いにする。
inline bool is_valid_preset_file(const std::string& path)
{
    namespace fs = std::filesystem;
    std::error_code ec;
    if (!fs::is_regular_file(path, ec) || ec)
    {
        return false;
    }
    const auto size = fs::file_size(path, ec);
    return !ec && size > 0;
}

// root の末尾に '/' が無ければ補う (正規化)。
inline std::string normalize_root(const std::string& root)
{
    if (root.empty() || root.back() == '/' || root.back() == '\\')
    {
        return root;
    }
    return root + "/";
}

// name を roots 配下の presets/<name>/ から解決する。
//   各 root r について normalize_root(r) + "presets/" + name + "/" を dir とし、以下
//   4 ファイルが全部揃えば最初に揃った root で返す (揃わなければ次の root へ):
//     unet_weights.safetensors
//     vae_weights.safetensors
//     text-encoder-l/model_ov.xml
//     text-encoder-g/model_ov.xml
//   名前不正・どの root でも揃わなければ nullopt。
inline std::optional<PresetPaths> resolve_preset_paths(
    const std::string& name, const std::vector<std::string>& roots)
{
    if (!is_valid_preset_name(name))
    {
        return std::nullopt;
    }

    for (const auto& root : roots)
    {
        const std::string dir   = normalize_root(root) + "presets/" + name + "/";
        const std::string unet  = dir + "unet_weights.safetensors";
        const std::string vae   = dir + "vae_weights.safetensors";
        const std::string enc_l = dir + "text-encoder-l/model_ov.xml";
        const std::string enc_g = dir + "text-encoder-g/model_ov.xml";

        if (is_valid_preset_file(unet) && is_valid_preset_file(vae) &&
            is_valid_preset_file(enc_l) && is_valid_preset_file(enc_g))
        {
            PresetPaths result;
            result.unet  = unet;
            result.vae   = vae;
            result.enc_l = enc_l;
            result.enc_g = enc_g;
            result.dir   = dir;
            return result;
        }
    }

    return std::nullopt;
}

} // namespace dollama

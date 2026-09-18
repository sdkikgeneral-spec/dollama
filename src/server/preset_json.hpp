// 2-6e: preset.json (prompt_prefix / negative_prefix) の読み込み (nlohmann/json 依存)。
//
// preset.hpp から分離した理由:
//   preset.hpp (PresetPaths / resolve_preset_paths) は backend_image_generator.hpp
//   越しに広く include されるため std::filesystem のみに保ちたい。JSON 読み込みを
//   要するのは preset.json を実際に読む cli_generate.hpp (build_image_generator) だけ
//   なので、json_dep を要求する本ヘッダはそちらだけが include する。
#pragma once

#include <filesystem>
#include <fstream>
#include <optional>
#include <string>

#include <nlohmann/json.hpp>

#include "server/preset.hpp" // PresetPrefix

namespace dollama
{

// dir (resolve_preset_paths の PresetPaths::dir・末尾 '/' 付き) 配下の "preset.json" を
// 読み、"prompt_prefix" / "negative_prefix" (いずれも省略可・文字列) を取り出す。
//   - ファイル不在 / JSON パース失敗 / 非 object / 両フィールドとも欠落 or 空 → nullopt
//     (throw しない。preset.json は checkpoint 一式にとって任意の付帯情報)。
//   - 片方のみ非空でも値を返す (呼び出し側で空側は無視する)。
//   - フィールドが存在しても文字列でなければ (例: 数値) そのフィールドは無視する
//     (もう片方が有効なら返る)。
inline std::optional<PresetPrefix> read_preset_prefix(const std::string& dir)
{
    namespace fs = std::filesystem;
    const std::string path = dir + "preset.json";

    std::error_code ec;
    if (!fs::is_regular_file(path, ec) || ec)
    {
        return std::nullopt;
    }

    std::ifstream ifs(path, std::ios::binary);
    if (!ifs)
    {
        return std::nullopt;
    }

    nlohmann::json j;
    try
    {
        ifs >> j;
    }
    catch (const std::exception&)
    {
        return std::nullopt; // 壊れた JSON
    }

    if (!j.is_object())
    {
        return std::nullopt;
    }

    PresetPrefix prefix;
    if (j.contains("prompt_prefix") && j["prompt_prefix"].is_string())
    {
        prefix.prompt = j["prompt_prefix"].get<std::string>();
    }
    if (j.contains("negative_prefix") && j["negative_prefix"].is_string())
    {
        prefix.negative = j["negative_prefix"].get<std::string>();
    }

    if (prefix.prompt.empty() && prefix.negative.empty())
    {
        return std::nullopt; // 両方欠落/空なら付帯情報なし扱い
    }
    return prefix;
}

} // namespace dollama

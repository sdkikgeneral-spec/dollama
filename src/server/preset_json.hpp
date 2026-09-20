// 2-6e: preset.json (prompt_prefix / negative_prefix) の読み込み (nlohmann/json 依存)。
//
// preset.hpp から分離した理由:
//   preset.hpp (PresetPaths / resolve_preset_paths) は backend_image_generator.hpp
//   越しに広く include されるため std::filesystem のみに保ちたい。JSON 読み込みを
//   要するのは preset.json を実際に読む cli_generate.hpp (build_image_generator) だけ
//   なので、json_dep を要求する本ヘッダはそちらだけが include する。
#pragma once

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>

#include <nlohmann/json.hpp>

#include "server/preset.hpp" // PresetPrefix

namespace dollama
{

// dir (resolve_preset_paths の PresetPaths::dir・末尾 '/' 付き) 配下の "preset.json" を
// 読み、"prompt_prefix" / "negative_prefix" (いずれも省略可・文字列) と
// "vae_scaling_factor" (E-0・省略可・数値) を取り出す。
//   - ファイル不在 / JSON パース失敗 / 非 object / 全フィールドとも欠落 or 空/不正 → nullopt
//     (throw しない。preset.json は checkpoint 一式にとって任意の付帯情報)。
//   - いずれか 1 つでも有効なら値を返す (呼び出し側で個別に無視できる)。
//   - フィールドが存在しても型が違えば (例: prompt_prefix が数値) そのフィールドは無視する
//     (他が有効なら返る)。
//   - vae_scaling_factor はキー不在時は静かに無視 (nullopt) するが、キーはあるのに
//     非数値・0 以下・非有限値のときは呼び出し側が既定 (kServerDefaultVaeScalingFactor) へ
//     フォールバックできるよう nullopt にしつつ、原因調査用に [warn] を stderr へ出す
//     (prompt_prefix/negative_prefix の型不一致は従来通り無警告で無視する既存挙動を維持)。
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

    // E-0: vae_scaling_factor。数値かつ 0 より大きく有限であることを検証する。
    if (j.contains("vae_scaling_factor"))
    {
        const auto& v = j["vae_scaling_factor"];
        const double d = v.is_number() ? v.get<double>() : 0.0;
        if (v.is_number() && d > 0.0 && std::isfinite(d))
        {
            prefix.vae_scaling_factor = static_cast<float>(d);
        }
        else
        {
            std::cerr << "[warn] preset.json '" << path
                      << "' vae_scaling_factor が不正 (非数値または 0 以下) — "
                         "既定 " << kServerDefaultVaeScalingFactor << " にフォールバックします\n";
        }
    }

    if (prefix.prompt.empty() && prefix.negative.empty() && !prefix.vae_scaling_factor)
    {
        return std::nullopt; // 全フィールド欠落/空/不正なら付帯情報なし扱い
    }
    return prefix;
}

} // namespace dollama

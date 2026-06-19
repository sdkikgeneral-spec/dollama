// PNG メタ往復 (character_bible-spec §7) — tEXt 焼き込み / 読み戻し
//
// 目的: 生成済み PNG にキャラ設定 (CharacterIdentity / SceneSpec / OutputSpec)
//   を JSON で焼き込み、次コマで読み戻して同一性を引き継ぐ往復を実現する。
//   identity_features など大物は焼かず、§7 スキーマ列挙フィールドのみを対象とする。
//
// 構造:
//   - bible_to_json / json_to_bible: 構造体 ↔ §7 JSON スキーマ
//   - embed_bible_in_png / read_bible_from_png: PNG ↔ tEXt チャンク
//   - write_bible_png / read_bible_png: 構造体 ↔ PNG 一発ラッパ
//
// 設計判断:
//   - tEXt の keyword は "dollama/bible"。値は JSON を ASCII エスケープ
//     (ensure_ascii=true) して Latin-1 制約を回避し、日本語 name も安全に往復する。
//   - チャンクの追記/CRC は src/server/png.hpp の detail を再利用する。
//     読み出し (チャンク走査) は png.hpp を汚さずここに新規実装する。
//   - CUDA / OpenVINO 依存なしの純 C++。JSON は nlohmann/json。
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "core/character.hpp"
#include "server/png.hpp"  // detail::put_chunk / put_be32 / crc32_update を再利用

namespace dollama
{

// このローダー/ライターが書き込むスキーマのメジャー版。
// 読み出しはメジャー不一致のみ拒否 (前方互換)。未知フィールドは無視する。
constexpr int kBibleSchemaVersion = 1;

namespace detail
{

// Sex enum → 固定文字列 (spec §7: female/male/other)。数値では焼かない。
inline const char* sex_to_str(Sex s)
{
    switch (s)
    {
    case Sex::Female:
        return "female";
    case Sex::Male:
        return "male";
    case Sex::Other:
        return "other";
    }
    return "female";  // 到達しない (安全側デフォルト)
}

// 文字列 → Sex。逆引き不明は安全側デフォルト Sex::Female。
inline Sex str_to_sex(const std::string& s)
{
    if (s == "male")
    {
        return Sex::Male;
    }
    if (s == "other")
    {
        return Sex::Other;
    }
    return Sex::Female;
}

// MattingMode enum → 固定文字列 (spec §7: none/segment/native)。
inline const char* matting_to_str(MattingMode m)
{
    switch (m)
    {
    case MattingMode::None:
        return "none";
    case MattingMode::Segment:
        return "segment";
    case MattingMode::Native:
        return "native";
    }
    return "segment";  // 到達しない (安全側デフォルト)
}

// 文字列 → MattingMode。逆引き不明は安全側デフォルト MattingMode::Segment。
inline MattingMode str_to_matting(const std::string& s)
{
    if (s == "none")
    {
        return MattingMode::None;
    }
    if (s == "native")
    {
        return MattingMode::Native;
    }
    return MattingMode::Segment;
}

// JSON から「キーがあり、型が期待どおりなら」取り出す小物群。
// 欠落・型不一致時は dst を変更せず (= 構造体デフォルトのまま) 残す (欠落許容)。
inline void get_str(const nlohmann::json& j, const char* key, std::string& dst)
{
    auto it = j.find(key);
    if (it != j.end() && it->is_string())
    {
        dst = it->get<std::string>();
    }
}

inline void get_int(const nlohmann::json& j, const char* key, int& dst)
{
    auto it = j.find(key);
    if (it != j.end() && it->is_number_integer())
    {
        dst = it->get<int>();
    }
}

inline void get_i32(const nlohmann::json& j, const char* key, int32_t& dst)
{
    auto it = j.find(key);
    if (it != j.end() && it->is_number_integer())
    {
        dst = it->get<int32_t>();
    }
}

inline void get_u64(const nlohmann::json& j, const char* key, uint64_t& dst)
{
    auto it = j.find(key);
    if (it != j.end() && it->is_number_integer())
    {
        dst = it->get<uint64_t>();
    }
}

inline void get_bool(const nlohmann::json& j, const char* key, bool& dst)
{
    auto it = j.find(key);
    if (it != j.end() && it->is_boolean())
    {
        dst = it->get<bool>();
    }
}

// 文字列配列を取り出す。欠落・非配列なら dst を空にする (= デフォルト)。
inline void get_str_array(const nlohmann::json& j, const char* key,
                          std::vector<std::string>& dst)
{
    auto it = j.find(key);
    if (it == j.end() || !it->is_array())
    {
        return;
    }
    dst.clear();
    for (const auto& e : *it)
    {
        if (e.is_string())
        {
            dst.push_back(e.get<std::string>());
        }
    }
}

// PNG シグネチャ 89 50 4E 47 0D 0A 1A 0A を検証する。
inline bool check_png_signature(const std::vector<uint8_t>& png)
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
    return true;
}

} // namespace detail

// ----------------------------------------------------------------
// bible_to_json: 構造体 → §7 JSON スキーマ
//   identity{name,age,sex,canonical_tags,forbidden_tags,digits_per_hand,
//            embedding_slot,seed}
//   scene{pose_tags,expression_tags,composition,scene_seed}
//   output{matting,isolation_tag,emit_alpha}
//   identity_features 等スキーマ外フィールドは焼かない。
// ----------------------------------------------------------------
inline nlohmann::json bible_to_json(const CharacterIdentity& id,
                                    const SceneSpec& scene,
                                    const OutputSpec& out)
{
    nlohmann::json j;
    j["dollama_bible_version"] = kBibleSchemaVersion;

    nlohmann::json ji;
    ji["name"]            = id.name;
    ji["age"]             = id.age;
    ji["sex"]             = detail::sex_to_str(id.sex);
    ji["canonical_tags"]  = id.canonical_tags;
    ji["forbidden_tags"]  = id.forbidden_tags;
    ji["digits_per_hand"] = id.digits_per_hand;
    ji["embedding_slot"]  = id.embedding_slot;
    ji["seed"]            = id.seed;
    j["identity"]         = ji;

    nlohmann::json js;
    js["pose_tags"]       = scene.pose_tags;
    js["expression_tags"] = scene.expression_tags;
    js["composition"]     = scene.composition;
    js["scene_seed"]      = scene.scene_seed;
    j["scene"]            = js;

    nlohmann::json jo;
    jo["matting"]       = detail::matting_to_str(out.matting);
    jo["isolation_tag"] = out.isolation_tag;
    jo["emit_alpha"]    = out.emit_alpha;
    j["output"]         = jo;

    return j;
}

// ----------------------------------------------------------------
// json_to_bible: §7 JSON → 構造体 (欠落許容・未知キー無視)
//   メジャー版不一致のみ拒否 (前方互換)。欠落フィールドは構造体デフォルト。
//   スキーマ外フィールド (aliases/matting_device/...) は触らずデフォルトのまま。
//   戻り値: 受理できれば true、メジャー版不一致なら false。
// ----------------------------------------------------------------
inline bool json_to_bible(const nlohmann::json& j,
                          CharacterIdentity& id,
                          SceneSpec& scene,
                          OutputSpec& out)
{
    if (!j.is_object())
    {
        return false;
    }

    // メジャー版チェック (前方互換: 未知の上位版は拒否)。
    {
        auto it = j.find("dollama_bible_version");
        if (it != j.end() && it->is_number_integer())
        {
            if (it->get<int>() != kBibleSchemaVersion)
            {
                return false;
            }
        }
        // 欠落時は寛容に受理する (前方互換)。
    }

    // identity
    {
        auto it = j.find("identity");
        if (it != j.end() && it->is_object())
        {
            const nlohmann::json& ji = *it;
            detail::get_str(ji, "name", id.name);
            detail::get_int(ji, "age", id.age);
            std::string sex_str;
            detail::get_str(ji, "sex", sex_str);
            if (!sex_str.empty())
            {
                id.sex = detail::str_to_sex(sex_str);
            }
            detail::get_str_array(ji, "canonical_tags", id.canonical_tags);
            detail::get_str_array(ji, "forbidden_tags", id.forbidden_tags);
            detail::get_int(ji, "digits_per_hand", id.digits_per_hand);
            detail::get_i32(ji, "embedding_slot", id.embedding_slot);
            detail::get_u64(ji, "seed", id.seed);
        }
    }

    // scene
    {
        auto it = j.find("scene");
        if (it != j.end() && it->is_object())
        {
            const nlohmann::json& js = *it;
            detail::get_str_array(js, "pose_tags", scene.pose_tags);
            detail::get_str_array(js, "expression_tags", scene.expression_tags);
            detail::get_str(js, "composition", scene.composition);
            detail::get_u64(js, "scene_seed", scene.scene_seed);
        }
    }

    // output
    {
        auto it = j.find("output");
        if (it != j.end() && it->is_object())
        {
            const nlohmann::json& jo = *it;
            std::string matting_str;
            detail::get_str(jo, "matting", matting_str);
            if (!matting_str.empty())
            {
                out.matting = detail::str_to_matting(matting_str);
            }
            detail::get_str(jo, "isolation_tag", out.isolation_tag);
            detail::get_bool(jo, "emit_alpha", out.emit_alpha);
        }
    }

    return true;
}

// ----------------------------------------------------------------
// embed_bible_in_png: tEXt チャンク (keyword "dollama/bible") を IEND 直前に挿入。
//   既存チャンク (IHDR/IDAT 等) は一切変更しない。
//   JSON は ensure_ascii=true で dump し Latin-1 制約 (tEXt) を満たす。
//   入力 png が IEND を持たない/不正な場合は入力をそのまま返す (破壊しない)。
// ----------------------------------------------------------------
inline std::vector<uint8_t> embed_bible_in_png(const std::vector<uint8_t>& png,
                                               const nlohmann::json& bible)
{
    // tEXt データ = keyword + 0x00 + text (Latin-1)。
    // text は ASCII エスケープ済み JSON 文字列なので全て ASCII (Latin-1 安全)。
    const std::string keyword = "dollama/bible";
    const std::string text    = bible.dump(-1, ' ', /*ensure_ascii=*/true);

    std::vector<uint8_t> chunk_data;
    chunk_data.reserve(keyword.size() + 1 + text.size());
    chunk_data.insert(chunk_data.end(), keyword.begin(), keyword.end());
    chunk_data.push_back(0x00);  // null separator
    chunk_data.insert(chunk_data.end(), text.begin(), text.end());

    // 完成した tEXt チャンクバイト列 (length + "tEXt" + data + CRC) を組み立てる。
    std::vector<uint8_t> text_chunk;
    detail::put_chunk(text_chunk, "tEXt", chunk_data);

    // IEND チャンクの開始位置を探す。チャンクを正しく辿って length フィールドを尊重する。
    // PNG: シグネチャ(8) のあと [length(4)][type(4)][data][crc(4)] の繰り返し。
    if (!detail::check_png_signature(png))
    {
        return png;  // PNG でなければ何もしない
    }

    size_t pos       = 8;
    size_t iend_pos  = static_cast<size_t>(-1);
    const size_t n   = png.size();
    while (pos + 8 <= n)
    {
        const uint32_t len = (static_cast<uint32_t>(png[pos]) << 24) |
                             (static_cast<uint32_t>(png[pos + 1]) << 16) |
                             (static_cast<uint32_t>(png[pos + 2]) << 8) |
                             static_cast<uint32_t>(png[pos + 3]);
        const char* type = reinterpret_cast<const char*>(&png[pos + 4]);
        // チャンク全体長 = 4(len) + 4(type) + len + 4(crc)。オーバーフロー/境界外を弾く。
        const uint64_t chunk_total = static_cast<uint64_t>(len) + 12u;
        if (static_cast<uint64_t>(pos) + chunk_total > static_cast<uint64_t>(n))
        {
            break;  // 壊れた length。安全側で中断。
        }
        if (type[0] == 'I' && type[1] == 'E' && type[2] == 'N' && type[3] == 'D')
        {
            iend_pos = pos;
            break;
        }
        pos += static_cast<size_t>(chunk_total);
    }

    if (iend_pos == static_cast<size_t>(-1))
    {
        return png;  // IEND が見つからない (不正)。破壊せずそのまま返す。
    }

    // IEND 直前に tEXt チャンクを差し込む。
    std::vector<uint8_t> result;
    result.reserve(png.size() + text_chunk.size());
    result.insert(result.end(), png.begin(), png.begin() + iend_pos);
    result.insert(result.end(), text_chunk.begin(), text_chunk.end());
    result.insert(result.end(), png.begin() + iend_pos, png.end());
    return result;
}

// ----------------------------------------------------------------
// read_bible_from_png: シグネチャ検証 → チャンク走査で keyword "dollama/bible"
//   の tEXt を探し、JSON へパースする。
//   見つからない / パース不能 / 壊れた PNG では false (クラッシュなし)。
// ----------------------------------------------------------------
inline bool read_bible_from_png(const std::vector<uint8_t>& png,
                                nlohmann::json& bible)
{
    if (!detail::check_png_signature(png))
    {
        return false;
    }

    const std::string keyword = "dollama/bible";
    const size_t n = png.size();
    size_t pos = 8;
    while (pos + 8 <= n)
    {
        const uint32_t len = (static_cast<uint32_t>(png[pos]) << 24) |
                             (static_cast<uint32_t>(png[pos + 1]) << 16) |
                             (static_cast<uint32_t>(png[pos + 2]) << 8) |
                             static_cast<uint32_t>(png[pos + 3]);
        const char* type = reinterpret_cast<const char*>(&png[pos + 4]);
        const uint64_t chunk_total = static_cast<uint64_t>(len) + 12u;
        if (static_cast<uint64_t>(pos) + chunk_total > static_cast<uint64_t>(n))
        {
            return false;  // 壊れた length。境界外読みを避けて中断。
        }

        if (type[0] == 't' && type[1] == 'E' && type[2] == 'X' && type[3] == 't')
        {
            const size_t data_begin = pos + 8;
            const size_t data_end   = data_begin + len;
            // tEXt データ内の null セパレータを探す。
            size_t sep = data_begin;
            while (sep < data_end && png[sep] != 0x00)
            {
                ++sep;
            }
            if (sep < data_end)
            {
                const std::string kw(reinterpret_cast<const char*>(&png[data_begin]),
                                     sep - data_begin);
                if (kw == keyword)
                {
                    const size_t text_begin = sep + 1;
                    const std::string text(
                        reinterpret_cast<const char*>(&png[text_begin]),
                        data_end - text_begin);
                    nlohmann::json parsed =
                        nlohmann::json::parse(text, nullptr, /*allow_exceptions=*/false);
                    if (parsed.is_discarded())
                    {
                        return false;  // JSON 不正
                    }
                    bible = std::move(parsed);
                    return true;
                }
            }
        }

        if (type[0] == 'I' && type[1] == 'E' && type[2] == 'N' && type[3] == 'D')
        {
            break;  // IEND 到達。これ以降にデータはない。
        }
        pos += static_cast<size_t>(chunk_total);
    }
    return false;
}

// ----------------------------------------------------------------
// write_bible_png: 構造体 → PNG (薄いラッパ)
//   既存 png に §7 JSON を tEXt として焼き込む。
// ----------------------------------------------------------------
inline std::vector<uint8_t> write_bible_png(const std::vector<uint8_t>& png,
                                            const CharacterIdentity& id,
                                            const SceneSpec& scene,
                                            const OutputSpec& out)
{
    return embed_bible_in_png(png, bible_to_json(id, scene, out));
}

// ----------------------------------------------------------------
// read_bible_png: PNG → 構造体 (薄いラッパ)
//   tEXt を読み出して §7 対象フィールドを構造体に展開する。
//   読み出せなければ false (構造体は呼び出し側の初期状態のまま)。
// ----------------------------------------------------------------
inline bool read_bible_png(const std::vector<uint8_t>& png,
                           CharacterIdentity& id,
                           SceneSpec& scene,
                           OutputSpec& out)
{
    nlohmann::json bible;
    if (!read_bible_from_png(png, bible))
    {
        return false;
    }
    return json_to_bible(bible, id, scene, out);
}

} // namespace dollama

#pragma once

// Model B — 数・位相 QA チェッカー Stage 1 (WD14 異常タグ soft ゲート)
//
// 既存 Wd14Tagger 出力 (sigmoid 済みスコアベクタ [1, N_TAGS]) を消費して、
// 数・位相異常 (extra_arms / multiple_heads / bad_anatomy 等) を soft に検出する。
// character-bible-spec §11 の QA 検査二段構成の一段目 (Stage 1)。
//
// 設計判断:
//   - **OV 非依存**で書く。WD14 スコアベクタを引数で受けるので OpenVINO は不要。
//     matting.hpp の compose_rgba が OV ガード外に置かれテスト実走できるのと同方針。
//     OV 無し開発機でもテスト緑にできるのが本タスクの肝。
//   - 異常タグは danbooru タグ**名**で保持し、index は CSV から解決する
//     (WD14 のバージョン差異に頑健・tokenizer.hpp が vocab.json を名前駆動で読むのと同哲学)。
//   - **soft に徹する**。ハードゲート (棄却) は実装しない (§11「異常度の高いコマのみ拾う」)。
//     flagged は将来の FB ループ (B-5) の入力かつ蒸留スコアラの teacher ラベル生成器。
//   - DIGITS_UNCOUNTABLE (肉球/鉤爪/ミトン) のキャラは Digits/Hands 軸の検査を無効化する
//     (指数カウントが意味を持たないため。番兵値は character.hpp の DIGITS_UNCOUNTABLE を使う)。

#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/character.hpp" // CharacterIdentity / DIGITS_UNCOUNTABLE

namespace dollama
{

// ------------------------------------------------------------
// AnomalyAxis: 数・位相異常の軸
// WD14 v3 語彙に index 付きで実在する粗い数・位相異常タグを軸ごとに束ねる。
// ------------------------------------------------------------
enum class AnomalyAxis
{
    Hands,         // 手の異常 (bad_hands)
    Limbs,         // 腕・四肢の数/欠損異常
    Head,          // 頭・顔の数異常
    Eyes,          // 目の数異常
    Ears,          // 耳の数異常
    Mouth,         // 口の数異常
    Digits,        // 指の本数異常 (WD14 は fewer_digits のみ・過剰は Stage 2)
    GlobalAnatomy, // 全体的な解剖・比率異常
};

// ------------------------------------------------------------
// AnomalyHit: 検出された 1 件の異常
// ------------------------------------------------------------
struct AnomalyHit
{
    std::string name;  // danbooru タグ名 (例 "extra_arms")
    AnomalyAxis axis;  // 軸
    float       score; // WD14 sigmoid スコア [0,1]
};

// ------------------------------------------------------------
// QaReport: 1 コマぶんの QA 判定結果
// soft flag のみ。棄却はしない。
// ------------------------------------------------------------
struct QaReport
{
    float                   overall_anomaly = 0.0f;   // hit スコアの最大値 (hit 無しは 0)
    std::vector<AnomalyHit> hits;                     // 閾値超の異常 (軸付き)
    bool                    flagged = false;          // overall_anomaly >= flag_threshold
};

// ------------------------------------------------------------
// QualityGate: WD14 異常タグ soft ゲート
// CSV (selected_tags.csv) から異常タグ名→出力 index を解決し、
// evaluate() で WD14 スコアベクタを採点する。
// ------------------------------------------------------------
class QualityGate
{
public:
    // 異常タグカタログの 1 エントリ (名前 → 軸)。
    struct CatalogEntry
    {
        std::string name;
        AnomalyAxis axis;
    };

    // 異常タグカタログ (danbooru タグ名 → 軸)。
    // index は持たない (CSV から解決する)。WD14 のバージョン差異に頑健。
    static const std::vector<CatalogEntry>& catalog()
    {
        static const std::vector<CatalogEntry> kCatalog = {
            // Hands
            {"bad_hands", AnomalyAxis::Hands},
            // Limbs
            {"extra_arms", AnomalyAxis::Limbs},
            {"missing_limb", AnomalyAxis::Limbs},
            {"disembodied_limb", AnomalyAxis::Limbs},
            {"severed_limb", AnomalyAxis::Limbs},
            {"amputee", AnomalyAxis::Limbs},
            {"double_amputee", AnomalyAxis::Limbs},
            // Head
            {"multiple_heads", AnomalyAxis::Head},
            {"disembodied_head", AnomalyAxis::Head},
            {"severed_head", AnomalyAxis::Head},
            {"extra_faces", AnomalyAxis::Head},
            // Eyes / Ears / Mouth
            {"extra_eyes", AnomalyAxis::Eyes},
            {"extra_ears", AnomalyAxis::Ears},
            {"extra_mouth", AnomalyAxis::Mouth},
            // Digits
            {"fewer_digits", AnomalyAxis::Digits},
            // GlobalAnatomy
            {"bad_anatomy", AnomalyAxis::GlobalAnatomy},
            {"bad_proportions", AnomalyAxis::GlobalAnatomy},
            {"deformed", AnomalyAxis::GlobalAnatomy},
        };
        return kCatalog;
    }

    // 解決済みカタログエントリ (CSV で index が引けたものだけ保持する)。
    struct ResolvedEntry
    {
        std::string name;
        AnomalyAxis axis;
        size_t      index; // WD14 出力ベクタの index
    };

    // selected_tags_csv: WD14 の selected_tags.csv パス
    //   (tag_id,name,category,count・header 1 行・データ行 i = 出力 index i)。
    // flag_threshold: flagged 判定および hit 収集の閾値 (既定 0.4)。
    explicit QualityGate(const std::string& selected_tags_csv,
                         float flag_threshold = 0.4f)
        : flag_threshold_(flag_threshold)
    {
        // CSV から name → 出力 index のマップを構築する。
        std::unordered_map<std::string, size_t> name_to_index = load_name_to_index(selected_tags_csv);

        // カタログ各タグ名を index 解決する。CSV に無い名は黙って skip (語彙差異に頑健)。
        for (const auto& e : catalog())
        {
            auto it = name_to_index.find(e.name);
            if (it == name_to_index.end())
            {
                continue; // CSV に存在しないタグはスキップ
            }
            resolved_.push_back({e.name, e.axis, it->second});
        }
    }

    // WD14 スコアベクタを採点する。
    // wd14_scores: [N_TAGS] sigmoid 済みスコア (Wd14Tagger::infer の出力)。
    // id:          キャラ同一性 (digits_per_hand で Digits/Hands 軸スキップ判定)。
    QaReport evaluate(const std::vector<float>& wd14_scores,
                      const CharacterIdentity& id) const
    {
        // DIGITS_UNCOUNTABLE (肉球/鉤爪/ミトン) のキャラは指の本数カウントが
        // 意味を持たないため、Digits/Hands 軸の検査をスキップする。
        const bool skip_digits = (id.digits_per_hand == DIGITS_UNCOUNTABLE);

        QaReport report;
        for (const auto& e : resolved_)
        {
            // uncountable キャラでは Digits/Hands 軸を検査しない。
            if (skip_digits &&
                (e.axis == AnomalyAxis::Digits || e.axis == AnomalyAxis::Hands))
            {
                continue;
            }

            // wd14_scores の範囲外 index は安全に skip (CSV と入力長の不整合に備える)。
            if (e.index >= wd14_scores.size())
            {
                continue;
            }

            const float score = wd14_scores[e.index];
            if (score >= flag_threshold_)
            {
                report.hits.push_back({e.name, e.axis, score});
                if (score > report.overall_anomaly)
                {
                    report.overall_anomaly = score;
                }
            }
        }

        // soft flag のみ。棄却はしない。
        report.flagged = report.overall_anomaly >= flag_threshold_;
        return report;
    }

    // 解決できたカタログ件数 (CSV で index が引けた異常タグ数)。
    // テストの smoke (実 CSV で全カタログが解決できるか) に使う。
    size_t resolved_count() const noexcept
    {
        return resolved_.size();
    }

    // flag 閾値 (テスト用 getter)。
    float flag_threshold() const noexcept
    {
        return flag_threshold_;
    }

private:
    // CSV (tag_id,name,category,count) を読み name → 出力 index マップを構築する。
    // データ行 i (header 除く) が WD14 出力 index i に対応する。
    // CSV パースは最小自前 (ifstream + ',' split)。safetensors.hpp の最小パーサと同流儀。
    static std::unordered_map<std::string, size_t> load_name_to_index(const std::string& csv_path)
    {
        std::unordered_map<std::string, size_t> name_to_index;

        std::ifstream ifs(csv_path);
        if (!ifs)
        {
            // 開けない場合は空マップを返す (全カタログが未解決 = skip)。
            return name_to_index;
        }

        std::string line;
        bool        header = true;
        size_t      index  = 0;
        while (std::getline(ifs, line))
        {
            if (header)
            {
                header = false; // ヘッダ行 (tag_id,name,category,count) を読み飛ばす
                continue;
            }

            // 行末 \r (CRLF) を除去する。
            if (!line.empty() && line.back() == '\r')
            {
                line.pop_back();
            }

            // 2 列目 (name) を取り出す。
            std::stringstream ss(line);
            std::string       col;
            std::getline(ss, col, ','); // tag_id
            std::getline(ss, col, ','); // name

            // CSV の name は前後空白を持たない前提。重複時は先勝ち (先に出た index を保持)。
            name_to_index.emplace(col, index);
            ++index;
        }
        return name_to_index;
    }

    float                      flag_threshold_;
    std::vector<ResolvedEntry> resolved_;
};

} // namespace dollama

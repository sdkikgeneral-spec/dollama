// QualityGate (Model B Stage 1: WD14 異常タグ soft ゲート) 単体テスト
//
// OV 非依存。合成 selected_tags.csv (一時ファイル) + 合成 WD14 スコアベクタで実走する。
// 本開発機 (OV C++ 無し) でもテスト緑になるのが本タスクの肝。
//
// 5 サブテスト:
//   1. 異常検出       — extra_arms 高スコアで hit/flagged を確認 (軸 = Limbs)。
//   2. クリーン       — 全タグ低スコアで flagged == false / overall ≈ 0。
//   3. uncountable    — DIGITS_UNCOUNTABLE で Digits/Hands hit が出ず、他軸は依然検出。
//   4. 閾値境界       — スコア = 閾値ちょうどで flagged (>=)、わずか下で非 flagged。
//   5. CSV 欠落 skip  — カタログにあるが CSV に無い名があってもクラッシュせず解決済み分のみ評価。
//
// 全テスト通過時は "[test_quality_gate] ALL PASSED" を出力して return 0。
// 失敗時は std::cerr に出力して return 1。
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "infer/quality_gate.hpp"

namespace fs = std::filesystem;

namespace dollama
{

// 合成 CSV を一時ファイルへ書き出すヘルパ。
// tag_names[i] が WD14 出力 index i に対応する (header 1 行 + データ行)。
// 戻り値: 書き出したファイルパス。
static std::string write_synthetic_csv(const std::vector<std::string>& tag_names,
                                       const std::string& path)
{
    std::ofstream ofs(path);
    ofs << "tag_id,name,category,count\n"; // header
    for (size_t i = 0; i < tag_names.size(); ++i)
    {
        // tag_id / category / count はパースで使わないのでダミー値で良い。
        ofs << (1000000 + i) << "," << tag_names[i] << ",0,123\n";
    }
    ofs.close();
    return path;
}

// 一時 CSV のレイアウトを共通化する。
// index と name の対応を固定し、各テストはこの index にスコアを置いてテストする。
// 主要な異常タグを含め、無害タグも混ぜる (実 CSV の縮小版)。
static std::vector<std::string> synthetic_tag_names()
{
    return {
        "1girl",            // 0  無害
        "solo",             // 1  無害
        "extra_arms",       // 2  Limbs
        "multiple_heads",   // 3  Head
        "bad_hands",        // 4  Hands
        "fewer_digits",     // 5  Digits
        "bad_anatomy",      // 6  GlobalAnatomy
        "smile",            // 7  無害
        "long_hair",        // 8  無害
        "extra_eyes",       // 9  Eyes
    };
}

// index ヘルパ (synthetic_tag_names の順序に依存)。
static constexpr size_t IDX_EXTRA_ARMS     = 2;
static constexpr size_t IDX_MULTIPLE_HEADS = 3;
static constexpr size_t IDX_BAD_HANDS      = 4;
static constexpr size_t IDX_FEWER_DIGITS   = 5;

// hits の中に指定軸の hit が含まれるか。
static bool has_axis(const QaReport& r, AnomalyAxis axis)
{
    for (const auto& h : r.hits)
    {
        if (h.axis == axis)
        {
            return true;
        }
    }
    return false;
}

// ----------------------------------------------------------------
// テスト1: 異常検出 (extra_arms 高スコア → hit/flagged, 軸 = Limbs)
// ----------------------------------------------------------------
static bool test_anomaly_detection(const QualityGate& gate)
{
    std::vector<float> scores(synthetic_tag_names().size(), 0.05f);
    scores[IDX_EXTRA_ARMS] = 0.9f; // 高スコアの extra_arms

    CharacterIdentity id; // digits_per_hand = 5 (既定・人間)
    const QaReport    r = gate.evaluate(scores, id);

    if (!r.flagged)
    {
        std::cerr << "[test_anomaly_detection] flagged が false\n";
        return false;
    }
    if (r.hits.empty())
    {
        std::cerr << "[test_anomaly_detection] hits が空\n";
        return false;
    }
    if (!has_axis(r, AnomalyAxis::Limbs))
    {
        std::cerr << "[test_anomaly_detection] Limbs 軸の hit が無い\n";
        return false;
    }
    if (r.overall_anomaly < 0.89f || r.overall_anomaly > 0.91f)
    {
        std::cerr << "[test_anomaly_detection] overall_anomaly 不正: "
                  << r.overall_anomaly << "\n";
        return false;
    }

    std::cout << "[test_anomaly_detection] PASSED  (overall="
              << r.overall_anomaly << ", hits=" << r.hits.size() << ")\n";
    return true;
}

// ----------------------------------------------------------------
// テスト2: クリーン (全タグ低スコア → flagged == false, overall ≈ 0)
// ----------------------------------------------------------------
static bool test_clean_image(const QualityGate& gate)
{
    std::vector<float> scores(synthetic_tag_names().size(), 0.05f); // 全タグ低スコア

    CharacterIdentity id;
    const QaReport    r = gate.evaluate(scores, id);

    if (r.flagged)
    {
        std::cerr << "[test_clean_image] flagged が true\n";
        return false;
    }
    if (!r.hits.empty())
    {
        std::cerr << "[test_clean_image] hits が空でない: " << r.hits.size() << "\n";
        return false;
    }
    if (r.overall_anomaly != 0.0f)
    {
        std::cerr << "[test_clean_image] overall_anomaly が 0 でない: "
                  << r.overall_anomaly << "\n";
        return false;
    }

    std::cout << "[test_clean_image] PASSED  (overall=0, hits=0)\n";
    return true;
}

// ----------------------------------------------------------------
// テスト3: uncountable skip
// DIGITS_UNCOUNTABLE で fewer_digits / bad_hands を高スコアにしても
// Digits/Hands hit が出ないこと。他軸 (multiple_heads = Head) は依然検出されること。
// ----------------------------------------------------------------
static bool test_uncountable_skip(const QualityGate& gate)
{
    std::vector<float> scores(synthetic_tag_names().size(), 0.05f);
    scores[IDX_FEWER_DIGITS]   = 0.95f; // Digits (スキップされるべき)
    scores[IDX_BAD_HANDS]      = 0.95f; // Hands  (スキップされるべき)
    scores[IDX_MULTIPLE_HEADS] = 0.95f; // Head   (検出されるべき)

    CharacterIdentity id;
    id.digits_per_hand = DIGITS_UNCOUNTABLE; // 肉球/鉤爪/ミトン

    const QaReport r = gate.evaluate(scores, id);

    if (has_axis(r, AnomalyAxis::Digits))
    {
        std::cerr << "[test_uncountable_skip] Digits hit が出てしまった\n";
        return false;
    }
    if (has_axis(r, AnomalyAxis::Hands))
    {
        std::cerr << "[test_uncountable_skip] Hands hit が出てしまった\n";
        return false;
    }
    if (!has_axis(r, AnomalyAxis::Head))
    {
        std::cerr << "[test_uncountable_skip] Head hit が検出されなかった\n";
        return false;
    }
    if (!r.flagged)
    {
        std::cerr << "[test_uncountable_skip] flagged が false (Head で立つべき)\n";
        return false;
    }

    std::cout << "[test_uncountable_skip] PASSED  (Digits/Hands skip, Head 検出, hits="
              << r.hits.size() << ")\n";
    return true;
}

// ----------------------------------------------------------------
// テスト4: 閾値境界 (>= 境界で flagged、わずか下で非 flagged)
// ----------------------------------------------------------------
static bool test_threshold_boundary(const QualityGate& gate)
{
    const float thr = gate.flag_threshold();
    CharacterIdentity id;

    // 境界ちょうど: extra_arms = 閾値 → flagged (>=)
    {
        std::vector<float> scores(synthetic_tag_names().size(), 0.0f);
        scores[IDX_EXTRA_ARMS] = thr;
        const QaReport r = gate.evaluate(scores, id);
        if (!r.flagged)
        {
            std::cerr << "[test_threshold_boundary] 境界ちょうど (" << thr
                      << ") で flagged が false\n";
            return false;
        }
    }

    // わずか下: extra_arms = 閾値 - 0.01 → 非 flagged
    {
        std::vector<float> scores(synthetic_tag_names().size(), 0.0f);
        scores[IDX_EXTRA_ARMS] = thr - 0.01f;
        const QaReport r = gate.evaluate(scores, id);
        if (r.flagged)
        {
            std::cerr << "[test_threshold_boundary] 閾値直下 (" << (thr - 0.01f)
                      << ") で flagged が true\n";
            return false;
        }
        if (!r.hits.empty())
        {
            std::cerr << "[test_threshold_boundary] 閾値直下で hits が空でない\n";
            return false;
        }
    }

    std::cout << "[test_threshold_boundary] PASSED  (>= 境界判定, thr=" << thr << ")\n";
    return true;
}

// ----------------------------------------------------------------
// テスト5: CSV 欠落タグの安全 skip
// カタログにあるが CSV に無いタグ名があってもクラッシュせず、解決済み分だけ評価する。
// ----------------------------------------------------------------
static bool test_missing_csv_tags()
{
    // extra_arms と bad_anatomy だけ含む縮小 CSV (multiple_heads 等は欠落)。
    const std::vector<std::string> partial_names = {
        "1girl",       // 0
        "extra_arms",  // 1  Limbs
        "bad_anatomy", // 2  GlobalAnatomy
    };

    const std::string csv_path =
        (fs::temp_directory_path() / "dollama_qg_partial.csv").string();
    write_synthetic_csv(partial_names, csv_path);

    bool ok = true;
    try
    {
        QualityGate gate(csv_path, 0.4f);

        // カタログ全 18 件のうち CSV にあるのは extra_arms / bad_anatomy の 2 件のみ。
        if (gate.resolved_count() != 2)
        {
            std::cerr << "[test_missing_csv_tags] 解決数が想定外: "
                      << gate.resolved_count() << " (期待 2)\n";
            ok = false;
        }

        // extra_arms (index 1) を高スコアにして検出できること。
        std::vector<float> scores(partial_names.size(), 0.05f);
        scores[1] = 0.8f; // extra_arms
        CharacterIdentity id;
        const QaReport    r = gate.evaluate(scores, id);
        if (!r.flagged || !has_axis(r, AnomalyAxis::Limbs))
        {
            std::cerr << "[test_missing_csv_tags] 解決済みタグの検出に失敗\n";
            ok = false;
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "[test_missing_csv_tags] 例外発生: " << e.what() << "\n";
        ok = false;
    }

    std::remove(csv_path.c_str()); // 後始末

    if (ok)
    {
        std::cout << "[test_missing_csv_tags] PASSED  (欠落タグ安全 skip)\n";
    }
    return ok;
}

// ----------------------------------------------------------------
// 任意 smoke: 実 selected_tags.csv があればカタログ全 18 名の index 解決数を情報出力。
// ----------------------------------------------------------------
static void info_real_csv_resolve()
{
    const std::vector<std::string> candidates = {
        "../models/wd14-swinv2-tagger-v3/selected_tags.csv",
        "models/wd14-swinv2-tagger-v3/selected_tags.csv",
        "../../models/wd14-swinv2-tagger-v3/selected_tags.csv",
    };
    std::string real;
    for (const auto& p : candidates)
    {
        if (fs::exists(p))
        {
            real = p;
            break;
        }
    }
    if (real.empty())
    {
        std::cout << "[info_real_csv_resolve] 実 selected_tags.csv なし、スキップ\n";
        return;
    }

    QualityGate gate(real, 0.4f);
    std::cout << "[info_real_csv_resolve] 実 CSV (" << real << ") 解決数="
              << gate.resolved_count() << " / カタログ件数="
              << QualityGate::catalog().size() << "\n";
}

} // namespace dollama

int main()
{
    using namespace dollama;

    // 共通の合成 CSV を一時ファイルへ書き出す。
    const std::string csv_path =
        (fs::temp_directory_path() / "dollama_qg_main.csv").string();
    write_synthetic_csv(synthetic_tag_names(), csv_path);

    bool ok = true;
    try
    {
        QualityGate gate(csv_path, 0.4f);

        ok = test_anomaly_detection(gate) && ok;
        ok = test_clean_image(gate)       && ok;
        ok = test_uncountable_skip(gate)  && ok;
        ok = test_threshold_boundary(gate) && ok;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[test_quality_gate] 想定外の例外: " << e.what() << "\n";
        ok = false;
    }

    // テスト5 は独自 CSV を使うので gate に依存しない。
    ok = test_missing_csv_tags() && ok;

    // 任意: 実 CSV の解決数を情報出力 (判定外)。
    info_real_csv_resolve();

    std::remove(csv_path.c_str()); // 後始末

    if (!ok)
    {
        std::cerr << "[test_quality_gate] FAILED\n";
        return 1;
    }

    std::cout << "[test_quality_gate] ALL PASSED\n";
    return 0;
}

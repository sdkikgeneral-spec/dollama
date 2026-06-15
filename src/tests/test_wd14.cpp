// Wd14Tagger 単体テスト + 推論レイテンシ計測
// HAVE_OPENVINO 未定義時は [SKIP] を出力して終了する。
// モデルファイルが存在しない場合も [SKIP] で終了する。
// 全テスト通過時は "[test_wd14] ALL PASSED" を出力して return 0。
// 失敗時は std::cerr に出力して return 1。
//
// 前処理レンジについて:
//   probe8 (dollma_probe8_wd14.py) はダミー入力を
//   np.random.randint(0,255,...,uint8).astype(np.float32) で作る (0-255 レンジ・正規化なし)。
//   本テストもこれに合わせ、0-255 レンジの float を合成入力に用いる。
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#ifdef HAVE_OPENVINO
#include "infer/wd14.hpp"
#endif

namespace fs = std::filesystem;

// モデルファイルの候補パス (実行ディレクトリに応じて探索)
// meson test: build/ から実行 → ../models/...
// 直接実行: プロジェクトルートから → models/...
static std::string find_model_xml()
{
    const std::vector<std::string> candidates = {
        "../models/wd14-swinv2-tagger-v3/model_ov.xml",
        "models/wd14-swinv2-tagger-v3/model_ov.xml",
        "../../models/wd14-swinv2-tagger-v3/model_ov.xml",
    };
    for (const auto& p : candidates)
    {
        if (fs::exists(p))
        {
            return p;
        }
    }
    return candidates[0]; // 見つからない場合は先頭を返す (SKIP メッセージ用)
}

namespace dollama {

#ifdef HAVE_OPENVINO

using Clock    = std::chrono::steady_clock;
using MsDouble = std::chrono::duration<double, std::milli>;

// 合成入力を生成する (probe8 準拠: 0-255 レンジの float, 正規化なし)
// seed 固定で再現可能なランダム画素を作る。
static std::vector<float> make_input(uint32_t seed)
{
    std::vector<float> pixels(Wd14Tagger::INPUT_NUMEL);
    std::mt19937                          rng(seed);
    std::uniform_int_distribution<int>    dist(0, 254); // probe8: randint(0,255) = 0..254
    for (auto& v : pixels)
    {
        v = static_cast<float>(dist(rng)); // uint8 → float32 キャスト相当
    }
    return pixels;
}

// ----------------------------------------------------------------
// テスト1: 出力サイズ確認
// 出力ベクタのサイズがモデル出力の実 size (= N_TAGS = 10861) と一致すること。
// ----------------------------------------------------------------
static bool test_output_shape(Wd14Tagger& tagger)
{
    const auto pixels = make_input(1);
    const auto out    = tagger.infer(pixels);

    // 真の期待値は IR 出力の実 size = N_TAGS。CSV 由来定数と二重化しないが、
    // 定数 N_TAGS は selected_tags.csv の行数と一致している前提で照合する。
    const size_t expected = static_cast<size_t>(Wd14Tagger::N_TAGS);

    if (out.size() != expected)
    {
        std::cerr << "[test_output_shape] サイズ不正: got=" << out.size()
                  << " expected=" << expected << "\n";
        return false;
    }

    std::cout << "[test_output_shape] PASSED  (size=" << out.size() << ")\n";
    return true;
}

// ----------------------------------------------------------------
// テスト2: スコア値域確認
// 全スコアが [0, 1] に収まること (sigmoid 済み出力)。
// ----------------------------------------------------------------
static bool test_scores_range(Wd14Tagger& tagger)
{
    const auto pixels = make_input(2);
    const auto out    = tagger.infer(pixels);

    for (size_t i = 0; i < out.size(); ++i)
    {
        if (out[i] < 0.0f || out[i] > 1.0f)
        {
            std::cerr << "[test_scores_range] スコアが範囲外: idx=" << i
                      << " value=" << out[i] << "\n";
            return false;
        }
    }

    std::cout << "[test_scores_range] PASSED  (all in [0,1], N=" << out.size() << ")\n";
    return true;
}

// ----------------------------------------------------------------
// テスト3: 決定性確認
// 同一入力で 2 回推論し、出力が完全一致すること。
// ----------------------------------------------------------------
static bool test_determinism(Wd14Tagger& tagger)
{
    const auto pixels = make_input(3);
    const auto a      = tagger.infer(pixels);
    const auto b      = tagger.infer(pixels);

    if (a.size() != b.size())
    {
        std::cerr << "[test_determinism] サイズ不一致: " << a.size()
                  << " vs " << b.size() << "\n";
        return false;
    }

    for (size_t i = 0; i < a.size(); ++i)
    {
        if (a[i] != b[i])
        {
            std::cerr << "[test_determinism] 出力不一致: idx=" << i
                      << " " << a[i] << " vs " << b[i] << "\n";
            return false;
        }
    }

    std::cout << "[test_determinism] PASSED  (2 回完全一致)\n";
    return true;
}

// ----------------------------------------------------------------
// テスト4: 入力サイズガード
// 不正サイズの入力で std::invalid_argument が飛ぶこと。
// ----------------------------------------------------------------
static bool test_input_size_guard(Wd14Tagger& tagger)
{
    std::vector<float> bad(Wd14Tagger::INPUT_NUMEL - 1, 0.0f); // 1 要素足りない

    bool threw = false;
    try
    {
        tagger.infer(bad);
    }
    catch (const std::invalid_argument&)
    {
        threw = true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[test_input_size_guard] 想定外の例外: " << e.what() << "\n";
        return false;
    }

    if (!threw)
    {
        std::cerr << "[test_input_size_guard] 例外が飛ばなかった\n";
        return false;
    }

    std::cout << "[test_input_size_guard] PASSED  (invalid_argument を捕捉)\n";
    return true;
}

// ----------------------------------------------------------------
// ベンチマーク: 推論レイテンシ (中央値)
// ウォームアップ 5 回 + 計測 100 回。PASS/FAIL 判定外、数値のみ出力。
// 期待 CPU ~101ms (probe8)。
// ----------------------------------------------------------------
static bool bench_infer_latency(Wd14Tagger& tagger)
{
    const auto pixels = make_input(42);

    // ウォームアップ
    for (int i = 0; i < 5; ++i)
    {
        tagger.infer(pixels);
    }

    // 計測 100 回
    constexpr int kN = 100;
    std::vector<double> times;
    times.reserve(kN);

    for (int i = 0; i < kN; ++i)
    {
        auto t0 = Clock::now();
        tagger.infer(pixels);
        double ms = MsDouble(Clock::now() - t0).count();
        times.push_back(ms);
    }

    std::sort(times.begin(), times.end());
    double median = times[kN / 2];

    std::cout << "[bench_infer_latency]\n";
    std::cout << "  BENCH: wd14_infer median=" << median << " ms"
              << "  (min=" << times.front()
              << " ms, max=" << times.back() << " ms, N=" << kN << ")\n";
    return true;
}

// ----------------------------------------------------------------
// 任意: selected_tags.csv があれば top-10 タグ名を情報出力 (assert にしない)
// ----------------------------------------------------------------
static void info_top_tags(Wd14Tagger& tagger, const std::string& xml_path)
{
    // CSV パスは model_ov.xml と同じディレクトリの selected_tags.csv
    fs::path csv = fs::path(xml_path).parent_path() / "selected_tags.csv";
    if (!fs::exists(csv))
    {
        std::cout << "[info_top_tags] selected_tags.csv なし、スキップ\n";
        return;
    }

    // CSV 読み込み (name 列を抽出。ヘッダは tag_id,name,category,count を想定)
    std::ifstream ifs(csv);
    if (!ifs)
    {
        std::cout << "[info_top_tags] CSV を開けず、スキップ\n";
        return;
    }

    std::vector<std::string> names;
    std::string              line;
    bool                     header = true;
    while (std::getline(ifs, line))
    {
        if (header)
        {
            header = false; // ヘッダ行を読み飛ばす
            continue;
        }
        // 2 列目 (name) を取り出す
        std::stringstream ss(line);
        std::string       col;
        std::getline(ss, col, ','); // tag_id
        std::getline(ss, col, ','); // name
        names.push_back(col);
    }

    // probe8 STEP3 相当のダミー画像 (R=200,G=150,B=180 一様)
    std::vector<float> img(Wd14Tagger::INPUT_NUMEL);
    for (size_t i = 0; i < img.size(); i += 3)
    {
        img[i + 0] = 200.0f;
        img[i + 1] = 150.0f;
        img[i + 2] = 180.0f;
    }

    const auto out = tagger.infer(img);
    const auto top = Wd14Tagger::top_k(out, 10);

    std::cout << "[info_top_tags] 上位 10 タグ (ダミー画像 R200/G150/B180):\n";
    for (size_t rank = 0; rank < top.size(); ++rank)
    {
        size_t      idx  = top[rank];
        std::string name = (idx < names.size()) ? names[idx] : "<idx out of range>";
        std::cout << "  " << (rank + 1) << ". " << name
                  << "  (" << out[idx] << ")\n";
    }
}

#endif // HAVE_OPENVINO

} // namespace dollama

int main()
{
#ifndef HAVE_OPENVINO
    std::cout << "[SKIP] HAVE_OPENVINO が未定義です。"
                 "OpenVINO なしでビルドされているためテストをスキップします。\n";
    return 0;
#else
    // モデルファイル存在確認
    const std::string xml_path = find_model_xml();
    if (!fs::exists(xml_path))
    {
        std::cout << "[SKIP] モデルファイルが見つかりません: " << xml_path << "\n";
        return 0;
    }

    // Wd14Tagger を CPU でコンパイル (WD14 は CPU が最速)
    dollama::Wd14Tagger tagger(xml_path, "CPU");

    bool ok = true;
    ok = dollama::test_output_shape(tagger)     && ok;
    ok = dollama::test_scores_range(tagger)     && ok;
    ok = dollama::test_determinism(tagger)      && ok;
    ok = dollama::test_input_size_guard(tagger) && ok;
    ok = dollama::bench_infer_latency(tagger)   && ok;

    // 任意: top-10 タグ情報出力 (判定外)
    dollama::info_top_tags(tagger, xml_path);

    if (!ok)
    {
        std::cerr << "[test_wd14] FAILED\n";
        return 1;
    }

    std::cout << "[test_wd14] ALL PASSED\n";
    return 0;
#endif // HAVE_OPENVINO
}

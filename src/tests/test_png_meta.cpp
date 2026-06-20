// PNG メタ往復 (character_bible-spec §7) 単体テスト + ベンチ
// testing.md 形式: 各検証は if (!cond) { cerr; return false; }、
// main で ok = test_xxx() && ok 集約、成功は PASSED / 最後に ALL PASSED。
//
// 検証範囲 (dispatch プラン):
//   1. ラウンドトリップ (日本語 name 含む全フィールド)
//   2. 寄生健全性 (実画像へ注入後も read_png_size が幅高さを返す)
//   3. 頑健性 (複数 tEXt / tEXt なし / 壊れ length / 空配列・全デフォルト)
//   4. enum 全網羅 (Sex 3値・MattingMode 3値)
//   5. ベンチ (embed / read を N 回, ns/op)
#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "io/png_meta.hpp"
#include "server/png.hpp"

namespace dollama
{

// §7 対象フィールドのみを field-by-field で照合する (スキーマ外は対象外)。
static bool identity_eq(const CharacterIdentity& a, const CharacterIdentity& b)
{
    return a.name == b.name && a.age == b.age && a.sex == b.sex &&
           a.canonical_tags == b.canonical_tags &&
           a.forbidden_tags == b.forbidden_tags &&
           a.digits_per_hand == b.digits_per_hand &&
           a.embedding_slot == b.embedding_slot && a.seed == b.seed;
}

static bool scene_eq(const SceneSpec& a, const SceneSpec& b)
{
    return a.pose_tags == b.pose_tags &&
           a.expression_tags == b.expression_tags &&
           a.composition == b.composition && a.scene_seed == b.scene_seed;
}

static bool output_eq(const OutputSpec& a, const OutputSpec& b)
{
    return a.matting == b.matting && a.color_mode == b.color_mode &&
           a.isolation_tag == b.isolation_tag && a.emit_alpha == b.emit_alpha;
}

// テスト用に最小の正規 PNG (2x2 RGB) を作る。
static std::vector<uint8_t> make_dummy_png(int w = 2, int h = 2)
{
    std::vector<uint8_t> rgb(static_cast<size_t>(w) * h * 3, 0x80);
    return encode_png_rgb8(rgb, w, h);
}

// ----------------------------------------------------------------
// 1. ラウンドトリップ: 全フィールド埋め (日本語 name) → 焼き込み → 読み戻し一致
// ----------------------------------------------------------------
static bool test_roundtrip_full()
{
    CharacterIdentity id;
    id.name            = "アリア・星詠み";  // 日本語 name (ASCII エスケープ往復)
    id.age             = 17;
    id.sex             = Sex::Female;
    id.canonical_tags  = {"silver hair", "red eyes", "long hair", "twin tails"};
    id.forbidden_tags  = {"short hair", "blue eyes"};
    id.digits_per_hand = 5;
    id.embedding_slot  = 26;
    id.seed            = 998244353ull;

    SceneSpec scene;
    scene.pose_tags       = {"sitting"};
    scene.expression_tags = {"smile"};
    scene.composition     = "upper body";
    scene.scene_seed      = 41;

    OutputSpec out;
    out.matting       = MattingMode::Segment;
    out.isolation_tag = "simple background";
    out.emit_alpha    = true;

    std::vector<uint8_t> png = make_dummy_png();
    std::vector<uint8_t> with = write_bible_png(png, id, scene, out);

    CharacterIdentity id2;
    SceneSpec scene2;
    OutputSpec out2;
    if (!read_bible_png(with, id2, scene2, out2))
    {
        std::cerr << "[test_roundtrip_full] read_bible_png 失敗\n";
        return false;
    }

    if (!identity_eq(id, id2))
    {
        std::cerr << "[test_roundtrip_full] identity 不一致 (name=" << id2.name << ")\n";
        return false;
    }
    if (!scene_eq(scene, scene2))
    {
        std::cerr << "[test_roundtrip_full] scene 不一致\n";
        return false;
    }
    if (!output_eq(out, out2))
    {
        std::cerr << "[test_roundtrip_full] output 不一致\n";
        return false;
    }
    std::cout << "[test_roundtrip_full] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// 2. 寄生健全性: 実画像へ注入後も read_png_size が正しい幅高さを返す
//    (IHDR/IDAT 非破壊) + シグネチャ/IEND が残る
// ----------------------------------------------------------------
static bool test_png_integrity()
{
    const int W = 13, H = 7;
    std::vector<uint8_t> rgb(static_cast<size_t>(W) * H * 3);
    for (size_t i = 0; i < rgb.size(); ++i)
    {
        rgb[i] = static_cast<uint8_t>(i * 7 + 3);
    }
    std::vector<uint8_t> png = encode_png_rgb8(rgb, W, H);

    int w0 = 0, h0 = 0;
    if (!read_png_size(png, w0, h0) || w0 != W || h0 != H)
    {
        std::cerr << "[test_png_integrity] 注入前サイズ不正\n";
        return false;
    }

    CharacterIdentity id;
    id.name = "Aria";
    std::vector<uint8_t> with = write_bible_png(png, id, SceneSpec{}, OutputSpec{});

    // サイズ拡大 (チャンク追加分)
    if (with.size() <= png.size())
    {
        std::cerr << "[test_png_integrity] チャンクが追加されていない\n";
        return false;
    }

    // IHDR 非破壊 → read_png_size が同じ幅高さを返す
    int w1 = 0, h1 = 0;
    if (!read_png_size(with, w1, h1) || w1 != W || h1 != H)
    {
        std::cerr << "[test_png_integrity] 注入後サイズ不正: " << w1 << "x" << h1 << "\n";
        return false;
    }

    // 末尾 IEND チャンク (12 バイト: len=0, "IEND", crc) が残る
    if (with.size() < 12)
    {
        std::cerr << "[test_png_integrity] 短すぎ\n";
        return false;
    }
    const uint8_t* tail = with.data() + with.size() - 8;  // type(4)+crc(4)
    if (!(tail[0] == 'I' && tail[1] == 'E' && tail[2] == 'N' && tail[3] == 'D'))
    {
        std::cerr << "[test_png_integrity] 末尾が IEND でない\n";
        return false;
    }

    // 読み戻しが効く
    CharacterIdentity id2;
    SceneSpec s2;
    OutputSpec o2;
    if (!read_bible_png(with, id2, s2, o2) || id2.name != "Aria")
    {
        std::cerr << "[test_png_integrity] 注入後読み戻し失敗\n";
        return false;
    }

    std::cout << "[test_png_integrity] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// 3a. 複数 tEXt: 他 keyword の tEXt を先入れしても dollama/bible を正しく拾う
// ----------------------------------------------------------------
static bool test_multiple_text_chunks()
{
    std::vector<uint8_t> png = make_dummy_png();

    // 別 keyword の tEXt を先に注入 (detail::put_chunk を直接使って IEND 直前へ)。
    // 簡便のため embed_bible_in_png を「別 keyword」で再現せず、bible 注入を 2 回行う:
    // まず別キーの json を持つ tEXt 相当を手で作る代わりに、別 keyword チャンクを差し込む。
    {
        const std::string kw = "Software";
        const std::string tx = "dollama-test";
        std::vector<uint8_t> cd;
        cd.insert(cd.end(), kw.begin(), kw.end());
        cd.push_back(0x00);
        cd.insert(cd.end(), tx.begin(), tx.end());
        std::vector<uint8_t> chunk;
        detail::put_chunk(chunk, "tEXt", cd);

        // IEND 直前へ自前で差し込む (シグネチャ後を走査)。
        size_t pos = 8;
        size_t iend = static_cast<size_t>(-1);
        while (pos + 8 <= png.size())
        {
            uint32_t len = (uint32_t(png[pos]) << 24) | (uint32_t(png[pos + 1]) << 16) |
                           (uint32_t(png[pos + 2]) << 8) | uint32_t(png[pos + 3]);
            const char* t = reinterpret_cast<const char*>(&png[pos + 4]);
            if (t[0] == 'I' && t[1] == 'E' && t[2] == 'N' && t[3] == 'D')
            {
                iend = pos;
                break;
            }
            pos += 12 + len;
        }
        std::vector<uint8_t> tmp;
        tmp.insert(tmp.end(), png.begin(), png.begin() + iend);
        tmp.insert(tmp.end(), chunk.begin(), chunk.end());
        tmp.insert(tmp.end(), png.begin() + iend, png.end());
        png = std::move(tmp);
    }

    // bible を注入
    CharacterIdentity id;
    id.name = "Mixed";
    id.seed = 12345;
    std::vector<uint8_t> with = write_bible_png(png, id, SceneSpec{}, OutputSpec{});

    CharacterIdentity id2;
    SceneSpec s2;
    OutputSpec o2;
    if (!read_bible_png(with, id2, s2, o2) || id2.name != "Mixed" || id2.seed != 12345)
    {
        std::cerr << "[test_multiple_text_chunks] 複数 tEXt から拾えない\n";
        return false;
    }
    std::cout << "[test_multiple_text_chunks] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// 3b. tEXt なし PNG → read false (クラッシュなし)
// ----------------------------------------------------------------
static bool test_no_text_chunk()
{
    std::vector<uint8_t> png = make_dummy_png();
    nlohmann::json bible;
    if (read_bible_from_png(png, bible))
    {
        std::cerr << "[test_no_text_chunk] tEXt なしで true を返した\n";
        return false;
    }
    std::cout << "[test_no_text_chunk] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// 3c. 切り詰め / 壊れた length → false で境界外読みなし
// ----------------------------------------------------------------
static bool test_corrupt_png()
{
    // 非 PNG (シグネチャ不正)
    {
        std::vector<uint8_t> junk = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        nlohmann::json b;
        if (read_bible_from_png(junk, b))
        {
            std::cerr << "[test_corrupt_png] 非 PNG で true\n";
            return false;
        }
    }
    // 空
    {
        std::vector<uint8_t> empty;
        nlohmann::json b;
        if (read_bible_from_png(empty, b))
        {
            std::cerr << "[test_corrupt_png] 空で true\n";
            return false;
        }
    }
    // tEXt を注入後、途中で切り詰め (length が残バイトを超える)
    {
        CharacterIdentity id;
        id.name = "Trunc";
        std::vector<uint8_t> with =
            write_bible_png(make_dummy_png(), id, SceneSpec{}, OutputSpec{});
        // 末尾を半分に切る
        with.resize(with.size() / 2);
        nlohmann::json b;
        // クラッシュせず false (または途中で IEND 前に当たれば true もあり得るが、
        // 切り詰めにより tEXt の length が残量を超えるため false を期待)。
        bool r = read_bible_from_png(with, b);
        if (r)
        {
            std::cerr << "[test_corrupt_png] 切り詰めで true (境界外の懸念)\n";
            return false;
        }
    }
    // length を巨大値に細工 (シグネチャ後の最初のチャンク length を破壊)
    {
        std::vector<uint8_t> png = make_dummy_png();
        png[8]  = 0xFF;
        png[9]  = 0xFF;
        png[10] = 0xFF;
        png[11] = 0xFF;
        nlohmann::json b;
        if (read_bible_from_png(png, b))
        {
            std::cerr << "[test_corrupt_png] 巨大 length で true\n";
            return false;
        }
        // embed も破壊せず入力相当を返す (クラッシュなし)
        std::vector<uint8_t> out = embed_bible_in_png(png, nlohmann::json{{"x", 1}});
        (void)out;
    }
    std::cout << "[test_corrupt_png] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// 3d. 空配列・全デフォルト bible 往復
// ----------------------------------------------------------------
static bool test_defaults_roundtrip()
{
    CharacterIdentity id;   // 全デフォルト (name 空, age 0, digits 5, slot -1, seed 0)
    SceneSpec scene;        // 全デフォルト (空配列, composition 空, scene_seed 0)
    OutputSpec out;         // matting Segment, isolation "simple background", emit true

    std::vector<uint8_t> with =
        write_bible_png(make_dummy_png(), id, scene, out);

    CharacterIdentity id2;
    SceneSpec scene2;
    OutputSpec out2;
    // 読み戻し先をデフォルトと違う値にしておき、上書き/維持が正しいか確認
    id2.name = "dirty";
    id2.age = 99;
    if (!read_bible_png(with, id2, scene2, out2))
    {
        std::cerr << "[test_defaults_roundtrip] 読み戻し失敗\n";
        return false;
    }
    if (!identity_eq(id, id2) || !scene_eq(scene, scene2) || !output_eq(out, out2))
    {
        std::cerr << "[test_defaults_roundtrip] デフォルト往復不一致\n";
        return false;
    }
    std::cout << "[test_defaults_roundtrip] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// 4. enum 全網羅: Sex 3値・MattingMode 3値の往復
// ----------------------------------------------------------------
static bool test_enum_coverage()
{
    Sex sexes[3] = {Sex::Female, Sex::Male, Sex::Other};
    MattingMode mattings[3] = {MattingMode::None, MattingMode::Segment, MattingMode::Native};
    ColorMode colors[3] = {ColorMode::Color, ColorMode::Greyscale, ColorMode::Lineart};

    for (Sex s : sexes)
    {
        for (MattingMode m : mattings)
        {
            for (ColorMode c : colors)
            {
                CharacterIdentity id;
                id.name = "E";
                id.sex  = s;
                OutputSpec out;
                out.matting    = m;
                out.color_mode = c;
                std::vector<uint8_t> with =
                    write_bible_png(make_dummy_png(), id, SceneSpec{}, out);
                CharacterIdentity id2;
                SceneSpec s2;
                OutputSpec o2;
                if (!read_bible_png(with, id2, s2, o2))
                {
                    std::cerr << "[test_enum_coverage] 読み戻し失敗\n";
                    return false;
                }
                if (id2.sex != s)
                {
                    std::cerr << "[test_enum_coverage] Sex 不一致\n";
                    return false;
                }
                if (o2.matting != m)
                {
                    std::cerr << "[test_enum_coverage] MattingMode 不一致\n";
                    return false;
                }
                if (o2.color_mode != c)
                {
                    std::cerr << "[test_enum_coverage] ColorMode 不一致\n";
                    return false;
                }
            }
        }
    }

    // 不明文字列のフォールバック (Sex::Female / MattingMode::Segment)
    {
        nlohmann::json j;
        j["dollama_bible_version"] = 1;
        j["identity"]["sex"]       = "alien";
        j["output"]["matting"]     = "magic";
        j["output"]["color_mode"]  = "rainbow";
        CharacterIdentity id2;
        SceneSpec s2;
        OutputSpec o2;
        if (!json_to_bible(j, id2, s2, o2))
        {
            std::cerr << "[test_enum_coverage] フォールバック json 受理失敗\n";
            return false;
        }
        if (id2.sex != Sex::Female || o2.matting != MattingMode::Segment ||
            o2.color_mode != ColorMode::Color)
        {
            std::cerr << "[test_enum_coverage] 不明文字列のフォールバック不正\n";
            return false;
        }
    }

    // color_mode 欠落 JSON → 既定 Color のまま (前方互換)。
    {
        nlohmann::json j;
        j["dollama_bible_version"] = 1;
        j["output"]["matting"]     = "segment";  // color_mode は意図的に欠落
        CharacterIdentity id2;
        SceneSpec s2;
        OutputSpec o2;
        o2.color_mode = ColorMode::Lineart;  // 欠落時に上書きされないことを確認するため汚す
        if (!json_to_bible(j, id2, s2, o2))
        {
            std::cerr << "[test_enum_coverage] 欠落 color_mode json 受理失敗\n";
            return false;
        }
        if (o2.color_mode != ColorMode::Lineart)
        {
            std::cerr << "[test_enum_coverage] 欠落 color_mode が既定で上書きされた\n";
            return false;
        }
    }

    // メジャー版不一致は拒否
    {
        nlohmann::json j;
        j["dollama_bible_version"] = 99;
        CharacterIdentity id2;
        SceneSpec s2;
        OutputSpec o2;
        if (json_to_bible(j, id2, s2, o2))
        {
            std::cerr << "[test_enum_coverage] メジャー版不一致を受理した\n";
            return false;
        }
    }

    std::cout << "[test_enum_coverage] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// 5. ベンチ: embed / read を N 回, ns/op を標準出力
// ----------------------------------------------------------------
static void bench_embed_read()
{
    CharacterIdentity id;
    id.name            = "アリア";
    id.age             = 17;
    id.canonical_tags  = {"silver hair", "red eyes", "long hair", "twin tails"};
    id.forbidden_tags  = {"short hair"};
    id.embedding_slot  = 26;
    id.seed            = 998244353ull;
    SceneSpec scene;
    scene.pose_tags = {"sitting"};
    OutputSpec out;

    std::vector<uint8_t> png = make_dummy_png(8, 8);
    const int iters = 10'000;

    // embed
    {
        volatile size_t sink = 0;
        nlohmann::json j = bible_to_json(id, scene, out);
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i)
        {
            std::vector<uint8_t> w = embed_bible_in_png(png, j);
            sink += w.size();
        }
        auto t1 = std::chrono::steady_clock::now();
        double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
        std::cout << "[bench_embed] " << (ns / iters) << " ns/op (" << iters
                  << " iters, sink=" << sink << ")\n";
    }

    // read
    {
        std::vector<uint8_t> with = write_bible_png(png, id, scene, out);
        volatile size_t sink = 0;
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i)
        {
            nlohmann::json b;
            if (read_bible_from_png(with, b))
            {
                sink += b.size();
            }
        }
        auto t1 = std::chrono::steady_clock::now();
        double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
        std::cout << "[bench_read] " << (ns / iters) << " ns/op (" << iters
                  << " iters, sink=" << sink << ")\n";
    }
}

} // namespace dollama

int main()
{
    bool ok = true;
    ok = dollama::test_roundtrip_full()        && ok;
    ok = dollama::test_png_integrity()         && ok;
    ok = dollama::test_multiple_text_chunks()  && ok;
    ok = dollama::test_no_text_chunk()         && ok;
    ok = dollama::test_corrupt_png()           && ok;
    ok = dollama::test_defaults_roundtrip()    && ok;
    ok = dollama::test_enum_coverage()         && ok;

    dollama::bench_embed_read();

    if (!ok)
    {
        std::cerr << "[test_png_meta] FAILED\n";
        return 1;
    }
    std::cout << "[test_png_meta] ALL PASSED\n";
    return 0;
}

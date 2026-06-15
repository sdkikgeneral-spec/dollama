// CharacterBible / プロンプト合成 単体テスト + ベンチ
// testing.md 形式: 各検証は if (!cond) { cerr; return false; }、
// main で ok = test_xxx() && ok 集約、成功は PASSED / 最後に ALL PASSED。
#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include "core/character.hpp"

namespace dollama {

// ----------------------------------------------------------------
// put 後 find が同じ name を返す / 未登録 name は nullptr
// ----------------------------------------------------------------
static bool test_bible_put_find()
{
    CharacterBible bible;
    CharacterIdentity c;
    c.name = "Aria";
    bible.put(c);

    const CharacterIdentity* found = bible.find("Aria");
    if (!found)
    {
        std::cerr << "[test_bible_put_find] find('Aria') が nullptr\n";
        return false;
    }
    if (found->name != "Aria")
    {
        std::cerr << "[test_bible_put_find] name 不一致: " << found->name << "\n";
        return false;
    }
    if (bible.find("Nobody") != nullptr)
    {
        std::cerr << "[test_bible_put_find] 未登録 name が nullptr でない\n";
        return false;
    }
    std::cout << "[test_bible_put_find] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// 同名 put で上書きされ size が増えない
// ----------------------------------------------------------------
static bool test_bible_overwrite()
{
    CharacterBible bible;
    CharacterIdentity c;
    c.name = "Aria";
    c.age  = 17;
    bible.put(c);

    c.age = 25;
    bible.put(c);

    if (bible.size() != 1)
    {
        std::cerr << "[test_bible_overwrite] size が 1 でない: " << bible.size() << "\n";
        return false;
    }
    const CharacterIdentity* found = bible.find("Aria");
    if (!found || found->age != 25)
    {
        std::cerr << "[test_bible_overwrite] 上書きされていない\n";
        return false;
    }
    std::cout << "[test_bible_overwrite] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// positive[0] が canonical_tags 先頭、pose→expression→composition→isolation の順
// (age は外見へ展開されない: spec §5)
// ----------------------------------------------------------------
static bool test_compose_order()
{
    CharacterIdentity id;
    id.canonical_tags = {"silver hair", "red eyes"};
    id.age = 17; // age は外見に影響しない

    SceneSpec scene;
    scene.pose_tags       = {"sitting"};
    scene.expression_tags = {"smile"};
    scene.composition     = "upper body";

    OutputSpec out;
    out.isolation_tag = "simple background";

    PromptParts p = compose_prompt(id, scene, out);

    std::vector<std::string> expected = {
        "silver hair", "red eyes",   // canonical (先頭固定)
        "sitting",                   // pose
        "smile",                     // expression
        "upper body",                // composition
        "simple background",         // isolation
    };

    if (p.positive != expected)
    {
        std::cerr << "[test_compose_order] positive 順序が期待と異なる\n";
        for (const auto& t : p.positive)
        {
            std::cerr << "  got: " << t << "\n";
        }
        return false;
    }
    if (p.positive.front() != "silver hair")
    {
        std::cerr << "[test_compose_order] positive[0] が canonical 先頭でない\n";
        return false;
    }
    std::cout << "[test_compose_order] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// forbidden_tags が negative に入る
// ----------------------------------------------------------------
static bool test_compose_negative()
{
    CharacterIdentity id;
    id.forbidden_tags = {"short hair", "blue eyes"};

    SceneSpec scene;
    OutputSpec out;
    out.quality_negatives = false; // forbidden_tags のみを検証する

    PromptParts p = compose_prompt(id, scene, out);

    std::vector<std::string> expected = {"short hair", "blue eyes"};
    if (p.negative != expected)
    {
        std::cerr << "[test_compose_negative] negative が forbidden_tags と一致しない\n";
        return false;
    }
    std::cout << "[test_compose_negative] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// identity.seed!=0 ならそれを採用、0 なら scene.scene_seed
// ----------------------------------------------------------------
static bool test_compose_seed()
{
    SceneSpec scene;
    scene.scene_seed = 41;
    OutputSpec out;

    // identity.seed != 0 → identity.seed 採用
    CharacterIdentity id1;
    id1.seed = 998244353;
    if (compose_prompt(id1, scene, out).seed != 998244353)
    {
        std::cerr << "[test_compose_seed] identity.seed が採用されていない\n";
        return false;
    }

    // identity.seed == 0 → scene.scene_seed 採用
    CharacterIdentity id2;
    id2.seed = 0;
    if (compose_prompt(id2, scene, out).seed != 41)
    {
        std::cerr << "[test_compose_seed] scene.scene_seed が採用されていない\n";
        return false;
    }
    std::cout << "[test_compose_seed] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// isolation_tag が positive 末尾に入る (空文字なら入らない)
// ----------------------------------------------------------------
static bool test_isolation_tag()
{
    CharacterIdentity id;
    id.canonical_tags = {"silver hair"};
    SceneSpec scene;

    // 非空: 末尾に入る
    OutputSpec out;
    out.isolation_tag = "simple background";
    PromptParts p1 = compose_prompt(id, scene, out);
    if (p1.positive.empty() || p1.positive.back() != "simple background")
    {
        std::cerr << "[test_isolation_tag] isolation_tag が末尾にない\n";
        return false;
    }

    // 空文字: 入らない (composition も空なので末尾は canonical のまま)
    OutputSpec out_empty;
    out_empty.isolation_tag = "";
    PromptParts p2 = compose_prompt(id, scene, out_empty);
    for (const auto& t : p2.positive)
    {
        if (t == "simple background")
        {
            std::cerr << "[test_isolation_tag] 空文字なのに isolation_tag が混入\n";
            return false;
        }
    }
    if (p2.positive.back() != "silver hair")
    {
        std::cerr << "[test_isolation_tag] 空文字時の末尾が想定外: " << p2.positive.back() << "\n";
        return false;
    }
    std::cout << "[test_isolation_tag] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// composition / isolation_tag が空文字のとき positive に空要素が入らない
// ----------------------------------------------------------------
static bool test_compose_empty_skip()
{
    CharacterIdentity id;
    id.canonical_tags = {"silver hair", "red eyes"};

    SceneSpec scene;
    scene.pose_tags   = {"standing"};
    scene.composition = ""; // 空

    OutputSpec out;
    out.isolation_tag = ""; // 空

    PromptParts p = compose_prompt(id, scene, out);

    // 空要素 ("") が混入していないこと
    for (const auto& t : p.positive)
    {
        if (t.empty())
        {
            std::cerr << "[test_compose_empty_skip] positive に空要素が混入\n";
            return false;
        }
    }

    // canonical + pose のみ = 3 要素
    std::vector<std::string> expected = {"silver hair", "red eyes", "standing"};
    if (p.positive != expected)
    {
        std::cerr << "[test_compose_empty_skip] positive が想定外\n";
        for (const auto& t : p.positive)
        {
            std::cerr << "  got: '" << t << "'\n";
        }
        return false;
    }
    std::cout << "[test_compose_empty_skip] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// quality_negatives=true で default_quality_negatives() が negative に合流 /
// false で入らない
// ----------------------------------------------------------------
static bool test_quality_negatives()
{
    CharacterIdentity id;
    id.forbidden_tags = {"blue eyes"};
    SceneSpec scene;

    const std::vector<std::string> qn = default_quality_negatives();

    // true: forbidden_tags の後に default_quality_negatives() が続く
    OutputSpec out_on;
    out_on.quality_negatives = true;
    PromptParts p_on = compose_prompt(id, scene, out_on);

    std::vector<std::string> expected_on = {"blue eyes"};
    expected_on.insert(expected_on.end(), qn.begin(), qn.end());
    if (p_on.negative != expected_on)
    {
        std::cerr << "[test_quality_negatives] true 時に品質ネガティブが合流していない\n";
        return false;
    }

    // false: forbidden_tags のみ
    OutputSpec out_off;
    out_off.quality_negatives = false;
    PromptParts p_off = compose_prompt(id, scene, out_off);
    if (p_off.negative != std::vector<std::string>{"blue eyes"})
    {
        std::cerr << "[test_quality_negatives] false 時に品質ネガティブが混入\n";
        return false;
    }

    // 本数を仮定する語が default に含まれていないこと
    for (const auto& t : qn)
    {
        if (t == "extra fingers" || t == "fewer digits")
        {
            std::cerr << "[test_quality_negatives] 本数仮定語が default に混入: " << t << "\n";
            return false;
        }
    }
    std::cout << "[test_quality_negatives] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// digits_per_hand のデフォルトが 5、DIGITS_UNCOUNTABLE==0
// ----------------------------------------------------------------
static bool test_digits_default()
{
    CharacterIdentity id;
    if (id.digits_per_hand != 5)
    {
        std::cerr << "[test_digits_default] digits_per_hand のデフォルトが 5 でない: "
                  << id.digits_per_hand << "\n";
        return false;
    }
    if (DIGITS_UNCOUNTABLE != 0)
    {
        std::cerr << "[test_digits_default] DIGITS_UNCOUNTABLE が 0 でない: "
                  << DIGITS_UNCOUNTABLE << "\n";
        return false;
    }
    std::cout << "[test_digits_default] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// ベンチ: compose_prompt のスループット (ns/op)
// PASS/FAIL 判定はせず数値のみ出力する。
// ----------------------------------------------------------------
static void bench_compose_prompt()
{
    CharacterIdentity id;
    id.canonical_tags = {"silver hair", "red eyes", "long hair", "twin tails"};
    id.forbidden_tags = {"short hair", "blue eyes"};
    id.seed = 998244353;

    SceneSpec scene;
    scene.pose_tags       = {"sitting"};
    scene.expression_tags = {"smile", "blush"};
    scene.composition     = "upper body";
    scene.scene_seed      = 41;

    OutputSpec out; // デフォルト (isolation_tag / quality_negatives 有効)

    const int iters = 1'000'000;
    volatile size_t sink = 0; // 最適化除去防止

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i)
    {
        PromptParts p = compose_prompt(id, scene, out);
        sink += p.positive.size() + p.negative.size();
    }
    auto t1 = std::chrono::steady_clock::now();

    double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    std::cout << "[bench_compose_prompt] " << (ns / iters) << " ns/op ("
              << iters << " iters, sink=" << sink << ")\n";
}

// ----------------------------------------------------------------
// ベンチ: CharacterBible::find のルックアップ (ns/op)
// 10,000 体登録し 1M 回ルックアップする。数値のみ出力。
// ----------------------------------------------------------------
static void bench_bible_find()
{
    CharacterBible bible;
    const int n = 10'000;
    std::vector<std::string> names;
    names.reserve(n);
    for (int i = 0; i < n; ++i)
    {
        CharacterIdentity c;
        c.name = "char_" + std::to_string(i);
        names.push_back(c.name);
        bible.put(c);
    }

    const int iters = 1'000'000;
    volatile size_t hits = 0; // 最適化除去防止

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i)
    {
        const CharacterIdentity* p = bible.find(names[i % n]);
        if (p)
        {
            hits += 1;
        }
    }
    auto t1 = std::chrono::steady_clock::now();

    double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    std::cout << "[bench_bible_find] " << (ns / iters) << " ns/op ("
              << iters << " lookups, hits=" << hits << ")\n";
}

} // namespace dollama

int main()
{
    bool ok = true;
    ok = dollama::test_bible_put_find()     && ok;
    ok = dollama::test_bible_overwrite()    && ok;
    ok = dollama::test_compose_order()      && ok;
    ok = dollama::test_compose_negative()   && ok;
    ok = dollama::test_compose_seed()       && ok;
    ok = dollama::test_isolation_tag()      && ok;
    ok = dollama::test_compose_empty_skip() && ok;
    ok = dollama::test_quality_negatives()  && ok;
    ok = dollama::test_digits_default()     && ok;

    // ベンチ (PASS/FAIL 判定には含めない。数値出力のみ)
    dollama::bench_compose_prompt();
    dollama::bench_bible_find();

    if (!ok)
    {
        std::cerr << "[test_character] FAILED\n";
        return 1;
    }
    std::cout << "[test_character] ALL PASSED\n";
    return 0;
}

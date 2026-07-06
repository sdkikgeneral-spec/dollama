// FastConfig (FAST モードのフラグ枠・Pkg G-0b) 単体テスト。
//
// 純 cpp (OV/CUDA 非依存) ゆえ開発機でも SKIP なし実走する。fast_config.hpp の
// FastConfig 既定値と resolve_fast_config の env / CLI 合成・fp8→fast 含意を検証する。
//
// 全テスト通過時は "[test_fast_config] ALL PASSED" を出力して return 0。
// 失敗時は std::cerr に出力して return 1。

#include <cstdlib>
#include <iostream>
#include <string>

#include "server/fast_config.hpp" // FastConfig / resolve_fast_config / fast_env_true

namespace
{

int g_fail = 0;

void check(bool cond, const std::string& msg)
{
    if (!cond)
    {
        std::cerr << "[test_fast_config] FAIL: " << msg << std::endl;
        ++g_fail;
    }
}

// 環境変数を設定する (val が空文字なら「存在するが空」= 未設定相当扱い)。
//   fast_env_true は「存在かつ非空」で true とするため、空設定で off に戻せる。
void set_env(const char* name, const char* val)
{
#if defined(_WIN32)
    _putenv_s(name, val);
#else
    setenv(name, val, 1);
#endif
}

// テスト前に FAST 系 env を必ず空へ戻す (プロセス内で env が持続するため)。
void clear_fast_env()
{
    set_env("DOLLAMA_FAST", "");
    set_env("DOLLAMA_FP8", "");
}

} // namespace

int main()
{
    using dollama::FastConfig;
    using dollama::resolve_fast_config;

    // --- 1) FastConfig の既定は全 off ---
    {
        FastConfig c;
        check(!c.fast, "default fast");
        check(!c.fp8, "default fp8");
        check(!c.attn_fast, "default attn_fast");
        check(!c.batch2, "default batch2");
        check(!c.graphs, "default graphs");
        check(!c.epilogue, "default epilogue");
    }

    // --- 2) フラグ無し (env なし・CLI なし) = 全 off ---
    {
        clear_fast_env();
        FastConfig c = resolve_fast_config();
        check(!c.fast, "no-flag fast off");
        check(!c.fp8, "no-flag fp8 off");
        check(!c.attn_fast && !c.batch2 && !c.graphs && !c.epilogue,
              "no-flag sub off");
    }

    // --- 3) env DOLLAMA_FAST=1 が fast に落ちる (fp8 は立たない) ---
    {
        clear_fast_env();
        set_env("DOLLAMA_FAST", "1");
        FastConfig c = resolve_fast_config();
        check(c.fast, "env DOLLAMA_FAST -> fast");
        check(!c.fp8, "env DOLLAMA_FAST keeps fp8 off");
    }

    // --- 4) env DOLLAMA_FP8=1 が fp8 に落ち、fast を含意する ---
    {
        clear_fast_env();
        set_env("DOLLAMA_FP8", "1");
        FastConfig c = resolve_fast_config();
        check(c.fp8, "env DOLLAMA_FP8 -> fp8");
        check(c.fast, "env DOLLAMA_FP8 implies fast");
    }

    // --- 5) CLI --fp8 (base.fp8=true) が fast=true を含意する (env なし) ---
    {
        clear_fast_env();
        FastConfig base;
        base.fp8 = true; // --fp8 相当
        FastConfig c = resolve_fast_config(base);
        check(c.fp8, "cli --fp8 -> fp8");
        check(c.fast, "cli --fp8 implies fast");
    }

    // --- 6) CLI --fast (base.fast=true) 単独は fp8 を立てない ---
    {
        clear_fast_env();
        FastConfig base;
        base.fast = true; // --fast 相当
        FastConfig c = resolve_fast_config(base);
        check(c.fast, "cli --fast -> fast");
        check(!c.fp8, "cli --fast keeps fp8 off");
    }

    // --- 7) env が空文字なら off (存在かつ非空のみ true) ---
    {
        clear_fast_env();
        set_env("DOLLAMA_FAST", "");
        set_env("DOLLAMA_FP8", "");
        FastConfig c = resolve_fast_config();
        check(!c.fast, "empty env fast off");
        check(!c.fp8, "empty env fp8 off");
    }

    // 後片付け (他テストへの env 漏れ防止)。
    clear_fast_env();

    if (g_fail == 0)
    {
        std::cout << "[test_fast_config] ALL PASSED" << std::endl;
        return 0;
    }
    std::cerr << "[test_fast_config] " << g_fail << " check(s) FAILED" << std::endl;
    return 1;
}

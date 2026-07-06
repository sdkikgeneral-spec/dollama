// FastConfig — FAST モード (自作カーネル高速化) のフラグ枠 (Pkg G-0b・骨組み)。
//
// ----------------------------------------------------------------
// 設計意図 (非交渉):
//   default を byte-for-byte 無改変に保つのが芯。この枠は fast の実挙動を一切足さず、
//   フラグを拡散経路 (BackendConfig → make_backend → SDXLBackend → IDiffusionRunner →
//   DiffusionPipeline) へ「運ぶだけ」。fast の実分岐 (if (cfg.attn_fast) …) は後続 Pkg
//   (G-3k-wire 等) が初めて入れる。全 default false = 現行挙動そのまま。
//
//   純 cpp・CUDA/OpenVINO 非依存 (std のみ)。よって CUDA 非対応 TU からも常に include できる。
// ----------------------------------------------------------------
#pragma once

#include <cstdlib>
#include <string>

namespace dollama
{

// FAST モードのフラグ集合。
//   fast      : 精度ゼロコストの高速化群を有効化する総合スイッチ (`--fast` / DOLLAMA_FAST)。
//   fp8       : FP8 Tensor Core 経路 (`--fp8` / DOLLAMA_FP8)。fast を含意する (fp8 単独無効)。
//   attn_fast : #4 attention FlashAttn 級 (G-3k)。この Pkg では未配線 (常に既定)。
//   batch2    : #2 CFG batch=2 (G-2k)。      同上。
//   graphs    : #1 GPU 常駐 + CUDA Graphs (G-1k)。同上。
//   epilogue  : #5 epilogue 融合 (G-4k)。    同上。
//   全 default false = 現行 (default) 挙動。
struct FastConfig
{
    bool fast     = false;
    bool fp8      = false;
    bool attn_fast = false;
    bool batch2    = false;
    bool graphs    = false;
    bool epilogue  = false;
};

// 環境変数が「存在かつ非空」で true か (resolve_path と同じ std::getenv 作法)。
inline bool fast_env_true(const char* name)
{
    // MSVC は std::getenv に C4996 (非推奨) を出すが、ここは読み取り専用で
    // スレッド前 (起動時 1 回) のため安全。局所的に警告を抑止する (resolve_path と同流儀)。
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
    if (const char* v = std::getenv(name))
    {
        if (v[0] != '\0')
        {
            return true;
        }
    }
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
    return false;
}

// env (DOLLAMA_FAST / DOLLAMA_FP8) と CLI 由来 base を OR 合成し、fp8→fast 含意を適用する。
//   base       : CLI (--fast / --fp8) 由来の初期値。指定が無ければ既定 (全 off)。
//   戻り値     : env で立ったフラグを OR し、fp8 が立っていれば fast も強制した FastConfig。
//   attn_fast/batch2/graphs/epilogue は base をそのまま透過する (この Pkg では env で触らない)。
inline FastConfig resolve_fast_config(const FastConfig& base = FastConfig{})
{
    FastConfig cfg = base;

    // env を OR で重ねる (CLI で立っていれば env が無くても維持)。
    if (fast_env_true("DOLLAMA_FAST"))
    {
        cfg.fast = true;
    }
    if (fast_env_true("DOLLAMA_FP8"))
    {
        cfg.fp8 = true;
    }

    // fp8 は fast を含意する (FP8 単独は無効)。
    if (cfg.fp8)
    {
        cfg.fast = true;
    }

    return cfg;
}

} // namespace dollama

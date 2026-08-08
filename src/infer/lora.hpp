// ランタイム LoRA (L-2) — kohya 命名 LoRA を diffusers 命名 base UNet へ写像する
// 純 host ロジック (CUDA 非依存・ヘッダオンリー)。
//
// L-1 offline merge (scripts/dollma_merge_lora.py) の写像・scale 計算の C++ 移植であり、
// 数値の正典は Python 側にある:
//
//     W' = W + scale * (B @ A),  scale = strength * (alpha / rank),  rank = A.shape[0]
//
//   - LoRA は kohya 命名:
//       lora_unet_<flattened>.lora_down.weight  = A,  shape [rank, in(*kh*kw)]
//       lora_unet_<flattened>.lora_up.weight    = B,  shape [out, rank(,1,1)]
//       lora_unet_<flattened>.alpha             = スカラ (無ければ alpha=rank)
//   - base UNet は diffusers 命名 (UNet2DConditionModel.state_dict() の key)。
//   - kohya module → base key の解決は「base 全 key のドット除去索引」で直接一致させる
//     (Python build_base_dotless_index / resolve_base_key と同一。base が真実源)。
//     未解決 module は throw (命名ズレの早期検出・黙って落とさない)。
//   - lora_te1_* / lora_te2_* (テキストエンコーダ) はスコープ外 → skip して数える。
//
// 本ヘッダはデバイス適用 (unet_apply_loras) の入力 DTO (LoraModule) までを担当し、
// GEMM / 加算は unet.cu 側 (launch_gemm_fp16 + launch_add) が行う。
//
// 注意: .cu TU のホスト部 (MSVC /std:c++14) からも include されるため C++14 互換に保つこと。
#pragma once

#include <cstdint>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "io/safetensors.hpp"

namespace dollama
{

// ----------------------------------------------------------------
// FP32 ↔ FP16 (host・round-to-nearest-even)。
// numpy .astype(float16) と同じ RNE 丸め。fp32 LoRA を F16 化する経路と
// F16 alpha スカラの読み出しに使う (CUDA ヘッダ非依存で自前実装)。
// ----------------------------------------------------------------
inline uint16_t lora_f32_to_f16(float f)
{
    uint32_t x = 0;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t       mant = x & 0x007fffffu;
    const uint32_t e8   = (x >> 23) & 0xffu;

    if (e8 == 0xffu)
    {
        // Inf / NaN (NaN は quiet bit を立てて保存)
        return static_cast<uint16_t>(sign | 0x7c00u | (mant != 0 ? 0x200u : 0u));
    }

    const int32_t exp = static_cast<int32_t>(e8) - 127 + 15;
    if (exp >= 31)
    {
        // オーバーフロー → Inf
        return static_cast<uint16_t>(sign | 0x7c00u);
    }
    if (exp <= 0)
    {
        // サブノーマル / ゼロ
        if (exp < -10)
        {
            return static_cast<uint16_t>(sign);
        }
        mant |= 0x00800000u;  // implicit 1 を立てる
        const int      shift     = 14 - exp;  // 24bit 仮数 → 10bit へ
        uint32_t       half_mant = mant >> shift;
        const uint32_t rem       = mant & ((1u << shift) - 1u);
        const uint32_t halfway   = 1u << (shift - 1);
        if (rem > halfway || (rem == halfway && (half_mant & 1u) != 0))
        {
            ++half_mant;  // RNE
        }
        return static_cast<uint16_t>(sign | half_mant);
    }

    // 正規化数: 落とす下位 13bit を RNE 丸め (仮数繰り上がりは指数へ自然に伝播)
    uint32_t       half = sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13);
    const uint32_t rem  = mant & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (half & 1u) != 0))
    {
        ++half;
    }
    return static_cast<uint16_t>(half);
}

inline float lora_f16_to_f32(uint16_t h)
{
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exp  = (h >> 10) & 0x1fu;
    const uint32_t mant = h & 0x3ffu;
    uint32_t       x    = 0;
    if (exp == 0)
    {
        if (mant == 0)
        {
            x = sign;  // ±0
        }
        else
        {
            // サブノーマル → 正規化
            int      e = -1;
            uint32_t m = mant;
            do
            {
                m <<= 1;
                ++e;
            } while ((m & 0x400u) == 0);
            x = sign | (static_cast<uint32_t>(127 - 15 - e) << 23) | ((m & 0x3ffu) << 13);
        }
    }
    else if (exp == 31)
    {
        x = sign | 0x7f800000u | (mant << 13);  // Inf / NaN
    }
    else
    {
        x = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f = 0.0f;
    std::memcpy(&f, &x, 4);
    return f;
}

// ----------------------------------------------------------------
// LoraModule — デバイス適用 1 件分の DTO。
//   a_fp16 : A = lora_down [rank, in_flat] (FP16 生ビット・row-major・4D は 2D へ畳み済み)
//   b_fp16 : B = lora_up   [out_dim, rank] (同上)
//   scale  : strength * (alpha / rank)。delta = scale * (B @ A)。
// ----------------------------------------------------------------
struct LoraModule
{
    std::string           base_key;  // base UNet の diffusers .weight key
    std::vector<uint16_t> a_fp16;    // [rank, in_flat]
    std::vector<uint16_t> b_fp16;    // [out_dim, rank]
    int                   rank    = 0;
    int                   in_flat = 0;
    int                   out_dim = 0;
    float                 scale   = 1.0f;
};

// load_lora_modules の副次情報 (skip 件数等)。テスト・ログが消費する。
struct LoraLoadReport
{
    size_t te_skipped = 0;  // lora_te1_/lora_te2_/lora_te_ 由来 key (スコープ外 skip)
    size_t other_keys = 0;  // 解釈できない key
    size_t incomplete = 0;  // down/up 片欠けの不完全 module (skip)
    size_t modules    = 0;  // 解決・積載できた module 数
};

namespace lora_detail
{

// base 全 key の「ドット除去名 → 元 key」索引 (Python build_base_dotless_index 相当)。
inline std::map<std::string, std::string> build_base_dotless_index(
    const std::vector<std::string>& base_keys)
{
    std::map<std::string, std::string> idx;
    for (size_t i = 0; i < base_keys.size(); ++i)
    {
        std::string dotless = base_keys[i];
        for (size_t j = 0; j < dotless.size(); ++j)
        {
            if (dotless[j] == '.')
            {
                dotless[j] = '_';
            }
        }
        idx[dotless] = base_keys[i];
    }
    return idx;
}

// 接尾辞判定 (C++14 なので ends_with 自前)。
inline bool ends_with(const std::string& s, const std::string& suf)
{
    return s.size() >= suf.size() &&
           s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

inline bool starts_with(const std::string& s, const std::string& pre)
{
    return s.size() >= pre.size() && s.compare(0, pre.size(), pre) == 0;
}

// module 単位に束ねる中間表現 (down/up/alpha の key 名を保持し、実データは遅延参照)。
struct LoraGroup
{
    std::string down_key;   // 空 = 未出現
    std::string up_key;
    bool        has_alpha = false;
    float       alpha     = 0.0f;
};

// LoRA テンソルからスカラ alpha を読む (F32 / F16 のみ対応)。
inline float read_alpha_scalar(const SafeTensors& lora, const std::string& key)
{
    size_t         nbytes = 0;
    const uint8_t* p      = lora.tensor_bytes(key, nbytes);
    const StDtype  dt     = lora.dtype(key);
    if (dt == StDtype::F32 && nbytes >= 4)
    {
        float v = 0.0f;
        std::memcpy(&v, p, 4);
        return v;
    }
    if (dt == StDtype::F16 && nbytes >= 2)
    {
        uint16_t h = 0;
        std::memcpy(&h, p, 2);
        return lora_f16_to_f32(h);
    }
    throw std::runtime_error("lora: alpha '" + key + "' must be F32/F16 scalar (dtype=" +
                             std::string(st_dtype_name(dt)) + ")");
}

// LoRA テンソル (F16 or F32) を FP16 生ビット列へ読み出す。
inline std::vector<uint16_t> read_tensor_f16(const SafeTensors& lora, const std::string& key)
{
    size_t         nbytes = 0;
    const uint8_t* p      = lora.tensor_bytes(key, nbytes);
    const StDtype  dt     = lora.dtype(key);
    if (dt == StDtype::F16)
    {
        std::vector<uint16_t> out(nbytes / 2);
        std::memcpy(out.data(), p, nbytes);
        return out;
    }
    if (dt == StDtype::F32)
    {
        // fp32 LoRA は F16 化して積む (デバイス GEMM は FP16 入力・FP32 蓄積)
        const size_t          n = nbytes / 4;
        std::vector<uint16_t> out(n);
        for (size_t i = 0; i < n; ++i)
        {
            float v = 0.0f;
            std::memcpy(&v, p + (i << 2), 4);
            out[i] = lora_f32_to_f16(v);
        }
        return out;
    }
    throw std::runtime_error("lora: tensor '" + key + "' must be F16/F32 (dtype=" +
                             std::string(st_dtype_name(dt)) + ")");
}

// shape 先頭以降の積 (4D conv LoRA を 2D へ畳むときの残り次元)。
inline size_t trailing_numel(const std::vector<size_t>& shape)
{
    size_t n = 1;
    for (size_t i = 1; i < shape.size(); ++i)
    {
        n *= shape[i];
    }
    return n;
}

} // namespace lora_detail

// ----------------------------------------------------------------
// load_lora_modules — LoRA safetensors を base UNet へ写像し LoraModule 列を返す。
//
//   base     : base UNet safetensors (diffusers 命名・FP16 常駐対象)
//   lora     : LoRA safetensors (kohya 命名)
//   strength : 適用強度。scale = strength * (alpha / rank)。
//   report   : skip 件数等の副次情報 (nullptr 可)
//
// 例外:
//   - kohya module が base のどの key にも解決できない → std::runtime_error
//     (Python 正典と同じ「黙って落とさない」方針)
//   - A/B と base の shape 不整合 / 非対応 dtype → std::runtime_error
// te1/te2 は warn 相当 (report にカウント) で skip。down/up 片欠けも skip (カウント)。
// ----------------------------------------------------------------
inline std::vector<LoraModule> load_lora_modules(const SafeTensors& base,
                                                 const SafeTensors& lora,
                                                 float              strength,
                                                 LoraLoadReport*    report = nullptr)
{
    using namespace lora_detail;

    LoraLoadReport rep;

    // --- (1) kohya key を module 単位に束ねる (Python group_lora 相当) ---
    std::map<std::string, LoraGroup> groups;
    {
        const std::vector<std::string> keys = lora.names();
        for (size_t i = 0; i < keys.size(); ++i)
        {
            const std::string& k = keys[i];
            if (starts_with(k, "lora_te1_") || starts_with(k, "lora_te2_") ||
                starts_with(k, "lora_te_"))
            {
                ++rep.te_skipped;  // テキストエンコーダ LoRA はスコープ外
                continue;
            }
            if (!starts_with(k, "lora_unet_"))
            {
                ++rep.other_keys;
                continue;
            }
            const std::string body = k.substr(std::string("lora_unet_").size());

            // 接尾辞で down/up/alpha を判定 (ドット命名と完全平坦命名の両対応)
            if (ends_with(body, ".lora_down.weight"))
            {
                groups[body.substr(0, body.size() - 17)].down_key = k;
            }
            else if (ends_with(body, ".lora_up.weight"))
            {
                groups[body.substr(0, body.size() - 15)].up_key = k;
            }
            else if (ends_with(body, ".alpha"))
            {
                LoraGroup& g = groups[body.substr(0, body.size() - 6)];
                g.alpha      = read_alpha_scalar(lora, k);
                g.has_alpha  = true;
            }
            else if (ends_with(body, "_lora_down_weight"))
            {
                groups[body.substr(0, body.size() - 17)].down_key = k;
            }
            else if (ends_with(body, "_lora_up_weight"))
            {
                groups[body.substr(0, body.size() - 15)].up_key = k;
            }
            else if (ends_with(body, "_alpha"))
            {
                LoraGroup& g = groups[body.substr(0, body.size() - 6)];
                g.alpha      = read_alpha_scalar(lora, k);
                g.has_alpha  = true;
            }
            else
            {
                ++rep.other_keys;
            }
        }
    }

    // --- (2) base のドット除去索引 (base が真実源) ---
    const std::map<std::string, std::string> base_index =
        build_base_dotless_index(base.names());

    // --- (3) module ごとに解決・積載 ---
    std::vector<LoraModule> mods;
    for (std::map<std::string, LoraGroup>::const_iterator it = groups.begin();
         it != groups.end(); ++it)
    {
        const std::string& mod = it->first;
        const LoraGroup&   g   = it->second;

        if (g.down_key.empty() || g.up_key.empty())
        {
            ++rep.incomplete;  // 不完全 LoRA (down/up 片欠け) は skip
            continue;
        }

        // kohya 平坦名 + "_weight" = base key のドット除去名
        std::map<std::string, std::string>::const_iterator hit =
            base_index.find(mod + "_weight");
        if (hit == base_index.end())
        {
            // base に存在しない → 命名ズレの早期検出 (黙って落とさない)
            throw std::runtime_error(
                "lora: module 'lora_unet_" + mod +
                "' does not resolve to any base key (naming mismatch)");
        }
        const std::string& base_key = hit->second;

        // A/B の shape (4D conv LoRA は 2D へ畳む)
        const std::vector<size_t>& a_shape = lora.shape(g.down_key);
        const std::vector<size_t>& b_shape = lora.shape(g.up_key);
        if (a_shape.empty() || b_shape.empty())
        {
            throw std::runtime_error("lora: module 'lora_unet_" + mod + "' has empty A/B shape");
        }
        const size_t rank    = a_shape[0];
        const size_t in_flat = trailing_numel(a_shape);
        const size_t out_dim = b_shape[0];
        if (trailing_numel(b_shape) != rank)
        {
            throw std::runtime_error(
                "lora: module 'lora_unet_" + mod + "' B trailing dims (" +
                std::to_string(trailing_numel(b_shape)) + ") != rank (" +
                std::to_string(rank) + ")");
        }
        if (rank == 0 || in_flat == 0 || out_dim == 0)
        {
            throw std::runtime_error("lora: module 'lora_unet_" + mod + "' has zero dim");
        }

        // base との整合: 2D (Linear) は shape 完全一致、4D (Conv) 等は numel 一致
        //   (Python merge_one と同じ判定)
        const std::vector<size_t>& w_shape = base.shape(base_key);
        size_t                     w_numel = 1;
        for (size_t d = 0; d < w_shape.size(); ++d)
        {
            w_numel *= w_shape[d];
        }
        if (w_shape.size() == 2)
        {
            if (w_shape[0] != out_dim || w_shape[1] != in_flat)
            {
                throw std::runtime_error(
                    "lora: module 'lora_unet_" + mod + "' -> '" + base_key +
                    "' shape mismatch (Linear): base [" + std::to_string(w_shape[0]) + "," +
                    std::to_string(w_shape[1]) + "] vs B@A [" + std::to_string(out_dim) +
                    "," + std::to_string(in_flat) + "]");
            }
        }
        else if (w_numel != out_dim * in_flat)
        {
            throw std::runtime_error(
                "lora: module 'lora_unet_" + mod + "' -> '" + base_key +
                "' numel mismatch (Conv): base " + std::to_string(w_numel) +
                " vs B@A " + std::to_string(out_dim * in_flat));
        }

        // scale = strength * (alpha / rank)。alpha 欠如時 alpha=rank → scale=strength。
        const float alpha = g.has_alpha ? g.alpha : static_cast<float>(rank);
        const float scale = strength * (alpha / static_cast<float>(rank));

        LoraModule m;
        m.base_key = base_key;
        m.a_fp16   = read_tensor_f16(lora, g.down_key);
        m.b_fp16   = read_tensor_f16(lora, g.up_key);
        m.rank     = static_cast<int>(rank);
        m.in_flat  = static_cast<int>(in_flat);
        m.out_dim  = static_cast<int>(out_dim);
        m.scale    = scale;
        mods.push_back(std::move(m));
        ++rep.modules;
    }

    if (report != nullptr)
    {
        *report = rep;
    }
    return mods;
}

} // namespace dollama

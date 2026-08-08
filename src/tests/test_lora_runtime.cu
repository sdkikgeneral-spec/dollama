// ランタイム LoRA (L-2) 重みレベル検証テスト
// HAVE_CUDA 未定義時は [SKIP] で return 0。SDXL 実走は不要 (合成 base + LoRA)。
//
// ゲート:
//   [1] host 写像 (lora.hpp): module 数 / te skip / 不完全 skip / scale 値 /
//       未解決 module throw / alpha 欠如時 scale=strength
//   [2] parity: unet_apply_loras 適用後の常駐重み == CPU fp32 参照 W + scale*(B@A)
//       (L-1 merge_one と同式) を fp16 tol で全 touched key 比較
//   [3] revert bit-exact: apply → unet_clear_loras 後、常駐重みが base と memcmp 完全一致
//   [4] stack 加算性: LoRA×2 適用 == W + Σ scale_i*(B_i@A_i) (fp16 tol)、clear 後 bit-exact
//
// 合成データは LORA_TMP_DIR (ビルドディレクトリ配下) に safetensors として書き出し、
// 本番経路 (SafeTensors ファイルロード → load_lora_modules → unet_apply_loras) を
// そのまま通す (in-memory ショートカットで経路を偽らない)。
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef HAVE_CUDA
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "infer/lora.hpp"
#include "infer/unet.cuh"
#include "io/safetensors.hpp"
#include "kernels/utils.cuh"
#endif

namespace dollama
{

#ifdef HAVE_CUDA

// ----------------------------------------------------------------
// 合成 safetensors 書き出し (F16/F32 混在対応の最小ライタ)
// ----------------------------------------------------------------
struct StWriteRaw
{
    std::string          name;
    std::string          dtype;  // "F16" / "F32"
    std::vector<size_t>  shape;  // [] = スカラ
    std::vector<uint8_t> bytes;  // 生バイト
};

static void write_safetensors_raw(const std::string& path,
                                  const std::vector<StWriteRaw>& tensors)
{
    std::vector<uint8_t> body;
    std::string          json = "{";
    bool                 first = true;
    size_t               cursor = 0;
    for (const auto& t : tensors)
    {
        const size_t begin = cursor;
        const size_t end   = begin + t.bytes.size();
        body.insert(body.end(), t.bytes.begin(), t.bytes.end());
        cursor = end;

        if (!first)
        {
            json += ",";
        }
        first = false;
        json += "\"" + t.name + "\":{\"dtype\":\"" + t.dtype + "\",\"shape\":[";
        for (size_t i = 0; i < t.shape.size(); ++i)
        {
            if (i)
            {
                json += ",";
            }
            json += std::to_string(t.shape[i]);
        }
        json += "],\"data_offsets\":[" + std::to_string(begin) + ","
              + std::to_string(end) + "]}";
    }
    json += "}";

    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f)
    {
        throw std::runtime_error("write_safetensors_raw: cannot open: " + path);
    }
    const uint64_t n = static_cast<uint64_t>(json.size());
    uint8_t        lenbuf[8];
    for (int i = 0; i < 8; ++i)
    {
        lenbuf[i] = static_cast<uint8_t>((n >> (8 * i)) & 0xff);
    }
    f.write(reinterpret_cast<const char*>(lenbuf), 8);
    f.write(json.data(), static_cast<std::streamsize>(json.size()));
    if (!body.empty())
    {
        f.write(reinterpret_cast<const char*>(body.data()),
                static_cast<std::streamsize>(body.size()));
    }
    if (!f)
    {
        throw std::runtime_error("write_safetensors_raw: write failed: " + path);
    }
}

// FP16 乱数テンソル (生ビット列)。
static std::vector<uint8_t> make_f16_bytes(size_t n, unsigned seed, float scale)
{
    std::mt19937                    rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<uint8_t>            out(n * 2);
    uint16_t*                       p = reinterpret_cast<uint16_t*>(out.data());
    for (size_t i = 0; i < n; ++i)
    {
        const __half h = __float2half(dist(rng) * scale);
        std::memcpy(&p[i], &h, 2);
    }
    return out;
}

// F32 スカラ (alpha 用)。
static std::vector<uint8_t> f32_scalar_bytes(float v)
{
    std::vector<uint8_t> out(4);
    std::memcpy(out.data(), &v, 4);
    return out;
}

static StWriteRaw f16_tensor(const std::string& name, const std::vector<size_t>& shape,
                             unsigned seed, float scale)
{
    size_t n = 1;
    for (size_t d : shape)
    {
        n *= d;
    }
    return StWriteRaw{name, "F16", shape, make_f16_bytes(n, seed, scale)};
}

static StWriteRaw f32_alpha(const std::string& name, float v)
{
    return StWriteRaw{name, "F32", {}, f32_scalar_bytes(v)};
}

// ----------------------------------------------------------------
// CPU fp32 参照 (L-1 merge_one と同式): out = fp32(W) + scale * (fp32(B) @ fp32(A))
// ----------------------------------------------------------------
static std::vector<float> f16_bytes_to_f32(const uint8_t* p, size_t n)
{
    std::vector<float> out(n);
    const __half*      h = reinterpret_cast<const __half*>(p);
    for (size_t i = 0; i < n; ++i)
    {
        out[i] = __half2float(h[i]);
    }
    return out;
}

// delta += scale * (B @ A) を fp32 で加算 (B [out,rank], A [rank,in])
static void add_delta_ref(std::vector<float>& w, const std::vector<float>& b,
                          const std::vector<float>& a, int out_dim, int in_flat, int rank,
                          float scale)
{
    for (int o = 0; o < out_dim; ++o)
    {
        for (int i = 0; i < in_flat; ++i)
        {
            float acc = 0.0f;
            for (int k = 0; k < rank; ++k)
            {
                acc += b[static_cast<size_t>(o) * rank + k] *
                       a[static_cast<size_t>(k) * in_flat + i];
            }
            w[static_cast<size_t>(o) * in_flat + i] += scale * acc;
        }
    }
}

// GPU 常駐重みを D2H で回収して fp32 化。
static std::vector<float> fetch_device_f32(UnetWeightsHandle h, const std::string& name,
                                           size_t numel)
{
    const __half*       d = unet_weight_device_ptr(h, name);
    std::vector<__half> host(numel);
    CUDA_CHECK(cudaMemcpy(host.data(), d, numel * sizeof(__half), cudaMemcpyDeviceToHost));
    std::vector<float> out(numel);
    for (size_t i = 0; i < numel; ++i)
    {
        out[i] = __half2float(host[i]);
    }
    return out;
}

// GPU 常駐重みの生バイトを D2H 回収 (bit-exact 比較用)。
static std::vector<uint8_t> fetch_device_bytes(UnetWeightsHandle h, const std::string& name,
                                               size_t nbytes)
{
    const __half*        d = unet_weight_device_ptr(h, name);
    std::vector<uint8_t> out(nbytes);
    CUDA_CHECK(cudaMemcpy(out.data(), d, nbytes, cudaMemcpyDeviceToHost));
    return out;
}

// fp16 tol 比較 (atol + rtol)。max_abs_err と bad 数を返す。
struct CmpResult
{
    float  max_abs = 0.0f;
    size_t bad     = 0;
};

static CmpResult compare_f16_tol(const std::vector<float>& got, const std::vector<float>& ref,
                                 float atol, float rtol)
{
    CmpResult r;
    for (size_t i = 0; i < got.size(); ++i)
    {
        const float diff = std::fabs(got[i] - ref[i]);
        if (diff > r.max_abs)
        {
            r.max_abs = diff;
        }
        if (diff > atol + rtol * std::fabs(ref[i]))
        {
            ++r.bad;
        }
    }
    return r;
}

// ----------------------------------------------------------------
// 合成 base / LoRA の定義 (SDXL の代表 module 名を模す)
// ----------------------------------------------------------------
static const char* kKeyToQ  = "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight";
static const char* kKeyFF   = "down_blocks.0.attentions.0.transformer_blocks.0.ff.net.0.proj.weight";
static const char* kKeyConv = "down_blocks.0.resnets.0.conv1.weight";
static const char* kKeyUn   = "mid_block.resnets.0.conv2.weight";  // LoRA が触れない key

static const char* kModToQ  = "down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q";
static const char* kModFF   = "down_blocks_0_attentions_0_transformer_blocks_0_ff_net_0_proj";
static const char* kModConv = "down_blocks_0_resnets_0_conv1";

// to_q [64,48] / ff [96,32] / conv [24,16,3,3] / untouched [16,16,3,3]
static const size_t kToQOut = 64, kToQIn = 48;
static const size_t kFFOut = 96, kFFIn = 32;
static const size_t kConvOut = 24, kConvIn = 16, kConvK = 3;
static const size_t kConvInFlat = kConvIn * kConvK * kConvK;  // 144

int run()
{
    const std::string dir = LORA_TMP_DIR;
    const std::string base_path  = dir + "/lora_rt_base.safetensors";
    const std::string lora1_path = dir + "/lora_rt_lora1.safetensors";
    const std::string lora2_path = dir + "/lora_rt_lora2.safetensors";
    const std::string bad_path   = dir + "/lora_rt_bad.safetensors";

    // --- 合成 base (F16, diffusers 命名) ---
    std::vector<StWriteRaw> base_ts;
    base_ts.push_back(f16_tensor(kKeyToQ,  {kToQOut, kToQIn}, 100, 0.5f));
    base_ts.push_back(f16_tensor(kKeyFF,   {kFFOut, kFFIn},   101, 0.5f));
    base_ts.push_back(f16_tensor(kKeyConv, {kConvOut, kConvIn, kConvK, kConvK}, 102, 0.5f));
    base_ts.push_back(f16_tensor(kKeyUn,   {16, 16, 3, 3},    103, 0.5f));
    write_safetensors_raw(base_path, base_ts);

    // --- LoRA1 (kohya 命名): to_q rank4 alpha4 / ff rank8 alpha4 / conv rank4 alpha2 (4D) ---
    //     + te1 3 key (skip 対象) + 不完全 module 1 件 (down のみ)
    std::vector<StWriteRaw> l1;
    l1.push_back(f16_tensor(std::string("lora_unet_") + kModToQ + ".lora_down.weight",
                            {4, kToQIn}, 200, 0.05f));
    l1.push_back(f16_tensor(std::string("lora_unet_") + kModToQ + ".lora_up.weight",
                            {kToQOut, 4}, 201, 0.05f));
    l1.push_back(f32_alpha(std::string("lora_unet_") + kModToQ + ".alpha", 4.0f));
    l1.push_back(f16_tensor(std::string("lora_unet_") + kModFF + ".lora_down.weight",
                            {8, kFFIn}, 202, 0.05f));
    l1.push_back(f16_tensor(std::string("lora_unet_") + kModFF + ".lora_up.weight",
                            {kFFOut, 8}, 203, 0.05f));
    l1.push_back(f32_alpha(std::string("lora_unet_") + kModFF + ".alpha", 4.0f));
    // conv LoRA は kohya 実物と同じ 4D shape (down [r,in,kh,kw] / up [out,r,1,1])
    l1.push_back(f16_tensor(std::string("lora_unet_") + kModConv + ".lora_down.weight",
                            {4, kConvIn, kConvK, kConvK}, 204, 0.05f));
    l1.push_back(f16_tensor(std::string("lora_unet_") + kModConv + ".lora_up.weight",
                            {kConvOut, 4, 1, 1}, 205, 0.05f));
    l1.push_back(f32_alpha(std::string("lora_unet_") + kModConv + ".alpha", 2.0f));
    // te1 (スコープ外 skip)
    l1.push_back(f16_tensor("lora_te1_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight",
                            {4, 32}, 206, 0.05f));
    l1.push_back(f16_tensor("lora_te1_text_model_encoder_layers_0_self_attn_q_proj.lora_up.weight",
                            {32, 4}, 207, 0.05f));
    l1.push_back(f32_alpha("lora_te1_text_model_encoder_layers_0_self_attn_q_proj.alpha", 4.0f));
    // 不完全 module (down のみ → skip)
    l1.push_back(f16_tensor(std::string("lora_unet_") + kModFF + "_broken.lora_down.weight",
                            {4, kFFIn}, 208, 0.05f));
    write_safetensors_raw(lora1_path, l1);

    // --- LoRA2: to_q rank2 (alpha なし → scale=strength) / conv rank4 alpha4 ---
    std::vector<StWriteRaw> l2;
    l2.push_back(f16_tensor(std::string("lora_unet_") + kModToQ + ".lora_down.weight",
                            {2, kToQIn}, 300, 0.05f));
    l2.push_back(f16_tensor(std::string("lora_unet_") + kModToQ + ".lora_up.weight",
                            {kToQOut, 2}, 301, 0.05f));
    l2.push_back(f16_tensor(std::string("lora_unet_") + kModConv + ".lora_down.weight",
                            {4, kConvIn, kConvK, kConvK}, 302, 0.05f));
    l2.push_back(f16_tensor(std::string("lora_unet_") + kModConv + ".lora_up.weight",
                            {kConvOut, 4, 1, 1}, 303, 0.05f));
    l2.push_back(f32_alpha(std::string("lora_unet_") + kModConv + ".alpha", 4.0f));
    write_safetensors_raw(lora2_path, l2);

    // --- 未解決 module (base に無い) → throw 検証用 ---
    std::vector<StWriteRaw> lb;
    lb.push_back(f16_tensor("lora_unet_nonexistent_module_xyz.lora_down.weight", {4, 16}, 400, 0.05f));
    lb.push_back(f16_tensor("lora_unet_nonexistent_module_xyz.lora_up.weight", {16, 4}, 401, 0.05f));
    write_safetensors_raw(bad_path, lb);

    // ============================================================
    // [1] host 写像 (lora.hpp)
    // ============================================================
    SafeTensors base_st(base_path);
    SafeTensors lora1_st(lora1_path);
    SafeTensors lora2_st(lora2_path);

    const float kStrength1 = 0.8f;
    LoraLoadReport rep1;
    std::vector<LoraModule> mods1 = load_lora_modules(base_st, lora1_st, kStrength1, &rep1);

    bool ok = true;
    if (mods1.size() != 3 || rep1.modules != 3)
    {
        std::cerr << "[FAIL] [1] module count: got " << mods1.size() << " (expect 3)\n";
        ok = false;
    }
    if (rep1.te_skipped != 3)
    {
        std::cerr << "[FAIL] [1] te_skipped: got " << rep1.te_skipped << " (expect 3)\n";
        ok = false;
    }
    if (rep1.incomplete != 1)
    {
        std::cerr << "[FAIL] [1] incomplete: got " << rep1.incomplete << " (expect 1)\n";
        ok = false;
    }
    // scale 検証: to_q = 0.8*(4/4)=0.8 / ff = 0.8*(4/8)=0.4 / conv = 0.8*(2/4)=0.4
    for (const LoraModule& m : mods1)
    {
        float expect = -1.0f;
        if (m.base_key == kKeyToQ)       { expect = 0.8f; }
        else if (m.base_key == kKeyFF)   { expect = 0.4f; }
        else if (m.base_key == kKeyConv) { expect = 0.4f; }
        if (expect < 0.0f || std::fabs(m.scale - expect) > 1e-6f)
        {
            std::cerr << "[FAIL] [1] scale: key=" << m.base_key << " got " << m.scale
                      << " expect " << expect << "\n";
            ok = false;
        }
    }
    // alpha 欠如 (lora2 to_q): scale = strength
    {
        LoraLoadReport rep2;
        std::vector<LoraModule> mods2 = load_lora_modules(base_st, lora2_st, 0.5f, &rep2);
        bool found = false;
        for (const LoraModule& m : mods2)
        {
            if (m.base_key == kKeyToQ)
            {
                found = true;
                if (std::fabs(m.scale - 0.5f) > 1e-6f)
                {
                    std::cerr << "[FAIL] [1] scale w/o alpha: got " << m.scale
                              << " expect 0.5\n";
                    ok = false;
                }
            }
        }
        if (!found)
        {
            std::cerr << "[FAIL] [1] lora2 to_q module not loaded\n";
            ok = false;
        }
    }
    // 未解決 module → throw
    {
        SafeTensors bad_st(bad_path);
        bool thrown = false;
        try
        {
            (void)load_lora_modules(base_st, bad_st, 1.0f, nullptr);
        }
        catch (const std::exception&)
        {
            thrown = true;
        }
        if (!thrown)
        {
            std::cerr << "[FAIL] [1] unresolved module did not throw\n";
            ok = false;
        }
    }
    std::cout << "[1] host mapping: modules=" << rep1.modules << " te_skipped=" << rep1.te_skipped
              << " incomplete=" << rep1.incomplete << " scale/throw checks "
              << (ok ? "OK" : "NG") << "\n";
    if (!ok)
    {
        return 1;
    }

    // ============================================================
    // [2] parity: 適用後の常駐重み == CPU fp32 参照 (fp16 tol)
    // ============================================================
    UnetWeightsHandle h = unet_weights_create(base_st);

    unet_apply_loras(h, mods1.data(), static_cast<int>(mods1.size()));

    // CPU 参照を構築
    struct KeyDim { const char* key; size_t out_d; size_t in_f; };
    const KeyDim dims[3] = {
        {kKeyToQ,  kToQOut,  kToQIn},
        {kKeyFF,   kFFOut,   kFFIn},
        {kKeyConv, kConvOut, kConvInFlat},
    };

    const float atol = 2e-3f;
    const float rtol = 1e-2f;
    float       parity_max = 0.0f;
    size_t      parity_bad = 0;
    for (const KeyDim& kd : dims)
    {
        size_t         nbytes = 0;
        const uint8_t* wbytes = base_st.tensor_bytes(kd.key, nbytes);
        const size_t   numel  = nbytes / 2;
        std::vector<float> ref = f16_bytes_to_f32(wbytes, numel);

        // 対応 module の A/B を fp32 化して delta 加算
        for (const LoraModule& m : mods1)
        {
            if (m.base_key != kd.key)
            {
                continue;
            }
            std::vector<float> a(m.a_fp16.size()), b(m.b_fp16.size());
            for (size_t i = 0; i < a.size(); ++i)
            {
                a[i] = lora_f16_to_f32(m.a_fp16[i]);
            }
            for (size_t i = 0; i < b.size(); ++i)
            {
                b[i] = lora_f16_to_f32(m.b_fp16[i]);
            }
            add_delta_ref(ref, b, a, m.out_dim, m.in_flat, m.rank, m.scale);
        }

        std::vector<float> got = fetch_device_f32(h, kd.key, numel);
        CmpResult          cr  = compare_f16_tol(got, ref, atol, rtol);
        if (cr.max_abs > parity_max)
        {
            parity_max = cr.max_abs;
        }
        parity_bad += cr.bad;
    }
    std::cout << "[2] parity: max_abs_err=" << parity_max << " bad=" << parity_bad
              << " (atol=" << atol << " rtol=" << rtol << ")\n";
    if (parity_bad != 0)
    {
        std::cerr << "[FAIL] [2] parity gate NG\n";
        unet_weights_destroy(h);
        return 1;
    }

    // ============================================================
    // [3] revert bit-exact: clear 後、base と memcmp 完全一致 (全 key)
    // ============================================================
    unet_clear_loras(h);
    size_t revert_ok = 0;
    for (const std::string& name : base_st.names())
    {
        size_t         nbytes = 0;
        const uint8_t* src    = base_st.tensor_bytes(name, nbytes);
        std::vector<uint8_t> got = fetch_device_bytes(h, name, nbytes);
        if (std::memcmp(got.data(), src, nbytes) != 0)
        {
            std::cerr << "[FAIL] [3] revert: '" << name << "' not bit-exact vs base\n";
            unet_weights_destroy(h);
            return 1;
        }
        ++revert_ok;
    }
    std::cout << "[3] revert bit-exact: " << revert_ok << "/" << base_st.names().size()
              << " keys memcmp equal\n";

    // ============================================================
    // [4] stack 加算性: LoRA1 + LoRA2 適用 == W + Σ delta (fp16 tol) → clear で bit-exact
    // ============================================================
    const float kStrength2 = 0.5f;
    LoraLoadReport rep2b;
    std::vector<LoraModule> mods2 = load_lora_modules(base_st, lora2_st, kStrength2, &rep2b);

    unet_apply_loras(h, mods1.data(), static_cast<int>(mods1.size()));
    unet_apply_loras(h, mods2.data(), static_cast<int>(mods2.size()));

    float  stack_max = 0.0f;
    size_t stack_bad = 0;
    for (const KeyDim& kd : dims)
    {
        size_t         nbytes = 0;
        const uint8_t* wbytes = base_st.tensor_bytes(kd.key, nbytes);
        const size_t   numel  = nbytes / 2;
        std::vector<float> ref = f16_bytes_to_f32(wbytes, numel);

        // Σ_i scale_i * (B_i @ A_i) (両 LoRA)
        for (int pass = 0; pass < 2; ++pass)
        {
            const std::vector<LoraModule>& mm = (pass == 0) ? mods1 : mods2;
            for (const LoraModule& m : mm)
            {
                if (m.base_key != kd.key)
                {
                    continue;
                }
                std::vector<float> a(m.a_fp16.size()), b(m.b_fp16.size());
                for (size_t i = 0; i < a.size(); ++i)
                {
                    a[i] = lora_f16_to_f32(m.a_fp16[i]);
                }
                for (size_t i = 0; i < b.size(); ++i)
                {
                    b[i] = lora_f16_to_f32(m.b_fp16[i]);
                }
                add_delta_ref(ref, b, a, m.out_dim, m.in_flat, m.rank, m.scale);
            }
        }

        std::vector<float> got = fetch_device_f32(h, kd.key, numel);
        // stack は fp16 加算が 2 回挟まる (apply ごとに W へ丸め) ため tol を 1 段緩める
        CmpResult cr = compare_f16_tol(got, ref, atol * 2.0f, rtol);
        if (cr.max_abs > stack_max)
        {
            stack_max = cr.max_abs;
        }
        stack_bad += cr.bad;
    }
    std::cout << "[4] stack: max_abs_err=" << stack_max << " bad=" << stack_bad
              << " (atol=" << atol * 2.0f << " rtol=" << rtol << ")\n";
    if (stack_bad != 0)
    {
        std::cerr << "[FAIL] [4] stack gate NG\n";
        unet_weights_destroy(h);
        return 1;
    }

    // stack 後の clear も bit-exact
    unet_clear_loras(h);
    for (const std::string& name : base_st.names())
    {
        size_t         nbytes = 0;
        const uint8_t* src    = base_st.tensor_bytes(name, nbytes);
        std::vector<uint8_t> got = fetch_device_bytes(h, name, nbytes);
        if (std::memcmp(got.data(), src, nbytes) != 0)
        {
            std::cerr << "[FAIL] [4] revert after stack: '" << name << "' not bit-exact\n";
            unet_weights_destroy(h);
            return 1;
        }
    }
    std::cout << "[4] revert after stack bit-exact: OK\n";

    unet_weights_destroy(h);
    std::cout << "[PASS] test_lora_runtime: all gates green\n";
    return 0;
}

#endif // HAVE_CUDA

} // namespace dollama

int main()
{
#ifdef HAVE_CUDA
    try
    {
        return dollama::run();
    }
    catch (const std::exception& e)
    {
        std::cerr << "[FAIL] exception: " << e.what() << "\n";
        return 1;
    }
#else
    std::cout << "[SKIP] test_lora_runtime skipped (CUDA disabled build)\n";
    return 0;
#endif
}

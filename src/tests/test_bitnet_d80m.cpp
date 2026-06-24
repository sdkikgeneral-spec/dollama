// BitNet d80m config (容量増) 疎通テスト — 施策 D「容量増のコード化」。
//
// このテストは -DDOLLAMA_BITNET_ARCH=1 でビルドされ、アーキ次元が d80m
// (N_LAYERS=16 / FFN_DIM=2464・他は据え置き) に切り替わっていることを前提とする。
// 本訓練・GPU 実走・本番重み/golden の差し替えは別タスク (A 実ペア増待ち)。
// ここで検証するのは「コードが d80m 構成で矛盾なく構築・forward・ロードできるか」。
//
// testing.md 形式: 各検証は if (!cond) { cerr; return false; }、
// main で ok 集約、成功は PASSED / 最後に ALL PASSED。
//
// 検証項目 (完了条件):
//   - param_count 厳密アンカー: default 32,976,896 / d80m 79,908,864 (free 関数で両検証)
//   - ビルド config が d80m である (bitnet_arch::N_LAYERS==16 / FFN_DIM==2464)
//   - class の constexpr が d80m を見ている (BitNet / BitNetDenseInfer 3 ファイル同値)
//   - d80m 構成 BitNet::param_count() == 79,908,864
//   - d80m host forward 疎通: 形状 [S, 4999]・NaN/Inf なし・決定的
//   - safetensors 往復: d80m 重みを書き出し → BitNetDenseInfer が shape 照合エラー
//     なくロードでき、forward が完走する ([out,in] 規約・config 連動 expect shape)

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "infer/bitnet.hpp"          // BitNetDenseInfer (config 連動 expect shape)
#include "io/safetensors.hpp"
#include "models/bitnet.hpp"          // class BitNet (d80m constexpr)
#include "models/bitnet_config.hpp"   // bitnet_arch::param_count_for

namespace dollama
{

// scratchpad 等の一時ファイルパス (meson が -DD80M_TMP_PATH で埋め込む。
// 未定義時は build cwd 相対のファイル名にフォールバックする)。
#ifndef D80M_TMP_PATH
#define D80M_TMP_PATH "test_bitnet_d80m_tmp.safetensors"
#endif

// ----------------------------------------------------------------
// param_count 厳密アンカー (両 config を free 関数で一度に検証)
// ----------------------------------------------------------------
static bool test_param_count_anchors()
{
    constexpr size_t kDefaultParams =
        bitnet_arch::param_count_for(4999, 512, 8, 8, 1792);
    constexpr size_t kD80mParams =
        bitnet_arch::param_count_for(4999, 512, 16, 8, 2464);
    static_assert(kDefaultParams == 32976896ull,
                  "default param_count must be 32,976,896");
    static_assert(kD80mParams == 79908864ull,
                  "d80m param_count must be 79,908,864");

    std::cout << "[test_param_count_anchors] default(free) = " << kDefaultParams
              << " / d80m(free) = " << kD80mParams << "\n";

    if (kDefaultParams != 32976896ull || kD80mParams != 79908864ull)
    {
        std::cerr << "[test_param_count_anchors] free 関数の param_count 不一致\n";
        return false;
    }
    std::cout << "[test_param_count_anchors] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// このバイナリが d80m config でビルドされていること + 3 ファイル同値
// ----------------------------------------------------------------
static bool test_d80m_config_active()
{
    if (bitnet_arch::N_LAYERS != 16 || bitnet_arch::FFN_DIM != 2464)
    {
        std::cerr << "[test_d80m_config_active] d80m config でビルドされていない: "
                     "N_LAYERS=" << bitnet_arch::N_LAYERS
                  << " FFN_DIM=" << bitnet_arch::FFN_DIM
                  << " (-DDOLLAMA_BITNET_ARCH=1 が必要)\n";
        return false;
    }
    // 据え置き次元の確認。
    if (bitnet_arch::VOCAB_SIZE != 4999 || bitnet_arch::D_MODEL != 512
        || bitnet_arch::N_HEADS != 8 || bitnet_arch::HEAD_DIM != 64
        || bitnet_arch::MAX_SEQ_LEN != 64)
    {
        std::cerr << "[test_d80m_config_active] 据え置き次元が崩れている\n";
        return false;
    }
    // 3 ファイル (BitNet / BitNetDenseInfer / bitnet_arch) が同一次元を見ているか。
    if (BitNet::N_LAYERS != bitnet_arch::N_LAYERS
        || BitNet::FFN_DIM != bitnet_arch::FFN_DIM
        || BitNetDenseInfer::N_LAYERS != bitnet_arch::N_LAYERS
        || BitNetDenseInfer::FFN_DIM != bitnet_arch::FFN_DIM)
    {
        std::cerr << "[test_d80m_config_active] BitNet/BitNetDenseInfer/bitnet_arch の"
                     " 次元が不一致 (3 ファイル同値の保証が崩れている)\n";
        return false;
    }
    // class の param_count() も d80m。
    if (BitNet::param_count() != 79908864ull)
    {
        std::cerr << "[test_d80m_config_active] d80m BitNet::param_count() != "
                     "79,908,864: " << BitNet::param_count() << "\n";
        return false;
    }
    std::cout << "[test_d80m_config_active] N_LAYERS=" << BitNet::N_LAYERS
              << " FFN_DIM=" << BitNet::FFN_DIM
              << " param_count=" << BitNet::param_count() << "\n";
    std::cout << "[test_d80m_config_active] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// d80m host forward 疎通 (class BitNet, ternary 参照 forward)
//   形状 [S, 4999]・NaN/Inf なし・2 回実行で決定的。
// ----------------------------------------------------------------
static bool test_d80m_forward_smoke()
{
    BitNet model;
    model.init_random(20260624);

    const std::vector<int> tokens = {1, 5, 100, 4998, 7, 5, 200};  // seq=7
    const int S = static_cast<int>(tokens.size());

    std::vector<float> a = model.forward(tokens);
    std::vector<float> b = model.forward(tokens);

    const size_t want_n = static_cast<size_t>(S) * BitNet::VOCAB_SIZE;
    if (a.size() != want_n)
    {
        std::cerr << "[test_d80m_forward_smoke] 出力サイズ不一致: got " << a.size()
                  << " want " << want_n << " (VOCAB=" << BitNet::VOCAB_SIZE << ")\n";
        return false;
    }
    if (a != b)
    {
        std::cerr << "[test_d80m_forward_smoke] 2 回実行で不一致\n";
        return false;
    }
    for (float v : a)
    {
        if (!std::isfinite(v))
        {
            std::cerr << "[test_d80m_forward_smoke] 非有限 logit: " << v << "\n";
            return false;
        }
    }
    std::cout << "[test_d80m_forward_smoke] 形状 [" << S << ", "
              << BitNet::VOCAB_SIZE << "] OK / 決定的 OK / 有限 OK\n";
    std::cout << "[test_d80m_forward_smoke] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// 最小 safetensors (F32) ライタ — テスト内専用。
//   tensors: 名前 → (shape, data)。data は row-major float。
//   レイアウトは io/safetensors.hpp リーダーと一致 (8B len + JSON + raw body)。
// ----------------------------------------------------------------
struct StWriteTensor
{
    std::string         name;
    std::vector<size_t> shape;
    std::vector<float>  data;
};

static void write_safetensors_f32(const std::string& path,
                                  const std::vector<StWriteTensor>& tensors)
{
    // (1) body を連結しつつ各テンソルの [begin,end) を求める。
    std::vector<uint8_t> body;
    std::string json = "{";
    bool first = true;
    size_t cursor = 0;
    for (const auto& t : tensors)
    {
        const size_t nbytes = t.data.size() * sizeof(float);
        const size_t begin = cursor;
        const size_t end = begin + nbytes;

        const uint8_t* p = reinterpret_cast<const uint8_t*>(t.data.data());
        body.insert(body.end(), p, p + nbytes);
        cursor = end;

        if (!first)
        {
            json += ",";
        }
        first = false;
        json += "\"" + t.name + "\":{\"dtype\":\"F32\",\"shape\":[";
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

    // (2) 8 バイト LE ヘッダ長 + JSON + body を書き出す。
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f)
    {
        throw std::runtime_error("write_safetensors_f32: cannot open: " + path);
    }
    const uint64_t n = static_cast<uint64_t>(json.size());
    uint8_t lenbuf[8];
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
        throw std::runtime_error("write_safetensors_f32: write failed: " + path);
    }
}

// ----------------------------------------------------------------
// safetensors 往復: d80m dense 重みを書き出し → BitNetDenseInfer ロード → forward。
//   BitNetDenseInfer::load_f32 の expect_numel は config 連動 (N_LAYERS=16 /
//   FFN_DIM=2464) なので、shape 不一致なくロードできれば config 連動が証明される。
//   重みは決定的乱数で合成 (本番重み・golden には一切触れない)。
// ----------------------------------------------------------------
static bool test_d80m_safetensors_roundtrip()
{
    using cfg = BitNetDenseInfer;  // expect shape 元の constexpr
    const int V  = cfg::VOCAB_SIZE;
    const int D  = cfg::D_MODEL;
    const int L  = cfg::N_LAYERS;
    const int F  = cfg::FFN_DIM;

    std::mt19937_64 rng(424242);
    std::normal_distribution<float> wdist(0.0f, 0.02f);
    std::normal_distribution<float> ndist(1.0f, 0.01f);
    auto gen = [&](size_t n, bool norm)
    {
        std::vector<float> v(n);
        for (auto& x : v)
        {
            x = norm ? ndist(rng) : wdist(rng);
        }
        return v;
    };

    std::vector<StWriteTensor> ts;
    // embed [V, D] (tied lm_head)
    ts.push_back({"embed", {static_cast<size_t>(V), static_cast<size_t>(D)},
                  gen(static_cast<size_t>(V) * D, false)});
    // final_norm [D]
    ts.push_back({"final_norm", {static_cast<size_t>(D)},
                  gen(static_cast<size_t>(D), true)});
    // 各層 (74 テンソル相当・[out,in] 規約)
    for (int i = 0; i < L; ++i)
    {
        const std::string pfx = "layers." + std::to_string(i) + ".";
        ts.push_back({pfx + "attn_norm", {static_cast<size_t>(D)},
                      gen(static_cast<size_t>(D), true)});
        ts.push_back({pfx + "wq", {static_cast<size_t>(D), static_cast<size_t>(D)},
                      gen(static_cast<size_t>(D) * D, false)});
        ts.push_back({pfx + "wk", {static_cast<size_t>(D), static_cast<size_t>(D)},
                      gen(static_cast<size_t>(D) * D, false)});
        ts.push_back({pfx + "wv", {static_cast<size_t>(D), static_cast<size_t>(D)},
                      gen(static_cast<size_t>(D) * D, false)});
        ts.push_back({pfx + "wo", {static_cast<size_t>(D), static_cast<size_t>(D)},
                      gen(static_cast<size_t>(D) * D, false)});
        ts.push_back({pfx + "ffn_norm", {static_cast<size_t>(D)},
                      gen(static_cast<size_t>(D), true)});
        // w_gate / w_up : [FFN_DIM, D_MODEL] (out=FFN, in=D)
        ts.push_back({pfx + "w_gate", {static_cast<size_t>(F), static_cast<size_t>(D)},
                      gen(static_cast<size_t>(F) * D, false)});
        ts.push_back({pfx + "w_up", {static_cast<size_t>(F), static_cast<size_t>(D)},
                      gen(static_cast<size_t>(F) * D, false)});
        // w_down : [D_MODEL, FFN_DIM] (out=D, in=FFN)
        ts.push_back({pfx + "w_down", {static_cast<size_t>(D), static_cast<size_t>(F)},
                      gen(static_cast<size_t>(D) * F, false)});
    }

    const std::string path = D80M_TMP_PATH;
    try
    {
        write_safetensors_f32(path, ts);
    }
    catch (const std::exception& e)
    {
        std::cerr << "[test_d80m_safetensors_roundtrip] 書き出し失敗: " << e.what() << "\n";
        return false;
    }

    // BitNetDenseInfer でロード (config 連動の expect shape で照合)。
    try
    {
        BitNetDenseInfer infer(path);
        const std::vector<int> tokens = {1, 5, 6, 7, 8, 9};
        std::vector<float> logits = infer.forward(tokens);
        const size_t want_n =
            tokens.size() * static_cast<size_t>(cfg::VOCAB_SIZE);
        if (logits.size() != want_n)
        {
            std::cerr << "[test_d80m_safetensors_roundtrip] logits サイズ不一致: got "
                      << logits.size() << " want " << want_n << "\n";
            std::remove(path.c_str());
            return false;
        }
        for (float v : logits)
        {
            if (!std::isfinite(v))
            {
                std::cerr << "[test_d80m_safetensors_roundtrip] 非有限 logit: " << v << "\n";
                std::remove(path.c_str());
                return false;
            }
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "[test_d80m_safetensors_roundtrip] ロード/forward 失敗: "
                  << e.what() << "\n";
        std::remove(path.c_str());
        return false;
    }

    std::remove(path.c_str());
    std::cout << "[test_d80m_safetensors_roundtrip] d80m 重み (" << ts.size()
              << " テンソル) を書き出し → ロード → forward 完走 OK\n";
    std::cout << "[test_d80m_safetensors_roundtrip] PASSED\n";
    return true;
}

} // namespace dollama

int main()
{
    bool ok = true;
    try
    {
        ok = dollama::test_param_count_anchors()       && ok;
        ok = dollama::test_d80m_config_active()        && ok;
        ok = dollama::test_d80m_forward_smoke()        && ok;
        ok = dollama::test_d80m_safetensors_roundtrip() && ok;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[test_bitnet_d80m] 例外: " << e.what() << "\n";
        return 1;
    }

    if (!ok)
    {
        std::cerr << "[test_bitnet_d80m] FAILED\n";
        return 1;
    }
    std::cout << "[test_bitnet_d80m] ALL PASSED\n";
    return 0;
}

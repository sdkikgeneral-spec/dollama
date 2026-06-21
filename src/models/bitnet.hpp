#pragma once
// BitNet b1.58 モデル定義 + ホスト (CPU) 参照 forward + 量子化関数。
//
// 純ホストヘッダ (CUDA / OpenVINO 非依存・STL のみで自己完結)。
// scheduler.hpp と同じく、ここは「数値的に正しい参照実装」を目的とする。
// 4-5 CUDA ternary GEMM / 4-6 C++ 推論のゴールデン基準になる実装であり、
// 速度目標は課さない (naive ループで可)。
//
// ────────────────────────────────────────────────────────────────
// アーキテクチャ (確定値): BitNet b1.58, decoder-only transformer (LLaMA 系)
// ────────────────────────────────────────────────────────────────
//   VOCAB_SIZE  = 4999    (specials 5 + tags 4,994。dataset-spec.md §3.2)
//   D_MODEL     = 512
//   N_LAYERS    = 8
//   N_HEADS     = 8       (head_dim = 512 / 8 = 64)
//   FFN_DIM     = 1792    (SwiGLU: gate/up/down)
//   MAX_SEQ_LEN = 64      (text + tags、~32 token を余裕で収める)
//   RoPE base   = 10000
//   RMSNorm eps = 1e-5
//   embed tied  : 入力 embedding と出力 lm_head が同一重みを共有
//
// 構成要素:
//   - BitLinear : 重み ternary {-1,0,+1} (absmean 量子化) + 活性 int8 (absmax 量子化)
//   - RMSNorm   : pre-norm (各 sublayer の入力側)
//   - RoPE      : Q/K に回転位置埋め込み
//   - SwiGLU FFN: down( silu(gate(x)) * up(x) )
//   - causal self-attention
//
// パラメータ数 (要素数合計):
//   embed (tied)   : VOCAB_SIZE * D_MODEL           = 4999 * 512   = 2,559,488
//   per layer:
//     attn q/k/v/o : 4 * D_MODEL * D_MODEL          = 4*512*512    = 1,048,576
//     ffn g/u/d    : 3 * D_MODEL * FFN_DIM          = 3*512*1792   = 2,752,512
//     rmsnorm ×2   : 2 * D_MODEL                    =               =     1,024
//     ─ layer 計   :                                                = 3,802,112
//   layers (×8)    : 3,802,112 * 8                                  = 30,416,896
//   final rmsnorm  : D_MODEL                                        =       512
//   ─────────────────────────────────────────────────────────────────────────
//   合計           : 2,559,488 + 30,416,896 + 512    = 32,976,896  (≈ 33.0M)
//   (embed tied のため lm_head は別カウントしない。30M ≤ 33.0M ≤ 100M を満たす)

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

namespace dollama
{

// ================================================================
// 量子化関数 (BitNet b1.58 の本質。単体テスト可能なよう関数化)
// ================================================================

// ternary 重み量子化の結果。
//   scale  : α = mean(|W|)  (absmean)
//   qweight: clamp(round(W/α), -1, 1) ∈ {-1,0,+1}  (要素数は入力と同じ)
struct TernaryQuant
{
    float                scale = 0.0f;  // α (絶対平均)
    std::vector<int8_t>  q;             // 量子化重み {-1,0,+1}
};

// 重みを absmean ternary 量子化する。
//   α = mean(|W|)、W_q = clamp(round(W/α), -1, 1)
//   全 0 入力 (α=0) は量子化結果を全 0・scale 0 とする (ゼロ除算回避)。
inline TernaryQuant quantize_weight_ternary(const float* w, size_t n)
{
    TernaryQuant out;
    out.q.resize(n);
    if (n == 0)
    {
        return out;
    }
    double sum_abs = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        sum_abs += std::fabs(static_cast<double>(w[i]));
    }
    const double alpha = sum_abs / static_cast<double>(n);
    out.scale = static_cast<float>(alpha);
    if (alpha == 0.0)
    {
        // 全 0 入力: scale 0、量子化結果も全 0。
        for (size_t i = 0; i < n; ++i)
        {
            out.q[i] = 0;
        }
        return out;
    }
    const double inv = 1.0 / alpha;
    for (size_t i = 0; i < n; ++i)
    {
        double r = std::round(static_cast<double>(w[i]) * inv);
        if (r > 1.0)
        {
            r = 1.0;
        }
        else if (r < -1.0)
        {
            r = -1.0;
        }
        out.q[i] = static_cast<int8_t>(r);
    }
    return out;
}

// 活性 int8 量子化の結果。
//   scale: absmax(x) / 127
//   q    : clamp(round(x/scale), -127, 127)
struct Int8Quant
{
    float                scale = 0.0f;  // absmax / 127
    std::vector<int8_t>  q;             // 量子化活性 [-127,127]
};

// 活性を absmax int8 量子化する。
//   scale = absmax(x)/127、x_q = clamp(round(x/scale), -127, 127)
//   全 0 入力 (absmax=0 → scale=0) は量子化結果を全 0 とする (ゼロ除算回避)。
inline Int8Quant quantize_activation_int8(const float* x, size_t n)
{
    Int8Quant out;
    out.q.resize(n);
    if (n == 0)
    {
        return out;
    }
    double amax = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        const double a = std::fabs(static_cast<double>(x[i]));
        if (a > amax)
        {
            amax = a;
        }
    }
    const double scale = amax / 127.0;
    out.scale = static_cast<float>(scale);
    if (scale == 0.0)
    {
        for (size_t i = 0; i < n; ++i)
        {
            out.q[i] = 0;
        }
        return out;
    }
    const double inv = 1.0 / scale;
    for (size_t i = 0; i < n; ++i)
    {
        double r = std::round(static_cast<double>(x[i]) * inv);
        if (r > 127.0)
        {
            r = 127.0;
        }
        else if (r < -127.0)
        {
            r = -127.0;
        }
        out.q[i] = static_cast<int8_t>(r);
    }
    return out;
}

// ================================================================
// RMSNorm (pre-norm)
// ================================================================
// out[i] = x[i] / sqrt(mean(x^2) + eps) * weight[i]
// 全 0 入力でも eps によりゼロ除算しない。
inline void rms_norm(const float* x, const float* weight, float* out,
                     size_t n, double eps = 1e-5)
{
    double ms = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        const double v = static_cast<double>(x[i]);
        ms += v * v;
    }
    ms /= static_cast<double>(n);
    const double inv = 1.0 / std::sqrt(ms + eps);
    for (size_t i = 0; i < n; ++i)
    {
        out[i] = static_cast<float>(static_cast<double>(x[i]) * inv
                                    * static_cast<double>(weight[i]));
    }
}

// ================================================================
// BitNet モデル
// ================================================================
class BitNet
{
public:
    // ── アーキテクチャ定数 ─────────────────────────────────────
    static constexpr int VOCAB_SIZE  = 4999;  // specials 5 + tags 4994
    static constexpr int D_MODEL     = 512;
    static constexpr int N_LAYERS    = 8;
    static constexpr int N_HEADS     = 8;
    static constexpr int HEAD_DIM    = D_MODEL / N_HEADS;  // 64
    static constexpr int FFN_DIM     = 1792;
    static constexpr int MAX_SEQ_LEN = 64;
    static constexpr double ROPE_BASE = 10000.0;
    static constexpr double RMS_EPS   = 1e-5;

    // 1 層分の重み (host float)。BitLinear の重みも float で保持し、
    // forward 時にオンザフライで ternary 量子化 → 復元する (参照実装)。
    struct Layer
    {
        std::vector<float> attn_norm;  // [D_MODEL]
        std::vector<float> wq;         // [D_MODEL, D_MODEL]  (out, in) row-major
        std::vector<float> wk;         // [D_MODEL, D_MODEL]
        std::vector<float> wv;         // [D_MODEL, D_MODEL]
        std::vector<float> wo;         // [D_MODEL, D_MODEL]
        std::vector<float> ffn_norm;   // [D_MODEL]
        std::vector<float> w_gate;     // [FFN_DIM, D_MODEL]
        std::vector<float> w_up;       // [FFN_DIM, D_MODEL]
        std::vector<float> w_down;     // [D_MODEL, FFN_DIM]
    };

    BitNet()
    {
        embed_.assign(static_cast<size_t>(VOCAB_SIZE) * D_MODEL, 0.0f);
        final_norm_.assign(D_MODEL, 1.0f);
        layers_.resize(N_LAYERS);
        for (auto& L : layers_)
        {
            L.attn_norm.assign(D_MODEL, 1.0f);
            L.wq.assign(static_cast<size_t>(D_MODEL) * D_MODEL, 0.0f);
            L.wk.assign(static_cast<size_t>(D_MODEL) * D_MODEL, 0.0f);
            L.wv.assign(static_cast<size_t>(D_MODEL) * D_MODEL, 0.0f);
            L.wo.assign(static_cast<size_t>(D_MODEL) * D_MODEL, 0.0f);
            L.ffn_norm.assign(D_MODEL, 1.0f);
            L.w_gate.assign(static_cast<size_t>(FFN_DIM) * D_MODEL, 0.0f);
            L.w_up.assign(static_cast<size_t>(FFN_DIM) * D_MODEL, 0.0f);
            L.w_down.assign(static_cast<size_t>(D_MODEL) * FFN_DIM, 0.0f);
        }
    }

    // 全重み要素数の合計 (パラメータ数)。embed tied のため embed は 1 回だけ数える。
    static size_t param_count()
    {
        const size_t embed = static_cast<size_t>(VOCAB_SIZE) * D_MODEL;
        const size_t per_attn = 4ull * D_MODEL * D_MODEL;
        const size_t per_ffn  = 3ull * static_cast<size_t>(FFN_DIM) * D_MODEL;
        const size_t per_norm = 2ull * D_MODEL;
        const size_t per_layer = per_attn + per_ffn + per_norm;
        const size_t final_norm = D_MODEL;
        return embed + per_layer * N_LAYERS + final_norm;
    }

    // 固定 seed で全重みを擬似乱数初期化する (決定的)。
    // 重みは小さめの正規分布、norm 重みは 1.0 近傍にする。
    void init_random(uint64_t seed)
    {
        std::mt19937_64 rng(seed);
        std::normal_distribution<float> wdist(0.0f, 0.02f);
        std::normal_distribution<float> ndist(1.0f, 0.01f);

        auto fill_w = [&](std::vector<float>& v)
        {
            for (auto& x : v)
            {
                x = wdist(rng);
            }
        };
        auto fill_n = [&](std::vector<float>& v)
        {
            for (auto& x : v)
            {
                x = ndist(rng);
            }
        };

        fill_w(embed_);
        for (auto& L : layers_)
        {
            fill_n(L.attn_norm);
            fill_w(L.wq);
            fill_w(L.wk);
            fill_w(L.wv);
            fill_w(L.wo);
            fill_n(L.ffn_norm);
            fill_w(L.w_gate);
            fill_w(L.w_up);
            fill_w(L.w_down);
        }
        fill_n(final_norm_);
    }

    // embed tied の確認用: 入力 embedding 重みポインタ。
    const float* embed_weight() const
    {
        return embed_.data();
    }
    // 出力 lm_head の重みポインタ。embed tied のため embed_ と同一を返す。
    const float* lm_head_weight() const
    {
        return embed_.data();
    }

    const std::vector<Layer>& layers() const
    {
        return layers_;
    }
    std::vector<Layer>& layers()
    {
        return layers_;
    }

    // ── ホスト参照 forward ─────────────────────────────────────
    // 入力 : token 列 (各 id は [0, VOCAB_SIZE))。
    // 出力 : logits [seq_len * VOCAB_SIZE] (row-major、行 = 位置)。
    // causal self-attention + RoPE + SwiGLU + 各 BitLinear は ternary/int8 量子化を経る。
    std::vector<float> forward(const std::vector<int>& tokens) const
    {
        const int S = static_cast<int>(tokens.size());
        if (S <= 0)
        {
            throw std::runtime_error("BitNet::forward: empty token sequence");
        }
        if (S > MAX_SEQ_LEN)
        {
            throw std::runtime_error("BitNet::forward: seq_len exceeds MAX_SEQ_LEN");
        }
        for (int t : tokens)
        {
            if (t < 0 || t >= VOCAB_SIZE)
            {
                throw std::runtime_error("BitNet::forward: token id out of range");
            }
        }

        // hidden state h[S][D_MODEL]
        std::vector<float> h(static_cast<size_t>(S) * D_MODEL, 0.0f);
        for (int s = 0; s < S; ++s)
        {
            const float* e = embed_.data()
                             + static_cast<size_t>(tokens[s]) * D_MODEL;
            for (int d = 0; d < D_MODEL; ++d)
            {
                h[static_cast<size_t>(s) * D_MODEL + d] = e[d];
            }
        }

        for (const auto& L : layers_)
        {
            attention_block(L, h, S);
            ffn_block(L, h, S);
        }

        // 最終 RMSNorm → lm_head (tied embed)。
        std::vector<float> logits(static_cast<size_t>(S) * VOCAB_SIZE, 0.0f);
        std::vector<float> normed(D_MODEL);
        for (int s = 0; s < S; ++s)
        {
            rms_norm(&h[static_cast<size_t>(s) * D_MODEL], final_norm_.data(),
                     normed.data(), D_MODEL, RMS_EPS);
            // lm_head: logits[v] = dot(normed, embed_[v]) (tied、量子化はかけない)。
            for (int v = 0; v < VOCAB_SIZE; ++v)
            {
                const float* wv = embed_.data()
                                  + static_cast<size_t>(v) * D_MODEL;
                double acc = 0.0;
                for (int d = 0; d < D_MODEL; ++d)
                {
                    acc += static_cast<double>(normed[d])
                           * static_cast<double>(wv[d]);
                }
                logits[static_cast<size_t>(s) * VOCAB_SIZE + v] =
                    static_cast<float>(acc);
            }
        }
        return logits;
    }

private:
    std::vector<float>  embed_;       // [VOCAB_SIZE, D_MODEL] (tied で lm_head と共有)
    std::vector<float>  final_norm_;  // [D_MODEL]
    std::vector<Layer>  layers_;

    // BitLinear: y = (Wq・xq を float 復元して dot) の参照実装。
    //   重み W [out_dim, in_dim] を absmean ternary 量子化、
    //   入力 x [in_dim] を absmax int8 量子化し、整数積を float に復元する。
    //   y[o] = w_scale * x_scale * Σ_i ( Wq[o,i] * xq[i] )
    static void bitlinear(const float* w, const float* x,
                          int out_dim, int in_dim, float* y)
    {
        // 入力活性を int8 量子化 (行全体で 1 つの scale)。
        const Int8Quant xq = quantize_activation_int8(x, static_cast<size_t>(in_dim));
        for (int o = 0; o < out_dim; ++o)
        {
            const float* wrow = w + static_cast<size_t>(o) * in_dim;
            // 各出力行ごとに重みを ternary 量子化 (行ごとの α)。
            const TernaryQuant wq =
                quantize_weight_ternary(wrow, static_cast<size_t>(in_dim));
            long long acc = 0;
            for (int i = 0; i < in_dim; ++i)
            {
                acc += static_cast<long long>(wq.q[static_cast<size_t>(i)])
                       * static_cast<long long>(xq.q[static_cast<size_t>(i)]);
            }
            y[o] = static_cast<float>(static_cast<double>(acc)
                                      * static_cast<double>(wq.scale)
                                      * static_cast<double>(xq.scale));
        }
    }

    // RoPE: head 内ベクトル vec[HEAD_DIM] に位置 pos の回転を適用する (in-place)。
    // (i, i+HEAD_DIM/2) のペアを回す GPT-NeoX 系レイアウト。
    static void apply_rope(float* vec, int pos)
    {
        const int half = HEAD_DIM / 2;
        for (int i = 0; i < half; ++i)
        {
            const double freq = 1.0 / std::pow(ROPE_BASE,
                static_cast<double>(2 * i) / static_cast<double>(HEAD_DIM));
            const double angle = static_cast<double>(pos) * freq;
            const double cs = std::cos(angle);
            const double sn = std::sin(angle);
            const double a = static_cast<double>(vec[i]);
            const double b = static_cast<double>(vec[i + half]);
            vec[i]        = static_cast<float>(a * cs - b * sn);
            vec[i + half] = static_cast<float>(a * sn + b * cs);
        }
    }

    // causal multi-head self-attention (pre-RMSNorm + residual)。
    void attention_block(const Layer& L, std::vector<float>& h, int S) const
    {
        const int D = D_MODEL;
        // 1) pre-norm
        std::vector<float> x(static_cast<size_t>(S) * D);
        for (int s = 0; s < S; ++s)
        {
            rms_norm(&h[static_cast<size_t>(s) * D], L.attn_norm.data(),
                     &x[static_cast<size_t>(s) * D], D, RMS_EPS);
        }

        // 2) Q/K/V projection (BitLinear)。
        std::vector<float> Q(static_cast<size_t>(S) * D);
        std::vector<float> K(static_cast<size_t>(S) * D);
        std::vector<float> V(static_cast<size_t>(S) * D);
        for (int s = 0; s < S; ++s)
        {
            const float* xs = &x[static_cast<size_t>(s) * D];
            bitlinear(L.wq.data(), xs, D, D, &Q[static_cast<size_t>(s) * D]);
            bitlinear(L.wk.data(), xs, D, D, &K[static_cast<size_t>(s) * D]);
            bitlinear(L.wv.data(), xs, D, D, &V[static_cast<size_t>(s) * D]);
        }

        // 3) RoPE を Q/K の各 head に適用。
        for (int s = 0; s < S; ++s)
        {
            for (int hd = 0; hd < N_HEADS; ++hd)
            {
                float* q = &Q[static_cast<size_t>(s) * D + hd * HEAD_DIM];
                float* k = &K[static_cast<size_t>(s) * D + hd * HEAD_DIM];
                apply_rope(q, s);
                apply_rope(k, s);
            }
        }

        // 4) causal scaled dot-product attention (head ごと)。
        const double scale = 1.0 / std::sqrt(static_cast<double>(HEAD_DIM));
        std::vector<float> attn_out(static_cast<size_t>(S) * D, 0.0f);
        std::vector<double> scores(static_cast<size_t>(S));
        for (int hd = 0; hd < N_HEADS; ++hd)
        {
            const int off = hd * HEAD_DIM;
            for (int s = 0; s < S; ++s)
            {
                // s は j<=s のみ参照可 (causal)。
                double maxv = -1e300;
                for (int j = 0; j <= s; ++j)
                {
                    const float* q = &Q[static_cast<size_t>(s) * D + off];
                    const float* k = &K[static_cast<size_t>(j) * D + off];
                    double dot = 0.0;
                    for (int d = 0; d < HEAD_DIM; ++d)
                    {
                        dot += static_cast<double>(q[d]) * static_cast<double>(k[d]);
                    }
                    dot *= scale;
                    scores[static_cast<size_t>(j)] = dot;
                    if (dot > maxv)
                    {
                        maxv = dot;
                    }
                }
                // softmax (causal 範囲 0..s)。
                double sum = 0.0;
                for (int j = 0; j <= s; ++j)
                {
                    const double e = std::exp(scores[static_cast<size_t>(j)] - maxv);
                    scores[static_cast<size_t>(j)] = e;
                    sum += e;
                }
                const double inv = 1.0 / sum;
                float* outv = &attn_out[static_cast<size_t>(s) * D + off];
                for (int j = 0; j <= s; ++j)
                {
                    const double w = scores[static_cast<size_t>(j)] * inv;
                    const float* v = &V[static_cast<size_t>(j) * D + off];
                    for (int d = 0; d < HEAD_DIM; ++d)
                    {
                        outv[d] += static_cast<float>(w * static_cast<double>(v[d]));
                    }
                }
            }
        }

        // 5) 出力 projection (BitLinear) + residual。
        std::vector<float> proj(D);
        for (int s = 0; s < S; ++s)
        {
            bitlinear(L.wo.data(), &attn_out[static_cast<size_t>(s) * D],
                      D, D, proj.data());
            for (int d = 0; d < D; ++d)
            {
                h[static_cast<size_t>(s) * D + d] += proj[d];
            }
        }
    }

    // SwiGLU FFN (pre-RMSNorm + residual)。
    //   y = w_down( silu(w_gate(x)) * w_up(x) )
    void ffn_block(const Layer& L, std::vector<float>& h, int S) const
    {
        const int D = D_MODEL;
        const int F = FFN_DIM;
        std::vector<float> x(D);
        std::vector<float> g(F);
        std::vector<float> u(F);
        std::vector<float> inter(F);
        std::vector<float> y(D);
        for (int s = 0; s < S; ++s)
        {
            rms_norm(&h[static_cast<size_t>(s) * D], L.ffn_norm.data(),
                     x.data(), D, RMS_EPS);
            bitlinear(L.w_gate.data(), x.data(), F, D, g.data());
            bitlinear(L.w_up.data(),   x.data(), F, D, u.data());
            for (int f = 0; f < F; ++f)
            {
                // SiLU(gate) * up
                const double gv = static_cast<double>(g[f]);
                const double silu = gv / (1.0 + std::exp(-gv));
                inter[f] = static_cast<float>(silu * static_cast<double>(u[f]));
            }
            bitlinear(L.w_down.data(), inter.data(), D, F, y.data());
            for (int d = 0; d < D; ++d)
            {
                h[static_cast<size_t>(s) * D + d] += y[d];
            }
        }
    }
};

} // namespace dollama

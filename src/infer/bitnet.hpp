#pragma once
// BitNet dense (FP32) 推論クラス — Phase 4 dense #6。
//
// 訓練済み dense モデル (FP Linear) の重みを safetensors からロードし、
// CPU で forward + 自己回帰 greedy デコード (自然文 text → danbooru タグ) を行う。
//
// ──────────────────────────────────────────────────────────────────
// src/models/bitnet.hpp との棲み分け (重要):
//   models/bitnet.hpp::BitNet は BitLinear (ternary 重み + int8 活性) を通すため、
//   PyTorch dense と数値が一致しない。本クラスは流用せず dense FP Linear で実装する。
//   数値ヘルパ (rms_norm) と RoPE 方式・アーキ定数は models/bitnet.hpp に揃える。
//
// アーキ (models/bitnet.hpp BitNet と等価・dense):
//   VOCAB 4999 / D_MODEL 512 / N_LAYERS 8 / N_HEADS 8 / HEAD_DIM 64 /
//   FFN 1792 / MAX_SEQ 64 / RoPE base 10000 / RMSNorm eps 1e-5。
//   SwiGLU / causal attention / embed tied (lm_head = embed 共有)。
//   RoPE は GPT-NeoX ペアリング ((i, i+half) を回す半分割回転)。
//
// 重みレイアウト (row-major [out, in]、PyTorch nn.Linear.weight と一致):
//   y[o] = Σ_i w[o*in + i] * x[i]。
//
// 純ホスト C++ (CUDA / OpenVINO 非依存・STL + safetensors.hpp/tokenizer.hpp のみ)。
// 内部の積和は double 蓄積 (golden の FP32 突合で誤差を抑える)。

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "io/safetensors.hpp"
#include "io/tokenizer.hpp"
#include "models/bitnet.hpp"  // rms_norm / アーキ定数の参照元

namespace dollama
{

// dense FP32 推論クラス。
class BitNetDenseInfer
{
public:
    // ── アーキ定数 (models/bitnet.hpp BitNet と一致) ───────────────
    static constexpr int VOCAB_SIZE  = 4999;
    static constexpr int D_MODEL     = 512;
    static constexpr int N_LAYERS    = 8;
    static constexpr int N_HEADS     = 8;
    static constexpr int HEAD_DIM    = D_MODEL / N_HEADS;  // 64
    static constexpr int FFN_DIM     = 1792;
    static constexpr int MAX_SEQ_LEN = 64;
    static constexpr double ROPE_BASE = 10000.0;
    static constexpr double RMS_EPS   = 1e-5;

    // 1 層分の重み (host float, dense)。
    struct Layer
    {
        std::vector<float> attn_norm;  // [D_MODEL]
        std::vector<float> wq;         // [D_MODEL, D_MODEL]
        std::vector<float> wk;         // [D_MODEL, D_MODEL]
        std::vector<float> wv;         // [D_MODEL, D_MODEL]
        std::vector<float> wo;         // [D_MODEL, D_MODEL]
        std::vector<float> ffn_norm;   // [D_MODEL]
        std::vector<float> w_gate;     // [FFN_DIM, D_MODEL]
        std::vector<float> w_up;       // [FFN_DIM, D_MODEL]
        std::vector<float> w_down;     // [D_MODEL, FFN_DIM]
    };

    // safetensors ファイルから全重みをロードする (FP32 のみ受け付ける)。
    explicit BitNetDenseInfer(const std::string& weights_path)
    {
        SafeTensors st(weights_path);
        embed_ = load_f32(st, "embed",
                          static_cast<size_t>(VOCAB_SIZE) * D_MODEL);
        final_norm_ = load_f32(st, "final_norm", D_MODEL);
        layers_.resize(N_LAYERS);
        for (int i = 0; i < N_LAYERS; ++i)
        {
            const std::string p = "layers." + std::to_string(i) + ".";
            Layer& L = layers_[static_cast<size_t>(i)];
            L.attn_norm = load_f32(st, p + "attn_norm", D_MODEL);
            L.wq = load_f32(st, p + "wq",
                            static_cast<size_t>(D_MODEL) * D_MODEL);
            L.wk = load_f32(st, p + "wk",
                            static_cast<size_t>(D_MODEL) * D_MODEL);
            L.wv = load_f32(st, p + "wv",
                            static_cast<size_t>(D_MODEL) * D_MODEL);
            L.wo = load_f32(st, p + "wo",
                            static_cast<size_t>(D_MODEL) * D_MODEL);
            L.ffn_norm = load_f32(st, p + "ffn_norm", D_MODEL);
            L.w_gate = load_f32(st, p + "w_gate",
                                static_cast<size_t>(FFN_DIM) * D_MODEL);
            L.w_up = load_f32(st, p + "w_up",
                              static_cast<size_t>(FFN_DIM) * D_MODEL);
            L.w_down = load_f32(st, p + "w_down",
                                static_cast<size_t>(D_MODEL) * FFN_DIM);
        }
    }

    // ── ホスト dense forward ───────────────────────────────────────
    // 入力 : token 列 (各 id は [0, VOCAB_SIZE))。
    // 出力 : logits [seq_len * VOCAB_SIZE] (row-major、行 = 位置)。
    std::vector<float> forward(const std::vector<int>& tokens) const
    {
        const int S = static_cast<int>(tokens.size());
        if (S <= 0)
        {
            throw std::runtime_error("BitNetDenseInfer::forward: empty sequence");
        }
        if (S > MAX_SEQ_LEN)
        {
            throw std::runtime_error("BitNetDenseInfer::forward: seq exceeds MAX_SEQ_LEN");
        }
        for (int t : tokens)
        {
            if (t < 0 || t >= VOCAB_SIZE)
            {
                throw std::runtime_error("BitNetDenseInfer::forward: token id out of range");
            }
        }

        // hidden state h[S][D_MODEL] を embedding で初期化。
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

    // ── greedy デコード ────────────────────────────────────────────
    // prompt id 列 (= [<bos>] + encode_text_greedy(text) + [<sep>]) から
    // 次トークンを逐次 argmax で生成する。生成した tag id 列 (prompt・<eos>
    // を含まない pure な生成部) を返す。
    //   - 各ステップ: 列全体を forward → 最終位置 FP32 logits の argmax。
    //   - tie は最小 id。
    //   - 次トークン == <eos>(=2) → 出力に含めず停止。
    //   - 列長が MAX_SEQ_LEN(=64) に達したら打ち切り。
    //   - repeat 抑制なし。<eos> 以外の specials はそのまま積む。
    std::vector<int> generate(const std::vector<int>& prompt_ids) const
    {
        std::vector<int> seq = prompt_ids;
        std::vector<int> gen;
        while (static_cast<int>(seq.size()) < MAX_SEQ_LEN)
        {
            std::vector<float> logits = forward(seq);
            // 最終位置のロジット行。
            const size_t last = (seq.size() - 1) * static_cast<size_t>(VOCAB_SIZE);
            int best = 0;
            float bestv = logits[last];
            for (int v = 1; v < VOCAB_SIZE; ++v)
            {
                const float lv = logits[last + static_cast<size_t>(v)];
                // tie は小さい id を選ぶため、厳密に大きいときだけ更新する。
                if (lv > bestv)
                {
                    bestv = lv;
                    best = v;
                }
            }
            if (best == TOK_EOS)
            {
                break;  // <eos> は出力に含めず停止
            }
            gen.push_back(best);
            seq.push_back(best);
        }
        return gen;
    }

    // tokenizer から prompt を組んで生成 (text → tag id 列) する便宜関数。
    std::vector<int> generate_from_text(const Tokenizer& tok,
                                        const std::string& text) const
    {
        // encode_text は <bos> ... <eos> を付けるので、本仕様の
        // [<bos>] + encode_text_greedy(text) + [<sep>] を自前で組む。
        std::vector<int> framed = tok.encode_text(text, MAX_SEQ_LEN);
        std::vector<int> prompt;
        prompt.push_back(TOK_BOS);
        // framed は [<bos>, ...tags..., <eos>]。中身 (tags 部) だけ抜く。
        for (size_t i = 1; i + 1 < framed.size(); ++i)
        {
            prompt.push_back(framed[i]);
        }
        prompt.push_back(TOK_SEP);
        return generate(prompt);
    }

    const float* embed_weight() const
    {
        return embed_.data();
    }
    const std::vector<Layer>& layers() const
    {
        return layers_;
    }

private:
    std::vector<float>  embed_;       // [VOCAB_SIZE, D_MODEL] (tied lm_head)
    std::vector<float>  final_norm_;  // [D_MODEL]
    std::vector<Layer>  layers_;

    // safetensors から F32 テンソルを期待要素数で読み出す。
    static std::vector<float> load_f32(const SafeTensors& st,
                                       const std::string& name,
                                       size_t expect_numel)
    {
        if (st.dtype(name) != StDtype::F32)
        {
            throw std::runtime_error("BitNetDenseInfer: '" + name + "' must be F32");
        }
        size_t nbytes = 0;
        const uint8_t* p = st.tensor_bytes(name, nbytes);
        const size_t numel = nbytes / sizeof(float);
        if (numel != expect_numel)
        {
            throw std::runtime_error("BitNetDenseInfer: '" + name + "' numel="
                                     + std::to_string(numel) + " expected "
                                     + std::to_string(expect_numel));
        }
        std::vector<float> out(numel);
        // raw little-endian F32 をそのままコピー (safetensors はバイト透過)。
        const float* fp = reinterpret_cast<const float*>(p);
        for (size_t i = 0; i < numel; ++i)
        {
            out[i] = fp[i];
        }
        return out;
    }

    // dense Linear: y[o] = Σ_i w[o*in + i] * x[i] (double 蓄積)。
    static void linear(const float* w, const float* x,
                       int out_dim, int in_dim, float* y)
    {
        for (int o = 0; o < out_dim; ++o)
        {
            const float* wrow = w + static_cast<size_t>(o) * in_dim;
            double acc = 0.0;
            for (int i = 0; i < in_dim; ++i)
            {
                acc += static_cast<double>(wrow[i]) * static_cast<double>(x[i]);
            }
            y[o] = static_cast<float>(acc);
        }
    }

    // RoPE (GPT-NeoX 系・(i, i+half) ペア)。models/bitnet.hpp と同式。
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

        // 2) Q/K/V projection (dense Linear)。
        std::vector<float> Q(static_cast<size_t>(S) * D);
        std::vector<float> K(static_cast<size_t>(S) * D);
        std::vector<float> V(static_cast<size_t>(S) * D);
        for (int s = 0; s < S; ++s)
        {
            const float* xs = &x[static_cast<size_t>(s) * D];
            linear(L.wq.data(), xs, D, D, &Q[static_cast<size_t>(s) * D]);
            linear(L.wk.data(), xs, D, D, &K[static_cast<size_t>(s) * D]);
            linear(L.wv.data(), xs, D, D, &V[static_cast<size_t>(s) * D]);
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

        // 5) 出力 projection (dense Linear) + residual。
        std::vector<float> proj(D);
        for (int s = 0; s < S; ++s)
        {
            linear(L.wo.data(), &attn_out[static_cast<size_t>(s) * D],
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
            linear(L.w_gate.data(), x.data(), F, D, g.data());
            linear(L.w_up.data(),   x.data(), F, D, u.data());
            for (int f = 0; f < F; ++f)
            {
                const double gv = static_cast<double>(g[f]);
                const double silu = gv / (1.0 + std::exp(-gv));
                inter[f] = static_cast<float>(silu * static_cast<double>(u[f]));
            }
            linear(L.w_down.data(), inter.data(), D, F, y.data());
            for (int d = 0; d < D; ++d)
            {
                h[static_cast<size_t>(s) * D + d] += y[d];
            }
        }
    }
};

} // namespace dollama

#pragma once
// BitNet dense INT8 推論クラス — Phase 4 圧縮実験 (重みのみ INT8)。
//
// 既存 dense FP32 推論 (src/infer/bitnet.hpp BitNetDenseInfer) のドロップイン互換版。
// 同じ FP32 safetensors をロードし、各層の射影 Linear 8 本 (wq/wk/wv/wo,
// w_gate/w_up/w_down) だけを「per-output-row 対称 absmax INT8」に量子化して保持する。
// 残り (embed/lm_head(tied)・RMSNorm 重み・RoPE/attention/softmax/SwiGLU 非線形) は
// FP32 据え置き (models/bitnet.hpp の BitLinear と同方針 = 射影だけ整数化)。
//
// ──────────────────────────────────────────────────────────────────
// 数値経路 (BitNetDenseInfer との差分は射影 Linear だけ):
//   - 射影: 重みは per-row INT8 (ロード時に量子化)、活性は forward 時に
//     quantize_activation_int8 (models/bitnet.hpp 流用) で int8 化。
//     y[o] = w_scale[o] · x_scale · Σ_i ( qw[o,i] · xq[i] )  (int64 蓄積)。
//   - lm_head: BitNetDenseInfer と同じく FP32 dot (double 蓄積)。embed tied。
//   - forward 構造 / RoPE 式 / causal mask / softmax / SwiGLU / greedy 停止規則は
//     BitNetDenseInfer と完全一致 (量子化部分だけ差し替え)。
//
// src/infer/bitnet.hpp は無改変 (dense #6 / A3 golden 非回帰を保証)。
//
// 純ホスト C++ (CUDA / OpenVINO 非依存・STL + safetensors.hpp/tokenizer.hpp /
// models/bitnet.hpp のみ)。

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "io/safetensors.hpp"
#include "io/tokenizer.hpp"
#include "models/bitnet.hpp"  // rms_norm / quantize_*・アーキ定数の参照元

namespace dollama
{

// dense INT8 (重みのみ INT8) 推論クラス。
// public API は BitNetDenseInfer と一致 (ドロップイン互換)。
class BitNetInt8Infer
{
public:
    // ── アーキ定数 (BitNetDenseInfer / models/bitnet.hpp と一致) ────
    static constexpr int VOCAB_SIZE  = 4999;
    static constexpr int D_MODEL     = 512;
    static constexpr int N_LAYERS    = 8;
    static constexpr int N_HEADS     = 8;
    static constexpr int HEAD_DIM    = D_MODEL / N_HEADS;  // 64
    static constexpr int FFN_DIM     = 1792;
    static constexpr int MAX_SEQ_LEN = 64;
    static constexpr double ROPE_BASE = 10000.0;
    static constexpr double RMS_EPS   = 1e-5;

    // INT8 量子化済みの射影重み 1 本分 (qweight + per-row scale + 形状)。
    struct QLinear
    {
        std::vector<int8_t> q;      // [out_dim*in_dim] row-major
        std::vector<float>  scale;  // [out_dim]
        int                 out_dim = 0;
        int                 in_dim  = 0;
    };

    // 1 層分の重み。norm は FP32・射影 8 本は INT8。
    struct Layer
    {
        std::vector<float> attn_norm;  // [D_MODEL] (FP32)
        QLinear            wq;         // [D_MODEL, D_MODEL]
        QLinear            wk;         // [D_MODEL, D_MODEL]
        QLinear            wv;         // [D_MODEL, D_MODEL]
        QLinear            wo;         // [D_MODEL, D_MODEL]
        std::vector<float> ffn_norm;   // [D_MODEL] (FP32)
        QLinear            w_gate;     // [FFN_DIM, D_MODEL]
        QLinear            w_up;       // [FFN_DIM, D_MODEL]
        QLinear            w_down;     // [D_MODEL, FFN_DIM]
    };

    // safetensors (FP32) から全重みをロードし、射影 8 本を per-row INT8 量子化する。
    // BitNetDenseInfer と同一の 74 テンソル [out,in] レイアウトを受け付ける。
    explicit BitNetInt8Infer(const std::string& weights_path)
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
            L.wq = load_qlinear(st, p + "wq", D_MODEL, D_MODEL);
            L.wk = load_qlinear(st, p + "wk", D_MODEL, D_MODEL);
            L.wv = load_qlinear(st, p + "wv", D_MODEL, D_MODEL);
            L.wo = load_qlinear(st, p + "wo", D_MODEL, D_MODEL);
            L.ffn_norm = load_f32(st, p + "ffn_norm", D_MODEL);
            L.w_gate = load_qlinear(st, p + "w_gate", FFN_DIM, D_MODEL);
            L.w_up   = load_qlinear(st, p + "w_up",   FFN_DIM, D_MODEL);
            L.w_down = load_qlinear(st, p + "w_down", D_MODEL, FFN_DIM);
        }
    }

    // INT8 重みの総バイト数 (qweight int8 + scale float)。フットプリント報告用。
    size_t int8_weight_bytes() const
    {
        size_t bytes = 0;
        for (const Layer& L : layers_)
        {
            const QLinear* qs[8] = {&L.wq, &L.wk, &L.wv, &L.wo,
                                    &L.w_gate, &L.w_up, &L.w_down, nullptr};
            for (int k = 0; k < 7; ++k)
            {
                bytes += qs[k]->q.size() * sizeof(int8_t);
                bytes += qs[k]->scale.size() * sizeof(float);
            }
        }
        return bytes;
    }

    // 量子化対象 (射影 8 本) を FP32 で持っていた場合のバイト数。削減量算出用。
    static size_t fp32_proj_bytes()
    {
        const size_t per_attn = 4ull * D_MODEL * D_MODEL;      // wq/wk/wv/wo
        const size_t per_ffn  = 3ull * static_cast<size_t>(FFN_DIM) * D_MODEL;
        const size_t per_layer = per_attn + per_ffn;
        return per_layer * N_LAYERS * sizeof(float);
    }

    // ── ホスト forward (射影だけ INT8・他は FP32) ──────────────────
    std::vector<float> forward(const std::vector<int>& tokens) const
    {
        const int S = static_cast<int>(tokens.size());
        if (S <= 0)
        {
            throw std::runtime_error("BitNetInt8Infer::forward: empty sequence");
        }
        if (S > MAX_SEQ_LEN)
        {
            throw std::runtime_error("BitNetInt8Infer::forward: seq exceeds MAX_SEQ_LEN");
        }
        for (int t : tokens)
        {
            if (t < 0 || t >= VOCAB_SIZE)
            {
                throw std::runtime_error("BitNetInt8Infer::forward: token id out of range");
            }
        }

        // hidden state h[S][D_MODEL] を embedding (FP32) で初期化。
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

        // 最終 RMSNorm → lm_head (tied embed、FP32 dot)。
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

    // ── greedy デコード (BitNetDenseInfer と完全一致の停止規則) ─────
    std::vector<int> generate(const std::vector<int>& prompt_ids) const
    {
        std::vector<int> seq = prompt_ids;
        std::vector<int> gen;
        while (static_cast<int>(seq.size()) < MAX_SEQ_LEN)
        {
            std::vector<float> logits = forward(seq);
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

    // tokenizer から prompt を組んで生成 (text → tag id 列)。
    std::vector<int> generate_from_text(const Tokenizer& tok,
                                        const std::string& text) const
    {
        std::vector<int> framed = tok.encode_text(text, MAX_SEQ_LEN);
        std::vector<int> prompt;
        prompt.push_back(TOK_BOS);
        for (size_t i = 1; i + 1 < framed.size(); ++i)
        {
            prompt.push_back(framed[i]);
        }
        prompt.push_back(TOK_SEP);
        return generate(prompt);
    }

    // ── 同一性条件付き prompt 組み立て (BitNetDenseInfer と同一) ─────
    std::vector<int> build_identity_prompt(
        const Tokenizer& tok,
        const std::vector<std::string>& identity_tags,
        const std::string& scene_text) const
    {
        std::vector<int> prompt;
        prompt.push_back(TOK_BOS);

        for (const std::string& t : identity_tags)
        {
            prompt.push_back(tok.tag_to_id(t));
        }
        prompt.push_back(TOK_SEP);

        std::vector<int> framed = tok.encode_text(scene_text, MAX_SEQ_LEN);
        for (size_t i = 1; i + 1 < framed.size(); ++i)
        {
            prompt.push_back(framed[i]);
        }
        prompt.push_back(TOK_SEP);

        return prompt;
    }

    std::vector<int> generate_with_identity(
        const Tokenizer& tok,
        const std::vector<std::string>& identity_tags,
        const std::string& scene_text) const
    {
        return generate(build_identity_prompt(tok, identity_tags, scene_text));
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
    std::vector<float>  embed_;       // [VOCAB_SIZE, D_MODEL] (tied lm_head, FP32)
    std::vector<float>  final_norm_;  // [D_MODEL] (FP32)
    std::vector<Layer>  layers_;

    // safetensors から F32 テンソルを期待要素数で読み出す。
    static std::vector<float> load_f32(const SafeTensors& st,
                                       const std::string& name,
                                       size_t expect_numel)
    {
        if (st.dtype(name) != StDtype::F32)
        {
            throw std::runtime_error("BitNetInt8Infer: '" + name + "' must be F32");
        }
        size_t nbytes = 0;
        const uint8_t* p = st.tensor_bytes(name, nbytes);
        const size_t numel = nbytes / sizeof(float);
        if (numel != expect_numel)
        {
            throw std::runtime_error("BitNetInt8Infer: '" + name + "' numel="
                                     + std::to_string(numel) + " expected "
                                     + std::to_string(expect_numel));
        }
        std::vector<float> out(numel);
        const float* fp = reinterpret_cast<const float*>(p);
        for (size_t i = 0; i < numel; ++i)
        {
            out[i] = fp[i];
        }
        return out;
    }

    // safetensors から FP32 射影重み [out,in] をロードし per-row INT8 量子化する。
    static QLinear load_qlinear(const SafeTensors& st, const std::string& name,
                                int out_dim, int in_dim)
    {
        const std::vector<float> w =
            load_f32(st, name, static_cast<size_t>(out_dim) * in_dim);
        const Int8RowQuant rq = quantize_weight_int8_perrow(
            w.data(), static_cast<size_t>(out_dim), static_cast<size_t>(in_dim));
        QLinear ql;
        ql.q       = rq.q;       // [out*in]
        ql.scale   = rq.scale;   // [out]
        ql.out_dim = out_dim;
        ql.in_dim  = in_dim;
        return ql;
    }

    // INT8 Linear: 活性 x[in_dim] を absmax int8 量子化 → 各 row int64 蓄積 →
    //   y[o] = scale[o] · x_scale · Σ_i ( qw[o,i] · xq[i] )。
    static void linear_int8(const QLinear& ql, const float* x, float* y)
    {
        const int out_dim = ql.out_dim;
        const int in_dim  = ql.in_dim;
        // 入力活性を int8 量子化 (行全体で 1 つの scale)。
        const Int8Quant xq =
            quantize_activation_int8(x, static_cast<size_t>(in_dim));
        for (int o = 0; o < out_dim; ++o)
        {
            const int8_t* qrow = ql.q.data() + static_cast<size_t>(o) * in_dim;
            long long acc = 0;
            for (int i = 0; i < in_dim; ++i)
            {
                acc += static_cast<long long>(qrow[i])
                       * static_cast<long long>(xq.q[static_cast<size_t>(i)]);
            }
            y[o] = static_cast<float>(static_cast<double>(acc)
                                      * static_cast<double>(ql.scale[static_cast<size_t>(o)])
                                      * static_cast<double>(xq.scale));
        }
    }

    // RoPE (GPT-NeoX 系・(i, i+half) ペア)。BitNetDenseInfer と同式。
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
    // 射影 Q/K/V/O のみ INT8、attention 本体は FP32 (BitNetDenseInfer と同構造)。
    void attention_block(const Layer& L, std::vector<float>& h, int S) const
    {
        const int D = D_MODEL;
        std::vector<float> x(static_cast<size_t>(S) * D);
        for (int s = 0; s < S; ++s)
        {
            rms_norm(&h[static_cast<size_t>(s) * D], L.attn_norm.data(),
                     &x[static_cast<size_t>(s) * D], D, RMS_EPS);
        }

        std::vector<float> Q(static_cast<size_t>(S) * D);
        std::vector<float> K(static_cast<size_t>(S) * D);
        std::vector<float> V(static_cast<size_t>(S) * D);
        for (int s = 0; s < S; ++s)
        {
            const float* xs = &x[static_cast<size_t>(s) * D];
            linear_int8(L.wq, xs, &Q[static_cast<size_t>(s) * D]);
            linear_int8(L.wk, xs, &K[static_cast<size_t>(s) * D]);
            linear_int8(L.wv, xs, &V[static_cast<size_t>(s) * D]);
        }

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

        std::vector<float> proj(D);
        for (int s = 0; s < S; ++s)
        {
            linear_int8(L.wo, &attn_out[static_cast<size_t>(s) * D], proj.data());
            for (int d = 0; d < D; ++d)
            {
                h[static_cast<size_t>(s) * D + d] += proj[d];
            }
        }
    }

    // SwiGLU FFN (pre-RMSNorm + residual)。射影 gate/up/down のみ INT8。
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
            linear_int8(L.w_gate, x.data(), g.data());
            linear_int8(L.w_up,   x.data(), u.data());
            for (int f = 0; f < F; ++f)
            {
                const double gv = static_cast<double>(g[f]);
                const double silu = gv / (1.0 + std::exp(-gv));
                inter[f] = static_cast<float>(silu * static_cast<double>(u[f]));
            }
            linear_int8(L.w_down, inter.data(), y.data());
            for (int d = 0; d < D; ++d)
            {
                h[static_cast<size_t>(s) * D + d] += y[d];
            }
        }
    }
};

} // namespace dollama

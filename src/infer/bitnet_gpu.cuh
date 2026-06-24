// BitNet dense (FP32) GPU 推論 — Phase 4 #6-GPU。
//   T1 (bitnet_gpu.cu) が device 経路 (重み常駐 + forward GPU 実行本体) を実装。
//   T2 (cpp-implementer) が class BitNetGpuInfer を完成させ public API を結線する。
//
// ──────────────────────────────────────────────────────────────────
// T1/T2 境界 (T1 提案):
//   - device 重み構造体 / 作業バッファ構造体は本ヘッダで共有 (下記)。
//     ポインタの cudaMalloc/cudaFree (所有) は T2 (クラス) 側で行う。T1 は計算のみ。
//   - forward の GPU 実行本体は bitnet_gpu_forward(...) (T1 実装)。
//   - 純 FP32 を担保するため cublasHandle は T2 が LM ローカルに 1 個生成し、
//     bitnet_gpu_forward に渡す (共有 kernels/gemm.cu のハンドルは使わない)。
//
// public API は CPU 版 BitNetDenseInfer と同形:
//   forward(host tokens) → host logits / generate / generate_from_text /
//   build_identity_prompt / generate_with_identity。
//   host↔device コピーはクラス内に隠蔽する。
//   重みローダは BitNetDenseInfer::load_f32 規約 (safetensors FP32・74 テンソル・
//   名前 embed / layers.N.{attn_norm,wq,wk,wv,wo,ffn_norm,w_gate,w_up,w_down} /
//   final_norm) を踏襲する (host へロード → cudaMemcpy で device 常駐)。
//
// ──────────────────────────────────────────────────────────────────
// __CUDACC__ ガードの理由:
//   class BitNetGpuInfer は safetensors.hpp / tokenizer.hpp を引くが、これらは
//   C++17 以降前提 (std::string::data() の非 const 版等)。一方 .cu (T1) は nvcc が
//   ホストパスを /std:c++14 でコンパイルする (CUDA 13.3 + MSVC の 0xC0000409 回避)
//   ため、これらヘッダを引くと C++14 非互換でエラーになる。.cu (= __CUDACC__) は
//   device 経路 (struct + bitnet_gpu_forward) だけ必要なので、class とその heavy
//   include は __CUDACC__ 未定義 (= cl が cpp として読む) のときだけ可視にする。
//   これで bitnet_gpu.cu は struct/forward 宣言のみ、利用側 (.cpp) は class が見える。
#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

// アーキ次元の単一情報源。純粋な constexpr / 整数マクロのみで構成されており、
// nvcc (.cu・/std:c++14) でも cl (.cpp) でも安全に引ける (heavy include 非依存)。
#include "models/bitnet_config.hpp"

namespace dollama
{

// ── アーキ定数 (CPU 版 BitNetDenseInfer / models/bitnet.hpp と一致) ──
//   値は bitnet_config.hpp (単一情報源) を引く。DOLLAMA_BITNET_ARCH で
//   default (33M) / d80m (80M) を切替・3 ファイル同値を保証する。
namespace bitnet_gpu_cfg
{
constexpr int    VOCAB_SIZE  = bitnet_arch::VOCAB_SIZE;
constexpr int    D_MODEL     = bitnet_arch::D_MODEL;
constexpr int    N_LAYERS    = bitnet_arch::N_LAYERS;
constexpr int    N_HEADS     = bitnet_arch::N_HEADS;
constexpr int    HEAD_DIM    = bitnet_arch::HEAD_DIM;  // 64
constexpr int    FFN_DIM     = bitnet_arch::FFN_DIM;
constexpr int    MAX_SEQ_LEN = bitnet_arch::MAX_SEQ_LEN;
constexpr double ROPE_BASE   = bitnet_arch::ROPE_BASE;
constexpr double RMS_EPS     = bitnet_arch::RMS_EPS;
} // namespace bitnet_gpu_cfg

// ── device 常駐重み (FP32・[out,in] row-major) ──────────────────────
//   ポインタ所有・cudaMalloc/cudaFree は T2 (クラス) 側。
struct BitNetGpuLayerDev
{
    float* attn_norm = nullptr;  // [D_MODEL]
    float* wq = nullptr;         // [D_MODEL, D_MODEL]
    float* wk = nullptr;         // [D_MODEL, D_MODEL]
    float* wv = nullptr;         // [D_MODEL, D_MODEL]
    float* wo = nullptr;         // [D_MODEL, D_MODEL]
    float* ffn_norm = nullptr;   // [D_MODEL]
    float* w_gate = nullptr;     // [FFN_DIM, D_MODEL]
    float* w_up = nullptr;       // [FFN_DIM, D_MODEL]
    float* w_down = nullptr;     // [D_MODEL, FFN_DIM]
};

struct BitNetGpuWeightsDev
{
    float*                          embed = nullptr;       // [VOCAB_SIZE, D_MODEL] (tied lm_head)
    float*                          final_norm = nullptr;  // [D_MODEL]
    std::vector<BitNetGpuLayerDev>  layers;
};

// ── forward 中間テンソル device バッファ (最大 seq で確保し使い回す) ──
//   cudaMalloc/cudaFree は T2 (クラス) 側。確保サイズの目安:
//     d_tokens : MAX_SEQ_LEN * sizeof(int)
//     d_h/d_x/d_Q/d_K/d_V/d_attn/d_proj : MAX_SEQ_LEN * D_MODEL * sizeof(float)
//     d_g/d_u/d_inter : MAX_SEQ_LEN * FFN_DIM * sizeof(float)
//     d_logits : MAX_SEQ_LEN * VOCAB_SIZE * sizeof(float)
struct BitNetGpuScratch
{
    int*   d_tokens = nullptr;
    float* d_h = nullptr;
    float* d_x = nullptr;
    float* d_Q = nullptr;
    float* d_K = nullptr;
    float* d_V = nullptr;
    float* d_attn = nullptr;
    float* d_proj = nullptr;
    float* d_g = nullptr;
    float* d_u = nullptr;
    float* d_inter = nullptr;
    float* d_logits = nullptr;
};

// ── T1 実装: GPU forward 本体 ───────────────────────────────────────
//   入力 host tokens [S] (各 id ∈ [0,VOCAB_SIZE)・S ∈ [1, MAX_SEQ_LEN])
//   → 出力 host logits [S * VOCAB_SIZE] (row-major、行 = 位置)。
//   handle は T2 が LM ローカルに生成した純 FP32 用 cuBLAS ハンドル。
//   w / sc は T2 が確保済みで渡す。host↔device コピーは本関数内で行う。
void bitnet_gpu_forward(cublasHandle_t handle,
                        const BitNetGpuWeightsDev& w,
                        BitNetGpuScratch& sc,
                        const int* host_tokens, int S,
                        float* host_logits_out);

// ────────────────────────────────────────────────────────────────────
// T2 本体: class BitNetGpuInfer (cl が cpp として読むときだけ可視)
//   safetensors FP32 を host へロード → device 常駐 + scratch 確保 + cuBLAS 生成。
//   public API は CPU 版 BitNetDenseInfer と同形・同挙動 (forward だけ GPU 経路)。
//   inline 定義 (ヘッダオンリー)。所有する device ptr はデストラクタで解放する。
// ────────────────────────────────────────────────────────────────────
#ifndef __CUDACC__

} // namespace dollama

#include "io/safetensors.hpp"
#include "io/tokenizer.hpp"

namespace dollama
{

class BitNetGpuInfer
{
public:
    static constexpr int VOCAB_SIZE  = bitnet_gpu_cfg::VOCAB_SIZE;
    static constexpr int D_MODEL     = bitnet_gpu_cfg::D_MODEL;
    static constexpr int N_LAYERS    = bitnet_gpu_cfg::N_LAYERS;
    static constexpr int N_HEADS     = bitnet_gpu_cfg::N_HEADS;
    static constexpr int HEAD_DIM    = bitnet_gpu_cfg::HEAD_DIM;
    static constexpr int FFN_DIM     = bitnet_gpu_cfg::FFN_DIM;
    static constexpr int MAX_SEQ_LEN = bitnet_gpu_cfg::MAX_SEQ_LEN;

    // safetensors FP32・74 テンソルを host へロード → device 常駐 + scratch 確保
    // + 純 FP32 用 cuBLAS ハンドル生成。失敗時は throw (部分確保は free_all で解放)。
    explicit BitNetGpuInfer(const std::string& weights_path)
    {
        try
        {
            load_weights_to_device(weights_path);
            alloc_scratch();
            cublasStatus_t st = cublasCreate(&handle_);
            if (st != CUBLAS_STATUS_SUCCESS)
            {
                throw std::runtime_error(
                    "BitNetGpuInfer: cublasCreate failed ("
                    + std::to_string(static_cast<int>(st)) + ")");
            }
        }
        catch (...)
        {
            // コンストラクタで投げる場合デストラクタは呼ばれないため、ここで解放。
            free_all();
            throw;
        }
    }

    // device ptr 全解放 + cuBLAS 破棄。例外は投げない。
    ~BitNetGpuInfer()
    {
        free_all();
    }

    // コピー禁止 (device ptr 所有のため)。
    BitNetGpuInfer(const BitNetGpuInfer&) = delete;
    BitNetGpuInfer& operator=(const BitNetGpuInfer&) = delete;

    // ── GPU forward (host tokens → host logits) ────────────────────
    // 入力検証 (空 / MAX_SEQ 超過 / range) は CPU 版と厳密一致。
    std::vector<float> forward(const std::vector<int>& tokens) const
    {
        const int S = static_cast<int>(tokens.size());
        if (S <= 0)
        {
            throw std::runtime_error("BitNetGpuInfer::forward: empty sequence");
        }
        if (S > MAX_SEQ_LEN)
        {
            throw std::runtime_error("BitNetGpuInfer::forward: seq exceeds MAX_SEQ_LEN");
        }
        for (int t : tokens)
        {
            if (t < 0 || t >= VOCAB_SIZE)
            {
                throw std::runtime_error("BitNetGpuInfer::forward: token id out of range");
            }
        }

        std::vector<float> out(static_cast<size_t>(S) * VOCAB_SIZE);
        bitnet_gpu_forward(handle_, w_, sc_, tokens.data(), S, out.data());
        return out;
    }

    // ── greedy デコード (CPU 版 generate と同挙動・forward だけ GPU) ──
    //   最終位置 argmax・tie は小 id・<eos>(=2) 停止/除外・MAX_SEQ 打切。
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
                // tie は小さい id を優先するため、厳密に大きいときだけ更新。
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

    // ── 同一性条件付き prompt 組み立て (Phase 4 A / A3) ─────────────────
    //   <bos> [identity ids] <sep> [scene greedy タグ] <sep>。CPU 版と同手順。
    std::vector<int> build_identity_prompt(
        const Tokenizer& tok,
        const std::vector<std::string>& identity_tags,
        const std::string& scene_text) const
    {
        std::vector<int> prompt;
        prompt.push_back(TOK_BOS);

        // identity 区間: 各タグを tag_to_id で id 化して直接積む。
        for (const std::string& t : identity_tags)
        {
            prompt.push_back(tok.tag_to_id(t));
        }
        prompt.push_back(TOK_SEP);

        // scene 区間: 自然文を greedy 最長一致でタグ化。<bos>/<eos> を剥がす。
        std::vector<int> framed = tok.encode_text(scene_text, MAX_SEQ_LEN);
        for (size_t i = 1; i + 1 < framed.size(); ++i)
        {
            prompt.push_back(framed[i]);
        }
        prompt.push_back(TOK_SEP);

        return prompt;
    }

    // 同一性条件付き生成: prompt を組んで既存 greedy デコードへ流す便宜関数。
    std::vector<int> generate_with_identity(
        const Tokenizer& tok,
        const std::vector<std::string>& identity_tags,
        const std::string& scene_text) const
    {
        return generate(build_identity_prompt(tok, identity_tags, scene_text));
    }

private:
    cublasHandle_t           handle_ = nullptr;  // 純 FP32 用 LM ローカルハンドル
    BitNetGpuWeightsDev      w_;
    mutable BitNetGpuScratch sc_;                // forward が書き換えるため mutable

    // ── cuda エラーチェック (throw 版) ─────────────────────────────
    static void cuda_check(cudaError_t err, const char* what)
    {
        if (err != cudaSuccess)
        {
            throw std::runtime_error(std::string("BitNetGpuInfer CUDA error: ")
                                     + cudaGetErrorString(err) + " (" + what + ")");
        }
    }

    // safetensors から F32 テンソルを host にロード (BitNetDenseInfer::load_f32 規約)。
    static std::vector<float> load_f32(const SafeTensors& st,
                                       const std::string& name,
                                       size_t expect_numel)
    {
        if (st.dtype(name) != StDtype::F32)
        {
            throw std::runtime_error("BitNetGpuInfer: '" + name + "' must be F32");
        }
        size_t nbytes = 0;
        const uint8_t* p = st.tensor_bytes(name, nbytes);
        const size_t numel = nbytes / sizeof(float);
        if (numel != expect_numel)
        {
            throw std::runtime_error("BitNetGpuInfer: '" + name + "' numel="
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

    // host float ベクタを device に確保してコピーし、その device ptr を返す。
    float* upload(const std::vector<float>& host)
    {
        float* d = nullptr;
        const size_t bytes = host.size() * sizeof(float);
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d), bytes),
                   "cudaMalloc weight");
        cuda_check(cudaMemcpy(d, host.data(), bytes, cudaMemcpyHostToDevice),
                   "cudaMemcpy weight H2D");
        return d;
    }

    // 74 テンソルを host ロード → device 常駐 (w_)。
    void load_weights_to_device(const std::string& weights_path)
    {
        using namespace bitnet_gpu_cfg;
        SafeTensors st(weights_path);

        w_.embed = upload(load_f32(st, "embed",
                                   static_cast<size_t>(VOCAB_SIZE) * D_MODEL));
        w_.final_norm = upload(load_f32(st, "final_norm", D_MODEL));

        w_.layers.resize(N_LAYERS);
        for (int i = 0; i < N_LAYERS; ++i)
        {
            const std::string p = "layers." + std::to_string(i) + ".";
            BitNetGpuLayerDev& L = w_.layers[static_cast<size_t>(i)];
            L.attn_norm = upload(load_f32(st, p + "attn_norm", D_MODEL));
            L.wq = upload(load_f32(st, p + "wq",
                                   static_cast<size_t>(D_MODEL) * D_MODEL));
            L.wk = upload(load_f32(st, p + "wk",
                                   static_cast<size_t>(D_MODEL) * D_MODEL));
            L.wv = upload(load_f32(st, p + "wv",
                                   static_cast<size_t>(D_MODEL) * D_MODEL));
            L.wo = upload(load_f32(st, p + "wo",
                                   static_cast<size_t>(D_MODEL) * D_MODEL));
            L.ffn_norm = upload(load_f32(st, p + "ffn_norm", D_MODEL));
            L.w_gate = upload(load_f32(st, p + "w_gate",
                                       static_cast<size_t>(FFN_DIM) * D_MODEL));
            L.w_up = upload(load_f32(st, p + "w_up",
                                     static_cast<size_t>(FFN_DIM) * D_MODEL));
            L.w_down = upload(load_f32(st, p + "w_down",
                                       static_cast<size_t>(D_MODEL) * FFN_DIM));
        }
    }

    // forward 作業バッファを MAX_SEQ_LEN 基準で確保 (sc_)。
    void alloc_scratch()
    {
        using namespace bitnet_gpu_cfg;
        const size_t S   = MAX_SEQ_LEN;
        const size_t SD  = S * D_MODEL;
        const size_t SF  = S * FFN_DIM;
        const size_t SV  = S * VOCAB_SIZE;

        auto malloc_f = [this](float** p, size_t n)
        {
            cuda_check(cudaMalloc(reinterpret_cast<void**>(p), n * sizeof(float)),
                       "cudaMalloc scratch");
        };

        cuda_check(cudaMalloc(reinterpret_cast<void**>(&sc_.d_tokens),
                              S * sizeof(int)),
                   "cudaMalloc d_tokens");
        malloc_f(&sc_.d_h, SD);
        malloc_f(&sc_.d_x, SD);
        malloc_f(&sc_.d_Q, SD);
        malloc_f(&sc_.d_K, SD);
        malloc_f(&sc_.d_V, SD);
        malloc_f(&sc_.d_attn, SD);
        malloc_f(&sc_.d_proj, SD);
        malloc_f(&sc_.d_g, SF);
        malloc_f(&sc_.d_u, SF);
        malloc_f(&sc_.d_inter, SF);
        malloc_f(&sc_.d_logits, SV);
    }

    // 全 device ptr / cuBLAS ハンドルを解放 (デストラクタ・コンストラクタ失敗時共用)。
    // 例外を投げない (cudaFree 結果は破棄)。
    void free_all() noexcept
    {
        // 重み。
        cudaFree(w_.embed);      w_.embed = nullptr;
        cudaFree(w_.final_norm); w_.final_norm = nullptr;
        for (BitNetGpuLayerDev& L : w_.layers)
        {
            cudaFree(L.attn_norm);
            cudaFree(L.wq);
            cudaFree(L.wk);
            cudaFree(L.wv);
            cudaFree(L.wo);
            cudaFree(L.ffn_norm);
            cudaFree(L.w_gate);
            cudaFree(L.w_up);
            cudaFree(L.w_down);
        }
        w_.layers.clear();

        // scratch。
        cudaFree(sc_.d_tokens); sc_.d_tokens = nullptr;
        cudaFree(sc_.d_h);      sc_.d_h = nullptr;
        cudaFree(sc_.d_x);      sc_.d_x = nullptr;
        cudaFree(sc_.d_Q);      sc_.d_Q = nullptr;
        cudaFree(sc_.d_K);      sc_.d_K = nullptr;
        cudaFree(sc_.d_V);      sc_.d_V = nullptr;
        cudaFree(sc_.d_attn);   sc_.d_attn = nullptr;
        cudaFree(sc_.d_proj);   sc_.d_proj = nullptr;
        cudaFree(sc_.d_g);      sc_.d_g = nullptr;
        cudaFree(sc_.d_u);      sc_.d_u = nullptr;
        cudaFree(sc_.d_inter);  sc_.d_inter = nullptr;
        cudaFree(sc_.d_logits); sc_.d_logits = nullptr;

        // cuBLAS。
        if (handle_ != nullptr)
        {
            cublasDestroy(handle_);
            handle_ = nullptr;
        }
    }
};

#endif // __CUDACC__

} // namespace dollama

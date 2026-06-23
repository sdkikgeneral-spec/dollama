#pragma once

// SDXL テキスト条件付け (txt2img の prompt → UNet 入力埋め込み)
//
// 役割: prompt / negative_prompt の 2 文字列を受け、SDXL の dual encoder 経由で
//   ① encoder_hidden_states [1,77,2048] (= CLIP-L penultimate 768 ++ bigG penultimate 1280)
//   ② text_embeds           [1,1280]    (= bigG pooled)
//   ③ time_ids              [6]         (= original/crop/target サイズの定数)
// の cond / uncond 2 セットを host float で構築し、diffusion.cu (E) に渡す。
//
// 確定レシピ (Stage A golden):
//   encoder_hidden_states[...,   :768] = CLIP-L penultimate
//   encoder_hidden_states[..., 768:  ] = bigG  penultimate
//   text_embeds                        = bigG  pooled
//   time_ids = (orig_h, orig_w, crop_top, crop_left, target_h, target_w)
//            = (1024, 1024, 0, 0, 1024, 1024)
//
// デバイス選定 (Stage C) は並行進行中のため、各エンコーダ/トークナイザのデバイスは
// パラメータ化する。bigG は当面 "NPU" を第一候補とし、後で差し替え可能にする。

#ifdef HAVE_OPENVINO

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "infer/clip_encoder2.hpp"
#include "infer/clip_tokenizer.hpp"

namespace dollama
{

// ------------------------------------------------------------
// SDXL dual encoder の確定定数
// ------------------------------------------------------------
constexpr int kSdxlSeqLen      = 77;
constexpr int kSdxlHiddenL     = 768;
constexpr int kSdxlHiddenG     = 1280;
constexpr int kSdxlHiddenTotal = kSdxlHiddenL + kSdxlHiddenG;  // 2048
constexpr int kSdxlPooledDim   = 1280;
constexpr int kSdxlTimeIdsDim  = 6;

// ------------------------------------------------------------
// 1 プロンプト分の条件付け結果 (cond または uncond)。
//   encoder_hidden_states: [SEQ_LEN * 2048] row-major ([s][2048] の順)
//   text_embeds:           [1280]
//   token_ids_l / _g:      [77] (検証・デバッグ用に保持)
// ------------------------------------------------------------
struct TextConditioningSide
{
    std::vector<float>   encoder_hidden_states;  // [77 * 2048]
    std::vector<float>   text_embeds;            // [1280]
    std::vector<int64_t> token_ids_l;            // [77]
    std::vector<int64_t> token_ids_g;            // [77]
};

// ------------------------------------------------------------
// cond / uncond 両方 + 共有の time_ids をまとめた最終出力構造体。
// これが diffusion.cu (E) の消費 API。
// ------------------------------------------------------------
struct TextConditioning
{
    TextConditioningSide cond;     // prompt 側
    TextConditioningSide uncond;   // negative_prompt 側
    std::vector<float>   time_ids; // [6] (cond/uncond 共通)
};

// ------------------------------------------------------------
// TextConditioner: 2 トークナイザ + 2 エンコーダを保持し、
// prompt / negative_prompt → TextConditioning を構築する。
// ------------------------------------------------------------
class TextConditioner
{
public:
    // 各モデルパスとデバイスをパラメータで受ける (Stage C のデバイス差し替えに対応)。
    // tokenizers_dll: openvino_tokenizers.dll の絶対パス。
    // device_l / device_g: CLIP-L / bigG エンコーダの実行デバイス。
    //   bigG は当面 "NPU" 第一候補。CLIP-L は test_clip 実績どおり "NPU" 既定。
    TextConditioner(const std::string& tokenizer_l_xml,
                    const std::string& tokenizer_g_xml,
                    const std::string& encoder_l_xml,
                    const std::string& encoder_g_xml,
                    const std::string& tokenizers_dll,
                    const std::string& device_l = "NPU",
                    const std::string& device_g = "NPU")
        : tok_l_(tokenizer_l_xml, tokenizers_dll),
          tok_g_(tokenizer_g_xml, tokenizers_dll),
          enc_l_(encoder_l_xml, kSdxlHiddenL, /*has_pooled=*/false, device_l),
          enc_g_(encoder_g_xml, kSdxlHiddenG, /*has_pooled=*/true,  device_g)
    {
    }

    // prompt と negative_prompt を条件付けする。
    // time_ids はデフォルトで (1024,1024,0,0,1024,1024)。
    TextConditioning encode(const std::string& prompt,
                            const std::string& negative_prompt,
                            int                target_h = 1024,
                            int                target_w = 1024)
    {
        TextConditioning out;
        out.cond   = encode_side(prompt);
        out.uncond = encode_side(negative_prompt);

        // time_ids = (orig_h, orig_w, crop_top, crop_left, target_h, target_w)
        out.time_ids = {
            static_cast<float>(target_h), static_cast<float>(target_w),
            0.0f, 0.0f,
            static_cast<float>(target_h), static_cast<float>(target_w),
        };
        return out;
    }

private:
    // 1 プロンプトを tokenize(L,G) → encode(L,G) → concat する。
    TextConditioningSide encode_side(const std::string& prompt)
    {
        TextConditioningSide side;

        // トークナイズ (L/G で pad 挙動が違うため別モデル)。
        side.token_ids_l = tok_l_.tokenize(prompt);
        side.token_ids_g = tok_g_.tokenize(prompt);

        // エンコード。
        std::vector<float> penult_l, dummy_pool_l;
        std::vector<float> penult_g, pooled_g;
        enc_l_.infer(side.token_ids_l, penult_l, dummy_pool_l);  // [77*768]
        enc_g_.infer(side.token_ids_g, penult_g, pooled_g);      // [77*1280], pooled [1280]

        // concat: 各トークン s について [L(768) | G(1280)] を並べて [77*2048]。
        side.encoder_hidden_states.resize(
            static_cast<size_t>(kSdxlSeqLen) * static_cast<size_t>(kSdxlHiddenTotal));
        for (int s = 0; s < kSdxlSeqLen; ++s)
        {
            float*       row = side.encoder_hidden_states.data()
                             + static_cast<size_t>(s) * kSdxlHiddenTotal;
            const float* lr  = penult_l.data() + static_cast<size_t>(s) * kSdxlHiddenL;
            const float* gr  = penult_g.data() + static_cast<size_t>(s) * kSdxlHiddenG;

            for (int c = 0; c < kSdxlHiddenL; ++c)
            {
                row[c] = lr[c];
            }
            for (int c = 0; c < kSdxlHiddenG; ++c)
            {
                row[kSdxlHiddenL + c] = gr[c];
            }
        }

        // text_embeds = bigG pooled。
        side.text_embeds = std::move(pooled_g);
        return side;
    }

    ClipTokenizer   tok_l_;
    ClipTokenizer   tok_g_;
    DualClipEncoder enc_l_;
    DualClipEncoder enc_g_;
};

} // namespace dollama

#endif // HAVE_OPENVINO

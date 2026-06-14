#pragma once

// CLIP text encoder (OpenVINO C++ API)
// NPU 推論: [1, 77] int64 → [1, 77, 768] float32
// 計測ベースライン: NPU 7.85ms (probe9)

#ifdef HAVE_OPENVINO

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <openvino/openvino.hpp>

namespace dollama {

// ------------------------------------------------------------
// ClipEncoder: CLIP-L text encoder の OpenVINO 推論ラッパー
// モデル入力:  [1, SEQ_LEN] int64 (token ids)
// モデル出力:  [1, SEQ_LEN, HIDDEN] float32 (hidden states)
// ------------------------------------------------------------
class ClipEncoder
{
public:
    // CLIP-L 固定定数
    static constexpr int SEQ_LEN = 77;
    static constexpr int HIDDEN  = 768;

    // コンストラクタ: モデルをロードしてデバイスにコンパイルする
    // model_xml: OV IR の .xml ファイルパス
    // device:    "NPU" / "CPU" / "GPU.0" など
    explicit ClipEncoder(const std::string& model_xml,
                         const std::string& device = "NPU")
    {
        // モデルを読み込んでコンパイル
        auto ov_model = core_.read_model(model_xml);
        compiled_      = core_.compile_model(ov_model, device);
        request_       = compiled_.create_infer_request();
    }

    // 推論メソッド
    // ids: [SEQ_LEN] の token id 列 (BOS / content / EOS / PAD=0)
    // 戻り値: SEQ_LEN * HIDDEN = 59136 要素の float ベクタ
    std::vector<float> infer(const std::array<int32_t, SEQ_LEN>& ids)
    {
        // OV が所有するバッファにコピーして渡す
        // (NPU プラグインが入力バッファに書き込む場合があるため、スタック変数の生ポインタは渡さない)
        //
        // モデルの input_ids は element_type="i64" shape [1,77] で静的。
        // i32 を渡すと NPU プラグインが要素あたり 8 バイト読もうとして
        // 領域外読み出し (0xC0000409) を起こすため、必ず i64 で生成する。
        ov::Tensor input_tensor(ov::element::i64, {1, static_cast<size_t>(SEQ_LEN)});

        // int32 の各要素を int64 へ明示変換しながら i64 バッファへコピーする
        int64_t* dst = input_tensor.data<int64_t>();
        for (int i = 0; i < SEQ_LEN; ++i)
        {
            dst[i] = static_cast<int64_t>(ids[i]); // i32 → i64 への明示拡張変換
        }

        request_.set_input_tensor(input_tensor);
        request_.infer();

        // 出力0: last_hidden_state [1, SEQ_LEN, HIDDEN]
        auto         output = request_.get_output_tensor(0);
        const float* ptr    = output.data<float>();
        return std::vector<float>(ptr, ptr + SEQ_LEN * HIDDEN);
    }

private:
    ov::Core           core_;
    ov::CompiledModel  compiled_;
    ov::InferRequest   request_;
};

} // namespace dollama

#endif // HAVE_OPENVINO

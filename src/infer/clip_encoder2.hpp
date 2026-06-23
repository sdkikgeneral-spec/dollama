#pragma once

// SDXL dual text encoder 用の汎用 CLIP エンコーダラッパー (OpenVINO C++ API)
//
// 既存の clip.hpp (ClipEncoder) は CLIP-L の last_hidden_state(出力0)・HIDDEN=768 固定で、
// test_clip がそれに依存している。本ファイルはそれを壊さず、SDXL の
// dual encoder (CLIP-L penultimate [1,77,768] / CLIP-bigG penultimate [1,77,1280] + pooled
// [1,1280]) に対応する別クラスを提供する。
//
// Stage B が用意した OV アセット (確定仕様):
//   - sdxl-text-encoder-l: 入力 input_ids i64 [1,77]、出力 OUT[0]="penultimate" [1,77,768] f32 のみ
//   - sdxl-text-encoder-g: 入力 input_ids i64 [1,77]、出力 OUT[0]="penultimate" [1,77,1280] f32 /
//                          OUT[1]="pooled" [1,1280] f32
//
// 型厳守: 入力 input_ids は i64。i32 を渡すと NPU が要素あたり 8 バイト読もうとして
//         領域外読み出し (0xC0000409) を起こす (タスク5 の既知事例)。token id は int64 へ明示変換。

#ifdef HAVE_OPENVINO

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <openvino/openvino.hpp>

namespace dollama
{

// ------------------------------------------------------------
// DualClipEncoder: SDXL の片側 text encoder (L または G) の推論ラッパー
//   入力:  [1, SEQ_LEN] int64 (token ids)
//   出力0: penultimate  [1, SEQ_LEN, hidden] float32
//   出力1: pooled       [1, hidden]          float32 (G のみ。has_pooled=true)
// HIDDEN / pooled の有無 / デバイスをコンストラクタで設定可能に一般化する。
// ------------------------------------------------------------
class DualClipEncoder
{
public:
    static constexpr int SEQ_LEN = 77;

    // model_xml:   OV IR の .xml パス
    // hidden:      penultimate の最終次元 (L=768 / G=1280)
    // has_pooled:  出力1 に pooled [1,hidden] を持つか (G=true / L=false)
    // device:      "NPU" / "CPU" / "GPU.0" など (Stage C のデバイス選定でパラメータ化)
    DualClipEncoder(const std::string& model_xml,
                    int                hidden,
                    bool               has_pooled,
                    const std::string& device)
        : hidden_(hidden), has_pooled_(has_pooled), device_(device)
    {
        auto ov_model = core_.read_model(model_xml);
        compiled_     = core_.compile_model(ov_model, device);
        request_      = compiled_.create_infer_request();
    }

    int  hidden() const { return hidden_; }
    bool has_pooled() const { return has_pooled_; }
    const std::string& device() const { return device_; }

    // 推論本体。
    // ids:               [SEQ_LEN] の token id 列 (i64 へ拡張して渡す)
    // out_penultimate:   [SEQ_LEN * hidden] へリサイズして書き込む
    // out_pooled:        has_pooled_ のとき [hidden] へリサイズして書き込む。L では空のまま。
    void infer(const std::vector<int64_t>& ids,
               std::vector<float>&          out_penultimate,
               std::vector<float>&          out_pooled)
    {
        if (static_cast<int>(ids.size()) != SEQ_LEN)
        {
            throw std::runtime_error("DualClipEncoder::infer: ids のサイズが SEQ_LEN ではありません");
        }

        // OV が所有する i64 バッファへコピーして渡す (IR の element_type=i64 に厳密一致)。
        ov::Tensor input_tensor(ov::element::i64, {1, static_cast<size_t>(SEQ_LEN)});
        int64_t*   dst = input_tensor.data<int64_t>();
        for (int i = 0; i < SEQ_LEN; ++i)
        {
            dst[i] = ids[i];
        }

        request_.set_input_tensor(input_tensor);
        request_.infer();

        // 出力0: penultimate [1, SEQ_LEN, hidden]
        const auto   penult = request_.get_output_tensor(0);
        const float* pptr   = penult.data<float>();
        const size_t pcount = static_cast<size_t>(SEQ_LEN) * static_cast<size_t>(hidden_);
        out_penultimate.assign(pptr, pptr + pcount);

        // 出力1: pooled [1, hidden] (G のみ)
        if (has_pooled_)
        {
            const auto   pooled = request_.get_output_tensor(1);
            const float* qptr   = pooled.data<float>();
            out_pooled.assign(qptr, qptr + static_cast<size_t>(hidden_));
        }
        else
        {
            out_pooled.clear();
        }
    }

private:
    ov::Core          core_;
    ov::CompiledModel compiled_;
    ov::InferRequest  request_;
    int               hidden_;
    bool              has_pooled_;
    std::string       device_;
};

} // namespace dollama

#endif // HAVE_OPENVINO

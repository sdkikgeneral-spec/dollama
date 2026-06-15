#pragma once

// WD14 SwinV2 Tagger v3 (OpenVINO C++ API)
// CPU 推論: [1, 448, 448, 3] float32 (NHWC, 0-255 レンジ) → [1, 10861] float32 (sigmoid 済みスコア 0-1)
// 計測ベースライン: CPU 101ms (probe8) ※CPU が iGPU/NPU より速い (Window Attention が NPU 不向き)
//
// モデル入出力仕様 (OV IR 実測):
//   入力  "input"  : element_type f32, shape [1, 448, 448, 3] (NHWC 静的) → reshape 不要
//   出力  "output" : element_type f32, shape [1, 10861]       (各タグの sigmoid 済みスコア 0-1)
//
// 前処理 (probe8 準拠): uint8 (0-255) → float32 へキャストのみ。正規化なし。

#ifdef HAVE_OPENVINO

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <openvino/openvino.hpp>

namespace dollama {

// ------------------------------------------------------------
// Wd14Tagger: WD14 SwinV2 tagger の OpenVINO 推論ラッパー
// モデル入力:  [1, INPUT_SIZE, INPUT_SIZE, CHANNELS] float32 (NHWC, 0-255)
// モデル出力:  [1, N_TAGS] float32 (sigmoid 済みスコア)
// ------------------------------------------------------------
class Wd14Tagger
{
public:
    // WD14 SwinV2 v3 固定定数
    static constexpr int INPUT_SIZE = 448; // 入力解像度 (固定形状)
    static constexpr int CHANNELS   = 3;   // RGB
    // N_TAGS: selected_tags.csv のタグ行数 (header 除く 10861 行) と一致。
    // ※ テストでの真の期待値は IR 出力の実 size (get_output_tensor(0).get_size()) を用いる。
    static constexpr int N_TAGS = 10861;

    // 1 枚分の入力要素数 = 448 * 448 * 3 = 602112
    static constexpr size_t INPUT_NUMEL =
        static_cast<size_t>(INPUT_SIZE) * INPUT_SIZE * CHANNELS;

    // コンストラクタ: モデルをロードしてデバイスにコンパイルする
    // model_xml: OV IR の .xml ファイルパス
    // device:    "CPU" (WD14 は CPU が最速) / "GPU.0" / "NPU"
    explicit Wd14Tagger(const std::string& model_xml,
                        const std::string& device = "CPU")
    {
        // IR は既に静的形状 [1,448,448,3] のため reshape 不要。
        auto ov_model = core_.read_model(model_xml);
        compiled_     = core_.compile_model(ov_model, device);
        request_      = compiled_.create_infer_request();
    }

    // 推論メソッド
    // pixels: [INPUT_SIZE * INPUT_SIZE * CHANNELS] = 602112 要素の float (NHWC, 0-255 レンジ)
    // 戻り値: N_TAGS 要素の float ベクタ (各タグの sigmoid 済みスコア 0-1)
    std::vector<float> infer(const std::vector<float>& pixels)
    {
        // 入力サイズ検証 (CLIP の i32/i64 教訓: 要素型・要素数は IR と厳密一致させる)
        if (pixels.size() != INPUT_NUMEL)
        {
            throw std::invalid_argument(
                "Wd14Tagger::infer: 入力サイズ不正 (期待 " +
                std::to_string(INPUT_NUMEL) + ", 実 " +
                std::to_string(pixels.size()) + ")");
        }

        // OV が所有する f32 テンソルを生成してコピーで渡す。
        // IR の入力 element_type は f32。uint8/i64 を渡してはならない。
        ov::Tensor input_tensor(
            ov::element::f32,
            {1, static_cast<size_t>(INPUT_SIZE),
             static_cast<size_t>(INPUT_SIZE),
             static_cast<size_t>(CHANNELS)});

        float* dst = input_tensor.data<float>();
        std::copy(pixels.begin(), pixels.end(), dst);

        request_.set_input_tensor(input_tensor);
        request_.infer();

        // 出力0: [1, N_TAGS]
        auto         output = request_.get_output_tensor(0);
        const float* ptr    = output.data<float>();
        return std::vector<float>(ptr, ptr + output.get_size());
    }

    // 軽量ヘルパ: スコア上位 K 件のインデックスを返す (CSV 非依存)
    // scores: infer() の戻り値。降順 (スコア大きい順) で k 件返す。
    static std::vector<size_t> top_k(const std::vector<float>& scores, size_t k)
    {
        const size_t n = scores.size();
        if (k > n)
        {
            k = n;
        }

        std::vector<size_t> idx(n);
        for (size_t i = 0; i < n; ++i)
        {
            idx[i] = i;
        }

        // 上位 k 件だけを部分ソート (降順)
        std::partial_sort(
            idx.begin(), idx.begin() + k, idx.end(),
            [&scores](size_t a, size_t b)
            {
                return scores[a] > scores[b];
            });

        idx.resize(k);
        return idx;
    }

private:
    ov::Core          core_;
    ov::CompiledModel compiled_;
    ov::InferRequest  request_;
};

} // namespace dollama

#endif // HAVE_OPENVINO

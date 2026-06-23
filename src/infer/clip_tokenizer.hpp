#pragma once

// SDXL CLIP トークナイザの OpenVINO ラッパー
//
// Stage B が用意した OV トークナイザ (openvino_tokenizers 拡張で読む):
//   - sdxl-tokenizer-l: pad=eos=49407
//   - sdxl-tokenizer-g: pad=0
//   - 両者: 入力 string [?] (バッチ次元動的)、出力 "input_ids" i64 [?,77] + "attention_mask" i64 [?,77]
//
// 重要: これらの IR は openvino_tokenizers の独自 opset を含むため、
//       core.add_extension("openvino_tokenizers.dll") を read_model 前に呼ばないと
//       unsupported opset で read_model が失敗する。DLL の場所は呼び出し側から渡す。
//
// 出力 input_ids は i64 [1,77]。これをそのまま DualClipEncoder へ渡す (型一致で 0xC0000409 回避)。

#ifdef HAVE_OPENVINO

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <openvino/openvino.hpp>

namespace dollama
{

// ------------------------------------------------------------
// ClipTokenizer: 1 つの OV トークナイザモデル (L または G) のラッパー
//   prompt 文字列 → [SEQ_LEN] int64 token id 列
// ------------------------------------------------------------
class ClipTokenizer
{
public:
    static constexpr int SEQ_LEN = 77;

    // tokenizer_xml:    sdxl-tokenizer-{l,g}/openvino_tokenizer.xml
    // tokenizers_dll:   openvino_tokenizers.dll の絶対パス (add_extension に渡す)
    // device:           トークナイズは軽量なので既定 "CPU" (NPU/iGPU 不要)
    ClipTokenizer(const std::string& tokenizer_xml,
                  const std::string& tokenizers_dll,
                  const std::string& device = "CPU")
    {
        // 独自 opset を解決するため拡張 DLL を先に登録する。
        core_.add_extension(tokenizers_dll);

        auto ov_model = core_.read_model(tokenizer_xml);
        compiled_     = core_.compile_model(ov_model, device);
        request_      = compiled_.create_infer_request();

        // 出力ポートのうち "input_ids" のインデックスを特定する
        // (出力順は input_ids / attention_mask だが、名前で引いて取り違えを防ぐ)。
        input_ids_index_ = 0;
        const auto& outs = compiled_.outputs();
        for (size_t i = 0; i < outs.size(); ++i)
        {
            const auto& names = outs[i].get_names();
            if (names.count("input_ids") != 0)
            {
                input_ids_index_ = i;
                break;
            }
        }
    }

    // prompt をトークナイズして [SEQ_LEN] の i64 token id 列を返す。
    std::vector<int64_t> tokenize(const std::string& prompt)
    {
        // 入力は string テンソル [1] (バッチ 1)。
        ov::Tensor in(ov::element::string, {1});
        in.data<std::string>()[0] = prompt;

        request_.set_input_tensor(in);
        request_.infer();

        const auto out = request_.get_output_tensor(input_ids_index_);

        // 出力 shape は [1, 77] (静的化済の想定だが念のため要素数で検証)。
        const size_t numel = out.get_size();
        if (numel != static_cast<size_t>(SEQ_LEN))
        {
            throw std::runtime_error(
                "ClipTokenizer::tokenize: input_ids の要素数が " +
                std::to_string(SEQ_LEN) + " ではありません (got " +
                std::to_string(numel) + ")");
        }

        const int64_t* ptr = out.data<int64_t>();
        return std::vector<int64_t>(ptr, ptr + SEQ_LEN);
    }

private:
    ov::Core          core_;
    ov::CompiledModel compiled_;
    ov::InferRequest  request_;
    size_t            input_ids_index_ = 0;
};

} // namespace dollama

#endif // HAVE_OPENVINO

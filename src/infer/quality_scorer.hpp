#pragma once

// ScorerNet 品質スコアラ — OpenVINO C++ API グルー (Phase 4 Model B / B-3d)
//
// B-3b で蒸留訓練した純 conv backbone (ResNet-18 級 11.18M・attention 皆無) を、
// B-3c で OV IR 静的化したものを C++ から推論する。matting.hpp の Matter と対称設計。
//
// モデル入出力仕様 (B-3c / golden_scorernet_meta.json 実測):
//   入力  "x"        : element_type f32, shape [1, 3, 512, 512] (NCHW 静的) → reshape 不要
//   出力  "output.0" : element_type f32, shape [1, 9] = logits (sigmoid 前)
//     index0    = quality logit (美的品質。B-3b では teacher null ゆえ head 凍結中)
//     index1..8 = 解剖 8 軸 logit (AnomalyAxis 順: Hands/Limbs/Head/Eyes/Ears/Mouth/Digits/GlobalAnatomy)
//
// 前処理 (train_scorer 厳守): RGB uint8 (0-255) → float32 へ /255 のみ (追加正規化なし)、レンジ [0,1]。
//   入力レイアウトは NCHW (R 面 → G 面 → B 面)。RGB であり BGR ではない。
//
// 計測ベースライン (B-3c 実測・512² 中央値): NPU 8.32ms / iGPU(GPU.0) 6.28ms / CPU 11.7ms。
//   純 conv ゆえ NPU フレンドリー (WD14 window attention 268ms と対照)。
//
// 設計判断 (matting.hpp に倣う):
//   - device 引数 (既定 "NPU")。拡散中 (RTX5080 が UNet で数秒占有) に遊休 NPU で並列採点する
//     のが本スコアラの狙い。iGPU(GPU.0) はマッティングが専有するため NPU を既定にする。
//     ただし golden の正は FP32 IR の CPU 出力ゆえ、テストは "CPU" 指定で突合する。
//   - 入力 element_type は IR と厳密一致 (f32)。CLIP の i32/i64 取り違えで 0xC0000409
//     (STATUS_STACK_BUFFER_OVERRUN) クラッシュした教訓に倣い、必ず ov::element::f32 で生成する。
//   - 入力解像度は 512 固定前提。≠512 の RGB が来たら最近傍リサイズで 512 に整える
//     (golden は 512 前処理済みなので、テスト経路ではリサイズは発火しない)。
//   - HAVE_OPENVINO 未定義時は OV 依存をすべて閉じる (この開発 PC は OV C++ 無し → コンパイルは通る)。
//
// logits → 確率変換ヘルパ (logits_to_result) は OV 非依存の純 C++ なので HAVE_OPENVINO の外に置く
// (OV 不在の開発機でも sigmoid 変換テストを実走できるようにするため。matting.hpp の compose_rgba と同方針)。

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "infer/quality_gate.hpp" // AnomalyAxis を再利用

namespace dollama
{

// ------------------------------------------------------------
// ScorerResult: ScorerNet 出力 (sigmoid 済み確率 [0,1])
//   quality  : 美的品質確率 (index0)
//   axis[0..7]: 解剖 8 軸の異常確率 (AnomalyAxis 順)
//     axis[0]=Hands / axis[1]=Limbs / axis[2]=Head / axis[3]=Eyes /
//     axis[4]=Ears  / axis[5]=Mouth / axis[6]=Digits / axis[7]=GlobalAnatomy
// ------------------------------------------------------------
struct ScorerResult
{
    float quality;
    float axis[8];
};

// ------------------------------------------------------------
// logits_to_result: 生 logits [9] (sigmoid 前) → ScorerResult (sigmoid 済み [0,1])
//   logits9: [9] = index0 quality logit + index1..8 解剖 8 軸 logit
//   sigmoid(x) = 1 / (1 + exp(-x))
//   size != 9 は std::invalid_argument。
// ------------------------------------------------------------
inline ScorerResult logits_to_result(const std::vector<float>& logits9)
{
    if (logits9.size() != 9)
    {
        throw std::invalid_argument(
            "logits_to_result: logits サイズ不正 (期待 9, 実 " +
            std::to_string(logits9.size()) + ")");
    }

    auto sigmoid = [](float x) -> float
    {
        return 1.0f / (1.0f + std::exp(-x));
    };

    ScorerResult r;
    r.quality = sigmoid(logits9[0]); // index0 = quality
    for (int a = 0; a < 8; ++a)
    {
        r.axis[a] = sigmoid(logits9[1 + a]); // index1..8 = 解剖 8 軸
    }
    return r;
}

} // namespace dollama

#ifdef HAVE_OPENVINO

#include <algorithm>
#include <string>

#include <openvino/openvino.hpp>

namespace dollama
{

// ------------------------------------------------------------
// QualityScorer: ScorerNet 品質スコアラの OpenVINO 推論ラッパー
// モデル入力:  [1, 3, INPUT_SIZE, INPUT_SIZE] f32 (NCHW, [0,1])
// モデル出力:  [1, N_OUT] f32 = logits (sigmoid 前。logits_to_result で確率化)
// ------------------------------------------------------------
class QualityScorer
{
public:
    // ScorerNet 固定定数
    static constexpr int INPUT_SIZE = 512; // 入力解像度 (固定形状 NCHW)
    static constexpr int CHANNELS   = 3;   // RGB
    static constexpr int N_OUT      = 9;   // 出力 logits 数 (quality + 8 軸)
    static constexpr int N_AXES     = 8;   // 解剖 8 軸

    // 1 枚分の入力要素数 = 3 * 512 * 512 = 786432
    static constexpr size_t INPUT_NUMEL =
        static_cast<size_t>(CHANNELS) * INPUT_SIZE * INPUT_SIZE;

    // コンストラクタ: モデルをロードしてデバイスにコンパイルする
    // model_xml: OV IR の .xml ファイルパス (model_ov_fp32.xml が golden の正)
    // device:    "NPU" (既定・拡散中遊休 NPU で並列採点) / "CPU" (golden 突合) / "GPU.0" など
    explicit QualityScorer(const std::string& model_xml,
                           const std::string& device = "NPU")
    {
        // IR は既に静的形状 [1,3,512,512] のため reshape 不要。
        auto ov_model = core_.read_model(model_xml);
        compiled_     = core_.compile_model(ov_model, device);
        request_      = compiled_.create_infer_request();
    }

    // 前処理済み NCHW 入力で推論する (golden 突合用の直接経路)。
    // nchw: [INPUT_NUMEL] = 3*512*512 の float (NCHW, [0,1] 前処理済み)。
    // 戻り値: [N_OUT] = 9 の float (logits・sigmoid 前)。
    std::vector<float> infer_nchw(const std::vector<float>& nchw)
    {
        // 入力サイズ検証 (CLIP の教訓: 要素型・要素数は IR と厳密一致させる)
        if (nchw.size() != INPUT_NUMEL)
        {
            throw std::invalid_argument(
                "QualityScorer::infer_nchw: 入力サイズ不正 (期待 " +
                std::to_string(INPUT_NUMEL) + ", 実 " +
                std::to_string(nchw.size()) + ")");
        }

        // OV が所有する f32 テンソルを生成してコピーで渡す。
        // IR の入力 element_type は f32。i32/i64/u8 を渡してはならない (0xC0000409 教訓:
        // CLIP で i32 を渡すと NPU プラグインが i64 として 8 バイト読み STATUS_STACK_BUFFER_OVERRUN)。
        ov::Tensor input_tensor(
            ov::element::f32,
            {1, static_cast<size_t>(CHANNELS),
             static_cast<size_t>(INPUT_SIZE),
             static_cast<size_t>(INPUT_SIZE)});

        float* dst = input_tensor.data<float>();
        std::copy(nchw.begin(), nchw.end(), dst);

        request_.set_input_tensor(input_tensor);
        request_.infer();

        // 出力0: logits [1, N_OUT]
        auto         output = request_.get_output_tensor(0);
        const float* ptr    = output.data<float>();
        return std::vector<float>(ptr, ptr + output.get_size());
    }

    // RGB8 画像から推論する (実運用経路)。
    // rgb: [w*h*3] RGB8 (行優先, 0-255)。w/h が INPUT_SIZE と異なれば最近傍リサイズで整える。
    // 戻り値: [N_OUT] = 9 の float (logits・sigmoid 前)。
    std::vector<float> infer(const std::vector<uint8_t>& rgb, int width, int height)
    {
        // 前処理: RGB8 HWC (0-255) → NCHW f32 (/255)。必要なら 512 へリサイズする。
        std::vector<float> nchw = preprocess(rgb, width, height);
        return infer_nchw(nchw);
    }

    // 前処理: RGB8 HWC (0-255) → NCHW f32 [0,1]。
    // 入力解像度が INPUT_SIZE≠ の場合は最近傍で INPUT_SIZE×INPUT_SIZE にリサイズする
    // (bilinear ではなく最近傍。golden 経路では発火しないため簡潔さを優先した。
    //  matting.hpp::preprocess と同流儀)。
    static std::vector<float> preprocess(const std::vector<uint8_t>& rgb,
                                         int width, int height)
    {
        const size_t expect = static_cast<size_t>(width) * height * CHANNELS;
        if (rgb.size() != expect)
        {
            throw std::invalid_argument(
                "QualityScorer::preprocess: rgb サイズが width*height*3 と不一致");
        }

        std::vector<float> nchw(INPUT_NUMEL);
        const size_t plane = static_cast<size_t>(INPUT_SIZE) * INPUT_SIZE;

        for (int y = 0; y < INPUT_SIZE; ++y)
        {
            // 最近傍リサイズ (width==height==INPUT_SIZE なら sy==y / sx==x で恒等)
            int sy = (height == INPUT_SIZE)
                         ? y
                         : static_cast<int>(static_cast<int64_t>(y) * height / INPUT_SIZE);
            if (sy >= height)
            {
                sy = height - 1;
            }
            for (int x = 0; x < INPUT_SIZE; ++x)
            {
                int sx = (width == INPUT_SIZE)
                             ? x
                             : static_cast<int>(static_cast<int64_t>(x) * width / INPUT_SIZE);
                if (sx >= width)
                {
                    sx = width - 1;
                }

                const size_t src = (static_cast<size_t>(sy) * width + sx) * CHANNELS;
                const size_t dst = static_cast<size_t>(y) * INPUT_SIZE + x;

                // NCHW: R 面 → G 面 → B 面。/255 のみ (正規化なし)。
                nchw[0 * plane + dst] = static_cast<float>(rgb[src + 0]) / 255.0f;
                nchw[1 * plane + dst] = static_cast<float>(rgb[src + 1]) / 255.0f;
                nchw[2 * plane + dst] = static_cast<float>(rgb[src + 2]) / 255.0f;
            }
        }
        return nchw;
    }

private:
    ov::Core          core_;
    ov::CompiledModel compiled_;
    ov::InferRequest  request_;
};

} // namespace dollama

#endif // HAVE_OPENVINO

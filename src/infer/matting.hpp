#pragma once

// ISNet-anime マッティング (α 抽出) — OpenVINO C++ API グルー
// CPU 推論: img [1,3,1024,1024] f32 (NCHW, [0,1]) → mask [1,1,1024,1024] f32 (sigmoid 済み soft マスク [0,1])
//
// モデル入出力仕様 (M-1 / golden_isnet_meta.json 実測):
//   入力  "img"  : element_type f32, shape [1, 3, 1024, 1024] (NCHW 静的) → reshape 不要
//   出力  "mask" : element_type f32, shape [1, 1, 1024, 1024]            (sigmoid 済み soft マスク [0,1])
//
// 前処理 (M-1 厳守): RGB uint8 (0-255) → float32 へ /255 のみ (正規化なし)、レンジ [0,1]。
//   入力レイアウトは NCHW (R 面 → G 面 → B 面)。RGB であり BGR ではない。
//
// 計測ベースライン (M-1 Python): OV IR(FP32) CPU 583ms。HW 比較 (NPU/iGPU) は研究機 M-5。
//
// 設計判断:
//   - device 引数 (既定 "CPU")。研究機が "NPU" / "GPU.0" を差せるよう wd14/clip と同じ形にする。
//   - 入力 element_type は IR と厳密一致 (f32)。CLIP の i32/i64 取り違えで 0xC0000409
//     (STATUS_STACK_BUFFER_OVERRUN) クラッシュした教訓に倣い、必ず ov::element::f32 で生成する。
//   - 入力解像度は 1024 固定前提。≠1024 の RGB が来たら最近傍リサイズで 1024 に整える
//     (golden は 1024 前処理済みなので、テスト経路ではリサイズは発火しない)。
//   - HAVE_OPENVINO 未定義時は OV 依存をすべて閉じる (この開発 PC は OV C++ 無し → コンパイルは通る)。
//
// RGBA 合成ヘルパ (compose_rgba) は OV 非依存の純 C++ なので HAVE_OPENVINO の外に置く。
// (テストの test_rgba_encode が OV 不在でも実行できるようにするため。)

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace dollama
{

// ------------------------------------------------------------
// RGBA 合成ヘルパ (OV 非依存・純 C++)
// rgb:  [W*H*3] RGB8 (行優先, 0-255)
// mask: [W*H]   soft マスク float [0,1] (二値化しない)
// 戻り値: [W*H*4] RGBA8 (ストレート α・premultiply なし)。
//   α = round(mask * 255)。RGB はそのまま残す (合成側でストレート α として扱う)。
// src/server/png.hpp の encode_png_rgba8 にそのまま渡せる。
// ------------------------------------------------------------
inline std::vector<uint8_t> compose_rgba(const std::vector<uint8_t>& rgb,
                                         const std::vector<float>& mask,
                                         int width, int height)
{
    const size_t npix = static_cast<size_t>(width) * static_cast<size_t>(height);
    if (rgb.size() != npix * 3)
    {
        throw std::runtime_error("compose_rgba: rgb サイズが width*height*3 と不一致");
    }
    if (mask.size() != npix)
    {
        throw std::runtime_error("compose_rgba: mask サイズが width*height と不一致");
    }

    std::vector<uint8_t> rgba(npix * 4);
    for (size_t i = 0; i < npix; ++i)
    {
        rgba[i * 4 + 0] = rgb[i * 3 + 0]; // R
        rgba[i * 4 + 1] = rgb[i * 3 + 1]; // G
        rgba[i * 4 + 2] = rgb[i * 3 + 2]; // B

        // soft α: [0,1] を 0-255 へ。範囲外はクランプ (推論誤差で僅かに超える場合に備える)。
        float a = mask[i];
        if (a < 0.0f)
        {
            a = 0.0f;
        }
        if (a > 1.0f)
        {
            a = 1.0f;
        }
        rgba[i * 4 + 3] = static_cast<uint8_t>(a * 255.0f + 0.5f);
    }
    return rgba;
}

} // namespace dollama

#ifdef HAVE_OPENVINO

#include <string>

#include <openvino/openvino.hpp>

namespace dollama
{

// ------------------------------------------------------------
// Matter: ISNet-anime マッティングの OpenVINO 推論ラッパー
// モデル入力:  [1, 3, INPUT_SIZE, INPUT_SIZE] f32 (NCHW, [0,1])
// モデル出力:  [1, 1, INPUT_SIZE, INPUT_SIZE] f32 (sigmoid 済み soft マスク [0,1])
// ------------------------------------------------------------
class Matter
{
public:
    // ISNet-anime 固定定数
    static constexpr int INPUT_SIZE = 1024; // 入力解像度 (固定形状 NCHW)
    static constexpr int CHANNELS   = 3;    // RGB

    // 1 枚分の入力要素数 = 3 * 1024 * 1024 = 3145728
    static constexpr size_t INPUT_NUMEL =
        static_cast<size_t>(CHANNELS) * INPUT_SIZE * INPUT_SIZE;

    // 1 枚分の出力要素数 = 1 * 1024 * 1024 = 1048576
    static constexpr size_t MASK_NUMEL =
        static_cast<size_t>(INPUT_SIZE) * INPUT_SIZE;

    // コンストラクタ: モデルをロードしてデバイスにコンパイルする
    // model_xml: OV IR の .xml ファイルパス (model_ov_fp32.xml が golden の正)
    // device:    "CPU" (既定・golden の正) / "NPU" / "GPU.0" (研究機 M-5 で差し替え)
    explicit Matter(const std::string& model_xml,
                    const std::string& device = "CPU")
    {
        // IR は既に静的形状 [1,3,1024,1024] のため reshape 不要。
        auto ov_model = core_.read_model(model_xml);
        compiled_     = core_.compile_model(ov_model, device);
        request_      = compiled_.create_infer_request();
    }

    // 前処理済み NCHW 入力で推論する (golden 突合用の直接経路)。
    // nchw: [INPUT_NUMEL] = 3*1024*1024 の float (NCHW, [0,1] 前処理済み)。
    // 戻り値: [MASK_NUMEL] = 1024*1024 の float (soft マスク [0,1])。
    std::vector<float> infer_nchw(const std::vector<float>& nchw)
    {
        // 入力サイズ検証 (CLIP の教訓: 要素型・要素数は IR と厳密一致させる)
        if (nchw.size() != INPUT_NUMEL)
        {
            throw std::invalid_argument(
                "Matter::infer_nchw: 入力サイズ不正 (期待 " +
                std::to_string(INPUT_NUMEL) + ", 実 " +
                std::to_string(nchw.size()) + ")");
        }

        // OV が所有する f32 テンソルを生成してコピーで渡す。
        // IR の入力 element_type は f32。i32/i64/u8 を渡してはならない (0xC0000409 教訓)。
        ov::Tensor input_tensor(
            ov::element::f32,
            {1, static_cast<size_t>(CHANNELS),
             static_cast<size_t>(INPUT_SIZE),
             static_cast<size_t>(INPUT_SIZE)});

        float* dst = input_tensor.data<float>();
        std::copy(nchw.begin(), nchw.end(), dst);

        request_.set_input_tensor(input_tensor);
        request_.infer();

        // 出力0: mask [1, 1, INPUT_SIZE, INPUT_SIZE]
        auto         output = request_.get_output_tensor(0);
        const float* ptr    = output.data<float>();
        return std::vector<float>(ptr, ptr + output.get_size());
    }

    // RGB8 画像から推論する (実運用経路)。
    // rgb: [w*h*3] RGB8 (行優先, 0-255)。w/h が INPUT_SIZE と異なれば最近傍リサイズで整える。
    // 戻り値: [MASK_NUMEL] = 1024*1024 の float (soft マスク [0,1])。
    std::vector<float> infer(const std::vector<uint8_t>& rgb, int width, int height)
    {
        // 前処理: RGB8 HWC (0-255) → NCHW f32 (/255)。必要なら 1024 へリサイズする。
        std::vector<float> nchw = preprocess(rgb, width, height);
        return infer_nchw(nchw);
    }

    // 前処理: RGB8 HWC (0-255) → NCHW f32 [0,1]。
    // 入力解像度が INPUT_SIZE≠ の場合は最近傍で INPUT_SIZE×INPUT_SIZE にリサイズする
    // (bilinear ではなく最近傍。golden 経路では発火しないため簡潔さを優先した。
    //  実運用で品質が要れば bilinear へ差し替え可能。M-6 結線時に判断)。
    static std::vector<float> preprocess(const std::vector<uint8_t>& rgb,
                                         int width, int height)
    {
        const size_t expect = static_cast<size_t>(width) * height * CHANNELS;
        if (rgb.size() != expect)
        {
            throw std::invalid_argument(
                "Matter::preprocess: rgb サイズが width*height*3 と不一致");
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

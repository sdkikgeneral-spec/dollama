// IMatter の OpenVINO 実体 — Matter (ISNet-anime) をラップして make_matter を実装する。
//   (Phase 4 M-6 / ウェーブ2)
//
// 依存隔離:
//   本 .cpp は OV 依存を内部に閉じ込める。OV 有効ビルド (openvino_dep.found()) のとき
//   exe / test にリンクされる。OV 無効ビルドでは matter_runner_stub.cpp が代わりに
//   リンクされ make_matter は常に nullptr を返す (両ビルドで matter_runner.hpp の
//   宣言は満たされる)。
//
// HAVE_OPENVINO の二重ガード:
//   meson は OV 有効時にこの .cpp をリンクするが、念のため #ifdef HAVE_OPENVINO で
//   実体を囲い、未定義時 (誤って OV 無で本 .cpp がリンクされた場合) でも nullptr 返しで
//   コンパイル・リンクが通るようにする。
#include "server/matter_runner.hpp"

#ifdef HAVE_OPENVINO

#include <cstddef>
#include <cstdint>

#include "infer/matting.hpp" // Matter (OV 依存・HAVE_OPENVINO ガード越し)

namespace dollama
{

namespace
{

// 1024 固定の Matter 出力 mask を、呼び出し側の w*h 解像度へ整える。
//   Matter は内部で常に 1024x1024 の mask を返す (MASK_NUMEL)。M-6 の生成器は
//   1024x1024 を生成するため通常は w==h==1024 で恒等コピーになる。
//   ≠1024 が来た場合に備え最近傍リサイズで w*h に合わせる (matting.hpp::preprocess と同方針)。
std::vector<float> resize_mask_to(const std::vector<float>& src1024, int w, int h)
{
    const int S = Matter::INPUT_SIZE; // 1024

    // 恒等ケース (生成器の通常経路): そのまま返す。
    if (w == S && h == S)
    {
        return src1024;
    }

    std::vector<float> dst(static_cast<size_t>(w) * static_cast<size_t>(h));
    for (int y = 0; y < h; ++y)
    {
        int sy = static_cast<int>(static_cast<int64_t>(y) * S / h);
        if (sy >= S)
        {
            sy = S - 1;
        }
        for (int x = 0; x < w; ++x)
        {
            int sx = static_cast<int>(static_cast<int64_t>(x) * S / w);
            if (sx >= S)
            {
                sx = S - 1;
            }
            dst[static_cast<size_t>(y) * w + x] =
                src1024[static_cast<size_t>(sy) * S + sx];
        }
    }
    return dst;
}

// IMatter の OV 実体。Matter を値で保持し matte() で α を返す。
class MatterRunner : public IMatter
{
public:
    MatterRunner(const std::string& model_xml, const std::string& device)
        : matter_(model_xml, device)
    {
    }

    std::vector<float> matte(const std::vector<uint8_t>& rgb, int w, int h) override
    {
        // Matter::infer は内部で 1024 へ前処理し [1024*1024] soft マスクを返す。
        std::vector<float> mask1024 = matter_.infer(rgb, w, h);
        return resize_mask_to(mask1024, w, h);
    }

private:
    Matter matter_;
};

} // namespace

std::unique_ptr<IMatter> make_matter(const std::string& model_xml,
                                     const std::string& device)
{
    // モデル xml 未指定なら構築しない (nullptr 契約)。
    if (model_xml.empty())
    {
        return nullptr;
    }
    try
    {
        return std::make_unique<MatterRunner>(model_xml, device);
    }
    catch (...)
    {
        // モデル不在 / ロード失敗 / デバイス未対応 等は nullptr に倒す。
        return nullptr;
    }
}

} // namespace dollama

#else // HAVE_OPENVINO 未定義

namespace dollama
{

// OV 無効でこの .cpp がリンクされた場合でもリンクが通るよう nullptr を返す。
std::unique_ptr<IMatter> make_matter(const std::string& model_xml,
                                     const std::string& device)
{
    (void)model_xml;
    (void)device;
    return nullptr;
}

} // namespace dollama

#endif // HAVE_OPENVINO

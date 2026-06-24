// make_matter のスタブ実装 (純 cpp・OpenVINO 無効ビルド用) — M-6
//
// OpenVINO 無効ビルド (openvino_dep 未解決) では matting.hpp の Matter (OV 依存) を
// コンパイルできないため、matter_runner.cpp の代わりにこの cpp スタブをリンクする。
// make_matter は常に nullptr を返し、呼び出し側 (matting_postprocess.hpp) が
// 不透明 PNG (encode_png_rgb8) へフォールバックする。これにより OV 無効でも
// マッティング結線を含む exe / test がリンク・実行できる。
#include "server/matter_runner.hpp"

namespace dollama
{

std::unique_ptr<IMatter> make_matter(const std::string& model_xml,
                                     const std::string& device)
{
    // OV 無効: マッティングは使えない。常に nullptr (→ 不透明 PNG フォールバック)。
    (void)model_xml;
    (void)device;
    return nullptr;
}

} // namespace dollama

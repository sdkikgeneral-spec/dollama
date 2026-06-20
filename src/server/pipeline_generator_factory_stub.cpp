// PipelineGenerator ファクトリのスタブ実装 (純 cpp・CUDA 無効ビルド用) — ST-5
//
// CUDA 無効ビルド (-Dwith_cuda=false) では .cu をコンパイルできない / リンクしないため、
// pipeline_generator_factory.cu の代わりにこの cpp スタブをリンクする。
// make_pipeline_generator は常に nullptr を返し、呼び出し側 (main.cpp) が
// StubGenerator にフォールバックする。これにより CUDA 無効でも main がリンクできる。
#include "server/pipeline_generator_factory.hpp"

namespace dollama
{

std::unique_ptr<IImageGenerator> make_pipeline_generator(
    const std::string& unet_weights_path,
    const std::string& vae_weights_path,
    const std::string& embeds_path,
    bool               deterministic)
{
    // CUDA 無効: 本 txt2img は使えない。常に nullptr (→ StubGenerator フォールバック)。
    (void)unet_weights_path;
    (void)vae_weights_path;
    (void)embeds_path;
    (void)deterministic;
    return nullptr;
}

} // namespace dollama

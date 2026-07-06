// DiffusionRunner ファクトリのスタブ実装 (純 cpp・CUDA 無効ビルド用)
//   (Phase 2-6b Stage F / ウェーブ2)
//
// CUDA 無効ビルド (-Dwith_cuda=false) では diffusion_runner.cu (nvcc) を
// コンパイル / リンクできないため、その代わりにこの cpp スタブをリンクする
// (meson 側で diffusion_runner.cu と排他)。
//
// make_diffusion_runner は常に nullptr を返し、呼び出し側 (Txt2ImgGenerator /
// main.cpp の DI) が PipelineGenerator / StubGenerator へフォールバックできる。
// これにより CUDA 無効でもビルドがリンクできる
// (pipeline_generator_factory_stub.cpp と同パターン)。
#include "server/diffusion_runner.hpp"

namespace dollama
{

std::unique_ptr<IDiffusionRunner> make_diffusion_runner(
    const std::string& unet_weights,
    const std::string& vae_weights,
    const std::string& embeds_path,
    const FastConfig&  fast_cfg)
{
    // CUDA 無効: DiffusionRunner は存在しない。常に nullptr。
    (void)unet_weights;
    (void)vae_weights;
    (void)embeds_path;
    (void)fast_cfg;
    return nullptr;
}

} // namespace dollama

// PipelineGenerator ファクトリ実装 (CUDA TU) — ST-5
//
// CUDA 隔離設計:
//   pipeline_generator.hpp は HAVE_CUDA ガード越しに diffusion.cuh (__half / CUDA 型) を
//   include する。これを include できるのは nvcc でコンパイルされる .cu のみ。
//   そこで PipelineGenerator の「構築」を本 .cu に閉じ込め、main.cpp (cpp TU) には
//   pipeline_generator_factory.hpp (純 cpp 宣言) だけを見せることで、cpp 側へ CUDA が
//   漏れない。これにより CUDA 無効ビルドでも main がリンクできる
//   (無効時は pipeline_generator_factory_stub.cpp の nullptr 返し実装をリンクする)。
//
// 重み不在チェック:
//   重み 3 ファイル (unet / vae / golden 埋め込み) のいずれかが開けない場合は nullptr を
//   返し、呼び出し側 (main.cpp) が StubGenerator にフォールバックする。これにより
//   重みが無い環境でも HTTP サーバは起動し続ける (回帰防止)。
#include "server/pipeline_generator_factory.hpp"

#include <fstream>
#include <iostream>

#ifdef HAVE_CUDA
#include "server/pipeline_generator.hpp"
#endif

namespace dollama
{

namespace
{

// ファイルが読み出し可能か (存在し開けるか) を判定する。
bool file_readable(const std::string& path)
{
    if (path.empty())
    {
        return false;
    }
    std::ifstream f(path, std::ios::binary);
    return f.good();
}

} // namespace

std::unique_ptr<IImageGenerator> make_pipeline_generator(
    const std::string& unet_weights_path,
    const std::string& vae_weights_path,
    const std::string& embeds_path,
    bool               deterministic)
{
#ifdef HAVE_CUDA
    // 重み/golden の存在チェック。1 つでも欠ければ nullptr (→ Stub フォールバック)。
    if (!file_readable(unet_weights_path) ||
        !file_readable(vae_weights_path) ||
        !file_readable(embeds_path))
    {
        std::cerr << "[factory] 重み/golden が不足しています — StubGenerator にフォールバック\n";
        std::cerr << "  unet  = " << unet_weights_path
                  << (file_readable(unet_weights_path) ? " [OK]" : " [MISSING]") << "\n";
        std::cerr << "  vae   = " << vae_weights_path
                  << (file_readable(vae_weights_path) ? " [OK]" : " [MISSING]") << "\n";
        std::cerr << "  embed = " << embeds_path
                  << (file_readable(embeds_path) ? " [OK]" : " [MISSING]") << "\n";
        return nullptr;
    }

    // 重み (5GB) のロードは PipelineGenerator の構築時に 1 回だけ走る。
    // 本番は deterministic=false (時刻ベース seed) で構築し毎回異なる画像を出す。
    try
    {
        return std::make_unique<PipelineGenerator>(
            unet_weights_path, vae_weights_path, embeds_path, deterministic);
    }
    catch (const std::exception& e)
    {
        // 重みロード失敗等。HTTP 起動を止めないため Stub にフォールバックさせる。
        std::cerr << "[factory] PipelineGenerator 構築に失敗 — StubGenerator にフォールバック: "
                  << e.what() << "\n";
        return nullptr;
    }
#else
    // CUDA 無効ビルド: PipelineGenerator は存在しない。常に nullptr。
    // (実際にはこの .cu は CUDA 有効時のみリンクされるが、念のためガードを置く。)
    (void)unet_weights_path;
    (void)vae_weights_path;
    (void)embeds_path;
    (void)deterministic;
    return nullptr;
#endif
}

} // namespace dollama

// make_pipeline_generator フォールバック判定テスト (純 cpp・CUDA 不要) — ST-5
//
// 検証内容:
//   存在しない重みパスを渡したとき make_pipeline_generator が nullptr を返すこと。
//   これは「重み不在 → StubGenerator フォールバック」という main.cpp の分岐前提を保証する。
//
// このテストは純 cpp。リンクするファクトリ実装は meson が切り替える:
//   - CUDA 無効ビルド: pipeline_generator_factory_stub.cpp (常に nullptr)。
//   - CUDA 有効ビルド: pipeline_generator_factory.cu (不在ファイルは file_readable で
//     弾かれ、重いロード前に nullptr を返す)。
//   いずれの実装でも「不在パス → nullptr」は成立するため、CUDA 有無に関わらず緑になる。
#include <iostream>
#include <memory>
#include <string>

#include "server/generator.hpp"
#include "server/pipeline_generator_factory.hpp"

int main()
{
    bool ok = true;

    // 確実に存在しないパスを 3 つ渡す → nullptr のはず。
    std::unique_ptr<dollama::IImageGenerator> gen =
        dollama::make_pipeline_generator(
            "/dollama_nonexistent_unet.safetensors",
            "/dollama_nonexistent_vae.safetensors",
            "/dollama_nonexistent_embeds.safetensors",
            /*deterministic=*/true);

    if (gen != nullptr)
    {
        std::cerr << "[test_pipeline_factory] FAIL: 不在パスで nullptr が返りませんでした\n";
        ok = false;
    }
    else
    {
        std::cout << "[test_pipeline_factory] 不在パス → nullptr OK (Stub フォールバック経路)\n";
    }

    // 空パスでも nullptr のはず。
    std::unique_ptr<dollama::IImageGenerator> gen2 =
        dollama::make_pipeline_generator("", "", "", /*deterministic=*/false);
    if (gen2 != nullptr)
    {
        std::cerr << "[test_pipeline_factory] FAIL: 空パスで nullptr が返りませんでした\n";
        ok = false;
    }
    else
    {
        std::cout << "[test_pipeline_factory] 空パス → nullptr OK\n";
    }

    if (ok)
    {
        std::cout << "[test_pipeline_factory] ALL PASSED\n";
        return 0;
    }
    std::cerr << "[test_pipeline_factory] FAILED\n";
    return 1;
}

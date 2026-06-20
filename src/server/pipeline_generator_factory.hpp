// PipelineGenerator ファクトリ (純 cpp 宣言・CUDA 非依存) — ST-5
//
// 目的:
//   main.cpp (cpp TU) は CUDA に依存させない。PipelineGenerator の構築は
//   diffusion.cuh (__half / CUDA 型) を含むため CUDA TU (.cu) でしか行えない。
//   そこで「構築だけ」をこのファクトリ関数の背後に隠す。
//
//   - 実装 (CUDA 有効): pipeline_generator_factory.cu が PipelineGenerator を生成。
//   - 実装 (CUDA 無効): pipeline_generator_factory_stub.cpp が常に nullptr を返す。
//   どちらをリンクするかは meson が cuda_enabled で切り替える。
//
//   この宣言ヘッダ自体は CUDA を一切 include しないため、main.cpp / テストなど
//   任意の cpp TU から安全に include できる (CUDA 漏れなし)。
//
// 契約:
//   重み/golden いずれかが解決できない・ファイル不在・CUDA 無効ビルドの場合は
//   nullptr を返す。呼び出し側は nullptr のとき StubGenerator にフォールバックする。
#pragma once

#include <memory>
#include <string>

#include "server/generator.hpp"

namespace dollama
{

// unet/vae 重みと golden 埋め込みのパスから PipelineGenerator を構築する。
//   deterministic=false なら時刻ベース seed (本番の多様性確保)。
//   いずれかのファイルが不在 / CUDA 無効ビルドのときは nullptr を返す
//   (この場合は呼び出し側が StubGenerator へフォールバックする)。
std::unique_ptr<IImageGenerator> make_pipeline_generator(
    const std::string& unet_weights_path,
    const std::string& vae_weights_path,
    const std::string& embeds_path,
    bool               deterministic);

} // namespace dollama

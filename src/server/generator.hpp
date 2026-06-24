// 画像生成器の抽象 interface (HTTP ハンドラと生成本体の境界)
//
// HTTP サーバ層は生成器の中身 (Pipeline / CUDA / OpenVINO) を一切知らない。
// 2-6 完了時に PipelineGenerator を実装し、main.cpp の DI を 1 箇所差し替える
// だけで本 txt2img へ移行できるようにするための純粋仮想 interface。
//
// 責務分離:
//   - 生成器の責務は PNG バイト列まで。
//   - base64 化は配管側 (サーバ責務) で行う (generator は base64 を知らない)。
// CUDA / OpenVINO 依存は一切ない純 C++ (IMatter も純 cpp interface)。
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "server/matter_runner.hpp" // IMatter (純 cpp interface・set_matter 注入用)

namespace dollama
{

// 生成リクエスト DTO (HTTP レイヤがパースして渡す)
struct GenRequest
{
    std::string prompt;          // 必須
    std::string negative_prompt; // 省略時 ""
    int n      = 1;              // 生成枚数 (現在 1 のみ)
    int steps  = 20;             // 拡散ステップ数
    int width  = 1024;           // 出力幅
    int height = 1024;           // 出力高
    bool matting = true;         // M-6: マッティング既定 ON (透過 PNG)。--no-matting で OFF。
};

// 生成結果 DTO (PNG バイト列まで; base64 化はサーバ層)
struct GenResult
{
    std::vector<uint8_t> png_bytes;
};

// 画像生成器 interface
struct IImageGenerator
{
    virtual ~IImageGenerator() = default;

    // 1 リクエスト分の生成を行い PNG バイト列を返す。
    // 失敗時は std::runtime_error 等を投げてよい (サーバ層が 500 に変換)。
    virtual GenResult generate(const GenRequest& req) = 0;

    // モデル識別子 (GET /v1/models で返す)。
    virtual std::string model_id() const = 0;

    // M-6: マッティング器 (α 抽出) を後付け注入する (既定 no-op)。
    //   3 段 DI フォールバックで生成器の所有権が確定した後、build_image_generator が
    //   1 回だけ呼ぶ。マッティングをサポートする生成器 (Txt2Img/Pipeline) のみ override し、
    //   StubGenerator 等は no-op のまま (不透明 PNG)。
    virtual void set_matter(std::unique_ptr<IMatter>) {}
};

} // namespace dollama

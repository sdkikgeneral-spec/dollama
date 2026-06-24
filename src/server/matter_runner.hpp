// IMatter — ISNet-anime マッティング (α 抽出) を純 cpp 層へ繋ぐ抽象 interface。
//   (Phase 4 M-6 / ウェーブ1)
//
// ----------------------------------------------------------------
// 設計意図 (重要):
//   matting.hpp の Matter クラスは OpenVINO (ov::Core / ov::CompiledModel 等) を
//   ヘッダに直接抱えるため #ifdef HAVE_OPENVINO でガードされ、OV 有効 TU からしか
//   include できない。一方、本 interface は OV 非対応の翻訳単位 (cpp TU、例:
//   pipeline_generator.hpp を含む .cu / api.cpp 等) からも常に参照できる必要が
//   あるため、IDiffusionRunner と同様に以下を厳守する:
//     - CUDA 型 (__half 等) も OpenVINO 型も一切シグネチャに露出させない。
//     - CUDA / OV ヘッダを include しない (std と cstdint 等の標準ヘッダのみ)。
//     - #ifdef HAVE_OPENVINO でガードしない (interface 宣言は常に可視)。
//   これにより任意の cpp/.cu TU は IMatter 越しに α 抽出を呼べ、実体 (OV 依存) は
//   matter_runner.cpp (make_matter の OV 実装) に隔離される。
// ----------------------------------------------------------------
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace dollama
{

// マッティング (α 抽出) の実行 interface。実体は matter_runner.cpp 側で
// Matter (ISNet-anime / OpenVINO) をラップして実装する。
struct IMatter
{
    virtual ~IMatter() = default;

    // RGB8 画像から soft α マスクを抽出する。
    //   rgb : [w*h*3] の uint8 RGB ([y,x,c] = row-major HWC, 0-255)。
    //   w,h : 入力解像度。Matter は内部で 1024 固定へ整える (≠1024 はリサイズ)。
    //   戻り値: [w*h] の float soft α マスク ([0,1])。
    //           マスク解像度は入力 w*h に対応する (Matter が 1024 出力を返す前提で
    //           実体側が呼び出し側の w*h に合わせる)。
    virtual std::vector<float> matte(const std::vector<uint8_t>& rgb, int w, int h) = 0;
};

// モデル xml とデバイスを受けて IMatter (Matter ラッパ) を構築する。
//
// nullptr 契約:
//   - model_xml が空 / 不在・ロード不能の場合は nullptr を返す (例外は投げない)。
//   - OpenVINO 無効ビルド (stub リンク時) は常に nullptr を返す。
//   - 構築途中の OV 例外も握って nullptr に倒す。
//   呼び出し側はこれを検知して「matting 無効 (不透明 PNG)」へフォールバックできる。
//
//   device: "CPU" / "NPU" / "GPU.0" (Intel Xe iGPU) 等。M-5 で iGPU 最速確定。
std::unique_ptr<IMatter> make_matter(const std::string& model_xml,
                                     const std::string& device);

} // namespace dollama

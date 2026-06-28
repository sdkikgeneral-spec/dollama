// IScorer — ScorerNet 品質スコアラ (純 conv backbone) を純 cpp 層へ繋ぐ抽象 interface。
//   (Phase 4 Model B / B-5-1)
//
// ----------------------------------------------------------------
// 設計意図 (重要):
//   quality_scorer.hpp の QualityScorer クラスは OpenVINO (ov::Core / ov::CompiledModel 等)
//   をヘッダに直接抱えるため #ifdef HAVE_OPENVINO でガードされ、OV 有効 TU からしか
//   include できない。一方、本 interface は OV 非対応の翻訳単位 (cpp TU、例: 生成器結線や
//   .cu 等) からも常に参照できる必要があるため、IMatter / IDiffusionRunner と同様に以下を
//   厳守する:
//     - CUDA 型 (__half 等) も OpenVINO 型も一切シグネチャに露出させない。
//     - CUDA / OV ヘッダを include しない (std と cstdint 等の標準ヘッダのみ)。
//     - #ifdef HAVE_OPENVINO でガードしない (interface 宣言は常に可視)。
//   これにより任意の cpp/.cu TU は IScorer 越しに採点を呼べ、実体 (OV 依存) は
//   scorer_runner.cpp (make_scorer の OV 実装) に隔離される。
//
// IMatter との非対称点 (意図的):
//   matter_runner.hpp は戻り値を自前の vector<float> で宣言したが、本 interface は
//   ScorerResult を再定義せず quality_scorer.hpp のものを include して再利用する。
//   ScorerResult{float quality; float axis[8];} は HAVE_OPENVINO ガードの外に定義されて
//   いる (OV 非依存) ため、純 cpp TU から include しても OV シンボルは出ない
//   (transitive include の infer/quality_gate.hpp も OV 非依存)。構造体を共有することで
//   将来の軸数・意味の乖離を防ぐ。
// ----------------------------------------------------------------
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "infer/quality_scorer.hpp" // ScorerResult を再利用 (OV 非依存・HAVE_OPENVINO ガードの外)

namespace dollama
{

// 品質スコアリングの実行 interface。実体は scorer_runner.cpp 側で QualityScorer
// (ScorerNet / OpenVINO) をラップして実装する。
struct IScorer
{
    virtual ~IScorer() = default;

    // RGB8 画像を採点して ScorerResult (sigmoid 済み確率 [0,1]) を返す。
    //   rgb : [w*h*3] の uint8 RGB ([y,x,c] = row-major HWC, 0-255)。
    //   w,h : 入力解像度。QualityScorer は内部で 512 固定へ整える (≠512 はリサイズ)。
    //   戻り値: ScorerResult{quality, axis[8]} (index0=quality / axis[0..7]=解剖 8 軸)。
    virtual ScorerResult score(const std::vector<uint8_t>& rgb, int w, int h) = 0;
};

// モデル xml とデバイスを受けて IScorer (QualityScorer ラッパ) を構築する。
//
// nullptr 契約 (make_matter に倣う):
//   - model_xml が空 / 不在・ロード不能の場合は nullptr を返す (例外は投げない)。
//   - OpenVINO 無効ビルド (stub リンク時) は常に nullptr を返す。
//   - 構築途中の OV 例外も握って nullptr に倒す。
//   呼び出し側はこれを検知して「品質採点なし」へフォールバックできる。
//
//   device: "NPU" (既定の意図) / "CPU" (golden 突合) / "GPU.0" (Intel Xe iGPU) 等。
//     既定の狙いは「拡散中 (RTX5080 が UNet で数秒占有) に遊休 NPU で並列採点」。
std::unique_ptr<IScorer> make_scorer(const std::string& model_xml,
                                     const std::string& device);

} // namespace dollama

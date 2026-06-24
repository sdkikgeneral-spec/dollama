// マッティング後処理 — RGB8 生成画像を「透過 PNG (matter 有) / 不透明 PNG (matter 無)」へ
// エンコードする純 cpp グルー (Phase 4 M-6 / ウェーブ1)。
//
// ----------------------------------------------------------------
// 設計意図:
//   生成器 (Txt2ImgGenerator / PipelineGenerator) の最終段で
//   out.png_bytes = encode_png_rgb8(rgb,w,h) を本ヘルパ 1 行に置換し、マッティングを
//   既定 ON にする。本ヘルパは IMatter 越し (純 cpp interface) に α を取るため
//   CUDA/OV 非依存に保たれ、CUDA TU (.cu) からも OV TU (.cpp) からも include できる。
//
//   フォールバック方針 (生成を絶対に失敗させない):
//     - matter == nullptr            → 不透明 PNG (encode_png_rgb8)
//     - matte() の返す mask が w*h と不一致 → 不透明 PNG
//     - matte() が例外を投げた        → 不透明 PNG (catch して握りつぶす)
//   いずれのフォールバックでも例外を呼び出し側へ伝播させず、必ず PNG バイト列を返す。
//   (マッティングはあくまで仕上げ。失敗しても画像生成そのものは成功させる。)
//
//   compose_rgba は matting.hpp の OV 非依存部 (HAVE_OPENVINO の外) なので、本ヘッダは
//   OV 無効ビルドでもコンパイル・実行できる。
// ----------------------------------------------------------------
#pragma once

#include <cstdint>
#include <vector>

#include "infer/matting.hpp" // compose_rgba (OV 非依存部)
#include "server/matter_runner.hpp"
#include "server/png.hpp"

namespace dollama
{

// RGB8 生成画像を PNG バイト列へエンコードする。matter が有効なら α を抽出して
// 透過 PNG (RGBA8 / color_type=6) を、無効・失敗時は不透明 PNG (RGB8 / color_type=2)
// を返す。例外は外へ漏らさない (常に PNG を返す)。
//   matter : 有効なら IMatter::matte で α を取る。nullptr なら不透明。
//   rgb    : [w*h*3] RGB8 (行優先)。
//   w,h    : 解像度。
inline std::vector<uint8_t> encode_png_maybe_transparent(
    IMatter* matter, const std::vector<uint8_t>& rgb, int w, int h)
{
    // matter 無効 → 従来通り不透明 PNG。
    if (matter == nullptr)
    {
        return encode_png_rgb8(rgb, w, h);
    }

    try
    {
        std::vector<float> mask = matter->matte(rgb, w, h);

        // マスク解像度が画像と一致しなければ合成できない → 不透明 PNG へフォールバック。
        if (mask.size() != static_cast<size_t>(w) * static_cast<size_t>(h))
        {
            return encode_png_rgb8(rgb, w, h);
        }

        // soft α を合成して透過 PNG (color_type=6) を返す。
        std::vector<uint8_t> rgba = compose_rgba(rgb, mask, w, h);
        return encode_png_rgba8(rgba, w, h);
    }
    catch (...)
    {
        // matte / 合成のいずれかが失敗しても生成は成功させる (不透明 PNG)。
        return encode_png_rgb8(rgb, w, h);
    }
}

} // namespace dollama

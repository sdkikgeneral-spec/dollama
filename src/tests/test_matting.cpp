// Matter (ISNet-anime マッティング) 単体テスト + 推論レイテンシ計測
// HAVE_OPENVINO 未定義時は [SKIP] を出力して終了する。
// モデルファイル / golden が存在しない場合も [SKIP] で終了する。
// 全テスト通過時は "[test_matting] ALL PASSED" を出力して return 0。
// 失敗時は std::cerr に出力して return 1。
//
// golden (src/tests/data/matting/golden_isnet.safetensors):
//   input [1,3,1024,1024] F32 = 前処理済み NCHW [0,1]  → Matter::infer_nchw に渡す
//   mask  [1,1,1024,1024] F32 = OV IR(FP32) CPU の期待出力 → IoU / MAE で突合
// golden は同じ FP32 IR の出力を正とするため、C++ Matter は高精度一致するはず。
//
// パス埋め込み (cwd 非依存): meson の cpp_args で
//   -DMATTING_MODEL_XML="..." / -DMATTING_GOLDEN_PATH="..." を渡す。

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "infer/matting.hpp" // compose_rgba は OV 非依存 (常に使える)
#include "io/safetensors.hpp"
#include "server/png.hpp"

#ifndef MATTING_MODEL_XML
#define MATTING_MODEL_XML "models/isnet-anime/model_ov_fp32.xml"
#endif
#ifndef MATTING_GOLDEN_PATH
#define MATTING_GOLDEN_PATH "src/tests/data/matting/golden_isnet.safetensors"
#endif

namespace fs = std::filesystem;

namespace dollama
{

// ----------------------------------------------------------------
// OV 非依存テスト: compose_rgba → encode_png_rgba8 が color_type=6 PNG を吐く
// (M-3 PNG エンコーダとの結線確認。OV 不在でも実行できる)
// ----------------------------------------------------------------
static bool test_rgba_encode()
{
    const int w = 4;
    const int h = 3;

    // RGB8 と soft マスクを合成
    std::vector<uint8_t> rgb(static_cast<size_t>(w) * h * 3);
    std::vector<float>   mask(static_cast<size_t>(w) * h);
    for (size_t i = 0; i < mask.size(); ++i)
    {
        rgb[i * 3 + 0] = static_cast<uint8_t>(i * 7 % 256);
        rgb[i * 3 + 1] = static_cast<uint8_t>(i * 13 % 256);
        rgb[i * 3 + 2] = static_cast<uint8_t>(i * 23 % 256);
        mask[i]        = static_cast<float>(i) / static_cast<float>(mask.size() - 1); // 0..1
    }

    std::vector<uint8_t> rgba = compose_rgba(rgb, mask, w, h);
    if (rgba.size() != static_cast<size_t>(w) * h * 4)
    {
        std::cerr << "[test_rgba_encode] RGBA サイズ不正: " << rgba.size() << "\n";
        return false;
    }

    // α が soft (二値化されていない) こと: 中間画素の α が 0/255 以外を含む
    bool has_mid = false;
    for (size_t i = 0; i < mask.size(); ++i)
    {
        // RGB がそのまま残っていること
        if (rgba[i * 4 + 0] != rgb[i * 3 + 0] ||
            rgba[i * 4 + 1] != rgb[i * 3 + 1] ||
            rgba[i * 4 + 2] != rgb[i * 3 + 2])
        {
            std::cerr << "[test_rgba_encode] RGB が改変された idx=" << i << "\n";
            return false;
        }
        uint8_t a = rgba[i * 4 + 3];
        if (a != 0 && a != 255)
        {
            has_mid = true;
        }
    }
    if (!has_mid)
    {
        std::cerr << "[test_rgba_encode] α が二値 (soft でない)\n";
        return false;
    }

    // PNG エンコード → シグネチャ + IHDR color_type=6 を検証
    std::vector<uint8_t> png = encode_png_rgba8(rgba, w, h);
    const uint8_t sig[8] = {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A};
    if (png.size() < 26 || std::memcmp(png.data(), sig, 8) != 0)
    {
        std::cerr << "[test_rgba_encode] PNG シグネチャ不正\n";
        return false;
    }
    // IHDR は png[8..11]=length, [12..15]="IHDR", [16..19]=W, [20..23]=H, [24]=bitdepth, [25]=colortype
    if (png[25] != 6)
    {
        std::cerr << "[test_rgba_encode] color_type が 6 でない: " << static_cast<int>(png[25]) << "\n";
        return false;
    }
    int pw = 0, ph = 0;
    if (!read_png_size(png, pw, ph) || pw != w || ph != h)
    {
        std::cerr << "[test_rgba_encode] PNG サイズ不正: " << pw << "x" << ph << "\n";
        return false;
    }

    std::cout << "[test_rgba_encode] PASSED  (color_type=6, " << pw << "x" << ph
              << ", PNG " << png.size() << " bytes)\n";
    return true;
}

#ifdef HAVE_OPENVINO

using Clock    = std::chrono::steady_clock;
using MsDouble = std::chrono::duration<double, std::milli>;

// golden safetensors から input [1,3,1024,1024] と mask [1,1,1024,1024] を float 配列で取り出す。
static bool load_golden(std::vector<float>& input, std::vector<float>& mask)
{
    SafeTensors st(MATTING_GOLDEN_PATH);

    if (!st.contains("input") || !st.contains("mask"))
    {
        std::cerr << "[load_golden] golden に input/mask テンソルがない\n";
        return false;
    }
    if (st.dtype("input") != StDtype::F32 || st.dtype("mask") != StDtype::F32)
    {
        std::cerr << "[load_golden] golden の dtype が F32 でない\n";
        return false;
    }

    size_t nb = 0;
    const uint8_t* pin = st.tensor_bytes("input", nb);
    if (nb != Matter::INPUT_NUMEL * sizeof(float))
    {
        std::cerr << "[load_golden] input バイト数不正: " << nb << "\n";
        return false;
    }
    input.resize(Matter::INPUT_NUMEL);
    std::memcpy(input.data(), pin, nb);

    const uint8_t* pmk = st.tensor_bytes("mask", nb);
    if (nb != Matter::MASK_NUMEL * sizeof(float))
    {
        std::cerr << "[load_golden] mask バイト数不正: " << nb << "\n";
        return false;
    }
    mask.resize(Matter::MASK_NUMEL);
    std::memcpy(mask.data(), pmk, nb);
    return true;
}

// MAE: 平均絶対誤差
static double compute_mae(const std::vector<float>& a, const std::vector<float>& b)
{
    double acc = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        acc += std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
    }
    return acc / static_cast<double>(a.size());
}

// IoU: しきい値 0.5 で二値化した前景マスクの Intersection over Union
static double compute_iou(const std::vector<float>& a, const std::vector<float>& b, float thr = 0.5f)
{
    size_t inter = 0;
    size_t uni   = 0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        bool fa = a[i] >= thr;
        bool fb = b[i] >= thr;
        if (fa && fb)
        {
            ++inter;
        }
        if (fa || fb)
        {
            ++uni;
        }
    }
    if (uni == 0)
    {
        return 1.0; // 双方とも全背景 → 完全一致扱い
    }
    return static_cast<double>(inter) / static_cast<double>(uni);
}

// ----------------------------------------------------------------
// テスト: golden 突合 (IoU / MAE)
// golden input を Matter に通し、出力 mask を golden mask と突合する。
// 同じ FP32 IR の出力が正なので高精度一致するはず。
// ----------------------------------------------------------------
static bool test_golden_iou_mae(Matter& matter,
                                const std::vector<float>& gin,
                                const std::vector<float>& gmask)
{
    std::vector<float> out = matter.infer_nchw(gin);

    if (out.size() != gmask.size())
    {
        std::cerr << "[test_golden_iou_mae] 出力サイズ不一致: got=" << out.size()
                  << " expected=" << gmask.size() << "\n";
        return false;
    }

    const double mae = compute_mae(out, gmask);
    const double iou = compute_iou(out, gmask);

    // しきい値: 同一 FP32 IR の再現なので極めて高い一致を要求する。
    // FP32 IR(ONNX) vs OV-FP32 の max_abs ~6e-6 (meta.json) を踏まえ、MAE は十分小さいはず。
    const double iou_thr = 0.99;
    const double mae_thr = 1e-3;

    std::cout << "[test_golden_iou_mae] IoU=" << iou << " (thr>=" << iou_thr << ")  "
              << "MAE=" << mae << " (thr<=" << mae_thr << ")\n";

    if (iou < iou_thr)
    {
        std::cerr << "[test_golden_iou_mae] IoU が閾値未満\n";
        return false;
    }
    if (mae > mae_thr)
    {
        std::cerr << "[test_golden_iou_mae] MAE が閾値超過\n";
        return false;
    }

    std::cout << "[test_golden_iou_mae] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// テスト: 出力値域 [0,1] (sigmoid 済み soft マスク)
// ----------------------------------------------------------------
static bool test_mask_range(Matter& matter, const std::vector<float>& gin)
{
    std::vector<float> out = matter.infer_nchw(gin);
    for (size_t i = 0; i < out.size(); ++i)
    {
        if (out[i] < 0.0f || out[i] > 1.0f)
        {
            std::cerr << "[test_mask_range] 値域外: idx=" << i << " value=" << out[i] << "\n";
            return false;
        }
    }
    std::cout << "[test_mask_range] PASSED  (all in [0,1], N=" << out.size() << ")\n";
    return true;
}

// ----------------------------------------------------------------
// テスト: 決定性 (同一入力 2 回で完全一致)
// ----------------------------------------------------------------
static bool test_determinism(Matter& matter, const std::vector<float>& gin)
{
    std::vector<float> a = matter.infer_nchw(gin);
    std::vector<float> b = matter.infer_nchw(gin);
    if (a.size() != b.size())
    {
        std::cerr << "[test_determinism] サイズ不一致\n";
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i)
    {
        if (a[i] != b[i])
        {
            std::cerr << "[test_determinism] 出力不一致: idx=" << i
                      << " " << a[i] << " vs " << b[i] << "\n";
            return false;
        }
    }
    std::cout << "[test_determinism] PASSED  (2 回完全一致)\n";
    return true;
}

// ----------------------------------------------------------------
// テスト: 入力サイズガード (不正サイズで invalid_argument)
// ----------------------------------------------------------------
static bool test_input_size_guard(Matter& matter)
{
    std::vector<float> bad(Matter::INPUT_NUMEL - 1, 0.0f);
    bool threw = false;
    try
    {
        matter.infer_nchw(bad);
    }
    catch (const std::invalid_argument&)
    {
        threw = true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[test_input_size_guard] 想定外の例外: " << e.what() << "\n";
        return false;
    }
    if (!threw)
    {
        std::cerr << "[test_input_size_guard] 例外が飛ばなかった\n";
        return false;
    }
    std::cout << "[test_input_size_guard] PASSED  (invalid_argument を捕捉)\n";
    return true;
}

// ----------------------------------------------------------------
// ベンチ: 推論レイテンシ (中央値)。warmup 3 + 計測 N。
// ISNet-anime は重いため N を控えめにする (期待 CPU ~583ms / M-1 Python)。
// ----------------------------------------------------------------
static bool bench_infer(Matter& matter, const std::vector<float>& gin)
{
    for (int i = 0; i < 3; ++i)
    {
        matter.infer_nchw(gin);
    }

    constexpr int kN = 7;
    std::vector<double> times;
    times.reserve(kN);
    for (int i = 0; i < kN; ++i)
    {
        auto t0 = Clock::now();
        matter.infer_nchw(gin);
        times.push_back(MsDouble(Clock::now() - t0).count());
    }
    std::sort(times.begin(), times.end());
    double median = times[kN / 2];

    std::cout << "[bench_infer]\n";
    std::cout << "  BENCH: matting_infer median=" << median << " ms"
              << "  (min=" << times.front()
              << " ms, max=" << times.back() << " ms, N=" << kN << ")\n";
    return true;
}

#endif // HAVE_OPENVINO

} // namespace dollama

int main()
{
    // OV 非依存テスト (compose_rgba → PNG) は常に実行する。
    bool ok = true;
    ok = dollama::test_rgba_encode() && ok;

#ifndef HAVE_OPENVINO
    std::cout << "[SKIP] HAVE_OPENVINO が未定義です。"
                 "OpenVINO なしでビルドされているため推論テストをスキップします。\n";
    if (!ok)
    {
        std::cerr << "[test_matting] FAILED\n";
        return 1;
    }
    std::cout << "[test_matting] ALL PASSED\n";
    return 0;
#else
    // モデル / golden の存在確認
    const std::string xml_path    = MATTING_MODEL_XML;
    const std::string golden_path = MATTING_GOLDEN_PATH;
    if (!fs::exists(xml_path))
    {
        std::cout << "[SKIP] モデルファイルが見つかりません: " << xml_path << "\n";
        if (!ok)
        {
            std::cerr << "[test_matting] FAILED\n";
            return 1;
        }
        std::cout << "[test_matting] ALL PASSED\n";
        return 0;
    }
    if (!fs::exists(golden_path))
    {
        std::cout << "[SKIP] golden が見つかりません: " << golden_path << "\n";
        if (!ok)
        {
            std::cerr << "[test_matting] FAILED\n";
            return 1;
        }
        std::cout << "[test_matting] ALL PASSED\n";
        return 0;
    }

    // golden ロード
    std::vector<float> gin;
    std::vector<float> gmask;
    if (!dollama::load_golden(gin, gmask))
    {
        std::cerr << "[test_matting] golden ロード失敗\n";
        return 1;
    }

    // Matter を CPU でコンパイル (golden の正は FP32 IR の CPU 出力)
    dollama::Matter matter(xml_path, "CPU");

    ok = dollama::test_golden_iou_mae(matter, gin, gmask) && ok;
    ok = dollama::test_mask_range(matter, gin)            && ok;
    ok = dollama::test_determinism(matter, gin)           && ok;
    ok = dollama::test_input_size_guard(matter)           && ok;
    ok = dollama::bench_infer(matter, gin)                && ok;

    if (!ok)
    {
        std::cerr << "[test_matting] FAILED\n";
        return 1;
    }
    std::cout << "[test_matting] ALL PASSED\n";
    return 0;
#endif // HAVE_OPENVINO
}

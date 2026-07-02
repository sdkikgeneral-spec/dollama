// make_backend の実体 (registry factory)。
//
// ----------------------------------------------------------------
// 設計意図:
//   diffusion_backend.hpp は純 cpp (OV/CUDA 非露出) の宣言のみ。SDXLBackend の実体は
//   OV+CUDA 依存ゆえ、その include と構築をこの .cpp に隔離する。OV 有無で sdxl_backend.hpp
//   の参照をガードし、OV 無効ビルドでは "sdxl" は nullptr を返す (matter_runner_stub と同思想)。
//
//   本 .cpp は cl (MSVC) でコンパイルされる cpp TU。SDXLBackend が使う IDiffusionRunner の
//   実体 (make_diffusion_runner) は別途 diffusion_runner.cu / _stub.cpp でリンクされる。
// ----------------------------------------------------------------
#include "server/diffusion_backend.hpp"

#include <memory>

#include "server/sd35_backend.hpp"

#ifdef HAVE_OPENVINO
#include "server/sdxl_backend.hpp"
#endif

namespace dollama
{

std::unique_ptr<IDiffusionBackend> make_backend(const BackendConfig& cfg)
{
    // "sdxl": OV 有効時のみ SDXLBackend を構築する。
    //   構築途中の例外 (runner nullptr → throw / OV ロード失敗) は握って nullptr に倒す
    //   (make_matter / make_scorer の nullptr 契約に倣う)。
    if (cfg.backend_name == "sdxl")
    {
#ifdef HAVE_OPENVINO
        try
        {
            return std::make_unique<SDXLBackend>(
                cfg.tok_l, cfg.tok_g, cfg.enc_l, cfg.enc_g, cfg.tok_dll,
                cfg.unet_weights, cfg.vae_weights, cfg.embeds,
                cfg.device_l.empty() ? "NPU" : cfg.device_l,
                cfg.device_g.empty() ? "NPU" : cfg.device_g);
        }
        catch (...)
        {
            // 重み不在 / OV ロード失敗 → 呼び出し側が段2/3 へフォールバックできる。
            return nullptr;
        }
#else
        // OV 無効ビルドでは SDXL は構築できない。
        return nullptr;
#endif
    }

    // "sd35": 常に構築可 (生成は未実装で generate() が throw する拡張点)。
    if (cfg.backend_name == "sd35")
    {
        return std::make_unique<SD35Backend>();
    }

    // 未知の backend 名 → nullptr。
    return nullptr;
}

} // namespace dollama

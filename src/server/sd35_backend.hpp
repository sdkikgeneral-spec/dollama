// SD35Backend — SD3.5 (Stable Diffusion 3.5) アーキの拡張点 stub。
//
// ----------------------------------------------------------------
// 設計意図:
//   プラグイン枠が実際に別アーキを受け入れられることを型レベルで示すための stub。
//   generate() は throw して「未実装」を明示する (registry には登録し、info() の静的
//   メタは正しい値を返す)。SD3.5 の実生成 (MMDiT グラフ・T5-XXL 配置・flow matching・
//   16ch VAE) は本 stub のスコープ外 (docs/diffusion-backend-spec.md の残作業節を参照)。
//
//   本ヘッダは純 cpp (OV/CUDA 非依存)。SD3.5 は latent 16ch・T5 必須ゆえ info() で
//   SDXL と異なる値 (latent_channels=16 / needs_t5=true) を返す。
// ----------------------------------------------------------------
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "server/diffusion_backend.hpp"

namespace dollama
{

// SD3.5 拡張点 stub。generate は未実装 (throw)・info/model_id はメタのみ返す。
class SD35Backend : public IDiffusionBackend
{
public:
    // SD3.5 は MMDiT + flow matching + 16ch VAE + T5-XXL が未実装ゆえ生成できない。
    void generate(const std::string& /*prompt*/, const std::string& /*negative*/,
                  int /*steps*/, uint64_t /*seed*/, float /*cfg*/, int /*w*/, int /*h*/,
                  std::vector<uint8_t>& /*rgb_out*/, int& /*w_out*/, int& /*h_out*/) override
    {
        throw std::runtime_error("sd35 backend: 未実装 (拡張点)");
    }

    // SD3.5 の静的メタ情報: latent 16ch / native 1024 / T5 必須。
    BackendInfo info() const override
    {
        return {"sd35", 16, 1024, true};
    }

    std::string model_id() const override
    {
        return "sd35";
    }
};

} // namespace dollama

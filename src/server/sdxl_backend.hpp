// SDXLBackend — SDXL アーキを IDiffusionBackend として実装する (プラグイン化の SDXL 実体)。
//
// ----------------------------------------------------------------
// 依存隔離 (Txt2ImgGenerator の依存構造をそのまま踏襲):
//   - TextConditioner (text_conditioner.hpp) は OpenVINO 依存のため、本ヘッダ全体を
//     #ifdef HAVE_OPENVINO でガードする。HAVE_OPENVINO が無い TU から include しても
//     空になりコンパイルが壊れない。
//   - 拡散ループ (CUDA / __half) は純 cpp interface IDiffusionRunner 越しに呼ぶため、
//     本ヘッダに CUDA 型は一切露出しない。runner の実体 (CUDA) は make_diffusion_runner
//     (.cu / _stub.cpp) に隔離される。
//
//   従来 Txt2ImgGenerator が担っていた「SDXL encode + 拡散」の中核だけをここへ移設し、
//   共通後処理 (解像度 reject / seed / 採点ログ / matting PNG 化) は
//   BackendImageGenerator へ切り出した。
// ----------------------------------------------------------------
#pragma once

#ifdef HAVE_OPENVINO

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "infer/text_conditioner.hpp"
#include "server/diffusion_backend.hpp"
#include "server/diffusion_runner.hpp"
#include "server/fast_config.hpp"

namespace dollama
{

// TextConditioner + IDiffusionRunner を結線し、prompt → RGB8 を生成する SDXL backend。
class SDXLBackend : public IDiffusionBackend
{
public:
    // CFG (classifier-free guidance) スケール。SDXL の一般的な既定値。
    static constexpr float kGuidanceScale = 7.5f;

    // 各モデルパス・デバイス・重みパスを受けて TextConditioner と IDiffusionRunner を構築する。
    //   (引数規約は従来 Txt2ImgGenerator の ctor をそのまま踏襲)
    //   fast_cfg (G-0b): FAST フラグを make_diffusion_runner → DiffusionPipeline へ運ぶだけ
    //     (既定 = 全 off = 現行挙動)。この Pkg では generate() に fast 分岐を一切足さない。
    //   runner (= make_diffusion_runner) が nullptr の場合 (重み不在 / CUDA 無効) は
    //   構築失敗として std::runtime_error を投げる。make_backend がこれを握って nullptr へ倒す。
    SDXLBackend(const std::string& tokenizer_l_xml,
                const std::string& tokenizer_g_xml,
                const std::string& encoder_l_xml,
                const std::string& encoder_g_xml,
                const std::string& tokenizers_dll,
                const std::string& unet_weights,
                const std::string& vae_weights,
                const std::string& embeds_path,
                const std::string& device_l = "NPU",
                const std::string& device_g = "NPU",
                const FastConfig&  fast_cfg = FastConfig{})
        : tc_(tokenizer_l_xml, tokenizer_g_xml, encoder_l_xml, encoder_g_xml,
              tokenizers_dll, device_l, device_g),
          runner_(make_diffusion_runner(unet_weights, vae_weights, embeds_path, fast_cfg))
    {
        if (!runner_)
        {
            // 重み不在 / ロード失敗 / CUDA 無効ビルド。make_backend が握って nullptr へ倒す。
            throw std::runtime_error(
                "SDXLBackend: DiffusionRunner の構築に失敗 (重み不在 / CUDA 無効ビルド)");
        }
    }

    SDXLBackend(const SDXLBackend&)            = delete;
    SDXLBackend& operator=(const SDXLBackend&) = delete;

    // prompt / negative → RGB8。cfg<=0 のときは SDXL 既定 (7.5) を使う。
    void generate(const std::string& prompt, const std::string& negative,
                  int steps, uint64_t seed, float cfg, int /*w*/, int /*h*/,
                  std::vector<uint8_t>& rgb_out, int& w_out, int& h_out) override
    {
        const int   real_steps = (steps > 0) ? steps : 20;
        const float scale       = (cfg > 0.0f) ? cfg : kGuidanceScale;

        // --- テキスト条件付け (prompt / negative → cond/uncond 埋め込み) ---
        const TextConditioning tc = tc_.encode(prompt, negative);

        // --- 拡散ループ (CFG) 実行 ---
        rgb_out.clear();
        w_out = 0;
        h_out = 0;
        runner_->run_txt2img(
            real_steps, seed, scale,
            tc.cond.encoder_hidden_states.data(),
            tc.cond.text_embeds.data(),
            tc.uncond.encoder_hidden_states.data(),
            tc.uncond.text_embeds.data(),
            tc.time_ids.data(),
            rgb_out, w_out, h_out);
    }

    // ----------------------------------------------------------------
    // ランタイム LoRA (L-2)。name → ファイルパスを解決して runner へ委譲する。
    //   探索先: 環境変数 DOLLAMA_LORA_DIR (既定 "models/loras") 配下の <name>.safetensors。
    //   未存在は std::invalid_argument (HTTP 層が 4xx に変換する契約)。
    //   デバイス適用の実体は runner (DiffusionRunner → DiffusionPipeline →
    //   unet_apply_loras) に隔離され、本ヘッダに CUDA 型は露出しない。
    // ----------------------------------------------------------------
    void apply_loras(const std::vector<LoraRequest>& reqs) override
    {
        if (reqs.empty())
        {
            return;
        }
        std::vector<LoraFileRequest> files;
        files.reserve(reqs.size());
        for (const LoraRequest& r : reqs)
        {
            const std::string path = resolve_lora_path(r.name);
            std::ifstream     f(path, std::ios::binary);
            if (!f.good())
            {
                throw std::invalid_argument(
                    "SDXLBackend: LoRA '" + r.name + "' が見つかりません: " + path);
            }
            files.push_back(LoraFileRequest{path, r.strength});
        }
        runner_->apply_loras(files);
    }

    // patch 済み常駐重みを base へ bit-exact 復元する。
    void clear_loras() override
    {
        runner_->clear_loras();
    }

    // SDXL の静的メタ情報: latent 4ch / native 1024 / T5 不要。
    BackendInfo info() const override
    {
        return {"sdxl", 4, 1024, false};
    }

    std::string model_id() const override
    {
        return "sdxl-1.0";
    }

private:
    // LoRA 名 → ファイルパス解決。DOLLAMA_LORA_DIR (既定 "models/loras") 配下の
    // <name>.safetensors。存在確認は呼び出し側 (apply_loras) が行う。
    //
    // name は HTTP リクエスト由来のユーザー入力なので、連結前に厳格に検証して
    // ディレクトリ脱出 (path traversal) を封じる: 空・先頭ドット・区切り文字を含む
    // ものは即 reject し、許可文字は [A-Za-z0-9_.-] のみ (これにより '/' '\\' '..' は
    // 構造的に混入不能)。不正名は std::invalid_argument → HTTP 層が 4xx に変換する。
    static std::string resolve_lora_path(const std::string& name)
    {
        if (name.empty() || name[0] == '.')
        {
            throw std::invalid_argument("SDXLBackend: 不正な LoRA 名: '" + name + "'");
        }
        for (const char c : name)
        {
            const bool allowed = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
                                 (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.';
            if (!allowed)
            {
                throw std::invalid_argument("SDXLBackend: 不正な LoRA 名: '" + name + "'");
            }
        }

        const char*       env = std::getenv("DOLLAMA_LORA_DIR");
        const std::string dir = (env != nullptr && env[0] != '\0') ? env : "models/loras";
        return dir + "/" + name + ".safetensors";
    }

    TextConditioner                   tc_;
    std::unique_ptr<IDiffusionRunner> runner_;
};

} // namespace dollama

#endif // HAVE_OPENVINO

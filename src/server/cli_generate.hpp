// CLI 生成モードと HTTP モードで共有する画像生成器 DI ヘルパ。
//
// 目的:
//   従来 main.cpp の `if (http_mode)` ブロック内にインライン展開されていた
//   生成器の 3 段フォールバック DI を 1 関数 build_image_generator() に括り出し、
//   --http と新 CLI 生成モード (--prompt) の両方が同一経路を共有する。
//
//   段順・env/パス解決・NPU→CPU フォールバック・各段の条件・ログ文言は
//   従来 (HTTP) と bitwise 等価に移植 (挙動非回帰)。
//
//   3 段:
//     段1) OV アセット (tokenizer/encoder L/G + tokenizers.dll) + unet/vae/embeds が
//          揃う → Txt2ImgGenerator (prompt を反映する本 txt2img・NPU 第一→失敗時 CPU)。
//     段2) unet/vae 重みのみ → PipelineGenerator (golden 埋め込み)。
//     段3) いずれも無 → StubGenerator。
//
//   M-6: 3 段で gen が確定した後、マッティング器 (IMatter) を 1 回だけ後付け注入する。
//        所有権が 3 段 DI を跨ぐため、set_matter() (既定 no-op) で gen 確定後に注入する。
//        make_matter は OV 無効ビルドで stub が nullptr を返すためガード不要。
//
//   この宣言ヘッダ自体は CUDA を一切 include しない (PipelineGenerator の構築は
//   make_pipeline_generator ファクトリ越し)。Txt2ImgGenerator のみ OV 依存のため
//   HAVE_OPENVINO && HAVE_CUDA ガード内で参照する。
#pragma once

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#ifdef HAVE_OPENVINO
#include <filesystem>
#endif

#include "server/generator.hpp"
#include "server/matter_runner.hpp"
#include "server/pipeline_generator_factory.hpp"
#include "server/stub_generator.hpp"

#if defined(HAVE_OPENVINO) && defined(HAVE_CUDA)
#include "server/txt2img_generator.hpp"
#endif

namespace dollama
{

#ifdef HAVE_OPENVINO
// モデル .xml の候補パスを探索する (実行ディレクトリ差を吸収)。
inline std::string find_model_xml(const std::string& rel)
{
    namespace fs = std::filesystem;
    const std::string candidates[] = {
        "../models/" + rel,
        "models/" + rel,
        "../../models/" + rel,
    };
    for (const auto& p : candidates)
    {
        if (fs::exists(p))
        {
            return p;
        }
    }
    return ""; // 見つからなければ空文字
}
#endif

// 環境変数で上書き可能な重みパスを解決する。
//   env が空でなければそれを使い、無ければ fallback を使う。
inline std::string resolve_path(const char* env_name, const std::string& fallback)
{
    // MSVC は std::getenv に C4996 (非推奨) を出すが、ここは読み取り専用で
    // スレッド前 (起動時 1 回) のため安全。局所的に警告を抑止する。
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
    if (const char* v = std::getenv(env_name))
    {
        if (v[0] != '\0')
        {
            return std::string(v);
        }
    }
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
    return fallback;
}

// 画像生成器を 3 段フォールバックで構築する (HTTP / CLI 共有)。
//   log: 各段の選択を出力するストリーム (HTTP は std::cout / CLI も std::cout)。
//   ログ文言・段順・条件は従来 HTTP DI と bitwise 等価。
inline std::unique_ptr<IImageGenerator> build_image_generator(std::ostream& log)
{
    // 重み/golden パスを解決 (env 変数で上書き可。既定は test data パス)。
    // 既定は src/tests/data。本番の重み配置先が決まったら DEFAULT を差し替える。
    const std::string unet_w =
        resolve_path("DOLLAMA_UNET_WEIGHTS", "src/tests/data/unet_weights.safetensors");
    const std::string vae_w =
        resolve_path("DOLLAMA_VAE_WEIGHTS", "src/tests/data/vae_weights.safetensors");
    const std::string embeds =
        resolve_path("DOLLAMA_EMBEDS", "src/tests/data/unet_io.safetensors");

    std::unique_ptr<IImageGenerator> gen;

    // ----------------------------------------------------------------
    // DI 3 段フォールバック:
    //   段1) OV アセット (tokenizer/encoder L/G + tokenizers.dll) + unet/vae 重みが
    //        揃う → Txt2ImgGenerator (prompt を反映する本 txt2img)。
    //   段2) unet/vae 重みのみ (OV 無 / アセット欠) → PipelineGenerator (golden 埋め込み)。
    //   段3) いずれも無 → StubGenerator。
    // Txt2ImgGenerator は HAVE_OPENVINO かつ runner (CUDA) が必要。両ガードが揃わない
    // ビルドでは段1 をコンパイル時に丸ごと無効化し、段2/3 へ落ちる。
    // ----------------------------------------------------------------
#if defined(HAVE_OPENVINO) && defined(HAVE_CUDA)
    {
        // OV アセットパスを解決 (env 優先・既定は models/ ツリー)。
        const std::string tok_l =
            resolve_path("DOLLAMA_TOKENIZER_L",
                         find_model_xml("sdxl-tokenizer-l/openvino_tokenizer.xml"));
        const std::string tok_g =
            resolve_path("DOLLAMA_TOKENIZER_G",
                         find_model_xml("sdxl-tokenizer-g/openvino_tokenizer.xml"));
        const std::string enc_l =
            resolve_path("DOLLAMA_ENCODER_L",
                         find_model_xml("sdxl-text-encoder-l/model_ov.xml"));
        const std::string enc_g =
            resolve_path("DOLLAMA_ENCODER_G",
                         find_model_xml("sdxl-text-encoder-g/model_ov.xml"));
        // openvino_tokenizers.dll は env のみ (空なら段1 をスキップ)。
        const std::string tok_dll = resolve_path("DOLLAMA_OV_TOKENIZERS_DLL", "");

        namespace fs = std::filesystem;
        const bool ov_ready =
            !tok_l.empty() && fs::exists(tok_l) &&
            !tok_g.empty() && fs::exists(tok_g) &&
            !enc_l.empty() && fs::exists(enc_l) &&
            !enc_g.empty() && fs::exists(enc_g) &&
            !tok_dll.empty() && fs::exists(tok_dll) &&
            fs::exists(unet_w) && fs::exists(vae_w) && fs::exists(embeds);

        if (ov_ready)
        {
            // NPU 第一・失敗時 CPU フォールバックで Txt2ImgGenerator を構築。
            try
            {
                gen = std::make_unique<dollama::Txt2ImgGenerator>(
                    tok_l, tok_g, enc_l, enc_g, tok_dll,
                    unet_w, vae_w, embeds, "NPU", "NPU");
                log << "dollama HTTP server (txt2img generator — NPU)\n";
            }
            catch (const std::exception& e)
            {
                log << "[warn] NPU での Txt2ImgGenerator 構築に失敗 ("
                    << e.what() << ") → CPU を試します。\n";
                try
                {
                    gen = std::make_unique<dollama::Txt2ImgGenerator>(
                        tok_l, tok_g, enc_l, enc_g, tok_dll,
                        unet_w, vae_w, embeds, "CPU", "CPU");
                    log << "dollama HTTP server (txt2img generator — CPU)\n";
                }
                catch (const std::exception& e2)
                {
                    log << "[warn] CPU でも構築に失敗 (" << e2.what()
                        << ") → 段2/3 へフォールバックします。\n";
                }
            }
        }
    }
#endif

    // 段2) Txt2ImgGenerator が立たなければ PipelineGenerator を試みる。
    //   ファクトリは重み不在 / CUDA 無効なら nullptr (→ 段3)。本番なので deterministic=false。
    if (!gen)
    {
        gen = dollama::make_pipeline_generator(
            unet_w, vae_w, embeds, /*deterministic=*/false);
        if (gen)
        {
            log << "dollama HTTP server (pipeline generator — golden 埋め込み)\n";
            log << "  weights: unet='" << unet_w << "' vae='" << vae_w
                << "' embeds='" << embeds << "'\n";
        }
    }

    // 段3) いずれも立たなければ StubGenerator (回帰防止・必ず PNG が出る)。
    if (!gen)
    {
        gen = std::make_unique<dollama::StubGenerator>();
        log << "dollama HTTP server (stub generator — 重み未解決のためフォールバック)\n";
    }

    // ----------------------------------------------------------------
    // M-6: マッティング器 (IMatter) を後付け注入する (gen 確定後に 1 回のみ)。
    //   - モデル xml: env DOLLAMA_MATTING_WEIGHTS 優先。既定は HAVE_OPENVINO 時のみ
    //     models/ ツリーを find_model_xml で探索 (OV 無時は空文字 → stub nullptr)。
    //   - device: env DOLLAMA_MATTING_DEVICE 優先。既定は "GPU.0" (Intel Xe iGPU・M-5 最速)。
    //   make_matter は OV 無効ビルドで stub が常に nullptr を返すためガード不要。
    //   null なら従来通り不透明 PNG・非 null なら set_matter で透過 PNG 有効化。
    {
#ifdef HAVE_OPENVINO
        const std::string matting_xml =
            resolve_path("DOLLAMA_MATTING_WEIGHTS",
                         find_model_xml("isnet-anime/model_ov_fp32.xml"));
#else
        const std::string matting_xml = resolve_path("DOLLAMA_MATTING_WEIGHTS", "");
#endif
        const std::string matting_dev = resolve_path("DOLLAMA_MATTING_DEVICE", "GPU.0");

        std::unique_ptr<IMatter> m = make_matter(matting_xml, matting_dev);
        if (m)
        {
            gen->set_matter(std::move(m));
            log << "  matting: " << matting_dev << " (model='" << matting_xml << "')\n";
        }
        else
        {
            log << "  matting: 無効 (モデル/OV 無 — 不透明 PNG)\n";
        }
    }

    return gen;
}

// PNG バイト列をファイルへ書き出す (ios::binary)。成功で true。
inline bool write_png_file(const std::string& path, const std::vector<uint8_t>& bytes)
{
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs)
    {
        return false;
    }
    ofs.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
    return static_cast<bool>(ofs);
}

} // namespace dollama

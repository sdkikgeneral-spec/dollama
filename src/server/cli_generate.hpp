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
//          揃う → make_backend(...) → BackendImageGenerator (prompt を反映する本 txt2img)。
//          backend は env DOLLAMA_BACKEND で選択 (既定 "sdxl")。NPU 第一→失敗時 CPU。
//     段2) unet/vae 重みのみ → PipelineGenerator (golden 埋め込み)。
//     段3) いずれも無 → StubGenerator。
//
//   M-6: 3 段で gen が確定した後、マッティング器 (IMatter) を 1 回だけ後付け注入する。
//        所有権が 3 段 DI を跨ぐため、set_matter() (既定 no-op) で gen 確定後に注入する。
//        make_matter は OV 無効ビルドで stub が nullptr を返すためガード不要。
//
//   この宣言ヘッダ自体は CUDA を一切 include しない (PipelineGenerator の構築は
//   make_pipeline_generator ファクトリ越し)。段1 の backend 実体 (SDXL) は OV+CUDA
//   依存だが make_backend (diffusion_backend.cpp) の内側に隔離される。BackendImageGenerator
//   は純 cpp ゆえ本ヘッダはガード不要で include できる (段1 の構築判定のみ OV&&CUDA でガード)。
#pragma once

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#ifdef HAVE_OPENVINO
#include <filesystem>
#endif

#include "server/backend_image_generator.hpp" // 純 cpp・IDiffusionBackend アダプタ (段1)
#include "server/diffusion_backend.hpp"        // BackendConfig / make_backend (registry)
#include "server/generator.hpp"
#include "server/matter_runner.hpp"
#include "server/preset.hpp"                   // 2-6d: preset 解決 (resolve_preset_paths)
#include "server/preset_json.hpp"              // 2-6e: preset.json 読込 (read_preset_prefix)
#include "server/scorer_runner.hpp"
#include "server/pipeline_generator_factory.hpp"
#include "server/stub_generator.hpp"

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
//   preset_cli: CLI 由来の preset 名 (--preset)。空なら env DOLLAMA_BACKEND_PRESET を見る。
//   既定 preset は "illustrious-xl" (ユーザー決裁)。base 重みへ明示的に戻したいときは
//   `--preset base` / `DOLLAMA_BACKEND_PRESET=base` を指定する (preset なし の明示値。
//   `DOLLAMA_BACKEND_PRESET=` (空文字) は「未指定」= 既定 preset に落ちる点に注意)。
//   ログ文言・段順・条件は preset 解決成功時のみ変化し、"base" 明示 / preset 未配置時は
//   preset 関連ログ行が増えるのみで既存ログは不変。
inline std::unique_ptr<IImageGenerator> build_image_generator(
    std::ostream& log, const FastConfig& cli_fast = FastConfig{},
    const std::string& preset_cli = "")
{
    // G-0b: env (DOLLAMA_FAST/DOLLAMA_FP8) と CLI 由来フラグを OR 合成し fp8→fast を適用。
    //   default (全 off) は現行挙動そのまま。fast 分岐はこの Pkg では一切足さない。
    const FastConfig fast_cfg = resolve_fast_config(cli_fast);
    (void)fast_cfg; // 段1 (OV&&CUDA) 以外では未使用。段2/3 は fast 非対象。

    // 2-6d: preset 名を解決する (CLI 優先・無ければ env・既定 "illustrious-xl")。
    //   "base" は「preset なし = base 重み」の明示値として扱う (resolve 前に空へ変換)。
    std::string preset = !preset_cli.empty()
                              ? preset_cli
                              : resolve_path("DOLLAMA_BACKEND_PRESET", "illustrious-xl");
    if (preset == "base")
    {
        log << "[preset] base (explicit)\n";
        preset.clear();
    }
    std::optional<PresetPaths> preset_paths;
    if (!preset.empty())
    {
        preset_paths = resolve_preset_paths(preset, default_model_roots());
        if (preset_paths)
        {
            log << "[preset] " << preset << " dir=" << preset_paths->dir << "\n";
            log << "  unet='" << preset_paths->unet << "'\n";
            log << "  vae='" << preset_paths->vae << "'\n";
            log << "  enc_l='" << preset_paths->enc_l << "'\n";
            log << "  enc_g='" << preset_paths->enc_g << "'\n";
        }
        else
        {
            log << "[warn] preset '" << preset
                << "' が models/presets/ に見つからない "
                   "(unet/vae/text-encoder-l/g の 4 ファイルが要る) → base にフォールバック\n";
        }
    }

    // 重み/golden パスを解決 (env 変数で上書き可。既定は test data パス。preset が
    // 解決できていれば preset のパスを fallback にする・個別 env は従来通り優先)。
    // 既定は src/tests/data。本番の重み配置先が決まったら DEFAULT を差し替える。
    const std::string unet_w = resolve_path(
        "DOLLAMA_UNET_WEIGHTS",
        preset_paths ? preset_paths->unet : "src/tests/data/unet_weights.safetensors");
    const std::string vae_w = resolve_path(
        "DOLLAMA_VAE_WEIGHTS",
        preset_paths ? preset_paths->vae : "src/tests/data/vae_weights.safetensors");
    const std::string embeds =
        resolve_path("DOLLAMA_EMBEDS", "src/tests/data/unet_io.safetensors");

    std::unique_ptr<IImageGenerator> gen;

    // ----------------------------------------------------------------
    // DI 3 段フォールバック:
    //   段1) OV アセット (tokenizer/encoder L/G + tokenizers.dll) + unet/vae 重みが
    //        揃う → make_backend(...) → BackendImageGenerator (prompt を反映する本 txt2img)。
    //   段2) unet/vae 重みのみ (OV 無 / アセット欠) → PipelineGenerator (golden 埋め込み)。
    //   段3) いずれも無 → StubGenerator。
    // 段1 の backend 実体 (SDXL) は HAVE_OPENVINO かつ runner (CUDA) が必要。両ガードが
    // 揃わないビルドでは段1 をコンパイル時に丸ごと無効化し、段2/3 へ落ちる。
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
        const std::string enc_l = resolve_path(
            "DOLLAMA_ENCODER_L",
            preset_paths ? preset_paths->enc_l
                         : find_model_xml("sdxl-text-encoder-l/model_ov.xml"));
        const std::string enc_g = resolve_path(
            "DOLLAMA_ENCODER_G",
            preset_paths ? preset_paths->enc_g
                         : find_model_xml("sdxl-text-encoder-g/model_ov.xml"));
        // openvino_tokenizers.dll は env のみ (空なら段1 をスキップ)。
        const std::string tok_dll = resolve_path("DOLLAMA_OV_TOKENIZERS_DLL", "");

        // 中1: 段1 で最終的に使う 4 パスをログする (preset 非空のときだけ・preset 空時は不変)。
        //   個別 env が preset の解決値を上書きしていれば (env override) を注記する。
        if (!preset.empty())
        {
            auto log_resolved = [&](const char* field, const std::string& final_v,
                                     const std::string& preset_v)
            {
                const bool env_override = preset_paths && final_v != preset_v;
                log << "[preset] resolved: " << field << "='" << final_v << "'"
                    << (env_override ? " (env override)" : "") << "\n";
            };
            log_resolved("unet", unet_w, preset_paths ? preset_paths->unet : "");
            log_resolved("vae", vae_w, preset_paths ? preset_paths->vae : "");
            log_resolved("enc_l", enc_l, preset_paths ? preset_paths->enc_l : "");
            log_resolved("enc_g", enc_g, preset_paths ? preset_paths->enc_g : "");
        }

        // backend 選択 (env 既定 "sdxl"・resolve_path と同流儀で getenv)。
        const std::string backend_name = resolve_path("DOLLAMA_BACKEND", "sdxl");

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
            // BackendConfig を組み立てる共通ラムダ (device のみ差し替えて NPU→CPU する)。
            auto make_cfg = [&](const std::string& device) -> BackendConfig
            {
                BackendConfig cfg;
                cfg.backend_name = backend_name;
                cfg.preset       = preset_paths ? preset : ""; // 2-6d: フォールバック時は名前を残さない
                cfg.unet_weights = unet_w;
                cfg.vae_weights  = vae_w;
                cfg.embeds       = embeds;
                cfg.tok_l        = tok_l;
                cfg.tok_g        = tok_g;
                cfg.enc_l        = enc_l;
                cfg.enc_g        = enc_g;
                cfg.tok_dll      = tok_dll;
                cfg.device_l     = device;
                cfg.device_g     = device;
                cfg.fast_cfg     = fast_cfg; // G-0b: 拡散経路へ運ぶだけ
                return cfg;
            };

            // preset が実際に解決できたときだけログにタグを足す (preset 空 or 解決失敗 → 従来通り無タグ)。
            //   4 本すべて preset のパスが最終採用されていれば [preset=<name>]、
            //   1 本でも個別 env が上書きしていれば [preset=<name>+env] にする。
            std::string preset_tag;
            std::optional<PresetPrefix> preset_prefix; // 2-6e: 既定 nullopt (preset 無/env override 時)
            if (preset_paths)
            {
                const bool all_from_preset =
                    unet_w == preset_paths->unet && vae_w == preset_paths->vae &&
                    enc_l == preset_paths->enc_l && enc_g == preset_paths->enc_g;
                preset_tag = all_from_preset ? ("[preset=" + preset + "] ")
                                              : ("[preset=" + preset + "+env] ");

                // 2-6e: preset.json の prompt_prefix/negative_prefix。4 本すべてが preset
                //   由来のときだけ読む (個別 env で 1 本でも上書きされていれば preset.json も
                //   別ものである可能性が高いため skip・ログで明示する)。
                if (all_from_preset)
                {
                    preset_prefix = read_preset_prefix(preset_paths->dir);
                    if (!preset_prefix)
                    {
                        // 不在 (ファイルそのものが無い) と、存在するが読めない/空
                        // (壊れた JSON・非 object・両フィールドとも欠落 or 空) を
                        // ログ上で切り分ける (原因調査を容易にするため)。
                        std::error_code ec;
                        const bool exists = fs::is_regular_file(
                            preset_paths->dir + "preset.json", ec);
                        if (exists && !ec)
                        {
                            log << "[preset] prefix: none (preset.json が読めない/空)\n";
                        }
                        else
                        {
                            log << "[preset] prefix: none (preset.json 不在)\n";
                        }
                    }
                }
                else
                {
                    log << "[preset] prefix skipped (env override)\n";
                }
            }

            // NPU 第一・失敗時 CPU フォールバックで backend を構築 → BackendImageGenerator。
            //   make_backend は nullptr 契約 (未知名 / OV 無 / 構築失敗 → nullptr)。
            std::unique_ptr<IDiffusionBackend> backend = make_backend(make_cfg("NPU"));
            if (backend)
            {
                gen = std::make_unique<BackendImageGenerator>(std::move(backend), preset_prefix);
                log << "dollama HTTP server (" << backend_name << " backend "
                    << preset_tag << "— NPU)\n";
            }
            else
            {
                log << "[warn] NPU での backend '" << backend_name
                    << "' 構築に失敗 → CPU を試します。\n";
                backend = make_backend(make_cfg("CPU"));
                if (backend)
                {
                    gen = std::make_unique<BackendImageGenerator>(std::move(backend), preset_prefix);
                    log << "dollama HTTP server (" << backend_name << " backend "
                        << preset_tag << "— CPU)\n";
                }
                else
                {
                    log << "[warn] CPU でも構築に失敗 → 段2/3 へフォールバックします。\n";
                }
            }
        }
    }
#endif

    // 段2) 段1 が立たなければ PipelineGenerator を試みる。
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

    // ----------------------------------------------------------------
    // B-5-3: 品質スコアラ (IScorer) を後付け注入する (gen 確定後・matting 注入の後に 1 回)。
    //   - モデル xml: env DOLLAMA_SCORER_WEIGHTS 優先。既定は HAVE_OPENVINO 時のみ
    //     models/ ツリーを find_model_xml で探索 (OV 無時は空文字 → stub nullptr)。
    //   - device: env DOLLAMA_SCORER_DEVICE 優先。既定 "NPU" (拡散中遊休 NPU で並列採点・
    //     iGPU は matting が専有するため NPU を既定にする。quality_scorer.hpp 設計意図)。
    //   make_scorer は OV 無効ビルドで stub が常に nullptr → set_scorer no-op で不採点。
    {
#ifdef HAVE_OPENVINO
        const std::string scorer_xml =
            resolve_path("DOLLAMA_SCORER_WEIGHTS",
                         find_model_xml("scorer-net/model_ov_fp32.xml"));
#else
        const std::string scorer_xml = resolve_path("DOLLAMA_SCORER_WEIGHTS", "");
#endif
        const std::string scorer_dev = resolve_path("DOLLAMA_SCORER_DEVICE", "NPU");

        std::unique_ptr<IScorer> s = make_scorer(scorer_xml, scorer_dev);
        if (s)
        {
            gen->set_scorer(std::move(s));
            log << "  scorer: " << scorer_dev << " (model='" << scorer_xml << "')\n";
        }
        else
        {
            log << "  scorer: 無効 (モデル/OV 無 — 不採点)\n";
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

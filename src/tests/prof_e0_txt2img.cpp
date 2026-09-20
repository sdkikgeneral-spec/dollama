// E-0: illustrious-xl preset で実プロンプト txt2img A/B (計測専用・meson test 非登録)。
//
// 目的:
//   cli_generate.hpp の NPU 既定経路が現在の研究機で OV ドライバ都合により
//   ZE_RESULT_ERROR_INVALID_NATIVE_BINARY を出す (E-0 とは無関係の既存環境事情) ため、
//   text encoder (CLIP-L/bigG) だけ device=CPU を明示して SDXLBackend を直接構築し、
//   同一 prompt/negative/seed で vae_scaling_factor=0.13025 (A) / 0.18215 (B) を
//   4 seed 分生成して PNG を吐く。BackendConfig::vae_scaling_factor の貫通確認も兼ねる。
//
// 使い方:
//   prof_e0_txt2img.exe <out_dir>
//   (アセットパスは models/ ツリー・src/tests/data 前提でハードコードする。
//    cwd はリポジトリルート (main dollama) を想定。)
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "server/diffusion_backend.hpp"

#ifdef HAVE_OPENVINO
#include "server/sdxl_backend.hpp"
#include "server/png.hpp"
#include "server/scorer_runner.hpp"
#endif

int main(int argc, char** argv)
{
#if defined(HAVE_OPENVINO) && defined(HAVE_CUDA)
    if (argc < 2)
    {
        std::cerr << "usage: prof_e0_txt2img <out_dir>\n";
        return 1;
    }
    const std::string out_dir = argv[1];
    std::filesystem::create_directories(out_dir);

    const std::string preset_dir = "models/presets/illustrious-xl/";
    const std::string tok_l = "models/sdxl-tokenizer-l/openvino_tokenizer.xml";
    const std::string tok_g = "models/sdxl-tokenizer-g/openvino_tokenizer.xml";
    const std::string enc_l = preset_dir + "text-encoder-l/model_ov.xml";
    const std::string enc_g = preset_dir + "text-encoder-g/model_ov.xml";
    const std::string tok_dll =
        "C:\\Users\\sdkik\\AppData\\Local\\Python\\pythoncore-3.14-64\\Lib\\site-packages\\"
        "openvino_tokenizers\\lib\\openvino_tokenizers.dll";
    const std::string unet_w = preset_dir + "unet_weights.safetensors";
    const std::string vae_w  = preset_dir + "vae_weights.safetensors";
    const std::string embeds = "src/tests/data/unet_io.safetensors"; // ctor 要件のみ (txt2img では未使用)

    const std::string prompt   = "1girl, solo, silver hair, blue eyes, simple background";
    const std::string negative = "lowres, bad anatomy, worst quality";
    const std::vector<uint64_t> seeds = {1234, 42, 777, 2024};
    const std::vector<float> sfs = {0.13025f, 0.18215f};

    // 既存 ScorerNet (B-3〜B-5) で 8 軸+quality を採点する。NPU は現在このマシンで
    // ZE_RESULT_ERROR_INVALID_NATIVE_BINARY を出すため device=CPU (golden 突合用途とも一致)。
    const std::string scorer_xml = "models/scorer-net/model_ov_fp32.xml";
    std::unique_ptr<dollama::IScorer> scorer = dollama::make_scorer(scorer_xml, "CPU");
    if (!scorer)
    {
        std::cerr << "[prof_e0_txt2img] [warn] ScorerNet 構築に失敗 — 採点なしで続行\n";
    }

    for (float sf : sfs)
    {
        const char* tag = (sf > 0.15f) ? "B_0.18215" : "A_0.13025";
        std::cout << "=== building SDXLBackend (CPU/CPU, vae_scaling_factor=" << sf
                  << ") ===\n";
        try
        {
            dollama::SDXLBackend backend(
                tok_l, tok_g, enc_l, enc_g, tok_dll,
                unet_w, vae_w, embeds,
                "CPU", "CPU", dollama::FastConfig{}, "illustrious-xl", sf);

            for (uint64_t seed : seeds)
            {
                std::vector<uint8_t> rgb;
                int w = 0, h = 0;
                // seed は backend.generate() のシグネチャに無いため、DOLLAMA_SEED 経由でなく
                // ここでは backend が内部の乱数無し (generate 自体は seed 引数なし) ため、
                // IDiffusionBackend::generate は seed を取る (steps, seed, cfg, w, h)。
                backend.generate(prompt, negative, /*steps=*/20, seed, /*cfg=*/7.5f,
                                 1024, 1024, rgb, w, h);
                std::cout << "  seed=" << seed << " w=" << w << " h=" << h
                          << " bytes=" << rgb.size() << "\n";

                if (scorer)
                {
                    try
                    {
                        const dollama::ScorerResult sr = scorer->score(rgb, w, h);
                        std::cout << "  [scorer] quality=" << sr.quality << " axis=[";
                        for (int i = 0; i < 8; ++i)
                        {
                            std::cout << sr.axis[i] << (i < 7 ? "," : "");
                        }
                        std::cout << "]\n";
                    }
                    catch (const std::exception& e)
                    {
                        std::cerr << "  [scorer] FAIL: " << e.what() << "\n";
                    }
                }

                // PNG 化は server/png.hpp の不透明 PNG エンコーダをそのまま使う。
                std::vector<uint8_t> png = dollama::encode_png_rgb8(rgb, w, h);
                const std::string path = out_dir + "/" + tag + "_s" + std::to_string(seed) + ".png";
                std::ofstream ofs(path, std::ios::binary);
                ofs.write(reinterpret_cast<const char*>(png.data()),
                          static_cast<std::streamsize>(png.size()));
                std::cout << "  wrote " << path << " (" << png.size() << " bytes)\n";
            }
        }
        catch (const std::exception& e)
        {
            std::cerr << "[prof_e0_txt2img] FAIL (" << tag << "): " << e.what() << "\n";
            return 1;
        }
    }
    return 0;
#else
    (void)argc; (void)argv;
    std::cerr << "[prof_e0_txt2img] HAVE_OPENVINO && HAVE_CUDA 必須\n";
    return 1;
#endif
}

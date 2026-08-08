// FAST モード診断用 warm ベンチ (計測専用・既存テスト無改変)
//
// 目的: test_unet_fast の 1step 計測は後方互換 API (launch_unet(SafeTensors&,...)) を
//   使っており、毎 call で DeviceWeights を再構築 = 全重み 4.9GB を再 H2D するため
//   cold 相当 (attention 比率が weight upload で希釈される)。
//   本ベンチは常駐ハンドル API (unet_weights_create) で重みを一度だけ upload し、
//   warm な per-step レイテンシを fast on/off で直接比較する。
//
// 出力:
//   - default / fast の warm per-step ms (中央値 + min/max, cudaEvent)
//   - DOLLAMA_PROFILE=1 のときは cat_attention_sec (attention 段のみの累積) も
//     fast on/off で分けて報告 (段2 profile の 4.621s/20step = 231ms/step に対応)。
//     ※ profile 有効時は ScopedSyncTimer の cudaDeviceSynchronize が入るため
//        per-step wall は素の値より膨らむ。純 wall は profile 無効で読むこと。
//
// 使い方:
//   build/src/prof_unet_fast_warm.exe                  … 純 warm wall 比較
//   DOLLAMA_PROFILE=1 build/src/prof_unet_fast_warm.exe … attention 内訳つき

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#ifdef HAVE_CUDA
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "infer/unet.cuh"
#include "infer/profile.cuh"
#include "kernels/utils.cuh"
#include "io/safetensors.hpp"
#endif

namespace dollama
{

#ifdef HAVE_CUDA

static std::vector<__half> load_f16(const SafeTensors& st, const std::string& name)
{
    size_t nbytes = 0;
    const uint8_t* p = st.tensor_bytes(name, nbytes);
    std::vector<__half> out(nbytes / sizeof(__half));
    std::memcpy(out.data(), p, nbytes);
    return out;
}

static std::vector<float> load_f32(const SafeTensors& st, const std::string& name)
{
    size_t nbytes = 0;
    const uint8_t* p = st.tensor_bytes(name, nbytes);
    std::vector<float> out(nbytes / sizeof(float));
    std::memcpy(out.data(), p, nbytes);
    return out;
}

static int run_bench()
{
    const std::string wpath  = UNET_WEIGHTS_PATH;
    const std::string iopath = UNET_IO_PATH;
    {
        std::ifstream fw(wpath, std::ios::binary), fio(iopath, std::ios::binary);
        if (!fw.good() || !fio.good())
        {
            std::cout << "[prof_unet_fast_warm] [SKIP] golden data not found\n";
            return 0;
        }
    }

    SafeTensors weights(wpath);
    SafeTensors io(iopath);

    std::vector<__half> h_latent = load_f16(io, "input_sample_scaled_step0_f16");
    std::vector<__half> h_ehs    = load_f16(io, "input_encoder_hidden_states_f16");
    std::vector<__half> h_txt    = load_f16(io, "input_text_embeds_f16");
    std::vector<__half> h_tids   = load_f16(io, "input_time_ids_f16");
    std::vector<float>  h_ts     = load_f32(io, "input_timestep_f32");
    const float timestep = h_ts[0];

    const size_t latent_n = (size_t)4 * 128 * 128;
    const size_t ehs_n    = (size_t)77 * 2048;

    __half *d_latent = nullptr, *d_ehs = nullptr, *d_txt = nullptr, *d_tids = nullptr, *d_np = nullptr;
    CUDA_CHECK(cudaMalloc(&d_latent, latent_n * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_ehs,    ehs_n    * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_txt,    1280     * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_tids,   6        * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_np,     latent_n * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_latent, h_latent.data(), latent_n * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ehs,    h_ehs.data(),    ehs_n    * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_txt,    h_txt.data(),    1280     * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tids,   h_tids.data(),   6        * sizeof(__half), cudaMemcpyHostToDevice));

    // 常駐ハンドル: 全重みを一度だけ upload (以降の step は重み転送ゼロ = warm)。
    std::cout << "[prof_unet_fast_warm] uploading weights (once)...\n";
    UnetWeightsHandle handle = unet_weights_create(weights);

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    const bool prof = profile_enabled();
    const int  warmup = 3;
    const int  iters  = 10;

    // G-4k 再 profile: resnet バケット (cat_resnet_sec) を per-step + ×20step で報告し
    //   ≤0.95s ゲート判定に供する。epilogue フラグも貫通させ fast+epilogue の resnet を測る。
    const int gen_steps = 20;  // 正典 20step 生成でのバケット換算

    auto bench = [&](bool attn_fast, bool epilogue)
    {
        for (int i = 0; i < warmup; ++i)
        {
            launch_unet(handle, d_latent, timestep, d_ehs, d_txt, d_tids, d_np, attn_fast, epilogue);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        if (prof)
        {
            profile_counters().reset();
        }
        std::vector<float> t;
        t.reserve(iters);
        for (int i = 0; i < iters; ++i)
        {
            CUDA_CHECK(cudaEventRecord(ev0));
            launch_unet(handle, d_latent, timestep, d_ehs, d_txt, d_tids, d_np, attn_fast, epilogue);
            CUDA_CHECK(cudaEventRecord(ev1));
            CUDA_CHECK(cudaEventSynchronize(ev1));
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
            t.push_back(ms);
        }
        std::vector<float> s = t;
        std::sort(s.begin(), s.end());
        const float med = s[s.size() / 2];
        std::cout << "[prof_unet_fast_warm] attn_fast=" << (attn_fast ? 1 : 0)
                  << " epilogue=" << (epilogue ? 1 : 0)
                  << " warm per-step: median=" << med << " ms"
                  << " min=" << s.front() << " max=" << s.back()
                  << " (N=" << iters << ", warmup=" << warmup << ")\n";
        if (prof)
        {
            const double attn_ms   = profile_counters().cat_attention_sec   * 1000.0 / iters;
            const double tfmr_ms   = profile_counters().cat_transformer_sec * 1000.0 / iters;
            const double resnet_ms = profile_counters().cat_resnet_sec      * 1000.0 / iters;
            const double resnet_bucket = resnet_ms * gen_steps / 1000.0;  // 20step バケット [s]
            std::cout << "[prof_unet_fast_warm]   [PROFILE] attention/step=" << attn_ms
                      << " ms  transformer/step=" << tfmr_ms
                      << " ms  resnet/step=" << resnet_ms << " ms"
                      << "  weight_upload=" << profile_counters().weight_upload_sec
                      << " s (must be ~0 when warm)\n";
            std::cout << "[prof_unet_fast_warm]   [RESNET-BUCKET] resnet x" << gen_steps
                      << "step=" << resnet_bucket << " s  (gate: <=0.95s / stretch <=0.85s"
                      << " / baseline 1.225s)\n";
        }
    };

    bench(false, false);         // default
    bench(true,  false);         // fast (attn only)
    bench(true,  true);          // fast + epilogue (G-4k S1a/S1b/S2)
    bench(false, false);         // 順序影響 (クロック/温度) の確認用に default を再計測

    unet_weights_destroy(handle);
    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    CUDA_CHECK(cudaFree(d_latent));
    CUDA_CHECK(cudaFree(d_ehs));
    CUDA_CHECK(cudaFree(d_txt));
    CUDA_CHECK(cudaFree(d_tids));
    CUDA_CHECK(cudaFree(d_np));
    return 0;
}

#endif // HAVE_CUDA

} // namespace dollama

int main()
{
#ifndef HAVE_CUDA
    std::cout << "[prof_unet_fast_warm] [SKIP] HAVE_CUDA undefined\n";
    return 0;
#else
    try
    {
        return dollama::run_bench();
    }
    catch (const std::exception& e)
    {
        std::cerr << "[prof_unet_fast_warm] exception: " << e.what() << "\n";
        return 1;
    }
#endif
}

// EulerDiscreteScheduler (SDXL) ゴールデン突合テスト (純 C++ / CUDA 不要)
// testing.md 形式: 各検証は if (!cond) { cerr; return false; }、
// main で ok = test_xxx() && ok 集約、成功は PASSED / 最後に ALL PASSED。
//
// golden は src/tests/data/unet_io.safetensors (S1 が生成)。突合キー:
//   sched_sigmas_f32              [21]            set_timesteps(20) の sigmas
//   sched_timesteps_f32          [20]            set_timesteps(20) の timesteps
//   input_sample_f32             [1,4,128,128]   step0 入力 latent (未スケール)
//   input_sample_scaled_step0_f32                scale_model_input(step0) 期待値
//   noise_pred_f32               [1,4,128,128]   step0 の UNet 出力 (epsilon)
//   sched_latent_after_step0_f32                 step(step0) の prev_sample 期待値
//
// golden 不在時は [SKIP] で return 0 (test_vae_decode と同じ作法)。
// パスは UNET_IO_PATH を meson で -D 埋め込み (cwd 非依存)。

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "infer/scheduler.hpp"
#include "io/safetensors.hpp"

#ifndef UNET_IO_PATH
#define UNET_IO_PATH "data/unet_io.safetensors"
#endif

namespace dollama
{

static const std::string g_golden = UNET_IO_PATH;

// SafeTensors から F32 テンソルを std::vector<float> に展開。
static std::vector<float> load_f32(const SafeTensors& st, const std::string& name)
{
    if (st.dtype(name) != StDtype::F32)
    {
        throw std::runtime_error("load_f32: '" + name + "' is not F32");
    }
    size_t nbytes = 0;
    const uint8_t* p = st.tensor_bytes(name, nbytes);
    const size_t n = nbytes / sizeof(float);
    std::vector<float> out(n);
    std::memcpy(out.data(), p, nbytes);
    return out;
}

// 2 配列の最大絶対誤差。
static double max_abs_err(const std::vector<float>& a, const std::vector<float>& b)
{
    double m = 0.0;
    const size_t n = a.size() < b.size() ? a.size() : b.size();
    for (size_t i = 0; i < n; ++i)
    {
        const double e = std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        if (e > m)
        {
            m = e;
        }
    }
    return m;
}

// ----------------------------------------------------------------
// set_timesteps(20) の sigmas / timesteps が golden と一致するか
// ----------------------------------------------------------------
static bool test_set_timesteps(const SafeTensors& st)
{
    std::vector<float> g_sigmas = load_f32(st, "sched_sigmas_f32");
    std::vector<float> g_ts     = load_f32(st, "sched_timesteps_f32");

    EulerDiscreteScheduler sched;
    sched.set_timesteps(20);

    const std::vector<float>& sigmas = sched.sigmas();
    const std::vector<float>& ts     = sched.timesteps();

    if (sigmas.size() != g_sigmas.size())
    {
        std::cerr << "[test_set_timesteps] sigmas サイズ不一致: got "
                  << sigmas.size() << " want " << g_sigmas.size() << "\n";
        return false;
    }
    if (ts.size() != g_ts.size())
    {
        std::cerr << "[test_set_timesteps] timesteps サイズ不一致: got "
                  << ts.size() << " want " << g_ts.size() << "\n";
        return false;
    }

    const double tol = 1e-4;
    const double sig_err = max_abs_err(sigmas, g_sigmas);
    const double ts_err  = max_abs_err(ts, g_ts);

    std::cout << "[test_set_timesteps] sigmas max_abs_err=" << sig_err
              << " timesteps max_abs_err=" << ts_err << "\n";

    if (sig_err > tol)
    {
        std::cerr << "[test_set_timesteps] sigmas 誤差超過: " << sig_err
                  << " > " << tol << "\n";
        return false;
    }
    if (ts_err > tol)
    {
        std::cerr << "[test_set_timesteps] timesteps 誤差超過: " << ts_err
                  << " > " << tol << "\n";
        return false;
    }
    std::cout << "[test_set_timesteps] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// scale_model_input(step0) が golden と一致するか
// ----------------------------------------------------------------
static bool test_scale_model_input(const SafeTensors& st)
{
    std::vector<float> sample = load_f32(st, "input_sample_f32");
    std::vector<float> golden = load_f32(st, "input_sample_scaled_step0_f32");

    EulerDiscreteScheduler sched;
    sched.set_timesteps(20);

    sched.scale_model_input(sample.data(), sample.size(), 0);

    // tol は F32 往復 + sqrt 演算誤差相応に緩める。
    const double tol = 1e-4;
    const double err = max_abs_err(sample, golden);
    std::cout << "[test_scale_model_input] max_abs_err=" << err << "\n";
    if (err > tol)
    {
        std::cerr << "[test_scale_model_input] 誤差超過: " << err << " > " << tol << "\n";
        return false;
    }
    std::cout << "[test_scale_model_input] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// step(step0) の prev_sample が golden と一致するか
// (diffusers の step は未スケールの input_sample を使う)
// ----------------------------------------------------------------
static bool test_step(const SafeTensors& st)
{
    std::vector<float> sample = load_f32(st, "input_sample_f32");
    std::vector<float> noise  = load_f32(st, "noise_pred_f32");
    std::vector<float> golden = load_f32(st, "sched_latent_after_step0_f32");

    if (sample.size() != noise.size() || sample.size() != golden.size())
    {
        std::cerr << "[test_step] 入力サイズ不一致\n";
        return false;
    }

    EulerDiscreteScheduler sched;
    sched.set_timesteps(20);

    std::vector<float> prev(sample.size());
    sched.step(noise.data(), 0, sample.data(), prev.data(), sample.size());

    const double tol = 1e-4;
    const double err = max_abs_err(prev, golden);
    std::cout << "[test_step] max_abs_err=" << err << "\n";
    if (err > tol)
    {
        std::cerr << "[test_step] 誤差超過: " << err << " > " << tol << "\n";
        return false;
    }
    std::cout << "[test_step] PASSED\n";
    return true;
}

} // namespace dollama

int main()
{
    // golden 不在なら SKIP (CUDA 不要だが golden が無い環境向けの作法を踏襲)
    {
        std::ifstream f(dollama::g_golden, std::ios::binary);
        if (!f)
        {
            std::cout << "[test_scheduler] [SKIP] golden が見つからない: "
                      << dollama::g_golden << "\n";
            return 0;
        }
    }

    bool ok = true;
    try
    {
        dollama::SafeTensors st(dollama::g_golden);
        ok = dollama::test_set_timesteps(st)     && ok;
        ok = dollama::test_scale_model_input(st) && ok;
        ok = dollama::test_step(st)              && ok;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[test_scheduler] 例外: " << e.what() << "\n";
        return 1;
    }

    if (!ok)
    {
        std::cerr << "[test_scheduler] FAILED\n";
        return 1;
    }
    std::cout << "[test_scheduler] ALL PASSED\n";
    return 0;
}

#pragma once
// SDXL 用 EulerDiscreteScheduler の純 C++ 実装 (ヘッダオンリー)。
//
// diffusers の EulerDiscreteScheduler を SDXL デフォルト設定で再現する。
// CUDA カーネルは使わず host 側 float (内部は double 蓄積) で計算する。
// latent は float 配列 (host) で受け渡す設計。UNet 連携は S5/S6 で行い、
// ここはスケジューラ単体の数値的正しさを目的とする。
//
// 設定 (diffusers SDXL base 既定):
//   num_train_timesteps = 1000
//   beta_start = 0.00085, beta_end = 0.012
//   beta_schedule = "scaled_linear"
//   timestep_spacing = "leading", steps_offset = 1
//   use_karras_sigmas = false
//   prediction_type = "epsilon"
//
// 検証 (test_scheduler.cpp):
//   golden unet_io.safetensors の sched_sigmas / sched_timesteps /
//   input_sample_scaled_step0 / sched_latent_after_step0 と突合済み。
//   spacing は "leading" が golden (sigmas[0]=11.028, timesteps[0]=951→[19]=1)
//   と一致するため採用 (linspace / trailing は不一致)。

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace dollama
{

class EulerDiscreteScheduler
{
public:
    // SDXL 既定設定でコンストラクト。alphas_cumprod から学習時 sigma 列を構築する。
    EulerDiscreteScheduler(int    num_train_timesteps = 1000,
                           double beta_start          = 0.00085,
                           double beta_end            = 0.012)
        : num_train_timesteps_(num_train_timesteps)
    {
        if (num_train_timesteps_ <= 0)
        {
            throw std::runtime_error("EulerDiscreteScheduler: num_train_timesteps must be > 0");
        }

        // scaled_linear: beta = linspace(sqrt(beta_start), sqrt(beta_end), N)^2
        const double sqrt_start = std::sqrt(beta_start);
        const double sqrt_end   = std::sqrt(beta_end);

        train_sigmas_.resize(static_cast<size_t>(num_train_timesteps_));
        double cumprod = 1.0;  // alphas_cumprod の逐次積
        for (int i = 0; i < num_train_timesteps_; ++i)
        {
            const double t    = (num_train_timesteps_ == 1)
                                    ? 0.0
                                    : static_cast<double>(i) / (num_train_timesteps_ - 1);
            const double sqrt_beta = sqrt_start + (sqrt_end - sqrt_start) * t;
            const double beta      = sqrt_beta * sqrt_beta;
            const double alpha     = 1.0 - beta;
            cumprod *= alpha;
            // sigma_i = sqrt((1 - alphas_cumprod_i) / alphas_cumprod_i)
            train_sigmas_[static_cast<size_t>(i)] =
                std::sqrt((1.0 - cumprod) / cumprod);
        }
    }

    // 推論ステップ数を指定し、sigmas / timesteps 配列を構築する。
    // timestep_spacing = "leading", steps_offset = 1 (diffusers SDXL 既定)。
    void set_timesteps(int num_inference_steps)
    {
        if (num_inference_steps <= 0)
        {
            throw std::runtime_error("EulerDiscreteScheduler: num_inference_steps must be > 0");
        }
        num_inference_steps_ = num_inference_steps;

        // leading spacing:
        //   step_ratio = num_train_timesteps // num_inference_steps
        //   timesteps  = (arange(0, num) * step_ratio).round() + steps_offset
        //   その後に降順 (reverse) へ。
        const int step_ratio   = num_train_timesteps_ / num_inference_steps;
        const int steps_offset = 1;

        std::vector<double> ts_ascending(static_cast<size_t>(num_inference_steps));
        for (int i = 0; i < num_inference_steps; ++i)
        {
            const double raw = std::round(static_cast<double>(i) * step_ratio)
                               + steps_offset;
            ts_ascending[static_cast<size_t>(i)] = raw;
        }

        // sigmas を学習 sigma 列から線形補間 (np.interp 相当)。
        // x 軸 = arange(num_train_timesteps), y 軸 = train_sigmas_。
        // その後 reverse して末尾に 0.0 を付与する。
        sigmas_.resize(static_cast<size_t>(num_inference_steps) + 1);
        timesteps_.resize(static_cast<size_t>(num_inference_steps));

        for (int i = 0; i < num_inference_steps; ++i)
        {
            // 昇順 timesteps を reverse して降順へ (j = 末尾から)。
            const int    j      = num_inference_steps - 1 - i;
            const double t       = ts_ascending[static_cast<size_t>(j)];
            timesteps_[static_cast<size_t>(i)] = static_cast<float>(t);
            sigmas_[static_cast<size_t>(i)]    =
                static_cast<float>(interp_train_sigma(t));
        }
        sigmas_[static_cast<size_t>(num_inference_steps)] = 0.0f;
    }

    // step_index における sigma で sample をスケールする。
    //   scaled = sample / sqrt(sigma^2 + 1)
    // sample は in-place 更新する (要素数 n)。
    void scale_model_input(float* sample, size_t n, int step_index) const
    {
        check_step_index(step_index);
        const double sigma = sigmas_[static_cast<size_t>(step_index)];
        const double scale = 1.0 / std::sqrt(sigma * sigma + 1.0);
        for (size_t k = 0; k < n; ++k)
        {
            sample[k] = static_cast<float>(static_cast<double>(sample[k]) * scale);
        }
    }

    // Euler 法 1 ステップ。prediction_type = "epsilon"。
    //   pred_original = sample - sigma * model_output
    //   derivative    = (sample - pred_original) / sigma = model_output
    //   dt            = sigma_next - sigma
    //   prev          = sample + derivative * dt
    // model_output / sample は要素数 n。結果 prev_sample[n] を書き込む。
    void step(const float* model_output, int step_index,
              const float* sample, float* prev_sample, size_t n) const
    {
        check_step_index(step_index);
        const double sigma      = sigmas_[static_cast<size_t>(step_index)];
        const double sigma_next = sigmas_[static_cast<size_t>(step_index) + 1];
        const double dt         = sigma_next - sigma;
        for (size_t k = 0; k < n; ++k)
        {
            // derivative = model_output (epsilon 予測時)
            const double deriv = static_cast<double>(model_output[k]);
            prev_sample[k] = static_cast<float>(
                static_cast<double>(sample[k]) + deriv * dt);
        }
    }

    // sigmas 配列 (要素数 num_inference_steps + 1、末尾は 0)。
    const std::vector<float>& sigmas() const
    {
        return sigmas_;
    }

    // timesteps 配列 (要素数 num_inference_steps、降順)。
    const std::vector<float>& timesteps() const
    {
        return timesteps_;
    }

    int num_inference_steps() const
    {
        return num_inference_steps_;
    }

private:
    int                num_train_timesteps_  = 1000;
    int                num_inference_steps_  = 0;
    std::vector<double> train_sigmas_;  // 学習時 sigma 列 (index 0..N-1)
    std::vector<float>  sigmas_;        // 推論用 sigma 列 (+末尾 0)
    std::vector<float>  timesteps_;     // 推論用 timestep 列 (降順)

    void check_step_index(int step_index) const
    {
        if (step_index < 0 || step_index >= num_inference_steps_)
        {
            throw std::runtime_error("EulerDiscreteScheduler: step_index out of range");
        }
    }

    // train_sigmas_ を x=arange(N) 上で線形補間する (np.interp 相当)。
    // t は [0, N-1] の実数。範囲外は端点でクランプ。
    double interp_train_sigma(double t) const
    {
        const int N = num_train_timesteps_;
        if (t <= 0.0)
        {
            return train_sigmas_.front();
        }
        if (t >= static_cast<double>(N - 1))
        {
            return train_sigmas_.back();
        }
        const int    lo   = static_cast<int>(std::floor(t));
        const int    hi   = lo + 1;
        const double frac = t - static_cast<double>(lo);
        const double a    = train_sigmas_[static_cast<size_t>(lo)];
        const double b    = train_sigmas_[static_cast<size_t>(hi)];
        return a + (b - a) * frac;
    }
};

} // namespace dollama

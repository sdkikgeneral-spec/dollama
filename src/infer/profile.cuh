// 拡散パイプライン プロファイラ — S0 (profiling-first) 計時基盤
// 対象: RTX5080 (Blackwell / sm_120) / CUDA Runtime API のみ
//
// 目的: 20step 生成の内訳を実測で割る。「重み転送/malloc」と「純カーネル計算」を
//       分離し、段グループ別 (embed / down / mid / up / conv_out)・VAE・host 往復・
//       総時間を秒で出す。
//
// 方針:
//   - 環境変数 DOLLAMA_PROFILE が立っているときだけ計時を有効化 (既定オフ・本番不変)。
//   - 重み転送 (DeviceWeights::get のキャッシュミス = cudaMalloc + H2D memcpy) は
//     同期 H2D なので CPU 時計 (std::chrono) で累積計時する。
//   - 段グループ / VAE / UNet step 全体は cudaDeviceSynchronize で区切り、CPU 時計で
//     累積計時する (cudaEvent でも可だが、段境界で既に同期が要るため壁時計で十分)。
//   - 全カウンタはグローバルシングルトン。launch_unet / launch_vae_decode /
//     DiffusionPipeline::generate から参照する。
//
// 注意: DOLLAMA_PROFILE 無効時は計時コードはほぼノーオペ (getenv キャッシュ判定のみ)。
#pragma once

#include <chrono>
#include <cstdint>
#include <cstdlib>

namespace dollama
{

// ----------------------------------------------------------------
// プロファイル有効判定 (環境変数 DOLLAMA_PROFILE。getenv は初回のみ・以後キャッシュ)。
// ----------------------------------------------------------------
inline bool profile_enabled()
{
    static int e = -1;
    if (e < 0)
    {
        e = std::getenv("DOLLAMA_PROFILE") != nullptr ? 1 : 0;
    }
    return e == 1;
}

// ----------------------------------------------------------------
// プロファイル累積カウンタ (秒・バイト・回数)。グローバルシングルトン。
// ----------------------------------------------------------------
struct ProfileCounters
{
    // 重み転送 (DeviceWeights::get キャッシュミス時の malloc + H2D)。
    double weight_upload_sec   = 0.0;  // 累積
    uint64_t weight_upload_bytes = 0;  // 累積バイト
    uint64_t weight_upload_count = 0;  // 累積回数 (= cudaMalloc 回数)

    // UNet step 全体 (launch_unet 1 回ぶんの壁時計。重み転送込み)。
    double unet_total_sec = 0.0;

    // 段グループ別 (壁時計・重み転送込み)。
    double unet_embed_sec    = 0.0;  // time embedding + add embedding
    double unet_down_sec     = 0.0;  // conv_in + down_blocks 0..2
    double unet_mid_sec      = 0.0;  // mid_block
    double unet_up_sec       = 0.0;  // up_blocks 0..2
    double unet_convout_sec  = 0.0;  // conv_norm_out + conv_out

    // カーネルカテゴリ別 (壁時計・段グループと直交した集計)。
    //   resnet_block (GroupNorm + conv2d + SiLU 主体) と transformer2d
    //   (attention + GEMM 主体) を分離し、up/down/mid の中で conv 律速か
    //   attention 律速かを判定する。
    double cat_resnet_sec      = 0.0;  // resnet_block 累計 (conv/groupnorm 主体)
    double cat_transformer_sec = 0.0;  // transformer2d 累計 (attention/gemm 主体)
    double cat_attention_sec   = 0.0;  // launch_attention のみ累計 (transformer の部分集合)

    // VAE decode 全体。
    double vae_sec = 0.0;

    // host 往復 (scale_model_input / FP16変換 / H2D / D2H / scheduler step)。
    double host_roundtrip_sec = 0.0;

    // 総時間 (generate 全体)。
    double total_sec = 0.0;

    int unet_steps = 0;  // launch_unet 呼び出し回数

    void reset()
    {
        *this = ProfileCounters{};
    }
};

// 単一の翻訳単位 (unet.cu) に実体を置き、他は extern 参照する。
ProfileCounters& profile_counters();

// ----------------------------------------------------------------
// 簡易スコープタイマ: 構築時に cudaDeviceSynchronize して開始、
// stop() で再度同期して経過秒を target に加算する。
// CUDA を直接 include したくないヘッダなので関数本体は .cu 側に置く。
// ----------------------------------------------------------------
class ScopedSyncTimer
{
public:
    // accum: 加算先 (nullptr なら無効)。enabled=false でもノーオペ。
    ScopedSyncTimer(double* accum, bool enabled);
    void stop();
    ~ScopedSyncTimer();

private:
    double* accum_;
    bool    enabled_;
    bool    stopped_;
    std::chrono::high_resolution_clock::time_point t0_;
};

} // namespace dollama

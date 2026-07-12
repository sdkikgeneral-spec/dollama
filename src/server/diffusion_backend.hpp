// IDiffusionBackend — 拡散アーキ (SDXL / SD3.5 / Stable Cascade ...) を差し替え可能に
// する純 cpp interface (プラグイン化の枠)。
//
// ----------------------------------------------------------------
// 設計意図 (重要):
//   dollama の品質天井は自作カーネルではなく「拡散重み (アーキ)」にある。芯 (自作 Tensor /
//   GEMM / Attention / CUDA カーネル) を汚さずに新アーキへ口を開けるため、prompt→RGB の
//   境界を 1 枚の純 cpp interface に切り出す。ComfyUI / diffusers のような breadth は追わず、
//   「2D キャラ生成に必要なアーキだけを、芯を共有しつつ差し替える」棲み分けにする。
//
//   本 interface は IDiffusionRunner / IMatter / IScorer と同じ純 cpp 規約を厳守する:
//     - CUDA 型 (__half 等) も OpenVINO 型も一切シグネチャに露出させない。
//     - CUDA / OV ヘッダを include しない (std と cstdint 等の標準ヘッダのみ)。
//     - #ifdef ガードで宣言を隠さない (interface は常に可視)。
//   これにより任意の cpp/.cu TU は IDiffusionBackend 越しに生成を呼べ、実体
//   (SDXL = OV+CUDA 依存) は sdxl_backend.hpp / diffusion_backend.cpp に隔離される。
//
//   境界は prompt → RGB。RGB 規約は IDiffusionRunner::run_txt2img の rgb_out/w/h を踏襲
//   ([H*W*3] の uint8 RGB, [y,x,c] = row-major HWC)。
// ----------------------------------------------------------------
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "server/fast_config.hpp" // FAST モードのフラグ枠 (G-0b・拡散経路へ運ぶだけ)

namespace dollama
{

// backend の静的メタ情報 (registry / UI / 上流の埋め込み整形が参照する)。
//   name             : backend 識別名 ("sdxl" / "sd35" 等)。
//   latent_channels  : VAE latent のチャネル数 (SDXL=4 / SD3.5=16)。
//   native_resolution: 学習解像度 (現状いずれも 1024)。
//   needs_t5         : T5-XXL テキストエンコーダを要するか (SD3.5=true)。
struct BackendInfo
{
    std::string name;
    int         latent_channels   = 0;
    int         native_resolution = 0;
    bool        needs_t5          = false;
};

// ランタイム LoRA (L-2) のリクエスト DTO (純 cpp・CUDA/OV 型なし)。
//   name     : LoRA 識別名。backend が DOLLAMA_LORA_DIR (既定 models/loras) 配下の
//              <name>.safetensors へ解決する。
//   strength : 適用強度 (scale = strength * alpha / rank)。
struct LoraRequest
{
    std::string name;
    float       strength = 1.0f;
};

// 拡散 backend の実行 interface。実体はアーキごとに (sdxl_backend.hpp 等で) 実装する。
struct IDiffusionBackend
{
    virtual ~IDiffusionBackend() = default;

    // ランタイム LoRA (L-2): 生成前に常駐重みへマージ / 生成後に復元する。
    //   default no-op (sd35 / stub は無改修で LoRA 無効)。SDXL が override する。
    //   name が解決できない場合は std::invalid_argument を投げる (HTTP 層が 4xx に変換)。
    virtual void apply_loras(const std::vector<LoraRequest>& reqs) { (void)reqs; }
    virtual void clear_loras() {}

    // prompt / negative から 1 枚生成し RGB8 を返す。
    //   prompt   : ポジティブプロンプト。
    //   negative : ネガティブプロンプト (空可)。
    //   steps    : 推論ステップ数 (1 未満は実体側が既定へ丸めてよい)。
    //   seed     : 初期ノイズのシード。
    //   cfg      : classifier-free guidance スケール (<=0 は実体側の既定を使ってよい)。
    //   w, h     : 要求解像度 (実体側が対応外を reject してよい)。
    //   rgb_out  : [h*w*3] の uint8 RGB ([y,x,c] = row-major HWC) で返す。
    //   w_out, h_out : 実際に生成した解像度。
    // 失敗時は std::runtime_error 等を投げてよい (呼び出し側が握る)。
    virtual void generate(const std::string& prompt, const std::string& negative,
                          int steps, uint64_t seed, float cfg, int w, int h,
                          std::vector<uint8_t>& rgb_out, int& w_out, int& h_out) = 0;

    // backend の静的メタ情報。
    virtual BackendInfo info() const = 0;

    // モデル識別子 (GET /v1/models で返す・"sdxl-1.0" 等)。
    virtual std::string model_id() const = 0;
};

// backend 構築設定 (registry factory が受ける)。
//   backend_name : 選択する backend ("sdxl" / "sd35")。
//   preset       : 将来のプリセット名 (現状未使用・拡張点)。
//   unet/vae/embeds : 拡散重み (SDXL が使う)。
//   tok_l/g・enc_l/g・tok_dll : OV text encoder アセット (SDXL が使う)。
//   device_l/g   : text encoder の実行デバイス ("NPU" / "CPU" 等)。
//   fast_cfg     : FAST モードフラグ (G-0b)。全 default false = 現行挙動。拡散経路へ運ぶだけ。
struct BackendConfig
{
    std::string backend_name;
    std::string preset;
    std::string unet_weights;
    std::string vae_weights;
    std::string embeds;
    std::string tok_l;
    std::string tok_g;
    std::string enc_l;
    std::string enc_g;
    std::string tok_dll;
    std::string device_l;
    std::string device_g;
    FastConfig  fast_cfg;
};

// backend_name に対応する IDiffusionBackend を構築する (registry factory)。
//
// nullptr 契約 (make_matter / make_scorer に倣う):
//   - 未知の backend_name → nullptr。
//   - "sdxl" は OpenVINO 無効ビルドでは構築できないため nullptr。
//   - 構築途中の例外は握って nullptr に倒す。
//   呼び出し側はこれを検知して段2/3 (PipelineGenerator / StubGenerator) へフォールバックできる。
//
// 実体は diffusion_backend.cpp に置き、OV 有無で sdxl_backend.hpp を条件参照する。
std::unique_ptr<IDiffusionBackend> make_backend(const BackendConfig& cfg);

} // namespace dollama

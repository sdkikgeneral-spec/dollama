// SDXL UNet2DConditionModel forward — 実装 (Phase 2 マイルストーン 2-5)
// 対象: RTX5080 (Blackwell / sm_120) / CUDA Runtime API のみ
//
// 既存プリミティブ (conv2d / group_norm / layer_norm / silu / gelu / geglu /
// attention / gemm / add / bias_add / timestep_embedding / upsample_nearest2x)
// を組み上げて latent[1,4,128,128] -> noise_pred[1,4,128,128] を生成する。
// 中間活性はすべて FP16 (各カーネル内部は FP32 蓄積)。S1 報告で UNet 全中間が
// FP16 範囲に収まることを確認済みのため、VAE のような FP32 経路は不要。
//
// 数値方針メモ:
//   - Linear (x@W^T) は launch_gemm_fp16(..., transB=true) を使い、bias は
//     launch_bias_add_rowvec で別途加算する (gemm に bias 引数が無いため)。
//   - ResnetBlock の time_emb_proj は temb[1280] を SiLU 後 Linear して [Cout] に
//     射影し、各チャネル平面へ broadcast 加算 (launch_bias_add_channel)。
//   - Transformer2D の attention は head_dim=64 固定。[tokens, heads*64] を
//     [heads, tokens, 64] へ転置してから launch_attention に渡す (B=1)。
//
// VRAM 設計: 段間受け渡しは局所 Scratch で都度確保し、スコープ脱出で解放する。
// UNet の最大解像度は 320ch x 128x128 = 5.24M 要素 (skip concat で 2560ch x 32x32
// = 2.6M など) で VAE より遥かに小さいため、ping-pong 大バッファは使わない。

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "infer/unet.cuh"
#include "infer/profile.cuh"
#include "kernels/conv2d.cuh"
#include "kernels/groupnorm.cuh"
#include "kernels/layernorm.cuh"
#include "kernels/activation.cuh"
#include "kernels/geglu.cuh"
#include "kernels/attention.cuh"
#include "kernels/gemm.cuh"
#include "kernels/elementwise.cuh"
#include "kernels/bias_add.cuh"
#include "kernels/timeembed.cuh"
#include "kernels/utils.cuh"

namespace dollama
{

// ----------------------------------------------------------------
// プロファイラ実体 (profile.cuh の宣言に対応)。
// ----------------------------------------------------------------
ProfileCounters& profile_counters()
{
    static ProfileCounters g;
    return g;
}

ScopedSyncTimer::ScopedSyncTimer(double* accum, bool enabled)
    : accum_(accum), enabled_(enabled), stopped_(false)
{
    if (enabled_)
    {
        // 開始前に既存カーネルを排出してから計時開始 (壁時計の純度を上げる)。
        cudaDeviceSynchronize();
        t0_ = std::chrono::high_resolution_clock::now();
    }
}

void ScopedSyncTimer::stop()
{
    if (enabled_ && !stopped_)
    {
        cudaDeviceSynchronize();
        const auto t1 = std::chrono::high_resolution_clock::now();
        const double sec = std::chrono::duration<double>(t1 - t0_).count();
        if (accum_ != nullptr) { *accum_ += sec; }
        stopped_ = true;
    }
}

ScopedSyncTimer::~ScopedSyncTimer()
{
    stop();
}

// ----------------------------------------------------------------
// 段ごとのゴールデン突合フック (テストが登録する)。
// launch_unet 内の各段で dbg_stat(name, ptr, n) が呼ばれるたびに、
// フックが登録されていれば (name, デバイスポインタ, 要素数) を渡す。
// test_unet がこのフックで段ごとに golden と突合する (粒度の細かい配線検証)。
// ----------------------------------------------------------------
using UnetStageHook = void (*)(const char* name, const __half* d_buf, size_t n);
static UnetStageHook g_stage_hook = nullptr;

void unet_set_stage_hook(UnetStageHook hook)
{
    g_stage_hook = hook;
}

// ----------------------------------------------------------------
// デバッグ用統計 (環境変数 UNET_DEBUG で有効化)。配線切り分け専用。
// ----------------------------------------------------------------
static bool unet_debug_enabled()
{
    static int e = -1;
    if (e < 0)
    {
        e = std::getenv("UNET_DEBUG") != nullptr ? 1 : 0;
    }
    return e == 1;
}

static void dbg_stat(const char* name, const __half* d_buf, size_t n)
{
    if (g_stage_hook != nullptr)
    {
        g_stage_hook(name, d_buf, n);
    }
    if (!unet_debug_enabled())
    {
        return;
    }
    cudaDeviceSynchronize();
    std::vector<__half> h(n);
    cudaMemcpy(h.data(), d_buf, n * sizeof(__half), cudaMemcpyDeviceToHost);
    float mn = 1e30f, mx = -1e30f;
    long bad = 0;
    for (size_t i = 0; i < n; ++i)
    {
        float v = __half2float(h[i]);
        if (std::isnan(v) || std::isinf(v)) { ++bad; continue; }
        if (v < mn) mn = v;
        if (v > mx) mx = v;
    }
    std::printf("[UNET_DEBUG] %-32s n=%zu min=%g max=%g bad=%ld head=%g\n",
                name, n, mn, mx, bad, __half2float(h[0]));
    std::fflush(stdout);
}

// ----------------------------------------------------------------
// 転置カーネル: [C, S] (row-major, S=H*W) <-> [S, C]
// Transformer2D で [1,C,H,W] を系列×チャネル [S=tokens, C] に並べ替える。
// ----------------------------------------------------------------
static __global__ void k_chw_to_sc(const __half* in, __half* out, int C, int S)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < C * S)
    {
        const int c = idx / S;
        const int s = idx - c * S;
        out[s * C + c] = in[idx];
    }
}

static __global__ void k_sc_to_chw(const __half* in, __half* out, int C, int S)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < C * S)
    {
        const int s = idx / C;
        const int c = idx - s * C;
        out[c * S + s] = in[idx];
    }
}

static void launch_chw_to_sc(const __half* d_in, __half* d_out, int C, int S)
{
    const int threads = 256;
    const int blocks  = ceil_div(C * S, threads);
    k_chw_to_sc<<<blocks, threads>>>(d_in, d_out, C, S);
    CUDA_CHECK_KERNEL();
}

static void launch_sc_to_chw(const __half* d_in, __half* d_out, int C, int S)
{
    const int threads = 256;
    const int blocks  = ceil_div(C * S, threads);
    k_sc_to_chw<<<blocks, threads>>>(d_in, d_out, C, S);
    CUDA_CHECK_KERNEL();
}

// ----------------------------------------------------------------
// マルチヘッド転置: [tokens, heads*Dh] <-> [heads, tokens, Dh]
// diffusers のヘッド分割: head h はチャネル [h*Dh:(h+1)*Dh] を取り、
// attention は [B=1, heads, tokens, Dh] レイアウトで計算する。
//   split:  in[t, h*Dh + d] -> out[(h*tokens + t)*Dh + d]
//   merge:  in[(h*tokens + t)*Dh + d] -> out[t, h*Dh + d]
// ----------------------------------------------------------------
static __global__ void k_split_heads(const __half* in, __half* out, int tokens, int heads, int Dh)
{
    const long total = (long)tokens * heads * Dh;
    for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += (long)gridDim.x * blockDim.x)
    {
        const int d = (int)(idx % Dh);
        long t1 = idx / Dh;
        const int h = (int)(t1 % heads);
        const int t = (int)(t1 / heads);
        // in は [tokens, heads*Dh] = [t, h*Dh + d]
        out[((long)h * tokens + t) * Dh + d] = in[(long)t * heads * Dh + h * Dh + d];
    }
}

static __global__ void k_merge_heads(const __half* in, __half* out, int tokens, int heads, int Dh)
{
    const long total = (long)tokens * heads * Dh;
    for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += (long)gridDim.x * blockDim.x)
    {
        const int d = (int)(idx % Dh);
        long t1 = idx / Dh;
        const int h = (int)(t1 % heads);
        const int t = (int)(t1 / heads);
        // out は [tokens, heads*Dh] = [t, h*Dh + d]
        out[(long)t * heads * Dh + h * Dh + d] = in[((long)h * tokens + t) * Dh + d];
    }
}

static void launch_split_heads(const __half* d_in, __half* d_out, int tokens, int heads, int Dh)
{
    const long total = (long)tokens * heads * Dh;
    long bl = (total + 255) / 256;
    const int blocks = (int)(bl < 65535 ? bl : 65535);
    k_split_heads<<<blocks, 256>>>(d_in, d_out, tokens, heads, Dh);
    CUDA_CHECK_KERNEL();
}

static void launch_merge_heads(const __half* d_in, __half* d_out, int tokens, int heads, int Dh)
{
    const long total = (long)tokens * heads * Dh;
    long bl = (total + 255) / 256;
    const int blocks = (int)(bl < 65535 ? bl : 65535);
    k_merge_heads<<<blocks, 256>>>(d_in, d_out, tokens, heads, Dh);
    CUDA_CHECK_KERNEL();
}

// ----------------------------------------------------------------
// channel 次元の concat: out[0:Ca] = a, out[Ca:Ca+Cb] = b ([N=1,C,H,W])
// up_block の skip concat (現活性 + down 経路 skip) に使う。
// ----------------------------------------------------------------
static __global__ void k_concat_chw(const __half* a, const __half* b, __half* out,
                                    int Ca, int Cb, int HW)
{
    const long total = (long)(Ca + Cb) * HW;
    for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += (long)gridDim.x * blockDim.x)
    {
        const int c = (int)(idx / HW);
        const int s = (int)(idx % HW);
        if (c < Ca)
        {
            out[idx] = a[(long)c * HW + s];
        }
        else
        {
            out[idx] = b[(long)(c - Ca) * HW + s];
        }
    }
}

static void launch_concat_chw(const __half* d_a, const __half* d_b, __half* d_out,
                              int Ca, int Cb, int HW)
{
    const long total = (long)(Ca + Cb) * HW;
    long bl = (total + 255) / 256;
    const int blocks = (int)(bl < 65535 ? bl : 65535);
    k_concat_chw<<<blocks, 256>>>(d_a, d_b, d_out, Ca, Cb, HW);
    CUDA_CHECK_KERNEL();
}

// ----------------------------------------------------------------
// ODR 注意 (T1b / C0 修正): 以下は本 TU (translation unit) ローカルの実装型で、
// 必ず匿名 namespace の中に置くこと。外に出すと ODR 違反で実行時に落ちる。
//
//   機序: unet.cu と vae_decode.cu は同名の DeviceWeights / Scratch を
//   それぞれ独自レイアウトで定義している。namespace dollama 直下 (外部リンケージ)
//   に置くと、暗黙 inline のデストラクタが COMDAT 弱シンボルとして出力され、
//   リンカが同名 dtor のうち「片方だけ」を選んで両 TU の全 call site に適用する。
//   どちらが勝つかはリンク単位ごとに変わるため、あるバイナリでは 0xC0000005
//   (アクセス違反)、別のバイナリでは単なるメモリリークとして現れる。
//
//   発火点: L-2 ランタイム LoRA で unet 側 DeviceWeights にだけ 4 メンバ目
//   std::set<std::string> patched_ を足したことでレイアウトが決定的にずれ、
//   ~DiffusionPipeline (VAE ハンドルと UNet ハンドルが同居して破棄される経路) で
//   AV が顕在化した。Scratch も ptrs_ が両方先頭なので現状は無症状なだけの同型の地雷。
//
//   よって「他 TU から参照されていないから外でもよい」ではなく、参照されていない
//   からこそ匿名 namespace が正解。善意で外に出すと再発する。
// ----------------------------------------------------------------
namespace
{
// ----------------------------------------------------------------
// 重み管理 (vae_decode と同方針): 名前から FP16 デバイスポインタを引く。
// ----------------------------------------------------------------
class DeviceWeights
{
public:
    explicit DeviceWeights(const SafeTensors& st) : st_(st) {}

    ~DeviceWeights()
    {
        for (__half* p : ptrs_)
        {
            cudaFree(p);
        }
    }

    const __half* get(const std::string& name)
    {
        auto it = cache_.find(name);
        if (it != cache_.end())
        {
            return it->second;
        }
        if (st_.dtype(name) != StDtype::F16)
        {
            throw std::runtime_error("unet: weight '" + name + "' is not FP16");
        }
        size_t nbytes = 0;
        const uint8_t* src = st_.tensor_bytes(name, nbytes);
        __half* d_ptr = nullptr;
        // S0 プロファイル: cudaMalloc + H2D (重み転送) を CPU 時計で累積計時。
        const bool prof = profile_enabled();
        std::chrono::high_resolution_clock::time_point wt0;
        if (prof) { wt0 = std::chrono::high_resolution_clock::now(); }
        CUDA_CHECK(cudaMalloc(&d_ptr, nbytes));
        CUDA_CHECK(cudaMemcpy(d_ptr, src, nbytes, cudaMemcpyHostToDevice));
        if (prof)
        {
            const auto wt1 = std::chrono::high_resolution_clock::now();
            ProfileCounters& pc = profile_counters();
            pc.weight_upload_sec   += std::chrono::duration<double>(wt1 - wt0).count();
            pc.weight_upload_bytes += nbytes;
            pc.weight_upload_count += 1;
        }

        ptrs_.push_back(d_ptr);
        cache_[name] = d_ptr;
        return d_ptr;
    }

    // S1: 常駐ハンドル生成時に全 FP16 重みを先読みして upload しておく。
    // こうすると拡散ループ (generate) 内の計測区間では重み転送が一切発生せず、
    // PROFILE の weight upload 内訳が 0 になる (= 純カーネル時間が見える)。
    // F16 以外のテンソル (もしあれば) は UNet では使わないのでスキップする。
    void upload_all()
    {
        for (const std::string& name : st_.names())
        {
            if (st_.dtype(name) == StDtype::F16)
            {
                (void)get(name);
            }
        }
    }

    // ----------------------------------------------------------------
    // ランタイム LoRA (L-2): 常駐重みへ delta = scale*(B@A) を焼き込む。
    //   forward 経路は 1 命令も変えない (重みバッファの中身だけが変わる)。
    //   scratch (A/B/delta) は都度確保・解放 (適用は generate 律速の外・sub 秒)。
    // ----------------------------------------------------------------
    void apply_lora(const LoraModule& m)
    {
        // base の numel と B@A の numel 整合 (load_lora_modules 済みだが防御的に再検証)
        const std::vector<size_t>& shp   = st_.shape(m.base_key);
        size_t                     numel = 1;
        for (size_t d : shp)
        {
            numel *= d;
        }
        const size_t delta_numel =
            static_cast<size_t>(m.out_dim) * static_cast<size_t>(m.in_flat);
        if (numel != delta_numel)
        {
            throw std::runtime_error("unet lora: '" + m.base_key + "' numel mismatch (base " +
                                     std::to_string(numel) + " vs delta " +
                                     std::to_string(delta_numel) + ")");
        }
        if (m.a_fp16.size() != static_cast<size_t>(m.rank) * m.in_flat ||
            m.b_fp16.size() != static_cast<size_t>(m.out_dim) * m.rank)
        {
            throw std::runtime_error("unet lora: '" + m.base_key + "' A/B buffer size mismatch vs shape");
        }

        // cache miss の key は先に常駐化してから patch する
        __half* d_w = const_cast<__half*>(get(m.base_key));

        // 変異前に patched_ へ登録する。GEMM/加算が途中失敗すると W は
        // 中途半端に書き換わり得るため、失敗しても unet_clear_loras が
        // base host バイトから必ず復元できるよう先に追跡する
        // (未変異のまま登録されても revert は base 再 upload なので無害)。
        patched_.insert(m.base_key);

        // scratch: A [rank, in_flat] / B [out, rank] / delta [out, in_flat]
        __half* d_a     = nullptr;
        __half* d_b     = nullptr;
        __half* d_delta = nullptr;
        try
        {
            CUDA_CHECK(cudaMalloc(&d_a, m.a_fp16.size() * sizeof(__half)));
            CUDA_CHECK(cudaMalloc(&d_b, m.b_fp16.size() * sizeof(__half)));
            CUDA_CHECK(cudaMalloc(&d_delta, delta_numel * sizeof(__half)));
            CUDA_CHECK(cudaMemcpy(d_a, m.a_fp16.data(), m.a_fp16.size() * sizeof(__half),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_b, m.b_fp16.data(), m.b_fp16.size() * sizeof(__half),
                                  cudaMemcpyHostToDevice));

            // delta = scale * (B @ A): [out, rank] @ [rank, in_flat] -> [out, in_flat]
            //   alpha=scale, beta=0 → delta は初めからスケール済み (FP32 蓄積)。
            launch_gemm_fp16(d_b, d_a, d_delta,
                             /*M=*/m.out_dim, /*N=*/m.in_flat, /*K=*/m.rank,
                             /*alpha=*/m.scale, /*beta=*/0.0f,
                             /*transA=*/false, /*transB=*/false);

            // W += delta (launch_add は out==a の in-place 安全)
            launch_add(d_w, d_delta, d_w, static_cast<int>(delta_numel));
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        catch (...)
        {
            if (d_a != nullptr)     { cudaFree(d_a); }
            if (d_b != nullptr)     { cudaFree(d_b); }
            if (d_delta != nullptr) { cudaFree(d_delta); }
            throw;
        }
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_delta);
    }

    // patch 済みの全 key を base SafeTensors の host バイトから再 upload して
    // bit-exact 復元する (st_ が base の真実源なのでデバイス側コピーは不要)。
    void revert_all()
    {
        for (const std::string& key : patched_)
        {
            auto it = cache_.find(key);
            if (it == cache_.end())
            {
                continue;  // 論理上到達しない (patch 時に必ず常駐化している)
            }
            size_t         nbytes = 0;
            const uint8_t* src    = st_.tensor_bytes(key, nbytes);
            CUDA_CHECK(cudaMemcpy(it->second, src, nbytes, cudaMemcpyHostToDevice));
        }
        patched_.clear();
    }

private:
    const SafeTensors&             st_;
    std::map<std::string, __half*> cache_;
    std::vector<__half*>           ptrs_;
    std::set<std::string>          patched_;  // L-2: LoRA patch 済み key (clear で復元)
};

// ----------------------------------------------------------------
// 局所スクラッチプール (FP16)。段内で確保しスコープ脱出で全解放。
//   attn_fast: FAST モード (G-3k)。true のとき transformer_block の attention を
//              launch_attention_fast へ切り替える。既定 false = 現行 (default) 挙動。
//              Scratch は段 (down/mid/up ブロック) 単位で生成され transformer_block まで
//              流れるため、フラグの運搬役をそのまま兼ねる (追加の引数配線を避ける最小差分)。
//   epilogue : FAST モード (G-4k S1a)。true のとき resnet_block の norm1/norm2 の
//              GroupNorm を multi-block 版 (launch_group_norm_mb) へ切り替える。
//              既定 false = 現行 (default) 挙動。運搬方式は attn_fast の厳密ミラー。
// ----------------------------------------------------------------
class Scratch
{
public:
    Scratch() = default;
    explicit Scratch(bool attn_fast, bool epilogue = false)
        : attn_fast_(attn_fast), epilogue_(epilogue) {}

    __half* alloc(size_t n)
    {
        __half* p = nullptr;
        CUDA_CHECK(cudaMalloc(&p, n * sizeof(__half)));
        ptrs_.push_back(p);
        return p;
    }

    // FAST モードで attention 高速バリアントを使うか。
    bool attn_fast() const { return attn_fast_; }

    // FAST モードで GroupNorm multi-block 版 (epilogue 経路) を使うか。
    bool epilogue() const { return epilogue_; }

    ~Scratch()
    {
        for (__half* p : ptrs_)
        {
            cudaFree(p);
        }
    }

private:
    std::vector<__half*> ptrs_;
    bool                 attn_fast_ = false;
    bool                 epilogue_  = false;
};
} // namespace (匿名): TU ローカル型 DeviceWeights / Scratch ここまで

// ----------------------------------------------------------------
// Linear (x @ W^T + b): [M, K] @ [N, K]^T -> [M, N]
//   W は diffusers の [out, in] = [N, K] そのまま (transB=true で W^T)。
//   bias != nullptr のとき row-vector broadcast 加算。
// ----------------------------------------------------------------
static void linear(DeviceWeights& w, const std::string& wname, const std::string& bname,
                   const __half* d_in, __half* d_out, int M, int N, int K, bool has_bias)
{
    launch_gemm_fp16(d_in, w.get(wname), d_out, M, N, K, 1.0f, 0.0f, false, true);
    if (has_bias)
    {
        launch_bias_add_rowvec(d_out, w.get(bname), d_out, M, N);
    }
}

// ----------------------------------------------------------------
// ResnetBlock2D (diffusers, time embedding 付き)
//   h = conv1(silu(norm1(x)))                        norm:GN32 eps1e-5, conv:3x3 pad1
//   h += time_emb_proj(silu(temb)) を channel broadcast
//   h = conv2(silu(norm2(h)))
//   skip = (Cin==Cout) ? x : conv_shortcut(x)         (1x1)
//   out = h + skip
//   d_temb_silu: silu(temb) [1280] (呼び側で 1 回計算して共有)
//   d_out は x と別バッファ。スクラッチは sc から確保。
// ----------------------------------------------------------------
static void resnet_block(DeviceWeights& w, const std::string& prefix,
                         const __half* d_x, int B, int Cin, int Cout, int H, int W,
                         const __half* d_temb_silu, __half* d_out, Scratch& sc)
{
    // S0 カテゴリ計時: resnet_block 全体を resnet バケットへ (conv/groupnorm 主体)。
    ScopedSyncTimer cat_t(profile_enabled() ? &profile_counters().cat_resnet_sec : nullptr,
                          profile_enabled());
    const float eps    = 1e-5f;
    const int   groups = 32;
    const int   HW     = H * W;
    const size_t chw_in  = (size_t)Cin  * HW;
    const size_t chw_out = (size_t)Cout * HW;

    // G-2k S2: 中間バッファは B サンプル分。B=1 では従来と同一サイズ・同一呼び出し。
    __half* d_a = sc.alloc((size_t)B * (chw_in > chw_out ? chw_in : chw_out));
    __half* d_b = sc.alloc((size_t)B * chw_out);

    // norm1 -> silu -> conv1 (Cin -> Cout)。group_norm/conv2d は N=B を渡すだけ。
    // G-4k S1a/S1b: epilogue=on のみ multi-block GN + SiLU 融合 1 パス
    // (mb→別 launch_silu の 2 パスとビット一致・default から隔離)。
    if (sc.epilogue())
    {
        launch_group_norm_silu(d_x, w.get(prefix + "norm1.weight"),
                               w.get(prefix + "norm1.bias"),
                               d_a, B, Cin, H, W, groups, eps);
    }
    else
    {
        launch_group_norm(d_x, w.get(prefix + "norm1.weight"), w.get(prefix + "norm1.bias"),
                          d_a, B, Cin, H, W, groups, eps);
        launch_silu(d_a, d_a, B * Cin * HW);
    }
    // G-4k S2: epilogue=on のみ conv1.bias を後段融合カーネルへ移す (conv は bias なし)。
    //   融合が GEMM 経路の 2 段丸め列 (conv_bias_add_rows → bias_add_channel) と
    //   bit 一致するのは conv が GEMM 経路のときのみ。direct 経路 shape (GEMM 下限割れ)
    //   は bias の丸め列が違う (FP32 蓄積へ畳む単一丸め) ため、ガードで従来 2 パスへ
    //   フォールバックする (静かな bit 崩れ防止)。epilogue=off は従来経路そのまま。
    const bool fuse1 = sc.epilogue()
        && conv2d_uses_gemm_bias_path(B, Cin, H, W, Cout, 3, 3, 1, 1, 1, 1, 1, 1);
    if (fuse1)
    {
        launch_conv2d(d_a, w.get(prefix + "conv1.weight"), /*d_bias=*/nullptr,
                      d_b, B, Cin, H, W, Cout, 3, 3, 1, 1, 1, 1, 1, 1);
    }
    else
    {
        launch_conv2d(d_a, w.get(prefix + "conv1.weight"), w.get(prefix + "conv1.bias"),
                      d_b, B, Cin, H, W, Cout, 3, 3, 1, 1, 1, 1, 1, 1);
    }

    // time embedding: silu(temb) -> Linear -> [B,Cout] -> channel broadcast add。
    //   temb proj は per-b で異なる (temb は cond/uncond で別) ため M=B で射影し、
    //   bias_add_channel は bias が per-b で異なるので per-b に加算する (カーネル無改修)。
    __half* d_tproj = sc.alloc((size_t)B * Cout);
    linear(w, prefix + "time_emb_proj.weight", prefix + "time_emb_proj.bias",
           d_temb_silu, d_tproj, B, Cout, 1280, true);
    if (fuse1)
    {
        // G-4k S2 P1: conv1.bias broadcast + per-(n,c) time-bias を 1 カーネルに融合
        //   (2 段丸めを 1:1 再現し per-b ループ B+1 発を 1 発に置換)。
        launch_conv_bias_biasch(d_b, w.get(prefix + "conv1.bias"), d_tproj,
                                d_b, B, Cout, H, W);
    }
    else
    {
        for (int b = 0; b < B; ++b)
        {
            launch_bias_add_channel(d_b + (size_t)b * Cout * HW, d_tproj + (size_t)b * Cout,
                                    d_b + (size_t)b * Cout * HW, 1, Cout, H, W);
        }
    }

    // norm2 -> silu -> conv2 (Cout -> Cout)
    // G-4k S1b: norm1 と同じく epilogue=on は GN+SiLU 融合 1 パス。
    if (sc.epilogue())
    {
        launch_group_norm_silu(d_b, w.get(prefix + "norm2.weight"),
                               w.get(prefix + "norm2.bias"),
                               d_a, B, Cout, H, W, groups, eps);
    }
    else
    {
        launch_group_norm(d_b, w.get(prefix + "norm2.weight"), w.get(prefix + "norm2.bias"),
                          d_a, B, Cout, H, W, groups, eps);
        launch_silu(d_a, d_a, B * Cout * HW);
    }
    // G-4k S2: epilogue=on のみ conv2.bias + residual add を後段融合 (ガードは conv1 と同じ
    //   GEMM 経路保証のみ。false なら従来 2 パス)。conv_shortcut 自体は無改修 (bias 込み)。
    const bool fuse2 = sc.epilogue()
        && conv2d_uses_gemm_bias_path(B, Cout, H, W, Cout, 3, 3, 1, 1, 1, 1, 1, 1);
    if (fuse2)
    {
        launch_conv2d(d_a, w.get(prefix + "conv2.weight"), /*d_bias=*/nullptr,
                      d_out, B, Cout, H, W, Cout, 3, 3, 1, 1, 1, 1, 1, 1);
        // residual 決定: Cin==Cout は x そのまま、それ以外は 1x1 conv_shortcut(x) を d_b へ。
        const __half* d_res = d_x;
        if (Cin != Cout)
        {
            launch_conv2d(d_x, w.get(prefix + "conv_shortcut.weight"),
                          w.get(prefix + "conv_shortcut.bias"),
                          d_b, B, Cin, H, W, Cout, 1, 1, 1, 1, 0, 0, 1, 1);
            d_res = d_b;
        }
        // G-4k S2 P2: conv2.bias broadcast + residual add を 1 カーネルに融合
        //   (conv_bias_add_rows → add_fp16 の 2 段丸めを 1:1 再現)。
        launch_conv_bias_residual(d_out, w.get(prefix + "conv2.bias"), d_res,
                                  d_out, B, Cout, H, W);
    }
    else
    {
        launch_conv2d(d_a, w.get(prefix + "conv2.weight"), w.get(prefix + "conv2.bias"),
                      d_out, B, Cout, H, W, Cout, 3, 3, 1, 1, 1, 1, 1, 1);

        // skip
        if (Cin == Cout)
        {
            launch_add(d_out, d_x, d_out, B * Cout * HW);
        }
        else
        {
            launch_conv2d(d_x, w.get(prefix + "conv_shortcut.weight"),
                          w.get(prefix + "conv_shortcut.bias"),
                          d_b, B, Cin, H, W, Cout, 1, 1, 1, 1, 0, 0, 1, 1);
            launch_add(d_out, d_b, d_out, B * Cout * HW);
        }
    }
    (void)chw_out;
}

// ----------------------------------------------------------------
// BasicTransformerBlock (diffusers, 1 block)
//   入力 d_x: [tokens, C] (in-place 更新)
//   (1) norm1 -> self-attn(attn1) -> +residual
//   (2) norm2 -> cross-attn(attn2, encoder_hidden_states) -> +residual
//   (3) norm3 -> GEGLU FF -> +residual
//   attn1: to_q/k/v bias なし, to_out.0 bias あり, K/V も自身 (Sk=tokens)
//   attn2: to_q[C,C], to_k/to_v[C, ctx_dim=2048], K/V は encoder_hidden_states
//   head_dim=64。
//   d_ctx: encoder_hidden_states [ctx_tokens, ctx_dim] (cross K/V 源)
// ----------------------------------------------------------------
static void transformer_block(DeviceWeights& w, const std::string& prefix,
                              __half* d_x, int B, int tokens, int C,
                              const __half* d_ctx, int ctx_tokens, int ctx_dim,
                              Scratch& sc)
{
    const float ln_eps = 1e-5f;
    const int   Dh     = 64;
    const int   heads  = C / Dh;
    const float scale  = 1.0f / sqrtf((float)Dh);
    const size_t tc    = (size_t)tokens * C;   // 1 サンプルの [tokens,C]
    // G-2k S2: レイアウトは [B,tokens,C] 連続 = [B*tokens, C]。linear は M=B*tokens、
    //   split/merge heads は per-b オフセット呼び (in+b*tokens*C)、attention は B を渡す。
    //   B=1 では M=tokens・offset 0 の 1 周で従来と完全同一。
    const int    Mt    = B * tokens;           // GEMM の行数 (B サンプル束ね)

    __half* d_norm = sc.alloc((size_t)B * tc);
    __half* d_q    = sc.alloc((size_t)B * tc);
    __half* d_k    = sc.alloc((size_t)B * tc);  // self: B x tokens x C
    __half* d_v    = sc.alloc((size_t)B * tc);
    __half* d_qh   = sc.alloc((size_t)B * tc);  // split-heads Q [B,heads,tokens,Dh]
    __half* d_kh   = sc.alloc((size_t)B * tc);
    __half* d_vh   = sc.alloc((size_t)B * tc);
    __half* d_oh   = sc.alloc((size_t)B * tc);  // attn 出力 (B,heads,tokens,Dh)
    __half* d_o    = sc.alloc((size_t)B * tc);  // merge-heads 出力 (B,tokens,C)

    // ---- (1) self-attn ----
    launch_layer_norm(d_x, w.get(prefix + "norm1.weight"), w.get(prefix + "norm1.bias"),
                      d_norm, Mt, C, ln_eps);
    linear(w, prefix + "attn1.to_q.weight", "", d_norm, d_q, Mt, C, C, false);
    linear(w, prefix + "attn1.to_k.weight", "", d_norm, d_k, Mt, C, C, false);
    linear(w, prefix + "attn1.to_v.weight", "", d_norm, d_v, Mt, C, C, false);
    for (int b = 0; b < B; ++b)
    {
        const size_t off = (size_t)b * tokens * C;
        launch_split_heads(d_q + off, d_qh + off, tokens, heads, Dh);
        launch_split_heads(d_k + off, d_kh + off, tokens, heads, Dh);
        launch_split_heads(d_v + off, d_vh + off, tokens, heads, Dh);
    }
    {
        // S0 計時: self-attention のみ attention バケットへ。
        ScopedSyncTimer at(profile_enabled() ? &profile_counters().cat_attention_sec : nullptr,
                           profile_enabled());
        // FAST: attn_fast のとき高速バリアント (数学同一・SSIM=1.0)。既定は従来経路 (無改変)。
        if (sc.attn_fast())
        {
            launch_attention_fast(d_qh, d_kh, d_vh, d_oh, B, heads, tokens, tokens, Dh, scale);
        }
        else
        {
            launch_attention(d_qh, d_kh, d_vh, d_oh, B, heads, tokens, tokens, Dh, scale);
        }
    }
    for (int b = 0; b < B; ++b)
    {
        const size_t off = (size_t)b * tokens * C;
        launch_merge_heads(d_oh + off, d_o + off, tokens, heads, Dh);
    }
    // to_out.0 (Linear + bias)
    linear(w, prefix + "attn1.to_out.0.weight", prefix + "attn1.to_out.0.bias",
           d_o, d_norm, Mt, C, C, true);
    launch_add(d_x, d_norm, d_x, (int)((size_t)B * tc));

    // ---- (2) cross-attn ----
    launch_layer_norm(d_x, w.get(prefix + "norm2.weight"), w.get(prefix + "norm2.bias"),
                      d_norm, Mt, C, ln_eps);
    // Q from d_norm [B*tokens,C]; K/V from d_ctx [B,ctx_tokens,ctx_dim] -> [B,ctx_tokens,C]
    linear(w, prefix + "attn2.to_q.weight", "", d_norm, d_q, Mt, C, C, false);
    __half* d_kc = sc.alloc((size_t)B * ctx_tokens * C);
    __half* d_vc = sc.alloc((size_t)B * ctx_tokens * C);
    linear(w, prefix + "attn2.to_k.weight", "", d_ctx, d_kc, B * ctx_tokens, C, ctx_dim, false);
    linear(w, prefix + "attn2.to_v.weight", "", d_ctx, d_vc, B * ctx_tokens, C, ctx_dim, false);
    __half* d_kch = sc.alloc((size_t)B * ctx_tokens * C);
    __half* d_vch = sc.alloc((size_t)B * ctx_tokens * C);
    for (int b = 0; b < B; ++b)
    {
        launch_split_heads(d_q  + (size_t)b * tokens * C,     d_qh  + (size_t)b * tokens * C,
                           tokens,     heads, Dh);
        launch_split_heads(d_kc + (size_t)b * ctx_tokens * C, d_kch + (size_t)b * ctx_tokens * C,
                           ctx_tokens, heads, Dh);
        launch_split_heads(d_vc + (size_t)b * ctx_tokens * C, d_vch + (size_t)b * ctx_tokens * C,
                           ctx_tokens, heads, Dh);
    }
    {
        // S0 計時: cross-attention のみ attention バケットへ。
        ScopedSyncTimer at(profile_enabled() ? &profile_counters().cat_attention_sec : nullptr,
                           profile_enabled());
        // FAST: cross-attn も同様に切り替え (非対応 Dh は内部で従来経路へフォールバック)。
        if (sc.attn_fast())
        {
            launch_attention_fast(d_qh, d_kch, d_vch, d_oh, B, heads, tokens, ctx_tokens, Dh, scale);
        }
        else
        {
            launch_attention(d_qh, d_kch, d_vch, d_oh, B, heads, tokens, ctx_tokens, Dh, scale);
        }
    }
    for (int b = 0; b < B; ++b)
    {
        const size_t off = (size_t)b * tokens * C;
        launch_merge_heads(d_oh + off, d_o + off, tokens, heads, Dh);
    }
    linear(w, prefix + "attn2.to_out.0.weight", prefix + "attn2.to_out.0.bias",
           d_o, d_norm, Mt, C, C, true);
    launch_add(d_x, d_norm, d_x, (int)((size_t)B * tc));

    // ---- (3) GEGLU FF ----
    //   ff.net.0.proj: Linear [C -> 2*ff]  (+bias)
    //   GEGLU: [B*tokens, 2*ff] -> [B*tokens, ff]
    //   ff.net.2: Linear [ff -> C]  (+bias)
    const int ff = 4 * C;  // SDXL: ff_inner = 4*C (640->2560 proj は 5120=2*2560)
    launch_layer_norm(d_x, w.get(prefix + "norm3.weight"), w.get(prefix + "norm3.bias"),
                      d_norm, Mt, C, ln_eps);
    __half* d_proj = sc.alloc((size_t)Mt * 2 * ff);
    linear(w, prefix + "ff.net.0.proj.weight", prefix + "ff.net.0.proj.bias",
           d_norm, d_proj, Mt, 2 * ff, C, true);
    __half* d_geglu = sc.alloc((size_t)Mt * ff);
    launch_geglu(d_proj, d_geglu, Mt, ff);
    linear(w, prefix + "ff.net.2.weight", prefix + "ff.net.2.bias",
           d_geglu, d_norm, Mt, C, ff, true);
    launch_add(d_x, d_norm, d_x, (int)((size_t)B * tc));
}

// ----------------------------------------------------------------
// Transformer2D (diffusers, SDXL は norm が GroupNorm + proj_in/out が Linear)
//   residual = x ([1,C,H,W])
//   h = group_norm(x) (eps 1e-6, 32群)
//   [C,HW] -> [HW=tokens, C] 転置
//   h = proj_in(h)  (Linear C->C +bias)
//   transformer_blocks x num_layers (in-place on [tokens,C])
//   h = proj_out(h) (Linear C->C +bias)
//   [tokens,C] -> [C,HW] 転置
//   x = h + residual (in-place)
// ----------------------------------------------------------------
static void transformer2d(DeviceWeights& w, const std::string& prefix,
                          __half* d_x, int B, int C, int H, int W, int num_layers,
                          const __half* d_ctx, int ctx_tokens, int ctx_dim,
                          Scratch& sc)
{
    // S0 カテゴリ計時: transformer2d 全体を transformer バケットへ (attention/gemm 主体)。
    ScopedSyncTimer cat_t(profile_enabled() ? &profile_counters().cat_transformer_sec : nullptr,
                          profile_enabled());
    const float gn_eps = 1e-6f;
    const int   groups = 32;
    const int   S      = H * W;       // tokens (1 サンプル)
    const size_t cs    = (size_t)C * S;
    // G-2k S2: group_norm は N=B、chw<->sc 転置は per-b オフセット呼び、proj は M=B*S。
    const int    Ms    = B * S;

    __half* d_gn  = sc.alloc((size_t)B * cs);
    __half* d_sc  = sc.alloc((size_t)B * cs);     // [B, tokens, C]
    __half* d_pin = sc.alloc((size_t)B * cs);     // proj_in 出力 [B, tokens, C]

    // group_norm (CHW, N=B) -> per-b で [C,HW] を [tokens,C] へ転置
    launch_group_norm(d_x, w.get(prefix + "norm.weight"), w.get(prefix + "norm.bias"),
                      d_gn, B, C, H, W, groups, gn_eps);
    for (int b = 0; b < B; ++b)
    {
        launch_chw_to_sc(d_gn + (size_t)b * cs, d_sc + (size_t)b * cs, C, S);
    }

    // proj_in (Linear C->C +bias)
    linear(w, prefix + "proj_in.weight", prefix + "proj_in.bias",
           d_sc, d_pin, Ms, C, C, true);

    // transformer_blocks
    for (int l = 0; l < num_layers; ++l)
    {
        const std::string bp = prefix + "transformer_blocks." + std::to_string(l) + ".";
        transformer_block(w, bp, d_pin, B, S, C, d_ctx, ctx_tokens, ctx_dim, sc);
    }

    // proj_out (Linear C->C +bias) -> d_sc
    linear(w, prefix + "proj_out.weight", prefix + "proj_out.bias",
           d_pin, d_sc, Ms, C, C, true);

    // [tokens, C] -> [C, HW] に per-b で戻し residual 加算
    for (int b = 0; b < B; ++b)
    {
        launch_sc_to_chw(d_sc + (size_t)b * cs, d_gn + (size_t)b * cs, C, S);
    }
    launch_add(d_x, d_gn, d_x, (int)((size_t)B * cs));
}

// ----------------------------------------------------------------
// Downsample2D: 3x3 stride2 pad1 conv (Cin->Cin)。H,W -> H/2,W/2
// ----------------------------------------------------------------
static void downsample(DeviceWeights& w, const std::string& prefix,
                       const __half* d_in, int B, int C, int H, int W, __half* d_out)
{
    launch_conv2d(d_in, w.get(prefix + "conv.weight"), w.get(prefix + "conv.bias"),
                  d_out, B, C, H, W, C, 3, 3, 2, 2, 1, 1, 1, 1);
}

// ----------------------------------------------------------------
// Upsample2D: nearest2x + 3x3 pad1 conv (Cin->Cin)。H,W -> 2H,2W
// ----------------------------------------------------------------
static void upsample(DeviceWeights& w, const std::string& prefix,
                     const __half* d_in, int B, int C, int H, int W, __half* d_out, Scratch& sc)
{
    const int H2 = H << 1, W2 = W << 1;
    __half* d_up = sc.alloc((size_t)B * C * H2 * W2);
    launch_upsample_nearest2x(d_in, d_up, B, C, H, W);
    launch_conv2d(d_up, w.get(prefix + "conv.weight"), w.get(prefix + "conv.bias"),
                  d_out, B, C, H2, W2, C, 3, 3, 1, 1, 1, 1, 1, 1);
}

// TU ローカル型 (上の ODR 注意と同じ理由で匿名 namespace に置く)。
namespace
{
// ----------------------------------------------------------------
// skip スタック: down 経路の各 resnet/downsample 出力を順に push し、
// up 経路で逆順に pop する。各エントリは (デバイスバッファ, C, H, W)。
// ----------------------------------------------------------------
struct SkipEntry
{
    __half* ptr;
    int     C;
    int     H;
    int     W;
};
} // namespace (匿名): TU ローカル型 SkipEntry ここまで

// ----------------------------------------------------------------
// UNet 本体 (S1: 重みは DeviceWeights& で受け取り、常駐/都度構築の両方から共有)
// ----------------------------------------------------------------
static void launch_unet_impl(DeviceWeights&     w,
                             int                B,
                             const __half*      d_latent,
                             float              timestep,
                             const __half*      d_encoder_hidden_states,
                             const __half*      d_text_embeds,
                             const __half*      d_time_ids,
                             __half*            d_noise_pred_out,
                             bool               attn_fast,
                             bool               epilogue)
{

    // ============ S0 プロファイル計時 (DOLLAMA_PROFILE 有効時のみ) ============
    // 段グループ別の壁時計 (重み転送込み)。prof_lap で「前のマーク以降」を加算しマーク更新。
    const bool prof = profile_enabled();
    ProfileCounters& pc = profile_counters();
    std::chrono::high_resolution_clock::time_point prof_t_total;
    std::chrono::high_resolution_clock::time_point prof_mark;
    if (prof)
    {
        cudaDeviceSynchronize();
        prof_t_total = std::chrono::high_resolution_clock::now();
        prof_mark    = prof_t_total;
        pc.unet_steps += 1;
    }
    auto prof_lap = [&](double* bucket)
    {
        if (!prof) { return; }
        cudaDeviceSynchronize();
        const auto now = std::chrono::high_resolution_clock::now();
        *bucket += std::chrono::duration<double>(now - prof_mark).count();
        prof_mark = now;
    };


    const int   ctx_tokens = 77;
    const int   ctx_dim    = 2048;
    const int   C0 = 320, C1 = 640, C2 = 1280;

    // ============ time embedding ============
    // sinusoidal(320) -> linear_1(1280,320)+SiLU -> linear_2(1280,1280) = temb[B,1280]
    //   G-2k S2: timestep は cond/uncond 共通スカラ。sinusoidal を [B,320] に複製して
    //   linear を M=B で通す。add embedding は text_embeds が per-b で異なるので per-b に
    //   埋める。B=1 では複製ループが回らず従来と完全同一 (バッファも 320/1280/2816)。
    __half* d_temb = nullptr;
    {
        Scratch sc;
        __half* d_sin = sc.alloc((size_t)B * 320);
        launch_timestep_embedding(d_sin, timestep, 320, 10000.0f);
        for (int b = 1; b < B; ++b)
        {
            CUDA_CHECK(cudaMemcpy(d_sin + (size_t)b * 320, d_sin, 320 * sizeof(__half),
                                  cudaMemcpyDeviceToDevice));
        }
        __half* d_h1 = sc.alloc((size_t)B * 1280);
        linear(w, "time_embedding.linear_1.weight", "time_embedding.linear_1.bias",
               d_sin, d_h1, B, 1280, 320, true);
        launch_silu(d_h1, d_h1, B * 1280);
        CUDA_CHECK(cudaMalloc(&d_temb, (size_t)B * 1280 * sizeof(__half)));
        linear(w, "time_embedding.linear_2.weight", "time_embedding.linear_2.bias",
               d_h1, d_temb, B, 1280, 1280, true);
        dbg_stat("time_embedding_out", d_temb, (size_t)B * 1280);

        // ============ add embedding (SDXL) ============
        // time_ids[6] を各成分 sinusoidal(256) 展開し concat[1536]、text_embeds[b][1280] を
        // 前に置いて [B,2816] -> linear_1+SiLU -> linear_2 -> temb に加算。
        std::vector<__half> h_time_ids(6);
        CUDA_CHECK(cudaMemcpy(h_time_ids.data(), d_time_ids, 6 * sizeof(__half),
                              cudaMemcpyDeviceToHost));
        __half* d_addin = sc.alloc((size_t)B * 2816);
        for (int b = 0; b < B; ++b)
        {
            // 先頭 1280 = text_embeds[b] (cond/uncond で異なる)
            CUDA_CHECK(cudaMemcpy(d_addin + (size_t)b * 2816, d_text_embeds + (size_t)b * 1280,
                                  1280 * sizeof(__half), cudaMemcpyDeviceToDevice));
            // 後続 1536 = 6 x sinusoidal(256) (time_ids は共有)
            for (int i = 0; i < 6; ++i)
            {
                const float tid = __half2float(h_time_ids[i]);
                launch_timestep_embedding(d_addin + (size_t)b * 2816 + 1280 + i * 256,
                                          tid, 256, 10000.0f);
            }
        }
        __half* d_ah = sc.alloc((size_t)B * 1280);
        linear(w, "add_embedding.linear_1.weight", "add_embedding.linear_1.bias",
               d_addin, d_ah, B, 1280, 2816, true);
        launch_silu(d_ah, d_ah, B * 1280);
        __half* d_aout = sc.alloc((size_t)B * 1280);
        linear(w, "add_embedding.linear_2.weight", "add_embedding.linear_2.bias",
               d_ah, d_aout, B, 1280, 1280, true);
        dbg_stat("add_embedding_out", d_aout, (size_t)B * 1280);
        // temb += add_embed
        launch_add(d_temb, d_aout, d_temb, B * 1280);
    }
    dbg_stat("temb_final", d_temb, (size_t)B * 1280);

    // ResnetBlock 内で使う silu(temb) を 1 回計算して共有。
    __half* d_temb_silu = nullptr;
    CUDA_CHECK(cudaMalloc(&d_temb_silu, (size_t)B * 1280 * sizeof(__half)));
    launch_silu(d_temb, d_temb_silu, B * 1280);
    prof_lap(&pc.unet_embed_sec);

    // skip スタック (down 経路 push, up 経路 pop)。
    std::vector<SkipEntry> skips;
    auto push_skip = [&](const __half* d_src, int C, int H, int W)
    {
        // G-2k S2: skip エントリは [B,C,H,W]。B=1 では従来と同一サイズ。
        __half* p = nullptr;
        const size_t sz = (size_t)B * C * H * W;
        CUDA_CHECK(cudaMalloc(&p, sz * sizeof(__half)));
        CUDA_CHECK(cudaMemcpy(p, d_src, sz * sizeof(__half), cudaMemcpyDeviceToDevice));
        skips.push_back({p, C, H, W});
    };

    // ============ conv_in (3x3 pad1, 4->320), 128x128 ============
    __half* d_cur = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cur, (size_t)B * C0 * 128 * 128 * sizeof(__half)));
    launch_conv2d(d_latent, w.get("conv_in.weight"), w.get("conv_in.bias"),
                  d_cur, B, 4, 128, 128, C0, 3, 3, 1, 1, 1, 1, 1, 1);
    dbg_stat("conv_in_out", d_cur, (size_t)B * C0 * 128 * 128);
    push_skip(d_cur, C0, 128, 128);   // res_samples[0] = conv_in 出力

    // ============ down_blocks[0]: DownBlock2D (resnet x2, downsample) 128->64 ============
    {
        const int H = 128, Wd = 128;
        Scratch sc(attn_fast, epilogue);
        __half* d_r0 = sc.alloc((size_t)B * C0 * H * Wd);
        resnet_block(w, "down_blocks.0.resnets.0.", d_cur, B, C0, C0, H, Wd, d_temb_silu, d_r0, sc);
        dbg_stat("down_block_0_resnet0_out", d_r0, (size_t)B * C0 * H * Wd);
        push_skip(d_r0, C0, H, Wd);
        __half* d_r1 = sc.alloc((size_t)B * C0 * H * Wd);
        resnet_block(w, "down_blocks.0.resnets.1.", d_r0, B, C0, C0, H, Wd, d_temb_silu, d_r1, sc);
        push_skip(d_r1, C0, H, Wd);
        // downsample 128->64
        CUDA_CHECK(cudaFree(d_cur));
        CUDA_CHECK(cudaMalloc(&d_cur, (size_t)B * C0 * 64 * 64 * sizeof(__half)));
        downsample(w, "down_blocks.0.downsamplers.0.", d_r1, B, C0, H, Wd, d_cur);
        push_skip(d_cur, C0, 64, 64);
        dbg_stat("down_block_0_out", d_cur, (size_t)B * C0 * 64 * 64);
    }

    // ============ down_blocks[1]: CrossAttnDownBlock (resnet+T2D[L=2] x2, downsample) 64->32 ============
    {
        const int H = 64, Wd = 64, Ln = 2;
        Scratch sc(attn_fast, epilogue);
        // resnet0: 320->640
        __half* d_r0 = sc.alloc((size_t)B * C1 * H * Wd);
        resnet_block(w, "down_blocks.1.resnets.0.", d_cur, B, C0, C1, H, Wd, d_temb_silu, d_r0, sc);
        dbg_stat("down_block_1_resnet0_out", d_r0, (size_t)B * C1 * H * Wd);
        transformer2d(w, "down_blocks.1.attentions.0.", d_r0, B, C1, H, Wd, Ln,
                      d_encoder_hidden_states, ctx_tokens, ctx_dim, sc);
        dbg_stat("down_block_1_attn0_out", d_r0, (size_t)B * C1 * H * Wd);
        push_skip(d_r0, C1, H, Wd);
        // resnet1+T2D: 640->640
        __half* d_r1 = sc.alloc((size_t)B * C1 * H * Wd);
        resnet_block(w, "down_blocks.1.resnets.1.", d_r0, B, C1, C1, H, Wd, d_temb_silu, d_r1, sc);
        transformer2d(w, "down_blocks.1.attentions.1.", d_r1, B, C1, H, Wd, Ln,
                      d_encoder_hidden_states, ctx_tokens, ctx_dim, sc);
        push_skip(d_r1, C1, H, Wd);
        // downsample 64->32
        CUDA_CHECK(cudaFree(d_cur));
        CUDA_CHECK(cudaMalloc(&d_cur, (size_t)B * C1 * 32 * 32 * sizeof(__half)));
        downsample(w, "down_blocks.1.downsamplers.0.", d_r1, B, C1, H, Wd, d_cur);
        push_skip(d_cur, C1, 32, 32);
        dbg_stat("down_block_1_out", d_cur, (size_t)B * C1 * 32 * 32);
    }

    // ============ down_blocks[2]: CrossAttnDownBlock (resnet+T2D[L=10] x2, downsample なし) 32x32 ============
    {
        const int H = 32, Wd = 32, Ln = 10;
        Scratch sc(attn_fast, epilogue);
        // resnet0: 640->1280
        __half* d_r0 = sc.alloc((size_t)B * C2 * H * Wd);
        resnet_block(w, "down_blocks.2.resnets.0.", d_cur, B, C1, C2, H, Wd, d_temb_silu, d_r0, sc);
        dbg_stat("down_block_2_resnet0_out", d_r0, (size_t)B * C2 * H * Wd);
        transformer2d(w, "down_blocks.2.attentions.0.", d_r0, B, C2, H, Wd, Ln,
                      d_encoder_hidden_states, ctx_tokens, ctx_dim, sc);
        dbg_stat("down_block_2_attn0_out", d_r0, (size_t)B * C2 * H * Wd);
        push_skip(d_r0, C2, H, Wd);
        __half* d_r1 = sc.alloc((size_t)B * C2 * H * Wd);
        resnet_block(w, "down_blocks.2.resnets.1.", d_r0, B, C2, C2, H, Wd, d_temb_silu, d_r1, sc);
        transformer2d(w, "down_blocks.2.attentions.1.", d_r1, B, C2, H, Wd, Ln,
                      d_encoder_hidden_states, ctx_tokens, ctx_dim, sc);
        push_skip(d_r1, C2, H, Wd);
        // d_cur <- d_r1 (block 出力, downsample なし)
        CUDA_CHECK(cudaFree(d_cur));
        CUDA_CHECK(cudaMalloc(&d_cur, (size_t)B * C2 * H * Wd * sizeof(__half)));
        CUDA_CHECK(cudaMemcpy(d_cur, d_r1, (size_t)B * C2 * H * Wd * sizeof(__half),
                              cudaMemcpyDeviceToDevice));
        dbg_stat("down_block_2_out", d_cur, (size_t)B * C2 * H * Wd);
    }

    prof_lap(&pc.unet_down_sec);
    // ============ mid_block: Resnet -> T2D[L=10] -> Resnet (1280, 32x32) ============
    {
        const int H = 32, Wd = 32, Ln = 10;
        Scratch sc(attn_fast, epilogue);
        __half* d_r0 = sc.alloc((size_t)B * C2 * H * Wd);
        resnet_block(w, "mid_block.resnets.0.", d_cur, B, C2, C2, H, Wd, d_temb_silu, d_r0, sc);
        dbg_stat("mid_block_resnet0_out", d_r0, (size_t)B * C2 * H * Wd);
        transformer2d(w, "mid_block.attentions.0.", d_r0, B, C2, H, Wd, Ln,
                      d_encoder_hidden_states, ctx_tokens, ctx_dim, sc);
        dbg_stat("mid_block_attn0_out", d_r0, (size_t)B * C2 * H * Wd);
        resnet_block(w, "mid_block.resnets.1.", d_r0, B, C2, C2, H, Wd, d_temb_silu, d_cur, sc);
        dbg_stat("mid_block_out", d_cur, (size_t)B * C2 * H * Wd);
    }

    prof_lap(&pc.unet_mid_sec);
    // pop ヘルパ
    auto pop_skip = [&]() -> SkipEntry
    {
        SkipEntry e = skips.back();
        skips.pop_back();
        return e;
    };

    // ============ up_blocks[0]: CrossAttnUpBlock (resnet+T2D[L=10] x3, upsample) 32->64 ============
    // 各 resnet 入力 = concat(現活性, pop skip)。
    {
        const int H = 32, Wd = 32, Ln = 10, HW = H * Wd;
        for (int r = 0; r < 3; ++r)
        {
            Scratch sc(attn_fast, epilogue);
            SkipEntry sk = pop_skip();  // 1280,1280,640
            __half* d_cat = sc.alloc((size_t)B * (C2 + sk.C) * HW);
            // concat は per-b (レイアウト [B,Ca+Cb,HW])。B=1 は offset 0 の 1 周。
            for (int b = 0; b < B; ++b)
            {
                launch_concat_chw(d_cur + (size_t)b * C2 * HW, sk.ptr + (size_t)b * sk.C * HW,
                                  d_cat + (size_t)b * (C2 + sk.C) * HW, C2, sk.C, HW);
            }
            CUDA_CHECK(cudaFree(sk.ptr));
            __half* d_out = sc.alloc((size_t)B * C2 * HW);
            resnet_block(w, "up_blocks.0.resnets." + std::to_string(r) + ".",
                         d_cat, B, C2 + sk.C, C2, H, Wd, d_temb_silu, d_out, sc);
            if (r == 0) { dbg_stat("up_block_0_resnet0_out", d_out, (size_t)B * C2 * HW); }
            transformer2d(w, "up_blocks.0.attentions." + std::to_string(r) + ".",
                          d_out, B, C2, H, Wd, Ln, d_encoder_hidden_states, ctx_tokens, ctx_dim, sc);
            if (r == 0) { dbg_stat("up_block_0_attn0_out", d_out, (size_t)B * C2 * HW); }
            CUDA_CHECK(cudaFree(d_cur));
            CUDA_CHECK(cudaMalloc(&d_cur, (size_t)B * C2 * HW * sizeof(__half)));
            CUDA_CHECK(cudaMemcpy(d_cur, d_out, (size_t)B * C2 * HW * sizeof(__half),
                                  cudaMemcpyDeviceToDevice));
        }
        // upsample 32->64
        Scratch sc;
        __half* d_up = sc.alloc((size_t)B * C2 * 64 * 64);
        upsample(w, "up_blocks.0.upsamplers.0.", d_cur, B, C2, H, Wd, d_up, sc);
        CUDA_CHECK(cudaFree(d_cur));
        CUDA_CHECK(cudaMalloc(&d_cur, (size_t)B * C2 * 64 * 64 * sizeof(__half)));
        CUDA_CHECK(cudaMemcpy(d_cur, d_up, (size_t)B * C2 * 64 * 64 * sizeof(__half),
                              cudaMemcpyDeviceToDevice));
        dbg_stat("up_block_0_out", d_cur, (size_t)B * C2 * 64 * 64);
    }

    // ============ up_blocks[1]: CrossAttnUpBlock (resnet+T2D[L=2] x3, upsample) 64->128 ============
    {
        const int H = 64, Wd = 64, Ln = 2, HW = H * Wd;
        int curC = C2;
        for (int r = 0; r < 3; ++r)
        {
            Scratch sc(attn_fast, epilogue);
            SkipEntry sk = pop_skip();  // 640,640,320
            __half* d_cat = sc.alloc((size_t)B * (curC + sk.C) * HW);
            for (int b = 0; b < B; ++b)
            {
                launch_concat_chw(d_cur + (size_t)b * curC * HW, sk.ptr + (size_t)b * sk.C * HW,
                                  d_cat + (size_t)b * (curC + sk.C) * HW, curC, sk.C, HW);
            }
            CUDA_CHECK(cudaFree(sk.ptr));
            __half* d_out = sc.alloc((size_t)B * C1 * HW);
            resnet_block(w, "up_blocks.1.resnets." + std::to_string(r) + ".",
                         d_cat, B, curC + sk.C, C1, H, Wd, d_temb_silu, d_out, sc);
            if (r == 0) { dbg_stat("up_block_1_resnet0_out", d_out, (size_t)B * C1 * HW); }
            transformer2d(w, "up_blocks.1.attentions." + std::to_string(r) + ".",
                          d_out, B, C1, H, Wd, Ln, d_encoder_hidden_states, ctx_tokens, ctx_dim, sc);
            if (r == 0) { dbg_stat("up_block_1_attn0_out", d_out, (size_t)B * C1 * HW); }
            CUDA_CHECK(cudaFree(d_cur));
            CUDA_CHECK(cudaMalloc(&d_cur, (size_t)B * C1 * HW * sizeof(__half)));
            CUDA_CHECK(cudaMemcpy(d_cur, d_out, (size_t)B * C1 * HW * sizeof(__half),
                                  cudaMemcpyDeviceToDevice));
            curC = C1;
        }
        // upsample 64->128
        Scratch sc;
        __half* d_up = sc.alloc((size_t)B * C1 * 128 * 128);
        upsample(w, "up_blocks.1.upsamplers.0.", d_cur, B, C1, H, Wd, d_up, sc);
        CUDA_CHECK(cudaFree(d_cur));
        CUDA_CHECK(cudaMalloc(&d_cur, (size_t)B * C1 * 128 * 128 * sizeof(__half)));
        CUDA_CHECK(cudaMemcpy(d_cur, d_up, (size_t)B * C1 * 128 * 128 * sizeof(__half),
                              cudaMemcpyDeviceToDevice));
        dbg_stat("up_block_1_out", d_cur, (size_t)B * C1 * 128 * 128);
    }

    // ============ up_blocks[2]: UpBlock2D (resnet x3, upsample なし) 128x128 ============
    {
        const int H = 128, Wd = 128, HW = H * Wd;
        int curC = C1;
        for (int r = 0; r < 3; ++r)
        {
            Scratch sc(attn_fast, epilogue);
            SkipEntry sk = pop_skip();  // 320,320,320
            __half* d_cat = sc.alloc((size_t)B * (curC + sk.C) * HW);
            for (int b = 0; b < B; ++b)
            {
                launch_concat_chw(d_cur + (size_t)b * curC * HW, sk.ptr + (size_t)b * sk.C * HW,
                                  d_cat + (size_t)b * (curC + sk.C) * HW, curC, sk.C, HW);
            }
            CUDA_CHECK(cudaFree(sk.ptr));
            __half* d_out = sc.alloc((size_t)B * C0 * HW);
            resnet_block(w, "up_blocks.2.resnets." + std::to_string(r) + ".",
                         d_cat, B, curC + sk.C, C0, H, Wd, d_temb_silu, d_out, sc);
            CUDA_CHECK(cudaFree(d_cur));
            CUDA_CHECK(cudaMalloc(&d_cur, (size_t)B * C0 * HW * sizeof(__half)));
            CUDA_CHECK(cudaMemcpy(d_cur, d_out, (size_t)B * C0 * HW * sizeof(__half),
                                  cudaMemcpyDeviceToDevice));
            curC = C0;
            if (r == 0) { dbg_stat("up_block_2_resnet0_out", d_cur, (size_t)B * C0 * HW); }
        }
        dbg_stat("up_block_2_out", d_cur, (size_t)B * C0 * HW);
    }

    prof_lap(&pc.unet_up_sec);
    // ============ conv_norm_out (GN32 eps1e-5) -> SiLU -> conv_out (3x3 pad1, 320->4) ============
    {
        const int H = 128, Wd = 128;
        Scratch sc;
        __half* d_n = sc.alloc((size_t)B * C0 * H * Wd);
        // G-4k S1a: epilogue=on のみ multi-block GN。default は従来経路を無改変で通す。
        if (epilogue)
        {
            launch_group_norm_mb(d_cur, w.get("conv_norm_out.weight"), w.get("conv_norm_out.bias"),
                                 d_n, B, C0, H, Wd, 32, 1e-5f);
        }
        else
        {
            launch_group_norm(d_cur, w.get("conv_norm_out.weight"), w.get("conv_norm_out.bias"),
                              d_n, B, C0, H, Wd, 32, 1e-5f);
        }
        // golden conv_norm_out_out は SiLU 前 (GroupNorm 出力) を捕捉している。
        dbg_stat("conv_norm_out_out", d_n, (size_t)B * C0 * H * Wd);
        launch_silu(d_n, d_n, B * C0 * H * Wd);
        launch_conv2d(d_n, w.get("conv_out.weight"), w.get("conv_out.bias"),
                      d_noise_pred_out, B, C0, H, Wd, 4, 3, 3, 1, 1, 1, 1, 1, 1);
        dbg_stat("noise_pred", d_noise_pred_out, (size_t)B * 4 * H * Wd);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    prof_lap(&pc.unet_convout_sec);
    if (prof)
    {
        const auto prof_end = std::chrono::high_resolution_clock::now();
        pc.unet_total_sec += std::chrono::duration<double>(prof_end - prof_t_total).count();
    }
    // 残った skip があれば解放 (正常時は空)。
    for (SkipEntry& e : skips)
    {
        cudaFree(e.ptr);
    }
    CUDA_CHECK(cudaFree(d_cur));
    CUDA_CHECK(cudaFree(d_temb));
    CUDA_CHECK(cudaFree(d_temb_silu));
}

// ----------------------------------------------------------------
// 後方互換 public API: 呼び出しごとに DeviceWeights を構築 (= 全重み都度転送)。
// test_unet が使用。常駐版が無かった頃と完全同一の挙動。
// ----------------------------------------------------------------
void launch_unet(const SafeTensors& weights,
                 const __half*      d_latent,
                 float              timestep,
                 const __half*      d_encoder_hidden_states,
                 const __half*      d_text_embeds,
                 const __half*      d_time_ids,
                 __half*            d_noise_pred_out,
                 bool               attn_fast,
                 bool               epilogue)
{
    DeviceWeights w(weights);
    launch_unet_impl(w, 1, d_latent, timestep, d_encoder_hidden_states,
                     d_text_embeds, d_time_ids, d_noise_pred_out, attn_fast, epilogue);
}

// ----------------------------------------------------------------
// 常駐重みハンドル API (S1)。
//   DeviceWeights を 1 個ヒープに確保して不透明ポインタとして返す。get() の
//   name->デバイスポインタ キャッシュにより、初回 launch で全重みを upload した後は
//   以降の step で再 malloc / 再 H2D が一切発生しない。
// ----------------------------------------------------------------
UnetWeightsHandle unet_weights_create(const SafeTensors& weights)
{
    DeviceWeights* w = new DeviceWeights(weights);
    // 全重みをデバイスへ常駐させる (一度きり)。以降の get() は即返し。
    w->upload_all();
    return static_cast<UnetWeightsHandle>(w);
}

void unet_weights_destroy(UnetWeightsHandle handle)
{
    if (handle == nullptr)
    {
        return;
    }
    delete static_cast<DeviceWeights*>(handle);
}

// ----------------------------------------------------------------
// ランタイム LoRA (L-2) C-API。DeviceWeights (private) へのハンドル越し操作。
// ----------------------------------------------------------------
void unet_apply_loras(UnetWeightsHandle handle, const LoraModule* mods, int n)
{
    if (handle == nullptr || mods == nullptr || n <= 0)
    {
        return;
    }
    DeviceWeights& w = *static_cast<DeviceWeights*>(handle);
    for (int i = 0; i < n; ++i)
    {
        w.apply_lora(mods[i]);
    }
}

void unet_clear_loras(UnetWeightsHandle handle)
{
    if (handle == nullptr)
    {
        return;
    }
    static_cast<DeviceWeights*>(handle)->revert_all();
}

const __half* unet_weight_device_ptr(UnetWeightsHandle handle, const std::string& name)
{
    if (handle == nullptr)
    {
        return nullptr;
    }
    return static_cast<DeviceWeights*>(handle)->get(name);
}

void launch_unet(UnetWeightsHandle  handle,
                 const __half*      d_latent,
                 float              timestep,
                 const __half*      d_encoder_hidden_states,
                 const __half*      d_text_embeds,
                 const __half*      d_time_ids,
                 __half*            d_noise_pred_out,
                 bool               attn_fast,
                 bool               epilogue)
{
    DeviceWeights& w = *static_cast<DeviceWeights*>(handle);
    launch_unet_impl(w, 1, d_latent, timestep, d_encoder_hidden_states,
                     d_text_embeds, d_time_ids, d_noise_pred_out, attn_fast, epilogue);
}

// ----------------------------------------------------------------
// バッチ (B>1) 版 public API (G-2k S2)。CFG cond/uncond の B=2 束ねを 1 forward で回す。
//   d_latent               : [B,4,128,128] (b ごとに cond/uncond)
//   d_encoder_hidden_states: [B,77,2048]
//   d_text_embeds          : [B,1280]
//   d_time_ids             : [6] (全 b 共有)
//   d_noise_pred_out       : [B,4,128,128] (事前確保)
//   B=1 で呼べば launch_unet(handle,...) と各カーネル起動列・数値が完全同一。
// ----------------------------------------------------------------
void launch_unet_batched(UnetWeightsHandle  handle,
                         int                B,
                         const __half*      d_latent,
                         float              timestep,
                         const __half*      d_encoder_hidden_states,
                         const __half*      d_text_embeds,
                         const __half*      d_time_ids,
                         __half*            d_noise_pred_out,
                         bool               attn_fast,
                         bool               epilogue)
{
    DeviceWeights& w = *static_cast<DeviceWeights*>(handle);
    launch_unet_impl(w, B, d_latent, timestep, d_encoder_hidden_states,
                     d_text_embeds, d_time_ids, d_noise_pred_out, attn_fast, epilogue);
}

} // namespace dollama

// device_arena — チャンク連結型 bump アリーナ (G-8k S1a 基盤)
// 対象: RTX5080 (Blackwell / sm_120) / CUDA Runtime API のみ
//
// ----------------------------------------------------------------
// 目的:
//   step ループ内の cudaMalloc / cudaFree を撲滅する。cudaFree は暗黙のデバイス
//   同期点であり、occupancy/latency 律速の阻害要因。さらに CUDA Graphs (G-1k) は
//   capture 中の malloc/free を許さないため、その前提条件でもある。
//
// 設計の急所 (これを外すと壊れる):
//   本アリーナの利用者は「同時生存が複数本」ある (unet.cu の Scratch は
//   transformer_block 1 回で 9-13 本が同時に生きている)。したがって
//   groupnorm.cu の g_mb_buf のような「grow したら古い領域を cudaFree して
//   再確保する」grow-only 単一バッファ方式は **絶対に一般化してはならない**
//   (生存中のポインタが dangling になり、memcmp が通ったり通らなかったりする)。
//   本実装は「現チャンクが足りなければ新チャンクを追加確保し、既存チャンクは
//   解放しない」チャンク連結型にすることで、**既に配ったポインタが無効化される
//   ことが原理的に無い** 構造になっている。
//
// 使い方 (S2 での Scratch 置換を想定):
//   {
//       DeviceArenaScope sc(DeviceArenaId::UNet);   // = mark()
//       __half* p = sc.alloc<__half>(n);            // 呼び出し側は従来と同形
//       ...
//   }                                              // デストラクタで rewind()
//
// キルスイッチ:
//   環境変数 DOLLAMA_POOL=0 (存在かつ "0") のとき、alloc は素の cudaMalloc、
//   rewind は対応する cudaFree にフォールバックし、旧経路が完全に復元される。
//
// スレッド契約:
//   単一スレッド前提。初回使用時に所有スレッド id を記録し、以後不一致なら
//   throw する (暗黙の仮定ではなく「落ちる契約」にする)。
//
// ODR 注意:
//   本ヘッダは TU 横断の外部リンケージを意図的に持つ (C0 の DeviceWeights /
//   Scratch 二重定義の再演を避けるため、シンボルは dollama::device_arena_* に
//   統一し、実体は device_arena.cu 1 本だけに置く)。ヘッダ内に __global__ は
//   置かない (置く場合は static 必須)。
// ----------------------------------------------------------------
#pragma once

#include <cstddef>
#include <cstdint>

namespace dollama
{

// ----------------------------------------------------------------
// アリーナ識別子。VAE の GB 級チャンクを UNet と混ぜないため 2 本に分ける
// (S3 だけ独立に revert 可能・trim 粒度も分離できる)。
// ----------------------------------------------------------------
enum class DeviceArenaId : int
{
    UNet = 0,
    VAE  = 1,
};

// 本数 (内部テーブルの寸法)。
// (C++14 前提のため inline 変数は使わない。TU ローカル定数で足りる)
static constexpr int kDeviceArenaCount = 2;

// 表示名 ("unet" / "vae")。プロファイル出力・エラーメッセージ用。
const char* device_arena_name(DeviceArenaId id);

// 全アロケーションのアラインメント (バイト)。最低 256B (要件)。
static constexpr size_t kDeviceArenaAlign = 256;

// ----------------------------------------------------------------
// mark / rewind のトークン。LIFO でのみ使う (取得順の逆順に rewind する)。
//   chunk / offset : プール有効時のカーソル位置。
//   fallback_count : DOLLAMA_POOL=0 時の「その時点までに配ったポインタ本数」。
// ----------------------------------------------------------------
struct DeviceArenaMark
{
    DeviceArenaId id             = DeviceArenaId::UNet;
    size_t        chunk          = 0;
    size_t        offset         = 0;
    size_t        fallback_count = 0;
};

// ----------------------------------------------------------------
// 統計カウンタ (S4 で DOLLAMA_PROFILE=1 の [ALLOC] 行として出す)。
// ----------------------------------------------------------------
struct DeviceArenaStats
{
    uint64_t cuda_malloc_calls  = 0;  // 実 cudaMalloc 回数 (累積)
    uint64_t cuda_free_calls    = 0;  // 実 cudaFree 回数 (累積)
    uint64_t chunk_alloc_calls  = 0;  // 新規チャンク確保回数 (累積・プール経路のみ)
    uint64_t alloc_calls        = 0;  // alloc() 呼び出し回数 (累積)
    size_t   total_capacity     = 0;  // 現在の総容量 (バイト・全チャンク合計)
    size_t   live_chunks        = 0;  // 現在のチャンク本数
    size_t   bytes_in_use       = 0;  // 現在の使用中バイト (bump カーソル基準)
    size_t   peak_bytes_in_use  = 0;  // 使用中バイトのピーク
};

// ----------------------------------------------------------------
// 低レベル API
// ----------------------------------------------------------------

// プールが有効か (DOLLAMA_POOL=0 なら false = 素の cudaMalloc/cudaFree)。
// getenv は初回のみ・以後キャッシュ。
bool device_arena_pool_enabled();

// 現在のカーソル位置を記録する。
DeviceArenaMark device_arena_mark(DeviceArenaId id);

// bytes 分をアリーナから切り出す (256B アライン)。bytes==0 は nullptr。
void* device_arena_alloc(DeviceArenaId id, size_t bytes);

// mark 時点までカーソルを巻き戻す。**チャンクは解放しない** (grow は単調)。
// DOLLAMA_POOL=0 のときのみ、対応する cudaFree を行う。
void device_arena_rewind(const DeviceArenaMark& mark);

// 指定アリーナの全チャンクを解放し、カーソルを初期化する。
// 定期 trim / step 間 trim は **実装しない** (目的を殺すため)。明示呼び出し専用。
void device_arena_release(DeviceArenaId id);

// 統計の読み取り。
DeviceArenaStats device_arena_stats(DeviceArenaId id);

// 累積カウンタのみゼロクリア (容量・チャンクは保持)。ベンチ区間の切り出し用。
void device_arena_reset_counters(DeviceArenaId id);

// 最初のチャンクの既定サイズ (バイト)。成長則の起点。テストが
// 「チャンク跨ぎ」を形状非依存に強制するために公開する。
size_t device_arena_first_chunk_bytes(DeviceArenaId id);

// ----------------------------------------------------------------
// RAII スコープ: 構築で mark()、破棄で rewind()。
//   S2 では unet.cu の Scratch をこれに置き換えるだけで、alloc の呼び出し側は
//   1 行も変わらない (要素数 n を受けて T* を返す形を維持)。
// ----------------------------------------------------------------
class DeviceArenaScope
{
public:
    explicit DeviceArenaScope(DeviceArenaId id)
        : mark_(device_arena_mark(id))
    {
    }

    ~DeviceArenaScope()
    {
        // デストラクタからは投げない (rewind は throw しうるため握り潰す)。
        // 所有スレッド違反はこの時点で既に alloc 側が検出している。
        try
        {
            device_arena_rewind(mark_);
        }
        catch (...)
        {
        }
    }

    DeviceArenaScope(const DeviceArenaScope&)            = delete;
    DeviceArenaScope& operator=(const DeviceArenaScope&) = delete;

    // 要素数 count 個ぶんの T 配列を切り出す (従来の Scratch::alloc と同形)。
    template <typename T>
    T* alloc(size_t count)
    {
        return static_cast<T*>(device_arena_alloc(mark_.id, count * sizeof(T)));
    }

    // バイト数指定版。
    void* alloc_bytes(size_t bytes)
    {
        return device_arena_alloc(mark_.id, bytes);
    }

    DeviceArenaId id() const { return mark_.id; }

private:
    DeviceArenaMark mark_;
};

} // namespace dollama

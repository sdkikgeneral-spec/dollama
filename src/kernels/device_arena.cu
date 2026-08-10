// device_arena — チャンク連結型 bump アリーナ 実体 (G-8k S1a)
// 対象: RTX5080 (Blackwell / sm_120) / CUDA Runtime API のみ
//
// 設計は device_arena.cuh のヘッダコメント参照。要点だけ再掲:
//   - 現チャンクが足りなければ「新チャンクを追加確保」し、既存チャンクは解放しない。
//     → 既に配ったポインタが無効化されることが原理的に無い (同時生存 9-13 本に必須)。
//   - rewind はカーソルを戻すだけ。チャンクの解放は device_arena_release() だけが行う
//     (定期 trim / step 間 trim は目的を殺すので実装しない)。
//   - DOLLAMA_POOL=0 で素の cudaMalloc / cudaFree に完全フォールバックする。

#include "kernels/device_arena.cuh"
#include "kernels/utils.cuh"

#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace dollama
{

namespace
{

// ----------------------------------------------------------------
// チャンク 1 本。ptr は確保後 release まで不変 (この不変条件が本アリーナの芯)。
// ----------------------------------------------------------------
struct ArenaChunk
{
    char*  ptr   = nullptr;
    size_t bytes = 0;
};

// DOLLAMA_POOL=0 (キルスイッチ) 経路で配った素の cudaMalloc ブロック。
struct FallbackBlock
{
    void*  ptr   = nullptr;
    size_t bytes = 0;
};

// ----------------------------------------------------------------
// アリーナ本体 (プロセス常駐・アリーナ id ごとに 1 つ)。
// ----------------------------------------------------------------
struct Arena
{
    std::vector<ArenaChunk> chunks;      // 追加のみ / 途中挿入あり (要素は移動しても ptr は不変)
    size_t                  cur    = 0;  // 現チャンク index
    size_t                  offset = 0;  // 現チャンク内 bump オフセット

    // DOLLAMA_POOL=0 用: 素の cudaMalloc で配ったポインタ (rewind で末尾から free)。
    std::vector<FallbackBlock> fallback_ptrs;

    // 所有スレッド (初回使用時に確定。以後不一致なら throw)。
    std::thread::id owner;
    bool            owner_set = false;

    DeviceArenaStats stats;
};

Arena& arena_of(DeviceArenaId id)
{
    // 関数内 static: 初期化順序問題を避ける (TU 横断で参照されるため)。
    static Arena g_arenas[kDeviceArenaCount];
    const int    i = static_cast<int>(id);
    if (i < 0 || i >= kDeviceArenaCount)
    {
        throw std::runtime_error("device_arena: 不正な DeviceArenaId");
    }
    return g_arenas[i];
}

// 256B 切り上げ。
inline size_t align_up(size_t bytes)
{
    return (bytes + kDeviceArenaAlign - 1) & ~(kDeviceArenaAlign - 1);
}

// ----------------------------------------------------------------
// チャンク成長ポリシー
//   - 初期チャンク: UNet 64MiB / VAE 256MiB (VAE は 512MB 級のキャリーがあるため大きめ)。
//   - 2 本目以降: 「直前までのチャンク本数」で倍々に伸ばし、上限でクランプする
//     (UNet 512MiB / VAE 1GiB)。要求が上限より大きければ要求サイズそのもの。
//   - 断片化より「実 cudaMalloc 回数 0 への収束」を優先する設計。
// ----------------------------------------------------------------
size_t first_chunk_bytes(DeviceArenaId id)
{
    switch (id)
    {
    case DeviceArenaId::UNet:
        return (size_t)64 << 20;   // 64MiB
    case DeviceArenaId::VAE:
        return (size_t)256 << 20;  // 256MiB
    case DeviceArenaId::UNetPersist:
        // G-8k S2: skip 9 本 (B=2 で ~102MiB) + d_cur 固定バッファ (B=2 で 42MiB) +
        // temb。B=2 まで 1 チャンクで収まる 256MiB を初期値にして実 cudaMalloc を 1 回に。
        return (size_t)256 << 20;  // 256MiB
    }
    return (size_t)64 << 20;
}

size_t max_chunk_bytes(DeviceArenaId id)
{
    switch (id)
    {
    case DeviceArenaId::UNet:
        return (size_t)512 << 20;         // 512MiB
    case DeviceArenaId::VAE:
        return (size_t)1024 << 20;        // 1GiB
    case DeviceArenaId::UNetPersist:
        return (size_t)512 << 20;         // 512MiB
    }
    return (size_t)512 << 20;
}

// 新チャンクのサイズ決定 (need を必ず満たす)。
size_t next_chunk_bytes(DeviceArenaId id, const Arena& a, size_t need)
{
    size_t       want = first_chunk_bytes(id);
    const size_t cap  = max_chunk_bytes(id);
    for (size_t k = 0; k < a.chunks.size(); ++k)
    {
        if (want >= cap)
        {
            want = cap;
            break;
        }
        want <<= 1;  // 2 のべき乗倍は左シフト (プロジェクト規約)
    }
    if (want > cap)
    {
        want = cap;
    }
    if (want < need)
    {
        want = need;
    }
    return align_up(want);
}

// 現在の使用中バイト (カーソル基準・チャンク跨ぎの捨て分も含む実効フットプリント)。
size_t in_use_bytes(const Arena& a)
{
    size_t sum = 0;
    for (size_t i = 0; i < a.cur && i < a.chunks.size(); ++i)
    {
        sum += a.chunks[i].bytes;
    }
    return sum + a.offset;
}

// 静止状態か (生存中の確保がゼロ = カーソルが基点かつ fallback 空)。
// 静止状態でだけ所有権の移譲を許す (G-8k S2)。
bool is_quiescent(const Arena& a)
{
    return a.cur == 0 && a.offset == 0 && a.fallback_ptrs.empty();
}

// 所有スレッド検査 (G-8k S2 で「所有権移譲つき」に格上げ)。
//   - 生存中の確保があるのに別スレッドが触った = 同時進入 → throw (落ちる契約は維持)。
//   - 静止状態なら所有権を現スレッドへ移譲する。HTTP サーバー (cpp-httplib) は
//     リクエストごとにプールの別ワーカーで生成を回すため、リクエスト間でスレッドが
//     変わる。生成自体は逐次なので、この移譲が無いと 2 リクエスト目で必ず落ちる。
void check_thread(DeviceArenaId id, Arena& a)
{
    const std::thread::id me = std::this_thread::get_id();
    if (!a.owner_set)
    {
        a.owner     = me;
        a.owner_set = true;
        return;
    }
    if (a.owner == me)
    {
        return;
    }
    if (is_quiescent(a))
    {
        // 引き渡し (前の所有スレッドの作業は既に全て rewind 済み)。
        a.owner = me;
        return;
    }
    throw std::runtime_error(std::string("device_arena: アリーナ '")
                             + device_arena_name(id)
                             + "' へ生存確保のある状態で別スレッドが進入した"
                               " (同時使用は単一スレッドのみ)");
}

} // namespace (匿名)

// ----------------------------------------------------------------
// 表示名
// ----------------------------------------------------------------
const char* device_arena_name(DeviceArenaId id)
{
    switch (id)
    {
    case DeviceArenaId::UNet:
        return "unet";
    case DeviceArenaId::VAE:
        return "vae";
    case DeviceArenaId::UNetPersist:
        return "unet_persist";
    }
    return "?";
}

size_t device_arena_first_chunk_bytes(DeviceArenaId id)
{
    return first_chunk_bytes(id);
}

// ----------------------------------------------------------------
// キルスイッチ: DOLLAMA_POOL=0 でプール無効 (素の cudaMalloc/cudaFree)。
//   作法は fast_config.hpp の fast_env_true に倣うが、kernels 層を独立させるため
//   依存は持たせない (getenv は初回のみ・以後キャッシュ)。
// ----------------------------------------------------------------
bool device_arena_pool_enabled()
{
    static int e = -1;
    if (e < 0)
    {
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
        const char* v = std::getenv("DOLLAMA_POOL");
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
        // 未設定 / 空 は既定 (プール有効)。"0" のときだけ無効化する。
        e = (v != nullptr && std::strcmp(v, "0") == 0) ? 0 : 1;
    }
    return e == 1;
}

// ----------------------------------------------------------------
// mark
// ----------------------------------------------------------------
DeviceArenaMark device_arena_mark(DeviceArenaId id)
{
    Arena& a = arena_of(id);
    check_thread(id, a);

    DeviceArenaMark m;
    m.id             = id;
    m.chunk          = a.cur;
    m.offset         = a.offset;
    m.fallback_count = a.fallback_ptrs.size();
    return m;
}

// ----------------------------------------------------------------
// alloc
// ----------------------------------------------------------------
void* device_arena_alloc(DeviceArenaId id, size_t bytes)
{
    Arena& a = arena_of(id);
    check_thread(id, a);

    if (bytes == 0)
    {
        return nullptr;
    }
    a.stats.alloc_calls++;

    const size_t need = align_up(bytes);

    // --- キルスイッチ経路: 素の cudaMalloc (旧経路の完全復元) ---
    if (!device_arena_pool_enabled())
    {
        void* p = nullptr;
        CUDA_CHECK(cudaMalloc(&p, need));
        a.stats.cuda_malloc_calls++;
        FallbackBlock fb;
        fb.ptr   = p;
        fb.bytes = need;
        a.fallback_ptrs.push_back(fb);
        a.stats.bytes_in_use += need;
        if (a.stats.bytes_in_use > a.stats.peak_bytes_in_use)
        {
            a.stats.peak_bytes_in_use = a.stats.bytes_in_use;
        }
        a.stats.total_capacity += need;
        return p;
    }

    // --- プール経路: 現チャンク → 後続チャンク → 新チャンク の順で探す ---
    while (a.cur < a.chunks.size())
    {
        const ArenaChunk& c = a.chunks[a.cur];
        if (a.offset + need <= c.bytes)
        {
            char* p = c.ptr + a.offset;
            a.offset += need;
            a.stats.bytes_in_use = in_use_bytes(a);
            if (a.stats.bytes_in_use > a.stats.peak_bytes_in_use)
            {
                a.stats.peak_bytes_in_use = a.stats.bytes_in_use;
            }
            return p;
        }
        // 現チャンクの残りでは足りない。
        if (a.offset == 0)
        {
            // 先頭から使ってなお入らない = このチャンクは容量不足。
            // ここに新チャンクを「挿入」する (既存チャンクは解放しない)。
            break;
        }
        // 残りを捨てて次チャンクへ (bump アリーナの通常動作)。
        a.cur++;
        a.offset = 0;
    }

    // 新チャンクを a.cur の位置に挿入する。
    //   - 末尾なら push_back と同義。
    //   - 途中挿入でも、既に発行済みのポインタは各チャンクの ptr に紐づくため無効化されない
    //     (vector が持つのはチャンク記述子であり、デバイスメモリ本体は動かない)。
    //   - LIFO 前提より、生存中の mark は chunk <= a.cur しか無く、挿入で index が
    //     ずれるのは a.cur 以降だけ → 巻き戻し先は依然として「その時点以降の全確保」を覆う。
    ArenaChunk nc;
    nc.bytes = next_chunk_bytes(id, a, need);
    void* raw = nullptr;
    CUDA_CHECK(cudaMalloc(&raw, nc.bytes));
    nc.ptr = static_cast<char*>(raw);

    a.stats.cuda_malloc_calls++;
    a.stats.chunk_alloc_calls++;
    a.stats.total_capacity += nc.bytes;

    a.chunks.insert(a.chunks.begin() + (ptrdiff_t)a.cur, nc);
    a.offset = need;

    a.stats.live_chunks  = a.chunks.size();
    a.stats.bytes_in_use = in_use_bytes(a);
    if (a.stats.bytes_in_use > a.stats.peak_bytes_in_use)
    {
        a.stats.peak_bytes_in_use = a.stats.bytes_in_use;
    }
    return nc.ptr;
}

// ----------------------------------------------------------------
// rewind (チャンクは解放しない = grow は単調)
// ----------------------------------------------------------------
void device_arena_rewind(const DeviceArenaMark& mark)
{
    Arena& a = arena_of(mark.id);
    check_thread(mark.id, a);

    if (!device_arena_pool_enabled())
    {
        // キルスイッチ経路: mark 以降に配ったポインタを LIFO で cudaFree する。
        while (a.fallback_ptrs.size() > mark.fallback_count)
        {
            FallbackBlock fb = a.fallback_ptrs.back();
            a.fallback_ptrs.pop_back();
            if (fb.ptr != nullptr)
            {
                cudaFree(fb.ptr);
                a.stats.cuda_free_calls++;
            }
            a.stats.bytes_in_use   -= fb.bytes;
            a.stats.total_capacity -= fb.bytes;
        }
        return;
    }

    if (mark.chunk > a.chunks.size())
    {
        throw std::runtime_error("device_arena: rewind 先が不正 (LIFO 違反の疑い)");
    }
    a.cur                = mark.chunk;
    a.offset             = mark.offset;
    a.stats.bytes_in_use = in_use_bytes(a);
}

// ----------------------------------------------------------------
// release (全チャンク解放)。定期 trim は実装しない。明示呼び出し専用。
// ----------------------------------------------------------------
void device_arena_release(DeviceArenaId id)
{
    Arena& a = arena_of(id);
    check_thread(id, a);

    for (size_t i = 0; i < a.chunks.size(); ++i)
    {
        if (a.chunks[i].ptr != nullptr)
        {
            cudaFree(a.chunks[i].ptr);
            a.stats.cuda_free_calls++;
        }
    }
    a.chunks.clear();

    for (size_t i = 0; i < a.fallback_ptrs.size(); ++i)
    {
        if (a.fallback_ptrs[i].ptr != nullptr)
        {
            cudaFree(a.fallback_ptrs[i].ptr);
            a.stats.cuda_free_calls++;
        }
    }
    a.fallback_ptrs.clear();

    a.cur                 = 0;
    a.offset              = 0;
    a.stats.total_capacity = 0;
    a.stats.live_chunks    = 0;
    a.stats.bytes_in_use   = 0;
    // 所有スレッドの記録自体は維持する (release 後は静止状態なので、次に別スレッドが
    // 触れば check_thread が正規の移譲を行う)。
}

// ----------------------------------------------------------------
// 統計
// ----------------------------------------------------------------
DeviceArenaStats device_arena_stats(DeviceArenaId id)
{
    Arena& a = arena_of(id);
    // 読み取りは所有スレッド検査を行わない (プロファイル出力から呼ぶため)。
    a.stats.live_chunks = a.chunks.size();
    return a.stats;
}

void device_arena_reset_counters(DeviceArenaId id)
{
    Arena& a = arena_of(id);
    a.stats.cuda_malloc_calls = 0;
    a.stats.cuda_free_calls   = 0;
    a.stats.chunk_alloc_calls = 0;
    a.stats.alloc_calls       = 0;
    a.stats.peak_bytes_in_use = a.stats.bytes_in_use;
}

} // namespace dollama

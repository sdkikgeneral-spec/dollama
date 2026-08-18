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

#include <cstdio>
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

    // 同時生存要求バイト (アライン後の合計)。捨て分を含まない真の live 量 (S2b)。
    size_t req_live = 0;

    // 所有スレッド (初回使用時に確定。以後不一致なら throw)。
    std::thread::id owner;
    bool            owner_set = false;

    // G-8k S3b: reserve 済みなのに跨ぎが起きた (= reserve 不足) ことを
    // DOLLAMA_PROFILE=1 のとき 1 回だけ知らせるためのフラグ。黙って太らせない。
    bool            reserve_warned = false;

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
// チャンク成長ポリシー (G-8k S2b で倍々成長を撤去)
//   - チャンクサイズ = max(刻み幅, 要求サイズ)。**倍々に伸ばさない**。
//   - 刻み幅: UNet 256MiB / UNetPersist 64MiB / VAE 256MiB。
//   - 理由 (S2b): 倍々成長は「あと少し足りない」ために 512MiB を丸ごと生やすため、
//     UNet で capacity 1984MiB に対し実需 (POOL=0 の live peak) 1095MiB = 捨て分
//     872MiB という収支になっていた。VAE (S3) の GB 級を足す前に VRAM ゲートを
//     割るため、要求に沿って刻む方式へ変更。
//   - **刻み幅は「単発の最大要求」以上でなければならない** (実測で確定):
//     conv2d の im2col バッファは IM2COL_TILE_BYTES = 256MiB まで来る。刻みを
//     64MiB / 128MiB にすると、この 1 本が入らないチャンクを次々に読み飛ばして
//     捨て分が激増する (実測 capacity: 刻み 64MiB=1876MiB / 128MiB=1722MiB /
//     **256MiB=1280MiB**)。よって UNet の刻みは im2col タイル上限と同値に揃える。
//   - 代償はチャンク本数の増加 (実 cudaMalloc は初回 forward に集中) だけで、
//     **既存チャンクを解放しない = 生存中ポインタを無効化しない**という設計の急所は
//     一切変えていない。定常状態の実 cudaMalloc 0 も変わらない。
// ----------------------------------------------------------------
size_t first_chunk_bytes(DeviceArenaId id)
{
    switch (id)
    {
    case DeviceArenaId::UNet:
        return (size_t)256 << 20;  // 256MiB (S2b 実測合わせ = im2col タイル上限と同値)
    case DeviceArenaId::VAE:
        return (size_t)256 << 20;  // 256MiB
    case DeviceArenaId::UNetPersist:
        // skip 9 本 + d_cur 固定バッファ + temb で B=1 ~68MiB / B=2 ~136MiB。
        // 刻み 64MiB で 2-3 本に収める (S2 の 256MiB 一括は捨て分が大きすぎた)。
        return (size_t)64 << 20;   // 64MiB
    }
    return (size_t)64 << 20;
}

// 新チャンクのサイズ決定 (need を必ず満たす)。
size_t next_chunk_bytes(DeviceArenaId id, const Arena& a, size_t need)
{
    (void)a;  // 本数依存の成長 (倍々) は S2b で廃止した。
    size_t want = first_chunk_bytes(id);
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
// DOLLAMA_PROFILE の有無 (infer/profile.cuh と同じ判定を kernels 層で独立に持つ。
// 依存を張らないのは device_arena が最下層だから)。getenv は初回のみ・以後キャッシュ。
// ----------------------------------------------------------------
static bool arena_profile_enabled()
{
    static int e = -1;
    if (e < 0)
    {
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
        e = std::getenv("DOLLAMA_PROFILE") != nullptr ? 1 : 0;
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
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
    m.req_bytes      = a.req_live;
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

    // 真の同時生存量 (捨て分を含まない)。プール経路とキルスイッチ経路で共通。
    a.req_live += need;
    a.stats.live_request_bytes = a.req_live;
    if (a.req_live > a.stats.peak_request_bytes)
    {
        a.stats.peak_request_bytes = a.req_live;
    }

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
    // G-8k S3b: reserve 済みなのに新チャンクが要る = reserve 不足。
    // 黙って太らせず、DOLLAMA_PROFILE=1 のとき 1 回だけ報告する
    // (正しさは不変。ここから先は従来どおりチャンクを足して続行する)。
    if (a.stats.reserved_bytes != 0 && !a.reserve_warned)
    {
        a.reserve_warned = true;
        if (arena_profile_enabled())
        {
            std::printf("[ALLOC] reserve shortage: arena=%s peak_request %zu MiB"
                        " > reserve %zu MiB (falling back to chunk growth)\n",
                        device_arena_name(id),
                        (a.req_live > a.stats.peak_request_bytes
                             ? a.req_live : a.stats.peak_request_bytes) >> 20,
                        a.stats.reserved_bytes >> 20);
            std::fflush(stdout);
        }
    }

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

    a.req_live                 = mark.req_bytes;
    a.stats.live_request_bytes = a.req_live;

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
// reserve (G-8k S3b): 静止状態で「live peak を包む 1 本」を先に確保する。
//   チャンク跨ぎ = 捨て分の唯一の発生源を、原理的に消すのが目的。
//   - 非静止状態 (生存確保あり) で呼ばれたら throw する (落ちる契約は維持)。
//     既存チャンクを解放するため、生存ポインタがあると dangling になるから。
//   - DOLLAMA_POOL=0 / bytes==0 は no-op。
// ----------------------------------------------------------------
void device_arena_reserve(DeviceArenaId id, size_t bytes)
{
    Arena& a = arena_of(id);
    check_thread(id, a);

    if (bytes == 0)
    {
        return;
    }
    if (!device_arena_pool_enabled())
    {
        // キルスイッチ経路は素の cudaMalloc/cudaFree。旧経路を汚さない。
        return;
    }
    if (!is_quiescent(a))
    {
        throw std::runtime_error(std::string("device_arena: アリーナ '")
                                 + device_arena_name(id)
                                 + "' の reserve は静止状態でのみ可能"
                                   " (生存中の確保があるとポインタが dangling になる)");
    }

    // 既存チャンクを畳んでから 1 本にまとめ直す。
    for (size_t i = 0; i < a.chunks.size(); ++i)
    {
        if (a.chunks[i].ptr != nullptr)
        {
            cudaFree(a.chunks[i].ptr);
            a.stats.cuda_free_calls++;
        }
    }
    a.chunks.clear();
    a.stats.total_capacity = 0;

    ArenaChunk nc;
    nc.bytes  = align_up(bytes);
    void* raw = nullptr;
    CUDA_CHECK(cudaMalloc(&raw, nc.bytes));
    nc.ptr = static_cast<char*>(raw);
    a.chunks.push_back(nc);

    a.stats.cuda_malloc_calls++;
    a.stats.chunk_alloc_calls++;
    a.stats.total_capacity  = nc.bytes;
    a.stats.live_chunks     = a.chunks.size();
    a.stats.reserved_bytes  = nc.bytes;

    a.cur                = 0;
    a.offset             = 0;
    a.stats.bytes_in_use = 0;
    a.reserve_warned     = false;
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
    a.req_live            = 0;
    a.stats.live_request_bytes = 0;
    a.stats.total_capacity = 0;
    // reserve 済みの 1 本もここで手放したので、reserve 状態は解除する
    // (再 reserve するかは呼び出し側の判断)。
    a.stats.reserved_bytes = 0;
    a.reserve_warned       = false;
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
    a.stats.peak_request_bytes = a.req_live;
}

} // namespace dollama

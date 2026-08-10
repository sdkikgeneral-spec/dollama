// device_arena 単体テスト (G-8k S1a)
// HAVE_CUDA 未定義時は [SKIP] で return 0。
//
// 最重要ケース = 「同時生存の安全性」。13 本を連続 alloc してチャンク跨ぎを強制し、
// 全ポインタが最後まで有効であること (= 既に配った領域が新チャンク確保で
// 壊れない / 重ならない) を、既知パターンの書き込み → 全 alloc 後の読み返しで検査する。
// これが groupnorm.cu の grow-only 単一バッファ方式を一般化してはならない、という
// 設計の急所そのものの回帰テスト。
//
// DOLLAMA_POOL=0 (キルスイッチ) 経路は、同一 exe を env 付きで別プロセス実行して検査する
// (device_arena_pool_enabled() は getenv を初回キャッシュするため、1 プロセス内で
//  両経路を混ぜられない)。meson 側で 'device_arena' と 'device_arena_pool_off' の
// 2 test として登録する。
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#ifdef HAVE_CUDA
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "kernels/device_arena.cuh"
#include "kernels/utils.cuh"
#endif

namespace dollama
{

#ifdef HAVE_CUDA

// 同時生存させる本数 (unet.cu の transformer_block が 9-13 本なのでその上限)。
static constexpr int LIVE_N = 13;

static bool g_ok = true;

static void check(bool cond, const char* what)
{
    if (!cond)
    {
        std::cerr << "[test_device_arena] FAIL: " << what << "\n";
        g_ok = false;
    }
}

// 領域の先頭 / 中央 / 末尾 256B を読み返し、全て val で埋まっているか検査する。
static bool verify_pattern(void* p, size_t bytes, unsigned char val)
{
    const size_t probe = 256;
    size_t offs[3];
    offs[0] = 0;
    offs[1] = (bytes >> 1) & ~(size_t)0xFF;
    offs[2] = bytes - probe;

    std::vector<unsigned char> host(probe);
    for (int i = 0; i < 3; ++i)
    {
        CUDA_CHECK(cudaMemcpy(host.data(), (char*)p + offs[i], probe, cudaMemcpyDeviceToHost));
        for (size_t k = 0; k < probe; ++k)
        {
            if (host[k] != val)
            {
                std::cerr << "  pattern mismatch: expect " << (int)val
                          << " got " << (int)host[k]
                          << " at offset " << (offs[i] + k) << "\n";
                return false;
            }
        }
    }
    return true;
}

// ----------------------------------------------------------------
// 1) 同時生存の安全性 (最重要)。チャンク跨ぎを強制して 13 本すべてが生き残るか。
//    戻り値: 配ったポインタ列 (後続テストで再利用する)。
// ----------------------------------------------------------------
static void test_concurrent_liveness()
{
    const DeviceArenaId id = DeviceArenaId::UNet;

    // 1 本あたり「初期チャンク / 8 + 1KB」→ 8 本で 1 チャンクを必ず溢れさせる。
    const size_t first = device_arena_first_chunk_bytes(id);
    const size_t each  = (first >> 3) + 1024;

    const DeviceArenaStats before = device_arena_stats(id);

    DeviceArenaMark m = device_arena_mark(id);
    void*           ptrs[LIVE_N];
    for (int i = 0; i < LIVE_N; ++i)
    {
        ptrs[i] = device_arena_alloc(id, each);
        check(ptrs[i] != nullptr, "alloc returned nullptr");
        // 既知パターン: i+1 で全域を埋める。
        CUDA_CHECK(cudaMemset(ptrs[i], (unsigned char)(i + 1), each));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // 全 alloc が終わった「後で」全領域を読み返す。ここが急所の回帰点。
    bool all = true;
    for (int i = 0; i < LIVE_N; ++i)
    {
        all = verify_pattern(ptrs[i], each, (unsigned char)(i + 1)) && all;
    }
    check(all, "liveness: all 13 pointers stay valid to the end");

    const DeviceArenaStats after = device_arena_stats(id);
    check(after.chunk_alloc_calls > before.chunk_alloc_calls,
          "chunk crossing happened (new chunk allocs > 0)");
    check(after.live_chunks >= 2, "live chunks >= 2");

    std::cout << "[1] liveness " << LIVE_N << " allocs / each=" << (each >> 10) << "KiB"
              << " / chunks=" << after.live_chunks
              << " / capacity=" << (after.total_capacity >> 20) << "MiB\n";

    device_arena_rewind(m);
}

// ----------------------------------------------------------------
// 2) mark/rewind の LIFO 動作: rewind 後に再 alloc すると同じアドレスが返る。
// ----------------------------------------------------------------
static void test_lifo()
{
    const DeviceArenaId id = DeviceArenaId::UNet;

    DeviceArenaMark m  = device_arena_mark(id);
    void*           p1 = device_arena_alloc(id, 1 << 20);
    const uint64_t  mal1 = device_arena_stats(id).cuda_malloc_calls;
    device_arena_rewind(m);

    void*          p2   = device_arena_alloc(id, 1 << 20);
    const uint64_t mal2 = device_arena_stats(id).cuda_malloc_calls;
    device_arena_rewind(m);

    check(p1 == p2, "rewind 後の再 alloc が同一アドレスを返すこと");
    check(mal1 == mal2, "rewind 後の再 alloc で実 cudaMalloc が増えないこと");
    std::cout << "[2] LIFO: p1==p2 / cudaMalloc " << mal1 << " -> " << mal2 << "\n";
}

// ----------------------------------------------------------------
// 3) grow の単調性: 一度育てたチャンクは rewind では解放されない。
// ----------------------------------------------------------------
static void test_grow_monotonic()
{
    const DeviceArenaId id = DeviceArenaId::UNet;

    const DeviceArenaStats s0 = device_arena_stats(id);

    DeviceArenaMark m = device_arena_mark(id);
    const size_t each = (device_arena_first_chunk_bytes(id) >> 3) + 1024;
    for (int i = 0; i < LIVE_N; ++i)
    {
        (void)device_arena_alloc(id, each);
    }
    device_arena_rewind(m);

    const DeviceArenaStats s1 = device_arena_stats(id);
    check(s1.total_capacity == s0.total_capacity,
          "rewind でチャンク容量が減らないこと (grow は単調)");
    check(s1.live_chunks == s0.live_chunks, "rewind でチャンク本数が減らないこと");
    check(s1.cuda_free_calls == s0.cuda_free_calls, "rewind が cudaFree を呼ばないこと");
    check(s1.bytes_in_use == s0.bytes_in_use, "rewind で使用中バイトが戻ること");
    std::cout << "[3] grow monotonic: capacity=" << (s1.total_capacity >> 20)
              << "MiB / chunks=" << s1.live_chunks
              << " / cudaFree=" << s1.cuda_free_calls << "\n";
}

// ----------------------------------------------------------------
// 4) カウンタ: 定常状態 (2 周目以降) で実 cudaMalloc 回数が 0 になること。
// ----------------------------------------------------------------
static void test_steady_state()
{
    const DeviceArenaId id   = DeviceArenaId::UNet;
    const size_t        each = (device_arena_first_chunk_bytes(id) >> 3) + 1024;

    // 1 周目 (ここまでで既に暖まっているが、明示的にもう 1 周する)。
    {
        DeviceArenaMark m = device_arena_mark(id);
        for (int i = 0; i < LIVE_N; ++i)
        {
            (void)device_arena_alloc(id, each);
        }
        device_arena_rewind(m);
    }

    device_arena_reset_counters(id);

    // 2 周目以降 (= 定常状態)。
    for (int rep = 0; rep < 3; ++rep)
    {
        DeviceArenaMark m = device_arena_mark(id);
        for (int i = 0; i < LIVE_N; ++i)
        {
            (void)device_arena_alloc(id, each);
        }
        device_arena_rewind(m);
    }

    const DeviceArenaStats s = device_arena_stats(id);
    check(s.cuda_malloc_calls == 0, "steady state: real cudaMalloc == 0");
    check(s.cuda_free_calls == 0, "steady state: real cudaFree == 0");
    check(s.alloc_calls == (uint64_t)LIVE_N * 3, "alloc_calls counter is correct");
    std::cout << "[4] steady state: cudaMalloc=" << s.cuda_malloc_calls
              << " cudaFree=" << s.cuda_free_calls
              << " alloc=" << s.alloc_calls << "\n";
}

// ----------------------------------------------------------------
// 5) アリーナ分離: "unet" の rewind が "vae" に影響しないこと。
// ----------------------------------------------------------------
static void test_arena_isolation()
{
    const size_t sz = 4u << 20;

    DeviceArenaMark mv = device_arena_mark(DeviceArenaId::VAE);
    void*           pv = device_arena_alloc(DeviceArenaId::VAE, sz);
    CUDA_CHECK(cudaMemset(pv, 0x5A, sz));
    const DeviceArenaStats v0 = device_arena_stats(DeviceArenaId::VAE);

    DeviceArenaMark mu = device_arena_mark(DeviceArenaId::UNet);
    void*           pu = device_arena_alloc(DeviceArenaId::UNet, sz);
    CUDA_CHECK(cudaMemset(pu, 0xA5, sz));
    device_arena_rewind(mu);

    CUDA_CHECK(cudaDeviceSynchronize());
    const DeviceArenaStats v1 = device_arena_stats(DeviceArenaId::VAE);

    check(verify_pattern(pv, sz, 0x5A), "unet rewind 後も vae の領域が無傷であること");
    check(v0.bytes_in_use == v1.bytes_in_use, "unet rewind が vae のカーソルを動かさないこと");
    check(pv != pu, "isolation: two arenas use distinct memory");
    std::cout << "[5] isolation: vae in_use=" << (v1.bytes_in_use >> 10) << "KiB (unchanged)\n";

    device_arena_rewind(mv);
}

// ----------------------------------------------------------------
// 6) アラインメント: 返るポインタが 256B 境界に載っていること。
// ----------------------------------------------------------------
static void test_alignment()
{
    const DeviceArenaId id = DeviceArenaId::UNet;
    const size_t sizes[6]  = { 1, 3, 255, 257, 1000, 65537 };

    DeviceArenaMark m  = device_arena_mark(id);
    bool            ok = true;
    for (int i = 0; i < 6; ++i)
    {
        void* p = device_arena_alloc(id, sizes[i]);
        ok = (p != nullptr) && (((uintptr_t)p & (kDeviceArenaAlign - 1)) == 0) && ok;
    }
    device_arena_rewind(m);

    check(ok, "alignment: all pointers are 256B aligned");
    check(device_arena_alloc(id, 0) == nullptr, "zero-size alloc returns nullptr");
    std::cout << "[6] alignment: 256B OK\n";
}

// ----------------------------------------------------------------
// 7) device_arena_release(): 総容量が 0 に戻り、その後も再利用できること。
// ----------------------------------------------------------------
static void test_release()
{
    const DeviceArenaId id = DeviceArenaId::VAE;

    DeviceArenaMark m = device_arena_mark(id);
    void*           p = device_arena_alloc(id, 4u << 20);
    check(p != nullptr, "alloc before release");
    device_arena_rewind(m);

    device_arena_release(id);
    const DeviceArenaStats s = device_arena_stats(id);
    check(s.total_capacity == 0, "release: total_capacity back to 0");
    check(s.live_chunks == 0, "release: live_chunks back to 0");
    check(s.bytes_in_use == 0, "release: bytes_in_use back to 0");

    // 再利用できること。
    DeviceArenaMark m2 = device_arena_mark(id);
    void*           p2 = device_arena_alloc(id, 4u << 20);
    check(p2 != nullptr, "release: arena is reusable afterwards");
    CUDA_CHECK(cudaMemset(p2, 0x33, 4u << 20));
    CUDA_CHECK(cudaDeviceSynchronize());
    check(verify_pattern(p2, 4u << 20, 0x33), "release: reused region works");
    device_arena_rewind(m2);
    device_arena_release(id);

    std::cout << "[7] release: capacity back to 0 -> reusable OK\n";
}

// ----------------------------------------------------------------
// 8) thread guard (G-8k S2 で契約改訂):
//    (a) **生存中の確保がある**状態で別スレッドが触ったら throw (同時進入の検出)。
//    (b) 静止状態 (全 rewind 済み) なら別スレッドへ所有権が移譲される。
//        HTTP サーバーはリクエストごとにプールの別ワーカーで生成を回すため、
//        (b) が無いと 2 リクエスト目で必ず落ちる。
// ----------------------------------------------------------------
static void test_thread_guard()
{
    const DeviceArenaId id = DeviceArenaId::UNet;

    // --- (a) 生存確保あり → 別スレッドは throw ---
    DeviceArenaMark m = device_arena_mark(id);
    void*           p = device_arena_alloc(id, 1024);
    check(p != nullptr, "thread guard: precondition alloc");

    bool threw = false;
    std::thread t([&threw, id]
                  {
                      try
                      {
                          (void)device_arena_alloc(id, 1024);
                      }
                      catch (const std::exception&)
                      {
                          threw = true;
                      }
                  });
    t.join();
    check(threw, "thread guard: alloc from foreign thread throws while live");

    bool threw_mark = false;
    std::thread t2([&threw_mark, id]
                   {
                       try
                       {
                           (void)device_arena_mark(id);
                       }
                       catch (const std::exception&)
                       {
                           threw_mark = true;
                       }
                   });
    t2.join();
    check(threw_mark, "thread guard: mark from foreign thread throws while live");

    // --- (b) 静止状態 → 別スレッドへ移譲される ---
    device_arena_rewind(m);
    bool handoff_ok = false;
    std::thread t3([&handoff_ok, id]
                   {
                       try
                       {
                           DeviceArenaScope sc(id);
                           void* q = sc.alloc_bytes(1024);
                           handoff_ok = (q != nullptr);
                       }
                       catch (const std::exception&)
                       {
                           handoff_ok = false;
                       }
                   });
    t3.join();
    check(handoff_ok, "thread guard: quiescent arena hands ownership to foreign thread");

    // --- 元スレッドへも (静止状態なので) 戻せる ---
    bool back_ok = true;
    try
    {
        DeviceArenaScope sc(id);
        back_ok = (sc.alloc_bytes(1024) != nullptr);
    }
    catch (const std::exception&)
    {
        back_ok = false;
    }
    check(back_ok, "thread guard: ownership can be handed back");

    std::cout << "[8] thread guard: concurrent entry throws / quiescent handoff OK\n";
}

// ----------------------------------------------------------------
// 9) DeviceArenaScope (RAII): S2 で Scratch を置換する形の疎通。
// ----------------------------------------------------------------
static void test_scope_raii()
{
    const DeviceArenaId id = DeviceArenaId::UNet;

    const DeviceArenaStats s0 = device_arena_stats(id);
    __half*                p0 = nullptr;
    {
        DeviceArenaScope sc(id);
        p0 = sc.alloc<__half>(1024);
        check(p0 != nullptr, "scope.alloc<__half>");
    }
    const DeviceArenaStats s1 = device_arena_stats(id);
    check(s0.bytes_in_use == s1.bytes_in_use, "scope: cursor restored on scope exit");

    {
        DeviceArenaScope sc(id);
        __half* p1 = sc.alloc<__half>(1024);
        // 同一アドレス保証はプール経路のみ (POOL=0 は素の cudaMalloc なので保証しない)。
        if (device_arena_pool_enabled())
        {
            check(p1 == p0, "scope: re-entry returns same address");
        }
        check(p1 != nullptr, "scope: re-entry alloc");
    }
    std::cout << "[9] DeviceArenaScope: mark/rewind RAII OK\n";
}

// ----------------------------------------------------------------
// 10) キルスイッチ (DOLLAMA_POOL=0) 経路: 素の cudaMalloc/cudaFree で同じ結果。
// ----------------------------------------------------------------
static void test_pool_off()
{
    const DeviceArenaId id   = DeviceArenaId::UNet;
    const size_t        each = 1u << 20;

    device_arena_reset_counters(id);
    const DeviceArenaStats s0 = device_arena_stats(id);

    DeviceArenaMark m = device_arena_mark(id);
    void*           ptrs[LIVE_N];
    for (int i = 0; i < LIVE_N; ++i)
    {
        ptrs[i] = device_arena_alloc(id, each);
        check(ptrs[i] != nullptr, "POOL=0: alloc");
        CUDA_CHECK(cudaMemset(ptrs[i], (unsigned char)(i + 1), each));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    bool all = true;
    for (int i = 0; i < LIVE_N; ++i)
    {
        all = verify_pattern(ptrs[i], each, (unsigned char)(i + 1)) && all;
    }
    check(all, "POOL=0: same result as pool path for 13 live blocks");

    const DeviceArenaStats s1 = device_arena_stats(id);
    check(s1.cuda_malloc_calls == (uint64_t)LIVE_N, "POOL=0: one real cudaMalloc per alloc");
    check(s1.chunk_alloc_calls == 0, "POOL=0: no chunk allocation");

    device_arena_rewind(m);
    const DeviceArenaStats s2 = device_arena_stats(id);
    check(s2.cuda_free_calls == (uint64_t)LIVE_N, "POOL=0: rewind が対応する cudaFree を行うこと");
    check(s2.total_capacity == s0.total_capacity, "POOL=0: rewind で容量が戻ること");

    std::cout << "[10] POOL=0: cudaMalloc=" << s1.cuda_malloc_calls
              << " cudaFree=" << s2.cuda_free_calls << " (legacy path restored)\n";
}

#endif // HAVE_CUDA

} // namespace dollama

int main()
{
#ifndef HAVE_CUDA
    std::cout << "[test_device_arena] [SKIP] HAVE_CUDA 未定義\n";
    return 0;
#else
    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0)
    {
        std::cout << "[test_device_arena] [SKIP] CUDA デバイスなし\n";
        return 0;
    }

    try
    {
        if (dollama::device_arena_pool_enabled())
        {
            std::cout << "[test_device_arena] mode = POOL (既定)\n";
            dollama::test_concurrent_liveness();
            dollama::test_lifo();
            dollama::test_grow_monotonic();
            dollama::test_steady_state();
            dollama::test_arena_isolation();
            dollama::test_alignment();
            dollama::test_release();
            dollama::test_scope_raii();
            dollama::test_thread_guard();
            dollama::device_arena_release(dollama::DeviceArenaId::UNet);
        }
        else
        {
            std::cout << "[test_device_arena] mode = POOL OFF (DOLLAMA_POOL=0)\n";
            dollama::test_pool_off();
            dollama::test_alignment();
            dollama::test_scope_raii();
            dollama::test_thread_guard();
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "[test_device_arena] 例外: " << e.what() << "\n";
        return 1;
    }

    if (!dollama::g_ok)
    {
        std::cerr << "[test_device_arena] FAILED\n";
        return 1;
    }
    std::cout << "[test_device_arena] ALL PASSED\n";
    return 0;
#endif
}

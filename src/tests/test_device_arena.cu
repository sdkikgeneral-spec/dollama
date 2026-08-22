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

    // --- (c) G-8k S2b: release (ハンドル破棄で呼ばれる) を挟んでも契約が壊れないこと ---
    //     release 後もアリーナは静止状態なので、別スレッドからの再利用が成立する。
    device_arena_release(id);
    bool after_release_ok = false;
    std::thread t4([&after_release_ok, id]
                   {
                       try
                       {
                           DeviceArenaScope sc(id);
                           void* q = sc.alloc_bytes(1 << 20);
                           after_release_ok = (q != nullptr);
                       }
                       catch (const std::exception&)
                       {
                           after_release_ok = false;
                       }
                   });
    t4.join();
    check(after_release_ok, "thread guard: usable from another thread after release");

    std::cout << "[8] thread guard: concurrent entry throws / quiescent handoff / post-release OK\n";
}

// ----------------------------------------------------------------
// 8b) G-8k T2 (F5): **破棄経路** の契約。
//     守りたい経路:
//       ~DiffusionPipeline -> destroy_resources -> unet_weights_destroy
//       -> device_arena_release -> check_thread throw
//     dtor から例外を出さないこと自体は destroy_resources() の try/catch が担保して
//     いるので、noexcept 版の役目は ① その二重防御 ② UNet 側が拒否されても
//     UNetPersist の解放へ進むこと (素の release だと 1 本目の throw で 2 本目に
//     到達しない) である。「noexcept 版が terminate を単独で防いでいる」という
//     読み方は誤り (T2 相互レビュー 中2)。本テストはこのラッパの契約だけを固定する。
//
//     (1) 生存確保あり + 別スレッド -> 素の device_arena_release は **throw する**
//         (落ちる契約は維持する。noexcept 版を足したことで緩んでいないことの明文化)。
//     (2) 同条件で device_arena_release_noexcept は **投げず false**。しかも
//         **1 バイトも解放していない** (統計据え置き + 生存ポインタの中身が無事) ので、
//         元スレッドへ戻ればそのまま使い続けられる。
//     (3) 静止状態での release_noexcept は **true** で、素の release と同じ結果
//         (容量 0 -> 再成長できる)。
//     (4) もう 1 つの throw 点 = arena_of の不正 id でも投げず false (軽5)。
//         check_thread だけを固定して「arena_of 経路は素通し」にしないため。
//
//     POOL / POOL OFF の両枠に登録する: DOLLAMA_POOL=0 でも生存確保は fallback_ptrs に
//     載るため is_quiescent が false になり、まったく同じ契約が成立する。片枠だけだと
//     device_arena_pool_off が素通しになる。
//
//     注: (2) と (4) で device_arena_release_noexcept が stderr に 1 行ずつ (計 2 行)
//     出す。**これは期待動作** (見送ったことを黙って隠さないための出力) であって、
//     新規の赤ではない。
//
//     真の「2 パイプライン同居」テストは書けない (重み 7067MB x2 + アリーナ 6.3GB が
//     16GB に入らない)。アリーナ層の等価形 = 「解放しようとした瞬間に別スレッドの
//     生存確保がある」状態で代替する。
// ----------------------------------------------------------------
static void test_release_noexcept_contract()
{
    const DeviceArenaId id = DeviceArenaId::UNet;
    const size_t        n  = 4u << 20;

    // 元スレッドが所有権を取り、静止状態から始める。
    device_arena_release(id);

    DeviceArenaMark m = device_arena_mark(id);
    void*           p = device_arena_alloc(id, n);
    check(p != nullptr, "release_noexcept: precondition alloc");
    CUDA_CHECK(cudaMemset(p, 0x5A, n));
    CUDA_CHECK(cudaDeviceSynchronize());

    const DeviceArenaStats before = device_arena_stats(id);
    check(before.total_capacity != 0, "release_noexcept: precondition capacity is non-zero");

    // --- (1) 素の release は foreign thread から throw する ---
    bool threw = false;
    std::thread t([&threw, id]
                  {
                      try
                      {
                          device_arena_release(id);
                      }
                      catch (const std::exception&)
                      {
                          threw = true;
                      }
                  });
    t.join();
    check(threw, "release_noexcept: raw release from foreign thread throws while live");

    // --- (2) noexcept 版は投げず false・アリーナは無傷 ---
    bool caught_any = false;
    bool ret        = true;
    std::thread t2([&ret, &caught_any, id]
                   {
                       try
                       {
                           ret = device_arena_release_noexcept(id);
                       }
                       catch (...)
                       {
                           caught_any = true;
                       }
                   });
    t2.join();
    check(!caught_any, "release_noexcept: never throws (a dtor would std::terminate)");
    check(!ret, "release_noexcept: returns false when the arena is not releasable");

    // 「1 バイトも解放していない」の検証:
    //   (a) cudaFree 回数が 1 回も増えていない
    //   (b) 容量 / チャンク本数 / カーソルが完全に据え置き
    //   (c) 生存ポインタの中身が読み返せる (解放 -> 再利用で壊されていない)
    // **ゲートとして効いているのは (a) と (b) だけ** (T2 相互レビュー 軽6)。
    //   (c) の verify_pattern は補助証拠にすぎない: 本当に cudaFree されていたら
    //   読み返しは UB であって「たまたま通る」方が普通なので、失敗を当てにできない。
    //   **将来 (a)(b) を削って (c) だけ残す縮小をしないこと** (ゲートが空になる)。
    const DeviceArenaStats after = device_arena_stats(id);
    check(after.cuda_free_calls == before.cuda_free_calls,
          "release_noexcept: no cudaFree happened on the refused path");
    check(after.total_capacity == before.total_capacity,
          "release_noexcept: total_capacity untouched on the refused path");
    check(after.live_chunks == before.live_chunks,
          "release_noexcept: live_chunks untouched on the refused path");
    check(after.bytes_in_use == before.bytes_in_use,
          "release_noexcept: cursor untouched on the refused path");
    check(verify_pattern(p, n, 0x5A), "release_noexcept: live region intact after false");

    // 元スレッドでそのまま使い続けられること (状態が壊れていない)。
    void* p2 = device_arena_alloc(id, n);
    check(p2 != nullptr, "release_noexcept: arena still usable after a refused release");
    CUDA_CHECK(cudaMemset(p2, 0x5B, n));
    CUDA_CHECK(cudaDeviceSynchronize());
    check(verify_pattern(p, n, 0x5A) && verify_pattern(p2, n, 0x5B),
          "release_noexcept: both live regions valid after a refused release");
    device_arena_rewind(m);

    // --- (3) 静止状態なら true。素の release と同じ結果になる ---
    const bool ok = device_arena_release_noexcept(id);
    check(ok, "release_noexcept: returns true when quiescent");
    const DeviceArenaStats q = device_arena_stats(id);
    check(q.total_capacity == 0, "release_noexcept: total_capacity back to 0");
    check(q.live_chunks == 0, "release_noexcept: live_chunks back to 0");
    check(q.bytes_in_use == 0, "release_noexcept: bytes_in_use back to 0");

    DeviceArenaMark m2 = device_arena_mark(id);
    void*           p3 = device_arena_alloc(id, n);
    check(p3 != nullptr, "release_noexcept: arena regrows after a successful release");
    CUDA_CHECK(cudaMemset(p3, 0x5C, n));
    CUDA_CHECK(cudaDeviceSynchronize());
    check(verify_pattern(p3, n, 0x5C), "release_noexcept: regrown region works");
    device_arena_rewind(m2);

    // --- (4) もう 1 つの throw 点 = arena_of の不正 id も noexcept で受け止める ---
    //     release_noexcept が握り潰す対象は check_thread だけではない (軽5)。
    //     不正 id は arena_of がテーブルを引く前に throw するので、こちらも
    //     「1 バイトも解放していない」= アリーナに一切触っていない。
    bool bad_threw = false;
    bool bad_ret   = true;
    try
    {
        bad_ret = device_arena_release_noexcept(static_cast<DeviceArenaId>(99));
    }
    catch (...)
    {
        bad_threw = true;
    }
    check(!bad_threw, "release_noexcept: invalid arena id does not throw either");
    check(!bad_ret, "release_noexcept: invalid arena id returns false");

    std::cout << "[8b] release_noexcept: live+foreign -> false (arena untouched)"
                 " / quiescent -> true (== release) / bad id -> false OK\n";

    // 後続テストの前提 (静止状態) を壊さずに抜ける。
    device_arena_release(id);
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

// ----------------------------------------------------------------
// 11) G-8k S3: GB 級の単発要求 (VAE のキャリー 512MiB / 1GiB がこれに当たる)。
//     成長則は「チャンクサイズ = max(刻み幅, 要求サイズ)」なので、刻み幅を超える
//     要求に対しては **要求サイズちょうど** のチャンクが 1 本だけ生えるはず。
//     (倍々成長が復活すると capacity がここで跳ねるので、その回帰検知でもある)
// ----------------------------------------------------------------
static void test_giant_alloc_exact_chunk()
{
    const DeviceArenaId id = DeviceArenaId::UNet;

    device_arena_release(id);
    const DeviceArenaStats s0 = device_arena_stats(id);

    const size_t giant = (size_t)1 << 30;  // 1GiB (VAE の FP32 キャリー 1 本と同寸)
    DeviceArenaMark m  = device_arena_mark(id);
    void*           p  = device_arena_alloc(id, giant);
    check(p != nullptr, "giant: alloc 1GiB");

    const DeviceArenaStats s1 = device_arena_stats(id);
    if (device_arena_pool_enabled())
    {
        check(s1.chunk_alloc_calls == s0.chunk_alloc_calls + 1,
              "giant: exactly one new chunk");
        check(s1.total_capacity == s0.total_capacity + giant,
              "giant: chunk size == request size (no doubling growth)");
        check(s1.live_chunks == 1, "giant: live_chunks == 1");
    }
    check(s1.peak_request_bytes >= giant, "giant: peak_request_bytes >= 1GiB");

    // 端まで実体があること (先頭/中央/末尾を書いて読み返す)。
    CUDA_CHECK(cudaMemset(p, 0x77, giant));
    CUDA_CHECK(cudaDeviceSynchronize());
    check(verify_pattern(p, giant, 0x77), "giant: whole 1GiB region is backed");

    std::cout << "[11] giant alloc: cap=" << (s1.total_capacity >> 20)
              << "MiB chunks=" << s1.live_chunks
              << " (要求 " << (giant >> 20) << "MiB ちょうど)\n";

    device_arena_rewind(m);
    device_arena_release(id);
}

// ----------------------------------------------------------------
// 12) G-8k S3 の急所: 「小確保を複数 → 全 rewind → GB 級確保 → 再び小確保」。
//     GB 級は現チャンクに収まらないため **チャンク列の途中に新チャンクが挿入** される。
//     挿入で vector 内の index はずれるが、デバイスメモリ本体は動かない設計なので、
//     挿入を跨いで生き続けているポインタの中身は壊れてはならない。
//     (VAE decode が正にこの形: 段の Scratch を rewind した後、外側スコープから
//      FP32 キャリー 1GiB x2 を切り、その後また段内の小確保が続く)
// ----------------------------------------------------------------
static void test_insert_across_live_pointers()
{
    const DeviceArenaId id    = DeviceArenaId::UNet;
    const size_t        first = device_arena_first_chunk_bytes(id);

    device_arena_release(id);

    // --- 事前に複数チャンクへ育てておく (挿入位置が「末尾」でなく「途中」になる条件) ---
    {
        DeviceArenaMark warm = device_arena_mark(id);
        const size_t    each = (first >> 3) + 1024;   // 8 本で 1 チャンクを溢れさせる寸法
        for (int i = 0; i < 12; ++i)
        {
            (void)device_arena_alloc(id, each);
        }
        device_arena_rewind(warm);                    // ここで全 rewind (チャンクは残る)
    }
    const DeviceArenaStats sw = device_arena_stats(id);
    check(!device_arena_pool_enabled() || sw.live_chunks >= 2,
          "insert: pre-grown to >= 2 chunks");

    // --- 小確保 (前半) ---
    const size_t    small = 1u << 20;
    DeviceArenaMark m     = device_arena_mark(id);
    void*           ps[6];
    for (int i = 0; i < 3; ++i)
    {
        ps[i] = device_arena_alloc(id, small);
        check(ps[i] != nullptr, "insert: small alloc (first half)");
        CUDA_CHECK(cudaMemset(ps[i], (unsigned char)(0x10 + i), small));
    }

    // --- GB 級 (ここでチャンク挿入が起きる) ---
    const size_t giant = (size_t)1 << 30;
    void*        pg    = device_arena_alloc(id, giant);
    check(pg != nullptr, "insert: giant alloc");
    CUDA_CHECK(cudaMemset(pg, 0x2A, giant));
    const DeviceArenaStats sg = device_arena_stats(id);
    check(!device_arena_pool_enabled() || sg.live_chunks > sw.live_chunks,
          "insert: giant alloc added one chunk");

    // --- 小確保 (後半) ---
    for (int i = 3; i < 6; ++i)
    {
        ps[i] = device_arena_alloc(id, small);
        check(ps[i] != nullptr, "insert: small alloc (second half)");
        CUDA_CHECK(cudaMemset(ps[i], (unsigned char)(0x10 + i), small));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- 全部読み返す: 挿入を跨いでも前半のポインタが無傷であること ---
    bool all = true;
    for (int i = 0; i < 6; ++i)
    {
        all = verify_pattern(ps[i], small, (unsigned char)(0x10 + i)) && all;
    }
    all = verify_pattern(pg, giant, 0x2A) && all;
    check(all, "insert: all pointers intact across chunk insertion");

    // 重なりが無いこと (アドレス範囲の総当たり)。
    bool        disjoint = true;
    const char* bases[7];
    size_t      lens[7];
    for (int i = 0; i < 6; ++i)
    {
        bases[i] = (const char*)ps[i];
        lens[i]  = small;
    }
    bases[6] = (const char*)pg;
    lens[6]  = giant;
    for (int i = 0; i < 7; ++i)
    {
        for (int j = i + 1; j < 7; ++j)
        {
            const bool ov = (bases[i] < bases[j] + lens[j]) && (bases[j] < bases[i] + lens[i]);
            disjoint      = disjoint && !ov;
        }
    }
    check(disjoint, "insert: handed-out regions are disjoint");

    std::cout << "[12] insert across live: chunks " << sw.live_chunks << " -> "
              << sg.live_chunks << " / 7 ptrs intact / disjoint\n";

    device_arena_rewind(m);
    device_arena_release(id);
}

// ----------------------------------------------------------------
// 13) G-8k S3: device_arena_release() の統計ゼロ化と再成長。
//     DOLLAMA_ARENA_RELEASE=1 (画像境界での明示解放) が踏む経路そのもの。
// ----------------------------------------------------------------
static void test_release_then_regrow()
{
    const DeviceArenaId id = DeviceArenaId::UNet;

    // 何かしら育てる。
    {
        DeviceArenaScope sc(id);
        void*            p = sc.alloc_bytes(64u << 20);
        check(p != nullptr, "regrow: alloc before release");
    }
    const DeviceArenaStats sb = device_arena_stats(id);
    check(!device_arena_pool_enabled() || sb.total_capacity > 0, "regrow: capacity > 0 before release");

    device_arena_release(id);
    const DeviceArenaStats s0 = device_arena_stats(id);
    check(s0.total_capacity == 0, "regrow: release -> total_capacity == 0");
    check(s0.live_chunks == 0, "regrow: release -> live_chunks == 0");
    check(s0.bytes_in_use == 0, "regrow: release -> bytes_in_use == 0");
    check(s0.live_request_bytes == 0, "regrow: release -> live_request_bytes == 0");
    check(!device_arena_pool_enabled() || s0.cuda_free_calls > sb.cuda_free_calls,
          "regrow: release issues real cudaFree");

    // 再成長 + 実データの往復。
    {
        DeviceArenaScope sc(id);
        void*            q = sc.alloc_bytes(64u << 20);
        check(q != nullptr, "regrow: alloc works after release");
        CUDA_CHECK(cudaMemset(q, 0x63, 64u << 20));
        CUDA_CHECK(cudaDeviceSynchronize());
        check(verify_pattern(q, 64u << 20, 0x63), "regrow: reused region is usable");
    }
    const DeviceArenaStats s1 = device_arena_stats(id);
    check(!device_arena_pool_enabled() || s1.total_capacity > 0, "regrow: capacity grew again");

    std::cout << "[13] release -> regrow: cap " << (sb.total_capacity >> 20) << "MiB -> 0 -> "
              << (s1.total_capacity >> 20) << "MiB\n";

    device_arena_release(id);
}

// ----------------------------------------------------------------
// 14) G-8k S3b: 事前 reserve が効いていること。
//     reserve 1 本の中に収まる確保を何度繰り返しても、新チャンクは 1 本も生えない
//     (= チャンク跨ぎ = 捨て分がゼロ)。S4 で VRAM ゲートを割った原因への直接の回帰。
// ----------------------------------------------------------------
static void test_reserve_no_chunk_growth()
{
    const DeviceArenaId id = DeviceArenaId::UNet;
    const size_t        rsv = (size_t)768 << 20;  // 768MiB を 1 本

    device_arena_release(id);
    device_arena_reserve(id, rsv);

    const DeviceArenaStats s0 = device_arena_stats(id);
    if (device_arena_pool_enabled())
    {
        check(s0.reserved_bytes == rsv, "reserve: reserved_bytes == request");
        check(s0.total_capacity == rsv, "reserve: capacity == reserve (single chunk)");
        check(s0.live_chunks == 1, "reserve: live_chunks == 1");
    }
    else
    {
        check(s0.reserved_bytes == 0, "reserve: no-op under DOLLAMA_POOL=0");
        check(s0.total_capacity == 0, "reserve: no capacity under DOLLAMA_POOL=0");
    }

    device_arena_reset_counters(id);

    // reserve 未満の確保を、寸法を変えながら何周も回す (跨ぎが起きるなら必ず出る形)。
    const size_t sizes[5] = { (size_t)200 << 20, (size_t)5 << 20, (size_t)300 << 20,
                              (size_t)1 << 20,   (size_t)120 << 20 };
    for (int rep = 0; rep < 4; ++rep)
    {
        DeviceArenaScope sc(id);
        for (int i = 0; i < 5; ++i)
        {
            void* p = sc.alloc_bytes(sizes[i]);
            check(p != nullptr, "reserve: alloc within reserve");
        }
    }
    const DeviceArenaStats s1 = device_arena_stats(id);
    if (device_arena_pool_enabled())
    {
        check(s1.chunk_alloc_calls == 0, "reserve: no new chunk while within reserve");
        check(s1.cuda_malloc_calls == 0, "reserve: no real cudaMalloc while within reserve");
        check(s1.cuda_free_calls == 0, "reserve: no real cudaFree while within reserve");
        check(s1.total_capacity == rsv, "reserve: capacity unchanged");
        // 捨て分 = capacity - live peak。reserve 内に収まる限り reserve 分だけ。
        check(s1.peak_request_bytes <= s1.total_capacity, "reserve: live peak <= capacity");
    }

    std::cout << "[14] reserve: cap=" << (s1.total_capacity >> 20)
              << "MiB reserved=" << (s1.reserved_bytes >> 20)
              << "MiB live_peak=" << (s1.peak_request_bytes >> 20)
              << "MiB chunk_alloc=" << s1.chunk_alloc_calls << "\n";

    device_arena_release(id);
}

// ----------------------------------------------------------------
// 15) G-8k S3b: reserve を **超えた** 要求はフォールバック (チャンク追加) し、
//     そのとき既に配ってある生存ポインタの内容が壊れないこと。
//     reserve は速度/VRAM の最適化であって正しさの前提ではない、の回帰。
// ----------------------------------------------------------------
static void test_reserve_overflow_fallback()
{
    const DeviceArenaId id  = DeviceArenaId::UNet;
    const size_t        rsv = (size_t)512 << 20;

    device_arena_release(id);
    device_arena_reserve(id, rsv);
    device_arena_reset_counters(id);

    const size_t small = 64u << 20;
    DeviceArenaMark m  = device_arena_mark(id);

    void* ps[3];
    for (int i = 0; i < 3; ++i)
    {
        ps[i] = device_arena_alloc(id, small);
        check(ps[i] != nullptr, "overflow: alloc within reserve");
        CUDA_CHECK(cudaMemset(ps[i], (unsigned char)(0x40 + i), small));
    }

    // reserve の残りに入らない要求 -> フォールバックで新チャンクが生える。
    // **注 (G-8k T2 / F3)**: この 1 本で device_arena.cu が stderr へ
    //   "[ALLOC] reserve shortage: ..." を 1 行出す。**本テストが意図的に
    //   reserve 不足を起こしているので、これは期待動作**であり新規の赤ではない
    //   (F3 以前は DOLLAMA_PROFILE=1 のときだけ stdout に出ていた)。
    const size_t over = (size_t)768 << 20;
    void*        po   = device_arena_alloc(id, over);
    check(po != nullptr, "overflow: fallback alloc beyond reserve");
    CUDA_CHECK(cudaMemset(po, 0x7E, over));

    // フォールバック後にもう 1 本 (挿入後のカーソルからの確保)。
    void* pl = device_arena_alloc(id, small);
    check(pl != nullptr, "overflow: alloc after fallback");
    CUDA_CHECK(cudaMemset(pl, 0x43, small));
    CUDA_CHECK(cudaDeviceSynchronize());

    bool all = true;
    for (int i = 0; i < 3; ++i)
    {
        all = verify_pattern(ps[i], small, (unsigned char)(0x40 + i)) && all;
    }
    all = verify_pattern(po, over, 0x7E) && all;
    all = verify_pattern(pl, small, 0x43) && all;
    check(all, "overflow: all live pointers intact across fallback chunk insertion");

    const DeviceArenaStats s = device_arena_stats(id);
    if (device_arena_pool_enabled())
    {
        check(s.chunk_alloc_calls >= 1, "overflow: fallback added a chunk");
        check(s.reserved_bytes == rsv, "overflow: reserved_bytes stays as reserved");
    }
    std::cout << "[15] reserve overflow: chunk_alloc=" << s.chunk_alloc_calls
              << " cap=" << (s.total_capacity >> 20)
              << "MiB reserved=" << (s.reserved_bytes >> 20) << "MiB / pointers intact\n";

    device_arena_rewind(m);
    device_arena_release(id);
}

// ----------------------------------------------------------------
// 16) G-8k S3b: 非静止状態での reserve は throw すること。
//     reserve は既存チャンクを解放するので、生存ポインタがあると dangling になる。
//     (POOL=0 では no-op が正なので throw しない = そちらも検査する)
// ----------------------------------------------------------------
static void test_reserve_requires_quiescent()
{
    const DeviceArenaId id = DeviceArenaId::UNet;

    device_arena_release(id);

    DeviceArenaMark m = device_arena_mark(id);
    void*           p = device_arena_alloc(id, 4u << 20);
    check(p != nullptr, "reserve guard: precondition alloc");

    bool threw = false;
    try
    {
        device_arena_reserve(id, (size_t)256 << 20);
    }
    catch (const std::exception&)
    {
        threw = true;
    }
    if (device_arena_pool_enabled())
    {
        check(threw, "reserve guard: reserve while live throws");
    }
    else
    {
        check(!threw, "reserve guard: no-op (no throw) under DOLLAMA_POOL=0");
    }

    device_arena_rewind(m);

    // 静止状態なら通ること。
    bool ok = true;
    try
    {
        device_arena_reserve(id, (size_t)256 << 20);
    }
    catch (const std::exception&)
    {
        ok = false;
    }
    check(ok, "reserve guard: reserve succeeds when quiescent");
    // bytes == 0 は no-op (capacity も reserved_bytes も動かない)。
    const DeviceArenaStats sb = device_arena_stats(id);
    device_arena_reserve(id, 0);
    const DeviceArenaStats sa = device_arena_stats(id);
    check(sb.total_capacity == sa.total_capacity && sb.reserved_bytes == sa.reserved_bytes,
          "reserve guard: bytes==0 is a no-op");

    std::cout << "[16] reserve guard: live -> throw / quiescent -> ok\n";

    device_arena_release(id);
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
            // G-8k T2 (F5): 破棄経路 (release_noexcept) の契約。
            dollama::test_release_noexcept_contract();
            // G-8k S3 追加分 (GB 級チャンク / 挿入跨ぎ / release -> 再成長)。
            dollama::test_giant_alloc_exact_chunk();
            dollama::test_insert_across_live_pointers();
            dollama::test_release_then_regrow();
            // G-8k S3b 追加分 (事前 reserve)。
            dollama::test_reserve_no_chunk_growth();
            dollama::test_reserve_overflow_fallback();
            dollama::test_reserve_requires_quiescent();
            dollama::device_arena_release(dollama::DeviceArenaId::UNet);
        }
        else
        {
            std::cout << "[test_device_arena] mode = POOL OFF (DOLLAMA_POOL=0)\n";
            dollama::test_pool_off();
            dollama::test_alignment();
            dollama::test_scope_raii();
            dollama::test_thread_guard();
            // G-8k T2 (F5): POOL=0 でも生存確保は fallback_ptrs に載る =
            //   is_quiescent が false になるので、破棄経路の契約は同一。
            dollama::test_release_noexcept_contract();
            // G-8k S3: キルスイッチ経路でも「挿入跨ぎ」相当の同時生存と release -> 再成長が
            // 同じ結果になること (チャンク本数の検査はプール経路のみで判定する)。
            dollama::test_insert_across_live_pointers();
            dollama::test_release_then_regrow();
            // G-8k S3b: キルスイッチ経路では reserve が no-op であること。
            dollama::test_reserve_no_chunk_growth();
            dollama::test_reserve_requires_quiescent();
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

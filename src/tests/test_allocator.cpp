// Allocator 単体テスト + 速度計測
// 失敗時は std::cerr に出力して return 1。
// 全テスト通過時は "[test_allocator] ALL PASSED" を出力して return 0。
// 速度計測結果は BENCH: プレフィックスで stdout に出力する。
#include <iostream>
#include <chrono>
#include <stdexcept>
#include <utility>

#include "core/allocator.hpp"

namespace dollama {

using Clock    = std::chrono::steady_clock;
using MsDbl    = std::chrono::duration<double, std::milli>;

// ----------------------------------------------------------------
// テスト1: PinnedAllocator — alloc/free が正常動作する
// HAVE_CUDA 未定義時は ::operator new/delete にフォールバックする
// ----------------------------------------------------------------
static bool test_pinned_alloc_free()
{
    void* p = PinnedAllocator::alloc(1024);
    if (p == nullptr)
    {
        std::cerr << "[test_pinned_alloc_free] alloc(1024) が nullptr を返した\n";
        return false;
    }
    PinnedAllocator::free(p);

    std::cout << "[test_pinned_alloc_free] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// テスト2: UniqueBuffer 基本動作 — get() 非 null、size() が正しい
// ----------------------------------------------------------------
static bool test_unique_buffer_basic()
{
    UniqueBuffer<PinnedAllocator> buf(1024);

    if (buf.get() == nullptr)
    {
        std::cerr << "[test_unique_buffer_basic] get() が nullptr\n";
        return false;
    }
    if (buf.size() != 1024)
    {
        std::cerr << "[test_unique_buffer_basic] size() が " << buf.size()
                  << " (期待: 1024)\n";
        return false;
    }

    std::cout << "[test_unique_buffer_basic] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// テスト3: UniqueBuffer ムーブコンストラクタ
// 新側が元ポインタを持ち、旧側が nullptr / size==0 になる
// ----------------------------------------------------------------
static bool test_unique_buffer_move_ctor()
{
    UniqueBuffer<PinnedAllocator> a(512);
    void* orig = a.get();

    UniqueBuffer<PinnedAllocator> b(std::move(a));

    if (b.get() != orig)
    {
        std::cerr << "[test_unique_buffer_move_ctor] ムーブ後: b.get() が元のポインタと異なる\n";
        return false;
    }
    if (b.size() != 512)
    {
        std::cerr << "[test_unique_buffer_move_ctor] ムーブ後: b.size() = "
                  << b.size() << " (期待: 512)\n";
        return false;
    }
    if (a.get() != nullptr)
    {
        std::cerr << "[test_unique_buffer_move_ctor] ムーブ後: 旧側 a.get() が nullptr でない\n";
        return false;
    }
    if (a.size() != 0)
    {
        std::cerr << "[test_unique_buffer_move_ctor] ムーブ後: 旧側 a.size() = "
                  << a.size() << " (期待: 0)\n";
        return false;
    }

    std::cout << "[test_unique_buffer_move_ctor] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// テスト4: UniqueBuffer ムーブ代入演算子
// 旧バッファが解放され、代入先が元ポインタを引き継ぐ
// ----------------------------------------------------------------
static bool test_unique_buffer_move_assign()
{
    UniqueBuffer<PinnedAllocator> a(256);
    void* orig = a.get();

    UniqueBuffer<PinnedAllocator> b(128);
    b = std::move(a);

    if (b.get() != orig)
    {
        std::cerr << "[test_unique_buffer_move_assign] 代入後: b.get() が元のポインタと異なる\n";
        return false;
    }
    if (b.size() != 256)
    {
        std::cerr << "[test_unique_buffer_move_assign] 代入後: b.size() = "
                  << b.size() << " (期待: 256)\n";
        return false;
    }
    if (a.get() != nullptr)
    {
        std::cerr << "[test_unique_buffer_move_assign] 代入後: 旧側 a.get() が nullptr でない\n";
        return false;
    }

    std::cout << "[test_unique_buffer_move_assign] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// テスト5: 自己ムーブ代入でクラッシュしないことを確認
// 自己代入後の状態は nullptr になるが、例外・クラッシュがなければ PASSED
// ----------------------------------------------------------------
static bool test_unique_buffer_self_assign()
{
    UniqueBuffer<PinnedAllocator> b(256);

    // 警告を避けるため間接的に自己代入する
    UniqueBuffer<PinnedAllocator>& ref = b;
    b = std::move(ref);

    // 自己代入後は get()==nullptr になる可能性があるが、それは正常
    std::cout << "[test_unique_buffer_self_assign] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// テスト6: HAVE_CUDA 未定義時に CudaAllocator::alloc が std::runtime_error
// ----------------------------------------------------------------
static bool test_cuda_alloc_no_cuda_throws()
{
#ifdef HAVE_CUDA
    std::cout << "[test_cuda_alloc_no_cuda_throws] SKIP (HAVE_CUDA あり)\n";
    return true;
#else
    try
    {
        void* p = CudaAllocator::alloc(1);
        (void)p;
        std::cerr << "[test_cuda_alloc_no_cuda_throws] 例外が投げられなかった\n";
        return false;
    }
    catch (const std::runtime_error&)
    {
        // 期待通り
    }
    catch (const std::exception& e)
    {
        std::cerr << "[test_cuda_alloc_no_cuda_throws] 予期しない例外型: " << e.what() << "\n";
        return false;
    }

    std::cout << "[test_cuda_alloc_no_cuda_throws] PASSED\n";
    return true;
#endif
}

// ----------------------------------------------------------------
// テスト7: PinnedBuffer エイリアスが UniqueBuffer<PinnedAllocator> と同等に動く
// ----------------------------------------------------------------
static bool test_pinned_buffer_alias()
{
    PinnedBuffer buf(2048);

    if (buf.get() == nullptr)
    {
        std::cerr << "[test_pinned_buffer_alias] get() が nullptr\n";
        return false;
    }
    if (buf.size() != 2048)
    {
        std::cerr << "[test_pinned_buffer_alias] size() = " << buf.size()
                  << " (期待: 2048)\n";
        return false;
    }

    std::cout << "[test_pinned_buffer_alias] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// ベンチ1: PinnedAllocator alloc/free スループット (1MB × 1000 回)
// ----------------------------------------------------------------
static bool bench_pinned_alloc_free()
{
    static constexpr size_t kBytes = 1 * 1024 * 1024;  // 1MB
    static constexpr int    kN     = 1000;

    auto t0 = Clock::now();
    for (int i = 0; i < kN; ++i)
    {
        void* p = PinnedAllocator::alloc(kBytes);
        PinnedAllocator::free(p);
    }
    double elapsed_ms = MsDbl(Clock::now() - t0).count();
    double total_gb   = static_cast<double>(kN) * kBytes / 1e9;
    double gbps       = total_gb / (elapsed_ms / 1000.0);

    std::cout << "[bench_pinned_alloc_free]\n";
    std::cout << "  BENCH: " << kN << " x alloc/free (1MB)"
              << "  total = " << elapsed_ms << " ms"
              << "  throughput = " << gbps << " GB/s\n";
    return true;
}

// ----------------------------------------------------------------
// ベンチ2: UniqueBuffer 生成・解放コスト (小サイズ 4KB と 大サイズ 1MB)
// ----------------------------------------------------------------
static bool bench_unique_buffer_create()
{
    static constexpr int kN = 10000;

    // 小サイズ (4KB)
    {
        auto t0 = Clock::now();
        for (int i = 0; i < kN; ++i)
        {
            UniqueBuffer<PinnedAllocator> buf(4096);
            (void)buf;
        }
        double elapsed_ms = MsDbl(Clock::now() - t0).count();
        std::cout << "[bench_unique_buffer_create]\n";
        std::cout << "  BENCH: " << kN << " x UniqueBuffer(4KB)"
                  << "  total = " << elapsed_ms << " ms"
                  << "  per = " << elapsed_ms / kN * 1000.0 << " us/op\n";
    }

    // 大サイズ (1MB)
    {
        auto t0 = Clock::now();
        for (int i = 0; i < kN; ++i)
        {
            UniqueBuffer<PinnedAllocator> buf(1024 * 1024);
            (void)buf;
        }
        double elapsed_ms = MsDbl(Clock::now() - t0).count();
        std::cout << "  BENCH: " << kN << " x UniqueBuffer(1MB)"
                  << "  total = " << elapsed_ms << " ms"
                  << "  per = " << elapsed_ms / kN * 1000.0 << " us/op\n";
    }

    return true;
}

} // namespace dollama

int main()
{
    bool ok = true;

    ok = dollama::test_pinned_alloc_free()           && ok;
    ok = dollama::test_unique_buffer_basic()         && ok;
    ok = dollama::test_unique_buffer_move_ctor()     && ok;
    ok = dollama::test_unique_buffer_move_assign()   && ok;
    ok = dollama::test_unique_buffer_self_assign()   && ok;
    ok = dollama::test_cuda_alloc_no_cuda_throws()   && ok;
    ok = dollama::test_pinned_buffer_alias()         && ok;
    ok = dollama::bench_pinned_alloc_free()          && ok;
    ok = dollama::bench_unique_buffer_create()       && ok;

    if (!ok)
    {
        std::cerr << "[test_allocator] FAILED\n";
        return 1;
    }

    std::cout << "[test_allocator] ALL PASSED\n";
    return 0;
}

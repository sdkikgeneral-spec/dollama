// Tensor クラス単体テスト + 速度計測
// 失敗時は std::cerr に出力して return 1。
// 全テスト通過時は "[test_tensor] ALL PASSED" を出力して return 0。
// 速度計測結果は BENCH: プレフィックスで stdout に出力する。
#include <iostream>
#include <chrono>
#include <stdexcept>
#include <cstring>

#include "core/tensor.hpp"

namespace dollama {

using Clock = std::chrono::steady_clock;
using MsDbl = std::chrono::duration<double, std::milli>;

// ----------------------------------------------------------------
// テスト1: デフォルト構築 — numel()==0, ndim()==0
// ----------------------------------------------------------------
static bool test_default_ctor()
{
    Tensor t;

    if (t.numel() != 0)
    {
        std::cerr << "[test_default_ctor] numel() が " << t.numel()
                  << " (期待: 0)\n";
        return false;
    }
    if (t.ndim() != 0)
    {
        std::cerr << "[test_default_ctor] ndim() が " << t.ndim()
                  << " (期待: 0)\n";
        return false;
    }

    std::cout << "[test_default_ctor] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// テスト2: CPU Tensor の形状確認
// ----------------------------------------------------------------
static bool test_shape_cpu()
{
    Tensor t({2, 3, 4}, Device::CPU);

    if (t.ndim() != 3)
    {
        std::cerr << "[test_shape_cpu] ndim() が " << t.ndim()
                  << " (期待: 3)\n";
        return false;
    }
    if (t.dim(0) != 2)
    {
        std::cerr << "[test_shape_cpu] dim(0) が " << t.dim(0)
                  << " (期待: 2)\n";
        return false;
    }
    if (t.numel() != 24)
    {
        std::cerr << "[test_shape_cpu] numel() が " << t.numel()
                  << " (期待: 24)\n";
        return false;
    }
    if (t.nbytes() != 96)
    {
        std::cerr << "[test_shape_cpu] nbytes() が " << t.nbytes()
                  << " (期待: 96)\n";
        return false;
    }

    std::cout << "[test_shape_cpu] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// テスト3: CPU Tensor のゼロ初期化と書き込み・読み戻し
// ----------------------------------------------------------------
static bool test_data_zero_init()
{
    Tensor t({4, 4}, Device::CPU);
    float* p = t.data();

    // ゼロ初期化の確認
    for (size_t i = 0; i < t.numel(); ++i)
    {
        if (p[i] != 0.0f)
        {
            std::cerr << "[test_data_zero_init] p[" << i << "] が "
                      << p[i] << " (期待: 0.0)\n";
            return false;
        }
    }

    // 書き込み後に読み戻せることを確認
    for (size_t i = 0; i < t.numel(); ++i)
    {
        p[i] = static_cast<float>(i) * 1.5f;
    }
    for (size_t i = 0; i < t.numel(); ++i)
    {
        float expected = static_cast<float>(i) * 1.5f;
        if (p[i] != expected)
        {
            std::cerr << "[test_data_zero_init] 読み戻し失敗: p[" << i
                      << "] = " << p[i] << " (期待: " << expected << ")\n";
            return false;
        }
    }

    std::cout << "[test_data_zero_init] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// テスト4: PINNED Tensor で data() が使える (例外を投げない)
// ----------------------------------------------------------------
static bool test_pinned_device()
{
    // PINNED は HAVE_CUDA 有無にかかわらず ::operator new にフォールバックする
    try
    {
        Tensor t({8}, Device::PINNED);
        float* p = t.data();
        if (p == nullptr)
        {
            std::cerr << "[test_pinned_device] data() が nullptr を返した\n";
            return false;
        }
        // 書き込みテスト
        p[0] = 42.0f;
        if (p[0] != 42.0f)
        {
            std::cerr << "[test_pinned_device] data() 書き込み/読み出しに失敗\n";
            return false;
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "[test_pinned_device] 例外が発生: " << e.what() << "\n";
        return false;
    }

    std::cout << "[test_pinned_device] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// テスト5: CUDA Tensor で data() が std::logic_error を投げる
// ----------------------------------------------------------------
static bool test_cuda_data_throws()
{
    Tensor t({4, 4}, Device::CUDA);
    bool threw = false;

    try
    {
        (void)t.data();
    }
    catch (const std::logic_error&)
    {
        threw = true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[test_cuda_data_throws] 予期しない例外: " << e.what() << "\n";
        return false;
    }

    if (!threw)
    {
        std::cerr << "[test_cuda_data_throws] std::logic_error が投げられなかった\n";
        return false;
    }

    std::cout << "[test_cuda_data_throws] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// テスト6: NPU Tensor で data() が std::logic_error を投げる
// ----------------------------------------------------------------
static bool test_npu_data_throws()
{
    Tensor t({4, 4}, Device::NPU);
    bool threw = false;

    try
    {
        (void)t.data();
    }
    catch (const std::logic_error&)
    {
        threw = true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[test_npu_data_throws] 予期しない例外: " << e.what() << "\n";
        return false;
    }

    if (!threw)
    {
        std::cerr << "[test_npu_data_throws] std::logic_error が投げられなかった\n";
        return false;
    }

    std::cout << "[test_npu_data_throws] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// テスト7: set_data_ptr() 後に data_ptr() が同じポインタを返す
// ----------------------------------------------------------------
static bool test_set_data_ptr()
{
    Tensor t({2, 2}, Device::CUDA);

    // ダミーポインタで set_data_ptr をテスト
    float dummy[4] = {0};
    t.set_data_ptr(dummy);

    void* got = t.data_ptr();
    if (got != static_cast<void*>(dummy))
    {
        std::cerr << "[test_set_data_ptr] data_ptr() が設定したポインタと一致しない\n";
        return false;
    }

    std::cout << "[test_set_data_ptr] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// テスト8: CUDA Tensor で set_data_ptr() 未呼び出し → data_ptr() が logic_error
// ----------------------------------------------------------------
static bool test_data_ptr_cuda_no_ptr_throws()
{
    Tensor t({4, 4}, Device::CUDA);
    bool threw = false;

    try
    {
        (void)t.data_ptr();
    }
    catch (const std::logic_error&)
    {
        threw = true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[test_data_ptr_cuda_no_ptr_throws] 予期しない例外: "
                  << e.what() << "\n";
        return false;
    }

    if (!threw)
    {
        std::cerr << "[test_data_ptr_cuda_no_ptr_throws] "
                     "std::logic_error が投げられなかった\n";
        return false;
    }

    std::cout << "[test_data_ptr_cuda_no_ptr_throws] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// テスト9: CPU Tensor で data_ptr() が cpu_buf_.data() と一致する
// static_cast<float*>(t.data_ptr()) == t.data() で確認
// ----------------------------------------------------------------
static bool test_data_ptr_cpu_returns_buf()
{
    Tensor t({3, 3}, Device::CPU);

    float* via_data     = t.data();
    float* via_data_ptr = static_cast<float*>(t.data_ptr());

    if (via_data_ptr != via_data)
    {
        std::cerr << "[test_data_ptr_cpu_returns_buf] "
                     "data_ptr() と data() が異なるポインタを返した\n";
        return false;
    }

    std::cout << "[test_data_ptr_cpu_returns_buf] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// ベンチ1: Tensor({1024,1024}, CPU) を 10000 回生成して ms を報告
// ----------------------------------------------------------------
static bool bench_tensor_create()
{
    static constexpr int kN = 10000;

    auto t0 = Clock::now();
    for (int i = 0; i < kN; ++i)
    {
        Tensor t({1024, 1024}, Device::CPU);
        // 最適化で除去されないようポインタを利用
        volatile float* p = t.data();
        (void)p;
    }
    double elapsed_ms = MsDbl(Clock::now() - t0).count();

    std::cout << "[bench_tensor_create]\n";
    std::cout << "  BENCH: " << kN << " x Tensor({1024,1024},CPU) total = "
              << elapsed_ms << " ms"
              << "  per create = " << elapsed_ms / kN * 1000.0 << " us\n";
    return true;
}

// ----------------------------------------------------------------
// ベンチ2: CPU Tensor 読み書きスループット (1M 要素 × 100 回)
// ----------------------------------------------------------------
static bool bench_data_access()
{
    static constexpr int    kN  = 100;
    static constexpr size_t kSz = 1024 * 1024;

    Tensor t({kSz}, Device::CPU);
    float* p = t.data();

    auto t0 = Clock::now();
    for (int iter = 0; iter < kN; ++iter)
    {
        for (size_t i = 0; i < kSz; ++i)
            p[i] += 1.0f;
    }
    double elapsed_ms = MsDbl(Clock::now() - t0).count();
    // read + write で 2 倍のデータ量
    double gb   = static_cast<double>(kN) * kSz * sizeof(float) * 2.0 / 1e9;
    double gbps = gb / (elapsed_ms / 1000.0);

    std::cout << "[bench_data_access]\n";
    std::cout << "  BENCH: " << kN << " x " << kSz / 1024 << "K-float RW"
              << "  total = " << elapsed_ms << " ms"
              << "  throughput = " << gbps << " GB/s\n";
    return true;
}

} // namespace dollama

int main()
{
    bool ok = true;

    ok = dollama::test_default_ctor()                  && ok;
    ok = dollama::test_shape_cpu()                     && ok;
    ok = dollama::test_data_zero_init()                && ok;
    ok = dollama::test_pinned_device()                 && ok;
    ok = dollama::test_cuda_data_throws()              && ok;
    ok = dollama::test_npu_data_throws()               && ok;
    ok = dollama::test_set_data_ptr()                  && ok;
    ok = dollama::test_data_ptr_cuda_no_ptr_throws()   && ok;
    ok = dollama::test_data_ptr_cpu_returns_buf()      && ok;
    ok = dollama::bench_tensor_create()                && ok;
    ok = dollama::bench_data_access()                  && ok;

    if (!ok)
    {
        std::cerr << "[test_tensor] FAILED\n";
        return 1;
    }

    std::cout << "[test_tensor] ALL PASSED\n";
    return 0;
}

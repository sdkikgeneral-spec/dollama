// SPSCQueue 単体テスト + 速度計測
// 失敗時は std::cerr に出力して return 1。
// 全テスト通過時は "[test_queue] ALL PASSED" を出力して return 0。
// 速度計測結果は BENCH: プレフィックスで stdout に出力する。
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>

#include "core/queue.hpp"

namespace dollama {

using Clock    = std::chrono::steady_clock;
using NsDouble = std::chrono::duration<double, std::nano>;
using MsCount  = std::chrono::milliseconds;

// ----------------------------------------------------------------
// テスト1: シングルスレッド — push/pop FIFO 動作確認
// ----------------------------------------------------------------
static bool test_single_fifo()
{
    SPSCQueue<int, 4> q;

    // 1〜4 を push して全件成功することを確認
    for (int i = 1; i <= 4; ++i)
    {
        if (!q.push(i))
        {
            std::cerr << "[test_single_fifo] push(" << i << ") が予期せず失敗\n";
            return false;
        }
    }

    // 満杯時の push は false を返す
    if (q.push(5))
    {
        std::cerr << "[test_single_fifo] 満杯時の push(5) が true を返した (期待: false)\n";
        return false;
    }

    // pop で 1〜4 が FIFO 順に取れることを確認
    for (int expected = 1; expected <= 4; ++expected)
    {
        int val = 0;
        if (!q.pop(val))
        {
            std::cerr << "[test_single_fifo] pop が予期せず失敗 (expected=" << expected << ")\n";
            return false;
        }
        if (val != expected)
        {
            std::cerr << "[test_single_fifo] pop の値が不正 (got=" << val
                      << ", expected=" << expected << ")\n";
            return false;
        }
    }

    // 空のキューで pop は false を返す
    {
        int dummy = 0;
        if (q.pop(dummy))
        {
            std::cerr << "[test_single_fifo] 空のキューで pop が true を返した\n";
            return false;
        }
    }

    std::cout << "[test_single_fifo] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// テスト2: シングルスレッド — push_wait タイムアウト確認
// 速度計測: タイムアウト精度 (ms)
// ----------------------------------------------------------------
static bool test_push_wait_timeout()
{
    SPSCQueue<int, 4> q;

    for (int i = 1; i <= 4; ++i)
        q.push(i);

    // 満杯状態で push_wait(99, 5ms) が false を返し、約 5ms で戻ることを確認
    auto t0     = Clock::now();
    bool result = q.push_wait(99, MsCount{5});
    auto elapsed_ms = std::chrono::duration_cast<MsCount>(Clock::now() - t0).count();

    if (result)
    {
        std::cerr << "[test_push_wait_timeout] 満杯時に push_wait が true を返した (期待: false)\n";
        return false;
    }
    if (elapsed_ms < 4 || elapsed_ms > 50)
    {
        std::cerr << "[test_push_wait_timeout] タイムアウト時間が範囲外 (elapsed="
                  << elapsed_ms << "ms, expected ~5ms)\n";
        return false;
    }

    std::cout << "[test_push_wait_timeout] PASSED\n";
    std::cout << "  BENCH: push_wait timeout elapsed = " << elapsed_ms << " ms"
              << "  (target 5ms, overhead " << elapsed_ms - 5 << " ms)\n";
    return true;
}

// ----------------------------------------------------------------
// テスト3: マルチスレッド — producer/consumer 2 スレッド
// 速度計測: スループット (Mops/s)
// ----------------------------------------------------------------
static bool test_multithread_producer_consumer()
{
    static constexpr int kN = 10000;
    static constexpr MsCount kTimeout{5000};

    SPSCQueue<int, 16> q;
    std::vector<int> received;
    received.reserve(kN);

    bool producer_ok = true;
    bool consumer_ok = true;

    auto t0 = Clock::now();

    std::thread producer([&]()
    {
        for (int i = 0; i < kN; ++i)
        {
            if (!q.push_wait(i, kTimeout))
            {
                std::cerr << "[test_multithread] producer: push_wait タイムアウト (i=" << i << ")\n";
                producer_ok = false;
                return;
            }
        }
    });

    std::thread consumer([&]()
    {
        for (int i = 0; i < kN; ++i)
        {
            int val = 0;
            if (!q.pop_wait(val, kTimeout))
            {
                std::cerr << "[test_multithread] consumer: pop_wait タイムアウト (i=" << i << ")\n";
                consumer_ok = false;
                return;
            }
            received.push_back(val);
        }
    });

    producer.join();
    consumer.join();

    double elapsed_us = std::chrono::duration<double, std::micro>(Clock::now() - t0).count();

    if (!producer_ok || !consumer_ok)
        return false;

    if (static_cast<int>(received.size()) != kN)
    {
        std::cerr << "[test_multithread] 受信件数が不正 (got=" << received.size()
                  << ", expected=" << kN << ")\n";
        return false;
    }
    for (int i = 0; i < kN; ++i)
    {
        if (received[i] != i)
        {
            std::cerr << "[test_multithread] 値または順序が不正 (received[" << i
                      << "]=" << received[i] << ", expected=" << i << ")\n";
            return false;
        }
    }

    double throughput_mops = kN / elapsed_us;  // items/us = Mops/s
    std::cout << "[test_multithread_producer_consumer] PASSED\n";
    std::cout << "  BENCH: " << kN << " items in " << static_cast<long long>(elapsed_us) << " us"
              << "  throughput = " << throughput_mops << " Mops/s\n";
    return true;
}

// ----------------------------------------------------------------
// 速度計測専用: シングルスレッド push/pop レイテンシ
// SPSCQueue<int, 64> で 1M 回 push→pop のラウンドトリップを計測する。
// 正確性チェックは行わず、ns/op を報告する。
// ----------------------------------------------------------------
static bool bench_single_thread_latency()
{
    static constexpr int kN = 1'000'000;
    SPSCQueue<int, 64> q;

    // ウォームアップ (キャッシュを温める)
    for (int i = 0; i < 64; ++i) q.push(i);
    for (int i = 0; i < 64; ++i) { int v; q.pop(v); }

    // 計測: push 1件 → pop 1件 を kN 回繰り返す
    auto t0 = Clock::now();
    for (int i = 0; i < kN; ++i)
    {
        q.push(i);
        int v;
        q.pop(v);
    }
    double elapsed_ns = NsDouble(Clock::now() - t0).count();

    double ns_per_op = elapsed_ns / kN;
    std::cout << "[bench_single_thread_latency]\n";
    std::cout << "  BENCH: " << kN / 1000 << "K push+pop  total = "
              << static_cast<long long>(elapsed_ns / 1000) << " us"
              << "  per roundtrip = " << ns_per_op << " ns/op\n";
    return true;
}

} // namespace dollama

int main()
{
    bool ok = true;

    ok = dollama::test_single_fifo()                    && ok;
    ok = dollama::test_push_wait_timeout()              && ok;
    ok = dollama::test_multithread_producer_consumer()  && ok;
    ok = dollama::bench_single_thread_latency()         && ok;

    if (!ok)
    {
        std::cerr << "[test_queue] FAILED\n";
        return 1;
    }

    std::cout << "[test_queue] ALL PASSED\n";
    return 0;
}

// SPSC (Single Producer / Single Consumer) ロックフリーキュー
// スレッド間のゼロコピー受け渡しに使用。
// Capacity は必ず 2の冪数を指定すること (static_assert で強制)。
// 計測値: CPU pinned memory 経由で 3.4% オーバーヘッド (probe2)
#pragma once
#include <atomic>
#include <array>
#include <chrono>
#include <thread>
#include <cstddef>

namespace dollama {

// SPSC ロックフリーキュー
// プロデューサスレッドとコンシューマスレッドがそれぞれ 1 本ずつの場合のみ安全。
template<typename T, size_t Capacity>
class SPSCQueue
{
    // Capacity は 2の冪数でなければならない
    static_assert(Capacity >= 2, "Capacity は 2 以上でなければなりません");
    static_assert((Capacity & (Capacity - 1)) == 0,
                  "Capacity は 2の冪数でなければなりません");

    // インデックスマスク (& kMask で剰余を高速化)
    static constexpr size_t kMask = Capacity - 1;

public:
    SPSCQueue()                              = default;
    ~SPSCQueue()                             = default;

    // コピー・ムーブ禁止 (キャッシュライン分離構造を維持するため)
    SPSCQueue(const SPSCQueue&)              = delete;
    SPSCQueue& operator=(const SPSCQueue&)  = delete;
    SPSCQueue(SPSCQueue&&)                  = delete;
    SPSCQueue& operator=(SPSCQueue&&)       = delete;

    // ----------------------------------------------------------------
    // push (コピー版) — ノンブロッキング。満杯なら false を返す
    // ----------------------------------------------------------------
    bool push(const T& item)
    {
        // tail は自スレッド(プロデューサ)のみが更新するので relaxed で読む
        size_t tail = tail_.load(std::memory_order_relaxed);
        // head はコンシューマが更新するので acquire で観測する
        size_t head = head_.load(std::memory_order_acquire);
        if (tail - head >= Capacity)
        {
            // バッファ満杯
            return false;
        }
        buf_[tail & kMask] = item;
        // コンシューマに可視にするため release で格納
        tail_.store(tail + 1, std::memory_order_release);
        return true;
    }

    // ----------------------------------------------------------------
    // push (ムーブ版) — ノンブロッキング。満杯なら false を返す
    // ----------------------------------------------------------------
    bool push(T&& item)
    {
        size_t tail = tail_.load(std::memory_order_relaxed);
        size_t head = head_.load(std::memory_order_acquire);
        if (tail - head >= Capacity)
        {
            return false;
        }
        buf_[tail & kMask] = std::move(item);
        tail_.store(tail + 1, std::memory_order_release);
        return true;
    }

    // ----------------------------------------------------------------
    // pop — ノンブロッキング。空なら false を返す
    // ----------------------------------------------------------------
    bool pop(T& item)
    {
        // head は自スレッド(コンシューマ)のみが更新するので relaxed で読む
        size_t head = head_.load(std::memory_order_relaxed);
        // tail はプロデューサが更新するので acquire で観測する
        size_t tail = tail_.load(std::memory_order_acquire);
        if (tail == head)
        {
            // バッファ空
            return false;
        }
        item = std::move(buf_[head & kMask]);
        // プロデューサに可視にするため release で格納
        head_.store(head + 1, std::memory_order_release);
        return true;
    }

    // ----------------------------------------------------------------
    // push_wait — 100µs ポーリング。timeout 内に push できなければ false
    // ----------------------------------------------------------------
    bool push_wait(const T& item, std::chrono::milliseconds timeout)
    {
        auto deadline = std::chrono::steady_clock::now() + timeout;
        while (true)
        {
            if (push(item))
            {
                return true;
            }
            if (std::chrono::steady_clock::now() >= deadline)
            {
                return false;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }

    // ----------------------------------------------------------------
    // pop_wait — 100µs ポーリング。timeout 内に pop できなければ false
    // ----------------------------------------------------------------
    bool pop_wait(T& item, std::chrono::milliseconds timeout)
    {
        auto deadline = std::chrono::steady_clock::now() + timeout;
        while (true)
        {
            if (pop(item))
            {
                return true;
            }
            if (std::chrono::steady_clock::now() >= deadline)
            {
                return false;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }

    // ----------------------------------------------------------------
    // size / empty / full — 近似値 (スナップショット)
    // ----------------------------------------------------------------
    size_t size() const noexcept
    {
        size_t head = head_.load(std::memory_order_relaxed);
        size_t tail = tail_.load(std::memory_order_relaxed);
        return tail - head;
    }

    bool empty() const noexcept
    {
        return size() == 0;
    }

    bool full() const noexcept
    {
        return size() >= Capacity;
    }

private:
    // false sharing 防止のためキャッシュライン (64 バイト) で分離
    alignas(64) std::atomic<size_t> head_{0};  // コンシューマスレッドのみ更新
    alignas(64) std::atomic<size_t> tail_{0};  // プロデューサスレッドのみ更新
    std::array<T, Capacity> buf_{};
};

} // namespace dollama

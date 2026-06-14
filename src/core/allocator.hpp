#pragma once
#include <cstddef>
#include <stdexcept>
#include <new>

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif

namespace dollama {

// ------------------------------------------------------------
// CPU Pinned Memory (CUDA DMA 転送用・3.4% オーバーヘッドで済む)
// 計測値: 100MB → 3.46ms / 30.3 GB/s (probe2)
// ------------------------------------------------------------
struct PinnedAllocator
{
    static void* alloc(size_t bytes)
    {
#ifdef HAVE_CUDA
        void* ptr = nullptr;
        if (cudaMallocHost(&ptr, bytes) != cudaSuccess)
        {
            throw std::runtime_error("cudaMallocHost failed");
        }
        return ptr;
#else
        return ::operator new(bytes);
#endif
    }

    static void free(void* ptr) noexcept
    {
#ifdef HAVE_CUDA
        cudaFreeHost(ptr);
#else
        ::operator delete(ptr);
#endif
    }
};

// ------------------------------------------------------------
// CUDA VRAM (RTX5080 / Blackwell / sm_120)
// 計測値: system RAM → VRAM 12MB = 0.254ms / 49.6 GB/s (probe4)
// ------------------------------------------------------------
struct CudaAllocator
{
    static void* alloc(size_t bytes)
    {
#ifdef HAVE_CUDA
        void* ptr = nullptr;
        if (cudaMalloc(&ptr, bytes) != cudaSuccess)
        {
            throw std::runtime_error("cudaMalloc failed");
        }
        return ptr;
#else
        throw std::runtime_error("CUDA not compiled");
#endif
    }

    static void free(void* ptr) noexcept
    {
#ifdef HAVE_CUDA
        cudaFree(ptr);
#else
        (void)ptr;
#endif
    }
};

// ------------------------------------------------------------
// RAII ラッパー
// ------------------------------------------------------------
template<typename Alloc>
class UniqueBuffer
{
public:
    explicit UniqueBuffer(size_t bytes) : bytes_(bytes)
    {
        ptr_ = Alloc::alloc(bytes);
    }

    ~UniqueBuffer()
    {
        if (ptr_)
        {
            Alloc::free(ptr_);
        }
    }

    UniqueBuffer(const UniqueBuffer&)            = delete;
    UniqueBuffer& operator=(const UniqueBuffer&) = delete;

    // [Bug-5] ムーブコンストラクタで bytes_ をゼロリセットする
    UniqueBuffer(UniqueBuffer&& o) noexcept
        : ptr_(o.ptr_), bytes_(o.bytes_)
    {
        o.ptr_   = nullptr;
        o.bytes_ = 0;
    }

    // [Bug-6] ムーブ代入演算子を追加 (旧バッファを Alloc::free で解放)
    UniqueBuffer& operator=(UniqueBuffer&& o) noexcept
    {
        if (this == &o)
        {
            return *this;
        }
        if (ptr_)
        {
            Alloc::free(ptr_);
        }
        ptr_     = o.ptr_;
        bytes_   = o.bytes_;
        o.ptr_   = nullptr;
        o.bytes_ = 0;
        return *this;
    }

    void*  get()  const noexcept { return ptr_; }
    size_t size() const noexcept { return bytes_; }

private:
    void*  ptr_   = nullptr;
    size_t bytes_ = 0;
};

using PinnedBuffer = UniqueBuffer<PinnedAllocator>;
using CudaBuffer   = UniqueBuffer<CudaAllocator>;

} // namespace dollama

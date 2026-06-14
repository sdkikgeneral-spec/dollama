#pragma once
#include <vector>
#include <cstddef>
#include <numeric>
#include <stdexcept>

namespace dollama {

enum class Device
{
    CPU,     // 通常ヒープ
    PINNED,  // CUDA pinned memory (DMA 転送用)
    CUDA,    // VRAM (RTX5080)
    NPU,     // OpenVINO RemoteTensor (将来)
};

// 軽量テンソルクラス
// 現フェーズは CPU / PINNED のみ STL で管理。
// CUDA / NPU バッファは allocator.hpp で別途確保し、data_ptr で参照する。
class Tensor
{
public:
    Tensor() = default;
    explicit Tensor(std::vector<size_t> shape, Device dev = Device::CPU);

    // 要素数
    size_t numel() const noexcept;

    // 形状
    const std::vector<size_t>& shape() const noexcept
    {
        return shape_;
    }

    size_t ndim() const noexcept
    {
        return shape_.size();
    }

    size_t dim(size_t i) const
    {
        return shape_.at(i);
    }

    // CPU / PINNED データポインタ
    // [Bug-2] CUDA / NPU デバイスで呼ぶと logic_error を投げる
    float* data()
    {
        if (device_ != Device::CPU && device_ != Device::PINNED)
        {
            throw std::logic_error("data() は CPU/PINNED 専用");
        }
        return cpu_buf_.data();
    }

    const float* data() const
    {
        if (device_ != Device::CPU && device_ != Device::PINNED)
        {
            throw std::logic_error("data() は CPU/PINNED 専用");
        }
        return cpu_buf_.data();
    }

    // 外部バッファ (CUDA / NPU) を参照させる場合
    void set_data_ptr(void* ptr) noexcept
    {
        ext_ptr_ = ptr;
    }

    // [Bug-1] CUDA / NPU で set_data_ptr() 未呼び出しの場合 logic_error を投げる
    void* data_ptr()
    {
        if ((device_ == Device::CUDA || device_ == Device::NPU) && ext_ptr_ == nullptr)
        {
            throw std::logic_error(
                "data_ptr(): CUDA/NPU Tensor に set_data_ptr() が呼ばれていない");
        }
        return ext_ptr_ ? ext_ptr_ : cpu_buf_.data();
    }

    Device device() const noexcept
    {
        return device_;
    }

    size_t nbytes() const noexcept
    {
        return numel() * sizeof(float);
    }

private:
    std::vector<size_t> shape_;
    std::vector<float>  cpu_buf_;  // CPU / PINNED バッファ
    void*               ext_ptr_ = nullptr;
    Device              device_  = Device::CPU;
};

inline size_t Tensor::numel() const noexcept
{
    if (shape_.empty())
    {
        return 0;
    }
    return std::accumulate(shape_.begin(), shape_.end(),
                           size_t{1}, std::multiplies<size_t>{});
}

inline Tensor::Tensor(std::vector<size_t> shape, Device dev)
    : shape_(std::move(shape)), device_(dev)
{
    if (dev == Device::CPU || dev == Device::PINNED)
    {
        cpu_buf_.assign(numel(), 0.0f);
    }
    // CUDA / NPU は allocator.hpp で確保してから set_data_ptr() で渡す
}

} // namespace dollama

# Tensor クラス仕様 — dollama C++ 実装

## 設計方針

- STL + CUDA Runtime API のみ使用 (PyTorch / LibTorch 不使用)
- CPU / PINNED (CUDA pinned) / CUDA (VRAM) / NPU の4デバイスを統一インターフェースで扱う
- CUDA / NPU バッファの確保は `allocator.hpp` (UniqueBuffer) に委譲
- float32 固定 (量子化テンソルは別クラスで後述)

## ファイル

```
src/core/tensor.hpp      — Tensor クラス (ヘッダオンリー)
src/core/allocator.hpp   — PinnedAllocator / CudaAllocator / UniqueBuffer
```

## Device 列挙

```cpp
enum class Device
{
    CPU,     // 通常ヒープ (std::vector<float>)
    PINNED,  // CUDA pinned memory — cudaMallocHost / DMA 転送用
    CUDA,    // RTX5080 VRAM — cudaMalloc
    NPU,     // OpenVINO RemoteTensor (将来; 現在は set_data_ptr で渡す)
};
```

## Tensor クラス インターフェース

```cpp
namespace dollama {

class Tensor
{
public:
    Tensor() = default;
    explicit Tensor(std::vector<size_t> shape, Device dev = Device::CPU);

    // 形状
    const std::vector<size_t>& shape() const noexcept;
    size_t numel() const noexcept;   // 要素数
    size_t ndim()  const noexcept;
    size_t dim(size_t i) const;      // shape_[i] (bounds-checked)
    size_t nbytes() const noexcept;  // numel() * sizeof(float)
    Device device() const noexcept;

    // CPU / PINNED 専用ポインタ — CUDA/NPU で呼ぶと logic_error
    float*       data();
    const float* data() const;

    // 外部バッファ (CUDA / NPU) のセット・取得
    // CUDA/NPU Tensor で set_data_ptr() 前に data_ptr() を呼ぶと logic_error
    void  set_data_ptr(void* ptr) noexcept;
    void* data_ptr();
};

} // namespace dollama
```

## 所有権モデル

| device | バッファ所有者 | 確保方法 | 解放方法 |
|---|---|---|---|
| CPU | Tensor 内 `std::vector<float>` | コンストラクタで自動 | デストラクタで自動 |
| PINNED | Tensor 内 `std::vector<float>` | コンストラクタで自動 (将来: cudaMallocHost) | デストラクタで自動 |
| CUDA | 呼び出し側 `CudaBuffer` | `CudaAllocator::alloc` | `CudaBuffer` RAII |
| NPU | 呼び出し側 (OpenVINO) | `ov::Tensor` | OpenVINO が管理 |

CUDA / NPU の場合は外部で確保したバッファを `set_data_ptr()` で渡す。
Tensor はポインタを保持するだけで解放しない。

## 使用例

```cpp
// CPU テンソル (embedding 受け取り用)
dollama::Tensor emb({1, 77, 768}, dollama::Device::CPU);
float* ptr = emb.data();  // OK

// CUDA テンソル (VRAM 上の latent)
dollama::CudaBuffer vram_buf(1 * 4 * 128 * 128 * sizeof(float));
dollama::Tensor latent({1, 4, 128, 128}, dollama::Device::CUDA);
latent.set_data_ptr(vram_buf.get());
void* vram_ptr = latent.data_ptr();  // OK

// CUDA テンソルに data() を呼ぶと例外
latent.data();  // throws std::logic_error
```

## 将来拡張 (現フェーズ対象外)

- `dtype` フィールド (float16 / int8 / int4 等)
- `Tensor::to(Device)` — デバイス間コピー
- スライス / reshape (view)
- `QuantizedTensor` — BitNet b1.58 用 ternary 重み表現

## allocator.hpp — UniqueBuffer

```cpp
template<typename Alloc>
class UniqueBuffer
{
public:
    explicit UniqueBuffer(size_t bytes);
    UniqueBuffer(UniqueBuffer&& o) noexcept;   // bytes_ もゼロリセット
    UniqueBuffer& operator=(UniqueBuffer&& o) noexcept;  // 旧ptr を free

    void*  get()  const noexcept;  // nullptr なら未確保 or move 済み
    size_t size() const noexcept;  // move 後は 0
};

using PinnedBuffer = UniqueBuffer<PinnedAllocator>;  // cudaMallocHost
using CudaBuffer   = UniqueBuffer<CudaAllocator>;    // cudaMalloc
```

**注意**: move 後のオブジェクトは `get()==nullptr` かつ `size()==0`。

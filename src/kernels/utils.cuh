// CUDA 共通ユーティリティ — エラーチェック・カーネル起動基盤
// 対象: RTX5080 (Blackwell / sm_120) / CUDA Runtime API のみ
#pragma once

#include <cuda_runtime.h>
#include <cassert>
#include <stdexcept>
#include <string>

namespace dollama
{

// ----------------------------------------------------------------
// CUDA エラーチェックマクロ
// ----------------------------------------------------------------
// CUDA Runtime API 呼び出しを包み、失敗時に file/line 付きで例外を投げる。
#define CUDA_CHECK(call)                                              \
    do                                                                \
    {                                                                 \
        cudaError_t err = (call);                                     \
        if (err != cudaSuccess)                                       \
        {                                                             \
            throw std::runtime_error(                                 \
                std::string("CUDA error: ") + cudaGetErrorString(err) \
                + " at " __FILE__ ":" + std::to_string(__LINE__));    \
        }                                                             \
    } while (0)

// カーネル起動直後の検査。
// 常に cudaGetLastError() で起動エラーを拾い、デバッグビルド (NDEBUG 未定義) では
// cudaDeviceSynchronize() で実行時エラーも同期的に検出する。
#ifdef NDEBUG
#define CUDA_CHECK_KERNEL()             \
    do                                  \
    {                                   \
        CUDA_CHECK(cudaGetLastError()); \
    } while (0)
#else
#define CUDA_CHECK_KERNEL()                  \
    do                                       \
    {                                        \
        CUDA_CHECK(cudaGetLastError());      \
        CUDA_CHECK(cudaDeviceSynchronize()); \
    } while (0)
#endif

// ----------------------------------------------------------------
// 1D グリッド起動用ヘルパー: ceil(a / b)
// 事前条件: b > 0 (除数 0 は整数 0 除算 = UB)。a >= 0 を想定。
// host/device 両用なので例外は使えず assert でトラップ (release では消える)。
// ----------------------------------------------------------------
__host__ __device__ inline int ceil_div(int a, int b)
{
    assert(b > 0);
    return (a + b - 1) / b;
}

// ----------------------------------------------------------------
// vector_add — 疎通用カーネル: out[i] = a[i] + b[i]
// ----------------------------------------------------------------
__global__ void vector_add(const float* a, const float* b, float* out, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        out[idx] = a[idx] + b[idx];
    }
}

// ホストラッパー: デバイスポインタ (d_a, d_b, d_out) を受け取り vector_add を起動する。
// 起動後に CUDA_CHECK_KERNEL() でエラー検査を行う。
inline void launch_vector_add(const float* d_a, const float* d_b, float* d_out, int n)
{
    const int threads = 256;
    const int blocks  = ceil_div(n, threads);
    vector_add<<<blocks, threads>>>(d_a, d_b, d_out, n);
    CUDA_CHECK_KERNEL();
}

} // namespace dollama

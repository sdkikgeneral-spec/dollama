// dense FP16 GEMM — ホストラッパー宣言 (Phase 2 マイルストーン 2-2-1)
// 対象: RTX5080 (Blackwell / sm_120) / CUDA Runtime API のみ
#pragma once

#include <cuda_fp16.h>

namespace dollama
{

// ----------------------------------------------------------------
// dense FP16 GEMM: C = alpha*(op(A) op(B)) + beta*C
// ----------------------------------------------------------------
// レイアウトは row-major 固定。呼び出し側は常に row-major で考えること。
//
//   transA=false, transB=false:
//     A: [M, K] row-major (A[i*lda + k], lda=K)
//     B: [K, N] row-major (B[k*ldb + j], ldb=N)
//     C: [M, N] row-major (C[i*ldc + j], ldc=N)
//     C[i*N + j] = alpha * Σ_k A[i*K + k] * B[k*N + j] + beta * C[i*N + j]
//
//   transB=true (SDXL Linear `x @ W^T` の主役):
//     A: [M, K] row-major (lda=K)
//     B: [N, K] row-major (B[j*ldb + k], ldb=K) ← 論理的には B^T が [K, N]
//     C: [M, N] row-major (ldc=N)
//     C[i*N + j] = alpha * Σ_k A[i*K + k] * B[j*K + k] + beta * C[i*N + j]
//
//   transA=true: 引数だけ用意 (Attention/SDXL で将来必要)。現状は false 経路のみ実装。
//
// dtype: A, B, C は __half (FP16)。accumulator は内部で必ず FP32。
//        acc = Σ __half2float(a) * __half2float(b)
//        C   = __float2half(alpha*acc + beta*__half2float(C_in))
//
// 引数:
//   d_A, d_B  : 入力デバイスポインタ (FP16)
//   d_C       : 入出力デバイスポインタ (FP16)。beta!=0 のとき読み込まれる。
//   M, N, K   : 論理次元 (row-major、transpose 適用後の op(A)=[M,K], op(B)=[K,N])
//   alpha, beta : スカラ (FP32)
//   transA, transB : 転置フラグ
//
// すべてデバイスポインタを受け取る (launch_vector_add と同規約、内部で cudaMalloc しない)。
// 起動後に CUDA_CHECK_KERNEL() でエラー検査を行う。
void launch_gemm_fp16(const __half* d_A,
                      const __half* d_B,
                      __half*       d_C,
                      int           M,
                      int           N,
                      int           K,
                      float         alpha,
                      float         beta,
                      bool          transA,
                      bool          transB);

} // namespace dollama

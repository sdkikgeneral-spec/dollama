// dense FP16 GEMM — ホストラッパー宣言 (Phase 2 マイルストーン 2-2-1)
// 対象: RTX5080 (Blackwell / sm_120) / CUDA Runtime API のみ
#pragma once

#include <cuda_fp16.h>

#include <cstdint>

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

// ----------------------------------------------------------------
// FP32 dense GEMM (S3-E): C[M,N] = alpha*op(A) @ op(B) + beta*C。
// すべて FP32 (CUDA_R_32F) デバイスポインタ。VAE up2/up3 の im2col + GEMM 用。
// cuBLAS GemmEx に委譲する (既定 TF32 Tensor Core / env DOLLAMA_VAE_GEMM=fp32 で純 FP32)。
// レイアウト・transA/transB の意味は launch_gemm_fp16 と同一。transA=true は非対応。
// 起動後に CUDA_CHECK_KERNEL() は不要 (cuBLAS は status で検査済み)。
void launch_gemm_f32(const float* d_A,
                     const float* d_B,
                     float*       d_C,
                     int          M,
                     int          N,
                     int          K,
                     float        alpha,
                     float        beta,
                     bool         transA,
                     bool         transB);

// ----------------------------------------------------------------
// batched FP16 GEMM (G-10k T3)。同形状の GEMM を batch 個まとめて 1 発で発行する。
//
//   n = 0 .. batch-1 について
//     C_n = alpha * (A_n op(B_n)) + beta * C_n
//   ここで
//     A_n = d_A + n*stride_a   (論理 [M, K] row-major、lda = K)
//     B_n = d_B + n*stride_b   (transB=false: [K, N] ldb=N / transB=true: [N, K] ldb=K)
//     C_n = d_C + n*stride_c   (論理 [M, N] row-major、ldc = N)
//
// レイアウト規約は launch_gemm_fp16 と完全に同一 (row-major 固定・FP32 蓄積)。
// transA は本経路では非対応 (記述子に持たない = 常に false)。
//
// stride は「要素数」単位 (バイトではない)。stride_a = 0 は合法で、
// 全 item が同一の A を共有する conv2d の重み共有形態を表す。
//
// 負 stride について (未発火経路):
//   gemm.cu の batched_span (gemm.cu:611-612) は stride が負でも union 区間を畳めるよう
//   min/max を取って書いてあるが、この負 stride 側 (off_lo = last) は
//   【実走で確認した経路ではない】。T3 のケース構成に負 stride は 1 件も無く、
//   docs/logs/g10k-t3/t3_default_run.log:6-17 の [H7] 12 件も含めすべて非負である。
//   T4 の conv2d 側からも非負 stride しか渡さない契約 (docs/g10k-plan.md §6 T4 行) のため、
//   この分岐は T4 完了後も未発火のまま残る。
//
// 数値: 既存 launch_gemm_fp16 と同じく FP16 入力 / FP32 蓄積 / FP16 出力。
//       cuBLAS の compute type も既存 gemm_cublas と同じ CUBLAS_COMPUTE_32F で固定する
//       (TF32 や COMPUTE_16F にすると蓄積の丸め単位が変わりゲートの前提が崩れる)。
// ----------------------------------------------------------------
struct GemmBatchedDesc
{
    int       batch    = 1;     // item 数 (>= 1)
    int       M        = 0;     // 出力行数
    int       N        = 0;     // 出力列数
    int       K        = 0;     // 縮約長
    long long stride_a = 0;     // A の item 間 stride (要素数)。0 = 全 item 共有 (重み共有)
    long long stride_b = 0;     // B の item 間 stride (要素数)
    long long stride_c = 0;     // C の item 間 stride (要素数)
    float     alpha    = 1.0f;
    float     beta     = 0.0f;  // != 0 のとき C を読む (= C は入出力)
    bool      trans_b  = false;
};

// 記述子と実ポインタの検査結果。Ok 以外はラッパが stderr へ理由を出して abort する。
enum class GemmBatchedValidation
{
    Ok = 0,
    BadBatch,     // batch < 1
    BadDims,      // M / N / K のいずれかが <= 0
    NullPointer,  // d_A / d_B / d_C のいずれかが null
    OverlapC,     // C の item 同士が重なっている (|stride_c| < M*N、または stride_c==0 かつ batch>1)
    OverlapCA,    // C の占有区間が A の占有区間と重なっている
    OverlapCB,    // C の占有区間が B の占有区間と重なっている
};

// ----------------------------------------------------------------
// 記述子の純関数 validator (副作用なし・GPU 不要・host から単体で呼べる)。
//
// 検査内容:
//   - batch >= 1 / M, N, K > 0 / ポインタ非 null
//   - C の item 範囲が互いに素であること (一様 stride なので閉形式で判定できる)
//   - C の全 item が占める区間が、A の全 item 区間・B の全 item 区間の
//     いずれとも重ならないこと (beta != 0 で C を「読む」ため read/write を問わず禁止)
//
// 明示的に許可するもの:
//   - A の stride 0 (conv2d の重み共有形態)
//   - A と B が互いに重なること (禁止するのは C との重なりのみ)
//
// 区間は「union 区間」(最初の item の先頭 〜 最後の item の末尾) で見る保守的判定。
// ----------------------------------------------------------------
GemmBatchedValidation gemm_batched_validate(const GemmBatchedDesc& desc,
                                            const __half*          d_A,
                                            const __half*          d_B,
                                            const __half*          d_C);

// validator の結果を人間可読な英語 1 語へ (.cu の文字列リテラルは ASCII 必須)。
const char* gemm_batched_validation_str(GemmBatchedValidation v);

// ----------------------------------------------------------------
// 分岐カウンタ (G-10k T3 / [H6])。どの枝を実際に通ったかを test から差分で読むための計器。
// DOLLAMA_PROFILE には依存しない (常時生存)。
// 前提: 既存 GEMM 経路と同じく単一 stream・単一スレッドから呼ばれること
//       (プロセス global の非 atomic カウンタなので、複数スレッドから叩くと値が壊れる)。
// ----------------------------------------------------------------
struct GemmBatchedStats
{
    uint64_t wrapper_calls        = 0;  // launch_gemm_fp16_batched の呼び出し回数 (累積)
    uint64_t cublas_batched_calls = 0;  // cuBLAS strided batched を実際に発行した回数 (累積)
    uint64_t fallback_loops       = 0;  // フォールバック直列ループに入った回数 (累積)
    uint64_t fallback_items       = 0;  // そのループが回した item 数の合計 (累積)
};

// 現在値をコピーで返す。
GemmBatchedStats gemm_batched_stats();

// 全カウンタを 0 に戻す。
void gemm_batched_stats_reset();

// batched ラッパ本体。desc を先頭で必ず validate し、違反なら stderr へ出して abort する。
// cuBLAS 不適格形状 (既存 use_cublas が false になる形状) では、
// 既存 launch_gemm_fp16 を per-item にそのまま呼ぶ直列ループへフォールバックする。
void launch_gemm_fp16_batched(const __half*          d_A,
                              const __half*          d_B,
                              __half*                d_C,
                              const GemmBatchedDesc& desc);

} // namespace dollama

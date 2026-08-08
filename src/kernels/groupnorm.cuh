// GroupNorm — ホストラッパー宣言 (Phase 2 マイルストーン 2-2-3)
// 対象: RTX5080 (Blackwell / sm_120) / CUDA Runtime API のみ
#pragma once

#include <cuda_fp16.h>

namespace dollama
{

// ----------------------------------------------------------------
// GroupNorm: チャネルをグループに分割しグループ単位で正規化 → affine
// ----------------------------------------------------------------
// レイアウトは row-major [N, C, H, W] 固定。呼び出し側は常にこのレイアウトで考えること。
//   要素アドレス: in[((n*C + c)*H + h)*W + w]
//
// グループ割り当て:
//   チャネル数 C をグループ数 num_groups で等分する。1 グループのチャネル数は
//   cpg = C / num_groups。チャネル c の属するグループは g = c / cpg。
//   正規化はサンプル n・グループ g 単位、すなわち [cpg, H, W] = cpg*H*W 要素の
//   集合で mean / var を取る (PyTorch nn.GroupNorm と同一仕様)。
//   前提条件: C % num_groups == 0 (内部で assert)。
//
// affine (PyTorch 仕様):
//   gamma[C], beta[C] はチャネルごと (グループごとではない)。
//   out = (x - mean_g) / sqrt(var_g + eps) * gamma[c] + beta[c]
//
// dtype:
//   入出力は __half (FP16)。ただし内部計算は必ず FP32。
//   部分和 (sum, sum-sq) は float で蓄積し、var = E[x^2] - E[x]^2、
//   var = max(var, 0.0f) でクランプする (浮動小数の桁落ちで負になり得るため)。
//
// in-place 安全:
//   d_in == d_out を許容する。各ブロックは担当グループの全要素を読み切って
//   mean/var を確定してから書き戻すため、入出力が同一バッファでもレースは発生しない。
//
// 起動:
//   1 グループ = 1 ブロックのリダクション戦略。グリッド = N * num_groups ブロック。
//   ブロック内でグリッドストライドして部分和を集め、shared memory + warp shuffle の
//   ツリーリダクションで mean/var を求め、同ブロックが正規化 + affine を書き戻す。
//   すべてデバイスポインタを受け取る (launch_gemm_fp16 / launch_silu と同規約、
//   内部で cudaMalloc しない)。起動後に CUDA_CHECK_KERNEL() でエラー検査を行う。
//
// 引数:
//   d_in    : 入力デバイスポインタ (FP16, [N,C,H,W])
//   d_gamma : affine scale (FP16, [C])。チャネルごと。
//   d_beta  : affine bias  (FP16, [C])。チャネルごと。
//   d_out   : 出力デバイスポインタ (FP16, [N,C,H,W])。d_in と同一でも可。
//   N,C,H,W : 論理次元
//   num_groups : グループ数 (SDXL 標準 32)。C % num_groups == 0 必須。
//   eps     : 分散の数値安定化項 (1e-6 級)
void launch_group_norm(const __half* d_in,
                       const __half* d_gamma,
                       const __half* d_beta,
                       __half*       d_out,
                       int           N,
                       int           C,
                       int           H,
                       int           W,
                       int           num_groups,
                       float         eps);

// ----------------------------------------------------------------
// multi-block GroupNorm (G-4k S1a・FAST epilogue 経路専用)
// ----------------------------------------------------------------
// 数学・引数仕様は launch_group_norm と同一。1 グループを複数ブロックで分担する
// 「2 段の決定的集約」(partial sum → finalize → normalize の 3 カーネル) で
// SM 占有率と実効帯域を引き上げる (1-block 版は grid=N*num_groups=32 で 73 GB/s)。
//   - atomic 不使用。grid/block 形状とリダクション木は入力形状のみで決まる固定順
//     なので、同一入力に対して run-to-run 完全ビット一致 (非交渉ゲート)。
//   - 蓄積順が 1-block 版と異なるため出力は 1-block 版と FP16 で ~1 ULP 級ずれる。
//     default (epilogue=off) 経路では使わないこと (byte-for-byte 無改変の原則)。
//   - 中間バッファ (~65KB 上界) は TU 内 grow-only 静的常駐。step ループ内の
//     cudaMalloc/cudaFree は発生しない (G-8k と整合)。単一スレッド前提。
//   - d_in == d_out の in-place 可 (stats 確定後の normalize は純 map)。
void launch_group_norm_mb(const __half* d_in,
                          const __half* d_gamma,
                          const __half* d_beta,
                          __half*       d_out,
                          int           N,
                          int           C,
                          int           H,
                          int           W,
                          int           num_groups,
                          float         eps);

// ----------------------------------------------------------------
// multi-block GroupNorm + SiLU 融合 (G-4k S1b・FAST epilogue 経路専用)
// ----------------------------------------------------------------
// 引数仕様は launch_group_norm_mb と完全同一。partial / finalize カーネルは
// mb 版をそのまま共用し、normalize カーネルのみ末尾 map に SiLU を畳んだ 1 パス版。
//   - bit-exact パリティ: normalize は GN 結果を一旦 __half へ丸めてから float に
//     戻して SiLU (launch_silu と同式・同精度 = x / (1 + __expf(-x))) を掛けるため、
//     「launch_group_norm_mb → launch_silu」の 2 パスと完全ビット一致する。
//   - SiLU は reduction を伴わない末尾 map なので、mb 版の run-to-run 完全
//     ビット一致 (決定的 2 段集約) はそのまま維持される (非交渉ゲート)。
//   - default (epilogue=off) 経路では使わないこと (byte-for-byte 無改変の原則)。
//   - 中間バッファは mb 版と共用 (grow-only 静的常駐・step 内 cudaMalloc なし)。
//   - d_in == d_out の in-place 可 (stats 確定後の normalize+SiLU は純 map)。
void launch_group_norm_silu(const __half* d_in,
                            const __half* d_gamma,
                            const __half* d_beta,
                            __half*       d_out,
                            int           N,
                            int           C,
                            int           H,
                            int           W,
                            int           num_groups,
                            float         eps);

} // namespace dollama

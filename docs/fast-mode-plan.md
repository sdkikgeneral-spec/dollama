# FAST モード計画 — 自作カーネル高速化 (default 温存 / --fast / --fast --fp8)

> Phase 2 の再オープン。現状 `dollama.exe` は SDXL 1024²/20step/CFG を **19.5s/枚** で生成しており、
> 同一 GPU・同一重み・同一 step で diffusers/ComfyUI が **~3.8s** を出す ([[docs/measurements-log.md]] probe10) のに対し
> **RTX5080 を RTX3060 相当の速度で走らせている (ハードを約5倍ドブに捨てている)**。物理限界ではなく、
> 自作カーネルが Blackwell の演算器を埋めていない構造損失。本 doc はその回収計画 (分割タスク台帳)。
>
> **走行注意 (2026-07-04 現在)**: 研究機ターミナルで F-0b G-1 rollout (`dollma_rollout_bestofn.py`) が
> `dollama.exe` を回して走行中。本計画のビルド/実走は**その完走・退避を待ってから**着手する
> (`docs/f0b-rejection-sft-plan.md` G-1 と資源競合するため)。

## 芯 (設計方針)

**default を今まで通りに温存し、高速化はフラグで opt-in する。** ComfyUI の `--fast` / `--fp8_e4m3fn` と同じ正攻法。
default を byte-for-byte 現行のまま残すことで **golden-SSIM 0.999998 の回帰アンカー**が保たれ、「速くした結果どこかが壊れた」を
常に基準線と突き合わせて検出できる。

### 3段モード (速度/精度の曲線上に 3 点)

| モード | 中身 | 精度 | 狙い |
|---|---|---|---|
| **default (フラグ無し)** | 現行カーネル・現行 CFG ループ | golden 0.999998 完全保存 | 回帰アンカー・基準線・無改変 |
| **`--fast`** | #1 GPU常駐+Graphs / #2 CFG batch=2 / #4 FlashAttn級 / #5 epilogue融合 | FP32 蓄積維持 = **絵は実質同一** | ~2倍・リスク無しの常用 (将来 default 昇格候補) |
| **`--fast --fp8`** | 上記 + FP8 を選択的 GEMM に | reward順位/最終RGB で gated | 最速・崩れたら層単位で FP16 に戻す |

- **精度ゼロコストと FP8 を混ぜない** のが肝。速度の大半は FP8 抜き (#1/#2/#4/#5・FP32 蓄積維持) で取れるので、
  「速いけど精度は保ちたい」点 (`--fast`) を必ず選べるようにする。FP8 は本当に最後の一段に隔離。
- フラグ表面: 環境変数 `DOLLAMA_FAST=1` / `DOLLAMA_FP8=1` もしくは HTTP/CLI フラグ。`--fp8` は `--fast` を含意 (FP8 単独は無効)。

## 現状診断 (律速の在処・実装根拠)

| # | 損失源 | 根拠 (file:line) | 性質 |
|---|---|---|---|
| CFG | cond/uncond を**逐次 2 forward** | `src/infer/diffusion.cu:534,543` | GPU を半分しか使わない |
| Loop | 毎 step で latent H2D / noise D2H / host で CFG合成・Euler・scale = **full sync × 20** | `src/infer/diffusion.cu:523-571` | カーネルの背後実行が全部潰れる |
| Attn | flash+wmma だが **1 block = 1 warp (32スレッド)** で低占有率 | `src/kernels/attention.cu:653` | SM が空く。4.60s の本体 |
| Attn | P·V epilogue が store_matrix→shared→acc の往復 | `src/kernels/attention.cu:571-578` | 余分な shared 往復 |
| UNet | `launch_unet` が **単一 latent 前提 (batch 引数なし)** | `src/infer/unet.cuh:88` | CFG batch=2 化に batch 次元の貫通が要る |
| FP8 | Blackwell FP8 tensor core **未使用** | — | 理論 2× を握ったまま |

> 「attn 4.60s」は 2-6 時点・CFGなし・stub の値。**実 checkpoint + CFG 込みの内訳は G-0 で取り直す** (下記)。

### 現状 GPU 稼働ベースライン (実測 2026-07-04・G-1 走行中の生成を nvidia-smi で観測)

| 指標 | 実測 | 解釈 |
|---|---|---|
| **消費電力** | **154W / 360W ≈ 43%** | 真の稼働指標。**パワーを半分も引いていない** = SM が満杯でない |
| SM 稼働率 (sm%) | 平均 ~78% (67-91%) | 「1warp でも動いた時間割合」= 満杯率ではない。sm% に騙されない |
| メモリ帯域 | ~11% | 帯域律速でもない |
| SM クロック | 2857 MHz | ブースト正常 (サボりではない) |
| VRAM | 7.9GB / 16GB | 余裕。VRAM は非ボトルネック |
| 候補間 | SM→0 / 126W | LM(CPU)+採点(CPU) の間 GPU が完全遊休 |

**結論**: 電力 43% + 帯域 11% = compute でも帯域でも律速せず **occupancy/latency 律速** (1warp/block attention + 依存待ち)。
「5080 を 3060 で走らせている」の定量的裏付け。#1(同期撲滅→SM谷を消す)/#4(multi-warp→power↑)/#2(遊休SMにuncond同居)で
**FP8 前に電力を 43%→300W超、時間を半減**が射程。目標 = **構造を壊さず GPU を飽和させる (power を上限近くまで引く)**。

## レバー詳細

### #1 拡散ループ GPU 常駐化 + CUDA Graphs (精度ゼロコスト)
- `scale_model_input` / CFG 合成 / Euler step を小さな elementwise カーネルにして latent を GPU 常駐のまま回す。
  per-step の D2H/H2D と 20 回の full sync を撲滅。
- さらに 20step × 数百カーネルの起動列を **1 本の CUDA Graph にキャプチャ**してローンチオーバーヘッド除去。
- 触るファイル: `src/infer/diffusion.cu` (ループ), `src/infer/scheduler.hpp` の host 実装を .cu elementwise へ, 新規 `src/kernels/elementwise` 追記。
- 精度: **ビット一致** (Graph 化・同期除去は数値不変。elementwise 化は演算順不変)。

### #2 CFG batch=2 (精度ゼロコスト)
- cond/uncond を batch=2 の 1 forward へ。`launch_unet` に batch 次元を通す。
  attention は既に `B` 対応、GEMM は M 倍化、conv2d/groupnorm は N 拡張、bias/act/norm は要素数 2 倍。
- 触るファイル: `src/infer/unet.cuh` / `src/infer/unet.cu` / `src/kernels/{gemm,conv2d,groupnorm,activation,geglu,bias_add}` の N/M 汎用性確認, `src/infer/diffusion.cu` の呼び出し。
- 精度: **数学的に同一** (束ねるだけ)。golden は再ベースライン (バッチ軸追加で並びが変わるだけ)。
- 効果: **素で ~2×**。出荷経路そのものが速くなる恒久資産。

### #4 attention を FlashAttention-2 級に (精度ゼロコスト)
- 現状 1 warp/block を **multi-warp block** 化し query 行タイルを増やして K/V 再ロードを償却
  (Sq=Sk=4096 の self-attn は K/V を query タイルごとに読み直しており DRAM 帯域律速)。
- **cp.async のダブル/トリプルバッファ**で K/V ロードと wmma をパイプライン化 (sm_120)。
- P·V epilogue の shared 往復除去 (fragment を直接 acc へ)。
- 触るファイル: `src/kernels/attention.cu` / `attention.cuh`。
- 精度: **FP32 蓄積維持** (reduction 順が変わり golden は微差・SSIM ゲートは通る想定)。

### #5 epilogue 融合 (精度ゼロコスト)
- bias 加算 + SiLU/GeGLU + GroupNorm を GEMM/Conv の epilogue に畳み込み、カーネル起動と DRAM 往復を削る。
- 触るファイル: `src/kernels/{gemm,conv2d}` に epilogue バリアント, 呼び出し側 `src/infer/unet.cu` / `vae_decode`。
- 精度: 融合は数値ほぼ不変 (中間の FP16 往復が減る分むしろ改善方向)。

### #3 FP8 Tensor Core (精度トレードオフ・最内 opt-in)
- **選択的 FP8**: 計算律速の大 GEMM のみ入力を FP8 (E4M3)、**蓄積は FP32** (既存規約と整合)。
  softmax / GroupNorm / time embed / 最初と最後の conv は **FP16 のまま残す** (敏感層)。
- 崩れたら**層単位で FP16 に戻せる** (可逆・段階適用)。
- 触るファイル: `src/kernels/gemm.cu` に FP8 経路, `src/infer/unet.cu` で層ごとの精度選択テーブル。
- 精度: **golden-SSIM 0.999998 には届かない (E4M3 は実質2桁)** → 下記の FP8 専用ゲートで判定。
- 効果: FP16 wmma の理論 2×。diffusers 既定は FP8 未使用なので**自作で追い抜ける唯一の軸**。ComfyUI の `--fp8_e4m3fn` 実績あり (実用上わずかな劣化)。

## 検証・ゲート (モードごとに物差しを変える)

| モード | ゲート |
|---|---|
| **default** | 現行 golden-SSIM 0.999998 を**無改変で維持** (test_unet / test_vae_decode / test_diffusion 全緑) |
| **`--fast`** | golden 再ベースライン後、noise_pred SSIM ≥ 0.9999 / 最終 RGB SSIM ≥ 0.999 (絵が実質同一であること) |
| **`--fast --fp8`** | golden は**使わない**。① 最終 RGB を FP16 経路と LPIPS/SSIM で比較 (知覚等価) ② **reward 順位相関** (best-of-N は絶対値でなく順位のみ使用・崩れなければ実害ゼロ)。**通らなければ FP8 不採用** (層単位で戻す) |

## 実測手段と SAC 制約 (非交渉)

- 研究機は SAC で**再ビルドした exe の新ハッシュがブロック** ([[reference_wdac_new_exe_block]])。カーネルを直して
  リビルドすると `dollama.exe` の実走緑が取りにくい。着手前に **allow-list 更新をユーザーへ依頼**するか、
  **開発機ビルド緑 + Python/OV 経由の数値検証**で回す運用を確定させる。
- profile: 拡散ループに CUDA events を仕込み、UNet 内の attn/gemm/conv/groupnorm の実測内訳を取る (G-0)。

## 分割タスク台帳

| Pkg | 内容 | 担当 | 依存 | 精度 | status |
|---|---|---|---|---|---|
| **G-0** | profile 計装 (CUDA events) で実 checkpoint+CFG 込みの内訳確定 + FAST フラグ枠 (env/CLI・default 無改変) を骨組み | cuda-kernel-dev + cpp-implementer | G-1完走待ち | — | 🔲 |
| **G-1k** | #1 ループ GPU 常駐化 + CUDA Graphs | cuda-kernel-dev | G-0 | ゼロコスト | 🔲 |
| **G-2k** | #2 CFG batch=2 (`launch_unet` batch 貫通) | cuda-kernel-dev + cpp-implementer | G-1k | ゼロコスト | 🔲 |
| **G-3k** | #4 attention FlashAttn級 (multi-warp/cp.async/epilogue) | cuda-kernel-dev | G-0 | ゼロコスト | 🔲 |
| **G-4k** | #5 epilogue 融合 (gemm/conv) | cuda-kernel-dev | G-2k | ゼロコスト | 🔲 |
| **G-5k** | #3 FP8 選択的経路 + 層別精度テーブル + FP8 ゲート | cuda-kernel-dev | G-2k/G-4k | トレードオフ | 🔲 |
| **G-6k** | 3段モードの実測まとめ + 出荷判定 (`--fast` の default 昇格是非 / FP8 採否) | PL + gpu-benchmarker | 全部 | — | 🔲 |

> 各 Pkg は着手時に **test も実装** (CLAUDE.md ルール4・`src/tests/test_*`・meson 緑)。

## 順序 (掛け算で効く)

```
G-0 (profile+枠) ──► G-1k (常駐/Graphs) ──► G-2k (CFG batch2) ──► G-4k (融合)
                 └──► G-3k (attention) ────────────────────────► G-5k (FP8) ──► G-6k (判定)
```
まず **精度ゼロコスト4手 (#1/#2/#4/#5) で ~8-10s を確定** (絵は今と同一)。そのうえで **FP8 は別立ての gated 実験**。
A+B (#1/#2) だけでも ~19.5s→~10-11s、4手で ~8-10s、FP8 が通れば diffusers 級 (~3.8s) かそれ以下が射程。

## 安全弁 (非交渉・既定)

1. **default は無改変** (回帰アンカー)。全高速化は `--fast` 以降に隔離。
2. **走行中 G-1 に触れない**。ビルド/実走は G-1 完走・退避後。`data/rollouts/` 無干渉。
3. **FP8 は可逆** (層単位で FP16 に戻せる)。ゲート未通過なら不採用でクローズ。
4. **コミットは巻き込み禁止** (in-flight ファイルがあるため `git add -A` を使わず対象ファイルのみ明示 stage)。

## スコープ外

- **steps 削減 (Turbo/Lightning/LCM 蒸留・高速 sampler)** = カーネルでなくモデル/サンプラ層の直交レバー。
  最大の乗数だが eval 基準 steps=20 を動かす判断が別途要るため本 doc では扱わない (要ユーザー決裁)。
- **cuBLASLt / cuDNN / CUTLASS 置換** = 最速だが**自作カーネル研究の芯を殺す**ため不採用 ([[project_output_quality_over_features]])。
  上記 #1-5+FP8 は全部「自作のまま」で diffusers 級を狙う構成。

## 参照
- 診断根拠: `src/infer/diffusion.cu` / `src/kernels/attention.cu` / `src/infer/unet.cuh` / `src/kernels/gemm.cuh`
- ベースライン: CLAUDE.md 計測表 (probe10 3.80s / 自作フル拡散 11.30s / test_unet SSIM 0.999998) / `docs/measurements-log.md`
- 走行中の依存: `docs/f0b-rejection-sft-plan.md` (G-1 rollout・`dollama.exe` 使用中)
- SAC 制約: [[reference_wdac_new_exe_block]] / CUDA ビルド: [[reference_cuda_meson_build]] / カーネル規約: [[reference_cuda_kernel_conventions]]

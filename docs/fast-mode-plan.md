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

> 「attn 4.60s」は 2-6 時点・CFGなし・stub の値。**実 checkpoint での内訳を G-0 で取り直した (下表)**。実重み 4.9GB でも
> attention は 4.621s とほぼ不変 (コストは shape 駆動で重み値に非依存) → 律速の在処は stub 時と同じと確定。

### G-0 実測 (2026-07-05・既存 allow-list 済み dollama.exe・リビルド無し)

`DOLLAMA_PROFILE=1` + OV アセット無効化で**段2 PipelineGenerator (計装済み非 CFG `generate`)** を実 UNet 重み (`unet_weights.safetensors` 4.9GB) で 20step/1024² 実走。CFG 版 `generate_txt2img` は未計装だが、UNet 内カーネルの律速内訳は CFG/非 CFG で不変 (CFG は同 UNet を 2 forward するだけ) ゆえこれで律速は確定できる。

| 項目 | 実測 | 比率 | 解釈 |
|---|---|---|---|
| TOTAL (非 CFG) | 11.469 s | 100% | CFG 本番 ≈ 2×UNet+VAE ≈ **21.7s** (doc 冒頭 19.5s 帯) |

> **✅ e2e ベースライン一本化 (2026-07-09・G-2k ベンチで確定)**: `test_diffusion_batch2 DB2_BENCH=1` (20step/1024²/guidance7.5/**warm**=UNet+VAE 重み常駐/VAE decode 込み/**OV encode・cold 重み転送は付帯外**/SAC OFF 実 exe) で 3 構成を min 実測: **default (逐次 2 forward) 20.93s** / batch2 単独 17.59s (x1.19) / **--fast (attn_fast+batch2) 15.88s (x1.32)**。冒頭 19.5s / G-0 推定 21.7s と同帯 (warm ゆえ僅かに軽い)。G-3k 注記の 43.83s は cold 重み転送込みの段1 CFG 本番値で別物 (付帯込み)。**G-6k 出荷判定の分母 = warm default 20.93s / diffusers 3.8s は cold+OV 込みの別条件ゆえ最終比較は同条件で取り直す**。
| UNet step total (20) | 10.254 s | 89.4% | per-step 0.513s |
| └ transformer (attn/gemm) | 7.007 s | 61.1% | うち gemm ≈ 2.39s |
| 　└ **attention only** | **4.621 s** | **40.3%** | **単一最大の律速** = #4 (G-3k) が最大レバーと裏取れ |
| └ resnet (conv/groupnorm) | 1.225 s | 10.7% | 二次律速 (#5 G-4k 対象) |
| VAE decode | 1.197 s | 10.4% | — |
| **host roundtrip** | **0.007 s** | **0.06%** | **ほぼゼロ** → #1 の「同期/D2H 撲滅」の時間直接効果は小。#1 の主眼は Graph ローンチ削減 + batch2 の下地に補正 |
| 段グループ | up 5.65s(49%) > down 3.61s(31%) > mid 0.97s | | up-block 律速 |

**優先度の実測結論**: **#4 attention (G-3k) が単独最大 40%** → 次いで **#2 batch=2 (G-2k・2×forward 削減+occupancy)**。**#1 の時間直接効果は小さい (host 0.007s)** ため順序を #4/#2 先行へ寄せる (下記台帳決裁: G-0b フラグ枠 + G-3k を先行)。

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

- **2026-07-06 更新: SAC OFF・実 exe 実走で全 49/49 緑を確認**。以後の全ゲート (G-2k の SSIM / CFG e2e 秒数、G-5k の LPIPS/reward 順位相関) は**実 exe 実測を正**とする。下記の旧運用 (開発機ビルド緑 + Python/OV 検証) は SAC 再有効化時のフォールバックとして温存。
- (旧) 研究機は SAC で**再ビルドした exe の新ハッシュがブロック** ([[reference_wdac_new_exe_block]])。カーネルを直して
  リビルドすると `dollama.exe` の実走緑が取りにくい。着手前に **allow-list 更新をユーザーへ依頼**するか、
  **開発機ビルド緑 + Python/OV 経由の数値検証**で回す運用を確定させる。
- profile: 拡散ループに CUDA events を仕込み、UNet 内の attn/gemm/conv/groupnorm の実測内訳を取る (G-0)。

## 分割タスク台帳

| Pkg | 内容 | 担当 | 依存 | 精度 | status |
|---|---|---|---|---|---|
| **G-0** | profile で実 checkpoint 内訳確定 (既存計装・リビルド無しで実測) | — | G-1完走待ち | — | ✅ (上表・attention 40% 最大) |
| **G-0b** | FAST フラグ枠 (env `DOLLAMA_FAST`/`DOLLAMA_FP8` + CLI・`FastConfig` を拡散経路へ貫通・default byte-for-byte 無改変) | cpp-implementer | G-0 | 無改変 | ✅ (test_fast_config 緑・golden 無改変で緑) |
| **G-1k** | #1 ループ GPU 常駐化 + CUDA Graphs (時間直接効果は小・Graph ローンチ削減主) | cuda-kernel-dev | G-2k/G-4k 後の再profile | ゼロコスト | 🔲 **条件付き降格**: host 0.007s ゆえ時間直接効果ほぼ無。G-2k で batch 次元が変わると Graph 再キャプチャ=手戻り。再profile で launch 谷が実測で残る場合のみ着手・残らなければ不着手クローズ可 |
| **G-2k** | #2 CFG batch=2 (`launch_unet` batch 貫通) | cuda-kernel-dev + cpp-implementer | G-0b | ゼロコスト (fast側 SSIM=1 目標: 行単位 K-loop 不変なら G-3k 同様ビット一致狙い・届かねば ≥0.9999 ゲート) | ✅ **完了 (S1〜S3c・下記注)。parity gate g=1.0 SSIM 0.9995≥0.999・batch2 e2e 1.19x (期待 ~2× に非到達)・合成 --fast 1.32x** |
| **G-3k** | #4 attention FlashAttn級 (multi-warp/cp.async・`launch_attention_fast` 別 TU・既存温存) + call site 結線 (`--fast`→attn_fast) | cuda-kernel-dev | G-0b | ゼロコスト | ✅ **カーネル SSIM=1.0 ビット一致・self4096 単体 1.41x / warm e2e UNet 1.20x (下記注)** |
| **G-3kf** | G-3k follow-up: `test_unet_fast` を常駐ハンドル warm 経路へ (毎 call 4.9GB 再転送を止め・ゲート数値を実態 1.20x に是正) | cpp-implementer | G-3k | — | ✅ **完了**: warm ハンドル (unet_weights_create) で計測・default vs golden SSIM 0.999996/bad0・fast vs default bit-exact・warm 1step default 526ms/fast 437ms = **1.20x** |
| **G-4k** | #5 epilogue 融合 (gemm/conv) | cuda-kernel-dev | G-2k | ゼロコスト | 🔲 |
| **G-5k** | #3 FP8 選択的経路 + 層別精度テーブル + FP8 ゲート | cuda-kernel-dev | G-2k/G-4k | トレードオフ | 🔲 |
| **G-6k** | 3段モードの実測まとめ + 出荷判定 (`--fast` の default 昇格是非 / FP8 採否) | PL + gpu-benchmarker | 全部 | — | 🔲 |

> 各 Pkg は着手時に **test も実装** (CLAUDE.md ルール4・`src/tests/test_*`・meson 緑)。

### G-3k 結果の注記 (2026-07-05・実走緑・warm 実測で確定)

- **正しさ**: `launch_attention_fast` は既存 `launch_attention` と **SSIM=1.0 / MAE=0 (ビット一致)**。reduction が同じ 16-タイル順のため。UNet 全段でも default vs fast で **noise_pred SSIM=1** = 絵は完全同一。default (fast off) は golden 無改変で緑。→ **精度リスクゼロで恒久資産**。
- **速度 (warm 実測・`prof_unet_fast_warm.cu` + 段1 e2e)**:
  - attention 段/step **231→161ms = 1.43x** (単体 micro 1.41x と一致)、UNet forward/step **452→382ms = 1.19x** (−72ms/step)、attention 以外 (GEMM/resnet) は不変 = 副作用なし。
  - 段1 CFG 本番 e2e (交互3回): **43.83→40.20s = −3.63s** (全ペアで fast 勝ち・逆転なし)。CFG は 1step 2 forward ゆえ理論 −2.9s と整合。
- **「1.03x」は計測アーティファクトだった (訂正)**: 初回報告の e2e 1step 1.03x は `test_unet_fast` が後方互換 `launch_unet(const SafeTensors&)` を使い**毎 call で 4.9GB を H2D 再アップロード (cold)** → attention 比率が 40%→20% に希釈された値。カーネルは遅くない。→ **要 follow-up: test_unet_fast を常駐ハンドル warm 経路に直し、ゲート数値を実態 (1.19x) に合わせる**。
- **含意**: G-0 の「attention 40% 最大律速」を fast が 1.43x で回収済み。単独 40% 律速は解消 (fast 後 warm 内訳 = attn 161ms / GEMM ~130ms / resnet ~60ms)。**次の最大レバーは計画どおり #2 CFG batch=2** (2 forward→1 batched・occupancy+host 半減)。ソースレビューの「shared 占有率律速で 1.09x 上限」説は warm 実測で否定 (両者を回して確定)。

### G-2k 結果の注記 (2026-07-09・SAC OFF 実 exe 実走で確定・S1〜S3c)

- **実装 (S1〜S3b)**: `launch_unet_batched(B)` で batch 次元を UNet 全段に貫通 (S2: attention は既に B 対応・linear GEMM は M=B*tokens・conv2d は per-n ループ・groupnorm/bias/act は要素数倍化)。`generate_txt2img` に `use_batch2` 分岐を新設 (S3b): batch2 時は cond/uncond を連続 [2,...] に束ね `launch_unet_batched(2)` を 1 step 1 回で回す。default (batch2 off) は逐次 2 forward を **byte-for-byte 無改変**で保持 (回帰アンカー)。フラグは env `DOLLAMA_BATCH2` / `--fast` 含意 (S3a)。
- **正しさ (S3c parity gate・`test_diffusion_batch2`)**: batch2 は数学的同一変換だが linear の GEMM が M=tokens→B*tokens に変わり cuBLAS タイル選択=蓄積順が変わるため FP16 で ~1 ULP/要素の差 (S2 per-sample MAE 6.4e-5・**ビット一致には非到達**)。この微小差は CFG 合成 `noise=uncond+g*(cond-uncond)` で **guidance 倍に増幅**され step 累積 → 最終 RGB 発散は guidance に**単調** (実測 sweep g=1:SSIM0.9995 / g=3:0.9988 / g=7.5:0.9966・平均差≤0.3/255=絵は実質同一)。**ノイズ床 (off@g=1.0 x2) は完全 bit-exact** → 発散源は 100% batch2 の tiling 差と切り分け確定。**ゲート (ユーザー決裁 2026-07-09)**: CFG 増幅を排し batch2 単独の数学的等価性を測るため **guidance=1.0 の最終 RGB SSIM ≥ 0.999 をハードゲート** (計画書 --fast RGB 基準)・realistic な g=3/7.5 は characterization 出力 (regression 監視)。gate g=1.0 SSIM 0.9994 PASS。
- **速度 (`DB2_BENCH=1` warm e2e・上記ベースライン表)**: batch2 単独 20.93→17.59s = **1.19x** (attn_fast off)・**期待「素で ~2×」には非到達**。主因: **conv2d は per-n 直列で batch されない** (resnet ~10.7% は恩恵ゼロ)・batch されるのは attn+linear GEMM のみで占有率ゲインも部分的 = 「逐次 2 forward で GPU 半遊休だから 2×」の仮説ほど単純でなかった。精度ゼロコストで 1.19x は正の恒久資産。**合成 --fast (attn_fast+batch2) = 20.93→15.88s = 1.32x**。
- **含意**: 精度ゼロコスト 2 手 (#4 attn ✅ + #2 batch2 ✅) の合成が **1.32x** (20.93→15.88s)。計画冒頭の「4 手で ~8-10s (~2倍超)」は batch2 が 1.19x に留まったぶん下方修正が要る。残る精度ゼロコストは **#5 epilogue 融合 (G-4k)** = resnet/conv 部 (batch2 が効かなかった ~10.7%+GEMM epilogue) が主対象で相補的。その後 FP8 (G-5k) が diffusers 級への最大レバー。

## 順序 (掛け算で効く)

```
G-0/G-0b (profile+枠) ──► G-3k ✅ (attention) ──► G-2k ✅ (CFG batch2) ──► G-4k (融合) ──► [G-1k 条件付き] ──► G-5k (FP8) ──► G-6k (判定)
```
> **順序改訂 (2026-07-06)**: G-0 実測で host roundtrip 0.007s と判明したため G-1k (同期撲滅/Graphs) を主線先頭から**末尾の条件付き**へ降格。attention (G-3k) は完了済ゆえ次は #2 CFG batch=2 (G-2k) = UNet 15.3s 全体に効く単独最大レバー。
> **G-2k クローズ (2026-07-09)**: batch2 1.19x/合成 --fast 1.32x で完了。**次は #5 epilogue 融合 (G-4k)** = batch2 が効かなかった conv/resnet 部 + GEMM epilogue が対象 (相補的)。その後 FP8 (G-5k)。

まず **精度ゼロコスト4手 (#4✅/#2/#5/#1) で ~8-10s を確定** (絵は今と同一)。そのうえで **FP8 は別立ての gated 実験**。
A+B (#1/#2) だけでも ~19.5s→~10-11s、4手で ~8-10s、FP8 が通れば diffusers 級 (~3.8s) かそれ以下が射程。

## 安全弁 (非交渉・既定)

1. **default は無改変** (回帰アンカー)。全高速化は `--fast` 以降に隔離。
2. ~~走行中 G-1 に触れない~~ **解消済み (2026-07-06)**: F-0b は不採用クローズ (a3b129d merge) 済みで G-1 rollout はもう走っていない。`data/rollouts/` 制約は撤去。
3. **G-2k 着手前に G-0b/G-3k を先行コミット** (対象ファイル明示 stage)。ワークツリーに未確定成果を残したまま次 Pkg に入ると安全弁4 が守れず、問題時のロールバック点も失う。
4. **FP8 は可逆** (層単位で FP16 に戻せる)。ゲート未通過なら不採用でクローズ。
5. **コミットは巻き込み禁止** (in-flight ファイルがあるため `git add -A` を使わず対象ファイルのみ明示 stage)。

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

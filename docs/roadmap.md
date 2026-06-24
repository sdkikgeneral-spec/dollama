# dollama 実装ロードマップ

**スコープ (確定)**: 生成対象は**キャラクターのみ**。背景は外部 (Grok/Gemini/SD) +
CLIP Studio で合成し、出力は**切り抜き済み透過 PNG**。キャラ設定構造・切り抜き・
手指品質・学習ループの設計は `docs/character-bible-spec.md` を参照。

## Phase 1 — パイプライン骨格 (現在)

OpenVINO C++ API で動くパーツから順に実装し、スレッド骨格を完成させる。
SDXL / BitNet は含まない。

| # | 実装物 | ファイル | 状態 |
|---|---|---|---|
| 1 | Tensor クラス | `src/core/tensor.hpp` | ✅ 完了 |
| 2 | メモリアロケーター | `src/core/allocator.hpp` | ✅ 完了 |
| 3 | SPSC キュー | `src/core/queue.hpp` | ✅ 完了 |
| 4 | CLIP NPU 推論 | `src/infer/clip.hpp` | ✅ 完了 (NPU 7.82ms) |
| 5 | キャラ台帳 (CharacterBible, authored 層) | `src/core/character.hpp` | ✅ 完了 |
| 6 | WD14 CPU 推論 | `src/infer/wd14.hpp` | ✅ 完了 (CPU 105ms) |
| 7 | スレッド骨格 + CPU アフィニティ | `src/main.cpp` + `src/core/affinity.hpp` + `src/pipeline.hpp` | ✅ 完了 (9.13 frames/s, WD14 律速) |

**Phase 1 完了の定義**: stub (LLM なし) → CLIP(NPU) → queue → WD14(CPU) のループが
マルチスレッドで回り、タグ文字列が出力されること。

---

## Phase 2 — SDXL 自作 CUDA カーネル

最大の実装物。CUDA カーネルをゼロから書き、diffusers なしで画像生成を実現する。
SDXL UNet/VAE は **FP16 dense** なので ternary GEMM は使わない (Phase 4 BitNet へ移動)。

**前提 (ブロッカー)**: CUDA Toolkit 12.8 (nvcc) のインストールが必須。現状ドライバ
(610.47) のみで Toolkit 未導入。probe10 は PyTorch 同梱ランタイムで動いていた。
Blackwell sm_120 は CUDA 12.8+ が必須。導入後 `meson.build` の project 言語に `'cuda'`
を追加し `-arch=sm_120` でコンパイルする。

**カーネル方針 (確定)**: 完全自作を目指す。到達困難になった重い GEMM/Conv のみ
cuBLAS/cuDNN フォールバックを許容 (自作版に後で置換可能な形で実装)。Attention・
正規化・活性化は自作。3.80s 同等は段階的目標とし、まず正しさ・絵が出ることを優先。

**検証戦略**: 生 CUDA は参照無しでは数値デバッグ不能 → Python(probe10 環境)で中間
テンソルをダンプ → C++ カーネルがロードして許容誤差比較する**ゴールデンテスト**を各段に置く。

| 段 | 実装物 | ファイル | 検証 | 状態 |
|---|---|---|---|---|
| 2-0 | Toolkit + meson CUDA 言語 + 疎通 (vector add) | `meson.build`, `src/kernels/utils.cuh` | test_cuda_smoke | ✅ 完了 (CUDA 13.3 / sm_120) |
| 2-1 | エラーチェック + カーネル基盤 (CUDA_CHECK/CUDA_CHECK_KERNEL/ceil_div) | `src/kernels/utils.cuh` | test_cuda_smoke (マクロ経由) | ✅ 完了 |
| 2-2 | primitives (下記 2-2-1〜2-2-5 に分割。1つずつ 実装→ゴールデンテスト→ベンチ) | `src/kernels/*.cu` | 各 test、CPU 参照と tol 比較 | ✅ 完了 (GEMM/活性化/GroupNorm/Conv2d/Attention 全緑) |
| 2-2-1 | dense FP16 GEMM (他カーネルの検証土台) | `src/kernels/gemm.cu` | test_gemm、CPU 参照と tol 比較 | ✅ 完了 (shared-mem タイリング, 1024³ 4730 GFLOPS) |
| 2-2-2 | SiLU / GeLU 活性化 (GeLU erf 主・tanh 併設) | `src/kernels/activation.cu` | test_activation | ✅ 完了 (FFN 544 GB/s, in-place 安全) |
| 2-2-3 | GroupNorm (1グループ=1ブロック, 1パス FP32 リダクション) | `src/kernels/groupnorm.cu` | test_groupnorm | ✅ 完了 (UNet 75 GB/s, SiLU 融合は 2-4/2-5 で検討) |
| 2-2-4 | Conv2d (最重量・計算量の大半) | `src/kernels/conv2d.cu` | test_conv2d | ✅ 完了 (direct conv, UNet C320 64² 3×3 1807 GFLOPS / VAE C128 512² 3×3 1667 GFLOPS。1×1=GEMM・3×3=im2col/タイリング昇格は 2-4/2-5 で実測後) |
| 2-2-5 | Attention (self + cross、GEMM+softmax) | `src/kernels/attention.cu` | test_attention | ✅ 完了 (per-(b,h,row) block + shared scores, FP32 softmax。self 1024² Dh80 1631 GFLOPS / cross Sk77 1748 GFLOPS。flash/Tensor Core 委譲は後続) |
| 2-3 | safetensors 重みローダー | `src/io/safetensors.hpp` | test_safetensors、golden 突合 | ✅ 完了 (ヘッダオンリー・自作最小 JSON パーサ・F32/F16/BF16/I64/I8 突合, load 19.0 µs/op) |
| 2-4 | **VAE decode** (latent→画像、自己完結・初の実画像) | `src/kernels/vae_decode.cu` | probe10 latent → 正解画像比較 | ✅ 完了 (final SSIM 0.999992 / decode 7.96s/枚 / up2以降 FP32 中間) |
| 2-5 | **SDXL UNet** + スケジューラ (Euler/DDIM) | `src/infer/unet.cu`/`.cuh`, `src/infer/scheduler.hpp` | 1step ごと latent を PyTorch 比較 | ✅ 完了 (noise_pred SSIM 0.999998 / 24段ゴールデン全緑 / 1step ~9.2s。Euler scheduler max_err≤5e-6) |
| 2-6a | フル C++ 拡散パイプライン結線 (UNet×Nstep+Euler+VAE) + PipelineGenerator/HTTP DI + 対 3.80s 計測 | `src/infer/diffusion.cu`, `src/server/pipeline_generator.hpp` + factory, `src/main.cpp` | test_diffusion (2step smoke 緑 / 20step DOLLAMA_BENCH) + test_pipeline_generator + test_pipeline_factory | ✅ 完了 (20step 実画像 **84.07s** = probe10 3.80s 比 22.1x。golden 埋め込み・CFG なし) |
| 2-6 最適化 | direct conv→im2col/Tensor Core GEMM・naive attention→flash で 84s→3.80s | `unet.cu` / `vae_decode.cu` / `conv2d.cu` / `attention.cu` | 既存 golden 突合維持 + 速度再計測 | 🟡 **一旦クローズ (2026-06-22)**: 84.07s→**11.30s** (7.44x 改善・probe10 比 2.97x)。S2(conv im2col+wmma)/S3-0(ビルドフラグ)/S3-C(attn wmma)/S3-B(GEMM cuBLAS)/S3-A(VAE mid flash)/S3-D(VAE 重み常駐)/S3-E(VAE up2/up3 FP32 direct conv→im2col+cuBLAS TF32 GEMM, 5.21s→1.16s) 完了。残 (UNet attention 4.60s) は cuBLAS/cuDNN ライブラリフォールバック余地として保留。研究の本丸を **Phase 4 独自 LM (BitNet)** に移すため、ここで自作 CUDA の詰めは打ち切り。golden 維持 (UNet SSIM 0.999996 / VAE SSIM 0.999988) |
| 2-6b | 本 txt2img: CLIP-G OV 化 (dual encoder 2048+pooled1280) + CFG (negative) + prompt→embeds 結線 (**prompt 供給元は将来 Phase 4 A の自作タグ生成 LM**。Phase1 縦通しの stub を差し替え) | `infer/{clip_tokenizer,clip_encoder2,text_conditioner}.hpp`, `server/diffusion_runner.{hpp,cu}` + stub, `server/txt2img_generator.hpp`, `infer/diffusion.cu` (generate_txt2img), `main.cpp` | test_text_conditioner + test_txt2img + golden 突合 + 生成確認 | ✅ 完了 (2026-06-23, c3b1dac)。Stage A golden / B OV変換 (tokenizer・encoder L/G) / C bigG device probe / D TextConditioner (L768++G1280=2048 concat・G pooled=text_embeds・time_ids) / E generate_txt2img (cond/uncond 2回UNet→host CFG 合成) / F 結線 (純cpp IDiffusionRunner 境界で OV/CUDA 分離・main DI 3段フォールバック)。NPU で CLIP-L/bigG、guidance 7.5・20step、PNG 1024² var=2300。meson test 35/35 全緑 |

**最初の "絵が出る" 山は 2-4 (VAE decode)**。UNet より小さく自己完結で、probe10 の
latent を入力に正解画像と比較できるため、最初の実画像マイルストーンに置く。

**Phase 2 完了の定義**: フル C++ パイプラインで 1024×1024 画像が生成されること。
目標: probe10 ベースライン (3.80s / 20steps) と同等以上。
**Phase 2 後の接続**: 生成画像は §11 品質スコアラ (Phase 4 B) の入口になる (生成→採点→A へ FB)。
Phase 2 の作業内容自体は本方針変更で増減しない (ternary は元から Phase 2 非対象)。

---

## Phase 3 — HTTP サーバー

外部クライアント (WebUI 等) から呼べるようにする。
**配管は自作しない**: HTTP/JSON は定番のヘッダオンリーライブラリを使う
(自作は HW 研究コアに限定、CLAUDE.md「実装方針」参照)。

| # | 実装物 | ライブラリ / ファイル | 状態 |
|---|---|---|---|
| 1 | HTTP サーバー | **cpp-httplib 0.47.0** (単一ヘッダ・Winsock2/POSIX 吸収) | ✅ 完了 |
| 2 | JSON 入出力 | **nlohmann/json 3.12.0** (ヘッダオンリー) | ✅ 完了 |
| 3 | エンドポイント実装 | `src/server/api.cpp` (上記2ライブラリを使用) | ✅ 完了 (generations/health/models, edits=501) |
| 4 | Base64 (PNG 返却用) | `src/server/base64.hpp` (自己完結) | ✅ 完了 |

依存はヘッダオンリーのみ採用 → 単一バイナリ・重量級フレームワーク不使用の方針は維持。
meson subproject (wrap) で取り込む。API 仕様: `docs/http-api-spec.md` 参照。
`POST /v1/images/generations` で OpenAI Images API 互換。

**生成本体の抽象境界**: `src/server/generator.hpp` の `IImageGenerator` (純粋仮想) 越しに
生成器を注入。生成器の責務は PNG バイト列まで・base64 化はサーバ層。現状は
`StubGenerator` (prompt ハッシュ決定色のダミー画像)。**2-6 完了時に `PipelineGenerator`
を追加し main.cpp の DI 1 箇所差し替えで本 txt2img へ移行** (Pipeline は本フェーズで未改変)。

**Phase 3 完了の定義**: `curl` で叩いて PNG (base64) が返ってくること。
→ ✅ 達成。test_http で自己リクエスト全緑、スタブ生成+HTTP 往復 2.11ms (256×192)。

---

## Phase 4 — 自作タグ生成 LM (旧「BitNet b1.58 LLM」)

現在の LLM stub (またはQwen2 Python) を自作モデルに置き換える。

> **方向性レビュー (2026-06)**: 当初は「BitNet b1.58 を作る」が目的だったが見直した。
> 経緯・論点 (要点):
> - **核は decoder-only 小型タグ生成 LM** (`bitnet.hpp` 33M, #1/#2 実装済み)。これは維持。
> - **ternary (b1.58) は目的ではなく圧縮の研究軸に降格**。まず FP16/INT8 dense で品質を出し、
>   ternary は後段の実験 (`ternary_gemm.cu`)。33M 規模では旨味が限定的。
> - **置き場は GPU が第一** (速度 + 既存自作 CUDA カーネル gemm/attention/layernorm/geglu を流用。
>   LM は拡散の上流で逐次 → GPU 競合ほぼ無し)。**CPU は代替** (フレーム並列時の裏生成用)。
>   **NPU は自己回帰不可で除外** (probe6/investigation-log)。
> - **先行実装あり**: DanTagGen (400M LLaMA) / TIPO が「自然文+タグ → danbooru タグ」を実現済み
>   → 素の text→tags は新規性が薄い。これらは蒸留教師・品質基準として使う。
> - **2D 特化の独自性 (A/B)**: ① キャラ**同一性条件付き**タグ生成 (character-bible を条件入力) /
>   ② アニメ**品質スコアラ** (NPU・固定形状, §11) — ここが DanTagGen に無い価値。

本線は「**dense で動くタグ生成 LM**」。その上に 拡張(A)・評価(B)・圧縮(#5 ternary) を積む。
番号は実装 ID で不変 (`bitnet.hpp` 等が 4-5=ternary / 4-6=推論 を参照するため)。

| 段 | # | 実装物 | ファイル / 作業 | 状態 |
|---|---|---|---|---|
| 基盤 | 1 | 訓練データ収集 | user text → danbooru tags ペア (`data/bitnet/`, `scripts/dollma_{build_vocab,make_pairs}.py`, `docs/dataset-spec.md`) | ✅ 完了 (5,000 ペア / vocab 4,994 タグ / 実 danbooru タグ共起 + 合成テンプレ / OOV0・負語0・順序0・リーク0 / tokenizer 往復 UNK0) |
| 基盤 | 2 | モデル定義 (30-100M params) | `src/models/bitnet.hpp` | ✅ 完了 (decoder-only LLaMA系: BitLinear(ternary absmean + 活性int8 absmax)/RMSNorm/RoPE/SwiGLU/causal attn・embed tied。確定アーキ d_model=512/n_layers=8/n_heads=8/ffn=1792/vocab=4999/max_seq=64 = **32.98M params**。純ホスト参照 forward (4-5/4-6 のゴールデン基準)・test_bitnet 全緑) |
| dense 本線 | 3 | トークナイザー (タグ単位完全一致・旧称「BPE」) | `src/io/tokenizer.hpp` | ✅ 完了 (vocab.json 駆動・ヘッダオンリー純ホスト C++。specials id 0..4 + tags[i].id==5+i 検証ロード・encode/decode/encode_text(greedy 最長一致)・§6 正規化(`long_hair`→`long hair`・顔文字 `_` 保持)。実 pairs 5,000行/77,195タグ **UNK 0**・往復完全一致。encode 365 / decode 168 / encode_text 2655 ns/op。test_tokenizer 全緑。サブワード BPE は VOCAB_SIZE 再設計を伴う別タスクとして切離) |
| dense 本線 | 4 | 訓練スクリプト (Python) | `scripts/train_bitnet.py` | ✅ 完了 (hard CE・蒸留なし初版。bitnet.hpp 等価 dense PyTorch 32.98M を GTX1080Ti FP32 6epoch/45.3s 訓練 → val_loss 2.41 収束・top10 tag recall 0.777=random の 388x。重み `data/bitnet/bitnet_dense{,_fp32}.safetensors` を [out,in] 規約で 1:1 出力・74テンソル。`docs/training-spec.md`。**系列レベル蒸留 (Qwen2 text 多様化 3000ペア混合) を試行したが過学習が悪化 (gap 1.09→2.07・recall 微減) し採用せず — 負の結果を training-spec §10 / dataset-spec §12 に記録。soft-label KL (D5/D6) は下行「4-蒸留」参照**) |
| dense 本線 | 4-蒸留 | 蒸留 A/B 深掘り (D2–D6・recall 天井突破の探索) | `scripts/train_bitnet.py` + `scripts/dollma_d6_teacher_cache.py` | 🔬 4 路線評価済 — **D2/D4** (Qwen2 text 多様化 hard CE 混合) = 過学習悪化で**負**。**D5 案A** (corpus 共起 soft teacher・`--distill-kl`) = 過学習は実抑制 (gap 1.09→−0.21) だが recall 0.758<#1 で**不採用**。**D6 案c** (外部教師 **TIPO-200M** 生成タグを自作 vocab 写像・`--distill-ext`) = 単一 seed では #1 を僅かに上回ったが **seed 頑健性 sweep (4 seed) で再現せず不採用** (delta 平均 −0.0008・符号反転・#1 の seed 分散内)。**全 4 路線で蒸留の recall 利得は否定** — 再現する効果は過学習抑制 (gap 1.07→0.19) のみ = 正則化ノブ。詳細 training-spec §10–12 / dataset-spec §12 |
| dense 本線 | 6 | C++ 推論 (CPU dense 完了 / GPU は #6-GPU で完了 / 量子化は後段) | `src/infer/bitnet.hpp` | ✅ CPU dense 完了 (`BitNetDenseInfer`: safetensors FP32 ロード→forward+greedy デコード text→tags。models/bitnet.hpp の ternary は不使用・dense FP matmul double 蓄積。RoPE NeoX/tied lm_head)。PyTorch golden 突合: logits max abs err ≤2.29e-5・corr 1.0 (seq 8/32/63)、greedy 5/5 完全一致。test_bitnet_infer 緑。1 forward ~253ms (naive)。GPU=#6-GPU で完了・INT8 は別タスク |
| dense 本線 | 6-GPU | C++ GPU 推論 (RTX5080 で自作 CUDA カーネル流用) | `src/infer/bitnet_gpu.cu` (T1 device forward) + `src/infer/bitnet_gpu.cuh` (T2 `BitNetGpuInfer`) | ✅ 完了 (純 FP32 で CPU 版 `BitNetDenseInfer` と数値一致)。device forward = embed gather / RMSNorm / RoPE / causal SDPA / SwiGLU の static `__global__` + Linear/lm_head は **LM ローカル純 FP32 cuBLAS (CUBLAS_COMPUTE_32F・TF32 厳禁)**・リダクション系は double 蓄積で CPU(double) と桁合わせ。74 テンソルを device 常駐・LM ローカル cublas ハンドル。**本番重み突合**: logits seq 8/32/63 で max abs err 1.49e-6〜6.44e-6 / corr 1.0、greedy `generate` 3/3・`generate_with_identity` 2/2 完全一致。**forward 中央値 (DOLLAMA_BENCH=1, warmup3/iters30)**: seq8 2.43ms (CPU 113.78ms=**46.8x**) / seq32 9.90ms (46.2x) / seq63 10.29ms (900.83ms=**87.5x**)。class と heavy include は `#ifndef __CUDACC__` で .cu(nvcc/C++14)非汚染。test_bitnet_gpu 緑 (重み/GPU 不在時 [SKIP]) |
| 拡張 (A) | A | **同一性条件付きタグ生成** (character-bible を条件入力) | `bitnet.hpp` 拡張 + `character.hpp` 結線 + **同一性条件付きデータ** (dataset-spec §13) | ✅ 完了 (機構 = (a-1) prompt prefix `<bos> identity <sep> scene <sep> target <eos>`・`<sep>` 2回流用で vocab/tokenizer/アーキ無変更)。A1 §13 データ 5000ペア (identity/scene 分離・identity⊆target・validate 全件0)・A2 混合訓練 + **identity retention 0.947**・A3 `generate_with_identity` + CharacterBible 結線。identity golden 突合 logits corr 1.0 / greedy 5/5 一致。test_bitnet_infer 緑 |
| 評価 (B) | B | **アニメ品質スコアラ** = §11 蒸留 QA スコアラ (生成画像を採点 → A へ FB)。数・位相 QA は二段構成 (Stage 1 = WD14 異常タグ soft ゲート / Stage 2 = NPU 正確カウント) | `src/infer/quality_gate.hpp` (Stage 1) + 将来 `src/infer/` + OV (clip.hpp/wd14.hpp グルー流用) | 🔄 **Stage 1 完了** (`QualityGate`: 既存 Wd14Tagger 出力 [1,N_TAGS] を消費する OV 非依存 soft QA ゲート。`extra_arms`/`multiple_heads`/`bad_anatomy` 等 18 異常タグを軸 (`AnomalyAxis` Hands/Limbs/Head/Eyes/Ears/Mouth/Digits/GlobalAnatomy) で束ね、CSV から名前駆動で index 解決・WD14 版差異に頑健。`evaluate` は閾値超を hit 収集し soft flag のみ・棄却なし。`DIGITS_UNCOUNTABLE` キャラは Digits/Hands 軸スキップ。実 selected_tags.csv で 18/18 解決・test_quality_gate 5 サブテスト緑 (OV 無し開発機で実走)。flag は B-5 FB ループ入力 + 蒸留 teacher ラベル生成器)。**NPU 実行性 probe 済 → scorer_device=NPU 妥当** (`scripts/dollma_probe_quality_scorer.py`: 純 conv backbone 11.18M で NPU 448² 4.62ms < iGPU 5.48 < CPU 8.35、NPU/CPU 0.55x。WD14 268ms は Window Attention 由来と切り分け確定・純 conv は NPU フレンドリー)。**Stage 2 切り出し** (NPU 正確カウント検出 = DWPose 等・指の過剰カウント。モデル入手前提・別タスク。WD14 に `extra_digits`/`fused_fingers` が無く `fewer_digits` のみ = 語彙が二段構成を裏付け)。**前提: 実スコアラ本体を純 conv で設計** (attention head は NPU 不利に戻す) |
| 圧縮 | 5 | Ternary GEMM (重み{-1,0,+1}) — **圧縮実験** (目的ではない) | `src/kernels/ternary_gemm.cu` | ⏳ 降格 (dense が動いた後の研究軸) |

**Phase 4 完了の定義**: user text → danbooru タグ変換が C++ (CPU/GPU) で動き、品質が
DanTagGen / Qwen2 蒸留基準に遜色ないこと。さらに **A (同一性条件付け)** が character-bible
入力で機能し、**B (= §11 蒸留 QA スコアラ)** が FB ループを閉じること。ternary 化は完了条件
には含めない (圧縮実験として別評価)。NPU は **CLIP-L 専任が確定**、B を NPU に載せるかは
conv probe 次第。

---

## キャラクター品質・一貫性 (画像生成後の段、Phase 2+ で並行)

キャラを「コマ間でブレさせない」「手指を崩さない」ための段。設計は
`docs/character-bible-spec.md` 参照。authored 層 (character.hpp) は完了済みで、
以下は learned 層・後処理段として段階的に実装する。

| 項目 | 内容 | spec | 時期 |
|---|---|---|---|
| 切り抜き (マッティング) | 透過 PNG 出力。anime-segmentation (isnet 系)。乗せる HW は probe 比較 | §3, §9 | 🔄 CPU 完了 (M0-4): **ISNet-anime 採用** (CPU 1.14s vs BiRefNet 15.7s・髪 soft α 最適・Apache-2.0)。OV IR FP32/FP16 静的化 (ONNX↔OV 6.1e-6)、`encode_png_rgba8` 追加、`src/infer/matting.hpp` Matter グルー + compose_rgba (soft α・ストレート) + test_matting (golden IoU/MAE は OV 有効ビルドで実走)。**M-5 ✅ 完了** (研究機 4 device 実測 `scripts/dollma_probe_matting_device.py` / `_matting_device_report.json`): iGPU(Xe) **99.96ms** 最速 < NPU 142.96 < CPU 204.20 < RTX5080-OV 220.47 → **matting_device = "iGPU" 確定**。ISNet は純 conv-UNet ゆえ NPU も 1024² で compile 成功 (WD14 Window Attention と対照) だが iGPU が上。RTX5080 は OV 遅延+拡散専任で除外。**M-6 (生成器結線) ✅ 完了**: 純 cpp `IMatter` + OV 隔離 factory (`src/server/matter_runner.*`) + `matting_postprocess.hpp::encode_png_maybe_transparent` (matter null/サイズ不一致/例外で不透明フォールバック) を Txt2Img/Pipeline 両生成器に結線・`IImageGenerator::set_matter` 後付け注入で 3 段 DI を跨ぐ所有権を解消・CLI `--no-matting` (既定 ON)・device 既定 `GPU.0`。開発機 (OV 無 stub) で test_matting 3 サブテスト緑 + 非回帰確認。**研究機 (isnet IR 配置済) で end-to-end 実走確認済** ✅: 実 Matter golden(CPU) IoU=1/MAE 1e-8、`dollama --prompt` → `matting: GPU.0` + color_type=6 透過 PNG 出力 (配管疎通)、iGPU(GPU.0) 実推論 108.8ms (M-5 一致域)・実アニメで前景 25.4%・soft α ありの意味あるマスク。意味ある透過キャラ出力は実 SDXL 重み配置が前提 (配管完了) |
| 手指 L1 (予防) | 品質ネガティブ注入 (`default_quality_negatives`) | §10 | ✅ 完了 (器) |
| 手指 L2/L3 (修復・検査) | 手検出→インペイント再生成 / 指数を `digits_per_hand` と照合し再生成 | §10 | Phase 2 |
| 学習層 `CharacterMemory` | 生成→学習→FB ループ。記憶層 (seed/pose 蓄積・重心) → **蒸留 QA スコアラ (= Phase 4 B。NPU 載せは conv probe 次第)** → fine-tune | §11 | Phase 2/3 |
| 背景プラグイン | 外部背景生成 (Grok/Gemini/SD) + 自動合成。宿主は HTTP サーバ層 | §9 | Phase 3 |

---

## マイルストーン一覧

| マイルストーン | 内容 | 目標 |
|---|---|---|
| **M1** | Phase 1 完了: C++ でタグ抽出ループが動く | Phase 1 全完了後 |
| **M2** | Phase 2 完了: フル C++ で画像生成 | SDXL カーネル完成後 |
| **M3** | Phase 3 完了: HTTP 経由で画像生成 | サーバー完成後 |
| **M4** | Phase 4 完了: 自作タグ生成 LM (dense) + A 同一性条件付け + B 品質スコアラ込みの end-to-end | dense 訓練 → A/B 後 |

---

## 技術的リスク・未確定事項

| 項目 | リスク | 対策 |
|---|---|---|
| SDXL UNet 自作カーネル | 実装規模が最大・デバッグが困難 | 小さいモデル (64×64 latent) で動作確認してからスケール |
| ~~タグ生成 LM 基礎データ~~ | quality / quantity 未確定 | ✅ #1 で 5,000 ペア確保・解消 |
| A 同一性条件付けデータ | 現 dataset は同一性を target 除外 (dataset-spec §4) → 条件付きペアが無い | dataset-spec §13 で新規設計 (dataset-curator) |
| B 品質スコアラの正解ラベル | 「良い絵」の教師信号をどう得るか (難問) | §11 の合格/不合格蓄積 + 大型評価器を teacher に蒸留 |
| ~~B の NPU 実行性~~ | ~~conv 系は NPU で遅い前科 (iGPU 8x・probe4)~~ | ✅ 解消 (probe 済)。純 conv は **NPU 最速** (448² 4.62ms・NPU/CPU 0.55x)。WD14 268ms は Window Attention 由来と切り分け。scorer_device=NPU・実スコアラは純 conv 設計が前提 (`dollma_probe_quality_scorer.py`) |
| safetensors パーサー | バイナリ仕様の正確な実装が必要 | 既存仕様書とテストファイルで検証 |
| Linux 対応 (HTTP) | ~~Winsock2/POSIX 二重実装~~ | cpp-httplib がクロスプラットフォーム吸収 → 解消 |

---

## 将来の探索テーマ (バックログ)

ロードマップ本線には載せないが、研究価値があり時期未定の項目。

| テーマ | 概要 | 評価 |
|---|---|---|
| **MoE × HW 分散配置 / アンサンブル・オブ・スペシャリスト** | **得意分野の違う完結 LM を HW ごとに分散**し並列協調する (例: Qwen2/CPU=NL 意図解釈・自作 LM/GPU=タグ語彙共起・A=同一性注入)。dollama の芯 (全 HW 協調) に直結。**◎ 価値高**。設計の要点: ① 強みが**相補的**であること (出力を側面で分担=外見/ポーズ構図/同一性 → character-bible 層構造と対応)、② **束ね役**が要る (concat / 多数決 / **アービタ = Model B 品質スコアラ**)、③ 「別完結モデルを丸ごと配置」する粗粒度 MoE なので**古典 MoE のトークン単位動的ルーティング (NPU 静的形状で不可) を回避**できる。④ 導入順: **まず GPU 本線 A を動かして弱点を実測 → 穴を埋める専門家だけ B で束ねて足す** (穴を知る前のメッシュ化は過剰投資)。⑤ 注意: 全待ち設計だとレイテンシ=最遅モデル・強み重複は冗長 | 時期未定 (A 計測後) |
| 拡散 UNet の timestep-expert | eDiff-I / DiT-MoE 系。20step を「初期=構図 / 後期=ディテール」で別エキスパートに分割。Phase 2 で dense を動かした後に検討。 | ◯ |
| タグ生成 LLM の **内部** MoE 化 | **1モデル内部にエキスパート+トークン単位ルーティング**を持つ古典 MoE の話 (上行の「別完結モデル分散」とは別物)。単体 30–100M 規模には過剰 (MoE は数B〜で真価)・ルーティング損失/分岐コストが見合わない・NPU 静的形状とも非両立。**得意分野別の協調がしたいなら上行 (別完結 LM の HW 分散) を採る**。 | △ 過剰 |
| **NPU 骨格/部位検出による解剖メタ整合検査** | 生成画像を NPU で部位検出し、**「数・位相」だけ**を `CharacterIdentity` の宣言値と照合 (指数/四肢の本数・有無/重複欠損/左右の本数対称)。**角度・比率・ポーズ自然さは見ない** (2D のパース・デフォルメで誤検出するため)。L3 指数検査の一般化。NPU は拡散中 3.8s ほぼ遊休 → 裏で実質ゼロコスト採点。詳細は character-bible-spec §11。 | **◎ 価値高 (Phase 2)** |
| **MCP 公開 / Claude 連携** | (a) dollama を **MCP サーバとして公開** → Claude 等が画像生成をツール呼び出し。Phase 3 の OpenAI 互換 HTTP サーバ (cpp-httplib) の薄いラッパで済み、自作・単一バイナリの美学と両立。(b) **プロンプト解析を Claude に**やらせる案は研究コア (自作 BitNet) の代替ではなく、BitNet の**訓練データ収集 / 品質上限の評価基準**として位置づける (Qwen2 蒸留と同じ役割)。 | (a) ◎ Phase 3 / (b) Phase 4 のデータ・評価文脈 |
| **タグ生成 LM 学習強化プログラム (C/B/A/D/F)** | 蒸留 4 路線が全滅 (recall は学習レシピでなくデータ/容量で頭打ち・training-spec §10–12) した後の、recall 底上げの統合プログラム。**5 施策は独立メニューでなく依存連鎖**。詳細は下記サブセクション。 | ◎ 本命 (Phase 4 継続・C が起点) |

**共通の制約**: NPU は静的形状のみ → 古典的な token-level dynamic routing は不可。
回避策は (a) 全エキスパート dense 計算 + マスク合成 (容量メリット消失)、または
(b) リクエスト/スタイル単位の**固定ルーティング**で形状を静的に保つ。後者は
「キャラ系統ごとに別エキスパート」運用と噛み合い NPU 制約とも両立する。

### タグ生成 LM 学習強化プログラム (C/B/A/D/F)

**背景 (診断)**: 蒸留 4 路線 (D2/D4 hard CE 混合・D5 共起 soft・D6 外部教師 TIPO) が
**いずれも top10 recall を動かせなかった** (training-spec §10–12)。正則化も soft label も
外部知識転移も効かなかった = **33M モデルは 4,500 ペアから学べる分を学び切った**。recall を
上げる筋は **① データ ② 容量 ③ そもそもの測り方** に限られ、レシピ側 (蒸留含む) は枯れた。

**見落としやすい罠**: 現行評価は固定 val 500・**テンプレ 3 種**・one-hot recall@10。これは
「3 テンプレに合うか」を測っており「実ユーザーの自由文に汎化するか」ではない。D2 の
Qwen2 文多様化が「過学習悪化」に見えたのも、LLM 文が val のテンプレ分布から外れたためで
**proxy 由来の見かけ**だった可能性が高い。→ **0.777 を 0.79 にする作業の研究価値は薄い**
(素の text→tags は DanTagGen が実現済・新規性が薄いと §Phase4 レビューでも確認)。上げるべきは
proxy 数値でなく**実用品質**。

**5 施策と依存連鎖** (独立メニューではない):

```
   C 評価を実目標に作り直す ──────────── 全施策の前提・最初・低コスト
        │ これで初めて B/A/D の効果が「実は何点か」で測れる
        ▼
   B 入力(自然文)多様化 ──┐  C なしだと D2 の二の舞 (proxy 上は悪化に見える)
   A 実ペア増 8200→数万 ──┤  ※法務/ToS ゲート (PL 経由・dataset-spec §1.3)・リードタイムあり
        │                 │
        ▼                 │
   D 容量 33M→60-100M ─────┘  D は A と必ずセット (単独は過学習・gap 1.09 が開く)
        │
        ▼
   F 品質フィードバック学習 ── 到達点・本命。B スコアラ→生成→SDXL→採点→fine-tune
```

| 施策 | 機構 | 依存 / ゲート |
|---|---|---|
| **C** 評価作り直し | テンプレ外の多様な val (LLM/人手・**タグは実 danbooru のまま**=LLM にタグを推測させない不変方針) で set-F1 / recall@k / Jaccard。**生成ベース** (greedy 生成タグ集合 vs gold) も測る (現行は teacher-forcing recall)。固定 val 500 は**不変**で残し**加算的**に追加 (#1/D5/D6 突合の再現性保全) | ✅ **完了** (C-1〜C-4 / training-spec §13 / dataset-spec §14) |
| **B** 入力多様化 | タグ集合固定 (tags-stay-real) で自然文を多様化。テンプレ 3 種の偏りを解消し実世界汎化を狙う。`source:"llm_distill"` スキーマ (dataset-spec §12/§15) 流用 | ✅ **件数拡大まで完了** (Claude 著述 Replace **500→2,000**・diverse F1 が件数増で単調拡大=**スケール則**・全判定軸頑健・著者交絡は D2 で否定・**本線昇格は未決** / training-spec §14 / dataset-spec §15) |
| **A** 実ペア増 | danbooru harvest 8,200→数万 posts・ユニークペア天井 6,400 を突破。recall 天井そのものを上げる唯一の確実筋 | **法務/ToS ゲート** (dataset-spec §1.3: 5,000 超は PL 経由専門家確認) |
| **D** 容量増 | 33M→60-100M (設計レンジ §LLM の将来像 内・RTX5080 で訓練)。A で天井を上げてから取りに行く | A 必須 (単独は過学習) |
| **F** 品質ループ | B 品質スコアラ (アニメ品質・NPU/CPU・§技術リスク表) を作り、生成プロンプト→SDXL 画像→採点→報酬で LM を fine-tune (CharacterMemory ループ)。recall でなく「良い絵を生む」方向に学習軸を移す | Phase 2 (済 11.3s) + B スコアラ実装 |

**C と F は同じ軸の両端**: C = より良いオフライン proxy、F = 本物のオンライン信号 (良い絵か)。
このプログラムの背骨は**物差しを proxy→実品質へ動かすこと**で、recall という枯れた数値から
卒業する話。**推奨着手順**: ① C 着手 + A の法務ゲートを PL に並行起票 (互いに待たない) →
② C 完了後 B を新 val で評価 (D2「悪化」の決着) → ③ 法務 GO 後 A+D 一括 → ④ F。
**評価**: ◎ 本命だが、実装は CLAUDE.md ルール (プランモード設計→承認→PL 振り分け) に従う。

> **C 完了 (2026-06-23) — 物差し変更が D5 判定の符号を反転させた**: diverse-val (テンプレ外
> 自然文・tags-stay-real) + 生成 set-metrics + eval-only ハーネス + seed sweep を実装 (C-1〜C-4)。
> **旧 proxy (テンプレ teacher-forcing recall@10) では D5 (soft-label KL) が最下位 (0.667)
> だったのが、新 proxy (diverse 生成 F1) では最上位に反転** — C の仮説「テンプレ recall が D5 の
> 実力を隠していた」を実データで裏付け。seed sweep で D5−#1 の diverse F1/Jaccard delta は
> **全 4 seed で正・各 seed の paired CI が 0 を除外** (D6 の recall 上振れが符号反転した seed
> ノイズだったのと対照的) = **小幅だが統計的に頑健**な実効果 (delta +0.009〜+0.012 F1・絶対値は
> #1 の seed 分散帯以下)。**確定事項**: recall@10 (テンプレ) を主要数値から退役させ、**diverse 生成
> set-F1 を新オフライン主指標**に据える。**未決**: D5 を本線昇格させるかは新物差しの下で別途判断
> (絶対値はなお ~0.18–0.22 と低く edge は小さい → A 実ペア増 / D 容量増と束ねて再評価が妥当)。
> 本番重みは #1 据え置き・無改変。詳細 training-spec §13。

> **B パイロット完了 (2026-06-23) — 入力多様化が新物差しで大幅かつ頑健な改善を出した**:
> C で据えた diverse-val + 生成 set-F1 物差しの上で、施策 B の最初のパイロット (タグ固定 =
> tags-stay-real・自然文だけ多様化・**Claude 著述 Replace 500**・総件数 4,500 維持) を実施。
> **diverse 生成 macro F1 が #1 を大幅に上回る** (diverse_a 0.1800→0.2675 / diverse_b
> 0.1921→0.3039・in-dist pairs.val は −0.009 で退行なし・legacy recall ≈同値)。seed 頑健性
> sweep (4 seed・6ep paired) で delta (B−#1) は **全 4 seed 正・分散帯の 4–6 倍・各 seed の
> paired CI が 0 を除外** = 判定 (a)(b)(c) すべて成立 (D5 は (b) 不成立の小幅・D6 は符号反転
> seed ノイズだったのと**桁違いに大きく頑健**)。**著者分布交絡は否定**: 旧 D2 (Qwen2 著述) を
> diverse-val 再採点 (B-0) しても同等に改善 (diverse_a 0.2701 / b 0.3134) → 「Claude train が
> Claude test に似て上がった」では説明できない = 多様化そのものの効果。**旧 proxy では D2 同様
> 却下されたはずで、C の物差しなしには可視化されなかった** (依存連鎖 C→B の実証)。**未決**:
> 本番重みは #1 据え置き・別名 `bitnet_dense_diverse_b` 出力。絶対値はなお diverse F1 ~0.26–0.31
> と低帯域 → B 著述件数拡大 (500→数千) / A 実ペア増 / D 容量増と束ねて本線昇格を再評価が妥当。
> 詳細 training-spec §14 / dataset-spec §15。

> **B 件数拡大完了 (2026-06-24) — 入力多様化のスケール則を確認**: パイロット (Replace 500) と
> 同方式・同物差し (diverse-val 生成 set-F1)・同 sweep で**著述件数だけを 2,000 に拡大** (Replace で
> 総件数 4,500 維持・著述 2,000 + synthetic 2,500・tags-stay-real)。**diverse 生成 macro F1 が件数増で
> 単調に拡大** (diverse_a 0.2675→0.3212 / diverse_b 0.3039→0.3670)・in-dist は誤差内据え置き
> (out-of-template だけ伸びる=汎化方向)。seed sweep (4 seed) で delta (B-2−#1) は diverse_a
> **+0.1472±0.0102** / diverse_b **+0.1788±0.0029** = 500版 (+0.096/+0.126) の ~1.4–1.5x に拡大
> しつつ seed sd は縮小 (効果が強まり頑健性も増加・全判定軸成立)。**頭打ちは見えない**。**未決**:
> 本番重みは #1 据え置き・別名 `bitnet_dense_diverse_b2000` 出力。絶対値はなお diverse F1 ~0.32–0.37
> と低帯域 → A 実ペア増 / D 容量増と束ねて本線昇格を再評価の既定方針を維持。詳細 training-spec §14.8
> / dataset-spec §15.6。

### ポーズデータ取得方針 (探索テーマの補足)

上記「解剖メタ整合検査」とは別に、**生成側のポーズ語彙/多様性**を増やすデータ源の方針。
動画は QA (カウント検査) には不要 — あくまで ControlNet OpenPose 条件 + §11 記憶層の
pose バイアス源としての「ポーズ辞書」用途。オフラインのバッチ前処理で、リアルタイム
パイプライン (NPU 7.85ms 枠) には入らない。

- **本命: 3D モーション (Mixamo / VRM / MMD)**。任意アングルにレンダして
  (2Dポーズ + 参照画像) ペアを自動生成。実写ギャップ無し・正解骨格が既知・権利クリーン。
- **実写↔2D ギャップ**: 実写動画のポーズは硬く 2D の誇張/パースが乗らない → 転写すると
  棒立ちになりやすい。アニメ系 (DWPose) を当て、3D 合成を優先。
- **法務 (日本)**: 著作権は **30条の4 (情報解析)** が ML 学習を広く許容。さらに
  **派生した骨格座標だけ保持しピクセルは破棄**すれば著作物から遠く、リスクは小さい。
  ただし **30条の4 は契約(ToS) を上書きしない**。
- **YouTube**: 規約が自動コンテンツ採取を禁止 → `yt-dlp` も、埋め込み/画面キャプチャ/
  ヘッドレス自動操作/`captureStream` も「取り口」が違うだけで**規約上は同じく灰色〜黒**。
  「保存しない/別ブラウザ経由/再生のみ」は著作権側には効くが**契約違反は解消しない**。
  公式 IFrame は cross-origin サンドボックスで `canvas`/`captureStream` が tainted になり
  フレーム取得不可 — 回り込むには画面キャプチャ=保護回避になる。
- **クリーンな道**: 「再生→その場で骨格抽出→座標だけ保存→ピクセル破棄」という手法自体は
  優秀。向ける先を **CC ライセンス動画 / 正規に再生権を持つ素材** に限定すれば著作権も契約も
  クリア。本命は最初から配布されている 3D モーション。(商用化前は専門家確認)

---

## 参照ドキュメント

- `docs/pipeline-spec.md` — スレッド構成・キュー設計・タイミング試算
- `docs/tensor-spec.md` — Tensor クラス詳細設計
- `docs/http-api-spec.md` — HTTP API 仕様
- `docs/cpu-topology.md` — CPU コアアフィニティ設定
- `docs/archives/investigation-log.md` — probe1〜10 調査ログ

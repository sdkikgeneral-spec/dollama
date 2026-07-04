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
| 2-6c | 拡散 backend プラグイン枠: prompt→RGB 境界を純 cpp interface `IDiffusionBackend` に切り出し registry 化。品質天井は自作カーネルでなく拡散アーキ (重み) にあるため、芯 (自作 Tensor/GEMM/Attention/CUDA) を汚さず新アーキへ口を開ける。ComfyUI 的 breadth は追わず「2D キャラ生成に要るアーキだけ差し替える」棲み分け | `server/diffusion_backend.{hpp,cpp}` (registry `make_backend`), `server/sdxl_backend.hpp` (OV+CUDA 隔離), `server/sd35_backend.hpp` (拡張点 stub), `server/backend_image_generator.hpp` (共通後処理), `server/cli_generate.hpp` (段1 DI 差し替え), `meson.build` | test_diffusion_backend (純 cpp・fake backend で PNG/DI/registry 検証) | ✅ 完了 (2026-07-02)。`IDiffusionBackend` = generate(prompt/negative→RGB8)+info()+model_id() の純 cpp interface (CUDA/OV 非露出)。`make_backend` nullptr 契約 (未知名/OV無/構築失敗→nullptr で段2/3 フォールバック)。`SDXLBackend` は従来 `Txt2ImgGenerator` の encode+拡散中核を移設 (HAVE_OPENVINO ガード)、共通後処理 (解像度 reject/seed/採点ログ/matting PNG化) は `BackendImageGenerator` へ分離し新アーキで無改修再利用。`SD35Backend` は latent 16ch/T5必須の info を返す型レベル実証 stub (generate throw)。段1 DI を env `DOLLAMA_BACKEND` (既定 "sdxl") で backend 選択・NPU→CPU フォールバック踏襲。旧 `txt2img_generator.hpp`/`test_txt2img` は回帰用に残置。**全 46 test 緑** |
| 2-6d | 実 checkpoint 差し替え = **アニメ特化 SDXL を 3 preset 対応** (NoobAI-XL / Animagine XL 4.0 / Illustrious XL)。品質天井は素 SDXL base 1.0 (2023) にあり、最新アニメ特化 checkpoint 載せ替えが絵の質の最大レバー。3 つとも SDXL アーキゆえ `SDXLBackend` は**無改修**、違いは重みのみ → backend を増やさず「重みセットを選べる」化。`BackendConfig.preset` (既存の未使用拡張点) を preset→重みパス解決に使う | (計画) `server/diffusion_backend.cpp` (preset→パス解決グルー・`find_model_xml`/`resolve_path` 流儀), `server/cli_generate.hpp` (env `DOLLAMA_BACKEND_PRESET`), `THIRD_PARTY_NOTICES.md` (3 件帰属), 変換スクリプト流用 (SDXL テンソル名マッピング) | test_diffusion_backend に preset 分岐テスト追加 | 🔲 **計画確定・未着手** (2026-07-02)。**段取り**: ① 3 checkpoint ライセンス確認 (NoobAI/Animagine=Fair AI Public License 1.0-SD 商用可・帰属のみ / Illustrious は版依存で 1.0+ 独自条項要確認)・自ホスト参照+再配布なしゆえ Fair AI のコピーレフトは非トリガー → THIRD_PARTY_NOTICES 帰属で足りる ② 各 checkpoint の unet/vae を自作ローダー名レイアウトへ変換 (~6-7GB×3≒20GB・同時常駐せず 1 つずつロードで VRAM 問題なし) ③ preset 解決グルー実装 ④ `dollama --prompt`→透過PNG→WD14 ラベル化でループ確認・撮り比べ。danbooru タグ native ゆえ自作 LM のタグ出力と直結 ([[project-generation-quality-bar]] 最大レバー / [[project-output-quality-over-features]])。**DL+変換+生成の実走は GPU セッション (PL+cpp-implementer/gpu-benchmarker) で別途** |

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
| 拡張 (A) | A | **同一性条件付きタグ生成** (character-bible を条件入力) | `bitnet.hpp` 拡張 + `character.hpp` 結線 + **同一性条件付きデータ** (dataset-spec §13) | ✅ 完了 (機構 = (a-1) prompt prefix `<bos> identity <sep> scene <sep> target <eos>`・`<sep>` 2回流用で vocab/tokenizer/アーキ無変更)。A1 §13 データ 5000ペア (identity/scene 分離・identity⊆target・validate 全件0)・A2 混合訓練 + **identity retention 0.947**・A3 `generate_with_identity` + CharacterBible 結線。identity golden 突合 logits corr 1.0 / greedy 5/5 一致。test_bitnet_infer 緑。**実ペア増 a12k 4 seed sweep でクローズ (二分結論・training-spec §9.10)**: ① **diverse-val 生成 F1 は 12k で seed ノイズ** (4 set/metric とも判定 NO・seed 42 のみ符号反転・across delta −0.015±0.032 < #1 帯 sd 0.022 = D6 と同型・施策 B ~2,000 飽和と整合)。② **identity retention は頑健に 0.975 達成** (across-seed 0.9748±0.0010・全 seed・base ~0.58–0.63 から) = **identity 条件付けの機能基盤**。→ A の効果は diverse-val F1 でなく retention。a25k は回さず未使用保持・本番 #1 即時差し替えなし・`[B-merge-at-A]` でまとめ焼き |
| 評価 (B) | B | **アニメ品質スコアラ** = §11 蒸留 QA スコアラ (生成画像を採点 → A へ FB)。数・位相 QA は二段構成 (Stage 1 = WD14 異常タグ soft ゲート / Stage 2 = NPU 正確カウント) | `src/infer/quality_gate.hpp` (Stage 1) + 将来 `src/infer/` + OV (clip.hpp/wd14.hpp グルー流用) | 🔄 **Stage 1 完了** (`QualityGate`: 既存 Wd14Tagger 出力 [1,N_TAGS] を消費する OV 非依存 soft QA ゲート。`extra_arms`/`multiple_heads`/`bad_anatomy` 等 18 異常タグを軸 (`AnomalyAxis` Hands/Limbs/Head/Eyes/Ears/Mouth/Digits/GlobalAnatomy) で束ね、CSV から名前駆動で index 解決・WD14 版差異に頑健。`evaluate` は閾値超を hit 収集し soft flag のみ・棄却なし。`DIGITS_UNCOUNTABLE` キャラは Digits/Hands 軸スキップ。実 selected_tags.csv で 18/18 解決・test_quality_gate 5 サブテスト緑 (OV 無し開発機で実走)。flag は B-5 FB ループ入力 + 蒸留 teacher ラベル生成器)。**NPU 実行性 probe 済 → scorer_device=NPU 妥当** (`scripts/dollma_probe_quality_scorer.py`: 純 conv backbone 11.18M で NPU 448² 4.62ms < iGPU 5.48 < CPU 8.35、NPU/CPU 0.55x。WD14 268ms は Window Attention 由来と切り分け確定・純 conv は NPU フレンドリー)。**Stage 2 切り出し** (NPU 正確カウント検出 = DWPose 等・指の過剰カウント。モデル入手前提・別タスク。WD14 に `extra_digits`/`fused_fingers` が無く `fewer_digits` のみ = 語彙が二段構成を裏付け)。**前提: 実スコアラ本体を純 conv で設計** (attention head は NPU 不利に戻す)。**実スコアラ本体: E-1 (研究機コーパス生成・実 SDXL PNG 180枚) → B-3b 蒸留訓練 ✅ 完了 (2026-06-28)** = `scripts/train_scorer.py` で ScorerNet (純 conv 11.18M・出力 [1,1+8]) を E-1 コーパスに蒸留 (axis 8軸 BCE soft・train_loss ep0 0.0933→ep5 0.00747・CPU 95s・seed bitwise 一致・実画像実ロード・quality 全 null → B head 凍結)。出力 `data/scorer/scorer_net{,_fp32}.safetensors`・test_dollma_train_scorer 9/9 緑 (CLAUDE.md 計測表参照)。**B-3c (OV 変換) ✅ 完了 (2026-06-28・研究機)** = `scripts/dollma_convert_scorer.py` で OV IR 静的化 `[1,3,512,512]→[1,9]` FP32/FP16・**NPU compile+推論成功 (8.32ms・NPU/CPU 0.711x = 純 conv が NPU に載る B の芯を実証)**・iGPU 6.28ms 最速・PyTorch vs OV FP32 1.34e-05 (eval() で BN 正常)・golden `src/tests/data/scorer/golden_scorernet.safetensors` 確定。**残: B-3d (C++ グルー `src/infer/quality_scorer.hpp`・clip/wd14 対称・要 OV 有効ビルド) → B-3e (golden 突合 test) → B-5 (FB ループ)。quality head は美的モデル (waifu-scorer-v4-beta) のライセンス整理後に再訓練で有効化** |
| 圧縮 | 5 | Ternary GEMM (重み{-1,0,+1}) — **圧縮実験** (目的ではない) | `src/kernels/ternary_gemm.cu` | ⏳ 降格 (dense が動いた後の研究軸)。なお **INT8 dense CPU 推論は別途完了** (`src/infer/bitnet_int8.hpp`・重みのみ per-row INT8・量子化耐性高く greedy 完全一致・training-spec §15) |

**Phase 4 完了の定義**: user text → danbooru タグ変換が C++ (CPU/GPU) で動き、品質が
DanTagGen / Qwen2 蒸留基準に遜色ないこと。さらに **A (同一性条件付け)** が character-bible
入力で機能し、**B (= §11 蒸留 QA スコアラ)** が FB ループを閉じること。ternary 化は完了条件
には含めない (圧縮実験として別評価)。NPU は **CLIP-L 専任が確定**、B を NPU に載せるかは
conv probe 次第。

---

## Phase 5 — ポージング・パース (コマ単位の自由な画角)

**動機 (確定)**: 漫画は**コマごとに角度・パース (あおり/俯瞰/前後の圧縮 = foreshortening)・
ダイナミックポーズ**が変わる。これがクリエイティブの核なのに、拡散モデルが最も苦手とする
領域 (ユーザー指摘 2026-06-29)。

**問題の切り分け (なぜタグ生成 LM では解けないか)**: 「学習にないパターン」には ①未見の
**組み合わせ** (補間・合成、生成可) と ②分布の**外** (外挿、基本不可) があり、創造性の実体は
①。パース崩壊の原因は 2 つ:

- **データ分布の偏り**: danbooru は「立ち・正面・目線・バストアップ」に激しく偏り、`from_below` /
  `foreshortening` / `dynamic_pose` 等の語彙は**存在するが学習サンプルが桁違いに少ない** →
  タグ LM が emit しにくく (施策 B の diverse-val 汎化問題と同型)、与えても拡散が崩す。
- **拡散は 3D 構造を持たない**: SDXL は 2D 画像統計の補間で、カメラ・骨格・奥行きの内部モデルが
  無い。極端な角度ほど補間近傍が枯れて解剖が破綻する。これは**幾何の問題で、タグでは解けない** →
  **構造を外から与える**のが筋 (industry 標準 = structural conditioning)。

**3 段の依存連鎖 (5-1 → 5-2 → 5-3、独立メニューではない)**:

```text
   5-1 崩壊境界の実測 (probe) ──── まず「どの画角でどれだけ崩れるか」を地図化・低コスト
        │ これが 5-2/5-3 の必要量と投資判断を決める (Aで足りるか / Bまで要るか)
        ▼
   5-2 img2img 幾何注入 ────────── 安い・既存パス (iGPU VAE encode 79ms) 流用・新カーネル不要
        │ 5-2 で救えない強パースが実需として残るなら…
        ▼
   5-3 構造条件付け (ControlNet) ── 重い本丸・自作 UNet に制御ブランチ増設 (2-5 相当の中規模)
```

| 段 | # | 実装物 | 内容 / 流用 | 状態 |
|---|---|---|---|---|
| 実測 | 5-1 | **崩壊境界 probe** | 固定キャラ × 画角/パースタグ (`from_above`/`from_below`/`dutch_angle`/`foreshortening`/`perspective`/`dynamic_pose`/`reaching_towards_viewer`) の直積を自作 SDXL で生成 → **既存 QualityGate (Phase 4 B Stage 1) の異常タグ hit 率** (`bad_anatomy`/`extra_arms`/`extra_digits`/`multiple_heads`) を崩壊スコアとして画角別に集計 → ヒートマップ。**採点器は新規不要** (WD14→QualityGate を流用)。`scripts/dollma_probe_pose_breakage.py` 想定 | ⏳ 未着手 (起点) |
| 安価解 | 5-2 | **img2img 幾何注入ワークフロー** | 3D ソフト/ラフ (DesignDoll/Blender/VRoid) のアタリで構図・パース・骨格を**幾何で固定** → img2img 下絵に投入 → 拡散は清書に専念 + タグ LM は同一性 (Phase 4 A) + 質感。**研究の核心 = denoising strength のスイートスポット** (低すぎ=下絵のラフさ残存 / 高すぎ=描き直して崩壊)。「ポーズ保存度 vs 清書品質」のトレードオフ曲線を実測。img2img パスは既存 (新カーネル不要・設計とパラメータ詰めが主) | ⏳ 未着手 (5-1 後) |
| 本丸 | 5-3 | **構造条件付け (ControlNet)** | 下絵すら無しで骨格 (OpenPose) / 深度 (depth) / 線画 (lineart) で構造強制。自作 UNet CUDA に**制御ブランチ** (各 down/mid の residual に制御信号を加算) を増設 + ControlNet 重みを既存 safetensors ローダーで読む + 制御入力 (骨格/深度推定) を OV グルーで 1 段 (CLIP/WD14/ISNet と同立ち位置)。VRAM: SDXL ControlNet 1 本 ~+1.2GB → 10.49+1.2 で 16GB 内。既存自作 conv/attention カーネル流用は効く | ⏳ 未着手 (5-2 で不足が確認されたら正式タスク化・決裁要) |

**他段との関係**:

- **Phase 4 B (QualityGate)**: 5-1 の自動採点器として直接流用。レア構図で崩れる前提で**後段で破綻コマを弾く/再生成**する §11 の発想と一致。
- **Phase 4 A (同一性条件付け)**: 角度を振っても顔・特徴が別人化しないための軸。5-2/5-3 と直交補完 (構造=幾何、同一性=タグ条件)。
- **LoRA (本線)**: 見た目/画風ロックは LoRA、ポーズ/構図/画角は本 Phase。これも直交。
- **「ポーズデータ取得方針」(下記バックログ)**: 5-3 ControlNet OpenPose の条件入力辞書/§11 記憶層 pose バイアス源。本命は **3D モーション (Mixamo/VRM/MMD)** からの (2D ポーズ + 参照画像) 合成 (法務・実写ギャップの整理済)。
- **character-bible §11 解剖メタ整合検査**: NPU 部位検出で「数・位相」のみ照合 (角度・比率・ポーズ自然さは見ない=パース誤検出回避)。5-1/5-3 の破綻検出と地続き。

⚠️ **HW 前提**: SDXL 実走は研究機 (RTX5080) が必須。本開発 PC (GTX1080Ti) は SDXL 本走に不向き
([[dev-pc-hardware]])。probe/スクリプトはここで書けるが**実走は研究機**。

**着手順 (推奨)**: ① **5-1 を最初にプラン化** (崩壊地図なしに 5-2 の denoise 詰めも 5-3 の投資判断も
勘になる) → ② 結果を見て 5-2 の denoise スイートスポット実験 → ③ 5-2 の残課題を見て 5-3 ControlNet を
正式タスク化するか決裁。実装着手は CLAUDE.md ルール (プランモード設計→承認→PL 振り分け) に従う。

**Phase 5 完了の定義**: コマ単位で指定した画角・パース・ポーズの透過キャラ PNG が、解剖破綻を
QualityGate 許容内に抑えて出力できること。最低ラインは 5-1 (崩壊地図) + 5-2 (img2img で強パースを
実用品質に救えることの実証)。5-3 ControlNet は 5-2 で不足が出た場合の拡張。

### 5-D — デッサンモード (トレース用あたり生成・5-3 の応用)

**動機 (確定 2026-07-04)**: 本番キャラ生成とは別に、**下書きのトレース元**として使う
「デッサン人形をポージングさせたあたり図」を出したい。CLIP STUDIO の 3D 人形モードは
① ポーズ付けが 3D 操作で面倒 ② 出力が 3DCG 臭くアニメ絵の比率・タッチと合わない、で
「いまいち」というのが起点。dollama の勝ち筋は **3D 人形をレンダリングせず、拡散で最初から
アニメ絵の比率・タッチのあたりを出す**こと (5-3 ControlNet の直接応用)。

**設計の芯 = 役割分離** (汎用 3D 人形に無い「そのキャラ比率のあたり」が新規性):

| 要素 | 担保する場所 |
|---|---|
| プロポーション (頭身・骨長) | **スケルトンの幾何** ← character-bible 頭身スペックから骨長を自動生成 |
| 同一性 (顔/髪/色) | bible タグ + 同一性条件付き LM (Phase 4 A, retention 0.975) |
| ポーズ | OpenPose キーポイント (下記 3 入力すべてここに収束) |
| 絵柄 (グレー人形) | 「gray mannequin / neutral figure」prompt preset |

**ポーズ入力 3 種 (すべて対応・ユーザー確定)** → 共通中間表現 = **OpenPose キーポイント図**:

- **2D スケルトン編集**: 棒人間を 2D ドラッグで関節操作 (3D 回転の煩わしさを回避)。Blazor UI 側。
- **参照画像からポーズ抽出**: 写真/イラストを DWPose 等でキーポイント化 → そのポーズであたり生成。
  新規 OV 推論モデル 1 段 (固定形状寄りで NPU/iGPU 配置候補・CLIP/WD14/ISNet と同立ち位置)。
- **テキスト指定**: 自然文 → prompt 直行、またはポーズライブラリ (下記「ポーズデータ取得方針」の
  Mixamo/VRM/MMD 由来) から選択。骨格を持てないので精度は上 2 つに劣る。

**拘束方式の決定 = ControlNet-OpenPose 一択** (ユーザー決裁 2026-07-04): T2I-Adapter (軽いが
精度落ち) は却下。デッサンでもポーズがふらつくと下書きに使えず、質優先 ([[project-output-quality-over-features]])。
= Phase 5-3 の ControlNet 制御ブランチをそのまま使う (新方式の追加ではない)。

**着手の段取り (質優先だが博打回避・probe→確認→実装の流儀)**:

1. **質検証 probe** (軽・gpu-benchmarker): diffusers の ControlNet-OpenPose + アニメ特化 checkpoint +
   「gray mannequin」prompt で、狙ったグレー人形のあたりが実際に出るか + 頭身をスケルトンで
   拘束できるかを数枚で確認。
2. probe 合格 → **本結線**: 5-3 の自作 CUDA UNet への ControlNet 統合 (制御ブランチ増設) +
   グレー人形 preset + 頭身→骨長マッパー + (2D スケルトン編集器は Blazor UI) + ポーズ検出器 OV グルー。

**他段との関係**: 5-3 ControlNet (基盤・同一)、Phase 4 A (同一性)、「ポーズデータ取得方針」
バックログ (テキスト入力のライブラリ源)、ui/ Blazor (2D スケルトン編集器)。
**着手は CLAUDE.md ルール準拠** (probe→質確認→プランモード設計→承認→PL 振り分け)。未着手 (5-3 に依存)。

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
| **M5** | Phase 5: コマ単位の画角・パース指定で解剖破綻を抑えた透過キャラ出力 | 5-1 崩壊地図 + 5-2 img2img 実証 (5-3 は拡張) |

---

## 技術的リスク・未確定事項

| 項目 | リスク | 対策 |
|---|---|---|
| ~~SDXL UNet 自作カーネル~~ | ~~実装規模が最大・デバッグが困難~~ | ✅ 解消 (タスク 2-5)。「小さいモデルで確認してからスケール」を完遂。noise_pred **SSIM 0.999998**・24 段ゴールデン全緑・test_unet 緑 |
| ~~タグ生成 LM 基礎データ~~ | quality / quantity 未確定 | ✅ #1 で 5,000 ペア確保・解消 |
| ~~A 同一性条件付けデータ~~ | ~~現 dataset は同一性を target 除外 (dataset-spec §4) → 条件付きペアが無い~~ | ✅ 解消。dataset-spec §13 で新規設計→A1 retention 0.947→**a12k 4 seed sweep でクローズ** (identity retention **0.975** で同一性条件付けの機能基盤確定・diverse-val F1 は seed ノイズ・dataset-spec §17.7 / training-spec §9.10) |
| B 品質スコアラの正解ラベル | 「良い絵」の教師信号をどう得るか (難問) | 🟡 前進。§11 軸ラベルは **WD14 soft ラベルで 8 軸蒸留** (B-3b〜B-3d 完了・ScorerNet 11.18M・OV/NPU 載り実証)。**quality (美的) head はライセンス整理待ちで凍結中** (waifu-scorer-v4-beta 既定・§16.9) → ここだけ未解消 |
| ~~B の NPU 実行性~~ | ~~conv 系は NPU で遅い前科 (iGPU 8x・probe4)~~ | ✅ 解消 (probe 済)。純 conv は **NPU 最速** (448² 4.62ms・NPU/CPU 0.55x)。WD14 268ms は Window Attention 由来と切り分け。scorer_device=NPU・実スコアラは純 conv 設計が前提 (`dollma_probe_quality_scorer.py`) |
| ~~safetensors パーサー~~ | ~~バイナリ仕様の正確な実装が必要~~ | ✅ 解消 (タスク 2-3)。golden 5 テンソル突合・**19.0 µs/op**・test_safetensors 緑 |
| Linux 対応 (HTTP) | ~~Winsock2/POSIX 二重実装~~ | cpp-httplib がクロスプラットフォーム吸収 → 解消 |

---

## LoRA 対応 (生成エンジン拡張・本線)

2D イラスト AI 生態系 (NovelAI / Civitai / ComfyUI) で LoRA は事実上の標準装備。dollama の
用途 (自分の OC で同人) では、tag ベースの同一性条件付け (Phase 4 A) だけでは OC をキャラ
LoRA ほど固定できず、**画風 / OC を重みベースで持つ手段としてほぼ必須**と判断
(ユーザー 2026-06-24・バックログから本線昇格)。**本 txt2img (2-6b) は完了済みゆえ依存は
クリア・着手可能**。tag ベース同一性 (A) とは直交する補完 (LoRA = 見た目 / 画風ロック・
tag = シーン / ポーズ / 表情)。研究機 (RTX5080 + CUDA) 前提。

| 段 | 実装物 | 難易度 | 状態 |
|---|---|---|---|
| L-1 | **offline merge**: LoRA を base safetensors に事前マージ (W′=W+scale·BA, scale=strength·alpha/rank) → 合成 checkpoint をロード。**自作 CUDA 推論は無改修**・前処理ツール 1 本 (`scripts/dollma_merge_lora.py`) で済み、単一バイナリ / 自作美学と完全両立。「画風 / OC を 1 組焼き込む」入口 (安く capability を得る) | 低 | ✅ **完了** (2026-06-24, `aec741e`)。host 側 safetensors 重み演算のみ (SDXL 推論不要・この PC で完結)・fp32 計算→出力 dtype round・kohya→diffusers key 写像 (base 真実源)・UNet のみ (lora_te1/te2 は warn+skip)・Conv 4D reshape 加算。test_dollma_merge_lora.py **12/12 緑** (W+scale·BA vs numpy max err 0・key 写像 golden・触れない key bitwise 不変・safetensors 往復)。合成ベンチ rank32 0.38 GB/s → 5.1GB 外挿 ~13-16s + I/O。**実 SDXL/LoRA でのマージ + 実画像確認は研究機・別タスク** (本番アセット無改変) |
| L-2 | **ランタイム LoRA**: base + ΔW=α·BA を層ごと実行時加算。生態系標準の本命ワークフロー = **生成ごとに LoRA 選択 / スタック / 強度可変 + UI**。自作 UNet に LoRA 認識 + ローダ + 低ランク積加算を追加 (自作 GEMM が扱う演算ゆえ難所ではなく手間)。SDXL 機能完全性寄りで芯とは別軸 | 中 | 未着手 |

**データ要件**: **使うだけなら学習データ不要** (既存/他者作 LoRA の .safetensors を merge / 適用するだけ)。
**自分の OC / 画風 LoRA を作る場合のみ画像データが要る** — 画像 + 各画像のキャプション (danbooru 風タグ)・
キャラは 10〜50 枚をポーズ/角度/服違いで・画風は自分の絵。これは `data/bitnet` (text→tag) とは**別データセット**で、
既存 text→tag 学習データは無駄にならない (層が違う・上流のタグ生成は LoRA 不変)。**OC の鶏卵問題** (一貫画像を出したいから
LoRA が要るのに学習画像が無い) は ① 手描き/発注の参照絵、② **記憶層ブートストラップ** (タグ+A で量産 → 品質ゲート §11 で
合格選別 → そのセットで学習) で解く。**既存資産が効く**: WD14 tagger で学習画像を**自動キャプション**・品質ゲートが
**素材キュレーター**。学習は軽量 (数十枚・rank 低・研究機で数分〜1h・offline で生成と非競合)。

進め方: **L-1 (安い capability) → L-2 (本命体験)**。character-bible-spec §記憶層 / 本訓練層の
「合格画像バッチ fine-tune (LoRA / 埋め込み / リワード)」とも接続。UI 側 (Blazor) は L-2 で
LoRA 選択 / 強度のチップ・プリセット機構を流用できる。

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
| **ガヤ (群衆) 複数人出力** | 漫画/イラストの背景モブを出したい用途。**まず案 B = 1人ずつ生成 → 各自を既存 ISNet 単一前景マッティングで透過 PNG 化 → 下流 (CLIP Studio) で重ねる**。dollama の芯 (単一キャラ + 透過切り抜き) にそのまま乗り、新規モデル不要・各キャラが完全立ち絵・分離/前後/配置自由・オクルージョン問題なし。実装は主にオーケストレーション (UI に枚数指定 → N 生成 → N 枚透過出力)。**足りなくなったら案 A = instance segmentation で 1 枚の群衆を人物分離**を追加方向で (アニメ向け instance-seg のモデル成熟度低・縁再仕上げ・隠れ部分は inpainting 必要=ハード)。一体感のある群衆を 1 枚で出しつつ分離もしたい場合のみ A。**起点は B 確定** (ユーザー判断 2026-06-24)。 | ◯ (B は低コスト・時期未定) |
| **HW 環境抽象化 / 実行モード (`--cpu` `--npu` `--dgpu` 等)** | dollama を Intel 研究機以外（**Ryzen 無印 + NVIDIA dGPU** など）でも動かすため、搭載 HW を宣言して各処理段のデバイス割り当てを切り替えるモード/フラグ体系。**NVIDIA dGPU を保つ限り CUDA 研究コアは無傷**（sm_86 再コンパイルのみ）だが、**NPU/iGPU は Intel 提供ゆえ非 Intel 環境で消える** → `--npu=none`/`--igpu=none` を宣言でき、NPU/iGPU 段を CUDA/CPU へ自動退避（フォールバックチェーン）させる。`--vram=6g` で SDXL を自動ダイエット（1024²→512²/offload）。3 層構成（HW 宣言 / 環境プロファイル / 段ごと上書き）+ 起動時解決ロジック。既存の散在 env（`DOLLAMA_MATTING_DEVICE` 等）を統一デバイス計画に集約。**AMD Radeon (ROCm) は対象外**（フラグ予約のみ・CUDA→HIP 移植は別途巨大タスク）。設計詳細・対応環境マトリクスは [docs/hw-environment-spec.md](hw-environment-spec.md)。下行「遠隔 HW ノード」は本抽象を**LAN 越しの別マシン**まで延伸した姉妹テーマ（こちらは 1 台内のデバイス計画）。 | ◯ (Ryzen+NVIDIA 対応の核・段階導入可・時期未定) |
| **遠隔 HW ノード (LAN 越し第 2 マシンの協調)** | 余剰ノート PC (**Ryzen 7 5700/5800 = Zen3 8C/16T + RTX3060 Laptop = sm_86 / VRAM 6GB + 64GB RAM・2.5GbE LAN・NPU なし**) をプロジェクトに足す案 (ユーザー 2026-06-29)。**まず原則の切り分け**: ❌ **密結合 (1 画像の step 内でモデルを機械間分割) は不可** — attention だけ別機等は per-layer 往復でネットワークレイテンシ律速。⭕ **粗粒度 (モデル 1 個 = 1 stage を機械にまたいで置く / ジョブ単位) は成立**。**帯域は非ボトルネック**: 2.5GbE 実効 ~280 MB/s で stage 間ペイロード (latent [1,4,128,128] FP16 128KB ~0.5ms / CLIP embeds 308KB ~1ms / 1024² PNG ~1MB ~4ms) は全部 ms 級・重み 5.1GB は起動時 1 回常駐で per-request では流さない。残る制約はレイテンシ (層単位分割不可) と「5080 が速すぎて laptop に振る価値のある stage が限られる」点。**64GB RAM が VRAM 6GB の壁をほどく**: sequential offload で 6GB の 3060 でも SDXL 1024² 本番がフル解像度で回る (遅いが落ちない) → 下書き専任に縛られず本番ノード化可。**共通土台 = `RemoteNode` 抽象** (Phase 3 の cpp-httplib/json を流用し laptop を dollama 常駐サーバ化 + 主機にクライアント。上行 L278/L221「HW 環境抽象化」を *remote HW* まで延伸した形)。**設計前提 (ユーザー必須要件 2026-06-29): 協調 PC のあり・なしを簡単に切替できること** — 遠隔ノードは**完全 opt-in で既定は単機 (遠隔なしで全機能成立・現状の挙動は無改変)**、宣言 1 つ (例 `--remote-node=host:port` / 既定 `none`) で足す/外す。**不在時は遠隔に振る予定だった stage/ジョブをローカル HW へ自動フォールバック** (L278 のフォールバックチェーンと同一哲学・遠隔の有無でコードパスを分岐させず「デバイス計画に remote ノードが居るか」だけの差にする)。これにより laptop は「**無くても全機能が動き、有れば正確性が増す**」加速器であって依存先にはならない (ユーザー意図 2026-06-29: **遠隔ノードの本命は速度でなく "精度の上乗せ"**・後述③ critic の深度が遠隔の有無で graceful に劣化するだけで生成は常に成立)。その上に 2 つの消費者が乗る (排他でなく ①→③ の一本道): **① 訓練/sweep 分散 (即効・最大効率・③の de-risking)** — 3060 (Ampere/TensorCore/FP16 フル) は開発機 GTX1080Ti (Pascal・FP16 1:64 で実質 FP32 強制・[[dev-pc-hardware]]) より小型 LM 訓練が素直に速く混合精度も使える。D 容量増 80M (~3.8h・eval 律速) や 4 seed sweep は機械間通信ゼロで割れる典型 → 2 台分散で壁時計 ~半減・npz 集約は無負荷。これが sm_86 ビルド (自作 CUDA・wmma 含む) と LAN ジョブ配管を枯らし③の足場を兼ねる。**③ レビュー/critic ノード (研究本命・遠隔ノードの本命用途) = 生成は研究機 / 講評は遠隔** — producer/critic を 2 台に分離した形で、Phase 4 F (品質フィードバック学習) の *機械間* 版・MoE×HW (L271) の「アービタ = 品質スコアラ」を機械間に出した形。研究機が batch 生成 → PNG (~1MB/~4ms) を送る → 遠隔が採点 (異常 flag / 同一性照合 / **解剖・ポーズの整合**) → verdict (極小) を返す → 不合格を再生成 / 報酬で LM fine-tune。pipeline 化 (N+1 生成の裏で N を採点) も batch offline (F 訓練) も両対応・レイテンシ非律速。**RTX3060 Laptop はそれなりの計算力**ゆえ critic は安直ヒューリスティックに留まらず **DWPose 等の keypoint 抽出 / 学習済み解剖・ポーズ分類器 / 小型 VLM 講評 (64GB offload) といった本物の学習済みレビュアーを載せられる** (ユーザー指摘 2026-06-29)。**旗艦例 = 解剖/ポーズ critic を 3 Tier で段階化**し、遠隔の有無で**レビュー深度が graceful degradation する** (上記必須要件の具体形・遠隔なしでも生成は動き精度だけ下がる): **Tier A 数・位相** (指/四肢の本数・重複・欠損・左右対称) = WD14/QualityGate で軽く **研究機の遊休 NPU 常駐**・遠隔不要。**Tier B 骨格レベル** (DWPose 等で 2D keypoint 抽出 → 連結・本数・四肢貫通/関節の粗い不可能を判定) = 重い・**遠隔 (3060 で実走可)**。**Tier C 物理的妥当性** (3D リフト / 学習済み解剖 prior / VLM critic で「このカメラでこのポーズは物理的にありうるか」) = 最重・誤検出しやすい研究フロンティア・**遠隔**。**§11 解剖メタ整合検査 (L274) の線引きを継承**: A は確定スコープ、B/C は §11 が *意図的に避けた* 角度・比率・ポーズ自然さ領域 → **foreshortening/デフォルメを罰しない設計が必須** (強い 2D パースを「崩壊」と誤検出すると Phase 5 の攻めた画角を殺す)。**Phase 5 5-1 の採点器を格上げ**: 現 5-1 は QualityGate 異常タグ hit 率で崩壊スコア化 (タグ粒度) → B/C critic はより強い崩壊検出器になり 5-1→5-2/5-3 の投資判断を支える。増分は stage 分割プロトコル + scheduler (5080 が拡散で詰まる裏の遊休へ何を逃がすか = 芯「全 HW 使い切り」の機械間版)。**着手順 ①→③** (① が独立した見返りを持ちつつ③の sm_86 ビルド/LAN 配管リスクを潰す)。**HW 前提**: SDXL 本走は研究機 (RTX5080) 必須だが、① の小型 LM 訓練・sweep と ③ の critic は 3060 laptop で可。 | ◎ (①即効・③=遠隔の本命は精度の上乗せ・無くても動く graceful degradation・①→③一本道・時期未定) |
| **CPU 側 LM 推論の速度最適化** | ✅ **Tier 1 完了 (2026-06-25)**。**「lm_head 律速」仮説は実測で否定**: `prof_bitnet`(`src/tests/prof_bitnet.cpp`・本番重み・i7-10700)で区間分解したところ、律速は [src/infer/bitnet.hpp](../src/infer/bitnet.hpp) の `linear()`(double 蓄積・単スレッド・SIMD なし三重ループ)で **FFN ~67% + attention ~25% = ~92%**、lm_head はわずか **~7.5%**。対策はデュアルパス: `forward()`/`linear()`(double=golden 参照)を**完全無改変で温存**し、本番 `generate*()` を新設 `forward_fast()`/`linear_fast()`(float32 蓄積 + AVX2/FMA 明示 intrinsics + 末尾スカラ・lm_head は generate 時 last_only)に差し替え。**全 seq ~5x**(seq8 263→54.9ms 4.79x / seq32 1038→187.7ms 5.53x / seq63 2028→380.5ms 5.33x)。golden 非回帰(logit corr 1.0 / greedy synthetic 5/5・identity 5/5 / 新 double-vs-fast サブテスト max_abs ~5e-6・corr 1.0)。**Tier 2 はデータ駆動で設計確定 + 今は実装見送り**(下記バックログへ): トポロジ自動検出ベンチ(`src/core/cpu_topology.hpp` / `prof_cpu_topology`)で実測 — ① per-thread クリーン基準(seq8 51/seq32 186/seq63 364ms)が affinity 未固定 Tier 1 値と同帯 = **Tier 1 ~5x は実質1スレッド速度**と確定。② 独立 forward の物理コア並列スループットは N=5–7 で ~5x 飽和・N=8 は膨張(帯域/OS 競合)。③ HT 兄弟は別物理2本の 68–83% = disjoint 物理コア割当が基本。これらから **当初案「linear 内 out_dim 分割」(単一 forward レイテンシ削減)は却下** — ② が測ったのはレイテンシでなくスループットで、CPU LM の役割(拡散の裏で複数フレーム先行生成)では単発レイテンシは GPU 版(#6-GPU 87.5x)が担うため動機なし・同期コスト無駄。**Tier 2 確定設計 = (A) 独立 forward ワーカー方式**(linear 内は Tier 1 単スレッドのまま・複数フレームの forward を disjoint 物理コア(上限 5–6・HT 非動員)に pin した embarrassingly parallel)。**実装見送りは憶測でなく構造的事実**: (A) が効くのは複数フレーム同時投入時だが `pipeline.hpp` にその駆動側が無く(LM 段は stub・フレーム逐次)使用箇所ゼロ → 今コミットは保守コスト先払い。**発動条件 = pipeline に複数フレーム先行生成が実装されボトルネック化したとき**(下行バックログ)。BitLinear 再量子化排除/int8 GEMM は別タスク(圧縮実験)。 | ✅ Tier 1 完了 (~5x・golden 維持)・Tier 2 設計確定/留保 |
| **LM 複数フレーム先行生成 + Tier 2(A) 独立 forward ワーカー** | CPU LM 推論 Tier 2(上行 Tier 1 = AVX2 単スレッド ~5x の続き)の**確定設計 = (A) 独立 forward ワーカー方式**: `linear` 内は Tier 1 単スレッドのまま、複数フレームの forward を **disjoint 物理コア(`cpu_topology.hpp` 自動検出・上限 5–6・HT 兄弟は非動員)に pin した embarrassingly parallel ワーカー**で回す(プールは forward 単位のタスク投入・linear 内分割はしない)。実測ベンチ(②独立 forward スループットが物理コア ~5x まで素直にスケール・③HT 非効率)が裏付け。**当初案「linear 内 out_dim 分割」は却下**(単一 forward レイテンシは GPU 版が担う・役割に動機なし)。**着手は `pipeline.hpp` に複数フレーム先行生成(LM で N+1/N+2 を拡散の裏で生成する駆動側)が実装されボトルネック化したとき** — 現状その駆動側が無く実装しても使用箇所ゼロゆえ留保。L221「HW 環境抽象化」で決め打ちマスクを `cpu_topology.hpp` 自動検出へ置換する際に (A) の物理コア割当も同基盤へ束ねる。MoE × HW 分散検討とも地続き。**【2026-06-28 計測クローズ】駆動側を `src/core/multi_frame_pipeline.hpp` (MultiFramePipeline・複数フレーム先行生成の汎用骨格) として実装しテスト+並列ベンチを与えた (test_multi_frame_pipeline・42/42 全緑)。実デバイス比率スタブで実測した結果、発動条件 (LM 段ボトルネック化) は単一 GPU 構成では成立しない**: per_frame 3879.8ms ≈ SDXL 単段 3800ms (理論 GPU 上限の 98% = GPU バウンド)・`queue_bclip_to_bsdxl` 待ち 0.0002ms ≈0 (GPU 飢餓なし)・LM (404ms) は SDXL の裏に完全隠蔽。QueueDepth {2,4,8} スイープも 2 で飽和 (look-ahead 既定 2 が最適)。**= 単一 GPU では stage A は飢餓を起こさず Tier 2(A) の動機なし。留保継続を計測で裏取り。再評価は SDXL がライブラリ fallback 等で桁違いに速くなった世界のみ** (CLAUDE.md 計測ベースライン表参照)。 | ◯ (設計確定・**計測で留保継続を裏取り 2026-06-28**・発動条件は単一 GPU で不成立) |
| **プレビュー用低解像度ドラフトモード** | UI のプロンプト/タグ試行錯誤を速くする用途。**本番と同じ SDXL 重み・同じステップ数のまま、解像度だけ下げて**軽い下書きを出す (例: プレビュー 768²／本番 1024²)。狙いは「タグの当たり付け」専用で、**本番との完全一致は狙わない** (解像度が変わると SDXL は構図が変わる・512² は人物複製等で崩れやすいので 768² 推奨)。**うちの律速 (UNet attention) に特に効く**: self-attention は空間トークン数の 2乗なので 1024²→512² でトークン 1/4・attention コスト 1/16。**ステップ削減は不採用** (最終出力が変わるため・ユーザー判断 2026-06-25)。**latent preview (本番生成の途中 decode で忠実プレビュー) も今回は見送り** (同上)。実装は UI 側 ([ui/Components/Pages/Generate.razor](../ui/Components/Pages/Generate.razor) のサイズ切替にプレビュー/本番モードを足すだけ・C++ 無改修)・配管は既存。 | ✅ **完了 (2026-06-29)**: 2 ボタン方式 (「生成」=選択サイズ・「下書き(高速プレビュー)」=768² 固定)。送信サイズ決定は純ロジック `ui/Services/DraftPreview.cs` `ResolveDraftSize` (幅>768→768²・≤768 据え置き・パース不能→768²・例外なし) に切出し ui.Tests でカバー。`GenerateAsync(bool draft)` で draft 時のみ `req.Size` をローカル上書き (`_size` 本体・`Steps` は不変)・直近モードを `.gen-mode` バッジ表示。DTO/クライアント/C++ 無改修。`dotnet build ui` 0 エラー・`dotnet test ui.Tests` 39 緑 (DraftPreviewTests 7 ケース含む) |
| **成人向け後処理: モザイク/バー修正** | 同人誌頒布用途。日本の成人向け頒布は無修正不可 (刑法175条) なので頒布前提なら**事実上必須**。dollama 本体にコンテンツフィルタは無く、生成可否は載せる SDXL チェックポイント次第 (danbooru 系 fine-tune は対応)。修正は matting と同じ後処理段に **2 パーツで乗る**: ① **NSFW 領域検出** = アニメ向け検出モデルを OV グルーで 1 段 (ISNet/WD14/CLIP と同じ立ち位置・**山はモデル選定**)、② **修正処理** = モザイク (領域ブロック平均) or バー (黒/白塗り)・**モデル不要の純画像処理**・粗さは法的慣行 (画素数基準) / プラットフォーム規定 (DLsite/FANZA 等) に合わせて可変。`OutputSpec` に `censor: none/mosaic/bar` + ブロックサイズを追加。タグでの擬似修正 (`mosaic censoring` 等は vocab にあり) は位置・粗さ不定で頒布要件を満たさない。**絶対線: 成人キャラのみ (未成年不可)**。(ユーザー判断 2026-06-24 で積む) | ◯ (修正処理は低コスト・検出モデル選定が前提・時期未定) |

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

**プログラム結論 (2026-06-29・C/B/A/D 決着)**: C (評価作り直し) で据えた diverse-val 生成
set-F1 の上で B/A/D を 4 seed paired sweep で測った結果、**diverse-val F1 を頑健に押し上げたのは
施策 B (入力多様化) のみ**で、しかも **~2,000 件で飽和** (B-3・training-spec §14.9)。**施策 A
(実ペア増) は diverse-val F1 に seed ノイズで非寄与**だが identity retention 0.975 を頑健に成立
させる機能基盤 (§9.10)。**施策 D (容量増 33M→80M) は陰性確定** — F1 は seed ノイズ内 (符号反転)・
retention 床割れ・in-dist 微退行で 80M 不採用 (§16・勝者 = 33M b2000∧identity)。**蒸留 4 路線
(D2/D4/D5/D6) も全て recall/F1 非寄与**。→ **データ件数でも容量でもない別軸 (データ多様性の質・
アーキ・損失設計、そして本命 F の実品質オンライン信号) が残る diverse-val 低帯域
(diverse_a ~0.31 / diverse_b ~0.36) を取りに行く次のフロンティア**。

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
| **B** 入力多様化 | タグ集合固定 (tags-stay-real) で自然文を多様化。テンプレ 3 種の偏りを解消し実世界汎化を狙う。`source:"llm_distill"` スキーマ (dataset-spec §12/§15) 流用 | ✅ **件数拡大まで完了** (Claude 著述 Replace **500→2,000→10,000**・diverse F1 は seed 頑健に正だが **~2,000 で飽和** (2,000→10,000 は平坦・B-3 で「頭打ちなし」を訂正・training-spec §14.9)・著者交絡は D2 で否定・**本線昇格=レシピ既定化を確定 (今後の訓練 A/D/F は多様化入力=tags-stay-real を既定)・アーティファクト本体 (`bitnet_dense{,_fp32}.safetensors`) と C++ 推論 golden (test_bitnet_infer/gpu) の差し替えは A 実ペアと束ねる次の出荷リトレインまで遅延** (2026-06-24 ユーザー決裁) / training-spec §14 / dataset-spec §15) |
| **A** 実ペア増 | danbooru harvest 8,200→数万 posts。**a12k 4 seed sweep でクローズ (二分結論・training-spec §9.10)**: diverse-val 生成 F1 は **12k で seed ノイズ** (4 set/metric 判定 NO・seed 42 のみ反転・~2,000 飽和の B と整合) → **diverse F1 を上げる手ではない**。一方 **identity retention は頑健に 0.975** (across-seed 0.9748±0.0010・全 seed) = **同一性条件付けの機能基盤**。a25k は回さず未使用保持。recall 天井底上げは D 容量増側で取りに行く | ✅ **a12k で評価完了・クローズ** (本番 #1 即時差し替えなし・**法務/ToS ゲート** dataset-spec §1.3 / `[B-merge-at-A]` 遅延タスク=下記注記でまとめ焼き) |
| **D** 容量増 | 33M→**80M** (`DOLLAMA_BITNET_ARCH=d80m`・N_LAYERS 8→16 / FFN_DIM 1792→2464 = 79.91M)。両アーム同一レシピ (b2000 ∧ a12k identity)・`--arch` だけ差・4 seed 6ep paired sweep で diverse-val F1 優位の seed 頑健性を測定。**陰性確定・80M 不採用 (training-spec §16)**: diverse-val F1/Jaccard は 4 set/metric とも判定 NO (seed 20260620 で負・seed 7 で正と符号反転・across 平均 −0.002〜−0.004 で c33 seed 分散帯 sd 以下 = A12k/D6 と同型の seed ノイズ)・retention は 3/4 seed が床 0.975 割れ・in-dist 微退行。**容量 (33M→80M) では diverse-val F1 は取れない (データ律速・施策 B ~2,000 飽和と整合)** → **勝者 = c33 (33M・b2000 ∧ a12k identity) = #1 超え出荷候補**。80M は forward ~2x の対価に見合う実利なし | ✅ **陰性確定・クローズ** (打ち切り基準どおり 80M 不出荷・正典化は `[B-merge-at-A]` で勝者 33M を 1 回まとめ焼き) |
| **F** 品質ループ | B 品質スコアラ (アニメ品質・NPU/CPU・§技術リスク表) を作り、生成プロンプト→SDXL 画像→採点→報酬で LM を fine-tune (CharacterMemory ループ)。recall でなく「良い絵を生む」方向に学習軸を移す | Phase 2 (済 11.3s) + B スコアラ実装 |
| **F-0a** 信号ゲート | **✅ 実走 80/80 → 判定 = 信号弱 (補強してから)** (2026-07-02・研究機 gpu-benchmarker)。reward std 0.0377 / best−worst 0.2031 (PL 閾値 std>0.1 かつ best−worst>0.3 に未達)。worst-axis argmax **Limbs 77/80** で他 7 軸ほぼ死 (ScorerNet dynamic range が Limbs 単軸) + worst 帯は多人数/背景/mecha/文字焼き込み等スコープ外題材への confound (画像照合: 単独素直題材は解剖正常で reward≈0)。ただし生成 prompt の clean vs clutter で **|r| 4倍分離** (0.007 vs 0.0285) = 弱いが本物の勾配源・−1 飽和帯ではない (checkpoint エスカレーション不要)。詳細 measurements-log.md | ✅ 実走・判定済 |
| **F-0b** SFT | **保留 (F-0a 補強後)**。最小補強順: ① **quality head 有効化 (Q-2 waifu 再正規化 / deepghs 合流)** で生存中の直交軸を reward に足す (最小・最大レバー) → ② **7 死軸の分解能診断** (B 側 ScorerNet) → ③ 補強後 F-0a smoke 再走で std→>0.1 or clean/clutter 分離維持を確認 → 立てば SFT 着手 | ⏸ 補強待ち |

**C と F は同じ軸の両端**: C = より良いオフライン proxy、F = 本物のオンライン信号 (良い絵か)。
このプログラムの背骨は**物差しを proxy→実品質へ動かすこと**で、recall という枯れた数値から
卒業する話。**推奨着手順**: ① C 着手 + A の法務ゲートを PL に並行起票 (互いに待たない) →
② C 完了後 B を新 val で評価 (D2「悪化」の決着) → ③ 法務 GO 後 A+D 一括 → ④ F。
**評価**: ◎ 本命だが、実装は CLAUDE.md ルール (プランモード設計→承認→PL 振り分け) に従う。

> **ラベル化 (生成画像→WD14 タグ) の C++化は F でブロック解除** (2026-06-24・PL 判断):
> 研究機での SDXL生成→ラベル化 end-to-end 実走は **Python offline ツール
> `scripts/dollma_label_image.py` (d83fd66) を当面の正規経路**として確定。本番のメモリ上
> 結線 (生成器の生ピクセル → WD14 → タグ列 → LM FB) は **唯一の消費者が F** のため、
> F 着手まで C++化は保留 (消費者不在の先行配管は死にコード化リスク・PNG デコーダ連鎖も招く。
> `src/server/png.hpp` はエンコード専用)。**F 着手時の C++化スコープ (最小)**:
> ① 純関数 `dollma_resize_to_wd14` (raw RGBA/RGB → 448 BGR float, 白合成→正方形パディング→
> リサイズ, ヘッダオンリー + test)。前処理正準は SmilingWolf (RGBA→白合成→正方形パディング→
> 448→BGR→float32 0-255 **正規化なし**)・`scripts/dollma_label_image.py` 準拠。
> ② `src/infer/wd14.hpp Wd14Tagger` に `selected_tags.csv` 名前マッピングを追加 (実装パターンは
> `src/infer/quality_gate.hpp` の CSV 名前→index 解決を流用)。
> ③ `src/pipeline.hpp` のダミー乱数画像入力 (L258) と `tag<idx>` ダミー解決 (L314) を ①② で置換。

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
> しつつ seed sd は縮小 (効果が強まり頑健性も増加・全判定軸成立)。**頭打ちは見えない**。**決裁済 (2026-06-24・ユーザー)**:
> **レシピ既定化を確定** — 今後の訓練 (A 実ペア増 / D 容量増 / F 品質ループ) は多様化入力 (tags-stay-real) を
> **既定レシピ**とする。B は A/D と直交 (依存連鎖 C→{B,A}→D→F の並列枝) ゆえ、劣るレシピ (#1 系) の上に A/D が積まれる
> 事故を防ぐためレシピ確定を先に行う。**正典重み `bitnet_dense{,_fp32}.safetensors` と C++ 推論 golden
> (test_bitnet_infer/gpu) の差し替えは A 実ペアと束ねる次の出荷リトレインで1回** (golden チャーンを 2〜3 回払うのを回避)。
> 当面 #1 重みは `bitnet_dense.safetensors` のまま据え置き・別名 `bitnet_dense_diverse_b2000` は実験出力のまま。
> **保留が解けた根拠**: ① Pareto 改善 (主指標 diverse-val 生成 F1 で strictly 改善・in-dist pairs.val / legacy recall は退行なし)
> ② seed sweep 全 3 軸成立 ③ スケール則 (件数増で単調・頭打ちなし)。当初の保留理由 (edge が seed 分散帯以下・絶対値が低い)
> は B-2000 で消滅 — delta は分散帯の数倍に拡大し、絶対値が低いのは #1 がさらに低いだけで #1 を選ぶ理由にならない。
> **残る論点 (絶対品質 diverse F1 ~0.32–0.37)** は A 実ペア増 / D 容量増で取りに行く (本線の良し悪しの話とは別軸)。
> 詳細 training-spec §14.8 / dataset-spec §15.6。

> **B-3 件数拡大 (2026-06-25) — スケール則は ~2,000 件で飽和 (前ノートの「頭打ちなし」を訂正)**:
> 同方式・同物差し・同 sweep で**著述件数を 10,000 に拡大** (P=2,500 post × k=4 variant・B-2 の
> スーパーセット・Replace で総 train 12,000=著述 10,000+synthetic 2,000・tags-stay-real)。`make` を
> `--n-posts`/`--k-per-post` に一般化 (k=1 で B-1/B-2 bitwise 非回帰)。seed sweep 4 seed の delta(B10k−#1)
> は全 set/metric で **判定 YES (seed 頑健・本物)** だが、**2,000→10,000 の 5 倍増で平坦** — diverse_a F1
> +0.1472→**+0.1411**・diverse_b F1 +0.1788→**+0.1761** (seed 分散内・むしろ微減)、b 絶対値も 0.319→0.313 /
> 0.361→0.359 で頭打ち。**前ノート (B-2) の「単調・頭打ちなし」は 500→2,000 の 2 点外挿の誤りで、3 点目
> (10,000) で飽和が判明**。**運用結論**: 入力多様化単体の伸びしろは ~2,000 で尽きる → 残る低帯域
> (diverse_a ~0.31 / diverse_b ~0.36) は **B 件数ではなく A 実ペア増 / D 容量増**で取る。**B 著述を 2,000 超に
> 積む価値は薄く、`[B-merge-at-A]` の既定多様化ファイルは `pairs.train.diverse_b2000.jsonl` で足りる**
> (b10k 不要)。本線昇格決裁 (2026-06-24) は不変・本番重み #1 据え置き・別名 `bitnet_dense_diverse_b10k`。
> 本 sweep は 2026-06-25 の PC ハングから冪等再開 (`_results/*.npz` 存在 skip) で完走。
> 詳細 training-spec §14.9 / dataset-spec §15.8。

> **`[B-merge-at-A]` (A 出荷リトレイン時のチェックリスト・2026-06-24 決裁の遅延条項)**: 施策 B の正典化は
> A 実ペア増と束ねる次の出荷リトレインで1回にまとめる (golden チャーンを集約)。その回で必ず行うこと:
> ① train ソースを多様化版に切り替える (`pairs.train.diverse_b2000.jsonl` を `--train-file` で既定指定・
> A の新規実ペアは同じ tags-stay-real 機構で diverse train へ合流させる)。それまで `scripts/train_bitnet.py` の
> `--train-file` default=None (=#1 経路) は**意図的に据え置く** (今書き換えると別名出力分岐に落ち、bitwise 非回帰
> アンカー・golden 据え置きという遅延条項を破るため)。② 正典 `data/bitnet/bitnet_dense{,_fp32}.safetensors` を
> **この1回で**差し替える。③ C++ 推論 golden (test_bitnet_infer / test_bitnet_gpu・corr 1.0 突合) を**同時に**再生成する。
> ④ legacy 非回帰アンカー (pairs.val recall ~0.777) の役割を「#1 アンカー」→「新本線アンカー」へ変更する。
> ⑤ **A (同一性条件付け) も同じこの1回で焼く** (2026-06-26 A クローズ後の追記): A は a12k 4 seed sweep で評価完了し (training-spec §9.10)、効果は diverse-val F1 でなく **identity retention 0.975 (全 seed 頑健)** と確定。よってこの出荷リトレインは `--identity` で b2000 多様化 + identity_cond 混合を同時に焼き、retention を持つ本線を 1 回で作る (A の実ペアは a12k を使用・a25k は未使用保持)。出荷重み = B(多様化) ∧ A(identity 条件付け) のまとめ焼き 1 本。
>
> **✅ 完了 (2026-07-03) — `[B-merge-at-A]` クローズ**: 上記①〜⑤を1回のまとめ焼きで実施済。勝者 33M で
> B(b2000) ∧ A(a12k identity) を merged 混合1本に訓練 (MIXED train=15300 [diverse_b2000 4500 ∪ a12k
> identity 10800] / val=1700 [synthetic 500 + identity_cond 1200]・6ep FP32・val_loss ep3 底 1.9996・
> train 203.1s・param 32,976,896)。**ゲート4指標** (seed 20260620・`data/bitnet/_merge_ba/eval_report_merged.json`):
> identity retention **0.9807** / diverse_a 生成 macro F1 **0.3332** / diverse_b 生成 macro F1 **0.3804** /
> in-dist pairs.val 生成 macro F1 **0.4552** — 各単体参照 (a12k retention 0.9748 / b2000 diverse_a 0.3212 /
> diverse_b 0.3670) を全軸で上回り合格。① `--train-file=diverse_b2000` + `--identity` の merged 分岐で train
> ソース多様化 + identity 混合。② 正典 `bitnet_dense{,_fp32}` / `bitnet_dense_identity{,_fp32}` を merged と
> 同一バイトへ差し替え (sha256 FP16 5780fe10 / FP32 5043772d・旧は .pre_merge 退避)。③ C++ 推論 golden 再生成
> (`data/bitnet/golden/` 6 ファイル merged 基準・旧 golden_pre_merge/ 退避)・test_bitnet_infer corr 1.0 /
> greedy 5/5 (synthetic+identity)・meson test 25/25 緑。④ legacy 非回帰アンカー (pairs.val recall) の役割を
> 「#1 アンカー」→「新本線 (merged) アンカー」へ変更。⑤ A(identity) を同回で焼成 (a12k 使用・a25k 未使用保持)。
> `--train-file` default は None 据え置きのまま、正典再現は明示コマンドで行う (training-spec §17 再現手順)。
> **follow-up (非ブロッキング)**: 研究機 RTX5080 (sm_120・`with_cuda=true`) で test_bitnet_gpu の GPU golden
> corr 1.0 再確認 (CPU 版 BitNetDenseInfer は確認済)。詳細 training-spec §17 / dataset-spec §19。

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

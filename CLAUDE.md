# dollama — Claude 向けプロジェクトコンテキスト

## プロジェクト概要

**芯**: CPU / NPU / iGPU / RTX5080 — 搭載する全 HW を使い切りながら、
2D イラスト/漫画の**キャラクター描画**生成にたどり着くことを研究する。最短実装ではなく、
各 HW をどう活かし、どう協調させるかがこのプロジェクトの本質。

**スコープ (確定)**: 生成対象は**キャラクターのみ**。背景は生成しない
(Grok / Gemini / Stable Diffusion + CLIP Studio Paint で合成)。出力は
**切り抜き済み透過 PNG**。マッティング (α 抽出) も HW 協調パイプラインの一段とする。
キャラ設定の管理構造は `docs/character-bible-spec.md` 参照 (同一性層/シーン層/出力層)。

**HW 役割分担 (研究中・随時更新)**

| HW | 役割 | 状態 |
|---|---|---|
| CPU | Qwen2-1.5B LLM (プロンプト生成) | ✅ 64-71 tok/s 確認済み |
| NPU | **CLIP text encoder 7.85ms** / WD14 タグ抽出 268ms | ✅ CLIP が CPU の 2.5倍速 |
| iGPU (Intel Xe) | VAE encode (img2img、79ms) + マッティング ISNet (99.96ms) | ✅ どちらも CPU より速い (matting_device=iGPU 確定 M-5) |
| **RTX5080** | SDXL UNet + VAE decode | ✅ **3.80s/image** (1024×1024, 20steps, 5.3 it/s) |

**パイプライン (確定構成)**

txt2img:
```
CPU: Qwen2-1.5B (暫定) / 将来: 自作タグ生成 LM (bitnet.hpp 33M, GPU 主・CUDA カーネル流用 / CPU 可・NPU 不可)
  自然文 → danbooru タグ列 (~2s / 将来 <10ms)
    │
    ▼
NPU: CLIP-L text encoder (7.85ms)
  テキスト → embedding [1, 77, 768]
    │
    ▼
RTX5080: SDXL UNet × 20steps + VAE decode (3.80s / 1024×1024)
    │
    ├─ CPU: WD14 SwinV2 tagger (101ms) ← GPU 生成中に並列
    │       → danbooru タグ → LLM フィードバックループ
    └─ 出力画像
```

img2img (追加パス):
```
入力画像
  ├─→ iGPU: VAE encode (79ms)  → latent ─┐
  └─→ CPU:  LLM (~2s)          ──────────┤ (並列)
                                          ▼
                               NPU: CLIP (7.85ms)
                                          ▼
                               RTX5080: SDXL UNet + VAE decode (3.80s)
```
iGPU の VAE encode は CPU LLM と並列に走るため待ち時間ゼロ。

**デバイス選定根拠 (probe 実測)**
- CLIP: NPU 7.85ms < iGPU 14ms < CPU 20ms → **NPU 採用**
- WD14: CPU 101ms < iGPU 104ms < NPU 268ms → **CPU 採用** (Window Attention が NPU に不向き)
- VAE decode: CPU 126ms << iGPU 995ms → **RTX5080 採用**
- VAE encode: iGPU 79ms < CPU 117ms → **iGPU 採用** (img2img パスのみ)

**VRAM 収支と多 HW の根拠 (重要)**

RTX5080 = **16GB**。常駐物の概算:

| 常駐物 | VRAM |
|---|---|
| SDXL 20steps ピーク (probe10) | **10.49GB** (実質ここが全部食う) |
| CLIP-L (123M FP16) | ~0.25GB |
| CLIP-G (694M FP16) | ~1.4GB |
| 自作タグ生成 LM (33M FP16) | ~0.066GB (誤差) |
| 合計 | ~12.2GB < 16GB |

- **LM に関して VRAM はボトルネックですらない** (33M=66MB)。GPU-only は 5080 で既に成立し、
  大容量 VRAM カードは不要。VRAM が効くのは「大型モデルを大量同時常駐」する別世界の話。
- **多 HW に逃がす理由は VRAM 不足の回避ではない**: ① 速度・並列 (CLIP は NPU 7.85ms で、
  拡散で詰まる数秒の裏で走り GPU を空ける) ② 拡散中に遊休の NPU/CPU/iGPU を使い切る
  ③ 研究そのもの (プロジェクトの芯)。
- **CPU/NPU offload = 金をかけずに容量を増やす手**: CPU 側専門家はシステム RAM (安・大量)、
  NPU はオンチップメモリを使う → VRAM を買い足さずモデルを足せる。「VRAM が高い」という
  制約への回答が、この「小型専門モデル + 全 HW 協調」設計そのもの。
- 関連: CPU+GPU マルチ LM / アンサンブル → roadmap バックログ「MoE × HW 分散配置」。

## 環境

- OS: Windows 11
- CPU/NPU: Intel Core Ultra 9 285 (NPU = Intel AI Boost, DEVICE_ARCHITECTURE: 3720)
- GPU: NVIDIA GeForce RTX 5080
- Python: 3.14
- OpenVINO: 2024.x 以降 (openvino.runtime は廃止済み)
- PyTorch: cu128 ビルド (RTX5080 = Blackwell / sm_120 = CUDA 12.8 必須)

## 確定済みアーキテクチャ決定

### CPU 経由パイプライン (現在の確定構成)

調査したゼロコピーの全ルートと結果:

| ルート | 結果 | 理由 |
|---|---|---|
| CUDA Virtual Memory + Win32ハンドル → NPU | ❌ | OpenVINO NPU に CUDA ハンドル import API なし |
| D3D12 クロスアダプター (RTX5080 → iGPU → NPU) | ❌ | Intel iGPU が DXGI に非表示 (BIOS でコンピュート専用) |
| CPU pinned memory | ✅ | 3.4% オーバーヘッド・マルチスレッドで隠蔽可能 |

**重要: OpenVINO の 'GPU' デバイスについて (probe4 で確認)**
- BIOS で iGPU を有効化すると `['CPU', 'GPU.0', 'GPU.1', 'NPU']` の4デバイスが見える
- `GPU.0` = Intel(R) Graphics (INTEGRATED) = Intel Xe iGPU
- `GPU.1` = NVIDIA GeForce RTX 5080 (DISCRETE)
- iGPU は FP16 / INT8 / GPU_USM_MEMORY 対応

**iGPU のパフォーマンス (probe4 STEP2 で確認)**
- VAE デコードスタブ (ConvTranspose2d 4→512→256→128→3): iGPU 995ms、CPU 126ms
- **iGPU は CPU の 8倍遅い** → 大規模 Conv モデルには向かない
- iGPU は軽量な前処理・後処理向け。VAE decode は RTX5080 または CPU が適切

**NPU ↔ iGPU ゼロコピー (probe4 STEP3 で確認)**
- NPU 出力 (231KB) をそのまま iGPU に渡す場合と .copy() の差: 0.158ms = 誤差範囲
- システムRAM共有によるゼロコピー実証済み ✅

**OpenVINO NPU プラグインのメモリ interop (ソース調査で確認)**
- Level Zero ベースの外部メモリインポート実装あり (`ZeroRemoteTensor`)
- `SHARED_BUF` / `CPU_VA` / NT handle / DMA-BUF をサポート
- ただし CUDA interop は実装なし → RTX5080 との直接共有は不可

- 計測済み転送オーバーヘッド: 3.4% → 問題なし
- GPU 拡散処理 >> NPU 推論 なので、マルチスレッドで完全に隠蔽可能

### NPU の制約

- 静的形状のみ受け付ける
- `ov_model.reshape([batch, seq_len])` をコンパイル前に必ず実行
- `convert_model` はデフォルトで動的形状を出力するため、reshape が必須

### OpenVINO C++ 入力テンソルの要素型 (タスク5 で確認)

- **OV IR の入力 `element_type` を必ず確認してテンソルを生成すること。** CLIP-L の `input_ids` は `i64` shape `[1,77]` (静的)。
- C++ で `ov::element::i32` テンソルを渡すと、NPU プラグインが i64 として要素あたり 8 バイト読もうとし、領域外読み出しで **0xC0000409 (STATUS_STACK_BUFFER_OVERRUN)** クラッシュする。型は IR と厳密に一致させる (token id は int64 へ明示変換してコピー)。
- CLIP は出力が2つ: `last_hidden_state [1,77,768]` = 出力0、`pooler_output [1,768]` = 出力1。hidden states は `get_output_tensor(0)`。
- WD14 等の後続 OV モデル実装時も同様に i32/i64 取り違えに注意。

### OpenVINO API (2024.x)

- `import openvino.runtime` は廃止 → `import openvino as ov` を使う
- モデル構築: `ov.convert_model(torch_model, example_input=...)` が推奨
- NPU コンパイル: `core.compile_model(ov_model, "NPU")`

### NPU が適切な用途

- 固定形状・encoder-only モデル: CLIP text encoder (77トークン固定)、Whisper encoder
- 小型分類・回帰ネット (probe2 で 512dim MLP = 0.88ms 確認済み)
- **不適**: LLM 自己回帰推論 (KV-cache でシーケンス長が動的に増加するため)

### パイプライン構造 (C++ 実装)

```cpp
// GIL なし真のマルチスレッド、STL + CUDA API のみ
std::thread llm_thread([&]  { /* CPU: 自作タグ生成 LM (bitnet.hpp) */   });
std::thread clip_thread([&] { /* NPU: 自作 CLIP 推論 */       });
std::thread sdxl_thread([&] { /* GPU: 自作 UNet CUDA カーネル */ });
std::thread tag_thread([&]  { /* NPU: 自作 WD14 推論 */       });

// SPSC lock-free queue でゼロコピー受け渡し
```

**CPU アフィニティは自己ピン留め型** (`src/core/affinity.hpp` の `set_current_thread_affinity(mask)`)。各ワーカーが起動直後に自スレッドへ設定する。当初の「親が子 jthread の `native_handle()` を `SetThreadAffinityMask` に渡す」設計は、`std::thread::native_handle_type` が MSVC=`HANDLE` / MinGW-W64 posix=`pthread_t` で **MinGW でコンパイル不可**だったため変更 (winpthreads は `pthread_getw32threadhandle_np`/`pthread_setaffinity_np` 未提供)。`GetCurrentThread()` 擬似ハンドルで native_handle 非依存にし MSVC/MinGW 両対応・両環境で実ピン留め可。詳細は `docs/cpu-topology.md`。

### 実装方針

**自作は HW を叩く研究コアに限定し、配管 (HTTP/JSON/Base64) は定番のヘッダオンリー
ライブラリを使う。** 配管を自作してもバグ表面と保守コストが増えるだけで研究価値がない。
「重量級フレームワークを使わない・単一バイナリ」の美学はヘッダオンリー採用で維持する。

| 使う | 使わない |
|---|---|
| STL 全般 | PyTorch / LibTorch |
| CUDA Runtime API | diffusers / stable-diffusion.cpp |
| 自作 Tensor / GEMM / Attention | llama.cpp / OpenVINO (probe のみ) |
| 自作 CUDA カーネル / BitNet / OV 推論グルー | Drogon 等 **重量級** HTTP フレームワーク |
| **cpp-httplib** (HTTP, 単一ヘッダ・Winsock2/POSIX 吸収) | 手書き Winsock2 ボイラープレート |
| **nlohmann/json** (JSON, ヘッダオンリー) | 手書き JSON パーサ |
| Base64: httplib 付属 or 数十行の小物 | — |

### LLM の将来像

- 現在: Qwen2-1.5B (Python probe 用) — CPU 64-71 tok/s
- 自作モデル: **小型タグ生成 LM** (`src/models/bitnet.hpp`, decoder-only LLaMA 系 33M, 実装済み・test 緑)
  - 30-100M params、user text → danbooru タグ生成特化
  - **置き場は GPU が第一** (速度 + **既存の自作 CUDA カーネル gemm/attention/layernorm/geglu をそのまま流用**)。LM は拡散の上流で逐次実行 → GPU 競合ほぼ無し・33M は VRAM も誤差。**CPU は代替** (複数フレームのパイプライン並列時に拡散の裏で先行生成する用)。**NPU は自己回帰不可で除外** (probe6 で Phi-3 がオンチップメモリ超過・investigation-log。NPU 化は非自己回帰版が要る・未確定)
  - **ternary (b1.58) は目的ではなく圧縮の研究軸**: まず FP16/INT8 dense で品質を出し、ternary は乗算削減の実験として後段で被せる (`ternary_gemm.cu`)。33M 規模では ternary の旨味は限定的
  - **先行実装あり**: DanTagGen (400M LLaMA) / TIPO が「自然文+タグ → danbooru タグ」を既に実現 → 蒸留教師・品質基準として活用 (Qwen2 蒸留と同じ役割)
  - **新規性の方向 (2D 特化・独自)**: ① キャラ**同一性条件付き**タグ生成 (character-bible を条件入力・DanTagGen に無い) ② アニメ**品質スコアラ** (NPU・固定形状, §11)。詳細・経緯は roadmap Phase 4

## 計測ベースライン

| 指標 | 値 | probe |
|---|---|---|
| CPU→VRAM (10MB) | 0.76ms | probe2 |
| CPU→VRAM (100MB) | 3.46ms / 30.3 GB/s | probe2 |
| NPU推論 (512dim MLP, 静的形状) | 0.88ms | probe2 |
| NPU出力 (2048B) → GPU | 0.031ms (3.4%) | probe2 |
| iGPU VAE decode stub | 995ms (CPU 126ms → RTX5080 採用) | probe4 |
| NPU→iGPU ゼロコピー差分 (231KB) | 0.158ms (誤差範囲) | probe4 |
| system RAM → RTX5080 latent (256KB) | 0.030ms / 8.7 GB/s | probe4 |
| system RAM → RTX5080 image (12MB) | 0.254ms / 49.6 GB/s | probe4 |
| iGPU VAE encode (1024→128, img2img) | **79ms** (CPU 117ms → iGPU 採用) | probe5 |
| Qwen2-1.5B INT4 CPU tok/s | 64-71 tok/s | probe7 |
| Qwen2-1.5B INT4 ロード時間 | 1.1s | probe7 |
| WD14 SwinV2 (448×448): CPU / iGPU / NPU | 101ms / 104ms / 268ms → **CPU 採用** | probe8 |
| CLIP-L text encoder (77token): CPU / iGPU / NPU | 20ms / 14ms / **7.85ms** → **NPU 採用** | probe9 |
| CLIP-L NPU (C++ ClipEncoder, 中央値/N=100) | **7.82ms** (min 7.61 / max 12.15) | test_clip |
| WD14 (C++ CPU Wd14Tagger, 中央値/N=100) | **105.3ms** (min 99.1 / max 132.8) | test_wd14 |
| SDXL 20steps 1024×1024 RTX5080 | **3.80s** / 5.3 it/s / VRAM ピーク 10.49GB | probe10 |
| Pipeline 縦通し (stub→CLIP NPU→queue→WD14 CPU, C++ jthread) | **9.13 frames/s** / per_frame 109ms / 単発レイテンシ中央値 157ms (WD14 CPU 律速) | test_pipeline |
| compose_prompt (C++ CharacterBible, 1M iters) | **242 ns/op** | test_character |
| CharacterBible::find (10,000体, 1M lookups) | **10.5 ns/op** | test_character |
| vector_add 疎通ベンチ (N=16.7M, H2D×2+D2H, pageable, RTX5080) | **14.9ms 中央値 / 13.5 GB/s** (pinned 化で probe2 30GB/s 帯に上がる余地) | test_cuda_smoke |
| 自作 FP16 GEMM 1024³ (shared-mem タイリング, FP32 蓄積, RTX5080) | **0.45ms 中央値 / 4730 GFLOPS** (max_rel 5e-4) | test_gemm |
| 自作 FP16 GEMM SDXL Linear transB (M=4096 N=K=1280) | **3.19ms / 4208 GFLOPS** | test_gemm |
| 自作 活性化 SiLU/GeLU(erf) FFN (4096×5120, FP32 内部, RTX5080) | **544 GB/s** (UNet FM は起動律速で ~230 GB/s) | test_activation |
| 自作 GroupNorm (1グループ=1ブロック, 1パス FP32 リダクション, RTX5080) | UNet C1280 **73 GB/s** / C640 75 GB/s / VAE FM C128H512 48 GB/s (占有率制約) | test_groupnorm |
| 自作 Conv2d direct (1スレッド=出力1画素, FP32 蓄積, RTX5080) | UNet C320 64² 3×3 **4.18ms / 1807 GFLOPS** / VAE C128 512² 3×3 **46.4ms / 1667 GFLOPS** (1×1=GEMM・3×3=タイリング昇格余地) | test_conv2d |
| 自作 Attention (per-(b,h,row) block + shared scores, FP32 softmax, RTX5080) | self 1024² Dh80 **1.65ms / 1631 GFLOPS** / cross Sk77 Dh80 **0.116ms / 1748 GFLOPS** (flash/Tensor Core 委譲は後続) | test_attention |
| safetensors ローダー (ヘッダオンリー, 自作最小 JSON パーサ, ifstream 全読み) | golden 5テンソル (F32/F16/BF16/I64/I8) ロード **19.0 µs/op** (N=10000) | test_safetensors |
| 自作 VAE decoder 全段 (SDXL AutoencoderKL, latent[1,4,128,128]→image[1,3,1024,1024], RTX5080) | final **SSIM 0.999992** / MAE 5.45e-4 / decode **中央値 7.96s** (cudaEvent N=7)。up2 以降は FP32 中間 (FP16 中間は up3 で Inf→GroupNorm NaN 伝播のため不可)。naive attention(Sq=Sk=16384)+ direct conv(1024²)が律速で最適化余地大 | test_vae_decode |
| 自作 SDXL UNet 全段 (UNet2DConditionModel, latent[1,4,128,128]+CLIP[1,77,2048]+added_cond → noise_pred[1,4,128,128], RTX5080) | final noise_pred **SSIM 0.999998** / MAE 7.40e-5 / 24段ゴールデン全緑 (time/add embed〜各down/mid/up〜conv_out, corr≥0.999997)。中間は全段 FP16 で可 (VAE と違い FP16 範囲内)。1step **中央値 ~9.2s** (cudaEvent N=7、direct conv + per-row attention 律速・Tensor Core/im2col-GEMM/flash 化が最適化余地)。重み 2.57B params (FP16 5.1GB) | test_unet |
| 自作 EulerDiscreteScheduler (SDXL, scaled_linear betas, timestep_spacing=leading, 20steps) | sigmas max_err 4.77e-6 / timesteps 完全一致 / scale_model_input 1.79e-7 / step0 1.43e-6 (diffusers golden 突合) | test_scheduler |
| 自作 LayerNorm / GEGLU / sinusoidal time embed / broadcast bias add (UNet 補助カーネル, RTX5080) | LayerNorm 4096×1280 **0.043ms / 486 GB/s** / GEGLU 4096×2560 **0.055ms / 1138 GB/s** / time embed dim320 0.011ms / bias_add rowvec 4096×1280 0.071ms (max_rel<2e-3) | test_layernorm/geglu/timeembed/bias_add |
| HTTP サーバー (Phase 3, cpp-httplib 0.47.0 + nlohmann/json 3.12.0, ヘッダオンリー wrap) | スタブ生成 + HTTP 往復 (256×192) **2.11ms** (PNG 147KB, ローカル自己リクエスト)。OpenAI Images 互換 `/v1/images/generations` で PNG base64 返却。生成本体は `IImageGenerator` 越し (2-6a で PipelineGenerator 差し替え可能・main.cpp フォールバック DI) | test_http |
| PNG メタ往復 (character-bible-spec §7, tEXt "dollama/bible", ensure_ascii JSON, IHDR/IDAT 非破壊で IEND 直前挿入, 純 C++) | embed **1961 ns/op** / read **4376 ns/op** (N=10000)。CharacterIdentity/SceneSpec/OutputSpec ⇔ §7 JSON ⇔ PNG。日本語 name 往復・非PNG/壊れlength/切り詰めを境界外読みなしで弾く | test_png_meta |
| 自作 BitNet b1.58 モデル定義 (Phase 4-2, decoder-only LLaMA系, BitLinear ternary+int8 / RMSNorm / RoPE / SwiGLU / causal attn / embed tied, 純ホスト参照 forward, GTX1080Ti 開発機 GCC) | d_model=512/n_layers=8/n_heads=8/ffn=1792/vocab=4999/max_seq=64 = **32.98M params**。forward 中央値 **~18,982 ms/forward** (seq=32, N=5。naive 参照: BitLinear 毎回 re-quantize + lm_head 4999×512 全走査。速度目標なし・4-5 CUDA ternary GEMM / 4-6 推論のゴールデン基準)。test_bitnet 8 サブテスト全緑 (param 範囲/ternary/int8/RMSNorm/embed tied/決定的 forward/logit 健全性) |
| 自作トークナイザー (Phase 4-3, vocab.json 駆動・タグ単位完全一致・ヘッダオンリー純ホスト C++, GTX1080Ti 開発機 GCC) | encode **365 ns/op** / decode **168 ns/op** / encode_text(greedy 最長一致) **2655 ns/op** (中央値, warmup後)。specials id 0..4 + tags[i].id==5+i 検証ロード (総語彙 4999)。§6 正規化 (`long_hair`→`long hair`・顔文字 `_` 保持)。実 pairs 5,000行/77,195タグ **UNK 0**・encode→decode 往復完全一致。test_tokenizer 全緑 |
| 自作 dense タグ生成 LM 訓練 (Phase 4-4, scripts/train_bitnet.py, bitnet.hpp 等価 dense PyTorch 32.98M, hard CE・蒸留なし初版, GTX1080Ti FP32 6epoch) | val_loss 3.22→**2.41** 単調収束 / train_loss 1.32 / **top10 tag recall 0.777** (random 0.002 の 388x) / 訓練 **45.3s** (seed 20260620 で決定的・再現バイト一致)。params 32,976,896 = bitnet.hpp::param_count 完全一致。重み `data/bitnet/bitnet_dense{,_fp32}.safetensors` (FP16 62.9MB / FP32 125.8MB, [out,in] 規約 74 テンソル, gitignore・再生成可)。蒸留は D2/D4 (hard CE 混合・不採用) と D5 (soft-label KL・下行) で検証済 |
| 自作 soft-label KL 蒸留 (Phase 4-D5, scripts/train_bitnet.py `--distill-kl`, 案A=`cache/danbooru_posts.jsonl` 8200posts の prefix 条件付き共起 teacher・温度付き soft target, `L=α·T²·KL(teacher‖student)+(1−α)·hardCE`・target 区間マスク共有, GTX1080Ti FP32 seed 20260620) | **過学習を実際に抑制**: train-val gap #1 1.09 → D5-XL 0.27 / D5-L **−0.21 (消失)**、val_loss 反転 ep4→ep6 後退 (D2/D4 が両軸悪化させたのと対照)。だが最良 top10 recall 0.758 < #1 0.777 (soft 追従と one-hot val recall がトレードオフ・両取り不可) → **不採用・#1 6ep 本線維持**。α=0 で plain hardCE と全 epoch bitwise 一致 (非回帰)、dropout は eval forward 不変 (max abs diff 0.0)=golden 非回帰。別名 `bitnet_dense_kl*` 出力・本番重み/golden 無改変。小規模 split/early-stopping 運用で再利用価値ありの正の機構。training-spec §11 / dataset-spec §12.3 |
| 自作 外部教師 soft-label KL 蒸留 (Phase 4-D6 = 案c, scripts/train_bitnet.py `--distill-ext` + scripts/dollma_d6_teacher_cache.py, 外部教師 **TIPO-200M** (KBlueLeaf・apache-2.0) のタグ補完生成を自作4999vocabに完全一致写像=案b-tagset・OOV drop+再正規化 (保持率0.79・エントロピー0.85nats), 訓練時 TIPO 非ロード=position軸付き COO npz を順送り読込, KL plumbing は §11 流用, GTX1080Ti FP32 seed 20260620) | **不採用 (recall 上振れは seed ノイズと確定・#1 本線維持)**: 単一 seed では #1 を僅かに上回って見えた (α=0.1 T=1.5 で 0.7818) が、**seed 頑健性 sweep (20260620/1/42/7) で再現せず** — delta=D6−#1 が +0.0013/−0.0027/+0.0015/−0.0031 (符号反転・平均 **−0.0008**)、最大 |delta| 0.0031 は #1 自身の seed 分散 (sd 0.0020・range 0.0055) に埋もれる。方法論補正: epoch 最大 recall で統一すると原値 #1 best は 0.7787 (D6 0.7800 との差 +0.0013)。**再現する D6 効果は過学習抑制のみ** (gap 全 seed 1.07→0.19・~5.5x): D5 と同じ正則化ノブで one-hot val recall に非寄与 (soft 追従 vs recall トレードオフ)。小規模 split/early-stopping 運用で再利用価値あり。別名 `bitnet_dense_d6*`・sweep は scratch `_seedsweep/`・本番重み/golden 無改変。training-spec §12 / dataset-spec §12.4 |
| 自作 dense タグ生成 LM 推論 (Phase 4-6, src/infer/bitnet.hpp BitNetDenseInfer, safetensors FP32 ロード→forward+greedy デコード text→tags, CPU 純ホスト, GTX1080Ti/i7-10700 MSVC) | PyTorch golden 突合: logits seq 8/32/63 で max abs err **2.86e-6〜2.29e-5** / **corr 1.0**、greedy デコード **5/5 完全一致** (語彙外日本語含む)。RoPE NeoX・[out,in]・tied lm_head・eps を golden 一致で検証。1 forward (seq8) **~253ms** (naive 参照・lm_head 4999×512 全走査支配・GPU 版 #6-GPU で seq8 2.43ms=46.8x に短縮済・量子化は後段)。models/bitnet.hpp の ternary BitLinear 不使用・dense FP matmul double 蓄積。test_bitnet_infer 緑 (golden/重み不在時 [SKIP]) |
| 自作 dense タグ生成 LM 推論 GPU 版 (Phase 4-6-GPU, src/infer/bitnet_gpu.cu/.cuh BitNetGpuInfer, safetensors FP32 ロード→device 常駐→GPU forward+greedy デコード, RTX5080 sm_120, T1 CUDA カーネル + T2 クラス) | 純 FP32 で CPU 版 BitNetDenseInfer と数値一致: device forward は embed gather / RMSNorm / RoPE / causal SDPA / SwiGLU を static `__global__`・Linear/lm_head は **LM ローカル純 FP32 cuBLAS (CUBLAS_COMPUTE_32F・TF32 厳禁)**・リダクション系は double 蓄積で CPU(double) と桁合わせ。**本番重み突合**: logits seq 8/32/63 で max abs err **1.49e-6〜6.44e-6 / corr 1.0**、greedy `generate` 3/3・`generate_with_identity` 2/2 完全一致。**forward レイテンシ中央値 (warmup3/iters30, DOLLAMA_BENCH=1)**: seq8 **2.43ms** (CPU 113.78ms = **46.8x**) / seq32 **9.90ms** (456.87ms = 46.2x) / seq63 **10.29ms** (900.83ms = **87.5x**)。class と heavy include は `#ifndef __CUDACC__` で囲み .cu(nvcc/C++14)非汚染。test_bitnet_gpu 緑 (重み/GPU 不在時 [SKIP]) | test_bitnet_gpu |
| 自作 INT8 dense タグ生成 LM 推論 (Phase 4 圧縮実験, src/infer/bitnet_int8.hpp BitNetInt8Infer, 重みのみ per-row 対称INT8・ロード時量子化・射影8本のみ量子化/embed・lm_head(tied)・RMSNorm・RoPE・attn・softmax は FP32 据え置き, GTX1080Ti/i7-10700 CPU 純ホスト) | **量子化耐性が高い** (圧縮の研究軸であって目的ではない — CLAUDE.md 方針)。**INT8 logits vs FP32 golden**: seq8 max_abs_err 0.0493 / corr **0.999964**、seq32 0.2126 / 0.999944、seq63 0.4608 / **0.999873** (ハードゲート corr≥0.99 を**3桁上回る**)。**greedy 生成 vs gen_golden 全5ケース完全一致** (16/16・8/8・14/14・9/9・8/8 = トークン一致率 **1.0 (55/55)・EXACT 5/5**・語彙外日本語含む)。**フットプリント**: 射影8本 FP32 121,634,816 B → INT8重み+per-row scale 30,605,312 B = **74.84%減** (削減 91,029,504 B・残存比 0.2516≈1/4)。**seq8 forward レイテンシ**: INT8 **~152.9ms** vs FP32 ~393.3ms = **0.389x** (int8 内積 int64 蓄積が FP32 double dot より軽い・lm_head は両者 FP32 同条件)。dense FP32 経路 (`src/infer/bitnet.hpp`) は**無改変**で #6/A3 golden 非回帰確認済。`src/models/bitnet.hpp` に `quantize_weight_int8_perrow()`/`Int8RowQuant` 追加・活性は既存 `quantize_activation_int8` 流用。GPU INT8 / ternary GEMM (#5) / INT4 は別タスク。test_bitnet_int8 全サブテスト緑・全22テスト緑・Skipped 0 | test_bitnet_int8 |
| 自作 dense タグ生成 LM 推論 CPU AVX2 高速化 (Phase 4 #6-CPU-fast, src/infer/bitnet.hpp `forward_fast`/`linear_fast` + `attention_block_fast`/`ffn_block_fast`, float32 蓄積 + AVX2/FMA 明示 intrinsics `_mm256_fmadd_ps` + 末尾スカラ, i7-10700 Comet Lake = AVX2 あり/AVX-512 なし) | **「lm_head 律速」仮説を実測で否定**: `prof_bitnet`(`src/tests/prof_bitnet.cpp`・本番重み)で区間分解すると律速は `linear()`(double 蓄積・単スレッド・SIMD なし三重ループ)で **FFN ~67% + attention ~25% = ~92%**、lm_head は **~7.5%**。**デュアルパス**: `forward()`/`linear()`(double=golden 参照)を**完全無改変で温存**し、本番 `generate*()` を `forward_fast`(matmul のみ `linear_fast` 化・RoPE/softmax は double 据え置き・lm_head は generate 時 `last_only` で最終位置のみ)に差し替え。**forward 中央値 (本番重み)**: seq8 **54.9ms** (double 263ms = **4.79x**) / seq32 **187.7ms** (1038ms = **5.53x**) / seq63 **380.5ms** (2028ms = **5.33x**) = 単スレッド AVX2 で全 seq ~5x。**golden 非回帰**: logit golden corr 1.0 / greedy synthetic 5/5・identity 5/5 完全一致 / 新 double-vs-fast サブテスト max_abs ~5e-6・corr 1.0。MSVC は `/arch:AVX2` 前提・GCC/Clang は関数単位 `__attribute__((target("avx2,fma")))`・非対応環境は float32 スカラ fallback をコンパイル時選択。**Tier 2 はデータ駆動で (A) 独立 forward ワーカー方式に設計確定 + 実装見送り**(下行 cpu_topology ベンチ + roadmap L222)。test_bitnet_infer 緑・prof_bitnet は計測専用 exe | test_bitnet_infer / prof_bitnet |
| 自作 CPU トポロジ検出 + forward_fast スケーリングベンチ (Phase 4 #6-CPU-topo, `src/core/cpu_topology.hpp`=`GetLogicalProcessorInformationEx` で物理コア/HT兄弟/EfficiencyClass を実機申告・OS依存隔離・決め打ちマスク排除 + `src/tests/prof_cpu_topology.cpp`=affinity ピン留め計測, i7-10700 Comet Lake homogeneous 8C/16T) | Tier 2 設計確定の前提計測。**① per-thread クリーン基準**(単一論理 pin): seq8 51 / seq32 186 / seq63 364ms = affinity 未固定 Tier 1 値(54.9/187.7/380.5)と同帯 → **Tier 1 ~5x は実質1スレッド速度**と確定。**② 物理コア 1→N 独立 forward スループット**: N=5–7 で **~5x 飽和**・N=8 全載せは per-forward 膨張(seq32 186→455ms・帯域/OS競合律速)。**③ HT 係数**: 兄弟2本 = 別物理2本の **68–83%** → disjoint 物理コア割当が基本・HT 同居は非効率。**④ P/E 比**: homogeneous 機ゆえ N/A(ハイブリッド機=13/14世代 Raptor Lake・研究機 Arrow Lake no-HT は実機引き渡し)。`forward_fast` は const・重み read-only で**並列呼び出し安全**(共有 mutable なし)。**Tier 2 = (A) 独立 forward ワーカー**(linear 内分割でなく forward 単位を物理コア pin)に確定・発動は pipeline 複数フレーム先行生成の実装後(現状 LM 段 stub で駆動側なし)。`cpu_topology.hpp` は roadmap L221「HW 環境抽象化」のトポロジ検出基盤として採用。test_cpu_topology 緑(構造的健全性)・prof_cpu_topology は計測専用 exe | test_cpu_topology / prof_cpu_topology |
| 自作 同一性条件付きタグ生成 (Phase 4-A, character-bible を条件入力・2D 特化の核, 機構 (a-1) prompt prefix `<bos> identity <sep> scene <sep> target <eos>`・`<sep>` 2回流用で vocab/tokenizer/アーキ無変更, GTX1080Ti FP32) | A1 §13 データ 5000ペア (identity/scene 分離・identity⊆target=retention 100% 教師・validate 全件0・train/val identity 重複 0.253)。A2 #1+§13 混合 6epoch 訓練で **identity retention 0.947** (生成 target が入力 identity の約95%を再現・条件付け実証)・source 別 val recall synthetic 0.900/identity 0.946。A3 C++ `BitNetDenseInfer::generate_with_identity` + `CharacterBible.find->canonical_tags` 結線 (tokenizer/character 無改修)。identity golden 突合: logits corr 1.0 / greedy 5/5 完全一致。test_bitnet_infer 緑 |
| 自作 評価作り直し (Phase 4-C, scripts/dollma_make_eval_diverse.py + train_bitnet.py `--eval-only` + dollma_c_seedsweep{,_analyze}.py, diverse-val 1500×2 + greedy 生成 set-metrics + seed sweep, GTX1080Ti FP32/CPU) | テンプレ teacher-forcing recall@10 を **greedy 生成タグ集合 vs gold の set-F1/Jaccard** へ置換し、テンプレ外自然文 val (tags-stay-real・Pool A val 由来 / Pool B 未使用 post・リーク0・gold⊆vocab) で採点。**物差し変更で D5 判定が符号反転**: 旧 recall で最下位 (D5 0.667 < #1 0.7767) が新 diverse F1 で最上位 (diverse_a **0.2024** vs #1 0.18 / diverse_b **0.2192** vs 0.1921・precision 0.38–0.41 vs 0.25)。in-dist (pairs.val) は 3 者同点 ~0.47。seed sweep (4 seed・6ep paired) で D5−#1 の diverse F1/Jaccard delta **全 4 seed 正・各 seed paired bootstrap 95%CI が 0 を除外** (p<8e-3) = **小幅だが頑健**な実効果 (+0.009〜+0.012 F1・絶対値は #1 seed 分散帯以下で strict 判定(b)のみ不成立)。D6 の符号反転 seed ノイズ (§12.5) と対照的。**確定: recall@10 退役・diverse 生成 set-F1 を新主指標**。本番重み/golden/既存 val 無改変・eval_report は provenance 付き。test_dollma_eval_diverse 緑 / test_dollma_eval_setmetrics は要 torch。training-spec §13 / dataset-spec §14 |
| 自作 入力多様化パイロット (Phase 4-B, scripts/dollma_make_diverse_train.py + train_bitnet.py `--train-file` + dollma_b_seedsweep{,_analyze}.py, **Claude 著述 Replace 500** (タグ固定=tags-stay-real・総件数 4500 維持)・施策 C の diverse-val 物差し上で採点, GTX1080Ti FP32 seed 20260620) | 施策 C (§13) で据えた **diverse-val + 生成 set-F1** の上で入力(自然文)多様化を測る。**diverse 生成 macro F1 が #1 を大幅に上回る**: diverse_a 0.1800→**0.2675** (+0.0875) / diverse_b 0.1921→**0.3039** (+0.1118)・precision も改善 (a +0.030 / b +0.052)・in-dist pairs.val −0.009 で**退行なし**・legacy recall@10 ≈同値 (タグ集合同一)。**seed 頑健性 sweep (4 seed 20260620/21/42/7・6ep paired・n≈1500/seed)** で delta (B−#1) は **全 4 seed 正 (a) / 分散帯の 4–6 倍 (b) / 各 seed paired CI が 0 を除外 (c) すべて成立**: diverse_a F1 **+0.0957±0.0184** / diverse_b F1 **+0.1263±0.0214** (paired t 全 16 セル p<8e-142)。**D5 (§13・(b)不成立の +0.009 小幅)・D6 (§12.5・符号反転 seed ノイズ) と桁違いに大きく頑健**。**著者交絡を否定**: 旧 D2 (Qwen2 著述・旧 recall で過学習悪化と却下) を diverse-val 再採点 (B-0) しても同等に改善 (diverse_a **0.2701** / b **0.3134** vs #1 0.18/0.1921) → 「Claude train が Claude test に似て上がった」では説明不可 = 多様化そのものの効果。旧 proxy なら却下されたはずで **C の物差しなしには可視化されなかった** (依存連鎖 C→B 実証)。**本番重みは #1 据え置き・無改変** (B-pilot 重みは別名 `bitnet_dense_diverse_b{,_fp32}`・本線昇格は project-leader/ユーザー決裁・絶対値なお ~0.26–0.31 と低帯域 → B 件数拡大/A 実ペア増/D 容量増と束ねて再評価)。Python のため C++ meson test 対象外 (test_dollma_make_diverse_train.py 6/6 緑・`--train-file` 既定で #1 経路と bitwise 非回帰・本番非汚染スナップショット diff)。training-spec §14 / dataset-spec §15 |
| 自作 入力多様化 件数拡大 (Phase 4-B-2, Replace 500→2,000・**Claude 著述 2,000 + synthetic 2,500**・tags-stay-real・施策C diverse-val 物差し, GTX1080Ti FP32 seed 20260620) | パイロット 500→2,000 で diverse 生成 macro F1 が**単調拡大**: diverse_a 0.2675→**0.3212** / diverse_b 0.3039→**0.3670**・precision も改善 (a 0.282→0.336 / b 0.316→0.377)・in-dist pairs.val 0.4539 (−0.009 退行なし)・legacy recall@10 0.7702 (同帯)。params 32,976,896 一致・val_loss ep4 底 2.456・訓練 116.7s。**seed sweep (4 seed 20260620/21/42/7・6ep paired・n=1500/seed)** delta(B-2−#1) diverse_a F1 **+0.1472±0.0102** / diverse_b F1 **+0.1788±0.0029** (Jaccard +0.105/+0.131) = 500版 (+0.096/+0.126) の **~1.4–1.5x に拡大しつつ seed sd は縮小** (diverse_b F1 sd 0.0214→0.0029)・判定 3 軸 (a)全seed正 (b)分散帯 ~6.7–8x (c)全 paired CI が 0 除外 すべて成立・paired t≈35–47 p<1e-269。**入力多様化はスケール則 (件数増で単調に強まり頭打ちなし・in-dist 据え置きで out-of-template のみ伸びる汎化方向)** — D5 (小幅・(b)不成立)・D6 (符号反転 seed ノイズ) と桁違いに対照的。絶対値はなお diverse_a ~0.32 / diverse_b ~0.37 と低帯域ゆえ**本線昇格は未決** (A 実ペア増 / D 容量増と束ね再評価)。本番重み #1 据え置き・別名 `bitnet_dense_diverse_b2000{,_fp32}` (gitignore)・sweep は scratch `_seedsweep_b2000/`。training-spec §14.8 / dataset-spec §15.6 |
| 自作 入力多様化 件数拡大 (Phase 4-B-3, Replace 2,000→10,000・**Claude 著述 10,000 + synthetic 2,000 = 総 train 12,000**・P=2,500 post × k=4 variant・B-2 のスーパーセット・tags-stay-real・施策C diverse-val 物差し, GTX1080Ti FP32) | **スケール則は ~2,000 件で飽和** (B-2 の「頭打ちなし」を訂正)。`make` を `--n-posts`/`--k-per-post` に一般化 (k=1 で B-1/B-2 bitwise 非回帰・test 14/14緑)。**seed sweep 4 seed (20260620/21/42/7・6ep paired・n=1500/seed)** delta(B10k−#1) は全 set/metric で **判定 YES (seed 頑健)**: diverse_a F1 **+0.1411±0.0206** / diverse_b F1 **+0.1761±0.0199** (Jaccard +0.102/+0.133)・(a)全seed正 (b)#1分散帯sd超え (c)全 paired CI が 0 除外・paired t≈28–46。**だが 2,000→10,000 の 5 倍増は追加利得ゼロ**: delta は B-2 (+0.1472/+0.1788) から平坦 (seed 分散内・むしろ微減)、b 絶対値も 0.319→0.313 (a) / 0.361→0.359 (b) で頭打ち。3 点 (500/2,000/10,000) で **500→2,000 大幅・2,000→10,000 飽和**が確定 — B-2 の「単調・頭打ちなし」は 2 点外挿の誤り。**運用結論**: 入力多様化単体の伸びしろは ~2,000 で尽きる → 残低帯域 (diverse_a ~0.31 / diverse_b ~0.36) は **B 件数でなく A 実ペア増 / D 容量増**で取る・**B 著述を 2,000 超に積む価値は薄く `[B-merge-at-A]` の既定多様化は b2000 で足りる** (b10k 不要)。本線昇格決裁 (2026-06-24) 不変・本番重み #1 据え置き・別名 `bitnet_dense_diverse_b10k{,_fp32}` (gitignore)・sweep scratch `_seedsweep_b10k/`。**PC ハング (2026-06-25・seed 20260621 b-arm eval 中) から冪等再開 (`_results/*.npz` skip) で完走**。training-spec §14.9 / dataset-spec §15.8 |
| マッティング/切り抜き (透過 PNG, character-bible §3 出力層, 学習済みセグメンテーション OV グルー = CLIP/WD14 と同立ち位置, GTX1080Ti/i7-10700 開発機) | **ISNet-anime 採用** (probe 比較: CPU 1.14s vs BiRefNet 15.7s=13.8x・アニメ髪 soft α 最適 soft_px/perim 14.1 vs 4.6・Apache-2.0・176MB)。OV IR FP32/FP16 静的 `[1,3,1024,1024]→[1,1,1024,1024]` (前処理 img/255 のみ・ONNX↔OV-FP32 max abs err **6.1e-6**・OV CPU 583-721ms)。`src/infer/matting.hpp` Matter (device 引数・OV ガード・f32 厳密一致) + compose_rgba (soft α・二値化なし・ストレート α premultiply なし) + `encode_png_rgba8` (png.hpp, color_type=6)。golden 突合 IoU≥0.99/MAE≤1e-3 は **OV 有効ビルドで実走** (本開発機は OV C++ 無で [SKIP]・compose_rgba は実走緑)。test_matting/png 緑。**M-5 (NPU/iGPU HW 比較・matting_device 確定) ✅ 完了 = 下行**。**M-6 (生成器結線) ✅ 完了**: 純 cpp interface `IMatter` + OV 隔離 factory (`src/server/matter_runner.{hpp,cpp,_stub.cpp}`・`make_matter`) を `IDiffusionRunner` と対称に新設し、`matting_postprocess.hpp` の `encode_png_maybe_transparent` (matter null/mask サイズ不一致/例外で不透明 PNG フォールバック) を Txt2ImgGenerator/PipelineGenerator 両方の最終段に結線。所有権が 3 段 DI を跨ぐ問題は `IImageGenerator::set_matter` (既定 no-op) の後付け注入で解消。CLI `--no-matting` で OFF (既定 ON)・`cli_generate.hpp` が `DOLLAMA_MATTING_WEIGHTS`/`DOLLAMA_MATTING_DEVICE` (既定 `GPU.0`=iGPU) で `make_matter` 1 回。開発機 (OV 無→stub nullptr) で test_matting 新サブテスト 3 件緑 (透過 color_type=6・不透明 color_type=2・サイズ不一致フォールバック) + `dollama --prompt` 不透明 PNG 出力で非回帰確認。**研究機 (OV+CUDA, isnet-anime IR 配置済) で end-to-end 実走確認済** ✅: ① 実 Matter golden (CPU) IoU=1 / MAE 9.99e-09 / 決定的 ② `dollama --prompt` → `matting: GPU.0` ログ + **color_type=6 透過 PNG** 出力 (生成器→iGPU マッティング→α 抽出→透過 PNG の配管が通る・テスト golden 重みは非キャラゆえ α≈0 だが配管は正) ③ iGPU(GPU.0) 実推論を実アニメ入力で検証 = **108.8ms** (M-5 99.96ms 一致域・CPU 261ms / 同一 mask)・mask range[0,1]・前景 25.4%・soft 中間α 4.6% = 意味のあるキャラ切り抜き。意味のある透過キャラ出力には実 SDXL 重み配置 (`DOLLAMA_UNET_WEIGHTS`) が前提 (配管は完了) | test_matting |
| マッティング ISNet-anime HW 比較 (Phase 4 M-5, `scripts/dollma_probe_matting_device.py`, isnetis.onnx を OV IR 静的化 `[1,3,1024,1024]→[1,1,1024,1024]` → CPU/iGPU(Xe)/RTX5080/NPU 比較, 研究機 Ultra9 285K/iGPU Xe/NPU AI Boost/RTX5080・OV2026.2) | 中央値: **iGPU(Xe) 99.96ms** < NPU 142.96 < CPU 204.20 < RTX5080(OV) 220.47。**iGPU/CPU 0.49x** (iGPU が CPU の ~2.0x 速)・NPU/CPU 0.70x。**matting_device = "iGPU" 確定**。ISNet は純 conv-UNet ゆえ **NPU も 1024² で compile/推論成功** (WD14 268ms の Window Attention と対照・純 conv は NPU フレンドリー) だが iGPU が最速。RTX5080 は OV plugin オーバヘッドで遅く本来拡散専任 (3.8s/VRAM 10.49GB) ゆえ除外。マッティングは拡散後段・1 枚 1 回で iGPU の VAE encode とも時間的非競合。レイテンシは重み/入力値非依存。`scripts/_matting_device_report.json` | (probe) M-5 |
| 品質スコアラ (Model B) NPU 実行性 probe (Phase 4 B, roadmap リスク「B の NPU 実行性」回収, `scripts/dollma_probe_quality_scorer.py`, 純 conv 代表 backbone ResNet-18 級 11.18M・attention 皆無を OV IR 静的化 `[1,3,H,W]` → CPU/iGPU(Xe)/RTX5080/NPU 比較, RTX5080 研究機/OV2026.2) | **NPU 最速・scorer_device=NPU 妥当**。中央値: 448² で **NPU 4.62ms** < iGPU 5.48 < CPU 8.35 < RTX5080(OV) 11.51 / 512² で **NPU 6.14ms** < iGPU 6.63 < CPU 10.54 < RTX5080 13.0。**NPU/CPU = 0.55〜0.58x (NPU が CPU の ~1.8x 速)**。**WD14 が NPU 268ms だったのは conv でなく Window Attention 由来と切り分け確定** — 純 conv なら NPU フレンドリー (WD14 比 ~58x)。拡散中 NPU 遊休に並列で実質ゼロコスト採点 → F 品質 FB ループに理想的。**前提: 実 §11 スコアラを純 conv で設計すること** (attention head を足すと NPU 不利に戻る恐れ)。RTX5080 が OV で遅いのは plugin オーバヘッド+小モデルゆえ (本来拡散専任で無問題)。レイテンシは重み値非依存 (乱数重み)。`scripts/_quality_scorer_report.json` | (probe) |
| 自作フル拡散パイプライン (タスク 2-6a, DiffusionPipeline = UNet×Nstep + Euler scheduler + VAE decode 結線, golden 埋め込み入力, CFG なし guidance=1, RTX5080) | 20step 1024×1024 実画像 初版 **84.07s** → **タスク 2-6 最適化後 11.30s** (probe10 3.80s 比 2.97x・累積 7.44x 改善)。最適化内訳は S2(conv im2col+wmma)/S3-0/A/B/C/D/E (下記 2-6 最適化 行)。最適化後の律速は UNet attention 4.60s / VAE decode 1.16s。出力 var=1244 / 全画素 [0,255] / NaN・Inf なし。VAE 画像化 (x*0.5+0.5→clamp→×255)。UNet 5.1GB+VAE 同時常駐可 (空き 14.9GB)。2step smoke が CI 緑判定・20step は DOLLAMA_BENCH=1 で計測のみ | test_diffusion |
| 自作 本番 txt2img (タスク 2-6b, prompt→画像の本結線, SDXL dual text encoder + CFG, NPU CLIP-L/bigG + RTX5080 拡散) | NPU で CLIP-L penult 768 ++ bigG penult 1280 = 2048 concat / bigG pooled 1280 = text_embeds、CFG guidance 7.5 で各 step cond/uncond の UNet 2 回 → host 合成。20step 1024² 実画像 PNG **var=2300** / 全画素 [0,255] / NaN・Inf なし。`IDiffusionRunner` (純cpp) を境界に OV (`Txt2ImgGenerator`/.cpp) と CUDA (`DiffusionRunner`/.cu) を隔離。main DI 3段フォールバック (Txt2Img→Pipeline(golden)→Stub)。test_txt2img は実アセットで 20step 実走 (NPU 構築失敗時 CPU フォールバック)・無ければ [SKIP]。meson test 35/35 全緑 | test_txt2img / test_text_conditioner |
| 自作 同一性条件付き実ペア増 (Phase 4-A Phase1, scripts/dollma_make_identity_pairs.py `--out-tag`/`--exclude-post-ids`, danbooru タグメタのみ・画像非取得, seed 20260620・B 多様化非適用で A 単体) | 5k→**12k/25k** 生成。**a12k** train 10,800/val 1,200・**a25k** train 22,500/val 2,500、両者 teacher_retention **1.0**・vocab retention **1.0** (OOV 0)・tokenizer 往復 UNK 0 完全一致。**リーク 0 証跡** (frozen eval 1000 post_id 直読との train/all 交差 0・excluded 交差 0・train/val post_id&text disjoint)。**identity 重複率 (val基準)** 5k 0.253 → 12k 0.2871 → 25k 0.3003 と件数増で漸増 (val の ~70% は train 未見 identity)。25k は cache を 38000→55200 posts へ過去方向延伸 (fetch-factor 2.2)・12k は cache 充足で API 取得なし。本番重み/golden/A1 5k/凍結 eval/#1 本線 train/val 無改変・cache 着手前を `.preA` に退避。成果物は別名 a12k/a25k (gitignore)。dataset-spec §17 | dollma_make_identity_pairs |
| 自作 同一性条件付き実ペア増 評価クローズ (Phase 4-A Phase2, scripts/dollma_a_seedsweep.py + dollma_a_seedsweep_analyze.py, base(#1 plain) vs a(A2 `--identity` 混合) を a12k で 4 seed (20260620/20260621/42/7) 6ep paired・凍結 diverse-val 生成 set-F1 + identity retention の二物差し, GTX1080Ti FP32) | **A の効果は二分・diverse-val F1 ではなく retention と確定 (A クローズ)**。**① diverse-val 生成 F1/Jaccard = seed ノイズ (頑健でない)**: 4 set/metric (diverse_a/b × F1/Jaccard) すべて判定 NO — diverse_a F1 per-seed delta(a−base) = [−0.0118, −0.0358, **+0.0286 (seed42 反転)**, −0.0405]・across **−0.0149±0.0316** < #1 帯 sd 0.0221 (b NO)・符号不一貫 (a NO)・各 seed CI は 0 除外 (c YES) → **D6 と同型の seed ノイズ**・施策 B ~2,000 飽和とも整合。**② identity retention = 全 seed 頑健に 0.975**: a arm across-seed **0.9748±0.0010** (base ~0.576–0.631・n_cases=1200) = **identity 条件付けの機能基盤**。in-dist pairs.val F1 ≈ base・recall@10 0.78→0.84・a の val_loss < base。**a25k は回さず未使用保持** (12k が seed ノイズである以上 25k 反転安定化は見込み薄)。本番 #1/identity/golden/凍結 eval 無改変・出力は `_seedsweep_a12k/` のみ (gitignore)。**本番即時差し替えなし・`[B-merge-at-A]` で B(b2000)+A(identity) を出荷リトレイン 1 回でまとめ焼き**。training-spec §9.10 / dataset-spec §17.7 / roadmap Phase 4 | dollma_a_seedsweep |
| 自作 容量増 seed sweep (Phase 4-D, scripts/dollma_d_seedsweep.py + dollma_d_seedsweep_analyze.py, c33(33M) vs d80(80M `DOLLAMA_BITNET_ARCH=d80m`・N_LAYERS 8→16/FFN 1792→2464=79,908,864) を**両アーム同一レシピ** (b2000 多様化 ∧ a12k identity)・`--arch` だけ差で 4 seed (20260620/20260621/42/7) 6ep paired・凍結 diverse-val 生成 set-F1 主指標 + retention/in-dist ガードレール, GTX1080Ti FP32) | **陰性確定・80M 不採用・勝者 = c33(33M)**。**① diverse-val F1/Jaccard = seed ノイズ (全 4 set/metric 判定 NO)**: delta(d80−c33) diverse_a F1 per-seed = [−0.0240, −0.0047, −0.0008, **+0.0133 (seed7 反転)**]・across **−0.0040±0.0154** < c33 帯 sd 0.0114 (b NO)・seed 20260620 負/seed 7 正で符号反転 (a NO)・各 seed CI は 0 除外がバラける (c 一部のみ) → **A12k/D6 と同型の seed ノイズ**・施策 B ~2,000 飽和と整合。diverse_b F1 across −0.0021±0.0181 も同様。**② ガードレール側も 80M 不利**: retention c33 across 0.9778 (全 seed ≥0.975 ✅) vs d80 0.9741 (**3/4 seed 床割れ** 0.9744/0.9739/0.9711)・in-dist pairs.val F1 c33 0.4599 > d80 0.4564 (微退行)。**容量 (33M→80M) では diverse-val F1 は取れない (データ律速)** — 打ち切り基準どおり 80M 不出荷。forward ~2x の対価に見合う実利なし。**施策 B (~2,000 飽和) のみが diverse-val F1 を頑健に押し上げ、A(retention 専)/D(陰性)/蒸留4路線(非寄与) は F1 非寄与**と確定。残低帯域 (diverse_a ~0.31/diverse_b ~0.36) は容量/件数でなく別軸 (多様性の質・アーキ・損失・本命 F) で取る。正典化は `[B-merge-at-A]` で勝者 33M を 1 回まとめ焼き。本番重み/golden/凍結 eval 無改変・出力 `_seedsweep_d80m/` のみ (gitignore)・test 8/8 緑。training-spec §16 / dataset-spec §18 / roadmap Phase 4 | dollma_d_seedsweep |

## 次のタスク

**C++ 実装フェーズ (Phase 1 — パイプライン骨格)**

| # | 実装物 | ファイル | 状態 |
|---|---|---|---|
| 1 | Meson ビルド + src/ 構造 | `meson.build`, `src/` | ✅ 完了 |
| 2 | Tensor クラス + テスト | `src/core/tensor.hpp`, `test_tensor.cpp` | ✅ 完了 |
| 3 | Allocator + テスト | `src/core/allocator.hpp`, `test_allocator.cpp` | ✅ 完了 |
| 4 | SPSC キュー + テスト | `src/core/queue.hpp`, `test_queue.cpp` | ✅ 完了 |
| 5 | CLIP NPU 推論 + テスト | `src/infer/clip.hpp`, `test_clip.cpp` | ✅ 完了 (NPU 7.82ms) |
| 5.5 | キャラ台帳 character.hpp + テスト | `src/core/character.hpp`, `test_character.cpp` | ✅ 完了 |
| 6 | WD14 CPU 推論 + テスト | `src/infer/wd14.hpp`, `test_wd14.cpp` | ✅ 完了 (CPU 105.3ms) |
| 7 | スレッド骨格 + CPU アフィニティ | `src/main.cpp` + `src/core/affinity.hpp` + `src/pipeline.hpp` | ✅ 完了 (9.13 frames/s) |

**Phase 2 以降 (詳細は `docs/roadmap.md` 参照)**
- `src/io/safetensors.hpp` — safetensors 重みローダー ✅ 完了 (タスク 2-3, golden 突合, 19.0 µs/op)
- `src/kernels/vae_decode.cu` — VAE decode ✅ 完了 (タスク 2-4, SSIM 0.999992)
- `src/infer/unet.cu` + `src/infer/scheduler.hpp` — SDXL UNet + Euler scheduler ✅ 完了 (タスク 2-5, noise_pred SSIM 0.999998, 1step ~9.2s)
- `src/server/api.cpp` — cpp-httplib OpenAI 互換 HTTP サーバー ✅ 完了 (Phase 3, 生成本体は IImageGenerator 越し)
- `src/infer/diffusion.cu` + `src/server/pipeline_generator.hpp` — フル C++ 拡散パイプライン統合 ✅ 完了 (タスク **2-6a**, 20step 実画像 84.07s)。`DiffusionPipeline` (UNet×Nstep+Euler+VAE) を `PipelineGenerator` で `IImageGenerator` 化し、main.cpp が `pipeline_generator_factory` 経由で DI (重み/golden 揃えば本 txt2img・無ければ StubGenerator フォールバック・env DOLLAMA_UNET_WEIGHTS/VAE_WEIGHTS/EMBEDS で上書き)。CUDA 隔離は HAVE_CUDA ガード + .cu factory で cpp TU 非汚染
- **タスク 2-6 最適化**: 🟡 **一旦クローズ (2026-06-22)**。84.07s → **11.30s** (7.44x 改善・probe10 3.80s 比 2.97x)。S2(conv im2col+wmma) / S3-0(ビルドフラグ) / S3-C(attn wmma) / S3-B(GEMM cuBLAS GemmEx) / S3-A(VAE mid flash) / S3-D(VAE 重み常駐ハンドル) / S3-E(VAE up2/up3 FP32 direct conv → im2col+cuBLAS TF32 GEMM, 5.21s→1.16s) を実施。残る律速は UNet attention 4.60s。**研究の本丸を Phase 4 独自 LM (BitNet) に移すため、自作 CUDA の速度詰めはここで打ち切り** (残りは cuBLAS/cuDNN ライブラリフォールバック余地)。golden 維持 (UNet noise_pred SSIM 0.999996 / VAE SSIM 0.999988 / meson test 28/28 全緑)
- **2-6b (テキスト→画像の本結線)**: ✅ **完了 (2026-06-23, c3b1dac)**。2-6a の golden 埋め込み使い回し・CFG なしを解消し、prompt→実画像を dual encoder + CFG で一気通貫。① **CLIP-G の OV 化** (SDXL dual encoder: CLIP-L penult 768 ++ bigG penult 1280 → concat 2048・bigG pooled 1280 = text_embeds) ② CFG (negative prompt) で各 step cond/uncond の UNet 2 回 → host で `noise=uncond+scale·(cond-uncond)` 合成 ③ prompt→embeds 結線。Stage A golden ダンプ / B OV変換 (tokenizer・encoder L/G) / C bigG device probe / D `TextConditioner` (clip_tokenizer・clip_encoder2・text_conditioner.hpp) / E `DiffusionPipeline::generate_txt2img` / F 結線 (純cpp `IDiffusionRunner` を境界に OV/CUDA 分離・`Txt2ImgGenerator`(OV)→runner→`DiffusionRunner`(.cu)・main DI 3段フォールバック Txt2Img→Pipeline(golden)→Stub)。NPU で CLIP-L/bigG、guidance 7.5・20step、PNG 1024² var=2300・NaN/Inf なし。meson test 35/35 全緑。guidance_scale は定数 7.5 (GenRequest フィールド化は後続 TODO)。**prompt 供給元は将来 Phase 4 A の自作タグ生成 LM に差し替え**
- 部位構造化プロンプト ([[project-part-structured-prompt]], character-bible-spec §1 改訂) — 2-6 後に §11 QA ループ・案B embedding と一緒に設計
- **Phase 4 (自作タグ生成 LM)**: bitnet.hpp 33M を土台に ① tokenizer ✅ ② 訓練 (`scripts/train_bitnet.py`, hard CE dense) ✅ ③ CPU 推論 (`src/infer/bitnet.hpp` BitNetDenseInfer, golden 突合 corr 1.0) ✅ ④ **同一性条件付き化** (character-bible 入力・prompt prefix 方式・A1 retention 0.947) ✅ + **実ペア増 a12k 4 seed sweep でクローズ** — **A の効果は二分**: diverse-val 生成 F1 は **12k で seed ノイズ** (4 set/metric 判定 NO・seed 42 のみ符号反転・across −0.015±0.032 < #1 帯 sd 0.022・D6 と同型・B ~2,000 飽和と整合) で **F1 を上げる手ではない**が、**identity retention は全 seed 頑健に 0.975** (across 0.9748±0.0010・base ~0.58–0.63) = **同一性条件付けの機能基盤**。a25k は未使用保持・本番 #1 即時差し替えなし・`[B-merge-at-A]` で B(b2000)+A(identity) を出荷リトレイン 1 回でまとめ焼き (training-spec §9.10 / dataset-spec §17.7) ✅ — **dense 本線 + 2D 特化 A が text→tags / identity→tags で動作**。**蒸留 2 路線とも検証済**: 系列レベル hard CE 混合 (D2/D4, Qwen2 text 多様化) は過学習悪化で不採用 (負の結果・training-spec §10)。**soft-label KL (D5, 案A=`cache/danbooru_posts.jsonl` の prefix 条件付き共起 teacher・温度付き)** は過学習を**実際に抑制** (train-val gap 1.09→0.27、D5-L で −0.21=消失・反転 ep4→ep6) したが top10 recall は #1 を超えず (最良 0.758<0.777・soft 追従と one-hot val recall がトレードオフ) **不採用・#1 6ep 本線維持** (training-spec §11 / dataset-spec §12.3 に機構・A/B 記録)。D5 は小規模 split / early-stopping 運用で再利用価値ありの正の機構。**案C = 外部教師 TIPO-200M (D6・`--distill-ext`)** も実施・**不採用**: 案b-tagset (TIPO 生成タグを自作 vocab に完全一致写像) で単一 seed では #1 を僅かに上回って見えたが、**seed 頑健性 sweep (4 seed) で再現せず** (delta 平均 −0.0008・符号反転・#1 の seed 分散内)。再現する効果は過学習抑制のみ (gap 1.07→0.19・全 seed) = D5 と同じ正則化ノブで recall 非寄与 (training-spec §12 / dataset-spec §12.4)。**蒸留 4 路線 (D2/D4/D5/D6) は旧 proxy (テンプレ teacher-forcing recall@10) では全て不採用**。⑤ **施策 C (評価作り直し) 完了** (C-1〜C-4・training-spec §13 / dataset-spec §14): diverse-val (テンプレ外自然文・tags-stay-real) + greedy 生成 set-metrics + eval-only ハーネス + seed sweep を実装。**物差しを変えると D5 判定の符号が反転**した — 旧 recall で最下位 (0.667) の D5 が新 diverse 生成 F1 では最上位 (diverse_a 0.2024 vs #1 0.18・precision 0.38 vs 0.25)、seed sweep で D5−#1 の F1/Jaccard delta が**全 4 seed 正・各 seed paired CI が 0 を除外** (D6 の符号反転 seed ノイズと対照) = **小幅だが統計的に頑健**な実効果 (delta +0.009〜+0.012・絶対値は #1 の seed 分散帯以下)。**確定: recall@10 を主要数値から退役させ diverse 生成 set-F1 を新オフライン主指標に据える**。**未決: D5 本線昇格は新物差しで別途判断** (絶対値なお ~0.18–0.22 と低い → A 実ペア増 / D 容量増と束ねて再評価)。本番重みは #1 据え置き・無改変。⑥ **施策 B (入力多様化) パイロット完了** (B-0〜B-1-d・training-spec §14 / dataset-spec §15): C で据えた diverse-val 物差しの上で、タグ固定 (tags-stay-real)・自然文だけ多様化・**Claude 著述 Replace 500** (総件数 4,500 維持) を測ったところ、**diverse 生成 F1 が #1 を大幅に上回り** (diverse_a 0.1800→0.2675 / diverse_b 0.1921→0.3039・in-dist 退行なし)、**seed sweep (4 seed) で全判定軸 (a)(b)(c) 成立** (delta +0.06〜+0.13 = 分散帯の 4–6 倍・D5 の小幅・D6 の seed ノイズと桁違いに対照的) = **大幅かつ頑健な本物の効果**。**著者交絡は D2 で否定** (旧 Qwen2 著述 D2 を diverse-val 再採点 = B-0 しても同等改善 diverse_a 0.2701 / b 0.3134 → 「Claude 同士で似て上がった」では説明不可)。**旧 proxy では D2 同様却下されたはずで C の物差しなしには可視化されなかった** (蒸留4路線→C→B の流れ・依存連鎖の実証)。**本番重みは #1 据え置き・別名 `bitnet_dense_diverse_b` 出力**。**B-2 件数拡大 (500→2,000) ✅ + 本線昇格決裁済** (2026-06-24 ユーザー・多様化入力=tags-stay-real を既定レシピ化・正典差し替えは A 出荷リトレインで 1 回=遅延条項 `[B-merge-at-A]`)。**B-3 件数拡大 (2,000→10,000) ✅ = スケール則は ~2,000 件で飽和**: seed sweep 4 seed で delta は全 set/metric 判定 YES (seed 頑健) だが 2,000→10,000 の 5 倍増は追加利得ゼロ (delta 平坦・絶対値頭打ち) → B-2 の「頭打ちなし」を訂正。**入力多様化単体の伸びしろは ~2,000 で尽き、残低帯域 (diverse_a ~0.31 / diverse_b ~0.36) は B 件数でなく A 実ペア増 / D 容量増で取る** (B 著述を 2,000 超に積む価値は薄い・既定多様化は b2000 で足りる)。training-spec §14.9 / dataset-spec §15.8。⑦ **#6-GPU (RTX5080 で CUDA カーネル流用) ✅ 完了** (`src/infer/bitnet_gpu.cu/.cuh` BitNetGpuInfer・純 FP32 cuBLAS で CPU 版と数値一致 corr 1.0・本番重み突合済・forward seq63 で **87.5x** = 10.29ms vs CPU 900.83ms・test_bitnet_gpu 緑)。⑥ **Model B 品質スコアラ Stage 1 完了** (`src/infer/quality_gate.hpp` `QualityGate`): §11 数・位相 QA を二段構成に切り分け、一段目 = 既存 Wd14Tagger 出力 [1,N_TAGS] を消費する **OV 非依存 soft QA ゲート**を実装。`extra_arms`/`multiple_heads`/`bad_anatomy` 等 18 異常タグを軸 (`AnomalyAxis` Hands/Limbs/Head/Eyes/Ears/Mouth/Digits/GlobalAnatomy) で束ね、selected_tags.csv から名前駆動で index 解決 (WD14 版差異に頑健・CSV 欠落名は安全 skip)。`evaluate` は閾値超を hit 収集し soft flag のみ (棄却なし・§11「異常度の高いコマのみ拾う」)・`DIGITS_UNCOUNTABLE` キャラは Digits/Hands 軸スキップ。flag は B-5 FB ループ入力 + 蒸留 teacher ラベル生成器。実 selected_tags.csv で 18/18 解決・test_quality_gate 5 サブテスト緑 (OV 無し開発機で実走・新規モデル入手ゼロ)。NPU 実行性は 4-B probe 済 (scorer_device=NPU 妥当・純 conv は NPU フレンドリー)。**Stage 2 切り出し** = NPU 正確カウント検出 (DWPose 等・指の過剰カウント。モデル入手前提・別タスク。WD14 に `extra_digits`/`fused_fingers` 無く `fewer_digits` のみ = 語彙が二段構成を裏付け)。残: 実スコアラ本体 (純 conv backbone・蒸留訓練 B-3) / FB ループ B-5。B/A/D/F は diverse-val 上で測る。ternary GEMM (`src/kernels/ternary_gemm.cu`) は圧縮実験。方向性の経緯は roadmap Phase 4

## 実装作業のルール

1. **プランモードで設計を提示し、ユーザーのレビューと承認を得てから着手する。** 承認なしに勝手にコードを書き始めない。
2. **承認後は必ず `project-leader` エージェントを呼び出し、作業を担当エージェントに振り分けてもらう。** Claude 自身が直接実装せず、PL の指示のもと専門エージェントが実装する。
3. **ゴールが設定された場合、承認作業は `project-leader` が行う。** ユーザーへの判断依頼は PL が迷ったときのみ。
4. **コンポーネントを実装したら、必ずテストも実装する。** `src/tests/test_<component>.cpp` を作成し、`meson test -C build` が通ることを確認してから完了とする。テスト規約は `docs/testing.md` 参照。

## コーディング規約

- ファイル名プレフィックス: `dollma_` (dollama のプロジェクト内ファイル)
- プローブスクリプトは `scripts/dollma_probe*.py`
- 本実装は `src/` 以下に C++ で記述
- ビルド: Meson (`meson setup build && meson compile -C build`)
- コメントは日本語で書く

### C++ スタイル

開き波括弧 `{` は必ず改行して次の行に置く (Allman スタイル):

```cpp
void abc()
{
    // ...
}
```

`switch` 文の `case` ラベルは `switch` と同じタブ位置に揃える:

```cpp
switch (x)
{
case 1:
    break;
case 2:
    break;
}

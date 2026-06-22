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
| iGPU (Intel Xe) | VAE encode (img2img、79ms) | ✅ CPU 117ms より速い |
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
| 自作 dense タグ生成 LM 推論 (Phase 4-6, src/infer/bitnet.hpp BitNetDenseInfer, safetensors FP32 ロード→forward+greedy デコード text→tags, CPU 純ホスト, GTX1080Ti/i7-10700 MSVC) | PyTorch golden 突合: logits seq 8/32/63 で max abs err **2.86e-6〜2.29e-5** / **corr 1.0**、greedy デコード **5/5 完全一致** (語彙外日本語含む)。RoPE NeoX・[out,in]・tied lm_head・eps を golden 一致で検証。1 forward (seq8) **~253ms** (naive 参照・lm_head 4999×512 全走査支配・速度は #6-GPU/量子化で改善)。models/bitnet.hpp の ternary BitLinear 不使用・dense FP matmul double 蓄積。test_bitnet_infer 緑 (golden/重み不在時 [SKIP]) |
| 自作 同一性条件付きタグ生成 (Phase 4-A, character-bible を条件入力・2D 特化の核, 機構 (a-1) prompt prefix `<bos> identity <sep> scene <sep> target <eos>`・`<sep>` 2回流用で vocab/tokenizer/アーキ無変更, GTX1080Ti FP32) | A1 §13 データ 5000ペア (identity/scene 分離・identity⊆target=retention 100% 教師・validate 全件0・train/val identity 重複 0.253)。A2 #1+§13 混合 6epoch 訓練で **identity retention 0.947** (生成 target が入力 identity の約95%を再現・条件付け実証)・source 別 val recall synthetic 0.900/identity 0.946。A3 C++ `BitNetDenseInfer::generate_with_identity` + `CharacterBible.find->canonical_tags` 結線 (tokenizer/character 無改修)。identity golden 突合: logits corr 1.0 / greedy 5/5 完全一致。test_bitnet_infer 緑 |
| マッティング/切り抜き (透過 PNG, character-bible §3 出力層, 学習済みセグメンテーション OV グルー = CLIP/WD14 と同立ち位置, GTX1080Ti/i7-10700 開発機) | **ISNet-anime 採用** (probe 比較: CPU 1.14s vs BiRefNet 15.7s=13.8x・アニメ髪 soft α 最適 soft_px/perim 14.1 vs 4.6・Apache-2.0・176MB)。OV IR FP32/FP16 静的 `[1,3,1024,1024]→[1,1,1024,1024]` (前処理 img/255 のみ・ONNX↔OV-FP32 max abs err **6.1e-6**・OV CPU 583-721ms)。`src/infer/matting.hpp` Matter (device 引数・OV ガード・f32 厳密一致) + compose_rgba (soft α・二値化なし・ストレート α premultiply なし) + `encode_png_rgba8` (png.hpp, color_type=6)。golden 突合 IoU≥0.99/MAE≤1e-3 は **OV 有効ビルドで実走** (本開発機は OV C++ 無で [SKIP]・compose_rgba は実走緑)。test_matting/png 緑。**NPU/iGPU HW 比較 (matting_device 確定) と PipelineGenerator 結線は研究機 (M-5/M-6)** | test_matting |
| 自作フル拡散パイプライン (タスク 2-6a, DiffusionPipeline = UNet×Nstep + Euler scheduler + VAE decode 結線, golden 埋め込み入力, CFG なし guidance=1, RTX5080) | 20step 1024×1024 実画像 **84.07s** (cudaEvent, probe10 3.80s 比 **22.1x slower**)。出力 var=1244 / 全画素 [0,255] / NaN・Inf なし。律速は UNet 1step ~9.2s×20 + VAE ~8s (direct conv + per-row attention)。VAE 画像化 (x*0.5+0.5→clamp→×255)。UNet 5.1GB+VAE 同時常駐可 (空き 14.9GB)。2step smoke が CI 緑判定・20step は DOLLAMA_BENCH=1 で計測のみ | test_diffusion |

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
- **次 (タスク 2-6 最適化)**: 84s → 3.80s が速度の本丸。**direct conv → im2col/Tensor Core GEMM**・**naive attention (per-row) → flash** が UNet/VAE 両方の律速。FP16 Tensor Core (cuBLAS フォールバック許容) で大幅短縮見込み
- **2-6b (テキスト→画像の本結線・別タスク)**: 現状 2-6a は golden 埋め込み使い回し・CFG なし。本物の txt2img には ① **CLIP-G の OV 化** (SDXL dual encoder: CLIP-L 768 + CLIP-G 1280 → concat 2048・pooled 1280) ② CFG (negative prompt) でバッチ2相当の 2 回 UNet ③ prompt→embeds 結線 (PipelineGenerator の TODO(2-6b)) が要る。model-converter + npu-benchmarker + cpp-implementer を巻き込む
- 部位構造化プロンプト ([[project-part-structured-prompt]], character-bible-spec §1 改訂) — 2-6 後に §11 QA ループ・案B embedding と一緒に設計
- **Phase 4 (自作タグ生成 LM)**: bitnet.hpp 33M を土台に ① tokenizer ✅ ② 訓練 (`scripts/train_bitnet.py`, hard CE dense) ✅ ③ CPU 推論 (`src/infer/bitnet.hpp` BitNetDenseInfer, golden 突合 corr 1.0) ✅ ④ **同一性条件付き化** (character-bible 入力・prompt prefix 方式・retention 0.947) ✅ — **dense 本線 + 2D 特化 A が text→tags / identity→tags で動作**。**蒸留 2 路線とも検証済**: 系列レベル hard CE 混合 (D2/D4, Qwen2 text 多様化) は過学習悪化で不採用 (負の結果・training-spec §10)。**soft-label KL (D5, 案A=`cache/danbooru_posts.jsonl` の prefix 条件付き共起 teacher・温度付き)** は過学習を**実際に抑制** (train-val gap 1.09→0.27、D5-L で −0.21=消失・反転 ep4→ep6) したが top10 recall は #1 を超えず (最良 0.758<0.777・soft 追従と one-hot val recall がトレードオフ) **不採用・#1 6ep 本線維持** (training-spec §11 / dataset-spec §12.3 に機構・A/B 記録)。D5 は小規模 split / early-stopping 運用で再利用価値ありの正の機構。残る蒸留路線は **案C = DanTagGen/TIPO 外部教師 (D6・要 vocab 整合・真の知識転移で recall 底上げ狙い・roadmap バックログ)**。残: #6-GPU (RTX5080 で CUDA カーネル流用)、⑤ アニメ**品質スコアラ** (B, NPU, §11)。ternary GEMM (`src/kernels/ternary_gemm.cu`) は圧縮実験。方向性の経緯は roadmap Phase 4

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

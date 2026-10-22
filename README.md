# dollama

**[English](README_en.md)** | **日本語**

**CPU / NPU / iGPU / RTX5080 — 搭載する全 HW を使い切りながら、2D イラスト/漫画のキャラクター描画生成にたどり着くことを研究するプロジェクト**

最短実装ではなく、各 HW をどう活かし、どう協調させるかを探ることがこのプロジェクトの本質。
**推論パス**は PyTorch / diffusers / llama.cpp に頼らず、C++ と自作 CUDA カーネル (+ cuBLAS / OpenVINO ランタイム) で実装する。
モデルの訓練・変換・golden 生成は Python (PyTorch / diffusers / OpenVINO) で行う。

> ⚠️ **研究中のプロジェクトです (完成品ではありません)。** **研究プロセスの公開**を目的としており、すぐ使えるツールではありません。
> API・ファイル構成・計測値は今後も変わります。
>
> - 現在できること: 任意テキスト → **1024×1024 固定**の画像を、自作 C++/CUDA の SDXL 経路 (dual text encoder + CFG + UNet + Euler + VAE decode) で生成し、
>   iGPU のマッティング (ISNet-anime) で **切り抜き済み透過 PNG** にする。既定の重みはアニメ特化 SDXL の preset `illustrious-xl`。
> - 速度: 出荷経路 (CFG g=7.5・`--fast`・CFG batch=2) の実測は **10.65〜10.96 s/枚** (2026-09-04・研究機・UNet 20step + VAE decode・テキストエンコードを除く)。
>   diffusers による参考値 (probe10) **3.80s** はテキストエンコード込みの別条件の値で、自作経路は**それより約 2.8 倍遅い** (条件差があるため目安)。
>   速度改善は継続中 (G-10k は 2026-09-17 に陰性クローズ・次の G-14k attention 書き直しは計画段階)。
> - 未完成のもの: 自作タグ生成 LM の**パイプライン結線** (現状 prompt は直入力)・img2img (`/v1/images/edits` は 501)・WD14 タグのフィードバック。

> **本研究は Intel Core Ultra 9 285 (Intel AI Boost NPU + Intel Xe iGPU) と NVIDIA RTX 5080 を組み合わせた 1 台の構成で行っている。**
> NPU / iGPU の活用 (OpenVINO 経由) は Intel プラットフォーム固有の特性に依存しており、計測値・デバイス選定はこの環境を前提とする。

---

## 概要と現在地

dollama は、1 台のマシンに載った **CPU・NPU・iGPU・dGPU を 1 つの推論パイプラインとして協調**させ、
2D イラスト/漫画の**キャラクター**画像 (背景なし・切り抜き済み透過 PNG) を生成する研究プロジェクト。
背景は生成しない (外部ツールで合成する前提)。

**コア・アイデア — "時間方向の協調" (構想):** HW を直結 (ゼロコピー) でつなぐのではなく、
GPU が拡散処理に数秒かける間に、遊休している NPU/CPU/iGPU で別の段 (テキスト埋め込み・採点・タグ抽出など) を回す。
各 HW を**得意なタスクに割り当て**、待ち時間を互いに埋め合う。
現状の出荷 txt2img は 1 リクエストを逐次処理しており、並列化は実寸スタブでの検証段階 (下記「計測ベースライン」)。

| フェーズ | 内容 | 状態 |
|---|---|---|
| Phase 1 | C++ パイプライン骨格 (Tensor / Allocator / SPSC Queue / CLIP-NPU / WD14-CPU / スレッド骨格 + アフィニティ) | ✅ 完了 (縦通し 9.13 frames/s・WD14 律速) |
| Phase 2 (2-0〜2-6a) | 自作 CUDA カーネル + safetensors + VAE decode + SDXL UNet + Euler + フル拡散ループ結線 | ✅ 完了 (初版 20step 84.07s) |
| 2-6 最適化 | direct conv → im2col + GEMM・naive attention → flash + wmma・巨大 GEMM の cuBLAS 化 | 🟡 2026-06-22 に一旦クローズ (84.07s → 11.30s・ただし**非 CFG・B=1・`generate()` 経路の値で現在は陳腐化**)。以後も速度改善を継続: `--fast` に含まれるもの (G-2k CFG batch2 / G-3k attention / G-4k epilogue 融合)・既定 ON で全経路に効くもの (G-8k VRAM アリーナ・キルスイッチ `DOLLAMA_POOL=0`)・env opt-in (G-10k conv 真 batch2 `DOLLAMA_CONV_BATCH=1`・陰性クローズ) |
| 2-6b | 本 txt2img: SDXL dual encoder (CLIP-L + CLIP-G) + CFG + negative prompt + マッティング → 透過 PNG | ✅ 完了 (2026-06-23・`c3b1dac`) |
| 2-6c | 拡散 backend プラグイン枠 (`IDiffusionBackend` registry: `"sdxl"` 稼働 / `"sd35"` stub) | ✅ 完了 (2026-07-02) |
| 2-6d | アニメ特化 SDXL 3 preset (NoobAI-XL 1.1 / Animagine XL 4.0 / Illustrious XL v0.1)。全て SDXL アーキゆえカーネル無改修・重みのみ差し替え | ✅ 完了 (2026-09-17・既定 = `illustrious-xl`・素 SDXL は `--preset base`) |
| 2-6e | preset の `prompt_prefix` / `negative_prefix` を生成時に自動付与 | ✅ 完了 (2026-09-19・`4d4a405`) |
| 2-6f | Blazor UI から複数の dollama サーバーを実行時に切替 | ✅ 完了 (2026-09-19) |
| 2-6 残件 | HTTP API の preset フィールド / UI の preset 選択 | 🔲 未着手 |
| Phase 3 | OpenAI Images 互換 HTTP サーバ (cpp-httplib / nlohmann-json) | ✅ 完了 (生成器は `BackendImageGenerator` → `PipelineGenerator` → `StubGenerator` の 3 段フォールバックで DI) |
| LoRA (L-1 / L-2) | offline merge スクリプト + ランタイム LoRA (HTTP `loras` / UI 選択チップ) | ✅ 完了 |
| Phase 4 | 自作タグ生成 LM (33M) + 同一性条件付け (A) + アニメ品質スコアラ (Model B) + 品質フィードバック (F) | ⏳ 進行中 (下記「自作タグ生成 LM」) |

> 詳細なロードマップは [`docs/roadmap.md`](docs/roadmap.md)、計測の完全版は [`docs/measurements-log.md`](docs/measurements-log.md)。

---

## 何が分かったか

probe と自作実装で得た要点。**数字より結論を先に**:

1. **HW 選定は「タスク × アーキテクチャの相性」で決まる — 事前予想は当てにならない。**
   NPU は固定形状の encoder (CLIP-L 77token) で CPU の 2.5 倍速 (7.85ms) と最速。
   一方、同じ "推論" でも WD14 SwinV2 (Window Attention) では NPU が**最遅** (268ms、CPU 101ms)、
   LLM の自己回帰 (Phi-3 mini / Qwen2-1.5B INT4) は NPU で**コンパイルが通らなかった** (動的形状・KV-cache が NPU の設計外)。
   「NPU = 速い」ではなく「固定形状の純粋 GEMM チェーンだけ速い」。→ 全モデルを実測してから割り当てた。

2. **HW 間のゼロコピー直結は (この構成では) 不可能。CPU pinned 経由が現実解。**
   CUDA↔NPU の interop API は存在せず、iGPU は BIOS 設定で DXGI に非表示 (D3D12 クロスアダプター不可)。
   CPU pinned memory 経由の転送オーバーヘッドは **3.4%** で、拡散処理の裏に隠せる規模だった。

3. **iGPU は "方向" で使い分ける。** 大規模 Conv (VAE decode スタブ) は CPU の 8 倍遅い (995ms vs 126ms) が、
   VAE **encode** は CPU 117ms に対し **79ms** と速い (probe5)。マッティング (ISNet-anime) も iGPU が最速 (99.96ms) で、出荷経路で iGPU 担当になった。

4. **diffusers (PyTorch) の SDXL は RTX5080 で 3.80s/枚 (1024², 20steps, CFG あり, テキストエンコード込み)、VRAM ピーク 10.49GB (probe10)。**
   これが自作経路の比較基準。自作経路は現状 10.65〜10.96 s/枚 (テキストエンコード除く) で、まだ届いていない。

5. **ML フレームワーク無しの自作 CUDA カーネルでフル拡散パイプラインが動き、golden と一致する。**
   GEMM / 活性化 / GroupNorm / Conv2d / Attention を実装し、diffusers からダンプした中間テンソルとのゴールデンテストで正当性を確認。
   自作 VAE decode は SSIM **0.999988**、自作 SDXL UNet は noise_pred SSIM **0.999996** (いずれも 2-6 最適化後の値。2-4/2-5 当時はそれぞれ 0.999992 / 0.999998。VAE の変化点は `b3fe139`)。

6. **数値ゲートが全部緑でも、速くなるとは限らない。** G-10k (conv2d の真 batch2 化) は bit-exact を含む数値ゲートを全て通したが、
   resnet バケットは +3.38% / +1.87% と悪化 (秒中立) → opt-in (`DOLLAMA_CONV_BATCH=1`) に降格して陰性クローズした (`c3ee3ca`)。律速は GEMM / im2col が N に比例すること。

7. **タグ生成 LM の汎化を頑健に押し上げたのは「入力の多様化」だけだった。** 蒸留 4 路線・実ペア増・容量増 (33M→80M) は
   主指標 (diverse 生成 set-F1) を seed ノイズを超えて動かせず、自然文入力の多様化 (施策 B) のみが効き、しかも ~2,000 件で飽和した。

→ 結論 (現時点): **「全 HW を使い切る」は、実測に基づいて割り当てれば成立しうる。** 鍵は直結ではなく、
"各 HW を得意タスクに割り当て、GPU 生成中の遊休を他の HW で埋める" 時間方向の協調にある。ただし並列化そのものは未実装。

---

## ターゲット環境

| コンポーネント | 詳細 |
|---|---|
| CPU / NPU | Intel Core Ultra 9 285 (NPU = Intel AI Boost, DEVICE_ARCHITECTURE: 3720) |
| GPU | NVIDIA GeForce RTX 5080 (Blackwell / sm_120, VRAM 16GB) |
| iGPU | Intel Xe Graphics (OpenVINO `GPU.0`、システム RAM 共有) |
| OS | Windows 11 |
| CUDA | 要件 12.8+ (sm_120) / 検証 13.3 |
| 調査・訓練 | Python 3.14 + PyTorch (cu128) / diffusers / OpenVINO |
| 本実装 | C++20 + Meson (MSVC で検証)。UI は .NET 10 (Blazor Server) |

---

## HW の割り当て

### 実装済み (出荷 txt2img の経路)

`src/server/cli_generate.hpp` の `build_image_generator` が組み立てる、CLI / HTTP 共通の生成経路。

| HW | 担当 | 備考 |
|---|---|---|
| **CPU** | OV tokenizer (CLIP-L / CLIP-G のトークン化) | 毎リクエスト CPU で実行 (`src/infer/clip_tokenizer.hpp` 既定 device) |
| **NPU** | SDXL dual text encoder (CLIP-L + CLIP-G) | NPU での構築に失敗したら CPU へ退避 |
| **NPU** | ScorerNet 品質採点 (自作 11.18M・解剖 8 軸) | 採点結果は**ログ出力のみ** (消費者は未実装)。NPU 8.32ms (B-3c) |
| **iGPU** (`GPU.0`) | ISNet-anime マッティング (α 抽出 → 透過 PNG) | 99.96ms (M-5 probe)。`--no-matting` で OFF |
| **RTX5080** | 自作 SDXL UNet × 20step (CFG) + Euler + VAE decode | `--fast` で 10.65〜10.96 s/枚 (下記) |

```
prompt (直入力。preset の prompt_prefix を自動付与)
  │
  ▼
[NPU] CLIP-L + CLIP-G (SDXL dual encoder)  ※失敗時 CPU
  │
  ▼
[RTX5080] 自作 SDXL UNet ×20step (CFG) + VAE decode → RGB 1024×1024
  │
  ├─ [NPU]  ScorerNet 採点 (ログのみ)
  └─ [iGPU] ISNet-anime マッティング → 透過 PNG
```

後処理 (採点 → マッティング) は生成の後に逐次実行される (`src/server/backend_image_generator.hpp`)。

### 構想 (未実装)

| 段 | 想定 HW | 現状 |
|---|---|---|
| 自作タグ生成 LM (自然文 → danbooru タグ) | GPU 主 (LM 専用の FP32 cuBLAS + 専用カーネル `src/infer/bitnet_gpu.cu`・拡散用カーネルは流用していない) / CPU 可 / NPU 不可 | モデル・C++ 推論は実装済み (GPU forward seq8 2.43ms) だが**パイプライン未結線**。現状は prompt を直入力 (`src/server/backend_image_generator.hpp` → `sdxl_backend.hpp` の text encoder へそのまま渡る) |
| WD14 タグ抽出 → フィードバック | CPU | WD14 推論グルー自体は Phase 1 骨格で実装済み (C++ 105.3ms) だが、出荷経路には未結線 |
| img2img (入力画像 → latent) | iGPU で VAE encode (probe5 79ms) | 未実装 (`POST /v1/images/edits` は 501) |
| GPU 生成中の他 HW 並列化 | 全 HW | 実寸スタブで検証段階 (`test_multi_frame_pipeline`) |

### デバイス選定根拠 (probe 実測)

| モデル | CPU | iGPU | NPU | 採用 | 理由 |
|---|---|---|---|---|---|
| CLIP-L text encoder | 20ms | 14ms | **7.85ms** | NPU | 純粋 GEMM チェーン |
| WD14 SwinV2 tagger | **101ms** | 104ms | 268ms | CPU | Window Attention (gather/scatter) が NPU に不向き |
| VAE decode (スタブ) | 126ms | 995ms | - | RTX5080 | iGPU は 8 倍遅い |
| VAE encode (img2img) | 117ms | **79ms** | - | iGPU | encode 方向は iGPU が有利 |
| マッティング ISNet-anime | 204.20ms | **99.96ms** | 142.96ms | iGPU | 純 conv で NPU にも載るが iGPU が最速 (M-5) |
| LLM (Phi-3 mini / Qwen2-1.5B INT4) | 64-71 tok/s (Qwen2) | - | ❌ コンパイル失敗 | CPU (probe のみ) | 動的形状・KV-cache は NPU の設計外 |

---

## 計測ベースライン

### probe (Python / diffusers・OpenVINO・調査フェーズ)

| 指標 | 値 | probe |
|---|---|---|
| CPU→VRAM (10MB / 100MB) | 0.76ms / 3.46ms (30.3 GB/s) | probe2 |
| NPU 推論 (512dim MLP 静的形状) / 出力→GPU | 0.88ms / 0.031ms (3.4%) | probe2 |
| system RAM → RTX5080 (latent 256KB / image 12MB) | 0.030ms / 0.254ms | probe4 |
| NPU→iGPU ゼロコピー差分 (231KB) | 0.158ms (誤差) | probe4 |
| iGPU VAE decode スタブ | 995ms (CPU 126ms) | probe4 |
| iGPU VAE encode (1024→128) | **79ms** (CPU 117ms) | probe5 |
| Qwen2-1.5B INT4 CPU (参考値・本実装は非依存) | 64-71 tok/s | probe7 |
| WD14 SwinV2 (448×448) | CPU 101ms / iGPU 104ms / NPU 268ms | probe8 |
| CLIP-L text encoder (77token) | CPU 20ms / iGPU 14ms / **NPU 7.85ms** | probe9 |
| SDXL 20steps 1024×1024 (diffusers・CFG あり・テキストエンコード込み) | **3.80s** / 5.3 it/s / VRAM ピーク 10.49GB (`torch.cuda.max_memory_allocated`) | probe10 |

### 自作経路の e2e (RTX5080・1024×1024・20step)

| 構成 | 値 | 条件・出典 |
|---|---|---|
| 出荷 `--fast` (attention 高速版 + CFG batch=2 + epilogue 融合)・g=7.5 | **10.65〜10.96 s/枚** (2 走行 × 2 枚) | 2026-09-04 研究機・`src/tests/prof_arena_e2e.cu` (計測専用ハーネス)・HEAD `e17374c`。UNet + VAE decode の区間で、テキストエンコードは含まない (golden 埋め込み入力)。生ログ `docs/logs/g10k-baseline/e2e_run1.log` / `e2e_run2.log` |
| 既定 (非 `--fast`)・g=7.5 | 14.1s (14066.9ms) | 2026-09-16 研究機・`test_diffusion_batch2` (DB2_BENCH 20step)・`DOLLAMA_CONV_BATCH=0` (= 現在の既定と同じ per-n)・HEAD `c0f6d8f` (E-0 より前)。**`DOLLAMA_PROFILE=1` で計装 ON の参考値**。同走行の `--fast` は 10799.8ms。生ログ `docs/logs/g10k-e2e/t7_p2_convbatch0.log` |
| VRAM ピーク (`--fast`) | 13,397MB | 上記 e2e。`cudaMemGetInfo` の used (デバイス全体) のピークで、CUDA コンテキストと VRAM アリーナの常駐分を含む。probe10 の 10.49GB とは計り方が異なる |
| 旧値 (参考): 2-6 最適化のクローズ時 | ~~11.30s~~ | 非 CFG (g=1.0)・B=1・`generate()` 経路・2026-06-22。**出荷経路の値ではなく、現在は陳腐化** |

### 自作実装の単体計測 (`test_*`)

| 指標 | 値 | test |
|---|---|---|
| CLIP-L (NPU, `ClipEncoder`) | 7.82ms (中央値 / N=100) | test_clip |
| WD14 (CPU, `Wd14Tagger`) | 105.3ms (中央値 / N=100) | test_wd14 |
| Phase 1 縦通し (stub→CLIP→queue→WD14) | 9.13 frames/s | test_pipeline |
| マルチフレーム並列 (実寸スタブ) | 0.258 fps = GPU 上限の 98% (GPU バウンド) | test_multi_frame_pipeline |
| 自作 VAE decode (golden 比) | SSIM 0.999988 | test_vae_decode |
| 自作 SDXL UNet (golden 比・default 経路) | noise_pred SSIM 0.999996 | test_unet / test_unet_fast |
| EulerDiscreteScheduler | sigmas max_err 4.77e-6 | test_scheduler |
| HTTP 往復 (スタブ生成・256×192) | 2.11ms | test_http |

### 自作 CUDA カーネル (Phase 2 初版の値)

以下は **Phase 2 で書いた初版カーネル**の計測値。現在の既定経路は、大きな GEMM が cuBLAS、
Conv2d が im2col + GEMM、Attention が flash + wmma に置き換わっており、この表はその置き換え前の値である。

| カーネル | 値 |
|---|---|
| FP16 GEMM (shared-mem tiling, FP32 蓄積) | 1024³ **4730 GFLOPS** / SDXL Linear 4208 GFLOPS |
| SiLU / GeLU | FFN 544 GB/s |
| GroupNorm (1 group = 1 block) | 73 GB/s → G-4k S1a の multi-block 版 (`--fast` の epilogue 経路のみ) で **444.6 GB/s** |
| Conv2d (direct, FP32 蓄積) | UNet C320 64² 3×3 **1807 GFLOPS** / VAE C128 512² 3×3 1667 GFLOPS |
| Attention (FP32 softmax) | self **1631 GFLOPS** / cross 1748 GFLOPS |

---

## 出力画像サイズ

**現状 1024×1024 固定。** `--width` / `--height` や HTTP の `size` は受け付けるが、1024 以外は生成器が
`runtime_error` で拒否する (`src/server/backend_image_generator.hpp` ほか。黙って差し替えない方針)。
拡散ループが latent `[1,4,128,128]` 固定で結線されているため。

参考 (将来対応): SDXL 一般の学習解像度は 1024×1024 前後のアスペクト違い (832×1216 / 1216×832 など)。

---

## 自作タグ生成 LM (Phase 4)

汎用 LLM (probe で試した Qwen2-1.5B) の代わりに、**自然文 → danbooru タグ**に特化した小型 LM を自作する。
核は `src/models/bitnet.hpp` (decoder-only LLaMA 系 d512/L8 = **32.98M**)。語彙はタグ単位完全一致トークナイザ `src/io/tokenizer.hpp` (vocab.json 駆動)。

| | Qwen2-1.5B (probe 参考) | 自作タグ生成 LM (本線) |
|---|---|---|
| パラメータ | 1.5B | 32.98M |
| サイズ | 873MB (INT4) | ~66MB (FP16) |
| デバイス | CPU (64-71 tok/s) | GPU 主 / CPU 可 / NPU 不可 (自己回帰のため) |
| タスク | 汎用 | user text → danbooru タグ特化 |

- **訓練は PyTorch** (`scripts/train_bitnet.py`)、**推論は C++** (`src/infer/bitnet.hpp` CPU / `bitnet_gpu.cu` GPU / `bitnet_int8.hpp` INT8)。
  PyTorch golden と logits corr 1.0・greedy デコード一致。GPU forward は CPU 版比 seq8 46.8x / seq63 87.5x、INT8 は重み 74.84% 減で corr 0.9999。
- **正典 = `[B-merge-at-A]`** (2026-07-03): 入力多様化 B (2,000 件) と同一性条件付け A (a12k identity) を 33M に一括で焼いたモデル。
  昇格ゲート 4 指標を全通過 (単一 seed 20260620): identity retention 0.9807 / diverse_a set-F1 0.3332 / diverse_b set-F1 0.3804 / in-dist 0.4552。
  (A 単体の retention は 0.975。)
- **不採用に終わったもの**: 蒸留 4 路線 (Qwen2 / soft-label KL / TIPO-200M 等・過学習抑制のみで F1 非寄与)、容量増 33M→80M (陰性クローズ)。
  DanTagGen / TIPO は品質基準として参照する。
- **未結線**: LM 段はまだ txt2img パイプラインに繋がっていない。
- **ternary (b1.58) は目的ではなく圧縮の研究軸**: まず dense で品質を出し、ternary GEMM は後段の実験 (`src/kernels/ternary_gemm.cu` は未作成)。

### アニメ品質スコアラ (Model B) と品質フィードバック (F)

- **ScorerNet** (自作 11.18M 純 conv・NPU 8.32ms) を C++ に結線済み。採点は現状ログのみ。
- **Q-2**: quality 判定を CLIP 埋め込み上の別枝 (QualityMLP) に分離し、ScorerNet は解剖 (anatomy) 用途に限定。
  これで reward が信号ゲートを通過 (std 0.1038 / best−worst 0.3575、閾値 >0.1 / >0.3)。QualityMLP は Python 側の reward で使い、C++ には未結線。
- **F-0a** (信号ゲート): 初版 reward は信号が弱かった (std 0.0377 / best−worst 0.2031)。
- **F-0b** (RAFT-SFT: best-of-8 → SFT): **効果を検出できず不採用**。射程は限定的で、現 reward 設計 (解剖 8 軸のうち Limbs 以外の 7 軸がほぼ動かない)・単一 seed・検出力不足の条件下での不検出であり、手法一般の無効性は示していない。
- 次のレバー: reward 設計 / 解剖スコアの死軸の分解能診断 / 日本語条件付けの改修 / SDXL seed 制御 ([`docs/roadmap.md`](docs/roadmap.md) 「タグ生成 LM 学習強化プログラム」節)。

**2D 特化の独自性**: ① キャラ**同一性条件付き**タグ生成 (character-bible を条件入力・DanTagGen に無い) ② アニメ品質スコアラ (NPU・固定形状)。
キャラ設定の構造は [`docs/character-bible-spec.md`](docs/character-bible-spec.md)。

---

## 実装方針

**自作は HW を叩く研究コアに限定し、配管 (HTTP / JSON / Base64) は定番のヘッダオンリーライブラリを使う。**

| 使う | 使わない (推論パス) |
|---|---|
| STL | PyTorch / LibTorch |
| CUDA Runtime API + **cuBLAS** (大きな GEMM の既定経路。`DOLLAMA_GEMM=wmma` で自作 wmma に切替) | diffusers / stable-diffusion.cpp |
| 自作 CUDA カーネル (Conv2d / Attention / GroupNorm / VAE decode 等) | llama.cpp |
| **OpenVINO ランタイム** (NPU / iGPU / CPU の推論グルー: CLIP・WD14・ISNet・ScorerNet) | 重量級 HTTP フレームワーク |
| **cpp-httplib** / **nlohmann/json** (ヘッダオンリー) | 手書き Winsock2 / 手書き JSON パーサ |

Windows では cpp-httplib のために `ws2_32` をリンクするが、dollama のコードが Winsock2 を直接使う箇所は無い。

### ゼロコピー調査結果 (probe1-4)

| ルート | 結果 | 理由 |
|---|---|---|
| CUDA Virtual Memory + Win32 ハンドル → NPU | ❌ | OpenVINO NPU に CUDA ハンドル import API なし |
| D3D12 クロスアダプター (RTX5080 → iGPU → NPU) | ❌ | Intel iGPU が DXGI に非表示 |
| CPU pinned memory | ✅ 採用 | オーバーヘッド 3.4% |

---

## src ツリー

```
src/
├── main.cpp         エントリポイント (引数なし = デバイスチェック / --http / --prompt)
├── pipeline.hpp     Phase 1 マルチスレッド骨格 (CLIP=NPU → queue → WD14=CPU)
├── core/            Tensor・Allocator (CPU / pinned / VRAM)・SPSC キュー・CharacterBible (キャラ台帳)・
│                    CPU アフィニティ / トポロジ検出・マルチフレーム orchestrator
├── kernels/         自作 CUDA カーネル: gemm (cuBLAS 既定 + 自作 wmma / tiling)・conv2d (direct / 1x1 GEMM /
│                    im2col + GEMM)・attention (flash + wmma) / attention_fast (`--fast`)・groupnorm・activation・
│                    layernorm・geglu・timeembed・bias_add・elementwise・vae_decode・device_arena (VRAM bump アリーナ)
├── infer/           推論: SDXL UNet (unet.cu)・拡散ループ (diffusion.cu)・Euler scheduler・SDXL dual text encoder
│                    (clip_tokenizer / clip_encoder2 / text_conditioner)・CLIP-L 単体 (clip.hpp)・WD14・マッティング・
│                    品質スコアラ / QA ゲート・ランタイム LoRA 写像・タグ生成 LM 推論 (bitnet CPU / GPU / INT8)・プロファイラ
├── io/              safetensors ローダー・PNG メタ (tEXt にキャラ設定を焼き込み)・タグ単位トークナイザ
├── models/          タグ生成 LM のモデル定義 (bitnet.hpp / bitnet_config.hpp)
├── server/          HTTP API (api.cpp)・生成器 DI (cli_generate.hpp)・BackendImageGenerator・拡散 backend registry
│                    (sdxl_backend / sd35_backend stub)・diffusion_runner・preset 解決 (preset / preset_json)・
│                    FastConfig・マッティング / 採点の後処理と runner・PipelineGenerator / StubGenerator・PNG / Base64
└── tests/           test_*.cpp / test_*.cu (meson test) と prof_*.cu / prof_*.cpp (計測専用・meson test 未登録のものを含む)
```

`.cu` を含む部分は CUDA 有効ビルドのみ。OpenVINO / CUDA 依存は純 C++ interface (`IDiffusionRunner` / `IMatter` / `IScorer` 等) の向こうに隔離し、
無効ビルドでは stub に差し替わる。

---

## ビルド

### 一括インストーラー (Windows・推奨)

まっさらな Windows 機なら、必要なもの一式 (VS Build Tools / CUDA / OpenVINO / Python 依存) を winget + pip で導入できる:

```powershell
powershell -ExecutionPolicy Bypass -File install_windows.ps1
```

- NVIDIA dGPU 検出時のみ CUDA Toolkit を導入し torch=cu128。無ければスキップして torch=CPU・`-Dwith_cuda=false` を案内する。
- `-DryRun` / `-CheckOnly` / `-SkipCuda` / `-SkipSdk` / `-PythonExe <path>` / `-Force` を指定可能。
- 最後に環境に合わせた推奨 `meson setup` コマンドを出力する。

### 手動

**前提**

- [Meson](https://mesonbuild.com/) + Ninja、C++20 コンパイラ (Windows 11 + MSVC で検証)
- CUDA Toolkit (要件 12.8+ / 検証 13.3)。`nvcc` を `PATH` に通す
- OpenVINO 2024.x+ ランタイム。**テキスト→画像に必須** (無いと golden 埋め込み経路かスタブにフォールバックする)

```bash
# 既定の SDK パスは meson.build にある (CUDA v13.3 / openvino_2024)。違う場所なら -D で指定する
meson setup build \
  -Dgpu_sdk_root="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3" \
  -Dnpu_sdk_root="C:/Program Files (x86)/Intel/openvino_2024"

meson compile -C build
meson test -C build            # 単体テスト + カーネルのゴールデンテスト (重み・golden が無いものは [SKIP])
```

`-Dwith_cuda=false` / `-Dwith_openvino=false` / `-Dwith_http=false` で各機能を無効化できる。テスト規約は [`docs/testing.md`](docs/testing.md)。

---

## 実行

重み・golden の既定パスは**リポジトリルートからの相対パス**なので、ルートで実行する。

### CLI

```powershell
# 1 枚生成して PNG 保存 (既定: preset illustrious-xl・マッティング ON = 透過 PNG)
build\src\dollama.exe --prompt "1girl, solo, smile" --negative "lowres" --out out.png --seed 1234 --fast
```

| フラグ | 意味 |
|---|---|
| (引数なし) | デバイスチェック (OpenVINO / CUDA デバイス列挙 + Phase 1 骨格の縦通し) |
| `--prompt <str>` / `--negative <str>` / `--out <path>` | CLI 生成モード (既定の出力は `out.png`) |
| `--http` / `--port N` | HTTP サーバを起動 (`127.0.0.1`、既定ポート 8080) |
| `--preset <name>` | `models/presets/<name>/` の重み一式を使う (既定 `illustrious-xl`。`base` で素 SDXL) |
| `--no-preset-prefix` | preset の prompt / negative 接頭辞の自動付与を OFF |
| `--fast` | 高速モード (attention 高速版 + CFG batch=2 + epilogue 融合)。既定 OFF |
| `--seed <uint64>` / `--cfg <float>` | seed / guidance scale (未指定なら env `DOLLAMA_SEED` → 時刻、cfg は 7.5) |
| `--steps N` | ステップ数 (既定 20) |
| `--width` / `--height` | 受け付けるが、現状 1024 以外は拒否される |
| `--no-matting` | マッティングを OFF (不透明 PNG) |

各重みのパスは環境変数 (`DOLLAMA_UNET_WEIGHTS` / `DOLLAMA_VAE_WEIGHTS` / `DOLLAMA_ENCODER_L` など) で上書きできる。
**本 txt2img 経路には `DOLLAMA_OV_TOKENIZERS_DLL` (openvino_tokenizers の DLL パス) の指定が必要**で、
未指定・アセット不足のときは golden 埋め込み経路 → スタブの順にフォールバックする (起動ログにどの段で立ったかが出る)。

### HTTP (OpenAI Images 互換)

```bash
curl -s http://127.0.0.1:8080/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"prompt":"1girl, solo","negative_prompt":"lowres","seed":1234,"guidance_scale":7.5}'
```

- `POST /v1/images/generations` (txt2img・base64 PNG) / `GET /health` / `GET /v1/models` / `GET /` (簡易 Web UI)。`POST /v1/images/edits` は 501。
- 任意フィールド: `negative_prompt` / `steps` / `size` (1024x1024 のみ) / `seed` / `guidance_scale` / `preset_prefix` (bool) /
  `loras: [{"name": ..., "strength": ...}]` (名前は `DOLLAMA_LORA_DIR` 既定 `models/loras` 配下に限定)。
- 仕様は [`docs/http-api-spec.md`](docs/http-api-spec.md)。

### UI (Blazor Server・.NET 10)

```powershell
.\run-ui.ps1          # UI: http://localhost:5074 → C++ サーバー http://127.0.0.1:8080 を叩く
```

C++ サーバー (`dollama --http`) は別プロセスで起動しておく。複数サーバーの切替 (2-6f)・LoRA 選択などの機能と設定は [`ui/README.md`](ui/README.md)。

---

## モデル

モデル重みは本リポジトリに**含まれない** (`models/` は Git 管理外)。各自で取得・変換する。

### 配置

| パス | 中身 | 作り方 |
|---|---|---|
| `models/sdxl-text-encoder-l/` / `sdxl-text-encoder-g/` | SDXL の CLIP-L / CLIP-G (OV IR) | `scripts/dollma_convert_sdxl_text_encoders.py` |
| `models/sdxl-tokenizer-l/` / `sdxl-tokenizer-g/` | 対応するトークナイザ (openvino-tokenizers) | 同上 |
| `models/presets/<name>/` | preset 一式: `unet_weights.safetensors` / `vae_weights.safetensors` / `text-encoder-l/` / `text-encoder-g/` (+ 任意の `preset.json`) | `scripts/dollma_export_sdxl_preset.py --preset <name>` |
| `src/tests/data/unet_weights.safetensors` / `vae_weights.safetensors` | 素 SDXL の UNet / VAE (`--preset base` 時・preset 未配置時のフォールバック) | `scripts/dollma_dump_unet_golden.py` / `dollma_dump_vae_golden.py` |
| `src/tests/data/unet_io.safetensors` | golden 入出力 (本 txt2img 経路の起動条件にも含まれる) | `scripts/dollma_dump_unet_golden.py` |
| `models/isnet-anime/` | マッティング (OV IR) | `scripts/dollma_convert_matting.py` |
| `models/scorer-net/` | ScorerNet (OV IR) | `scripts/dollma_convert_scorer.py` |
| `models/clip-l-text-encoder/` / `wd14-swinv2-tagger-v3/` | Phase 1 骨格 (デバイスチェック / test_pipeline) 用 | probe 時代の変換 (`scripts/archives/`) |
| `models/loras/` | ランタイム LoRA | 既存の `.safetensors` を置く |

`scripts/archives/` の probe スクリプトは記録のために残しており、再実行は想定していない。
タグ生成 LM の正典重みは `data/bitnet/` 配下 (Git 管理外・`scripts/train_bitnet.py` で再生成)。

### ライセンス

本リポジトリの**コードは Apache-2.0** だが、各モデルの**重みは配布元それぞれのライセンスに従う**。利用者がその条件の遵守に責任を負う。
帰属とライセンスの正本は [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) (リリース前の最終照合が必要な「要照合」項目を含む)。主なもの:

| モデル | 役割 | ライセンス |
|---|---|---|
| Stable Diffusion XL base 1.0 | 拡散 (UNet + VAE)・dual text encoder (`--preset base`) | CreativeML Open RAIL++-M |
| **Illustrious XL v0.1** (early-release-v0) | **既定 preset** `illustrious-xl` | Fair AI Public License 1.0-SD (コピーレフト条項あり。重みを再配布しない運用) |
| NoobAI-XL 1.1 | preset `noobai-xl` | Fair AI Public License 1.0-SD |
| Animagine XL 4.0 | preset `animagine-xl-4` | CreativeML Open RAIL++-M |
| CLIP ViT-L/14 | text encoder / 美的スコアラの image embed | MIT |
| WD14 SwinV2 Tagger | タグ抽出 (CPU)・解剖ラベル教師 | Apache-2.0 |
| ISNet-anime (anime-segmentation) | マッティング (iGPU) | Apache-2.0 |
| waifu-scorer-v4-beta | 品質スコアラの教師 (ラベル生成のみ) | Apache-2.0 |
| deepghs/anime_aesthetic | 品質スコアラの教師候補 (使用制限条項あり) | OpenRAIL |
| TIPO-200M / Qwen2-1.5B / DanTagGen | 蒸留教師・probe・品質基準のみ (配布物に含まない) | THIRD_PARTY_NOTICES.md 参照 |

CLIP-G (bigG) は SDXL base 1.0 (および各 preset checkpoint) に同梱の `text_encoder_2` から変換している。
教師モデルは soft ラベル生成にのみ使い (既定は waifu-scorer-v4-beta。anime_aesthetic は使用制限条項のある教師候補)、その出力を自作 ScorerNet へ蒸留する。教師の重みは再配布しない。

---

## セットアップ (Python・訓練 / 変換 / probe)

Windows なら `install_windows.ps1` が `requirements.txt` 基準で pip 依存も導入する (torch は GPU 有無で index を自動分岐)。手動の場合:

```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu128   # NVIDIA GPU がある場合 (無ければ pip install torch)
```

---

## ファイル構成

```
dollama/
  src/                 C++ 本実装 (上記「src ツリー」)
  ui/  ui.Tests/       Blazor Server UI (.NET 10) とそのテスト
  scripts/             変換・golden 生成・訓練・評価スクリプト (dollma_*.py / train_*.py)
    archives/          完了済みの probe (`dollma_probe*.py`) とクローズ済み実験のスクリプト (再実行は非想定)
  subprojects/         Meson wrap (cpp-httplib / nlohmann_json)
  models/              変換済みモデル (Git 管理外)
  data/                訓練データ・評価データ (重みは Git 管理外)
  docs/
    roadmap.md             実装ロードマップ (今どこで次に何をやるか)
    roadmap-decisions.md   完了アイテムの経緯・決裁記録
    measurements-log.md    計測の完全版 (条件・陳腐化注記つき)
    training-spec.md       タグ生成 LM の訓練仕様と結果
    dataset-spec.md        データセット仕様
    fast-mode-plan.md      高速化 (G-2k〜G-8k・`--fast` 系と G-8k VRAM アリーナ) の計画と記録
    g10k-plan.md / g14k-plan.md   conv 真 batch2 (陰性クローズ) / attention 書き直し (計画中)
    image-quality-plan.md  画像品質の改善計画
    character-bible-spec.md  キャラ設定データ構造 (同一性 / シーン / 出力層)
    http-api-spec.md       HTTP API 仕様
    pipeline-spec.md / tensor-spec.md / cpu-topology.md / hw-environment-spec.md
    testing.md             テスト規約
    logs/                  計測の一次証拠として退避した生ログ
    archives/investigation-log.md   probe1-10 の詳細調査ログ
  THIRD_PARTY_NOTICES.md   第三者モデル / ライブラリのライセンス表示
  SECURITY.md / SECURITY_jp.md
  install_windows.ps1 / run-ui.ps1 / publish-ui.ps1
  CLAUDE.md                Claude Code 向けコンテキスト
```

---

## ライセンス

[Apache License 2.0](LICENSE) の下で公開。詳細は [`LICENSE`](LICENSE) を参照。第三者モデル・ライブラリは [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。

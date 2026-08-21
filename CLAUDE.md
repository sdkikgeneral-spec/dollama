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
| CPU | WD14 タグ抽出 105.3ms / **LM 段は未結線** (Qwen2 は Python probe 専用) | ⏳ 自作 LM の結線が残 |
| NPU | **CLIP text encoder 7.85ms** / WD14 タグ抽出 268ms | ✅ CLIP が CPU の 2.5倍速 |
| iGPU (Intel Xe) | VAE encode (img2img、79ms) + マッティング ISNet (99.96ms) | ✅ どちらも CPU より速い (matting_device=iGPU 確定 M-5) |
| **RTX5080** | SDXL UNet + VAE decode | ✅ **3.80s/image** (1024×1024, 20steps, 5.3 it/s) |

**パイプライン (確定構成)**

txt2img:
```
GPU/CPU: 自作タグ生成 LM (bitnet.hpp 33M, GPU 主・CUDA カーネル流用 / CPU 可・NPU 不可) — ★未結線
  自然文 → danbooru タグ列 (GPU seq8 2.43ms 実測) / 現状はこの段を通さず prompt 直入力
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
  └─→ CPU:  LM 段 (未結線)      ──────────┤ (並列)
                                          ▼
                               NPU: CLIP (7.85ms)
                                          ▼
                               RTX5080: SDXL UNet + VAE decode (3.80s)
```
iGPU の VAE encode は CPU 側の LM 段と並列に走る設計のため待ち時間ゼロ (LM 結線後)。

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

- Qwen2-1.5B は **Python probe 専用** (`scripts/archives/`・CPU 64-71 tok/s) — **C++ 実装には非搭載** (`docs/pipeline-spec.md` ‘LibTorch 不使用方針’)。現状の txt2img は `req.prompt` を素で dual encoder に渡す (`src/server/txt2img_generator.hpp`) = **LM 段は未結線**
- 自作モデル: **小型タグ生成 LM** (`src/models/bitnet.hpp`, decoder-only LLaMA 系 33M, 実装済み・test 緑)
  - 30-100M params、user text → danbooru タグ生成特化
  - **置き場は GPU が第一** (速度 + **既存の自作 CUDA カーネル gemm/attention/layernorm/geglu をそのまま流用**)。LM は拡散の上流で逐次実行 → GPU 競合ほぼ無し・33M は VRAM も誤差。**CPU は代替** (複数フレームのパイプライン並列時に拡散の裏で先行生成する用)。**NPU は自己回帰不可で除外** (probe6 で Phi-3 がオンチップメモリ超過・investigation-log。NPU 化は非自己回帰版が要る・未確定)
  - **ternary (b1.58) は目的ではなく圧縮の研究軸**: まず FP16/INT8 dense で品質を出し、ternary は乗算削減の実験として後段で被せる (`ternary_gemm.cu`)。33M 規模では ternary の旨味は限定的
  - **先行実装あり**: DanTagGen (400M LLaMA) / TIPO が「自然文+タグ → danbooru タグ」を既に実現 → 蒸留教師・品質基準として活用 (Qwen2 蒸留 D2 と同じ役割・ただし D2 自体は不採用)
  - **新規性の方向 (2D 特化・独自)**: ① キャラ**同一性条件付き**タグ生成 (character-bible を条件入力・DanTagGen に無い) ② アニメ**品質スコアラ** (NPU・固定形状, §11)。詳細・経緯は roadmap Phase 4

## 計測ベースライン

> 各行の詳細経緯・条件・seed sweep・採否理由は **`docs/measurements-log.md`** に完全版を退避。
> ここは芯となる数値のみ。深掘りは `docs/roadmap.md` / `training-spec.md` / `dataset-spec.md`。

**HW 疎通・デバイス選定 (probe2–10)**

| 指標 | 値 | probe |
|---|---|---|
| CPU→VRAM (10MB / 100MB) | 0.76ms / 3.46ms (30.3 GB/s) | probe2 |
| NPU推論 (512dim MLP 静的形状) / 出力→GPU | 0.88ms / 0.031ms (3.4%) | probe2 |
| iGPU VAE decode stub | 995ms (CPU 126ms → RTX5080 採用) | probe4 |
| NPU→iGPU ゼロコピー差分 (231KB) | 0.158ms (誤差) | probe4 |
| iGPU VAE encode (img2img) | **79ms** (CPU 117ms → iGPU 採用) | probe5 |
| Qwen2-1.5B INT4 CPU (probe 参考値・**本実装非依存**) | 64-71 tok/s / ロード 1.1s | probe7 |
| WD14 SwinV2: CPU/iGPU/NPU | 101/104/268ms → **CPU 採用** | probe8 |
| CLIP-L: CPU/iGPU/NPU | 20/14/**7.85**ms → **NPU 採用** | probe9 |
| SDXL 20step 1024² | **3.80s** / 5.3 it/s / VRAM 10.49GB | probe10 |

**C++ 実装の確定値 (test_*)**

| 指標 | 値 | test |
|---|---|---|
| CLIP-L NPU ClipEncoder | **7.82ms** (中央値/N=100) | test_clip |
| WD14 CPU Wd14Tagger | **105.3ms** (中央値/N=100) | test_wd14 |
| Pipeline 縦通し (stub→CLIP→queue→WD14) | **9.13 frames/s** (WD14 律速) | test_pipeline |
| マルチフレーム並列 (実寸スタブ) | **0.258 fps** = GPU 上限 98% = GPU バウンド。look-ahead 2 で飽和・Tier2 独立 forward は単一 GPU では非発動 | test_multi_frame_pipeline |
| CharacterBible compose_prompt / find | 242 ns/op / 10.5 ns/op | test_character |
| 自作 FP16 GEMM 1024³ / SDXL Linear | 0.45ms 4730 GFLOPS / 3.19ms 4208 GFLOPS | test_gemm |
| 自作 SiLU/GeLU FFN / GroupNorm / Conv2d direct | 544 GB/s / 73 GB/s / 1807 GFLOPS | test_activation/groupnorm/conv2d |
| 自作 Attention (self/cross) | 1.65ms 1631 GFLOPS / 0.116ms 1748 GFLOPS | test_attention |
| safetensors ローダー | 19.0 µs/op | test_safetensors |
| 自作 VAE decoder 全段 | SSIM **0.999992** / decode 7.96s | test_vae_decode |
| 自作 SDXL UNet 全段 | noise_pred SSIM **0.999998** / 1step ~9.2s / 2.57B params | test_unet |
| UNet バッチ (G-2k S2) | `launch_unet_batched(B=2)` per-sample パリティ **実走緑** (SAC OFF): sample0/1 とも MAE**6.4e-05**/bad0・wmma M=154 タイル境界も異常なし。ビット一致には非到達=FP16 tol 内 (linear の M=B*tokens 単発 GEMM が cuBLAS タイル選択を変え蓄積順が変わるため・1 ULP)。conv2d S1 は per-n ループで **BIT-EXACT** | test_unet/conv2d |
| CFG batch2 パリティ+速度 (G-2k S3c **完了**) | **parity gate g=1.0 SSIM 0.9994≥0.999 PASS** (SAC OFF 実走): ノイズ床 (off@g=1.0 x2) 完全 bit-exact → 発散源 100% batch2 tiling 差。guidance sweep g=1/3/7.5 = 0.9995/0.9988/0.9966 (FP16 tiling が CFG で単調増幅・平均差≤0.3/255)。**正典 CFG e2e (20step/warm/VAE 込)**: default **20.93s** / batch2 単独 **17.59s(1.19x)** / **--fast(attn+batch2) 15.88s(1.32x)**。batch2 が ~2× 未達なのは conv2d per-n 直列が batch されないため | test_diffusion_batch2 |
| UNet fast warm (G-3kf 完了) | warm ハンドル (unet_weights_create) で cold 4.9GB 再転送を撲滅・default vs golden SSIM 0.999996/bad0・fast vs default bit-exact・warm 1step default 526ms/fast 437ms = **1.20x** (cold 希釈を是正) | test_unet_fast |
| GroupNorm multi-block (G-4k S1a 完了) | `launch_group_norm_mb` 2段決定的集約 (atomic 禁止): 帯域 unet_320_128 **444.6 GB/s** (1-block 105 → **4.2x**)・parity MAE~3e-7・bitexact 3runs 一致。epilogue 配線 (resnet norm1/2+conv_norm_out): epilogue vs default **SSIM 0.999999**/bad0・default 無改変 (fast vs default bit-exact 維持)・1step 差はノイズ内 (resnet ゲート合否は S1b/S2 後の再 profile) | test_groupnorm/test_unet_fast |
| GroupNorm+SiLU 融合 (G-4k S1b 完了) | `launch_group_norm_silu` = mb normalize 末尾に SiLU を融合 (partial/finalize は共用)・resnet epilogue の norm1/2 で mb+silu 2 パス→1 パス。GN 結果を half 丸め→同式 SiLU で **5 形状すべて BIT-EXACT vs mb+silu** (memcmp)・融合 bitexact 3runs 一致。UNet: epilogue vs default **SSIM 0.999999**/MAE6.6e-5/bad0・default 無改変維持 (fast vs default bit-exact)・1step 差ノイズ内 (conv_norm_out は golden 捕捉保護で据え置き) | test_groupnorm/test_unet_fast |
| conv2d 後段融合 (G-4k S2 完了) | resnet epilogue の conv 後段を融合 2 本に: `launch_conv_bias_biasch` (P1=conv1.bias broadcast + per-(n,c) time-bias・per-b ループ B 発+bias パスを 1 発) / `launch_conv_bias_residual` (P2=conv2.bias broadcast + residual add・bias パス+launch_add を 1 発)。**急所=`launch_conv2d` は形状で丸め列が違う** (GEMM 経路=2 段丸め / direct 経路=単一丸め) → conv を bias 抜きで呼び中間を register で half 丸めし bias 後付け再現、GEMM 経路のみ bit 一致。`conv2d_uses_gemm_bias_path` ガード (conv2d.cu 純追加・本体不変) で非保証 shape は従来 2 パスへフォールバック。P1/P2 **全 6 形状 (B1/2×320_128/640_64/1280_32) BIT-EXACT vs GEMM 2 段丸め参照** (memcmp)・in-place bit-exact・3runs 一致。UNet: epilogue vs default **SSIM 0.999999**/MAE6.6e-5/bad0・default 無改変維持 (fast vs default bit-exact)・1step warm default 469.5/fast 402.3/fast+epi 400.7ms。**resnet ≤0.95s 再 profile 完了 (2026-07-14 研究機実走)=不合格**: resnet ×20 バケット default 1.226s→fast+epi **1.212s (−1.1%=ノイズ床内)** で ≤0.95s 未達。バケットは conv2d 質量・GN 4.2x は非寄与 → ゲートは G-4k スコープ外と確定・合否は **G-10k(conv 真batch2)/G-8k(im2col malloc撲滅)** 後へ再割当 (`prof_unet_fast_warm.exe` DOLLAMA_PROFILE=1 [RESNET-BUCKET] で恒久化) | test_bias_add/test_unet_fast/prof_unet_fast_warm |
| **epilogue 出荷結線 (G-4k S3 完了 = G-4k クローズ)** | `DOLLAMA_EPILOGUE` env を `FastConfig` に追加 (`--fast` が含意) + `DiffusionPipeline` へ貫通 → **出荷 `--fast` = attn_fast+batch2+epilogue**。**ゲート再設計 (急所)**: 初版は fast+epi vs default を **g=7.5** で SSIM 判定 = ①batch2 単独ですら g=7.5 で 0.9966 ゆえ epilogue 正常でも落ちる ②測っている変数が epilogue でない ③epilogue 自身も GN mb の蓄積順差で bit 一致でない (MAE6.6e-5 = batch2 と同オーダー) → **ハードゲートは g=1.0 の parity 節 (steps=4・meson test 自動枠) に 4 本**へ移設・g=3/7.5 は characterization (合否なし)・DB2_BENCH は速度計測に純化 (`DB2_BENCH_G` 追加)・旧ゲート②(default 二連走) は parity `:300` と重複で削除 (~60-70s 節約)。**教訓: CFG 増幅下 (g>1) で FP16 微差をゲートしない。被験変数は g=1.0 で分離する**。**実走 (SAC OFF・全 PASS)**: GATE1 off@g=1.0 二連走 **bit-exact** / GATE2 batch2 vs off SSIM **0.999474** / GATE3 epilogue determinism **bit-exact** / GATE4 **(fast+epi) vs fast @g=1.0 SSIM 0.999477** (≥0.999)。characterization: (fast+epi) vs fast は g=3 0.998765 / **g=7.5 0.996533** (=初版ゲートなら FAIL の実証)。**e2e DB2_BENCH 20step warm**: default 24662.8ms / attn+batch2 (composite・**not CLI-reachable**) 18557.8ms **x1.32898** / **fast+epilogue (shipping --fast) 18482.4ms x1.33439** = epilogue 上乗せ +0.4% ノイズ床内。**注: 本走行は機体クロック/熱で全体が ~18% 遅くドリフト** (同一走行内で default resnet バケットが 1.28564→1.41299s = +9.9%) → **相対倍率 x1.33 を主指標とし絶対秒は条件付き** (G-2k の x1.32 と整合)。batch2 単独は harness に config 無く未測定。test_unet_fast: default vs golden SSIM 0.999996 / **fast vs default bit-exact (default 無改変維持)** / epilogue vs default SSIM 0.999999・1step warm 485.7/476.3/475.9ms (ドリフトで fast の効きが x1.20→x1.02 に縮んで見える) | test_diffusion_batch2/test_unet_fast/prof_unet_fast_warm |
| 自作 EulerDiscreteScheduler | sigmas max_err 4.77e-6 / golden 一致 | test_scheduler |
| 自作 LayerNorm/GEGLU/time embed | 0.043ms / 0.055ms / 0.011ms | test_layernorm 等 |
| HTTP サーバー (cpp-httplib+nlohmann) | 生成+往復 2.11ms (OpenAI Images 互換) | test_http |
| PNG メタ往復 (tEXt bible) | embed 1961 / read 4376 ns/op | test_png_meta |
| ランタイム LoRA (L-2 完了) | kohya→diffusers 写像 (`load_lora_modules`) + 常駐重み apply-time マージ (`unet_apply_loras`: `launch_gemm_fp16` で delta=scale*(B@A) → `launch_add` in-place) / `unet_clear_loras` bit-exact 復元。**全ゲート PASS** (SAC OFF 実走): [1]host写像 modules/te_skip/incomplete/scale/throw OK・[2]parity max_abs**4.9e-4**/bad0・[3]revert 4/4 memcmp bit-exact・[4]stack max_abs**9.6e-4**/bad0+revert bit-exact。数値正典=L-1 offline merge (dollma_merge_lora.py) `W+strength*(alpha/rank)*(B@A)`。HTTP `loras:[{name,strength}]` 結線 (name は allowlist 検証で path traversal 封止)。**⚠ 後日判明した ODR 違反 (G-4k S3 期の C0 で修正)**: `src/infer/unet.cu` と `src/kernels/vae_decode.cu` が**同名・異レイアウトの `class DeviceWeights` を外部リンケージで**定義しており (元から ODR 違反)、L-2 で追加した `patched_` メンバが両者のレイアウトを決定的にずらして**顕在化** → VAE+UNet 両ハンドル同居時の `~DiffusionPipeline` で **0xC0000005**。C0 で両 TU の `DeviceWeights` を**匿名 namespace 化**して解消 (`test_diffusion_batch2` 90.80s 完走・exit 0・破棄後 VRAM free 14913MB)。**教訓: 単体ハンドルの test は、複数ハンドル同居時の破棄経路を検査しない** — `test_lora_runtime` は UNet ハンドルしか持たないため L-2 の地雷を素通しし、VAE+UNet 同居の `test_diffusion_batch2` で初めて露見した。同居破棄を通す test が既に回帰ゲートとして機能しているため C0 に専用 test は新設していない | test_lora_runtime (+回帰ゲート: test_diffusion_batch2) |

**Phase 4 自作 LM (BitNet, 詳細は measurements-log.md §Phase4)**

| 指標 | 値 | test |
|---|---|---|
| BitNet b1.58 モデル定義 | 32.98M params / 参照 forward | test_bitnet |
| 自作トークナイザー (vocab.json 駆動) | encode 365 / decode 168 ns/op / UNK 0 | test_tokenizer |
| dense LM 訓練 (#1, hard CE 6ep) | val_loss 2.41 / top10 recall 0.777 = **本線** | (train_bitnet.py) |
| dense LM 推論 CPU / GPU / AVX2 / INT8 | corr 1.0 / GPU 46.8–87.5x / AVX2 ~5x / INT8 74.84%減 corr 0.9999 | test_bitnet_infer/_gpu/_int8 |
| CPU トポロジ検出 + Tier2 ベンチ | 物理 5–7 コアで ~5x 飽和・(A) 独立 forward 方式に確定 | test_cpu_topology |
| 同一性条件付き (4-A) | retention 0.947 → 実ペア増で **0.975 (頑健)** / diverse-F1 は seed ノイズ | test_bitnet_infer |
| 評価作り直し (4-C) | recall@10 退役 → **diverse 生成 set-F1 を主指標**化 | test_dollma_eval_diverse |
| 入力多様化 (4-B 500/2k/10k) | diverse-F1 +0.06→+0.15 で **~2,000 件飽和**・本線昇格決裁済 (`[B-merge-at-A]`) | (train_bitnet.py) |
| 蒸留 D5(KL)/D6(TIPO 外部教師) | 両不採用 (過学習抑制のみ・recall 非寄与) | (train_bitnet.py) |
| 容量増 33M→80M (4-D) | **陰性クローズ・80M 不採用/勝者=33M** (F1 seed ノイズ・retention 3/4 床割れ・in-dist 微退行) | (dollma_d_seedsweep) |
| 正典化まとめ焼き (`[B-merge-at-A]`) | 勝者33Mで B(b2000)∧A(a12k identity) を merged 1本に正典化✅・ゲート4指標全通過 (retention 0.9807/diverse_a F1 0.3332/diverse_b F1 0.3804/in-dist 0.4552・各単体参照を全軸超)・正典差し替え+golden 再生成・corr 1.0/meson 25/25 | test_bitnet_infer |

**Phase 4 Model B 品質スコアラ + マッティング (詳細は measurements-log.md)**

| 指標 | 値 | test |
|---|---|---|
| マッティング ISNet-anime (HW 比較 M-5) | iGPU(Xe) **99.96ms** < NPU < CPU → **matting_device=iGPU** | (probe M-5) |
| マッティング 生成器結線 (M-6) | `IMatter`/make_matter で透過 PNG 結線・研究機 e2e 実走済 | test_matting |
| 品質スコアラ NPU probe (B) | 純 conv 448² **NPU 4.62ms** (NPU/CPU 0.55x) | (probe) |
| ScorerNet 訓練 (B-3b) / OV 変換 (B-3c) | 11.18M / NPU 8.32ms / PyTorch↔OV err 1.34e-5 | test_dollma_train_scorer |
| 品質スコアラ C++ 結線 (B-3d→B-5) | `IScorer`/scorer_runner/scoring_postprocess で生成器結線完了 (採点はログのみ・消費者 F 未着手) | test_scoring_postprocess |
| **品質 FB ループ F-0a 信号ゲート (実走 80/80)** | reward min-0.203/max-0.0/mean-0.0253/**std0.0377**/best−worst**0.2031** → **信号弱 (補強してから)**。argmax **Limbs 77/80**・他7軸ほぼ死 (max<0.012)。clean vs clutter prompt で **|r| 4倍分離** (0.007 vs 0.0285)。**判定: F-0b 保留・quality head 有効化 (Q-2) を先に** | (dollma_collect_rollouts) |

**統合パイプライン**

| 指標 | 値 | test |
|---|---|---|
| 自作フル拡散 (2-6a) | 20step 1024² 84.07s → **11.30s** (2-6 最適化後・律速 UNet attn 4.60s) | test_diffusion |
| 本番 txt2img (2-6b) | prompt→画像 dual encoder+CFG / var=2300 / NaN・Inf なし | test_txt2img |
| 拡散 backend プラグイン枠 (2-6c) | `IDiffusionBackend` registry (`make_backend` "sdxl"/"sd35") + `BackendImageGenerator` 共通後処理 / 純 cpp・全 46 test 緑 | test_diffusion_backend |

## 次のタスク

> 経緯・採否・詳細は `docs/roadmap.md`。完了済みの計測は上表＋ `docs/measurements-log.md`。

**Phase 1 (パイプライン骨格) ✅ 全完了**

| # | 実装物 | ファイル |
|---|---|---|
| 1 | Meson + src/ 構造 | `meson.build`, `src/` |
| 2-4 | Tensor / Allocator / SPSC キュー | `src/core/{tensor,allocator,queue}.hpp` |
| 5 | CLIP NPU 推論 (7.82ms) | `src/infer/clip.hpp` |
| 5.5 | キャラ台帳 | `src/core/character.hpp` |
| 6 | WD14 CPU 推論 (105.3ms) | `src/infer/wd14.hpp` |
| 7 | スレッド骨格 + アフィニティ (9.13 fps) | `src/main.cpp`, `src/core/affinity.hpp`, `src/pipeline.hpp` |

**Phase 2-3 ✅ 完了** (詳細 `docs/roadmap.md`)
- safetensors ローダー / VAE decode / SDXL UNet + Euler scheduler — 全 golden 突合済
- cpp-httplib OpenAI 互換 HTTP サーバー (生成は `IImageGenerator` 越し)
- **2-6a** フル C++ 拡散統合 → **2-6 最適化** 84.07s→11.30s で一旦クローズ (律速 UNet attn 4.60s・以降はライブラリ余地で保留・本丸は Phase 4 へ)
- **2-6b** prompt→画像 本結線 (dual encoder + CFG・`IDiffusionRunner` で OV/CUDA 隔離) ✅ — prompt 供給元は将来 Phase 4 A の自作 LM に差し替え
- **2-6c** 拡散 backend プラグイン枠 ✅ — 品質天井は自作カーネルでなく拡散アーキ (重み) にあるため、prompt→RGB 境界を純 cpp interface `IDiffusionBackend` に切り出し registry (`make_backend`) 化。`SDXLBackend` (OV+CUDA 隔離) + `SD35Backend` (拡張点 stub・generate throw) + `BackendImageGenerator` (解像度 reject/seed/採点ログ/matting PNG 化の共通後処理を集約)。段1 DI を `Txt2ImgGenerator` から差し替え (env `DOLLAMA_BACKEND` で選択・既定 "sdxl")。ComfyUI 的 breadth は追わず「2D キャラ生成に要るアーキだけ芯を共有して差し替える」棲み分け ([[project-output-quality-over-features]])
- **2-6d** 実 checkpoint 差し替え = アニメ特化 SDXL を **3 preset 対応** (NoobAI-XL / Animagine XL 4.0 / Illustrious XL) 🔲 計画確定・未着手 — 3 つとも SDXL アーキゆえ `SDXLBackend` 無改修・`BackendConfig.preset` で重みセット選択。素 base 1.0 の質天井を超える最大レバー ([[project-generation-quality-bar]])。DL+変換+生成は GPU セッションで別途。詳細 roadmap 2-6d
- 部位構造化プロンプト ([[project-part-structured-prompt]]) — §11 QA・案B embedding と一緒に設計 (未着手バックログ)

**Phase 4 (自作タグ生成 LM + Model B) — 進行中**
- 自作 LM: tokenizer ✅ / 訓練 #1 (hard CE 6ep, 本線) ✅ / 推論 CPU・GPU・AVX2・INT8 ✅ / 同一性条件付き A (retention 0.975 頑健) ✅
- 評価: recall@10 退役 → **diverse 生成 set-F1 が主指標** (4-C) ✅
- 入力多様化 B: ~2,000 件で飽和・本線昇格決裁済 (出荷リトレイン `[B-merge-at-A]` で A と一括焼き) ✅
- **正典化 `[B-merge-at-A]` 完了 (2026-07-03)**: 勝者 33M で B(b2000)∧A(a12k identity) を merged 1本にまとめ焼き・ゲート4指標全通過 (retention 0.9807/diverse_a F1 0.3332/diverse_b F1 0.3804/in-dist 0.4552 で各単体参照を全軸超) → 正典 `bitnet_dense{,_fp32}`/identity を merged と同一バイトへ差し替え・golden 再生成・test_bitnet_infer corr 1.0/meson 25/25 緑・legacy アンカーを #1→新本線へ (training-spec §17 / dataset-spec §19)。follow-up=研究機で test_bitnet_gpu GPU golden 再確認 ✅
- 蒸留 D2/D4/D5/D6: 全不採用 (過学習抑制のみ・recall 非寄与) ✅
- 容量増 D (33M→80M): **陰性クローズ・80M 不採用/勝者=33M** (4seed sweep で F1 は seed ノイズ・retention 3/4 床割れ・in-dist 微退行) ✅ — diverse-val F1 を頑健に押し上げたのは B(~2,000 飽和) のみで、A(retention 専)/D(陰性)/蒸留4路線(非寄与) は F1 非寄与と確定
- Model B 品質スコアラ: QA ゲート Stage1 ✅ / ScorerNet 訓練・OV 変換・C++ 結線 (B-3b〜B-5) ✅ (採点はログのみ) / **Q-1 quality ラベル実採点 ✅** (waifu_scorer_v4 apache-2.0・quality=null は実行ギャップで解消・分布 0.0/med0.061/0.178 std0.0753・非退化だが [0,0.18] 圧縮 → Q-2 で head 凍結解除・raw_waifu から再正規化)
- 品質 FB ループ F: **F-0a 計装 ✅ + 実走 80/80 ✅ → 信号ゲート判定 = 信号弱 (補強してから) ✅** = reward std0.038 / best−worst0.203 (両方 PL 閾値 std>0.1・>0.3 に未達)。argmax **Limbs 77/80**・他7軸ほぼ死 (ScorerNet の分解能が Limbs のみ生存)。worst 帯は多人数/背景/mecha/文字焼き込み等**スコープ外題材**への発火が主 (画像照合: 単独素直題材は解剖正常で reward≈0)。ただし **clean vs clutter prompt で |r| 4倍分離** (0.007 vs 0.0285) = 弱いが本物の prompt 品質勾配あり (flat noise ではない=飽和 -1 帯でもない)
- **残 (F-0b 保留・補強を先に)**: ① **quality head 有効化 (Q-2, waifu 再正規化/deepghs 合流)** で reward に生存中の直交軸を足す (最小・最大レバー) ② **7 死軸の分解能診断** (B 側 ScorerNet: anatomy が Limbs 以上になれるか) ③ 補強後 F-0a smoke 再走で std→>0.1 or 分離維持を確認 → **F-0b SFT** / 残低帯域は A 実ペア増・D 容量増 / ternary GEMM は圧縮実験

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

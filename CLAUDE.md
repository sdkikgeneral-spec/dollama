# dollama — Claude 向けプロジェクトコンテキスト

## プロジェクト概要

**芯**: CPU / NPU / iGPU / RTX5080 — 搭載する全 HW を使い切りながら、
2D イラスト生成にたどり着くことを研究する。最短実装ではなく、
各 HW をどう活かし、どう協調させるかがこのプロジェクトの本質。

**HW 役割分担 (研究中・随時更新)**

| HW | 役割 | 状態 |
|---|---|---|
| CPU | Qwen2-1.5B LLM (プロンプト生成) | ✅ 64-71 tok/s 確認済み |
| NPU | **CLIP text encoder 7.85ms** / WD14 タグ抽出 268ms | ✅ CLIP が CPU の 2.5倍速 |
| iGPU (Intel Xe) | VAE encode (img2img、79ms) | ✅ CPU 117ms より速い |
| **RTX5080** | SDXL UNet + VAE decode | ✅ **3.80s/image** (1024×1024, 20steps, 5.3 it/s) |

**パイプライン構想**
```
User
 ↓
CPU: Qwen2-1.5B ─────────────────────────── text prompt
                                                │
iGPU: CLIP text encode ←───────────────────────┘
                                                │
RTX5080: SDXL UNet × steps ─────────────────── latent
       └── VAE decode ───────────────────────── image
                                                │
NPU: WD14 tagger ────────────── tags ──────────┤
   └── Aesthetic scorer ─────── score ─────────┘
                                                │
              (tags → CPU/LLM フィードバックループ)
```

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
std::thread llm_thread([&]  { /* CPU: 自作 BitNet b1.58 */   });
std::thread clip_thread([&] { /* NPU: 自作 CLIP 推論 */       });
std::thread sdxl_thread([&] { /* GPU: 自作 UNet CUDA カーネル */ });
std::thread tag_thread([&]  { /* NPU: 自作 WD14 推論 */       });

// SPSC lock-free queue でゼロコピー受け渡し
```

### 実装方針

| 使う | 使わない |
|---|---|
| STL 全般 | PyTorch / LibTorch |
| CUDA Runtime API | diffusers / stable-diffusion.cpp |
| Winsock2 (HTTP server) | llama.cpp / OpenVINO (probe のみ) |
| 自作 Tensor / GEMM / Attention | Drogon 等 HTTP フレームワーク |

### LLM の将来像

- 現在: Qwen2-1.5B (Python probe 用) — CPU 64-71 tok/s
- 目標: **自作 BitNet b1.58** — 重み {-1,0,+1}、multiply 不要、NPU 対応
  - 30-100M params、~20MB、user text → danbooru タグ生成特化

## 計測ベースライン

| 指標 | 値 | probe |
|---|---|---|
| CPU→VRAM (10MB) | 0.76ms | probe2 |
| CPU→VRAM (100MB) | 3.46ms / 30.3 GB/s | probe2 |
| NPU推論 (512dim MLP, 静的形状) | 0.88ms | probe2 |
| NPU出力 (2048B) → GPU | 0.031ms | probe2 |
| 転送オーバーヘッド | 3.4% | probe2 |
| iGPU VAE decode stub (Conv 4→512→...→3, 128→1024) | 995ms | probe4 |
| CPU VAE decode stub (同上) | 126ms | probe4 |
| NPU→iGPU ゼロコピー差分 (231KB) | 0.158ms (誤差) | probe4 |
| system RAM → RTX5080 latent (256KB) | 0.030ms / 8.7 GB/s | probe4 |
| system RAM → RTX5080 image (12MB) | 0.254ms / 49.6 GB/s | probe4 |

## 計測ベースライン (追加)

| 指標 | 値 | probe |
|---|---|---|
| Qwen2-1.5B INT4 CPU tok/s (英語) | 64-71 tok/s | probe7 |
| Qwen2-1.5B INT4 CPU ロード時間 | 1.1s | probe7 |
| Phi-3 mini INT4 CPU tok/s | 13-29 tok/s | probe6 |

## 次のタスク (C++ 実装フェーズ)

1. `src/` + `meson.build` 構築
2. `src/core/tensor.hpp` — 独自 Tensor クラス (STL ベース)
3. `src/kernels/ternary_gemm.cu` — BitNet ternary GEMM CUDA カーネル
4. `src/server/http.cpp` — Winsock2 OpenAI 互換 HTTP サーバー
5. 自作 BitNet モデルの訓練データ収集

## コーディング規約

- ファイル名プレフィックス: `dollma_` (dollama のプロジェクト内ファイル)
- プローブスクリプトは `dollma_probe*.py`
- 本実装は `dollma_pipeline.py` 等に分離予定
- コメントは日本語で書く

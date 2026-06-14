# dollama

**CPU / NPU / iGPU / RTX5080 — 搭載する全 HW を使い切る二次元特化の画像生成パイプライン研究**

各ハードウェアの特性を活かして協調させることが目的。最短実装ではなく、最適な HW 割り当てを探る。
ML フレームワークに頼らず C++ でフルスクラッチ実装を目指す。

---

## ターゲット環境

| コンポーネント | 詳細 |
|---|---|
| CPU / NPU | Intel Core Ultra 9 285 (NPU = Intel AI Boost, DEVICE_ARCHITECTURE: 3720) |
| GPU | NVIDIA GeForce RTX 5080 (Blackwell / sm_120, CUDA 12.8, VRAM 15.9GB) |
| iGPU | Intel Xe Graphics (OpenVINO GPU.0、システム RAM 共有) |
| OS | Windows 11 |
| 調査フェーズ | Python 3.14 + OpenVINO / diffusers (probe スクリプト) |
| 本実装 | C++ + Meson、STL + CUDA API + Winsock2 のみ (ML フレームワーク不使用) |

---

## HW 役割分担 (全て計測済み)

| HW | 担当タスク | 計測値 |
|---|---|---|
| **NPU** | CLIP-L text encoder (77token 固定) | **7.85ms** ← CPU の 2.5倍速 |
| **NPU** | WD14 SwinV2 tagger (448×448 固定) | 268ms (GPU 生成中に並列実行) |
| **iGPU** | VAE encode — img2img 用 (入力画像→latent) | **79ms** ← CPU 117ms より速い |
| **CPU** | LLM プロンプト生成 (暫定 Qwen2-1.5B → 将来 自作 BitNet b1.58) | 64-71 tok/s |
| **RTX5080** | SDXL UNet (20steps) + VAE decode | **3.80s** / 1024×1024 |

### NPU の得意 / 不得意

| モデル | アーキテクチャ | NPU 結果 | 理由 |
|---|---|---|---|
| CLIP-L text encoder | 標準 MHA、固定 77token | **7.85ms 最速** | 純粋 GEMM チェーン |
| WD14 SwinV2 | Window Attention | 268ms (CPU 101ms より遅い) | gather/scatter 操作が多い |
| LLM (Qwen2 等) | 自己回帰 KV-cache | ❌ コンパイル失敗 | 動的形状、NPU 設計外 |

---

## パイプライン構想

### txt2img

```
[CPU] Qwen2-1.5B (暫定) / 将来: 自作 BitNet b1.58 on NPU
  自然文 → danbooru タグ列 (~2s / 将来 <10ms)
    │
    ▼
[NPU] CLIP-L text encoder (7.85ms)
  テキスト → embedding [1, 77, 768]
    │
    ▼
[RTX5080] SDXL UNet × 20steps + VAE decode (3.80s / 1024×1024)
    │
    ├─ [CPU] WD14 SwinV2 tagger (101ms) ← GPU 生成中に並列実行
    │         生成画像 → danbooru タグ → LLM フィードバックループ
    └─ 出力画像
```

### img2img (追加パス)

```
入力画像
    ├─→ [iGPU] VAE encode (79ms)   ─→ latent ─┐
    │                                           │
    └─→ [CPU]  LLM テキスト生成 (~2s) ─────────┤ (並列)
                                               │
                                    [NPU] CLIP (7.85ms)
                                               │
                                    [RTX5080] SDXL UNet + VAE decode (3.80s)
                                               │
                                           出力画像
```

iGPU の VAE encode (79ms) は CPU の LLM 生成 (~2s) と並列に走るため、待ち時間ゼロ。

### デバイス選定根拠

| モデル | CPU | iGPU | NPU | 採用 | 理由 |
|---|---|---|---|---|---|
| CLIP-L text encoder | 20ms | 14ms | **7.85ms** | NPU | 純粋 GEMM チェーン |
| WD14 SwinV2 tagger | **101ms** | 104ms | 268ms | CPU | Window Attention が NPU に不向き |
| VAE decode | 126ms | 995ms | - | RTX5080 | iGPU は 8倍遅い |
| VAE encode (img2img) | 117ms | **79ms** | - | iGPU | encode 方向は iGPU が有利 |

---

## 出力画像サイズ

デフォルト **1024×1024** px。起動引数 `--width` / `--height` またはリクエストの `size` フィールドで変更可能。

SDXL の訓練解像度に合わせた推奨サイズ一覧 (8の倍数であれば任意指定も可):

| サイズ | アスペクト比 | 用途 |
|---|---|---|
| **1024×1024** | 1:1 | 正方形 (デフォルト、probe10 計測済み) |
| 1152×896 / 896×1152 | 9:7 | 横長 / 縦長 |
| 1216×832 / 832×1216 | 3:2 | |
| **832×1216** | 2:3 | 縦長ポートレート (2D イラスト向け) |
| 1344×768 / 768×1344 | 16:9 | ワイドスクリーン |
| 1536×640 / 640×1536 | 12:5 | 超横長 / 縦長 |

RTX5080 の VRAM ピークは 1024×1024 / 20steps で **10.49GB** (16GB 中)。1536×640 程度まで余裕あり。

### Stable Diffusion 各世代の学習解像度と品質上限

| モデル | 学習解像度 | 品質を維持できる上限 | 超えると |
|---|---|---|---|
| SD 1.x | 512×512 | ~512px | 破綻しやすい |
| SD 2.x | 768×768 | ~768px | 同上 |
| **SDXL (本プロジェクト採用)** | **1024×1024** | **長辺 ~1536px** | 同構図の繰り返しアーティファクト |

ソフト的な上限はなく VRAM 次第で任意サイズを指定できるが、学習解像度を大きく超えると品質が劣化する。  
4K 相当 (4096×4096 等) が必要な場合は **1024×1024 生成 → AI アップスケーラー (Real-ESRGAN / waifu2x) で 4× 拡大** が現実的な構成。

---

## 計測ベースライン (probe 実測値)

| 指標 | 値 | probe |
|---|---|---|
| CPU→VRAM (100MB) | 3.46ms / 30.3 GB/s | probe2 |
| NPU 推論 (512dim MLP) | 0.88ms | probe2 |
| NPU 出力 → GPU | 0.031ms (3.4%) | probe2 |
| system RAM → RTX5080 latent (256KB) | 0.030ms / 8.7 GB/s | probe4 |
| system RAM → RTX5080 image (12MB) | 0.254ms / 49.6 GB/s | probe4 |
| iGPU VAE decode stub | 995ms (CPU 比 8倍遅い ❌) | probe4 |
| iGPU VAE encode (1024→128) | **79ms** (CPU 117ms より速い ✅) | probe5 |
| Qwen2-1.5B INT4 CPU tok/s | 64-71 tok/s | probe7 |
| **WD14 SwinV2 (448×448)** | CPU 101ms / iGPU 104ms / **NPU 268ms** | probe8 |
| **CLIP-L text encoder (77token)** | CPU 20ms / iGPU 14ms / **NPU 7.85ms** | probe9 |
| **SDXL 20steps 1024×1024** | **3.80s** / 5.3 it/s / VRAM ピーク 10.49GB | probe10 |

---

## LLM の将来像 — 自作 BitNet b1.58

汎用 LLM (Qwen2-1.5B, 873MB, CPU ~2s) を目的特化の超軽量モデルに置き換える:

```
重み W ∈ {-1, 0, +1}  (log₂3 ≈ 1.58 bit)
演算: y = x_pos - x_neg  ← multiply 不要、XNOR + popcount
```

| | Qwen2-1.5B (現状) | 自作 BitNet (目標) |
|---|---|---|
| パラメータ | 1.5B | 30-100M |
| サイズ | 873MB | ~20MB |
| デバイス | CPU | NPU (固定形状) |
| レイテンシ | ~2s | <10ms |
| タスク | 汎用 | user text → danbooru タグ特化 |

訓練データ: Danbooru キャプション + Qwen2 蒸留

---

## 確定済み設計判断

### ゼロコピー調査結果 (probe1-4)

| ルート | 結果 | 理由 |
|---|---|---|
| CUDA Virtual Memory + Win32ハンドル → NPU | ❌ | OpenVINO NPU に CUDA ハンドル import API なし |
| D3D12 クロスアダプター (RTX5080 → iGPU → NPU) | ❌ | Intel iGPU が DXGI に非表示 |
| CPU pinned memory | ✅ 採用 | オーバーヘッド 3.4%、マルチスレッドで隠蔽可能 |

---

## 実装方針 (C++ フルスクラッチ)

```
src/
├── core/
│   ├── tensor.hpp        — 独自 Tensor (STL ベース)
│   └── allocator.hpp     — CPU / pinned / VRAM メモリ管理
├── kernels/
│   ├── ternary_gemm.cu   — BitNet ternary GEMM (CUDA)
│   ├── attention.cu      — 自作 Multi-Head Attention
│   └── rms_norm.cu
├── models/
│   ├── tokenizer.cpp     — 自作 BPE
│   ├── bitnet.cpp        — BitNet b1.58 推論
│   └── clip.cpp          — CLIP text encoder
└── server/
    └── http.cpp          — Winsock2 + OpenAI API 互換
```

**使うもの**: STL / CUDA API / Winsock2  
**使わないもの**: PyTorch / OpenVINO / diffusers / llama.cpp 等の ML フレームワーク

---

## セットアップ (調査フェーズ / Python probe)

```bash
pip install openvino openvino-genai openvino-tokenizers
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install diffusers transformers accelerate optimum[openvino]
pip install huggingface_hub
```

---

## ファイル構成

```
dollama/
  scripts/
    dollma_probe8_wd14.py      # WD14 SwinV2 NPU/iGPU/CPU 比較 → NPU 268ms
    dollma_probe9_clip.py      # CLIP-L text encoder → NPU 7.85ms (最速)
    dollma_probe10_sdxl.py     # SDXL RTX5080 → 3.80s/image
    archives/                  # probe1-7b (完了済み調査スクリプト)
  src/                         # C++ 本実装 (構築中)
  models/                      # 変換済みモデル (Git 管理外)
    clip-l-text-encoder/       # OV IR (NPU 用)
    qwen2-1.5b-int4-npu/       # 暫定 LLM
    wd14-swinv2-tagger-v3/     # OV IR (NPU 用)
  outputs/                     # 生成画像 (Git 管理外)
  logs/                        # probe 実行ログ
  docs/
    investigation-log.md       # 詳細調査ログ
  CLAUDE.md                    # Claude Code 向けコンテキスト
```

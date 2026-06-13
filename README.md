# dollama

**CPU / NPU / iGPU / RTX5080 — 搭載する全 HW を使い切る二次元特化の画像生成パイプライン研究**

各ハードウェアの特性を活かして協調させることが目的。最短実装ではなく、最適な HW 割り当てを探る。

---

## ターゲット環境

| コンポーネント | 詳細 |
|---|---|
| CPU / NPU | Intel Core Ultra 9 285 (NPU = Intel AI Boost, DEVICE_ARCHITECTURE: 3720) |
| GPU | NVIDIA GeForce RTX 5080 (Blackwell / sm_120, CUDA 12.8) |
| iGPU | Intel Xe Graphics (OpenVINO GPU.0) |
| OS | Windows 11 |
| Python | 3.14 (調査フェーズ) |
| 本実装 | C++ + Meson (Windows / Linux 両対応予定) |

---

## HW 役割分担

| HW | 担当 | 状態 |
|---|---|---|
| CPU | Qwen2-1.5B INT4 LLM (プロンプト生成) | ✅ 64-71 tok/s 確認済み |
| NPU | WD14 SwinV2 タグ抽出 / CLIP-L text encoder / Aesthetic scorer | 🔬 計測中 |
| iGPU | 軽量前処理のみ (VAE decode は CPU 比 8倍遅いため不採用) | ✅ 性能確認済み |
| RTX5080 | SDXL / SD3.5 UNet + VAE decode | ⏳ 未着手 |

---

## パイプライン構想

```
ユーザー入力
    │
    ▼
[Thread-A: CPU]
  Qwen2-1.5B INT4 → 英語プロンプト生成 (64-71 tok/s)
    │
    │  CPU pinned memory 経由 (転送オーバーヘッド 3.4%)
    ▼
[Thread-B: RTX5080]
  CLIP-L text encoder → embedding
  SDXL UNet × steps  → latent
  VAE decode          → RGB image
    │
    ▼
[NPU: サイドパス]
  WD14 SwinV2  → danbooru tags
  Aesthetic scorer → quality score
    │
    └─ フィードバック → CPU LLM (次回生成に反映)
```

---

## 確定済み設計判断

### ゼロコピー調査結果

| ルート | 結果 | 理由 |
|---|---|---|
| CUDA Virtual Memory + Win32ハンドル → NPU | ❌ | OpenVINO NPU に CUDA ハンドル import API なし |
| D3D12 クロスアダプター (RTX5080 → iGPU → NPU) | ❌ | Intel iGPU が DXGI に非表示 (BIOS でコンピュート専用) |
| CPU pinned memory | ✅ 採用 | オーバーヘッド 3.4%・マルチスレッドで隠蔽可能 |

### NPU の制約

- 静的形状のみ受け付ける → `ov_model.reshape(...)` をコンパイル前に必須
- LLM 自己回帰推論には不適 (KV-cache でシーケンス長が動的に増加)
- NPU が適切な用途: 固定入力形状の encoder (WD14=448px / CLIP=77tokens)

---

## 計測ベースライン

| 指標 | 値 | probe |
|---|---|---|
| CPU→VRAM (10MB) | 0.76ms | probe2 |
| CPU→VRAM (100MB) | 3.46ms / 30.3 GB/s | probe2 |
| NPU推論 (512dim MLP, 静的形状) | 0.88ms | probe2 |
| NPU出力 (2KB) → GPU | 0.031ms (3.4%) | probe2 |
| system RAM → RTX5080 latent (256KB) | 0.030ms / 8.7 GB/s | probe4 |
| system RAM → RTX5080 image (12MB) | 0.254ms / 49.6 GB/s | probe4 |
| iGPU VAE decode stub | 995ms (CPU 比 8倍遅い) | probe4 |
| NPU→iGPU ゼロコピー差分 (231KB) | 0.158ms (誤差範囲) | probe4 |
| Qwen2-1.5B INT4 CPU tok/s | 64-71 tok/s | probe7 |
| Qwen2-1.5B INT4 ロード時間 | 1.1s | probe7 |

---

## 技術スタック

| レイヤー | 技術 |
|---|---|
| NPU推論 | OpenVINO 2024.x (`import openvino as ov`) |
| GPU推論 | PyTorch cu128 + diffusers |
| LLM | llama.cpp / transformers (Qwen2-1.5B INT4 on CPU) |
| ビルド (本実装) | C++ + Meson (Windows / Linux) |
| 調査スクリプト | Python 3.14 |

---

## セットアップ (調査フェーズ)

```bash
# OpenVINO
pip install openvino

# PyTorch (RTX5080 = Blackwell / cu128 必須)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# diffusers / transformers
pip install diffusers transformers accelerate huggingface_hub
```

---

## ファイル構成

```
dollama/
  scripts/
    dollma_probe8_wd14.py     # WD14 SwinV2 NPU/iGPU/CPU 比較
    dollma_probe9_clip.py     # CLIP-L text encoder NPU/iGPU/CPU 比較
    dollma_probe10_sdxl.py    # SDXL RTX5080 動作確認
    archives/                 # probe1-7 (完了済み調査)
  .claude/
    agents/                   # Claude Code サブエージェント定義
      project-leader.md       # タスク分割・調整 (コーディングなし)
      npu-benchmarker.md      # NPU 計測・OpenVINO 変換
      gpu-benchmarker.md      # RTX5080 計測・diffusers 推論
      model-converter.md      # ONNX → OV IR 変換・量子化
      pipeline-debugger.md    # スレッド間デバッグ・ボトルネック診断
      prompt-engineer.md      # SD プロンプト最適化・danbooru タグ変換
  docs/
    investigation-log.md      # 調査ログ
  models/                     # ローカルモデル置き場 (Git 管理外)
  outputs/                    # 生成画像 (Git 管理外)
  CLAUDE.md                   # Claude Code 向けプロジェクトコンテキスト
```

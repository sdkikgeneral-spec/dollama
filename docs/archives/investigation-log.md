# dollama 調査ログ

NPU + GPU パイプライン構築に向けた技術調査の記録。

---

## probe1 — NPU/GPU 環境確認 (STEP 1-4)

**目的**: OpenVINO デバイス列挙・CUDA VRAM・Virtual Memory API・転送レイテンシの確認

**結果**:

| 項目 | 結果 |
|---|---|
| OpenVINO デバイス | `['CPU', 'GPU', 'NPU']` |
| NPU | Intel AI Boost / DEVICE_ARCHITECTURE: 3720 / FP16・INT8対応 |
| CUDA VRAM 割り当て | OK (cuMemAlloc_v2) |
| Virtual Memory API 粒度 | 2MB (Win32ハンドル対応) |
| CPU→VRAM (100MB) | 3.46ms / 30.3 GB/s |

**判明事項**: Virtual Memory API は利用可能。Win32ハンドルで共有できる可能性あり → probe2 へ。

---

## probe2 — Win32ハンドル + NPU パイプライン (STEP 5-7)

**目的**: cuMemCreate + Win32ハンドル共有の確認 / NPU推論パイプラインの実測

**発生したエラーと修正**:

| エラー | 原因 | 修正 |
|---|---|---|
| `cuMemCreate INVALID_VALUE` | `CUmemAllocationProp` の構造体レイアウト不正 | `CUmemLocation` をネストした正しい構造体に修正 |
| `cuMemCreate INVALID_VALUE` (再) | `win32HandleMetaData` が NULL | `SECURITY_ATTRIBUTES` を作成してポインタを渡す |
| `cannot access local variable 'ctypes'` | `import ctypes.wintypes` を関数内で行っていた | モジュールレベルに移動 |
| `No module named 'openvino.runtime'` | OpenVINO 2024.x で廃止 | `import openvino as ov` に変更 |
| `Missing upper bound` (NPU compile) | NPU は動的形状不可 | `ov_model.reshape([1, 512])` を追加 |

**計測結果**:

| 指標 | 値 |
|---|---|
| NPU推論 (512dim MLP) | 0.88ms |
| NPU出力 (2048B) → GPU | 0.031ms |
| 転送オーバーヘッド | 3.4% |

**判定**: CPU経由で十分。マルチスレッドで隠蔽可能。

**NPU パイプラインの確定パターン**:
```python
ov_model = ov.convert_model(torch_model, example_input=example_input)
ov_model.reshape([1, 512])  # NPU は静的形状必須
compiled = core.compile_model(ov_model, "NPU")
infer_req = compiled.create_infer_request()

npu_output = infer_req.get_output_tensor(0).data
pinned = torch.from_numpy(npu_output.copy()).pin_memory()
gpu_tensor = pinned.cuda(non_blocking=True)
```

---

## probe3 — D3D12 クロスアダプター調査

**目的**: RTX5080 → Intel iGPU → NPU のゼロコピー経路が存在するか確認

**結論**: **不可**

- Intel iGPU が `IDXGIFactory6` (MINIMUM_POWER 列挙) に表示されない
- BIOSでiGPUがコンピュート専用モードに設定されているため DXGI に非表示
- D3D12 クロスアダプターはこの構成では実装不可

---

## OpenVINO NPU プラグイン ソースコード調査

**目的**: OpenVINO をフォークして CUDA ハンドルインポートを追加できるか確認

**調査対象**: `openvinotoolkit/openvino` GitHub

**発見**:

NPU プラグイン (`src/plugins/intel_npu/`) に Level Zero ベースの外部メモリ interop が実装済み:
- ファイル: `zero_remote_tensor.cpp`, `zero_mem.cpp`
- サポート: `SHARED_BUF` / `CPU_VA` / NT handle (Windows) / DMA-BUF (Linux)
- 実装クラス: `ZeroRemoteTensor`

GPU プラグイン (`src/plugins/intel_gpu/`) に D3D12 共有ハンドルサポートあり:
- `SharedMemType::BUFFER_FROM_HANDLE` で NT handle インポート可能

**CUDA interop の可否**:
- NPU ↔ CUDA: 実装なし → 不可
- iGPU ↔ CUDA (D3D12経由): ドライバーレベルで相互運用なし → 不可
- Level Zero は Intel GPU 専用で NVIDIA GPU を対象外

**結論**: フォークで CUDA interop を追加するには NPU プラグインに `cuImportExternalMemory` 経路を実装する必要があるが、NVIDIA ドライバーと Level Zero の相互運用が存在しないためドライバー層で詰まる。現実的でない。

---

## probe4 — Intel SoC クラスター検証 (iGPU BIOS有効化後)

**前提**: BIOSでiGPUを DXGI に表示するよう設定変更後

**STEP 1: デバイス識別**

```
['CPU', 'GPU.0', 'GPU.1', 'NPU']
  GPU.0: Intel(R) Graphics  [INTEGRATED]  FP16/INT8/GPU_USM_MEMORY
  GPU.1: NVIDIA GeForce RTX 5080  [DISCRETE]
```

**STEP 2: iGPU VAE デコード速度**

| デバイス | 速度 |
|---|---|
| Intel iGPU (GPU.0) | 995ms |
| CPU | 126ms |
| iGPU/CPU 比 | **0.1x (iGPU が 8倍遅い)** |

→ 大規模 ConvTranspose2d は iGPU に向かない。VAE decode は RTX5080 か CPU が適切。

**STEP 3: NPU → iGPU ゼロコピー**

| パターン | 時間 |
|---|---|
| ゼロコピー (直接渡し) | 1.09ms |
| コピーあり | 0.93ms |
| 差分 | 0.158ms (誤差範囲) |

→ **システムRAM共有確認**。NPU出力をそのまま iGPU に渡せる (コピーコスト≈0)。

**STEP 4: iGPU → RTX5080 転送コスト**

| データ | サイズ | 時間 |
|---|---|---|
| text embedding | 231KB | 0.026ms |
| latent 128x128x4 | 256KB | 0.030ms |
| image 1024x1024x3 | 12MB | 0.254ms |

→ latent の PCIe 転送コストは拡散1ステップに対して誤差レベル。

---

## アーキテクチャ決定まとめ

```
現在の確定構成:

  ユーザー入力
      ↓
  NPU (OpenVINO)        : テキストエンコーダー (LLM / CLIP)
      ↓ システムRAM (ゼロコピー)
  CPU (pinned memory)   : 中継バッファ
      ↓ PCIe (~0.03ms)
  RTX5080 (PyTorch)     : 拡散モデル UNet + VAE decode
      ↓
  出力画像
```

**廃案になったルート**:

| ルート | 理由 |
|---|---|
| CUDA Win32ハンドル → NPU | NPU に CUDA interop なし |
| D3D12 クロスアダプター | iGPU が DXGI 非表示 (BIOS設定依存) |
| iGPU VAE decode | CPU の 8倍遅い (probe4) |
| OpenVINO フォーク | ドライバー層で CUDA interop なし |

**iGPU の位置づけ**:
- アクセス可能 (BIOS有効化後) ✅
- NPU とゼロコピー共有 ✅
- 大規模 Conv は CPU に負ける ❌
- 軽量前処理・リサイズ・正規化なら有効かもしれない (未検証)

---

## probe5 — iGPU 活用候補の実測比較

**目的**: iGPU を使うべきタスクを実測で決める

**結果**:

| 候補 | iGPU | CPU | 比 | 判定 |
|---|---|---|---|---|
| A: ControlNet depth (512px) | 57.9ms | 15.7ms | 0.3x | ❌ CPU の方が速い |
| B: CLIP image encoder (224px) | 3.3ms | 1.8ms | 0.5x | ❌ CPU の方が速い |
| C: VAE Encode 1024→128 | **79.2ms** | 117.4ms | **1.5x** | ✅ iGPU が速い |
| D: Aesthetic Scorer MLP | 0.1ms | 0.0ms | 0.3x | ❌ CPU の方が速い |

**probe4 との対比 (VAE 非対称性)**:

| 操作 | iGPU | CPU | 理由 |
|---|---|---|---|
| VAE Encode (1024→128, Conv downscale) | 79ms ✅ | 117ms | 出力小 (256KB)、書き込みが少ない |
| VAE Decode (128→1024, ConvTranspose upscale) | 995ms ❌ | 126ms | 出力大 (12MB)、共有RAM帯域を圧迫 |

**iGPU の弱点**: 大量書き込みを伴う操作 (ConvTranspose + 大出力) はシステムRAM 帯域がボトルネックになる。

**結論**:
- **iGPU 採用**: VAE Encode (img2img ワークフロー限定)
- **iGPU 不採用**: VAE Decode / ControlNet / CLIP / Aesthetic MLP → CPU か RTX5080

**採用時の流れ (img2img)**:
```
入力画像
  ↓ iGPU: VAE Encode (79ms)   ← CPU より 38ms 速い
  ↓ PCIe → RTX5080 (0.030ms)
  ↓ RTX5080: noise + UNet 30steps (~1500ms)
  ↓ RTX5080: VAE Decode (native PyTorch, 高速)
出力画像
```

---

## アーキテクチャ決定まとめ (最新)

```
text2img:
  NPU  : テキストエンコーダー
  CPU  : 中継 (pinned memory)
  RTX5080: UNet拡散 + VAE Decode

img2img:
  iGPU : VAE Encode (入力画像 → latent)  ← iGPU が唯一有利な箇所
  NPU  : テキストエンコーダー
  CPU  : 中継
  RTX5080: UNet拡散 + VAE Decode
```

**廃案になったルート**:

| ルート | 理由 |
|---|---|
| CUDA Win32ハンドル → NPU | NPU に CUDA interop なし |
| D3D12 クロスアダプター | iGPU が DXGI 非表示 (BIOS設定依存) |
| iGPU VAE Decode | CPU の 8倍遅い (probe4) |
| OpenVINO フォーク | ドライバー層で CUDA interop なし |
| iGPU ControlNet / CLIP / MLP | CPU の方が速い (probe5) |

---

---

## Phi-3 mini INT4 変換トラブルシューティング

**環境**: Python 3.14 + optimum 2.2.0 + transformers 5.0.0 + openvino 2026.2.0

**エラー**:
```
TypeError: NormalizedConfig.__init__() got multiple values for argument 'allow_new'
```

**根本原因**: Python 3.14 の `functools.partial` が C 実装になり **descriptor protocol (`__get__`) を追加した**。

```python
import functools
def f(x): pass
p = functools.partial(f, 1)
print(hasattr(p, '__get__'))  # True (Python 3.14 のみ)
```

クラス属性に `functools.partial` を置くと、インスタンスからアクセスした際に通常の関数と同様に `self` がバインドされる:

```python
class Foo:
    method = functools.partial(f, allow_new=True)

foo = Foo()
# foo.method → bound method、self が第一引数にバインドされる
# foo.method(config) → f(foo, config, allow_new=True)
# NormalizedConfig.__init__(self, phi3ov, config, allow_new=True)
#   → config が allow_new のポジションを埋め、さらに allow_new=True → 二重渡し
```

**影響範囲**: optimum の `NormalizedConfig.with_args()` は `functools.partial` を返してクラス属性に設定する設計。Python 3.14 でこのパターン全体が壊れる。

**修正**: `NormalizedTextConfigWithGQA` は既に `NUM_KEY_VALUE_HEADS = "num_key_value_heads"` を持つため、`with_args` なしで使用可能。

```python
# dollma_convert.py に追加
from optimum.utils.normalized_config import NormalizedTextConfigWithGQA
from optimum.exporters.openvino.model_configs import Phi3OpenVINOConfig, PhiMoEOpenVINOConfig
Phi3OpenVINOConfig.NORMALIZED_CONFIG_CLASS = NormalizedTextConfigWithGQA
PhiMoEOpenVINOConfig.NORMALIZED_CONFIG_CLASS = NormalizedTextConfigWithGQA
```

---

---

## probe6 — Phi-3 mini INT4 NPU/CPU 推論計測

**モデル**: `microsoft/Phi-3-mini-4k-instruct` INT4 (optimum-intel)  
**ファイルサイズ**: openvino_model.bin 1984 MB

### STEP 1: ファイル構成

```
models/phi3-mini-int4-npu/
  openvino_model.bin       1984 MB  (INT4 量子化済み)
  openvino_model.xml          2.6 MB
  openvino_tokenizer.xml      0.0 MB  (openvino_tokenizers で自動生成)
  openvino_detokenizer.xml    0.0 MB
  tokenizer.json              3.5 MB  (HF tokenizer)
```

**量子化分布** (NNCF):
| モード | 割合 |
|---|---|
| INT4_asym group_size=128 | 95% (128/130 layers) |
| INT8_asym per-channel | 5% (2/130 layers) |

### STEP 2: OV Tokenizer デバイス速度

OV Tokenizer は `SpecialTokensSplit` カスタム op を使用するため、`import openvino_tokenizers` で extension 登録が必要。

| デバイス | 速度 | 備考 |
|---|---|---|
| CPU | 0.200ms ✅ | 動作 |
| iGPU (GPU.0) | NG | StringTensorUnpack (opset15) 非対応 |
| NPU | NG | Incorrect precision: string |

**結論**: トークナイザーは CPU のみ。文字列型テンソルはアクセラレーター非対応。

### STEP 3: LLMPipeline (NPU) 

**結果**: コンパイル失敗

```
[vpux-compiler] StopLocationVerifierPass Pass failed: Found 184 duplicated names
Compilation failed. Level0 pfnCreate2 result: ZE_RESULT_ERROR_INVALID_NULL_POINTER
```

**原因の仮説**:
1. Phi-3 mini (3.8B params, INT4で2GB) が NPU のオンチップメモリ容量を超えている
2. VPUX コンパイラの Phi-3 アーキテクチャに対するバグ
3. 動的な形状 / ページドアテンション機構が NPU と非互換

### STEP 4: LLMPipeline (CPU)

**ロード時間**: 1.2s

| プロンプト | 時間 | 推定 tok/s |
|---|---|---|
| "こんにちは" (短文) | 3179ms | 13.2 |
| "1girl, magical girl..." | 2033ms | 29.0 |
| "Generate a SD prompt..." | 1326ms | 26.4 |

**CPU まとめ**: Phi-3 mini INT4 は CPU で約 13-29 tok/s。プロンプト生成用途には十分な速度。

### 判断: LLM を何で動かすか

| 選択肢 | tok/s | TTFT | 問題 |
|---|---|---|---|
| Phi-3 mini on CPU | ~20 tok/s | 1.2s | NPU が遊ぶ |
| Phi-3 mini on NPU | ❌ コンパイル失敗 | - | VPUX バグ or OOM |
| 小型モデル (Phi-2 / Qwen 1.8B) on NPU | 未計測 | - | 要調査 |

→ **現状**: CPU で Phi-3 mini INT4 を動かすのが最も確実。NPU は別モデルで再調査。

---

## probe7 — Qwen2-1.5B-Instruct INT4 NPU コンパイル検証

**目的**: Phi-3 mini の NPU 失敗がモデルサイズ起因か VPUX コンパイラバグかを切り分ける

**モデル**: Qwen2-1.5B-Instruct INT4 (873MB, Phi-3 mini の半分)

### STEP 2+3: LLMPipeline (NPU)

**結果**: コンパイル失敗 (4.8s)

```
[vpux-compiler] StopLocationVerifierPass Pass failed: Found 142 duplicated names
Compilation failed. Level0 pfnCreate2 result: ZE_RESULT_ERROR_INVALID_NULL_POINTER
```

| モデル | サイズ | 重複名数 | 結果 |
|---|---|---|---|
| Phi-3 mini INT4 | 1984MB | 184 重複 | ❌ |
| Qwen2-1.5B INT4 | 873MB | 142 重複 | ❌ |

→ 重複名数がモデルサイズ(層数)と比例 = **モデル層数に比例した系統的なバグ**

### STEP 4: LLMPipeline (CPU)

| プロンプト | 時間 | 推定 tok/s |
|---|---|---|
| "こんにちは" | 859ms | 8.1 |
| "1girl, magical girl..." | 1032ms | 64.9 |
| "Generate SD prompt..." | 1027ms | 71.0 |

CPU では Qwen2-1.5B は **64-71 tok/s** と Phi-3 mini の 2-3倍高速。

---

## probe7b — NPU コンパイル失敗の根本原因解析

**目的**: `LLMPipeline` の内部処理 vs 低レベル `compile_model` のどちらが失敗源か切り分け

### STEP 1+2: `OVModelForCausalLM` (use_cache=False)

**結果**: `use_cache=False` は export 時に設定が必要。保存済みモデルは変更不可。

### STEP 3: `core.compile_model` で openvino_model.xml を NPU に直接コンパイル

**結果**: `to_shape was called on a dynamic shape` → NPU が動的形状を拒否

### 結論

```
openvino_model.xml:
  動的形状 (seq_len = ?)
      ↓ LLMPipeline が reshape (静的化)
      ↓ VPUX compile_model
      ↗ StopLocationVerifierPass: 142 重複名 ❌  ← バグはここ
```

**根本原因 (修正)**: VPUX コンパイラバグではなく、**NPU の設計哲学との根本的な不一致**。

Intel AI Boost (NPU) は固定形状・事前コンパイル型の DSP アクセラレーター。
自己回帰 LLM (KV-cache でシーケンス長が token ごとに動的に増加) とは設計上非互換。

| 特性 | Intel NPU | GPU |
|---|---|---|
| 形状 | 静的のみ (コンパイル時確定) | 動的対応 |
| KV-cache | 非対応 (シーケンスが伸びる) | 対応 |
| 用途 | 画像分類・音声認識・固定 CNN | 汎用行列演算 |
| LLM 自己回帰 | ❌ 設計外 | ✅ |

**正しい NPU の用途 (dollama コンテキスト)**:
- CLIP テキストエンコーダー (77トークン固定) ← probe2 で NPU は MLP 0.88ms で動作確認済
- Whisper encoder (音声入力)
- VAE encoder 的な固定解像度処理

---

## 確定アーキテクチャ (LLM 配置の修正)

```
旧想定:  NPU = LLM (テキストエンコーダー)   ← 誤り、NPU は LLM に非対応
新確定:  CPU = LLM (Qwen2-1.5B, 64-71 tok/s)
         NPU = CLIP encoder (77トークン固定形状) ← 要検証
         RTX5080 = SDXL/SD3.5 UNet + VAE
```

---

## probe8 — WD14 SwinV2 Tagger NPU/iGPU/CPU 比較

**モデル**: SmilingWolf/wd-swinv2-tagger-v3 (ONNX 445.8MB → OV IR)  
**入力**: `[1, 448, 448, 3]` 固定形状 / **タグ数**: 10,861 クラス

**変換**: `ONNX → core.read_model() → reshape([1,448,448,3]) → save_model()`  
→ 固定形状 CNN は LLM と違い reshape が素直に通る

| デバイス | コンパイル | 推論 (中央値) |
|---|---|---|
| CPU | 0.9s | **101ms** |
| iGPU (GPU.0) | 4.5s | 104ms |
| **NPU** | **43s** | 268ms ✅ |

**重要**: NPU で実モデルが初めてコンパイル・推論成功。LLM 失敗は KV-cache 動的形状が原因であり、固定形状モデルは NPU で動く。

NPU が CPU より遅い理由: SwinV2 の Window Attention は gather/scatter 操作が多く、NPU 得意の Conv/GEMM チェーンではない。ConvNeXt 系なら改善の可能性あり。

**パイプライン評価**: RTX5080 UNet (1-2s) >> NPU WD14 (268ms) → 前の画像を NPU でタグ付けしながら GPU が次を生成 → 並列成立 ✅  
NPU コンパイル 43s は `CACHE_DIR` で初回のみに削減可能。

---

## probe9 — CLIP-L text encoder NPU/iGPU/CPU 比較

**モデル**: openai/clip-vit-large-patch14 (text encoder のみ)  
**変換**: `CLIPModel → text_model → ov.convert_model() → reshape([1,77])`  
**入力**: `[1, 77]` int32 固定 / **出力**: `[1,77,768]` + `[1,768]`

| デバイス | コンパイル | 推論 (中央値) |
|---|---|---|
| CPU | 0.7s | 20.04ms |
| iGPU (GPU.0) | 2.2s | 14.11ms |
| **NPU** | **5.8s** | **7.85ms ← 最速** |

### 重要発見: NPU の得意領域が確定

| モデル | アーキテクチャ | NPU結果 |
|---|---|---|
| WD14 SwinV2 | Window Attention (gather/scatter 多用) | 268ms ← CPU より遅い |
| CLIP text encoder | 標準 MHA + 固定 77 token | **7.85ms ← CPU の 2.5倍速** |

→ **NPU は純粋な GEMM チェーン (標準 MHA + FFN) が得意。Window Attention など複雑なインデックス操作は苦手。**

SDXL は CLIP-L + OpenCLIP-bigG の 2 エンコーダー構成 → 両方 NPU に乗せられる可能性あり。

---

## 確定: dollama 4-HW タスク割り当て

| HW | タスク | 計測値 |
|---|---|---|
| **NPU** | CLIP-L text encode (SD テキスト条件付け) | **7.85ms** |
| **iGPU** | VAE encode (img2img、入力画像→latent) | **79ms** (CPU 117ms より速い) |
| **CPU** | Qwen2-1.5B LLM (プロンプト生成) | **64-71 tok/s** |
| **RTX5080** | SDXL UNet + VAE decode | 未計測 |

---

## probe10 — SDXL on RTX5080 (初の実画像生成)

**プロンプト**: `1girl, anime, silver hair, magical girl, twin tails, detailed eyes, soft lighting, pastel colors, masterpiece, best quality`

| 指標 | 値 |
|---|---|
| ロード時間 | 43.0s (初回、モデルキャッシュ後は短縮) |
| 生成時間 (20steps, 1024×1024) | **3.80s** (中央値) |
| スループット | **5.3 steps/s** (190ms/step) |
| VRAM ロード時 | 6.57 GB / 15.9 GB |
| VRAM ピーク | 10.49 GB (残 5.4 GB) |

**結果**: 銀髪ツインテール魔法少女の生成に成功 ✅ (`outputs/probe10_sdxl_test.png`)

**4-HW タイミング試算**:
```
CPU: LLM (Qwen2-1.5B)  |────── ~2s ──────|
NPU: CLIP encode                           |8ms|
GPU: SDXL 20steps                               |──── 3.80s ────|
NPU: WD14 feedback tag                                           |268ms|

→ LLM が SDXL の中に収まる → 2枚目以降の実効レイテンシ ≈ 3.80s
```

**VRAM 余裕**: ピーク 10.49GB / 15.9GB → 5.4GB 空き。SDXL Refiner や LoRA の同時ロードも可能。

---

## 確定: 4-HW 全タスク割り当て (計測済み)

| HW | タスク | 計測値 |
|---|---|---|
| **NPU** | CLIP-L text encode | **7.85ms** (CPU の 2.5倍速) |
| **NPU** | WD14 タグ抽出 (フィードバック) | **268ms** |
| **iGPU** | VAE encode (img2img) | **79ms** (CPU 117ms より速い) |
| **CPU** | Qwen2-1.5B LLM | **64-71 tok/s** (~2s/60token) |
| **RTX5080** | SDXL UNet 20steps + VAE decode | **3.80s** (1024×1024) |

---

## 次のステップ

1. ~~SDXL on RTX5080~~ → probe10 完了 (3.80s/image, 5.3 steps/s)
2. **probe11: 4-HW キューパイプライン** ← 次
   - CPU(LLM) → NPU(CLIP) → GPU(SDXL) → NPU(WD14) のスレッド接続
   - キューのバックプレッシャー計測
   - 実効スループット vs 各 HW 単体の測定
3. OpenCLIP-bigG (SDXL 2本目 encoder) も NPU で計測
4. `torch.compile` で SDXL UNet を高速化 (5.3 → 目標 8+ steps/s)

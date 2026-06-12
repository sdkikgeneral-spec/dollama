# dollama — Claude 向けプロジェクトコンテキスト

## プロジェクト概要

Intel NPU + NVIDIA GPU を使った二次元特化の軽量画像生成パイプライン。
NPU (Intel AI Boost) で LLM 推論、RTX5080 で拡散モデルを動かし、
マルチスレッドで並列処理する。

## 環境

- OS: Windows 11
- CPU/NPU: Intel Core Ultra 9 285 (NPU = Intel AI Boost, DEVICE_ARCHITECTURE: 3720)
- GPU: NVIDIA GeForce RTX 5080
- Python: 3.14
- OpenVINO: 2024.x 以降 (openvino.runtime は廃止済み)
- PyTorch: cu128 ビルド (RTX5080 = Blackwell / sm_120 = CUDA 12.8 必須)

## 確定済みアーキテクチャ決定

### CPU 経由パイプライン (ゼロコピーは採用しない) ← 全ルート調査済み・確定

調査したゼロコピーの全ルートと結果:

| ルート | 結果 | 理由 |
|---|---|---|
| CUDA Virtual Memory + Win32ハンドル → NPU | ❌ | OpenVINO NPU に CUDA ハンドル import API なし |
| D3D12 クロスアダプター (RTX5080 → iGPU → NPU) | ❌ | Intel iGPU が DXGI に非表示 (BIOS でコンピュート専用) |
| CPU pinned memory | ✅ | 3.4% オーバーヘッド・マルチスレッドで隠蔽可能 |

Intel iGPU は Level Zero / OpenCL 経由でのみアクセス可能。
IDXGIFactory6 の MINIMUM_POWER 列挙でも DXGI に出てこない。

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

### パイプライン構造

```python
# Thread-A: NPU
npu_output = infer_req.infer({0: input_np})
pinned = torch.from_numpy(npu_output.copy()).pin_memory()
gpu_tensor = pinned.cuda(non_blocking=True)
queue.put(gpu_tensor)

# Thread-B: GPU
conditioning = queue.get()
image = diffusion_model(conditioning)
```

## 計測ベースライン

| 指標 | 値 |
|---|---|
| CPU→VRAM (10MB) | 0.76ms |
| CPU→VRAM (100MB) | 3.46ms / 30.3 GB/s |
| NPU推論 (512dim MLP, 静的形状) | 0.88ms |
| NPU出力 (2048B) → GPU | 0.031ms |
| 転送オーバーヘッド | 3.4% |

## 次のタスク

1. 実 LLM (Phi-3 mini INT4) を NPU で動かして本番レイテンシを計測
2. 拡散モデル (SDXL / SD3.5) を RTX5080 で動かす
3. threading + queue でパイプラインを接続

## コーディング規約

- ファイル名プレフィックス: `dollma_` (dollama のプロジェクト内ファイル)
- プローブスクリプトは `dollma_probe*.py`
- 本実装は `dollma_pipeline.py` 等に分離予定
- コメントは日本語で書く

# dollama

**NPU + GPU を使った二次元特化の軽量画像生成パイプライン**

Intel Core Ultra の NPU (AI Boost) でテキスト処理・LLM推論を行い、
NVIDIA GPU で拡散モデルを動かす。CPU 経由のパイプラインをマルチスレッドで構成することで、
NPU と GPU の処理を並列化する。

---

## ターゲット環境

| コンポーネント | 詳細 |
|---|---|
| CPU / NPU | Intel Core Ultra 9 285 (AI Boost / Intel AI NPU) |
| GPU | NVIDIA GeForce RTX 5080 |
| OS | Windows 11 |
| Python | 3.14 |

## アーキテクチャ

```
入力テキスト
    │
    ▼
[Thread-A: NPU]
  OpenVINO GenAI
  LLM (Phi-3 mini 等) INT4
  テキスト → conditioning vector
    │
    │  CPU pinned memory 経由 (0.031ms / ~2KB)
    ▼
[Thread-B: GPU]
  PyTorch / diffusers
  RTX5080 で拡散モデル実行
    │
    ▼
出力画像
```

### 設計判断

- **NPU→GPU ゼロコピーは採用しない**
  - CUDA Virtual Memory API (Win32ハンドル) は RTX5080 側で確保可能だが、
    OpenVINO NPU 側に CUDA ハンドルをインポートする API が存在しない
  - CPU 経由転送のオーバーヘッドは **3.4%** と計測済み → 問題なし

- **マルチスレッドで転送コストを隠蔽**
  - GPU の拡散ステップ (~数百ms) が NPU 推論 (~1ms〜数十ms) より圧倒的に遅い
  - Thread-A が次フレームを準備している間に Thread-B が現フレームを処理

- **NPU は静的形状のみ対応**
  - `ov_model.reshape([batch, seq_len])` でコンパイル前に形状を固定する必要あり

## 計測済みベースライン

| 指標 | 値 |
|---|---|
| CPU→VRAM (10MB) | 0.76ms / 13.9 GB/s |
| CPU→VRAM (100MB) | 3.46ms / 30.3 GB/s |
| NPU推論 (512dim MLP) | 0.88ms |
| NPU出力→GPU転送 | 0.031ms (3.4%) |

## 技術スタック

- **NPU推論**: OpenVINO / OpenVINO GenAI
- **GPU推論**: PyTorch + diffusers
- **パイプライン**: Python threading + queue

## セットアップ

```bash
pip install openvino openvino-genai
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install diffusers transformers accelerate
```

## ファイル構成

```
dollama/
  agents/
    base.py            # BaseAgent / GenerationRequest / GenerationResult
    director.py        # Claude API でユーザー意図を解析 (claude-opus-4-8)
    prompt_builder.py  # NPU (OpenVINO GenAI / Phi-3 mini INT4) → SD プロンプト生成
    style_selector.py  # アートスタイル・品質タグ・LoRA のルールベース選択
    image_generator.py # RTX5080 で SDXL を動かし画像を生成 (diffusers)
    quality_checker.py # CLIP で画像とプロンプトの一致度をスコアリング
  models/              # ローカルモデル置き場 (Git 管理外)
  outputs/             # 生成画像の出力先
  dollama_pipeline.py  # NPU/GPU 二スレッドパイプラインのエントリポイント
  dollma_probe.py      # 環境調査 STEP1-4
  dollma_probe2.py     # Win32ハンドル + NPU パイプライン調査 STEP5-7
  dollma_probe3.py     # D3D12 クロスアダプター調査 (不可と確認)
  README.md
  CLAUDE.md
```

## エージェント構成

```
ユーザー入力 (日本語/英語)
    │
    ├─ Thread-A (Claude API + NPU)
    │    DirectorAgent  ─ Claude claude-opus-4-8 で意図を構造化抽出
    │         ↓
    │    PromptBuilderAgent ─ NPU 上の Phi-3 mini INT4 で SD プロンプト生成
    │         ↓ queue.Queue(maxsize=2)
    │
    └─ Thread-B (CPU → GPU)
         StyleSelectorAgent  ─ 品質タグ・LoRA をルールで選択
              ↓
         ImageGeneratorAgent ─ RTX5080 で SDXL 推論 (fp16)
              ↓
         QualityCheckerAgent ─ CLIP スコアで品質判定
              ↓
         outputs/*.png
```

## 実行

```bash
# 環境変数
set ANTHROPIC_API_KEY=sk-ant-...

# モデルのダウンロード (初回)
optimum-cli export openvino \
    --model microsoft/Phi-3-mini-4k-instruct \
    --weight-format int4 models/phi3-mini-int4-npu

# 単発生成
python dollama_pipeline.py "ツインテールの魔法少女、夕焼けの中で戦っている"
```

# dollama

**[English](README.md)** | **日本語**

**CPU / NPU / iGPU / RTX5080 — 搭載する全 HW を使い切る二次元特化の画像生成パイプライン研究**

各ハードウェアの特性を活かして協調させることが目的。最短実装ではなく、最適な HW 割り当てを探る。
ML フレームワークに頼らず C++ でフルスクラッチ実装を目指す。

> ⚠️ **実装中 — 完成品ではなく研究中のプロジェクトです。**
> フル C++ 拡散パイプライン (自作 UNet + Euler スケジューラ + VAE decode) が **任意テキストから実画像 1024×1024 を生成**し
> (CLIP-G を加えた SDXL dual encoder + CFG・タスク 2-6b)、結果を **切り抜き済み透過 PNG** にマッティング (iGPU ISNet-anime) するようになりました。
> 速度は当初の 84s から **20steps 11.3s** へ短縮済み (im2col / Tensor Core GEMM + cuBLAS フォールバック・probe10 の 3.80s 比 約3倍)。
> 残るカーネル律速は UNet attention ですが、自作 CUDA の速度詰めはここで一旦打ち切り、研究の本丸を **自作タグ生成 LM (Phase 4)** に移しています。
> API・ファイル構成・計測値は今後変わります。**研究プロセスの公開**を目的としており、すぐ使えるツールではありません。

> **本研究は Intel 環境 (Core Ultra 9 285 — Intel AI Boost NPU + Intel Xe iGPU) と
> NVIDIA RTX5080 を組み合わせた構成で行っている。** NPU / iGPU の活用 (OpenVINO 経由) は
> Intel プラットフォーム固有の特性に依存しており、計測値・デバイス選定はこの環境を前提とする。

---

## 概要

dollama は、1 台のマシンに載った **CPU・NPU・iGPU・dGPU を 1 つの推論パイプラインとして協調**させ、
2D イラスト/漫画の**キャラクター**画像 (背景なし・切り抜き済み透過 PNG) を生成する研究プロジェクト。
PyTorch / diffusers などの ML フレームワークに頼らず、推論カーネルから HTTP サーバまで **C++ でフルスクラッチ実装**する。

**コア・アイデア — "時間方向の協調":** HW を直結 (ゼロコピー) でつなぐのではなく、
GPU が拡散処理に数秒かける間に、遊休している NPU/CPU で次のリクエストの CLIP 埋め込みやタグ抽出を
並列で回す。各 HW を**得意なタスクに割り当て**、待ち時間を互いに埋め合う。

```
[CPU] LLM (プロンプト生成) / WD14 タグ抽出
[NPU] CLIP text encoder (固定形状に最速)
[iGPU] VAE encode (img2img 入力、CPU と並列)
[RTX5080] SDXL UNet 20steps + VAE decode (生成本体)
```

**現在の進捗:**

| フェーズ | 内容 | 状態 |
|---|---|---|
| Phase 1 | C++ パイプライン骨格 (Tensor/Allocator/Queue/CLIP-NPU/WD14-CPU/スレッド骨格) | ✅ 完了 (9.13 frames/s) |
| Phase 2 | 自作 CUDA カーネル + safetensors + VAE decode + SDXL UNet + Euler + フル拡散パイプライン結線 | ✅ 完了 (実画像 1024² を生成・**20steps 11.3s** へ最適化) |
| Phase 2-6b | 任意テキスト → 画像 (CLIP-G を加えた SDXL dual encoder + CFG + negative prompt) + マッティング → 透過 PNG | ✅ 完了 (prompt → 実画像 1024² 透過 PNG を一気通貫) |
| Phase 3 | OpenAI 互換 HTTP サーバ (cpp-httplib / nlohmann-json) | ✅ 完了 (PipelineGenerator を DI、フォールバック付き) |
| Phase 4 | 自作タグ生成 LM (bitnet.hpp 33M) + 同一性条件付け/品質スコアラ。ternary は圧縮実験 | ⏳ 進行中 — dense LM が C++ で訓練・推論 (CPU/GPU・golden corr 1.0・GPU 87.5x・INT8 / AVX2 経路)、同一性条件付け (A) はクローズ (retention 0.975)、入力多様化 (B) は本線昇格、品質スコアラ (Model B) は解剖 8 軸で蒸留済 |

> **次の本丸**: 研究の焦点は **自作タグ生成 LM (Phase 4)** — 暫定 Qwen2 段を 33M フルスクラッチモデル
> (自然文 → danbooru タグ・同一性条件付き) に置き換え、Model B のアニメ品質フィードバックループを閉じる。
> 出力品質への最大レバーは、より新しいアニメ特化 SDXL checkpoint への差し替え (カーネル無改修)。

> 詳細なロードマップは [`docs/roadmap.md`](docs/roadmap.md)、HW 役割の根拠は下記「何が分かったか」を参照。

---

## 何が分かったか (probe 結果サマリ)

10 本の probe と Phase 2 の自作カーネル実装で得た要点。**数字より結論を先に**:

1. **HW 選定は「タスク × アーキテクチャの相性」で決まる — 事前予想は当てにならない。**
   NPU は固定形状の encoder (CLIP-L 77token) では CPU の 2.5 倍速 (7.85ms) で最速。
   一方、同じ "推論" でも WD14 SwinV2 (Window Attention) では NPU が **最遅** (268ms、CPU 101ms)、
   LLM の自己回帰 (動的 KV-cache) は**コンパイルすら通らない**。「NPU = 速い」ではなく
   「固定形状の純粋 GEMM チェーンだけ速い」。→ 全モデルを実測してから割り当てた。

2. **HW 間のゼロコピー直結は (この構成では) 不可能。CPU pinned 経由が現実解。**
   CUDA↔NPU の interop API は存在せず、iGPU は BIOS 設定で DXGI に非表示 (D3D12 クロスアダプター不可)。
   残った CPU pinned memory 経由の転送オーバーヘッドは **3.4%** で、GPU 拡散処理に対し
   マルチスレッドで完全に隠蔽できる → 直結に固執する必要はなかった。

3. **iGPU は "方向" で使い分ける。** 大規模 Conv (VAE decode) は CPU の 8 倍遅く使い物にならないが、
   VAE **encode** (img2img) では CPU 117ms に対し **79ms** と速い。しかもこれは CPU の LLM 生成 (~2s) と
   並列に走るため img2img パスでは待ち時間ゼロ。

4. **RTX5080 (Blackwell/sm_120) は SDXL を 3.80s/枚 (1024², 20steps) でこなし、VRAM ピークは 10.49GB**。
   16GB 中なので長辺 ~1536px まで余裕。生成本体は GPU に集約し、NPU/CPU は生成中の遊休時間に
   CLIP・タグ抽出を並列で回す構成が成立する。

5. **ML フレームワーク無しの自作 CUDA カーネルでフル拡散パイプラインが動き、実画像が出た。**
   Phase 2 で GEMM / 活性化 / GroupNorm / Conv2d / Attention をフルスクラッチ実装し、
   CPU 参照とのゴールデンテストで正当性を確認 (GEMM 4730 GFLOPS / Conv2d 1807 GFLOPS / Attention 1631 GFLOPS)。
   その上で自作 VAE decode (final SSIM 0.999992) と自作 SDXL UNet (noise_pred SSIM 0.999998) を実装し、
   Euler スケジューラと結線して **20steps で実画像 1024² を生成 (84s)**。正しさは出た — 速度 (direct conv +
   naive attention 律速で probe10 比 22倍) が次の課題で、Tensor Core / flash 化が本丸。

→ 結論: **「全 HW を使い切る」は理想論ではなく、実測に基づけば成立する。** 鍵は直結ではなく
"各 HW を得意タスクに割り当て、GPU 生成中の遊休を NPU/CPU で埋める" 時間方向の協調にある。

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
| **CPU** | LLM プロンプト生成 (暫定 Qwen2-1.5B → 将来 自作タグ生成 LM bitnet.hpp) | 64-71 tok/s |
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
[CPU] Qwen2-1.5B (暫定) / 将来: 自作タグ生成 LM (bitnet.hpp 33M, GPU 主・CUDA カーネル流用 / CPU 可・NPU 不可)
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

## LLM の将来像 — 自作タグ生成 LM

汎用 LLM (Qwen2-1.5B, 873MB, CPU ~2s) を目的特化の超軽量モデルに置き換える。
核は `src/models/bitnet.hpp` (decoder-only LLaMA 系 **33M**, モデル定義+ホスト参照は実装済)。
語彙はタグ単位完全一致トークナイザ `src/io/tokenizer.hpp` (vocab.json 駆動・実装済, Phase 4-3) が担う。

| | Qwen2-1.5B (現状) | 自作タグ生成 LM (目標) |
|---|---|---|
| パラメータ | 1.5B | 33M (30-100M) |
| サイズ | 873MB | ~66MB (FP16) / ~20MB (ternary 実験時) |
| デバイス | CPU | GPU 主 (CUDA カーネル流用) / CPU 可・NPU 不可 |
| レイテンシ | ~2s | 目標 <10ms |
| タスク | 汎用 | user text → danbooru タグ特化 |

訓練データ: Danbooru タグ共起 + Qwen2 / DanTagGen 蒸留 (先行実装 DanTagGen 400M / TIPO を教師・品質基準に)

**ternary (b1.58) は圧縮の研究軸** (目的ではない): まず FP16/INT8 dense で品質を出し、
重み W ∈ {-1,0,+1} (≈1.58bit, `y = x_pos − x_neg` で乗算不要) は後段の圧縮実験として
被せる (`src/kernels/ternary_gemm.cu`)。33M 規模では旨味は限定的。

**2D 特化の独自性**: ① キャラ同一性条件付きタグ生成 (character-bible 入力・DanTagGen に無い)
② アニメ品質スコアラ (NPU・固定形状)。詳細は `docs/roadmap.md` Phase 4。

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

実装済み (✅) と計画中 (⏳) を区別して記載 (実装中のため随時変化する):

```
src/
├── core/
│   ├── tensor.hpp        ✅ 独自 Tensor (STL ベース)
│   ├── allocator.hpp     ✅ CPU / pinned / VRAM メモリ管理
│   ├── queue.hpp         ✅ SPSC ロックフリーキュー
│   ├── character.hpp     ✅ キャラ台帳 (CharacterBible) + プロンプト合成 + カラーモード
│   └── affinity.hpp      ✅ CPU コアアフィニティ
├── infer/
│   ├── clip.hpp          ✅ CLIP-L text encoder (NPU / OpenVINO)
│   ├── wd14.hpp          ✅ WD14 SwinV2 tagger (CPU / OpenVINO)
│   ├── scheduler.hpp     ✅ EulerDiscreteScheduler (SDXL)
│   ├── unet.cu/.cuh      ✅ SDXL UNet 全段 (noise_pred SSIM 0.999998)
│   └── diffusion.cu/.cuh ✅ フル拡散ループ結線 (UNet×Nstep + Euler + VAE)
├── kernels/              ✅ 自作 CUDA カーネル (Phase 2)
│   ├── gemm.cu           ✅ dense FP16 GEMM
│   ├── activation.cu     ✅ SiLU / GeLU
│   ├── groupnorm.cu      ✅ GroupNorm
│   ├── conv2d.cu         ✅ direct Conv2d
│   ├── attention.cu      ✅ scaled dot-product attention (self / cross)
│   ├── vae_decode.cu     ✅ SDXL VAE decode (final SSIM 0.999992)
│   └── ternary_gemm.cu   ⏳ ternary GEMM 圧縮実験 (Phase 4)
├── io/
│   ├── safetensors.hpp   ✅ safetensors 重みローダー (golden 突合)
│   ├── png_meta.hpp      ✅ PNG キャラ設定メタ往復 (tEXt 焼き込み)
│   └── tokenizer.hpp     ✅ タグ単位完全一致トークナイザ (vocab.json 駆動, Phase 4-3)
├── server/              ✅ cpp-httplib + OpenAI 互換 API (Phase 3)
│   ├── api.cpp           ✅ /v1/images/generations 他
│   ├── png.hpp           ✅ PNG エンコード
│   ├── stub_generator.hpp        ✅ ダミー生成器 (フォールバック)
│   └── pipeline_generator.hpp    ✅ 本生成器 (DiffusionPipeline を IImageGenerator 化)
├── pipeline.hpp          ✅ マルチスレッド骨格
├── main.cpp              ✅ エントリポイント (生成器をフォールバック付き DI)
└── models/bitnet.hpp     ✅ タグ生成 LM 定義+ホスト参照 (Phase 4-2) / 推論 ⏳
```

**使うもの**: STL / CUDA API / Winsock2 (HTTP/JSON はヘッダオンリーの定番ライブラリ)  
**使わないもの**: PyTorch / OpenVINO (probe のみ) / diffusers / llama.cpp 等の ML フレームワーク

---

## ビルド (C++)

> 現状ビルドできるもの: core・CLIP(NPU)/WD14(CPU) 推論グルー・スレッド骨格・Phase 2 CUDA カーネル群・
> 自作 VAE decode / SDXL UNet / Euler スケジューラ・フル拡散パイプライン・OpenAI 互換 HTTP サーバ、と各テスト。
> **golden 埋め込みからの end-to-end 画像生成は動きます** (20steps 84s)。任意テキストからの生成は 2-6b で結線予定。

### 一括インストーラー (Windows・推奨)

まっさらな Windows 機なら、必要なライブラリ一式 (VS Build Tools / CUDA / OpenVINO / Python 依存) を
winget + pip で一括導入できます:

```powershell
powershell -ExecutionPolicy Bypass -File install_windows.ps1
```

- NVIDIA dGPU 検出時のみ CUDA Toolkit を導入し torch=cu128。無ければ自動でスキップし torch=CPU・`-Dwith_cuda=false` を案内します。
- `-DryRun` (実行せずコマンド表示) / `-CheckOnly` (存在チェックのみ) / `-SkipCuda` / `-SkipSdk` / `-PythonExe <path>` / `-Force` を指定可能。
- 最後に環境に合わせた **推奨 `meson setup` コマンド**を出力します。

手動で揃える場合は以下の前提・手順に従ってください。

**前提**

- [Meson](https://mesonbuild.com/) + Ninja、C++20 コンパイラ (Windows 11 + MSVC で検証)
- **CUDA Toolkit 13.x** (`nvcc`) — RTX5080 は `sm_120` (CUDA 12.8+) が必須。`nvcc` を `PATH` に通す。
- **OpenVINO 2024.x+** ランタイム — CLIP/WD14 の NPU パス用 (任意、`-Dwith_openvino` で切替)

**ビルド & テスト**

```bash
# 構成 (各ツールキットのパスを自環境に合わせる)
meson setup build \
  -Dwith_cuda=true     -Dgpu_sdk_root="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3" \
  -Dwith_openvino=true -Dnpu_sdk_root="C:/Program Files (x86)/Intel/openvino_2024"

meson compile -C build
meson test -C build            # 全単体テスト + カーネルのゴールデンテスト/ベンチを実行
```

`-Dwith_cuda=false` で CUDA を無効化できる (その場合 `.cu` テストはスキップ)。
テスト内訳は [`docs/testing.md`](docs/testing.md) を参照。

---

## モデル

モデル重みは本リポジトリに **含まれません** (`models/` は Git 管理外)。CLIP / WD14 / SDXL を動かすには
各自で取得・変換が必要:

- **CLIP-L text encoder** / **WD14 SwinV2 tagger** → OpenVINO IR (NPU / CPU) に `scripts/` の probe スクリプト
  (`optimum-intel` / OpenVINO 経由) で変換。
- 暫定 LLM の **Qwen2-1.5B (INT4)**。
- 拡散段の **SDXL** (自作 CUDA カーネルで画像生成可能。任意テキストからの本結線は 2-6b)。

本リポジトリの **コードは Apache-2.0** だが、各モデルの **重みは配布元それぞれのライセンスに従う**。
利用者がその条件の遵守に責任を負う。

### 第三者モデルのライセンス表記

| モデル | 役割 | 配布元ライセンス |
|---|---|---|
| SDXL | 拡散 (UNet + VAE) | Stability AI Community / CreativeML OpenRAIL-M (checkpoint による) |
| CLIP-L / CLIP-bigG | テキストエンコーダ (NPU) | MIT (OpenAI CLIP) / OpenCLIP |
| WD14 SwinV2 tagger v3 | タグ抽出 (CPU) | Apache-2.0 |
| Qwen2-1.5B | 暫定 LLM プロンプト段 | Apache-2.0 |
| ISNet-anime (anime-segmentation) | マッティング / 透過 PNG (iGPU) | Apache-2.0 |
| TIPO-200M (KBlueLeaf) | 蒸留教師の実験 (4-D6・本番不採用) | Apache-2.0 |
| **Eugeoter/waifu-scorer-v4-beta** | **Model B 品質スコアラ — 美的教師 (既定・ラベル生成のみ)** | **Apache-2.0** |
| deepghs/anime_aesthetic | Model B 品質スコアラ — 代替の美的教師 (切替可) | OpenRAIL |

> **Model B 美的教師について:** 既定教師 (waifu-scorer-v4-beta) は Apache-2.0。切替可能な代替
> (deepghs/anime_aesthetic) は OpenRAIL で、商用利用・再配布・改変を許可し、制限は付属書の行動ベース
> 禁止条項のみ (違法 / 差別 / 加害的用途の禁止)。いずれの場合も dollama はモデルを **教師として soft
> ラベルを生成する用途のみ**に使い、その出力を自作 ScorerNet へ一から蒸留する。教師モデルの重み自体は
> 本プロジェクトでは **再配布しない**。

---

## セットアップ (調査フェーズ / Python probe)

Windows なら上記の `install_windows.ps1` が以下の pip 依存もまとめて導入します
(`requirements.txt` 基準・torch は GPU 有無で index 自動分岐)。手動で入れる場合:


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
    roadmap.md                 # 実装ロードマップ (Phase 1-4)
    character-bible-spec.md    # キャラ設定データ構造 (同一性/シーン/出力層・カラーモード)
    http-api-spec.md           # OpenAI 互換 HTTP API 仕様
    pipeline-spec.md           # パイプライン構成
    tensor-spec.md             # Tensor 仕様
    cpu-topology.md            # CPU コアトポロジ / アフィニティ
    testing.md                 # テスト規約
    archives/
      investigation-log.md     # probe1-10 の詳細調査ログ
  CLAUDE.md                    # Claude Code 向けコンテキスト
```

---

## ライセンス

[Apache License 2.0](LICENSE) の下で公開。詳細は [`LICENSE`](LICENSE) を参照。

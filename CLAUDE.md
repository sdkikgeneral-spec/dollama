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
CPU: Qwen2-1.5B (暫定) / 将来: 自作 BitNet b1.58 on NPU
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
| SDXL 20steps 1024×1024 RTX5080 | **3.80s** / 5.3 it/s / VRAM ピーク 10.49GB | probe10 |
| compose_prompt (C++ CharacterBible, 1M iters) | **242 ns/op** | test_character |
| CharacterBible::find (10,000体, 1M lookups) | **10.5 ns/op** | test_character |

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
| **6** | **WD14 CPU 推論 + テスト** | **`src/infer/wd14.hpp`, `test_wd14.cpp`** | **⏳ 次** |
| 7 | スレッド骨格 + CPU アフィニティ | `src/main.cpp` 拡張 | ⏳ 未着手 |

**Phase 2 以降 (詳細は `docs/roadmap.md` 参照)**
- `src/kernels/ternary_gemm.cu` — BitNet ternary GEMM CUDA カーネル
- `src/server/http.cpp` — Winsock2 OpenAI 互換 HTTP サーバー
- 自作 BitNet b1.58 の訓練データ収集・学習

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

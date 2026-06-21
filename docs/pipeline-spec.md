# パイプライン仕様 — dollama C++ 実装

## 最終アーキテクチャ

4つの HW (CPU / NPU / iGPU / RTX5080) をスレッドで並列動作させ、
txt2img / img2img の推論パイプラインを実現する。

```
┌─────────────────────────────────────────────────────────────┐
│  Thread: llm_thread  (CPU — P-core, mask=0x00C03C03)        │
│  自作タグ生成 LM: user text → danbooru tags (<10ms)         │
│         ↓  tags (string)                                     │
│         put → llm_to_clip_queue                             │
└─────────────────────────────────────────────────────────────┘
         ↓ SPSC queue (pinned memory)
┌─────────────────────────────────────────────────────────────┐
│  Thread: clip_thread  (NPU — OpenVINO C++ API)              │
│  CLIP-L text encoder: tags [1,77] int32 → embedding [1,77,768] │
│  計測: 7.85ms / NPU (probe9)                                 │
│         ↓  embedding (float32, ~231KB, pinned memory)        │
│         put → clip_to_sdxl_queue                            │
└─────────────────────────────────────────────────────────────┘
         ↓ SPSC queue (pinned memory → VRAM transfer)
┌─────────────────────────────────────────────────────────────┐
│  Thread: sdxl_thread  (RTX5080 — 自作 CUDA カーネル)        │
│  UNet × 20steps + VAE decode                                │
│  計測目標: ≦3.80s / 1024×1024 (probe10 baseline)           │
│         ↓  image (RGB, 1024×1024, ~12MB, pinned memory)     │
│         put → sdxl_to_wd14_queue                            │
└─────────────────────────────────────────────────────────────┘
         ↓ SPSC queue (pinned memory)
┌─────────────────────────────────────────────────────────────┐
│  Thread: wd14_thread  (CPU — E-core, mask=0x003FC3FC)        │
│  WD14 SwinV2 tagger (OpenVINO C++ API): image → tags        │
│  計測: 101ms / CPU (probe8)                                  │
│         ↓  tags → llm_thread へフィードバック                │
└─────────────────────────────────────────────────────────────┘
```

### img2img 追加スレッド

```
┌─────────────────────────────────────────────────────────────┐
│  Thread: vae_enc_thread  (iGPU — OpenVINO GPU.0)            │
│  VAE encode: input image 1024×1024 → latent [1,4,128,128]   │
│  計測: 79ms / iGPU (probe5) — CPU 117ms より速い            │
│  CPU LLM と並列実行するため待ち時間ゼロ                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 実装フェーズ

Python プロトタイプは作らず直接 C++ に入る。
各 HW の計測は probe1〜11 で完了済み。

### Phase 1 — キュー + 推論パーツ単体 (即着手可能)

OpenVINO C++ API で動くパーツから実装する。SPSC キューが全体の骨格。

| 実装物 | ファイル | 依存 |
|---|---|---|
| SPSC lock-free queue | `src/core/queue.hpp` | STL のみ |
| CLIP NPU 推論 | `src/infer/clip.hpp` | OpenVINO C++ |
| WD14 CPU 推論 | `src/infer/wd14.hpp` | OpenVINO C++ |
| スレッド骨格 + アフィニティ | `src/pipeline.hpp` + `src/core/affinity.hpp` | STL thread (各ワーカーが起動時に自己ピン留め) |

LLM スレッドはこの時点で **スタブ** (ユーザー入力をそのまま tags として渡す)。
Qwen2-1.5B は Python probe 専用で C++ に持ち込まない (LibTorch 不使用方針)。

**縦通し確認**: `stub → CLIP(NPU) → queue → WD14(CPU)` で推論ループが回ること。

### Phase 2 — SDXL 自作 CUDA カーネル

最大の実装物。CUDA カーネルをゼロから書く。

| 実装物 | ファイル | 備考 |
|---|---|---|
| CUDA エラーチェック・ユーティリティ | `src/kernels/utils.cuh` | CUDA_CHECK マクロ |
| ternary GEMM (BitNet 向け基礎) | `src/kernels/ternary_gemm.cu` | `cuda-kernel-dev` 担当 |
| Transformer Attention | `src/kernels/attention.cu` | |
| VAE decode | `src/kernels/vae_decode.cu` | |
| UNet スケジューラ | `src/infer/unet.hpp` | |

モデル重みのロード: GGUF or safetensors を独自パーサーで読む。

### Phase 4 — 自作タグ生成 LM (旧 BitNet b1.58)

| 実装物 | 備考 |
|---|---|
| 訓練データ収集 (user text → danbooru tags) | ✅ #1 完了 |
| モデル定義 30-100M params (bitnet.hpp 33M) | ✅ #2 完了 |
| トークナイザ (タグ単位完全一致, tokenizer.hpp) | ✅ #3 完了 |
| GPU 推論 (自己回帰・CUDA カーネル流用が第一) / CPU 代替 | NPU は自己回帰不可で除外。NPU 化は非自己回帰版が要る・未確定 |
| ternary GEMM カーネル | dense が動いた後の圧縮実験 (目的ではない) |
| 同一性条件付け / アニメ品質スコアラ | 2D 特化の独自スコープ (A/B) |

---

## スレッドアフィニティ (probe11 計測値)

| スレッド | コア種別 | マスク (Windows) | 理由 |
|---|---|---|---|
| llm_thread | P-core | `0x00C03C03` | 重い逐次処理 |
| sdxl_thread | P-core | `0x00C03C03` | GPU 同期のレイテンシ優先 |
| wd14_thread | E-core | `0x003FC3FC` | GPU 生成中の並列処理 |
| clip_thread / vae_enc_thread | E-core | `0x003FC3FC` | 軽量・短時間 |

詳細なビットパターンは `docs/cpu-topology.md` を参照。

---

## キュー設計

### 基本方針

- SPSC (Single-Producer Single-Consumer) lock-free ring buffer
- データはすべて **CPU pinned memory** 経由 (転送オーバーヘッド 3.4%、GPU処理で隠蔽可能)
- backpressure: capacity=2 で生産者をブロック

### キュー一覧

| キュー名 | 生産者 | 消費者 | データ | capacity |
|---|---|---|---|---|
| `llm_to_clip_queue` | llm_thread | clip_thread | トークン列 `[1,77]` int32 (308B) | 2 |
| `clip_to_sdxl_queue` | clip_thread | sdxl_thread | embedding `[1,77,768]` float32 (~231KB) | 2 |
| `sdxl_to_wd14_queue` | sdxl_thread | wd14_thread | image `[3,1024,1024]` uint8 (~3MB) | 1 |
| `wd14_to_llm_queue` | wd14_thread | llm_thread | タグ文字列 (1KB) | 4 |
| `vae_enc_queue` (img2img) | vae_enc_thread | sdxl_thread | latent `[1,4,128,128]` float16 (~128KB) | 1 |

### C++ インターフェース (`src/core/queue.hpp`)

```cpp
template<typename T, size_t Capacity>
class SPSCQueue
{
public:
    bool push(T item);   // 満杯なら false (ノンブロッキング)
    bool pop(T& item);   // 空なら false (ノンブロッキング)
    bool push_wait(T item, std::chrono::milliseconds timeout);
    bool pop_wait(T& item, std::chrono::milliseconds timeout);
    size_t size() const noexcept;
private:
    std::array<T, Capacity> buf_;
    alignas(64) std::atomic<size_t> head_{0};
    alignas(64) std::atomic<size_t> tail_{0};
};
```

---

## データ転送フロー

```
CPU pinned memory (embedding ~231KB)
  ↓ cudaMemcpyAsync  0.030ms (probe4)
RTX5080 VRAM
  ↓ UNet 20steps     3.80s   (probe10 baseline)
RTX5080 VRAM
  ↓ cudaMemcpyAsync  0.254ms / 12MB (probe4)
CPU pinned memory (image ~12MB)
```

## タイミング試算 (txt2img, 定常状態)

```
フレーム N:
  CPU(stub/LLM) |──── <10ms or 2s ────|
  NPU(CLIP)                             |8ms|
  GPU(SDXL)                                  |──── 3.80s ────|
  CPU(WD14)                                                   |101ms|

フレーム N+1:
  CPU(LLM) は GPU(3.80s) の中に収まる → 実効レイテンシ ≈ 3.80s / image
```

## エラーハンドリング

- 各スレッドは例外をキャッチしてシャットダウンフラグ (`std::atomic<bool>`) を立てる
- メインスレッドがフラグを監視し、全スレッドを join して終了
- CUDA エラー: `CUDA_CHECK` マクロで即 throw (`src/kernels/utils.cuh`)
- OpenVINO エラー: `try/catch (ov::Exception&)` でラップ

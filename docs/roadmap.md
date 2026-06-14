# dollama 実装ロードマップ

## Phase 1 — パイプライン骨格 (現在)

OpenVINO C++ API で動くパーツから順に実装し、スレッド骨格を完成させる。
SDXL / BitNet は含まない。

| # | 実装物 | ファイル | 状態 |
|---|---|---|---|
| 1 | Tensor クラス | `src/core/tensor.hpp` | ✅ 完了 |
| 2 | メモリアロケーター | `src/core/allocator.hpp` | ✅ 完了 |
| 3 | SPSC キュー | `src/core/queue.hpp` | ✅ 完了 |
| 4 | CLIP NPU 推論 | `src/infer/clip.hpp` | ⏳ 未着手 |
| 5 | WD14 CPU 推論 | `src/infer/wd14.hpp` | ⏳ 未着手 |
| 6 | スレッド骨格 + CPU アフィニティ | `src/main.cpp` 拡張 | ⏳ 未着手 |

**Phase 1 完了の定義**: stub (LLM なし) → CLIP(NPU) → queue → WD14(CPU) のループが
マルチスレッドで回り、タグ文字列が出力されること。

---

## Phase 2 — SDXL 自作 CUDA カーネル

最大の実装物。CUDA カーネルをゼロから書き、diffusers なしで画像生成を実現する。

| # | 実装物 | ファイル | 状態 |
|---|---|---|---|
| 1 | CUDA エラーユーティリティ | `src/kernels/utils.cuh` | ⏳ 未着手 |
| 2 | Ternary GEMM (BitNet 基礎) | `src/kernels/ternary_gemm.cu` | ⏳ 未着手 |
| 3 | Multi-Head Attention | `src/kernels/attention.cu` | ⏳ 未着手 |
| 4 | VAE decode | `src/kernels/vae_decode.cu` | ⏳ 未着手 |
| 5 | SDXL UNet 推論 + スケジューラ | `src/infer/unet.hpp` | ⏳ 未着手 |
| 6 | モデル重みローダー (safetensors) | `src/io/safetensors.hpp` | ⏳ 未着手 |

**Phase 2 完了の定義**: フル C++ パイプラインで 1024×1024 画像が生成されること。
目標: probe10 ベースライン (3.80s / 20steps) と同等以上。

---

## Phase 3 — HTTP サーバー

外部クライアント (WebUI 等) から呼べるようにする。

| # | 実装物 | ファイル | 状態 |
|---|---|---|---|
| 1 | Winsock2 HTTP サーバー | `src/server/http.cpp` | ⏳ 未着手 |
| 2 | Base64 エンコーダ | `src/server/base64.hpp` | ⏳ 未着手 |
| 3 | JSON パーサー (最小実装) | `src/server/json.hpp` | ⏳ 未着手 |

API 仕様: `docs/http-api-spec.md` 参照。
`POST /v1/images/generations` で OpenAI Images API 互換。

**Phase 3 完了の定義**: `curl` で叩いて PNG (base64) が返ってくること。

---

## Phase 4 — 自作 BitNet b1.58 LLM

現在の LLM stub (またはQwen2 Python) を自作モデルに置き換える。

| # | 実装物 | ファイル / 作業 | 状態 |
|---|---|---|---|
| 1 | 訓練データ収集 | user text → danbooru tags ペア | ⏳ 未着手 |
| 2 | モデル定義 (30-100M params) | `src/models/bitnet.hpp` | ⏳ 未着手 |
| 3 | BPE トークナイザー | `src/io/tokenizer.hpp` | ⏳ 未着手 |
| 4 | 訓練スクリプト (Python) | `scripts/train_bitnet.py` | ⏳ 未着手 |
| 5 | C++ 推論 (ternary GEMM 流用) | `src/infer/bitnet.hpp` | ⏳ 未着手 |

**Phase 4 完了の定義**: user text → danbooru タグ変換が C++ で動き、
Qwen2 Python に対して遜色ない品質であること。目標レイテンシ: <10ms (BitNet b1.58)。

---

## マイルストーン一覧

| マイルストーン | 内容 | 目標 |
|---|---|---|
| **M1** | Phase 1 完了: C++ でタグ抽出ループが動く | Phase 1 全完了後 |
| **M2** | Phase 2 完了: フル C++ で画像生成 | SDXL カーネル完成後 |
| **M3** | Phase 3 完了: HTTP 経由で画像生成 | サーバー完成後 |
| **M4** | Phase 4 完了: フル自作スタックで end-to-end | BitNet 訓練後 |

---

## 技術的リスク・未確定事項

| 項目 | リスク | 対策 |
|---|---|---|
| SDXL UNet 自作カーネル | 実装規模が最大・デバッグが困難 | 小さいモデル (64×64 latent) で動作確認してからスケール |
| BitNet b1.58 訓練データ | quality / quantity が未確定 | Danbooru + Qwen2 蒸留で初期データを確保 |
| safetensors パーサー | バイナリ仕様の正確な実装が必要 | 既存仕様書とテストファイルで検証 |
| Linux 対応 (Winsock2 → POSIX) | 二重実装が必要 | `#ifdef _WIN32` 分岐で吸収 |

---

## 参照ドキュメント

- `docs/pipeline-spec.md` — スレッド構成・キュー設計・タイミング試算
- `docs/tensor-spec.md` — Tensor クラス詳細設計
- `docs/http-api-spec.md` — HTTP API 仕様
- `docs/cpu-topology.md` — CPU コアアフィニティ設定
- `docs/archives/investigation-log.md` — probe1〜10 調査ログ

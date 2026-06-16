# dollama 実装ロードマップ

**スコープ (確定)**: 生成対象は**キャラクターのみ**。背景は外部 (Grok/Gemini/SD) +
CLIP Studio で合成し、出力は**切り抜き済み透過 PNG**。キャラ設定構造・切り抜き・
手指品質・学習ループの設計は `docs/character-bible-spec.md` を参照。

## Phase 1 — パイプライン骨格 (現在)

OpenVINO C++ API で動くパーツから順に実装し、スレッド骨格を完成させる。
SDXL / BitNet は含まない。

| # | 実装物 | ファイル | 状態 |
|---|---|---|---|
| 1 | Tensor クラス | `src/core/tensor.hpp` | ✅ 完了 |
| 2 | メモリアロケーター | `src/core/allocator.hpp` | ✅ 完了 |
| 3 | SPSC キュー | `src/core/queue.hpp` | ✅ 完了 |
| 4 | CLIP NPU 推論 | `src/infer/clip.hpp` | ✅ 完了 (NPU 7.82ms) |
| 5 | キャラ台帳 (CharacterBible, authored 層) | `src/core/character.hpp` | ✅ 完了 |
| 6 | WD14 CPU 推論 | `src/infer/wd14.hpp` | ✅ 完了 (CPU 105ms) |
| 7 | スレッド骨格 + CPU アフィニティ | `src/main.cpp` + `src/core/affinity.hpp` + `src/pipeline.hpp` | ✅ 完了 (9.13 frames/s, WD14 律速) |

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
**配管は自作しない**: HTTP/JSON は定番のヘッダオンリーライブラリを使う
(自作は HW 研究コアに限定、CLAUDE.md「実装方針」参照)。

| # | 実装物 | ライブラリ / ファイル | 状態 |
|---|---|---|---|
| 1 | HTTP サーバー | **cpp-httplib** (単一ヘッダ・Winsock2/POSIX 吸収) | ⏳ 未着手 |
| 2 | JSON 入出力 | **nlohmann/json** (ヘッダオンリー) | ⏳ 未着手 |
| 3 | エンドポイント実装 | `src/server/api.cpp` (上記2ライブラリを使用) | ⏳ 未着手 |
| 4 | Base64 (PNG 返却用) | httplib 付属 or 数十行の小物 | ⏳ 未着手 |

依存はヘッダオンリーのみ採用 → 単一バイナリ・重量級フレームワーク不使用の方針は維持。
meson subproject (wrap) で取り込む。API 仕様: `docs/http-api-spec.md` 参照。
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

## キャラクター品質・一貫性 (画像生成後の段、Phase 2+ で並行)

キャラを「コマ間でブレさせない」「手指を崩さない」ための段。設計は
`docs/character-bible-spec.md` 参照。authored 層 (character.hpp) は完了済みで、
以下は learned 層・後処理段として段階的に実装する。

| 項目 | 内容 | spec | 時期 |
|---|---|---|---|
| 切り抜き (マッティング) | 透過 PNG 出力。anime-segmentation (isnet 系)。乗せる HW は probe 比較 | §3, §9 | Phase 2 |
| 手指 L1 (予防) | 品質ネガティブ注入 (`default_quality_negatives`) | §10 | ✅ 完了 (器) |
| 手指 L2/L3 (修復・検査) | 手検出→インペイント再生成 / 指数を `digits_per_hand` と照合し再生成 | §10 | Phase 2 |
| 学習層 `CharacterMemory` | 生成→学習→FB ループ。記憶層 (seed/pose 蓄積・重心) → 蒸留 QA スコアラ (NPU) → fine-tune | §11 | Phase 2/3 |
| 背景プラグイン | 外部背景生成 (Grok/Gemini/SD) + 自動合成。宿主は HTTP サーバ層 | §9 | Phase 3 |

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
| Linux 対応 (HTTP) | ~~Winsock2/POSIX 二重実装~~ | cpp-httplib がクロスプラットフォーム吸収 → 解消 |

---

## 参照ドキュメント

- `docs/pipeline-spec.md` — スレッド構成・キュー設計・タイミング試算
- `docs/tensor-spec.md` — Tensor クラス詳細設計
- `docs/http-api-spec.md` — HTTP API 仕様
- `docs/cpu-topology.md` — CPU コアアフィニティ設定
- `docs/archives/investigation-log.md` — probe1〜10 調査ログ

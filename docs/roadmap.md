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

## 将来の探索テーマ (バックログ)

ロードマップ本線には載せないが、研究価値があり時期未定の項目。

| テーマ | 概要 | 評価 |
|---|---|---|
| **MoE × HW 分散配置** | MoE のエキスパートは独立計算できるため、NPU/iGPU/CPU/RTX5080 に**エキスパートを分散配置**し協調効率を研究する。dollama の芯 (全 HW 協調) に直結する固有テーマ。**◎ 価値高** | 時期未定 |
| 拡散 UNet の timestep-expert | eDiff-I / DiT-MoE 系。20step を「初期=構図 / 後期=ディテール」で別エキスパートに分割。Phase 2 で dense を動かした後に検討。 | ◯ |
| タグ生成 LLM の MoE 化 | 単体では 30–100M 規模に対し過剰 (MoE は数B〜で真価)。ルーティング損失・分岐コストが見合わない。 | △ 過剰 |
| **NPU 骨格/部位検出による解剖メタ整合検査** | 生成画像を NPU で部位検出し、**「数・位相」だけ**を `CharacterIdentity` の宣言値と照合 (指数/四肢の本数・有無/重複欠損/左右の本数対称)。**角度・比率・ポーズ自然さは見ない** (2D のパース・デフォルメで誤検出するため)。L3 指数検査の一般化。NPU は拡散中 3.8s ほぼ遊休 → 裏で実質ゼロコスト採点。詳細は character-bible-spec §11。 | **◎ 価値高 (Phase 2)** |

**共通の制約**: NPU は静的形状のみ → 古典的な token-level dynamic routing は不可。
回避策は (a) 全エキスパート dense 計算 + マスク合成 (容量メリット消失)、または
(b) リクエスト/スタイル単位の**固定ルーティング**で形状を静的に保つ。後者は
「キャラ系統ごとに別エキスパート」運用と噛み合い NPU 制約とも両立する。

### ポーズデータ取得方針 (探索テーマの補足)

上記「解剖メタ整合検査」とは別に、**生成側のポーズ語彙/多様性**を増やすデータ源の方針。
動画は QA (カウント検査) には不要 — あくまで ControlNet OpenPose 条件 + §11 記憶層の
pose バイアス源としての「ポーズ辞書」用途。オフラインのバッチ前処理で、リアルタイム
パイプライン (NPU 7.85ms 枠) には入らない。

- **本命: 3D モーション (Mixamo / VRM / MMD)**。任意アングルにレンダして
  (2Dポーズ + 参照画像) ペアを自動生成。実写ギャップ無し・正解骨格が既知・権利クリーン。
- **実写↔2D ギャップ**: 実写動画のポーズは硬く 2D の誇張/パースが乗らない → 転写すると
  棒立ちになりやすい。アニメ系 (DWPose) を当て、3D 合成を優先。
- **法務 (日本)**: 著作権は **30条の4 (情報解析)** が ML 学習を広く許容。さらに
  **派生した骨格座標だけ保持しピクセルは破棄**すれば著作物から遠く、リスクは小さい。
  ただし **30条の4 は契約(ToS) を上書きしない**。
- **YouTube**: 規約が自動コンテンツ採取を禁止 → `yt-dlp` も、埋め込み/画面キャプチャ/
  ヘッドレス自動操作/`captureStream` も「取り口」が違うだけで**規約上は同じく灰色〜黒**。
  「保存しない/別ブラウザ経由/再生のみ」は著作権側には効くが**契約違反は解消しない**。
  公式 IFrame は cross-origin サンドボックスで `canvas`/`captureStream` が tainted になり
  フレーム取得不可 — 回り込むには画面キャプチャ=保護回避になる。
- **クリーンな道**: 「再生→その場で骨格抽出→座標だけ保存→ピクセル破棄」という手法自体は
  優秀。向ける先を **CC ライセンス動画 / 正規に再生権を持つ素材** に限定すれば著作権も契約も
  クリア。本命は最初から配布されている 3D モーション。(商用化前は専門家確認)

---

## 参照ドキュメント

- `docs/pipeline-spec.md` — スレッド構成・キュー設計・タイミング試算
- `docs/tensor-spec.md` — Tensor クラス詳細設計
- `docs/http-api-spec.md` — HTTP API 仕様
- `docs/cpu-topology.md` — CPU コアアフィニティ設定
- `docs/archives/investigation-log.md` — probe1〜10 調査ログ

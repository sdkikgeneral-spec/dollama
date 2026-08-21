# dollama 実装ロードマップ

**スコープ (確定)**: 生成対象は**キャラクターのみ**。背景は外部 (Grok/Gemini/SD) +
CLIP Studio で合成し、出力は**切り抜き済み透過 PNG**。キャラ設定構造・切り抜き・
手指品質・学習ループの設計は `docs/character-bible-spec.md` を参照。

> **この文書は「今どこで次に何をやるか」の地図**。完了アイテムの経緯・決裁記録・実測の
> 詳細ナラティブは `docs/roadmap-decisions.md` (決定・完了アーカイブ) に退避した。数値の
> 完全版は `training-spec.md` / `dataset-spec.md` / `measurements-log.md`。各段の「状態」列は
> 短い現状 + 詳細への→ポインタに留める。

---

## Phase 1 — パイプライン骨格 ✅ 全完了

OpenVINO C++ API で動くパーツから順に実装し、スレッド骨格を完成させた (SDXL/BitNet 含まず)。

| # | 実装物 | ファイル | 状態 |
|---|---|---|---|
| 1 | Tensor クラス | `src/core/tensor.hpp` | ✅ |
| 2 | メモリアロケーター | `src/core/allocator.hpp` | ✅ |
| 3 | SPSC キュー | `src/core/queue.hpp` | ✅ |
| 4 | CLIP NPU 推論 | `src/infer/clip.hpp` | ✅ NPU 7.82ms |
| 5 | キャラ台帳 (CharacterBible, authored 層) | `src/core/character.hpp` | ✅ |
| 6 | WD14 CPU 推論 | `src/infer/wd14.hpp` | ✅ CPU 105ms |
| 7 | スレッド骨格 + CPU アフィニティ | `src/main.cpp` + `src/core/affinity.hpp` + `src/pipeline.hpp` | ✅ 9.13 fps (WD14 律速) |

**完了の定義**: stub → CLIP(NPU) → queue → WD14(CPU) のループがマルチスレッドで回りタグ出力。

---

## Phase 2 — SDXL 自作 CUDA カーネル ✅ 完了

CUDA カーネルをゼロから書き diffusers なしで画像生成を実現。SDXL UNet/VAE は **FP16 dense**
(ternary は Phase 4 BitNet へ)。**検証戦略**: Python(probe10) で中間テンソルをダンプ →
C++ カーネルがロードして許容誤差比較する**ゴールデンテスト**を各段に置く。

| 段 | 実装物 | ファイル | 状態 |
|---|---|---|---|
| 2-0/2-1 | Toolkit + meson CUDA 言語 + カーネル基盤 | `meson.build`, `src/kernels/utils.cuh` | ✅ CUDA 13.3 / sm_120 |
| 2-2 | primitives (GEMM/活性化/GroupNorm/Conv2d/Attention) | `src/kernels/*.cu` | ✅ 全緑 (計測は CLAUDE.md 表) |
| 2-3 | safetensors 重みローダー | `src/io/safetensors.hpp` | ✅ 19.0 µs/op・golden 突合 |
| 2-4 | **VAE decode** (初の実画像) | `src/kernels/vae_decode.cu` | ✅ SSIM 0.999992 |
| 2-5 | **SDXL UNet** + Euler/DDIM scheduler | `src/infer/unet.cu`/`.cuh`, `src/infer/scheduler.hpp` | ✅ noise_pred SSIM 0.999998・24段 golden 全緑 |
| 2-6a | フル C++ 拡散パイプライン結線 + HTTP DI | `src/infer/diffusion.cu`, `src/server/pipeline_generator.hpp` | ✅ 20step **84.07s** (golden 埋め込み・CFG なし) |
| 2-6 最適化 | direct conv→im2col/wmma・naive attn→flash | `unet.cu`/`vae_decode.cu`/`conv2d.cu`/`attention.cu` | 🟡 一旦クローズ **84s→11.30s** (7.44x)。残律速=UNet attn 4.60s (ライブラリ余地で保留)。本丸は Phase 4 へ |
| 2-6b | 本 txt2img: dual encoder (CLIP-L+G) + CFG + prompt→embeds 結線 | `infer/{clip_tokenizer,clip_encoder2,text_conditioner}.hpp`, `server/diffusion_runner.*`, `server/txt2img_generator.hpp` | ✅ (c3b1dac)・NPU L/bigG・guidance 7.5・20step・PNG 1024²・test 35/35 緑 |
| 2-6c | 拡散 backend プラグイン枠 `IDiffusionBackend` registry | `server/diffusion_backend.*`, `sdxl_backend.hpp`, `sd35_backend.hpp` (stub), `backend_image_generator.hpp` | ✅ (2026-07-02)・純 cpp 境界で OV/CUDA 隔離・env `DOLLAMA_BACKEND`・全 46 test 緑 |
| 2-6d | **実 checkpoint 差し替え = アニメ特化 SDXL 3 preset** (NoobAI-XL / Animagine XL 4.0 / Illustrious XL) | `server/diffusion_backend.cpp` (preset→パス解決), `cli_generate.hpp` (env `DOLLAMA_BACKEND_PRESET`), `THIRD_PARTY_NOTICES.md` | 🔲 **計画確定・未着手**。3 つとも SDXL アーキゆえ `SDXLBackend` 無改修・`BackendConfig.preset` で重み選択。素 base 1.0 の質天井を超える最大レバー ([[project-generation-quality-bar]])。段取り: ①ライセンス確認 ②unet/vae 変換 ③preset 解決グルー ④ループ確認。**実走は GPU セッションで別途** |

**完了の定義**: フル C++ で 1024² 画像生成・probe10 (3.80s/20steps) 同等以上。
**接続**: 生成画像は §11 品質スコアラ (Phase 4 B) の入口 (生成→採点→A へ FB)。

---

## Phase 3 — HTTP サーバー ✅ 完了

外部クライアントから呼べるようにする。**配管は自作しない** (HTTP/JSON は定番ヘッダオンリー・
CLAUDE.md「実装方針」)。API 仕様: `docs/http-api-spec.md`。`POST /v1/images/generations` で
OpenAI Images API 互換。

| # | 実装物 | ライブラリ / ファイル | 状態 |
|---|---|---|---|
| 1 | HTTP サーバー | **cpp-httplib 0.47.0** | ✅ |
| 2 | JSON 入出力 | **nlohmann/json 3.12.0** | ✅ |
| 3 | エンドポイント実装 | `src/server/api.cpp` | ✅ generations/health/models (edits=501) |
| 4 | Base64 (PNG 返却) | `src/server/base64.hpp` | ✅ |

**生成本体の抽象境界**: `src/server/generator.hpp` の `IImageGenerator` (純粋仮想) 越しに注入。
生成器の責務は PNG バイト列まで・base64 化はサーバ層。**完了の定義**: `curl` で PNG(base64) が
返る → ✅ (test_http 全緑・往復 2.11ms)。

---

## Phase 4 — 自作タグ生成 LM (旧「BitNet b1.58 LLM」) — 進行中

現状の prompt 直入力 (LM 段なし・Qwen2 は Python probe 専用) を自作モデルで結線する。本線は「**dense で動くタグ生成 LM**」、
その上に 拡張(A)・評価(B)・圧縮(#5 ternary) を積む。番号は実装 ID で不変。

> **方向性 (2026-06 レビュー)**: 核は decoder-only 小型タグ生成 LM (`bitnet.hpp` 33M)。**ternary
> (b1.58) は目的でなく圧縮の研究軸に降格**。置き場は **GPU 第一** (自作 CUDA カーネル流用・拡散上流で
> 逐次)・**CPU は代替**・**NPU は自己回帰不可で除外**。素の text→tags は DanTagGen/TIPO が既存 → 蒸留
> 教師・品質基準として使う。**2D 特化の独自性 = ① 同一性条件付け (A) / ② 品質スコアラ (B, §11)**。

| 段 | # | 実装物 | 状態 |
|---|---|---|---|
| 基盤 | 1 | 訓練データ収集 (user text → danbooru tags) | ✅ 5,000 ペア / vocab 4,994 / OOV0 (dataset-spec) |
| 基盤 | 2 | モデル定義 `src/models/bitnet.hpp` | ✅ decoder-only LLaMA系 d512/L8/h8/ffn1792 = **32.98M**・純ホスト参照 forward |
| dense | 3 | トークナイザー `src/io/tokenizer.hpp` | ✅ vocab.json 駆動・UNK 0・往復一致 (§6 正規化) |
| dense | 4 | 訓練スクリプト `scripts/train_bitnet.py` | ✅ hard CE 6ep・val_loss 2.41・top10 recall 0.777 (training-spec §10) |
| dense | 4-蒸留 | 蒸留 D2–D6 (recall 天井突破の探索) | 🔬 **全 4 路線で recall 利得は否定** (過学習抑制のみ・training-spec §10–12) |
| dense | 6 | C++ 推論 CPU `src/infer/bitnet.hpp` | ✅ golden corr 1.0・greedy 5/5・~253ms→Tier1 で ~5x |
| dense | 6-GPU | C++ GPU 推論 `src/infer/bitnet_gpu.cu`/`.cuh` | ✅ CPU 版と数値一致・forward 46.8–87.5x・test_bitnet_gpu 緑 |
| 拡張 (A) | A | **同一性条件付けタグ生成** (character-bible 入力) | ✅ **クローズ (二分結論)**: diverse-val F1 は seed ノイズ・**identity retention 0.975 が機能基盤** (roadmap-decisions §A / training-spec §9.10) |
| 評価 (B) | B | **アニメ品質スコアラ** (§11 蒸留 QA・生成画像を採点) | 🔄 Stage1 QualityGate ✅ / ScorerNet 訓練・OV/NPU 化 (B-3b〜c) ✅ (NPU 8.32ms 純 conv 実証) / C++ 結線 **B-3d〜B-5 完了** (`IScorer`/scorer_runner/scoring_postprocess で生成器結線・採点はログのみ)。quality は Q-2 で **CLIP-embed 別枝 (QualityMLP)** に分離・ScorerNet は anatomy 専用化 (F 決定ログ)。**残 = 消費者 F** |
| 圧縮 | 5 | Ternary GEMM (重み{-1,0,+1}) — 圧縮実験 | ⏳ 降格 (dense 後の研究軸)。INT8 dense CPU 推論は別途完了 (`bitnet_int8.hpp`・training-spec §15) |

**学習強化プログラム C/B/A/D/F の決着 (2026-06〜07)**: → 下記「タグ生成 LM 学習強化プログラム」節。
**完了の定義**: user text → danbooru タグ変換が C++ (CPU/GPU) で動き、A が character-bible 入力で
機能し、B が FB ループを閉じること。ternary 化は完了条件に含めない。

---

## Phase 5 — ポージング・パース (コマ単位の自由な画角)

**動機**: 漫画はコマごとに角度・パース (あおり/俯瞰/foreshortening)・ダイナミックポーズが変わる。
拡散が最も苦手とする領域。**なぜタグ LM で解けないか**: ① danbooru は立ち・正面に偏り語彙が
存在しても学習サンプルが桁違いに少ない ② 拡散は 3D 構造を持たず極端な角度で解剖破綻 = **幾何の
問題**。→ **構造を外から与える** (industry 標準 = structural conditioning)。

**3 段の依存連鎖 (5-1 → 5-2 → 5-3)**:

```text
   5-1 崩壊境界の実測 (probe) ──── まず「どの画角でどれだけ崩れるか」を地図化・低コスト
        │ これが 5-2/5-3 の必要量と投資判断を決める
        ▼
   5-2 img2img 幾何注入 ────────── 安い・既存パス (iGPU VAE encode 79ms) 流用・新カーネル不要
        │ 5-2 で救えない強パースが実需として残るなら…
        ▼
   5-3 構造条件付け (ControlNet) ── 重い本丸・自作 UNet に制御ブランチ増設
```

| 段 | # | 実装物 | 内容 / 流用 | 状態 |
|---|---|---|---|---|
| 実測 | 5-1 | **崩壊境界 probe** | 固定キャラ × 画角/パースタグの直積を自作 SDXL 生成 → 既存 QualityGate 異常タグ hit 率で崩壊スコア化 → ヒートマップ。**採点器は新規不要**。`scripts/dollma_probe_pose_breakage.py` 想定 | ⏳ 未着手 (起点) |
| 安価解 | 5-2 | **img2img 幾何注入** | 3D ソフトのアタリで構図/パース/骨格を幾何固定 → img2img 下絵 → 拡散は清書。**核心 = denoising strength スイートスポット** (ポーズ保存度 vs 清書品質のトレードオフ実測)。既存パス | ⏳ 未着手 (5-1 後) |
| 本丸 | 5-3 | **構造条件付け (ControlNet)** | 骨格 (OpenPose)/深度/線画で構造強制。自作 UNet に制御ブランチ増設 + ControlNet 重みを既存ローダーで読む + 制御入力 OV グルー 1 段。VRAM +1.2GB で 16GB 内 | ⏳ 未着手 (5-2 で不足確認後・決裁要) |

**他段との関係**: Phase 4 B (QualityGate = 5-1 採点器)・A (同一性・5-2/5-3 と直交)・LoRA (見た目/画風
ロック・直交)・ポーズデータ取得方針 (下記バックログ = 5-3 条件入力源)・§11 解剖メタ整合検査 (数・位相のみ)。
⚠️ **HW 前提**: SDXL 実走は研究機 (RTX5080) 必須・開発機 (GTX1080Ti) は本走不向き ([[dev-pc-hardware]])。
**着手順**: 5-1 プラン化 → 5-2 denoise 実験 → 5-3 決裁。**完了の定義**: コマ単位で指定した画角の透過
キャラ PNG を解剖破綻 QualityGate 許容内に抑えて出力。最低ライン = 5-1 + 5-2。

### 5-D — デッサンモード (トレース用あたり生成・5-3 の応用)

**動機 (2026-07-04)**: 本番生成とは別に、下書きのトレース元「デッサン人形のあたり図」を出したい。
CLIP STUDIO の 3D 人形は 3D 操作が面倒 + 3DCG 臭い。dollama の勝ち筋は **3D をレンダせず拡散で最初から
アニメ絵比率のあたりを出す** (5-3 の直接応用)。

**役割分離** (キャラ比率のあたりが新規性): プロポーション=スケルトン幾何 (character-bible 頭身から
骨長生成) / 同一性=bible タグ + Phase 4 A (retention 0.975) / ポーズ=OpenPose キーポイント / 絵柄=
「gray mannequin」preset。**ポーズ入力 3 種** (全対応・共通中間表現 = OpenPose 図): ① 2D スケルトン編集
(Blazor UI) ② 参照画像から DWPose 抽出 (OV 推論 1 段・NPU/iGPU 候補) ③ テキスト指定。**拘束方式 =
ControlNet-OpenPose 一択** (2026-07-04 決裁・T2I-Adapter 却下・質優先)。**段取り**: ① 質検証 probe
(diffusers ControlNet-OpenPose + アニメ checkpoint で狙ったあたりが出るか) → ② 本結線 (5-3 の CUDA UNet
統合 + preset + 頭身→骨長マッパー + UI + ポーズ検出 OV)。未着手 (5-3 に依存)。

---

## キャラクター品質・一貫性 (画像生成後の段、Phase 2+ で並行)

キャラを「コマ間でブレさせない」「手指を崩さない」段。設計は `docs/character-bible-spec.md`。
authored 層 (character.hpp) は完了、以下は learned 層・後処理段。

| 項目 | 内容 | spec | 状態 |
|---|---|---|---|
| 切り抜き (マッティング) | 透過 PNG 出力。ISNet-anime (Apache-2.0) | §3, §9 | ✅ **M-6 完了**: matting_device=iGPU (99.96ms・M-5 確定)・`IMatter`/make_matter で生成器結線・研究機 e2e 実走済 (透過 PNG 出力) |
| 手指 L1 (予防) | 品質ネガティブ注入 | §10 | ✅ (器) |
| 手指 L2/L3 (修復・検査) | 手検出→インペイント / 指数照合 | §10 | Phase 2 |
| 学習層 `CharacterMemory` | 生成→学習→FB ループ (記憶層 → QA スコアラ → fine-tune) | §11 | Phase 2/3 |
| 背景プラグイン | 外部背景生成 + 自動合成 | §9 | Phase 3 |

---

## マイルストーン一覧

| M | 内容 | 目標 |
|---|---|---|
| **M1** | Phase 1 完了: C++ でタグ抽出ループが動く | ✅ |
| **M2** | Phase 2 完了: フル C++ で画像生成 | ✅ |
| **M3** | Phase 3 完了: HTTP 経由で画像生成 | ✅ |
| **M4** | Phase 4 完了: 自作 LM (dense) + A + B の end-to-end | dense→A/B 後 |
| **M5** | Phase 5: 画角/パース指定で解剖破綻を抑えた透過キャラ | 5-1 + 5-2 (5-3 は拡張) |

---

## 技術的リスク・未確定事項

| 項目 | 状態 |
|---|---|
| ~~SDXL UNet 自作カーネル~~ | ✅ 解消 (2-5・SSIM 0.999998) |
| ~~タグ生成 LM 基礎データ~~ | ✅ 解消 (#1・5,000 ペア) |
| ~~A 同一性条件付けデータ~~ | ✅ 解消 (dataset-spec §13→a12k で retention 0.975 クローズ) |
| B 品質スコアラの正解ラベル | 🟡 前進。§11 軸は WD14 soft 8 軸蒸留 (B-3b〜B-5 完了)・**quality は Q-2 で CLIP-embed 別枝 (QualityMLP) に分離** (ScorerNet=anatomy 専用・waifu 蒸留・F 決定ログ) |
| ~~B の NPU 実行性~~ | ✅ 解消。純 conv は NPU 最速 (448² 4.62ms) |
| ~~safetensors パーサー~~ | ✅ 解消 (2-3・19.0 µs/op) |
| ~~Linux 対応 (HTTP)~~ | ✅ cpp-httplib が吸収 |

---

## タグ生成 LM 学習強化プログラム (C/B/A/D/F)

**背景 (診断)**: 蒸留 4 路線が top10 recall を動かせず (training-spec §10–12) = **33M は 4,500 ペアから
学べる分を学び切った**。recall を上げる筋は ① データ ② 容量 ③ 測り方 に限られ、レシピ側は枯れた。
現行評価 (固定 val 500・テンプレ 3 種・recall@10) は「テンプレに合うか」を測っており実用品質でない。

**プログラム結論 (2026-06〜07)**: C (評価作り直し) の上で B/A/D を 4 seed sweep → **diverse-val F1 を
頑健に押し上げたのは施策 B (入力多様化) のみ・~2,000 件で飽和**。**A は F1 非寄与だが identity retention
0.975 の機能基盤**。**D (33M→80M) は陰性確定 (勝者 = 33M b2000∧identity)**。→ 残る低帯域 (diverse_a
~0.31 / diverse_b ~0.36) はデータ件数でも容量でもない別軸 (多様性の質・アーキ・損失設計・本命 F の
実品質信号) が次のフロンティア。**各施策の決着経緯・決裁記録は `docs/roadmap-decisions.md`**。

```
   C 評価を実目標に作り直す ──────────── 全施策の前提・最初・低コスト
        │ これで初めて B/A/D の効果が「実は何点か」で測れる
        ▼
   B 入力(自然文)多様化 ──┐  C なしだと D2 の二の舞 (proxy 上は悪化に見える)
   A 実ペア増 8200→数万 ──┤  ※法務/ToS ゲート (PL 経由・dataset-spec §1.3)
        │                 │
        ▼                 │
   D 容量 33M→60-100M ─────┘  D は A と必ずセット (単独は過学習)
        │
        ▼
   F 品質フィードバック学習 ── 到達点・本命。B スコアラ→生成→SDXL→採点→fine-tune
```

| 施策 | 機構 | 状態 |
|---|---|---|
| **C** 評価作り直し | テンプレ外の多様な val (tags-stay-real) で生成 set-F1。固定 val 500 は不変で加算的に追加 | ✅ 完了 (C-1〜4・training-spec §13) |
| **B** 入力多様化 | タグ固定で自然文を多様化。テンプレ偏りを解消し実世界汎化 | ✅ 完了 (500→2,000→10,000・**~2,000 で飽和**・本線昇格決裁済・training-spec §14) |
| **A** 実ペア増 | danbooru harvest 8,200→数万。**a12k でクローズ** (F1 は seed ノイズ・retention 0.975 が機能基盤) | ✅ 評価完了 (法務ゲート dataset-spec §1.3・`[B-merge-at-A]` で焼成済) |
| **D** 容量増 | 33M→80M。**陰性確定・80M 不採用** (F1 seed ノイズ・retention 床割れ・勝者=33M) | ✅ クローズ (training-spec §16) |
| **F** 品質ループ | B スコアラ→SDXL 画像→採点→報酬で LM fine-tune。recall でなく「良い絵を生む」方向へ | 🔄 進行 (F-0a/F-0b 下記) |
| **F-0a** 信号ゲート | reward 収集 80/80 → **判定 = 信号弱** (std 0.038 / best−worst 0.203)。clean vs clutter で \|r\| 4倍分離 = 弱いが本物の勾配 | ✅ 実走・判定済 (roadmap-decisions) |
| **F-0b** SFT | RAFT best-of-8→SFT を end-to-end 実走 → **不採用クローズ** (reward +0.017 が set-F1 −0.02 を正当化できず・正典無改変) | ✅ 不採用クローズ (f0b-rejection-sft-plan.md) |

**C と F は同じ軸の両端**: C = より良いオフライン proxy、F = 本物のオンライン信号。背骨は**物差しを
proxy→実品質へ動かすこと**。**次レバー** (F-0b 後): reward 設計 / 日本語条件付け改修 / seed 制御。
**着手は CLAUDE.md ルール** (プランモード設計→承認→PL 振り分け)。

---

## LoRA 対応 (生成エンジン拡張・本線)

2D イラスト AI 生態系で LoRA は事実上の標準装備。tag ベース同一性 (Phase 4 A) だけでは OC を固定
しきれず、**画風/OC を重みベースで持つ手段としてほぼ必須** (ユーザー 2026-06-24・本線昇格)。tag ベース
同一性 (A) とは直交補完 (LoRA = 見た目/画風ロック・tag = シーン/ポーズ/表情)。研究機前提。

| 段 | 実装物 | 難易度 | 状態 |
|---|---|---|---|
| L-1 | **offline merge**: LoRA を base に事前マージ (W′=W+scale·BA) → 合成 checkpoint ロード。**自作推論は無改修**・前処理ツール 1 本 (`scripts/dollma_merge_lora.py`) | 低 | ✅ 完了 (aec741e)・test 12/12 緑・**実マージ+実画像は研究機で別途** |
| L-2 | **ランタイム LoRA**: 生成ごとに LoRA 選択/スタック/強度可変。kohya→diffusers 写像 (`load_lora_modules`) + 常駐重み apply-time マージ (`unet_apply_loras`: `launch_gemm_fp16` で delta=scale·BA → `launch_add` in-place) / `unet_clear_loras` bit-exact 復元。HTTP `loras:[{name,strength}]` 結線 (name allowlist で path traversal 封止) | 中 | ✅ **完了** (2dc4181 / path traversal 修正 4fa6ca5)・test_lora_runtime 全ゲート PASS (parity max_abs 4.9e-4・revert memcmp bit-exact・stack+revert bit-exact)・数値正典=L-1 offline merge。**UI (Blazor) 選択チップも完了**: 静的カタログ (`ui/wwwroot/loras.json`)・トグルチップ+強度スライダ (0.0–1.5)・空選択で `loras` キー非送出=従来経路無改変・ui.Tests 46 緑 (サーバー無改修) |

**データ要件**: 使うだけなら学習データ不要 (既存 .safetensors を merge/適用)。**自分の OC/画風 LoRA を
作る場合のみ画像が要る** (キャラ 10〜50 枚 or 自分の絵)。OC の鶏卵問題は ① 参照絵 ② 記憶層ブートストラップ
(タグ+A で量産 → §11 品質ゲートで選別 → 学習) で解く。既存資産が効く (WD14=自動キャプション・品質ゲート=
素材キュレーター)。進め方: **L-1 → L-2**。UI (Blazor) は L-2 で LoRA 選択/強度チップを流用。

---

## 将来の探索テーマ (バックログ)

本線に載せないが研究価値があり時期未定の項目。設計エッセイの長文は `docs/roadmap-decisions.md`
「バックログ深掘り」へ退避。

| テーマ | 概要 | 評価 |
|---|---|---|
| **MoE × HW 分散配置** | 得意分野の違う完結 LM を HW ごとに分散し並列協調 (Qwen2/CPU=NL 意図・自作 LM/GPU=タグ・A=同一性)。芯 (全 HW 協調) に直結。粗粒度 MoE で NPU 静的形状の制約を回避。導入順: まず GPU 本線 A を動かし弱点を実測 → 穴を埋める専門家だけ B (品質スコアラ=アービタ) で束ねる | ◎ (A 計測後) |
| 拡散 UNet の timestep-expert | eDiff-I / DiT-MoE 系。20step を初期=構図/後期=ディテールで別エキスパート分割 | ◯ |
| タグ生成 LM の内部 MoE 化 | 1モデル内部の古典 MoE。30–100M には過剰・NPU 静的形状と非両立。協調は上行 (別完結 LM 分散) を採る | △ 過剰 |
| **NPU 骨格/部位検出による解剖メタ整合検査** | 生成画像を NPU で部位検出し**「数・位相」だけ**を宣言値と照合 (指数/四肢本数/重複欠損/左右対称)。角度・比率・ポーズ自然さは見ない (2D 誤検出回避)。拡散中 3.8s ほぼ遊休 → 裏で実質ゼロコスト。character-bible-spec §11 | **◎ (Phase 2)** |
| **MCP 公開 / Claude 連携** | (a) dollama を MCP サーバ公開 → Claude 等がツール呼び出し (Phase 3 HTTP の薄ラッパ) (b) プロンプト解析を Claude に = BitNet の代替でなく訓練データ収集/評価基準 | (a) ◎ Phase 3 / (b) Phase 4 |
| **ガヤ (群衆) 複数人出力** | **案 B = 1人ずつ生成 → ISNet 透過 PNG 化 → 下流で重ねる** (芯にそのまま乗る・新規モデル不要・各キャラ完全立ち絵)。足りなければ 案 A = instance segmentation (成熟度低・ハード)。起点は B 確定 | ◯ (時期未定) |
| **HW 環境抽象化 / 実行モード (`--cpu` `--npu` `--dgpu`)** | 非 Intel 環境 (Ryzen+NVIDIA 等) でも動かすためのデバイス割り当て切替。NVIDIA dGPU なら CUDA 無傷 (sm_86 再ビルドのみ)・NPU/iGPU は宣言で CPU/CUDA 退避。`--vram=6g` で SDXL 自動ダイエット。詳細 [docs/hw-environment-spec.md](hw-environment-spec.md) | ◯ (時期未定) |
| **遠隔 HW ノード (LAN 越し第 2 マシン)** | 余剰ノート (Ryzen+RTX3060 Laptop+64GB) を **完全 opt-in** で足す。密結合(層分割)は不可・粗粒度(stage/ジョブ)は成立・帯域非ボトルネック。消費者 ①訓練/sweep 分散 (即効) → ③ critic ノード (研究本命=生成は研究機/講評は遠隔・解剖ポーズ 3 Tier で graceful degradation)。**本命は速度でなく精度の上乗せ**・無くても全機能成立。詳細 roadmap-decisions | ◎ (①即効/③本命・時期未定) |
| **CPU 側 LM 推論の速度最適化** | ✅ **Tier 1 完了** (AVX2 単スレッド ~5x・golden 維持)。律速は FFN+attn ~92% (lm_head は 7.5%)。**Tier 2(A) 独立 forward ワーカーは設計確定/留保** — 発動条件 (LM 段ボトルネック化) は **単一 GPU では計測で不成立** (LM は SDXL 裏に完全隠蔽・2026-06-28)。詳細 roadmap-decisions | ✅ Tier1 / Tier2 留保 (裏取り済) |
| **プレビュー用低解像度ドラフトモード** | 本番と同じ重み・同ステップで解像度だけ下げる (768²)。UNet attention に効く (トークン 1/4・attn 1/16)。ステップ削減は不採用 | ✅ 完了 (2026-06-29・UI 2 ボタン・`DraftPreview.cs`・ui.Tests 39 緑) |
| **成人向け後処理: モザイク/バー修正** | 頒布前提なら事実上必須 (刑法175条)。matting と同じ後処理段に 2 パーツ: ① NSFW 領域検出 (OV グルー 1 段・山はモデル選定) ② 修正処理 (モザイク/バー・純画像処理)。`OutputSpec` に `censor` 追加。**絶対線: 成人キャラのみ** | ◯ (検出モデル選定が前提・時期未定) |

**共通の制約**: NPU は静的形状のみ → token-level dynamic routing 不可。回避は (a) 全エキスパート dense +
マスク合成 (容量メリット消失) または (b) リクエスト/スタイル単位の**固定ルーティング** (キャラ系統ごとに
別エキスパート運用と噛み合う)。

### ポーズデータ取得方針 (探索テーマの補足)

**生成側のポーズ語彙/多様性**を増やすデータ源の方針 (ControlNet OpenPose 条件 + §11 記憶層の pose
バイアス源用途)。オフラインのバッチ前処理でリアルタイムパイプラインには入らない。

- **本命: 3D モーション (Mixamo / VRM / MMD)**。任意アングルにレンダして (2Dポーズ + 参照画像) を自動
  生成。実写ギャップ無し・正解骨格既知・権利クリーン。実写は硬く 2D の誇張/パースが乗らない → 3D 優先。
- **法務 (日本)**: 著作権は **30条の4 (情報解析)** が ML 学習を広く許容・派生骨格座標だけ保持しピクセル
  破棄すればリスク小。ただし **30条の4 は契約(ToS) を上書きしない**。YouTube は規約が自動採取を禁止 →
  取り口を問わず契約上は灰色〜黒。**クリーンな道** = CC ライセンス動画 / 正規に再生権を持つ素材に限定
  (本命は最初から配布されている 3D モーション・商用化前は専門家確認)。

---

## 参照ドキュメント

- `docs/roadmap-decisions.md` — **決定・完了アーカイブ** (本文書から退避した経緯・決裁記録)
- `docs/pipeline-spec.md` — スレッド構成・キュー設計・タイミング試算
- `docs/tensor-spec.md` — Tensor クラス詳細設計
- `docs/http-api-spec.md` — HTTP API 仕様
- `docs/cpu-topology.md` — CPU コアアフィニティ設定
- `docs/hw-environment-spec.md` — HW 環境抽象化・対応環境マトリクス
- `docs/training-spec.md` / `docs/dataset-spec.md` / `docs/measurements-log.md` — LM 訓練/データ/計測の完全版
- `docs/archives/investigation-log.md` — probe1〜10 調査ログ

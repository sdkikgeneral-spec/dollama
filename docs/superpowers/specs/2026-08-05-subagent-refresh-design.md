# サブエージェント定義の現状化 (設計) — 2026-08-05

## 背景

`.claude/agents/*.md` の 11 体は 2026-06-20〜06-26 に著述された。それ以降に
**Phase 2 完了 (2-6a/b/c)・Phase 4 一巡 (A/B/C/D/蒸留/F-0a/Q-2/F-0b)・fast-mode 計画の起票・
二機体制の確立 (開発機 GTX1080Ti / 研究機 RTX5080)** が起き、定義の前提が実装と乖離した。
モデル世代も当時と異なり、旧世代向けの冗長な自己チェック指示や長いコード例が残っている。

放置すると害が出るのは主に `project-leader` で、**「次に着手すべきタスク 1〜7」が全部完了済み**の
まま残っており、PL がそれを読んで指示すると完了済みタスクへ誘導する。

## 決定事項 (ユーザー決裁 2026-08-05)

| 論点 | 決定 |
|---|---|
| スコープ | **全 11 体を全面改訂 + 統廃合の判断も含む** |
| 体制 | **案A = 最小の統廃合 (11 → 10 体) + 共通ルール 1 枚** |
| `model:` 指定 | **重い判断のみ上位固定・他は inherit (書かない)** — モデル名を全定義に焼くと世代交代で再び陳腐化するため |
| `model-trainer` の訓練機 | **RTX5080 主・開発機 1080Ti も併記** (33M 規模は 1080Ti FP32 で回る。実績は全訓練が 1080Ti・記録は `training-spec.md:101`) |

## 現状の乖離 (改訂の根拠)

### 全体に共通する欠落

- `model:` 指定がどのファイルにも無い (全部が親モデル継承)
- **`Edit` ツールが全エージェントに無い** (`Bash/Read/Write/Glob/Grep` のみ) → 既存ファイルの部分修正を
  `Write` 全文上書きでやらせている。主シェルが PowerShell なのに `Bash` のみ、も実務と不一致
- **二機体制の明示が `dataset-curator` だけ**。他は研究機前提か機械の区別なし
- テスト必須ルール (CLAUDE.md ルール4)・`hooks/dollama_protect_artifacts.py` による正典保護・
  NAS 搬送・研究機の SAC 制約 — どれも定義に書かれていない

### 個別の事実誤り

| エージェント | 現状との矛盾 |
|---|---|
| `project-leader` | 「次に着手すべきタスク 1〜7」が全部完了済み。エージェント表に `dataset-curator`/`csharp-ui-implementer`/`model-trainer` が無い |
| `cpp-implementer` | 担当が `tensor.hpp`/`http.cpp (Winsock2)` 止まり。HTTP は cpp-httplib 確定なのに「Winsock2 を使う」と明記。Phase 1 当時の「既知のバグ 6 件」を今も抱える |
| `cuda-kernel-dev` | 主役が ternary GEMM (実際は降格・未着手)。gemm/attention/conv2d/unet/vae は完成済み。「cuBLAS 使わない」も方針変更済み。fast-mode 計画未反映 |
| `gpu-benchmarker` / `npu-benchmarker` | 「調査フェーズ・本実装は C++ + **LibTorch**」← LibTorch 不使用が確定方針 |
| `model-trainer` | BPE 学習が担当に残る (タグ単位完全一致で確定・BPE は切離)。正典無改変・NAS 搬送ルール未記載 |
| `model-converter` | 変換対象が Qwen2/WD14/CLIP-L/「Aesthetic 検討中」止まり。ISNet/ScorerNet/QualityMLP/CLIP-image 未反映 |
| `pipeline-debugger` | 中身が Python probe 時代 (`queue.Queue`/`torch.cuda`)。役割自体が計測クローズで枯れている |
| `prompt-engineer` | 前提が「Qwen2 が英語プロンプトを作る」。実際は自作 33M LM。役割が `dataset-curator` と重複 |

## S1. 成果物の構成

```
docs/agent-common.md          (新規1枚) 全エージェント共通の非交渉ルール ← 保守点をここに集約
.claude/agents/
  project-leader.md           改訂 (最重要)
  cpp-implementer.md          改訂
  cuda-kernel-dev.md          改訂
  csharp-ui-implementer.md    改訂 (軽微)
  model-trainer.md            改訂
  dataset-curator.md          改訂
  model-converter.md          改訂
  npu-benchmarker.md          改訂
  gpu-benchmarker.md          改訂
  perf-profiler.md            新規 (pipeline-debugger.md を置換・旧ファイル削除)
  prompt-engineer.md          削除
```

各定義は「共通ルールを読む」1 行 + 自分の固有知識だけを持つ。CLAUDE.md と重複する規約・
長いコード例は削除する。**共通ルールを 1 枚に集約するのは、今回の陳腐化の根本原因
(同じ規約が 11 箇所にコピーされ、11 箇所とも腐る) への手当てである。**

## S2. `docs/agent-common.md` の中身

| 節 | 内容 |
|---|---|
| 走る機械の判定 | 開発機 (GTX1080Ti sm_61 / i7-10700 / NPU なし / nvcc なし / `with_cuda=with_openvino=False`) と研究機 (Core Ultra 9 285 + NPU + Intel Xe iGPU + RTX5080 sm_120) の識別方法 (`nvidia-smi` / `nvcc --version` / `build/meson-info/intro-buildoptions.json`) と、**研究機前提のタスクを開発機で振られたらスクリプト著述までで止めて報告する**規律 |
| コーディング規約 | Allman・`switch` の case 位置・日本語コメント (C++/C# 共通)・`dollma_` は scripts のみ |
| テスト必須 | CLAUDE.md ルール4。`meson test -C build` / `dotnet test` が緑になるまで完了としない |
| 正典アーティファクト保護 | hook がブロックする対象 = `data/bitnet/bitnet_dense{,_fp32}.safetensors` / `data/bitnet/golden/` 配下 / `pairs.train.jsonl` / `pairs.val.jsonl` / `pairs.eval_diverse_[ab].jsonl`。実験は別名 (`bitnet_dense_<suffix>.safetensors`) へ出す |
| 重み搬送 | NAS 経由 (`train_bitnet.py --copy/--publish`)。cross-GPU 再生成は bit 非一致ゆえ exact コピー必須 |
| 研究機の SAC 制約 | 再ビルドした exe の新ハッシュがブロックされる → allow-list 更新をユーザーへ依頼するか、開発機ビルド緑 + Python/OV 経由の数値検証で回す |
| 実装方針 | 使う/使わない表 (cpp-httplib + nlohmann 採用・LibTorch/diffusers/llama.cpp 不使用・cuBLAS は到達困難な重い GEMM/Conv のみ許容) |
| docs 更新の分担 | CLAUDE.md = 芯の数値のみ / `measurements-log.md` = 計測全文 / `roadmap.md` = 経緯と採否 / `training-spec.md`・`dataset-spec.md` = 手順詳細 |
| 報告フォーマット | 完了時に「何を・どの機械で・どの数値で・どの test が緑か」を返す。CLAUDE.md 計測表への追記が要るものは追記案を出す |

## S3. 10 体の改訂方針

`model` 列の「上位固定」は間違いのコストが高い 3 体のみ。残りは指定を書かず
セッションのモデルに追従させる (陳腐化防止)。

| エージェント | model | 主な改訂点 |
|---|---|---|
| `project-leader` | 上位固定 | **「次に着手すべきタスク 1〜7」を撤去**し「`roadmap.md` / CLAUDE.md を読んで現状から判断する」に置換 (具体タスクを焼くから腐る)。エージェント表を 10 体へ更新。二機体制での振り分け判断 (このタスクはどちらの機械か) を追加。承認権限は維持 |
| `cpp-implementer` | 上位固定 | 担当ツリーを実態へ (`core/` 7・`io/` 3・`infer/` 18・`server/` 28・`models/` 2・`tests/` 48 + `main.cpp`/`pipeline.hpp`)。**HTTP = cpp-httplib + nlohmann に訂正** (Winsock2 記述を削除)。Phase 1 の「既知のバグ 6 件」撤去。**OV/CUDA 隔離の作法** (純 cpp interface + `*_stub.cpp` の二重実装 + `#ifndef __CUDACC__`) を追加 — 現在の最重要な固有知識。開発機では `with_cuda=with_openvino=False` でビルドする旨 |
| `cuda-kernel-dev` | 上位固定 | ternary 主役をやめ、完成済みカーネル一覧 (gemm/activation/groupnorm/conv2d/attention/layernorm/geglu/bias_add/elementwise/timeembed/vae_decode + `infer/unet.cu`/`diffusion.cu`/`bitnet_gpu.cu`) と **fast-mode G-0〜G-6k を主タスク**に。cuBLAS 方針を訂正。golden 非交渉 (default 無改変・UNet SSIM 0.999998 / VAE 0.999988 アンカー)。nvcc は研究機のみ = **開発機ではコンパイル不可**を明記 |
| `csharp-ui-implementer` | inherit | `ui/` 実態 (Telemetry・`data/thumbs`・`DraftPreview`) と `ui.Tests` 39 緑を反映。ドラフトモード/サムネ実装済みを既存挙動として記載 |
| `model-trainer` | inherit | **訓練機 = RTX5080 主・開発機 1080Ti 併記** (1080Ti は sm_61 で FP16 native 非対応 → FP32 固定・NAS 搬送が必要)。BPE 学習を撤去。Phase 4 確定レシピ (b2000 多様化 ∧ identity・4 seed paired sweep・eval が律速で ~895s/本)・正典無改変/別名出力・`[B-merge-at-A]` 完了を反映。`train_bitnet.py` の主要フラグ (`--train-file`/`--identity`/`--arch`/`--sft-rejection`/`--copy`/`--publish`) を記載 |
| `dataset-curator` | inherit | §12〜§19 の現状 (`pairs.train.diverse_b2000.jsonl` / `pairs.identity.train.a12k.jsonl` / `pairs.eval_diverse_[ab].jsonl` 凍結アンカー) へ。hook 保護対象の凍結ファイルを明記。**廃止する `prompt-engineer` の日本語→danbooru タグ変換表を語彙規約として引き取る** |
| `model-converter` | inherit | 変換対象を実態へ (ISNet-anime / ScorerNet / QualityMLP / CLIP-image ViT-L / SDXL text encoders)。`models/` 実構成。落とし穴 (CLIP-image の MHA fastpath 無効化・`eval()` で BN 正常化・NPU は静的形状必須) |
| `npu-benchmarker` | inherit | 確定値へ更新 (CLIP-L 7.85ms / WD14 268ms=不採用 / ScorerNet 8.32ms / QualityMLP 0.553ms / CLIP-image 85.55ms / ISNet は iGPU 99.96ms が最速)。**LibTorch 記述を削除**。研究機専用を明記。「純 conv は NPU フレンドリー・Window Attention は不向き」という切り分け結論を前提知識に |
| `gpu-benchmarker` | inherit | 「調査フェーズ / LibTorch」を削除。担当を実態へ = SDXL 実走・rollout 収集 (best-of-N)・reward 採点・GPU golden 再確認・`nvidia-smi` での電力/稼働率診断。SAC 制約と実測 19.5s/枚 (実 checkpoint + CFG)。研究機専用 |
| `perf-profiler` (新) | inherit | 全面新規。`src/infer/profile.cuh` を使った CUDA events 計装・occupancy/latency 律速の診断 (電力 43% / 帯域 11% / sm% に騙されない読み方)・計測クローズ事実 (単一 GPU では multi-frame Tier2 は不発動・LM は SDXL の裏に完全隠蔽) を前提知識に。Python probe 時代の記述は全削除 |

### tools の統一

全体を `Bash, PowerShell, Read, Write, Edit, Glob, Grep` に揃える。
`project-leader` だけは `Read, Glob, Grep` を維持する (コードを書かせないため)。

## S4. 各定義の共通テンプレート

frontmatter:

```yaml
---
name: <name>
description: <何を担当し・いつ呼ぶか・どこが境界か (他エージェントとの棲み分けを明記)>
tools: Bash, PowerShell, Read, Write, Edit, Glob, Grep
model: opus   # 上位固定の 3 体のみ。他は行ごと書かない (セッション追従)
---
```

`model` に書くのは**エイリアス (`opus`) のみ**とし、具体バージョン (`claude-opus-5` 等) は書かない。
エイリアスは世代交代で最新を指し続けるため陳腐化しない — これが「モデル名を焼くと腐る」への手当てである。

本文 6 節:

1. **役割と境界** — 何をするか / **何をしないか** (他エージェントに渡すもの)
2. **走る機械** — 開発機 / 研究機 / 両方。研究機必須なら開発機での振られ方への対処も
3. **担当ファイル** — 実在するパスのみ
4. **固有知識・落とし穴** — そのエージェントだけが要る知識 (共通ルールに書けないもの)
5. **完了条件 (DoD)** — 何が緑なら完了か
6. **共通ルール参照** — `docs/agent-common.md` を読む旨

## S5. 検証方法

1. 各定義が挙げるファイルパスが**実在すること**を機械的に確認 (存在しないパスを書かない)
2. 数値主張を docs (`measurements-log.md` / `training-spec.md` / `roadmap.md`) と突合する
3. 廃止する `prompt-engineer` の内容 (日本語→タグ変換表) が `dataset-curator` に移設されたことを確認
4. (ユーザー承認の上で) `perf-profiler` を 1 回試験的に呼び、共通ルールを読んで自機 (開発機) を
   正しく判定し「研究機必須ゆえ著述まで」と返すかを確認する。承認が得られない場合は
   定義の記述レビュー (1〜3・5) のみで完了とする
5. `.claude/agents/*.md` の frontmatter が全件パースされ、エージェント一覧に 10 体が出ることを確認

## スコープ外 (別タスクとして記録)

改訂作業中に見つかったが、本 spec では扱わない。放置すると効くため別途決裁する。

- **`.claude/settings.json` の `permissions.allow` と `additionalDirectories` が全て旧パス**
  `e:\Develop\Projects\dollama` (実体は `E:\Projects\dollama`) → 許可エントリが 1 つもマッチせず死んでいる。
  併せて中身も probe 時代のワンショットコマンド 20 件超で、現在の作業 (meson/dotnet/python 訓練) と無関係。
- **hook の起動が `py -3.12` 固定** (環境は Python 3.14)。`dollama_protect_artifacts.py` は fail-open 設計
  ゆえ、3.12 が無ければ**静かに保護が無効化**される。正典重み保護という役割の重さと釣り合わない。

## 参照

- 現行定義: `.claude/agents/*.md` (11 体)
- 保護 hook: `.claude/hooks/dollama_protect_artifacts.py`
- 現状の真実源: `CLAUDE.md` / `docs/roadmap.md` / `docs/measurements-log.md` /
  `docs/training-spec.md` / `docs/dataset-spec.md` / `docs/fast-mode-plan.md` /
  `docs/f0b-rejection-sft-plan.md` / `docs/q2-quality-branch-plan.md`
- 二機体制: [[dev-pc-hardware]] / 重み搬送: [[weights-nas-transport]]

## 完了 (2026-08-05)

本 spec の内容は `docs/superpowers/plans/2026-08-05-subagent-refresh.md` の 9 タスクとして実装・完了した。

- **10 体構成になった**: `cpp-implementer` / `csharp-ui-implementer` / `cuda-kernel-dev` /
  `dataset-curator` / `gpu-benchmarker` / `model-converter` / `model-trainer` / `npu-benchmarker` /
  `perf-profiler` (新設) / `project-leader`。`pipeline-debugger` と `prompt-engineer` は削除
  (前者は perf-profiler が置換・後者の日本語→タグ写像は dataset-curator が引き取り)。
- **共通ルールを `docs/agent-common.md` に集約**。各定義は末尾 1 行で参照するのみとし、
  Allman 規約・使う/使わない表・テスト必須・正典保護・NAS 搬送・SAC 制約の重複記述を全定義から除去した。
- **`scripts/test_dollma_agent_defs.py` を新設し 9/9 緑**。frontmatter (name/description/tools/model)・
  tools 契約・model エイリアス限定・10 体の在不在・共通ルール参照・**本文パスの実在**・
  禁止語 (LibTorch / Winsock2 / openvino.runtime) を検証する。回帰テストとして常設され、
  今後の陳腐化を機械的に検出する。実際に着手時点で 3 件の存在しないパス参照を検出した。
- 実装中の逸脱は 1 点のみ: 検証スクリプトのパス実在チェックが `test_<component>.cpp` のような
  プレースホルダ表記を誤検出したため、`<` `>` を含むトークンを glob と同じくスキップ対象に加えた。

### 追補: `2d27d3d` の実装知識を拾い直し (2026-08-05)

並行改訂 `2d27d3d` を上書きした後、そこに含まれていた実装現場の知識を本構成へ統合した
(ユーザー判断)。10 体構成・共通ルール 1 枚・検証スクリプトの枠組みは維持したまま、中身を厚くした形。

| 移した先 | 拾った内容 |
|---|---|
| `docs/agent-common.md` | 研究機のビルド手順 (CUDA v13.3 の PATH・meson フルパス・**PowerShell では通らず Bash ツールが要る**)・**SAC の正しい運用** (`meson test` の前にユーザーへ OFF を依頼・`WinError 4551`・コードを疑う前に既存 exe で切り分け)・Edit 原則 (大きいファイルを Write で全文置換しない)・ライセンス非委譲・**生成スコープ** (キャラのみ / 背景タグ禁止 / 単独キャラ原則 / `simple background` はマッティング精度に効く) |
| `cuda-kernel-dev` | **数値パリティ規約**一式 (FP32 蓄積必須・tol は `1e-2*sqrt(K)`・入力ビット一致・row-major と transB・融合は memcmp bit-exact・**`launch_conv2d` の丸め列がGEMM/direct で違うためガードでフォールバック**・UNet は SSIM≥0.999/bad0 かつ default 無改変・**CFG 増幅下でゲートせず g=1.0 で分離**)・**多 TU リンク** (ヘッダ内 `__global__` の LNK2005・`DeviceWeights` の ODR 違反で `0xC0000005`・`#ifndef __CUDACC__` 隔離)・**新規 `.cu` の cuda_args** (`/utf-8` で C1070 回避・`/std:c++14` で cudafe++ の `0xC0000409` 回避・`-DHAVE_CUDA` は cuda_args 側)・cuBLAS は column-major をラッパーに封じ込め |
| `cpp-implementer` | **既存実装の確定挙動** (`tensor.hpp` の `data()`/`data_ptr()` は `logic_error` を投げる = サイレント nullptr を返さない防御を外さない・`UniqueBuffer` は move 実装済みでコピー禁止・LoRA host 写像の正典は offline merge) |
| `perf-profiler` | **大原則 4 つ** (計測なき最適化を許さない・同一条件の前後比較・**ノイズ床を先に測る**・パリティが壊れた高速化は改善でない)・prof 系計測 exe と `[RESNET-BUCKET]`・**攻め筋 G-10k (conv 真 batch2) / G-8k (im2col の cudaMalloc 撲滅)**・診断手順 6 段 (**改善余地が薄ければ「触るな」と結論するのも仕事**)・測定環境のドリフトは相対倍率で報告 |
| `model-converter` | 変換対象表 (**SDXL / 自作 LM は OV 変換しない** = 依頼が来たら差し戻す)・**入出力の申し送り義務** (報告に入力名/shape/element_type と出力名/shape/index を必ず含める) |
| `npu-benchmarker` / `gpu-benchmarker` | 位置づけと境界 (C++ 本線は実装済み・**Python / diffusers は golden 生成 / 新 checkpoint 下見 / HW 特性の 3 用途に限る**・自作パイプラインの profile は `perf-profiler` の担当) |
| `dataset-curator` | 廃止した `prompt-engineer` の規約 (**情景語を変換表に載せない**・情景を指定されたら `orange backlight` 等キャラに乗る要素へ落とす・スタイルプリセットはタッチのみで情景プリセットは持たない・**表情の忠実度の穴 3 層**) |

初版の誤記 1 件を訂正した: SAC 制約を「allow-list 更新を依頼」と書いていたが、正しくは
「**`meson test` の前に SAC を OFF にしてもらう** (即時反映・再起動不要)」。

**未処理 (スコープ外のまま)**: 上記「スコープ外」節の 2 件 — `.claude/settings.json` の
`permissions.allow` / `additionalDirectories` が旧パス `e:\Develop\Projects\dollama` で全滅している件と、
hook 起動が `py -3.12` 固定で fail-open ゆえ保護が静かに無効化され得る件。いずれも別途決裁する。

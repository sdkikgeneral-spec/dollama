# F-0b 再検証プラン (分割実行・Z-1〜Z-5)

> 2026-09-27 の記録監査 (record-auditor・BLOCK) が F-0b の**手続き不備**を指摘した。
> 本 doc はその是正を**セッションをまたいで拾える分割タスク台帳**にしたもの。
> 各 Package は担当/担当機/依存/入口出口が自己完結。次セッションは「現在地」を見て未完 Package から着手する。
> 元台帳 = `docs/f0b-rejection-sft-plan.md` (G-1〜G-3・実走記録)。本 doc は**その上に立つ再検証だけ**を扱う。

## 方針 (ユーザー承認済み)

**採用に仕向けるのではなく、手続きの不備 (anatomy 軸の分解能不足・seed 未制御・プラセボ対照の欠如) を
是正して公正に再検証する。結果次第で依然として不採用という結論もあり得る。**

- F-0b の**処置**(正典 `bitnet_dense` 無改変・SFT 重み隔離) は妥当だったので変更しない。
  再検証の対象は**主張の強さ**と、その主張を支える**手続き**のみ。
- 本 doc に「採用になる見込み」等の予断は書かない。Z-1〜Z-5 の結果は出るまで空欄 (status 🔲 未)。

## 是正対象 (監査 2026-09-27 の指摘と一次証拠)

| 系 | 監査が指摘した不備 | 一次証拠 (このセッションで確認済) |
|---|---|---|
| **A 系** | reward Δ の「~2.4σ」は**疑似反復**によるアーティファクト。日本語 54 件は同一プロンプト 1 組の複製 (uniq_prompts=1) → クラスタ補正で t=0.955・95%CI [−0.0115, +0.0348] (0 を含む) = **有意でない** | `data/rollouts/g2b_prepost.jsonl` の `lang=ja` 54 行は pre/post 各アームで `prompt` uniq=**1** (en は 46 行 uniq 46)。機構 = 日本語入力の空条件化 (`encode_text_greedy`) で greedy が同一プロンプトへ収束 |
| **B 系** | set-F1 退行の「構造的」は**同一 seed 内の一致**にすぎない。seed 間分散は未測定で、施策 D が seed ノイズと断じた最大幅 −0.0240 とほぼ同値 | `data/bitnet/_g2a_eval/eval_report_*.json` 6 本すべて provenance `"seed": 20260620`。`scripts/train_bitnet.py` の `--seed` 既定も 20260620。施策 D の −0.0240 は `docs/measurements-log.md` Phase 4-D 行 |
| **C 系** | 「不採用」の射程が広すぎる。報酬の実効寄与は quality 軸が支配 (監査再計算で anatomy 寄与率 7.5%)・anatomy 8 軸中 5 軸は死軸 → 棄却できるのは「この reward 設計・この単一 seed・この検出力での効果不検出」まで | `g2b_prepost.jsonl` 200 行の `axes` argmax = Limbs 148 / Hands 35 / Head 17 (**5/8 軸は一度も argmax にならない**)。Δ 内訳 quality +0.0155 / anatomy +0.0016 (元台帳 G-2b 行) |

> 数値のうち **t=0.955 / 95%CI / 寄与率 7.5%** は**監査 (2026-09-27) による再解析値**としての引用。
> 独立再現は **Z-2 / Z-1 の仕事**であり、本 doc 執筆時点では再現していない。

## 分割タスク台帳

| Pkg | 内容 | 担当 | 担当機 | 依存 | status |
|---|---|---|---|---|---|
| **Z-1** | **quality 抜き勝者一致率・寄与率の再計算**: best-of-N の勝者選抜が quality 単独でどれだけ決まるか (quality を除いた reward での勝者と一致する率) と、reward Δ への anatomy/quality 寄与率を既存データから算出。監査値 (anatomy 7.5%) の独立再現も兼ねる | model-trainer | 本機 | なし (既存データのみ・生成ゼロ) | 🔲 未 |
| **Z-2** | **g2b クラスタ補正の再解析 + σ_seed の直接推定**: `g2b_prepost.jsonl` をクラスタ (同一プロンプト) 単位で再解析し t / 95%CI を独立再現。加えて ja 54 件 (= 同一プロンプト × SDXL seed 変動) から **SDXL seed 由来の reward 分散 σ_seed** を直接推定する (GPU 実走の必要枚数を逆算するための入力) | model-trainer | 本機 | なし (既存データのみ・生成ゼロ) | 🔲 未 |
| **Z-3** | **set-F1 の per-case paired bootstrap CI**: `train_bitnet.py --eval-only --dump-persample` (CLI 実在・`scripts/train_bitnet.py`) で canon / SFT の per-case F1 を出し、対応付き bootstrap で退行 (−0.0174/−0.0241) の CI を出す。同一 seed 内での「退行が 0 と識別できるか」を明示 | model-trainer | 本機 | なし (隔離重み `bitnet_dense_sft*` 既存・再走は eval のみ) | 🔲 未 |
| **Z-4** | **プラセボ対照 SFT**: 勝者ペアを reward と無関係な選抜 (例: N 候補からランダム 1 本) に差し替えて同レシピで SFT し、set-F1 退行が **RAFT 固有か「低 LR 追加 SFT を掛けたこと」の副作用か**を切り分ける。訓練あり・SDXL 生成ゼロ | model-trainer | 本機 | Z-3 (物差しの CI が先) | 🔲 未 |
| **Z-5** | **SFT seed sweep 4 seed**: SFT アームを 4 seed (既存慣行 20260620/20260621/42/7) で焼き、diverse set-F1 の **seed 間分散**を測る → 退行幅 −0.0174/−0.0241 が seed noise 帯の内か外かを判定 (施策 A/D と同じ判定 3 軸: 符号一貫性 / 分散帯比 / 各 seed の paired CI)。訓練あり・SDXL 生成ゼロ | model-trainer | 本機 | Z-3 | 🔲 未 |
| **(保留)** | **GPU 実走 (seed 固定 pre/post reward 再測)**: SDXL seed を固定して疑似反復と seed 交絡を除いた reward 前後比を測り直す。**未起票** — 必要枚数を **Z-2 の σ_seed から逆算してから**起票する (逆算前に走らせると再び検出力不足になる) | 未定 (gpu-benchmarker 想定) | 研究機 (GPU) | Z-2 | ⏸ 保留 (未起票) |

> **Z-1〜Z-5 は全て本機・SDXL 生成ゼロ** (Z-4/Z-5 は LM の訓練のみ)。GPU 実走を要するのは保留行だけ。
> **安全弁 (継承)**: 正典 `bitnet_dense{,_fp32}`/identity/golden は無改変。実験成果物は別名 + scratch ディレクトリへ。

### Z-1 — quality 抜き勝者一致率・寄与率
- 入口: `data/rollouts/candidates_bestofn*.jsonl` (N 候補の axes/quality) + `data/rollouts/g2b_prepost.jsonl`。
- 出口: ① quality 抜き reward での勝者と現行勝者の一致率 ② reward Δ の anatomy/quality 寄与率 (監査値 7.5% の再現可否)。
- 意味: 「RAFT が学んだのは anatomy ではなく quality だった」を**数値で確定**させる (C 系の限定の土台)。

### Z-2 — クラスタ補正の再解析 + σ_seed 推定
- 入口: `data/rollouts/g2b_prepost.jsonl` (200 行)。
- 出口: ① クラスタ (同一プロンプト) 単位の t / 95%CI (監査値 t=0.955 の再現可否) ② **σ_seed** (同一プロンプトを SDXL seed だけ変えたときの reward ばらつき) ③ σ_seed から「Δ=+0.017 を検出するのに要る独立プロンプト数」の逆算。
- 意味: 保留中の GPU 実走の**枚数設計**がここで決まる。

### Z-3 — set-F1 per-case paired bootstrap CI
- 入口: 隔離重み `data/bitnet/bitnet_dense_sft_fp32.safetensors` + 正典 + 凍結 diverse-val (`data/bitnet/_g2a_eval/pairs.eval_diverse_{a,b}.jsonl`)。
- 出口: canon vs SFT の per-case F1 差の 95%CI (diverse_a / diverse_b)。seed は既存と同じ 20260620 に固定し **同一 seed 内の識別力**だけを見る (seed 間は Z-5)。

### Z-4 — プラセボ対照 SFT
- 入口: `data/rollouts/candidates_bestofn*.jsonl` (N=8 候補) + `train_bitnet.py --sft-rejection`。
- 出口: ランダム選抜 SFT の set-F1 4 指標。**RAFT 版と同幅で退行するなら退行は選抜方針ではなく追加 SFT の副作用**、有意に浅いなら reward 起因。
- 注意: 選抜以外 (LR / epoch / データ件数 / seed) を本採用条件と厳密に一致させる。

### Z-5 — SFT seed sweep (4 seed)
- 入口: Z-3 と同じ資産 + `--seed` 切り替え。
- 出口: 4 seed の diverse set-F1 (canon/SFT paired)・across 平均 ± sd・判定 3 軸の充足表。
- 意味: B 系の核。ここが揃うまで「退行は構造的」とは書けない (`docs/measurements-log.md` の施策 A/D と同じ作法)。

## 現在地 (最終更新: 2026-09-28)
- ✅ 監査 BLOCK (2026-09-27) の指摘を記録側で是正 (元台帳 / roadmap / measurements-log / CLAUDE.md / model-trainer.md)。
- 🔲 Z-1〜Z-5 は**すべて未実施**。着手は CLAUDE.md 実装ルール (プラン → 承認 → PL 振り分け) に従う。
- ⏸ GPU 実走は **Z-2 の σ_seed 待ちで未起票**。

## 参照
- 元台帳・実走記録: `docs/f0b-rejection-sft-plan.md`
- reward: `scripts/dollma_reward.py` / quality 枝: `docs/q2-quality-branch-plan.md`
- LM/SFT: `scripts/train_bitnet.py` (`--sft-rejection` / `--eval-only` / `--dump-persample` / `--seed` 既定 20260620)
- seed noise 判定の先例 (物差し): `docs/measurements-log.md` Phase 4-A (a12k) / Phase 4-D (容量増) 行
- F 全体: roadmap 「Phase 4」F 行 / CLAUDE.md 計測表 Phase 4 F 行 / [[project_phase4_F_status]]

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
| **A 系** | reward Δ の「~2.4σ」は**疑似反復**によるアーティファクト。日本語 54 件は同一プロンプト 1 組の複製 (uniq_prompts=1) → クラスタ補正で t=0.955・95%CI [−0.0115, +0.0348] (0 を含む) = **有意でない**。★**推定量 = 同一プロンプトを 1 観測に集約した unweighted cluster-mean の 1 標本 t 検定 (G=47)** (2026-09-28 追記・監査 中①) | `data/rollouts/g2b_prepost.jsonl` の `lang=ja` 54 行は pre/post 各アームで `prompt` uniq=**1** (en は 46 行 uniq 46)。機構 = 日本語入力の空条件化 (`encode_text_greedy`) で greedy が同一プロンプトへ収束 |
| **B 系** | set-F1 退行の「構造的」は**同一 seed 内の一致**にすぎない。seed 間分散は未測定で、施策 D が seed ノイズと断じた最大幅 −0.0240 とほぼ同値 | **訓練 seed (2026-09-28 追加・監査 軽微2)**: `data/bitnet/train_stats_sft.json` の `"seed": 20260620` (`"mode": "sft_rejection"` / epochs3・lr2e-05・batch32 / train 400) = SFT 訓練そのものの seed の厳密な記録。**評価 seed**: `data/bitnet/_g2a_eval/eval_report_*.json` 6 本すべて provenance `"seed": 20260620`。`scripts/train_bitnet.py` の `--seed` 既定も 20260620。施策 D の −0.0240 は `docs/measurements-log.md` Phase 4-D 行 |
| **C 系** | 「不採用」の射程が広すぎる。報酬の実効寄与は quality 軸が支配 (**Δ の 9 割超が quality**)・anatomy は argmax が 3 軸のみ / 軸別 max 値では Limbs 以外の 7 軸が死 → 棄却できるのは「この reward 設計・この単一 seed・この検出力での効果不検出」まで | `g2b_prepost.jsonl` 200 行の `axes` argmax = Limbs 148 / Hands 35 / Head 17 (**5/8 軸は一度も argmax にならない**)。**軸別 max (2026-09-28 再計算)**: Limbs 0.09347 / Head 0.00993 / Ears 0.00173 / Digits 0.00164 / Eyes 0.00154 / GlobalAnatomy 0.00150 / Hands 0.00141 / Mouth 0.00117 = F-0a と同じ物差し (max<0.012) では **Limbs 以外 7 軸が死**。Δ 内訳 quality +0.015494 / anatomy +0.001586 (元台帳 G-2b 行) |

> 数値のうち **t=0.955 / 95%CI** は**監査 (2026-09-27) による再解析値**の引用だが、**2026-09-28 に
> `g2b_prepost.jsonl` 200 行から独立再計算して一致を確認した** (naive t=2.4025 / cluster-mean t=0.9549 /
> **CR sandwich t=2.7592 (CR0)・2.7297 (CR1)** / **en46 のみ t=0.9154・Δ+0.0112**)。
> ★**「クラスタ補正」という語は推定量を一意に決めない** — cluster-mean と CR sandwich で**結論が逆になる**ので、
> 以後の記述では必ず推定量名を添えること。Z-2 はこの 3 通りを並べて出すのが出口条件 (下記)。
> ★**「anatomy 寄与率 7.5%」は撤回扱い (2026-09-28・監査 中③)**: 監査から引用した値だが定義が未記載で、
> Δ 基準 9.29% / 水準基準 1.13% のどちらでも再現しない。**主表現は「Δ の 9 割超が quality」**とし、
> 寄与率の**定義を決めて再計算するのは Z-1 の宿題**。

## 分割タスク台帳

| Pkg | 内容 | 担当 | 担当機 | 依存 | status |
|---|---|---|---|---|---|
| **Z-1** | **quality 抜き勝者一致率・寄与率の再計算**: best-of-N の勝者選抜が quality 単独でどれだけ決まるか (quality を除いた reward での勝者と一致する率) と、reward Δ への anatomy/quality 寄与率を**定義を明文化した上で**既存データから算出 (**監査引用値 7.5% は Δ 基準 9.29%/水準基準 1.13% のどちらでも再現しないため再現目標にしない** — 2026-09-28 是正) | model-trainer | 本機 | なし (既存データのみ・生成ゼロ) | 🔲 未 |
| **Z-2** | **g2b クラスタ補正の再解析 + σ_seed の直接推定**: `g2b_prepost.jsonl` をクラスタ (同一プロンプト) 単位で再解析し t / 95%CI を独立再現。加えて ja 54 件 (= 同一プロンプト × SDXL seed 変動) から **SDXL seed 由来の reward 分散 σ_seed** を直接推定する (GPU 実走の必要枚数を逆算するための入力) | model-trainer | 本機 | なし (既存データのみ・生成ゼロ) | 🔲 未 |
| **Z-3** | **set-F1 の per-case paired bootstrap CI**: `train_bitnet.py --eval-only --dump-persample` (CLI 実在・`scripts/train_bitnet.py`) で canon / SFT の per-case F1 を出し、対応付き bootstrap で退行 (−0.0174/−0.0241) の CI を出す。同一 seed 内での「退行が 0 と識別できるか」を明示 | model-trainer | 本機 | なし (隔離重み `bitnet_dense_sft*` 既存・再走は eval のみ) | 🔲 未 |
| **Z-4** | **プラセボ対照 SFT**: 勝者ペアを reward と無関係な選抜 (例: N 候補からランダム 1 本) に差し替えて同レシピで SFT し、set-F1 退行が **RAFT 固有か「低 LR 追加 SFT を掛けたこと」の副作用か**を切り分ける。訓練あり・SDXL 生成ゼロ | model-trainer | 本機 | Z-3 (物差しの CI が先) | 🔲 未 |
| **Z-5** | **SFT seed sweep 4 seed**: SFT アームを 4 seed (既存慣行 20260620/20260621/42/7) で焼き、diverse set-F1 の **seed 間分散**を測る → 退行幅 −0.0174/−0.0241 が seed noise 帯の内か外かを判定 (施策 A/D と同じ判定 3 軸: 符号一貫性 / 分散帯比 / 各 seed の paired CI)。訓練あり・SDXL 生成ゼロ | model-trainer | 本機 | Z-3 | 🔲 未 |
| **(保留)** | **GPU 実走 (seed 固定 pre/post reward 再測)**: SDXL seed を固定して疑似反復と seed 交絡を除いた reward 前後比を測り直す。**未起票** — 必要枚数を **Z-2 の σ_seed から逆算してから**起票する (逆算前に走らせると再び検出力不足になる) | 未定 (gpu-benchmarker 想定) | 研究機 (GPU) | Z-2 | ⏸ 保留 (未起票) |

> **Z-1〜Z-5 は全て本機・SDXL 生成ゼロ** (Z-4/Z-5 は LM の訓練のみ)。GPU 実走を要するのは保留行だけ。
> **安全弁 (継承)**: 正典 `bitnet_dense{,_fp32}`/identity/golden は無改変。実験成果物は別名 + scratch ディレクトリへ。

### Z-1 — quality 抜き勝者一致率・寄与率
- 入口: `data/rollouts/candidates_bestofn*.jsonl` (N 候補の axes/quality) + `data/rollouts/g2b_prepost.jsonl`。
- 出口: ① quality 抜き reward での勝者と現行勝者の一致率 ② reward Δ の anatomy/quality 寄与率 — ★**まず「寄与率」の定義を明文化してから計算する** (Δ 基準 = Δanatomy/(Δanatomy+Δquality) なら 9.29%、水準基準 = 平均絶対寄与比なら 1.13%。監査引用値 7.5% はどちらでも再現しないため、**7.5% の再現を目標にしない**)。
- 意味: 「RAFT が学んだのは anatomy ではなく quality だった」を**数値で確定**させる (C 系の限定の土台)。

### Z-2 — クラスタ補正の再解析 + σ_seed 推定
- 入口: `data/rollouts/g2b_prepost.jsonl` (200 行)。
- 出口: ① ★**推定量を明示して 3 通りを並べる**: **(i) naive (n=100 独立扱い) / (ii) unweighted cluster-mean (G=47) / (iii) cluster-robust sandwich (CR0/CR1・G=47)** の t / 95%CI を全部出し、**どれを結論の根拠に採るかを理由付きで選ぶ** (2026-09-28 時点の暫定再計算は 2.4025 / 0.9549 / 2.7592・2.7297)。**加えて en のみ 46 プロンプト (疑似反復ゼロ) の t / CI を独立系列として必ず併記** (暫定 t=0.9154・Δ+0.0112)。**「クラスタ補正で有意でない」と書くだけの出口は不可** (推定量で結論が逆になる) ② **σ_seed** (同一プロンプトを SDXL seed だけ変えたときの reward ばらつき) ③ σ_seed から「Δ=+0.017 を検出するのに要る独立プロンプト数」の逆算。
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
- ✅ 監査 BLOCK (2026-09-27) の指摘を記録側で是正 (元台帳 / roadmap / measurements-log / CLAUDE.md / model-trainer.md) = commit `351648d` / `951148b`。
- ✅ **その是正への 2 巡目監査 BLOCK (2026-09-28・重大1/中3/軽微2) を是正**: ① `docs/superpowers/plans/2026-08-05-subagent-refresh.md` に残っていた撤回済み断定 (「set-F1 が構造的に退行」) をポインタ化 (**同 plan は 36 チェックボックス全未チェック = 再実行可能体裁なので、agent 定義へ旧断定が復活する経路だった**) ② t=0.955 の**推定量を全該当 doc に明記** + en46 独立系列を併記 ③ 「anatomy 8 軸中 5 軸が死軸」を F-0a と同じ物差し (軸別 max) へ統一 ④ 「anatomy 寄与率 7.5%」を格下げ ⑤ 訓練 seed の一次証拠に `train_stats_sft.json` を追加 ⑥ 下記のバイト数訂正。
- ⚠️ **commit `351648d` 本文の自己申告「+1.5KB 増」は不正確 (2026-09-28 訂正)**。commit 本文は書き換えないので、ここに実測を残す:
  CLAUDE.md は git blob (LF) で **34,215 → 35,394 バイト = 追加 1,626 / 削除 447 / net +1,179**
  (作業ツリーは CRLF ゆえ 34,560 → 35,741 = +1,181)。一次証拠 = `git show 351648d~1:CLAUDE.md | wc -c` / `git show 351648d:CLAUDE.md | wc -c` / `git diff 351648d~1 351648d -- CLAUDE.md` の追加・削除行のバイト数。
  なお本 2 巡目の是正で CLAUDE.md F-0b 行はさらに増える (推定量と死軸の物差しを書き足したため。**どちらも監査が「これを書かないと誤読が復活する」と指摘した実体**であり、ポインタで代替できない)。
- 🔲 Z-1〜Z-5 は**すべて未実施**。着手は CLAUDE.md 実装ルール (プラン → 承認 → PL 振り分け) に従う。
- ⏸ GPU 実走は **Z-2 の σ_seed 待ちで未起票**。
- 🔲 **未起票の別残債 (Q-2 由来)**: E-2 の clean/clutter 分離判定は**入力題材基準**で計算されており F-0a の 4x (**生成 prompt 基準**) とは被験変数が違う = **分離維持/悪化は未判定**。詳細と一次証拠は `docs/q2-quality-branch-plan.md` Package E 節の 2026-09-28 注記。**Q-2 のゲート通過自体は std 0.1038>0.1 で正当**。

## 参照
- 元台帳・実走記録: `docs/f0b-rejection-sft-plan.md`
- reward: `scripts/dollma_reward.py` / quality 枝: `docs/q2-quality-branch-plan.md`
- LM/SFT: `scripts/train_bitnet.py` (`--sft-rejection` / `--eval-only` / `--dump-persample` / `--seed` 既定 20260620)
- seed noise 判定の先例 (物差し): `docs/measurements-log.md` Phase 4-A (a12k) / Phase 4-D (容量増) 行
- F 全体: roadmap 「Phase 4」F 行 / CLAUDE.md 計測表 Phase 4 F 行 / [[project_phase4_F_status]]

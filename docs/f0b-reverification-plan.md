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

> 数値のうち **t 値 4 本は 2026-09-28 に `g2b_prepost.jsonl` 200 行から独立再計算して一致を確認した**
> (naive t=2.4025 / cluster-mean t=0.9549 / **CR sandwich t=2.7592 (CR0)・2.7297 (CR1)** /
> **en46 のみ t=0.9154・Δ+0.0112**)。
> ★**cluster-mean の 95%CI [−0.0115, +0.0348] は監査 (2026-09-27) からの引用で、本セッションでは再現していない**
> (2026-09-28 是正・監査 中1。**上表 A 系セルの CI もこの引用値**)。同じ 200 行の recompute は
> **Δ+0.011386 / se 0.011923 / G=47** で、ここから引いた CI は **t(46) 基準 [−0.0126, +0.0354] /
> z 基準 [−0.0120, +0.0348]** = **下端が記録値 (−0.0115) と一致しない** (上端 0.0348 は z 基準と一致)。
> **「0 を含む = 有意でない」という結論はどの分位点でも変わらない** (t=0.9549 < t(46)0.975 分位 2.0129) が、
> **CI の数値を引くときは分位点を明記し、記録値が未再現であることを承知で引くこと**。CI の確定は Z-2 の宿題。
> → **2026-09-29 Z-2 の帰結 (下記「### Z-2」節)**: 解析的 CI は Z-2 でも **t(46) [−0.01261, +0.03539] / z [−0.01198, +0.03475]** で、
> 下端 −0.0115 は**再現しない**。cluster bootstrap percentile CI (B=20000) では下端が乱数 seed により
> −0.0111〜−0.0119 に振れ、−0.0115 はその幅に入る。ただし記録値の計算条件 (手法・B・seed) が残っていないので、
> **これは「bootstrap なら出うる値」という示唆にとどまり、再現の証明ではない**。記録値 [−0.0115, +0.0348] は**出自未確定の引用値のまま**扱う。
> cluster-mean 推定量の CI が「0 を含む」ことは、t / z / bootstrap のどれでも同じ (CR sandwich の CI は 0 を含まない。推定量が違う)。
> なお **en46 の 95%CI は t(45) 基準で [−0.0134, +0.0357] = 記録値に一致** (本セッションで独立再計算済)。
> ★**「クラスタ補正」という語は推定量を一意に決めない** — cluster-mean と CR sandwich で**結論が逆になる**ので、
> 以後の記述では必ず推定量名を添えること。Z-2 はこの 3 通りを並べて出すのが出口条件 (下記)。
> ★**「anatomy 寄与率 7.5%」は撤回扱い (2026-09-28・監査 中③)**: 監査から引用した値だが定義が未記載で、
> Δ 基準 9.29% / 水準基準 1.13% のどちらでも再現しない。**主表現は「Δ の 9 割超が quality」**とし、
> 寄与率の**定義を決めて再計算するのは Z-1 の宿題**。
> → **2026-09-29 Z-1 で定義を明文化して再計算済み** (下記「### Z-1」節の「結果」)。
> **Δ 基準 (主) anatomy 9.287% / 水準基準 (副) anatomy 1.126%** で、上の 9.29% / 1.13% と一致する。
> 7.5% はどちらの定義でも再現しないので、**撤回扱いのまま**とする。

## 分割タスク台帳

| Pkg | 内容 | 担当 | 担当機 | 依存 | status |
|---|---|---|---|---|---|
| **Z-1** | **quality 抜き勝者一致率・寄与率の再計算**: best-of-N の勝者選抜が quality 単独でどれだけ決まるか (quality を除いた reward での勝者と一致する率) と、reward Δ への anatomy/quality 寄与率を**定義を明文化した上で**既存データから算出 (**監査引用値 7.5% は Δ 基準 9.29%/水準基準 1.13% のどちらでも再現しないため再現目標にしない** — 2026-09-28 是正) | model-trainer | 本機 | なし (既存データのみ・生成ゼロ) | ✅ 算出済 (2026-09-29・数値と定義のみ・**判定なし**。結果は下記 Z-1 節) |
| **Z-2** | **g2b クラスタ補正の再解析 + σ_seed の直接推定**: `g2b_prepost.jsonl` をクラスタ (同一プロンプト) 単位で再解析し t / 95%CI を独立再現。加えて ja 54 件 (= 同一プロンプト × SDXL seed 変動) から **SDXL seed 由来の reward 分散 σ_seed** を直接推定する (GPU 実走の必要枚数を逆算するための入力) | model-trainer | 本機 | なし (既存データのみ・生成ゼロ) | ✅ 算出済 (2026-09-29・**判定なし**。出口①の「どれを結論の根拠に採るか」の選択も未了。結果は下記 Z-2 節) |
| **Z-3** | **set-F1 の per-case paired bootstrap CI**: `train_bitnet.py --eval-only --dump-persample` (CLI 実在・`scripts/train_bitnet.py`) で canon / SFT の per-case F1 を出し、対応付き bootstrap で退行 (−0.0174/−0.0241) の CI を出す。同一 seed 内での「退行が 0 と識別できるか」を明示 | model-trainer | 本機 | なし (隔離重み `bitnet_dense_sft*` 既存・再走は eval のみ) | 🔲 未 |
| **Z-4** | **プラセボ対照 SFT**: 勝者ペアを reward と無関係な選抜 (例: N 候補からランダム 1 本) に差し替えて同レシピで SFT し、set-F1 退行が **RAFT 固有か「低 LR 追加 SFT を掛けたこと」の副作用か**を切り分ける。訓練あり・SDXL 生成ゼロ | model-trainer | 本機 | Z-3 (物差しの CI が先) | 🔲 未 |
| **Z-5** | **SFT seed sweep 4 seed**: SFT アームを 4 seed (既存慣行 20260620/20260621/42/7) で焼き、diverse set-F1 の **seed 間分散**を測る → 退行幅 −0.0174/−0.0241 が seed noise 帯の内か外かを判定 (施策 A/D と同じ判定 3 軸: 符号一貫性 / 分散帯比 / 各 seed の paired CI)。訓練あり・SDXL 生成ゼロ | model-trainer | 本機 | Z-3 | 🔲 未 |
| **(保留)** | **GPU 実走 (seed 固定 pre/post reward 再測)**: SDXL seed を固定して疑似反復と seed 交絡を除いた reward 前後比を測り直す。**未起票** — 必要枚数を **Z-2 の σ_seed から逆算してから**起票する (逆算前に走らせると再び検出力不足になる) | 未定 (gpu-benchmarker 想定) | 研究機 (GPU) | Z-2 | ⏸ 保留 (σ_seed 算出済・起票は未。枚数は未決定) |

> **Z-1〜Z-5 は全て本機・SDXL 生成ゼロ** (Z-4/Z-5 は LM の訓練のみ)。GPU 実走を要するのは保留行だけ。
> **安全弁 (継承)**: 正典 `bitnet_dense{,_fp32}`/identity/golden は無改変。実験成果物は別名 + scratch ディレクトリへ。

### Z-1 — quality 抜き勝者一致率・寄与率
- 入口: `data/rollouts/candidates_bestofn*.jsonl` (N 候補の axes/quality) + `data/rollouts/g2b_prepost.jsonl`。
- 出口: ① quality 抜き reward での勝者と現行勝者の一致率 ② reward Δ の anatomy/quality 寄与率 — ★**まず「寄与率」の定義を明文化してから計算する** (Δ 基準 = Δanatomy/(Δanatomy+Δquality) なら 9.29%、水準基準 = 平均絶対寄与比なら 1.13%。監査引用値 7.5% はどちらでも再現しないため、**7.5% の再現を目標にしない**)。
- 意味: 「RAFT が学んだのは anatomy ではなく quality だった」を**数値で確定**させる (C 系の限定の土台)。

#### Z-1 結果 (2026-09-29 算出・**数値と定義のみ。採否・上記「意味」の成否の判定はしていない**)
- **一次成果物**: スクリプト `scripts/dollma_f0b_z1_quality_contribution.py` / 出力 `docs/logs/f0b-z1/z1_result.json` /
  実行記録 `docs/logs/f0b-z1/z1_run_log.txt` (以上 3 本は commit `11962db`) / 出自と再現手順 `docs/logs/f0b-z1/README.md`。
  既存データの再解析のみで、SDXL 生成も訓練もしていない。
- **入力** (sha256 は JSON `inputs.*.sha256`。record-writer が 2026-09-29 に `sha256sum` で再計算し、3 本とも一致を確認):
  `data/rollouts/candidates_bestofn.jsonl` (G-1 の best-of-8 候補 3200 行 = 400 group × N=8) /
  `data/rollouts/sft_bestofn.jsonl` (400 行・lang の結合に使う) / `data/rollouts/g2b_prepost.jsonl` (G-2b の held-out 100 入力 × pre/post = 200 行)。
- **定義** (スクリプト冒頭の docstring と JSON `contribution.*.formula` / `contribution.primary_choice_reason` に明文化してある):
  - reward 式は `scripts/dollma_reward.py` と同じ。`anatomy_reward = -(0.95·max(axes) + 0.05·mean(axes))`、
    `reward = 0.6·anatomy_reward + 0.4·(quality − 1)`。
  - **quality 抜き reward := `anatomy_reward` のみ**。各 group で `anatomy_reward` の argmax を「quality 抜き勝者」とし、
    `is_winner=True` の現行勝者と一致した group の割合を**勝者一致率**とする。
    タイ方針は「float 完全一致のときだけ同率とみなし、最小 candidate index を採る」(JSON `winner_match.tie_policy`)。
  - **寄与率・Δ 基準 (主)** = |mean Δanatomy_contribution| / (|mean Δanatomy_contribution| + |mean Δquality_contribution|)。
    `g2b_prepost.jsonl` の post_id ごとの pre→post 差 (100 ペア) を使う。
  - **寄与率・水準基準 (副)** = mean|anatomy_contribution| / (mean|anatomy_contribution| + mean|quality_contribution|)。
    同じファイルの 200 行 (pre + post) を使う。
  - 主を Δ 基準にしたのは**スクリプト実行者の選択**で、理由は JSON `contribution.primary_choice_reason` にある
    (F-0b の主張は pre→post の変化の内訳なので、変化量を分母分子に置く方が直接対応する。水準基準は anatomy と quality の
    スケール差を受ける)。PL の決裁は経ていない。
- **数値**:

  | 指標 | 値 | JSON キー |
  |---|---|---|
  | 勝者一致率 (全体) | **44/400 = 11.0%** | `winner_match.overall` |
  | 勝者一致率 (ja) | **14/184 = 7.6%** | `winner_match.ja` |
  | 勝者一致率 (en) | **30/216 = 13.9%** | `winner_match.en` |
  | タイ group 数 | 0 | `winner_match.{overall,ja,en}.n_tie_groups` |
  | is_winner 異常 group (勝者フラグが 1 本でない) | 0 | `winner_match.overall.n_groups_official_winner_anomaly` |
  | lang を結合できなかった group | 0 | `winner_match.n_groups_lang_unknown` |
  | 検算: 再計算した full reward の argmax と is_winner の不一致 | **0/400** | `winner_match.sanity_full_reward_argmax_mismatch_vs_is_winner` |
  | ランキングから除外した空候補 (`axes=null`) | 1 件 | `winner_match.n_empty_candidates_excluded_from_ranking` |
  | Δ reward mean (100 ペア) | +0.017081 | `contribution.delta_reward_mean` |
  | Δ anatomy_contribution mean / Δ quality_contribution mean | +0.001586 / +0.015494 | `contribution.delta_{anatomy,quality}_contribution_mean` |
  | **寄与率 Δ 基準 (主)**: anatomy / quality | **9.287% / 90.713%** | `contribution.delta_based.{anatomy,quality}_share` |
  | mean\|anatomy_contribution\| / mean\|quality_contribution\| (200 行) | 0.003253 / 0.285610 | `contribution.level_based.mean_abs_*` |
  | **寄与率 水準基準 (副)**: anatomy / quality | **1.126% / 98.874%** | `contribution.level_based.{anatomy,quality}_share` |

- **record-writer による独立検算 (2026-09-29・スクリプトを通さず生データから再計算)**:
  勝者一致率 ja 14/184・en 30/216、argmax 不一致 0、タイ 0 がすべて再現した。
  `candidates_bestofn.jsonl` の保存済み `reward` 列と、上の式による再計算値は全候補で差 0 だった。
  寄与率 0.09287305891828718 / 0.01125984005772925 も再現した。
  既記載の 9.29% / 1.13% (上の注記・`docs/measurements-log.md` F-0b 行 ③) とも一致する。
- **読み方の注意** (★引用するときに誤読しやすい点):
  - **一致率 11.0% は「quality の寄与が 89%」という意味ではない**。「quality 項を抜くと、400 group 中 356 group で勝者が入れ替わる」という意味である。
  - 参考までに、8 候補から一様ランダムに選んだときの期待一致率は 1/8 = 12.5% (空候補のある 1 group は 1/7)。
    これは算術上の比較値で、**検定はしていない**。
  - Δ 基準は符号を落として比べている (絶対値)。全体では Δanatomy と Δquality の mean がどちらも正なので、影響はない。
    ただし en 46 ペアだけで見ると Δanatomy mean は **−0.000646** で負 (JSON `contribution.distributions.delta_anatomy_contribution_en.mean`)。
    言語別に寄与率を引き直すときは、符号の扱いを先に決めること (本 Z-1 では言語別の寄与率は出していない)。
- **限界 (疑義として残す・未検証の部分を明示)**:
  1. **lang の結合**: `candidates_bestofn.jsonl` には lang 列がない。そのため `sft_bestofn.jsonl` の `meta.post_id → lang` で結合した。
     record-writer が確認できたのは次の 2 点。
     - 結合の整合: 3200 行すべてで `input_text` が `sft_bestofn.jsonl` の `text` と一致した。
     - lang ラベルと本文の整合: 400 件すべてで、lang=ja と「本文に仮名・漢字を含む」が一致した。

     **lang ラベルそのものの出自** (上流データの lang 列の付け方) は**未検証**。
  2. **空候補 1 件の除外**: post_id 11615011 の candidate 0 (`empty=true`・`axes=null`・非勝者・en)。
     スコアが付けられないので、quality 抜きランキングから外し、この group は 7 候補で比べた。
     **この除外方針は本 plan に明記されていない** (スクリプト側の判断)。
     影響は最大でも 1 group (= 一致率 ±0.25pt) だが、方針として承認されたものではない。
  3. **一致率と寄与率は別データセット**:
     - 一致率は G-1 の訓練側 rollout (`candidates_bestofn.jsonl`・400 入力) から出している。
     - 寄与率は G-2b の held-out (`g2b_prepost.jsonl`・100 入力) から出している。
     - 両者の post_id の重なりは **0** (record-writer 確認)。

     **2 つを合算したり、同じ母集団の値として並べて比べたりしないこと**。
     なお、`g2b_prepost.jsonl` にある ja の疑似反復 (54 行が同一プロンプト・上記 A 系) は、Δ 基準・水準基準の両方の母数にそのまま含まれている。
     これを補正した寄与率は出していない。
     一方 `candidates_bestofn.jsonl` の ja は、1472 行で prompt が 1471 種ある (record-writer 確認)。つまり g2b 型の同一プロンプト収束は起きていない。
  4. **採否・意味付けの判定はしていない**。上記「意味」の一文 (「RAFT が学んだのは anatomy ではなく quality だった」) が
     この数値で確定したかどうかは、本節では判断しない。
- **行番号の注意**: スクリプト docstring と JSON `plan_ref` の「l.45, l.55-58」は、**本 doc を 2026-09-29 に更新する前の行番号**である。
  今回の加筆で行がずれているため、節名 (「分割タスク台帳」の Z-1 行 / 「### Z-1」節) で辿ること。

### Z-2 — クラスタ補正の再解析 + σ_seed 推定
- 入口: `data/rollouts/g2b_prepost.jsonl` (200 行)。
- 出口: ① ★**推定量を明示して 3 通りを並べる**: **(i) naive (n=100 独立扱い) / (ii) unweighted cluster-mean (G=47) / (iii) cluster-robust sandwich (CR0/CR1・G=47)** の t / 95%CI を全部出し、**どれを結論の根拠に採るかを理由付きで選ぶ** (2026-09-28 時点の暫定再計算は 2.4025 / 0.9549 / 2.7592・2.7297)。**加えて en のみ 46 プロンプト (疑似反復ゼロ) の t / CI を独立系列として必ず併記** (暫定 t=0.9154・Δ+0.0112)。**「クラスタ補正で有意でない」と書くだけの出口は不可** (推定量で結論が逆になる) ② **σ_seed** (同一プロンプトを SDXL seed だけ変えたときの reward ばらつき) ③ σ_seed から「Δ=+0.017 を検出するのに要る独立プロンプト数」の逆算。
- 意味: 保留中の GPU 実走の**枚数設計**がここで決まる。

#### Z-2 結果 (2026-09-29 算出・**数値のみ。有意・採否・GPU 実走の枚数決定はしていない**)
- **一次成果物** (commit `985a409`): スクリプト `scripts/dollma_f0b_z2_seed_variance.py` / 出力 `docs/logs/f0b-z2/z2_result.json` /
  実行記録 `docs/logs/f0b-z2/z2_run_log.txt` / 出自と再現手順 `docs/logs/f0b-z2/README.md`。
  既存データの再解析のみで、SDXL 生成も訓練もしていない。
- **入力**: `data/rollouts/g2b_prepost.jsonl` (200 行 = 100 ペア)。sha256 `ba2db3be…9e89` (JSON `inputs.g2b_prepost.sha256`)。
  record-writer が 2026-09-29 に `sha256sum` で再計算して一致を確認した。Z-1 と同一ファイル。
- **クラスタ定義**: `(pre.prompt, post.prompt)` の組をキーにする (lang は使わない)。結果は ja 54 ペアが 1 クラスタ、en 46 ペアが各 1 件のクラスタで、
  **G=47** (JSON `exit1_estimators.cluster_mean.{G,cluster_sizes_distinct}`)。
- **出口① 推定量別の Δreward** (JSON `exit1_estimators.*`):

  | 推定量 | Δ | se | t | df | p (両側) | 95%CI (分位点) |
  |---|---|---|---|---|---|---|
  | (i) naive (n=100 を独立扱い) | +0.017081 | 0.007110 | **2.4025** | 99 | 0.0181 | t(99) [+0.00297, +0.03119] / z [+0.00315, +0.03102] |
  | (ii) unweighted cluster-mean (G=47) | +0.011386 | 0.011923 | **0.9549** | 46 | 0.3446 | t(46) [−0.01261, +0.03539] / z [−0.01198, +0.03475] / cluster bootstrap percentile (B=20000・seed 20260620) [−0.01192, +0.03475] |
  | (iii) CR sandwich CR0 (G=47) | +0.017081 | 0.006190 | **2.7592** | 46 | 0.0083 | t(46) [+0.00462, +0.02954] |
  | (iii) CR sandwich CR1 (G=47) | +0.017081 | 0.006257 | **2.7297** | 46 | 0.0090 | t(46) [+0.00449, +0.02968] |
  | en のみ 46 ペア (疑似反復なし) | +0.011152 | 0.012183 | **0.9154** | 45 | 0.3649 | t(45) [−0.01338, +0.03569] / z [−0.01273, +0.03503] |

  t 値 5 本は本 doc 冒頭注記の 2026-09-28 記録値と一致する。
- **出口② σ_seed** (同一プロンプト文字列を SDXL に繰り返し投げたときの reward の sd。ja 54 件から。JSON `exit2_sigma_seed.*`):

  | 対象 | sd | 95%CI χ² | 95%CI bootstrap (B=20000・seed 20260620) |
  |---|---|---|---|
  | reward pre (n=54) | 0.04091 | [0.03439, 0.05050] | [0.03171, 0.04846] |
  | reward post (n=54) | 0.05172 | [0.04348, 0.06385] | [0.04156, 0.05969] |
  | **reward pooled (df=106)** | **0.04663** | [0.04111, 0.05388] | [0.03984, 0.05209] |
  | quality_contribution pooled | 0.04640 | [0.04090, 0.05361] | [0.03958, 0.05184] |
  | anatomy_contribution pooled | 0.001364 | [0.001203, 0.001576] | [0.001078, 0.001596] |
  | anatomy_contribution pre / post | 0.001928 / **0.0000667** | — | — |

- **出口③ 必要な独立プロンプト数の逆算** (JSON `exit3_required_n`):
  - 式: n = ceil((z₀.₉₇₅ + z₀.₈)² · Var(Δᵢ) / Δ²)。定数 (1.95996 + 0.84162)² = 7.8489。α=0.05 両側・検出力 0.8。t 分布の補正はしていない。
  - Var(Δᵢ) の中身:
    - 設計 (a) = pre/post で seed が別 (現行と同じ): τ² + 2σ_seed²/k (k = 1 プロンプトあたりの seed 数)。
    - 設計 (b) = seed を固定して pre/post を撮る: τ² のみ (**ρ=1 = seed ノイズが完全に相殺される楽観仮定**)。
      ρ=0 なら (a) と同じ式になる。
  - τ² (プロンプト間の真の効果ばらつき) = Var(Δ_en46) − 2σ_seed² = 0.006827 − 2×0.002174 = **0.002478** (負にならず打ち切りなし)。
  - 総枚数 = プロンプト数 × k × 2 アーム。

  | 検出したい Δ | 設計 | k=1 | k=2 | k=4 | k=8 | k=16 |
  |---|---|---|---|---|---|---|
  | +0.017081 (naive 全体) | (a) プロンプト / 枚 | **184 / 368** | 126 / 504 | 96 / 768 | 82 / 1312 | 74 / 2368 |
  | +0.017081 (naive 全体) | (b) ρ=1 楽観 プロンプト / 枚 | **67 / 134** | 67 / 268 | 67 / 536 | 67 / 1072 | 67 / 2144 |
  | +0.011152 (en46) | (a) プロンプト / 枚 | **431 / 862** | 294 / 1176 | 226 / 1808 | 191 / 3056 | 174 / 5568 |
  | +0.011152 (en46) | (b) ρ=1 楽観 プロンプト / 枚 | **157 / 314** | 157 / 628 | 157 / 1256 | 157 / 2512 | 157 / 5024 |

- **record-writer による独立検算** (2026-09-29): スクリプトを通さず、生データから numpy で直接計算した。
  - 次の値がすべて再現した: G=47 (クラスタサイズは 1 と 54) / cluster-mean t / CR0 t / en46 の Δ・t・分散 /
    ja の prompt uniq (pre・post とも 1) / σ_seed の pre・post・pooled / anatomy post sd / τ² /
    必要プロンプト数 (Δ 2 種 × k=1・4・16 と設計 (b))。
  - cluster bootstrap の seed 依存を見るため、スクリプトの `_cluster_bootstrap_ci` を seed 10 通り (20260620, 0〜7, 42) で走らせた。
    B=20000 で下端 −0.01106〜−0.01192、上端 +0.03426〜+0.03507 だった。この値はログに残していない (本節が唯一の記録)。
- **読み方の注意** (★引用するときに誤読しやすい点):
  - **出口①の「どれを結論の根拠に採るかを理由付きで選ぶ」は Z-2 では済んでいない**。5 系列を並べただけである。
    推定量によって t が 0.92〜2.76 と割れる状況は変わっていない。選択は別途決裁が要る。
  - **記録値 CI [−0.0115, +0.0348] は再現していない**。解析的 CI (t・z) では下端が合わない。
    bootstrap では −0.0115 が seed のばらつきの幅に入るが、元の計算条件が残っていないので再現の証明にはならない。
    なお `docs/logs/f0b-z2/README.md` にある「seed によって −0.0114〜−0.0117」という幅は、実行記録に出典がない。
    上の record-writer の 10 seed 走行とも一致しない (より狭い)。
  - **Δ+0.017081 は naive 平均である**。ja の疑似反復 54 ペアを含んだまま計算している (ja のみの Δ 平均は +0.022131・JSON `exit1_estimators.ja_all_delta_mean`)。
    これは本節の出口③が「Δ=+0.017」を目標に指定したので置いた値で、**真の効果量として検証された値ではない**。
  - 設計 (a) の k=1 の Var(Δᵢ) は、定義上 en46 の観測分散 0.006827 そのものになる。
    設計 (b) ではモデル上 seed の寄与が 0 なので、k を増やしてもプロンプト数は減らず、枚数だけが増える。
  - **anatomy_contribution の post sd 0.0000667 は極小**である (pre は 0.001928)。
    「post が解剖的に安定した画像だけを出す」のか「ScorerNet の分解能不足でほぼ同じ値しか出ない」のかは、切り分けていない (JSON `limitations`)。
- **限界 (疑義として残す・未検証の部分を明示)**:
  1. **σ_seed の根拠は ja の単一プロンプト 1 組だけ**である (pre 側 1 種・post 側 1 種)。
     en プロンプトには「同一プロンプト × 複数 seed」の測定がないので、**en への一般化は未検証**。
     τ² と出口③の全行はこの一般化を前提にしている。
  2. **ρ (seed 固定時の pre/post 相関) の実測はない**。現行データでは pre/post が常に別 seed なので観測できない
     (seed は server wall-clock 由来で非制御。`scripts/dollma_g2b_reward_prepost.py` docstring「ペアの扱い (重要な制約)」)。
     設計 (b) の行は楽観側の上限として読むこと。
  3. ja 54 件は post_id が 54 通りとも違う。つまり「同一 post_id を seed だけ変えて撮り直した測定」ではなく、
     「同一プロンプト文字列に対する独立な SDXL 呼び出し」のばらつきである。
  4. pre と post の σ_seed は 0.0409 と 0.0517 で異なる。pooled はこの 2 つが同じだという前提に立つが、その前提は検定していない。
  5. CR sandwich の df は G−1 (=46) だけを採った (Cameron & Miller 2015 の慣行)。Satterthwaite 等の代替 df は計算していない。
  6. スクリプト docstring の「出口 (5): 全ての主要数値を 2 経路で計算」は実装と合わない。
     実際の `cross_check` が照合するのは **naive の mean と t だけ**である (関数 `cross_check`・run log も「naive mean/t」と記載)。
     他の値の 2 経路検算は、上の record-writer 独立検算が代わりになっている。

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

## 現在地 (最終更新: 2026-09-29・Z-2 反映)
- ✅ 監査 BLOCK (2026-09-27) の指摘を記録側で是正 (元台帳 / roadmap / measurements-log / CLAUDE.md / model-trainer.md) = commit `351648d` / `951148b`。
- ✅ **その是正への 2 巡目監査 BLOCK (2026-09-28・重大1/中3/軽微2) を是正**: ① `docs/superpowers/plans/2026-08-05-subagent-refresh.md` に残っていた撤回済み断定 (「set-F1 が構造的に退行」) をポインタ化 (**同 plan は 36 チェックボックス全未チェック = 再実行可能体裁なので、agent 定義へ旧断定が復活する経路だった**) ② t=0.955 の**推定量を全該当 doc に明記** + en46 独立系列を併記 ③ 「anatomy 8 軸中 5 軸が死軸」を F-0a と同じ物差し (軸別 max) へ統一 ④ 「anatomy 寄与率 7.5%」を格下げ ⑤ 訓練 seed の一次証拠に `train_stats_sft.json` を追加 ⑥ 下記のバイト数訂正。
- ⚠️ **commit `351648d` 本文の自己申告「+1.5KB 増」は不正確 (2026-09-28 訂正)**。commit 本文は書き換えないので、ここに実測を残す:
  CLAUDE.md は git blob (LF) で **34,215 → 35,394 バイト = 追加 1,626 / 削除 447 / net +1,179**
  (作業ツリーは CRLF ゆえ 34,560 → 35,741 = +1,181)。一次証拠 = `git show 351648d~1:CLAUDE.md | wc -c` / `git show 351648d:CLAUDE.md | wc -c` / `git diff 351648d~1 351648d -- CLAUDE.md` の追加・削除行のバイト数。
  なお本 2 巡目の是正 (commit `74f1a33`) で CLAUDE.md は git blob **35,394 → 35,902 バイト = net +508** さらに増えた (F-0b 行 1 行のみの書き換え)。増やした理由 = 推定量名・en46 独立系列・死軸の物差しはいずれも**監査が「これを書かないと誤読が復活する」と指摘した実体**で、ポインタでは代替できないため。
- ~~🔲 Z-1〜Z-5 は**すべて未実施**。~~ (2026-09-28 時点の記載。2026-09-29 に Z-1 を算出したため取り消す)
  着手は CLAUDE.md 実装ルール (プラン → 承認 → PL 振り分け) に従う。
- ✅ **Z-1 算出済 (2026-09-29)**: 勝者一致率は 44/400 = 11.0% (ja 14/184・en 30/216)。
  寄与率は Δ 基準 (主) で anatomy 9.287%、水準基準 (副) で 1.126%。
  **数値と定義のみで、判定はしていない**。定義・一次証拠・限界 (lang 結合の出自は未検証 / 空候補 1 件の除外は plan 未記載 / 一致率と寄与率は別データセット) は「### Z-1」節の「Z-1 結果」にある。
- ~~🔲 **Z-2〜Z-5 は未実施**。~~ (2026-09-29 Z-1 反映時点の記載。同日 Z-2 を算出したため取り消す)
- ✅ **Z-2 算出済 (2026-09-29)**: 推定量 5 系列の t は記録値と一致した。記録値 CI の下端 −0.0115 は解析的 CI では再現せず、出自は未確定のまま。
  σ_seed (reward・pooled) は 0.04663。必要プロンプト数の逆算表も出した。
  **数値のみで、判定はしていない** (出口①の推定量の選択も未了)。数値・読み方の注意・限界 (σ_seed は ja 1 プロンプト由来で en への一般化は未検証 / ρ の実測なし / CR の df は G−1 のみ) は「### Z-2」節の「Z-2 結果」にある。
- 🔲 **Z-3〜Z-5 は未実施**。
- ~~⏸ GPU 実走は **Z-2 の σ_seed 待ちで未起票**。~~ (2026-09-29 Z-2 算出により取り消す)
- ⏸ GPU 実走は **σ_seed 算出済・起票は未**。設計 (a)/(b)・k・枚数は決めていない。起票時に扱いが要る論点 (本 doc では決めない):
  「Z-2 結果」の限界 1・2 (σ_seed の en 一般化、ρ) と、出口①の推定量の選択。
- 🔲 **未起票の別残債 (Q-2 由来)**: E-2 の clean/clutter 分離判定は**入力題材基準**で計算されており F-0a の 4x (**生成 prompt 基準**) とは被験変数が違う = **分離維持/悪化は未判定**。詳細と一次証拠は `docs/q2-quality-branch-plan.md` Package E 節の 2026-09-28 注記。**Q-2 のゲート通過自体は std 0.1038>0.1 で正当**。

## 参照
- 元台帳・実走記録: `docs/f0b-rejection-sft-plan.md`
- reward: `scripts/dollma_reward.py` / quality 枝: `docs/q2-quality-branch-plan.md`
- LM/SFT: `scripts/train_bitnet.py` (`--sft-rejection` / `--eval-only` / `--dump-persample` / `--seed` 既定 20260620)
- seed noise 判定の先例 (物差し): `docs/measurements-log.md` Phase 4-A (a12k) / Phase 4-D (容量増) 行
- F 全体: roadmap 「Phase 4」F 行 / CLAUDE.md 計測表 Phase 4 F 行 / [[project_phase4_F_status]]

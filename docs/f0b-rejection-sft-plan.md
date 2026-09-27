# F-0b: rejection-sampling SFT プラン (分割実行・複数セッション/GPU)

> Phase 4 施策 F の本命 SFT。Q-2 で reward に quality 直交軸が入り**信号ゲート通過**
> (reward std 0.1038>0.1 ∧ best−worst 0.3575>0.3) したのを受け、F-0a の補強待ちを解除して着手。
> recall でなく「良い絵を生む」方向へ LM の学習軸を移す (RAFT / best-of-N SFT)。
> 本 doc は**セッションをまたいで拾える分割タスク台帳**。各 Package は担当機/依存/入口出口が自己完結。

## 芯 (PL 承認済み設計: オフライン best-of-N SFT・anatomy+美的報酬・Python-first)

```
各入力 text ──► LM (BitNet) が N 候補プロンプトを確率的生成 (temperature/top-k)
                    │ (× M 入力)
                    ▼
              SDXL 生成 ──► ScorerNet(anatomy) + CLIP image→QualityMLP(quality)
                    ▼
              reward_from_scorer(axes, quality)  [anatomy + quality w=0.4・Q-2 で開通]
                    ▼
              best-of-N 選抜 (top-1 of N = reward 最大の1本)
                    ▼
              SFT データ (input text → 勝者プロンプト) × M
                    ▼
              train_bitnet.py SFT 経路: 正典 bitnet_dense から層状に低LRで焼く
                    ▼
              評価: diverse set-F1 非退行 ∧ 平均 reward 前後比
```

## 確定パラメータ (ユーザー決裁 2026-07-04)
- **rollout 規模**: **N=8 候補 × M=400 入力 = 3,200 枚生成**。データ律速を押す (フル維持)。
- **所要時間 (実測訂正)**: 自作 C++ backend `dollama.exe` は **19.3s/枚** (probe10 の diffusers 3.8s は非該当) → **~17.7h**。
  steps=20 は eval 標準で据え置き・単一 GPU で SDXL は 98% GPU バウンドゆえ並列短縮不可。
- **チャンク実行 (ユーザー決裁)**: **50件ずつ resume 可能**に (`dollma_rollout_bestofn.py --limit 50` + 完了 post_id skip)。
  8チャンク×~2.2h で中断/再開可。総時間は同じだが監査・耐障害性が上がる。
- **採択ポリシー**: **top-1 of N (RAFT)** = 各入力で reward 最大の1本のみを SFT 教師に。SFT データ = 400 勝者ペア。
- **reward**: anatomy 8軸 + quality (CLIP image→QualityMLP・QUALITY_WEIGHT=0.4)。`dollma_reward.py` 現行。
- **既知の制約 (G-2a へ申し送り)**: 選定400入力は英216/日184。`encode_text_greedy` は英数字タグのみ拾うため
  **日本語入力(184)は空トークン化→[BOS,SEP] 空条件** (best-of-N は最良の*無条件*プロンプトを選抜)。
  英語入力(216)はタグ語で真に条件付け。eval_diverse も日本語は同条件ゆえ train/eval は内部整合。
  ~46%が無条件学習になる点は F-0b の効果解釈時に留意 (トークナイザ既存性質・F-0b 起因ではない)。

## 分割タスク台帳

| Pkg | 内容 | 担当 | 担当機 | 依存 | status |
|---|---|---|---|---|---|
| **G-1** | LM 確率的サンプリング (temperature/top-k) 追加 + best-of-N rollout 収集 (N8×M400→SDXL→reward→top-1 選抜) → SFT データセット | gpu-benchmarker | 研究機 (GPU ~3.4h) | Q-2✅ | 🔲 未 |
| **G-2a** | rejection-sampling SFT (`train_bitnet.py` に SFT 経路・正典 bitnet_dense から層状低LR) + **diverse set-F1 非退行** 評価 (生成不要) + test | model-trainer | 本機 | G-1 | 🔲 未 |
| **G-2b** | **平均 reward 前後比** の実測 (SDXL 生成を伴う→GPU 必須) | gpu-benchmarker | 研究機 (GPU) | G-2a | 🔲 未 |
| **G-3** | 出荷判定 (reward↑ ∧ 正典 set-F1 非退行なら正典化・満たさねば不採用でクローズ) | PL + model-trainer | 本機 | G-2a/b | 🔲 未 |

> **PL 条件付き承認 (2026-07-04)** の必須条件:
> 1. **G-2 は 2 分割**: 訓練+set-F1 は本機 (G-2a)、reward 前後比は SDXL 生成を伴うため研究機 GPU (G-2b)。
> 2. **リーク防止 (必須)**: G-1 の 400 rollout 入力は **diverse-val 評価セットと素性重複ゼロ** (disjoint)。
>    重複すると G-2a の set-F1 非退行ゲートが自己参照で嵩上げされ安全弁が無効化する。G-1 着手時に disjoint 保証。
> 3. **再現性**: 3,200 枚は入力×候補ごとに固定 seed で監査可能に。SDXL seed 固定・temperature/top-k も固定。
> 4. **搬送**: 400 ペア (G-1→G-2a) と SFT 済みモデル (G-2a→G-2b) は git 管理外 → `--copy/--publish` (train_bitnet.py) で搬送。
> 5. SAC/ライセンスはクリア (全経路 Python・reward の外部モデルは waifu apache のみ・追加導入なし)。

### Package G-1 — best-of-N rollout 収集 (研究機・GPU・~3.4h)
- 入口: Q-2 完了 (reward に quality 結線済み・`dollma_collect_rollouts.py` の新 quality 経路)。
- 作業:
  1. **LM 確率的サンプリング**: 現行 `dollma_collect_rollouts.py` / `train_bitnet.py` の生成は greedy (`_greedy_generate`・決定的単一)。temperature/top-k サンプリング生成を追加 (`_sample_generate` 等)。同一入力で N=8 の**多様な**候補が出ること (temperature 調整・重複率をログ)。seed 制御で再現的に。
  2. **入力 M=400**: diverse 自然文入力コーパス (tags-stay-real・`data/bitnet/diverse_train_*` or eval diverse セット) から 400 入力を採る。F-0a の 80 と重ならない/被覆広めが望ましい。
  3. **rollout**: 各入力 × N=8 候補 → SDXL 生成 → ScorerNet axes + CLIP image→QualityMLP quality → `reward_from_scorer`。**top-1 of N** で勝者選抜。
  4. **SFT データ出力**: (input text → 勝者プロンプト) × 400 を jsonl 化 (train_bitnet の SFT 経路が読める schema)。勝者 reward 分布・N候補の reward spread (best−worst per input)・選抜前後の平均 reward もログ。
- 出口: 400 勝者ペア SFT データセット + rollout 統計 (勝者 reward 分布・候補多様性)。
- 注意: SDXL seed は server wall-clock で非再現ゆえ画像は再生成不可 → 勝者プロンプトとスコアを確実に保存。data/rollouts/ は gitignore (PNG 再生成可)。

### Package G-2 — rejection-sampling SFT + 評価 (本機/研究機)
- 入口: G-1 完了 (400 勝者ペア)。
- 作業:
  1. **SFT 経路**: `train_bitnet.py` に rejection-sampling SFT を追加。**正典 `bitnet_dense` (出荷ベース) から層状に低 LR・少 epoch** で焼く (破滅的忘却回避)。SFT データ = G-1 の勝者ペア。既存の多様化入力レシピ (tags-stay-real) と非干渉に。
  2. **評価 (2軸)**:
     - **diverse set-F1 非退行**: 正典基準 (retention 0.9807 / diverse_a F1 0.3332 / diverse_b F1 0.3804 / in-dist 0.4552) を**割らない**こと (ゲート)。
     - **平均 reward 前後比**: SFT 前 (正典 LM) vs 後 (SFT LM) で、同一入力セットに best-of-N でなく greedy 1本を生成させ reward 平均を比較 → **上がっていること** (SFT の効果)。
  3. test: SFT 経路の単体 (決定性・schema・層状ロードが正典から始まること)。
- 出口: SFT 済み重み (隔離・正典未差し替え) + 評価数値 (set-F1 4指標 + reward 前後比)。
- 安全弁: **正典アーティファクト無改変** (G-3 の合格まで)。set-F1 が正典基準を割ったら不採用。

### Package G-3 — 出荷判定 (正典化 or クローズ)
- 入口: G-2 完了 (評価数値)。
- 合格条件: **平均 reward が有意に↑ ∧ diverse set-F1 4指標が正典基準を非退行**。
- 合格時: `[B-merge-at-A]` と同流儀で正典 bitnet_dense{,_fp32}/golden を SFT 版へまとめ焼き・corr 突合・meson 緑。
- 不合格時: 不採用でクローズ (SFT が set-F1 を割る/reward が上がらない)。原因を measurements-log に記録 → 次レバー (教師枚数増 = rollout M 拡大 / reward 設計 / ScorerNet anatomy 分解能) へ。

## 安全弁 (非交渉・既定)
- SFT は正典 `bitnet_dense` から層状 (低 LR・少 epoch)。diverse set-F1 が正典基準を割ったら不採用。
- 正典アーティファクトは G-3 合格まで無改変 (Q-2 と同じ隔離運用・[B-merge-at-A] とは別レイヤ)。

## スコープ外
- deepghs 合流 / ScorerNet anatomy 7死軸の分解能改善 = F-0b と独立の別レバー (必要時)。
- ternary GEMM 圧縮実験 = 別軸。

## 現在地 (最終更新: 2026-09-28 = 記録監査 BLOCK の是正。走行の追加なし)
- ✅ Q-2 quality 枝 (信号ゲート通過) → F-0b ゲート解除
- ✅ PL 条件付き承認 (G-2 分割・リーク防止・seed 固定が条件)
- ✅ **Package G-1 完走: 400/400 (2026-07-05 11:22 JST)**。
  - 成果物 `data/rollouts/sft_bestofn.jsonl` (400 勝者ペア・source=rejection_sft) = G-2a 入力。
  - 統計: winner reward mean **−0.1842**/std 0.0768/range[−0.343,−0.041]・**best−worst spread mean 0.1952** (best-of-8 で~0.2改善)・候補多様性 7.995/8・lang ja184/en216・disjoint OK。
  - resume/chunk 化 (`scripts/dollma_rollout_bestofn.py --limit` + `dollma_g1_driver.sh`)・`_sample_generate` in train_bitnet.py。
- ⚠️ **Package G-2a 完了・set-F1 非退行ゲート未達 (2026-07-05)**:
  - `train_bitnet.py --sft-rejection` 追加 (正典 bitnet_dense から層状 warm-start・低LR・破滅的忘却なし)。隔離重み `data/bitnet/bitnet_dense_sft{,_fp32}.safetensors`・正典無改変・test 3/3。
  - set-F1 前後: in-dist 0.4552→0.4570(+) / **diverse_a 0.3332→0.3158(−0.0174)** / **diverse_b 0.3804→0.3563(−0.0241)** / retention 0.9807→0.9784(−0.002)。
  - ⚠️ **「構造的 (チューニングノイズでない)」は撤回 (2026-09-28 是正・監査 2026-09-27 指摘)** → **同一 seed 内の全レシピ (本採用 + lr×ep sweep 4 条件) で退行の符号が揃った。ただし seed 間分散は未測定のため、seed noise 帯との識別はできていない。**
    - 一次証拠 (seed 未制御): `data/bitnet/_g2a_eval/eval_report_{canon_g2a,sft_g2a,cand_lr5e-6_ep1,cand_lr5e-6_ep2,cand_lr1e-5_ep1,cand_lr1e-5_ep2}.json` の provenance は **6 本すべて `"seed": 20260620`** (`scripts/train_bitnet.py` の `--seed` 既定も 20260620)。SFT アームの seed 変動走行はゼロ。
    - **既存の判定基準との並置**: 施策 D (容量増 33M→80M) は diverse_a F1 の per-seed delta 最大幅 **−0.0240** を「seed ノイズ」と断じている (`docs/measurements-log.md` Phase 4-D 行)。F-0b の退行幅 (−0.0174 / −0.0241) は**これとほぼ同値**であり、単一 seed の観測では seed noise 帯と区別できない。
  - 退行の**仮説** (未検証): best-of-N の reward(解剖+美的) と gold タグ set-F1 は非整合・日本語184空条件も同方向。
  - **論点**: F の狙いは元々「recall でなく良い絵へ学習軸を移す」= set-F1 卒業。set-F1 軽微退行は F 思想そのものとも言える。判定には reward 前後比 (G-2b) が必須。
- ✅ **Package G-2b 完了 (2026-07-05)**: held-out 100入力 (G-1訓練/eval と disjoint・en46/ja54) で greedy→SDXL→reward ペア比較 (200枚)。
  - reward mean pre −0.2974 → post −0.2803・**Δ mean +0.017 / median +0.013 / 正60%**。
  - ⚠️ **「~2.4σ」「弱い正」は撤回 (2026-09-28 是正・監査 2026-09-27 指摘)** → **有意差を検出できなかった (検出力不足)**。
    - 100 件を独立標本として扱った naive t≈2.40 は**疑似反復によるアーティファクト**。監査 (2026-09-27) によるクラスタ補正の再解析では **t=0.955・95%CI [−0.0115, +0.0348] = 0 を含む** = 有意でない。
    - 疑似反復の一次証拠: `data/rollouts/g2b_prepost.jsonl` の `lang=ja` 54 件は、pre/post いずれのアームでも `prompt` の uniq 数が **1** (en は 46 件で uniq 46)。つまり ja 分は「同一プロンプト 1 組を SDXL seed だけ変えて 54 回複製した」標本。
    - 機構 (既知性質・F-0b 起因ではない): **日本語入力は `encode_text_greedy` が英数字タグしか拾わず空条件化し、greedy 生成が同一プロンプトへ収束する** (本 doc「確定パラメータ」節の「既知の制約」・[[project_expression_fidelity_gap]])。
  - Δ 内訳: **ほぼ全量 quality 由来** (quality +0.0155 / anatomy +0.0016)。en +0.011 / ja +0.022。
  - 但し書き: SDXL seed 非再現 → per-input Δ に seed ノイズ (Δ std 0.071 の主因)。
  - スクリプト `scripts/dollma_g2b_reward_prepost.py` / `dollma_g2b_driver.sh`。
- ✅ **G-3 判定: 不採用でクローズ (ユーザー決裁 2026-07-05)**: reward↑ (+0.017・有意性は未確立) が信頼 proxy(set-F1) 退行 (−0.0174/−0.0241) を正当化できない、という判断。**正典 bitnet_dense 無改変維持・SFT 重みは隔離保存 (data/bitnet/bitnet_dense_sft*)・知見記録**。
  - **処置 (正典無改変・SFT 重み隔離) は妥当であり変更しない**。2026-09-28 の是正対象は処置の是非ではなく**主張の強さ**のみ。

## 🏁 F-0b 完遂・クローズ (2026-07-05・結論の強さを 2026-09-28 に是正)

**結論**: RAFT-SFT (best-of-N→top-1→SFT) を end-to-end で実装・実走・評価。**パイプラインは検証済み・再利用可能**。**この 1 走行では出荷に値する効果を検出できなかったため不採用でクローズした**。
- ⚠️ **棄却されたのは「RAFT-SFT という手法」ではない** (2026-09-28 是正)。棄却されたのは
  **「anatomy 軸が死んだ現行 reward 設計・単一 seed (20260620)・検出力不足という条件下で、M=400 の RAFT-SFT の効果を検出できなかった」**という一点。手法自体の一般的な無効性は示していない。
  - 報酬の実効寄与は **quality 軸に支配されている**: Δ 内訳 quality +0.0155 / anatomy +0.0016 (= Δ の 9 割超が quality 由来)・監査 (2026-09-27) の再計算では anatomy の寄与率 **7.5%**。
  - 一次証拠 (anatomy の分解能): `data/rollouts/g2b_prepost.jsonl` 200 行の `axes` argmax は Limbs 148 / Hands 35 / Head 17 = **8 軸のうち 5 軸は一度も argmax にならない** (F-0a の「7 死軸」と同型)。
  - **処置面は変更なし**: 正典無改変・SFT 重み隔離は不採用判定下で妥当。
- G-1 400ペア収集 (best-of-8 で spread ~0.2)・G-2a SFT (正典から層状・破滅的忘却なし)・G-2b reward 前後比 (+0.017・有意性未確立)・G-3 不採用。
- **確定した知見** (2026-09-28 に ①② の主張を証拠の範囲へ引き下げ): ① SFT 後に diverse set-F1 が下がる方向は**同一 seed 内の全レシピで一致した** (−0.0174/−0.0241)。ただし **seed 間分散は未測定**で、施策 D が seed ノイズと断じた最大幅 −0.0240 とほぼ同値ゆえ **seed noise 帯と識別できていない**。「reward と gold タグ set-F1 が非整合」は依然**仮説**。② reward シフトは**ほぼ全量 quality 由来** (anatomy は F-0a 同様ほぼ死・上記 argmax 参照)。③ SDXL seed 非再現 (SAC で再ビルド不可) が per-input reward 比較のノイズ源。④ 日本語184/400 が空条件化 ([[project_expression_fidelity_gap]]) → G-2b の held-out 100 件でも ja 54 件が**単一プロンプトへ収束**し疑似反復を生んだ (有意性主張が崩れた直接原因)。
- **再検証 (未実施)**: 監査指摘の手続き不備 (anatomy 軸の分解能不足・seed 未制御・プラセボ対照の欠如) の是正は
  **`docs/f0b-reverification-plan.md` の Z-1〜Z-5** に台帳化済み。全 status は 🔲 未。再検証の結果が
  **依然「不採用」になることも十分あり得る** (採用へ仕向ける作業ではない)。
- **次レバー (F-0b とは別軸・優先度順の仮説)**: (a) **reward 設計**の見直し (anatomy が死んでいる → quality 主体でよいか/新軸) (b) **日本語条件付けトークナイザ改修** (空条件を実条件に・[[project_expression_fidelity_gap]]) (c) SDXL seed 制御 (HTTP に seed 引数・SAC 制約下の実現方法) で reward 比較のノイズ除去 (d) 教師枚数増 M拡大 (効果量 +0.017 自体は seed 交絡で伸びにくい公算・優先度低)。
  ※ これらは F-0b の**再検証**とは別物。再検証 (手続き不備の是正) は `docs/f0b-reverification-plan.md` Z-1〜Z-5。
- 再利用資産: `train_bitnet.py --sft-rejection` / `dollma_rollout_bestofn.py` (resume/chunk) / `dollma_g2b_reward_prepost.py` / 各 driver・test。隔離重み `bitnet_dense_sft*` は将来の reward 改善時に再評価可。
- **ステータスは一意**: G-1 / G-2a / G-2b / G-3 は**すべて完了** (上記「分割タスク台帳」と「現在地」参照)。
  ※ 2026-09-28 是正: 本節末尾に「未着手: G-2a / G-2b / G-3」および G-1 の途中経過 (101 入力で継続中) が
  残っていて上部の ✅ と矛盾していた (監査 2026-09-27 指摘)。下記の通り**G-1 の途中経過の記録として位置づけ直した**
  (記録は消さない)。

**【履歴・G-1 の途中経過 (2026-07-05 完走前のスナップショット。現状ではない)】**
- モノリシック走行を 101 入力で停止・退避 `data/rollouts/sft_bestofn.done101.bak.jsonl` / `candidates_bestofn.done101.bak.jsonl`。
- 統計 (101件時点): winner reward mean −0.180 / std 0.076・best−worst mean 0.192・ja54/en47・候補多様性 8/8・disjoint OK。best-of-N は Q-2 quality 軸が選抜駆動。
- **実測 19.5s/枚**。当時の残り 299 件 (~13h)。**resume-skip+50件チャンク化**で 101→400 を継続 → **400/400 完走済**。
- ⚠️ 当時の script は起動時 `"w"` truncate → resume 改修前の再実行厳禁 (.done101.bak が保険)。

## 参照
- reward: `scripts/dollma_reward.py` (anatomy+quality w0.4) / rollout: `scripts/dollma_collect_rollouts.py` (Q-2 で quality 結線済)
- LM/SFT: `scripts/train_bitnet.py` (`_greedy_generate` / `build_sequence` / 正典 `data/bitnet/bitnet_dense*.safetensors`)
- quality 枝: `docs/q2-quality-branch-plan.md` / QualityMLP `data/scorer/quality_mlp*.safetensors` / CLIP image IR `models/clip-image/`
- 評価: diverse set-F1 = `scripts/dollma_make_eval_diverse.py` / `scripts/test_dollma_eval_diverse.py` (training-spec §13/§17)
- F 全体: CLAUDE.md 計測表 Phase4 F 行 / `docs/measurements-log.md` / [[project_phase4_F_status]] / roadmap F-0b 行
- 再検証台帳 (Z-1〜Z-5・全て未実施): `docs/f0b-reverification-plan.md`
- G-2b 生データ: `data/rollouts/g2b_prepost.jsonl` (200 行 = pre/post × 100 入力) / G-2a 評価 provenance: `data/bitnet/_g2a_eval/eval_report_*.json`

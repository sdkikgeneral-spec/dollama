# docs/logs/f0b-z4 — F-0b 再検証 Z-4 (プラセボ対照 SFT) の一次成果物

F-0b 再検証の Z-4 (勝者を reward argmax で選んだ R'_s と、候補から一様抽選した P_s を同レシピで SFT し、
diverse set-F1 の canon 比 Δ を 8 seed で比べる) の本走出力と実行記録を置く。Z-5 (定義変更付き) も同じ出力で判定している。
**設計・判定規則の正本は `docs/f0b-reverification-plan.md` の「Z-4 事前固定」節 (本走前に固定・commit `750ae6f`)、
結果・限定条件の正本は同 doc の「Z-4 結果」節**。本 README は成果物の説明と、ログの不備の注記だけを書く。

## 中身

| ファイル | 何 |
|---|---|
| `z4_result.json` | 解析出力 (UTF-8・`scripts/dollma_f0b_z4_analyze.py` の `--out`)。`datasets.{diverse_a,diverse_b}` 配下に `classification` (R'・P の seed 間 one-sample t・D の Welch 95%/90%CI・M・区分)・`per_seed.{r,p}` (seed ごとの canon 比 Δ と post_id 単位 / iid の CI)・`z5`・`z5_axis_c`・`z5_variance_band`・`reference_R_existing`・`reeval_check`・`mde_note`。トップに `combined` (a/b 合成区分)・`z5_pass`・`provenance` (canon / r_ref / R'_s 8 / P_s 8 = 18 アームの重みパスと sha256・R'_s と P_s の 16 本は互いに別 sha = `n_arm_weights_distinct` 16)・`retention`・`confound`・`power_prior`・`constants`・`env` |
| `z4_run_log.txt` | 2026-10-10 の本走ログ。開始時の点検 (HEAD=`750ae6f`・作業ツリー clean・main data の sha256 (前))・判定定数の機械照合 (`ALL_MATCH`)・prepare / placebo / train の全コマンド・決定論確認・eval の 3 本目のコマンド行まで |
| `resume_20261018.log` | 2026-10-18 の再開ログ。prepare (入力 sha256 の再出力)・決定論確認行の再出力・残り 16 本の eval・analyze |
| `_scratch_data/` | 入力のコピー・プラセボの勝者ペア・訓練済み重み (R'_s 8・P_s 8・`_detchk` 1 = 17 本 × fp16/fp32)・train stats・eval レポート / per-sample npz 18 本・eval ログ。gitignore 対象 (`.gitignore` の `docs/logs/*/_scratch_data/`)・repo には入らない |

生成元のスクリプト: `scripts/dollma_f0b_z4_run.py` (本走ランナー)・`scripts/dollma_f0b_z4_placebo_pairs.py` (プラセボ生成)・
`scripts/train_bitnet.py` (`--sft-rejection --deterministic` で訓練 / `--eval-only --dump-persample --dump-retention-detail` で評価)・
`scripts/dollma_f0b_z4_analyze.py` (判定)。

## 出自

- 実行場所: git worktree `E:\Develop\Projects\dollama\.claude\worktrees\f0b-z4-placebo` (HEAD `750ae6f`)。
- 正典 `bitnet_dense_fp32` と F-0b の隔離重み `bitnet_dense_sft_fp32`・勝者ペア・候補プールは、main checkout の `data/` から scratch へコピーして使った (prepare 段)。
  10-10 ログ冒頭の「main data sha256 (前)」5 本のうち、コピー元になった 4 本は scratch のコピーと sha256 が一致する (残る 1 本 `bitnet_dense_sft.safetensors` (fp16) はコピーしていない)。
  ログに出ている訓練・評価・解析・プラセボ生成のコマンドの出力先 (`--out-dir` / `--out`) はすべて worktree の scratch 配下である。
  本走後の main data の sha256 (「後」) はどちらのログにも記録が無い。代わりに record-writer が記録時 (2026-10-18・本走完了後) に同じ 5 本を `sha256sum` で再計算し、
  「前」の値と 5 本とも一致した (本走の前後で中身が変わっていない。途中で変わって戻った可能性までは排除しない)。
- 実行日と評価日:
  - 2026-10-10: prepare / プラセボ生成 / 訓練 17 本 / 決定論確認 (R'_20260620 の 2 回訓練で重み sha256 一致) / eval 2 本 (canon・r_ref)。
    **r_s20260620 の eval コマンド行を最後にログが途切れている。~~停止の原因はログに記載がなく、不明~~**
    → **ユーザー指示による停止 (2026-10-10 00:59 頃)** (2026-10-18 訂正)。★ログには記載が無い。出典はセッション記録 (main thread の memory `project_f0b_z4_status.md`) で、二次の記録である。
  - 2026-10-18: 再開。prepare は入力 10 ファイルの sha256 を再出力し、10-10 と全一致。プラセボ生成・訓練の段では新しいコマンドは出ていない
    (重みはすべて 10-10 作成のものを再利用)。残り 16 本 (R'_s 8・P_s 8) を eval し、analyze を実行した。
  - したがって **canon / r_ref は 10-10 評価、他の 16 本は 10-18 評価**。eval レポート 18 本の provenance はすべて `git_rev` 750ae6f・評価 seed 20260620・torch 2.11.0+cu128。
  - 日跨ぎの整合: R'_20260620 の重み sha256 は既存 R (`bitnet_dense_sft_fp32`・`3b2e2181…8e15`) と同一で、
    r_ref (10-10 評価) と r_s20260620 (10-18 評価) の per-sample 配列 8 本が完全一致する。

## ★ログの不備 (数値には影響しない)

1. **`resume_20261018.log` に開始時の点検見出しが無い**: 10-10 のログ冒頭にある「事前固定 SHA / HEAD・clean の確認」と「main data sha256 (前)」の見出しが、
   再開ログには無い。再開時の HEAD は 10-18 の eval レポート 16 本の provenance `git_rev` (= `750ae6f`) で、入力の同一性は prepare 段の sha256 再出力 (10-10 と全一致) で、それぞれ間接的に裏付けられる。
   main checkout の `data/` 側の sha256 を再開時に記録したものは無い (記録時の再計算で「前」と一致したことは上の「出自」に書いた)。
   また 10-10 の停止原因 (ユーザー指示による停止・上の「出自」) も、どちらのログにも書かれていない。
2. **10-18 のサブプロセス出力の和文が cp932 のバイト列で記録されている**: 再開ログは UTF-8 として不正なバイトを含み (`iconv -f utf-8` が決定論確認の行で失敗)、
   UTF-8 で開くと和文が化ける。その行を cp932 で読み直すと `[決定論確認] R'_20260620 2 回訓練 sha256 一致 = True (3b2e2181155122ef / 3b2e2181155122ef)` と復元できる (record-writer が `iconv -f cp932` で確認)。
   数値・sha256・コマンド行は ASCII なので、どちらで読んでも変わらない。
   `z4_result.json` 自体は UTF-8 で正しく読める (端末によっては表示で化けるので、読むときは UTF-8 を明示すること)。

## 要点 (判定と限定は plan「Z-4 結果」節が正本)

- 判定 = **② 選抜方針 (reward) 起因** (a/b 一致)。★②/③ 境界は僅差 (Δ_P の 95%CI 上限 a +0.00037 / b +0.00065)。
- ★「reward の中身が原因」とは確定していない (タグ長分布の違いと分離できない設計)。P は seed ごとに不均一。
- Z-5 (定義変更付き) = 合格。★R' の seed はシャッフル順しか変えないので、言えるのは「F-0b の単一 seed 退行は SFT のシャッフル seed では説明できない」まで。
- **F-0b の不採用判断は覆さない**。

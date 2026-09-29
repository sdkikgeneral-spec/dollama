# docs/logs/f0b-z2 — F-0b 再検証 Z-2 の一次成果物

F-0b 再検証の Z-2 (クラスタ補正の再解析 + σ_seed の直接推定) を算出したときの出力と実行記録を置く。
**定義・数値・読み方の注意・限界の正本は `docs/f0b-reverification-plan.md` の「### Z-2」節**
(記録が反映されていればそちら、まだなら本 README とサブエージェント報告)。

## 中身

| ファイル | 何 |
|---|---|
| `z2_result.json` | 本計算の出力 (UTF-8)。`exit1_estimators` (naive/cluster_mean/cr_sandwich/en_only)・`exit2_sigma_seed` (per_arm/pooled/limitations)・`exit3_required_n` (design a/b の必要プロンプト数)・`cross_check` |
| `z2_run_log.txt` | 実行記録。日時・環境・入力の sha256・コマンド・標準出力の要約 |

生成元のスクリプト: `scripts/dollma_f0b_z2_seed_variance.py`

## 出自

- 実行日: 2026-09-29
- 実行場所: git worktree `E:\Develop\Projects\dollama\.claude\worktrees\f0b-z2-sigma-seed`
- 既存データの再解析のみで、SDXL 生成も LM 訓練もしていない。
  main checkout の `data/` へは読み取りだけで、書き込んでいない。

## 入力 (main checkout・gitignore 対象のため repo には無い)

| ファイル | sha256 |
|---|---|
| `E:\Develop\Projects\dollama\data\rollouts\g2b_prepost.jsonl` | `ba2db3bec55157ab24e8f10c71e7498ae04997aeeba92c4b24eb326820699e89` |

`docs/logs/f0b-z1/README.md` に記載の値と一致 (Z-1 と同一ファイル)。

## 再現コマンド

リポジトリのルートで実行する。

```
python scripts/dollma_f0b_z2_seed_variance.py --selftest   # 合成データの自己検査・exit 0 で OK
python scripts/dollma_f0b_z2_seed_variance.py              # 本計算 → docs/logs/f0b-z2/z2_result.json を上書き
```

- 入力パスの既定値はスクリプト内に絶対パスで固定 (`E:\Develop\Projects\dollama\data\rollouts`)。
  別の場所で再現するときは `--g2b-prepost` / `--out` で差し替える。
- クラスタ bootstrap・σ_seed bootstrap は `numpy.random.default_rng(seed=20260620)` で決定的
  (同一環境・同一 numpy バージョンなら再実行しても同じ値になる想定。厳密なビット一致は保証しない)。

## 要点 (数値の断定的解釈はしていない・詳細は JSON / サブエージェント報告参照)

- 4 推定量 (naive / cluster-mean / CR0 / CR1 / en46) の t 値は、`docs/f0b-reverification-plan.md`
  の 2026-09-28 記録値 (2.4025 / 0.9549 / 2.7592 / 2.7297 / 0.9154) と全て一致した (`cross_check.ok=true`)。
- 記録に残る「cluster-mean 95%CI [-0.0115, +0.0348]」の下端は、解析的 t/z CI (t(46) → [-0.0126,+0.0354]・
  z → [-0.0120,+0.0348]) では再現しないが、**nonparametric cluster bootstrap percentile CI では
  seed によって -0.0114〜-0.0117 のレンジに入り、-0.0115 に近い** (本スクリプトの既定 seed=20260620 での
  1 回の実行では lo=-0.0119 で、乱数シードに依存した Monte Carlo 誤差の範囲内)。**この一致は示唆であって
  確定的な再現証明ではない** (元の CI がどの乱数シード・どの B で計算されたか記録が残っていないため)。
- σ_seed (reward, pooled ja 54+54) = 0.0466 (chi2 95%CI [0.0411, 0.0539] / bootstrap 95%CI [0.0398, 0.0521])。
  この推定は ja の単一プロンプトペアのみに基づき、en プロンプトへの一般化は未検証 (JSON `limitations`)。
- 必要プロンプト数の逆算は τ² (プロンプト間の真の効果ばらつき) を en46 の Δ 分散から σ_seed² 分を
  差し引いて推定しており、詳細は JSON `exit3_required_n` 参照。GPU 実走の起票・枚数の最終決定はしていない。

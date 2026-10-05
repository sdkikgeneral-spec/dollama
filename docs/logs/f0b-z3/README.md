# docs/logs/f0b-z3 — F-0b 再検証 Z-3 の一次成果物

F-0b 再検証の Z-3 (set-F1 の per-case paired bootstrap CI) を算出したときの出力と実行記録を置く。
**定義・数値・読み方の注意・限界の正本は `docs/f0b-reverification-plan.md` の「### Z-3」節**
(記録が反映されていればそちら、まだなら本 README とサブエージェント報告)。

## 中身

| ファイル | 何 |
|---|---|
| `z3_result.json` | 本計算の出力 (UTF-8)。`datasets.{diverse_a,diverse_b}` 配下に `reproduction_check` (元台帳/measurements-log 値との再現確認)・`diff_summary` (paired t・分布要約)・`bootstrap_ci95` (B=20000)・`bootstrap_ci95_seed_sweep` (10 seed)・`cross_check` |
| `z3_run_log.txt` | 実行記録。日時・環境・実行したコマンド 4 本 (canon 評価 / SFT 評価 / Z-3 本計算 / selftest)・入力の sha256・標準出力の要約 |
| `_scratch_data/` | eval に使った入力 (vocab.json / pairs.val.jsonl / pairs.eval_diverse_{a,b}.jsonl の worktree 内コピー) と出力 (eval_report_{canon,sft}_z3.json / eval_persample_{canon,sft}_z3.npz)。gitignore 対象 (`.gitignore` の `docs/logs/*/_scratch_data/`・2026-10-05 追加。それ以前は未追跡なだけで ignore されていなかった)・repo には入らない |

生成元のスクリプト: `scripts/dollma_f0b_z3_setf1_bootstrap.py`
(入力の生成には `scripts/train_bitnet.py --eval-only --dump-persample` を使用。既存 CLI・新規追加なし)

## 出自

- 実行日: 2026-09-29〜30
- 実行場所: git worktree `E:\Develop\Projects\dollama\.claude\worktrees\f0b-z3-setf1-bootstrap`
- 正典 (`bitnet_dense{,_fp32}`/identity/golden) は無改変。`--weights` は main checkout の
  隔離重みを読み取り専用で参照した (コピー・移動・上書きなし)。
  main checkout の `data/` へは一切書き込んでいない (出力は全て worktree の scratch 配下)。

## 入力 (sha256)

| ファイル | sha256 |
|---|---|
| `data/bitnet/bitnet_dense_fp32.safetensors` (正典・canon) | `5043772dd13f6fd854d9909191960d8e593ea830638ebd482e8d691df9ad0b61` |
| `data/bitnet/bitnet_dense_sft_fp32.safetensors` (F-0b 隔離重み・SFT) | `3b2e2181155122ef94a4cfd8d767696728076344ff642726e4b1cebb8eab8e15` |
| `data/bitnet/pairs.eval_diverse_a.jsonl` | `02c37d15fbe757d054d8d5e253ca2b5be52da66a381d4ca2ed199253d0bba828` |
| `data/bitnet/pairs.eval_diverse_b.jsonl` | `f5514da802437545d9581f9402641a546b6cb30981d3393b37e4bbd9244927c8` |
| `data/bitnet/pairs.val.jsonl` | `8548b250c6b65cf40cea45936a23705409cb939566395f46079d1fb1e7d2cb4c` |
| `data/bitnet/vocab.json` | `af520585b4414e82f21aadfff37a12d0e97c5f362ba7e7ea1f307bfe4b1e7d51` |

いずれも G-2a の評価 (`data/bitnet/_g2a_eval/`) で使ったものと同一ファイルである。根拠は 2 種類で、裏付けの範囲が違う。
- **重み 2 本と `pairs.val.jsonl`**: 既存の `eval_report_{canon_g2a,sft_g2a}.json` の provenance
  (`weights_sha256` / `val_sha256`) に記載された sha256 と一致する。
- **`pairs.eval_diverse_{a,b}.jsonl` と `vocab.json`**: provenance には sha256 が載っていない。
  record-writer が `data/bitnet/_g2a_eval/` の同名ファイルと `sha256sum` で直接比較し、同一バイトだと確認した (2026-10-05)。
(2026-10-05 訂正。旧記載は「いずれも provenance に記載の sha256 と一致する」で、provenance で裏付けられるのは重みと val だけだった)

## 再現コマンド

リポジトリのルートで実行する。

```
python scripts/dollma_f0b_z3_setf1_bootstrap.py --selftest   # 合成データの自己検査・exit 0 で OK

# per-case F1 配列の生成 (canon / SFT)。--weights は main checkout の絶対パスに読み替える。
python scripts/train_bitnet.py --eval-only \
  --data-dir docs/logs/f0b-z3/_scratch_data \
  --weights <path>/bitnet_dense_fp32.safetensors \
  --device cuda --seed 20260620 --eval-name canon_z3 --dump-persample
python scripts/train_bitnet.py --eval-only \
  --data-dir docs/logs/f0b-z3/_scratch_data \
  --weights <path>/bitnet_dense_sft_fp32.safetensors \
  --device cuda --seed 20260620 --eval-name sft_z3 --dump-persample

# Z-3 本計算 → docs/logs/f0b-z3/z3_result.json を上書き
python scripts/dollma_f0b_z3_setf1_bootstrap.py \
  --canon-npz docs/logs/f0b-z3/_scratch_data/eval_persample_canon_z3.npz \
  --sft-npz docs/logs/f0b-z3/_scratch_data/eval_persample_sft_z3.npz \
  --canon-report docs/logs/f0b-z3/_scratch_data/eval_report_canon_z3.json \
  --sft-report docs/logs/f0b-z3/_scratch_data/eval_report_sft_z3.json \
  --out docs/logs/f0b-z3/z3_result.json
```

- `--data-dir docs/logs/f0b-z3/_scratch_data` には事前に `data/bitnet/{vocab.json,pairs.val.jsonl,
  pairs.eval_diverse_a.jsonl,pairs.eval_diverse_b.jsonl}` をコピーしておく必要がある
  (train_bitnet.py --eval-only は data_dir 直下からこれらを読む)。
  これらは git 管理下で worktree に既に存在する小ファイルなので、コピーは正典の複製ではない。
- bootstrap は `numpy.random.default_rng(seed)` で決定的 (既定 seed=20260620・B=20000。
  同一環境・同一 numpy バージョンなら再実行しても同じ値になる想定。厳密なビット一致は保証しない)。

## 要点 (数値の断定的解釈はしていない・詳細は JSON / plan「### Z-3」節参照)

- **再現確認 = 成功**: canon/SFT を同一評価スクリプトで再走した結果、macro F1 は
  diverse_a 0.3332→0.3158・diverse_b 0.3804→0.3563 (Δ −0.0174 / −0.0241) と、
  元台帳 (`docs/f0b-rejection-sft-plan.md`「現在地」節の Package G-2a 完了の項の「set-F1 前後」行) の記録値に小数4桁まで一致した。
  原因調査 (再現しない場合の代替手順) は不要だった。
- **per-case pairing**: diverse_a/diverse_b とも 1500 件全てが有効ペア (`n_valid_paired=1500`)。
  スキップ (prompt 長超過) は 0 件、canon/SFT 間の NaN 不一致も 0 件。
- **同一 seed 内の paired bootstrap CI (95%・B=20000・seed=20260620)**:
  diverse_a `[-0.022167, -0.012620]`、diverse_b `[-0.029525, -0.018794]`。
  どちらも 0 を含まない。参考の paired t による CI (t(1499) 基準) もほぼ同じ区間
  (diverse_a `[-0.022244, -0.012579]`・diverse_b `[-0.029471, -0.018815]`)。
- **bootstrap の seed 依存**: 10 seed (20260620, 0–7, 42) で CI 端は
  diverse_a 下端 `[-0.022258, -0.022167]` (幅 0.000091)・上端 `[-0.012647, -0.012524]` (幅 0.000123)、
  diverse_b 下端 `[-0.029572, -0.029435]` (幅 0.000138)・上端 `[-0.018869, -0.018774]` (幅 0.000095) と、
  乱数 seed による揺れは CI の桁 (10^-2) に対して非常に小さい (n=1500 の恩恵)。
- **差の分布**: diverse_a は退行ケース 37.3%・改善 27.7%・同値 35.0% (同値の内訳と理由は下の「読み方の注意」)。
  diverse_b は退行 40.9%・改善 27.7%・同値 31.4%。平均は負だが、個々のケースでは改善も一定割合ある
  (退行が全ケース一様に起きているわけではない)。
- **cross_check**: 両データセットとも 14 項目 (mean/sd/se/t/CI 上下/正負ゼロ件数/中央値/四分位/
  bootstrap CI 上下) 全一致 (`ok=true`・`n_fail=0`)。

## ★読み方の注意 (誤読しやすい点)

- seed の一次証拠: 評価 seed は canon・SFT とも 20260620 (npz の `_seed`)。
  訓練 seed が一次証拠で確認できるのは **SFT だけ** (`data/bitnet/train_stats_sft.json` の `"seed": 20260620`)。
  **canon の訓練 seed は未確認**である。下の「訓練 20260620」は SFT の訓練を指す。
- **本節がカバーするのは「同一 seed (SFT 訓練 20260620・評価 20260620) 内で、この 1500 件の
  凍結 diverse-val に対して canon と SFT の F1 差が 0 か」という識別力だけ**である。
  CI が 0 を除外することは、**この 1 回の SFT 訓練・この 1 回の凍結評価セットの中では
  差が安定して観測できる**ことを示すが、**SFT を別の訓練 seed で焼き直したときに同じ幅で
  退行するか (= seed 間分散) は測っていない** (Z-5 の範囲)。plan の是正対象表が指摘した
  「seed ノイズ帯 (施策 D 最大幅 −0.0240) と識別できない」という論点は、**訓練 seed 間の分散**の
  話であり、本節の CI (同一訓練・同一評価内の per-case ばらつき) とは異なる軸である。
  **両者を混同して「Z-3 で seed ノイズ論点が解決した」と読まないこと** — 訓練 seed 間分散の
  測定は Z-5 が担当する。
- **プラセボ対照 (Z-4) もまだ無い**: 本節は「退行が 0 と識別できるか」までで、
  「退行が RAFT の選抜方針に起因するのか、低 LR 追加 SFT 一般の副作用なのか」は切り分けていない。
- F1 の差分布の同値 (diff=0) は diverse_a 525 件・diverse_b 471 件ある (JSON `diff_summary.n_zero`)。
  そのうち両モデルとも F1=0 のケースは **diverse_a 42/525・diverse_b 26/471** にとどまる
  (record-writer が npz から数えた値で、JSON には無い)。**大半は F1>0 で値が一致したケース**である。
  それが同じタグ集合を出したため (= そのケースではモデル差がない) なのか、別の集合で F1 だけ一致したのかは**未検証**。
  「同値は F1 の解像度の限界でモデル差の限界ではない」とは言えない。ここは判定材料として使っていない。
  (2026-10-05 訂正。旧記載は「交わり 0 件で両モデルとも F1=0」を例に、同値は解像度の限界だと断定していた)
- **判定 (採否・有意の断定) はしていない**。CI が 0 を除外するという事実の提示のみで、
  「よって構造的に退行する」と結論づけるのは本節の外 (それを言うには Z-4/Z-5 が要る、というのが
  plan の設計)。

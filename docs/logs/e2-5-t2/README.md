# docs/logs/e2-5-t2 — E-2.5 T1 (soft recall 指標) の検証ログ

判定・経緯の**正本は `docs/image-quality-plan.md` §E-2.5**。ここはファイルの索引と扱い上の注意のみ。

★**ディレクトリ名は `e2-5-t2` だが、中身は T1 (= `scripts/dollma_eval_image_grid.py` への
soft recall 実装) の検証ログである**。E-2.5 の T1/T2 の区分を定義した記録はリポ内に無い
(発注文はリポ外) ため、名前の「t2」が何を指すかは**未確定**。
★**新規の SDXL 生成は 1 枚も無い**。入力はすべて E-1 の既存 16 枚 (`docs/logs/e1/img/`)。
★**出自**: 本ディレクトリの 5 ファイル・`docs/logs/e1/grid_results_rescored.csv`・
`scripts/dollma_eval_image_grid.py` の変更の原本は、**共有 main checkout (`E:\Develop\Projects\dollama`) に
未 commit のまま残っていたもの** (mtime 2026-09-27 22:55〜22:57) である。2026-09-28 に main thread が
本 worktree へコピーした (コピーした主体は main thread の申告)。記録執筆時に両側 7 ファイルの sha256 を突合し、
**全ファイルがバイト同一**であることを確認した。main checkout 側の原本は、本記録の commit を push した後に
main thread が片付ける予定。
★**走行日・走行機・Python 実行系は記録されていない**。再採点の走行場所は **main checkout と推定**する
(WD14 IR `models/wd14-swinv2-tagger-v3/` は main checkout 側にだけあり、worktree には `models/` が無い)。
mtime はコピーで変わりうるので、走行日や走行場所の断定には使えない。
なお本 README 執筆時の `--selftest` 再実行は**研究機** (`KIK-WIN-RTX58`) で行った (開発機では未確認)。

## ファイル一覧

| ファイル | 出自 | 扱い |
|---|---|---|
| `selftest_stdout.log` | `scripts/dollma_eval_image_grid.py --selftest` の標準出力 (推定・コマンド行自体は保全されていない)。合成 fixture による純ヘルパ検査で、OpenVINO / PNG / exe に依存しない | **24 件 PASS + `ALL PASS`**。★文字コードは UTF-8 ではない (日本語部分が UTF-8 として読めない。Windows コンソール既定の cp932 と推定・未検証) |
| `grid_results_rescored.csv` | `--rescore-dir docs/logs/e1` の出力 (`run_rescore` は出力先を `--rescore-dir` 自身に書く) を本ディレクトリへ複製したもの | ★**`docs/logs/e1/grid_results_rescored.csv` とバイト同一** (sha256 `8791fdb46f54a13f0ec0a53f5384aa6b65b88d65b524e5d92ee4d6225dfc3d14` が両者で一致)。別走行ではない |
| `analyze_smoketest.log` | 実走者による**注釈付きの要約** (コマンド行 + `→` 以下の説明文)。生の stdout / traceback ではない (生出力は 11 行目の 1 行のみ・文字化けあり) | 下記「既知の欠陥」の記録。★traceback は保全されていない |
| `e1_with_fake_cfg_smoketest_only.csv` | `grid_results_rescored.csv` の全 16 行に**定数 `cfg=7.5` 列を人工的に足した**もの | ★★**偽データ。スモークテスト専用**。E-1 走行は cfg 列を持たず (E-2 T2 の cfg 軸追加より前の走行)、`7.5` は実測の条件ではなく「既定 7.5 相当だったはず」という仮置き。**この CSV を分析・判定の入力にしないこと** |
| `analysis_summary.txt` | E-1 の 16 行 (`grid_results_rescored.csv`) から distinct 数・語彙外語・seed 間 std・SNR を出した要約 | ★**生成したスクリプト/コマンドは保全されていない** (リポ全体を `soft_recall` で grep してヒットするスクリプトは `dollma_eval_image_grid.py` のみで、同 script に distinct 数や SNR を出す処理は無い)。数値の一部は記録執筆時に CSV から手計算で検算した (正本 §E-2.5 参照)。**characterization であって判定ではない** |

## 生成コマンド (保全されている範囲)

`analyze_smoketest.log` に残っているコマンド行 (すべて cwd = リポジトリ root と読める相対パス):

```
python scripts/dollma_eval_image_grid.py --rescore-dir docs/logs/e1
python scripts/dollma_e2_t2_analyze.py --csv docs/logs/e1/grid_results_rescored.csv --metrics recall soft_recall soft_recall_iv soft_recall_logit
python scripts/dollma_e2_t2_analyze.py --csv docs/logs/e2-5-t2/e1_with_fake_cfg_smoketest_only.csv --ref-cfg 7.5 --metrics recall soft_recall soft_recall_iv soft_recall_logit
```

`--selftest` と `analysis_summary.txt` の生成コマンドは記録が無い。

## 既知の欠陥 (未修正)

- **`--rescore-dir` は E-1 形式 (cfg 列なし) の CSV で集計段が落ちる**。per-row の再採点と
  `grid_results_rescored.csv` の書き出しは先に済むが、その後の集計で `KeyError: 'cfg'`
  (`scripts/dollma_eval_image_grid.py` の `run_rescore` 内 `cfgs = sorted(set(r["cfg"] for r in out_rows), ...)`)。
  → **`grid_summary_rescored.csv` は生成されていない** (`docs/logs/e1/` に存在しない)。
- `scripts/dollma_e2_t2_analyze.py` (本 T1 では無改変) も同じ理由で E-1 の CSV を読めない
  (`main` 内 `cfgs = sorted(set(r["cfg"] for r in rows), ...)`)。
- 偽 cfg 列を足した CSV では analyze が落ちないことを確認した、というのが `analyze_smoketest.log` の主張。
  ただし cfg が 1 値しか無いため**比較アームが無く、3 軸判定は 1 件も出力されていない**
  = **このスモークテストは「落ちない」ことしか示していない**。

## 取り違え注意

- `grid_results_rescored.csv` の `recall` / `recall_matched` 列は E-1 原本 (`docs/logs/e1/grid_results.csv`)
  と 16/16 行で同値 (WD14 を再実行して同じ値が再現した)。`axis_*` / `worst_anatomy` / `sec` 等は
  **再計算されておらず原本の値をそのまま複写**している (`rescore_row` は `dict(row)` に recall 系だけを上書きする)。
  anatomy 列の一致を「再現した」と読まないこと。
- E-1 は **illustrious-xl 単独・4 prompt × seed 1000-1003・cfg 既定・1 条件のみ**。
  条件間の比較 (レバーの有無) はこのデータからは出ない。

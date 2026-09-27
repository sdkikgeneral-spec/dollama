# docs/logs/e2-t2 — E-2 T2 実走ログ (研究機・SAC OFF・2026-09-27)

branch `feat/e2-sampling-knobs` (main 未 merge)。判定・経緯の**正本は
`docs/image-quality-plan.md` §E-2 「現況 (2026-09-27) — T2」**。ここはファイルの索引のみ。

## DoD 2 (既定経路の無改変ゲート)

- `p{1,2,3}_dod2.png` / `.log` — E-2 worktree 側の出力 (env `DOLLAMA_SEED=1234` /
  `--preset illustrious-xl` / `--steps 20` / **`--seed`・`--cfg` 未指定**)
- `baseline-main-766082a/p{1,2,3}_base.png` / `.log` — main `766082a` (= `git merge-base`) 側の同条件出力
- `dod2_notes.txt` — 実走メモ。★**同ファイル中の `docs/logs/e2-t2-baseline/` というパス表記は誤り**で、
  現物は本ディレクトリ配下 `baseline-main-766082a/`。
- 結果: **3/3 sha256 完全一致**。★突合相手は 2-6e ではなく `766082a`
  (2-6e は E-0 = `272e854` より前の exe 産で、原理的に一致しない)。理由は正本を参照。

## DoD 5 (スイープ)

- `cfg_sweep/` — 3 prompt × seed 1000-1003 × cfg {0.0(=未指定→既定 7.5 相当), 5.0, 10.0} = **36 走行**
- `steps_sweep_s12/` / `steps_sweep_s28/` — steps 12 / 28 を各 **12 走行**
  (**steps=20 の参照アームは `cfg_sweep` の cfg=0.0 の 12 走行を再利用**)
- 合計 **60 走行** (★「72 枚」は誤り。`grid_results.csv` の行数 36/12/12 で検算可)
- 各ディレクトリ: `grid_results.csv` (1 走行 1 行) / `grid_summary.csv` (条件ごと集計) /
  `log/*.log` (60 走行分) / `contact_sheets/` / `img/` ※下記
- `analysis_cfg.log` / `analysis_steps.log` — `scripts/dollma_e2_t2_analyze.py` の 3 軸判定出力。
  **4 条件すべて `頭打ち/ノイズ支配`・有効レバー 0/4**
- `cfg_sweep_run.log` / `steps_s12_run.log` / `steps_s28_run.log` — ハーネス走行ログ

## ★commit していないもの

**`*/img/*.png` (スイープ本体 60 枚・約 240MB)** は commit 対象外 (研究機ローカルのみ)。
リポジトリへ恒久追加するかは**ユーザー/PL の決裁待ち** (E-1 は img を全数 commit しており先例とは異なる)。
DoD 2 の PNG 6 枚と全ログ・CSV は commit 済み。

## 走行条件の注記

- **`--fast` / `--fp8` は未指定 (既定 OFF)**。★ただし `src/main.cpp` は fast の実効値を**ログに出さない**ため、
  スイープ以外 (手動実走の DoD 2 分) は **argv が保全されておらず二次証拠のみ**。
- **`--seed` は 60/60 走行でログに `[gen] seed=<値>(req)`** = CLI 経路が env より優先することを実走で確認。
- ★**`--cfg` の実効値はログに出ない**。cfg が効いた一次証拠は、同一 prompt/seed で
  cfg アームごとに PNG の sha256 が相違すること (正本に 3 値を記載)。

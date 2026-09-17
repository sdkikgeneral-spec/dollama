# G-10k T7 (S4c) 生ログ (研究機 KIK-WIN-RTX58・HEAD a9e0cf1)

`docs/g10k-plan.md` §6 T7 行・§6「アンカー 3 本と、アンカー 3 の位置づけ」の**一次証拠**。
**本ディレクトリは数値採取と生ログ提出のみ。合否・削減率の判定は T7b (perf-profiler) が行う。**

## 走行ヘッダ (各ログ共通・詳細は各ファイル先頭を参照)

- HEAD: ~~`a9e0cf1b4eba94561a42eb670b14a197adaf77b0` (走行前に採取。3 プロセスとも同一 HEAD で変化なし)~~
  ★**訂正 (T8・2026-09-17)**: 3 プロセスで同一ではない。**P1 = `a9e0cf1`** (`t7_p1_default.log:5`・porcelain に
  `.claude/agents/*.md` 11 本の `M`) / **P2・P3 = `c0f6d8f`** (`t7_p2_convbatch0.log:5` / `t7_p3_default.log:5`・porcelain は
  `?? docs/logs/g10k-e2e/` のみ)。P1 と P2 の間にその 11 本だけを `c0f6d8f` (chore(agents)) としてコミットしたため。
  **`src/` 無改変・exe / `conv2d.cu` / `test_diffusion_batch2.cu` の sha256 は 3 本同一** (各ログ `:9-11`) → 判定への影響なし。
  初版 (`703ea92`) の記述は履歴として取り消し線で残す。
- `git status --porcelain`: **空ではない** (★T8 注: これは **P1 のみ**。P2 / P3 は `?? docs/logs/g10k-e2e/` のみ) — `.claude/agents/*.md` 11 ファイルが `M` (エージェント定義の更新。
  `src/` 配下・`docs/g10k-plan.md` は無改変)。タスク指示は「porcelain 空を走行前に確認」だったが、
  実際の走行前状態はこの 11 ファイルの変更を含んでいた。**判定への影響はない** (対象は `src/` 無改変の exe 実行のみ)。
- exe: `build\src\test_diffusion_batch2.exe` (cwd `build`) — sha256 `87EA4C3AF0FCC8064388CFCD52D995CAF655CC181633EEEA521511A7ED3C271B`
  (3 プロセスとも同一バイナリ・T5/T6 で使われた既存ビルド。本セッションでビルドしていない)
- `src/kernels/conv2d.cu` sha256: 各ログ参照 (3 プロセスとも同一)
- `src/tests/test_diffusion_batch2.cu` sha256: 各ログ参照 (3 プロセスとも同一)
- SAC (Smart App Control) 状態: **未確認** (裏取りしていない。ヘッダに「未確認 (裏取りせず)」と明示・
  3 プロセスとも起動即成功しており SAC ブロックは発生しなかった)
- `nvidia-smi` (pre/post): 各ログ末尾・冒頭を参照 (温度 52-56℃・電力 173-186W 帯)
- env (共通): `DB2_BENCH=1 DB2_BENCH_STEPS=20 DB2_BENCH_ITERS=1 DOLLAMA_PROFILE=1`。他の
  `DOLLAMA_*`/`PROF_*`/`DB2_*` は全て `<unset>` (走行スクリプトが明示的にクリアしてから設定・各ログ `--- env ---` 節参照)。

## 生ログ索引

| ファイル | 構成 (`DOLLAMA_CONV_BATCH`) | exit |
|---|---|---|
| `t7_p1_default.log` | P1: 未設定 (既定) | 0 |
| `t7_p2_convbatch0.log` | P2: `=0` | 0 |
| `t7_p3_default.log` | P3: 未設定 (既定・P1 と同条件の再走) | 0 |
| `scripts/t7_run.ps1` | 走行スクリプト (T6 の `t6_run.ps1` と同型・出力先/対象ファイル一覧のみ変更) |

3 プロセスは連続実行 (間に他の GPU 負荷を挟んでいない)。各プロセスは 1 プロセス内で
`test_diffusion_batch2` の全体 (steps=4 parity 節 + steps=20 `DB2_BENCH` 節) を実行する。
`DB2_BENCH` 節は 3 構成 (default / attn+batch2 合成・CLI 到達不可 / fast+epilogue = 出荷 `--fast`) を
順に回し、構成ごとに warmup 1 回 + 本走 (ITERS=1) 1 回 = 2 回 `generate_txt2img` を呼ぶ
(`DOLLAMA_PROFILE=1` の dump が構成ごとに 2 ブロック出る)。**warmup 側は lazy init を含むので
判定に使わない (記録には残す)** — T2c と同じ規律。

## resnet バケット抜粋 (`DOLLAMA_PROFILE=1` dump の `resnet (conv/groupnorm)` 行・単位 s)

| 構成 | P1 (既定) warmup / 本走 | P2 (`CONV_BATCH=0`) warmup / 本走 | P3 (既定) warmup / 本走 |
|---|---|---|---|
| default (forwards=40) | 1.069 / 1.064 | 1.084 / 1.058 | 1.078 / 1.060 |
| attn+batch2 (合成・forwards=20) | 0.998 / 0.996 | 0.993 / 0.974 | 1.013 / 0.994 |
| fast+epilogue (出荷 `--fast`・forwards=20) | 0.874 / **0.886** | 0.877 / **0.857** | 0.892 / **0.873** |

出典: 各ログの `[DOLLAMA_PROFILE / generate_txt2img (CFG path)]` dump ブロック
(steps=20 節・`unet forwards=` 行の直後に続く `resnet (conv/groupnorm)` 行)。
**warmup (判定不使用)** と本走を列内で `/` 区切りで併記した。

## e2e (harness `steady_clock` の min ms・profile ON 下・T7b は使わない/T7c が正典を別採取)

| 構成 | P1 | P2 | P3 |
|---|---|---|---|
| default | 14106.1 | 14066.9 | 14042.8 |
| attn+batch2 (合成) | 10965.8 | 10914.9 | 10925.5 |
| fast+epilogue (出荷) | 10974.3 | 10799.8 | 10838.0 |

出典: 各ログの `[test_diffusion_batch2] BENCH e2e (min ms): …` 行。

## アンカー 3 の窓 (記録のみ・結論は書かない)

比較対象: **`fast+epilogue` 構成の resnet バケット本走秒**。
T2c (conv2d 変更前・`docs/g10k-plan.md` §13 T2c 節「採取値 — resnet バケット」表) の値:
A=0.854s / B=0.857s。
本走行 P2 (`DOLLAMA_CONV_BATCH=0`) の値: **0.857s**。

帯 = ±max(アンカー1の実測ドリフト幅, 10%) (`docs/g10k-plan.md:463-471`)。
T2c 内のセッション間差 (A vs B) は (0.857-0.854)/0.854 ≈ 0.35% であり、
床の 10% が適用されるので帯は ±10% (= T2c=0.857 を基準にすると 0.771〜0.943s)。
**本走行 P2 の値 0.857s と帯 [0.771, 0.943] の関係の記録のみ (帯内/帯外の判定・結論は T7b へ委ねる。ここでは書かない)。**

## SAC ブロック

3 プロセスとも起動即失敗のような症状はなく、`exit=0` で完走した。SAC によるブロックは**発生しなかった**。

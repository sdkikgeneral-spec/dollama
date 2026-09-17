# G-10k T6 / T6r 生ログ (研究機 KIK-WIN-RTX58・SAC OFF・HEAD 0ff28c9)

`docs/g10k-plan.md` §13「T6」「T6r」の**一次証拠**。数値の位置づけ・合否設計は同 §6 T6 行 / §8「ケース B' と T6r」を
見ること。ここは所在と読み方のみ。**判定は台帳の数値条件どおりに機械的に行い、それ以外の解釈は加えない。**

- 走行時 HEAD: `0ff28c9` (T6 / T6r とも・各ログのヘッダ `HEAD =` が一次証拠)。`src/kernels/conv2d.cu` sha256
  `31F6C8C6…` / `gemm.cu` `4EEC27B5…` は T4 最終 (`docs/logs/g10k-t4/t4_index.txt`) と同一。
  (監査 軽微-2 で出典を生ログへ) sha の一次は T4 生ログ `docs/logs/g10k-t4/t4_default_run2_final.log:9-11`
  (`:11` = gemm.cu) と T5 索引 `docs/logs/g10k-t5/t5_index.txt:36-37` (conv2d.cu / gemm.cu)。
  `t4_index.txt` は `:26` (conv2d.cu) / `:28` (gemm.cu) に同値。
- SAC: `VerifiedAndReputablePolicyState = 0` を各ヘッダで採取。
- 走行は全てフォアグラウンドで 1 本ずつ (並列なし)。

## T6 (2026-09-12) — 先行必須ゲート・合否あり

| ファイル | 内容 |
|---|---|
| `t6_conv2d_default_run1.log` | `build\src\test_conv2d.exe` (sha256 `1FEF7939…` = T5 と同一 exe) を env 全 `<unset>` (= `DOLLAMA_CONV_BATCH` 既定 = batched ON) で 1 プロセス実走。ヘッダ (HEAD / porcelain / exe・ソース sha256 / SAC / nvidia-smi / env 19 変数) + stdout+stderr (`2>&1`) + 末尾 `exit=0` |
| `t6_conv2d_default_run2.log` | 同条件の再走 (再現性確認) |
| `scripts/t6_run.ps1` | 上記の走行スクリプト (T5 の `t5_run.ps1` と同型・出力先のみ `g10k-resnet`) |

読み方:

- T6 の被験行は **`[bench_batch]` 5 本** (run1 / run2 とも `:205-209`)。形式は
  `[bench_batch] <形状> N=2 batch median=<ms> | N=1x2 seq median=<ms>`。合否は §6 T6 行の
  **batched median / seq median ≤ 0.95 を全形状で** (合否対象 = `rep_320_128` / `rep_640_64` / `rep_1280_32` /
  `G4_band_640to320_128`。`unet_c320_64` は G-2k 由来の参考行)。比はログに印字されないので自分で割る。
- 同じログには T4 のゲート `[G2a]`〜`[G5]` と `[bench_conv2d]` (N=1 ベンチ) も含まれるが、それらは T6 の被験ではない
  (`[G1:*]` は既定プロセスでは `n/a` = 設計どおり・`DOLLAMA_CONV_BATCH=0` プロセスは T5 の `t5_conv2d_convbatch0.log`)。
- **per-call ms であり e2e 秒には翻訳しない。** `prof_unet_fast_warm` は使っていない (§2 F1)。

## T6r (2026-09-15) — 律速診断・src コミットなし

計測 exe `prof_conv_breakdown` は**コミットしない作業ツリー限定** (§8 ケース B' の規定)。走行時の porcelain は
`M src/meson.build` + `?? src/tests/prof_conv_breakdown.cu` (+ 本ディレクトリ)。`conv2d.cu` / `gemm.cu` は無改変。

| ファイル | 内容 |
|---|---|
| `t6r/summary.log` | ヘッダ (HEAD / porcelain / 計測 exe と `conv2d.cu` `gemm.cu` `prof_conv_breakdown.cu` の sha256 / SAC / nvidia-smi / iters=50) + 10 走行の `[<tag>] exit=0` + 終了時刻 |
| `t6r/<形状>_<mode>.nsys-rep` (10 本) | nsys 2026.1.3 `profile --capture-range=cudaProfilerApi --capture-range-end=stop --trace=cuda` の原本。`<mode>` = `batched` (N=2 を 1 回) / `seq` (N=1 を 2 回) |
| `t6r/<形状>_<mode>_cuda_gpu_kern_sum.csv` (10 本) | `nsys stats --report cuda_gpu_kern_sum --format csv` のカーネル名別集計 (Total / Instances / Avg / Med / Min / Max / StdDev / Name) |
| `t6r/<形状>_<mode>.stdout.log` (10 本) | nsys ラッパの stdout。★**計測 exe 自身の `[prof_conv_breakdown] …` 行 (cudaEvent 中央値) はここに載っていない** (nsys 経由で失われた)。T6r の数値は CSV (= nsys 集計) だけが一次 |
| `t6r/cudaevent_direct.log` | ★留保是正 (同日 23:39): **同一 exe (sha256 `4E1B2B56…`) を nsys なしで直接実行**した `[prof_conv_breakdown]` 行 10 本 (5 形状 × batched/seq・iters=50・per-call median と min)。同一 exe 内の cudaEvent vs nsys 突合はこれが一次。★同一 exe だが**別プロセス** (nsys 走行 23:19 の 20 分後) かつ**モード別プロセス** (batched/seq を交互に採っていない・T6 は同一プロセス内で交互)。env 未採取 (下記)。ヘッダは start/HEAD/exe sha256/nvidia-smi (pre) のみ (porcelain・SAC・env・`exit=`・post 無し) |
| `t6r/breakdown_table.txt` | `scripts/t6r_table.py` が CSV から生成した per-call 内訳表 (us/call = `Total Time (ns)` / 1e3 / 50 = iters 平均・**中央値ではない**)。カーネル名 → 列: `im2col` / `cutlass` `nvjet` `gemm` → GEMM / `bias` / `scatter` |
| `scripts/t6r_nsys.ps1` | 5 形状 × 2 モードの実走 + CSV 生成スクリプト |
| `scripts/t6r_table.py` | 内訳表の生成スクリプト (`python docs/logs/g10k-resnet/scripts/t6r_table.py > docs/logs/g10k-resnet/t6r/breakdown_table.txt`) |
| `scripts/prof_conv_breakdown.cu` | 計測 exe のソース退避 (作業ツリーの `src/tests/prof_conv_breakdown.cu` と同一 sha256 `29E1ED56…`・`summary.log` ヘッダの値) |
| `scripts/prof_conv_breakdown.meson.diff` | `src/meson.build` への一時追加 (`prof_conv_breakdown` executable 1 ブロック) の diff |

読み方・注意:

- ★**T6r は nsys 走行・直接実行とも env を採取していない** (`t6r_nsys.ps1:11` は 4 変数を Remove するだけでヘッダに
  書かない・`summary.log` / `cudaevent_direct.log` に env 節なし)。batched 経路が効いた証拠は CSV のカーネル名
  (`im2col_fp16_batched` / `conv_bias_add_rows_batched` / `scatter_band_to_out_batched` が batched 側のみ) に求める。
  直接実行の方はカーネル名が無いので env 既定は未検証。
- **`.sqlite` (nsys stats の中間ファイル) は削除済**。CSV を作り直すときは
  `nsys stats --report cuda_gpu_kern_sum --format csv --force-export true -o <出力> <tag>.nsys-rep`。
- nsys 集計は GPU カーネル時間の合計で、カーネル間のギャップを含まない。T6 の `[bench_batch]` (cudaEvent) とは別計器。
  **突合は「比の向きと桁が同じ」まで** (§13 T6r 留保)。
- `G4_band_640to320_128` の batched 側は im2col / GEMM / scatter が Instances=100 (= 50 iters × 帯 2 本)、
  bias は 50。seq 側の Instances=100 は n=0,1 の 2 発。
- stdout ログの「CPU sampling requires administrative privileges, disabling」は非管理者実行による nsys の警告で、
  GPU カーネル時間の集計には影響しない。
- T6r 自体は §8 ケース B' の規定 (perf-profiler 主・cuda-kernel-dev 協働) と異なり main thread が単独で実施した
  (§13 T6r 冒頭に記載)。

## docs をコミットするときの手順 (T6r の作業ツリー改変を混ぜない)

T6r の計測 exe 用改変 `M src/meson.build` / `?? src/tests/prof_conv_breakdown.cu` は**コミットしない** (§8 ケース B')。
本ディレクトリ (`docs/logs/g10k-resnet/`) と `docs/g10k-plan.md` を commit するときは、**`git add -A` を使わず**
パスを明示して `git add docs/g10k-plan.md docs/logs/g10k-resnet` とする (§10 規律)。`src/meson.build` の改変は
退避 diff (`scripts/prof_conv_breakdown.meson.diff`) と一致することを確認したうえで `git checkout -- src/meson.build`
で戻してよい (`.cu` は未追跡のまま放置で commit に入らない)。commit 後に `git show --stat HEAD` で `src/` が 0 行で
あることを確認する。

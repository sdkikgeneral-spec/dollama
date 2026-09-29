# docs/logs/f0b-z1 — F-0b 再検証 Z-1 の一次成果物

F-0b 再検証の Z-1 (quality 抜き勝者一致率・寄与率) を算出したときの出力と実行記録を置く。
**定義・数値・読み方の注意・限界の正本は `docs/f0b-reverification-plan.md` の「### Z-1」節の「Z-1 結果」**である。
本 README は、出自と再現手順だけを扱う。

## 中身

| ファイル | 何 |
|---|---|
| `z1_result.json` | 本計算の出力 (UTF-8)。`inputs` (パスと sha256) / `env` / `constants` / `winner_match` (group ごとの `records` 400 件を含む) / `contribution` / `notes` |
| `z1_run_log.txt` | 実行記録。日時・環境・入力の sha256・コマンド・標準出力の要約 |

生成元のスクリプト: `scripts/dollma_f0b_z1_quality_contribution.py` (JSON・実行記録・スクリプトの 3 本は commit `11962db`)

## 出自

- 実行日: 2026-09-29 (`z1_run_log.txt` に記載)
- 実行場所: git worktree `E:\Develop\Projects\dollama\.claude\worktrees\agent-a3fa76555e564d19a` (JSON `env.cwd`)
- 環境: Python 3.14.6 / Windows-11-10.0.26200-SP0 (JSON `env`)
- 既存データの再解析のみで、SDXL 生成も LM 訓練もしていない。
  main checkout の `data/` へは読み取りだけで、書き込んでいない (`z1_run_log.txt` の申告)。

## 入力 (main checkout・gitignore 対象のため repo には無い)

| ファイル | sha256 |
|---|---|
| `E:\Develop\Projects\dollama\data\rollouts\candidates_bestofn.jsonl` | `a38fba8b9dadec8ec68e8a1d94fc3813653fe932e6cbca2623a7007f728142c5` |
| `E:\Develop\Projects\dollama\data\rollouts\sft_bestofn.jsonl` | `032470f2d4bff6e42489290a6d53a28bbe7ad8436dc1884281b0517055a57b0a` |
| `E:\Develop\Projects\dollama\data\rollouts\g2b_prepost.jsonl` | `ba2db3bec55157ab24e8f10c71e7498ae04997aeeba92c4b24eb326820699e89` |

上の値は JSON `inputs.*.sha256` と同じである。2026-09-29 に record-writer が `sha256sum` で再計算し、3 本とも一致を確認した。
同じディレクトリには `*.done101.bak.jsonl` という名前の別ファイルがある。**Z-1 の入力はそちらではない**。

## 再現コマンド

リポジトリのルートで実行する。

```
python3 scripts/dollma_f0b_z1_quality_contribution.py --selftest   # 合成データの自己検査・exit 0 で OK
python3 scripts/dollma_f0b_z1_quality_contribution.py              # 本計算 → docs/logs/f0b-z1/z1_result.json を上書き
```

- 入力パスの既定値は、スクリプト内に `E:\Develop\Projects\dollama\data\rollouts` として**絶対パスで固定**されている。
  別の場所で再現するときは `--candidates` / `--sft-bestofn` / `--g2b-prepost` / `--out` で差し替える。
- 再実行すると `z1_result.json` は上書きされる。記録を保持したまま検算したいときは、`--out` を別パスにすること。

## 注意点

- **コンソールでは `Δ` が文字化けする**が、JSON ファイル自体は UTF-8 で正しい。
  `z1_run_log.txt` の申告では U+0394 のバイト (`\xce\x94`) を確認済みとしている。
  JSON を Python で読むときは `encoding='utf-8'` を指定すること。
  Windows の既定コードページのまま `print` すると、日本語の値も化ける。
- **行番号参照が腐っている**: スクリプト docstring と JSON `plan_ref` にある「`docs/f0b-reverification-plan.md` l.45, l.55-58」は、
  同 doc を 2026-09-29 に更新する前の行番号である。現在は節名 (「分割タスク台帳」の Z-1 行 / 「### Z-1」節) で辿ること。
- **一致率と寄与率は別データセットから出している** (一致率 = G-1 訓練側 400 入力 / 寄与率 = G-2b held-out 100 入力・post_id の重なり 0)。
  合算や直接比較をしないこと。そのほかの限界 (lang 結合の出自は未検証・空候補 1 件の除外方針は plan 未記載) は plan の Z-1 節を参照。
- **採否判定は含まない** (JSON `notes`)。

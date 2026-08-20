# G-8k S4b 実走 生ログ (2026-08-19・研究機・SAC OFF)

`docs/measurements-log.md` の「G-8k S4b e2e アリーナ A/B 再走」行の**一次証拠**。
S5d (2026-08-20) で、消える temp (セッション ID 付き scratchpad) から repo へ**無改変のまま**退避した。

- 退避元 (原本の所在):
  `C:\Users\sdkik\AppData\Local\Temp\claude\e--Develop-Projects-dollama\a052f05e-4438-4868-9235-e41ffd700a76\scratchpad\`
- ハーネス: `src/tests/prof_arena_e2e.cu` (`build/src/prof_arena_e2e.exe`。**meson test には登録せず build ターゲットのまま**)
- HEAD: `acca803` (S3c) の**ソース無改変**・再ビルドなし
- 共通 env: `PROF_IMAGES=3 PROF_STEPS=20 PROF_G=7.5 PROF_FAST=1 PROF_SAMPLE_MS=5 DOLLAMA_PROFILE=1`

| ファイル | 走行 | 構成 |
|---|---|---|
| `s4b_roundA_1_pool0.log` | A-1 | 基準 `DOLLAMA_POOL=0` |
| `s4b_roundA_2_default.log` | A-2b | 既定 (reserve ON) ※下記「棄却」参照 |
| `s4b_roundA_3_reserve0.log` | A-3 | 対照 `DOLLAMA_ARENA_RESERVE_MB=0` (= S3 挙動) |
| `s4b_roundB_3_reserve0.log` | B-3 | 対照 (ラウンド B は逆順 3→2→1) |
| `s4b_roundB_2_default.log` | B-2 | 既定 |
| `s4b_roundB_1_pool0.log` | B-1 | 基準 |
| `s4b_roundC_steps2_default.log` | C | 既定・`PROF_STEPS=2` の補走 |
| `smi_timeline.txt` | — | 各走行の前後で採った `nvidia-smi` の温度 / SM クロック / 常駐 / 電力 |
| `DISCARDED_A2_overlap_risk.log` | A-2 (棄却) | **採用しない**。下記参照 |

## 読むときの注意 (誤読しやすい点)

- **`DISCARDED_A2_overlap_risk.log` は採用値ではない**。初回 A-2 をバックグラウンド起動したところ
  「即完了・出力なし」と返って実際は遅延起動し、同名ログへ 2 プロセスが書く恐れが出たため条件の清潔さを
  保証できず破棄し、A-2b として撮り直した。破棄分の値も採用値と整合していたが、**結論はこの再走に依存しない**。
  再現時は**フォアグラウンドで 1 本ずつ**走らせること。
- **`smi_timeline.txt` の pre エントリ 7 個は「採用 7 走行」と 1:1 対応しない**。内訳は
  A-2(棄却)・A-2b・A-3・B-3・B-2・B-1・C であり、**基準走行 A-1 の開始時状態は記録が無い**。
  また `pre-A2` / `post-A2` には時刻欄が無いため、A-1→A-2(棄却)→A-2b の 2 区間のクールダウンは未検証
  (時刻の取れた 5 区間の実測は 71/71/68/70/85s)。
- **`DOLLAMA_POOL=0` の走行でも `[ALLOC] reserve: unet=6080MiB unet_persist=176MiB` の行が出る**
  (`reserve_arenas()` の printf が callee 側の no-op 判定より手前にあるため)。実体は no-op なので、
  判定は**終了時の `cap` / `reserved` / `chunks`** で行うこと。
- `PEAK_USED` は `cudaMemGetInfo` の total−free = **device-wide** (他プロセス分を含む)。
  「プロセス GPU peak」と呼ばないこと。

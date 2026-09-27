# 絵の更新 (2-6d 後続) 施策台帳 — E-0 〜 E-5

起票 2026-09-19 (record-writer)。**この文書は台帳であって、決裁済みの完了記録ではない。**
各項目の状態は「発火条件」欄が真になるまで未着手。数値・状態はすべて下記「一次証拠」から取っており、
未測定の値は書かない (書くときは「未測定」と明示する)。

対象は **2-6d (アニメ特化 SDXL 3 preset・既定 illustrious-xl) の後続として「出力画像の質」を上げる施策**。

⚠ **ID の衝突注意**: `E-1` / `E-2` という記号は本台帳とは**別の意味**で既に使われている。判明分:

| 既存の用法 | 出典 |
|---|---|
| 「E-1 実走 = 実 PNG 180 枚 (良90/悪90)」= ScorerNet 用コーパス生成ジョブ | `docs/dataset-spec.md` l.766 |
| 「分布 (E-1 180 枚)」= 同コーパスの quality 分布 | `docs/dataset-spec.md` l.869 |
| 「E-1 研究機生成コーパス (実 SDXL 生成 PNG 180枚)」= B-3b ScorerNet 訓練の入力 | `docs/measurements-log.md` l.63 |
| 「E-1 コーパス 180枚」= Q-1 waifu 採点の対象 | `docs/measurements-log.md` l.68 |
| 「E-2 再計算」= Q-2 reward 実験 | `docs/q2-quality-branch-plan.md` l.123 |

★**旧 E-1 は本台帳の新 E-1 と機能も近い** (どちらも「SDXL で生成 → WD14 系で採点」)。
用途は違う (旧 = ScorerNet 訓練コーパス作り / 新 = 条件比較の物差し) が、文字列検索では区別できない。
本台帳の E-0〜E-5 を他所で参照するときは必ず **「image-quality-plan の E-n」** と書くこと。

---

## 0. 共通規律 (全項目に適用)

1. **既定経路の無改変ゲート**: 新しいノブ/機能を**未指定**のとき、出力 PNG の sha256 が現行と一致すること。
   流儀は 2-6e と同じ — 2-6e は出荷経路の PNG sha256 が 2-6d 走行と 3/3 一致することで
   「出荷経路が 2-6d 目視評価の条件と bit 同一」を示した (measurements-log l.555)。
   ⚠ **この基準線には未検証の穴が継承されている**: 2-6d 走行も 2-6e 検証走行も **`--fast` の有無が
   記録されていない** (「CLI 生成モードは fast フラグをログしない」= measurements-log l.543 の 2-6d 残債 2 /
   同 l.556 の 2-6e「未検証」欄)。したがって「sha256 が 2-6d/2-6e と一致した」で言えるのは
   **「拡散条件が当該走行と同一である」**ところまでで、**その条件が `--fast` 有か無かは未確定**。
   本台帳で sha256 一致をゲートに使うときは、この未検証を毎回引き継いで書くこと
   (「`--fast` 相当と確認済」と書かない)。新規走行では `--fast` の有無を**走行条件として明示記録**する。
2. **N=1 で優劣を書かない**。2-6d の 12 枚比較には
   「**N=1 (seed 1 本) で優劣の根拠にはならない**」(measurements-log l.528) と明記がある。
   preset/ノブ/checkpoint の優劣判定は**多 seed でのみ**行う (物差し = E-1)。
   N=1 の観察は「characterization (判定なし)」と明示して残す。
   ★**ただし「多 seed で測った」だけでは判定にならない。** 本プロジェクトが Phase 4 で確立した
   判定基準は**3 軸すべての成立**であり、これを画質評価にもそのまま適用する:

   | 軸 | 内容 | 使われた実例 |
   |---|---|---|
   | (a) 符号一致 | delta が**全 seed で同符号**。1 本でも反転したら seed ノイズ | D6 = 4 seed で +/−/+/− 符号反転 → 不採用 (measurements-log l.49)・A12k seed42 反転 (同 l.72)・D80 seed7 反転 (同 l.73) |
   | (b) 効果量 > 分散帯 | 効果量が**参照アームの seed 分散帯 (sd)** を上回る | B-2 = 分散帯の ~6.7–8x で成立 (同 l.58)・D5 = +0.009 で (b) のみ不成立ゆえ「小幅」止まり (同 l.56) |
   | (c) paired CI | **各 seed の paired bootstrap 95%CI が 0 を除外** | B (同 l.57)・B-2 (同 l.58) で成立 |

   このうち (b) を評価するには **ノイズ床 = 参照アームの「同条件・異 seed」sd** を同じ走行で測る必要がある。
   **参照アーム単独の多 seed 走行をノイズ床としてセットで回すこと**を全比較の前提にする
   (床を測らずに delta だけ出した走行は判定に使えない)。
   ⚠ **画質評価での注意**: 上の 3 軸はいずれも**タグ集合メトリクス (LM 側)** で確立したもので、
   画像側メトリクス (WD14 再現率 / anatomy 軸) に適用した実績は**本台帳時点で無い**。
   E-1 が最初の適用例になる (=しきい値感は E-1 の実測で校正する。**未検証**)。
3. **記録の分離**: docs / CLAUDE.md 計測表 / commit 本文は **record-writer が書き record-auditor が検査**
   (CLAUDE.md 「実装作業のルール」4)。実装・ベンチ担当エージェントに docs 本文を書かせない
   (実装側は生ログ・数値・実行条件を報告として返すところまで)。
4. **Sonnet 実装/実走の後は必ず Opus high レビュー**を挟む (プロジェクトのサブエージェント規定)。
5. **新規 exe の実走前に SAC OFF をユーザーへ依頼**する (研究機の Smart App Control が新規/変更 exe を
   ブロックするため)。依頼→実走→報告までを 1 まとまりにし、実走できなかった場合は
   「未実走」と書く (ビルド緑を実走の代わりに書かない)。
6. **秒は今回の対象外**。E-0〜E-5 で速度が動いても **CLAUDE.md 計測表に行を足さない**。
   秒はハーネス CSV に **characterization** として残すのみ。速度レバー (attention 4.60s など) は
   本台帳のスコープ外 (下記「スコープ外」)。

---

## 1. スコープ外 (この台帳では扱わない)

- **G-10k の後片付け**: worktree `dollama-g10k-wt` / ブランチ `feat/g10k-conv-true-batch2` の main merge。
- **attention 4.60s の速度レバー** (および resnet ≤0.95s ゲート)。秒数の施策は本台帳に入れない。
- **Phase 5-2 (img2img 幾何注入) / 5-3 (ControlNet)**。
  **Phase 5-1 (崩壊境界 probe)** については、E-1 が先行実装を兼ねられる**可能性がある**が
  **これは未検証の見込み** (後述 E-1 節。roadmap l.217 は 5-1 の**内容説明**であって、兼用可否の根拠ではない)。

**将来検討リスト (本台帳では扱わないが、既存記録に残っている画質レバー)**
台帳の E-0〜E-5 に入れていないもの。起票するかは別途決裁:

| レバー | 出典 | 本台帳との関係 |
|---|---|---|
| **reward 設計**の見直し (anatomy が死んでいる → quality 主体でよいか / 新軸) | `docs/f0b-rejection-sft-plan.md` l.116 (a) / `docs/roadmap.md` l.321 | Phase 4 F 側の軸。E-1 の採点器 (ScorerNet) の分解能問題と**同じ根**を持つ (共通規律 2 の注記参照) |
| **日本語条件付けトークナイザ改修** (日本語プロンプトが空条件化している) | 同 l.116 (b) / roadmap l.321 / `[[project_expression_fidelity_gap]]` | プロンプト側の質レバー。E-2 の sampling ノブとは直交 |
| **SDXL seed 制御 (HTTP に seed 引数)** | 同 l.116 (c) / roadmap l.321 | ★**本台帳の E-2 と同一物**。E-2 DoD に取り込み済 (下記 E-2・重複起票しないこと) |
| 教師枚数増 M 拡大 | 同 l.116 (d)・優先度低と自記 | Phase 4 F 側 |
| **HTTP API preset フィールド / UI preset 選択** (2-6d 残債 4・未着手) | `docs/measurements-log.md` l.545 | E-1 の格子が preset 軸を回す際、HTTP 経路では preset を指定できない制約の原因。E-1 が CLI 経路を採る理由の一つ (下記 E-1) |

---

## 2. 項目一覧 (概観)

| ID | 目的 (要旨) | 担当 | 機械 | 発火条件 | 依存 | 状態 (★2026-09-27 時点) |
|---|---|---|---|---|---|---|
| E-0 | 既定 preset の VAE `scaling_factor` 残債の解消 | cuda-kernel-dev → gpu-benchmarker | 研究機 | 即時 (発注済 ※) | なし | 完了・main merge 済 (実装 `272e854` / merge `2d442b3`) |
| E-1 | 多 seed 画質評価ハーネス | gpu-benchmarker | 研究機 | 即時 (発注済 ※) | なし | 完了・main merge 済 (`f72d595` / merge `d8f9c0b`)。★T2 で cfg 軸を追加 = **その変更は E-2 branch 上で未 merge** |
| E-2 | sampling ノブ (steps/CFG/**seed**) の露出 + スイープ。**HTTP `seed` を含む** | cpp-implementer → gpu-benchmarker | 実装=開発機可 / 実走=研究機 | E-1 完了後 | E-1 | **T1+T2 完了 = DoD 1-6 充足。ただし branch `feat/e2-sampling-knobs` は main 未 merge・未 push** (詳細は E-2 節「現況 (2026-09-27) — T2」) |
| E-3 | scheduler 拡張 (Karras / DPM++ 2M / v-pred) | cpp-implementer + model-converter → gpu-benchmarker | 実装=開発機可 / 画評価=研究機 | **条件発火**: E-2 が頭打ちを示したときのみ | E-1, E-2 | **発火条件は充足** (E-2 T2 で有効レバー 0/4)。**充足 = 着手決裁ではない**・E-3 が効く見込みは未裏付け。競合候補は E-3 節「次のレバー候補」 |
| E-4 | 追加 checkpoint 候補の調査→変換→評価 | model-converter → gpu-benchmarker | 研究機 | **条件発火**: E-1 の物差しが立ってから | E-1 (v-pred 候補は E-3) | 未着手 |
| E-5 | ランタイム LoRA (L-2) の実重み e2e 検証 | gpu-benchmarker | 研究機 | **E-2 の HTTP `seed` 完了後** (理由は E-5 節) | E-2 (HTTP seed)・E-1。重み入手/ライセンス = ユーザー決裁 | 未着手。発火条件は **branch 基準では充足・main 基準では未** (E-2 未 merge) |

※ **「発注済」の出典**: リポ内に対応する branch / commit / issue は**無い** (2026-09-20 時点で
`git log` / `git branch` に該当なし)。本欄は **main thread からの口頭発注 (2026-09-19・リポ外の状態)** を
record-writer が転記したものであり、**一次証拠はリポ内に存在しない**。裏取りが要るときは main thread に確認すること。
(※ E-1 については、2026-09-20 時点で `scripts/dollma_eval_image_grid.py` と `docs/logs/e1/` が
**未追跡ファイル (`git status` の `??`)** として実在する = 発注が実作業に至っていることの間接証拠にはなる。下記 E-1 節。)

---

## E-0 — 既定 preset (illustrious-xl) の VAE `scaling_factor` 残債

**目的**
既定 preset の VAE スケーリングの不整合を解消する。一次証拠の状態は次のとおり:

- illustrious-xl の HF `vae/config.json` は **`scaling_factor = 0.18215`** (measurements-log l.537 の実測記述)。
- 自作ランタイムは `src/infer/diffusion.cu:41` で
  `constexpr float kScalingFactor = 0.13025f;` の**固定値**。preset 別の切り替え経路は無い。
- `preset.json` は `vae_scaling_factor=0.18215` を持つが、`vae_weights.safetensors` の sha256 は
  animagine-xl-4 (config 0.13025) と**完全一致**。ただし
  「scaling_factor は UNet 学習側の規約であって VAE 重みの属性ではないため、**同一バイトから HF config の
  誤記とは結論できない**」(measurements-log l.540・record-auditor 指摘で論理撤回済)。
- 現状は `0.13025` で decode しており、2-6d の illustrious-xl 3 枚は**目視異常なし・定量未検証**、
  main thread 判断で「実害なし」扱い。**正しい scaling は未確定** (measurements-log l.541)。
- 処置 (export スクリプトの改修案) は **未実施** と明記されている (measurements-log l.542)。
- ★**再変換は現行スクリプトではできない**: 「commit 済スクリプト (`bafe0a5`) は 0.13025 以外で
  `SystemExit(1)` するため、**現行スクリプトでは illustrious-xl を再変換できない**」(measurements-log l.538)。
- ★**現物の preset は commit 前の版で作られている**: `preset.json` の `converted_at` 22:15 JST は
  スクリプト mtime 22:25 / commit 23:13 より**前** = 「commit 前の版で変換された成果物」
  (measurements-log l.538-539)。つまり `models/presets/illustrious-xl/` は
  **リポ内の現行スクリプトでは再現できない成果物**である。

したがって E-0 は「0.18215 が正しい」という前提から始めない。**どちらが正しいかを実測で決める**のが目的。
そして **DoD 1 (0.18215 側でも生成して比較) は、上の 2 点を先に片付けないと着手できない** —
0.18215 側の生成には ① export スクリプトのガード緩和 (measurements-log l.542 の処置案) か
② ランタイム側で scaling を差し替える手段 (現状 `diffusion.cu:41` の `constexpr` = 再ビルドが要る)
のどちらかが必要で、**どちらも未実施**。E-0 の第一歩はこの経路確保であって、生成ではない。

**DoD (何が緑なら完了か)**
0. **0.18215 側で生成できる経路が確保されている** (上記のとおり現状は不可)。どの手段を採ったか
   (export スクリプト改修 / ランタイム差し替え / 別経路) を記録する。
1. 0.13025 / 0.18215 の両方で同一 seed 集合の生成を行い、**多 seed** (E-1 の物差しがあればそれを使う。
   無ければ最低でも複数 seed) で差を定量化した生ログがある。**N=1 の目視で決めない**。
   判定は共通規律 2 の **3 軸 (符号一致 / 分散帯超え / paired CI)** で行い、**ノイズ床 (同条件・異 seed sd)**
   を同じ走行で測ること。
2. 採用値の根拠が書かれている。`preset.json` の config 値と実使用値が**両方**記録される
   (measurements-log l.542 の処置案の趣旨)。
3. 既定経路の無改変ゲート: 採用値が 0.13025 のままなら、出力 PNG sha256 が 2-6e の
   p1/p2/p3 (`136d388f…` / `3765065d…` / `48553e0b…`・measurements-log l.555) と一致すること。
   ⚠ 一致で言えるのは「2-6d/2-6e 走行と拡散条件が同一」までで、**その条件の `--fast` 有無は未確定**
   (共通規律 1 の注記)。新規走行の `--fast` 有無は条件として明示記録する。
   採用値を変える場合は「変わること」自体が意図であり、**変更前後の sha256 を両方記録**する。
   ★**採用値の変更にはユーザー決裁が要る** (E-2 DoD 5 / E-4 DoD 5 と同格)。
   `src/infer/diffusion.cu:41` の `kScalingFactor` を変えると**出荷される全画像のビットが変わる**ため、
   既定 preset の変更・既定ノブの変更と同格以上の既定変更にあたる。
   エージェントは実数値と推奨を提示するところまでで、自前で確定しない (中継された承認を根拠にしない)。
4. `src/` を触る場合、`meson test` が緑。
5. **未確定のまま閉じる場合**は「どちらが正しいか未確定」と明示して残債に残す (黙って断定しない)。

**担当エージェント**: cuda-kernel-dev (ランタイム側 / export スクリプト) → gpu-benchmarker (実走・比較)。
記録は record-writer、検査は record-auditor。
**走る機械**: 研究機 (実重み 5GB + 実生成が要る)。
**発火条件**: 即時 (発注済 — 出典は項目一覧の ※ 注。**リポ内に裏付けなし**)。
**依存**: なし。E-1 が先に立っていれば物差しとして使う (必須ではない)。

---

## E-1 — 多 seed 画質評価ハーネス (`scripts/dollma_eval_image_grid.py`)

**目的**
「N=1 で優劣を書けない」(measurements-log l.528) という 2-6d の制約を、**物差しの側で**解消する。
preset / ノブ / checkpoint / LoRA を **(条件 × seed) の格子**で走らせ、既存の採点資産で集計して
CSV を吐くハーネスを 1 本作る。

**実行経路 = CLI・1 プロセス 1 枚 (前提として明記)**
HTTP 経路は格子を回せない。一次証拠:
- **HTTP に seed が無い** ★**起票時 (2026-09-19/20) のスナップショット。E-2 T1 (branch
  `feat/e2-sampling-knobs`・main 未 merge) で HTTP `seed` は実装済み**のため、**本 branch 上ではこの
  一次証拠は成り立たない** (`src/server/generator.hpp` l.53-56 に `bool has_seed` / `uint64_t seed` /
  `bool has_guidance_scale` / `float guidance_scale` / `src/server/api.cpp` l.177-187 が
  `body.contains("seed")` → `gr.has_seed=true; gr.seed=...` で受理。同様に `guidance_scale` も
  l.166-176 で受理される。行番号・型はいずれも **`c00e72b` の現物**で 2026-09-22 に確認。
  ★初版 `4afbe2f` では `std::optional<uint64_t> seed;` / `std::optional<float> guidance_scale;`
  (同 commit の l.50-51) だったが、CUDA ビルド回帰のため
  `c00e72b` で POD 表現へ変更済み — 経緯は E-2 節「現況」) — 詳細は E-2 節「現況 (2026-09-22)」と
  §3 の同趣旨バナーを参照。**E-1 の結論 (格子は CLI で回す) は下の「HTTP に preset が無い」だけで
  今も成立する**ので変更しない。以下は起票時の記述をそのまま残す:
  (`src/server/generator.hpp` l.34-46 の `GenRequest` に seed フィールド無し /
  `src/server/api.cpp` l.128-217 の受理フィールドは prompt / negative_prompt / n / steps / size /
  preset_prefix / loras / response_format のみ)。seed は env `DOLLAMA_SEED` だけ
  (`src/server/backend_image_generator.hpp` l.53-62, l.131)。
- **HTTP に preset が無い** (2-6d 残債 4「HTTP API preset フィールド / UI preset 選択」= 未着手・
  measurements-log l.545)。
したがって E-1 の格子は **CLI 生成モードで 1 プロセス 1 枚**を回す。**コスト前提**:
- 参考 (2-6d 実測): 各枚が別プロセス起動・5GB 重みロード込みの **cold 相当** で
  「base 21〜22s vs preset 37〜41s」(measurements-log l.530, l.531)。★この差の原因は**未計測**で、
  生成速度の差とは言えない (同 l.531)。なお **21〜22s は base preset の値**であり、
  illustrious-xl 中心の格子には当たらない (preset 側は 37〜41s)。
- ★**見積りの分母はこちらを使う (2026-09-20 E-1 実走の実測)**: `docs/logs/e1/grid_results.csv` の `sec` 列は
  **41.16〜123.1s/枚** (16 行)、`docs/logs/e1/grid_summary.csv` の条件別平均は
  **41.93 / 54.9 / 58.75 / 66.92 s/枚** (illustrious-xl の p1〜p4)。
  → **36 枚なら 25〜40 分強**。上の 2-6d 値 (21〜41s) を分母にすると**約半分の楽観見積り**になるので使わない。

なお秒は比較指標にしない (共通規律 6。上は所要時間の見積り用であり、`grid_summary.csv` 側も
列名が `sec_mean_characterization_only` のとおり characterization 扱い)。

**既存の同型ハーネス 3 本 (重複実装を避ける)**
「サーバを起こして生成 → 画像を採点 → CSV/JSONL」という骨格は既に 3 本ある。E-1 は**新規に書き起こす前に
この 3 本の再利用可否を検討し、採否を記録する**こと:

| 既存 | 位置づけ | E-1 に効く点 |
|---|---|---|
| `scripts/dollma_gen_scorer_corpus.py` | 旧 E-1 (ScorerNet コーパス 180 枚生成) | ★docstring l.24 に **「画像 seed は server 内 time(秒)・非再現」と自書** = HTTP 経路で seed が効かない落とし穴の**実例**。E-1 が CLI 経路を採る理由そのもの。`DOLLAMA_OV_TOKENIZERS_DLL` 未設定で段2 golden に落ちる注意 (同 l.21-22) も再利用価値あり |
| `scripts/dollma_collect_rollouts.py` | F-0a の 80 rollout 収集 | ScorerNet 採点の前処理・`axes_from_logits` の実装 |
| `scripts/dollma_rollout_bestofn.py` | F-0b の best-of-N (resume/chunk 対応) | 長時間走行の resume / チャンク分割。E-1 の格子も数十分〜数時間になるため同じ問題を持つ |

**Phase 5-1 との関係 (未検証)**
roadmap l.217 の 5-1 は「固定キャラ × 画角/パースタグの直積を自作 SDXL 生成 → 既存 QualityGate 異常タグ
hit 率で崩壊スコア化 → ヒートマップ。**採点器は新規不要**」(想定ファイル名 `scripts/dollma_probe_pose_breakage.py`)
であり、**格子生成 + 集計という骨格は同型に見える**。本台帳では汎用ハーネスを `dollma_eval_image_grid.py`
として先に作り、5-1 はその軸を画角/パースに差し替えて使う想定。
★**ただし「E-1 が 5-1 を兼ねられる」は未検証**であり、本台帳はこれを前提にしない:
- roadmap l.217 は **5-1 の内容説明**であって、**兼用可否の根拠ではない** (兼用について述べた記述はリポ内に無い)。
- roadmap l.224 の着手順は「**5-1 プラン化** → 5-2 denoise 実験 → 5-3 決裁」。E-1 を 5-1 の先行実装と
  みなすなら**プラン化の前に実装が先行する**ことになる。この可否 (プラン前の先行実装が許されるか) は**未整理**。
- 判断は **5-1 起票時に再評価**する。E-1 の DoD には 5-1 兼用は含めない。

**現況 (2026-09-20)**: 起票時 (2026-09-19) には `scripts/dollma_eval_image_grid.py` は存在しなかったが、
現在は `scripts/dollma_eval_image_grid.py` と `docs/logs/e1/` (`img` / `log` / `contact_sheets` /
**`grid_results.csv`** / **`grid_summary.csv`**) が
**未追跡ファイル (`git status` の `??`)** として実在する = E-1 の実作業が進行中。
現物 CSV と DoD の対応 (2026-09-20 に現物を開いて確認・**充足判定ではない**):
- `grid_results.csv` = ヘッダ `preset,prompt_id,seed,sec,exit,log_ok,log_reason,png,recall_matched,recall_total,recall,axis_*,argmax_axis,worst_anatomy`・データ 16 行 =
  **DoD 1 (1 行 = 1 画像の CSV) に対応する現物あり**。
- `grid_summary.csv` = `recall_mean,recall_std,worst_anatomy_mean,worst_anatomy_std,sec_mean_characterization_only,sec_std_characterization_only` =
  **DoD 4 (指標ごとの sd を CSV に出す) に対応する現物あり**。
- 一方 **DoD 5 の paired CI 列は両 CSV に無く、未充足に見える** (判定 3 軸のうち (c) が CSV から機械的に評価できない)。
★**5-1 兼用について現物と台帳が不一致**: `scripts/dollma_eval_image_grid.py` の docstring l.10-13 は
既に「**Phase 5-1 (崩壊境界 probe) の先行実装を兼ねる**」と**断定**しており、本台帳の姿勢
(下記「Phase 5-1 との関係 (未検証)」= 未検証・前提にしない) と一致しない。
**E-1 レビュー時に兼用主張の扱い (docstring を落とすか / 兼用を決裁するか) を決める**。
**未 commit・未レビュー**のため本台帳では完了扱いにしない (DoD の充足判定は成果物の検査後)。

**★採点資産の実態 (過大評価しないこと)**
E-1 は「既存の採点資産で集計する」が、その資産には**既知の穴が 2 つ**ある。**ハーネスの設計前に把握すること**:

1. **WD14 再現率は「既存資産」と呼べる状態ではない**。現状スクリプト化されておらず、
   `docs/logs/2-6d/README.md` l.70-79 で **プロンプトごとの語リストを直書きした手計算**である
   (「各プロンプトの danbooru タグ相当語 (下記) のうち …検出した tag line に含まれる語の割合」)。
   同 l.78 には **数え落としの訂正実例**が残っている (「初版は `from side` を数え落として 9 語・分母 9 で
   計算していた。10 語で再計算 → base 8/10・animagine 7/10・illustrious 7/10・noobai 8/10。順位は不変」)。
   → **E-1 で再現率を使うなら、語リストの抽出と照合をスクリプト化するのが E-1 の仕事の一部**。
   既存資産として流用できるのは `scripts/dollma_label_image.py` (WD14 タグ検出・既定閾値) までで、
   **再現率の算出ロジックは新規実装**になる。
2. **ScorerNet (anatomy 8 軸) は 8 軸中 7 軸が死んでいる**。F-0a 実走 (80 rollout・measurements-log l.67)
   の実測: worst-axis argmax が **Limbs 77 / Hands 2 / Head 1** = 実質 Limbs 単軸。
   他 7 軸は Hands/Eyes/Ears/Mouth/Digits/GlobalAnat が **max < 0.002**・Head が max 0.0118 で
   >0.05 の発火 0。reward 全体でも **std 0.0377 / best−worst 0.2031** で、PL の信号あり閾値
   (std>0.1 かつ best−worst>0.3) に**両方とも未達**。同記録はこれを
   「**ScorerNet の dynamic range が Limbs のみ生存 = anatomy=Limbs 状態 (B 側の分解能不足)**」と結論している。
   → E-1 が anatomy 軸で条件差を検出できるのは **実質 Limbs のみ**。
   **7 軸の値を条件比較の根拠に使わないこと** (床に張り付いた軸の差は分解能不足の産物)。
   軸を増やす話は本台帳のスコープ外 (将来検討リストの「reward 設計」)。

**DoD**
1. `scripts/dollma_eval_image_grid.py` が存在し、(条件 × seed) 格子を走らせて
   **1 行 = 1 画像**の CSV (条件 / seed / 採点値 / PNG パス / exe sha256 / exit code) を出力する。
2. 生ログに **フォールバック検出**が入っている: `[warn]` / stub / 「構築に失敗」の grep が 0 件であることを
   ハーネス自身が確認して CSV に記録する (2-6d で StubGenerator 混入事故が起きたため・measurements-log l.532)。
   grep は**大文字小文字を区別**する (`MISSING` を無視大小で引くと negative の "missing fingers" に当たる・
   measurements-log l.555)。
3. seed は**明示制御**され、ログに残ること。現状 seed の指定手段は env `DOLLAMA_SEED` のみ
   (`src/server/backend_image_generator.hpp` l.53-62, l.131 で `DOLLAMA_SEED` を読む。CLI に `--seed` は無い —
   `src/main.cpp` l.174-222 の引数分岐は `--http/--port/--steps/--width/--height/--prompt/--negative/
   --out/--no-matting/--no-preset-prefix/--fast/--fp8/--preset` のみ)。E-2 で `--seed` が入ったら差し替える。
   ★**2026-09-27: この条件は満たされた** — E-2 T1 で `--seed`/`--cfg` が入り (branch のみ・main 未 merge)、
   T2 のスイープは env ではなく **CLI `--seed` 経由**で 60 走行すべてログに `[gen] seed=<値>(req)` を残している
   (E-2 節「現況 (2026-09-27) — T2」)。★**ただし `--cfg` の実効値はログに出ていない** = 本 DoD 3 の
   「ログに残ること」は seed については充足・**cfg については未充足** (E-2 の残債②)。
4. ★**ノイズ床が測れていること**: 参照アーム (既定 = illustrious-xl) の**同条件・異 seed** を
   最低 4 seed 回し、指標ごとの **sd を CSV に出す**。これが共通規律 2 の (b) の分母になる。
   seed 本数は 4 以上 (Phase 4 の seed sweep はすべて 4 seed = measurements-log l.56-58, l.72, l.73)。
5. ★**判定 3 軸を CSV から機械的に評価できる**こと: 集計出力が (a) per-seed の delta と符号
   (b) 参照アームの sd (c) 各 seed の paired CI を**すべて含む** (CI の算出法は
   Phase 4 の paired bootstrap に揃えるか、揃えない場合は算出法を明記する)。
   平均 ± std だけを出して 3 軸を評価できない CSV は DoD 未達。
6. **採点器の穴を CSV に明示する**: anatomy 8 軸を出すなら、Limbs 以外の 7 軸に
   「分解能不足 (F-0a 実測)」のフラグ / 注記を付け、条件比較の主指標にしない (上記「採点資産の実態」2)。
   WD14 再現率を出すなら、語リストの出所 (どのプロンプトの何語か) を CSV / ログに残す (同 1 の数え落とし対策)。
7. 2-6d と同条件 (illustrious-xl・seed 1234) を 1 セル含め、既知の PNG sha256 と突き合わせて
   **ハーネス自身が正しい経路を叩いていること**を示す。⚠ この突合の `--fast` 前提は未確定
   (共通規律 1)。ハーネスは自分の走行の `--fast` 有無を条件として記録すること。
8. **この段階では優劣を書かない**。ハーネスが回って CSV が出たら完了。判定は E-2 以降。

**担当エージェント**: gpu-benchmarker (Sonnet 可 → Opus high レビュー)。記録は record-writer。
**走る機械**: 研究機。
**発火条件**: 即時 (発注済 — 出典は項目一覧の ※ 注。**リポ内に裏付けなし**。ただし未追跡の成果物が実在)。
**依存**: なし (E-0 と並走可。ただし E-0 の比較に使えるなら使う)。

---

## E-2 — sampling ノブ (steps / CFG / seed) の露出とスイープ

**目的** (★以下の「現状」は**起票時 = 2026-09-19 の状態**。T1 実装後の状態は本節の
「現況 (2026-09-22)」を見ること)
現状、**絵に効く基本ノブが外から触れない**:

- CFG スケールは `src/server/sdxl_backend.hpp:57` で
  `static constexpr float kGuidanceScale = 7.5f;` の**コンパイル時定数**。
- CLI に `--cfg` / `--seed` は**無い** (`src/main.cpp` の引数分岐は
  `--http/--port/--steps/--width/--height/--prompt/--negative/--out/--no-matting/--no-preset-prefix/--fast/--fp8/--preset`)。
- HTTP も `guidance_scale` を受け付けない (`src/server/api.cpp` が読むのは
  `prompt` / `negative_prompt` / `n` / `steps` / `size` / `preset_prefix` / `loras` / `response_format`)。
- seed は env `DOLLAMA_SEED` のみ (`src/server/backend_image_generator.hpp` l.53-62, l.131)。
  **HTTP も seed を受け付けない** — `GenRequest` (`src/server/generator.hpp` l.34-46) に seed フィールドが無く、
  `api.cpp` l.128-217 の受理フィールドにも無い。

そこで ① CLI `--cfg` / `--seed` と HTTP `guidance_scale` / **`seed`** を追加し、② E-1 の格子で
steps / CFG / seed をスイープして**既定値の妥当性**を多 seed で確かめる。

★**HTTP `seed` を E-2 のスコープに含める理由 (重要。★この段落も起票時 = 2026-09-19 の状態で書かれている**
— 「seed は HTTP から指定できない」は T1 実装後は成り立たない。本節「現況 (2026-09-22)」と突き合わせること):
これは E-5 の前提条件である。
**LoRA は HTTP 専用で CLI から到達できない** (`grep -rn -i lora src/server/cli_generate.hpp src/main.cpp`
= **0 件**。`loras` は `api.cpp` l.179 のみ)。一方 seed は HTTP から指定できない。
したがって **HTTP `seed` が無い限り「LoRA strength 軸 × 多 seed」を 1 プロセス内で清潔に回せない** (E-5)。
CLI `--seed` だけ足しても E-5 は解けない。
これは新規の思い付きではなく、**既存記録に残っている未解決レバー**と同一物:
`docs/f0b-rejection-sft-plan.md` l.116 (c)「SDXL seed 制御 (HTTP に seed 引数・SAC 制約下の実現方法) で
reward 比較のノイズ除去」/ `docs/roadmap.md` l.321「**次レバー** (F-0b 後): reward 設計 / 日本語条件付け改修 /
**seed 制御**」。E-2 はこのレバーを画質評価側から実装する形になるため、**Phase 4 F 側と重複起票しないこと**
(将来検討リストにも同旨を記載)。

**DoD**
1. CLI `--cfg <float>` / `--seed <uint64>` と HTTP `"guidance_scale"` / **`"seed"`** が動く。
   HTTP は型不正を **400** で弾く (2-6e の `preset_prefix` 非 bool → 400 と同じ流儀・measurements-log l.554)。
   seed の優先順位は **`req.seed` (CLI / HTTP 共通の `GenRequest` 経由) > env `DOLLAMA_SEED` > 時刻ベース**
   の 3 段。ログに実効値を出す。
   ★「HTTP body > CLI」という段は**存在しない** — CLI も HTTP と同じ `GenRequest` を通るため
   (`src/main.cpp` l.255 で CLI が `GenRequest` を組み立てる / `src/server/cli_generate.hpp` に
   `GenRequest`・`generate(` の出現は 0 件)。両者は排他の入口であって優先順位関係にない。
   HTTP `seed` は `GenRequest` (`src/server/generator.hpp`) へのフィールド追加を伴う。
   ⚠ **`GenRequest` への seed 追加は末尾に足すか designated init 化すること** —
   `src/main.cpp` l.255 は `GenRequest req{prompt, negative, 1, steps, width, height};` という
   **位置指定の集成初期化**であり、`height` より前に `seed` を挿すと黙って意味がずれる
   (既に `matting` / `preset_prefix` は「並びがずれる」ため代入で設定されている・同 l.256-258)。
2. **既定経路の無改変ゲート**: 4 ノブ (steps/CFG/seed/HTTP seed) すべて未指定のとき PNG sha256 が現行と一致
   (共通規律 1)。具体的には 2-6e の p1/p2/p3 sha256 と 3/3 一致。
   ⚠ 一致で言えるのは「2-6d/2-6e 走行と拡散条件が同一」までで、**`--fast` 有無は未確定**のまま引き継がれる
   (共通規律 1 の注記)。走行条件として `--fast` 有無を明示記録する。
3. `docs/http-api-spec.md` の拡張フィールド表を追随更新する。
   ★**現状の仕様表には既に 2 件の乖離がある** — 実装が受理するのに仕様表に**1 行も無い**:
   `preset_prefix` (2-6e・`api.cpp` l.166) と `loras` (L-2・`api.cpp` l.179)。
   仕様表 `docs/http-api-spec.md` l.30-35 (ヘッダ l.28) に載っているのは
   `prompt` / `n` / `size` / `negative_prompt` (拡張と明記・l.33) / `steps` (拡張と明記・l.34) /
   `response_format` の 6 件のみ。
   → E-2 では `guidance_scale` / `seed` を足すだけでなく、**この 2 件の既存乖離も同時に解消する**
   (乖離を残したまま新フィールドだけ足さない)。
4. `meson test` 緑 + 新規ノブの test (境界値 / 不正入力 / 未指定時の既定一致)。
5. スイープ: E-1 の CSV に steps / CFG / seed 軸の結果があり、**多 seed** での比較になっていること。
   判定は共通規律 2 の **3 軸 (符号一致 / 分散帯超え / paired CI)**。**ノイズ床 (参照アームの同条件・異 seed sd)**
   を同じ走行で測り、それを分母にする。
   結論は「既定を変える / 変えない」のいずれかを実数値付きで書く。頭打ちなら
   「既存ノブでは頭打ち」と明示する (これが E-3 の発火条件)。
   ★**既定値を変える場合はユーザー決裁が要る** (下記 E-4 DoD 5 と同じ規律。既定の変更は出荷物の変更であり、
   エージェントの判断で確定しない)。
6. 秒が動いても CLAUDE.md 計測表に行を足さない (共通規律 6)。

**現況 (2026-09-22) — T1 (ノブ配線 + 仕様表更新) 実装完了・未レビュー**
実装場所は worktree `E:\Develop\Projects\dollama-wt-e2t1` / branch `feat/e2-sampling-knobs` (main 未 merge)。
★**以下の DoD 欄は commit `4afbe2f` 時点で書かれた**。その後 `c00e72b` で CUDA ビルド回帰を修正して
`GenRequest` の表現が変わっているため、**型・フィールド名は下記「CUDA ビルド回帰 (`c00e72b`)」を正とする**。

**CUDA ビルド回帰 (`c00e72b`・2026-09-22) — 記録監査をすり抜けた事故**

- **何が壊れていたか**: T1 初版 `4afbe2f` は `GenRequest` (`src/server/generator.hpp`) に
  `std::optional<uint64_t> seed` / `std::optional<float> guidance_scale` を追加した。この構造体は
  `src/server/pipeline_generator_factory.cu` からも間接 include される
  (`pipeline_generator_factory.hpp` / `pipeline_generator.hpp` がどちらも `server/generator.hpp` を include)。
  本プロジェクトの `.cu` は `src/meson.build` l.46 で **`-Xcompiler /std:c++14` を強制**している
  (同 l.37 の注記: CUDA 13.3 + MSVC で c++17/20 ヘッダ組合せの 0xC0000409 を回避するため) ので、
  **`std::optional` がコンパイルできず `with_cuda=true` のビルドが通らなかった**。
  ★**失敗ビルドのエラー出力は保全していない** (`c00e72b` 後の build ディレクトリで上書き済み)。
  上記の機構は現物から導出したもの = `src/meson.build` l.46 (`-Xcompiler '/std:c++14'`・注記は同 l.37) +
  include 連鎖 (`pipeline_generator_factory.cu` l.15 → `pipeline_generator_factory.hpp` l.23 /
  同 .cu l.21 → `pipeline_generator.hpp` l.41 → どちらも `server/generator.hpp`) + 初版 `4afbe2f` の
  `generator.hpp` が `#include <optional>` していたこと。**「ビルドが失敗した」事象そのものの一次証拠
  (コンパイラ出力) は無い**。
- **なぜ見逃したか**: **T1 に対する記録監査 3 ラウンドはすべて `with_cuda=false` 構成で行われた**
  (★この「3 ラウンドとも」は **PL からの申し送り**であり、本記録執筆時に一次証拠で数えたものではない。
  下記 DoD 4 走行 A の構成が `with_cuda=false` だったことについても**一次証拠は残っていない**
  — 根拠は当時の記録 (二次) のみ。走行 A のログは同一 build ディレクトリの走行 B で上書きされている)。
  この構成では `.cu` が 1 つもコンパイルされないため、回帰が発生しうる TU が
  **ビルドグラフに存在しない**。「`meson test` 32/32 緑」はこの構成の緑であり、
  CUDA ビルドの成否については何も言っていなかった。
- **どこで発覚したか**: **研究機での T2 実走準備で `with_cuda=true` としてビルドした時点**
  (★発覚の場面も PL からの申し送り。本記録側の一次証拠は、`c00e72b` が実際に
  `with_cuda=true`/`with_openvino=true` 構成の build ディレクトリを残していること = DoD 4 走行 B)。
  レビューや test ではなく、実走のためのビルドが最初の検出器になった。
- **どう直したか**: `c00e72b` で `std::optional<T>` を **POD のフラグ + 値フィールド**
  (`bool has_seed` / `uint64_t seed` ・ `bool has_guidance_scale` / `float guidance_scale`) に置換
  (`src/server/generator.hpp` l.53-56)。
  ★**「POD」なのは E-2 ノブ部の 4 フィールドだけ**で、`GenRequest` 自体は `std::string prompt` /
  `std::vector<LoraSpec> loras` を含むため POD / trivially-copyable ではない
  (現物 l.34-57 = 全 13 フィールド)。
  呼び出し側は `api.cpp` / `backend_image_generator.hpp` / `main.cpp` /
  `test_diffusion_backend.cpp` / `test_http.cpp` を追従。
  **意味論 (未指定→env/時刻・backend 既定へフォールバック) は不変**であることは
  `git diff 4afbe2f c00e72b` の全差分で確認済み (分岐条件が `req.seed` → `req.has_seed`、
  値取得が `*req.seed` → `req.seed` に替わっただけ)。
  条件コンパイルでフィールドを隠す案は **ODR 違反の恐れがある**ため採らなかった
  (一次証拠 = `src/server/generator.hpp` の当該コメント「ODR 事故防止のため POD 表現に固定する
  (フィールドを条件コンパイルで隠さない)」。同種の実害の前例は CLAUDE.md 計測表
  「ランタイム LoRA (L-2 完了)」行に記録の `class DeviceWeights` ODR 事故)。
- ★**教訓 (今後の同種事故を防ぐための本記録の主目的)**:
  **CUDA / OpenVINO を有効にしたビルドを一度も通していない状態で「`meson test` 全緑」を完了扱いにしない。**
  構成を落としたビルドの緑は「その構成で壊れていない」しか意味せず、**落とした部分の無傷を保証しない**。
  記録・レビュー側も、test 結果を引用するときは **どの build option 構成の走行か**を必ず併記する
  (本節 DoD 4 はその書き方に従う)。

DoD ごとの状態:

- **DoD 1 = 実装済み**。`GenRequest` (`src/server/generator.hpp`) に sampling ノブ 2 件を**末尾に**追加。
  ★**現物 (`c00e72b`) の E-2 ノブ部は POD のフラグ + 値フィールド** (構造体全体が POD という意味ではない
  = 上記「どう直したか」の ★) — l.53-56 に `bool has_seed = false` / `uint64_t seed = 0` /
  `bool has_guidance_scale = false` / `float guidance_scale = 0.0f`
  (初版 `4afbe2f` は `std::optional<uint64_t> seed` / `std::optional<float> guidance_scale` だった。
  変更理由は上記「CUDA ビルド回帰」)。
  **DoD 1 の ⚠ (集成初期化の非破壊) は `c00e72b` でも維持されている** — 一次証拠:
  ① 追加 4 フィールドは構造体の**末尾**にあり、`prompt/negative_prompt/n/steps/width/height` の並びは無改変
  (`git diff 4afbe2f c00e72b -- src/server/generator.hpp` = **2 hunk**: 末尾ブロックの差し替えと、
  先頭の `-#include <optional>` 削除。既存フィールドを触る hunk は無い)、
  ② `src/main.cpp` の `GenRequest req{prompt, negative, 1, steps, width, height};` という位置指定の
  集成初期化は**一切変更されていない** (同 diff の `src/main.cpp` 側は `--seed`/`--cfg` の代入部のみ)、
  ③ ノブは集成初期化に足さず `if (cli_seed) { req.has_seed = true; req.seed = *cli_seed; }` /
  `if (cli_cfg) { req.has_guidance_scale = true; req.guidance_scale = *cli_cfg; }` の**代入**で設定
  (`matting` / `preset_prefix` と同じ流儀)。
  なお `src/main.cpp` のローカル変数側は `std::optional<uint64_t> cli_seed` / `std::optional<float> cli_cfg`
  のままである (`main.cpp` は `.cpp` TU なので c++14 制約を受けない)。POD 化したのは
  `GenRequest` の E-2 ノブ部だけ。
  CLI `--seed <uint64>` / `--cfg <float>` を追加 (パース失敗時は**未指定扱い**でエラーにしない)。
  ⚠ **この 2 引数は実行未検証** — argv 解析を通す test が 0 件のため、言えるのはビルド成立までである
  (詳細と一次証拠は下記 DoD 4 の ★★)。また **`--http` 起動時は両引数とも捨てられる**
  (`src/main.cpp` は `http_mode` なら `GenRequest` を組む前に `start_server` へ return する)。
  HTTP は `api.cpp` で `guidance_scale` 非数値 → 400 / `seed` 非負整数以外 (負数・小数・文字列) → 400。
  seed 解決は `backend_image_generator.hpp` で **`req.seed` > env `DOLLAMA_SEED` > 時刻**の 3 段になり、
  実効値を `[gen] seed=<値>(req|env|time)` としてログに出す。CFG は未指定なら従来どおり `cfg=0.0f` を
  backend へ渡し既定 (SDXL `kGuidanceScale=7.5f`) へ委譲。
- **DoD 2 = 充足 (T2・2026-09-27。ただし突合相手を 2-6e から main `766082a` へ差し替えた** —
  理由と一次証拠は下記「現況 (2026-09-27) — T2」の「DoD 2」項)。
  ★以下の T1 時点の記述は経緯として残す。
- (T1 時点の状態) **DoD 2 = 未充足 (T2 = 研究機の担当として残る)**。T1 のユニット test で示せたのは
  ① `BackendImageGenerator` が未指定時に backend へ渡す**実効引数**が従来と同じ (`cfg==0.0f`・seed は
  env→時刻の非ゼロ値)、② HTTP で未指定なら `GenRequest` が未指定のまま
  (`c00e72b` 以降は `has_guidance_scale`/`has_seed` がともに false。`src/tests/test_http.cpp` l.375 の判定。
  ★同 l.380 のログ文字列は `[gen-knobs-unset] 未指定時 nullopt OK` と **stale** のまま = 残債・src 未修正)、
  の 2 点まで。
  いずれも fake/stub backend 上の検証であり、**2-6e の p1/p2/p3 PNG sha256 3/3 一致は未実施**。
  ★実重みでの sha256 突合と、その走行条件 (`--fast` 有無の明示記録) は T2 で行う。
- **DoD 3 = 実装済み**。`docs/http-api-spec.md` の拡張フィールド表に `guidance_scale` / `seed` を追加し、
  併せて既存乖離 2 件 (`preset_prefix` (2-6e) / `loras` (L-2)) も同時に解消した。
  同 doc の CLI 引数抜粋にも `--seed` / `--cfg` を追記。
- **DoD 4 = 部分充足**。test 追加は `src/tests/test_diffusion_backend.cpp` (`RecordingFakeBackend` で
  未指定/env のみ/req 明示の 3 ケース) と `src/tests/test_http.cpp` (`RecordingGenerator` で伝播・
  **未指定 = `has_seed`/`has_guidance_scale` がともに false** (`test_generations_sampling_knobs_unset`)・
  非数値 cfg 400・負数 seed 400)。
  **走行 A (初版 `4afbe2f`・記録として残す)**: `meson test` **32/32 緑**
  (当時の出典は `build/meson-logs/testlog.txt`・2026-09-22T23:20:16)。
  ★**走行 A のログは同一 build ディレクトリの走行 B で上書きされ現存しない** (現物 `testlog.txt` の
  1 行目は `2026-09-22T23:56:53` のみ = 走行 B)。したがって **32/32・`with_cuda=false` は当時の記録 (二次)
  であり、現時点で再検証できない**。
  当時の記録によればこの build 構成は **`with_cuda=false` / `with_openvino=false` / `with_http=true`** で、
  CUDA/OV 依存 test は登録されない = **実重み経路はこの走行で 1 件も走っていない**
  (「全緑」を実重み検証の代わりに読まないこと)。**この構成では CUDA ビルド回帰も検出できなかった**
  (上記「CUDA ビルド回帰」)。
  **走行 B (`c00e72b`・2026-09-22T23:56:53 開始)**: `with_cuda=true` / `with_openvino=true` /
  `with_http=true` (`build/meson-info/intro-buildoptions.json` で確認) で**ビルド成立**し、
  `meson test` **55/55 Ok / Fail 0** (`build/meson-logs/testlog.txt` の `Ok: 55 / Fail: 0`)。
  → **解消した限定**: ① CUDA 有効ビルドが通る = `.cu` TU 群も新 `GenRequest` を含めてコンパイル/リンク可、
  ② 登録 test が 32→55 に増え、**GPU カーネル test が実走** (例: `test_gemm` / `test_conv2d` /
  `test_attention` / `test_groupnorm` / `test_device_arena` がいずれも ALL PASSED)。
  ★`test_groupnorm` の `bench_groupnorm_mb` はここでは **`ALL PASSED` = ゲート通過の事実のみ**を採る。
  同 bench の GB/s 値は**閾値 (300 GB/s) を超えたかの合否判定用であり、性能比較の数値ではない**
  (同一行の実測が median 0.0624ms に対し min 0.04448 / max 7.37501ms = **測定ばらつきが大きい**)。
  CLAUDE.md 計測表「GroupNorm multi-block (G-4k S1a 完了)」の値と引き算して**退行/改善を論じないこと**。
  → ★**残る限定 (「実重み経路が緑」と読まないこと)**: この worktree には gitignore 対象の
  golden / 重みが無く (拡散系 = `src/tests/data/unet_io.safetensors` / `txt2img_io.safetensors` 等、
  LM 系 = `data/bitnet/bitnet_dense*_fp32.safetensors` — `data/bitnet/vocab.json` は現存)、
  **11 test が重み/golden 不在で `[SKIP]` を出している**
  (text_conditioner / scheduler / bitnet_infer / bitnet_int8 / vae_decode / unet /
  unet_fast / diffusion / diffusion_batch2 / pipeline_generator / bitnet_gpu)。
  ★**preset の大型重みは現存する** — `models/presets/illustrious-xl/unet_weights.safetensors` 5,135,149,736 B /
  `vae_weights.safetensors` 98,995,758 B が実体としてあり、testlog も
  `unet = ../models/presets/illustrious-xl/unet_weights.safetensors [OK]` / `vae = ... [OK]` を出している。
  **「worktree に大型テストデータが 1 つも無い」わけではない**。
  ★**SKIP のうち 2 件は重み/golden 不在とは別原因**なので上の 11 件に数えない:
  ① `test_allocator` の `test_cuda_alloc_no_cuda_throws` は `SKIP (HAVE_CUDA あり)` =
  **CUDA が有効だから**スキップする (CUDA 無効時に例外を投げる経路の test なので、この構成では恒久的に SKIP)、
  ② `test_txt2img` は `[SKIP] アセットが見つかりません (tokenizers.dll)` = env `DOLLAMA_OV_TOKENIZERS_DLL`
  未設定が原因。
  さらに `test_cli_generate` は stderr に
  `[factory] 重み/golden が不足しています — StubGenerator にフォールバック`
  (`embed = src/tests/data/unet_io.safetensors [MISSING]`) を出しており、**実重み拡散は走っていない**。
  したがって走行 B で言えるのは「CUDA 有効ビルドが通り、重みを要さない GPU test が緑」までで、
  **実重みでの PNG 生成 e2e と 2-6e sha256 一致は依然未実施 = DoD 2 は引き続き未充足**。
  ★★**`src/main.cpp` の引数解析を通す test は 0 件** (`grep -rn '"--seed"\|"--cfg"\|"--prompt"' src/tests/`
  = **0 件**・2026-09-22 実行。**`c00e72b` の現物で 2026-09-27 に再実行しても 0 件**)。
  上記 2 本はどちらも `GenRequest` を test 側で直接組み立てており、
  argv 解析を 1 行も通らない。したがって **`--seed` / `--cfg` については「コンパイル・リンクが通った」
  ことまでしか言えず、引数が実際に拾われるかは未検証**。→ T2 で `dollama --prompt ... --seed ...` の
  smoke を 1 本走らせて実効値 (`[gen] seed=<値>(req)`) を確認する。
- **DoD 5 (スイープ) = 充足 (T2・2026-09-27)。結論 = 既存ノブ (この走行の範囲: cfg 5.0-10.0 / steps 12-28)
  では頭打ち・既定値は変更しない** (したがって DoD 5 ★のユーザー決裁は発生しない)。
  実数値・3 軸判定の内訳・限定は下記「現況 (2026-09-27) — T2」の「DoD 5」項。
- **DoD 6 = 遵守**。CLAUDE.md 計測表に行は足していない。
- **既定値は 1 つも変えていない** (`kGuidanceScale=7.5f`・seed の env→時刻フォールバックとも無改変)
  ため、DoD 5 ★のユーザー決裁は T1 では発生しない。

**現況 (2026-09-27) — T2 (実重み DoD 2 突合 + DoD 5 スイープ) 実走完了**
実走は研究機 (SAC OFF)・worktree `E:\Develop\Projects\dollama-wt-e2t1` / branch `feat/e2-sampling-knobs`。
生成物とログは `docs/logs/e2-t2/` 配下。★**本節の数値は記録執筆時に一次証拠 (PNG の sha256 再計算 /
`grid_results.csv` / `analysis_*.log`) を開いて検算した**。

★**成果物の保全範囲 (要決裁の残件)**: `docs/logs/e2-t2/` の実体は **81 PNG / 約 268MB**。
このうち **commit したのは DoD 2 の突合 PNG 6 枚・全ログ (60 走行分の `log/*.log`)・CSV 3 組・
contact sheets・`analysis_*.log`・`dod2_notes.txt`** で、**スイープ本体の `*/img/*.png` 60 枚
(約 240MB) は commit していない** (研究機ローカルにのみ存在)。
E-1 は img を全数 commit している (`git ls-files docs/logs/e1/` = PNG 20 件) ため**先例とは異なる扱い**であり、
240MB をリポジトリに恒久追加するかは**記録係の裁量を超える** = **ユーザー/PL の決裁待ち**。
影響: 本節が引用する cfg アーム 3 枚の sha256 (`abeede2f…`/`4703af70…`/`7fa14633…`) は、
commit 済みの範囲からは**再計算できない**。再現が必要なら研究機ローカルか再走行が要る。

**準備段階の前提破損 (先に読むこと)**
- worktree に gitignore 対象の `src/tests/data/unet_io.safetensors` が無く、`dollama.exe --preset
  illustrious-xl` の実走が段 1 (`BackendImageGenerator` / sdxl backend) ではなく **段 3 (`StubGenerator`)
  へ静かにフォールバックしていた** (`build_image_generator` の ov_ready 判定が embeds パスの存在も
  要求する)。既知症状 = `[[feedback_worktree_missing_test_data]]` / T1 の DoD 4 ★も同じ不在を記録している。
- 対処: main checkout の同ファイル (498,395,472 B) からハードリンクを張って復旧。以後のログに
  `dollama HTTP server (sdxl backend [preset=illustrious-xl] — NPU)` が出ることで段 1 到達を確認
  (一次証拠: `docs/logs/e2-t2/p1_dod2.log`)。
- ★**教訓 (T1 の「構成を落とした緑」と同型)**: 実走の**段**を確認せずに出力を採らない。
  段 3 フォールバックは exit=0 / PNG 生成ありで**静かに**起こる。

**DoD 2 (既定経路の無改変ゲート) = 充足。ただし突合相手を差し替えた**
- ★**DoD 2 原文の「2-6e の p1/p2/p3 sha256 と 3/3 一致」は、この時点では原理的に成立しない**:
  2-6e の PNG は **E-0 (VAE `scaling_factor` を 0.18215 へ・commit `272e854`) より前**の exe で生成された
  もので、E-0 は既に main へ入っており (`git merge-base --is-ancestor 272e854 766082a` = 真)、
  E-2 branch は E-0 の**後**に分岐している (`git merge-base main feat/e2-sampling-knobs` = **`766082a`**)。
  つまり 2-6e との差は **E-0 由来の正当な差分**であり、E-2 の回帰とは無関係。
  実際に E-2 worktree 出力は 2-6e の値と不一致だった。
  → **突合相手を merge-base `766082a` (E-0 後・E-2 分岐前の main) に差し替えて判定した**。
  ★この差し替えは DoD 原文の緩和ではなく、**原文が意図した「既定経路に回帰なし」を測れる唯一の基準への
  訂正**である (2-6e 基準では E-0 の変更が回帰として誤検出される)。
- **A/B 比較結果 = p1/p2/p3 3/3 sha256 完全一致**。両アームとも
  env `DOLLAMA_SEED=1234` / `--preset illustrious-xl` / `--steps 20` / **`--seed`・`--cfg` 未指定**:

  | prompt | sha256 (E-2 worktree / main `766082a` 共通) |
  |---|---|
  | p1 | `88be13a7c1544f406d6ca6193bc7a08fc63e8c919d4390b0acb164e550de98fd` |
  | p2 | `f99a511ba049d428cdab0f2f32dd007de5530540bf501d4987546e18311122e1` |
  | p3 | `d8d3e2b137cc9827b04be73caa25ee2a67c975163e7543cd15fe49f675a1bb1b` |

  一次証拠 = `docs/logs/e2-t2/p{1,2,3}_dod2.png` (E-2 側) と
  `docs/logs/e2-t2/baseline-main-766082a/p{1,2,3}_base.png` (main 側) の sha256 を記録執筆時に再計算し
  6 値すべてを確認した。★**`dod2_notes.txt` 中の「`docs/logs/e2-t2-baseline/`」というパス表記は誤り**で、
  現物は `docs/logs/e2-t2/baseline-main-766082a/` (訂正済・ファイル自体は同一)。
- **走行条件の明示 (DoD 2 の ⚠ が要求している項目)**: **`--fast` / `--fp8` はいずれも未指定 = 既定 OFF**。
  ★ただし**この点をログから確認することはできない** — `src/main.cpp` は `--fast`/`--fp8` の
  実効値を出力しない (l.221-223 でフラグを立てるだけ)。根拠は起動コマンド側 (スイープは
  `scripts/dollma_eval_image_grid.py` の `cmd` に `--fast` が 0 件)。**DoD 2 の PNG は手動実走のため
  argv が保全されておらず、`--fast` 未指定は二次証拠 (実走者の記録) のみ**。
  → 残債: 実効 fast 構成をログに出す (T1 が seed に対して `[gen] seed=<値>(req|env|time)` を足したのと同じ流儀)。
- **`--seed` CLI 経路の実地検証 = 充足 (T1 DoD 4 ★★ の宿題)**。T1 では argv 解析を通す test が 0 件で
  「ビルドが通る」までしか言えていなかった。T2 のスイープ **60 走行すべてのログに `[gen] seed=<値>(req)`**
  が出ている (`grep -l "seed=.*(req)" docs/logs/e2-t2/*/log/*.log` = **60/60**) =
  **CLI `--seed` が env `DOLLAMA_SEED` より優先される 3 段解決が実走で確認された**。
  ★**`--cfg` の実効値はログに出ない** (`[gen]` 行に cfg の出力が無い)。`--cfg` が効いていることの
  一次証拠は**出力 PNG が cfg アームごとに異なること**で取った (同一 prompt/seed で
  cfg0 `abeede2f…` / cfg5 `4703af70…` / cfg10 `7fa14633…` = 3 値すべて相違)。
  → 残債: cfg の実効値もログに出す。
- ★**`--seed 9999 --cfg 7.5` の単発 smoke については、ログが保全されていない**
  (`docs/logs/e2-t2/` 配下で "9999" に一致するのは `dod2_notes.txt` = **二次証拠のみ**)。
  この smoke 自体を一次証拠として引かないこと。上記 60/60 の方が強い証拠なので判定には影響しない。

**ハーネス拡張 (T2 で加えた変更・`src/` は無改変)**
- `scripts/dollma_eval_image_grid.py`: **cfg 軸を追加** (`build_grid` が `(preset, pid, seed, cfg)` の直積・
  出力名に `__cfg<値>`・CSV に `cfg` 列)。`run_one` を env 依存から **CLI `--seed` / `--cfg` の明示指定**に変更
  (= T1 で足した引数経路を実際に叩く形にした。これが上記 60/60 の `(req)` を生んでいる)。
  ★`cfg=0.0` は「未指定と等価」の意味で参照アームに使っている — 一次証拠 =
  `src/server/backend_image_generator.hpp` l.180 `const float cfg = req.has_guidance_scale ?
  req.guidance_scale : 0.0f;` で未指定時も `0.0f` が渡り、backend 側が既定 (`kGuidanceScale=7.5f`) へ委譲する。
- `scripts/dollma_e2_t2_analyze.py` (**新規**): 共通規律 2 の 3 軸判定ツール。
  ① 符号一致 ② 参照アーム分散帯超え ③ paired 95%CI が 0 を跨がない を計算し、
  **3 軸すべて真のときだけ `有効レバー`**、それ以外は `頭打ち/ノイズ支配` と出す (同 script の `verdict` 行)。

**DoD 5 (スイープ) = 充足。結論 = 既存ノブでは頭打ち・既定変更なし**
- **走行スコープ**: preset `illustrious-xl` 単独 / prompts p1,p2,p3 / seed 1000-1003 (4 本) /
  cfg = [**0.0 (=未指定 → 既定 7.5 相当)**, 5.0, 10.0] / steps = [12, 20, 28]。
  **総走行数 = 60 枚** (cfg 軸 36 = 3 prompt × 4 seed × 3 cfg、steps=12 が 12、steps=28 が 12。
  **steps=20 アームは cfg=0.0 の 12 枚を参照として再利用**しているため 3×4×3+12+12 = 60)。
  ★**gpu-benchmarker 報告の「全 72 枚」は誤り** — `grid_results.csv` の行数は 36/12/12 = **60**、
  PNG 実数も 36/12/12 = **60** (記録執筆時に `wc -l` と `ls | wc -l` で計数)。
- **健全性 = 全 60 行で `exit=0` / `log_ok=True` / `log_reason` 空 / PNG 実在** (CSV 3 本を
  記録執筆時にパースして計数)。**段 3 フォールバック検出 0 件**。
- **3 軸判定 (主指標 = recall。参照アームは cfg=0.0 / steps=20)**:

  | 条件 | delta_mean | 95%CI | ①符号一致 | ②分散帯超え | ③CI が 0 を除外 | 判定 |
  |---|---|---|---|---|---|---|
  | cfg=5.0 vs 既定 | **0.0000** | [0.0000, 0.0000] | True (縮退) | False | False | 頭打ち/ノイズ支配 |
  | cfg=10.0 vs 既定 | **-0.0167** | [-0.0387, 0.0054] | True (縮退) | False | False | 頭打ち/ノイズ支配 |
  | steps=12 vs 20 | **-0.0093** | [-0.0479, 0.0294] | False | False | False | 頭打ち/ノイズ支配 |
  | steps=28 vs 20 | **0.0000** | [0.0000, 0.0000] | True (縮退) | False | False | 頭打ち/ノイズ支配 |

  各 n=12。**参照アームの分散帯 (②の分母) = recall で 0.0570** (prompt ごとの seed 間 std を平均した値・
  `analysis_cfg.log` / `analysis_steps.log` に同一値)。
  → **4 条件すべてで「有効レバー」に到達したものは無い (0/4)**。
  ★**「3 軸とも不成立」と書かないこと** — 上表のとおり**軸①は 3 条件で True** になっている。
  ただしそれは delta が全ペアで 0 の**縮退** (`sign_consistent` は非ゼロ符号が無ければ True を返す実装・
  `majority_sign=0`) であって、効果の証拠ではない。**総合判定を分ける決定打は②と③で、両方 False**。
- **副指標 `worst_anatomy` は参考値扱い**。理由 = F-0a 実測で ScorerNet の分解能が Limbs 軸のみ生存と
  判明しており (CLAUDE.md 計測表「品質 FB ループ F-0a 信号ゲート」行)、絶対値が 0.0000-0.0041 と
  極小。★**この軸では軸②③が個別に True になる条件がある** (例: cfg=10.0 の worst_anatomy は
  ②分散帯超え=True、steps=28 は ③CI が 0 を除外=True) が、**分散帯自体が 0.000487 と極小なため
  「超えた」ことに実質的な意味が無い**。総合判定はいずれも `頭打ち/ノイズ支配`。**この軸を根拠に
  既定変更を論じないこと。**
- **結論 = 既定値は 1 つも変更しない** (`kGuidanceScale=7.5f` / steps=20 / seed フォールバックとも無改変)。
  → DoD 5 ★のユーザー決裁は**発生しない**。
- ★**この結論の限定 (超えて読まないこと)**: 言えるのは
  「**この走行の範囲 (illustrious-xl / cfg 5.0-10.0 / steps 12-28 / 3 prompt × 4 seed / recall 主指標)**
  で 3 軸判定を通るレバーが無かった」まで。
  **「CFG と steps は画質に効かない」という一般命題は示していない** — cfg アームは PNG レベルでは
  明確に別画像 (上記 sha256 3 値相違) を出しており、**動いていないのはノブではなく指標**である。
  他 preset・より広い cfg 範囲・他プロンプト群は未測。

**E-2 全体の状態 = 完了 (DoD 1-6 すべて充足)。ただし main 未 merge・未 push。**
- DoD 1 実装済 / **DoD 2 充足 (T2・突合相手 `766082a`)** / DoD 3 実装済 / DoD 4 走行 B で 55/55 緑
  (★限定は DoD 4 の記述どおり — 実重み経路の一部は worktree の重み不在で SKIP) /
  **DoD 5 充足 (結論 = 頭打ち・既定変更なし)** / DoD 6 遵守。
- ★**branch `feat/e2-sampling-knobs` は main へ merge されておらず push もされていない** (2026-09-27 時点)。
  したがって**出荷物 (main) には `--seed`/`--cfg`/HTTP `guidance_scale`/`seed` はまだ入っていない**。
  E-5 の発火条件「HTTP `seed` が入った後」を main 基準で読む場合は未成立 (branch 基準では成立)。
- **残債 (T2 で新たに立った分・いずれも `src/` 変更を伴うため本記録では直さない)**:
  ① 実効 fast 構成 (`--fast`/`--fp8`) をログに出す ② `--cfg` の実効値をログに出す
  ③ 段 3 フォールバックを**静かに**起こさない (実走の段を機械判定可能にする)
  ④ T1 から継続: `src/tests/test_http.cpp` l.380 のログ文字列 `nullopt` が stale。

**担当エージェント**: cpp-implementer (ノブ実装・Sonnet 可 → Opus high レビュー) → gpu-benchmarker (スイープ)。
**走る機械**: ノブ実装とユニット test は**開発機可**。スイープ実走は**研究機** (新規 exe = SAC OFF 依頼)。
**発火条件**: **E-1 完了後**。
**依存**: E-1 (物差しが無いとスイープ結果を判定できない)。

---

## E-3 — scheduler 拡張 (Karras sigmas / DPM++ 2M / v-prediction 分岐)

**目的**
現状の sampler は **1 種・固定設定のみ**。`src/infer/scheduler.hpp` の設定コメントは
`timestep_spacing = "leading", steps_offset = 1` (l.13) / `use_karras_sigmas = false` (l.14) /
`prediction_type = "epsilon"` (l.15) で、`src/` 内に Karras / DPM++ の実装は無い
(`karras` の出現は上記コメント 1 箇所のみ)。
アニメ系 checkpoint は Karras / DPM++ 2M 系の sampler 前提で調整されていることが多い。
また v-prediction 系 checkpoint については、`prediction_type` 分岐が無いと**サンプリング式が合わず
出力が壊れると見込まれる** — ★**「そもそも読めない (ロードできない)」ではない**。
重みロード自体は epsilon 用の経路で通る見込みで、壊れるのは denoise の更新式のはず。
ただし **本プロジェクトで v-pred checkpoint を読ませた実測は無く、上記はいずれも未検証の見込み**。
(「アニメ系が Karras/DPM++ 前提」も一般論であって本プロジェクトでは**未検証**。E-2 の頭打ち実測が出るまで
この理由だけで着手しない。)

**DoD**
1. Karras sigmas / DPM++ 2M / v-prediction の各分岐が、**diffusers 由来の golden** と突合して一致する
   (許容誤差は既存 scheduler と同流儀。既存は `test_scheduler` で golden `unet_io.safetensors` の
   `sched_sigmas` 等と突合済 — `src/infer/scheduler.hpp` l.17-21)。
2. **既定は現行のまま** (Euler / leading / karras=false / epsilon)。未指定時の PNG sha256 が現行と一致
   (共通規律 1)。新 sampler は明示指定でのみ有効。
3. `meson test` 緑。golden 生成スクリプトが `scripts/` に commit されている。
4. 画評価: E-1 の格子で sampler 軸を多 seed 比較し、採用/不採用を実数値付きで書く。
   判定は共通規律 2 の **3 軸 (符号一致 / 分散帯超え / paired CI)**。**ノイズ床 (参照アームの同条件・異 seed sd)**
   を同じ走行で測り、それを分母にする。
   不採用でもその事実と数値を残す (記録を消さない)。

**担当エージェント**: cpp-implementer (実装) + model-converter (参照 golden 生成) → gpu-benchmarker (画評価)。
Sonnet 実装後は Opus high レビュー。
**走る機械**: 実装・golden 突合は**開発機可**。画評価は**研究機**。
**発火条件**: **条件発火** — E-2 が「既存ノブ (steps/CFG/seed) だけでは頭打ち」を示したときのみ。
E-2 で既定調整だけで改善が取れるなら E-3 は起票しない。
★**2026-09-27: この発火条件は E-2 T2 の DoD 5 スイープで満たされた** — 4 条件 (cfg 5.0 / cfg 10.0 /
steps 12 / steps 28) すべてが 3 軸判定で `頭打ち/ノイズ支配`・有効レバー **0/4**・既定変更なし
(実数値と限定は E-2 節「現況 (2026-09-27) — T2」の「DoD 5」項)。
⚠ **ただし発火条件の充足 = 着手の決裁ではない**。起票・着手はユーザー/PL の決裁を要する。
また★**上の目的節にある「アニメ系は Karras/DPM++ 前提」「v-pred は出力が壊れる見込み」は
T2 を経ても依然として未検証の一般論**である (T2 は scheduler を 1 つも試していない)。
発火したのは「既存ノブが頭打ち」までで、**E-3 が効くという見込みは何も裏付けられていない**。

★**T2 が示した、E-3 着手前に検討すべき材料 (次のレバー候補)** — E-3 と**競合**する候補を含む:
1. **評価軸 (recall) の分散帯が広い**: 参照アームの seed 間 std = **0.0570** に対し、観測された
   最大の delta が **-0.0167** = 分散帯の 3 分の 1 以下。**この物差しでは、仮に E-3 が効いても
   検出できない可能性がある** (E-3 を回す前に、seed 本数を増やす / 分散の小さい指標に替える方が先かもしれない)。
2. **採点器の分解能不足**: 副指標 `worst_anatomy` は F-0a 実測どおり Limbs 軸のみ生存で、
   絶対値 0.0000-0.0041・分散帯 0.000487 と極小 = 実質的に死んでいる。
   **採点器 (ScorerNet / quality head) の分解能を上げる方が、sampler を増やすより上流のレバーになりうる**
   (Phase 4 の Q-2 / 「7 死軸の分解能診断」と同じ対象)。
3. **より広い CFG 範囲**: T2 は cfg 5.0-10.0 の 3 点しか見ていない。**cfg 2-3 の低域や 12 以上の高域は未測**で、
   「既存ノブの頭打ち」は**測った範囲の話**にすぎない。E-3 より安い再測として先に潰せる。
4. (T2 のスコープ外) preset は `illustrious-xl` 単独・prompt は p1/p2/p3 のみ = **他 preset / 他題材は未測**。

**依存**: E-2 (発火判断)、E-1 (評価物差し)。E-4 の v-pred 系候補は E-3 に依存する (逆向きの依存に注意)。

---

## E-4 — 追加 checkpoint 候補の調査 → 変換 → 評価

**目的**
2-6d で 3 preset を入れ、ユーザー目視順位は **illustrious-xl > noobai-xl > animagine-xl-4** (measurements-log l.534)、
既定は illustrious-xl (`660e538`)。
E-4 は **E-1 の多 seed 物差しの上で**、① 既存 3 preset の順位を測り直し、② 追加候補を変換して同じ格子に載せる。

★**2-6d の「N=1」但し書きの射程を取り違えないこと (2 つは別事象)**:

| 事象 | 何に付いた記述か |
|---|---|
| **measurements-log l.528 の「N=1 (seed 1 本) で優劣の根拠にはならない」** | ★**WD14 再現率の行に付いた但し書き** (l.528 は「再現率 = プロンプトの danbooru 相当語のうち WD14 が検出した割合 …」という文の末尾)。つまり**再現率という定量指標**についての注記 |
| **既定 preset = illustrious-xl の決定** | 同 l.534 の**ユーザー目視決裁 (2026-09-17)** による (`660e538`)。「animagine はダイナミックだが四肢の崩れが残る」という目視所見が根拠 |

→ **l.528 の但し書きを既定 preset 決定そのものの否認として引かない**。既定は**ユーザーの目視決裁**で決まっており、
E-4 が定量で別の順位を出したとしても、**それだけで既定を差し替える権限はエージェントに無い** (下記 DoD 5)。
E-4 が測り直すのは「再現率等の定量指標での順位」であって、目視決裁の代替ではない。
(なお定量順位が N=1 で出せないのは事実なので、E-4 は多 seed で測る。)

**DoD**
1. 候補ごとに **ライセンス確認**が済んでいる。**採否決裁はユーザー**が行う
   (2-6d では Fair AI Public License 1.0-SD を 2026-09-17 ユーザー決裁「採用可」・
   `THIRD_PARTY_NOTICES.md` 記載という前例・measurements-log l.516)。
   エージェントはライセンス決裁を自前で行わない (中継された承認を根拠にしない)。
   `THIRD_PARTY_NOTICES.md` の記載も DoD に含む。
2. 変換が 2-6d と同じ 4 ファイル構成 (unet / vae / TE-L / TE-G) で成立し、照合が取れている
   (2-6d の照合項目: UNet キー名 + shape/dtype の base 一致・tokenizer sha256 base 一致・TE 変換誤差 —
   CLAUDE.md 計測表「アニメ特化 SDXL 3 preset (2-6d)」行)。
3. `preset.json` に `vae_scaling_factor` の **config 値と実使用値の両方**が記録される (E-0 の処置と整合)。
4. E-1 の格子で **多 seed** 比較し、順位を実数値付きで書く。既存 3 preset も同じ走行に含める。
   判定は共通規律 2 の **3 軸**・**ノイズ床 (参照アームの同条件・異 seed sd)** を同走行で測ること。
   ここで得られるのは**定量順位**であり、2-6d のユーザー目視順位 (l.534) の追認/否認は
   「定量指標ではこう見える」という形でのみ書く (目視決裁の上書きではない・上記の射程注意)。
5. ★**既定 preset の変更にはユーザー決裁が要る** — ライセンス採否 (DoD 1) と**対称**の扱いにする。
   既定は 2-6d でユーザーの目視決裁により決まっている (measurements-log l.534・`660e538`)。
   エージェントは定量結果と推奨を**提示するところまで**で、既定変更を自前で確定しない
   (中継された承認を根拠にしない・DoD 1 と同じ規律)。
   ユーザー決裁が下りた場合のみ既定変更 commit を分け、変更前後の PNG sha256 を両方残す
   (sha256 の `--fast` 前提未確定は共通規律 1 のとおり引き継ぐ)。

**担当エージェント**: model-converter (調査・変換) → gpu-benchmarker (評価)。**ライセンス決裁はユーザー**。
**走る機械**: 研究機。
**発火条件**: **条件発火** — E-1 の物差しが立ってから。
**依存**: E-1。**v-prediction 系の候補は E-3 (prediction_type 分岐) に依存**するため、
E-3 未着手のあいだは epsilon 系候補に限る。

---

## E-5 — ランタイム LoRA (L-2) を実重みで e2e 検証

**目的**
L-2 (ランタイム LoRA) は実装・test 完了だが、**検証は数値パリティ中心**である。
roadmap l.334 の L-1 (offline merge) 行には「test 12/12 緑・**実マージ+実画像は研究機で別途**」とあり、
**実重み + 実画像の e2e は未実施**と読める。L-2 のゲートも
parity max_abs 4.9e-4 / revert memcmp bit-exact / stack+revert bit-exact (roadmap l.335) という数値ゲートで、
「実 LoRA を当てて絵が意図どおり変わるか」は別の問い。
E-5 は **実 LoRA 重みを 1 本以上当てて、絵が変わること・外すと元に戻ることを画像で確認**する。

★**実行経路の制約 (E-5 の依存が E-2 に掛かる理由)**
E-1/E-5 の格子は原則 **CLI 1 プロセス 1 枚** (E-1 節のとおり) だが、**E-5 だけは CLI で回せない**:

| 制約 | 一次証拠 |
|---|---|
| **LoRA は HTTP 専用・CLI から到達できない** | `grep -rn -i lora src/server/cli_generate.hpp src/main.cpp` = **0 件**。`loras` の受理は `src/server/api.cpp` l.179 のみ。`GenRequest::loras` (`src/server/generator.hpp` l.43) は HTTP ハンドラからしか埋まらない |
| **seed は HTTP から指定できない** ★**起票時 (2026-09-19/20) のスナップショット。E-2 T1 (branch `feat/e2-sampling-knobs`・main 未 merge) で HTTP `seed` は実装済み** (`src/server/api.cpp` の `body.contains("seed")` → `gr.has_seed=true; gr.seed=...`・`GenRequest` の当該ノブ部は `bool has_seed` + `uint64_t seed` の POD 表現 (構造体全体は POD ではない)。`c00e72b` の現物で 2026-09-22 に確認。★T1 初版 `4afbe2f` では `std::optional<uint64_t> seed` だったが CUDA ビルド回帰のため `c00e72b` で POD 化・意味論は不変) — **本行の制約は T1 branch 上では既に解消**。§3 の同趣旨バナーも参照 | `GenRequest` (`src/server/generator.hpp` l.34-46) に seed フィールド無し・`api.cpp` l.128-217 の受理フィールドにも無し。seed は env `DOLLAMA_SEED` のみ (`src/server/backend_image_generator.hpp` l.53-62, l.131) ★**この出典行番号は起票時のもの。T1 後の現物とはずれる** |

env `DOLLAMA_SEED` は**サーバープロセスの環境変数**であり、**HTTP クライアント側からリクエスト単位で指定できない**
(`getenv` 自体は `generate()` ごとに評価される — `src/server/backend_image_generator.hpp` l.132 が
1 リクエストごとに l.56-62 の `resolve_seed_from_env()` を呼ぶ・キャッシュなし。
つまり制約はプロセス起動時読み取りではなく、**値の供給元がプロセス環境しかない**ことにある)。
一方 LoRA は HTTP 専用。したがって **「多 seed × strength 軸」を 1 プロセス内で回すには HTTP が seed を
受けられる必要がある**。
⚠ ここが重要: **CLI `--cfg` / `--seed` を足すだけでは E-5 は解けない** (LoRA が CLI に無いため)。
E-5 の発火条件は「E-2 完了」一般ではなく、**E-2 のうち HTTP `seed` の実装が入ったとき**である
(この要求は E-2 DoD 1 に取り込み済)。

**代替 (HTTP seed を入れない場合)**: seed ごとにサーバープロセスを起動し直す運用も理屈上は可能だが、
1 プロセス = 5GB 重みロードで cold 21〜41s (measurements-log l.530-531。★2026-09-20 の E-1 実走では
**41.9〜66.9s/枚・最大 123.1s** = `docs/logs/e1/grid_summary.csv` / `grid_results.csv`。E-1 節「コスト前提」参照)
が seed ごとに掛かり、
かつ「同一プロセス内で LoRA を着脱して比較する」という L-2 本来の検証形にならない。
**本台帳は HTTP `seed` の追加 (E-2) を前提とする**。

**DoD**
1. 実 LoRA 重み (ライセンス確認済) を 1 本以上使い、HTTP `loras:[{name,strength}]` 経路で生成できる。
2. **多 seed** で strength 軸 (例 0.0 / 0.5 / 1.0) を E-1 の格子に載せ、
   strength=0 相当 (= `loras` 未送出) の PNG sha256 が **LoRA 無し経路と一致**すること
   (共通規律 1 の無改変ゲート。UI 側も空選択で `loras` キー非送出 = 従来経路無改変・roadmap l.335)。
   ⚠ **E-1 のハーネスは CLI 経路を前提にしているため、E-5 では HTTP 経路の走行系が別途要る**
   (E-1 のハーネスをそのまま使えると仮定しない。CSV スキーマと集計部の共用に留める想定)。
   判定は共通規律 2 の 3 軸 + ノイズ床。
3. **L-1 (offline merge) との突合**: 同一 LoRA・同一 strength・同一 seed で L-1 マージ品と L-2 ランタイム適用の
   出力を比較する。数値正典は L-1 offline merge (roadmap l.335)。
   bit 一致を要求するかは走行前に決めて明記する (FP16 の蓄積順差で bit 一致しない可能性があるため、
   **bit 一致を期待値として先に書かない**)。
4. `unet_clear_loras` 後の生成が LoRA 無しと一致することを**画像側でも**確認する。
5. LoRA 重みの入手経路とライセンスを `THIRD_PARTY_NOTICES.md` 相当に記録。

**担当エージェント**: gpu-benchmarker。**LoRA 重みの入手/ライセンスはユーザー決裁**。
**走る機械**: 研究機。
**発火条件**: **E-2 のうち HTTP `seed` が入った後** (上記「実行経路の制約」。
CLI `--seed` だけでは不可 = LoRA が CLI に無いため)。
★**2026-09-22 現在: この条件は E-2 T1 (branch `feat/e2-sampling-knobs`) で満たされている**
(`src/server/api.cpp` の `body.contains("seed")` → `gr.seed`)。**ただし main 未 merge** のため、
E-5 を走らせる前に「どのツリーで走らせるか」を確認すること。
**依存**: **E-2 (特に HTTP `seed`)**、E-1 (指標定義・集計部)。ユーザーのライセンス決裁。

---

## 3. 一次証拠 (本文の主張の出典)

本台帳で引いた記述の出典。**行番号は `bc77447` 時点** (起票 2026-09-19 / record-auditor 指摘の是正 2026-09-20。
是正時に全行の行番号を再検算済)。

★**本表は起票時 (2026-09-19/20) のスナップショットであり、そのまま残す**。ただし E-2 T1
(2026-09-22・branch `feat/e2-sampling-knobs`・main 未 merge) 以降、下記 6 行は**現在の src の状態を
表さない** — 読むときは E-2 節「現況 (2026-09-22)」と突き合わせること:
「CLI に `--cfg` / `--seed` が無い」/「HTTP が受けるフィールドに `guidance_scale` も `seed` も無い」/
「`GenRequest` に seed フィールドが無い」/「seed は env `DOLLAMA_SEED` 経由のみ」/
「実装は受理するが仕様表に無い 2 件」(DoD 3 で解消済)/
「HTTP 拡張フィールド表の現況 (拡張と明記は 2 件)」(T1 で仕様表は **10 行・拡張明記 6 件**になった)。
「CFG スケールがコンパイル時定数」の行は、**既定値が `constexpr` である点は不変**だが
`guidance_scale` / `--cfg` による上書き経路が加わった。
★**この列挙は網羅を保証しない**。T1 は `src/main.cpp` に行を足しているため、**同ファイルを指す
行番号は全般にずれている** (例: 本表と DoD 1 が引く `src/main.cpp` l.255 の `GenRequest req{...}` は
T1 後の現物では **l.292** — 2026-09-22 に現物を開いて確認)。行番号を使う前に現物で検算すること。

| 主張 | 出典 |
|---|---|
| illustrious-xl の HF `vae/config.json` = `scaling_factor 0.18215`・ランタイムは 0.13025 固定 | `docs/measurements-log.md` l.537 |
| VAE sha256 一致からは「HF config の誤記」と結論できない (論理撤回済) | 同 l.540 |
| 0.13025 で decode・目視異常なし/定量未検証・正しい scaling は未確定 | 同 l.541 |
| scaling_factor 残債の処置は **未実施** | 同 l.542 |
| **N=1 (seed 1 本) で優劣の根拠にはならない** (★**WD14 再現率の行**に付いた但し書き) | 同 l.528 |
| 再変換不可 (`bafe0a5` は 0.13025 以外で `SystemExit(1)`) / `converted_at` 22:15 < mtime 22:25 < commit 23:13 = commit 前の版の成果物 | 同 l.538-539 |
| 2-6d の 12 枚は全て別プロセス起動 = cold 相当 / base 21〜22s vs preset 37〜41s (差の原因は未計測) | 同 l.530, l.531 |
| ★**E-1 実走の実測秒 (見積りの分母)**: `sec` 列 41.16〜123.1s/枚 (16 行) / 条件別平均 41.93・54.9・58.75・66.92 s | `docs/logs/e1/grid_results.csv` (`sec` 列) / `docs/logs/e1/grid_summary.csv` (`sec_mean_characterization_only`)。2026-09-20 に現物を開いて確認 |
| `scripts/dollma_eval_image_grid.py` の docstring が **「Phase 5-1 の先行実装を兼ねる」と断定**している (台帳の姿勢と不一致) | 同スクリプト docstring l.10-13 (2026-09-20 に現物を開いて確認・未 commit) |
| CLI も HTTP と同じ `GenRequest` を通る (「HTTP body > CLI」という段は存在しない) | `src/main.cpp` l.255 `dollama::GenRequest req{prompt, negative, 1, steps, width, height};` (**位置指定の集成初期化**・l.256-258 で matting/preset_prefix は代入) / `grep -n "GenRequest\|generate(" src/server/cli_generate.hpp` = **0 件** (2026-09-20 実行) |
| `--fast` の有無は log に記録が無く**未検証** (2-6d 残債 2 / 2-6e「未検証」欄) | 同 l.543, l.556 |
| 2-6d 残債 4 = HTTP API preset フィールド / UI preset 選択が未着手 | 同 l.545 |
| **ユーザー目視決裁 (2026-09-17)** で既定 preset = illustrious-xl (`660e538`) | 同 l.534 |
| 多 seed 判定 3 軸 (a)全 seed 符号一致 (b)効果量 > 参照アーム分散帯 (c)各 seed paired CI が 0 除外 | 同 l.56 (D5・(b)のみ不成立), l.57 (B・3 軸成立), l.58 (B-2・分散帯 6.7–8x), l.49 (D6・符号反転で不採用), l.72 (A12k・seed42 反転), l.73 (D80・seed7 反転)。seed sweep はいずれも **4 seed** |
| F-0a 実走: worst-axis argmax Limbs 77 / Hands 2 / Head 1・他 7 軸 max<0.002 (Head 0.0118)・reward std 0.0377 / best−worst 0.2031・「ScorerNet の dynamic range が Limbs のみ生存 = 分解能不足」 | 同 l.67 |
| 旧 E-1 = ScorerNet コーパス 180 枚 (ID 衝突) | 同 l.63, l.68 / `docs/dataset-spec.md` l.766, l.869 |
| 2-6d 走行で StubGenerator 混入事故 → フォールバック grep を有効条件に再走 | 同 l.532 |
| 2-6e で PNG sha256 3/3 一致 = 出荷経路が 2-6d 条件と bit 同一 / grep は大小区別 | 同 l.555 |
| HTTP 非 bool を 400 で弾く前例 (`preset_prefix`) | 同 l.554 |
| Fair AI Public License 1.0-SD のユーザー決裁・THIRD_PARTY_NOTICES 記載 | 同 l.516 |
| ランタイムの VAE scaling factor は固定値 | `src/infer/diffusion.cu:41` `constexpr float kScalingFactor = 0.13025f;` |
| sampler は 1 種・固定設定のみ (leading/steps_offset 1 / karras=false / epsilon) | `src/infer/scheduler.hpp` l.13 (spacing), l.14 (karras), l.15 (prediction_type)。golden 突合の記述は同 l.17-21 |
| CFG スケールがコンパイル時定数 | `src/server/sdxl_backend.hpp:57` `static constexpr float kGuidanceScale = 7.5f;` |
| CLI に `--cfg` / `--seed` が無い | `src/main.cpp` l.174-222 の引数分岐 (`--http`/`--port`/`--steps`/`--width`/`--height`/`--prompt`/`--negative`/`--out`/`--no-matting`/`--no-preset-prefix`/`--fast`/`--fp8`/`--preset` の 13 本のみ。`grep -n '"--' src/main.cpp` で全数確認) |
| HTTP が受けるフィールドに `guidance_scale` も **`seed`** も無い | `src/server/api.cpp` l.128-217 (prompt / negative_prompt / n / steps / size / preset_prefix / loras / response_format) |
| **`GenRequest` に seed フィールドが無い / `loras` はある** | `src/server/generator.hpp` l.34-46 (prompt / negative_prompt / n / steps / width / height / matting / **loras** (l.43) / preset_prefix) |
| seed は env `DOLLAMA_SEED` 経由のみ | `src/server/backend_image_generator.hpp` l.53-62, l.131 |
| ★**LoRA は CLI から到達できない** | `grep -rn -i lora src/server/cli_generate.hpp src/main.cpp` = **0 件** (2026-09-20 実行)。受理は `src/server/api.cpp` l.179 のみ |
| 実装は受理するが仕様表に無い 2 件 = `preset_prefix` / `loras` | 受理: `src/server/api.cpp` l.166 / l.179。仕様表: `docs/http-api-spec.md` l.30-35 (ヘッダ l.28) に両者とも**記載なし** |
| HTTP 拡張フィールド表の現況 (拡張と明記は `negative_prompt` l.33 と `steps` l.34 の 2 件) | `docs/http-api-spec.md` l.30-35 (ヘッダ l.28) (prompt / n / size / negative_prompt / steps / response_format の 6 行) |
| L-1 の「実マージ+実画像は研究機で別途」= 実重み e2e 未実施 | `docs/roadmap.md` l.334 |
| L-2 のゲートは数値パリティ中心 / UI 空選択で `loras` 非送出 = 従来経路無改変 | `docs/roadmap.md` l.335 |
| Phase 5-1 = 崩壊境界 probe・既存 QualityGate で採点・採点器は新規不要 (★**5-1 の内容説明であって「E-1 が兼ねられる」の根拠ではない**) | `docs/roadmap.md` l.217 |
| Phase 5 の着手順 = 「**5-1 プラン化** → 5-2 denoise 実験 → 5-3 決裁」 | `docs/roadmap.md` l.224 |
| Phase 5 は 5-1 → 5-2 → 5-3 の依存連鎖 | `docs/roadmap.md` l.203-212 |
| 既存の未解決レバー「**SDXL seed 制御 (HTTP に seed 引数)**」/ reward 設計 / 日本語条件付け改修 / M 拡大 | `docs/f0b-rejection-sft-plan.md` l.116 (a)(b)(c)(d) |
| 「**次レバー** (F-0b 後): reward 設計 / 日本語条件付け改修 / seed 制御」 | `docs/roadmap.md` l.321 |
| WD14 再現率は**スクリプト化されておらず語リスト直書きの手計算**・数え落としの訂正実例あり | `docs/logs/2-6d/README.md` l.70-79 (算出方法の節・l.75-77 が語リスト・l.78 が `from side` 数え落としの訂正) |
| 既存同型ハーネス 3 本の存在 | `scripts/dollma_gen_scorer_corpus.py` / `dollma_collect_rollouts.py` / `dollma_rollout_bestofn.py` (`ls scripts/` で実在確認) |
| 「画像 seed は server 内 time(秒)・非再現」= HTTP 経路で seed が効かない自書の実例 | `scripts/dollma_gen_scorer_corpus.py` docstring l.24 (`DOLLAMA_OV_TOKENIZERS_DLL` 注意は同 l.21-22) |
| `scripts/dollma_eval_image_grid.py` は起票時 (2026-09-19) 未存在 → **2026-09-20 時点で未追跡ファイルとして実在** | `git status --short` = `?? scripts/dollma_eval_image_grid.py` / `?? docs/logs/e1/` (2026-09-20 実行。`docs/logs/e1/` の中身は `img` / `log` / `contact_sheets` / `grid_results.csv` (16 データ行) / `grid_summary.csv`。`ls -la` + 両 CSV を直接開いて確認) |
| 記録は record-writer が書き record-auditor が検査 | `CLAUDE.md` 「実装作業のルール」4 |
| 2-6d の照合項目 (UNet キー/tokenizer sha256/TE 誤差) | `CLAUDE.md` 計測表「アニメ特化 SDXL 3 preset (2-6d)」行 |

**未検証として明示したもの** (断定していない):
- **E-1 の汎用ハーネスで Phase 5-1 を兼用できるか** (5-1 起票時に再評価)。roadmap l.217 は 5-1 の内容説明であり
  兼用可否を述べた記述はリポ内に無い。roadmap l.224 の「5-1 プラン化が先」との関係も**未整理**。
- 「アニメ系 checkpoint は Karras / DPM++ 前提」という一般論 (本プロジェクトでは未測定)。
- **v-prediction 系 checkpoint の挙動**: 「重みロードは通り、サンプリング式が合わず出力が壊れる」という
  見込みであって、本プロジェクトで v-pred を読ませた実測は無い (「そもそも読めない」という断定は撤回した)。
- E-0 の正しい scaling 値 (0.13025 / 0.18215 のどちらか)。
- **2-6d / 2-6e 走行の `--fast` 有無** (measurements-log l.543, l.556)。sha256 一致ゲートはこの穴を継承する。
- **判定 3 軸 (共通規律 2) を画像側メトリクスに適用した実績**。Phase 4 のタグ集合メトリクスでのみ確立しており、
  WD14 再現率 / anatomy 軸での閾値感は E-1 の実測で校正が要る。
- **項目一覧の「発注済」**: リポ内に branch / commit / issue の裏付けが無く、出典は main thread の口頭発注
  (リポ外)。E-1 のみ未追跡成果物という間接証拠がある。

**本台帳の反映範囲**: 今回は `docs/image-quality-plan.md` の新規起票 + record-auditor 指摘の是正のみ。
`CLAUDE.md` / `docs/roadmap.md` への反映は**台帳が固まってから別途** (未反映であることを明記して残す)。
未追跡の `scripts/dollma_eval_image_grid.py` / `docs/logs/e1/` は本台帳では**完了扱いにしていない**
(未 commit・未レビューのため)。

**残債 (本台帳のスコープ外・別途決裁が要るもの)**:
1. `docs/http-api-spec.md` の既存乖離 2 件 (`preset_prefix` / `loras` が仕様表に無い) — E-2 DoD 3 に
   取り込んだが、**E-2 の発火を待たずに単独で直す**選択肢もある (仕様と実装の乖離は現に存在するため)。
2. ID 衝突 (旧 E-1 / 旧 E-2) の恒久解消 — 本台帳は「image-quality-plan の E-n」という呼称規約で回避したが、
   既存 doc 側の記号をリネームする案は未検討。

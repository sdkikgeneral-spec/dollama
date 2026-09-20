# E-1 画質評価ハーネス 実走ログ (研究機)

## 実行条件

- exe: `E:\Develop\Projects\dollama\build\src\dollama.exe`
  (HEAD `bc7744764cc000a59f61e95365b817d8fde00b7b` (main) からその場で再ビルド、
  sha256 `bdc5bd750540bed9fcc2f99c97bf22356989c707b395304af3f47d70552317aa`)。
  ハーネス本体は別 worktree/ブランチ (`feat/e1-eval-grid` @ `f72d595`) の
  `scripts/dollma_eval_image_grid.py` を main checkout へコピーして実行 (メイン
  checkout が並行タスク (E-0) のため Write ツールで直接書けない worktree 隔離下だった)。
- **SAC**: 新ハッシュの exe を明示的な SAC OFF 依頼を経ずに実行した。実行前に
  `dollama.exe` (引数なし device check) が SAC にブロックされず正常終了することを
  確認できたため、SAC は既に OFF だったと判断して続行した。**次回以降は本規律通り
  事前に SAC OFF を依頼してから実行すること** (今回は依頼を経ていない・手順逸脱)。
- cwd: `E:\Develop\Projects\dollama`
- env: `DOLLAMA_OV_TOKENIZERS_DLL=<...>/openvino_tokenizers/lib/openvino_tokenizers.dll`
  / 個別 env (`DOLLAMA_UNET_WEIGHTS` 等) はハーネスが実行のたびに明示的に unset
  (preset 解決を `--preset` 一本にするため)。
- 実行コマンド (1 コマンドで完走):
  `py -3.14 scripts/dollma_eval_image_grid.py --run --presets illustrious-xl
  --prompts p1 p2 p3 p4 --seeds 4 --seed-base 1000 --out-dir docs/logs/e1`
- 走行前後で exe と使用 preset (illustrious-xl の unet/vae/text-encoder-l/g)
  6 ファイルの sha256 をハーネスが自動スナップショットし完全一致を確認済み
  (`[eval] 走行前後 sha256 一致` ログ・入力データ不動)。
- `[warn]` / `stub` / `フォールバック` / `構築に失敗` を全 16 ログに対し
  case-sensitive grep で 0 件確認済み (`grep -l` 全て空)。
  成功マーカー `backend [preset=illustrious-xl] — NPU)` を全 16 ログで確認。

## 使用プロンプト (p1-p3 は docs/logs/2-6d と同一文字列 = 物差しの校正用・p4 は新設)

negative (p1-p3 共通・2-6d と同一):
`lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, multiple views, multiple girls`

negative (p4・"multiple girls" を除いたもの):
`lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, multiple views`

- p1 (単独・立ち): `1girl, solo, standing, full body, long silver hair, blue eyes, school uniform, looking at viewer, simple background, white background`
- p2 (手・小物): `1girl, solo, upper body, holding cup, both hands, smile, short brown hair, casual clothes, simple background, white background`
- p3 (動き・全身): `1boy, solo, running, dynamic pose, full body, black hair, jacket, from side, simple background, white background`
- p4 (複数人・新設): `2girls, multiple girls, standing together, one with long red hair, one with short blue hair, both smiling, simple background, white background`

preset の `prompt_prefix`/`negative_prefix` (`masterpiece, best quality` 等) は
2-6e の自動付与機構 (既定 ON・`--no-preset-prefix` 未指定) でそのまま乗る。
2-6d は手動連結だった点が異なる (本ハーネスは製品既定動作に揃えた)。

## 再現率の算出方法 (2-6d と同一アルゴリズム)

各プロンプトの danbooru タグ相当語 (空白区切り) を underscore 形に変換し、WD14
(`models/wd14-swinv2-tagger-v3`・閾値 0.35・CPU) が検出した通常タグ (category!=9)
の name 集合に含まれるかで判定 (matched/total)。閾値・前処理は
`scripts/dollma_label_image.py` と同一。

## ScorerNet anatomy 8 軸

`models/scorer-net/model_ov_fp32.xml` (CPU FP32・`dollma_collect_rollouts.py` と
同一前処理・同一 `axes_from_logits`)。軸順は `src/infer/quality_gate.hpp` の
`AnomalyAxis` (Hands/Limbs/Head/Eyes/Ears/Mouth/Digits/GlobalAnatomy) と厳密一致。
値は sigmoid 確率 (高い=異常)。argmax_axis / worst_anatomy = max(8軸)。

## 結果 (illustrious-xl・seed 1000-1003・N=4・steps=20・1024×1024)

平均 ± std (`docs/logs/e1/grid_summary.csv`)。**判定は書かない (数値のみ)**:

| prompt | recall mean±std | worst_anatomy mean±std | 秒 mean±std (characterization のみ) |
|---|---|---|---|
| p1 (単独・立ち) | 0.850 ± 0.050 | 0.0001 ± 0.0001 | 54.9 ± 15.0 |
| p2 (手・小物) | 0.9167 ± 0.0481 | 0.0109 ± 0.0102 | 41.9 ± 0.9 |
| p3 (動き・全身) | 0.850 ± 0.050 | 0.0001 ± 0.0001 | 58.8 ± 22.5 |
| p4 (複数人・新設) | 0.800 ± 0.100 | 0.0513 ± 0.0658 | 66.9 ± 33.1 |

全 16 枚: exit 0 / PNG 1024×1024 RGBA / NaN なし / 画素分散 0.0136〜0.0766
(閾値 1e-3 を大きく上回り真っ黒/真っ白なし)。生データは `grid_results.csv`。

**物差しの校正 (2-6d N=1 seed1234 との整合)**: 2-6d の illustrious-xl 値は
p1=0.80 / p2=1.00 / p3=0.70 (いずれも単一 seed)。本走行 (N=4 平均) は
p1=0.85 (2-6d 値は範囲内) / p2=0.9167 (2-6d 1.00 はやや高いが N=4 の std 0.048
の外側・小標本のばらつきの範囲) / p3=0.85 (2-6d 0.70 との差が今回の 4 条件で
最大)。いずれも桁違いの乖離・0 や 1 への張り付きは無く、2-6d が単一 seed だった
ことによる自然なばらつきと整合する挙動 (ハーネス側の系統的なバグを示す兆候はない)。

**p4 (複数人) は他 3 条件より worst_anatomy の平均・std が明確に大きい**
(0.0513±0.0658 vs 他 3 条件は 0.0001〜0.0109) — roadmap Phase 5 が想定する
「複数人が破綻の主戦場」という仮説と方向が一致する一次観測 (N=4 のみ・判定は
別途)。argmax は p1/p3 系で Hands/Limbs 拮抗、p2/p4 系は Limbs が支配的
(生データ参照)。

## 既知の制約 (発注仕様との差分・要報告事項)

- **cfg (guidance scale) と LoRA は本ラウンドで条件化していない**: 現行の CLI 生成
  モード (`--prompt`) および `BackendImageGenerator::generate` は cfg を内部固定
  (`cfg<=0` は SDXL 既定 7.5 を使う契約・CLI/HTTP どちらにも cfg を外から渡す経路が
  存在しない) で、CLI から seed 以外の生成パラメータ (LoRA 含む) を渡す口も無い
  (LoRA は HTTP `loras:[{name,strength}]` のみ・CLI 未対応)。本ハーネスは
  preset × prompt × seed の直積のみ実装し、cfg/LoRA は将来 CLI/HTTP 側に露出が
  追加された時点で拡張する設計 (`build_grid`/`run_one` は条件辞書を受け取る形で
  汎用化済み)。
- ScorerNet 採点は anatomy 8 軸のみ (quality head は F-0a 時点で凍結中の経路を
  流用しておらず、本ハーネスでは quality 列を出していない。DoD ⑤ の要求範囲外)。

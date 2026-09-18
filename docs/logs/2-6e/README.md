# 2-6e: 出荷経路 prefix 自動付与 — 2-6d 評価条件との一致検証 (研究機実走)

## 条件

- exe: `E:\Develop\Projects\dollama-g10k-wt\build\src\dollama.exe`
- exe sha256: `4ed1d47042d4898d9f9b08584794501b619ab7425189be7cca67ed02b95f6f10`
- HEAD: `4d4a40596952100bd10a0fda63dddd2f2286723e` server(preset): 2-6e — preset.json の prompt_prefix/negative_prefix を生成時に自動付与
- env: `DOLLAMA_OV_TOKENIZERS_DLL` (openvino_tokenizers.dll フルパス) / `DOLLAMA_SEED=1234`。`DOLLAMA_UNET_WEIGHTS` / `DOLLAMA_VAE_WEIGHTS` / `DOLLAMA_ENCODER_L` / `DOLLAMA_ENCODER_G` / `DOLLAMA_BACKEND_PRESET` は未設定を確認済み (既定 preset=illustrious-xl)
- cwd: `E:\Develop\Projects\dollama` (models/presets 解決のため)
- SAC: OFF 済 (発注文の前提。本実走内では未再確認)
- src/scripts は無改変・ビルドなし (exe は既存ビルド成果物を使用)
- コマンドライン (`--fast` の有無を含む) は本 README・各 log に記録なし (CLI 生成モードはフラグをログしない)。`--fast` は発注条件だが**未検証** (measurements-log 「2-6e」小節と同じ扱い)

## 結果

| # | prompt/negative 全文一致 (vs 2-6d FULL_PROMPT/FULL_NEGATIVE) | PNG sha256 一致 (vs 2-6d) | 画素差 |
|---|---|---|---|
| p1 | 一致 | 一致 (`136d388f...`) | — |
| p2 | 一致 | 一致 (`3765065d...`) | — |
| p3 | 一致 | 一致 (`48553e0b...`) | — |
| p1_noprefix (`--no-preset-prefix`) | (prefix 無効化のため対象外・下記参照) | **不一致** (`7f1dc13e...` ≠ p1 `136d388f...`) | 未算出 (sha256 不一致で意図通り) |

- 各ログで `[warn]` / `stub` / `フォールバック` / `MISSING` / `構築に失敗` は **0 件**。
- 有効条件確認: `sdxl backend [preset=illustrious-xl] — NPU` 行 + `[gen] preset_prefix applied: prompt='...' negative='...'` 行、p1/p2/p3 すべてで出力あり。
- `[gen] preset_prefix applied` の prompt/negative 全文は、2-6d `docs/logs/2-6d/illustrious-xl/p{1,2,3}.log` の `FULL_PROMPT:` / `FULL_NEGATIVE:` 行と **文字列完全一致**。
- p1_noprefix (`--no-preset-prefix`): ログに `preset_prefix applied` 行は出力されず (OFF が意図通り機能)。生成 PNG sha256 は p1.png と異なる (`7f1dc13e...` != `136d388f...`) = prefix 無効化で出力が変わることを確認。

## 結論

出荷経路 (prefix 自動付与, preset 既定 illustrious-xl) は、2-6d で目視評価に使った条件 (prefix 付き prompt/negative・seed=1234) と**完全一致**する (`--fast` は発注条件だが log に記録なし = 未検証・measurements-log 「2-6e」小節と同じ扱い。PNG sha256 一致から拡散条件が 2-6d 走行と同一であることまでは言える)。`--no-preset-prefix` は正しく prefix 付与を無効化する。フォールバック・stub 発火は皆無。

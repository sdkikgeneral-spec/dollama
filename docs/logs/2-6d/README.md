# 2-6d アニメ特化 SDXL 3 preset 生成比較 (研究機実走ログ)

## 実行条件

- exe: `E:\Develop\Projects\dollama-g10k-wt\build\src\dollama.exe`
  (branch `feat/2-6d-anime-presets`, HEAD `23ed047465790c8f996e3816eeb23a050346ab85`,
  sha256 `2abdc75e3d41f0876f574438d1e892ec84681e1ea2910def7e3601fff06c9564`)
- cwd: `E:\Develop\Projects\dollama` (メイン checkout ルート。base 重み
  `src/tests/data/*.safetensors` と `models/` (base OV 資産 + `models/presets/{animagine-xl-4,illustrious-xl,noobai-xl}/`) の相対解決のため)
- env: `DOLLAMA_OV_TOKENIZERS_DLL=C:\Users\sdkik\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages\openvino_tokenizers\lib\openvino_tokenizers.dll`
  / `DOLLAMA_SEED=1234`。実走前に `DOLLAMA_UNET_WEIGHTS` / `DOLLAMA_VAE_WEIGHTS` / `DOLLAMA_ENCODER_L` /
  `DOLLAMA_ENCODER_G` / `DOLLAMA_BACKEND_PRESET` が環境に残っていないことを確認済み (残っていなかった)。
- **SAC 未確認**: 本タスクでは SAC の ON/OFF を自分では確認していない (事前に OFF 済みとして
  引き継いだ)。exe は既存ビルド済みバイナリをそのまま使用 (再ビルド・src/scripts 変更なし)。
- preset は `--preset <name>` で指定 (base は付けない)。各実行後に log を grep し
  `[warn]` / `stub` / `フォールバック` / `構築に失敗` が 1 行でもあれば無効として即停止する運用で確認、
  かつ `dollama HTTP server (sdxl backend [preset=<name>] — NPU)` 行の存在を有効条件とした
  (`+env` 付きは無効扱い・今回は全 9 枚とも該当なし)。
- 経緯: 初回実走時に `models/presets/` が別エージェントの作業により実走中に消失し、
  animagine-xl-4 の一部が base フォールバックのまま生成される事故が発生 → 復旧確認後に
  該当ディレクトリを削除し、本ログは上記の grep チェック付きで全 12 枚を再実走したもの。
  (補足: 事故時の animagine-xl-4 3 枚は無効として削除済みで本ディレクトリには残っていない・現 animagine-xl-4/p1〜p3 は再走分。)

## 使用プロンプト (全文)

negative 共通:
`lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, multiple views, multiple girls`

- p1 (単独・立ち): `1girl, solo, standing, full body, long silver hair, blue eyes, school uniform, looking at viewer, simple background, white background`
- p2 (手・小物): `1girl, solo, upper body, holding cup, both hands, smile, short brown hair, casual clothes, simple background, white background`
- p3 (動き・全身): `1boy, solo, running, dynamic pose, full body, black hair, jacket, from side, simple background, white background`

preset 条件は `models/presets/<name>/preset.json` の `prompt_prefix` / `negative_prefix` を
prompt/negative の先頭に付与 (exe 自体は自動付与しないため、CLI 引数側で手動連結)。

| preset | prompt_prefix | negative_prefix |
|---|---|---|
| animagine-xl-4 | `masterpiece, high score, great score, absurdres` | `lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, low score, bad score, average score` |
| illustrious-xl | `masterpiece, best quality` | `lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality` |
| noobai-xl | `masterpiece, best quality, newest` | `lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, very displeasing` |

## 結果表 (12 枚・判定は書かない)

~~condition ごとに p1 (cold, 重みロード込み) → p2/p3 (warm)。~~
**訂正 (record-writer 2026-09-18)**: 各枚は**別プロセス起動** (各 log に `dollama — CLI 生成モード` 起動行と `real` が独立に出る = 5GB 重みロード込み) のため**全 12 枚が cold 相当**で、初版の「p1 cold / p2・p3 warm」は誤り。base 21s 台 vs preset 37s 台 (p2/p3) の差は**生成速度の差とは言えず原因未計測** (初版が cold と呼んだ p1 との差も OS ファイルキャッシュ等の可能性までしか言えない)。秒は preset 比較の指標に使わないこと。下表の (cold)/(warm) 表記は初版のまま残置 (上記のとおり読み替える)。

| condition | prompt | 秒 (real) | exit | 再現率 (検出/プロンプト語数) | 異常 | PNG |
|---|---|---|---|---|---|---|
| base | p1 | 63.6s (cold) | 0 | 8/10 (0.80) | なし | `docs/logs/2-6d/base/p1.png` |
| base | p2 | 21.0s (warm) | 0 | 9/9 (1.00) | なし | `docs/logs/2-6d/base/p2.png` |
| base | p3 | 21.9s (warm) | 0 | 8/9 (0.89) | なし | `docs/logs/2-6d/base/p3.png` |
| animagine-xl-4 | p1 | 40.8s (cold) | 0 | 9/10 (0.90) | なし | `docs/logs/2-6d/animagine-xl-4/p1.png` |
| animagine-xl-4 | p2 | 37.7s (warm) | 0 | 7/9 (0.78) | なし | `docs/logs/2-6d/animagine-xl-4/p2.png` |
| animagine-xl-4 | p3 | 37.0s (warm) | 0 | 6/9 (0.67) | なし | `docs/logs/2-6d/animagine-xl-4/p3.png` |
| illustrious-xl | p1 | 75.7s (cold) | 0 | 8/10 (0.80) | なし | `docs/logs/2-6d/illustrious-xl/p1.png` |
| illustrious-xl | p2 | 37.9s (warm) | 0 | 9/9 (1.00) | なし | `docs/logs/2-6d/illustrious-xl/p2.png` |
| illustrious-xl | p3 | 37.1s (warm) | 0 | 7/9 (0.78) | なし | `docs/logs/2-6d/illustrious-xl/p3.png` |
| noobai-xl | p1 | 73.1s (cold) | 0 | 9/10 (0.90) | なし | `docs/logs/2-6d/noobai-xl/p1.png` |
| noobai-xl | p2 | 39.9s (warm) | 0 | 9/9 (1.00) | なし | `docs/logs/2-6d/noobai-xl/p2.png` |
| noobai-xl | p3 | 40.8s (warm) | 0 | 8/9 (0.89) | なし | `docs/logs/2-6d/noobai-xl/p3.png` |

~~warm 平均 (p2/p3 平均): base 21.4s / animagine-xl-4 37.4s / illustrious-xl 37.5s / noobai-xl 40.4s。~~ (「warm」ではない・上記訂正を参照。p2/p3 平均の数値自体はそのまま)

全 12 枚: PNG 1024×1024 / bit depth 8 / color type 6 (RGBA 透過) / bytes 4195716 (同寸なので同一値)。
NaN/Inf なし・画素分散 0.062〜0.154 (閾値 1e-3 を大きく上回り真っ黒/真っ白なし)。

## 再現率の算出方法

各プロンプトの danbooru タグ相当語 (下記) のうち、WD14 (`scripts/dollma_label_image.py` 既定閾値)
が検出した tag line に含まれる語の割合。判定はしない (数値のみ)。

- p1: `1girl, solo, standing, long hair, silver hair, blue eyes, school uniform, looking at viewer, simple background, white background` (10語)
- p2: `1girl, solo, upper body, cup, smile, short hair, brown hair, simple background, white background` (9語)
- p3: `1boy, solo, running, dynamic pose, full body, black hair, jacket, simple background, white background` (9語)

各 PNG の詳細タグは `docs/logs/2-6d/<condition>/<pid>.tags.txt` を参照。
各実行の生ログ (起動時 `[preset] resolved:` 4行・`[gen] seed=1234(env)`・生成秒・exit code) は
`docs/logs/2-6d/<condition>/<pid>.log` を参照。

# G-10k T2a 基準採取 生ログ (2026-09-04・研究機 KIK-WIN-RTX58・SSH 越し)

G-10k (conv true batch2) の着手**前**に採った **before 基準**の一次証拠。
T2b 以降で「profile OFF の出力が変わっていない」ことを示すための突合先。

**本ディレクトリは採取のみで、判定・合否は一切含まない。**

---

## 1. 採取環境

| 項目 | 値 |
|---|---|
| 採取日時 | 2026-09-04 20:33〜20:37 JST (UTC 11:33〜11:37) |
| hostname | `KIK-WIN-RTX58` (研究機) |
| OS | Microsoft Windows 11 Home 10.0.26200.0 |
| リモートシェル | PowerShell 7.6.5 (pwsh) |
| GPU | NVIDIA GeForce RTX 5080 / VRAM **16303 MiB** / driver **610.88** (KMD 610.88 / CUDA UMD 13.3) |
| GPU アイドル時 | 35℃ / SM 180MHz / 465MiB 常駐 / 11.5W (cap 360W) |
| nvcc | **release 13.3, V13.3.33** (`cuda_13.3.r13.3`) |
| meson | **1.11.1** (`C:\Users\sdkik\AppData\Local\Python\pythoncore-3.14-64\Scripts\meson.exe`) |

### 両機の HEAD (一致を確認済み)

| 機体 | repo | branch | HEAD | `git status --porcelain` |
|---|---|---|---|---|
| 研究機 KIK-WIN-RTX58 | `E:\Develop\Projects\dollama` | `feat/g10k-conv-true-batch2` | `e17374c5b40437e6e23f628e8952615a1c75297c` | **出力 0 行 (clean)** |
| 開発機 (本 repo) | `E:\Projects\dollama` | `feat/g10k-conv-true-batch2` | `e17374c5b40437e6e23f628e8952615a1c75297c` | (本ログ追加前は clean) |

→ **両機の HEAD は一致している** (`e17374c`)。

### SAC (スマートアプリコントロール) の状態

- **ユーザー申告で OFF。** 本タスクではその申告以上の確認はしていない。
- 本タスクで実走した exe は `meson compile` を**本タスク外**で通った既存ビルド成果物
  (`mtime` は 2026-09-04 20:18:45〜20:19:00 = 本走行の約 15 分前)。**本タスクでは
  `meson compile` を一度も実行していない。**
- ★**言えるのはここまで**: 「上記 exe の実走は 4 本すべて exit 0 で成功した」。
  **新規ビルド exe (T2b で作られるもの) について SAC が通るかは本タスクでは未検証。**
- `WinError 4551` /「アプリケーション制御ポリシーによってブロック」は**一度も出ていない**。

### 実走に使った exe の同一性 (T2b との突合用)

| exe | size | mtime | sha256 |
|---|---|---|---|
| `build/src/prof_arena_e2e.exe` | 1387008 | 2026-09-04 20:18:59 | `CA11AF279B729E23A2D928D8BDA2B14876AF16167B1711338F44EAA9BDDA6FDB` |
| `build/src/test_conv2d.exe` | 251392 | 2026-09-04 20:18:53 | `C14BB9E88B381D8A222B499B163F1D73FDEF1642EF38EA289B4EC5EB685FFC9C` |
| `build/src/test_diffusion_batch2.exe` (参考・未実走) | 1396736 | 2026-09-04 20:19:00 | `6A3B1B80E445343408E2BDC66A81B58CB7818803DDA3FEE68668A52343658F89` |
| `build/src/test_unet_fast.exe` (参考・未実走) | 1260032 | 2026-09-04 20:18:54 | `6C8F5B1F705D9D7D0E46E0FD0D7BD1BAD4C281E050BFF028EDEA466F802D4F0A` |
| `build/src/dollama.exe` (参考・未実走) | 1975808 | 2026-09-04 20:18:45 | `9133C56531CA07B150F7E17B08C1E2AB6BE8DF346D2F3D057001AD831A13C8FF` |

---

## 2. 計器の選定 (`generate_txt2img` 到達の一次証拠)

`docs/g10k-plan.md` §2 の規則どおり、**doc ではなくソースを開いて**確認した 3 点。

### 基準A: `prof_arena_e2e` (e2e・被験構成 batch2 ON)

| # | 問い | 一次証拠 |
|---|---|---|
| ① | どの関数が呼ばれるか | `src/tests/prof_arena_e2e.cu:233` — `pipe.generate_txt2img(steps, 1234ULL, g, ...)` |
| ② | 被験コード行に到達するか | `src/infer/diffusion.cu:732` = `DiffusionPipeline::generate_txt2img` の定義。その内側 `src/infer/diffusion.cu:872` = `launch_unet_batched(...)` |
| ③ | 被験構成でその枝を通るか | `src/tests/prof_arena_e2e.cu:212-215` が `PROF_FAST=1` (既定 1) で `cfg.attn_fast/batch2/epilogue = true` → `src/infer/diffusion.cu:755` `const bool use_batch2 = fast_cfg_.batch2;` → `:864` `if (use_batch2)` → `:872`。**実走ログでも `[S4] config ... fast=1` を確認済** |

### 基準B: `test_conv2d` (G-10k が触るカーネル `launch_conv2d` のバイト列)

| # | 問い | 一次証拠 |
|---|---|---|
| ① | どの関数が呼ばれるか | `src/tests/test_conv2d.cu:657` — `launch_conv2d(d_in, d_weight, d_bias, d_out, N, ...)` |
| ② | 生成物ファイルが出るか | `src/tests/test_conv2d.cu:683` — `dump_bytes(label, runs[0].data(), nbytes)` / `:583-605` が `<prefix>_<label>.bin` へ `out_n * sizeof(__half)` バイトを書く。ゲートは `:589` の `DOLLAMA_G8K_DUMP` |
| ③ | 被験構成 (N=2) を通るか | `src/tests/test_conv2d.cu:696-703` が **N=1 が 4 形状・N=2 が 3 形状**を呼ぶ (`b2_320_128` / `b2_640_64` / `b2_1280_32` が N=2) |

### ★ PL 推奨との差分・および採らなかった選択肢

- **PL 推奨の `prof_arena_e2e` は「画像ファイルを書き出さない」。** ソースを読んで確認した:
  出力は `src/tests/prof_arena_e2e.cu:247` の `rgb_hash=0x...` = 同 `:85` の **FNV-1a 64bit を stdout へ
  出すだけ**で、`ofstream` は 1 箇所も無い (`grep` で確認)。
- **`generate_txt2img` を呼ぶ実行物は本ツリーに 3 つしか無い**が、いずれも
  「固定 seed **かつ** 生成物ファイル書き出し」を同時に満たさない:

  | 呼び出し元 | seed | 生成物ファイル |
  |---|---|---|
  | `src/tests/prof_arena_e2e.cu:233` | **固定 1234** | **無し** (stdout に FNV のみ) |
  | `src/tests/test_diffusion_batch2.cu:249`, `:460` | **固定 1234** (`:237`) | **無し** (`ofstream` 皆無) |
  | `src/server/diffusion_runner.cu:65` (= 出荷 CLI `dollama.exe --prompt --out`) | **時刻ベース**: `src/server/backend_image_generator.hpp:84` `const uint64_t seed = static_cast<uint64_t>(std::time(nullptr));` (同種: `src/server/txt2img_generator.hpp:126`)。`src/main.cpp:135-198` の引数解析に **`--seed` は存在しない**・seed 上書き env も存在しない (`grep DOLLAMA_SEED` 該当なし) | PNG を書く (`src/main.cpp:242`) |

  → **`src/` 無改変のままでは「`generate_txt2img` 由来の決定的な画像ファイル」は採れない。**
- そこで **2 層で採った**: 基準A で e2e の内容ダイジェスト (固定 seed・batch2 ON)、
  基準B で **実ファイルのバイト列 sha256** (G-10k が実際に書き換える `launch_conv2d` の出力・N=2 を含む)。
- `src/` に `--seed` / seed 上書き env を足せば基準A も画像ファイル化できるが、
  **T2a は `src/` 不可侵**のため実施していない。要否は PL 判断。

---

## 3. ファイル一覧

| ファイル | 何の走行か |
|---|---|
| `env_snapshot.log` | 環境スナップショット (hostname / nvidia-smi フル / nvcc / meson / git / exe sha256) |
| `e2e_run1.log` | **基準A 走行 #1** — `prof_arena_e2e.exe`・2 枚生成 |
| `e2e_run2.log` | **基準A 走行 #2** — 同一条件の再走 (決定性確認用) |
| `conv2d_run1.log` | **基準B 走行 #1** — `test_conv2d.exe` + `DOLLAMA_G8K_DUMP`・7 形状の `.bin` を生成し sha256 を記録 |
| `conv2d_run2.log` | **基準B 走行 #2** — 同一条件の再走 (決定性確認用) |
| `baseline_sha256.txt` | 基準 A / B の値を 1 枚にまとめた集約表 |
| `scripts/t2a_env.ps1` | 環境スナップショット採取スクリプト (研究機で実行したもの・実体) |
| `scripts/t2a_e2e.ps1` | 基準A 採取スクリプト |
| `scripts/t2a_conv2d.ps1` | 基準B 採取スクリプト |
| `scripts/t2a_finalize.ps1` | 集約 + 回収前 sha256 採取スクリプト |

`.bin` ブロブ本体 (計 122MB) は repo に入れず**研究機に保管**:
`E:\Develop\logs\g10k-t2a\bin_run1\` (走行 #2 の分は決定性確認後に削除済み)。

---

## 4. 基準値

### 基準A — e2e 生成物の内容ダイジェスト

**採取条件**

| 項目 | 値 |
|---|---|
| exe | `build/src/prof_arena_e2e.exe` (sha256 `CA11AF27…6FDB`) |
| env | `PROF_IMAGES=2 PROF_STEPS=20 PROF_G=7.5 PROF_FAST=1 PROF_SAMPLE_MS=5` |
| env (未設定を明示) | `DOLLAMA_PROFILE` = **未設定 (profile OFF)** / `DOLLAMA_POOL` = 未設定 (既定 ON) / `DOLLAMA_ARENA_*` = 未設定 |
| seed | **1234** (`src/tests/prof_arena_e2e.cu:233` ハードコード) |
| 解像度 / steps / guidance | **1024×1024 / 20 step / 7.5** |
| 構成 | `FastConfig{attn_fast=true, batch2=true, epilogue=true}` (= 出荷 `--fast` 相当) |
| 出力先 | **ファイル出力なし** (stdout の `rgb_hash`) |

**値**

```
rgb_hash = 0x8a96690109d2b253      (1024x1024)
```

★ この値は **FNV-1a 64bit** であって sha256 ではない (`src/tests/prof_arena_e2e.cu:85`)。
走行 #1 の image1 / image2、走行 #2 の image1 / image2 の **計 4 枚すべてこの 1 値**。

### 基準B — 生成物ファイルのバイト列 sha256

**採取条件**

| 項目 | 値 |
|---|---|
| exe | `build/src/test_conv2d.exe` (sha256 `C14BB9E8…FC9C`) |
| env | `DOLLAMA_G8K_DUMP=<dir>\conv` / `DOLLAMA_PROFILE` = **未設定** / `DOLLAMA_POOL` = 未設定 (既定 ON) |
| seed | 形状ごとにハードコード **3101〜3107** (`src/tests/test_conv2d.cu:696-703`) |
| 上書き | 走行ごとに別ディレクトリ (`bin_run1` / `bin_run2`)・走行前に `*.bin` を削除 |

**値** (dtype = `__half`)

| ファイル | N | 形状 | bytes | sha256 |
|---|---|---|---|---|
| `conv_b1_320_128.bin` | 1 | C320 128×128 | 10485760 | `A8E5F3E3653FE43AEA48F01A774C9635B10B2C8A41CE17239299C7369E671008` |
| `conv_b2_320_128.bin` | **2** | C320 128×128 | 20971520 | `2CA054357670F2C93F6538AE575BCA8BC65CB8779C7DE19B1E2396CDC36E9F08` |
| `conv_b1_640_64.bin` | 1 | C640 64×64 | 5242880 | `FCC32AE4D5555898019013CC2A37072A3678041A2B9CF941F883E76326ABA0AF` |
| `conv_b2_640_64.bin` | **2** | C640 64×64 | 10485760 | `F67AECD2AF7009CBF842B27D63F9680753EB7C584ECDDDAAA0F887E609E59EE4` |
| `conv_b1_1280_32.bin` | 1 | C1280 32×32 | 2621440 | `479C9E7EFEFCC45596A394E7ED2FA6C8BFA50BA3BDCE237DC759B24D3AD19572` |
| `conv_b2_1280_32.bin` | **2** | C1280 32×32 | 5242880 | `E154131B7002C12716E2912A2B5782F391F38F44AB483EBE6175BC9FB2C023E3` |
| `conv_b1_128_512.bin` | 1 | C128 512×512 | 67108864 | `30B7B87FE0D6EA373D9DA4BEA81ED83507894826110F48FBDCCD880268794231` |

### 2 回走行の一致について

- 基準A: 4 枚すべて `rgb_hash` 一致。基準B: 7 ファイルすべて sha256 一致。
- ★これは **基準そのものが決定的であることの確認 (determinism)** であって、
  **無改変の証明ではない**。無改変の判定は T2b 側で本基準と突合して行う。

---

## 5. 走行時間・観測値 (characterization のみ・判定ではない)

| 走行 | exit | 経過秒 | 備考 |
|---|---|---|---|
| `e2e_run1` | 0 | **58.5s** | うち生成 image1 `10.96s` / image2 `10.65s`。残りは重みロード等 |
| `e2e_run2` | 0 | **29.6s** | image1 `10.75s` / image2 `10.66s`。#1 より短いのは OS ファイルキャッシュが温まったため (重みロード分) |
| `conv2d_run1` | 0 | **4.0s** | |
| `conv2d_run2` | 0 | **4.0s** | |

- VRAM: `used_after_weight_load=13323MB` / **`PEAK_USED=13397MB`** (device-wide total−free・`total=16302MB`)。
  `used_after_destroy=1391MB`。★`PEAK_USED` は `cudaMemGetInfo` 由来で**他プロセス分を含む** (g8k-s4b README と同じ注意)。
- device_arena: 画像 1 枚あたり `d_cudaMalloc=0 d_cudaFree=0 d_chunkAlloc=0`
  (`arena=unet` cap 6080MiB / `arena=unet_persist` cap 176MiB)。
- 電力・クロック: 走行前 35℃/180MHz/11.7W → 走行直後 46℃/**2865MHz**/**173.95W** (cap 360W)。
- **`[ALLOC] reserve shortage:` は全ログで 0 件。** stderr は 4 走行すべて空。
  例外・`WinError 4551`・SAC ブロックも 0 件。
- `test_conv2d` の `[g8k_arena]` は 7 形状すべて `BIT-EXACT`。

---

## 6. 回収経路 (SSH 運用ドライラン)

研究機でファイルへ落とす → `scp` で開発機へ → 本ディレクトリへ実体を配置 → **両側 sha256 突合**、
までを実際に通した。

- 研究機側の原本: `E:\Develop\logs\g10k-t2a\` (**repo 外**に置き、研究機 repo を clean のまま保った)
- 回収コマンド: `scp -o BatchMode=yes rtx58:'E:/Develop/logs/g10k-t2a/*.log' .`
- **突合結果: 6/6 ファイルで両側 sha256 完全一致**

| sha256 (両側同一) | ファイル |
|---|---|
| `32F077A6EBBCB378238B3A611B5817838CCC256F0CCB88631C80648423D02E02` | `baseline_sha256.txt` |
| `318BC0AB5D8B3F0D28F2F7197093CDC9AC08D3B2A62B348C0350DD30F8AD094F` | `conv2d_run1.log` |
| `9D64717242074CCDEC68D3F22C3619607458C5AF1CEF61B3EA6C79C8FA464F47` | `conv2d_run2.log` |
| `057A26BF93AD86CE2B89F74C64A69A560BC97FD4EA291BA805659E2D627C1C67` | `e2e_run1.log` |
| `8D9A94A05C2BB011DA279318634D6D3DD6C4310994F128327102F71C41C193C3` | `e2e_run2.log` |
| `49386C7F656BB84065324F10C51BFF7A84732075BB4DEEF9CD1233A1AB6CC4D5` | `env_snapshot.log` |

### 再現するときの注意

- ★**リモートの `.ps1` は `pwsh` (PowerShell 7) で実行すること。** `powershell` (5.1) は
  BOM 無し UTF-8 のスクリプトを **CP932 と解釈**して日本語文字列を壊す (初回に実際に起きて撮り直した)。
- ssh 引数は**シングルクォート**で包む。バッククォートは bash / PowerShell 双方でエスケープ文字なので使わない。
- 接続のたびに post-quantum 警告 3 行が出るが無害。

---

# G-10k T2c 採取 追記 (2026-09-04・研究機 KIK-WIN-RTX58・SSH 越し)

★**本節は追記のみ。上の T2a 節は一切書き換えていない。**
★**採取と提出のみ。判定・合否・結論は書かない** (T2c のタスク定義)。

## T2c-1. 採取環境と HEAD

| 項目 | 値 |
|---|---|
| 採取日時 | 2026-09-04 23:00〜23:45 JST |
| hostname | `KIK-WIN-RTX58` / Windows 11 Home 10.0.26200 / pwsh 7.6.5 |
| GPU | RTX 5080 / 16303 MiB / driver 610.88 (CUDA UMD 13.3) |
| nvcc | release 13.3 V13.3.33 / meson 1.11.1 |
| Nsight Systems | **2026.1.3.243** (`C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.1.3\target-windows-x64\nsys.exe`) |
| 両機 HEAD | **`f58ba2c10649806c11c3c36f5f5bd314975cdbb6`** (`feat/g10k-conv-true-batch2`) で一致 |
| 研究機 `git status --porcelain` | **走行前・全走行後とも 0 行 (clean)** |
| ビルド | `meson compile -C build` = **`ninja: no work to do.`** (T2b 成果物をそのまま使用・本タスクで再ビルドなし) |
| SAC | ユーザー申告で OFF。本タスクでの実走はすべて exit 0。`WinError 4551` / アプリ制御ブロックは 0 件 |

## T2c-2. ★保存した実行物 (アンカー 3 の 3 段手順 2 段目に必須)

**保存先 (repo 外・セッション ID を含まない永続パス)**: `E:\Develop\logs\g10k-t2c\exe\`

| 絶対パス | bytes | sha256 |
|---|---|---|
| `E:\Develop\logs\g10k-t2c\exe\test_diffusion_batch2.exe` | 1401344 | `3649AF0B7968BE303B4D393AAD734AC6ECC7B89263426F322117108711A6CE43` |
| `E:\Develop\logs\g10k-t2c\exe\prof_arena_e2e.exe` | 1388544 | `9D420AB189EDA9224FB440703A8151485F33A11366CB816CFAE02400F9961140` |

- 両者とも `build\src\` の原本と **sha256 一致** (`t2c_stage_exe.log`)。
- ★**OpenVINO DLL 21 本を同ディレクトリへ同梱している** (`src/meson.build:1-7` で `deps += openvino_dep` =
  この 2 つの exe は `openvino.dll` 等にリンクしており、exe だけコピーしても別ディレクトリからは起動しない)。
  計 21 DLL / ディレクトリ合計 207,460,184 bytes。
- ★**保存先からの起動確認 (コピーしただけで済ませていない)**:
  - `test_diffusion_batch2.exe` — **T2c 本走行そのものを保存先から起動した** (`t2c_run_meta.log` の
    `exe = E:\Develop\logs\g10k-t2c\exe\test_diffusion_batch2.exe` / `exit_code = 0`)。
    したがって「走行に使った exe」と「保存した exe」は同一実体。
  - `prof_arena_e2e.exe` — 全計測の**後**に `PROF_IMAGES=0` で保存先から起動し `exit_code = 0`
    (`t2c_launchcheck.log` / `t2c_launchcheck_meta.log`)。画像は生成していないので計測を汚さない。

**nsys 生成物 (大きいので repo に入れず研究機に保管)**

| 絶対パス | bytes |
|---|---|
| `E:\Develop\logs\g10k-t2c\nsys\g10k_t2c_nsys.nsys-rep` | 28513058 |
| `E:\Develop\logs\g10k-t2c\nsys\g10k_t2c_nsys.sqlite` | 289697792 |
| `E:\Develop\logs\g10k-t2c\nsys\g10k_t2c_nsys2.nsys-rep` | 28167601 |
| `E:\Develop\logs\g10k-t2c\nsys\g10k_t2c_nsys2.sqlite` | 282767360 |

## T2c-3. 使った env の実値

**T2c 本走行** (`t2c_run_meta.log` に実出力あり)

```
DB2_BENCH=1  DB2_BENCH_ITERS=1  DB2_BENCH_STEPS=20  DOLLAMA_PROFILE=1
```
★**`DB2_BENCH_G` は設定していない** (= harness 既定 7.5。`test_diffusion_batch2.cu:559` の
`float bg = 7.5f;` がそのまま使われ、`:562` の上書きは env 不在で発火しない)。ログ側にも
`DB2_BENCH_G = <unset>` を印字して確認済。seed も harness 既定 (`:237` の固定 1234)。
`DB2_STEPS` / `DB2_UNCOND_ZERO` / `DOLLAMA_POOL` / `DOLLAMA_EPILOGUE` / `DOLLAMA_FAST` /
`DOLLAMA_CONV_BATCH` / `DOLLAMA_GEMM` / `DOLLAMA_ARENA_RELEASE` / `DOLLAMA_G8K_DUMP` は
**すべて `<unset>` を実確認**。**1 プロセス**・実行時間 **154.96s** / `exit_code = 0`。

**nsys 走行** (2 本とも同一 env)

```
PROF_IMAGES=2  PROF_STEPS=20  PROF_G=7.5  PROF_FAST=1
DOLLAMA_PROFILE / DOLLAMA_POOL / DOLLAMA_ARENA_RELEASE / PROF_SAMPLE_MS = すべて <unset>
```

## T2c-4. ファイル一覧 (両側 sha256 一致を確認済 = 21/21)

| ファイル | 内容 |
|---|---|
| `t2c_env.log` | 環境スナップショット + `meson compile` no work to do + exe sha256 + nsys 所在 |
| `t2c_stage_exe.log` | exe/DLL の永続パスへの退避と sha256 突合 |
| `t2c_run_meta.log` | **T2c 本走行のラッパ**: env 実値 / 経過秒 / 前後 nvidia-smi / reserve shortage sweep |
| `t2c_db2bench.log` | ★**T2c 本体の生ログ** (`test_diffusion_batch2` stdout+stderr マージ) |
| `t2c_gpu_sample.csv` | 本走行中の nvidia-smi 2s サンプル (53 サンプル) |
| `t2c_nsys_meta.log` / `t2c_nsys2_meta.log` | nsys 走行 #1 / #2 のラッパ |
| `t2c_nsys_stats.log` / `t2c_nsys2_stats.log` | `nsys profile --stats=true` の stdout |
| `t2c_nsys_cuda_gpu_kern_sum.csv` / `t2c_nsys2_cuda_gpu_kern_sum.csv` | カーネル別 GPU 時間 (CSV 再出力) |
| `t2c_nsys_cuda_api_sum.csv` / `t2c_nsys_cuda_gpu_mem_time_sum.csv` | CUDA API / memcpy 集計 |
| `t2c_nsys_timeline.log` | 全体タイムライン解析 (総カーネル時間・クラスタ・grid dim) |
| `t2c_nsys_gen_split.log` / `t2c_nsys2_gen_split.log` | 生成 1 枚ごとの wall / Σkernel / launch 谷 |
| `t2c_nsys2_unet_vae_split.log` | 生成 #2 を UNet 20step と VAE decode に分割 |
| `t2c_nsys_target_stdout.log` / `t2c_nsys2_target_stdout.log` | ★nsys が飲み込んだ**被計測プロセスの stdout** を `ProcessStreams` から復元したもの |
| `t2c_nsys2_stats_csv.log` / `t2c_artifacts.txt` / `t2c_sha256.txt` | 付帯 |
| `scripts/t2c_*.ps1` / `scripts/t2c_*.py` | 研究機で実際に実行したスクリプト実体 |

## T2c-5. nsys 運用で踏んだ落とし穴 (再現時の注意)

1. ★**`nsys profile --stats=true` の stdout から `cuda_gpu_kern_sum` の表が丸ごと落ちる。**
   直前に `[libprotobuf ERROR] String field 'Agent.StatsReportExecutionInfo.output' contains
   invalid UTF-8 data` が出て、その report の出力が stdout に届かない (`osrt_sum` も同様に空)。
   → **`nsys stats --report cuda_gpu_kern_sum --format csv -o <base> <.sqlite>` で採り直す**
   (post-processing のみ・GPU 再走不要)。
2. ★**被計測プロセスの stdout がログに出ない。** `--show-output=true` (`-w true`) を付けても
   リダイレクト先には流れない (走行 #2 で実測)。nsys は `.sqlite` の **`ProcessStreams` テーブル**に
   格納しているので、そこから取り出す (`scripts/t2c_target_stdout.py`)。
3. `Get-ChildItem -Include` は `-Recurse` かパス末尾ワイルドカードが無いと何も返さない
   (`t2c_finalize.ps1` で空表になり `t2c_finalize2.ps1` で採り直した)。
4. ssh のリモート inline コマンドに `$var` や `|` を素で書くと**リモート側シェルに食われる**。
   スクリプトファイルを `scp` して `pwsh -NoProfile -File` で叩くのが安全。

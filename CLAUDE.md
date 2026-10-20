# dollama — Claude 向けプロジェクトコンテキスト

## プロジェクト概要

**芯**: CPU / NPU / iGPU / RTX5080 — 搭載する全 HW を使い切りながら、
2D イラスト/漫画の**キャラクター描画**生成にたどり着くことを研究する。最短実装ではなく、
各 HW をどう活かし、どう協調させるかがこのプロジェクトの本質。

**スコープ (確定)**: 生成対象は**キャラクターのみ**。背景は生成しない
(Grok / Gemini / Stable Diffusion + CLIP Studio Paint で合成)。出力は
**切り抜き済み透過 PNG**。マッティング (α 抽出) も HW 協調パイプラインの一段とする。
キャラ設定の管理構造は `docs/character-bible-spec.md` 参照 (同一性層/シーン層/出力層)。

**パイプライン — 実装済み (出荷 txt2img = CLI / HTTP `/v1/images/generations`・2026-10-20 現物)**
```
prompt (直入力・LM 段なし) ── preset の prompt_prefix を自動前置 (2-6e)
  ▼
NPU: CLIP-L + CLIP-G dual encoder (失敗時 CPU)     src/server/cli_generate.hpp:311-312,320,331
  ▼
RTX5080: SDXL UNet ×20step CFG (既定 = B=1 を cond/uncond 2 回 / --fast = B=2) + VAE decode
  ▼ (生 rgb。採点と matting は同じ rgb を入力とし互いに依存しない)
NPU: ScorerNet 採点 (matting 前の生 rgb・結果はログのみ)   src/server/backend_image_generator.hpp:212
  ▼
iGPU (GPU.0): ISNet-anime matting → 透過 PNG               同 :230
```
- 実行順の根拠は `backend_image_generator.hpp` (上記行)。`cli_generate.hpp` の :383 (matting) / :412 (scorer) は**注入順**で実行順ではない。
- CPU: OV tokenizer (CLIP-L/G) が毎リクエスト CPU で動く (`src/infer/clip_tokenizer.hpp` 既定 "CPU"・`text_conditioner.hpp` で device 指定なし構築)。encoder 構築失敗時も CPU へ退避。
- backend 構築失敗時は golden 埋め込み PipelineGenerator → StubGenerator へ退避 (`cli_generate.hpp`)。
- 速度: **目標 3.80s = probe10 (PyTorch/diffusers) の参照値**。自作の現行値は計測表「統合パイプライン」。

**パイプライン — 構想 (未実装・設計は `docs/pipeline-spec.md`)**
```
自然文 → 自作タグ LM (bitnet 33M・GPU 主/CPU 代替/NPU 不可・GPU seq8 2.43ms) → CLIP   ★未結線
生成画像 → CPU: WD14 (105.3ms) → タグ → LM へフィードバック   ★出荷経路から未呼出 (src/pipeline.hpp 骨格のみ)
入力画像 → iGPU: VAE encode (probe5 79ms) → UNet (img2img)     ★VAE encoder 未実装・/v1/images/edits は 501
```

**デバイス選定根拠 (probe 実測)**
- CLIP: NPU 7.85ms < iGPU 14ms < CPU 20ms → **NPU 採用** (probe9)
- WD14: CPU 101ms < iGPU 104ms < NPU 268ms → **CPU 採用** (Window Attention が NPU に不向き・probe8)
- VAE decode: CPU 126ms << iGPU 995ms → **RTX5080 採用** (probe4)
- VAE encode: iGPU 79ms < CPU 117ms → **iGPU 採用** (img2img のみ・未実装・probe5) / matting: iGPU 99.96ms 最速 (M-5)

**VRAM 収支 (RTX5080 = 16GB)**: 自作出荷経路の peak = **13649MB (POOL 既定 ON) / 13269MB (`DOLLAMA_POOL=0`)**
(G-8k S4b・`--fast` 構成 (S4b は全走 fast=1)・判定は同一セッションの POOL=0 比 delta)・2026-09-04 走行 `PEAK_USED=13397MB` (`docs/logs/g10k-baseline/e2e_run1.log`)。
S6 でも 13397MB が出ており、S4b との 252MB 差は**未説明** (measurements-log G-8k S6 行)。
CLIP-L/G は NPU なので VRAM を食わず、自作 LM 33M は ~66MB。probe10 (PyTorch) の 10.49GB は参考値。
多 HW に逃がす理由は VRAM 不足ではなく速度・並列と研究そのもの (旧論考は measurements-log 第 2 次退避節 (b)・roadmap「MoE × HW 分散配置」)。

## 環境 (2 台運用・OS は両機 Windows 11)

| | 研究機 `KIK-WIN-RTX58` (mDNS 可) | 開発機 (本 repo) |
|---|---|---|
| HW | RTX 5080 16GB (sm_120) / Core Ultra 9 285 の NPU (AI Boost 3720) | GTX 1080 Ti / i7-10700 = **NPU なし** |
| CUDA | Toolkit 13.3 (`meson.build` 既定・`docs/logs/g10k-baseline/env_snapshot.log`) | **未導入** (`.cu` はビルド不可) |
| Python | 3.14。PyTorch は **cu128 ビルド必須** (Blackwell = CUDA 12.8 以上・研究機側の要件) | 3.14 既定・torch は `py -3.12` (2.5.1+cu121)・OpenVINO 2026.2.1・.NET SDK 10 |
| 担当 | 実重み・GPU/NPU/iGPU の実走 (`.cu` ビルド・meson test・e2e)。**commit も push もしない** = 開発機の commit を同期してビルド・実走のみ・作業ツリー clean を維持 (`docs/g10k-plan.md` §10) | Phase 4 訓練・UI・純 C++・記録。**push はここから** (SSH エイリアス `github-private`) |

拡散重み: 既定 preset = `illustrious-xl`・素 SDXL は `--preset base` 明示。

## 確定済みアーキテクチャ決定

- OpenVINO のデバイス名 (BIOS で iGPU 有効時): **`GPU.0` = Intel Xe iGPU / `GPU.1` = RTX5080** (probe4)
- **GPU↔NPU/iGPU のゼロコピーは不可 → CPU pinned memory 経由** (転送 3.4%・マルチスレッドで隠蔽)。OV NPU に CUDA interop が無く、
  D3D12 クロスアダプターも iGPU が DXGI に出ず不可。NPU↔iGPU はシステム RAM 共有でゼロコピー (差 0.158ms)。全ルートと probe4 詳細は `docs/archives/investigation-log.md`。
- **iGPU は大規模 Conv に不向き** (probe4 で CPU の 8 倍遅い)。**NPU は固定形状・encoder-only 向き**で、LLM 自己回帰は不適 (KV-cache で形状が動的)。
- **スレッド骨格** = `src/pipeline.hpp` (`std::jthread` + SPSC キュー)。CPU アフィニティは自己ピン留め型 (`src/core/affinity.hpp`・MinGW 対応の経緯は `docs/cpu-topology.md`)。

### 落とし穴 (既知事項)

| 事項 | 内容 |
|---|---|
| OV 入力要素型 | IR の `element_type` に厳密一致。CLIP-L `input_ids` は i64 `[1,77]` — i32 で渡すと NPU で **0xC0000409**。CLIP 出力 0 = `last_hidden_state` / 1 = `pooler_output` (`src/infer/clip_encoder2.hpp` の「タスク5」は旧採番) |
| NPU 静的形状 | `convert_model` は動的形状を出す → compile 前に `ov_model.reshape([batch, seq_len])` 必須 |
| OV Python API | `openvino.runtime` は廃止 → `import openvino as ov` / `ov.convert_model` / `core.compile_model(m, "NPU")` |
| `.cu` | 文字列リテラルは **ASCII 必須** (nvcc が CP932 解釈・コメントは日本語可)。テストの cuda_args `/utf-8` + `/std:c++14` は必須 (`docs/testing.md`) |

### 実装方針

**自作は HW を叩く研究コアに限定し、配管 (HTTP/JSON/Base64) は定番のヘッダオンリーライブラリを使う。**
配管を自作してもバグ表面と保守コストが増えるだけで研究価値がない。単一バイナリの美学はヘッダオンリー採用で維持する。

| 使う | 使わない |
|---|---|
| STL / CUDA Runtime API / **cuBLAS** (GemmEx = 既定の高速 GEMM 経路・`src/kernels/gemm.cu`) | PyTorch / LibTorch (C++ 本体) |
| 自作 Tensor / GEMM / Attention / CUDA カーネル / BitNet | diffusers / stable-diffusion.cpp / llama.cpp |
| **OpenVINO ランタイム** (C++ 本体の NPU/iGPU 推論グルー: CLIP・WD14・matting・scorer) | Drogon 等 **重量級** HTTP フレームワーク |
| **cpp-httplib** / **nlohmann/json** / Base64 小物 | 手書き Winsock2 / JSON パーサ |

- `ui/` = Blazor Server (.NET 10)。HTTP で dollama サーバーを叩く別プロセス。Python (PyTorch/diffusers/OV) は `scripts/` の probe・変換・訓練のみ。

### LLM の将来像

- Qwen2-1.5B は **Python probe 専用** (C++ 非搭載)。出荷 txt2img は `req.prompt` を素で dual encoder に渡す = **LM 段は未結線**。
- 本線は自作 **小型タグ生成 LM** (`src/models/bitnet.hpp`・33M)。置き場は **GPU 主 / CPU 代替 / NPU 不可**。ternary は圧縮の研究軸
  (`src/kernels/ternary_gemm.cu` は**未作成**)。独自性 = ① 同一性条件付き (A) ② 品質スコアラ (B)。経緯は roadmap「Phase 4」冒頭。

## 計測ベースライン

> 芯の数値と test 名だけ。経緯・条件・教訓は `docs/measurements-log.md`。「退避節「〜」」= 同 doc
> 「CLAUDE.md 計測表セルの全文退避」節の同名小節 (経緯と禁止事項の正本)。2026-10-20 改訂前の全文は `git show b95f0f4:CLAUDE.md`。

**計器・分母の禁止事項**

| 禁止 | 正本 |
|---|---|
| 陳腐化値 (VAE decode 7.96s / UNet 1step ~9.2s / CFG e2e 20.93・17.59・15.88s / G-4k S3 e2e x1.33439 / 2-6a 11.30s) を見積り・合否の分母に使う | measurements-log G-8k S5 残債④ / 退避節 |
| 2-6a 11.30s (非 CFG・B=1) や probe10 3.80s (PyTorch) を出荷経路の自作値として引く | measurements-log 2-6a 行 |
| batch2 構成の resnet を B=1 計器 `prof_unet_fast_warm` で測る | measurements-log G-10k 節 規律 1 |
| g>1 (CFG 増幅下) で FP16 微差をゲートする (G-4k S3 初版 g=7.5 は正常でも 0.996533 で FAIL)。被験変数は g=1.0 で分離 | 退避節「epilogue 出荷結線」 |
| VRAM を絶対値で判定する (POOL=0 比 delta で判定) / 未発火の F4・F2 異常系を「実走で確認した」と書く | measurements-log「G-8k S6 (T2)」節 |
| 推定量を書かずに「クラスタ補正で有意でない」と引用する (cluster-mean t=0.955 と CR sandwich t=2.73 で逆) | measurements-log F-0b 行 |
| VAE SSIM の変化点を `b3fe139` 以外に帰属させる (取り違えの是正が 2 回続いた) | measurements-log G-8k S5b〜S5e 残債① |
| 「G-8k は秒数レバーではない」を「実測で秒が動かなかった」と読む | `docs/fast-mode-plan.md`「用語の解消」 |

**HW 疎通 (probe2–10・デバイス選定の 4 probe は冒頭)**

| 指標 | 値 | probe |
|---|---|---|
| CPU→VRAM (10MB / 100MB) | 0.76ms / 3.46ms (30.3 GB/s) | probe2 |
| NPU推論 (512dim MLP) / 出力→GPU | 0.88ms / 0.031ms (3.4%) | probe2 |
| Qwen2-1.5B INT4 CPU (**本実装非依存**) | 64-71 tok/s / ロード 1.1s | probe7 |
| SDXL 20step 1024² (PyTorch/diffusers・base・CFG・テキストエンコード/VAE 込) | **3.80s** / 5.3 it/s / VRAM 10.49GB = **目標値** | probe10 |

**C++ 実装の確定値 (test_*)**

| 指標 | 値 | test |
|---|---|---|
| CLIP-L NPU / WD14 CPU | **7.82ms** / **105.3ms** (中央値/N=100) | test_clip / test_wd14 |
| Pipeline 縦通し / マルチフレーム並列 | **9.13 frames/s** (WD14 律速) / **0.258 fps** = GPU 上限 98% (look-ahead 2 で飽和・Tier2 は単一 GPU で非発動) | test_pipeline / test_multi_frame_pipeline |
| 自作カーネル単体 | GEMM 1024³ 4730 GFLOPS / SDXL Linear 4208 GFLOPS / SiLU・GeLU 544 GB/s / GN 73 GB/s / Conv2d direct 1807 GFLOPS / Attention self 1631・cross 1748 GFLOPS / LayerNorm 0.043ms | test_gemm/activation/groupnorm/conv2d/attention/layernorm |
| その他単体 | Euler sigmas max_err 4.77e-6 / safetensors 19.0 µs / HTTP 往復 2.11ms / PNG tEXt embed 1961・read 4376 ns / CharacterBible 242 ns | test_scheduler/safetensors/http/png_meta/character |
| 自作 VAE decoder 全段 | SSIM **0.999988** (変化点 `b3fe139` 単独) / decode 秒 = **現行値なし** | test_vae_decode |
| 自作 SDXL UNet 全段 | noise_pred SSIM **0.999998** (2-5 当時)・2-6 後 **0.999996** / 1step 秒 = **現行値なし** / 2.57B params | test_unet |
| UNet バッチ (G-2k S2) | B=2 per-sample MAE **6.4e-05**/bad0 (FP16 tol)・conv2d S1 は per-n で **BIT-EXACT** | test_unet/conv2d |
| CFG batch2 パリティ+速度 (G-2k S3c 完了) | parity g=1.0 SSIM **0.9994** PASS (g=1/3/7.5 = 0.9995/0.9988/0.9966) / e2e 秒は**陳腐化** | test_diffusion_batch2 |
| UNet fast warm (G-3kf 完了) | fast vs default bit-exact・warm 1step 526→437ms = **1.20x** (G-3kf 当時) | test_unet_fast |
| GroupNorm multi-block (G-4k S1a 完了) / +SiLU 融合 (S1b) | 帯域 **444.6 GB/s** (4.2x) / S1b 5 形状 **BIT-EXACT**・ともに epilogue vs default SSIM **0.999999** | test_groupnorm/test_unet_fast |
| conv2d 後段融合 (G-4k S2 完了) | 全 6 形状 **BIT-EXACT** / **resnet ≤0.95s ゲートは未達** (1.226→1.212s = ノイズ床内・G-10k 後も未達) | test_bias_add/test_unet_fast |
| epilogue 出荷結線 (G-4k S3 完了 = G-4k クローズ) | 出荷 `--fast` (既定 OFF) = attn_fast+batch2+epilogue。GATE1/3 bit-exact・GATE2 **0.999474**・GATE4 **0.999477** 全 PASS / e2e x1.33439 は**陳腐化** | test_diffusion_batch2/test_unet_fast |
| device_arena e2e VRAM 収支 (G-8k S1〜S4b 完了 = G-8k クローズ / S6 レビュー是正済) | S4 の G4 **FAIL** (peak 16302MB) → S3b/S3c で解消・S4b **6 ゲート全 PASS** (peak 13649MB ≤ POOL=0 13269+512MB・step 内 cudaMalloc 0・出力 bit 一致)。S6: 53/53 緑・出力 sha256 全一致・delta **+340MB** / 定常 residency +6.5GB | test_device_arena/test_http |
| conv2d 真 batch2 (G-10k・陰性クローズ 2026-09-17) | 数値ゲート全緑だが resnet 削減率 **+3.38% / +1.87% (悪化)** → opt-in `DOLLAMA_CONV_BATCH=1` に降格 (`c3ee3ca`) | test_conv2d/test_diffusion_batch2 |
| ランタイム LoRA (L-2 完了) | 全 4 ゲート PASS (parity **4.9e-4**・revert bit-exact・stack **9.6e-4**)・HTTP `loras` 結線。後日の ODR 違反 (0xC0000005) は C0 で解消 | test_lora_runtime |

**Phase 4 自作 LM / Model B (詳細は measurements-log.md)**

| 指標 | 値 | test |
|---|---|---|
| モデル定義 / トークナイザー | 32.98M params / encode 365・decode 168 ns・UNK 0 | test_bitnet / test_tokenizer |
| dense LM 訓練 #1 (hard CE 6ep) | val_loss 2.41 / top10 recall 0.777 | (train_bitnet.py) |
| 推論 CPU / GPU / AVX2 / INT8 | corr 1.0 / GPU 46.8–87.5x / AVX2 ~5x / INT8 74.84%減 corr 0.9999 | test_bitnet_infer/_gpu/_int8 |
| 主指標・施策 | 4-C で diverse 生成 set-F1 を主指標化 / 4-A retention **0.975** (頑健) / 4-B ~2,000 件飽和 / 蒸留 D2〜D6 不採用 / 4-D 80M 陰性 (勝者 33M) | test_dollma_eval_diverse 他 |
| 正典化 `[B-merge-at-A]` | ゲート 4 指標全通過 (retention 0.9807 / diverse_a 0.3332 / diverse_b 0.3804 / in-dist 0.4552) | test_bitnet_infer |
| マッティング (M-5 / M-6) | iGPU **99.96ms** 採用 / `IMatter` で透過 PNG 結線・研究機 e2e 実走済 | test_matting |
| ScorerNet (B-3b〜B-5) | 11.18M・NPU 8.32ms・OV err 1.34e-5 / 生成器結線 (採点はログのみ) | test_dollma_train_scorer / test_scoring_postprocess |
| 品質 FB ループ F-0a 信号ゲート (80/80) → Q-2 | reward std **0.0377** (信号弱・argmax Limbs 77/80) → Q-2 quality 枝 0.0797 (weight 0.3) → **0.1038** (weight 0.4・ゲート通過・`docs/q2-quality-branch-plan.md`) | (dollma_collect_rollouts) |
| F-0b RAFT-SFT (クローズ・不採用) | **効果を検出できず** (正典無改変): reward Δ +0.017 は有意でない / set-F1 −0.0174/−0.0241 (単一 seed)。Z-4 = ② 選抜方針起因 (射程限定・不採用不変) | (dollma_g2b_reward_prepost) |

**統合パイプライン**

| 指標 | 値 | test |
|---|---|---|
| **自作 出荷経路 e2e (現行値)** | `--fast` (CFG g=7.5・B=2・20step 1024²): **10.65〜10.96 s/枚** (UNet+VAE・テキストエンコード/重みロード除く・**2026-09-04 の計装 OFF 実測**・HEAD `e17374c`・`docs/logs/g10k-baseline/e2e_run{1,2}.log`) / 既定 (非 fast): 計装 OFF の現行値なし。参考 = 計装 ON (`DOLLAMA_PROFILE=1`・test_diffusion_batch2 DB2_BENCH 20step): 2026-09-16 `CONV_BATCH=0` (現既定と同じ per-n 経路・HEAD `c0f6d8f`) default 14066.9 / fast 10799.8ms (`docs/logs/g10k-e2e/t7_p2_convbatch0.log`) / 2026-09-09 default 14053.4 / fast 10815ms (`docs/logs/g10k-baseline/t2c_0909_db2bench_profile.log`)。いずれも E-0 (scaling パラメータ化) 前 | prof_arena_e2e / test_diffusion_batch2 |
| 自作フル拡散 (2-6a) | 84.07s → 11.30s (**非 CFG・B=1・2026-06-22・陳腐化**) | test_diffusion |
| 本番 txt2img (2-6b) / backend 枠 (2-6c) | dual encoder+CFG・NaN/Inf なし / `IDiffusionBackend` registry + `BackendImageGenerator` | test_txt2img / test_diffusion_backend |
| アニメ特化 SDXL 3 preset (2-6d) | **既定 illustrious-xl**。照合済 (UNet 1680 キー名+shape/dtype・tokenizer sha256 base 一致・TE 誤差 1e-5 台)。12 枚比較 (seed 1234・N=1・**VAE scaling 0.13025 時代**) の目視順位 illustrious > noobai > animagine。2-6e prefix 自動付与 ✅。VAE scaling: E-0 (`272e854`・ユーザー決裁) で illustrious-xl の preset.json 値 0.18215 が効くようになり既定 preset の生成値が変わった (優先 env > preset.json > 既定)。**サーバーの既定定数は 0.13025 のまま** (`src/server/vae_scaling_default.hpp`)・どちらが正しいかの論証は無い・export ガードは 0.13025 固定のまま未是正 (`scripts/dollma_export_sdxl_preset.py` l.49 / 333-337)。詳細は measurements-log「2-6d」節 残債① | test_preset |

## 次のタスク (現在地・2026-10-20)

> 完了済みの段は `docs/roadmap.md` の各 Phase 節と上の計測表。状態の正本は各台帳。ただし `docs/image-quality-plan.md` 項目一覧の
> E-2.5「未 commit」・E-4「未着手」は陳腐化しており、下の記述 (2026-10-20 に git で確認) のほうが新しい。

- **速度 — G-14k (attention FA2・`docs/g14k-plan.md`)**: S0 実施中・**S1〜S4 未着手** (2026-09-20 合流時点・以後 commit なし)。resnet ≤0.95s は未達。
- **画質 — E 系 (`docs/image-quality-plan.md`)**: E-0 ✅ `2d442b3` / E-1 merge 済 (DoD 判定未実施) / E-2 ✅ `4ad8f87` / E-2.5 T1 merge 済 `2fa0c95` (レビュー記録なし・判定なし) /
  E-3・E-5 は発火条件充足 (**着手決裁ではない**) / **E-4 = 本 branch に WIP** (`3d2de2c`・`c882c22`・Illustrious v2.0 変換と Anima probe・記録/レビュー未実施)。
  2-6 残件: HTTP API の preset フィールド / UI の preset 選択 (未着手)。
- **Phase 4 (`docs/f0b-reverification-plan.md`)**: F-0b 再検証の残 = GPU 実走 (σ_seed 算出済・未起票) / anatomy 死軸の分解能診断 / 次レバー (reward 設計・日本語条件付け・SDXL seed 制御) / 自作 LM 段の結線。
- バックログ: 部位構造化プロンプト (**未起票**)。

台帳索引: `docs/roadmap.md` (全体) / `docs/fast-mode-plan.md` (G-0〜G-8k) / `docs/g10k-plan.md` (G-10k・陰性) / `docs/g14k-plan.md` (G-14k) /
`docs/image-quality-plan.md` (E-0〜E-5) / `docs/f0b-reverification-plan.md` (Z-1〜Z-5) / `docs/q2-quality-branch-plan.md` (Q-2) / `docs/measurements-log.md` (計測全文)

## 実装作業のルール

1. **プランモードで設計を提示し、ユーザーのレビューと承認を得てから着手する。** 承認なしに勝手にコードを書き始めない。
2. **承認後、複数段の作業 / 研究機での実走 / `src/` を変更する作業は `project-leader` を呼び、担当エージェントへ振り分けてもらう。** 単発の docs 修正・chore は直接行ってよい。委譲の段数・並列数は必要最小限にする。
3. **ゴールが設定された場合、各段の承認は `project-leader` が行ってよい** (ユーザーへの判断依頼は PL が迷ったときのみ)。ただし **① 既定値の変更** (既定 preset・既定ノブ・VAE scaling 等、出荷画像のビットが変わるもの) **② 正典の差し替え** (正典重み・golden) **③ push ④ ライセンス判断** はユーザー決裁を要し、PL も承認できない。エージェントは実数値と推奨の提示までに留め、**中継された承認を根拠にしない** (`docs/image-quality-plan.md` E-0 DoD 3 と同旨)。
4. **記録 (docs / CLAUDE.md 計測表 / commit 本文) の執筆・是正は `record-writer`、その検査は `record-auditor` に分ける。** 同一主体に書かせて検査させない (G-8k S5 で是正が新しい誤りを持ち込む形が 5 ラウンド連続で起き、毎回この分離が検出した)。
5. **コンポーネントを実装したら、必ずテストも実装し、緑を確認してから完了とする。** C++/`.cu` = `src/tests/test_<component>.*` + `meson test -C build` / C# = `ui.Tests/` + `dotnet test ui.Tests/Dollama.Ui.Tests.csproj` / Python = `scripts/test_dollma_*.py`・`scripts/tests/` (`python -m pytest <file> -q`・torch 依存のものは `py -3.12 -m pytest`)。規約は `docs/testing.md`。

## コーディング規約

- `scripts/` の Python は `dollma_` プレフィックス (テストは `test_dollma_*`・既存の例外 `train_bitnet.py` / `train_scorer.py`)。プローブは `scripts/dollma_probe*.py`
- 本実装は `src/` 以下に C++ で記述。ビルド: Meson (`meson setup build && meson compile -C build`)
- コメントは日本語で書く

### C++ スタイル

開き波括弧 `{` は必ず改行して次の行に置く (Allman スタイル):

```cpp
void abc()
{
    // ...
}
```

`switch` 文の `case` ラベルは `switch` と同じタブ位置に揃える:

```cpp
switch (x)
{
case 1:
    break;
case 2:
    break;
}

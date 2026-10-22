# Anima 自作実装トラック (A 系) 台帳

> 本書は Anima トラックの状態の正本。数値はすべて研究機の生ログ (`docs/logs/anima-a0/`・`docs/logs/anima-a1/`) から転記した。
> ログに無い解釈は「推測」と明記する。計画本文 (承認済) は開発機の plan ファイル (`~/.claude/plans/sequential-exploring-haven.md`
> の「Anima A2」節・「A0/A1 完了・参照用」節) だが**版管理外**なので、**計画のうち本書に転記した範囲が正本**。
> 新設 2026-10-23 (A2-3 の前半・record-writer)。

## 1. 目的と位置づけ

- **2026-10-22 ユーザー決裁**: Anima を**研究トラックとして採用**。**出荷既定は SDXL (`illustrious-xl`) のまま**
  (`DOLLAMA_BACKEND` 既定 "sdxl" 不変・既定の切替は別決裁)。Anima は明示選択時のみ動かす計画。
- **採用の根拠はユーザーの目視** (E-4 の contact sheet)。E-4 の定量値 (soft recall 等) ではない
  (`docs/image-quality-plan.md` E-4 節「Anima (characterization のみ・判定なし)」と同旨。seed 間 sd が SDXL 系より小さく、
  平均の高さを画質の優位と読まない)。
- Anima は SDXL ではない (Cosmos-Predict2 2B ベースの DiT + Qwen3 0.6B TE + Qwen-Image VAE + llm_adapter)。
  自作 C++/CUDA/OV で新規実装し、芯 (全 HW 協調) に沿って **tokenizer = CPU / TE = NPU / adapter・DiT・VAE = RTX5080 /
  matting = iGPU** を目指す (置き場は案。A5 で確定)。
- ライセンス: CircleStone Labs Non-Commercial License v1.2 + NVIDIA Open Model License。個人研究で研究利用可
  (`THIRD_PARTY_NOTICES.md` §2 の Anima 行)。

## 2. 段一覧 (状態は 2026-10-23 時点)

| 段 | 内容 | 状態 |
|---|---|---|
| A0 | 参照 golden ダンプ (FP32) + 活性値表 + VAE conv3d→conv2d 検証 | ✅ 完了 (研究機実走 2026-10-22/23・スクリプト `b605f2a`・ログ `30a7a61`) |
| A1 | Qwen3 TE の OV IR 変換 + NPU/iGPU/CPU ベンチ + tokenizer 変換 | 変換・ベンチは完了 (同上)。**C++ グルー (`qwen3_encoder.hpp`)・test_anima_te は未着手** (計画上は A1 に含む)。グルーは A2 計画に含まれず未割当 (回収段は A3 以降の計画再提示時に決める) |
| A2 | DiT 部品カーネル (A2-0 → A0' ∥ A2-1 → A2-2 → A2-3) | **A2 計画 = 2026-10-23 ユーザー承認** (依頼者申告・承認の一次記録は repo 外)。研究機未実走・結果なし。本書 = A2-3 の前半 |
| A3 | DiT 全段 + adapter + flow sampler (`anima_dit.cu`・`flow_scheduler.hpp`) | 未着手。**A3 以降は計画未承認** (A2-2 の結果を見て再提示する想定) |
| A4 | VAE decoder (`anima_vae.cu`・conv2d 畳み込み版) | 未着手 |
| A5 | backend 結線 (`AnimaBackend`・`DOLLAMA_BACKEND=anima` / `--preset anima-base` のみ・既定 SDXL 不変) + 研究機 e2e | 未着手 |
| A6 | 記録 (roadmap・CLAUDE.md 計測表) | 未着手 (本書の新設は A2-3 側で前倒し) |

- A2 のコード (A2-0 / A0' / A2-1) は本書とは別の commit で入る。研究機での実走結果が出るまで本書の結果欄は空欄とする。

## 3. A0 結果 — 参照 golden (研究機・2026-10-22/23)

**条件** (`anima_golden_meta.json`): `anima-base-v1.0`・prompt p2 (E-1 の「手・小物」)・seed 1234・1024²
(latent `[1,16,1,128,128]`)・30 step・Euler (CONST / rectified flow・split CFG)・sigma schedule simple・shift 3.0・CFG 4.0・
**FP32** (`fp32_precision=ieee`)・torch 2.11.0+cu128 / diffusers 0.38.0 / transformers 5.0.0・token 数 pos qwen 41 / t5 40・neg qwen 37 / t5 42。
★E-4 (er_sde・seed 1000-1003・bf16) とは画像を直接比較できない (meta の `comparability_note`)。
golden 本体 (`models/anima/golden/anima_io.safetensors`・613.8MB・166 tensor) は研究機の gitignore 下。

### 3.1 ゲート (FAIL で exit 1)

| ゲート | 値 (CUDA FP32 30step・`full1.log`) | 値 (CPU 1step・`cpu1.log`) | 閾値 |
|---|---|---|---|
| sigma_selfcheck (自前 simple sigmas vs diffusers_anima) | 0.0 | 0.0 | 1e-6 |
| vae_conv2d_max_diff (causal conv3d → conv2d 置換 decode) | 0.0 | 3.25e-5 | 1e-4 |
| vae_time_conv_calls (T=1 decode での time_conv 呼出) | 0 | 0 | 0 |
| b0_ctx_equals_pos_adapter_out | True | True | True |
| denoised_step0_vs_diffusers_anima (手書き vs `_predict_denoised_const`) | 0.0 | 0.0 | 1e-5 |

全 PASS (CUDA は `full1`・`full2`・`full3_vram` の 3 走とも全 PASS)。VAE conv2d 版の u8 不一致率は CUDA 0.0 / CPU 9.85e-6。

### 3.2 決定性

- **同一プロセス 2 回**: BIT-EXACT (`full1.log`「決定性: BIT-EXACT」・meta `determinism.mode = same-process-2runs`)。
- **プロセス跨ぎ**: `golden` (full1) vs `golden_b` (full2) で **166/166 tensor BIT-EXACT** (`compare.log`)。
- 注: `anima_io.safetensors` の**ファイル** sha256 は 3 走で異なる (`a8ba837b…` / `32158d80…` / `5d8067f8…`) が、
  tensor 比較は一致している。ファイル差の原因はログに無い (safetensors ヘッダの metadata 差と**推測**・未確認)。
  `golden_c` (full3) は tensor 比較をしていない。

### 3.3 VRAM・時間 (参考)

- VRAM: **used ピーク 11653 MiB** (`full3_vram.log` 走行・`vram.txt`。他プロセス分 ~526 MiB を含む GPU 全体の used と読める記載だが、
  採取方法はログに書かれていない)。FP32・hook 付きダンプの値であり、自作実装の VRAM 見積りには使わない。
- 時間 (`full1`・`timing_sec`): ロード 212.5s (2 回目以降の走行は 9.4〜12.4s)・step 1 = 24.4s / step 2〜29 = 1.59〜1.60s / step 30 = 20.5s・
  VAE decode 6.8s (元) / 0.43s (conv2d 版)。CPU 1step は 71.7s。
  step 1 と 30 が長いのは、活性値キーが `step0_*` / `last_*` を持つ (= その step で tensor 捕捉をする) ためと**推測**。
  VAE の 6.8s / 0.43s も実行順・warm 状態の記載が無く、**速度比較に使わない**。いずれも FP32・hook 付きで、性能値ではない。

### 3.4 活性値表 (全 30 step の running max・`allsteps/*`)

FP16 上限 65504。グループ判定 (meta `fp16_summary_by_group`): **TE / adapter / VAE = 「全段 FP16 余裕 4 倍以上」・DiT = 「FP16 不可の段あり」**。

| 部位 | max\|x\| | 所在 | FP16 |
|---|---|---|---|
| TE (Qwen3) 層出力 | 7,014 | `te_pos/te/layer25`。**token0 の値**。token0 以外の max は TE 層全体で 442.9 (`te_pos/te/layer27`・mlp_down_in を含めると 461 = `layer27_mlp_down_in`) | 可 (余裕 9.3 倍) |
| TE MLP down_proj 入力 | 3,665 | `te_pos/te/layer02_mlp_down_in` | 可 |
| adapter block 出力 | 63 | `adapter_*/adapter/block0` | 可 |
| VAE decode | 7.8 | `decode/vae/up_block3` | 可 |
| **DiT 残差 (block 出力)** | **272,435** | `block16_out`。**block12〜21 の出力が 65504 超** (block21 = 98,351) | **不可** |
| DiT 残差 (その他の block) | block0 = 74 / block1〜11 = 4,969〜11,646 / block22〜27 = 16,717〜45,080 | block22〜27 は余裕 4 倍未満 (meta `headroom_lt_4x`) | 境界 |
| DiT gate | 4,682 | `block15_norm2_gate` | 可 (ulp 4) |
| DiT scale / shift | 843.5 / 195.4 | `block10_norm2_scale` / `block18_norm2_shift` | 可 |
| DiT FF 出力 | 710 | `block12_ff` | 可 |
| DiT 変調後入力 (modulate) | 482 | `block23_norm2_mod` | 可 |
| DiT FF down 入力 | 231 | `block12_ff_down_in` | 可 |
| DiT attn1 出力 | 231 | `block22_attn1` | 可 |
| DiT q / k (qk-norm 後) | 49.5 | `block25_attn1_{q,k}_normed` | 可 |
| DiT attn2 出力 | 14 | `block21_attn2` | 可 |
| DiT temb / norm_out / proj_out | 82 / 32 / 5.3 | `dit/temb` 他 | 可 |

- 非有限値なし (全 3153 エントリで `nonfinite` 偽)。
- 残差の外れ値は少数要素に集中している: 例 `step0_cond/dit/block16_out` は max 251,564 に対し p99.9 = 2,204・rms 422。
- **未測定** (A0' で埋める計画): to_q/k/v (norm 前)・P·V 出力 (to_out 前)・FF pre-GELU (8192 幅)・adaLN LoRA 中間 (256)・
  adapter 内 q/k/v と pre-GELU・block 別 attention logit max。
- 結論 (A0 時点): 「全部 BF16」は不要で、**残差だけ FP32 にすれば既存 FP16 GEMM/attention を流用できる見込み**。
  ただし上の未測定段があるため「見込み」に留める (A2-2 で確定)。

## 4. A1 結果 — Qwen3 0.6B TE を OV へ (研究機・2026-10-22/23)

**IR** (`export_meta.json`): 静的 `[1,128]`・入力 `input_ids` / `attention_mask` とも **i64**・出力 `last_hidden_state` `[1,128,1024]` f32
(final RMSNorm 後・有効 token `[0:n]` のみ使用・右 pad・causal)・**FP16 (compress_to_fp16)・bin 1136.9MB** (FP32 版 2273.8MB も保存)。
変換グラフ自体の忠実度 (in-memory FP32 グラフ vs torch): rel ≤ 1.38e-5・cos_min 1.00000000 (`all.log`)。

### 4.1 デバイス別 (N=100・warmup 5・計測ケース real(n=41)・`bench_L128.json`)

| デバイス | compile | 中央値 | p95 | rel_max (real) | MAE (real) | cos_min (real) | token0 rel (real / full / half) |
|---|---|---|---|---|---|---|---|
| **NPU** | 22.06s | **56.2ms** | 57.1ms | **1.11e-2** | 5.28e-3 | **0.99987** | 2.59e-2 / 3.55e-2 / 3.45e-2 |
| iGPU (GPU.0) | 5.28s | 62.6ms | 64.1ms | 1.22e-2 | 1.28e-2 | 0.99986 | 2.83e-2 / 8.11e-2 / 3.41e-2 |
| CPU | 2.42s | 128.8ms | 158.8ms | 7.22e-5 | 9.58e-5 | 1.00000 | 8.4e-6 / 2.0e-5 / 1.4e-5 |

- 定義 (`scripts/dollma_convert_anima_text_encoder.py` の `compare_hidden`): rel_max = 有効 token 内の max\|誤差\| / max\|ref\|・
  token0 rel = token0 行の max\|誤差\| / token0 行の max\|ref\| (token0 を分けて見る理由: TE の**層出力**では token0 が桁違いに大きい (layer25 = 7,014 vs 他 token ≤ 443) ため。ただし比較対象の **IR 出力 (final RMSNorm 後)** では逆で、real(n=41) の token0 max\|ref\| 25.06 < 他 token 58.10)・cos_min = token 別 cosine の最小。
- CPU 行は FP16 IR を f32 推論精度で実行した値 (`all.log` の `fp16-ir/CPU-f32` と同値)。
- NPU の全ケース: rel_max 1.05e-2〜2.05e-2 (n=1 が最大)・cos_min 0.99984〜0.99996。iGPU の full(n=128) は cos_min 0.99930・token0 rel 8.11e-2 と NPU より悪い。
- ★probe7 では Qwen2-1.5B INT4 の NPU compile が失敗している (`docs/archives/investigation-log.md` probe7 節・VPUX の重複名エラー)。
  今回の Qwen3 0.6B FP16 静的形状は NPU で compile・推論とも成功した。

### 4.2 tokenizer ゲート

- Qwen3 / T5 の OV tokenizer を変換 (Qwen は `[1,128]` 右 pad・pad=151643 / T5 は可変長・eos=1 は C++ 側で付加)。
- **ゲート対象 = E-1 の 16 本 (base / aest × p1〜p4 × pos / neg) で Qwen・T5 とも全一致**。
- 情報扱い (ゲート外): tags-sym・japanese は一致 / **newline は T5 のみ不一致** / empty は判定なし
  (OV tokenizer は空文字で n=0 を返す → C++ 側で Qwen = pad 1 個・T5 = `[eos]` に置換する特殊処理が要る・`tokenizers_note`)。
- prompt 長 (`lengths.log`): 16 本の最大は qwen 42 / t5 42 → L の目安 64 (max×1.25 を 16 の倍数へ)。

### 4.3 NPU 採用の理由と精度コスト (A5 へ申し送り)

- **NPU を載せ先にする理由は速度ではなく「全 HW 協調」の研究目的** (CLIP と同じ方針)。NPU は iGPU より 6.4ms 速いだけで、
  CPU との差は 72.5ms/回。TE は 1 枚あたり cond/uncond の 2 回なので差は約 145ms/枚 (算術)。
  e2e に対する比率は **Anima の e2e 秒が未測定なので未確定** (計画の「~1.5%」は見積り)。
- **精度コスト**: NPU は rel_max 1.1%・token0 rel 2.6〜3.5%・cos_min 0.99987 (real)。CPU (FP16 IR・f32 推論) は rel_max 7.2e-5。
  NPU の誤差は CPU より 2 桁大きい。
- **A5 に置くゲート**: 「NPU TE を流した最終画像 vs FP32 TE (CPU 経路)」の **SSIM ゲート**。CPU 経路を精度基準として残す。
- **L=64 は推奨値で未測定**。ベンチは L=128 IR のみ。L=64 IR の NPU 精度・時間・L 超過時の挙動・空 prompt は **A5 で実測して確定**する。

## 5. A2 方式 — **推奨・A2-2 で確定**

> 計画本文 (2026-10-23 承認・Fable 独立レビュー「修正して承認」の修正 8 点を反映済) からの転記。方式は**推奨**であり、
> A2-2 の 4 block parity で確定する。BF16 全面化など方式変更が要る場合は再提示。

- **残差 x は FP32** (4096 token × 2048 × 4B = 32MB/バッファ・B=2 で 64MB)。
- **timestep / 変調経路は全部 FP32**: time_proj → t_embedder → AdaLN-LoRA → shift/scale/gate (と norm_out 変調) を FP32 `[B,C]` で持つ。
  理由: timestep には σ がそのまま入り (diffusers_anima の `ANIMA_SAMPLING_MULTIPLIER = 1.0`)、σ の最小は 0.0955。
  time_proj (2048 幅・half 1024) の最小周波数は 1e4^(-1023/1024) ≈ 1.0e-4 なので、sin 成分は 0.0955 × 1.0e-4 ≈ 1e-5 となり
  FP16 の正規化数の下限 (6.1e-5) を下回る = サブノーマル (算術)。
  既存 `launch_timestep_embedding` は FP16 出力 (`src/kernels/timeembed.cu` l.47-48)。gate 4,682 の FP16 ulp は 4。
  M=B の小 GEMM は純 FP32 の口を使う (既存 `launch_gemm_f32` は既定 TF32・純 FP32 は env `DOLLAMA_VAE_GEMM` 依存 = `src/kernels/gemm.cu` l.478-479 (HEAD `30a7a61`))。
- **block 内の重い演算は FP16**: `ln_modulate` (FP32 in → FP16 out) → q/k/v・FF は既存 `launch_gemm_fp16` (GemmEx・FP32 累積) と
  `launch_attention_fast` を流用。
- **枝出力 GEMM 3 本 (attn1.to_out / attn2.to_out / ff.net[2]) は FP16 in / FP32 out 案** (新規 `launch_gemm_fp16_out_f32`)。
  y の FP16 丸め × gate 4,682 の増幅を消す狙い。★**枝出力を FP16 out にするか FP32 out にするかは未確定** — A2-2 で両案を実測して決める。
- 重み: BF16 → FP16 変換ロード。範囲外・サブノーマルの有無は A2-0 で検出する。
  精度が足りない場合の保険 (精度レバー) は 枝出力 FP32 → q·k 経路 → 該当層 BF16 GEMM の順。

### 5.1 A2 の小段

| 小段 | 内容 | 合格条件 (計画) | 結果 |
|---|---|---|---|
| A2-0 | 重みレンジ検査 `scripts/dollma_check_anima_weight_range.py` (on-disk dtype・\|w\|>65504 件数・サブノーマル件数・0 化件数・tensor 種別別 min/max)。出力 `docs/logs/anima-a2/` | pytest 緑 + 実重みの件数表 (inf・0 化 0 件が期待値・サブノーマル件数は記録) | (未実施) |
| A0' | golden 再ダンプ: 捕捉 block {0,15,25,27} + block27 出力・未測定段の hook・**bf16 実行版**を同 capture で 1 本 (閾値の物差し) | ゲート全 PASS・既存 golden (block0) と bit 一致 | (未実施) |
| A2-1 | 新規カーネル (`src/kernels/` に追加のみ・各 `src/tests/test_<k>.cu`): ln_modulate / gated_residual_add / rms_norm_rows / rope_apply / gemm_fp16_out_f32 / 純 FP32 GEMM の口 / FP16→FP32 変換 / plain add / embedding gather | 研究機 `meson test -C build` で新規 test 緑 + 既存 test 全緑。単体 test は相対/ulp 閾値・massive activation パターン入力必須 | (未実施) |
| A2-2 | 4 block 部品 parity (block {0,15,25,27})。比較量は delta (= out − in) と枝ごとの gate⊙y。attn2 など小さい段は rms 比 MAE | 閾値 = bf16 実行 vs FP32 golden の乖離 (段別)。それ以下なら合格 | (未実施) |
| A2-3 | 記録: 本書の新設 + CLAUDE.md 計測表 A0/A1 行 | record-auditor の検査 | 本書 (検査前) |

## 6. 構造の確定値 (diffusers 0.38.0 現物照合・A2-2 の golden 突合で最終確認)

> 出典: 研究機 anima 実行環境の diffusers **0.38.0** `models/transformers/transformer_cosmos.py`・`attention_processor.py`・
> `diffusers-anima-src` の `modeling_anima_transformer.py`・`sampling.py`・`constants.py` を 2026-10-23 に SSH で現物照合 (record-writer)。
> 計画は「diffusers main で Fable が確認」と書くが、**main と 0.38.0 の差分は照合していない**。A2-2 の golden 突合が最終確認になる。
> この照合はログ・ファイルとして repo に残していない (照合した参照ソースは研究機の環境内にあり、本書の記述が唯一の記録)。

- LayerNorm (norm1/2/3・norm_out): **eps 1e-6・affine なし**。分散は 2 パス or Welford で計算する
  (外れ値 2.7e5 で 1 パスの E[x²]−E[x]² は桁落ちする。既存 `src/kernels/layernorm.cu` は 1 パス)。
- AdaLN-LoRA: `e = linear_2(linear_1(SiLU(embedded_timestep))) + temb` (temb 6144 全体を加算・linear は bias なし)・
  chunk 順 shift / scale / gate・`x·(1+scale)+shift`。norm_out は 2 chunk で `temb[..., :4096]` を加算。
- time_embed: `Timesteps(flip_sin_to_cos=True, downscale_freq_shift=0)` (既存カーネルと同規約)・temb = linear_2(SiLU(linear_1)) bias なし
  (出力 3×2048 = 6144)・embedded_timestep = RMSNorm(2048, eps 1e-6, weight) を timesteps_proj に掛けたもの。
- block 内の順序: `x += gate⊙attn1(n1(x))` → `x += gate⊙attn2(n2(x), ctx)` → `x += gate⊙ff(n3(x))`。to_q/k/v/out と FF は全て bias なし。
- qk RMSNorm は head 単位 128・**eps 1e-5** (diffusers `Attention` の既定)。他の RMSNorm は 1e-6。
- RoPE (DiT): **qk-norm の後・attn1 のみ** (attn2 には渡されない)。分割 t44 / h42 / w42 = 128、rotate_half は d↔d+64 ペア。
  rope_scale (1,4,4) は **NTK θ スケール** (θ_h = θ_w = 1e4·4^(42/40) ≈ 4.29e4・θ_t = 1e4)。T=1 なので t 成分は恒等。
  golden `dit_rope_cos/sin` と表を突合する。learnable pos embed は使わない (`extra_pos_embed_type=None`)。
- adapter: RMSNorm (1024 幅の pre-norm ×3 + out_proj 後の最終・qk-norm は head 64 幅・eps 1e-6)・RoPE θ=1e4・d↔d+32・
  **self と cross の両方に RoPE** (q = target 位置・k = Qwen 側の位置)。MLP・out_proj は bias あり。GELU は DiT・adapter とも **erf**
  (diffusers `GELU` 既定 `approximate="none"`・adapter は `nn.GELU()`)。出力に T5 weights を乗算し 512 長へ zero pad。
- cross-attn ctx は 512 全長で mask なし (zero pad 行も softmax 分母に寄与)。`rms_norm_rows` は全零行で 0 を返すこと (NaN 不可)。
- patch_embed (68→2048)・proj_out (2048→64) は bias なし。入力 17ch 目 (padding mask) は既定で全零。
  unpatchify は `permute(0,7,1,6,2,4,3,5)` (patch の逆順ではない・diffusers のコメントにも明記)。

## 7. A3 以降の落とし穴

- 既存資産は FP16 のみ: `DeviceWeights` は F16 以外で throw (`src/infer/unet.cu` の「is not FP16」)。BF16 重みは変換ロードが要る。
- split/merge heads は `src/infer/unet.cu` 内の static。流用するなら kernels へ移すか複製し、既存挙動を bit 不変に保つ。
- golden の比較量: 中盤 block の出力は残差が支配し corr がほぼ無条件に 1 になる → **delta と枝ごとの gate⊙y で比べる** (A2-2 と同じ)。
- golden 条件 (Euler・seed 1234・FP32) は E-4 の Anima 画像 (er_sde・seed 1000-1003・bf16) と比較できない。
- A0 の時間・VRAM は FP32・hook 付きの値で、自作実装の目標・分母に使わない。
- tokenizer: T5 は改行入り prompt で OV と不一致 (ゲート外)・空 prompt は C++ 側の特殊処理が要る・L 超過時の扱い (失敗させる or CPU/GPU 版へ退避) は未決 (A5)。
- ライセンス: A5 (backend 結線) 時に `THIRD_PARTY_NOTICES.md` で Anima を §1 へ移し、NVIDIA OML の "Built on NVIDIA Cosmos" 表示要否を判定する。
  また同ファイル「非商用ライセンスについて (Anima)」節のとおり、**A5 で HTTP 結線した後に第三者へ提供する構成は CircleStone §1.c の除外 (b)
  (第三者エンドユーザーと直接対話する・影響を与える利用) に当たり得る**。
  Qwen3 0.6B TE / Qwen-Image VAE / `hdae/diffusers-anima` のライセンスは**未取得** (同ファイル「未確認事項」節)。
  ライセンス判断はユーザー決裁 (CLAUDE.md ルール 3 ④)。
- 研究機は実走のみ・commit しない (成果物は scp → 開発機で commit)。GPU 実走は `GPU_BUSY.txt` で排他。`.cu` の文字列リテラルは ASCII。

## 8. 改訂履歴

- 2026-10-23: record-auditor 指摘 (重大 1・中 3・軽微 5) の是正。A1 を「完了」から「変換・ベンチ完了 / グルー未着手」へ、A2 承認主体・A3 以降未承認を明記、git status の列挙を削除、ライセンス除外 (b) を追記、TE token0 の記述・変調経路 FP32 の機序を補正。
- 2026-10-23: 新設 (A2-3 前半・record-writer)。A0/A1 結果・A2 方式 (推奨)・構造確定値・落とし穴を記載。A2-0 / A0' / A2-1 / A2-2 の結果欄は未実施のため空欄。

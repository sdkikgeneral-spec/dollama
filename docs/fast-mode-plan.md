# FAST モード計画 — 自作カーネル高速化 (default 温存 / --fast / --fast --fp8)

> Phase 2 の再オープン。起草時の概算で `dollama.exe` は SDXL 1024²/20step/CFG を **~19.5s/枚** としていた
> (現在の正典ベースラインは **warm default 20.93s**・下記「e2e ベースライン一本化」参照。同帯であり診断は不変)。
> 同一 GPU・同一重み・同一 step で diffusers/ComfyUI が **~3.8s** を出す ([[docs/measurements-log.md]] probe10) のに対し
> **RTX5080 を RTX3060 相当の速度で走らせている (ハードを約5倍ドブに捨てている)**。物理限界ではなく、
> 自作カーネルが Blackwell の演算器を埋めていない構造損失。本 doc はその回収計画 (分割タスク台帳)。
>
> ~~走行注意 (2026-07-04): F-0b G-1 rollout の完走を待て~~ → **解消済み (2026-07-06)**。F-0b は不採用クローズ済みで
> rollout はもう走っていない (安全弁 2 参照)。

## 芯 (設計方針)

**default を今まで通りに温存し、高速化はフラグで opt-in する。** ComfyUI の `--fast` / `--fp8_e4m3fn` と同じ正攻法。
default を byte-for-byte 現行のまま残すことで **golden-SSIM 0.999998 の回帰アンカー**が保たれ、「速くした結果どこかが壊れた」を
常に基準線と突き合わせて検出できる。

### 3段モード (速度/精度の曲線上に 3 点)

| モード | 中身 | 精度 | 狙い |
|---|---|---|---|
| **default (フラグ無し)** | 現行カーネル・現行 CFG ループ | golden 0.999998 完全保存 | 回帰アンカー・基準線・無改変 |
| **`--fast`** | #2 CFG batch=2 ✅ / #4 FlashAttn級 ✅ / #5 epilogue融合 / #1 GPU常駐+Graphs (条件付き・G-1k) | FP32 蓄積維持 = **絵は実質同一** | 当初狙い ~2倍 → **実測 1.32x (#5/#1 残)**・リスク無しの常用 (将来 default 昇格候補) |
| **`--fast --fp8`** | 上記 + FP8 を選択的 GEMM に | reward順位/最終RGB で gated | 最速・崩れたら層単位で FP16 に戻す |

- **精度ゼロコストと FP8 を混ぜない** のが肝。速度の大半は FP8 抜き (#1/#2/#4/#5・FP32 蓄積維持) で取れるので、
  「速いけど精度は保ちたい」点 (`--fast`) を必ず選べるようにする。FP8 は本当に最後の一段に隔離。
- フラグ表面: 環境変数 `DOLLAMA_FAST=1` / `DOLLAMA_FP8=1` もしくは HTTP/CLI フラグ。`--fp8` は `--fast` を含意 (FP8 単独は無効)。

## 現状診断 (律速の在処・実装根拠)

| # | 損失源 | 根拠 (file:line) | 性質 |
|---|---|---|---|
| CFG | cond/uncond を**逐次 2 forward** | `src/infer/diffusion.cu:534,543` | GPU を半分しか使わない |
| Loop | 毎 step で latent H2D / noise D2H / host で CFG合成・Euler・scale = **full sync × 20** | `src/infer/diffusion.cu:523-571` | カーネルの背後実行が全部潰れる |
| Attn | flash+wmma だが **1 block = 1 warp (32スレッド)** で低占有率 | `src/kernels/attention.cu:653` | SM が空く。4.60s の本体 |
| Attn | P·V epilogue が store_matrix→shared→acc の往復 | `src/kernels/attention.cu:571-578` | 余分な shared 往復 |
| UNet | `launch_unet` が **単一 latent 前提 (batch 引数なし)** | `src/infer/unet.cuh:88` | CFG batch=2 化に batch 次元の貫通が要る |
| FP8 | Blackwell FP8 tensor core **未使用** | — | 理論 2× を握ったまま |

> 上表は**起草時 (G-0 前) の診断スナップショット**。file:line は当時の行番号 (その後のコード変更でずれている。
> 例: diffusion.cu の逐次 2 forward は現在 594/604 行)。CFG 行と Attn 行は G-2k/G-3k で `--fast` 側は解消済み
> (default は回帰アンカーとして意図的に無改変残置)。
>
> 「attn 4.60s」は 2-6 時点・CFGなし・stub の値。**実 checkpoint での内訳を G-0 で取り直した (下表)**。実重み 4.9GB でも
> attention は 4.621s とほぼ不変 (コストは shape 駆動で重み値に非依存) → 律速の在処は stub 時と同じと確定。

### G-0 実測 (2026-07-05・既存 allow-list 済み dollama.exe・リビルド無し)

`DOLLAMA_PROFILE=1` + OV アセット無効化で**段2 PipelineGenerator (計装済み非 CFG `generate`)** を実 UNet 重み (`unet_weights.safetensors` 4.9GB) で 20step/1024² 実走。CFG 版 `generate_txt2img` は未計装だが、UNet 内カーネルの律速内訳は CFG/非 CFG で不変 (CFG は同 UNet を 2 forward するだけ) ゆえこれで律速は確定できる。

| 項目 | 実測 | 比率 | 解釈 |
|---|---|---|---|
| TOTAL (非 CFG) | 11.469 s | 100% | CFG 本番 ≈ 2×UNet+VAE ≈ **21.7s** (doc 冒頭 19.5s 帯) |
| UNet step total (20) | 10.254 s | 89.4% | per-step 0.513s |
| └ transformer (attn/gemm) | 7.007 s | 61.1% | うち gemm ≈ 2.39s |
| 　└ **attention only** | **4.621 s** | **40.3%** | **単一最大の律速** = #4 (G-3k) が最大レバーと裏取れ |
| └ resnet (conv/groupnorm) | 1.225 s | 10.7% | 二次律速 (#5 G-4k 対象) |
| VAE decode | 1.197 s | 10.4% | — |
| **host roundtrip** | **0.007 s** | **0.06%** | **ほぼゼロ** (ただし CPU 時間のみ・launch 谷は含まない。下記「profile の測り方の注意」) → #1 の「同期/D2H 撲滅」の時間直接効果は小。#1 の主眼は Graph ローンチ削減 + batch2 の下地に補正 |
| 段グループ | up 5.65s(49%) > down 3.61s(31%) > mid 0.97s | | up-block 律速 |

> **✅ e2e ベースライン一本化 (2026-07-09・G-2k ベンチで確定)**: `test_diffusion_batch2 DB2_BENCH=1` (20step/1024²/guidance7.5/**warm**=UNet+VAE 重み常駐/VAE decode 込み/**OV encode・cold 重み転送は付帯外**/SAC OFF 実 exe) で 3 構成を min 実測: **default (逐次 2 forward) 20.93s** / batch2 単独 17.59s (x1.19) / **--fast (attn_fast+batch2) 15.88s (x1.32)**。冒頭 19.5s / G-0 推定 21.7s と同帯 (warm ゆえ僅かに軽い)。G-3k 注記の 43.83s は cold 重み転送込みの段1 CFG 本番値で別物 (付帯込み)。**G-6k 出荷判定の分母 = warm default 20.93s / diffusers 3.8s は cold+OV 込みの別条件ゆえ最終比較は同条件で取り直す**。
>
> **profile の測り方の注意 (G-1k 判定に効く)**: 上表は `ScopedSyncTimer` = **ブロック境界ごとに cudaDeviceSynchronize を挿入した壁時計** (`src/infer/profile.cuh`)。よって①カーネル間の launch 谷は各バケット時間に**混入**して単独では見えない ②「host roundtrip 0.007s」は host 計算/転送 (scale_model_input/FP16 変換/H2D/D2H/scheduler step) の CPU 時間のみで「同期・launch オーバーヘッドの総量」ではない。ただし profile 推定 21.7s と非 profile warm 実測 20.93s の差 ~0.8s が「同期挿入コスト + launch 谷」の**上界**を与える = 大きくても e2e の ~4% (G-1k の直接効果上限の傍証)。

**優先度の実測結論**: **#4 attention (G-3k) が単独最大 40%** → 次いで **#2 batch=2 (G-2k・2×forward 削減+occupancy)**。**#1 の時間直接効果は小さい (host 0.007s)** ため順序を #4/#2 先行へ寄せる (下記台帳決裁: G-0b フラグ枠 + G-3k を先行)。

### 現状 GPU 稼働ベースライン (実測 2026-07-04・G-1 走行中の生成を nvidia-smi で観測)

| 指標 | 実測 | 解釈 |
|---|---|---|
| **消費電力** | **154W / 360W ≈ 43%** | 真の稼働指標。**パワーを半分も引いていない** = SM が満杯でない |
| SM 稼働率 (sm%) | 平均 ~78% (67-91%) | 「1warp でも動いた時間割合」= 満杯率ではない。sm% に騙されない |
| メモリ帯域 | ~11% | 帯域律速でもない |
| SM クロック | 2857 MHz | ブースト正常 (サボりではない) |
| VRAM | 7.9GB / 16GB | 余裕。VRAM は非ボトルネック |
| 候補間 | SM→0 / 126W | LM(CPU)+採点(CPU) の間 GPU が完全遊休 |

**結論**: 電力 43% + 帯域 11% = compute でも帯域でも律速せず **occupancy/latency 律速** (1warp/block attention + 依存待ち)。
「5080 を 3060 で走らせている」の定量的裏付け。#1(同期撲滅→SM谷を消す)/#4(multi-warp→power↑)/#2(遊休SMにuncond同居)で
**FP8 前に電力を 43%→300W超、時間を半減**が射程。目標 = **構造を壊さず GPU を飽和させる (power を上限近くまで引く)**。

## レバー詳細

### #1 拡散ループ GPU 常駐化 + CUDA Graphs (精度ゼロコスト)
- `scale_model_input` / CFG 合成 / Euler step を小さな elementwise カーネルにして latent を GPU 常駐のまま回す。
  per-step の D2H/H2D と 20 回の full sync を撲滅。
- さらに 20step × 数百カーネルの起動列を **1 本の CUDA Graph にキャプチャ**してローンチオーバーヘッド除去。
- 触るファイル: `src/infer/diffusion.cu` (ループ), `src/infer/scheduler.hpp` の host 実装を .cu elementwise へ, 新規 `src/kernels/elementwise` 追記。
- 精度: **ビット一致** (Graph 化・同期除去は数値不変。elementwise 化は演算順不変)。

### #2 CFG batch=2 (精度ゼロコスト)
- cond/uncond を batch=2 の 1 forward へ。`launch_unet` に batch 次元を通す。
  attention は既に `B` 対応、GEMM は M 倍化、conv2d/groupnorm は N 拡張、bias/act/norm は要素数 2 倍。
- 触るファイル: `src/infer/unet.cuh` / `src/infer/unet.cu` / `src/kernels/{gemm,conv2d,groupnorm,activation,geglu,bias_add}` の N/M 汎用性確認, `src/infer/diffusion.cu` の呼び出し。
- 精度: **数学的に同一** (束ねるだけ)。golden は再ベースライン (バッチ軸追加で並びが変わるだけ)。
- 効果: **素で ~2×**。出荷経路そのものが速くなる恒久資産。

### #4 attention を FlashAttention-2 級に (精度ゼロコスト)
- 現状 1 warp/block を **multi-warp block** 化し query 行タイルを増やして K/V 再ロードを償却
  (Sq=Sk=4096 の self-attn は K/V を query タイルごとに読み直しており DRAM 帯域律速)。
- **cp.async のダブル/トリプルバッファ**で K/V ロードと wmma をパイプライン化 (sm_120)。
- P·V epilogue の shared 往復除去 (fragment を直接 acc へ)。
- 触るファイル: `src/kernels/attention.cu` / `attention.cuh`。
- 精度: **FP32 蓄積維持** (reduction 順が変わり golden は微差・SSIM ゲートは通る想定)。

### #5 epilogue 融合 (精度ゼロコスト) — G-4k スコープ確定 (PL 決裁 2026-07-10)

起草時の構想は「bias + SiLU/GeGLU + GroupNorm を GEMM/Conv の epilogue に畳む」だったが、**実装現実に合わせて次の通り確定**:

- **前提 (コード実態)**: 大 GEMM 本体は既に cuBLAS GemmEx 委譲 (`src/kernels/gemm.cu` `use_cublas`・2-6 S3-B。自作 wmma は `DOLLAMA_GEMM=wmma` の比較経路)。GemmEx は epilogue を持たないため「GEMM の epilogue に畳む」は cuBLASLt 移行なしには成立しない。conv2d も 1x1=GEMM / 一般=im2col+GEMM (内部同じく cuBLAS) で、bias は GEMM 後の別パス `conv_bias_add_rows`・N>1 (batch2) は per-n 直列ループ (`src/kernels/conv2d.cu`)。
- **G-4k で採るもの (GEMM 本体に触らない融合のみ)**:
  - **(A) `launch_group_norm_silu` 融合**: resnet の GN→SiLU 2 パスを 1 パスに (GN カーネルは実測 73 GB/s と帯域非効率でここが伸びしろ)。
  - **(B) conv2d 後段パスの融合**: GEMM 後の bias 加算に time-bias channel broadcast / residual add を畳み、resnet 内の per-b ループ (`launch_bias_add_channel` ×B 等) を解消。
- **G-4k で採らないもの**: cuBLASLt 移行による FFN/attn linear の bias epilogue 融合 → **別 Pkg G-7k としてバックログ起票** (台帳参照。G-5k は FP8 で既に埋まっているため採番は G-7k)。
- **効く総面積の見積り (G-0 profile 準拠)**: (A)(B) の対象は **resnet バケット 1.225s / 10.7% (非 CFG 20step) のみ**。transformer バケット 7.007s ≈ attention 4.621s + 残 2.39s (cuBLAS GEMM + layernorm/geglu/bias_add) で、こちらの融合は cuBLASLt (G-7k) 側。CFG e2e (--fast 15.88s) 換算では resnet ≈ 61ms/step/forward ×40 ≈ **2.45s (~15%)** が面積上限 → resnet を仮にゼロにしても e2e は ~1.18x が理論天井、現実目標は下記ゲート (~1.02-1.04x)。resnet バケット外にも conv_in/out・down/up-sampler の bias パス・conv_norm_out の GN+SiLU が少量ある (段グループ計にのみ計上) が誤差帯。
- **ゲート (PL 設定・見積りと整合)**: resnet 1.225s → **≤0.95s (≥22% 削減)**・ストレッチ ≤0.85s。e2e は --fast 15.88s → 参考 ≤15.55s (~1.02x。resnet −0.275s×2 forward ≈ −0.55s → ~15.3s と掛け算が合う)。**e2e にハード合否は課さず resnet バケット + パリティ (default 無改変・fast SSIM ゲート) で判定**。
- 触るファイル: `src/kernels/{groupnorm,conv2d}` に融合バリアント, 呼び出し側 `src/infer/unet.cu` (VAE は G-4k 対象外)。
- 精度: 融合は数値ほぼ不変 (中間の FP16 往復が減る分むしろ改善方向)。

#### G-4k S1a 実測 (2026-07-11・GN multi-block + epilogue 配線・完了)

G-11k (GN multi-block 化) を S1a として G-4k に織り込み。`launch_group_norm_mb` (src/kernels/groupnorm.cu) =
partial→finalize→normalize の 3 カーネル **2 段決定的集約 (atomic 禁止・run-to-run bit 一致が非交渉ゲート)**。
`epilogue` フラグを `Scratch` 経由で `src/infer/unet.cu` に配線 (`attn_fast` の厳密ミラー)。対象 = resnet の
norm1/norm2 + conv_norm_out。default(off) は 1-block GN のまま byte-for-byte 無改変。

- **GN カーネル単体 (test_groupnorm 実走 ALL PASSED)**:
  - parity: MAE ~3e-7 / max_abs 0.0039 (ゲート <1e-3 / <4e-3) PASS
  - bitexact: 3 runs 完全ビット一致 PASS
  - 帯域: unet_320_128 mb **444.6 GB/s** ≥ 300 ゲート PASS (1-block 105 GB/s → **4.2x**)。
    B2 527 / 640_64 334 / 1280_32 226 GB/s
- **UNet 結線 (test_unet_fast 実走 ALL PASSED・warm ハンドル)**:
  - epilogue(attn_fast+epilogue) vs default: **SSIM 0.999999 ≥ 0.9999** / MAE 6.6e-05 / bad=0 PASS
  - default 無改変ゲート: fast vs default **bit-exact 維持** (MAE=0/SSIM=1)・default vs golden SSIM 0.999996 PASS
  - 1step warm 中央値 (最終検証 2 走): default **479.0/483.8ms** / fast 440.8/412.9ms / fast+epilogue
    **436.2/414.3ms**。epilogue vs fast の差分は **−4.6ms / +1.4ms = run-to-run ノイズ内** (パリティ系は
    2 走とも完全同値で決定的)
- **resnet ≤0.95s ゲートの扱い**: S1a 単体の 1step 寄与はノイズ床下 (GN 置換のみで SiLU は別パスのまま =
  パス数不変・帯域改善分だけ。GN 単体は 4.2x で実証済み)。回帰なしは確認。resnet バケット合否は
  S1b (GN+SiLU 融合) + S2 (conv2d 後段融合) 完了後の再 profile で判定する。

### #3 FP8 Tensor Core (精度トレードオフ・最内 opt-in)
- **選択的 FP8**: 計算律速の大 GEMM のみ入力を FP8 (E4M3)、**蓄積は FP32** (既存規約と整合)。
  softmax / GroupNorm / time embed / 最初と最後の conv は **FP16 のまま残す** (敏感層)。
- 崩れたら**層単位で FP16 に戻せる** (可逆・段階適用)。
- 触るファイル: `src/kernels/gemm.cu` に FP8 経路, `src/infer/unet.cu` で層ごとの精度選択テーブル。
- 精度: **golden-SSIM 0.999998 には届かない (E4M3 は実質2桁)** → 下記の FP8 専用ゲートで判定。
- 効果: FP16 wmma の理論 2×。diffusers 既定は FP8 未使用なので**自作で追い抜ける唯一の軸**。ComfyUI の `--fp8_e4m3fn` 実績あり (実用上わずかな劣化)。

## 検証・ゲート (モードごとに物差しを変える)

| モード | ゲート |
|---|---|
| **default** | 現行 golden-SSIM 0.999998 を**無改変で維持** (test_unet / test_vae_decode / test_diffusion 全緑) |
| **`--fast`** | golden 再ベースライン後、noise_pred SSIM ≥ 0.9999 / 最終 RGB SSIM ≥ 0.999 (絵が実質同一であること) |
| **`--fast --fp8`** | golden は**使わない**。① 最終 RGB を FP16 経路と LPIPS/SSIM で比較 (知覚等価) ② **reward 順位相関** (best-of-N は絶対値でなく順位のみ使用・崩れなければ実害ゼロ)。**通らなければ FP8 不採用** (層単位で戻す) |

## 実測手段と SAC 制約 (非交渉)

- **2026-07-06 更新: SAC OFF・実 exe 実走で全 49/49 緑を確認**。以後の全ゲート (G-2k の SSIM / CFG e2e 秒数、G-5k の LPIPS/reward 順位相関) は**実 exe 実測を正**とする。下記の旧運用 (開発機ビルド緑 + Python/OV 検証) は SAC 再有効化時のフォールバックとして温存。
- (旧) 研究機は SAC で**再ビルドした exe の新ハッシュがブロック** ([[reference_wdac_new_exe_block]])。カーネルを直して
  リビルドすると `dollama.exe` の実走緑が取りにくい。着手前に **allow-list 更新をユーザーへ依頼**するか、
  **開発機ビルド緑 + Python/OV 経由の数値検証**で回す運用を確定させる。
- profile: 拡散ループに CUDA events を仕込み、UNet 内の attn/gemm/conv/groupnorm の実測内訳を取る (G-0)。

## 分割タスク台帳

| Pkg | 内容 | 担当 | 依存 | 精度 | status |
|---|---|---|---|---|---|
| **G-0** | profile で実 checkpoint 内訳確定 (既存計装・リビルド無しで実測) | — | (当時) F-0b G-1 rollout 完走待ち・解消済 | — | ✅ (上表・attention 40% 最大) |
| **G-0b** | FAST フラグ枠 (env `DOLLAMA_FAST`/`DOLLAMA_FP8` + CLI・`FastConfig` を拡散経路へ貫通・default byte-for-byte 無改変) | cpp-implementer | G-0 | 無改変 | ✅ (test_fast_config 緑・golden 無改変で緑) |
| **G-1k** | #1 ループ GPU 常駐化 + CUDA Graphs (時間直接効果は小・Graph ローンチ削減主) | cuda-kernel-dev | G-2k/G-4k 後の再profile | ゼロコスト | 🔲 **条件付き降格**: 降格根拠は 2 本立て — ① host 計算/転送の CPU 時間 0.007s (ただしこれは launch 谷を含まない・「profile の測り方の注意」参照) ② profile 推定 21.7s と非 profile warm 実測 20.93s の差 ~0.8s が同期+launch 谷の**上界** = e2e の ~4% 止まり。加えて G-2k/G-4k で起動列が変わると Graph 再キャプチャ=手戻り。**再判定は ScopedSyncTimer でなく非侵襲 profile (nsys 等) で launch 谷を直接観測**し、実測で残る場合のみ着手・残らなければ不着手クローズ可 |
| **G-2k** | #2 CFG batch=2 (`launch_unet` batch 貫通) | cuda-kernel-dev + cpp-implementer | G-0b | ゼロコスト (fast側 SSIM=1 目標: 行単位 K-loop 不変なら G-3k 同様ビット一致狙い・届かねば ≥0.9999 ゲート) | ✅ **完了 (S1〜S3c・下記注)。parity gate g=1.0 SSIM 0.9994≥0.999 PASS (sweep g=1 は 0.9995・別走行)・batch2 e2e 1.19x (期待 ~2× に非到達)・合成 --fast 1.32x** |
| **G-3k** | #4 attention FlashAttn級 (multi-warp/cp.async・`launch_attention_fast` 別 TU・既存温存) + call site 結線 (`--fast`→attn_fast) | cuda-kernel-dev | G-0b | ゼロコスト | ✅ **カーネル SSIM=1.0 ビット一致・self4096 単体 1.41x / warm e2e UNet 1.20x (下記注)** |
| **G-3kf** | G-3k follow-up: `test_unet_fast` を常駐ハンドル warm 経路へ (毎 call 4.9GB 再転送を止め・ゲート数値を実態 1.20x に是正) | cpp-implementer | G-3k | — | ✅ **完了**: warm ハンドル (unet_weights_create) で計測・default vs golden SSIM 0.999996/bad0・fast vs default bit-exact・warm 1step default 526ms/fast 437ms = **1.20x** |
| **G-4k** | #5 epilogue 融合 = (A) `launch_group_norm_silu` + (B) conv2d 後段融合 (time-bias/residual・per-b ループ解消)。cuBLASLt は採らない (→G-7k) | cuda-kernel-dev | G-2k | ゼロコスト | 🔲 **着手中・S1a ✅** (S1a 実測は #5 節「S1a 実測」参照。残: S1b GN+SiLU 融合 / S2 conv2d 後段融合 / S3 配線) |
| **G-5k** | #3 FP8 選択的経路 + 層別精度テーブル + FP8 ゲート | cuda-kernel-dev | G-2k/G-4k | トレードオフ | 🔲 |
| **G-6k** | 3段モードの実測まとめ + 出荷判定 (`--fast` の default 昇格是非 / FP8 採否) | PL + gpu-benchmarker | 全部 | — | 🔲 |
| **G-7k** | cuBLASLt 移行による linear (FFN/attn to_q/k/v/out) の bias epilogue 融合 — transformer バケット残 ~2.39s が対象。GEMM 本体の呼び方が変わるため G-4k と分離 (PL 決裁 2026-07-10・当初仮称「G-5k」は FP8 と衝突するため G-7k で採番) | cuda-kernel-dev | G-4k | ゼロコスト | 🔲 **バックログ** (順序図の主線外・費用対効果を G-4k 後の再 profile で判断) |

> 各 Pkg は着手時に **test も実装** (CLAUDE.md ルール4・`src/tests/test_*`・meson 緑)。

### G-3k 結果の注記 (2026-07-05・実走緑・warm 実測で確定)

- **正しさ**: `launch_attention_fast` は既存 `launch_attention` と **SSIM=1.0 / MAE=0 (ビット一致)**。reduction が同じ 16-タイル順のため。UNet 全段でも default vs fast で **noise_pred SSIM=1** = 絵は完全同一。default (fast off) は golden 無改変で緑。→ **精度リスクゼロで恒久資産**。
- **速度 (warm 実測・`prof_unet_fast_warm.cu` + 段1 e2e)**:
  - attention 段/step **231→161ms = 1.43x** (単体 micro 1.41x と一致)、UNet forward/step **452→382ms = 1.19x** (−72ms/step)、attention 以外 (GEMM/resnet) は不変 = 副作用なし。
  - 段1 CFG 本番 e2e (交互3回): **43.83→40.20s = −3.63s** (全ペアで fast 勝ち・逆転なし)。CFG は 1step 2 forward ゆえ理論 −2.9s と整合。
- **「1.03x」は計測アーティファクトだった (訂正)**: 初回報告の e2e 1step 1.03x は `test_unet_fast` が後方互換 `launch_unet(const SafeTensors&)` を使い**毎 call で 4.9GB を H2D 再アップロード (cold)** → attention 比率が 40%→20% に希釈された値。カーネルは遅くない。→ follow-up は **G-3kf で完了済** (warm ハンドル化・実態 1.20x に是正・台帳参照)。
- **含意**: G-0 の「attention 40% 最大律速」を fast が 1.43x で回収済み。単独 40% 律速は解消 (fast 後 warm 内訳 = attn 161ms / GEMM ~130ms / resnet ~60ms)。**次の最大レバーは計画どおり #2 CFG batch=2** (2 forward→1 batched・occupancy+host 半減)。ソースレビューの「shared 占有率律速で 1.09x 上限」説は warm 実測で否定 (両者を回して確定)。

### G-2k 結果の注記 (2026-07-09・SAC OFF 実 exe 実走で確定・S1〜S3c)

- **実装 (S1〜S3b)**: `launch_unet_batched(B)` で batch 次元を UNet 全段に貫通 (S2: attention は既に B 対応・linear GEMM は M=B*tokens・conv2d は per-n ループ・groupnorm/bias/act は要素数倍化)。`generate_txt2img` に `use_batch2` 分岐を新設 (S3b): batch2 時は cond/uncond を連続 [2,...] に束ね `launch_unet_batched(2)` を 1 step 1 回で回す。default (batch2 off) は逐次 2 forward を **byte-for-byte 無改変**で保持 (回帰アンカー)。フラグは env `DOLLAMA_BATCH2` / `--fast` 含意 (S3a)。
- **正しさ (S3c parity gate・`test_diffusion_batch2`)**: batch2 は数学的同一変換だが linear の GEMM が M=tokens→B*tokens に変わり cuBLAS タイル選択=蓄積順が変わるため FP16 で ~1 ULP/要素の差 (S2 per-sample MAE 6.4e-5・**ビット一致には非到達**)。この微小差は CFG 合成 `noise=uncond+g*(cond-uncond)` で **guidance 倍に増幅**され step 累積 → 最終 RGB 発散は guidance に**単調** (実測 sweep g=1:SSIM0.9995 / g=3:0.9988 / g=7.5:0.9966・平均差≤0.3/255=絵は実質同一)。**ノイズ床 (off@g=1.0 x2) は完全 bit-exact** → 発散源は 100% batch2 の tiling 差と切り分け確定。**ゲート (ユーザー決裁 2026-07-09)**: CFG 増幅を排し batch2 単独の数学的等価性を測るため **guidance=1.0 の最終 RGB SSIM ≥ 0.999 をハードゲート** (計画書 --fast RGB 基準)・realistic な g=3/7.5 は characterization 出力 (regression 監視)。gate g=1.0 SSIM 0.9994 PASS。
- **速度 (`DB2_BENCH=1` warm e2e・上記ベースライン表)**: batch2 単独 20.93→17.59s = **1.19x** (attn_fast off)・**期待「素で ~2×」には非到達**。主因: **conv2d は per-n 直列で batch されない** (resnet ~10.7% は恩恵ゼロ)・batch されるのは attn+linear GEMM のみで占有率ゲインも部分的 = 「逐次 2 forward で GPU 半遊休だから 2×」の仮説ほど単純でなかった。精度ゼロコストで 1.19x は正の恒久資産。**合成 --fast (attn_fast+batch2) = 20.93→15.88s = 1.32x**。
- **含意**: 精度ゼロコスト 2 手 (#4 attn ✅ + #2 batch2 ✅) の合成が **1.32x** (20.93→15.88s)。計画冒頭の「4 手で ~8-10s (~2倍超)」は batch2 が 1.19x に留まったぶん下方修正が要る。残る精度ゼロコストは **#5 epilogue 融合 (G-4k)** = resnet/conv 部 (batch2 が効かなかった ~10.7%) が主対象で相補的 (GEMM 側の bias epilogue は cuBLASLt が要るため G-7k へ分離・#5 節)。その後 FP8 (G-5k) が diffusers 級への最大レバー。

## 順序 (掛け算で効く)

```
G-0/G-0b (profile+枠) ──► G-3k ✅ (attention) ──► G-2k ✅ (CFG batch2) ──► G-4k (融合) ──► [G-1k 条件付き] ──► G-5k (FP8) ──► G-6k (判定)
```
> **順序改訂 (2026-07-06)**: G-0 実測で host roundtrip 0.007s と判明したため G-1k (同期撲滅/Graphs) を主線先頭から**末尾の条件付き**へ降格。attention (G-3k) は完了済ゆえ次は #2 CFG batch=2 (G-2k) = UNet 15.3s 全体に効く単独最大レバー。
> **G-2k クローズ (2026-07-09)**: batch2 1.19x/合成 --fast 1.32x で完了。**次は #5 epilogue 融合 (G-4k)** = batch2 が効かなかった conv/resnet 部が対象 (相補的・スコープは #5 節の PL 決裁参照)。その後 FP8 (G-5k)。GEMM (linear) 側の bias epilogue は G-7k バックログ (主線外)。

**期待値の下方修正 (2026-07-09 実測反映)**: 起草時の「#1/#2 だけで ~10-11s・4手で ~8-10s」は batch2 が素で ~2× 効く前提の見積で、
**実測で撤回**。実際は #4✅+#2✅ の合成が **1.32x (20.93→15.88s)** に留まった (conv2d per-n 直列で resnet に batch2 が効かず・G-2k 注記)。
残る精度ゼロコストの現実的な着地は **#5 (G-4k) で ~15.1-15.5s** (面積 ~15%・#5 節の見積り)、#1 (G-1k) は上界 ~0.8s の条件付き。
つまり **精度ゼロコスト4手の合計は ~15s 帯 (~1.4x) が現実線**で、**diffusers 級 (~3.8s) への主レバーは FP8 (G-5k) と
G-7k/attention 続戦等の追加最適化**に移った。FP8 は従来どおり別立ての gated 実験 (絵の等価性はゲートで担保)。

## 安全弁 (非交渉・既定)

1. **default は無改変** (回帰アンカー)。全高速化は `--fast` 以降に隔離。
2. ~~走行中 G-1 に触れない~~ **解消済み (2026-07-06)**: F-0b は不採用クローズ (a3b129d merge) 済みで G-1 rollout はもう走っていない。`data/rollouts/` 制約は撤去。
3. **G-2k 着手前に G-0b/G-3k を先行コミット** (対象ファイル明示 stage)。ワークツリーに未確定成果を残したまま次 Pkg に入ると安全弁4 が守れず、問題時のロールバック点も失う。
4. **FP8 は可逆** (層単位で FP16 に戻せる)。ゲート未通過なら不採用でクローズ。
5. **コミットは巻き込み禁止** (in-flight ファイルがあるため `git add -A` を使わず対象ファイルのみ明示 stage)。

## スコープ外

- **steps 削減 (Turbo/Lightning/LCM 蒸留・高速 sampler)** = カーネルでなくモデル/サンプラ層の直交レバー。
  最大の乗数だが eval 基準 steps=20 を動かす判断が別途要るため本 doc では扱わない (要ユーザー決裁)。
- **cuDNN / CUTLASS への全面置換** = 最速だが**自作カーネル研究の芯を殺す**ため不採用 ([[project_output_quality_over_features]])。
  注意: 「全部自作のまま」ではない — 大 GEMM 本体は既に cuBLAS GemmEx 委譲済み (2-6 S3-B・CLAUDE.md の
  「配管/重実装は定番に委ねる」方針・`DOLLAMA_GEMM=wmma` で自作へ戻せる)。スコープ外なのは attention/conv 等の
  **研究コアまで**ライブラリに明け渡すこと。cuBLASLt の bias epilogue は「委譲済み GEMM の呼び方の改善」であり
  芯を殺さないため、G-7k としてバックログ化 (台帳参照)。

## 参照
- 診断根拠: `src/infer/diffusion.cu` / `src/kernels/attention.cu` / `src/infer/unet.cuh` / `src/kernels/gemm.cuh`
- ベースライン: CLAUDE.md 計測表 (probe10 3.80s / 自作フル拡散 11.30s / test_unet SSIM 0.999998) / `docs/measurements-log.md`
- (解消済み) 走行中だった依存: `docs/f0b-rejection-sft-plan.md` (F-0b G-1 rollout・2026-07-06 クローズ)
- SAC 制約: [[reference_wdac_new_exe_block]] / CUDA ビルド: [[reference_cuda_meson_build]] / カーネル規約: [[reference_cuda_kernel_conventions]]

## 未起票バックログ (G-4k 着手時点・fable 調査)

> 2026-07-11 read-only ソース調査で発掘した、既存台帳 (G-0〜G-7k) に**含まれない**高速化余地。
> 全項目 file:line で裏取り済み。効果見積りは G-0 profile / e2e ベースライン (default 20.93s / --fast 15.88s /
> VAE 1.197s / resnet 1.225s 非CFG) に紐づけ、根拠が薄いものは「要実測」と明記する。採番は G-8k 以降。
> **本節は調査記録であり着手決裁ではない** (着手時は PL 決裁 + test 実装のルール4に従う)。

### 候補台帳

| Pkg | 内容 | 対象バケット | 効果見積り | 難度/リスク | 裏取り |
|---|---|---|---|---|---|
| **G-8k** | step ループ内 cudaMalloc/cudaFree 撲滅 (スクラッチプール / VAE ワークスペース常駐) | 全バケットに分散混入 | **要 nsys 実測** (数百 ms 級の可能性・下記) | 低 (数値不変・配管のみ) | ✅ |
| **G-9k** | VAE decode 高速化パック (mid attn GEMM 化 / GN f32 占有率 / FP32 重み変換キャッシュ / BF16 gated) | VAE 1.197s | **1.197s → ~0.4-0.6s** (概算・内訳推定は下記) | 低〜中 (BF16 のみ gated) | ✅ |
| **G-10k** | conv2d 真の batch2 (im2col N 込み + cublasGemmStridedBatchedEx) | CFG e2e の resnet ~2.45s | **~0.5-1.0s** (batch2 恩恵ゼロ領域の解消) | 中 (パリティ ~1 ULP・G-2k S2 と同種) | ✅ |
| **G-11k** | GroupNorm multi-block 化 (G-4k(A) 補強・grid=32 block の占有率是正) — **G-4k S1a に吸収・実装完了** (帯域 105→444.6 GB/s = 4.2x・#5 節「S1a 実測」参照) | resnet 1.225s の GN 面積 | GN 実効帯域 73→数百 GB/s (面積は G-4k 再 profile で確定) | 低〜中 (2 段 reduction・数値は蓄積順変化) | ✅ |
| **G-12k** | transformer 脇役融合 (QKV fused GEMM / split-merge heads 転置削減) | transformer 非attn ~2.39s (GEMM 純増でない・下記) | 見積り不能 (バケット内訳の再 profile が先) | 中 | ✅ |
| **G-13k** | マルチ stream 独立枝並列 (QKV / conv_shortcut / time_emb_proj) | launch 谷・依存直列 | 見積り不能 (**nsys で谷を直接観測してから**・G-1k 再判定と同一前提) | 中〜高 (stream/event 配線・検証コスト) | ✅ |
| micro | time_ids 毎 step D2H 同期 / up ループ free→malloc→memcpy / push_skip D2D コピー | 微小 | 各 ~ms 級 | 低 | ✅ |

### G-8k: step ループ内 cudaMalloc/cudaFree 撲滅

**現状 (裏取り)**:
- `src/kernels/conv2d.cu:370/420` — im2col 経路は **conv 1 回ごとに** `d_col` を cudaMalloc/cudaFree する (conv2d.cuh:34 に明記どおり)。UNet 1 forward の 3x3 conv は resnet 17 個×2 + conv_in/out + sampler 4 ≈ **~40 回** → CFG 20step で **~1,600 ペア/画像**。up_blocks.2 の conv1 (Cin=640, 128²) は col が **189MB/回** で malloc 単価も大きい。
- `src/infer/unet.cu:351-379` — `Scratch` が段スコープごとに cudaMalloc し、スコープ脱出で cudaFree。transformer_block 1 回で 9-13 本 (unet.cu:490-498, 542-547, 587-590)。1 forward で 100 本超。
- `src/infer/unet.cu:934-937, 972-975, 1007-1010` — up ループ毎反復で `cudaFree(d_cur) → cudaMalloc → cudaMemcpy D2D`。**ポインタ swap にすれば malloc/free/コピーとも消える**。
- `src/kernels/vae_decode.cu:684-685, 775-776` — decode ごとにキャリー 512MB×2 (FP16) + **1GB×2 (FP32)** を malloc/free。加えて `Scratch`/`Scratch32` (243-289) が段ごとに GB 級を確保・解放。
- cudaFree は暗黙のデバイス同期点。電力 43% / 帯域 11% の「occupancy/latency 律速」診断と整合する阻害要因であり、G-1k (CUDA Graphs) の前提条件でもある (Graph は capture 中の cudaMalloc/Free を許さないため、**G-1k を再浮上させるなら本 Pkg が先行必須**)。
- **効果は nsys で要実測**: ScopedSyncTimer profile では malloc 時間が各バケットに混入して単独で見えない。概算 (~1,600 ペア × 20-100µs + VAE GB 級 malloc) で数百 ms 級はあり得るが、根拠のある秒数は出せない。
- **実装**: サイズクラス別の再利用プール (ハンドルに持たせる) or `cudaMallocAsync`/メモリプール。数値完全不変。default に触れるか `--fast` 側に隔離するかは PL 決裁 (数値不変なので default 適用も筋は通るが、無改変原則との整合を要判断)。

### G-9k: VAE decode 高速化パック (1.197s / 10.4%)

G-0 以降未着手のバケット。内訳の直接 profile は無いが、コードから 4 つの独立した非効率を特定:

1. **mid attn が低速経路** — `src/kernels/vae_decode.cu:657` で `launch_attention(..., B=1, H=1, Sq=Sk=16384, Dh=512, ...)`。FLOP は 2·S²·Dh·2 ≈ **550 GFLOP** で、既存 attention の実測 ~1.6 TFLOPS なら **~0.3s** = VAE バケットの 1/4 級。`launch_attention_fast` は Dh=512 で shared 超過フォールバックの公算 (attention.cuh:69-71) なので、ここは **cuBLAS 2-GEMM (QKᵀ→softmax→PV) の materialize 方式**が本命: scores 16384² FP16 = 512MB は VRAM 許容内、cuBLAS ~40 TFLOPS + softmax 帯域で **~30-40ms** 圏。単発 (decode 1 回/画像) なので旧方式温存 + 分岐でリスク小。
2. **GroupNorm f32 の占有率崩壊** — `launch_group_norm_f32` (vae_decode.cu:510-517) も 1 group=1 block・**grid=32** のまま 1024² を舐める (up3 は 1 block が 4.2M 要素×2 パス)。up2/up3 + conv_norm_out で GN ~13 回 ≈ 概算 **~0.25s**。G-11k と同じ multi-block 化で 1 桁改善余地。
3. **FP16→FP32 重み変換を conv ごと毎 decode 実施** — `src/kernels/conv2d.cu:744-766` (`launch_conv2d_f32_gemm`) が呼び出しごとに cudaMalloc + `weight_h2f` + cudaFree。VAE FP32 経路の conv は decode あたり ~16 回。**VaeWeightsHandle に FP32 版をキャッシュ**すれば消える (VRAM +~100MB・create 時 1 回)。
4. **FP32 中間の帯域 2 倍** — up2 以降 FP32 必須の根拠は**実測記録がコードにある** (vae_decode.cu:9-20: up3 で ±100000 超 → FP16 だと Inf→GN NaN 伝播。「根拠なし」ではない)。ただし **BF16 (range ~3e38) なら overflow せず帯域は FP16 並み**。蓄積は全カーネル FP32 のままなので影響は仮数 8bit の丸めのみ → `--fast` RGB SSIM ≥0.999 ゲートで gated 採否 (FP8 と同じ「可逆・不採用可」の型)。実装は f32 バリアントの dtype 差し替え (中規模)。

合算の現実目標: **1.197s → ~0.4-0.6s** (1+2+3 で ~0.5s 回収・4 は上積み)。e2e では −0.6〜0.8s = --fast 15.88→~15.1-15.3s 級。CFG 恩恵ゼロのバケットなので G-4k/G-5k 後は相対比率がさらに上がる。

### G-10k: conv2d 真の batch2 (G-2k 残課題の解消)

- **現状**: `src/kernels/conv2d.cu:483-503` — N>1 は per-n オフセットで N==1 ヘルパを **直列 N 回** 呼ぶ (G-2k S1 のビット一致優先設計)。これが「batch2 が resnet に効かない」主因 (G-2k 注記)。
- **提案**: ① im2col カーネルに n 次元を追加し col を [N, K, Hout·Wout] で 1 launch 生成 ② GEMM を `cublasGemmStridedBatchedEx` (strideA=0 で重み共有 / strideB=K·HW / strideC=Cout·HW) に置換。**出力 [N, Cout, HW] は strided batched の C stride と自然に一致し、追加 scatter 不要** (帯分割時のみ既存 scatter を N 対応)。1x1 経路も同型で置換可。
- **効果**: CFG e2e (--fast 15.88s) の resnet ≈ 2.45s が対象。per-n 直列 → batched 並列で占有率が上がり、bias/im2col の launch も半減。**~0.5-1.0s** (resnet 1.3-1.7x 相当) を見込む。正確な倍率は要実測。
- **リスク**: strided batched は単発 GEMM とタイル選択が変わり得る = **G-2k S2 と同種の ~1 ULP パリティ** (ビット一致は狙わない)。既存 g=1.0 SSIM ≥0.999 ハードゲートをそのまま流用。default (B=1) は無改変 (N>1 枝のみ差し替え)。難度中。G-4k(B) の per-b bias ループ解消とは相補 (あちらは conv 後段・こちらは conv 本体)。

### G-11k: GroupNorm multi-block 化 (G-4k(A) の補強)

- **現状**: `src/kernels/groupnorm.cu:175-177` — grid = **N×num_groups = 32 block** (B=1)・256 thread/block = 全 GPU で 8,192 thread のみ。RTX5080 の 84 SM の 4 割弱に 1 block ずつで残りは完全遊休。さらに入力を stats パスと正規化パスで **2 回 global 読み** (119-124, 139-147)。CLAUDE.md 計測の **73 GB/s (理論 ~960 GB/s の 8%)** の構造要因はこれ。groupnorm.cu:18-20 に「multi-block reduction へ昇格」と設計時から明記されている宿題でもある。
- **提案**: グループを複数 block で分担する 2 段 reduction (partial sum → 集約 → 正規化)。G-4k(A) の GN+SiLU 融合は **パス数を減らすだけで grid=32 は直らない**ため、G-4k 実装時に本項を織り込むか直後に補強するのが掛け算として正しい。
- **効果**: GN 実効帯域 73 → 数百 GB/s。resnet 1.225s 内の GN 面積は G-4k 再 profile で確定してから秒数を語る (概算では UNet 内 GN ~0.3s 級 + VAE 側 G-9k-2)。
- **リスク**: 蓄積順が変わり FP16 で微差 → fast 側 SSIM ゲート。default 無改変で fast 側に隔離可能。難度低〜中。

### G-12k: transformer 脇役融合 (「gemm ~2.39s」の正体)

- **裏取り**: G-0 の transformer 7.007s − attention 4.621s = 2.386s は **cuBLAS GEMM 純増ではなく** layernorm / chw↔sc 転置 (unet.cu:146-182) / split·merge heads (191-237, transformer_block 1 回に per-b×8 launch: 506-512, 527-531, 548-556, 571-575) / bias_add / geglu / add が全部混ざったバケット (ScopedSyncTimer の粒度)。まず CUDA events で GEMM 純分と脇役分を分離するのが先。
- **レバー候補**: ① **QKV fused GEMM** — self-attn の to_q/k/v (unet.cu:503-505) は同一入力×3 GEMM。ハンドル作成時に重みを [3C, C] へ concat すれば 1 GEMM 化 (cross の to_k/v も [2C]) → launch 1/3・GEMM 効率向上。出力 [M,3C] は split_heads がオフセット読みで吸収可。② split/merge heads・chw↔sc 転置は非 coalesced 書き (out[s*C+c]) → shared タイル転置化 or attention 側 stride 読み対応で pass ごと削除。③ 残りの bias/act は G-7k (cuBLASLt epilogue) と同域。
- **効果**: 内訳未計測につき**見積り不能** (脇役分が仮に ~1s なら大レバー、~0.3s なら micro)。再 profile が着手条件。難度中・数値は ① がビット一致狙い可 (GEMM 形状は変わるので実際は ~1 ULP)、② は数値不変。

### G-13k: マルチ stream 独立枝並列

- **裏取り (依存独立な枝の具体列挙)**: 全カーネルが default stream 直列 (cublasSetStream も未使用)。
  - resnet: conv_shortcut(x) (unet.cu:456-459) は norm1→conv1→norm2→conv2 の main 鎖と独立 (合流は launch_add)。
  - resnet: time_emb_proj linear (433-440) は norm1→silu→conv1 と独立 (合流は bias_add_channel)。
  - attn: QKV 3 linear (503-505) / cross の to_k·to_v (544-545, 入力 d_ctx) は to_q (入力 d_norm) と独立。
  - VAE resnet も同型 (vae_decode.cu:569-579)。
- **判断**: 電力 43%・帯域 11% の latency 律速下では埋め草として理屈は通るが、**launch 谷の実在量を nsys で直接観測してから** (G-1k 再判定と同じ前提・profile の測り方の注意参照)。G-8k (malloc 同期点の除去) が先行しないと stream 並列は同期点で潰される。効果見積りは現時点で不能。難度中〜高 (event 配線・検証コスト・fast 側隔離)。

### micro (単独 Pkg にしない小物)

- `src/infer/unet.cu:761-763` — time_ids を **毎 forward D2H 同期読み** (cudaMemcpy DeviceToHost)。step 間で不変なのでハンドル or 呼び出し側キャッシュで除去 (CFG 20step で 40 回の同期点)。
- `src/infer/unet.cu:934-937` 等 — 上記 G-8k の swap 化に含める。
- `src/infer/unet.cu:799-807` — push_skip が全 skip を D2D コピー (~40MB/forward)。所有権移動で削減可。~ms 級。
- `src/kernels/vae_decode.cu:382-438` — `k_conv2d_f32`/`launch_conv2d_f32` は resnet_block_f32 が GEMM 版へ移行済みで**未使用の死コード** (性能でなく保守の話)。

### 非発見の確認 (調査したが問題なしだったもの)

- `use_cublas` 閾値 (`src/kernels/gemm.cu:392-403`): M/N/K ≥ 16 で常に cuBLAS。自作 wmma/tiled に落ちるのは time embed の M=1(B) 級 GEMM (unet.cu:750-784) のみで、面積は誤差帯。**閾値の付け直しに旨味なし**。
- conv2d の direct フォールバック対象 (UNet conv_out Cout=4 / VAE conv_out Cout=3 / post_quant_conv): FLOP が小さく direct 実測 1807 GFLOPS で ~ms 級。放置で可。
- VAE up2/up3 の FP32 中間: 「本当に FP32 が要るか」の根拠は**コードに実測として明記あり** (vae_decode.cu:9-20)。撤去は不可、緩和は G-9k-4 (BF16 gated) のみ。

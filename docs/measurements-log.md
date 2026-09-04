# dollama 計測ベースライン詳細ログ

CLAUDE.md から退避した完全版。CLAUDE.md には要点のみ残し、各行の詳細経緯・条件・seed sweep 結果はここに保持する。
出典 spec: `docs/roadmap.md` / `docs/training-spec.md` / `docs/dataset-spec.md`。

## 計測ベースライン

| 指標 | 値 | probe |
|---|---|---|
| CPU→VRAM (10MB) | 0.76ms | probe2 |
| CPU→VRAM (100MB) | 3.46ms / 30.3 GB/s | probe2 |
| NPU推論 (512dim MLP, 静的形状) | 0.88ms | probe2 |
| NPU出力 (2048B) → GPU | 0.031ms (3.4%) | probe2 |
| iGPU VAE decode stub | 995ms (CPU 126ms → RTX5080 採用) | probe4 |
| NPU→iGPU ゼロコピー差分 (231KB) | 0.158ms (誤差範囲) | probe4 |
| system RAM → RTX5080 latent (256KB) | 0.030ms / 8.7 GB/s | probe4 |
| system RAM → RTX5080 image (12MB) | 0.254ms / 49.6 GB/s | probe4 |
| iGPU VAE encode (1024→128, img2img) | **79ms** (CPU 117ms → iGPU 採用) | probe5 |
| Qwen2-1.5B INT4 CPU tok/s | 64-71 tok/s | probe7 |
| Qwen2-1.5B INT4 ロード時間 | 1.1s | probe7 |
| WD14 SwinV2 (448×448): CPU / iGPU / NPU | 101ms / 104ms / 268ms → **CPU 採用** | probe8 |
| CLIP-L text encoder (77token): CPU / iGPU / NPU | 20ms / 14ms / **7.85ms** → **NPU 採用** | probe9 |
| CLIP-L NPU (C++ ClipEncoder, 中央値/N=100) | **7.82ms** (min 7.61 / max 12.15) | test_clip |
| WD14 (C++ CPU Wd14Tagger, 中央値/N=100) | **105.3ms** (min 99.1 / max 132.8) | test_wd14 |
| SDXL 20steps 1024×1024 RTX5080 | **3.80s** / 5.3 it/s / VRAM ピーク 10.49GB | probe10 |
| Pipeline 縦通し (stub→CLIP NPU→queue→WD14 CPU, C++ jthread) | **9.13 frames/s** / per_frame 109ms / 単発レイテンシ中央値 157ms (WD14 CPU 律速) | test_pipeline |
| マルチフレーム並列 計測クローズ (Phase 4-(1), `src/core/multi_frame_pipeline.hpp` MultiFramePipeline = pipeline.hpp 4 段 A→B-CLIP→B-SDXL→C を std::function 注入で汎用化した「複数フレーム先行生成」駆動側, sleep スタブで実デバイス比率を再現・OV/CUDA 非依存ゆえ開発機で SKIP なし実走, GTX1080Ti/i7-10700) | **現設計の並列は既に最適 = look-ahead 2 段で GPU 飢餓なしを実測で確定**。実寸スタブ (LM 400 / CLIP 8 / SDXL 3800 / WD14 100 ms, N=8): **0.258 frames/s** (per_frame 3879.8ms ≈ SDXL 単段 3800ms = 理論 GPU 上限 0.263 fps の **98%** = **GPU バウンド**)。**GPU 飢餓検出 (核心)**: `queue_bclip_to_bsdxl` (B-SDXL の入力待ち) 中央値 **0.0002ms ≈0** → 先行生成が GPU を飢えさせていない・**LM (stage_a 404ms) は SDXL の裏に完全隠蔽** (per_frame ≪ 直列合計 4308ms)。**QueueDepth {2,4,8} スイープ**: fps 0.2581/0.2580/0.2574 (実寸)・縮尺 2.495/2.502/2.506 で **2 で飽和** (相対ばらつき ≤0.43%) → 深さを積む価値なし・**look-ahead 既定 2 が最適**。**結論: roadmap L223 Tier 2(A) 独立 forward ワーカーの「発動条件 (LM 段ボトルネック化)」は単一 GPU 構成では成立せず** (stage A は飢餓を起こさず隠蔽済) = 計測で留保継続を裏取り。SDXL がライブラリ fallback 等で桁違いに速くなった世界でのみ再評価。`QueueDepth` 非型テンプレ param 追加 (既定 2 で完全非回帰)・PipelineStats/dump_stats/4 段構造/SPSC 契約は無改変。test 42/42 全緑 | test_multi_frame_pipeline |
| compose_prompt (C++ CharacterBible, 1M iters) | **242 ns/op** | test_character |
| CharacterBible::find (10,000体, 1M lookups) | **10.5 ns/op** | test_character |
| vector_add 疎通ベンチ (N=16.7M, H2D×2+D2H, pageable, RTX5080) | **14.9ms 中央値 / 13.5 GB/s** (pinned 化で probe2 30GB/s 帯に上がる余地) | test_cuda_smoke |
| 自作 FP16 GEMM 1024³ (shared-mem タイリング, FP32 蓄積, RTX5080) | **0.45ms 中央値 / 4730 GFLOPS** (max_rel 5e-4) | test_gemm |
| 自作 FP16 GEMM SDXL Linear transB (M=4096 N=K=1280) | **3.19ms / 4208 GFLOPS** | test_gemm |
| 自作 活性化 SiLU/GeLU(erf) FFN (4096×5120, FP32 内部, RTX5080) | **544 GB/s** (UNet FM は起動律速で ~230 GB/s) | test_activation |
| 自作 GroupNorm (1グループ=1ブロック, 1パス FP32 リダクション, RTX5080) | UNet C1280 **73 GB/s** / C640 75 GB/s / VAE FM C128H512 48 GB/s (占有率制約) | test_groupnorm |
| 自作 GroupNorm multi-block 化 (FAST G-4k S1a = G-11k 吸収, `launch_group_norm_mb` partial→finalize→normalize 3 カーネル 2 段決定的集約・atomic 禁止, `epilogue` フラグを Scratch 経由で unet.cu resnet norm1/norm2 + conv_norm_out に配線・default(off) は 1-block GN byte-for-byte 無改変, RTX5080, 2026-07-11) | GN 帯域 unet_320_128 **444.6 GB/s** ≥300 ゲート PASS (1-block 105 → **4.2x**)・B2 527 / 640_64 334 / 1280_32 226 GB/s。parity MAE ~3e-7 / max_abs 0.0039 PASS・bitexact 3 runs 完全ビット一致 PASS。UNet 結線 (warm): epilogue vs default **SSIM 0.999999** ≥0.9999 / bad=0 PASS・fast vs default **bit-exact 維持** (MAE=0/SSIM=1)・default vs golden SSIM 0.999996 無改変。1step warm 中央値 (最終検証 2 走): default 479.0/483.8ms / fast 440.8/412.9ms / fast+epilogue **436.2/414.3ms** — epilogue vs fast の差分 −4.6ms/+1.4ms = **run-to-run ノイズ内・回帰なし** (GN 置換のみでパス数不変のため想定通り。パリティ系は 2 走とも完全同値で決定的。resnet ≤0.95s 合否は S1b/S2 後の再 profile で判定) | test_groupnorm / test_unet_fast |
| 自作 Conv2d direct (1スレッド=出力1画素, FP32 蓄積, RTX5080) | UNet C320 64² 3×3 **4.18ms / 1807 GFLOPS** / VAE C128 512² 3×3 **46.4ms / 1667 GFLOPS** (1×1=GEMM・3×3=タイリング昇格余地) | test_conv2d |
| 自作 Attention (per-(b,h,row) block + shared scores, FP32 softmax, RTX5080) | self 1024² Dh80 **1.65ms / 1631 GFLOPS** / cross Sk77 Dh80 **0.116ms / 1748 GFLOPS** (flash/Tensor Core 委譲は後続) | test_attention |
| safetensors ローダー (ヘッダオンリー, 自作最小 JSON パーサ, ifstream 全読み) | golden 5テンソル (F32/F16/BF16/I64/I8) ロード **19.0 µs/op** (N=10000) | test_safetensors |
| 自作 VAE decoder 全段 (SDXL AutoencoderKL, latent[1,4,128,128]→image[1,3,1024,1024], RTX5080) | final **SSIM 0.999992** / MAE 5.45e-4 / decode **中央値 7.96s** (cudaEvent N=7)。up2 以降は FP32 中間 (FP16 中間は up3 で Inf→GroupNorm NaN 伝播のため不可)。naive attention(Sq=Sk=16384)+ direct conv(1024²)が律速で最適化余地大 | test_vae_decode |
| 自作 SDXL UNet 全段 (UNet2DConditionModel, latent[1,4,128,128]+CLIP[1,77,2048]+added_cond → noise_pred[1,4,128,128], RTX5080) | final noise_pred **SSIM 0.999998** / MAE 7.40e-5 / 24段ゴールデン全緑 (time/add embed〜各down/mid/up〜conv_out, corr≥0.999997)。中間は全段 FP16 で可 (VAE と違い FP16 範囲内)。1step **中央値 ~9.2s** (cudaEvent N=7、direct conv + per-row attention 律速・Tensor Core/im2col-GEMM/flash 化が最適化余地)。重み 2.57B params (FP16 5.1GB) | test_unet |
| 自作 EulerDiscreteScheduler (SDXL, scaled_linear betas, timestep_spacing=leading, 20steps) | sigmas max_err 4.77e-6 / timesteps 完全一致 / scale_model_input 1.79e-7 / step0 1.43e-6 (diffusers golden 突合) | test_scheduler |
| 自作 LayerNorm / GEGLU / sinusoidal time embed / broadcast bias add (UNet 補助カーネル, RTX5080) | LayerNorm 4096×1280 **0.043ms / 486 GB/s** / GEGLU 4096×2560 **0.055ms / 1138 GB/s** / time embed dim320 0.011ms / bias_add rowvec 4096×1280 0.071ms (max_rel<2e-3) | test_layernorm/geglu/timeembed/bias_add |
| HTTP サーバー (Phase 3, cpp-httplib 0.47.0 + nlohmann/json 3.12.0, ヘッダオンリー wrap) | スタブ生成 + HTTP 往復 (256×192) **2.11ms** (PNG 147KB, ローカル自己リクエスト)。OpenAI Images 互換 `/v1/images/generations` で PNG base64 返却。生成本体は `IImageGenerator` 越し (2-6a で PipelineGenerator 差し替え可能・main.cpp フォールバック DI) | test_http |
| PNG メタ往復 (character-bible-spec §7, tEXt "dollama/bible", ensure_ascii JSON, IHDR/IDAT 非破壊で IEND 直前挿入, 純 C++) | embed **1961 ns/op** / read **4376 ns/op** (N=10000)。CharacterIdentity/SceneSpec/OutputSpec ⇔ §7 JSON ⇔ PNG。日本語 name 往復・非PNG/壊れlength/切り詰めを境界外読みなしで弾く | test_png_meta |
| 自作 BitNet b1.58 モデル定義 (Phase 4-2, decoder-only LLaMA系, BitLinear ternary+int8 / RMSNorm / RoPE / SwiGLU / causal attn / embed tied, 純ホスト参照 forward, GTX1080Ti 開発機 GCC) | d_model=512/n_layers=8/n_heads=8/ffn=1792/vocab=4999/max_seq=64 = **32.98M params**。forward 中央値 **~18,982 ms/forward** (seq=32, N=5。naive 参照: BitLinear 毎回 re-quantize + lm_head 4999×512 全走査。速度目標なし・4-5 CUDA ternary GEMM / 4-6 推論のゴールデン基準)。test_bitnet 8 サブテスト全緑 (param 範囲/ternary/int8/RMSNorm/embed tied/決定的 forward/logit 健全性) |
| 自作トークナイザー (Phase 4-3, vocab.json 駆動・タグ単位完全一致・ヘッダオンリー純ホスト C++, GTX1080Ti 開発機 GCC) | encode **365 ns/op** / decode **168 ns/op** / encode_text(greedy 最長一致) **2655 ns/op** (中央値, warmup後)。specials id 0..4 + tags[i].id==5+i 検証ロード (総語彙 4999)。§6 正規化 (`long_hair`→`long hair`・顔文字 `_` 保持)。実 pairs 5,000行/77,195タグ **UNK 0**・encode→decode 往復完全一致。test_tokenizer 全緑 |
| 自作 dense タグ生成 LM 訓練 (Phase 4-4, scripts/train_bitnet.py, bitnet.hpp 等価 dense PyTorch 32.98M, hard CE・蒸留なし初版, GTX1080Ti FP32 6epoch) | val_loss 3.22→**2.41** 単調収束 / train_loss 1.32 / **top10 tag recall 0.777** (random 0.002 の 388x) / 訓練 **45.3s** (seed 20260620 で決定的・再現バイト一致)。params 32,976,896 = bitnet.hpp::param_count 完全一致。重み `data/bitnet/bitnet_dense{,_fp32}.safetensors` (FP16 62.9MB / FP32 125.8MB, [out,in] 規約 74 テンソル, gitignore・再生成可)。蒸留は D2/D4 (hard CE 混合・不採用) と D5 (soft-label KL・下行) で検証済 |
| 自作 soft-label KL 蒸留 (Phase 4-D5, scripts/train_bitnet.py `--distill-kl`, 案A=`cache/danbooru_posts.jsonl` 8200posts の prefix 条件付き共起 teacher・温度付き soft target, `L=α·T²·KL(teacher‖student)+(1−α)·hardCE`・target 区間マスク共有, GTX1080Ti FP32 seed 20260620) | **過学習を実際に抑制**: train-val gap #1 1.09 → D5-XL 0.27 / D5-L **−0.21 (消失)**、val_loss 反転 ep4→ep6 後退 (D2/D4 が両軸悪化させたのと対照)。だが最良 top10 recall 0.758 < #1 0.777 (soft 追従と one-hot val recall がトレードオフ・両取り不可) → **不採用・#1 6ep 本線維持**。α=0 で plain hardCE と全 epoch bitwise 一致 (非回帰)、dropout は eval forward 不変 (max abs diff 0.0)=golden 非回帰。別名 `bitnet_dense_kl*` 出力・本番重み/golden 無改変。小規模 split/early-stopping 運用で再利用価値ありの正の機構。training-spec §11 / dataset-spec §12.3 |
| 自作 外部教師 soft-label KL 蒸留 (Phase 4-D6 = 案c, scripts/train_bitnet.py `--distill-ext` + scripts/dollma_d6_teacher_cache.py, 外部教師 **TIPO-200M** (KBlueLeaf・apache-2.0) のタグ補完生成を自作4999vocabに完全一致写像=案b-tagset・OOV drop+再正規化 (保持率0.79・エントロピー0.85nats), 訓練時 TIPO 非ロード=position軸付き COO npz を順送り読込, KL plumbing は §11 流用, GTX1080Ti FP32 seed 20260620) | **不採用 (recall 上振れは seed ノイズと確定・#1 本線維持)**: 単一 seed では #1 を僅かに上回って見えた (α=0.1 T=1.5 で 0.7818) が、**seed 頑健性 sweep (20260620/1/42/7) で再現せず** — delta=D6−#1 が +0.0013/−0.0027/+0.0015/−0.0031 (符号反転・平均 **−0.0008**)、最大 |delta| 0.0031 は #1 自身の seed 分散 (sd 0.0020・range 0.0055) に埋もれる。方法論補正: epoch 最大 recall で統一すると原値 #1 best は 0.7787 (D6 0.7800 との差 +0.0013)。**再現する D6 効果は過学習抑制のみ** (gap 全 seed 1.07→0.19・~5.5x): D5 と同じ正則化ノブで one-hot val recall に非寄与 (soft 追従 vs recall トレードオフ)。小規模 split/early-stopping 運用で再利用価値あり。別名 `bitnet_dense_d6*`・sweep は scratch `_seedsweep/`・本番重み/golden 無改変。training-spec §12 / dataset-spec §12.4 |
| 自作 dense タグ生成 LM 推論 (Phase 4-6, src/infer/bitnet.hpp BitNetDenseInfer, safetensors FP32 ロード→forward+greedy デコード text→tags, CPU 純ホスト, GTX1080Ti/i7-10700 MSVC) | PyTorch golden 突合: logits seq 8/32/63 で max abs err **2.86e-6〜2.29e-5** / **corr 1.0**、greedy デコード **5/5 完全一致** (語彙外日本語含む)。RoPE NeoX・[out,in]・tied lm_head・eps を golden 一致で検証。1 forward (seq8) **~253ms** (naive 参照・lm_head 4999×512 全走査支配・GPU 版 #6-GPU で seq8 2.43ms=46.8x に短縮済・量子化は後段)。models/bitnet.hpp の ternary BitLinear 不使用・dense FP matmul double 蓄積。test_bitnet_infer 緑 (golden/重み不在時 [SKIP]) |
| 自作 dense タグ生成 LM 推論 GPU 版 (Phase 4-6-GPU, src/infer/bitnet_gpu.cu/.cuh BitNetGpuInfer, safetensors FP32 ロード→device 常駐→GPU forward+greedy デコード, RTX5080 sm_120, T1 CUDA カーネル + T2 クラス) | 純 FP32 で CPU 版 BitNetDenseInfer と数値一致: device forward は embed gather / RMSNorm / RoPE / causal SDPA / SwiGLU を static `__global__`・Linear/lm_head は **LM ローカル純 FP32 cuBLAS (CUBLAS_COMPUTE_32F・TF32 厳禁)**・リダクション系は double 蓄積で CPU(double) と桁合わせ。**本番重み突合**: logits seq 8/32/63 で max abs err **1.49e-6〜6.44e-6 / corr 1.0**、greedy `generate` 3/3・`generate_with_identity` 2/2 完全一致。**forward レイテンシ中央値 (warmup3/iters30, DOLLAMA_BENCH=1)**: seq8 **2.43ms** (CPU 113.78ms = **46.8x**) / seq32 **9.90ms** (456.87ms = 46.2x) / seq63 **10.29ms** (900.83ms = **87.5x**)。class と heavy include は `#ifndef __CUDACC__` で囲み .cu(nvcc/C++14)非汚染。test_bitnet_gpu 緑 (重み/GPU 不在時 [SKIP]) | test_bitnet_gpu |
| 自作 INT8 dense タグ生成 LM 推論 (Phase 4 圧縮実験, src/infer/bitnet_int8.hpp BitNetInt8Infer, 重みのみ per-row 対称INT8・ロード時量子化・射影8本のみ量子化/embed・lm_head(tied)・RMSNorm・RoPE・attn・softmax は FP32 据え置き, GTX1080Ti/i7-10700 CPU 純ホスト) | **量子化耐性が高い** (圧縮の研究軸であって目的ではない — CLAUDE.md 方針)。**INT8 logits vs FP32 golden**: seq8 max_abs_err 0.0493 / corr **0.999964**、seq32 0.2126 / 0.999944、seq63 0.4608 / **0.999873** (ハードゲート corr≥0.99 を**3桁上回る**)。**greedy 生成 vs gen_golden 全5ケース完全一致** (16/16・8/8・14/14・9/9・8/8 = トークン一致率 **1.0 (55/55)・EXACT 5/5**・語彙外日本語含む)。**フットプリント**: 射影8本 FP32 121,634,816 B → INT8重み+per-row scale 30,605,312 B = **74.84%減** (削減 91,029,504 B・残存比 0.2516≈1/4)。**seq8 forward レイテンシ**: INT8 **~152.9ms** vs FP32 ~393.3ms = **0.389x** (int8 内積 int64 蓄積が FP32 double dot より軽い・lm_head は両者 FP32 同条件)。dense FP32 経路 (`src/infer/bitnet.hpp`) は**無改変**で #6/A3 golden 非回帰確認済。`src/models/bitnet.hpp` に `quantize_weight_int8_perrow()`/`Int8RowQuant` 追加・活性は既存 `quantize_activation_int8` 流用。GPU INT8 / ternary GEMM (#5) / INT4 は別タスク。test_bitnet_int8 全サブテスト緑・全22テスト緑・Skipped 0 | test_bitnet_int8 |
| 自作 dense タグ生成 LM 推論 CPU AVX2 高速化 (Phase 4 #6-CPU-fast, src/infer/bitnet.hpp `forward_fast`/`linear_fast` + `attention_block_fast`/`ffn_block_fast`, float32 蓄積 + AVX2/FMA 明示 intrinsics `_mm256_fmadd_ps` + 末尾スカラ, i7-10700 Comet Lake = AVX2 あり/AVX-512 なし) | **「lm_head 律速」仮説を実測で否定**: `prof_bitnet`(`src/tests/prof_bitnet.cpp`・本番重み)で区間分解すると律速は `linear()`(double 蓄積・単スレッド・SIMD なし三重ループ)で **FFN ~67% + attention ~25% = ~92%**、lm_head は **~7.5%**。**デュアルパス**: `forward()`/`linear()`(double=golden 参照)を**完全無改変で温存**し、本番 `generate*()` を `forward_fast`(matmul のみ `linear_fast` 化・RoPE/softmax は double 据え置き・lm_head は generate 時 `last_only` で最終位置のみ)に差し替え。**forward 中央値 (本番重み)**: seq8 **54.9ms** (double 263ms = **4.79x**) / seq32 **187.7ms** (1038ms = **5.53x**) / seq63 **380.5ms** (2028ms = **5.33x**) = 単スレッド AVX2 で全 seq ~5x。**golden 非回帰**: logit golden corr 1.0 / greedy synthetic 5/5・identity 5/5 完全一致 / 新 double-vs-fast サブテスト max_abs ~5e-6・corr 1.0。MSVC は `/arch:AVX2` 前提・GCC/Clang は関数単位 `__attribute__((target("avx2,fma")))`・非対応環境は float32 スカラ fallback をコンパイル時選択。**Tier 2 はデータ駆動で (A) 独立 forward ワーカー方式に設計確定 + 実装見送り**(下行 cpu_topology ベンチ + roadmap L222)。test_bitnet_infer 緑・prof_bitnet は計測専用 exe | test_bitnet_infer / prof_bitnet |
| 自作 CPU トポロジ検出 + forward_fast スケーリングベンチ (Phase 4 #6-CPU-topo, `src/core/cpu_topology.hpp`=`GetLogicalProcessorInformationEx` で物理コア/HT兄弟/EfficiencyClass を実機申告・OS依存隔離・決め打ちマスク排除 + `src/tests/prof_cpu_topology.cpp`=affinity ピン留め計測, i7-10700 Comet Lake homogeneous 8C/16T) | Tier 2 設計確定の前提計測。**① per-thread クリーン基準**(単一論理 pin): seq8 51 / seq32 186 / seq63 364ms = affinity 未固定 Tier 1 値(54.9/187.7/380.5)と同帯 → **Tier 1 ~5x は実質1スレッド速度**と確定。**② 物理コア 1→N 独立 forward スループット**: N=5–7 で **~5x 飽和**・N=8 全載せは per-forward 膨張(seq32 186→455ms・帯域/OS競合律速)。**③ HT 係数**: 兄弟2本 = 別物理2本の **68–83%** → disjoint 物理コア割当が基本・HT 同居は非効率。**④ P/E 比**: homogeneous 機ゆえ N/A(ハイブリッド機=13/14世代 Raptor Lake・研究機 Arrow Lake no-HT は実機引き渡し)。`forward_fast` は const・重み read-only で**並列呼び出し安全**(共有 mutable なし)。**Tier 2 = (A) 独立 forward ワーカー**(linear 内分割でなく forward 単位を物理コア pin)に確定・発動は pipeline 複数フレーム先行生成の実装後(現状 LM 段 stub で駆動側なし)。`cpu_topology.hpp` は roadmap L221「HW 環境抽象化」のトポロジ検出基盤として採用。test_cpu_topology 緑(構造的健全性)・prof_cpu_topology は計測専用 exe | test_cpu_topology / prof_cpu_topology |
| 自作 同一性条件付きタグ生成 (Phase 4-A, character-bible を条件入力・2D 特化の核, 機構 (a-1) prompt prefix `<bos> identity <sep> scene <sep> target <eos>`・`<sep>` 2回流用で vocab/tokenizer/アーキ無変更, GTX1080Ti FP32) | A1 §13 データ 5000ペア (identity/scene 分離・identity⊆target=retention 100% 教師・validate 全件0・train/val identity 重複 0.253)。A2 #1+§13 混合 6epoch 訓練で **identity retention 0.947** (生成 target が入力 identity の約95%を再現・条件付け実証)・source 別 val recall synthetic 0.900/identity 0.946。A3 C++ `BitNetDenseInfer::generate_with_identity` + `CharacterBible.find->canonical_tags` 結線 (tokenizer/character 無改修)。identity golden 突合: logits corr 1.0 / greedy 5/5 完全一致。test_bitnet_infer 緑 |
| 自作 評価作り直し (Phase 4-C, scripts/dollma_make_eval_diverse.py + train_bitnet.py `--eval-only` + dollma_c_seedsweep{,_analyze}.py, diverse-val 1500×2 + greedy 生成 set-metrics + seed sweep, GTX1080Ti FP32/CPU) | テンプレ teacher-forcing recall@10 を **greedy 生成タグ集合 vs gold の set-F1/Jaccard** へ置換し、テンプレ外自然文 val (tags-stay-real・Pool A val 由来 / Pool B 未使用 post・リーク0・gold⊆vocab) で採点。**物差し変更で D5 判定が符号反転**: 旧 recall で最下位 (D5 0.667 < #1 0.7767) が新 diverse F1 で最上位 (diverse_a **0.2024** vs #1 0.18 / diverse_b **0.2192** vs 0.1921・precision 0.38–0.41 vs 0.25)。in-dist (pairs.val) は 3 者同点 ~0.47。seed sweep (4 seed・6ep paired) で D5−#1 の diverse F1/Jaccard delta **全 4 seed 正・各 seed paired bootstrap 95%CI が 0 を除外** (p<8e-3) = **小幅だが頑健**な実効果 (+0.009〜+0.012 F1・絶対値は #1 seed 分散帯以下で strict 判定(b)のみ不成立)。D6 の符号反転 seed ノイズ (§12.5) と対照的。**確定: recall@10 退役・diverse 生成 set-F1 を新主指標**。本番重み/golden/既存 val 無改変・eval_report は provenance 付き。test_dollma_eval_diverse 緑 / test_dollma_eval_setmetrics は要 torch。training-spec §13 / dataset-spec §14 |
| 自作 入力多様化パイロット (Phase 4-B, scripts/dollma_make_diverse_train.py + train_bitnet.py `--train-file` + dollma_b_seedsweep{,_analyze}.py, **Claude 著述 Replace 500** (タグ固定=tags-stay-real・総件数 4500 維持)・施策 C の diverse-val 物差し上で採点, GTX1080Ti FP32 seed 20260620) | 施策 C (§13) で据えた **diverse-val + 生成 set-F1** の上で入力(自然文)多様化を測る。**diverse 生成 macro F1 が #1 を大幅に上回る**: diverse_a 0.1800→**0.2675** (+0.0875) / diverse_b 0.1921→**0.3039** (+0.1118)・precision も改善 (a +0.030 / b +0.052)・in-dist pairs.val −0.009 で**退行なし**・legacy recall@10 ≈同値 (タグ集合同一)。**seed 頑健性 sweep (4 seed 20260620/21/42/7・6ep paired・n≈1500/seed)** で delta (B−#1) は **全 4 seed 正 (a) / 分散帯の 4–6 倍 (b) / 各 seed paired CI が 0 を除外 (c) すべて成立**: diverse_a F1 **+0.0957±0.0184** / diverse_b F1 **+0.1263±0.0214** (paired t 全 16 セル p<8e-142)。**D5 (§13・(b)不成立の +0.009 小幅)・D6 (§12.5・符号反転 seed ノイズ) と桁違いに大きく頑健**。**著者交絡を否定**: 旧 D2 (Qwen2 著述・旧 recall で過学習悪化と却下) を diverse-val 再採点 (B-0) しても同等に改善 (diverse_a **0.2701** / b **0.3134** vs #1 0.18/0.1921) → 「Claude train が Claude test に似て上がった」では説明不可 = 多様化そのものの効果。旧 proxy なら却下されたはずで **C の物差しなしには可視化されなかった** (依存連鎖 C→B 実証)。**本番重みは #1 据え置き・無改変** (B-pilot 重みは別名 `bitnet_dense_diverse_b{,_fp32}`・本線昇格は project-leader/ユーザー決裁・絶対値なお ~0.26–0.31 と低帯域 → B 件数拡大/A 実ペア増/D 容量増と束ねて再評価)。Python のため C++ meson test 対象外 (test_dollma_make_diverse_train.py 6/6 緑・`--train-file` 既定で #1 経路と bitwise 非回帰・本番非汚染スナップショット diff)。training-spec §14 / dataset-spec §15 |
| 自作 入力多様化 件数拡大 (Phase 4-B-2, Replace 500→2,000・**Claude 著述 2,000 + synthetic 2,500**・tags-stay-real・施策C diverse-val 物差し, GTX1080Ti FP32 seed 20260620) | パイロット 500→2,000 で diverse 生成 macro F1 が**単調拡大**: diverse_a 0.2675→**0.3212** / diverse_b 0.3039→**0.3670**・precision も改善 (a 0.282→0.336 / b 0.316→0.377)・in-dist pairs.val 0.4539 (−0.009 退行なし)・legacy recall@10 0.7702 (同帯)。params 32,976,896 一致・val_loss ep4 底 2.456・訓練 116.7s。**seed sweep (4 seed 20260620/21/42/7・6ep paired・n=1500/seed)** delta(B-2−#1) diverse_a F1 **+0.1472±0.0102** / diverse_b F1 **+0.1788±0.0029** (Jaccard +0.105/+0.131) = 500版 (+0.096/+0.126) の **~1.4–1.5x に拡大しつつ seed sd は縮小** (diverse_b F1 sd 0.0214→0.0029)・判定 3 軸 (a)全seed正 (b)分散帯 ~6.7–8x (c)全 paired CI が 0 除外 すべて成立・paired t≈35–47 p<1e-269。**入力多様化はスケール則 (件数増で単調に強まり頭打ちなし・in-dist 据え置きで out-of-template のみ伸びる汎化方向)** — D5 (小幅・(b)不成立)・D6 (符号反転 seed ノイズ) と桁違いに対照的。絶対値はなお diverse_a ~0.32 / diverse_b ~0.37 と低帯域ゆえ**本線昇格は未決** (A 実ペア増 / D 容量増と束ね再評価)。本番重み #1 据え置き・別名 `bitnet_dense_diverse_b2000{,_fp32}` (gitignore)・sweep は scratch `_seedsweep_b2000/`。training-spec §14.8 / dataset-spec §15.6 |
| 自作 入力多様化 件数拡大 (Phase 4-B-3, Replace 2,000→10,000・**Claude 著述 10,000 + synthetic 2,000 = 総 train 12,000**・P=2,500 post × k=4 variant・B-2 のスーパーセット・tags-stay-real・施策C diverse-val 物差し, GTX1080Ti FP32) | **スケール則は ~2,000 件で飽和** (B-2 の「頭打ちなし」を訂正)。`make` を `--n-posts`/`--k-per-post` に一般化 (k=1 で B-1/B-2 bitwise 非回帰・test 14/14緑)。**seed sweep 4 seed (20260620/21/42/7・6ep paired・n=1500/seed)** delta(B10k−#1) は全 set/metric で **判定 YES (seed 頑健)**: diverse_a F1 **+0.1411±0.0206** / diverse_b F1 **+0.1761±0.0199** (Jaccard +0.102/+0.133)・(a)全seed正 (b)#1分散帯sd超え (c)全 paired CI が 0 除外・paired t≈28–46。**だが 2,000→10,000 の 5 倍増は追加利得ゼロ**: delta は B-2 (+0.1472/+0.1788) から平坦 (seed 分散内・むしろ微減)、b 絶対値も 0.319→0.313 (a) / 0.361→0.359 (b) で頭打ち。3 点 (500/2,000/10,000) で **500→2,000 大幅・2,000→10,000 飽和**が確定 — B-2 の「単調・頭打ちなし」は 2 点外挿の誤り。**運用結論**: 入力多様化単体の伸びしろは ~2,000 で尽きる → 残低帯域 (diverse_a ~0.31 / diverse_b ~0.36) は **B 件数でなく A 実ペア増 / D 容量増**で取る・**B 著述を 2,000 超に積む価値は薄く `[B-merge-at-A]` の既定多様化は b2000 で足りる** (b10k 不要)。本線昇格決裁 (2026-06-24) 不変・本番重み #1 据え置き・別名 `bitnet_dense_diverse_b10k{,_fp32}` (gitignore)・sweep scratch `_seedsweep_b10k/`。**PC ハング (2026-06-25・seed 20260621 b-arm eval 中) から冪等再開 (`_results/*.npz` skip) で完走**。training-spec §14.9 / dataset-spec §15.8 |
| マッティング/切り抜き (透過 PNG, character-bible §3 出力層, 学習済みセグメンテーション OV グルー = CLIP/WD14 と同立ち位置, GTX1080Ti/i7-10700 開発機) | **ISNet-anime 採用** (probe 比較: CPU 1.14s vs BiRefNet 15.7s=13.8x・アニメ髪 soft α 最適 soft_px/perim 14.1 vs 4.6・Apache-2.0・176MB)。OV IR FP32/FP16 静的 `[1,3,1024,1024]→[1,1,1024,1024]` (前処理 img/255 のみ・ONNX↔OV-FP32 max abs err **6.1e-6**・OV CPU 583-721ms)。`src/infer/matting.hpp` Matter (device 引数・OV ガード・f32 厳密一致) + compose_rgba (soft α・二値化なし・ストレート α premultiply なし) + `encode_png_rgba8` (png.hpp, color_type=6)。golden 突合 IoU≥0.99/MAE≤1e-3 は **OV 有効ビルドで実走** (本開発機は OV C++ 無で [SKIP]・compose_rgba は実走緑)。test_matting/png 緑。**M-5 (NPU/iGPU HW 比較・matting_device 確定) ✅ 完了 = 下行**。**M-6 (生成器結線) ✅ 完了**: 純 cpp interface `IMatter` + OV 隔離 factory (`src/server/matter_runner.{hpp,cpp,_stub.cpp}`・`make_matter`) を `IDiffusionRunner` と対称に新設し、`matting_postprocess.hpp` の `encode_png_maybe_transparent` (matter null/mask サイズ不一致/例外で不透明 PNG フォールバック) を Txt2ImgGenerator/PipelineGenerator 両方の最終段に結線。所有権が 3 段 DI を跨ぐ問題は `IImageGenerator::set_matter` (既定 no-op) の後付け注入で解消。CLI `--no-matting` で OFF (既定 ON)・`cli_generate.hpp` が `DOLLAMA_MATTING_WEIGHTS`/`DOLLAMA_MATTING_DEVICE` (既定 `GPU.0`=iGPU) で `make_matter` 1 回。開発機 (OV 無→stub nullptr) で test_matting 新サブテスト 3 件緑 (透過 color_type=6・不透明 color_type=2・サイズ不一致フォールバック) + `dollama --prompt` 不透明 PNG 出力で非回帰確認。**研究機 (OV+CUDA, isnet-anime IR 配置済) で end-to-end 実走確認済** ✅: ① 実 Matter golden (CPU) IoU=1 / MAE 9.99e-09 / 決定的 ② `dollama --prompt` → `matting: GPU.0` ログ + **color_type=6 透過 PNG** 出力 (生成器→iGPU マッティング→α 抽出→透過 PNG の配管が通る・テスト golden 重みは非キャラゆえ α≈0 だが配管は正) ③ iGPU(GPU.0) 実推論を実アニメ入力で検証 = **108.8ms** (M-5 99.96ms 一致域・CPU 261ms / 同一 mask)・mask range[0,1]・前景 25.4%・soft 中間α 4.6% = 意味のあるキャラ切り抜き。意味のある透過キャラ出力には実 SDXL 重み配置 (`DOLLAMA_UNET_WEIGHTS`) が前提 (配管は完了) | test_matting |
| マッティング ISNet-anime HW 比較 (Phase 4 M-5, `scripts/dollma_probe_matting_device.py`, isnetis.onnx を OV IR 静的化 `[1,3,1024,1024]→[1,1,1024,1024]` → CPU/iGPU(Xe)/RTX5080/NPU 比較, 研究機 Ultra9 285K/iGPU Xe/NPU AI Boost/RTX5080・OV2026.2) | 中央値: **iGPU(Xe) 99.96ms** < NPU 142.96 < CPU 204.20 < RTX5080(OV) 220.47。**iGPU/CPU 0.49x** (iGPU が CPU の ~2.0x 速)・NPU/CPU 0.70x。**matting_device = "iGPU" 確定**。ISNet は純 conv-UNet ゆえ **NPU も 1024² で compile/推論成功** (WD14 268ms の Window Attention と対照・純 conv は NPU フレンドリー) だが iGPU が最速。RTX5080 は OV plugin オーバヘッドで遅く本来拡散専任 (3.8s/VRAM 10.49GB) ゆえ除外。マッティングは拡散後段・1 枚 1 回で iGPU の VAE encode とも時間的非競合。レイテンシは重み/入力値非依存。`scripts/_matting_device_report.json` | (probe) M-5 |
| 品質スコアラ (Model B) NPU 実行性 probe (Phase 4 B, roadmap リスク「B の NPU 実行性」回収, `scripts/dollma_probe_quality_scorer.py`, 純 conv 代表 backbone ResNet-18 級 11.18M・attention 皆無を OV IR 静的化 `[1,3,H,W]` → CPU/iGPU(Xe)/RTX5080/NPU 比較, RTX5080 研究機/OV2026.2) | **NPU 最速・scorer_device=NPU 妥当**。中央値: 448² で **NPU 4.62ms** < iGPU 5.48 < CPU 8.35 < RTX5080(OV) 11.51 / 512² で **NPU 6.14ms** < iGPU 6.63 < CPU 10.54 < RTX5080 13.0。**NPU/CPU = 0.55〜0.58x (NPU が CPU の ~1.8x 速)**。**WD14 が NPU 268ms だったのは conv でなく Window Attention 由来と切り分け確定** — 純 conv なら NPU フレンドリー (WD14 比 ~58x)。拡散中 NPU 遊休に並列で実質ゼロコスト採点 → F 品質 FB ループに理想的。**前提: 実 §11 スコアラを純 conv で設計すること** (attention head を足すと NPU 不利に戻る恐れ)。RTX5080 が OV で遅いのは plugin オーバヘッド+小モデルゆえ (本来拡散専任で無問題)。レイテンシは重み値非依存 (乱数重み)。`scripts/_quality_scorer_report.json` | (probe) |
| 品質スコアラ ScorerNet 蒸留訓練 (Phase 4 Model B / B-3b, `scripts/train_scorer.py`, probe の純 conv backbone (ResNet-18 級 11.18M・attention 皆無) を昇格・出力 [1,1+8] (index0=quality / index1..8=解剖8軸 AnomalyAxis 1:1), E-1 研究機生成コーパス (実 SDXL 生成 PNG 180枚=train162/val18・axis=WD14 max-sigmoid soft ラベル) を蒸留, GTX1080Ti/i7-10700 CPU FP32 seed 20260620) | **E-1 のつづき (§16.7 手順4) を実走完了**。axis 8軸 = BCE soft-label 蒸留・**quality は全 null (美的モデル別ライセンス) ゆえ B head 自動凍結 (§16.8 既定)**。**train_loss ep0 0.0933 → ep5 0.00747 急収束** (axis_loss と一致・quality_loss 全 epoch 0.0=凍結期待通り)・val_loss ep1 底 0.00389→ep5 0.00495 (val 18件 small-data)。訓練 **95.0s** (CPU)。**params 11,181,129 = 11.18113M (probe 11.18M 一致)**・実 PNG 180枚すべて実ロード (PIL 12.2.0・合成フォールバックなし)・**同 seed 再実行で FP32 safetensors sha256 完全一致 (bitwise 決定性)**。出力 `data/scorer/scorer_net{,_fp32}.safetensors` (FP16 22.4MB / FP32 44.8MB・122 テンソル=BN running stats 含む・gitignore・再生成可) + `scorer_train_stats.json` (provenance)。sanity_reload 緑 (head.weight (9,512)・NaN/Inf なし)・test_dollma_train_scorer 9/9 緑。**OV/NPU 非接触** (B-3c OV 変換は研究機・別タスク)。**残: B-3c (OV 変換) → B-3d (C++ グルー `src/infer/quality_scorer.hpp`・clip/wd14 対称・OV 隔離) → B-3e (test) → B-5 (FB ループ)**。quality head は美的モデルのライセンス整理後に再訓練で有効化 (waifu-scorer-v4-beta 既定・§16.9)。dataset-spec §16.7/§16.8 | test_dollma_train_scorer |
| 品質スコアラ ScorerNet OV 変換 + 全 HW 実測 (Phase 4 Model B / B-3c, `scripts/dollma_convert_scorer.py`, B-3b 重み `scorer_net_fp32.safetensors` を ScorerNet にロード→`model.eval()`→`ov.convert_model(example_input=)` で OV IR 静的化 `[1,3,512,512]→[1,9]` FP32/FP16, 研究機 Ultra9 285/iGPU Xe/NPU AI Boost/RTX5080・OV2026.2) | **B の芯「純 conv スコアラが NPU に静的形状で載る」を実走実証**。**全 HW レイテンシ中央値 (512²・N=11・warmup3)**: iGPU(Xe) **6.28ms** < NPU **8.32ms** < CPU 11.70 < RTX5080(OV) 13.16。**NPU compile+推論成功・NPU/CPU 0.711x** (probe 4-B 予測 6.14ms/0.55x を再現方向・純 conv は NPU フレンドリー = WD14 window attention 268ms と対照)。**精度: PyTorch eval vs OV CPU FP32 max abs err 1.34e-05** (≤1e-4 ゲート PASS = `model.eval()` で BN running stats 焼き込み正常)・OV FP32 vs FP16 7.41e-03。出力 `[1,9]` f32 (index0=quality logit / index1..8=8軸 AnomalyAxis・`get_output_tensor(0)`・C++ は `ov::element::f32` 厳密一致=i32/i64 教訓)。**注意点**: PyTorch 変換出力にポート名なし→index アクセス、**RTX5080(GPU.1) は FP16 IR の convert kernel 選択失敗→FP32 にフォールバック** (intel_gpu プラグインの NVIDIA 経路・本来拡散専任ゆえ非重大)。成果物: `models/scorer-net/model_ov_fp32.xml`(.bin 44.7MB)/`model_ov_fp16.xml`(22.4MB・gitignore 再生成可) + golden `src/tests/data/scorer/golden_scorernet.safetensors` (input+logits・matting と同形式・tree) + meta json。**残: B-3d (C++ グルー `src/infer/quality_scorer.hpp`・要 OV 有効ビルド) → B-3e (golden 突合 test) → B-5 (FB ループ)**。dataset-spec §16.7 | (convert) B-3c |
| 品質スコアラ C++ 結線チェーン (Phase 4 Model B / B-3d→B-5-2, matting (M-6) の interface→postprocess→生成器結線 3 段と完全対称, GTX1080Ti/i7-10700 開発機 MSVC) | **B-3d** `src/infer/quality_scorer.hpp QualityScorer` = ScorerNet OV IR を C++ から推論するグルー (matting.hpp Matter 対称・device 既定 NPU・入力 `ov::element::f32` 厳密一致・logits[9]→`logits_to_result` sigmoid)。**B-3e** golden 突合 test (OV 有効ビルドで実走・開発機は OV 無で [SKIP])。**B-5-1** `src/server/scorer_runner.{hpp,cpp,_stub.cpp}` = 純cpp `IScorer` interface + `make_scorer(xml,device)` OV 隔離 factory (空/不在 xml→nullptr 契約・IMatter/make_matter 対称・`ScorerResult` は再定義せず quality_scorer.hpp のを再利用)。**B-5-2** `src/server/scoring_postprocess.hpp` = 純cpp 採点後処理グルー (matting_postprocess.hpp 対称): `score_image_safe(IScorer*,rgb,w,h)→ScoringOutcome{scored,result}` は scorer==nullptr/例外 (サイズ不一致の `std::invalid_argument` 含む) を握り未採点へ倒す (生成を絶対に失敗させない)・`collect_anomalous_axes(ScorerResult,thr=0.5)→vector<AxisFlag>` は axis[0..7] を `>=` 閾値で AnomalyAxis 順のまま soft 収集 (棄却なし・quality は B-3b head 凍結中ゆえ対象外・thr は load-bearing でないノブ)。fake IScorer で null/例外/成功/閾値境界/quality 非寄与の 5 ケースを開発機実走緑 (OV 非依存=SAC 影響なし)。**B-5-3** = 生成器結線 (matting の set_matter/M-6 と完全対称・**B-5 三段完了**): `generator.hpp IImageGenerator::set_scorer` 既定 no-op + Txt2Img/Pipeline で `scorer_` メンバ・`set_scorer` override・`generate()` 内 rgb 確定後/matting 合成前に `score_image_safe`→`collect_anomalous_axes`→`std::clog` (stderr) で `[scorer] quality=.. anomaly:..` をログ出力 (`axis_name` ヘルパ追加)。StubGenerator は no-op (不採点)。`cli_generate.hpp` が matting 注入直後に対称の `make_scorer(xml,device)`→`set_scorer` DI (env `DOLLAMA_SCORER_WEIGHTS` 既定 find_model_xml("scorer-net/model_ov_fp32.xml") / `DOLLAMA_SCORER_DEVICE` 既定 "NPU"=拡散中遊休 NPU で並列採点・iGPU は matting 専有ゆえ)。meson は scorer_runner(_stub).cpp を exe(main) と test_cli_generate にリンク (B-5-1 の「成果物3まで未登録」解除)。**採点結果は現状ログのみ** (消費者 F 未着手ゆえ GenResult 添付/HTTP API 露出は死にコード回避で見送り)。**開発機緑**: test_scoring_postprocess (axis_name 8 軸検証追加・実走) + dollama/test_cli_generate ビルド・リンク緑 (cli_generate 実走は CUDA+OV 新規 exe ゆえ SAC ブロック=既知環境制約)。**generate() 内採点ログ経路の実走は研究機 (OV+CUDA) deferred = M-6 生成器統合と同じ分担** (開発機は txt2img=HAVE_OPENVINO / pipeline=.cu ガードで非コンパイル)。**残: F 品質 FB ループ** (採点→報酬で LM fine-tune)。test_scorer_runner / test_scoring_postprocess 緑 | test_scoring_postprocess / test_cli_generate |
| 品質 FB ループ 計装 (Phase 4 施策 F / F-0a, 報酬関数 + rollout 収集ハーネス + 信号ゲートの位置づけ, C++ 非接触=Python のみ, `scripts/dollma_reward.py` + `scripts/dollma_collect_rollouts.py`, GTX1080Ti/i7-10700 開発機 Python 3.14) | **閉路を 1 周回す計装** (fine-tune 本体は F-0b 後送り): LM タグ生成 → SDXL 生成 → ScorerNet 採点 → 報酬 → JSONL 蓄積。**報酬設計 (PL 確定・anatomy のみ)**: `reward_from_scorer(axes,quality=None)` = **worst-axis 主** `-((1-w)*max(axes) + w*mean(axes))` (w=MEAN_TIE_WEIGHT 0.05)。2D は手 1 本の破綻で全損ゆえ max 集約 (scoring_postprocess.hpp collect_anomalous_axes の「異常度の高い軸を拾う」と同哲学)・mean は同点 (worst 等値) プロンプトの識別に従として極小重みで混ぜる。**凸結合ゆえ境界は正確**: 全 0 軸→**reward 0.0** (最良) / 全 1 軸→**-1.0** (最悪)。**quality head 前方互換**: ScorerNet index0=美的 quality は B-3b で凍結中 (anatomy 8 軸のみ訓練・null)。waifu_scorer_v4 は **apache-2.0 で採用可** (quality 有効化は実行ギャップでライセンスの壁ではない) ゆえ `quality` 引数を残し (現状 None・将来非 None で `(1-0.3)*anatomy + 0.3*(quality-1)` 経路発火)・JSONL は `quality:null` フィールド常在。**rollout JSONL スキーマ**: `{input_text, prompt, seed, axes:[8], quality:null, reward}`。**収集ハーネス** = gen_scorer_corpus の HTTP/サーバ lifecycle + train_bitnet greedy 生成を再利用・ScorerNet は B-3c 検証済 OV IR (`model_ov_fp32.xml` [1,3,512,512]→[1,9]・logit→sigmoid)・素プロンプト (品質ネガ非注入) で anatomy 分散を測る。**信号ゲートの位置づけ**: 収集後に「プロンプト間で reward がばらつくか/best−worst に学習可能なギャップがあるか」(= anatomy-only が学習信号になるか) を測れるよう reward/axes を JSONL に残し、provenance に reward min/max/mean/std/best−worst を出す。**実走は研究機 (gpu-benchmarker)・50-100 rollout**: 開発機は OV/torch/重み (ScorerNet IR・bitnet_dense_fp32・vocab) 不在ゆえ **明示 [SKIP] で早期 return** (NotImplementedError なし)。**開発機緑**: 純ヘルパ単体テスト 13/13 (reward 決定性・worst 支配・境界 0/-1・worst 単調・mean 同点処理・quality 前方互換・空 axes ValueError / sigmoid 数値安定・axes_from_logits index1..8・quality_from_logits 凍結 None・rollout_row スキーマ・build_input_texts seed 決定性・main PLAN 非例外)。**残: F-0b 報酬で LM fine-tune** (信号ゲート結果待ち)。 | test_dollma_reward_rollouts |
| 品質 FB ループ 信号ゲート実走 (Phase 4 施策 F / F-0a 実走, `scripts/dollma_collect_rollouts.py` を研究機で 80 rollout 実走 gpu-benchmarker 回収, ScorerNet OV IR `model_ov_fp32.xml` + `bitnet_dense_fp32.safetensors`, 素プロンプト・quality_enabled=false, RTX5080 研究機) | **判定 = 信号弱 (補強してから)**。80/80 完走・異常なし。**reward 分布**: min −0.2032 / max −0.0 / mean −0.0253 / **std 0.0377** / **best−worst 0.2031** (PL 信号あり閾値 std>0.1 かつ best−worst>0.3 の両方に未達)・p50 −0.0119・|r|>0.05 は 13/80・>0.1 は 5/80。全て [−0.20, 0] で −1 飽和帯ではない (=SDXL 天井/checkpoint エスカレーションではない)。**axis 別発火 (across 80, AnomalyAxis 順)**: worst-axis argmax は **Limbs 77 / Hands 2 / Head 1** = 実質 Limbs 単軸。Limbs max0.2125 mean0.0264 発火 (>0.05) 13/80。他 7 軸は全滅 (Hands/Eyes/Ears/Mouth/Digits/GlobalAnat max<0.002・Head max0.0118 で >0.05 は 0)。= ScorerNet の dynamic range が Limbs のみ生存 = **anatomy=Limbs 状態 (B 側の分解能不足)**。**題材層別**: 入力意味 (easy 単独 vs hard 多人数/手/foreshortening) では **分離せず** (mean|r| 0.0260 vs 0.0246)。理由: bitnet 生成 prompt が入力意味を反映せず (例「two girls hugging」→「1girl solo robot mecha science fiction」)、reward は入力題材でなく **生成 prompt 内容**に駆動。**生成 prompt の clean vs clutter で分離**: clean tag 列 (n12) mean|r|0.0070 vs clutter (n68・science fiction/mecha/chinese text/2girls/blood 等) mean|r|0.0285 = **4倍**。**画像照合 (worst/best 帯)**: best 000006 (r≈0) = 単独キャラ白ドレス全身・解剖正常 → 正しく 0。worst 000052 (r−0.203) = mecha bodysuit+blood+文字焼き込み+背景クラッタで **腕/手自体は破綻せず** → Limbs 高得点は機械アーマー/クラッタ誤検出が主。000070 (r−0.143) = 多人数が軍用車に群がる背景過多で **四肢が実際に溶けており発火妥当** だがスコープ外題材。**見立て**: 弱さは (a) 7 死軸 = B 側分解能不足 + (b) worst 帯がスコープ外 (背景/多人数/mecha/文字) への発火に confound の両方。生成側は単独素直題材では概ね無難 (大崩れしない)。**go/no-go 提言 = F-0b(SFT) 保留・最小補強を先に (順序付き)**: ① **quality head 有効化 (Q-2, waifu 再正規化 or deepghs 合流)** で生存中の直交軸を reward に足す (最小コスト・最大レバー・現 quality=null で無効) ② **7 死軸の分解能診断** (B 側 ScorerNet: anatomy が Limbs 超えになれるか判定) ③ 補強後に同 80 prompt で F-0a smoke 再走し std→>0.1 or clean/clutter 分離維持を確認 → 立てば **F-0b SFT**。飽和帯ではないので checkpoint エスカレーションは不要。clean/clutter 4倍分離は弱いが本物の勾配源ゆえ、quality 合流で信号が立つ公算。 | (dollma_collect_rollouts F-0a 実走) |
| 品質スコアラ quality ラベル実採点 (Phase 4 Model B / Q-1, `scripts/dollma_score_quality_v4.py`, waifu-scorer-v4-beta apache-2.0 を E-1 コーパス 180枚に実走, RTX5080 cu128 + open_clip ViT-L-14-quickgelu openai) | **quality=null は「ライセンスの壁」でなく「実行ギャップ」だったと実証・Q-1 で解消**。CLIP ViT-L/14 image embed[768]→L2 正規化 (aesthetic-predictor `normalized` 相当)→waifu MLP (768→2048→512→256→128→32→1・BatchNorm 入り・model.safetensors header で strict 一致)→生スコア→/10 クランプ (`WaifuScorerV4Provider.normalize_score`)。transformers 経路は sklearn DLL が SAC ブロックで不可ゆえ open_clip 純 torch で回避。**分布 (N=180)**: raw min/med/max −1.71/0.61/1.78 (mean 0.74 std 0.81)・**quality_norm 0.0/0.061/0.178 (mean 0.077 std 0.0753)**・**非退化** (variance あり) だが **[0,0.18] に強圧縮** (raw 負 28/180 が 0 にクランプ)・histogram [0,0.1) 108 / [0.1,0.2) 72 のみ。good/bad 分離 +0.0102 (弱・正方向)。**原因**: 対象が base-SDXL-1.0 キャラ生成 = アニメ美的スコアラ (danbooru masterpiece 基準) が正当に低評価 (memory「素 SDXL 1.0 が天井」と整合)。**Q-2 判定: 続行可だが要再正規化** — `quality` は /10 正準写像で充填済だが各行に `raw_waifu` を保持したので Q-2 は raw からパーセンタイル/min-max 再正規化で dynamic range 回復を推奨 (clamp で worst 28 枚の順序が潰れている)。出力 `data/scorer/scorer.{train,val}.jsonl` (quality 非 null・元は `.qnull.bak.jsonl` 退避) + `scorer_quality_report.json`。**deepghs(openrail) アンサンブルはユーザー本人承認待ちで未実行** (`quality_waifu` 列を残し後付け可)。dataset-spec §16.9 Q-1 | (score) Q-1 |
| 自作フル拡散パイプライン (タスク 2-6a, DiffusionPipeline = UNet×Nstep + Euler scheduler + VAE decode 結線, golden 埋め込み入力, CFG なし guidance=1, RTX5080) | 20step 1024×1024 実画像 初版 **84.07s** → **タスク 2-6 最適化後 11.30s** (probe10 3.80s 比 2.97x・累積 7.44x 改善)。最適化内訳は S2(conv im2col+wmma)/S3-0/A/B/C/D/E (下記 2-6 最適化 行)。最適化後の律速は UNet attention 4.60s / VAE decode 1.16s。出力 var=1244 / 全画素 [0,255] / NaN・Inf なし。VAE 画像化 (x*0.5+0.5→clamp→×255)。UNet 5.1GB+VAE 同時常駐可 (空き 14.9GB)。2step smoke が CI 緑判定・20step は DOLLAMA_BENCH=1 で計測のみ。★**採取条件の明示 (2026-09-04 追記・一次証拠で裏取り)**: この 11.30s は **非 CFG (guidance=1.0)・B=1・`DiffusionPipeline::generate()` 経路・2026-06-22 時点**の値である。① `src/tests/test_diffusion.cu:159` が `pipe.generate(/*steps=*/20, /*seed=*/1234ULL, rgb20, w20, h20)` = 4 引数オーバーロード (`src/infer/diffusion.cuh:87`) を呼び、`src/infer/diffusion.cu:521-528` が `generate(steps, seed, 1.0f, ...)` へ委譲する ② `src/infer/diffusion.cu:544-550` は `guidance_scale != 1.0` を throw する (「CFG は generate_txt2img を使う」) ③ `generate` 内の UNet 呼出は `launch_unet(...)` (同 `:616`) で、`src/infer/unet.cu:1461-1463` が `launch_unet_impl(w, 1, ...)` = **B=1 固定** ④ 日付は `4853bf2` (2026-06-22) 「最適化を 84.07s→11.30s でクローズ記録 (S3-E まで)」。★**本番 txt2img は `generate_txt2img` = CFG・B=2 の別経路** (`src/server/diffusion_runner.cu:65-75` = 出荷 CLI/HTTP が呼ぶのはこちら) なので、**11.30s を本番経路の数字として引用してはいけない**。★**さらに pre-G-2k (2026-07-09) / pre-G-3k (2026-07-06) / pre-G-4k (2026-07-28) / pre-G-8k (2026-08-19) = 陳腐化・要再測**。★**再測していないので値は置き換えていない** (残債④と同じ方針)。 | test_diffusion |
| 自作 本番 txt2img (タスク 2-6b, prompt→画像の本結線, SDXL dual text encoder + CFG, NPU CLIP-L/bigG + RTX5080 拡散) | NPU で CLIP-L penult 768 ++ bigG penult 1280 = 2048 concat / bigG pooled 1280 = text_embeds、CFG guidance 7.5 で各 step cond/uncond の UNet 2 回 → host 合成。20step 1024² 実画像 PNG **var=2300** / 全画素 [0,255] / NaN・Inf なし。`IDiffusionRunner` (純cpp) を境界に OV (`Txt2ImgGenerator`/.cpp) と CUDA (`DiffusionRunner`/.cu) を隔離。main DI 3段フォールバック (Txt2Img→Pipeline(golden)→Stub)。test_txt2img は実アセットで 20step 実走 (NPU 構築失敗時 CPU フォールバック)・無ければ [SKIP]。meson test 35/35 全緑 | test_txt2img / test_text_conditioner |
| 自作 同一性条件付き実ペア増 (Phase 4-A Phase1, scripts/dollma_make_identity_pairs.py `--out-tag`/`--exclude-post-ids`, danbooru タグメタのみ・画像非取得, seed 20260620・B 多様化非適用で A 単体) | 5k→**12k/25k** 生成。**a12k** train 10,800/val 1,200・**a25k** train 22,500/val 2,500、両者 teacher_retention **1.0**・vocab retention **1.0** (OOV 0)・tokenizer 往復 UNK 0 完全一致。**リーク 0 証跡** (frozen eval 1000 post_id 直読との train/all 交差 0・excluded 交差 0・train/val post_id&text disjoint)。**identity 重複率 (val基準)** 5k 0.253 → 12k 0.2871 → 25k 0.3003 と件数増で漸増 (val の ~70% は train 未見 identity)。25k は cache を 38000→55200 posts へ過去方向延伸 (fetch-factor 2.2)・12k は cache 充足で API 取得なし。本番重み/golden/A1 5k/凍結 eval/#1 本線 train/val 無改変・cache 着手前を `.preA` に退避。成果物は別名 a12k/a25k (gitignore)。dataset-spec §17 | dollma_make_identity_pairs |
| 自作 同一性条件付き実ペア増 評価クローズ (Phase 4-A Phase2, scripts/dollma_a_seedsweep.py + dollma_a_seedsweep_analyze.py, base(#1 plain) vs a(A2 `--identity` 混合) を a12k で 4 seed (20260620/20260621/42/7) 6ep paired・凍結 diverse-val 生成 set-F1 + identity retention の二物差し, GTX1080Ti FP32) | **A の効果は二分・diverse-val F1 ではなく retention と確定 (A クローズ)**。**① diverse-val 生成 F1/Jaccard = seed ノイズ (頑健でない)**: 4 set/metric (diverse_a/b × F1/Jaccard) すべて判定 NO — diverse_a F1 per-seed delta(a−base) = [−0.0118, −0.0358, **+0.0286 (seed42 反転)**, −0.0405]・across **−0.0149±0.0316** < #1 帯 sd 0.0221 (b NO)・符号不一貫 (a NO)・各 seed CI は 0 除外 (c YES) → **D6 と同型の seed ノイズ**・施策 B ~2,000 飽和とも整合。**② identity retention = 全 seed 頑健に 0.975**: a arm across-seed **0.9748±0.0010** (base ~0.576–0.631・n_cases=1200) = **identity 条件付けの機能基盤**。in-dist pairs.val F1 ≈ base・recall@10 0.78→0.84・a の val_loss < base。**a25k は回さず未使用保持** (12k が seed ノイズである以上 25k 反転安定化は見込み薄)。本番 #1/identity/golden/凍結 eval 無改変・出力は `_seedsweep_a12k/` のみ (gitignore)。**本番即時差し替えなし・`[B-merge-at-A]` で B(b2000)+A(identity) を出荷リトレイン 1 回でまとめ焼き**。training-spec §9.10 / dataset-spec §17.7 / roadmap Phase 4 | dollma_a_seedsweep |
| 自作 容量増 seed sweep (Phase 4-D, scripts/dollma_d_seedsweep.py + dollma_d_seedsweep_analyze.py, c33(33M) vs d80(80M `DOLLAMA_BITNET_ARCH=d80m`・N_LAYERS 8→16/FFN 1792→2464=79,908,864) を**両アーム同一レシピ** (b2000 多様化 ∧ a12k identity)・`--arch` だけ差で 4 seed (20260620/20260621/42/7) 6ep paired・凍結 diverse-val 生成 set-F1 主指標 + retention/in-dist ガードレール, GTX1080Ti FP32) | **陰性確定・80M 不採用・勝者 = c33(33M)**。**① diverse-val F1/Jaccard = seed ノイズ (全 4 set/metric 判定 NO)**: delta(d80−c33) diverse_a F1 per-seed = [−0.0240, −0.0047, −0.0008, **+0.0133 (seed7 反転)**]・across **−0.0040±0.0154** < c33 帯 sd 0.0114 (b NO)・seed 20260620 負/seed 7 正で符号反転 (a NO)・各 seed CI は 0 除外がバラける (c 一部のみ) → **A12k/D6 と同型の seed ノイズ**・施策 B ~2,000 飽和と整合。diverse_b F1 across −0.0021±0.0181 も同様。**② ガードレール側も 80M 不利**: retention c33 across 0.9778 (全 seed ≥0.975 ✅) vs d80 0.9741 (**3/4 seed 床割れ** 0.9744/0.9739/0.9711)・in-dist pairs.val F1 c33 0.4599 > d80 0.4564 (微退行)。**容量 (33M→80M) では diverse-val F1 は取れない (データ律速)** — 打ち切り基準どおり 80M 不出荷。forward ~2x の対価に見合う実利なし。**施策 B (~2,000 飽和) のみが diverse-val F1 を頑健に押し上げ、A(retention 専)/D(陰性)/蒸留4路線(非寄与) は F1 非寄与**と確定。残低帯域 (diverse_a ~0.31/diverse_b ~0.36) は容量/件数でなく別軸 (多様性の質・アーキ・損失・本命 F) で取る。正典化は `[B-merge-at-A]` で勝者 33M を 1 回まとめ焼き。本番重み/golden/凍結 eval 無改変・出力 `_seedsweep_d80m/` のみ (gitignore)・test 8/8 緑。training-spec §16 / dataset-spec §18 / roadmap Phase 4 | dollma_d_seedsweep |
| **G-8k S4 e2e アリーナ A/B (2026-08-19 研究機実走・SAC OFF, `src/tests/prof_arena_e2e.cu` 計測専用ハーネス, 同一プロセスで 1024² 20step CFG g=7.5 `--fast` 相当 (attn_fast+batch2+epilogue) を 3 枚連続 (★S5h 注記: 「同一プロセスで」は直前の**「3 枚連続」に掛かる** — 1 走行 = 1 プロセス = 1 構成 (env は起動時固定・構成切替ループ無し) であり、**1 プロセス内で 3 構成を切り替えた統制比較ではない**。熱・常駐が揃った基準線として G-9k/G-10k の効果見積りに流用しないこと。出典: `src/tests/prof_arena_e2e.cu:4` 「同一プロセスで 1024x1024 / 20step の CFG 生成を N 枚 (既定 3) 連続実行し」・`:8` 「走行間で変えるのは DOLLAMA_POOL / DOLLAMA_ARENA_RELEASE のみ」。S4 の生ログは残っていないため、この実行形態はハーネス構造からの演繹 (生ログ不在の詳細は同セル末尾の S5h 注記と残債⑤)) × 3 構成 × 2 ラウンド (順序 1→2→3 / 3→2→1), seed/prompt 固定, プロセス GPU peak = cudaMemGetInfo(total−free) を 5ms 周期サンプラで最大取り, 被験コード (S3 作業ツリー) は無改変)** (★S5h 注記: **本 S4 の生ログは repo に無い** — `docs/logs/` に退避済なのは S4b 分 (`g8k-s4b/`) のみで、本行の数値 (peak 13309/13310・16302MB / 捨て分 4637MiB / 秒 11s→25s 等) は生ログでの再検証ができない。repo に無いことは物理確認済みだが、研究機 temp 原本の有無は本機から検証不能。生ログ不在の正典リストは残債⑤) | **G4 (VRAM 主ゲート) FAIL = ハード停止**。プロセス GPU peak: **POOL=0 13309/13310MB** vs **POOL=1 既定 16302MB** vs **POOL=1+ARENA_RELEASE=1 16302MB** (2 ラウンド完全再現) → 許容 POOL=0+512MB=13821MB に対し **+2993MB 超過**・かつ 16302MB = **物理 total と同値 (free=0 に張り付き)** ゆえ真の要求量は計測不能な上振れ (weights 常駐 7067MB + arena cap 10688MiB ≈ 17.7GB) = **WDDM のホストページングで賄われている**。**G5 (絶対 peak ≤15.0GB) も FAIL** (15.92GB・cudaErrorMemoryAllocation は 0)。捨て分 (`total_capacity − peak_request_bytes`): **unet 10496MiB − 5914MiB = 4582MiB** (収束前の 1 枚目でも 9472−5914 = 3558MiB) / **unet_persist 192MiB − 137MiB = 55MiB** → 合計 **4637MiB**。原因は成長則でなく **GB 級単発要求のチャンク跨ぎ** (VAE decode の FP16 キャリー 512MiB×2・FP32 キャリー 1024MiB×2・up2/up3 の Scratch32 512MiB–1024MiB 級が並ぶ。刻み 256MiB に対し `max(刻み, 要求)` で 512MiB/1024MiB のチャンクが立ち、直前チャンクの残りが丸ごと捨てられる)。**G1 (2 枚目 step 2..20 の実 cudaMalloc/cudaFree = 0) PASS** (POOL=1 両構成): チャンク確保数が **step 数に不変** (steps=2 でも steps=20 でも 1 枚目 18 本・2 枚目 +1 本) = 成長は step1 と VAE decode に限局し step2..20 は 0。POOL=0 は構造上 **25,082 malloc + 25,082 free/枚 (≈1,254/step)** でこれがベースライン。**G2 (既定・2 枚目で chunk_alloc=0 かつ cuda_free=0) FAIL**: 2 枚目に **chunk_alloc=1 本 (+1024MiB, cap 9472→10496MiB)**・cuda_free=0。**収束は 3 枚目** (3 枚目 chunk_alloc=0/cuda_free=0) で S3 申し送り「収束は decode#3」と一致。**G3 (release ON で step ループ内 0 維持) PASS**: step 内 0 は維持。画像境界の malloc/free は **21/21 per image** (unet 18 + persist 3) で毎枚フル再構築 (characterization)。秒 (**characterization のみ・速くなったとは主張しない**): POOL=0 15.71/15.77/15.85/16.12/16.38/16.42 (中央値 **15.98s**) / POOL=1 既定 **1 枚目 10.96–11.44s だが 2 枚目以降 24.25–26.07s へ悪化** (VRAM 飽和後のページング) / POOL=1+release **10.95–11.70s で 3 枚とも安定**。機体は各走行開始時 SM 232–607MHz/40–47℃ の idle から入り終了時 2857–2880MHz/46–52℃・178W 以下で、ラウンド間ドリフトは POOL=0 中央値 15.77→16.12s = **+2.2%** に収まる (G-4k S3 の +9.9% ドリフトのような比較不能状態ではない)。**出力は 3 構成 6 走行 18 枚すべて同一 FNV-1a ハッシュ = ビット一致** (steps=2 の smoke も一致)。**結論: S3 の共有アリーナは既定でも release ON でも VRAM ゲートを通らない。VAE の GB 級キャリーを bump アリーナに載せる方式そのものの再設計 (専用固定バッファ / チャンク跨ぎの解消) が要る** — PL 決裁待ち | prof_arena_e2e |
| **G-8k S4b e2e アリーナ A/B 再走 (2026-08-19 研究機実走・SAC OFF・HEAD `acca803` ソース無改変, `build/src/prof_arena_e2e.exe` を再ビルドなしで使用, 7 走行すべて独立プロセス・フォアグラウンド・走行間クールダウンは **時刻の取れた 5 区間で実測 68-85s** (`smi_timeline.txt` の post→pre 間隔 = 71/71/68/70/85s。★`pre-A2`/`post-A2` には時刻が無く、**A-1→A-2(棄却)→A-2b の 2 区間は未検証**。再現時もこの帯を守ること — 起草時の「45-60s」は宣言値で一次証拠と食い違っていたため S5b で是正)、走行開始時の状態は **SM 180-1252MHz / 41-46℃** (★**S5d 追記: この 7 値は採用 7 走行と 1:1 対応しない**。`smi_timeline.txt` の pre エントリの内訳は **A-2(棄却)・A-2b・A-3・B-3・B-2・B-1・C** であり、**基準走行 A-1 の開始時状態は記録が存在しない**。VRAM の結論は熱に依存しないが**G6 の秒の議論は依存する**ので、基準側の条件は未記録と理解して読むこと) (**S5c 是正**: 「全走行が SM 180-555MHz の idle から入った」は誤り。`pre-B1` のみ **1252MHz** で他は 180/225/180/532/555/255MHz。**B ラウンドは A より遅い走行** (基準 16.41→17.06s) であり、その中の B-1 が唯一高いクロックから入っている = 条件は完全に均一ではない。丸めて隠さないこと), 共通 env `PROF_IMAGES=3 PROF_STEPS=20 PROF_G=7.5 PROF_FAST=1 PROF_SAMPLE_MS=5 DOLLAMA_PROFILE=1`, ラウンド A は順序 1→2→3 / ラウンド B は逆順 3→2→1 で熱ドリフトの向きを打ち消し, 補走 C は steps=2)** | **6 ハードゲート全 PASS = S3b/S3c の是正で S4 の FAIL を解消・G-8k S3 系クローズ**。構成別 (PEAK_USED / sec 1-3 枚目): 基準 `DOLLAMA_POOL=0` **13269 / 13299MB**・16.7/16.2/16.3 (A) ・17.1/17.0/17.0 (B) / **既定 (reserve ON) 13649 / 13638MB**・10.9/10.8/10.8 (A)・11.3/11.3/11.2 (B) / 対照 `DOLLAMA_ARENA_RESERVE_MB=0` (= S3 挙動) **16302MB (物理全量)**・12.0/**38.0**/**35.7** (A)・11.8/34.3/34.3 (B) / 補走 C (既定 steps=2) 13644MB・2.35/2.23/2.16。基準 peak・既定 peak の採り方 (**S5b で是正**): **基準に max (13299) を採ると許容枠が 13781 → 13811MB へ広がる = 被験側に有利**であり、「被験側に不利な方を採用した」という当初の記述は**誤り**。**真に保守的な組合せは 基準 min 13269 × 既定 max 13649 = delta +380MB / 余裕 132MB**(max 基準では +350MB / 余裕 162MB)。**どちらでも ≤512MB で G4 PASS = 結論は不変**。なお**余裕 132-162MB は同一構成の走行間ばらつき (基準 13269↔13299 = 30MB) の 4-5 倍しかない** — 枠が潤沢なわけではなく、以後の改修で数百 MB 級を積めばこのゲートは容易に落ちる。**ゲート**: **G0 出力不変 PASS** = 6 走行 18 枚すべて `rgb_hash=0x8a96690109d2b253` の 1 値 (補走 C は steps 違いで別値 `0xbfa9db2aab196421`・C 内 3 枚は一致) / **G1 step ループ内 malloc 撲滅 PASS** = 既定の 2/3 枚目で `d_cudaMalloc=d_cudaFree=0` (unet/persist とも)、補走 C(steps=2) の `cum_cudaMalloc` が A-2(steps=20) と同値 = **チャンク確保が step 数に不変** / **G2' 強化 PASS** = 既定は **1 枚目から `d_chunkAlloc=0`**・実 cudaMalloc は初期化 reserve のみ (**アリーナ 1 本あたり 1 回** = `unet`+`unet_persist` で **アリーナ由来は計 2 回**。★**「プロセス全体で 2 回」ではない** — 重みロードで別途 GB 級の cudaMalloc がある。出典: `docs/logs/g8k-s4b/s4b_roundA_1_pool0.log:4` の `used_after_weight_load=7067MB` (ctx 1317MB からの差分 **5750MB** が重み。★**S5e で出典を訂正**: S5d は既定走行の 13323MB を引いていたが、この値は reserve 6256MiB を含むため「重みの質量」の証拠にならない)) / **G4 VRAM 主ゲート PASS** = 13649 ≤ 13269+512 = 13781MB → **保守側 delta +380MB (余裕 132MB)**・max 基準なら +350MB (余裕 162MB) / **G5' 実害 PASS** = 13649MB ≤ 15360MB かつ peak 時 free **2653MB** ≥ 512MB / **G6 秒の実害 PASS** = 既定 2/3 枚目が POOL=0 比 **+5% 以内** (A/B 両ラウンド)・**S4 のページング事故 (11→25s) は再現せず**。★**G3 (`DOLLAMA_ARENA_RELEASE=1` 経路で step ループ内 0 維持) は S4b のゲート集合から外れている**: S4 は 3 構成目に `POOL=1+ARENA_RELEASE=1` を置いて G3 PASS を記録したが、S4b は 3 構成目を対照 `RESERVE_MB=0` に差し替えたため **7 走行すべて `DOLLAMA_ARENA_RELEASE` は未設定 (既定 OFF)**。つまり **S3c 以後、release 経路の e2e/VRAM 検証は行っていない**。**release は既定 OFF の debug スイッチであり、VRAM 節約目的で有効化しないこと** — S3b 期に reserve と併用したとき **PEAK_USED が 14250MB → 16302MB (物理張り付き) へ最悪化した実測**がある (出典: `src/infer/diffusion.cu:415-419` のコメント。現在は release 直後に reserve をやり直して回避しているが、その組合せは S4b で計測していない)。捨て分 (cap − live_peak): 既定 **205MiB** (unet 166 + persist 39) vs `RESERVE_MB=0` **4637MiB**。★**reserve 定数の一次証拠に不整合 (S5c で注記)**: `8e2e48d` (S3b) 本文は「UNet 6656MiB = S4 実測 live peak **6051MiB** +10%」と書くが、`88b3ae7` / `acca803` 本文および現物 `src/infer/diffusion.cu:202` (`kArenaLivePeakUnetMiB`) は live peak **5914**MiB である。6051 は 5914 + persist live peak 137 の**合算とみられる** (推測)。**再測時に参照すべきは 5914 (unet 単体)** で、6051 を unet の live peak として引かないこと。reserve 不足警告 **0 件**・全 7 本 exit=0・例外なし。**信頼性の担保**: ① 対照 A-3/B-3 で **S4 の病理 (peak 物理張り付き・peak 時 free 0MB・2 枚目以降が既定の約 3.2 倍) が同一バイナリで完全再現** → 基準側に疑義なし・reserve が効いている差分だと分離できた ② 熱ドリフトは A→B で基準 +4.0% / 既定 +3.6% と**同方向** ゆえ順序効果では説明できず G6 の結論は堅い ③ **metric の定義と、それが構成間比較に使える実証 (S5b で書き分け)**: `PEAK_USED` は `cudaMemGetInfo` の total−free = **デバイス全体 (device-wide) の使用量**で他プロセス分を含む = 厳密には「プロセス GPU peak」ではなく、プロセス分は分離できていない。**7 走行すべて `used_after_ctx=1317MB` で完全一致** (`total=16302MB` も一致) してはいるが、**これを「非プロセス分が一定」の証拠として読まないこと (S5c で後退)**: 同じ `smi_timeline.txt` の常駐値は走行途中で **1229-1235MiB → 1139MiB (~95MB 減)** と実際に動いており、device-wide の metric なら `used_after_ctx` も動くはずである。動いていない = **metric が鈍いだけの可能性**があり、**非プロセス分の変動は分離できていない**。★ただし **G4 の結論自体は変わらない** — 対照 A-3/B-3 で S4 の病理が同一バイナリで完全再現している (下記 ①) ことが構成間比較を別途支えている。秒は **characterization のみ (G-8k は秒数レバーではない・「速くなった」とは主張しない。秒数の本命は G-10k = conv 真 batch2)**、かつ熱ドリフトがあるため**絶対秒を性能主張の根拠にしない**。★**ただしこれを「秒に効いていない」と読まないこと (S5d で是正)**: 上表の **A/B (同一バイナリ・env だけを変えた構成間比較)** は **既定 10.8s vs キルスイッチ `DOLLAMA_POOL=0` 16.2s = −34%** (★**S5e で訂正: S5d はこれを「同一プロセス A/B」と書いていたが誤り** — S4b は本行冒頭のとおり**独立プロセス 7 本**・走行間 68-85s であり、**基準走行 A-1 の開始時状態は未記録**。★**S5f で再訂正**: S5e はここに「同一プロセス内で構成を切り替えたのは S4 の方」と書いたが**誤り** — **S4 も 1 走行 1 プロセス**で、構成は起動時 env で固定される。S4/S4b のハーネスは同一の `src/tests/prof_arena_e2e.cu` (git 履歴上 `88b3ae7` の 1 版のみ) で、冒頭コメント `:8` に「走行間で変えるのは DOLLAMA_POOL / DOLLAMA_ARENA_RELEASE のみ」、env は `:150-158` で `run_prof()` 冒頭に 1 度だけ読み `main()` に構成ループは無い (`DOLLAMA_POOL` の実効判定 `src/kernels/device_arena.cu:209-226` も static キャッシュで getenv は初回のみ)。★S4 の生ログは残っていないため、S4 の実行形態はこのハーネス構造からの演繹である。直前の S4 行の「同一プロセスで」は「**1 プロセスで 3 枚連続生成**」の意味 (「同一プロセス」は**枚数**に掛かる) であって構成切替ではない。「同一プロセス」と書くと熱・常駐が揃った実際より強い統制下の値と誤認され、G-9k/G-10k の効果見積りの基準線に流用されうる。なお結論自体は A/B 両ラウンドで再現し熱ドリフト +3.6〜4.0% の桁を大きく超えるため堅い) であり、G6 の「+5% 以内」は**片側ゲート**ゆえ嘘ではないが、要約文言だけを読むと実データと逆向きの結論に至る (独立の傍証: `4dee0b7` 本文の 1step warm **479.3ms(POOL=0) / 329.0ms(プール)**)。これは**旧 malloc 経路との比較**であって正典ハーネス (`test_diffusion_batch2` の DB2_BENCH) での再測を経ていないため**出荷性能値としては使わない** — 禁止しているのは出荷性能の主張であり実測の記述ではない。**この効きをゼロと見なして reserve 縮小やアリーナ廃止を提案しないこと** (下記「既知のトレードオフ」の +6.5GB は見返りゼロのコストではない)。**既知のトレードオフ**: 既定構成は peak が +350MB でも **定常 residency が 7135 → 13599MB (+6.5GB)** に増える (reserve を保持し続けるため) → peak ゲートは通るが**他プロセスと VRAM を分け合う運用では空きが常時 ~2.7GB**。**ログの誤読注意**: `DOLLAMA_POOL=0` の走行でも `[ALLOC] reserve: unet=6080MiB unet_persist=176MiB` の行が出る (**是正 (S5b・現物確認)**: `reserve_arenas()` (`src/infer/diffusion.cu:262-281`) 自身に pool 判定は無く、printf (274-280) は `device_arena_reserve()` 呼出 (270-271) の**後**にある。no-op 判定は callee 側の早期 return (`src/kernels/device_arena.cu:436-440`) で、caller からは成否が見えないため printf だけが出る、が正しい説明)。実体は no-op で終了時は `[ALLOC] arena=unet cap=0MiB reserved=0MiB chunks=0` → **判定は終了時の cap/reserved/chunks で行う** (printf 位置の是正は G-8k スコープ外・別タスクの宿題)。★**`88b3ae7` (S3 checkpoint) のコミット本文にある「S3b が緑になるまで merge 禁止」は、S3c + 本 S4b 全緑 (2026-08-19) をもって解除済み = 現在は無効**。git 履歴は書き換えないため当該文面は remote に残るが、正本は本行と `docs/fast-mode-plan.md` の G-8k 実装記録。**VRAM 計測手順 (S4 で恒久化決裁済・以後の VRAM 計測はこの手順のみ)**: GPU peak = `cudaMemGetInfo` の total−free を **5ms 周期の別スレッドでサンプリングして最大取り** (★**この値は device-wide** = 他プロセス分を含む。「プロセス GPU peak」と呼ばないこと。構成間比較に使う前提として、走行ごとに `used_after_ctx` が一致していることを必ず確認する)、ハーネスは `src/tests/prof_arena_e2e.cu` (**meson test には登録せず build ターゲットのまま**・`prof_unet_fast_warm` の前例に倣う)。★**生ログ全文は `docs/logs/g8k-s4b/` に repo 退避済 (S5d)** — `s4b_roundA_{1_pool0,2_default,3_reserve0}.log` / `s4b_roundB_{3_reserve0,2_default,1_pool0}.log` / `s4b_roundC_steps2_default.log` / `smi_timeline.txt` / `smi_before_A2.txt` の 9 本と、採用しない隔離分 `DISCARDED_A2_overlap_risk.log` の**計 10 本 = 原本ディレクトリの全量** (★S5d は `smi_before_A2.txt` を取りこぼしており S5e で追加。退避分は全 10 本とも原本と byte-identical)。読み方の注意は同ディレクトリの `README.md`。原本の所在は temp (`.../Temp/claude/e--Develop-Projects-dollama/a052f05e-4438-4868-9235-e41ffd700a76/scratchpad/`) だが**セッション ID 付きなので消える** — 以後の参照は repo 内パスを使うこと (S5c までは temp しか指しておらず、消えた時点で S4b 全緑が検証不能な自己申告に降格する状態だった) | prof_arena_e2e |
| **G-8k S6 (T2) 静的レビュー是正 F1〜F7 の実走再確認 (2026-08-22 研究機 `KIK-WIN-RTX58` 実走・SAC OFF。**是正前ツリー**で V1〜V6 を 2 ラウンド → 書き手と検査者を分けた相互 read-only レビューの指摘 12 件を是正した**最終ツリー**で V1 1 ラウンド + V5 を再確認。e2e ハーネスは S4b と同一の `src/tests/prof_arena_e2e.cu` = **1 走行 1 プロセス 1 構成** (構成切替ループは無い。S4/S4b 行の同名注記と同じ読み方をすること)。★**本行の数値の出典は研究機での実走報告であり、生ログは研究機ローカル `E:\Develop\logs\g8k-t2-verify\` (是正前・再現スクリプト `v1.sh`/`v5.sh`/`v2v3.sh`/`v3b.sh`/`v3c.sh`/`hog.py`/`hog_active.py` 同梱) と `E:\Develop\logs\g8k-t2-verify-final\` (最終ツリー・`v1f.sh`/`v5f.sh`) にしか無い = **repo へ未退避・本機 (開発機) からは検証不能** (S4b と同型の残債。下記⑤に追加)。是正内容そのものの正本は `docs/fast-mode-plan.md` G-8k 実装記録の「S6」項、当初プランと実施差は `docs/g8k-review-fix-plan.md`)** | **meson test**: merge (`5da3bfb`) 直後の**是正前ベースライン 53/53 緑** / **最終ツリー compile rc=0・test rc=0・Ok 53 / Fail 0 / Timeout 0 / Skipped 0**。**新規 meson test は 0 本** (`src/meson.build` 未接触・F5 の 4 ゲートは既存 `test_device_arena` 内の test 関数として増えるため)。**ハードゲート (呼称 H4/H5/H6 は実走側が付けたもの・是正前ツリー / 最終ツリーの双方で全 PASS)**: **H4 (step ループ内 malloc 撲滅)** = image=1 から `d_cudaMalloc=0 d_cudaFree=0 d_chunkAlloc=0` (unet / unet_persist とも)。3 枚後の累計は**アリーナ別に 1 行**で出る (`print_stats` が `arena=` ごとに 1 行を出す・出典 `src/tests/prof_arena_e2e.cu`) → **unet / unet_persist それぞれ `cum_cudaMalloc=1 cum_cudaFree=0 cum_chunkAlloc=1`** = **プロセス合計では実 cudaMalloc 2 回** (アリーナ 1 本あたり 1 回。S4b 行の「プロセス全体で 2 回ではない」注意と同型)、`cap` と `chunks` は累計ではなく**その時点の値**で、`chunks` は両アリーナとも **1**・`cap` は unet **6080MiB** / unet_persist **176MiB**。★**`cap=6080/176MiB` のような「/」連結はログに存在しない合成表記**なので引用しないこと / **H5 (出力不変)** = `rgb_hash` **18/18 枚が `0x8a96690109d2b253`** (= **S4b と同一値**) + 外部 cmp **6/6 BIT-EXACT** / **H6 (VRAM)** = 既定 13397MB − `DOLLAMA_POOL=0` 13057MB = **delta +340MB** ≤ 512MB。★**VRAM は絶対値で書かないこと (今回 V2 で実証)** — 判定は必ず**同一セッションの `POOL=0` 比 delta**で行う。**V1 (順序 1→2→3 / 3→2→1)**: 順序依存なし (A1==B1 13057 / A2==B2 13397 が MB 単位一致)。★**S4b の絶対値 13620/13649MB は再現せず 13397MB。この 252MB 差は未説明である** (★**起草時の「同居量の差 (S4b 待機 1229MiB / 今回 616MiB) による」という帰属は 2026-08-23 の監査で撤回した**: `used_after_ctx=1317MB` は常駐 **1229 / 616 / 123 MiB** のいずれでも不変で、常駐差では説明できない (repo 内 S4b 生ログ 8 本すべてが `used_after_ctx=1317MB total=16302MB` で一致 — 出典 `docs/logs/g8k-s4b/*.log`)。算術も合わない: 常駐差 1229−616 = 613MiB ≒ 643MB に対し、実測差は 13649−13397 = **252MB** (POOL=0 側も 13269−13057 = **212MB**)。しかも**本 doc の S4b 行は S5c で既に**「`used_after_ctx` が動いていない = metric が鈍いだけの可能性があり**非プロセス分の変動は分離できていない**」と後退させており、S6 起草はその後退を無視して同じ因果を再導入していた)。**判定は同一セッションの `POOL=0` 比 delta なので合否には影響しない**。★**将来 200〜300MB 級の絶対値上振れを「同居量の差だろう」で却下しないこと** (G4 の余白は S4b 時点で 132〜162MB しかない = 実退行を見逃す)。**delta は S4b +380MB → 今回 +340MB でほぼ不変**。**V2 (同居 2048MiB の hog を上乗せ)**: POOL=0 15376MB / 既定 15716MB = **delta +340MB で清浄時と完全同値** → **絶対値は同居量ぶん平行移動するが delta は同居に不変**。S4b がゲートを「POOL=0 比 delta」で切った判断の実証。★**V2 が動かしたのは 2048MiB の「アクティブな CUDA 同居プロセス」**であって、**デスクトップ常駐 (`nvidia-smi memory.used`) の差ではない**。V2 は「常駐量の差で絶対値が動く」ことを示していないので、直前の S4b↔S6 の 252MB へ当てはめないこと。**V3 (ctor の 6080MiB 単発 cudaMalloc)**: 同居 hog を idle 2560〜10240MiB / **active (常時触り続ける) 8192・11264MiB** まで振っても **9 条件すべてで reserve 成功** (内訳: idle hog 2560/3072/3584/4096 の 4 本 + 6144/8192/10240 の 3 本 + active 8192/11264 の 2 本 = **9 本**で、9 本とも `reserved=6080MiB`。★起草時の「8 条件」は誤りで 2026-08-23 の監査で是正 — 再現する人が「1 条件が落ちた/除外された」と誤解するため)。★**F4 の異常系は再現しなかった** (WDDM の eviction による) = **F4 / F2 の異常系は一度も発火していない**。実走で確定したのは「**拡大した try が正常系の確保・破棄収支を動かさない**」ことまでで、**異常系の正しさはコード上の推論に依拠する。「F4 を実走で確認した」と書かないこと**。**対照 `DOLLAMA_ARENA_RESERVE_MB=0`**: 1 枚目 11.0s → 2 枚目以降が **A ラウンド 27.1s / B ラウンド 24.3s** = **S4 の病態を再現** (★**1 走行の中で 27.1 → 24.3 と回復したのではない**。実測は `A3_reserve0` = 11.0411 / 27.1192 / 27.185s・`B3_reserve0` = 10.9766 / 24.3435 / 24.314s で、**各ラウンド内では 2 枚目・3 枚目とも悪化したまま**。ラウンド間の 27.1 vs 24.3 の差を「収束」と読まないこと)。reserve が何を防いでいるかの現地証拠。**V6 (`test_unet_fast` の poison 歩行)**: `over = 245MiB` が **18 サンプル全部で固定** (ゲート 512MiB / 余白 267MiB)。S2〜S3c レビューが懸念した ~440MiB 接近は出ずフレークなし。新挙動 (F2 の `release skipped` / F3 の `reserve shortage`) の発火は **0 件** — **是正前ツリー走行の `.log` 24 本すべて**と、**最終ツリー走行の `.log` 4 本**の双方で 0 件。★**起草時の「全 21 ログ」はどちらの本数とも一致しない誤り** (2026-08-23 の監査で是正)。★また **24 本は是正前ツリーの走行**であって、最終ツリーで走らせたのは V1 1 ラウンド + V5 だけである — 「最終ツリーで 21 本走らせて新経路が一度も出なかった」とは読まないこと。**最終ツリーでの再確認 (V1 1 ラウンド + V5)**: H4/H5/H6 全 PASS・**delta +340MB を保持** (絶対値も今回はたまたま 13397 / 13057 で一致)。`rgb_hash 0x8a96690109d2b253` が是正前・S4b と同一。外部 cmp の **sha256 が全 10 本一致** (`uf_*_default` / `uf_*_fast` = `f887962883aa339b…` / `uf_*_epilogue` = `2fe3b4f57cb0983a…` / `vae_*` 4 本 = `5ef9f2d5beb67329…`) = **相互レビュー是正 12 件が数値経路に触れていない直接証拠**。数値パリティも桁まで同一: `default vs golden MAE=7.83832e-05 max_abs=0.000465386 bad=0 SSIM=0.999996` / `fast vs default bit-exact` / `epilogue vs default MAE=6.57082e-05 SSIM=0.999999`。★**「try 拡大が数値経路に触れていない」の主証拠は上の sha256 全 10 本一致と `rgb_hash` 一致**であり、`used_after_weight_load 13323MB` / `used_after_destroy 1391MB` の一致は**補助証拠**として読むこと。とくに `used_after_destroy` は**構成にほぼ感度が無い**指標である — repo 内の S4b 生ログ 8 本すべてで POOL=0 / 既定 / RESERVE_MB=0 / steps=2 の別なく `1391MB` (出典 `docs/logs/g8k-s4b/*.log`) → 一致しても情報量は小さい (**リークが出れば増える**ので片側の検査としては有効)。`used_after_weight_load` の方は構成に感度があり (同ログで既定 **13323MB** vs POOL=0 / RESERVE_MB=0 **7067MB**)、既定同士で一致したことは **reserve + 重みの常駐量が変わっていない**傍証になるが、これも数値経路の証明ではない。**F1 直列化ゲートの実走出力 (`test_http`)**: `[serial] 直列化 OK 同時実行 max=1 calls=2 / 個別レイテンシ 263.61+138.996 ms (max 263.61 >= 236.899 = 競合なし 140.899 +0.8*busy → 窓 OK) / PNG 9332,9332 bytes 各参照と一致 / 総 264.271 ms (busy 120 ms/回)`。**負のコントロール (開発機 MinGW g++ `-O3`・スクラッチ配下でビルドし repo は未変更)**: 現行 **8/8 PASS** / `lock_guard` 行のみ削除 + busy=120 → **8/8 FAIL (`同時実行 max=2`)** / ロック削除 + busy=50 (下限) → **8/8 FAIL** / ロック削除 + busy=1 → **コンパイル停止 (`static_assert(kBusyMs >= 50)`)** = 「ロックを外すと落ちる」ことと「窓を狭めると空撃ちになる」ことの両方を実証。**秒 (characterization のみ・合否にしない・主張は倍率で行う)**: 既定 20step 1024² CFG `--fast` 相当 **10.82 s/枚** vs `DOLLAMA_POOL=0` **16.09 s/枚** (★**この 2 つの絶対秒は同居 VRAM・熱条件付き。見積りや合否の分母に使わないこと**) = **1.487x** (是正前ツリー 1.495x・S4b と 1% 内)。`test_unet_fast` 1step warm (pool on) default **322.058** / fast **251.702** / fast+epi **247.241** ms。**ドリフト併記**: 走行前 40℃ → 後 57℃・終了時 SM 2827/2820MHz・電力 179.9/191.0W・走行間クールダウン 60s・走行内ばらつき ≤1.2%。**G-4k S3 のような同一走行内 ~18% ドリフトは未発生**。★**ただしこれを「ドリフトしていない」と読まないこと** — 走行前後で 40→57℃ 動いており、**絶対秒 (10.82 / 16.09 s/枚・322.058 / 251.702 / 247.241 ms) はいずれも同居・熱条件付き値**である。合否と見積りには**倍率だけ**を使う。★**本行の主張のうち一次証拠に当たれなかったもの** (merge 直後の「是正前ベースライン 53/53 緑」の testlog / 負のコントロール 8/8 の内訳 / 是正前ツリーのソース / 相互レビュー「12 件」の合計) **は下記「G-8k S6 (T2)」節の「★出所を格付けした主張」に列挙してある。そちらを読まずに本行の数値だけを引かないこと。**| prof_arena_e2e / test_device_arena / test_http / test_unet_fast |

### G-8k S5b / S5c / S5d / S5e — 記録監査の是正と残債 (2026-08-19 〜 2026-08-20。S5f/S5g の追記は各項と⑦に明記)

S5 (commit `25772bd`) の記録に対する監査で見つかった食い違いを是正した際の、**残っている宿題**。
コードは 1 行も触っていない (docs / CLAUDE.md のみ)。

**① VAE SSIM の旧値 0.999992 が 7 箇所残存 (未修正)**

正典 (CLAUDE.md) は **0.999988** へ訂正済みだが、下記は旧値のまま:

| 箇所 | 種別 |
|---|---|
| `docs/measurements-log.md:39` | 2-4 実装時の計測行 (**当時の実測値としては正しい** — 履歴としてこのままでも可) |
| **本 doc**「次のタスク」節「Phase 2 以降」箇条書きの 2-4 (VAE decode) 項 (★実体は表でなく箇条書き — S5f で呼称是正) | 2 次記述 (要追随)。★**同一 doc 内なので行番号では指さない** (下記の注記参照) |
| `docs/roadmap.md:43` | 2 次記述 (要追随) |
| `docs/testing.md:444` | 2 次記述 (要追随) |
| `README.md:90` / `README.md:289` | ★**対外文書**。正典と食い違った状態 |
| `README_en.md:96` | ★**対外文書**。正典と食い違った状態 |

- 値が動いた実体は **2-6 最適化**であって G-8k ではない。**真の変化点は `b3fe139` 単独** (2026-06-20 =
  conv を im2col+wmma GEMM 化 + 帯分割 stride バグ修正) で、この時点で既に 0.999988 である
  (同 commit 本文「noise_pred SSIM 0.999996 / **VAE SSIM 0.999988 復帰**」)。
  - ★**帰属を決めるのは直前 `f3f4625` (2-6 S2 の 2 本目 = attention flash 化) 本文の
    「noise_pred SSIM 0.999998 / **VAE SSIM 0.999992 維持**」である** (S5d で追加)。
    `b3fe139` 本文の「**復帰**」は、**同 commit 内で作り込んだ帯分割 stride バグ (VAE SSIM 0.0116) からの復帰**を
    指しており、**それ単独では「0.999992 から動いた」ことを示さない**。`f3f4625`(0.999992 維持) →
    `b3fe139`(0.999988) の **2 本をセットで読んで初めて変化点が確定する**。
    片方だけを見た担当が「`b3fe139` は 0.0116→0.999988 の話で 0.999992 とは無関係では」と判断して
    **4 度目の書き換え (S3-E 説への逆戻り) をしないよう、必ず両方を引くこと**。
  - ★**「変化点は S2」という書き方をしない** (S5d で是正)。2-6 S2 は件名に「(S2)」を持つ commit が
    **3 本** (`b492085` / `f3f4625` / `b3fe139`) あり、「S2」と書くと前 2 本まで容疑者に入る。
    **帰属は hash `b3fe139` 単独で書き、必要なら「2-6 S2 の 3 本目」と添える**。以降の
  S3-A (`502c936`) / S3-D (`2e079e1`) は本文が「golden 維持: VAE SSIM 0.999988」、
  S3-E (`a71898e`) は「TF32 既定で VAE SSIM=0.999988」と記録しており、いずれも **維持**。
  出典 commit `b3fe139` / `a71898e`。
  - ★**S3-E (VAE up2/up3 の TF32 GEMM 化) を変化点として名指ししないこと** — S5b (`6d8f1e1`) の
    記述はこれを誤って断定していた (S5c で是正)。犯人を S3-E に取り違えると、将来 VAE 精度を
    疑う担当が TF32 化を revert しに行き、**真の変化点である S2 の FP16 im2col+wmma conv を見逃す**。
  - **G-8k S3/S3c は VAE 数値を動かしていない** (G-8k S3 前後で出力 FP16 画像が memcmp BIT-EXACT・
    出典 commit `88b3ae7` 本文)。同本文には **「S3 前ベースラインとも同値」** の一次記録もある
    (「test_vae_decode SSIM 0.999988 (POOL=0 / S3 前ベースラインとも同値」)。S5b はこれを
    出典なしと誤認して削除したので、S5c で出典付きで復活させた。
  - ★ただし**同じ `88b3ae7` 本文の「CLAUDE.md の 0.999992 は陳腐化した記録値」という評価部分は
    誤りなので引かないこと**。0.999992 は 2-4 当時の正しい実測値であり、「陳腐化」ではなく
    2-6 S2 での実変化に正典表が追随していなかっただけである (次に `88b3ae7` を読む担当が
    同じ誤解をしないよう、この 1 行を残す)。
- S5 の決裁が「CLAUDE.md の訂正は SSIM 1 箇所のみ」だったため、上記 7 箇所は**今回スコープ外として据え置き**。
  ★とくに README / README_en は対外文書なので、次に触る担当が優先して追随させること。

**② CLAUDE.md に G-8k の行が無い**

CLAUDE.md の計測表には G-2k / G-3kf / G-4k の行はあるが **G-8k の行が無く**、
さらに G-4k S2 の行 (CLAUDE.md:230 付近) は今も
「resnet ≤0.95s の合否は **G-10k(conv 真batch2)/G-8k(im2col malloc撲滅) 後**へ再割当」と
**G-8k を未完了の前提**で書かれている。G-8k は S4b 全緑でクローズ済 (2026-08-19) なので、
次に CLAUDE.md を触る際に追随させること (再 profile 自体は G-10k 完了後で正しい —
G-8k は秒数レバーではないため)。
→ ★**「未完了前提」の側は S5g (2026-08-22) でクローズ**: G-4k S2 行の再割当文に G-8k クローズ済
(S1〜S4b・2026-08-19) の最小注記を追加した (6 回目監査の中4・ユーザー決裁は最小注記のみ)。
**「計測表に G-8k の行が無い」の側は未実施のまま残債** (行を足すか否かは未決裁。CLAUDE.md 肥大の
制約があるため黙って足さない)。
→ ★**S5h (2026-08-22・ユーザー決裁) で CLAUDE.md 計測表に G-8k 行を 1 行追加 = ②は両側クローズ**。
内容は S1〜S4b と S4b 6 ハードゲート全 PASS の芯のみ (正本は本 doc の G-8k S4/S4b 行と
`docs/fast-mode-plan.md` G-8k 実装記録)。test 列は `test_device_arena` — 自動枠 (meson test) があるのは
これのみで、`prof_arena_e2e` は計測専用・test 未登録 (`src/meson.build` の executable 定義に test() 呼出が
無いことを S5h で確認)。

★**追随対象は docs だけではない (S5d で追加)**: `.claude/agents/` のエージェント定義にも同型の残置がある。
`perf-profiler.md:99` は「現在の最有力は G-10k と **G-8k**」のままだった (S5d で是正済)。
エージェント定義は**起動時にそのまま文脈へ入る**ため、放置すると docs 側で防いだ重複起票が
エージェント経路から復活する。G-8k 以外の項目でも `grep -rn "最有力\|未着手\|バックログ" .claude/agents/` で
定期的に突き合わせること。

**③ UNet noise_pred SSIM に同型の陳腐化 (2-5 値 0.999998 が残存・S5c で追加)**

VAE と**同じ形の追随漏れ**が UNet 側にもある。2-6 最適化後の実測は **0.999996** (本 doc の
2-6 最適化行「golden 維持 (UNet noise_pred SSIM 0.999996 ...)」・`docs/fast-mode-plan.md:329` の
`test_unet_fast` 実走出力 `default vs golden ... SSIM=0.999996`) だが、下記は 2-5 当時の
**0.999998** のまま:

| 箇所 | 種別 |
|---|---|
| `CLAUDE.md:224` | ★**正典**。VAE と違い未訂正 (S5b/S5c の決裁が「CLAUDE.md は SSIM の VAE セルのみ」ゆえスコープ外) |
| `docs/measurements-log.md:40` | 2-5 実装時の計測行 (**当時の実測値としては正しい** — 履歴としてこのままでも可) |
| **本 doc**「次のタスク」節「Phase 2 以降」箇条書きの 2-5 (SDXL UNet) 項 / `docs/roadmap.md:44` / `docs/roadmap.md:182` / `docs/testing.md:445` | 2 次記述 (要追随) |
| `docs/fast-mode-plan.md:15` / `:22` / `:367` / `:374` / `:476` | 2 次記述 (要追随)。同一 doc 内で :329 の実測 0.999996 と食い違っている (★行番号は 2026-09-04 の R-1 加筆で `:333`/`:340`/`:438`/`:295` から繰り下がった。指している中身は同じ) |
| `README.md:90` / `README.md:281` / `README_en.md:96` | ★**対外文書**。正典と食い違い中 |
| `docs/superpowers/specs/2026-08-05-subagent-refresh-design.md:91` | 2 次記述 (S5d で追加。**S5c 時点の表は「全箇所」を装いながらこの 1 件が漏れていた**) |

VAE と同じく **0.999998 は 2-5 当時の正しい実測値であって誤記ではない**。次に CLAUDE.md を
触る担当は、VAE セルと同じ体裁 (旧値・変化点 commit・「誤記ではない」の明記) で追随させること。
なお **`CLAUDE.md:224` には S5d で「2-6 後の実測は 0.999996 = 本残債③」への注記のみ入れた** (値は未置換)。

**④ 秒数側にも同型の陳腐化 (CLAUDE.md の decode 7.96s / 1step ~9.2s・S5d で追加)**

★**S5/S5b/S5c は「SSIM の追随漏れ」だけを直し、同じセルに載っている秒を見落としていた。**
`CLAUDE.md:223` は「本表が追随していなかった分の訂正」と自己宣言しながら、**同じセル末尾の
`decode 7.96s` は 2-6 最適化に未追随**だった (`:224` の `1step ~9.2s` も同型)。

| 正典の記載 | 実際に動いた経緯 (一次証拠) | 食い違い |
|---|---|---|
| `CLAUDE.md:223` VAE `decode 7.96s` | `b3fe139` 本文「**VAE decode 7.96s→5.73s**」→ `a71898e` 本文「wall-clock: **VAE decode 5.21s→1.16s**」 | `docs/fast-mode-plan.md` の **G-9k は同じ対象を 1.197s** と見積っており **7 倍** |
| `CLAUDE.md:224` UNet `1step ~9.2s` | `f3f4625` 本文「**UNet 1step ~9.2s → 2.50s**」 | G-3kf の **warm 実測 default 469.5-526ms** と **19 倍** |

同型の秒は**正典以外にも残っている** (S5e で追加。①③ と同じく横断で押さえる):

| 箇所 | 記載 |
|---|---|
| **本 doc** 計測表の 2-4 計測行 (「自作 VAE decoder 全段」= ①の表の測定原行と同じ行・S5f で追加) | 「decode **中央値 7.96s**」 (**当時の実測値としては正しい**) |
| **本 doc** 計測表の 2-5 計測行 (「自作 SDXL UNet 全段」= ③の表の測定原行と同じ行・S5f で追加) | 「1step **中央値 ~9.2s**」 (**当時の実測値としては正しい**) |
| `docs/testing.md:444` | 「実測 … decode 中央値 **7.96s**」 |
| **本 doc**「次のタスク」節「Phase 2 以降」箇条書きの 2-5 項 | 「1step **~9.2s**」 |

いずれも**当時の実測値としては正しい**ので誤記ではない。再測時に①③④をまとめて追随させること。

- **誤読経路**: 正典表を分母にして G-9k / G-10k の効果見積りや合否設計を組むと、**桁違いの分母で
  合否が決まる**。SSIM と違い秒は「効果 %」の分母に直接使われるため実害が大きい。
- ★**S5d では数値を置き換えていない** — 現況の decode 秒 / 1step 秒は**今回測っていない**からである
  (未測定値を書けば陳腐化を新しい陳腐化で上書きするだけになる)。CLAUDE.md には
  **「陳腐化・要再測・分母に使うな」の注記のみ**を入れ、値は旧値に取り消し線を付けて残した。
- **宿題**: `test_vae_decode` / `test_unet_fast` の現況中央値を研究機で採り、正典表を実測値へ置き換える。
  そのとき **G-9k の 1.197s と同一ハーネス・同一条件で採ること** (条件が違えばまた比較不能な数値が増える)。
- **同型の再発防止**: 計測表のセルを訂正するときは、**そのセルに載っている他の指標も同時に検算する**。
  「SSIM だけ直す」という決裁は、同居する秒を暗黙に「正しい」と保証してしまう。

**⑤ S4b 生ログを repo へ退避済 (S5d で実施) / S3・S3b・S4・S6 (T2) の生ログは依然として無し (S4 は S5h で、S6 は S6 の記録時に列挙へ追加)**

- S4b の生ログ 10 本 (S5d 時点は 9 本・S5e で `smi_before_A2.txt` を追加) は消える temp (セッション ID 付き scratchpad) にしか無く、**S4b 全緑の唯一の一次証拠**が
  失われる寸前だった → S5d で **`docs/logs/g8k-s4b/` へ無改変のまま退避**した (来歴・読み方の注意は
  同ディレクトリの `README.md`)。以後 S4b 行から参照すべきはこの repo 内パス。
- ★**S3 の「96 対 / 102 本」と S3b の「12 枚実測で live peak 変動ゼロ」は生ログが残っていない**
  (根拠は commit 本文とコードコメントのみ)。解像度 / batch / step を変える改修時はこの 2 つを
  **再測してから**触ること。
- ★**S4 (6 走行・上記計測表「G-8k S4」行) の生ログも repo に無い (S5h で本リストへ追加)**:
  `docs/logs/` にあるのは S4b 分 (`g8k-s4b/`) のみ (物理確認済) で、S4 の peak 13309/13310・16302MB・
  捨て分 4637MiB・ページング事故 (11s→25s) は**生ログでの再検証ができない** (出典は当時の実走報告 =
  上記 S4 行のみ)。研究機 temp 原本の有無は本機から検証不能。S4 の実行形態 (1 走行 1 プロセス) も
  ハーネス構造からの演繹 (S4b 行の S5f 加筆参照)。なおこれは**下記の reserve 定数まわりの列挙とは別軸**
  (定数値 5914/137 自体が S4b 生ログで検証可能であることは変わらない)。
- ★**S6 (T2) の実走生ログも repo に無い (本 S6 で追加)**: 是正前ツリーの V1〜V6 は研究機ローカル
  `E:\Develop\logs\g8k-t2-verify\` (再現スクリプト同梱)、最終ツリー再確認は同 `...\g8k-t2-verify-final\` にあり、
  **`docs/logs/` 配下は依然 `g8k-s4b/` の 1 ディレクトリのみ** (物理確認済)。したがって上記計測表の
  「G-8k S6」行の数値 (13397/13057MB・`0x8a96690109d2b253`・sha256 群・10.82/16.09s 等) は
  **本機からは再検証できず、出典は研究機での実走報告のみ**である。S4b と同型の残債なので、
  **研究機に触れる次のセッションで `docs/logs/g8k-t2/` 相当へ退避すること** (S4b は退避しないまま
  temp が消えかけ、全緑が自己申告に降格する寸前だった)。
- ★**F1 の negative control も走行出力が残っていない (2026-08-23 の監査で判明・本 S6 で追加)**: 「現行 8/8 PASS /
  `lock_guard` 削除 + busy=120 → 8/8 FAIL / busy=50 → 8/8 FAIL / busy=1 → コンパイル停止」は
  **`test_http` の直列化ゲートが空撃ちでないことを支える唯一の証拠**だが、実体は開発機のスクラッチ配下
  (`...\scratchpad\nc\`) にあり、**現存するのは `run_nc2.sh` (`for i in $(seq 1 8)` = 8/8 の出所) /
  `nc2_compile.log` (static_assert 失敗の実物) / `nc1.exe` `nc4.exe` `pc.exe` だけ**で、
  **8 回分の走行出力はコンソールのみ = ファイルに落ちていない**。セッション ID 付きディレクトリごと消える。
  S4b 生ログが temp から消えかけ「全緑が自己申告に降格する寸前だった」のと**同型**なので、
  再走するなら出力をファイルへ落として repo へ退避すること。
- ★**ただし reserve 定数 5914 / 137 MiB の「値そのもの」は repo 内の S4b 生ログで検証できる** (S5g で是正。
  従来ここにあった「定数の**唯一の出典**が S3/S3b の commit 本文とコードコメント」は**過小申告**だった)。
  `docs/logs/g8k-s4b/` の `.log` 8 本 (採用 7 走行 + 棄却 `DISCARDED_A2_overlap_risk.log`) を全数確認:
  **`arena=unet live_peak=5914MiB` は 8 本すべて**・**`arena=unet_persist live_peak=137MiB` は 6 本**に印字
  (POOL=0 の 2 本は `live_peak=0MiB` = pool 無効時は persist アリーナを使わないため、値として正しい挙動)。
  生ログで検証できるのは**定数値そのものと、POOL / RESERVE_MB / steps に対する不変性**。
  **この定数まわりで生ログ無しのまま残るのは S3 の「96 対 / 102 本」と、S3b だけが持つ `DOLLAMA_ARENA_RELEASE=1`
  構成込みの不変性** (S3b の 12 枚に release 構成が入っていた出典は `acca803` 本文「4 構成 (POOL=0 /
  既定 / RESERVE_MB=0 / ARENA_RELEASE=1) x 3 枚」。S4b は 8 本すべてヘッダが
  `DOLLAMA_ARENA_RELEASE=(unset)` で release 経路を含まない)。

★**行番号は追記でズレる (S5d で 2 度実際にズレた)**: S5c 時点で正しかった `measurements-log:160/161` は
本節への加筆で `:217/:218` へ移動し、**それを書き直した直後の加筆でもう一度ズレた**。
**同一 doc 内を行番号で指すのは自己参照ゆえ構造的に腐る** → 上表では**行番号を使わず「どの節の何の行か」で指す**
形に変えた。突き合わせは `grep -rn "0.999992" --include=*.md .` の結果と照合すること
(他 doc への参照は、その doc を編集しない限り腐らないので行番号のままでよい)。

**⑥ 見送りを明示した項目 (S5d)**

- 「現状 (裏取り)」ブロックの旧行 (`conv2d.cu:370/420` 等) に**打ち消し線を付けるのは見送った**。
  ブロック冒頭の「S1〜S3c で全て解消済み」注記が十分に強く、また一次記録を後から装飾する方向は
  原本性を損なうため。**ただし grep で行だけ拾うと注記が見えない**ので、当該行を引くときは
  必ずブロック冒頭を併せて読むこと。
- ~~`.claude/agents/record-auditor.md` (記録監査人の定義) は repo 未追跡のまま~~
  → **追跡化済 (2026-08-20・ユーザー決裁)**。監査人自身の定義が記録に残っていないのは不備であり、
  この定義には**このプロジェクトで実際に起きた事故形態が埋め込まれている** (未達を達成と書く / 数値の陳腐化 /
  旧記述の残置 / 被験変数の取り違え / 比較条件の非対称 / 代理指標での合格宣言) ため、
  監査の再現性のうえでも repo に置くのが筋。
  ★**エージェント定義はセッション起動時にしか読み込まれない** (新設・改訂した直後のセッションでは
  `Agent type not found` や旧版のまま動く) 点は変わらないので、改訂したら次セッションで効く前提で運用する。


**⑦ 6 回目監査 (S5f 後・条件付き PASS / 重大 0・中 4・軽微 4) の扱い (S5g・2026-08-22)**

6 回目監査は「新たに持ち込まれた誤り: なし」= 5 ラウンド続いた「是正が新しい誤りを持ち込む」型が
初めて止まった。中 4 件のうち、ユーザー決裁で **中3 (上記⑤で是正)・中4 (残債②の「未完了前提」側を
是正) の 2 件を S5g で実施し、中1・中2 は見送り**。見送り分もここに記録する (指摘の消失防止)。
★その後のユーザー決裁により、**中1・中2 は S5h (2026-08-22) で実施済** (各項末尾に追記)。

- **中3 (是正済)**: reserve 定数 5914/137 の証拠所在の過小申告 → 上記⑤と `docs/fast-mode-plan.md`
  「G-8k 実装記録」S3c 項を是正。★監査人の申告「S4b 生ログ **7 本すべて**に 5914/137 が印字」は
  **本数・内訳とも不正確** (実数は `.log` 8 本・5914 は 8 本すべて・137 は 6 本で POOL=0 の 2 本は
  `0MiB`) で、S5g で全 8 本を開いて検算した値だけを採った。
- **中4 (是正済)**: CLAUDE.md が G-8k を未完了前提のまま (残債②で S5b から宿題化されていたが未実施
  だった)。兄弟コピー (`docs/fast-mode-plan.md` の「追記 (2026-08-19)」2 箇所・
  `docs/hw-accel-plan.md:121/130-134/152` (旧 `:117/126-130/145`・2026-09-04 の R-1 加筆で繰り下がった)・`.claude/agents/perf-profiler.md:101-102`) は是正済みで、
  毎セッション文脈に入る入口の CLAUDE.md だけが取り残されていた → G-4k S2 行に最小注記を追加
  (詳細は残債②)。
- **中1 (見送り → S5h・2026-08-22 で実施)**: 上記計測表の **G-8k S4 行**の冒頭「同一プロセスで」が、S4b 行で
  2 度是正した「同一プロセス」誤読 (S5e→S5f) の**発生源**なのに、発生源自体に注記が無い。
  S5f は台帳指定どおり隣の S4b 行と `docs/fast-mode-plan.md` だけを直した (スコープ遵守の結果)。
  誤読経路: 「同一プロセスで … 3 枚連続 × 3 構成 × 2 ラウンド」の「同一プロセス」は**枚数** (1 プロセスで
  3 枚連続) に掛かるが、構成 (= 1 プロセス内で構成切替) にも掛かると読める。一次証拠:
  `src/tests/prof_arena_e2e.cu:4` 「同一プロセスで … N 枚 (既定 3) 連続実行」・`:8` 「走行間で変えるのは
  DOLLAMA_POOL / DOLLAMA_ARENA_RELEASE のみ」・getenv は `:150-151` の 1 度だけ。
  → **S5h で是正**: S4 行の「3 枚連続」直後に「同一プロセスで」の係り先の注記を追加
  (`prof_arena_e2e.cu:4` / `:8` は S5h セッションで開き直して確認)。
- **中2 (見送り → S5h・2026-08-22 で実施)**: S5f が S4b 行に新設した「★S4 の生ログは残っていない (実行形態は
  ハーネス構造からの演繹)」が、生ログ不在の正典リスト (上記⑤) に入っていない = 片肺。
  ⑤は S5g 是正後も S3 / S3b (release 込み) しか列挙しておらず、⑤だけを読むと「S4 は生ログあり」と
  誤読しうる。一次証拠: S4b 行の当該文 (S5f 加筆) と⑤の現行文面。
  → **S5h で是正**: ⑤の見出しと列挙に S4 を追加し、S4 行 (本 doc 計測表) と `docs/fast-mode-plan.md` の
  S4 記述 (「S3 の未達 (S4 実測)」項) にも生ログ不在の注記を追加。`docs/logs/` 配下が `g8k-s4b/` のみで
  あることは物理確認済・研究機 temp 原本の有無は本機から検証不能 (S5f の限定と同じ)。

### G-8k S6 (T2) — S2〜S3c 静的レビューの是正 (F1〜F7) と残債 (2026-08-22)

由来は `docs/g8k-review-fix-plan.md` (2026-08-20 に**開発機**で実施した静的コードレビュー。
ビルド・実走なしで、**行番号はすべて当時の HEAD `acca803` 基準**)。ユーザー決裁でスコープを
**F1 / F3 / F4 / F2 は dtor try/catch のみ / F5 / F7** に確定し、**見送りゼロで全件実施**した。
実測値・ゲート合否は上記計測表の「G-8k S6」行が正本、コード側の実装記録は
`docs/fast-mode-plan.md` の G-8k 実装記録「S6」項。

**実施内容の一次証拠 (現物の位置)**

| 項目 | 実体 |
|---|---|
| F1 生成の直列化 | `src/server/api.cpp` の無名 namespace に `std::mutex g_generate_mutex` を置き、`result = gen.generate(gr);` **1 文だけ**を `lock_guard` で覆う。JSON 解析・base64・応答組み立てはロック外 |
| F2 (dtor のみ) | `bool device_arena_release_noexcept(DeviceArenaId) noexcept` を新設 (**宣言は `src/kernels/device_arena.cuh` 1 本・実体は `src/kernels/device_arena.cu` 1 TU** = C0 の ODR 事故の再演防止・薄いラッパで release 本体を複製しない)。`unet_weights_destroy` の release 2 本を差し替え + `~DiffusionPipeline` を `destroy_resources() noexcept` へ集約。★**`maybe_release_arenas()` は noexcept 化していない** (generate 経路であって dtor ではないため。握り潰すと壊れた状態のまま生成が続く) |
| F3 reserve 不足の可視化 | `arena_profile_enabled()` を撤去し **無条件 stderr へ 1 回**。1 回ガード (`reserve_warned`) は維持。粒度は 1 reserve サイクルにつき 1 行 (`DOLLAMA_ARENA_RELEASE=1` では画像ごと・既定構成ではプロセス通算 1 行) |
| F4 / F4b ctor の例外安全 | `src/infer/diffusion.cu` の try を **device 確保区間全体** (`load_f16`×3 → `cudaMalloc`×3 → `cudaMemcpy`×3 → `unet_weights_create` 5.1GB → `vae_weights_create` 92MB → `reserve_arenas()`) へ拡大し catch で `destroy_resources(); throw;`。★**当初案は `reserve_arenas()` だけを覆う形**だったが、相互レビューで「1 行上の `vae_weights_create` が throw すると同じ 5.1GB リークが残る」と指摘されて範囲を広げた。F4b = `device_arena_reserve` の `chunks.clear()` 直後に `reserved_bytes = 0` と `reserve_warned = false` を追加 (`total_capacity = 0` は既にあった) |
| F5 破棄経路のゲート | `src/tests/test_device_arena.cu` に **4 ゲートを新設** (`git show HEAD:src/tests/test_device_arena.cu` に `release_noexcept` は **0 件** = **HEAD からの差分としては 4 本すべてが新規**。「3 本で着手しレビュー指摘で 1 本追加」は**未コミットの作業中間状態**の話であり、S6 以前に 3 本あったわけではない)。①生存確保あり + foreign thread から素の `device_arena_release` → **throw する** (落ちる契約の維持) ②同条件で `release_noexcept` → 例外なし `false` かつ**アリーナ状態不変** (`cuda_free_calls` / 容量 / チャンク / カーソルの据え置きを assert) ③静止状態で `release_noexcept` → `true` (素の release と同結果) ④**不正 id (`static_cast<DeviceArenaId>(99)`) → `false`** = `arena_of` 経路も固定。**POOL 枠 / POOL OFF 枠の両方**に登録 |
| F7 コメント・スタイル | `device_arena.cuh` の「C++14 前提」是正 (実体は **MSVC ホストのときだけ** `src/meson.build` が `-Xcompiler /std:c++14` に落としている) / revert 手順の是正 (`src/kernels/conv2d.cu:374/704/783` が `DeviceArenaId::UNet` をハードコードしているので `vae_decode.cu` の `kVaeArena` 1 行 revert では戻り切らない) / `prof_arena_e2e.cu` の Allman 整形 **9 箇所** (挙動不変) |

**★履歴の訂正 — 「アリーナ化以前は同時 2 リクエストがメモリ安全に成立していた」は誤り**

レビュープラン F1 の症状記述にあったこの一文は、相互レビュー (F1 側の検査者 = cuda-kernel-dev) で
反証された。生成経路が握る**プロセス共有の無ロック可変状態は 3 つ**あり、アリーナはそのうち最も新しい 1 つ:

- `src/kernels/gemm.cu:363` `g_cublas_handle` — **遅延生成が非アトミック** (`cublas_handle()` の
  `if (g_cublas_handle == nullptr)` → `cublasCreate`)。導入は `fc97b76` (2026-06-22 / 2-6 S3-B
  「transformer GEMM を cuBLAS GemmEx に委譲」)。**`cublasSetStream` の呼び出しは repo に 0 箇所**
  = 全部デフォルトストリーム (現ツリーで `grep -rn cublasSetStream src/` が 1 件ヒットするのは
  api.cpp に今回書いた**当該コメント自身**であり、`git grep cublasSetStream HEAD -- src/` は 0 件)。
- `src/kernels/groupnorm.cu:223-238` `g_mb_buf` — **grow-only 再確保** (`floats > g_mb_buf_floats`
  で `cudaFree` → `cudaMalloc`)。導入は `42ec1be` (2026-07-11 / G-4k S1a)。**出荷 `--fast` は
  epilogue を含意**するので実運用経路。
- `src/kernels/device_arena` (G-8k) — owner / cur / offset / req_live が無ロック。

→ **HTTP 並行生成は G-8k の退行ではなく、少なくとも 2-6 から一貫して不成立**。G-8k は 3 つ目の
共有状態を足した。**このロックを外せる条件は「3 つすべてが個別にスレッド安全化されたとき」**で、
アリーナだけ直しても外せない (api.cpp の `g_generate_mutex` 定義コメントに 3 状態を列挙済み)。
なお**再入経路が無いこと**も現物で確認した: `IImageGenerator::generate` の production override は 4 実装
(`backend_image_generator` / `pipeline_generator` / `txt2img_generator` / `stub_generator`) で
どれも他の `IImageGenerator` を保持せず、production の呼び出し元は `src/server/api.cpp:225` と
`src/main.cpp:241` の 2 箇所のみ・`--http` と CLI は 1 プロセス内で排他 (`main.cpp` は `--http` を :224 で return し `--prompt` 分岐 :227 に到達しない)。
(`src/tests/test_http.cpp` のゲート用デコレータだけは inner 生成器を保持するが、ファネルへは再入しない)

**★F3 で警告文言が変わった (台帳・手順書が旧文言を引用していたら更新すること)**

- **S3b 以来の committed 文言** (`git show 5da3bfb:src/kernels/device_arena.cu` の 350-351 行):
  `[ALLOC] reserve shortage: arena=%s peak_request %zu MiB > reserve %zu MiB (falling back to chunk growth)`
  — **stdout・`DOLLAMA_PROFILE=1` 配下**。
- **S6 確定**: `... (falling back to chunk growth; reserve is undersized -- see reserve_arenas() in src/infer/diffusion.cu)`
  — **stderr・無条件 1 回**。**接頭辞 `[ALLOC] reserve shortage: arena=…` は不変**なので、
  接頭辞で grep している採取手順は壊れない。
- 文面を **arena 非依存**にしたのは相互レビュー 中4 の指摘による: `arena=unet_persist` で出たときに
  「`DOLLAMA_ARENA_RESERVE_MB` を上げろ」と誘導すると**嘘になる**。persist 側の予約量は
  `arena_reserve_persist_mb()` (`src/infer/diffusion.cu:252-260`) の **固定 176MiB** (= 137+32 を
  16MiB 境界へ切上げ) で、この env は persist に対しては「0 か否か」のキルスイッチとしてしか効かない
  → 従っても警告は消えず、UNet 側の reserve だけ膨らんで VRAM ゲートの余白を削る。
  ★なお「F3 初版の文面が `; raise DOLLAMA_ARENA_RESERVE_MB` だった」というのは**セッション中の
  未 commit の中間状態**についての伝聞であり、一次証拠 (commit / ログ) では追えない。**確定版に
  その文字列が存在しない**ことだけが現物で確認できる事実。

**★実走で確定していないこと (書き方の禁止事項)**

- **F4 / F2 の異常系は一度も発火していない。**「F4 を実走で確認した」と書かないこと。
  V3 で同居 hog を idle 2560〜10240MiB / active 8192・11264MiB まで振っても **9 条件すべてで
  reserve は成功**した (WDDM の eviction による)。実走で確定したのは
  **「拡大した try が正常系の確保・破棄収支を動かさない」**ことまで
  (`used_after_weight_load 13323MB` / `used_after_destroy 1391MB` が是正前と一致) で、
  **異常系の正しさはコード上の推論に依拠する**。
- **VRAM を絶対値で書かない。** V2 で「絶対値は同居量ぶん平行移動するが delta は同居に不変」が
  実証された。判定は**同一セッションの `POOL=0` 比 delta** (S6 +340MB / S4b +380MB)。
- **秒を単独で主張しない。** 倍率 (1.487x) + ドリフト併記で書く。

**merge の経緯 (被験変数の混入について)**

時系列は **① 着手前に merge (`5da3bfb`・2026-08-22) → ② その後に PL が「被験変数を増やさないため
`git merge origin/main` は後回し」と決裁 → ③ 既に済んでいたので、影響が無いことを確かめて続行**、の順。
(一次証拠で日付が取れるのは merge commit `5da3bfb` 自体だけで、②③ の順序は作業報告による。)取り込んだ **13 コミット** (レビュープラン起草時の見込みは
11 コミットだった) は `git diff --name-only $(git merge-base d3c00f6 5da3bfb^2) 5da3bfb^2` で
**`src/` と `meson.build` を 1 行も触っていない**ことを確認済 (実体は UI/Blazor + docs のみ)。
merge 後のベースラインが **53/53 緑**であることを確認したうえで続行した。

**★出所を格付けした主張 (一次証拠に当たれなかったもの・2026-08-23 の監査で追加)**

以下は**本 repo からは裏が取れない**。引用するときは必ずこの格付けごと引くこと。
結論 (S6 が数値を動かしていないこと) への影響は小さいが、**「検証済み」として再利用してはいけない**。

| 主張 | 格 | 検算できたこと / できなかったこと |
|---|---|---|
| 相互レビューの指摘は **12 件** | 合計は**未検証** | `src/` のコメントに番号が残っているのは **6 件だけ** (`grep -rno "相互レビュー" src/` = 中2 / 中4 / 中6 / 軽1 / 軽3 / 軽6。すべてアリーナ側 = `device_arena.cu` `device_arena.cuh` `unet.cu` `diffusion.cu` `test_device_arena.cu`)。**F1 側 (`api.cpp` / `test_http.cpp`) には番号付きコメントが 0 件**。個々の是正が実在することは現物で確認できるが、**合計 12 という数は確認できない** |
| merge 直後の**是正前ベースライン 53/53 緑** | **未検証** | 当該 testlog は後続走行で**上書き済みで現存しない**。最終ツリーの 53/53 は別途確認済みだが、**merge 直後の 1 回は自己申告** |
| 負のコントロール **8/8 PASS / 8/8 FAIL** | 内訳は**未保存** | スクリプト `run_nc2.sh` と `nc2_compile.log` (static_assert 失敗) は現存するが、**8 回分の走行出力はコンソールのみ**。詳細は残債⑤ |
| S6 **是正前ツリー**での V1〜V6 | ソースが**再現不能** | 是正前ツリーは **commit も stash も無い** (`git stash list` = 空・本 repo で物理確認)。ただし最終ツリーとの数値一致 (sha256 10/10・`used_after_weight_load 13323MB`・`used_after_destroy 1391MB`) は確認済みで、**結論への影響は小** |
| 開発機 `multi_frame_pipeline` 赤の「stash して再現確認済み」 | **未検証** | `mfp.log` が **0 バイト**で追試不能。原因の PATH 汚染そのものは環境要因であり S6 の変更とは無関係 (研究機では緑) |
| V1〜V6 / 最終ツリーの**全実測値** | 研究機ローカル | 生ログが repo に無い (残債⑤)。本機 (開発機) には `E:\Develop` 自体が存在しない |

**S6 で残した follow-up (今回スコープ外・別タスク)**

- **`gemm.cu:363` の `g_cublas_handle` 非アトミック遅延生成**と **`groupnorm.cu:223` の `g_mb_buf`
  grow-only 再確保** — F1 のファネル直列化で実害は塞がれているが、**競合そのものは残存**。
- **プロセス間は直列化されない**: `dollama.exe --http` と `--prompt` を同時起動すると reserve
  6080MiB + 重み ~5GB が 2 セット必要になり 16GB 板では VRAM が足りない (排他は 1 プロセス内のみ)。
- **matting (iGPU) と scoring (NPU) が `generate()` の内側 = ロック内に入った** → 「A の GPU 拡散」と
  「B の NPU 採点」を重ねられない。**HW 協調はこのプロジェクトの芯**なので、取り戻すなら
  backend 境界で切り直しが要る (2-6c の `IDiffusionBackend` 境界が候補)。
- `try_lock` 失敗時に **503 + `Retry-After`** を返す UX 改善余地 (現状はロック待ちで HTTP スレッドが張り付く)。
- **所有権設計 (参照カウント / 所有移動) は 2-6d (SDXL 3 preset) 着手時へ** — F2 の (a) 参照カウント化 /
  (b) 所有移動はユーザー決裁で**不採用**。同居を防いでいるのは値保持ではなく
  `src/server/cli_generate.hpp` の**排他フォールバック梯子**という不変条件である
  (`DiffusionPipeline` を値で持つラッパは `src/server/diffusion_runner.cu` と
  `src/server/pipeline_generator.hpp` の 2 つ。梯子を「両方作って良い方を選ぶ」形に書き換えた瞬間に同居が成立する)。
- `device_arena.cuh` の単行 Allman 1 箇所は据え置き (レビュープランのスコープ外)。
- **開発機の `multi_frame_pipeline` が赤** (`0xC0000139`) — `C:\Strawberry\c\bin` / `perl\bin` の古い
  `libstdc++-6.dll` が PATH 先行。`std::jthread` を使う唯一の test。**環境問題で S6 の変更とは無関係**
  (★**「stash して再現確認済み」は作業報告のみで追試不能** — 採取したはずの `mfp.log` が **0 バイト**。2026-08-23 の監査で格下げ)。**研究機では緑** (最終ツリー 53/53)。


## 次のタスク

**C++ 実装フェーズ (Phase 1 — パイプライン骨格)**

| # | 実装物 | ファイル | 状態 |
|---|---|---|---|
| 1 | Meson ビルド + src/ 構造 | `meson.build`, `src/` | ✅ 完了 |
| 2 | Tensor クラス + テスト | `src/core/tensor.hpp`, `test_tensor.cpp` | ✅ 完了 |
| 3 | Allocator + テスト | `src/core/allocator.hpp`, `test_allocator.cpp` | ✅ 完了 |
| 4 | SPSC キュー + テスト | `src/core/queue.hpp`, `test_queue.cpp` | ✅ 完了 |
| 5 | CLIP NPU 推論 + テスト | `src/infer/clip.hpp`, `test_clip.cpp` | ✅ 完了 (NPU 7.82ms) |
| 5.5 | キャラ台帳 character.hpp + テスト | `src/core/character.hpp`, `test_character.cpp` | ✅ 完了 |
| 6 | WD14 CPU 推論 + テスト | `src/infer/wd14.hpp`, `test_wd14.cpp` | ✅ 完了 (CPU 105.3ms) |
| 7 | スレッド骨格 + CPU アフィニティ | `src/main.cpp` + `src/core/affinity.hpp` + `src/pipeline.hpp` | ✅ 完了 (9.13 frames/s) |

**Phase 2 以降 (詳細は `docs/roadmap.md` 参照)**
- `src/io/safetensors.hpp` — safetensors 重みローダー ✅ 完了 (タスク 2-3, golden 突合, 19.0 µs/op)
- `src/kernels/vae_decode.cu` — VAE decode ✅ 完了 (タスク 2-4, SSIM 0.999992)
- `src/infer/unet.cu` + `src/infer/scheduler.hpp` — SDXL UNet + Euler scheduler ✅ 完了 (タスク 2-5, noise_pred SSIM 0.999998, 1step ~9.2s)
- `src/server/api.cpp` — cpp-httplib OpenAI 互換 HTTP サーバー ✅ 完了 (Phase 3, 生成本体は IImageGenerator 越し)
- `src/infer/diffusion.cu` + `src/server/pipeline_generator.hpp` — フル C++ 拡散パイプライン統合 ✅ 完了 (タスク **2-6a**, 20step 実画像 84.07s)。`DiffusionPipeline` (UNet×Nstep+Euler+VAE) を `PipelineGenerator` で `IImageGenerator` 化し、main.cpp が `pipeline_generator_factory` 経由で DI (重み/golden 揃えば本 txt2img・無ければ StubGenerator フォールバック・env DOLLAMA_UNET_WEIGHTS/VAE_WEIGHTS/EMBEDS で上書き)。CUDA 隔離は HAVE_CUDA ガード + .cu factory で cpp TU 非汚染
- **タスク 2-6 最適化**: 🟡 **一旦クローズ (2026-06-22)**。84.07s → **11.30s** (7.44x 改善・probe10 3.80s 比 2.97x)。S2(conv im2col+wmma) / S3-0(ビルドフラグ) / S3-C(attn wmma) / S3-B(GEMM cuBLAS GemmEx) / S3-A(VAE mid flash) / S3-D(VAE 重み常駐ハンドル) / S3-E(VAE up2/up3 FP32 direct conv → im2col+cuBLAS TF32 GEMM, 5.21s→1.16s) を実施。残る律速は UNet attention 4.60s。**研究の本丸を Phase 4 独自 LM (BitNet) に移すため、自作 CUDA の速度詰めはここで打ち切り** (残りは cuBLAS/cuDNN ライブラリフォールバック余地)。golden 維持 (UNet noise_pred SSIM 0.999996 / VAE SSIM 0.999988 / meson test 28/28 全緑)
- **2-6b (テキスト→画像の本結線)**: ✅ **完了 (2026-06-23, c3b1dac)**。2-6a の golden 埋め込み使い回し・CFG なしを解消し、prompt→実画像を dual encoder + CFG で一気通貫。① **CLIP-G の OV 化** (SDXL dual encoder: CLIP-L penult 768 ++ bigG penult 1280 → concat 2048・bigG pooled 1280 = text_embeds) ② CFG (negative prompt) で各 step cond/uncond の UNet 2 回 → host で `noise=uncond+scale·(cond-uncond)` 合成 ③ prompt→embeds 結線。Stage A golden ダンプ / B OV変換 (tokenizer・encoder L/G) / C bigG device probe / D `TextConditioner` (clip_tokenizer・clip_encoder2・text_conditioner.hpp) / E `DiffusionPipeline::generate_txt2img` / F 結線 (純cpp `IDiffusionRunner` を境界に OV/CUDA 分離・`Txt2ImgGenerator`(OV)→runner→`DiffusionRunner`(.cu)・main DI 3段フォールバック Txt2Img→Pipeline(golden)→Stub)。NPU で CLIP-L/bigG、guidance 7.5・20step、PNG 1024² var=2300・NaN/Inf なし。meson test 35/35 全緑。guidance_scale は定数 7.5 (GenRequest フィールド化は後続 TODO)。**prompt 供給元は将来 Phase 4 A の自作タグ生成 LM に差し替え**
- 部位構造化プロンプト ([[project-part-structured-prompt]], character-bible-spec §1 改訂) — 2-6 後に §11 QA ループ・案B embedding と一緒に設計
- **Phase 4 (自作タグ生成 LM)**: bitnet.hpp 33M を土台に ① tokenizer ✅ ② 訓練 (`scripts/train_bitnet.py`, hard CE dense) ✅ ③ CPU 推論 (`src/infer/bitnet.hpp` BitNetDenseInfer, golden 突合 corr 1.0) ✅ ④ **同一性条件付き化** (character-bible 入力・prompt prefix 方式・A1 retention 0.947) ✅ + **実ペア増 a12k 4 seed sweep でクローズ** — **A の効果は二分**: diverse-val 生成 F1 は **12k で seed ノイズ** (4 set/metric 判定 NO・seed 42 のみ符号反転・across −0.015±0.032 < #1 帯 sd 0.022・D6 と同型・B ~2,000 飽和と整合) で **F1 を上げる手ではない**が、**identity retention は全 seed 頑健に 0.975** (across 0.9748±0.0010・base ~0.58–0.63) = **同一性条件付けの機能基盤**。a25k は未使用保持・本番 #1 即時差し替えなし・`[B-merge-at-A]` で B(b2000)+A(identity) を出荷リトレイン 1 回でまとめ焼き (training-spec §9.10 / dataset-spec §17.7) ✅ — **dense 本線 + 2D 特化 A が text→tags / identity→tags で動作**。**蒸留 2 路線とも検証済**: 系列レベル hard CE 混合 (D2/D4, Qwen2 text 多様化) は過学習悪化で不採用 (負の結果・training-spec §10)。**soft-label KL (D5, 案A=`cache/danbooru_posts.jsonl` の prefix 条件付き共起 teacher・温度付き)** は過学習を**実際に抑制** (train-val gap 1.09→0.27、D5-L で −0.21=消失・反転 ep4→ep6) したが top10 recall は #1 を超えず (最良 0.758<0.777・soft 追従と one-hot val recall がトレードオフ) **不採用・#1 6ep 本線維持** (training-spec §11 / dataset-spec §12.3 に機構・A/B 記録)。D5 は小規模 split / early-stopping 運用で再利用価値ありの正の機構。**案C = 外部教師 TIPO-200M (D6・`--distill-ext`)** も実施・**不採用**: 案b-tagset (TIPO 生成タグを自作 vocab に完全一致写像) で単一 seed では #1 を僅かに上回って見えたが、**seed 頑健性 sweep (4 seed) で再現せず** (delta 平均 −0.0008・符号反転・#1 の seed 分散内)。再現する効果は過学習抑制のみ (gap 1.07→0.19・全 seed) = D5 と同じ正則化ノブで recall 非寄与 (training-spec §12 / dataset-spec §12.4)。**蒸留 4 路線 (D2/D4/D5/D6) は旧 proxy (テンプレ teacher-forcing recall@10) では全て不採用**。⑤ **施策 C (評価作り直し) 完了** (C-1〜C-4・training-spec §13 / dataset-spec §14): diverse-val (テンプレ外自然文・tags-stay-real) + greedy 生成 set-metrics + eval-only ハーネス + seed sweep を実装。**物差しを変えると D5 判定の符号が反転**した — 旧 recall で最下位 (0.667) の D5 が新 diverse 生成 F1 では最上位 (diverse_a 0.2024 vs #1 0.18・precision 0.38 vs 0.25)、seed sweep で D5−#1 の F1/Jaccard delta が**全 4 seed 正・各 seed paired CI が 0 を除外** (D6 の符号反転 seed ノイズと対照) = **小幅だが統計的に頑健**な実効果 (delta +0.009〜+0.012・絶対値は #1 の seed 分散帯以下)。**確定: recall@10 を主要数値から退役させ diverse 生成 set-F1 を新オフライン主指標に据える**。**未決: D5 本線昇格は新物差しで別途判断** (絶対値なお ~0.18–0.22 と低い → A 実ペア増 / D 容量増と束ねて再評価)。本番重みは #1 据え置き・無改変。⑥ **施策 B (入力多様化) パイロット完了** (B-0〜B-1-d・training-spec §14 / dataset-spec §15): C で据えた diverse-val 物差しの上で、タグ固定 (tags-stay-real)・自然文だけ多様化・**Claude 著述 Replace 500** (総件数 4,500 維持) を測ったところ、**diverse 生成 F1 が #1 を大幅に上回り** (diverse_a 0.1800→0.2675 / diverse_b 0.1921→0.3039・in-dist 退行なし)、**seed sweep (4 seed) で全判定軸 (a)(b)(c) 成立** (delta +0.06〜+0.13 = 分散帯の 4–6 倍・D5 の小幅・D6 の seed ノイズと桁違いに対照的) = **大幅かつ頑健な本物の効果**。**著者交絡は D2 で否定** (旧 Qwen2 著述 D2 を diverse-val 再採点 = B-0 しても同等改善 diverse_a 0.2701 / b 0.3134 → 「Claude 同士で似て上がった」では説明不可)。**旧 proxy では D2 同様却下されたはずで C の物差しなしには可視化されなかった** (蒸留4路線→C→B の流れ・依存連鎖の実証)。**本番重みは #1 据え置き・別名 `bitnet_dense_diverse_b` 出力**。**B-2 件数拡大 (500→2,000) ✅ + 本線昇格決裁済** (2026-06-24 ユーザー・多様化入力=tags-stay-real を既定レシピ化・正典差し替えは A 出荷リトレインで 1 回=遅延条項 `[B-merge-at-A]`)。**B-3 件数拡大 (2,000→10,000) ✅ = スケール則は ~2,000 件で飽和**: seed sweep 4 seed で delta は全 set/metric 判定 YES (seed 頑健) だが 2,000→10,000 の 5 倍増は追加利得ゼロ (delta 平坦・絶対値頭打ち) → B-2 の「頭打ちなし」を訂正。**入力多様化単体の伸びしろは ~2,000 で尽き、残低帯域 (diverse_a ~0.31 / diverse_b ~0.36) は B 件数でなく A 実ペア増 / D 容量増で取る** (B 著述を 2,000 超に積む価値は薄い・既定多様化は b2000 で足りる)。training-spec §14.9 / dataset-spec §15.8。⑦ **#6-GPU (RTX5080 で CUDA カーネル流用) ✅ 完了** (`src/infer/bitnet_gpu.cu/.cuh` BitNetGpuInfer・純 FP32 cuBLAS で CPU 版と数値一致 corr 1.0・本番重み突合済・forward seq63 で **87.5x** = 10.29ms vs CPU 900.83ms・test_bitnet_gpu 緑)。⑥ **Model B 品質スコアラ Stage 1 完了** (`src/infer/quality_gate.hpp` `QualityGate`): §11 数・位相 QA を二段構成に切り分け、一段目 = 既存 Wd14Tagger 出力 [1,N_TAGS] を消費する **OV 非依存 soft QA ゲート**を実装。`extra_arms`/`multiple_heads`/`bad_anatomy` 等 18 異常タグを軸 (`AnomalyAxis` Hands/Limbs/Head/Eyes/Ears/Mouth/Digits/GlobalAnatomy) で束ね、selected_tags.csv から名前駆動で index 解決 (WD14 版差異に頑健・CSV 欠落名は安全 skip)。`evaluate` は閾値超を hit 収集し soft flag のみ (棄却なし・§11「異常度の高いコマのみ拾う」)・`DIGITS_UNCOUNTABLE` キャラは Digits/Hands 軸スキップ。flag は B-5 FB ループ入力 + 蒸留 teacher ラベル生成器。実 selected_tags.csv で 18/18 解決・test_quality_gate 5 サブテスト緑 (OV 無し開発機で実走・新規モデル入手ゼロ)。NPU 実行性は 4-B probe 済 (scorer_device=NPU 妥当・純 conv は NPU フレンドリー)。**Stage 2 切り出し** = NPU 正確カウント検出 (DWPose 等・指の過剰カウント。モデル入手前提・別タスク。WD14 に `extra_digits`/`fused_fingers` 無く `fewer_digits` のみ = 語彙が二段構成を裏付け)。残: 実スコアラ本体 (純 conv backbone・蒸留訓練 B-3) / FB ループ B-5。B/A/D/F は diverse-val 上で測る。ternary GEMM (`src/kernels/ternary_gemm.cu`) は圧縮実験。方向性の経緯は roadmap Phase 4


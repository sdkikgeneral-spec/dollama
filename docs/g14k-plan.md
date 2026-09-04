# G-14k: attention FA2 真書き直し (#4 レバー第 2 段・G-3k 続戦) — 研究機セッション向け引き継ぎ書

## ステータス (2026-09-05 起草)

**S0 のみ実施中 (本書 = S0 の成果物)。S1 以降はすべて予定。**

| 段 | 機械 | 担当 | 状態 |
|---|---|---|---|
| **S0** 現物棚卸し + 計画書 + ゲート確定 | 開発機 | record-writer → record-auditor | 🔵 **実施中 (本書)** |
| S1 wmma のみで新カーネル純追加・単体ゲート [A1][A3][A4][A5] | 研究機 | cuda-kernel-dev | 🔲 未着手 |
| S2 [A2] + 単体速度ゲート。頭打ちなら PTX 投入判断 (**PL 再決裁点**) | 研究機 | cuda-kernel-dev | 🔲 未着手 |
| S3 call site 結線 (env) + e2e [A6][A7] + 速度 A/B | 研究機 | cuda-kernel-dev | 🔲 未着手 |
| S4 記録 | 開発機 | record-writer → record-auditor | 🔲 未着手 |

★**本書は実走した数値を 1 つも含まない。**「確認した」と読める記述は、
**起草時 (2026-09-05・開発機) にソースまたは repo 内の生ログを開いて確認した事実**であって、
G-14k の GPU 実走の結果ではない。**§3 と §11 に出てくる秒・％は、すべて G-10k T2c
(2026-09-04・HEAD `f58ba2c`) ほかの既存生ログからの引用**であり、**G-14k の分母ではない** (§5)。

**作業ブランチ**: `feat/g14k-attention-fa2` (開発機で checkout 済・起草時 HEAD `76c672d`)。
**起草**: 2026-09-05 / 開発機 (CUDA Toolkit なし) / `record-writer`。
★**G-3k は再オープンしない** (クローズ済 Pkg に別内容を継ぎ足すと記録が濁る)。G-3k の ✅ は
`#4` の**部分実現**であり、その未実装分を**新 Pkg G-14k が引き継ぐ**。
**設計の正本**: `docs/fast-mode-plan.md` の **「#4 attention を FlashAttention-2 級に」節の進捗表**
(項目 1 ✅ / 2 🟡 ダブルのみ / 3 ❌ 未実装)。
★**正本への行番号は本書に書かない** — 同ファイルは起草と同時期に `record-auditor` が R-1 監査中で、
行が動きうる。**節名で指すこと。**

---

## 0. この文書の位置づけ

- **開発機では `.cu` を 1 行もコンパイルできない** (本セッションで実測):
  `which nvcc` = not found / `CUDA_PATH` は空 / `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA`
  が存在しない。さらに**開発機の GPU は GTX 1080 Ti** (`nvidia-smi -L`) = **sm_61** で、
  `meson.build:130` の `-arch=sm_120` 成果物は**そもそも動かない**。
  → **実装・実走はすべて研究機 `KIK-WIN-RTX58` (`E:\Develop\Projects\dollama`) で行う。**
- ★**開発機で `.cu` を書かないこと (非交渉)**。コンパイルできない `.cu` を開発機で書くと、
  **未検証コードが「権威ある成果物」として研究機へ持ち込まれる**。これは `docs/g10k-plan.md` §1 が
  防いでいるのと同じ事故形である。
- 運用の形は G-10k T2a〜T2c で確立した **SSH + `scp`** をそのまま踏襲する
  (開発機で編集 → 開発機で commit → 研究機が同一 HEAD へ同期 → 開発機から SSH でビルド / 実走 →
  生ログを `scp` 回収 → 開発機で commit)。詳細と注意 (`pwsh` で実行する / ssh の inline に `$` や `|` を
  素で書かない) は `docs/g10k-plan.md` §0。
- 本書は正本 (`docs/fast-mode-plan.md` #4 節) を**実行手順・ゲート・不変条件へ落としたもの**。

---

## 1. ★本書に「書いていない」こと (非交渉)

**FA2 の実装設計は本書に一切書かない。** 具体的には次を書かない — 文章でも、疑似コードでも、
「たとえば」でも書かない:

- **タイル形状 (Br / Bc)、warp 数、warp への行割付**
- **fragment のレジスタ配置**、レーンごとの担当要素
- **online softmax の rescale をどのレーンにどう分担させるか**
- **`mma.sync` / `ldmatrix` の具体的な命令列・オペランド順・`.m16n8k16` 等の shape 選択**
- **shared memory のレイアウト / swizzle / bank conflict 回避策**

> **これらは研究機側で、現物 (`src/kernels/attention_fast.cu` / `attention.cu`) と
> FA2 の公開アルゴリズム解説を読んで自分で導出し、S1/S2 の単体ゲートで実測検証すること。
> 本書に答えは書かない。**

**理由**: 未検証の設計を文書化すると、**実装者はそれを検証済みの権威として扱い、自分で導出し直さない**。
G-8k S5 で「是正が新しい誤りを持ち込む」形が 5 ラウンド続いたのと同根 (doc の 1 行から推測して断定する型)。
とくに attention のレジスタ内 fragment レイアウトは、**間違っても「動くが数値が違う」形で通ってしまう**
ため、doc 由来の権威が最も危険な箇所である (§9-1)。

**書いてよいもの**: 到達目標 / 禁止事項 / ゲート定義 / 段構成 / **現物の事実** (`file:line` 付き・
「今こうなっている」) / **危険箇所の指摘** (答えは書かない = `docs/g10k-plan.md` §9 と同じ書式)。

---

## 2. 計器 (被験変数を通るかを `file:line` で確認したもの)

★**計器を指定する決裁の規則** (`docs/g10k-plan.md` §2 で規則化・G-4k S3 / F1 / T6② / F3 の
**同型 4 連**を受けたもの) を本 Pkg にもそのまま適用する:

> **計器を指定する決裁は、次の 3 点を `file:line` の一次証拠で示さない限り出さない・採らない。**
> ① どの関数が呼ばれるか ② その関数が被験コード行に到達するか ③ 被験構成で実際にその枝を通るか
> **計器名を書くときは必ず `(B=1)` / `(B=2)` を併記する。過去 4 回の事故はすべて B の見落ちだった。**

### 2-1. 現物の計器一覧 (起草時にソースを開いて確認)

| 計器 | B | ①呼び出し | ②到達先 | ③被験を通るか (現物) |
|---|---|---|---|---|
| `test_attention_fast` の **`correctness_case`** (CPU 参照) | **B=1 と B=2 の両方**あり | `src/tests/test_attention_fast.cu:384-391` (6 ケース) | `run_gpu(..., use_fast=true)` `:225` → `launch_attention_fast` `:144` | ✅ fast 経路へ到達。**ただし形状が小さい** — 最大 `Sq=256` / `Sk=77` (`:385`)。**SDXL 実 shape (Sq=4096) は CPU 参照で 1 度も突合していない** |
| 同 **`ssim_gate_case`** (vs `launch_attention`) | **B=1 と B=2** (`:396-399`) | `main` `:396-399` | base = `run_gpu(..., use_fast=false)` `:269` → `launch_attention` / fast = `:270` | ✅ SDXL 実 shape (self 4096 / cross 4096×77) を覆う。**ただし `H=1` に絞ってある** (理由はソースコメント `:395`「baseline (1 warp/block) を回すため」) |
| 同 **`bench_pair`** (速度) | ★**現状 B=1 のみ** | `main` `:408-410` の 3 本 (`(1,1,4096,4096,64)` / `(1,8,1024,1024,64)` / `(1,8,4096,77,64)`) | `run_once` `:318-334` | ⚠ **B=2 の呼出が 1 本も無い**・self は `H=1`。**PL が hard に指定した「SDXL self 実形状・B=2」を現状の計器は通らない** (§7-2 / §2-2) |
| `test_unet_fast` | ★**B=1** | `src/tests/test_unet_fast.cu:258` / `:280` が `launch_unet(uh, ...)` | `src/infer/unet.cu:1462` = `launch_unet_impl(w, **1**, ...)` | ✅ attn_fast=true で fast 経路に到達するが **B=1**。B=2 の被験は通らない |
| `test_diffusion_batch2` parity 節 (steps=4) | **B=2** (batch2 構成のみ) | `src/tests/test_diffusion_batch2.cu:251` steps=4 / `:262-268` `gen` ラッパ / `run_config` 呼出 4 箇所 `:343`(off) `:348`(batch2 単独) `:353`(fst=attn+batch2) `:360`(epi=fast+epi) | `generate_txt2img` → `use_batch2` (`src/infer/diffusion.cu:755`) が true なら `launch_unet_batched(handle, **2**, ...)` (`:887-888`) | ✅ `fst` / `epi` は `attn_fast=true` かつ `batch2=true` = **B=2 で fast 経路**。`off` は attn_fast=false = **回帰アンカー側** |
| 同 DB2_BENCH 節 (e2e 秒) | **B=2** | `:555` の `getenv("DB2_BENCH")` → `bench_pipe` `:567-598` (warmup `:584` + `iters` 回の min `:586-596`) | 同上 | ✅ 3 構成を **1 プロセス**で連続実行。窓は `steady_clock` (`:589` / `:592`)・その前後に `cudaDeviceSynchronize` (`:588` / `:591`) |
| **T2b の内訳 dump** `-> attention only` | **B=2** | `src/infer/diffusion.cu:1009` (`generate_txt2img` 内・`DOLLAMA_PROFILE=1` のとき) | `pc.cat_attention_sec` = `src/infer/unet.cu:717` (self) と `:761` (cross) の `ScopedSyncTimer` | ✅ **B=2 経路で attention の絶対秒が読める**。★**`ScopedSyncTimer` は ctor と stop で `cudaDeviceSynchronize()` する** (`src/infer/unet.cu:67-93`) = **同期入りの host 壁時計**。**characterization 専用** |
| `prof_arena_e2e` | **B=2** (`PROF_FAST=1` 既定) | `src/tests/prof_arena_e2e.cu:233` の `generate_txt2img` | 同上 | ✅ profile OFF の e2e 秒 / VRAM peak。**meson test 未登録** (`src/meson.build:889-890` は executable 定義のみ) |
| nsys (`nsys profile`) | 被計測は `prof_arena_e2e` = **B=2** | — | — | ✅ カーネル別 GPU 時間・grid/block・launch 数。T2c の採取条件は §11-2 |

### 2-2. ★S1 の最初にやること (非交渉)

**実装者は `bench_pair` のソース (`src/tests/test_attention_fast.cu:290-367`) を自分で読み、
「計器が被験変数を通るか」を確認してから使うこと。docs の 1 行で計器を選ぶことを禁止する。**
最低限、次の 3 点をソースで確認してから速度ゲートに使う:

1. **warmup と反復**: `median_of` は warmup 3 回 (`:338-340`) → `iters` 回 (既定 20 = `:290`) → 中央値 (`:348-349`)。
2. **同期位置**: `run_once` は `cudaEventRecord(start)` → 起動 → `cudaEventRecord(stop)` →
   **`cudaEventSynchronize(stop)`** (`:320-333`) = **1 回ごとに同期する**。連続起動のパイプライン化は測らない。
3. **経路の選択肢が 2 つしかない**: `run_once` の分岐は `use_fast` の bool 1 本で
   `launch_attention_fast` / `launch_attention` の **2 択** (`:321-328`)。
   → **「現行 fast」対「新カーネル」の 3 経路目を測るには、ハーネス側の拡張が要る**
   (実装は S1/S2 = `cuda-kernel-dev` の仕事。**本書は拡張方法を指定しない**)。

**この確認を飛ばして計器を選ぶことは、このプロジェクトで 4 回連続で起きた事故型である**
(G-4k S3 の「測っている変数が epilogue でない」/ G-10k の F1 / T6② / F3)。

---

## 3. スコープと狙い

### 3-1. 現物の事実 (起草時にソースを開いて確認・`file:line` 付き)

**対象**: `src/kernels/attention_fast.cu` の `attention_flash_wmma_fast_fp16` (`:58-286`) と、
そのホストラッパ `launch_attention_fast` (`:291-358`)。**新カーネルは純追加**で、
**既存 2 本 (`launch_attention` / `launch_attention_fast`) は無改変で残す** (§4-1)。

| # | 現物 | 一次証拠 |
|---|---|---|
| 1 | **K/V タイルが 16 行**。`WF_N = 16` (`:42`) / `WF_KV_ROWS = 16` (`:47`)。タイル数は `ntiles = ceil(Sk/16)` (`:132`) なので **Sk=4096 で 256 反復** | `src/kernels/attention_fast.cu:42`, `:47`, `:132` |
| 2 | **その 256 反復のそれぞれに `__syncthreads()` が 2 回**入る (タイル可視化 `:184` / 次タイルのバッファ再利用前 `:270`)。加えて warp 内に `__syncwarp()` が `:201` `:244` `:260` `:267` | 同 `:184`, `:270`, `:201`, `:244`, `:260`, `:267` |
| 3 | **出力アキュムレータ `acc` が shared の FP32**。動的 shared から切り出している (`:102`)。レイアウト設計は `:85-93` のコメント。warp ローカルのビューが `wacc` (`:162`) | 同 `:85-93`, `:102`, `:162` |
| 4 | **online softmax の rescale が 16 レーンのみ**で走る (`:204` の `if (lane < WF_WARP_ROWS)` = lane 0..15。warp の残り 16 レーンはこの区間で遊ぶ)。しかもその 1 レーンが **`Dh` を逐次ループ**する (`:227-230` の `for (int d = 0; d < Dh; ++d) wacc[m*Dh+d] *= corr;`)。P の生成も `:234-240` で `WF_N=16` を逐次 | 同 `:204`, `:227-230`, `:234-240` |
| 5 | **P·V が `store_matrix_sync` → shared → 加算 の往復のまま**。`:259` で `o_frag` を shared (`ws`) へ書き戻し、`:261-266` で shared の `wacc` に加算する | 同 `:250-267` (とくに `:259`, `:261-266`) |
| 6 | **使用している命令は wmma 16×16×16 と `__pipeline_memcpy_async` だけ**。`asm` / `__asm__` は **`src/` 全体で 0 件**、`ldmatrix` / `mma.sync` の PTX 直書きも 0 件、TMA (`cuTensorMap`) 0 件、cuBLASLt 0 件、CUDA Graph (`cudaGraph`) 0 件、非既定 stream (`cudaStream_t` / `cudaStreamCreate`) **0 件** | grep (本セッション実行)。`__pipeline_*` の実体は `attention_fast.cu:146-147`, `:153`, `:177-178`, `:182`。`#include <mma.h>` は **3 TU のみ** (`attention.cu:27` / `attention_fast.cu:29` / `gemm.cu:39`) |
| 7 | **FP8 のカーネルは存在しない**。`--fp8` / `DOLLAMA_FP8` は **CLI / FastConfig の配管だけ** (`src/server/fast_config.hpp:22`, `:76-79` (FP8 env), `:93-97` (fp8→fast 含意) / `src/main.cpp:203-213` / `src/infer/diffusion.cu:297`) で、`__nv_fp8` / `e4m3` / `e5m2` は `src/` に 0 件。FP8 は **G-5k のスコープ** | grep (本セッション実行) |
| 8 | **UNet の attention は Dh=64 固定**。`transformer_block` が `const int Dh = 64;` (`src/infer/unet.cu:684`)・`heads = C / Dh` (`:685`)。transformer が載る解像度レベルは **64² (`C=640` → heads=10, tokens=4096)** と **32² (`C=1280` → heads=20, tokens=1024)** の 2 つで、128² には transformer が無い。cross は `ctx_tokens = 77` / `ctx_dim = 2048` (`:937-938`) | `src/infer/unet.cu:684-685`, `:937-938`, `:1094-1110` (down1・transformer2d は `:1103` `:1110`) / `:1119-1134` (down2・`:1128` `:1134`) / `:1144-1152` (mid・`:1152`) / `:1167-1188` (up0・`:1188`) / `:1204-1224` (up1・`:1224`) |
| 9 | ★**`Dh=80` は現行の呼び出し側からは生成されない形状である**。`launch_attention*` の呼出は `src/` に **UNet の 4 箇所** (`unet.cu:722` / `:726` / `:766` / `:770`・すべて `Dh=64`) と **VAE mid の 1 箇所** (`vae_decode.cu:683`・`Dh=C=512`) しかない。`test_attention_fast.cu:388-389` のコメントは「Dh=80 (SDXL self-attn)」と書くが、**本実装の SDXL UNet は Dh=64 である** | 同上 + `src/kernels/vae_decode.cu:683` |
| 10 | **VAE mid attention は `--fast` でも default カーネルを通る**。`src/kernels/vae_decode.cu:683` が `launch_attention(...)` を**直接**呼んでおり、VAE 側に `launch_attention_fast` の呼出は 0 件。`launch_attention` は `Dh%16==0` かつ shared に乗れば `attention_flash_wmma_fp16<<<grid, **32**, shmem>>>` = **1 block = 1 warp** を選ぶ | `src/kernels/vae_decode.cu:683` / `src/kernels/attention.cu:618-658` (コメント `:636` / 起動 `:653`) |
| 11 | ★**VAE が default カーネルなのは「呼び先」の問題であって「fast が Dh=512 を扱えない」からではない**。`launch_attention_fast` のフォールバック条件は (a) `Dh % 16 != 0` (`:302-306`) と (b) `{4,2,1}` のどの nwarps でも shared 上限 (`kMaxOptinShmem` = 224KB `:310`) に乗らない (`:334-338`) の 2 つだけである | `src/kernels/attention_fast.cu:300-338` |

### 3-2. スコープの未確定点 (★PL 確認事項・本書では確定していない)

- **正本 `docs/fast-mode-plan.md` #4 節は「VAE mid attention も G-14k のスコープ」と書いている**が、
  本 Pkg の PL 決裁 (段構成 S1〜S4 / ゲート [A1]〜[A7] / 形状網羅) は
  **UNet 経路の形状しか列挙していない** (self 4096 / cross 4096×77 / Dh=80 / B=1,B=2)。
  → **VAE mid (B=1, H=1, Sq=Sk=16384, Dh=512・1 発 514ms) を S1〜S3 のどこで扱うかは未決。**
  **S1 着手前に PL へ上げること。** 勝手にスコープへ入れない・勝手に落とさない。

### 3-3. 面積 (★すべて既存生ログの引用。G-14k の分母ではない)

**出典**: `docs/logs/g10k-baseline/` (commit `1c85dd9`)。採取条件は §11-2。

- **nsys (profile 計装 OFF)・生成 #2 (重み転送の尾を含まない側)**:
  `attention_flash_wmma_fast_fp16` = **6.3430 s (走行#1) / 6.3438 s (走行#2)**、
  **n=2800 launch / 生成 1 枚**、**Σ kernel time 分母で 60.58% / 60.59%**。
  `attention_flash_wmma_fp16` (VAE mid) = **0.5147 s / 0.5143 s**、**n=1** = **単発 514ms**、
  同分母で **4.92% / 4.91%**。
  ★**この `%` の分母は Σ kernel time (10.4705 s / 10.4707 s) であって wall (10.6615 s / 10.6608 s) ではない。**
  attention 2 本の合計 **6.858 s** は **Σkernel 分母で 65.50% / wall 分母で 64.32%** = **同じ測定の分母違い**。
  **どちらの分母で言っているかを必ず併記する。**
  ★**この 6.858 s / 65.50% / 64.32% の 3 つは合計と除算による導出値**であり、ログにその行は無い
  (元の 4 値 6.3430 / 0.5147 / 10.4705 / 10.6615 はログにある = §11-2)。
  ★**`n=2800` は生成 1 枚あたり**である (nsys の窓が 1 枚ぶん)。**導出値による裏取り**:
  transformer_block は down1 2×L2 + down2 2×L10 + mid 1×L10 + up0 3×L10 + up1 3×L2 = **70 個**、
  各 block が self と cross を 1 回ずつ呼ぶ (`src/infer/unet.cu:722`/`:766`) ので
  70 × 2 × 20 step = **2800** で一致する (batch2 = 1 step 1 forward)。
- **UNet / VAE の分割** (走行#2): UNet 20step = wall 9.6599 s / Σkernel 9.4756 s
  (attention 6.3438 s = **UNet Σkernel の 66.95%**)、VAE decode = wall 1.0009 s / Σkernel 0.9951 s
  (attention 0.5143 s = **VAE Σkernel の 51.69%**)。
- **grid 別の内訳** (走行#2): `attention_flash_wmma_fast_fp16` は 2 種類しか出ない —
  `grid=(64,20) block=128` **n=400 / total 3.5174 s / avg 8.793 ms** と
  `grid=(16,40) block=128` **n=2400 / total 2.8264 s / avg 1.178 ms**。
  `grid.y = B*H` なので前者が **B=2 × H=10** (tokens=4096 レベル)、後者が **B=2 × H=20** (tokens=1024 レベル)。
  `block=128` = **4 warp** (**導出**: `nwarps = blockDim.x / 32` = `src/kernels/attention_fast.cu:69`)。
  ★★**この 2 行は self と cross を分離していない** — grid は `(qtiles, B*H)` で決まり
  **`Sk` を含まない**ため、**同じ tokens レベルの self と cross は同一 grid に潰れる**。
  **「self = X s / cross = Y s」を本ログから読み取ることはできない。断定しないこと。**
- **launch 谷**: **generation #2 で 0.1910 s = wall の 1.79% (走行#1) / 0.1901 s = 1.78% (走行#2)**、
  GPU busy **98.21% / 98.22%**。★**generation #1 は 3.54% / 3.43%** (重み転送の尾を含む区間) なので、
  **どちらの区間の値かを必ず明示すること。**
- **カーネル起動数**: 生成 1 枚あたり **67,032 発**。うち **UNet 20step が 65,889 発 = 3,294.4 発/step**、
  VAE decode が 1,143 発。
- **T2b 内訳 dump (profile ON・`ScopedSyncTimer` 同期入り・別条件)** の同一走行:
  出荷 `--fast` (fast+epilogue) 本走で `-> attention only` **6.387 s** / `transformer (attn/gemm)` 8.693 s /
  `UNet step total` 9.729 s / `resnet` 0.854 s。`default` (attn off・逐次 2 forward) 本走は
  `-> attention only` **9.218 s**。
  ★**profile ON の 6.387 s と nsys の 6.3438 s は別条件の別測定である** (前者は同期入り host 壁時計、
  後者は GPU カーネル時間)。**同じ表に並べて差を語らないこと。**
- **e2e (参考値)**: `prof_arena_e2e` (profile OFF・HEAD `e17374c`) が **10.9628 / 10.6465 s** (run1) と
  **10.7505 / 10.6561 s** (run2)。`test_diffusion_batch2` DB2_BENCH (**profile ON**・HEAD `f58ba2c`) が
  `fast+epilogue` **10735.9 ms**。★**いずれも G-14k の分母にしない** (§5)。

### 3-4. ★見込みは書かない

「FA2 級ならどれだけ縮むか」の概算は **本書に数値として書かない**。
親セッションが持っていた「−4.6〜−5.5 s」という値は**概算であり実測ではない**ため、
**合否にも見積りにも使わない** (`docs/g10k-plan.md` が「見込み 0.5〜1.0s を合否に使わない」と
書いたのと同型)。**回収量は S3 の A/B で初めて確定する。**

---

## 4. 不変条件 4 本 (破ったら不合格)

1. **既存 2 本を 1 バイトも変えない。**
   `launch_attention` (`src/kernels/attention.cu:602-706`) と
   `launch_attention_fast` (`src/kernels/attention_fast.cu:291-358`)、および両者が呼ぶ
   カーネル 4 本 (`attention_fp16` `attention.cu:137` / `attention_flash_fp16` `:249` /
   `attention_flash_wmma_fp16` `:393` / `attention_flash_wmma_fast_fp16` `attention_fast.cu:58`) は**無改変**。
   新カーネルは**純追加**とする。G-3k が `attention.cu` を 1 行も触らずに `attention_fast.cu` を
   新設した前例 (`f3b86fc` の `--stat` に `attention.cu` は現れず `attention.cuh` の +19 行のみ) と同じ形。
2. **キルスイッチ必須。** env で**旧経路へ完全にフォールバック**できること。
   既存の型は 2 つあり、どちらも **getenv キャッシュ (プロセス単位で固定)**:
   `cublas_disabled()` (`src/kernels/gemm.cu:377-386`・`DOLLAMA_GEMM=wmma`) と
   `device_arena_pool_enabled()` (`src/kernels/device_arena.cu:215-232`・`DOLLAMA_POOL=0`)。
   ★**プロセス単位固定ゆえ A/B は必ず別プロセスになる** (§5 / §7-2)。
   ★env 名は `src/` の既存 32 名と衝突させないこと。**`DOLLAMA_CONV_BATCH` は G-10k が予約済み**
   (現時点で `src/` に実体は無い = grep 0 件) なので使わない。**名前は実装者が決める** (本書は指定しない)。
3. **FP32 蓄積の維持は非交渉。** running max / running sum / O accumulator を FP16 に落とさない。
   **FP8 は G-5k のスコープであって G-14k ではない** (§3-1 の #7 = FP8 カーネルは現状 0 本)。
4. **default 経路 (attn off) の出力を動かさない。**
   `--fast` を付けない経路は `launch_attention` を通る (`src/infer/unet.cu:726` / `:770`) ので、
   新カーネルが default 側へ漏れていないことを [A7] で検査する。

### ★禁止事項 (PL 決裁 2・非交渉)

- **CUTLASS / CuTe / cuDNN / cuDNN-frontend のヘッダ・コードを取り込まない (`#include` も不可)。**
  ★注: nsys のカーネル名に `cutlass::Kernel2<...>` や `nvjet_sm120_...` が出るのは
  **cuBLAS の内部実装**であって、我々のソースが CUTLASS を include しているからではない
  (`src/` に `cutlass` / `cute` / `cudnn` の文字列は 0 件 = grep 済)。**この行を根拠に禁止を緩めない。**
- **FlashAttention 等、第三者実装のコードを移植・貼り付けしない。**
  論文・公開解説の**アルゴリズム**を読んで自分で実装するのは可。
- **inline PTX (`asm volatile` による `mma.sync` / `ldmatrix` / `cp.async` / TMA) は許可する。**
  `docs/fast-mode-plan.md` が禁じているのは「研究コアをライブラリに明け渡すこと」であり、
  PTX を直接書くのは**その逆 (より深く自作する側)** である。wmma API は同じ命令の薄いラッパにすぎない。
- ★**ただし着手順序を課す (非交渉)**: **S1 は wmma API のみで到達できる上限を実装・実測する。
  PTX は S2 以降、S1 の頭打ちが実測で示されてから。** 理由 = ①PTX は保守コストとバグ表面が大きく、
  必要性が未実測の段階で入れると**「速くならない PTX」が恒久債務として残る** ②
  **「wmma だけでどこまで行けたか」自体が研究成果になる**。

---

## 5. 数値方針

- ★**bit-exact は狙わない。明示的に放棄する。**
  G-3k が偶然 bit 一致した (`f3b86fc` 本文「カーネル SSIM=1.0 (ビット一致・16タイル reduction 順不変)」)
  のは **16 タイルの reduction 順を既存と同一に保ったため**であって、FA2 化で reduction 順が変われば
  **必ず崩れる**。**G-3k の SSIM=1.0 を G-14k の期待値として引き写さないこと**
  (この禁止は正本 `docs/fast-mode-plan.md` #4 節にも明記されている)。
- **既存 SSIM ゲート 0.9999 は floor として継承し、引き下げない。**
  現物は `src/tests/test_attention_fast.cu:281` の `const bool ok = (s >= 0.9999);`。
- **tol を緩めない。** `correctness_case` の物差しは
  `atol_for(K) = 3e-3 + 2e-5*K` / `rtol_for(K) = 5e-3 + 1e-5*K`・**K = Dh + Sk** (`:56-63`, `:228-230`)。
  ★**緩めたくなった状況 = バグである** (§8 ケース A)。
- **決定性を tol に逃がさない。** 非決定的 reduction / atomic の混入は
  **同一入力 3 runs の memcmp** ([A3]) で捕まえる。前例の型は
  `src/tests/test_conv2d.cu:548-564` (設計コメント) / `:692-719` (実装) と
  `src/tests/test_unet_fast.cu:330-405` (アリーナ bit 一致 3runs)。
- ★**g>1 で FP16 の微差をゲートしない。被験変数は g=1.0 で分離する。**
  出典は G-4k S3 の教訓 (`docs/fast-mode-plan.md` #5 節「ゲート設計の欠陥と是正」)。
  実害の実測もある: (fast+epi) vs fast は g=1.0 で 0.999477 だが **g=7.5 で 0.996533** =
  初版ゲートなら FAIL (CLAUDE.md 計測表 G-4k S3 行)。
- ★**分母は必ずそのセッション内で取り直す (非交渉)。**
  既報の **10.65〜10.96 s** (`prof_arena_e2e` / HEAD `e17374c` / profile OFF) と
  **10735.9 ms** (DB2_BENCH / HEAD `f58ba2c` / **profile ON**) は**参考値であり分母にしない**。
  理由: ①条件が違う (profile ON/OFF・ハーネスが違う) ②機体クロック / 熱で全体が動く
  (G-4k S3 で同一走行内 **+9.9%**・全体 ~18% のドリフト実績)。
- **判定は同一セッション内の比 (倍率 / 削減率) で行う。絶対秒は条件付きで併記する。**
- ★**profile ON の内訳 (T2b 形式) は characterization 専用。構成をまたいだ比較に使わない。**
  `ScopedSyncTimer` は ctor と stop で `cudaDeviceSynchronize()` する (`src/infer/unet.cu:67-93`) ため、
  同期の摂動が入り、かつ e2e 秒そのものが膨らむ。

---

## 6. タスク表 (S0〜S4・完全直列)

**実行順**: **S0 → S1 → S2 → S3 → S4**。**GPU 実走の並列は禁止** (同一 GPU を 2 プロセスで
叩いた秒は意味を失う)。★**S2 の末尾に PL 再決裁点がある** (PTX 投入判断)。

| 段 | 機械 | 担当 | 触るファイル | DoD |
|---|---|---|---|---|
| **S0** | 開発機 | record-writer → record-auditor | `docs/g14k-plan.md` (本書) / `.claude/agents/perf-profiler.md` | 現物棚卸しの一次証拠 + 計画書 + ゲート確定。**`src/` は 0 行**。監査が通るまで S1 に進まない |
| **S1** | 研究機 | cuda-kernel-dev | `src/kernels/` (新規 TU) / `src/tests/test_attention_fast.cu` | ★**wmma API のみで到達できる上限を実装・実測する。PTX は使わない。** 新カーネルは純追加 (§4-1)・キルスイッチ付き (§4-2)。単体ゲート **[A1] [A3] [A4] [A5]** を緑にする。★着手の最初に §2-2 の計器確認を済ませ、**確認したことを報告に書く**。生ログは `docs\logs\g14k-*\` へ (`2>&1` 必須) |
| **S2** | 研究機 | cuda-kernel-dev | 同上 | **[A2]** (SSIM ≥ 0.9999 floor) + **単体速度ゲート [S-1] [S-2]** (§7-2)。★**ここで「wmma だけでどこまで行けたか」を数値で確定させる** — これ自体が研究成果である。**頭打ちなら PTX 投入を PL へ上げる (再決裁点 = §8 ケース C)。PL 決裁が出るまで PTX を書き始めない** |
| **S3** | 研究機 | cuda-kernel-dev | `src/infer/unet.cu` (call site) ほか | env で新経路を選べるよう結線し、e2e ゲート **[A6] [A7]** + **速度 A/B** (§7-2)。★**着手前に最新を取り込むこと** — call site は G-10k の epilogue ガード周辺と近い |
| **S4** | 開発機 | record-writer → record-auditor | `docs/` / CLAUDE.md 計測表 / commit 本文 | 記録。★**合否は実数値とセットで書く。「改善」「効率化」で FAIL を包まない** |

★**段レビュー (非交渉)**: **各段の末尾に Fable モデルのレビュアーを実装者と別主体で立てる**
(`docs/g10k-plan.md` §10 と同じ規律)。**指摘が出たら次段に進まず、その段で是正する。**

---

## 7. ゲート

### 7-1. 数値ゲート (hard 7 本)

**実装方法は指定しない — 検査すべき性質だけを書く。**

| # | 検査すべき性質 | 強度 |
|---|---|---|
| **[A1]** | **CPU 参照パリティ**。`src/tests/test_attention_fast.cu` の既存 CPU 参照 (`cpu_attention` `:66-118` を使う `correctness_case` `:212-251`) を、**同一の物差しで**新カーネルに課す。★**tol を緩めることを禁止** (緩めたくなった状況 = バグ) | **hard** |
| **[A2]** | 既存 `launch_attention` との **SSIM ≥ 0.9999**。既存ゲート (`:281`) を **floor として継承**し、**引き下げ禁止** | **hard (floor)** |
| **[A3]** | **決定性**: 同一入力 **3 runs で memcmp bit 一致**。非決定的 reduction / atomic 混入の検出。★**tol に逃がさない主ゲート** | **hard** |
| **[A4]** | **フォールバック不変**: 新経路が扱わない形状が既存経路へ落ち、その出力が**現行と bit 一致**すること | **hard** |
| **[A5]** | **形状網羅**: self (`Sq=Sk=4096, Dh=64`) / cross (`Sq=4096, Sk=77`) / `Dh=80` / **B=1 と B=2 の両方** (★本番は batch2)。**上記のゲート集合がこの形状をすべて覆うこと** | **hard** |
| **[A6]** | **e2e @g=1.0 で SSIM ≥ 0.999**。★**g>1 でゲートしない** | **hard** |
| **[A7]** | **default (attn off) 無改変** — 出力 **bit 一致** (回帰アンカー) | **hard** |

**characterization (合否なし)**: g=3 / g=7.5 の SSIM・絶対秒・profile ON の内訳 (T2b 形式)。

★**[A5] についての現物の注意** (事実だけ・答えは書かない):

- 既存の **CPU 参照は最大 `Sq=256`** (`:385`) で、**SDXL 実 shape (Sq=4096) を CPU 参照で突合したことは
  一度も無い**。[A1] をその形状へ広げるかは実装判断だが、広げるなら **CPU 側の計算量が
  `Sq·Sk·Dh` で効く**ことを見積もってから決めること (`cpu_attention` は素朴 3 重ループ `:82-113`)。
  ★**この点は PL 決裁の文面だけからは一意に決まらない** ([A1] の物差しと [A5] の形状集合の交点)。
  **迷ったら PL へ上げる。** 勝手に「4096 は [A2] だけで足りる」と決めない。
- 既存の **SSIM ゲートは 4096 形状を `H=1` に絞っている** (`:394-399`・理由はソースコメント `:395`
  「baseline (1 warp/block) を回すため」)。**本番の head 数は 10 (tokens=4096) / 20 (tokens=1024)**
  である (§3-1 の #8) ため、**`H=1` の結果を「本番形状で検査した」と書かないこと。**
- **`Dh=80` は現行の呼び出し側からは生成されない形状**である (§3-1 の #9)。
  PL 決裁で [A5] に含まれているので**検査はする**が、**記録に「SDXL の実形状」と書かないこと。**

★**[A7] についての現物の注意**: **「同一ツリーの二連走 bit 一致」で代用してはいけない。**
それは **determinism 検査であって無改変証明ではない**。この罠は
`src/tests/test_diffusion_batch2.cu:46-49` (ファイル冒頭のゲート設計注記) と `:438-440` (GATE3 直前の注記) が明文化している。
無改変の証明は **改修前後の同一 seed 生成物の sha256 突合** (G-8k S6 / G-10k T2b で確立した型)。
なお `test_unet_fast` の default 側ゲートは **tol** である (`:277-278` = `MAE > 5e-3` / `SSIM < 0.99`) ので、
**「既存テストが default の bit 一致を守っている」と書かないこと。**

### 7-2. 速度ゲート

**単体 (hard・2 本)**

| # | 内容 |
|---|---|
| **[S-1]** | `bench_pair` を計器とし、**SDXL self 実形状・B=2** で **現行 `launch_attention_fast` 比 ≥ 2.0x** (暫定 hard) |
| **[S-2]** | **cross 形状 (`Sq=4096, Sk=77`) が ≥ 1.0x** (= 遅くしない) |

★**計器の現状 (S1 で拡張が要る)**: `bench_pair` の呼出は `main` `:408-410` の **3 本すべて B=1**、
かつ self は `H=1`。**[S-1] の「B=2」を通す呼出は現状 1 本も無い**。
また `run_once` の分岐は 2 択なので **「現行 fast」対「新カーネル」を測る 3 経路目が無い** (§2-2)。
→ **S1/S2 でハーネスを拡張すること。拡張方法は本書で指定しない。**

**e2e (判定用)**

- 計器: `test_diffusion_batch2` の **DB2_BENCH**。
- 条件: **profile OFF・warm・`--fast`・B=2・20 step**。
- 取り方: **A→B→A の 3 プロセス** (キルスイッチは getenv キャッシュ型 = プロセス単位固定 = §4-2)。
- 判定: **同一セッション内の比**。**絶対秒は条件付き記載**
  (機体クロック / 熱ドリフトの前例: G-4k S3 で全体 ~18%・同一走行内の default 再測で **+9.9%**)。
- ★**「同一走行で A/B を取った」と書いてはいけない** — この repo の正典用法では **走行 = プロセス**
  (`docs/measurements-log.md:75`)。

---

## 8. 不合格時の分岐 (着手前に固定・実走後に決めない)

### ケース A: [A1] または [A3] が赤 → **緩めず、実装バグとして切り分けて PL へ。**

- **tol を緩めない。閾値を入れない。memcmp を tol に置き換えない。**
  ★**緩めたくなった状況そのものがバグの証拠である。**
- 切り分けの順序 (**答えではなく順序だけを指定する**):
  1. **[A3] (3 runs 自己一致) を先に見る** — 非決定性か系統差かで原因の種類が変わる。
  2. **キルスイッチで旧経路に落として同じ入力が通るか**を見る ([A4] が緑なら旧経路側は健全)。
  3. **形状を減らして (B=1 / H=1 / 小 Sk) 再現最小形を作る。**
  4. それでも不明なら **PL へ上げる**。
- ★**g>1 での再判定は禁止** (§5)。**「g=7.5 では合う」は情報ではない。**

### ケース A': [A2] が floor (0.9999) を割った → **緩めず PL へ。**

floor は既存ゲートの継承であって新設ではない。割った時点で**「reduction 順の差」では
説明できない量**の可能性が高い。[A3] と [A4] の結果を添えて PL へ上げる。
**実装者や実走担当が現場で緩めてはいけない** (ゲートの自己成就の禁止)。

### ケース B: 全ゲート緑・秒が動かない → **陰性クローズを許容。**

- **revert はしない。** キルスイッチ付きの opt-in として残し、
  **「attention を書き直しても秒が動かなかった律速はどこか」の診断を記録に残す**
  (G-4k S2 / S3 の陰性クローズと同じ扱い・次の Pkg の入力にする)。
- ★**「wmma だけでどこまで行けたか」は陰性でも成果である。**必ず数値で残す。
- ★**陰性を「改善」「効率化」「短縮」の語で包まない** (合否は実数値とセット)。

### ケース C: S2 で wmma が頭打ち → **PTX 投入は PL 再決裁。**

- **実装者が現場で PTX へ進まない。**「頭打ち」の判断材料 ([S-1] の実測倍率・profile の内訳) を
  添えて PL へ上げ、**決裁が出てから S2 の続きに入る**。
- 理由は §4 の禁止事項節に記した 2 点 (恒久債務の回避 / wmma 上限自体が成果)。

---

## 9. 実装者への危険箇所 (現物確認済み・答えは書かない)

1. ★★**fragment レイアウトの取り違えは「動くが数値が違う」形で通る。**
   これは **G-10k の stride 写像と完全に同型のリスク**である。レジスタ内の fragment レイアウトを
   取り違えても、カーネルはクラッシュせず、それらしい値を出す。
   **[A1] の CPU 参照ゲートが、それを捕まえる唯一の網である。**
   → だから **[A1] を後回しにしない・[A1] の tol を緩めない・[A1] を「小さい形状だから」と軽く見ない。**
   ★**[A2] (SSIM) は既存カーネルとの比較なので、両方が同じ向きにずれたら気付けない。**
   [A2] は [A1] の代わりにならない。
2. **`bench_pair` は 1 回ごとに `cudaEventSynchronize` する** (`:330`)。
   したがって **連続起動のパイプライン化 (launch レイテンシの重なり) は測らない**。
   単体で 2.0x でも e2e が動かない可能性は原理上ある — **単体倍率を e2e 倍率に翻訳しないこと**
   (逆も同じ: 単体が伸びなくても e2e が動く保証も無い)。
3. **`launch_attention_fast` は「非対応なら `launch_attention` へ落ちる」構造**である
   (`:302-306` と `:334-338` の 2 箇所)。**新カーネルを同じ形にするなら、意図せずフォールバックしても
   無言で通る**ことに注意する。[A4] は**出力の bit 一致**しか見ないので、
   **「意図せずフォールバックしていた」は秒でしか検出できない**。
   ★**どの形状が新経路に乗ったかを実行時に観測できる手段を用意すること** (手段は指定しない)。
4. **cross は `Sk=77` = タイル数が self の 1/50 以下**になる (`ntiles = ceil(Sk/16)` = `:132`)。
   **self に最適な形が cross で退行しうる**ため [S-2] (≥1.0x) を hard に置いてある。
   ★**nsys の grid 別内訳では self と cross が同一 grid に潰れている** (§3-3) ので、
   **「cross は速い / 遅い」を T2c のログから読み取らないこと。**
5. **`__pipeline_wait_prior` の引数は「残す batch 数」**であって「待つ batch 数」ではない
   (現物の使い分けは `:178` の `(1)` と `:182` の `(0)`・どちらもコメント付き)。
   段数を増やすなら**この意味を取り違えないこと** — 取り違えても**動くが競合する** (= 1 と同型)。
6. **`--use_fast_math` が既定 ON** である (`meson.build:131-134` / `meson_options.txt` の
   `option('fast_math', ..., value: true)`)。現物カーネルも `__expf` を使っている
   (`attention_fast.cu:224`, `:237`)。**数値ゲートが赤になったとき、原因候補としてここを忘れないこと**
   (ただし **既存経路も同じフラグで通っている**ので、フラグ自体を被疑にする前に [A4] を見ること)。
7. **`.cu` の文字列リテラルは ASCII 必須** (nvcc が CP932 解釈)。**日本語コメントは可・リテラルは不可。**
8. **採取時は必ず `2>&1`** — このリポジトリのログは stream が 2 系統に割れている前例がある
   (`docs/testing.md:26-30`)。

---

## 10. 規律

- ブランチ `feat/g14k-attention-fa2`。**`git add -A` 禁止** — 対象ファイルのみパスを明示して stage。
- **commit も push も開発機のみ。研究機ではコミットしない** (G-10k で確立した運用)。
  研究機の作業ツリーは各段で **`git status --porcelain` 0 行 (clean)** を保つ
  — これが「実走した exe は commit 済みツリーのものだ」という主張の根拠になる。
- **`meson test` の前に SAC OFF をユーザーへ依頼する** (黙って走らせない)。
- ★**生ログは必ず repo 配下 `docs\logs\g14k-*\` へ退避する** (README 付き)。
  **temp に置いたままにしない** — temp はセッション ID 付きで消え、消えた時点で全緑が
  検証不能な自己申告に降格する (G-8k S5c→S5d の実例)。
- **段の完了ごとに Fable モデルのレビュアーを立てて検査させる。実装担当と同一主体に検査させない。**
- コメント・記録はすべて日本語。C++ は Allman ブレース・`switch` の `case` は `switch` と同じタブ位置。

### ★G-10k との干渉回避 (非交渉)

1. **`--fast` の e2e 秒を動かす Pkg は、同時に 1 つしか実走しない。**
   G-10k と G-14k は**同じ計器 (`test_diffusion_batch2` DB2_BENCH) を叩き、同じ秒を動かす**ため、
   並走させると寄与が分離できない。
2. **G-10k は T2c で保留中 (再開点 T3)。G-14k が GPU を占有している間は G-10k を再開しない。**
3. **G-14k が attention を書き換えたら、G-10k のアンカー (T2c の resnet バケット 0.854 s) と
   nsys 内訳は別ツリーの値になる。** `docs/g10k-plan.md` §13-5 の 4 番がこの点を既に警告している
   (「resnet バケットは attention 改修の影響を受けない**想定**だが、受けないと確かめた実測はまだ無い」)。
   **G-10k を再開する側が T7 のドリフト対照で確認する** — G-14k 側で「影響しない」と断定しない。

---

## 11. 本書が根拠にした一次証拠 (2026-09-05 開発機セッションで実際に開いたもの)

**docs の記述は引用元としてのみ使い、根拠にはしていない** (docs を引くときは出典として明記した)。

### 11-1. ソース / 物理状態

| 主張 | 一次証拠 |
|---|---|
| 開発機で `.cu` をコンパイルできない / 成果物も動かない | `which nvcc` = not found / `CUDA_PATH` 空 / CUDA Toolkit ディレクトリ不在 / `nvidia-smi -L` = **GTX 1080 Ti** / `meson.build:130` の `-arch=sm_120` |
| K/V タイル 16 行・タイル数 = ceil(Sk/16) | `src/kernels/attention_fast.cu:42`, `:47`, `:132` |
| ループ 1 反復に `__syncthreads` ×2 | 同 `:184`, `:270` (`__syncwarp` は `:201`, `:244`, `:260`, `:267`) |
| `acc` は shared の FP32 | 同 `:85-93` (レイアウト), `:102`, `:162` |
| rescale が 16 レーンのみ・Dh を逐次 | 同 `:204`, `:227-230`, `:234-240` |
| P·V が store→shared→加算の往復 | 同 `:250-267` (とくに `:259`, `:261-266`) |
| nwarps は `{4,2,1}` から shared 制約で選択 | 同 `:308-331` |
| fast のフォールバック条件は 2 つだけ | 同 `:300-306`, `:334-338` (上限定数は `:310`) |
| `__expf` を使っている | 同 `:224`, `:237` |
| default 経路は 1 block = 1 warp | `src/kernels/attention.cu:636` (コメント), `:653` (`<<<grid, 32, shmem>>>`) |
| VAE mid は `launch_attention` を直接呼ぶ | `src/kernels/vae_decode.cu:683` (`1, 1, S, S, C`) |
| UNet の attention 呼出は 4 箇所・Dh=64 固定 | `src/infer/unet.cu:684-685`, `:722`, `:726`, `:766`, `:770` |
| UNet の解像度レベルと head 数 | `src/infer/unet.cu:937-938` (`ctx_tokens=77` / `ctx_dim=2048`), `:1094`(down1 64²/L=2), `:1119`(down2 32²/L=10), `:1144`(mid 32²/L=10), `:1167`(up0 32²/L=10), `:1204`(up1 64²/L=2)。transformer2d の呼出は `:1103` `:1110` `:1128` `:1134` `:1152` `:1188` `:1224` の 7 箇所 |
| PTX / TMA / FP8 / CUTLASS / cuDNN / cuBLASLt / CUDA Graph / 非既定 stream が 0 件 | `src/` に対して `\basm\b|__asm__|mma\.sync|ldmatrix` / `cuTensorMap|TensorMap` / `__nv_fp8|e4m3|e5m2` / `cutlass|cute|cudnn` / `cublasLt` / `cudaGraph` / `cudaStream_t|cudaStreamCreate` を grep (本セッション) |
| `#include <mma.h>` は 3 TU | `attention.cu:27` / `attention_fast.cu:29` / `gemm.cu:39` |
| FP8 は配管だけ存在する | `src/server/fast_config.hpp:22`, `:76-79` (FP8 env), `:93-97` (fp8→fast 含意) / `src/main.cpp:203-213` / `src/infer/diffusion.cu:297` |
| **CPU 参照の関数名は `correctness_case`** (`check_vs_cpu` は存在しない) | `src/tests/test_attention_fast.cu:212-251` / CPU 参照本体は `:66-118` |
| CPU 参照の tol は K=Dh+Sk 依存 | 同 `:56-63`, `:228-230` |
| CPU 参照ケースは最大 Sq=256 | 同 `:384-391` |
| SSIM ゲートは 0.9999・4096 形状は H=1 | 同 `:281`, `:394-399` (理由コメントは `:395`) |
| `bench_pair` の warmup / iters / 同期 / 2 択 | 同 `:290`, `:318-334` (`:330` が `cudaEventSynchronize`), `:336-350` |
| **`bench_pair` の呼出は 3 本すべて B=1・self は H=1** | 同 `:408-410` |
| `test_unet_fast` は B=1 | `src/tests/test_unet_fast.cu:258`, `:280` + `src/infer/unet.cu:1462` (`launch_unet_impl(w, 1, ...)`) |
| `test_unet_fast` の default 側ゲートは tol | 同 `:277-278` (`MAE > 5e-3` / `SSIM < 0.99`) |
| fast vs default のゲートは SSIM ≥ 0.9999 | 同 `:300-301` |
| 二連走は determinism であって無改変証明ではない | `src/tests/test_diffusion_batch2.cu:46-49`, `:438-440` |
| parity 節の構成 4 本と g1_again | 同 `:251` (steps=4), `:252` (seed=1234), `:256` (guids 3 本), `:262-268` (`gen`), `:317` (guidance ループ), `:323-327` (g1_again 再走), `:343`, `:348`, `:353`, `:360` (`run_config` 4 本) |
| DB2_BENCH は 1 プロセス・warmup1 + min-of-iters・`steady_clock` 窓 | 同 `:555-563`, `:567-598` (warmup `:584` / min ループ `:586-596` / 窓 `:589`,`:592`) |
| `generate_txt2img` は `use_batch2` で B=2 | `src/infer/diffusion.cu:755`, `:879`, `:887-888` |
| T2b dump が B=2 経路で attention 秒を出す | 同 `:1006-1009` (`resnet` / `transformer` / `-> attention only`) |
| `ScopedSyncTimer` は前後で `cudaDeviceSynchronize` | `src/infer/unet.cu:67-93` / 使用箇所は `:717` (self) と `:761` (cross) |
| キルスイッチの getenv キャッシュ型 2 例 | `src/kernels/gemm.cu:377-386` / `src/kernels/device_arena.cu:215-232` |
| `DOLLAMA_*` の既存 env 名 (32 個) に `DOLLAMA_CONV_BATCH` は無い | `grep -rn "DOLLAMA_[A-Z0-9_]*" src/ -o` の一意化 (本セッション) |
| `attention_fast.cu` の meson 登録 / test 登録 | `src/meson.build:66` / `:641-648` (`test('attention_fast', ..., timeout : 300)` は `:648`) |
| `prof_arena_e2e` は meson test 未登録 | `src/meson.build:889-890` (executable のみ) |
| 登録テスト数 = 53 | `src/meson.build` の `test(` 実数カウント |
| G-3k は `attention.cu` を触っていない | `git show --stat f3b86fc` (`attention.cuh` +19 のみ・`attention.cu` は不在) / `git log --oneline -- src/kernels/attention.cu` の最新は `502c936` |
| `--use_fast_math` が既定 ON | `meson.build:131-134` / `meson_options.txt` の `option('fast_math', ..., value: true)` |

### 11-2. 生ログ (repo 内・`docs/logs/g10k-baseline/`)

★**採取条件** (これと 1 つでも違えば比較は成立しない):

- **nsys 2 本** (`t2c_nsys*` / `t2c_nsys2*`): 被計測は `prof_arena_e2e.exe`
  (sha256 `9D420AB1…1140`)、env は `PROF_IMAGES=2 PROF_STEPS=20 PROF_G=7.5 PROF_FAST=1`、
  `DOLLAMA_PROFILE` / `DOLLAMA_POOL` / `DOLLAMA_ARENA_RELEASE` / `PROF_SAMPLE_MS` は**すべて未設定**
  (= **profile 計装 OFF**)。HEAD **`f58ba2c`**。出典 `t2c_nsys_meta.log:5-18` / `t2c_nsys2_meta.log:3-7`。
- **DB2_BENCH 1 本** (`t2c_db2bench.log`): env `DB2_BENCH=1 DB2_BENCH_STEPS=20 DB2_BENCH_ITERS=1
  DOLLAMA_PROFILE=1`・`DB2_BENCH_G` 未設定 (harness 既定 7.5)・seed 1234・**1 プロセスで 3 構成**。
  同 HEAD。出典 `t2c_run_meta.log` / `t2c_env.log`。
- **e2e 2 本** (`e2e_run1.log` / `e2e_run2.log`): `prof_arena_e2e`・
  `PROF_IMAGES=2 PROF_STEPS=20 PROF_G=7.5 PROF_FAST=1 PROF_SAMPLE_MS=5`・`DOLLAMA_PROFILE` 未設定・
  HEAD **`e17374c`**。出典 `e2e_run1.log:6-10` および同ディレクトリ `README.md` の HEAD 表。

| 数値 | ログ:行 |
|---|---|
| 生成#2 wall 10.6615s / kernels 67032 / Σkernel 10.4705s / gap 0.1910s = 1.79% / busy 98.21% | `t2c_nsys_gen_split.log:11-15` |
| 同 走行#2: 10.6608 / 67032 / 10.4707 / 0.1901 = 1.78% / 98.22% | `t2c_nsys2_gen_split.log:11-15` |
| **生成#1 の gap は 3.54% (走行#1) / 3.43% (走行#2)** | `t2c_nsys_gen_split.log:39` / `t2c_nsys2_gen_split.log:39` |
| `attention_flash_wmma_fast_fp16` 6.3430s 60.58% **n=2800** | `t2c_nsys_gen_split.log:19` |
| 同 走行#2 6.3438s 60.59% n=2800 | `t2c_nsys2_gen_split.log:19` |
| `attention_flash_wmma_fp16` (VAE) 0.5147s 4.92% **n=1** | `t2c_nsys_gen_split.log:21` |
| 同 走行#2 0.5143s 4.91% n=1 | `t2c_nsys2_gen_split.log:21` |
| UNet 20step: wall 9.6599s / Σkern 9.4756s / **65889 kernels = 3294.4/step** / attention 66.95% | `t2c_nsys2_unet_vae_split.log:2-6` |
| VAE decode: wall 1.0009s / Σkern 0.9951s / 1143 kernels / attention 51.69% | 同 `:19-23` |
| **grid 別**: `(64,20) block=128 n=400 total 3.5174s avg 8.793ms` / `(16,40) block=128 n=2400 total 2.8264s avg 1.178ms` / VAE `(1024,1) block=32 n=1 total 0.5143s avg 514.333ms` | 同 `:36-39` |
| 出荷 `--fast` 本走: UNet 9.729s / resnet 0.854s / transformer 8.693s / **attention only 6.387s** | `t2c_db2bench.log:496`, `:504`, `:505`, `:506` |
| `default` 本走: UNet 12.996s / resnet 1.057s / **attention only 9.218s** | 同 `:396`, `:404`, `:406` |
| DB2_BENCH e2e min ms: default 13999.6 / attn+batch2 10879.7 / **fast+epilogue 10735.9** | 同 `:512` |
| `prof_arena_e2e` e2e (profile OFF・HEAD `e17374c`): 10.9628 / 10.6465 s | `e2e_run1.log:18`, `:22` |
| 同 run2: 10.7505 / 10.6561 s | `e2e_run2.log:18`, `:22` |

**docs からの引用 (出典として明記した二次記述)**:
`docs/fast-mode-plan.md` の **#4 節の進捗表** / **#5 節「ゲート設計の欠陥と是正」**
(★行番号は R-1 監査中で動きうるため本書では節名で指す) /
`docs/g10k-plan.md` §0 (SSH 運用)・§1 (理由節)・§2 (計器規則)・§9 (危険箇所の書式)・§10 (規律)・
§13-5 (再開時の注意) / `docs/measurements-log.md:75` (走行 = プロセス) /
`docs/testing.md:26-30` (`2>&1`) /
CLAUDE.md 計測表 G-4k S3 行 (g=7.5 の 0.996533 / x1.33439 / ドリフト ~18%) /
`f3b86fc` の commit 本文 (G-3k の SSIM=1.0 主張 = **二次証拠**)。

---

## 12. 残債 (本 Pkg のスコープ外・または未決)

1. ★**VAE mid attention (単発 514ms) の扱いが未決** (§3-2)。正本 `docs/fast-mode-plan.md` #4 節は
   G-14k のスコープと書くが、本 Pkg の PL 決裁 (段構成・ゲート・形状) には現れない。
   **S1 着手前に PL へ上げること。** 勝手に入れない・勝手に落とさない。
2. **`src/tests/prof_unet_fast_warm.cu:155-156` に退役済みゲートが出力文字列として残っている**
   (`(gate: <=0.95s / stretch <=0.85s / baseline 1.225s)`)。`docs/g10k-plan.md` §12-4 が
   「**次に fast-mode を触る段の先頭で処理**」を宿題の優先度としている。
   **G-14k がその「次の段」に当たるかは PL 決裁。**
   ★**走行ログにこの行が出ても分母として読まないこと。**
3. **`docs/hw-accel-plan.md:122` / `:136`** の攻め筋 (「GPU 側で次に残っているのは G-9k / G-10k →
   G-5k → G-6k」) に **G-14k が入っていない**。追随の要否は PL 決裁 (本タスクのファイル指定に無い)。
4. **`.claude/agents/cuda-kernel-dev.md:101`** が `--fast` を
   「ループ GPU 常駐 + CUDA Graphs / CFG batch=2 / FlashAttn 級 attention / epilogue 融合」と書いている。
   ★**「ループ GPU 常駐 + CUDA Graphs」は未実装** (`cudaGraph` は `src/` に 0 件 = G-1k は条件付きへ降格のまま)。
   **本タスクの決裁は `perf-profiler.md` に対してのみ出ているので触っていない。報告のみ。**
5. **`bench_pair` の B=2 / 実 head 数の呼出追加**は本 Pkg の S1/S2 スコープ (残債ではない) だが、
   ★**追加した時点で「過去の B=1 / H=1 の bench 値とは別系列になる」**ことを S4 の記録で明示すること
   (G-10k が B=1 の 1.212s を退役させたのと同型の分断)。
6. **G-10k T3 以降の再開**は G-14k が GPU を離すまで保留 (§10 の干渉回避)。

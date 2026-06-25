# HW 環境抽象化 / 実行モード設計案

dollama を **異なるハードウェア環境**（Intel 研究機以外）でも動かせるよう、
搭載 HW を宣言し、各処理段のデバイス割り当てを切り替える**モード/フラグ体系**の設計案。

ステータス: **設計案 (要レビュー承認)**。実装は未着手。着手時はプランモード承認 → `project-leader` 経由。

## 背景: なぜ要るか

現状、各処理段のデバイスは **Intel 研究機 (Core Ultra 9 285 + Intel AI Boost NPU + Intel Xe iGPU + RTX5080)** を前提に
ほぼ固定で割り当てられている (probe 実測で確定):

| 処理段 | 既定デバイス | 根拠 |
|---|---|---|
| LM (BitNet タグ生成) | RTX5080 (CUDA) 第一・CPU 代替 | CLAUDE.md §LLM の将来像 |
| CLIP text encoder | NPU (7.85ms) | probe9 |
| WD14 tagger | CPU (101ms) | probe8 |
| VAE encode (img2img) | iGPU GPU.0 (79ms) | probe5 |
| VAE decode | RTX5080 | probe4 |
| matting ISNet | iGPU GPU.0 (99.96ms) | M-5 |
| 品質スコアラ | NPU (4.62ms) | 4-B probe |
| UNet + 拡散 | RTX5080 (CUDA) | probe10 |

この固定割り当ては、別環境では破綻する (環境分析は §「対応環境マトリクス」)。
特に **Ryzen 無印 + NVIDIA dGPU 構成では NPU/iGPU が物理的に存在しない**ため、
NPU/iGPU 前提の段は別デバイスへ退避させる必要がある。既存の散在 env
(`DOLLAMA_MATTING_DEVICE` 既定 `GPU.0`・`DOLLAMA_*_WEIGHTS`) と CLI (`--no-matting`) を、
**統一された「デバイス計画 (device plan)」**の下にまとめ直すのが本設計。

## 対応環境マトリクス

| 環境 | CUDA コア | NPU/iGPU 段 | SDXL | 評価 |
|---|---|---|---|---|
| **Intel 研究機** (現行・RTX5080 16GB) | ✅ | ✅ Intel (OpenVINO) | ✅ 1024² | 基準。4 HW 協調が成立 |
| **Ryzen 無印 + RTX3060+ (12GB)** | ✅ 再コンパイル(sm_86)のみ | ❌ NPU/iGPU 無し → CPU/CUDA へ退避 | ✅ 1024² (際どい) | 動くが **2 ドメインに縮退** (CPU+dGPU)・研究の芯 (多 HW 協調) は半減 |
| **Ryzen 無印 + RTX (6GB)** | ✅ | ❌ 同上 | ⚠️ **1024² 不可** → 512²/offload/量子化必須 | 低解像度ドラフト ([roadmap バックログ](roadmap.md)) が必須化 |
| **Intel Arc dGPU** (B580 12GB / A770 16GB) | ❌ CUDA 不可 → **SYCL/oneAPI 移植 or OV-SDXL 代替** | ✅ **Arc も OV `GPU` で見える** → 既存 OV 段そのまま載る | OV-SDXL なら可 | NVIDIA と AMD の**中間**。OV 半分は無傷・CUDA 半分のみ移植/代替。**コスパ良** (≈RTX4070・安価) |
| **AMD Ryzen AI Max** (Strix Halo APU・iGPU+NPU+〜128GB) | ❌ CUDA 不可 → ROCm/SYCL 移植 | △ HW はある (RDNA3.5 iGPU + XDNA2 NPU) が **AMD スタック** (OV 不可) → 再構築 | iGPU/ROCm 次第 | **思想的に最も噛み合う単一 APU** (多 HW + 大容量共有メモリ) だが**配管が全部 AMD 製**で要移植 |
| **AMD CPU + Radeon dGPU** | ❌ **CUDA 不可** → HIP/ROCm 全面移植 | ❌ OpenVINO 不可 → ROCm/DirectML/VitisAI で再構築 | 要移植 | 思想は流用可・**実装は別プロジェクト規模の移植**・ROCm×Windows 未成熟が壁 |
| **CPU-only** (dGPU 無し) | ❌ | CPU のみ | 実用外 (拡散が極端に遅い) | LM 研究のみ可・画像生成は非現実的 |

要点:
- **NVIDIA dGPU を保持する限り CUDA 研究コアは無傷** (アーキフラグ変更のみ)。Radeon にすると研究コア全面移植。
- **Intel Arc は中間**: 既存 OpenVINO 推論段はそのまま載るが、自作 CUDA 拡散コアは SYCL 移植 or OV-SDXL 代替が要る。コスパが良く生成 AI 用途で有力。
- **NPU/iGPU は Intel 提供**ゆえ、非 Intel 環境では消える/別スタック。`--npu`/`--igpu` で「無い」を宣言できる必要がある。
- **6GB VRAM は SDXL 1024² の壁** (ピーク 10.49GB)。VRAM 宣言で自動ダイエットしたい。

## モード/フラグ体系の設計

3 層に分ける。**上から宣言 → 下ほど細粒度の上書き**。

### 層 1: HW ケイパビリティ宣言 (何が載っているか)

搭載 HW を申告する。未指定は現行 Intel 研究機の値を既定とする (後方互換)。

| フラグ | 値 | 意味 / 影響 |
|---|---|---|
| `--cpu` | `intel` \| `amd` \| `arm64` \| `generic` | CPU マイクロ最適化経路の選択。x86 (intel/amd) は AVX2/AVX-512/VNNI、**`arm64` (Windows on Arm / Snapdragon X) は NEON/SVE2** と SIMD が別系統 → CPU LM 最適化のコード経路が分岐する。当面は情報のみ |
| `--npu` | `intel-aiboost` \| `ryzen-xdna` \| `none` | NPU の種別。`intel-aiboost`=OpenVINO NPU プラグイン。`ryzen-xdna`=AMD XDNA2 (Ryzen AI / AI Max、VitisAI スタック・当面予約)。`none` で NPU 段を退避 |
| `--igpu` | `intel-xe` \| `amd-rdna` \| `none` | iGPU の種別。`intel-xe`=OpenVINO `GPU.0`。`amd-rdna`=AMD APU iGPU (**Ryzen AI Max / Strix Halo は RDNA3.5 最大 40CU の本物の演算 iGPU**・ROCm/DirectML スタック・当面予約)。`none` (無印 Ryzen の飾り iGPU 含む) で iGPU 段を退避 |
| `--dgpu` | `nvidia` \| `intel-arc` \| `amd-rocm` \| `none` | dGPU の種別。`nvidia`=CUDA 経路 (研究コアそのまま)。`intel-arc`=OpenVINO/SYCL (後述・OV 半分は流用可)。`amd-rocm`=HIP/ROCm (要全面移植・当面予約)。`none` で拡散・GPU LM 不可→CPU 退避 or 無効 |
| `--vram` | `6g` \| `8g` \| `12g` \| `16g` \| `24g` \| `32g` … | dGPU VRAM。SDXL 解像度/offload の自動判定に使う。**既定は CUDA から自動検出** (`cudaMemGetInfo`/`cudaGetDeviceProperties`)・フラグは上書き/強制用。24g (RTX3090/4090) / 32g (RTX5090) では SDXL 1024² が余裕 + LoRA 同時常駐や複数枚バッチの余地 |

> **Intel Arc dGPU は特別扱い**: 本プロジェクトは既に第二スタックとして OpenVINO を使っており
> (CLIP/WD14/matting/品質スコアラ)、**Arc は OpenVINO の `GPU` デバイスとして素直に見える**
> (iGPU `GPU.0` と同系統)。つまり**既存 OV 推論段はそのまま Arc dGPU に載る**。ただし拡散の
> 研究コア (自作 CUDA カーネル) は CUDA 専用ゆえ Arc では動かず、**SYCL/oneAPI への移植か、
> OpenVINO の SDXL 推論にフォールバック**が要る。位置づけは NVIDIA (CUDA コア無傷) と
> AMD (全面移植) の**中間** — OV 半分は再利用でき、CUDA 半分だけ移植/代替。コスパは良い
> (Arc B580 12GB ≈ RTX4070 性能で安価・A770 16GB) ので生成 AI 用途では有力。
>
> **AMD Ryzen AI Max (Strix Halo) は「強い AMD APU」**: 飾り iGPU の無印 Ryzen と違い、
> **RDNA3.5 最大 40CU の演算 iGPU (Radeon 8060S) + XDNA2 NPU (~50 TOPS) + 最大 128GB
> ユニファイドメモリ**を持つ = AMD 版「CPU+NPU+iGPU が 1 パッケージ」。**dollama の芯
> (多 HW 協調・小型専門モデルを大容量共有メモリに逃がして容量を稼ぐ) と思想的に最も噛み合う
> 単一 APU** で、潜在的には魅力大。**ただしソフトは全部 AMD スタック** (iGPU=ROCm/DirectML・
> NPU=VitisAI) ゆえ OpenVINO コードは載らず、`--igpu=amd-rdna`/`--npu=ryzen-xdna` は**当面予約**
> (実装は ROCm/VitisAI 対応の大タスク)。HW はあるが配管が別、という状態。
>
> **AMD Radeon dGPU も当面 `amd-rocm` 予約** (CUDA も OpenVINO も叩けず HIP/ROCm 全面移植)。
> 将来 ROCm/DirectML/VitisAI 対応で `--dgpu=amd-rocm`・`--igpu=amd-rdna`・`--npu=ryzen-xdna` を本格対応する。
>
> **Windows on Arm (Snapdragon X 系) も別世界**: CPU は arm64 (NEON/SVE2・AVX 無し) で CPU LM の
> SIMD 経路が別、iGPU=Adreno・NPU=Hexagon は **QNN/DirectML スタック** (OpenVINO 不可)、
> **NVIDIA dGPU は Arm 版 Windows でドライバ事情が限定的**で CUDA コアもほぼ載らない。
> 純ホスト部 (LM/tokenizer/safetensors/HTTP) は arm64 でビルドすれば動くが、GPU/NPU 配管は要再構築。
> `--cpu=arm64` で宣言だけ受ける (本格対応は当面予約)。

### 層 2: 環境プロファイル (層 1 の束) — 任意・利便用

よく使う構成をワンショットで指定。内部で層 1 へ展開。

| `--env` | 展開 |
|---|---|
| `intel-research` (既定) | `--cpu=intel --npu=intel-aiboost --igpu=intel-xe --dgpu=nvidia --vram=16g` |
| `ryzen-nvidia` | `--cpu=amd --npu=none --igpu=none --dgpu=nvidia`（`--vram` は別途指定推奨） |
| `cpu-only` | `--dgpu=none --npu=none --igpu=none` |

### 層 3: 処理段ごとのデバイス上書き (最細粒度)

プロファイル/宣言から導いた既定を、段単位で強制上書きする。既存 `DOLLAMA_MATTING_DEVICE` を一般化。

`--lm-device` / `--clip-device` / `--wd14-device` / `--vae-encode-device` /
`--vae-decode-device` / `--matting-device` / `--scorer-device`
（値: `cuda` \| `cpu` \| `npu` \| `igpu`）。

## 解決ロジック (起動時)

```
1. --env があれば層 1 へ展開 (無ければ intel-research 既定)
2. 個別 HW フラグ (--npu=none 等) で上書き
3. 段ごとの --*-device で上書き
4. 妥当性検査 + フォールバック:
     ある段の指定先デバイスが「無い」と宣言されていれば、定義済みチェーンで退避:
       NPU 段   : NPU → CUDA → CPU
       iGPU 段  : iGPU → CUDA → CPU
       CUDA 段  : CUDA → CPU (拡散は CPU 退避すると実用外 → 警告)
     退避は 1 行ログを出す (既存「matting null→不透明 PNG」「DI 3 段フォールバック」と同じ作法)
5. VRAM ゲート:
     --vram が SDXL 要求 (1024²=10.49GB) に足りなければ
       解像度を自動降格 (768²/512²) or UNet ブロック offload を有効化し、警告
```

### 退避先の既定 (NPU/iGPU が無い環境)

| 段 | Intel 機既定 | NPU/iGPU 無し時の退避 |
|---|---|---|
| CLIP | NPU | **CUDA** (小・~0.25GB) or CPU |
| WD14 | CPU | CPU のまま |
| VAE encode | iGPU | CUDA or CPU |
| matting | iGPU | CUDA or CPU (CPU は 261ms 実測) |
| 品質スコアラ | NPU | CUDA or CPU |

→ Ryzen 無印 + NVIDIA では、実質 **NPU/iGPU 段をすべて CUDA か CPU に寄せた「2 ドメイン構成」**に解決される。

## 実装方針 (着手時)

- **デバイス計画オブジェクトを 1 つ作り、起動時に解決して全ジェネレータへ注入**する。
  既存の `IImageGenerator` / `IDiffusionRunner` / `IMatter` の DI 配線に乗せる
  (`make_matter` が `DOLLAMA_MATTING_DEVICE` を読む箇所を、計画参照に置換)。
- 散在 env (`DOLLAMA_MATTING_DEVICE` 等) は当面**併存・後方互換**とし、フラグが優先。
- **段階導入**: ① 層 1 (HW 宣言) + フォールバック解決 + VRAM ゲート → ② 層 3 (段上書き) →
  ③ 層 2 (プロファイル) の順。①だけで Ryzen+NVIDIA は動くようになる。
- OpenVINO 段は `--npu=none`/`--igpu=none` 時に**そもそも OV を初期化しない**こと
  (OV 不在ビルド・非 Intel 環境でのリンク/ロード失敗を避ける。既存の OV ガードと整合)。

## スコープ外 / 非目標

- **AMD Radeon (ROCm) 実装はこの設計では扱わない** (フラグ予約のみ)。CUDA→HIP 移植は別途巨大タスク。
- 自動 HW 検出 (申告なしで載っている HW を見つける) は将来。初版は**明示宣言**で十分。
- CPU マイクロ最適化 (`--cpu=amd` 分岐の中身) は別タスク
  ([roadmap バックログ「CPU 側 LM 推論の速度最適化」](roadmap.md))。

## 関連

- [docs/roadmap.md](roadmap.md) バックログ「対応環境マトリクス / HW 環境抽象化」「プレビュー用低解像度ドラフトモード」「CPU 側 LM 推論の速度最適化」。
- CLAUDE.md §確定済みアーキテクチャ決定 (OpenVINO デバイス命名 `GPU.0`=Intel Xe / `GPU.1`=RTX5080)。

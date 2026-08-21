# CPU/NPU/iGPU 側高速化 計画書 (hw-accel-plan)

> **位置づけ**: GPU 側の高速化台帳は `docs/fast-mode-plan.md` (G-0〜G-13k)。本 doc は**非 GPU 側
> (CPU/NPU/iGPU) の高速化検討**を扱う。ただし結論を先に言うと、**現状パイプラインは GPU バウンドが
> 実測で確定しており、非 GPU 側をいくら速くしても e2e スループットは動かない** (アムダール診断 §1)。
> 本 doc の価値は「やる候補リスト」ではなく「**やらない根拠の明文化 + 意味を持つ局面の条件付け**」にある。
> 本 doc は調査・計画のみ (コード実装なし)。着手時は CLAUDE.md ルール (プラン承認 → PL 振り分け → test 実装) に従う。

## 1. アムダール診断 (最重要・結論)

### 1.1 実測根拠

| 実測 | 値 | 出典 |
|---|---|---|
| マルチフレーム並列スループット | **0.258 fps = 理論 GPU 上限 0.263 fps の 98% = GPU バウンド** | test_multi_frame_pipeline (実寸スタブ LM400/CLIP8/SDXL3800/WD14 100ms, N=8) |
| GPU 飢餓検出 (B-SDXL 入力待ち) | queue 待ち中央値 **0.0002ms ≈ 0** = 先行生成が GPU を飢えさせていない | 同上 |
| LM 段の隠蔽 | stage_a 404ms は SDXL の裏に**完全隠蔽** (per_frame 3879.8ms ≪ 直列合計 4308ms) | 同上 |
| look-ahead 深さ | QueueDepth {2,4,8} で fps 0.2581/0.2580/0.2574 = **2 で飽和**・積む価値なし | 同上 |
| 自作拡散 e2e (正典 CFG・warm) | default **20.93s** / `--fast` **15.88s** (20step/1024²/VAE 込) | test_diffusion_batch2 DB2_BENCH (fast-mode-plan) |
| 非 GPU 段の実測合計 | CLIP NPU 7.85ms + WD14 CPU 105ms + matting iGPU 99.96ms + (img2img 時 VAE encode iGPU 79ms) ≈ **~0.2-0.3s** | probe9/8/M-5/5 |

### 1.2 診断

- **スループット局面 (複数フレーム連続生成)**: SDXL 段 (現 15.9〜20.9s、diffusers 級でも 3.8s) が
  クリティカルパスの実質 100%。非 GPU 段は look-ahead 2 の先行生成で**既に完全隠蔽済み** (queue 待ち ≈0 実測)。
  → **非 GPU 側を何倍速くしても e2e fps は 1 も動かない**。アムダールの法則で並列化可能部分 (=非 GPU 段)
  の比率がゼロに漬されている状態。
- **レイテンシ局面 (単発生成の初動)**: 非 GPU 段が直列に乗るのはプロンプト前段 (LM + CLIP) と
  後段 (matting + PNG 化) のみ。実測ベースの直列内訳:

  | 段 | HW | 実測 | e2e 20.93s 比 | 備考 |
  |---|---|---|---|---|
  | LM タグ生成 | **未結線** | **0s (段が無い)** | 0% | 現状は prompt 直入力。参考: Qwen2-1.5B を CPU に載せると ~2s = 非 GPU 側で唯一の非誤差帯になるため**載せない**判断 (下記) |
  | └ 結線予定: 自作タグ生成 LM | GPU | seq8 **2.43ms** (実装済・test_bitnet_gpu) | ~0% | 結線しても誤差帯に収まる |
  | CLIP text encode (L+bigG) | NPU | **7.85ms** | 0.04% | 誤差帯 |
  | UNet×20 + VAE decode | **RTX5080** | **15.88-20.93s** | **~76-100%** | **律速の全部** |
  | WD14 タグ抽出 (FB 用) | CPU | 105.3ms | — | 生成後・次フレーム生成と並列可 = 非クリティカル |
  | matting (透過 PNG) | iGPU | 99.96ms | 0.5% | 出力前直列だが誤差帯 |
  | scorer 採点 | NPU | 8.32ms | — | ログのみ・非クリティカル |
  | VAE encode (img2img のみ) | iGPU | 79ms | — | CPU LLM と並列 = 待ち時間ゼロ (確定構成) |

- **結論**: **e2e を動かすのは GPU 側 (fast-mode-plan の G 系) だけ**。非 GPU 側でレイテンシに意味を持つ
  唯一の項目は LM 段の**置き場**であり (Qwen2 を CPU に載せれば ~2s / 自作 LM を GPU に結線すれば 2.43ms)、これは CPU 高速化ではなく
  Phase 4 本線の完遂 (正典 merged 重みは焼成済・パイプライン結線が残)。それ以外の非 GPU 段は
  全部足しても e2e の ~1-3%。

### 1.3 非 GPU 高速化に意味がある局面 (3 つ・条件付き)

| # | 局面 | 現状判定 |
|---|---|---|
| ① | **パイプライン並列の先行生成** (LM/CLIP/WD14/VAE encode/matting を拡散の裏へ) | **既に達成済**。look-ahead 2 で GPU 飢餓ゼロ実測。追加高速化の効果ゼロ。再評価条件 = 「SDXL がライブラリ fallback 等で桁違いに速くなった世界」(measurements-log の Tier2 留保と同一条件) |
| ② | **単発レイテンシ** | LM 差し替え (Phase 4 結線) のみ有意。他は誤差帯 |
| ③ | **GPU を空けて別ワークへ** | **設計済**。CLIP=NPU / WD14=CPU / matting=iGPU / scorer=NPU は「拡散中に遊休 HW を使う」配置そのもの。これ以上 GPU から剥がせる大物は無い (UNet/VAE decode は iGPU 8 倍遅・probe4 で棄却済) |

**数値で言い切る**: 仮に GPU 側が全部成功して diffusers 級 3.8s に達しても、非 GPU 直列分
(CLIP 7.85ms + matting 100ms) は e2e の **~2.8%**。閾値として「**SDXL 段が ~1s を切るまで非 GPU 側の
速度施策は起票しない**」を本 doc の既定とする (1s でようやく matting が 10% 帯に入る)。

## 2. HW 別高速化候補の棚卸し (e2e 効果の有無を明記)

### 2.1 CPU

| 候補 | 状態 | e2e 効果 | 判定 |
|---|---|---|---|
| BitNet LM AVX2 (単スレッド ~5x) | ✅ 完了 (Tier 1・golden corr 1.0) | なし (LM は GPU 第一・CPU は代替) | 済 |
| BitNet LM INT8 (重み 74.84% 減・corr 0.9999) | ✅ 完了 (圧縮の研究軸) | なし | 済 |
| Tier 2 独立 forward ワーカー (物理コア pin・~5x 飽和実測済) | 設計確定・**留保** | なし — 発動条件「LM 段ボトルネック化」は**単一 GPU では計測で不成立** (LM は SDXL 裏に完全隠蔽) | **留保継続** (①の再評価条件と同一) |
| WD14 CPU 105ms の更なる高速化 | 未着手 | なし (生成後・並列可・非クリティカル) | **不起票** |
| 自作 LM のパイプライン結線 | 正典重み焼成済・結線残 (現状 LM 段なし = prompt 直入力) | **なし (GPU 結線なら +2.43ms = 誤差帯)**。Qwen2 を CPU に載せる路線を採らないこと自体が ~2s の回避 | Phase 4 本線 (本 doc スコープ外・M4) |

### 2.2 NPU

| 候補 | 状態 | e2e 効果 | 判定 |
|---|---|---|---|
| CLIP-L/bigG text encode (7.85ms・CPU の 2.5x 速) | ✅ 採用済 | 直列だが誤差帯 | 済・これ以上の投資不要 |
| ScorerNet 品質採点 (8.32ms・純 conv・拡散中遊休 NPU で並列) | ✅ 結線済 (ログのみ) | なし (非クリティカル)・③の模範例 | 済 |
| WD14 の NPU 化 | 棄却済 (268ms・Window Attention が NPU 不向き・probe8) | — | **不採用確定** (再訪しない) |
| 解剖メタ整合検査 (骨格/部位検出・§11) | バックログ (roadmap) | なし (拡散中 NPU 遊休の裏・実質ゼロコスト) | 速度施策ではなく**品質施策** — 起票は roadmap 側 |
| LM の NPU 化 | 棄却済 (自己回帰 = 動的形状で NPU 不可・probe6) | — | 不採用確定 |

NPU の固定形状制約 (静的 reshape 必須) は既知。**NPU に載せて得する残件は「非自己回帰・固定形状・純 conv」
の新規モデルのみ** (scorer が既にその枠を占有)。速度目的の新規 NPU 施策は無し。

### 2.3 iGPU (Intel Xe)

| 候補 | 状態 | e2e 効果 | 判定 |
|---|---|---|---|
| VAE encode (79ms・img2img) | ✅ 採用済・LLM と並列で待ちゼロ | なし | 済 |
| matting ISNet (99.96ms・CPU の 2.0x 速・M-5 確定) | ✅ 採用済 | 直列 0.5% = 誤差帯 | 済・高速化不起票 |
| VAE decode / UNet の iGPU オフロード | 棄却済 (decode stub で iGPU 995ms = CPU の 8 倍遅・probe4) | — | 不採用確定 (大規模 Conv は iGPU 不向き) |
| 軽量前処理・後処理の追加オフロード | 未検討 | なし (現状 iGPU は matting+encode で十分空きだが、載せる意味のある直列負荷が存在しない) | **不起票** |

### 2.4 棚卸しの総括

**非 GPU 3 デバイスとも「probe で最速デバイスを選び終え、遊休時間に仕事を割り当て済み」の状態にある。**
速度目的の未着手候補で e2e に効くものは**ゼロ**。残っているのは (a) LM 差し替え結線 (Phase 4 本線)、
(b) 品質系の新規ワーク追加 (解剖検査等 = 速度でなく品質)、(c) GPU 律速が解けた後の条件付き再訪、の 3 種のみ。

## 3. GPU 律速を緩める前提でのオフロード戦略

- **VRAM は非ボトルネック** (実測 7.9GB / 16GB・常駐概算 ~12.2GB < 16GB・CLAUDE.md)。よって
  「VRAM が足りないから CPU/NPU へ逃がす」局面は現構成に存在しない。オフロードの根拠は
  ① 速度・並列 (拡散の裏で走らせ GPU を空ける) ② 遊休 HW の活用 ③ 研究そのもの (プロジェクトの芯)。
- **CPU/NPU offload = 金をかけずに容量を増やす手** という設計原理は将来のモデル追加時に効く:
  CPU 側専門家はシステム RAM (安・大量)、NPU はオンチップメモリ → VRAM を買い足さずモデルを足せる。
  発動するのは「小型専門モデルを複数常駐させる」局面 = **MoE × HW 分散配置** (roadmap バックログ・
  導入順は GPU 本線 A の弱点実測後)。本 doc からは新規起票しない。
- **遠隔 HW ノード** (LAN 越し第 2 マシン・critic ノード) も同系のバックログ (roadmap)。本命は速度でなく
  精度の上乗せ・完全 opt-in。こちらも起票済みのため本 doc は重複起票しない。

## 4. 優先度 — GPU 側台帳 (G 系) との関係

e2e を動かす投資は **fast-mode-plan.md の G 系が唯一の主線**。非 GPU 施策はすべてその後段に条件付ける。

```
[e2e を動かす道 = GPU 側のみ]
  G-4k (epilogue 融合・着手中) ──► G-8k (step ループ内 cudaMalloc 撲滅・未起票中の最有力)
      ──► G-9k (VAE decode) / G-10k (conv2d 真 batch2) ──► G-5k (FP8) ──► G-6k (出荷判定)

[非 GPU 側 (本 doc)]
  条件 A: SDXL 段 < ~1s ──► matting/CLIP の直列分が 10% 帯に入って初めて再訪
  条件 B: SDXL が桁違いに速くなる ──► Tier2 独立 forward / look-ahead 深化の再評価 (measurements-log 留保と同一)
  条件 C: パイプライン並列で新ワークを足す ──► 速度でなく品質施策として roadmap 側で起票 (解剖検査等)
```

- **G-8k (step ループ内 cudaMalloc/cudaFree 撲滅)** が未起票バックログ中の GPU 側最有力
  (CFG 20step で ~1,600 malloc/free ペア・cudaFree は暗黙同期点・G-1k CUDA Graphs の前提条件。
  効果は要 nsys 実測)。非 GPU 側の検討はこれら G 系と**競合させない** — 同じ工数を非 GPU に割く
  期待値は §1 の通りゼロだからである。
- **非 GPU 施策の起票条件 (本 doc の既定)**: 「GPU 律速が解けた後 (条件 A/B)」または
  「パイプライン並列局面で新ワークを載せる (条件 C・ただし速度でなく品質目的)」のいずれかを満たすこと。
  満たさない速度施策の起票は却下を既定とする。

## 5. 次アクション (probe/計測で埋めるべき未知数)

コード実装なし・計測のみ。優先順:

| # | 計測 | 目的 | 状態 |
|---|---|---|---|
| P-1 | **単発レイテンシの全体分解** (prompt→透過 PNG の cold/warm 別内訳: OV encoder 構築・CLIP・LM・拡散・matting・PNG encode) | §1.2 の直列表は各段の単体実測の寄せ集めで、**一気通貫の実測分解は未取得** (warm 20.93s は UNet+VAE のみ・OV encode 等は付帯外)。cold 初回の OV compile / 重み転送が単発レイテンシの主犯かを確定させる | 要 probe (研究機・SAC OFF 実 exe) |
| P-2 | nsys 非侵襲 profile で launch 谷・malloc 同期点の直接観測 | G-1k 再判定 / G-8k 効果見積り (GPU 側だが P-1 と同走可能) | 要 nsys (fast-mode-plan 記載済) |
| P-3 | 実 SDXL + 実 LM でのマルチフレーム queue 待ち再確認 | 0.258 fps / queue 待ち ≈0 は**実寸 sleep スタブ**での確定。実デバイス (OV/CUDA 競合・メモリ帯域共有) で崩れないことの裏取り | 要実走 (優先度低 — スタブ比率は実測値由来で崩れる要因は限定的) |
| P-4 | G-6k 出荷判定の同条件比較 (dollama vs diffusers を cold+OV 込みで揃える) | 現分母 warm 20.93s vs diffusers 3.8s (cold 込み) は条件不一致 (fast-mode-plan 明記) | G-6k 側で実施 |

P-1 が本 doc 唯一の固有アクション。結果が「非 GPU 直列分は合計 ~0.3s (誤差帯)」を裏付ければ §1 の
診断が完結し、本 doc は「条件 A/B/C の監視だけ残してクローズ」できる。逆に cold 側 (OV compile・
重み転送) が秒単位で出た場合のみ、常駐化・キャッシュ等の配管施策 (数値不変・低リスク) を起票する。

## 参照

- `docs/fast-mode-plan.md` — GPU 側高速化台帳 (G-0〜G-13k・e2e ベースライン 20.93s/15.88s)
- `docs/measurements-log.md` — test_multi_frame_pipeline 完全版 (GPU バウンド確定・Tier2 留保条件)
- `docs/roadmap.md` — MoE HW 分散 / 遠隔ノード / 解剖メタ検査 / CPU LM Tier2 のバックログ原籍
- CLAUDE.md — HW 役割分担・デバイス選定根拠 (probe2〜10)・VRAM 収支

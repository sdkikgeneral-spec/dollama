# dollama ロードマップ 決定・完了アーカイブ

`docs/roadmap.md` から退避した**決定ログ・完了ナラティブ**。roadmap 本体は「今どこで次に何を
やるか」の地図に保ち、ここには「なぜそう決めたか・どう完了したか」の経緯を置く。数値の完全版は
`training-spec.md` / `dataset-spec.md` / `measurements-log.md`、本アーカイブはその要約と決裁記録。

---

## タグ生成 LM 学習強化プログラム (C/B/A/D/F) — 決定ログ

依存連鎖・施策定義・現状サマリは roadmap.md の同名サブセクション参照。以下は各施策の
決着に至った経緯 (時系列)。

### C 完了 (2026-06-23) — 物差し変更が D5 判定の符号を反転させた

diverse-val (テンプレ外自然文・tags-stay-real) + 生成 set-metrics + eval-only ハーネス +
seed sweep を実装 (C-1〜C-4)。**旧 proxy (テンプレ teacher-forcing recall@10) では D5
(soft-label KL) が最下位 (0.667) だったのが、新 proxy (diverse 生成 F1) では最上位に反転** —
C の仮説「テンプレ recall が D5 の実力を隠していた」を実データで裏付け。seed sweep で D5−#1 の
diverse F1/Jaccard delta は **全 4 seed で正・各 seed の paired CI が 0 を除外** (D6 の recall
上振れが符号反転した seed ノイズだったのと対照的) = **小幅だが統計的に頑健**な実効果
(delta +0.009〜+0.012 F1・絶対値は #1 の seed 分散帯以下)。**確定事項**: recall@10 (テンプレ) を
主要数値から退役させ、**diverse 生成 set-F1 を新オフライン主指標**に据える。**未決**: D5 を本線
昇格させるかは新物差しの下で別途判断 (絶対値はなお ~0.18–0.22 と低く edge は小さい → A 実ペア増 /
D 容量増と束ねて再評価が妥当)。本番重みは #1 据え置き・無改変。詳細 training-spec §13。

### B パイロット完了 (2026-06-23) — 入力多様化が新物差しで大幅かつ頑健な改善を出した

C で据えた diverse-val + 生成 set-F1 物差しの上で、施策 B の最初のパイロット (タグ固定 =
tags-stay-real・自然文だけ多様化・**Claude 著述 Replace 500**・総件数 4,500 維持) を実施。
**diverse 生成 macro F1 が #1 を大幅に上回る** (diverse_a 0.1800→0.2675 / diverse_b
0.1921→0.3039・in-dist pairs.val は −0.009 で退行なし・legacy recall ≈同値)。seed 頑健性
sweep (4 seed・6ep paired) で delta (B−#1) は **全 4 seed 正・分散帯の 4–6 倍・各 seed の
paired CI が 0 を除外** = 判定 (a)(b)(c) すべて成立 (D5 は (b) 不成立の小幅・D6 は符号反転
seed ノイズだったのと**桁違いに大きく頑健**)。**著者分布交絡は否定**: 旧 D2 (Qwen2 著述) を
diverse-val 再採点 (B-0) しても同等に改善 (diverse_a 0.2701 / b 0.3134) → 「Claude train が
Claude test に似て上がった」では説明できない = 多様化そのものの効果。**旧 proxy では D2 同様
却下されたはずで、C の物差しなしには可視化されなかった** (依存連鎖 C→B の実証)。**未決**:
本番重みは #1 据え置き・別名 `bitnet_dense_diverse_b` 出力。絶対値はなお diverse F1 ~0.26–0.31
と低帯域 → B 著述件数拡大 (500→数千) / A 実ペア増 / D 容量増と束ねて本線昇格を再評価が妥当。
詳細 training-spec §14 / dataset-spec §15。

### B 件数拡大完了 (2026-06-24) — 入力多様化のスケール則を確認

パイロット (Replace 500) と同方式・同物差し (diverse-val 生成 set-F1)・同 sweep で**著述件数だけを
2,000 に拡大** (Replace で総件数 4,500 維持・著述 2,000 + synthetic 2,500・tags-stay-real)。
**diverse 生成 macro F1 が件数増で単調に拡大** (diverse_a 0.2675→0.3212 / diverse_b
0.3039→0.3670)・in-dist は誤差内据え置き (out-of-template だけ伸びる=汎化方向)。seed sweep
(4 seed) で delta (B-2−#1) は diverse_a **+0.1472±0.0102** / diverse_b **+0.1788±0.0029** =
500版 (+0.096/+0.126) の ~1.4–1.5x に拡大しつつ seed sd は縮小 (効果が強まり頑健性も増加・
全判定軸成立)。**決裁済 (2026-06-24・ユーザー)**: **レシピ既定化を確定** — 今後の訓練 (A 実ペア増 /
D 容量増 / F 品質ループ) は多様化入力 (tags-stay-real) を**既定レシピ**とする。B は A/D と直交ゆえ、
劣るレシピ (#1 系) の上に A/D が積まれる事故を防ぐためレシピ確定を先に行う。**正典重み
`bitnet_dense{,_fp32}.safetensors` と C++ 推論 golden (test_bitnet_infer/gpu) の差し替えは A 実ペアと
束ねる次の出荷リトレインで1回** (golden チャーンを 2〜3 回払うのを回避)。当面 #1 重みは据え置き・
別名 `bitnet_dense_diverse_b2000` は実験出力のまま。**保留が解けた根拠**: ① Pareto 改善 ② seed
sweep 全 3 軸成立 ③ スケール則 (件数増で単調)。詳細 training-spec §14.8 / dataset-spec §15.6。

### B-3 件数拡大 (2026-06-25) — スケール則は ~2,000 件で飽和 (前ノートの「頭打ちなし」を訂正)

同方式・同物差し・同 sweep で**著述件数を 10,000 に拡大** (P=2,500 post × k=4 variant・B-2 の
スーパーセット・Replace で総 train 12,000=著述 10,000+synthetic 2,000・tags-stay-real)。`make` を
`--n-posts`/`--k-per-post` に一般化 (k=1 で B-1/B-2 bitwise 非回帰)。seed sweep 4 seed の
delta(B10k−#1) は全 set/metric で **判定 YES (seed 頑健・本物)** だが、**2,000→10,000 の 5 倍増で
平坦** — diverse_a F1 +0.1472→**+0.1411**・diverse_b F1 +0.1788→**+0.1761** (seed 分散内・むしろ
微減)、b 絶対値も 0.319→0.313 / 0.361→0.359 で頭打ち。**前ノート (B-2) の「単調・頭打ちなし」は
500→2,000 の 2 点外挿の誤りで、3 点目 (10,000) で飽和が判明**。**運用結論**: 入力多様化単体の
伸びしろは ~2,000 で尽きる → 残る低帯域 (diverse_a ~0.31 / diverse_b ~0.36) は **B 件数ではなく
A 実ペア増 / D 容量増**で取る。**B 著述を 2,000 超に積む価値は薄く、`[B-merge-at-A]` の既定多様化
ファイルは `pairs.train.diverse_b2000.jsonl` で足りる** (b10k 不要)。本線昇格決裁 (2026-06-24) は
不変・本番重み #1 据え置き・別名 `bitnet_dense_diverse_b10k`。詳細 training-spec §14.9 / dataset-spec §15.8。

### `[B-merge-at-A]` — A 出荷リトレイン時のチェックリスト (2026-06-24 決裁の遅延条項)

施策 B の正典化は A 実ペア増と束ねる次の出荷リトレインで1回にまとめる (golden チャーンを集約)。
その回で必ず行うこと:

1. train ソースを多様化版に切り替える (`pairs.train.diverse_b2000.jsonl` を `--train-file` で既定指定・
   A の新規実ペアは同じ tags-stay-real 機構で diverse train へ合流)。それまで `train_bitnet.py` の
   `--train-file` default=None (=#1 経路) は**意図的に据え置く** (今書き換えると別名出力分岐に落ち、
   bitwise 非回帰アンカー・golden 据え置きという遅延条項を破るため)。
2. 正典 `data/bitnet/bitnet_dense{,_fp32}.safetensors` を**この1回で**差し替える。
3. C++ 推論 golden (test_bitnet_infer / test_bitnet_gpu・corr 1.0 突合) を**同時に**再生成する。
4. legacy 非回帰アンカー (pairs.val recall ~0.777) の役割を「#1 アンカー」→「新本線アンカー」へ変更。
5. **A (同一性条件付け) も同じこの1回で焼く** (2026-06-26 A クローズ後の追記): A は a12k 4 seed sweep で
   評価完了し、効果は diverse-val F1 でなく **identity retention 0.975 (全 seed 頑健)** と確定。よって
   この出荷リトレインは `--identity` で b2000 多様化 + identity_cond 混合を同時に焼き、retention を持つ
   本線を 1 回で作る (A の実ペアは a12k を使用・a25k は未使用保持)。出荷重み = B(多様化) ∧ A(identity) の
   まとめ焼き 1 本。

**✅ 完了 (2026-07-03)**: 上記①〜⑤を1回のまとめ焼きで実施。勝者 33M で B(b2000) ∧ A(a12k identity) を
merged 混合1本に訓練 (MIXED train=15300 [diverse_b2000 4500 ∪ a12k identity 10800] / val=1700
[synthetic 500 + identity_cond 1200]・6ep FP32・val_loss ep3 底 1.9996・train 203.1s・param
32,976,896)。**ゲート4指標** (seed 20260620・`data/bitnet/_merge_ba/eval_report_merged.json`):
identity retention **0.9807** / diverse_a 生成 macro F1 **0.3332** / diverse_b 生成 macro F1
**0.3804** / in-dist pairs.val 生成 macro F1 **0.4552** — 各単体参照 (a12k retention 0.9748 /
b2000 diverse_a 0.3212 / diverse_b 0.3670) を全軸で上回り合格。② 正典 `bitnet_dense{,_fp32}` /
`bitnet_dense_identity{,_fp32}` を merged と同一バイトへ差し替え (sha256 FP16 5780fe10 / FP32
5043772d・旧は .pre_merge 退避)。③ C++ 推論 golden 再生成・test_bitnet_infer corr 1.0 / greedy 5/5・
meson test 25/25 緑。④ legacy アンカーを新本線 (merged) 基準へ。⑤ A(identity) を同回で焼成。
`--train-file` default は None 据え置き、正典再現は明示コマンド (training-spec §17 再現手順)。
**follow-up (非ブロッキング)**: 研究機 RTX5080 で test_bitnet_gpu の GPU golden corr 1.0 再確認済。
詳細 training-spec §17 / dataset-spec §19。

### ラベル化 (生成画像→WD14 タグ) の C++化は F でブロック解除 (2026-06-24・PL 判断)

研究機での SDXL生成→ラベル化 end-to-end 実走は **Python offline ツール
`scripts/dollma_label_image.py` (d83fd66) を当面の正規経路**として確定。本番のメモリ上結線
(生成器の生ピクセル → WD14 → タグ列 → LM FB) は **唯一の消費者が F** のため、F 着手まで C++化は
保留 (消費者不在の先行配管は死にコード化リスク・PNG デコーダ連鎖も招く)。**F 着手時の C++化スコープ
(最小)**: ① 純関数 `dollma_resize_to_wd14` (raw RGBA/RGB → 448 BGR float, 白合成→正方形パディング→
リサイズ・正規化なし・`dollma_label_image.py` 準拠) ② `Wd14Tagger` に `selected_tags.csv` 名前
マッピング追加 (`quality_gate.hpp` の CSV 名前→index 解決を流用) ③ `pipeline.hpp` のダミー乱数画像入力
(L258) と `tag<idx>` ダミー解決 (L314) を ①② で置換。

---

## Phase 4 F 品質フィードバックループ — 決定ログ

### F-0a 信号ゲート (✅ 実走 80/80 → 判定 = 信号弱・補強してから / 2026-07-02・研究機)

reward std 0.0377 / best−worst 0.2031 (PL 閾値 std>0.1 かつ best−worst>0.3 に未達)。worst-axis
argmax **Limbs 77/80** で他 7 軸ほぼ死 (ScorerNet dynamic range が Limbs 単軸) + worst 帯は
多人数/背景/mecha/文字焼き込み等スコープ外題材への confound (画像照合: 単独素直題材は解剖正常で
reward≈0)。ただし生成 prompt の clean vs clutter で **|r| 4倍分離** (0.007 vs 0.0285) = 弱いが
本物の勾配源・−1 飽和帯ではない (checkpoint エスカレーション不要)。詳細 measurements-log.md。

### F-0b SFT (✅ 完遂・不採用クローズ / 2026-07-05・docs/f0b-rejection-sft-plan.md)

前提の Q-2 (quality を CLIP-embed 枝に分離・自作 QualityMLP を CLIP image embed 上で waifu 蒸留
OOF corr+0.53・NPU 疎通・reward std 0.038→0.104 で信号ゲート通過・docs/q2-quality-branch-plan.md)
完了後、RAFT (best-of-8→top-1→SFT) を G-1 400ペア→G-2a SFT(正典から層状)→G-2b reward前後比→
G-3 判定で end-to-end 実装・実走。**結果=不採用**: 便益 reward +0.017 (弱・60%正・~2.4σ・SDXL
seed 交絡・ほぼ全量 quality由来) が コスト diverse set-F1 −0.017/−0.024 (構造的・全レシピ) を
正当化できず。正典 bitnet_dense 無改変・SFT 重み隔離保存。知見: best-of-N reward(解剖+美的) と
gold タグ set-F1 は非整合 / anatomy ほぼ死 / SDXL seed 非再現(SAC)が比較ノイズ源 / 日本語空条件。
パイプライン再利用可。次レバー = reward設計 / 日本語条件付け改修 / seed制御。

---

## Phase 4 A/D — クローズ経緯

### A 同一性条件付け (a12k 4 seed sweep でクローズ・二分結論 / training-spec §9.10)

① **diverse-val 生成 F1 は 12k で seed ノイズ** (4 set/metric とも判定 NO・seed 42 のみ符号反転・
across delta −0.015±0.032 < #1 帯 sd 0.022 = D6 と同型・施策 B ~2,000 飽和と整合)。② **identity
retention は頑健に 0.975 達成** (across-seed 0.9748±0.0010・全 seed・base ~0.58–0.63 から) =
**identity 条件付けの機能基盤**。→ A の効果は diverse-val F1 でなく retention。a25k は回さず未使用
保持・本番 #1 即時差し替えなし・`[B-merge-at-A]` でまとめ焼き。

### D 容量増 33M→80M (陰性確定・80M 不採用 / training-spec §16)

`DOLLAMA_BITNET_ARCH=d80m`・N_LAYERS 8→16 / FFN_DIM 1792→2464 = 79.91M。両アーム同一レシピ
(b2000 ∧ a12k identity)・`--arch` だけ差・4 seed 6ep paired sweep。diverse-val F1/Jaccard は
4 set/metric とも判定 NO (seed 20260620 で負・seed 7 で正と符号反転・across 平均 −0.002〜−0.004 で
c33 seed 分散帯 sd 以下 = A12k/D6 と同型の seed ノイズ)・retention は 3/4 seed が床 0.975 割れ・
in-dist 微退行。**容量では diverse-val F1 は取れない (データ律速・施策 B ~2,000 飽和と整合)** →
**勝者 = c33 (33M・b2000 ∧ a12k identity) = #1 超え出荷候補**。80M は forward ~2x の対価に見合う
実利なし。蒸留 4 路線 (D2/D4 hard CE 混合・D5 共起 soft・D6 外部教師 TIPO) も全て recall/F1 非寄与
(training-spec §10–12) = 33M は 4,500 ペアから学べる分を学び切った。

---

## バックログ深掘り (設計エッセイの退避)

roadmap.md のバックログ表は結論のみ。以下は投資判断の背景となる長い設計論。

### 遠隔 HW ノード (LAN 越し第 2 マシンの協調)

余剰ノート PC (**Ryzen 7 5700/5800 = Zen3 8C/16T + RTX3060 Laptop = sm_86 / VRAM 6GB + 64GB RAM・
2.5GbE LAN・NPU なし**) をプロジェクトに足す案 (ユーザー 2026-06-29)。

**原則の切り分け**: ❌ **密結合 (1 画像の step 内でモデルを機械間分割) は不可** — attention だけ別機等は
per-layer 往復でネットワークレイテンシ律速。⭕ **粗粒度 (モデル 1 個 = 1 stage を機械にまたいで置く /
ジョブ単位) は成立**。**帯域は非ボトルネック**: 2.5GbE 実効 ~280 MB/s で stage 間ペイロード (latent
128KB ~0.5ms / CLIP embeds 308KB ~1ms / 1024² PNG ~1MB ~4ms) は全部 ms 級・重み 5.1GB は起動時 1 回
常駐で per-request では流さない。残る制約はレイテンシ (層単位分割不可) と「5080 が速すぎて laptop に
振る価値のある stage が限られる」点。**64GB RAM が VRAM 6GB の壁をほどく**: sequential offload で
6GB の 3060 でも SDXL 1024² 本番がフル解像度で回る (遅いが落ちない) → 下書き専任に縛られず本番ノード化可。

**共通土台 = `RemoteNode` 抽象** (Phase 3 の cpp-httplib/json を流用し laptop を dollama 常駐サーバ化 +
主機にクライアント。「HW 環境抽象化」を *remote HW* まで延伸)。**設計前提 (ユーザー必須要件
2026-06-29): 協調 PC のあり・なしを簡単に切替できること** — 遠隔ノードは**完全 opt-in で既定は単機
(遠隔なしで全機能成立・現状の挙動は無改変)**、宣言 1 つ (例 `--remote-node=host:port` / 既定 `none`)。
**不在時はローカル HW へ自動フォールバック** (デバイス計画に remote ノードが居るかだけの差にする)。
laptop は「**無くても全機能が動き、有れば正確性が増す**」加速器 = 依存先にはならない (**本命は速度でなく
"精度の上乗せ"**・後述③ critic の深度が遠隔の有無で graceful に劣化するだけで生成は常に成立)。

その上に 2 つの消費者 (排他でなく ①→③ の一本道):

- **① 訓練/sweep 分散 (即効・最大効率・③の de-risking)** — 3060 (Ampere/TensorCore/FP16 フル) は
  開発機 GTX1080Ti (Pascal・FP16 1:64 で実質 FP32 強制) より小型 LM 訓練が素直に速い。D 容量増 80M
  (~3.8h) や 4 seed sweep は機械間通信ゼロで割れる典型 → 2 台分散で壁時計 ~半減。これが sm_86 ビルド
  (自作 CUDA・wmma 含む) と LAN ジョブ配管を枯らし③の足場を兼ねる。
- **③ レビュー/critic ノード (研究本命) = 生成は研究機 / 講評は遠隔** — producer/critic を 2 台に分離。
  Phase 4 F の *機械間* 版・MoE×HW の「アービタ = 品質スコアラ」を機械間に出した形。研究機が batch 生成
  → PNG を送る → 遠隔が採点 (異常 flag / 同一性照合 / 解剖・ポーズ整合) → verdict を返す → 不合格を
  再生成 / 報酬で LM fine-tune。**RTX3060 Laptop はそれなりの計算力**ゆえ critic は DWPose 等の keypoint
  抽出 / 学習済み解剖・ポーズ分類器 / 小型 VLM 講評 (64GB offload) といった本物の学習済みレビュアーを
  載せられる。**旗艦例 = 解剖/ポーズ critic を 3 Tier で段階化**し遠隔の有無でレビュー深度が graceful
  degradation する: **Tier A 数・位相** (指/四肢の本数・重複・欠損・左右対称) = WD14/QualityGate で軽く
  **研究機の遊休 NPU 常駐**・遠隔不要。**Tier B 骨格レベル** (DWPose 等で 2D keypoint → 連結/本数/貫通の
  粗い不可能判定) = 重い・**遠隔**。**Tier C 物理的妥当性** (3D リフト / 学習済み解剖 prior / VLM critic) =
  最重・誤検出しやすい研究フロンティア・**遠隔**。**§11 解剖メタ整合検査の線引きを継承**: A は確定
  スコープ、B/C は §11 が *意図的に避けた* 角度・比率・ポーズ自然さ領域 → **foreshortening/デフォルメを
  罰しない設計が必須** (強い 2D パースを「崩壊」と誤検出すると Phase 5 の攻めた画角を殺す)。**Phase 5
  5-1 の採点器を格上げ**: 現 5-1 は QualityGate 異常タグ hit 率 (タグ粒度) → B/C critic はより強い崩壊
  検出器になり 5-1→5-2/5-3 の投資判断を支える。増分は stage 分割プロトコル + scheduler (5080 が拡散で
  詰まる裏の遊休へ何を逃がすか = 芯「全 HW 使い切り」の機械間版)。**着手順 ①→③** (① が独立した見返りを
  持ちつつ③の sm_86 ビルド/LAN 配管リスクを潰す)。

### CPU 側 LM 推論 Tier 2(A) 独立 forward ワーカー — 設計確定/留保の裏取り

Tier 1 (AVX2 単スレッド ~5x) 完了後の続き。**確定設計 = (A) 独立 forward ワーカー方式**: `linear` 内は
Tier 1 単スレッドのまま、複数フレームの forward を **disjoint 物理コア (`cpu_topology.hpp` 自動検出・
上限 5–6・HT 兄弟は非動員) に pin した embarrassingly parallel ワーカー**で回す (プールは forward 単位の
タスク投入・linear 内分割はしない)。**当初案「linear 内 out_dim 分割」は却下** (単一 forward レイテンシは
GPU 版 #6-GPU 87.5x が担う・役割に動機なし)。

**【2026-06-28 計測クローズ】** 駆動側を `src/core/multi_frame_pipeline.hpp` (MultiFramePipeline) として
実装しテスト+並列ベンチを与えた (test_multi_frame_pipeline・42/42 全緑)。実デバイス比率スタブで実測した
結果、発動条件 (LM 段ボトルネック化) は**単一 GPU 構成では成立しない**: per_frame 3879.8ms ≈ SDXL 単段
3800ms (理論 GPU 上限の 98% = GPU バウンド)・`queue_bclip_to_bsdxl` 待ち 0.0002ms ≈0 (GPU 飢餓なし)・
LM (404ms) は SDXL の裏に完全隠蔽。QueueDepth {2,4,8} スイープも 2 で飽和 (look-ahead 既定 2 が最適)。
**= 単一 GPU では stage A は飢餓を起こさず Tier 2(A) の動機なし。留保継続を計測で裏取り。再評価は SDXL が
ライブラリ fallback 等で桁違いに速くなった世界のみ**。

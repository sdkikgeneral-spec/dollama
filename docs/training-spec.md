# dollama タグ生成 LM 訓練仕様 (Phase 4 dense 本線 #4)

`scripts/train_bitnet.py` が `src/models/bitnet.hpp` と数値的に等価な dense (FP) モデルを
訓練し、重みを safetensors 出力する仕様。**初版は hard CE のみ・蒸留なし**。蒸留
(DanTagGen/Qwen2 教師・KL) は次イテレーション。

## 1. アーキ定数 (bitnet.hpp と厳密一致)

| 定数 | 値 |
|---|---|
| VOCAB_SIZE | 4999 (specials 5 + tags 4994) |
| D_MODEL | 512 |
| N_LAYERS | 8 |
| N_HEADS | 8 |
| HEAD_DIM | 64 |
| FFN_DIM | 1792 (SwiGLU gate/up/down) |
| MAX_SEQ_LEN | 64 |
| RoPE base | 10000 |
| RMSNorm eps | 1e-5 |
| embed tied | lm_head = embed 重み共有 (別パラメータ無し) |
| パラメータ数 | **32,976,896** (PyTorch 実測 = bitnet.hpp::param_count と一致) |

- **#4 は通常の FP `nn.Linear` (bias なし) で学習する。BitLinear / ternary / int8 QAT は
  訓練に入れない。** ternary 圧縮は dense が動いた後の #5 で後段被せ。
- bias は全 Linear で無し (bitnet.hpp BitLinear も bias 無し)。
- RMSNorm は重み乗算あり・bias 無し。mean(x^2) は FP32 で蓄積 (参照実装が double 蓄積のため)。
- **推論アーキは bitnet.hpp 等価・dropout 等の訓練時正則化は別 (D5 / §11)**: `--dropout`
  で入る residual dropout は **訓練時のみ** の正則化。`nn.Dropout` は `model.eval()` で恒等
  写像になるため **eval forward は dropout 率に依らず bitwise 完全一致** (実測 dropout 0.0 vs
  0.3 の eval forward max abs diff = 0.0)。golden dump は常に eval で走るので #4/#6/A3 golden は
  非回帰。dropout はアーキにも safetensors 重みにも痕跡を残さない (bitnet.hpp は dropout を
  持たない=推論等価)。既定 `--dropout 0.0` は従来と完全一致。

### RoPE — GPT-NeoX 系ペアリング (最重要整合点)

bitnet.hpp::apply_rope は `(i, i+HEAD_DIM/2)` のペアを回転する NeoX 系レイアウト。
PyTorch 側も同方式 (HF LLaMA の interleaved ではない):

```
half = HEAD_DIM/2
freq_i = ROPE_BASE^(-(2i)/HEAD_DIM),  i in [0, half)
angle  = pos * freq_i
out[i]      = x[i]*cos(angle) - x[i+half]*sin(angle)
out[i+half] = x[i]*sin(angle) + x[i+half]*cos(angle)
```

`train_bitnet.py::apply_rope` は `x[..., :half]` / `x[..., half:]` を a/b に取り、
上式そのままを適用する。**ここがズレると #6 のゴールデン突合が落ちる。**

## 2. 損失 (hard cross-entropy)

- 自己回帰次トークン予測。input = ids[:-1]、target = ids[1:]、ignore_index=-100。
- `--loss-mode tags` (既定・採用) : tags 部 (＜sep＞ の次以降を予測する位置) のみで loss。
  `--loss-mode all` (text 部も含め全位置) も実装済み。
- **採用根拠**: 5 epoch 比較で top10 recall は tags=0.7766 / all=0.7788 とほぼ同点 (差は noise)。
  生成目的 (text を読んで tags を吐く) に直結する tags-only を主軸に採用。val_loss は
  mode 間で含む位置が違い直接比較不可なので、tags 部のみで測る recall を判定指標とした。
- 勾配クリップ 1.0、AdamW (betas 0.9/0.95, weight_decay 0.01)、warmup 100 step +
  cosine decay (total = n_batches × epochs)。

## 3. データ整形 (dataset-spec §3 準拠)

- 入力 `data/bitnet/pairs.{train,val}.jsonl` (train 4500 / val 500)。`text`=自然文、`tags`=正準順序 target。
- **語彙は `data/bitnet/vocab.json` を唯一のソースに読む** (Python 側で二重定義しない)。
  specials id 0..4 / `tags[i].id == 5+i` / §6 正規化 (英数字に挟まれた `_` のみスペース、
  顔文字の `_` は保持) を tokenizer.hpp::normalize と同一ロジックで再現・ロード時検証する。
- 自己回帰列: **`<bos> text(タグ化) <sep> tags <eos>`**。
  - text 側は tokenizer.hpp::encode_text と同じ greedy 最長一致 (正規化 → 英数字連結語へ分割
    → 最長連結タグを貪欲一致、非語彙単語はスキップ)。
  - tags 側は正準順序 target を tag_to_id で id 化 (未知は `<unk>`=4)。
  - MAX_SEQ_LEN=64 で打ち切り (末尾を `<eos>` に詰める)。実測 seq_len min 9 / max 30 / mean 22.9。
- specials id: `<pad>`=0 `<bos>`=1 `<eos>`=2 `<sep>`=3 `<unk>`=4。pad 位置は ignore_index で loss 除外。

## 4. 重みレイアウト (safetensors・#6 がそのままロード)

`src/io/safetensors.hpp` (F32/F16/BF16) でロードする標準フォーマット。テンソル名・shape は
bitnet.hpp::Layer 構造に 1:1 対応、**row-major `[out, in]`** (= PyTorch `nn.Linear.weight` /
bitlinear の `w[o*in_dim + i]` と一致)。

| テンソル名 | shape | 対応 (bitnet.hpp) |
|---|---|---|
| `embed` | [4999, 512] | embed_ (tied lm_head 兼用・別名重複出力しない) |
| `layers.{i}.attn_norm` | [512] | Layer::attn_norm |
| `layers.{i}.wq` / `wk` / `wv` / `wo` | [512, 512] | Layer::wq/wk/wv/wo |
| `layers.{i}.ffn_norm` | [512] | Layer::ffn_norm |
| `layers.{i}.w_gate` / `w_up` | [1792, 512] | Layer::w_gate/w_up |
| `layers.{i}.w_down` | [512, 1792] | Layer::w_down |
| `final_norm` | [512] | final_norm_ |

- i は 0..7 (N_LAYERS=8)。総 74 テンソル。
- **命名は bitnet.hpp に明記が無いため本表を採用した**。#6 (C++ 推論) 実装者はこの名前で
  ロードすること。bitnet.hpp の Layer メンバ名と素直に対応する命名にしてある。
- 出力 2 種:
  - `data/bitnet/bitnet_dense.safetensors` — **FP16 本体** (62.9 MB)。
  - `data/bitnet/bitnet_dense_fp32.safetensors` — **FP32 golden** (125.8 MB、#6 の誤差切り分け用)。
- 訓練は FP32 で実施し、出力時に該当 dtype へキャストして保存。
- スクリプト末尾で読み戻しサニティ (全テンソルの name/shape/dtype 一致・NaN/Inf 無し) を実行。

## 5. 訓練環境・再現手順

- **開発機 GTX1080Ti (sm_61, FP16 native 非対応) / i7-10700 / NPU なし**。研究機 (RTX5080) とは別物。
- **訓練は FP32 で回す** (1080Ti で AMP/FP16 訓練は旨味なし)。出力時のみ FP16 へキャスト。
- Python 3.12 + `torch 2.5.1+cu121` (cu121 ビルドは sm_61 で動作確認済み) + `safetensors 0.8.0` + `numpy`。
  - Python 3.14 既定環境には torch 安定 wheel が無いため Python 3.12 を使用。
- seed = 20260620 固定 (`random` / `torch.manual_seed`)。

```sh
# smoke (パイプライン疎通・少数バッチ 1 epoch)
python scripts/train_bitnet.py --smoke
# 本訓練 (採用設定: 6 epoch・tags loss・bs32・lr3e-4)
python scripts/train_bitnet.py --epochs 6 --loss-mode tags --batch-size 32 --lr 3e-4
```

- Windows コンソールで日本語ログを出すなら `PYTHONIOENCODING=utf-8` を設定。
- 学習ログは `data/bitnet/train_stats.json` (epoch ごと train/val loss・top-k recall・
  ハイパラ・seed・モデルサイズ・tensor_keys)。

## 6. 初回ラン結果 (採用: 6 epoch / tags loss)

| 指標 | 値 |
|---|---|
| device | CUDA / GTX 1080 Ti (sm_61) |
| 訓練時間 | 45.3 s (6 epoch, ~7.5 s/epoch) |
| 初期 val_loss (ep0) | 3.22 |
| 最終 val_loss (ep5) | 2.41 (最小は ep4 = 2.3825) |
| 最終 train_loss | 1.32 |
| top10 tag recall (val, teacher forcing) | **0.7767** |
| random baseline (top10/4999) | 0.0020 |
| **recall vs random** | **388x** |
| val target tags 評価数 | 8,254 |
| モデルサイズ | FP16 62.9 MB / FP32 125.8 MB |
| params | 32,976,896 (= bitnet.hpp) |

- val_loss は ep0→ep4 で単調下降 (3.22→2.38) し収束。
- **過学習特性**: 4,500 ペアは小規模で、20 epoch まで延ばすと ep4 を底に val_loss が
  単調上昇 (ep19 = 3.35) し recall も低下 (0.73)。よって**採用は 6 epoch** (cosine を 6 で
  サイズし val 最小付近で LR を絞り切る)。これ以上は早期終了 or 蒸留/正則化が要る (次イテ)。

## 7. 未解決・次イテレーション

- **蒸留 (本タスク範囲外)**: DanTagGen/Qwen2 教師の soft label (KL) で過学習を抑え、
  小規模データでの汎化を上げる。`source:"llm_distill"` ペア追加 (dataset-spec §12) と併せて。
- **過学習対策**: dropout / より強い weight_decay / 早期終了 / データ拡張。
- **#6 ゴールデン突合**: bitnet.hpp の dense 等価 forward (BitLinear を素の FP Linear に
  置き換えた版) と本モデルの logits を突き合わせ、RoPE NeoX 方式・RMSNorm・SwiGLU・
  tied lm_head の数値一致を確認する (FP32 golden を使用)。

## 8. golden dump フォーマット仕様 (#6 C++ dense 推論の数値突合用)

`scripts/train_bitnet.py --dump-golden` は `data/bitnet/bitnet_dense_fp32.safetensors`
(6 epoch 採用重み・FP32) を `BitNetDense` にロードし、`data/bitnet/golden/` に以下を出力する。
このサブモードは**訓練を行わず、本番の重み・`train_stats.json` を一切書き換えない** (golden のみ書く)。
golden は `.gitignore` 済み (大容量・再生成可)。生成は CPU 既定 (決定性重視・1080Ti は FP16 native 非対応)。

### 8.1 出力ファイル一覧

| ファイル | 形式 | 内容 |
|---|---|---|
| `logits_golden.safetensors` | safetensors (little-endian raw) | (a) ロジット golden |
| `gen_golden.safetensors` | safetensors | (b) 生成 golden |
| `manifest.json` | JSON (UTF-8, `ensure_ascii=False`) | 全ケースの索引・停止規則・arch 定数 |

ロジット・id 列をすべて safetensors にしたのは、#6 が `src/io/safetensors.hpp` で
同じローダーで読め、dtype/shape/エンディアン取り違えを構造的に排除できるため。

### 8.2 `logits_golden.safetensors` のテンソル (6 テンソル)

各 seq_len ∈ {8, 32, 63} について 2 テンソル (63 は MAX_SEQ_LEN=64 直下の境界)。

| テンソル名 | shape | dtype | 内容 |
|---|---|---|---|
| `logits_s{sl}` | `[sl, 4999]` | F32 | input_ids を `BitNetDense` に通した FP32 logits (row-major [seq, vocab]) |
| `input_ids_s{sl}` | `[sl]` | I32 | 入力 token id 列 (1D) |

- `{sl}` ∈ `8 / 32 / 63`。
- input_ids は `pairs.val.jsonl` の自己回帰列 (`build_sequence`) を固定順に連結し
  seq_len ちょうどで切り出したもの (再現可能・全 id が seq 範囲内)。具体値は manifest に記録。

### 8.3 `gen_golden.safetensors` のテンソル (10 テンソル)

5 ケース × 2 テンソル。`gen_len==0` のケースは shape `[0]` の空テンソルで保存する。

| テンソル名 | shape | dtype | 内容 |
|---|---|---|---|
| `prompt_ids_c{ci}` | `[plen]` | I32 | プロンプト id 列 = `[<bos>] + encode_text_greedy(text) + [<sep>]` |
| `gen_ids_c{ci}` | `[gen_len]` | I32 | greedy 生成した tag id 列 (prompt・`<eos>` を**含まない** pure な生成部) |

`{ci}` ∈ `0..4`。各ケースの text/prompt_ids/gen_ids/gen_tags_readable は manifest に記録。

### 8.4 greedy 停止規則 (C++ #6 が完全再現すること)

`manifest.json` の `generation_golden.greedy_stop_rule` にも機械可読で記録。手順:

1. **prompt** = `[<bos>] + encode_text_greedy(text) + [<sep>]`
   (`<bos>`=1, `<sep>`=3。`encode_text_greedy` は §6 正規化 + 英数字連結語の greedy 最長一致・
   非語彙語スキップ。tokenizer.hpp::encode_text と同一)。
2. **各ステップ**: 現在列全体を `BitNetDense` に通し、**最終位置 (列末) の logits** を取る。
   その argmax を次トークンとする (**FP32 で argmax**・**ties は最小 id** を採用 = `torch.argmax` の挙動)。
3. 生成した id を列に追記し、生成 tag id 列にも記録する。
4. **停止条件** (どちらか満たしたら即停止):
   - 次トークン == `<eos>`(=2) → `<eos>` は生成列に**含めず**停止。
   - 列長が `MAX_SEQ_LEN`(=64) に達した → そこで打ち切り。
5. **repeat 抑制・重複除去は行わない** (素の greedy。同一タグが連続出力されうる。C++ も同一に)。
6. `<eos>` **以外**の specials (`<pad>`=0 / `<bos>`=1 / `<sep>`=3 / `<unk>`=4) が生成されても
   特別扱いせず、そのまま列に積む (**停止するのは `<eos>` のみ**)。

### 8.5 サニティ・自己整合 (dump 時に検証済み)

- ロジット自己整合: 同一入力を 2 回 forward し `max_abs == 0.0` (完全一致) を確認。
- 生成決定性: 同一 prompt を 2 回デコードし gen_ids 一致を確認。
- 読み戻し: 保存後に再ロードし shape/dtype/NaN・Inf なし・manifest の id 値一致を確認。

### 8.6 採用 golden の各ケース (6 epoch 重み・seed 20260620・CPU)

ロジット: seq_len 8 / 32 / 63 (各 logits `[sl,4999]` F32・input_ids `[sl]` I32)。

| case | text | prompt_len | gen_len |
|---|---|---|---|
| 0 | `a girl with long hair, looking at viewer, blue eyes.` | 5 | 16 |
| 1 | `a boy with black hair, smile.` | 4 | 8 |
| 2 | `1girl, twintails, school uniform, blush.` | 6 | 14 |
| 3 | `金髪ツインテールの少女が笑っている` | 2 | 9 |
| 4 | `two girls holding hands, blue sky.` | 4 | 8 |

(case 3 は語彙外の日本語のため `encode_text_greedy` がほぼ空 → prompt は `<bos><sep>` の 2 トークン。
読み取れる語彙が無くてもモデルは `<sep>` 以降の自己回帰で生成を続ける。)

## 9. Phase 4 A (A2) — 同一性条件付き混合訓練

`scripts/train_bitnet.py --identity` で #1 (synthetic) と A1 (identity_cond, dataset-spec §13)
を**混合訓練**し、同一性条件付きタグ生成を学習する。アーキ・vocab・specials・dtype・
重み safetensors レイアウト (74 テンソル [out,in]) は #4 と**完全に不変**。`--identity`
未指定なら従来の #4 純 synthetic 訓練 (出力名・挙動とも) のまま (非回帰)。

### 9.1 条件付け機構 (承認済み・厳守 / dataset-spec §13.1)

`<sep>`(id=3) を 2 回流用した prompt prefix 方式。`build_sequence` を source で分岐する:

- `source=="synthetic"` (#1・1-`<sep>`): `<bos> text(greedy タグ) <sep> tags <eos>`。
- `source=="identity_cond"` (A1・2-`<sep>`):
  `<bos> [identity ids] <sep> [scene text の greedy タグ] <sep> [target ids] <eos>`。
  - identity 区間 = `meta.identity_tags` を `tag_to_id` で直接 id 化。
  - scene 区間 = 自然文 `text` を `encode_text_greedy` で greedy 最長一致したタグ列
    (推論時 prompt は自然文しか来ないため scene 条件も自然文由来トークンで揃える)。
  - target 区間 = 正準順序 `tags` を id 化。
- vocab/specials/MAX_SEQ_LEN は不変。`<sep>` が 2 個出るのは仕様 (区間は **位置** で決まる)。

### 9.2 損失マスク (両形式とも「target 区間のみ hard CE」)

`build_sequence` が返す `tags_start` を「最後の構造区切り (`<sep>`) の次」に置く
(identity_cond は 2 個目 `<sep>` の次、synthetic は 1 個目の次)。`collate(loss_mode="tags")`
は `tags_start-1` 未満の target を `-100` (ignore_index) でマスクするため、identity prefix
と scene 区間は loss から除外され、両形式が同じ「target のみ CE」原則で揃う。

### 9.3 混合・source 別指標

- 混合 = `pairs.train.jsonl` + `pairs.identity.train.jsonl` を **単純結合 + シャッフル**
  (seed 固定)。比率は両ファイル 4500/500 で 1:1 相当。val も両方含む。
- `eval_loss_and_recall_by_source` が混合 val を 1 走査し source 別に val_loss / top-k recall
  を分離集計する (per-token CE を `reduction="none"` で取り source へ振り分け)。毎 epoch +
  最終に `synthetic` / `identity_cond` / `all` を報告し `train_stats_identity.json` に記録。

### 9.4 identity retention rate (A2 最重要新指標)

```
retention = mean over identity_cond val (
  |生成 target に出た identity 集合 (語彙内)| / |入力 identity 集合 (語彙内)| )
```

- `eval_identity_retention`: 各 identity_cond val 行で prompt =
  `<bos> identity <sep> scene_text(greedy) <sep>` を組み、`_greedy_generate`
  (GREEDY_STOP_RULE 準拠・`<eos>` で停止 / MAX_SEQ_LEN 打ち切り) で target を生成。
  生成集合に入力 identity id が何個含まれるかを測る。`<unk>` は分母から除外。

### 9.5 出力ファイル (設計判断 = #4 と別名)

identity 対応モデルは #4 の純 dense と**別名**にする。理由: #4 の
`bitnet_dense_fp32.safetensors` は #6 (`src/infer/bitnet.hpp`) の golden 突合に使われており
壊せない。混合訓練したモデルは別物なので名前を分け #4/#6 を保全する。

| 出力 | ファイル |
|---|---|
| FP16 本体 | `data/bitnet/bitnet_dense_identity.safetensors` |
| FP32 golden | `data/bitnet/bitnet_dense_identity_fp32.safetensors` |
| 学習統計 | `data/bitnet/train_stats_identity.json` (追跡・小) |
| smoke (上書き禁止) | `bitnet_dense_identity_smoke*.safetensors` / `train_stats_identity_smoke.json` |

`--smoke` は本番重み/stats を一切上書きしない (#4 と同じ footgun 対策)。

### 9.6 A2 ラン結果 (採用: 6 epoch / tags loss / 混合 9000 train・seed 20260620)

| 指標 | 値 |
|---|---|
| device | CUDA / GTX 1080 Ti (sm_61) |
| 訓練時間 | 110.9 s (6 epoch・混合 train 9000 / val 1000) |
| final val_loss (all) | 1.5102 |
| final top10 recall (all) | 0.9233 |
| **synthetic** val_loss / recall | **1.6708 / 0.9003** (n=500・target tags 8254) |
| **identity_cond** val_loss / recall | **1.3517 / 0.9460** (n=500・target tags 8363) |
| **identity retention rate** | **0.9474** (n_cases=500) |
| params / モデルサイズ | 32,976,896 / FP16 62.9 MB |

- **retention 0.9474**: identity 条件付け prompt を与えると、生成 target が入力 identity の
  ~95% を再現する。条件付け機構が効いていることの直接証拠。#4 (synthetic のみ) は identity
  条件入力の系列を学習していないため retention の同条件比較対象を持たない (#4 は
  `<bos> text <sep>` プロンプトで identity prefix を解さない) が、混合訓練後の絶対値 0.9474 は
  「2 個目 `<sep>` 以降の target が prefix identity を強く引き継ぐ」ことを示す。
- source 別では identity_cond の方が val_loss 低・recall 高 (prompt に identity が露出する分
  予測が容易)。両 source とも val_loss は ep0→ep5 で単調下降し収束。
- 混合で train が 2 倍 (9000) になったため #4 (4500・recall 0.7767) より過学習が緩く、
  6 epoch 時点の recall が高い。

### 9.7 identity golden dump (`--dump-golden-identity`, A3 C++ 突合用)

`bitnet_dense_identity_fp32.safetensors` を `BitNetDense` にロードし、identity 条件付き系列の
golden を `data/bitnet/golden/` に **identity_ 別ファイル**で出力する (#4 の synthetic golden
`logits_golden`/`gen_golden`/`manifest.json` は**非回帰**・byte 一致を確認済み):

| ファイル | 内容 |
|---|---|
| `logits_golden_identity.safetensors` | identity_cond 2-`<sep>` 系列 (seq 8/32/63) の FP32 logits + input_ids (I32)。seq63 は `<sep>` 4 個 (連結 2 系列分)。 |
| `gen_golden_identity.safetensors` | 5 ケース × (prompt_ids / gen_ids / identity_ids、各 I32)。prompt = identity prefix + scene greedy + `<sep>`。 |
| `manifest_identity.json` | 条件付け機構・各ケース (post_id/text/identity/gen/retained)・GREEDY_STOP_RULE。 |

- サニティ: logits 自己整合 (再 forward max_abs 0.0)・生成決定性・読み戻し shape/dtype/NaN/Inf・
  manifest 値一致を dump 時に検証 (全通過)。生成 5 ケースの identity_retained は
  9/9・7/7・7/7・5/5・5/5 (golden レベルでも retention=1.0 のケースが揃う)。

### 9.8 再現手順

```sh
# 混合訓練 (採用設定)
py -3.12 scripts/train_bitnet.py --identity --epochs 6 --loss-mode tags --batch-size 32 --lr 3e-4
# identity golden dump (A3 用)
py -3.12 scripts/train_bitnet.py --dump-golden-identity
# smoke (疎通・本番を上書きしない)
py -3.12 scripts/train_bitnet.py --identity --smoke
```

### 9.9 引き渡し (A3 = cpp-implementer)

- `src/infer/bitnet.hpp` は `bitnet_dense_identity_fp32.safetensors` (#4 と同一 74 テンソル
  [out,in] レイアウト) をロードし、`<bos> identity <sep> scene <sep>` prompt から
  `manifest_identity.json` の GREEDY_STOP_RULE で生成、`logits_golden_identity` /
  `gen_golden_identity` と突合する。tokenizer.hpp / vocab.json は変更不要 (`<sep>` 2 回は
  decode が構造トークンとして読み飛ばす)。

### 9.10 A 実ペア増 a12k seed sweep (Phase 4-A 実ペア増・A クローズ確定)

A2 (§9.6) の identity_cond ペアを Phase 1 (dataset-spec §17) で 5k -> **a12k** (train 10,800 /
val 1,200・teacher_retention 1.0・凍結 eval 除外・リーク 0) に増やし、施策 C (§13) で据えた
**凍結 diverse-val 生成 set-F1** + **identity retention** の二物差しで、実ペア増の効果が seed
横断で頑健かを確定した。base(#1 plain hard CE) vs a(A2 `--identity` 混合) を **4 seed
(20260620 / 20260621 / 42 / 7)・6ep paired** で回す
(`scripts/dollma_a_seedsweep.py --scale a12k` + `dollma_a_seedsweep_analyze.py --scale a12k`)。
両 arm とも val・eval_diverse_a/b は完全共通で、唯一の差分は a arm に identity_cond 行が混ざる
こと。出力は `data/bitnet/_seedsweep_a12k/` 配下のみ (本番 `bitnet_dense*`/`identity`・golden・#1
本線 pairs・凍結 eval を一切無改変)。

**判定軸** (施策 B/C/D と同一): (a) 全 seed で delta の符号が + で一貫 (b) delta 平均が #1 自身の
seed 分散帯 (base を seed 間で比べた band) の sd を超える (c) 各 seed の paired bootstrap 95%CI が
0 を除外。

**(1) diverse-val 生成 F1/Jaccard = seed ノイズ (頑健でない・D6 と同型)**

| set / metric | per-seed delta = a - base (20260620 / 20260621 / 42 / 7) | across 平均 +- sd | #1 band sd | (a)符号 | (b)>band | (c)CI除外 | 頑健 |
|---|---|---|---|---|---|---|---|
| diverse_a / F1 | -0.0118 / -0.0358 / **+0.0286** / -0.0405 | -0.0149 +- 0.0316 | 0.0221 | NO (seed42 反転) | NO | YES x4 | **NO** |
| diverse_a / Jaccard | -0.0040 / -0.0179 / **+0.0183** / -0.0200 | -0.0059 +- 0.0176 | 0.0124 | NO | NO | YES x4 | **NO** |
| diverse_b / F1 | -0.0112 / -0.0376 / **+0.0286** / -0.0431 | -0.0158 +- 0.0327 | 0.0223 | NO | NO | YES x4 | **NO** |
| diverse_b / Jaccard | -0.0030 / -0.0169 / **+0.0189** / -0.0206 | -0.0054 +- 0.0179 | 0.0125 | NO | NO | F/Y/Y/Y | **NO** |

全 4 set/metric で判定 NO。各 seed 内では paired CI が 0 を除外する (per-sample n=1500・|t| 大) が、
**seed 42 だけ符号が反転**し、across 平均は #1 自身の seed 分散帯 sd 以下に埋もれる。これは D6
(§12.5・外部教師 TIPO 蒸留) と**同型の seed ノイズ** — 「単一 seed では有意に見えるが seed を跨ぐと
符号が安定しない」。施策 B が ~2,000 件で diverse-val 利得が飽和したこと (§14.9) とも整合的で、
**A の実ペア増は diverse-val 生成 F1 を上げる手ではない**。in-dist `pairs.val` の F1 は base とほぼ同値・
legacy recall@10 は 0.78 -> 0.84 と上がり、a arm の val_loss は base より低い (識別は学習しているが、
凍結 diverse-val の生成集合一致では優位が seed 頑健に立たない)。

**(2) identity retention = 全 seed 頑健に 0.975 (A の本効果)**

| seed | base retention | a retention |
|---|---|---|
| 20260620 | 0.5893 | 0.9753 |
| 20260621 | 0.6314 | 0.9750 |
| 42 | 0.5770 | 0.9734 |
| 7 | 0.5760 | 0.9757 |
| **across-seed** | ~0.576-0.631 | **0.9748 +- 0.0010** |

a arm の retention は **across-seed 0.9748 +- 0.0010** と極めて低分散で全 seed 頑健。base
(synthetic のみ・identity prefix を解さない) の ~0.58-0.63 から実ペア増 + 条件付け学習で 0.975 に
到達する。`n_cases=1200` (a12k val 全件・A1 5k の n=500 から拡大しても retention は同帯で安定)。

**結論 (A クローズ確定)**: A 実ペア増の効果は **diverse-val 生成 F1 ではなく「同一性条件追従
(retention) を頑健に成立させる実ペア基盤」**。a25k は回さない (12k で diverse-val が seed ノイズ
である以上、25k で符号反転が安定化する見込みは薄く、B ~2,000 飽和とも整合)。生成済 a25k ペアは
破棄せず保持 (dataset-spec §17)。本番 #1 の即時差し替えはせず、`[B-merge-at-A]` 遅延条項どおり
B(b2000 多様化) + A(identity 条件付け) を同じ出荷リトレインで 1 回まとめ焼きする。インフラ
(`dollma_a_seedsweep{,_analyze}.py`・base npz スケール共有・`--only-seed` 小出し・冪等 skip) は
再利用可能。


## 10. 蒸留混合の A/B 評価 (D3/D4 — 採用せず・負の結果を記録)

`scripts/train_bitnet.py --distill` で synthetic(4500) + Qwen2 蒸留(3000) = train 7500 を混合
訓練 (val は #1 と同一 `pairs.val.jsonl` 500・**不変**)。出力は `bitnet_dense_distill*` /
`train_stats_distill.json` の**別名** (#1/#4/#6 の重み・golden は無改変)。seed 20260620・FP32・
GTX1080Ti。動機は §6/§7 の過学習 (#1 は ep4 を底に val_loss 反転) を蒸留で緩和できるかの検証。

### 10.1 設計判断

- **epoch**: 反転の底と後退を観察するため**蒸留は 10ep** で実行・採用重みとした。A/B の公平化に
  **#1 を同一 val・同 seed で 10ep/6ep とも非破壊再算出** (scratch data-dir・本番重み無改変)。
  蒸留 train=7500 は #1 train=4500 より 1.67x 多く、同 epoch では勾配ステップ数が多いため
  train_loss が速く落ちる交絡があるので、6ep/10ep の両方で突合した。
- **identity 併用**: `--distill` は単独 (synthetic+distill = #1 系の蒸留版) を主評価とした。
  `--identity --distill` 併用も実装済 (任意・`bitnet_dense_identity_distill*`) だが本評価対象外。
- **#1 重み**: 再算出はすべて scratch dir 出力で #4 の `bitnet_dense_fp32.safetensors` を保全。

### 10.2 A/B 結果 (同一固定 val 500・teacher forcing)

| 指標 | #1 6ep (採用本線) | #1 10ep (再算出) | 蒸留 6ep | 蒸留 10ep (採用重み) |
|---|---|---|---|---|
| 最終 val_loss | **2.411** | 2.749 | 2.599 | 3.108 |
| 最終 top10 recall | **0.7767** | 0.7527 | 0.7616 | 0.7402 |
| 最終 train_loss | 1.320 | 0.211 | 0.534 | 0.102 |
| val_loss 反転(底) epoch | **ep4** (2.382) | ep4 | **ep2** (2.434) | ep2 (2.454) |
| 最終 train-val gap | **1.09** | 2.54 | 2.07 | 3.01 |

### 10.3 生成多様性 (val prompt greedy・本体 tag id)

| | #1 6ep | 蒸留 6ep | 蒸留 10ep |
|---|---|---|---|
| コーパス unique tag 数 | 386 | 606 | **736** |
| 平均 per-seq unique 率 | 0.945 | 0.978 | **0.991** |
| 正規化エントロピー | **0.839** | 0.816 | 0.814 |

### 10.4 判定 — 過学習は緩和されなかった (採用しない)

- **悪化**: val_loss 反転が #1 ep4 → 蒸留 ep2 へ前進、同 epoch の train-val gap は拡大、最良
  val_loss/recall も微減。系列レベル hard CE 混合は本データでは**正則化にならない**。
- **原因**: (1) train 1.67x 増で同 epoch の勾配ステップ過多 → 暗記加速。(2) Qwen2 自由文が
  synthetic val 分布と乖離し、追加容量が train 暗記へ向かい synthetic val を助けない。
- **唯一の利得 (トレードオフ)**: 自由生成の多様性は向上 (unique tag 約 2x・per-seq 反復減)。
  teacher-forced 指標は悪いが、生成語彙の広さ・反復抑制という別軸では改善。
- **方針**: **#1 (6ep synthetic) を本線維持**・蒸留版は採用しない。次イテは ① soft-label KL 蒸留
  (温度付き教師 logits) ② dropout / weight_decay 増 ③ 蒸留と同分布の val 追加、を検討。

### 10.5 再現手順

```sh
# 蒸留混合 (10ep 採用重み・別名出力)
py -3.12 scripts/train_bitnet.py --distill --epochs 10 --loss-mode tags --batch-size 32 --lr 3e-4
# #1 を同 val・同 seed で非破壊再算出 (本番重みを汚さない scratch data-dir 推奨)
py -3.12 scripts/train_bitnet.py --data-dir <scratch> --epochs 6   # / --epochs 10
# smoke (疎通・本番を上書きしない)
py -3.12 scripts/train_bitnet.py --distill --smoke
```

## 11. D5 — soft-label KL 蒸留 (案A 共起 teacher)

`scripts/train_bitnet.py --distill-kl`。D2/D4 の hard CE 混合蒸留 (§10・採用せず) の
**欠陥 = target が hard one-hot のまま (入力側データ拡張で student がソフト知識を見ていない)**
を直す路線 (§10.4 ①)。**新規ペアは作らず**、各 target 位置の**正解分布そのものを軟化**する
soft-label KL 蒸留。②dropout / weight_decay 増・③蒸留分布 val 検討も同時投入。出力は
**別名** `bitnet_dense_kl{,_fp32}.safetensors` / `train_stats_kl.json` (#1/#4/#6/distill/identity
の重み・golden・stats は無改変・mtime で確認済み)。val は #1 と同一 `pairs.val.jsonl` 500 で**不変**。
seed 20260620・FP32・GTX1080Ti。

### 11.1 損失機構

```
L = alpha * T^2 * KL(teacher_softened || student_T) + (1 - alpha) * hardCE
```

- **target 区間マスク共有**: hardCE / KL とも `collate(loss_mode="tags")` の `tgt != -100`
  (= `tags_start-1` 未満を ignore) でマスクした **同じ位置のみ**に加算 (§9.2 厳守)。
  synthetic 1-`<sep>` / identity 2-`<sep>` どちらの形式でも tags_start が「最後の構造区切りの次」
  なので原則は崩れない (D5 本評価は synthetic のみ)。
- **温度 T**: 学生は `log_softmax(logits / T)`。teacher は確率分布なので `teacher^(1/T)` を
  再正規化して温度を反映 (logits を持たない teacher 版の温度)。勾配スケールを hardCE と
  揃えるため `T^2` を乗ずる (Hinton 蒸留の慣例)。
- **alpha=0 は従来 hardCE と bitwise 完全一致** (非回帰チェック・§11.4)。

### 11.2 案A 共起 teacher (prefix 条件付き soft label)

- teacher 分布 = `cache/danbooru_posts.jsonl` (8,200 posts) の**タグ共起経験分布**。各 target
  位置で gold 次タグ g を予測するとき: **主質量 `main_mass` を g に置き、残余 (1-main_mass) を
  「現プレフィクス (その系列で既に出た target 本体タグ集合) と corpus 上で共起するタグ」上位
  `topn` 件へ温度 `cooc_temp` 付き softmax で配る** (= 意味づけされた・共起で裏打ちされた
  ラベルスムージング)。共起候補が無い位置・specials (`<eos>`/`<sep>`) 予測位置は g に全質量
  (= hard と同じ)。
- 共起テーブル = post 内タグの無向ペア共起回数を vocab id 空間で集計 (8,200 posts → 中心タグ
  3,596・entries 1,570,314)。`cache/danbooru_posts.jsonl.cooc.npz` に COO 形式 (int32 配列・
  pickle なし) でキャッシュし再利用。**新規ペアファイルは作らない** (D2/D4 hard 混合とは別物)。
- soft target はサンプル単位で**1 度だけ事前計算**しキャッシュ (4,500 サンプル ~208s)。teacher は
  決定的なので epoch 不変 → 毎 epoch 再計算しない (数値不変・高速化のみ)。
- **採用ハイパラ (代表)**: alpha=0.2 / T=2.0 / main_mass=0.92 / cooc_temp=2.0 / topn=24 /
  dropout=0.0 / weight_decay=0.02。残余を入れすぎると正準順序学習が崩れるため main_mass は
  高め・alpha は控えめに置いた (探索の結論)。

### 11.3 A/B 評価 (固定 val 500・#1 6ep=2.41/0.777 基準)

代表 3 config を 6ep cosine (#1 採用 epoch) で比較 (XL は 10ep も併記し反転を観察)。

| 指標 | #1 6ep (本線) | D5 重 (a0.5/mm0.85/dp0.1/wd0.05) 10ep | D5-L (a0.3/mm0.9/dp0.05/wd0.02) 6ep | D5-XL (a0.2/mm0.92/dp0.0/wd0.02) 6ep | D5-XL 10ep |
|---|---|---|---|---|---|
| 最終 val_loss | 2.411 | 3.113 | 2.730 | 2.606 | 3.145 |
| **最良 val_loss (底)** | **2.382 (ep4)** | 3.081 (ep6) | 2.730 (ep5) | **2.579 (ep4)** | 2.602 (ep4) |
| 最終 top10 recall | **0.7767** | 0.6970 | 0.7302 | 0.7515 | 0.6673 |
| **最良 recall** | **0.7767** | 0.7022 | 0.7309 | **0.7584 (ep4)** | 0.7606 (ep4) |
| val_loss 反転(底) epoch | ep4 | ep6 | **ep5+ (未反転)** | ep4 | ep4 |
| 最終 train-val gap | 1.091 | 0.625 | **-0.210** | 0.269 | 2.235 |

生成多様性 (val prompt greedy・本体 tag・§10.3 同等):

| | #1 6ep | D5-L 6ep | D5-XL 6ep | D5-XL 10ep |
|---|---|---|---|---|
| コーパス unique tag 数 | 386 | 231 | 273 | **352** |
| 平均 per-seq unique 率 | 0.945 | 0.924 | 0.933 | 0.944 |
| 正規化エントロピー | 0.839 | **0.864** | **0.869** | 0.841 |

### 11.4 判定 — 過学習・多様性は改善するが recall は #1 に届かず (採用しない)

- **過学習 (本評価の主軸) は改善**: soft-label KL は **train-val gap を一貫して縮める**
  (#1 1.09 → D5-XL 0.27 / D5-L は **-0.21 = 過学習消失**)。重め config は **反転を ep4→ep6 へ
  後退**させる。D5 は §10.4 が掲げた「過学習抑制」を**実際に達成**する (D2/D4 が両軸悪化させたのと対照的)。
- **だが top10 recall は #1 (0.777) を超えない** (最良 D5-XL 0.758)。soft target に合わせる
  学習は one-hot val 上の hard-CE / recall を構造的に押し上げ/押し下げる: 軽くすれば recall は
  #1 に近づくが反転は ep4 に戻り、重くすれば反転は遅れるが recall が落ちる。**recall と
  反転後退はトレードオフ**で、両取りはできなかった。
- **多様性は中立〜微改善**: 正規化エントロピーは僅かに上 (0.839→0.86-0.87)・unique tag は
  config 次第 (重め長め 10ep で 352)。D4 ほどの大幅 unique 増は出ない (案A は target 軟化
  であって入力多様化ではないため)。
- **方針**: **#1 (6ep synthetic hard CE) を本線維持**・D5 (soft-label KL) は**採用しない**。
  ただし D2/D4 と違い D5 は過学習・gap・反転を**明確に改善する正の機構**で、データを増やせず
  汎化を底上げしたい将来局面 (例: より小規模 split・early-stopping 前提運用) で**再利用価値がある**。
  残路線 ③ (蒸留分布 val 追加) は本評価では固定 val 500 を不変に保つ制約から導入せず
  (案A は新ペアを作らないため「蒸留分布 val」は corpus 共起そのもので、固定 val とは別軸)。
- **負の結果ではなく「採用見送りの正の機構」**: 過学習を消せる代わりに recall を 0.02 落とす
  という性質が明確に切り分けられた。

### 11.5 dropout の golden 非回帰 (機構レベル実証)

- `BitNetDense(dropout=r)` は eval (`model.eval()`) で `nn.Dropout` が恒等になり、**dropout 率
  に依らず eval forward が bitwise 一致** (dropout 0.0 vs 0.3 の eval forward max abs diff = 0.0
  を実測)。train モードでのみ mask が効く。golden dump は常に eval なので #4/#6/A3 golden 非回帰。
- 実装: attention 出力・FFN 出力 (残差加算前) + embed 直後の residual dropout。dropout=0.0 なら
  train でも恒等で従来挙動と完全一致。

### 11.6 検証 (ルール#4・Python のため C++ meson test 対象外)

- ① `--distill-kl --smoke`: 疎通・本番非破壊 (smoke 別名 `bitnet_dense_kl_smoke*` へ出力)。
- ② **alpha=0 非回帰**: scratch data-dir で plain hardCE と `--distill-kl --kl-alpha 0` を
  同 seed 3ep 比較し、**train_loss/val_loss/recall が全 epoch bitwise 一致**を確認
  (8.3855/7.9328・7.6663/7.3642・7.1030/6.8621)。
- ③ A/B 数値 (§11.3) + dropout eval 不変 (§11.5)。

### 11.7 再現手順

```sh
# 採用代表 (D5-XL・10ep・本番別名へ出力)
py -3.12 scripts/train_bitnet.py --distill-kl --kl-alpha 0.2 --kl-temp 2.0 \
    --cooc-main-mass 0.92 --cooc-temp 2.0 --cooc-topn 24 \
    --dropout 0.0 --weight-decay 0.02 --epochs 10 --batch-size 32 --lr 3e-4
# alpha=0 非回帰チェック (scratch dir 推奨・plain と全 epoch 一致するはず)
py -3.12 scripts/train_bitnet.py --data-dir <scratch> --epochs 3                 # plain
py -3.12 scripts/train_bitnet.py --data-dir <scratch> --distill-kl --kl-alpha 0 --epochs 3
# smoke (疎通・本番を上書きしない)
py -3.12 scripts/train_bitnet.py --distill-kl --smoke
```

## 12. D6 — 外部教師 (TIPO-200M) soft-label KL 蒸留 (案c)

`scripts/train_bitnet.py --distill-ext` + `scripts/dollma_d6_teacher_cache.py`。D5 (案A 共起
teacher・§11) は過学習を抑えたが top10 recall を底上げしなかった (soft 追従と one-hot val
recall のトレードオフ・§11.4)。D6 は **真の外部知識転移** を狙い、外部教師 **TIPO-200M**
(KBlueLeaf/TIPO-200M・apache-2.0) が学習した「条件付きタグ集合」を自作 4999 vocab の疎 soft
target に注入する (案b-tagset)。KL plumbing (§11.1) は無改修流用・**新規ペアは作らない**・val は
#1 と同一 500 で**不変**。出力は**別名** `bitnet_dense_d6{,_fp32}.safetensors` /
`train_stats_d6.json` (#1/#4/#6/distill/identity/kl の重み・golden は無改変)。seed 20260620・
FP32・GTX1080Ti。

### 12.1 案b-tagset 教師 (絶対制約 — logit 直写像は却下)

TIPO の vocab32013 は SentencePiece BPE で「1 タグ = 複数 subword」。next-subword logits を
自作タグ単位 4999 分布へ直接写像してはいけない (案b-logit は却下)。必ず TIPO に**タグ補完を
生成**させ、出力**タグ文字列**を自作 vocab に写像する (`dollma_d6_teacher_cache.py`):

1. TIPO 出力をカンマで split (空白では割らない = `long hair` を壊さない)。
2. 各タグ片を `train_bitnet.Tokenizer.normalize` で正規化 (import 再利用・二重実装禁止)。
3. `data/bitnet/vocab.json` に**完全一致**引き。
4. vocab 外タグは drop し in-vocab 質量で**再正規化** (alias 表は作らない)。

teacher I/F は D5 `CoOccurrenceTeacher` と同一 `soft_target(prefix_ids, gold_id)`: gold に
`main_mass`、残余を「TIPO が予測した次タグ候補」上位 `topn` へ温度付きで配る (共起カウントの
代わりに **TIPO 生成由来のタグ頻度**を使う)。

- **コスト圧縮 (per-sample 粒度)**: generate は ~1.15s/sample (FP32 batched N=8)。per-position
  個別生成は破綻するので、1 サンプルにつき seed タグ群 (自然文 greedy タグ + 先頭 target タグ)
  を渡して N=8 回 generate → **サンプル単位のタグ頻度分布**を作り、そのサンプルの全 target
  位置で共有する。生成実測: train 4,500 件 5,338s (~1.5h)・val 500 件 574s (overnight 余裕)。
- **訓練時に TIPO 本体はロードしない**: 事前生成した position 軸付き COO npz
  (`cache/d6_teacher_soft.{train,val}.npz`) を `ExternalTagTeacher` が**順送り**で読むだけ
  (各 sample の soft 位置数を `bind(dataset)` で先に把握し、`precompute_soft_targets` の決定的
  反復順序に同期して COO 分布を返す)。teacher は決定的 → epoch 不変。
- **教師品質 (`cache/d6_teacher_stats.json`)**: OOV (vocab 外タグ) 保持率 **train 0.791 / val
  0.790** (in-vocab 質量で再正規化後)・平均エントロピー **0.85 nats**・val soft 位置 7,754。

### 12.2 A/B 評価 (固定 val 500・#1 6ep=2.41/0.777 基準・seed 20260620)

teacher 共通: main_mass=0.85 / 生成温度=2.0 / topn=32。student 側 `--kl-alpha` / `--kl-temp` を
掃引 (6ep cosine・#1 採用 epoch)。

| 指標 | #1 6ep (本線) | α=0.5 | α=0.35 | α=0.2 | α=0.1 T=2.0 | α=0.05 | **α=0.1 T=1.5** |
|---|---|---|---|---|---|---|---|
| 最終 val_loss | 2.411 | 2.690 | 2.569 | 2.473 | 2.421 | 2.408 | 2.413 |
| **最良 val_loss (底)** | 2.382 | 2.676 | 2.550 | 2.450 | 2.400 | **2.386** | 2.388 |
| 最終 top10 recall | 0.7767 | 0.7329 | 0.7550 | 0.7698 | 0.7790 | 0.7773 | 0.7797 |
| **最良 recall** | 0.7767 | 0.7349 | 0.7573 | 0.7716 | 0.7800 | **0.7811** | **0.7818** |
| val_loss 反転(底) epoch | ep4 | ep4 | ep4 | ep4 | ep4 | ep4 | ep4 |
| 最終 train-val gap | 1.091 | −2.476 | −1.566 | −0.559 | **0.183** | 0.599 | 0.721 |

(α=0.1 T=2.0 = 代表 `train_stats_d6.json`。生成多様性 α=0.1: unique tag 310・per-seq unique
0.947・norm entropy 0.869。)

> ⚠️ **この表の「#1 = 0.7767」は ≒ final epoch 測定**。seed sweep (§12.5) は epoch 最大 recall で
> 両 config を統一して測り、原値 seed の #1 best は **0.7787** (D6 0.7800 との差は +0.0013)。
> ここで見える「#1 を +0.004–0.005 上回る」は #1 を低く取った比較の産物で、再現しない (§12.3)。

### 12.3 判定 — recall 上振れは seed ノイズと確定・不採用 (#1 本線維持)

単一 seed の §12.2 では案C が #1 を僅かに上回って見えたが、**seed 頑健性 sweep (§12.5) で
recall 改善は再現せず**、D5 と同じ「過学習抑制はするが recall は上げない」性質に帰着した。

- **測定方法論の補正 (重要)**: §12.2 の「#1 = 0.7767」は別測定 (≒ final epoch) で、sweep の
  **epoch 最大 recall で両 config を統一して測る**と原値 seed の #1 best は **0.7787**。D6 0.7800
  との差は **+0.0013** (§12.2 で見えた +0.004–0.005 は #1 を低く取った比較の産物だった)。
- **recall 改善は seed ノイズ**: 4 seed (20260620/1/42/7) の delta = D6 − #1 は
  **+0.0013 / −0.0027 / +0.0015 / −0.0031** (符号が seed で反転・2 正 2 負)、平均 **−0.0008**。
  最大 |delta| 0.0031 は **#1 単独の seed 分散** (sd 0.0020・range [0.7764, 0.7819] = 幅 0.0055)
  に完全に埋もれる。**外部 TIPO soft-label KL に再現する recall 利得は無い**。
- **再現する D6 効果は過学習抑制のみ (recall 非寄与)**: final train-val gap を全 4 seed で
  #1 ≈ 1.06–1.09 → D6 ≈ 0.18–0.20 (~5.5x 縮小) と一貫して潰す。これは D5 と同じ正則化機構で、
  one-hot val recall には転化しない (soft 追従 vs val recall のトレードオフ・§11.4 と整合)。
- **α を上げると D5 同様 over-soft で劣化** (§12.2): gap は α とともに単調に負へ・recall は
  単調低下 (α=0.5 で 0.735)。
- **方針**: **#1 (6ep synthetic hard CE) を本線維持**・**D6 (案c) は採用しない**。D2/D4 (負) と
  違い D6/D5 は過学習を確実に潰す**正則化ノブ**として、データを増やせず汎化を底上げしたい
  局面 (小規模 split・early-stopping 前提) で再利用価値がある。本番 `bitnet_dense{,_fp32}`・
  golden は無改変 (sweep は scratch `_seedsweep/` で実行・本番非破壊)。

### 12.4 非回帰・検証 (ルール#4・Python のため C++ meson test 対象外)

- `--distill-ext --kl-alpha 0` は plain hardCE と数値一致 (§11.1 と同じ KL plumbing・α=0 で KL 項消失)。
- smoke (`--distill-ext --smoke`): 疎通・本番非破壊 (別名 `bitnet_dense_d6_smoke*`)。
- `test_dollma_d6_teacher_cache.py`: 案b-tagset 写像 (normalize → vocab 完全一致 → OOV drop +
  再正規化) と `ExternalTagTeacher` の順送り (`bind` の呼数カウント = `compute_sample_soft` の
  反復・soft 位置 0 sample のスキップ) を検証。

### 12.5 seed 頑健性 sweep (実施済・判定の根拠)

§12.2 の「α=0.05–0.1 で #1 を +0.005 上回る」が seed 横断で再現するかを検証。データ
(`pairs.*.jsonl`) を固定し**訓練 seed のみ**振り (`--seed`)、#1 (plain) と D6 (α=0.1 T=2.0 =
両立 sweet spot) を同一 seed 集合で対比。scratch data-dir `data/bitnet/_seedsweep/` (本番
pairs/vocab と同一・本番重み/golden 非破壊・teacher npz は seed 非依存で再利用)。epoch 最大
recall を両 config 統一で測定 (どちらも ep4 が最良)。

| seed | #1 best recall | D6 best recall | delta = D6 − #1 |
|---|---|---|---|
| 20260620 (原値) | 0.7787 | 0.7800 | +0.0013 |
| 1 | 0.7797 | 0.7771 | −0.0027 |
| 42 | 0.7764 | 0.7778 | +0.0015 |
| 7 | 0.7819 | 0.7788 | −0.0031 |

delta 平均 **−0.0008**・符号 2 正 2 負・最大 |delta| 0.0031。#1 単独の seed 分散 (sd 0.0020・
range 0.0055) が delta を完全に覆う → **recall 改善は seed ノイズ** (§12.3 判定)。一方 gap は
全 seed で #1 ≈ 1.06–1.09 → D6 ≈ 0.18–0.20 と一貫縮小 (過学習抑制は再現する正則化効果)。原値
seed は既存記録 (#1 ep4 0.7787 / D6 best 0.7800) を再現 = 再現性破れ無し。コード変更無し。

### 12.6 再現手順

```sh
# 1) 教師 soft npz 生成 (TIPO 本体を 1 度だけロード・overnight ~1.6h)
py -3.12 scripts/dollma_d6_teacher_cache.py            # train/val 両 npz + stats
py -3.12 scripts/dollma_d6_teacher_cache.py --probe-only  # teacher-alone recall A/B (訓練しない)
# 2) D6 訓練 (両立 sweet spot・本番別名へ出力)
py -3.12 scripts/train_bitnet.py --distill-ext --kl-alpha 0.1 --kl-temp 2.0 \
    --epochs 6 --batch-size 32 --lr 3e-4
# smoke (疎通・本番を上書きしない)
py -3.12 scripts/train_bitnet.py --distill-ext --smoke
```

## 13. 施策 C — 評価作り直し (diverse-val + 生成 set-metrics)

蒸留 4 路線 (D2/D4/D5/D6) は **いずれも従来 proxy = 固定 val 500・テンプレ 3 種・
teacher-forcing top10 recall** で #1 (0.777) を超えられず不採用とした (§10–12)。だが
この proxy は「3 テンプレに合うか」を測るだけで、実ユーザーの自由文への汎化を測っていない
(roadmap「見落としやすい罠」)。施策 C は **物差しそのものを実運用寄りに作り直し**、その新指標で
4 路線を採点し直す。`scripts/train_bitnet.py --eval-only` + `scripts/dollma_make_eval_diverse.py`
+ `scripts/dollma_c_seedsweep{,_analyze}.py`。**本番重み/golden/既存 val は無改変・加算のみ**。

### 13.1 構成 (C-1〜C-4)

- **C-1 diverse-val 構築** (`dollma_make_eval_diverse.py`, dataset-spec §14): テンプレ 3 種の
  偏りを排した多様な自然文 val。**tags-stay-real** (gold = 実 danbooru タグ固定・生成文から
  タグを推測させない不変方針) を厳守し、散文のみを多様化する 3 段 (段a gold→プロンプト出力 /
  段b main Claude が散文著述 / 段c 取り込み・検証・凍結)。Pool A = `pairs.val.jsonl` 由来
  (in-distribution の gold)・Pool B = train∪val 非交差の未使用 post (リーク 0)。各 post×3
  variant で `pairs.eval_diverse_a.jsonl` / `_b.jsonl` 各 **1,500 行**に凍結 (再現性アンカー)。
- **C-2 生成ベース set-metrics** (`eval_generation_setmetrics`): teacher-forcing recall を
  **greedy 自己回帰生成タグ集合 vs gold タグ集合の set-overlap** へ置換 (= 実運用 = 自由文→タグ集合
  に近い)。macro/micro の precision / recall / **F1** / Jaccard / recall@k。#1/diversity/retention
  と greedy・encode・Tokenizer を完全共有 (別スクリプト化せず単一ソース維持)。従来
  `eval_loss_and_recall` は**非回帰アンカーとして残す** (#1 既知値 0.777 再現経路)。
- **C-3 eval-only ハーネス** (`--eval-only --weights <export.safetensors>`): 訓練せず採点のみ。
  legacy TF recall + 生成 set-metrics (pairs.val / diverse_a / _b・**存在ガード付き**) +
  diversity + identity retention を provenance (重み/val sha256・seed・git rev) 付きで
  `eval_report_<name>.json` に書く。schema `dollama/eval_report/C-1`。`--dump-persample` で
  paired bootstrap/t 用の per-sample F1/Jaccard/precision/recall を `eval_persample_<name>.npz`
  (rows と同順・NaN=skip/未定義) に出力。
- **C-4 seed 頑健性 sweep** (`dollma_c_seedsweep.py` → `_analyze.py`): D6 の recall 上振れが
  seed ノイズだった (§12.5) のと同じ手続きで、新指標上の **D5 vs #1** が seed 頑健かを検定。
  4 seed (20260620/20260621/42/7) で各 arm 6ep FP32 訓練 → diverse で per-sample 採点 →
  **同 seed paired delta (D5−#1)・paired bootstrap 95%CI・paired t**。scratch
  `data/bitnet/_seedsweep/`・本番非破壊。判定軸 (a) 全 seed 符号一貫 (b) |delta 平均| が #1 自身の
  seed 分散帯を超える (c) 各 seed CI が 0 を除外。

### 13.2 本番重み採点 (eval-only・seed 20260620・CPU)

`bitnet_dense_fp32` (#1 6ep) / `bitnet_dense_kl_fp32` (D5 10ep) / `bitnet_dense_d6_fp32`
(D6 6ep) を同一ハーネスで採点。**legacy 列が旧 proxy・diverse 列が新 proxy**。

| 指標 | #1 (本線) | D5 (KL 10ep) | D6 (TIPO 6ep) |
|---|---|---|---|
| legacy TF recall@10 (テンプレ val 500) | **0.7767** | 0.6673 | 0.7790 |
| pairs.val 生成 macro F1 (in-dist) | 0.4715 | 0.4683 | 0.4703 |
| diverse_a 生成 macro F1 | 0.1800 | **0.2024** | 0.1842 |
| diverse_a 生成 macro precision | 0.2515 | **0.3762** | 0.3024 |
| diverse_b 生成 macro F1 | 0.1921 | **0.2192** | 0.1976 |
| diverse_b 生成 macro precision | 0.2644 | **0.4092** | 0.3249 |

**proxy が逆順に並ぶ**: 旧 recall では D5 が**最下位** (0.667) だが、新 diverse F1 では D5 が
**最上位**。in-distribution (pairs.val) では 3 者ほぼ同点 (~0.47) で、差は**テンプレ外の自由文**
でのみ開く。D5 は precision が突出 (0.38–0.41 vs #1 0.25–0.26) — soft-label KL で**短く確信の高い
タグ集合**を出すようになり、これがテンプレ外入力で効く。

### 13.3 C-4 seed sweep (4 seed・6ep paired・diverse)

per-sample paired delta = D5 − #1 (同 seed・両 arm 非 NaN 位置・n_pair≈1500/seed):

| set / metric | per-seed delta | across-seed 平均±sd | 全 seed 符号 | 各 seed CI が 0 を除外 |
|---|---|---|---|---|
| diverse_a / F1 | +0.0117 / +0.0091 / +0.0085 / +0.0050 | **+0.0086 ± 0.0028** | **+ 一貫** | **4/4 除外** |
| diverse_a / Jaccard | +0.0078 / +0.0054 / +0.0059 / +0.0034 | **+0.0057 ± 0.0018** | **+ 一貫** | **4/4 除外** |
| diverse_b / F1 | +0.0136 / +0.0135 / +0.0108 / +0.0097 | **+0.0119 ± 0.0019** | **+ 一貫** | **4/4 除外** |
| diverse_b / Jaccard | +0.0084 / +0.0087 / +0.0077 / +0.0065 | **+0.0078 ± 0.0010** | **+ 一貫** | **4/4 除外** |

paired t は全 set/metric/seed で p < 8e-3 (大半 p < 1e-6)。

### 13.4 判定 — 物差しを変えると D5 の符号が反転する (= C の本質的成果)

- **D6 (§12.5) との決定的対比**: D6 の recall delta は seed 間で符号反転 (2 正 2 負・平均 −0.0008)
  = **seed ノイズ**だった。D5 の diverse F1/Jaccard delta は **全 4 seed で正・各 seed の paired CI が
  0 を除外** = **再現する実効果** (seed ノイズではない)。判定 (a)+(c) は満たす。
- **strict 判定 (a∧b∧c) は "弱い"**: 判定 (b) のみ不成立 — |delta 平均| (~0.009–0.012 F1) が #1 自身の
  diverse seed 分散帯 (sd 0.022–0.022, range ~0.048) 以下。ただしこの band は **seed 42 の外れ値**
  (#1 diverse_a F1 が 0.139 へ低下) で膨張しており、paired 比較 (seed 効果を相殺済み) が有意な以上、
  (b) は過度に保守的な尺度。**結論: D5 の diverse 優位は「小幅だが統計的に頑健」**。
- **C の payoff**: **旧 proxy (template TF recall) は D5 を最下位に置き、新 proxy (diverse 生成 F1) は
  最上位に置く — 物差しの変更が D5 判定の符号を反転させた。** これは施策 C が掲げた仮説
  「テンプレ recall proxy が D5 の実力を隠していた可能性が高い」(roadmap) を実データで裏付ける。
  D2 の「過学習悪化に見えた」件も同根 (proxy 由来の見かけ) の可能性が高い。
- **採用判断は別件・本線は据え置き**: 本番重みは **#1 (6ep) を維持** (重み無改変)。C の成果物は
  「**より良い凍結オフライン物差し**」(diverse-val + 生成 set-metrics + eval-only + seed sweep
  方法論) であり、今後の施策 B/A/D/F はこの新指標で測る。**recall@10 (テンプレ val) を主要数値から
  退役させ、diverse 生成 set-F1 を新たな主要オフライン proxy に据える**ことが C の確定事項。
  D5 を本線へ昇格させるかは、この新物差しの下で改めて判断する (絶対値はなお低く ~0.18–0.22・
  edge は小さいため、A 実ペア増 / D 容量増と束ねて再評価するのが妥当)。

### 13.5 検証・非回帰 (ルール#4・Python のため C++ meson test 対象外)

- `scripts/test_dollma_eval_diverse.py` (C-1・torch 非依存部): EvalTokenizer の tb.Tokenizer
  バイト等価・段a スキーマ/リーク0/gold⊆vocab/Pool A バイト一致・段c 取り込み/検証/post_id 漏出弾き。**緑**。
- `scripts/test_dollma_eval_setmetrics.py` (C-2・**要 torch**・`py -3.12` で実走): 手置き gen/gold で
  空集合 0 除算ガード・recall@k 先頭 k 順序・macro/micro 定義・per-sample 配列が rows と同順 (NaN=skip)
  を検証。`train_bitnet` を import するため torch 必須 (torch 無しの素 python では import 段で停止
  = §11.6/§12.4 と同じく C++ meson test 対象外・py -3.12 で担保)。
- 本番重み/golden/既存 val・#1/D5/D6 の既存数値はすべて無改変 (eval-only は読むだけ・seed sweep は
  scratch `_seedsweep/`)。`--dump-persample` 既定 off で `eval_generation_setmetrics` の返り値・挙動は従来一致。

### 13.6 再現手順

```sh
# C-1: diverse-val 構築 (段a → main Claude が texts → 段c で凍結。詳細 dataset-spec §14)
py -3.12 scripts/dollma_make_eval_diverse.py --emit-prompts --n-variants 3 --seed 20260620
#   (段b: main Claude が data/bitnet/eval_diverse_texts.jsonl を著述)
py -3.12 scripts/dollma_make_eval_diverse.py --ingest
# C-2/C-3: 本番重みを新指標で採点 (provenance 付き eval_report_<name>.json)
py -3.12 scripts/train_bitnet.py --eval-only --weights data/bitnet/bitnet_dense_fp32.safetensors
py -3.12 scripts/train_bitnet.py --eval-only --weights data/bitnet/bitnet_dense_kl_fp32.safetensors
py -3.12 scripts/train_bitnet.py --eval-only --weights data/bitnet/bitnet_dense_d6_fp32.safetensors
# C-4: seed 頑健性 sweep (4 seed・本番非破壊) → 集計・判定
py -3.12 scripts/dollma_c_seedsweep.py
py -3.12 scripts/dollma_c_seedsweep_analyze.py
```

## 14. 施策 B — 入力多様化 (Claude 著述 Replace パイロット)

施策 C (§13) が据えた**新物差し = diverse-val 上の生成 set-F1** の上で、施策 B
(roadmap タグ生成 LM 学習強化プログラム) の最初のパイロットを実施する。B の狙いは
**タグ集合を実 danbooru のまま固定 (tags-stay-real) し、入力の自然文だけを多様化**して
テンプレ 3 種 (§3 / dataset-spec §3.3) の偏りを解消し、実ユーザーの自由文への汎化を上げること。
C で物差しを作り直していなければ、D2 と同じく「テンプレ val 上では悪化に見える」罠に
落ちる (roadmap 依存連鎖: B は C をゲートに持つ)。`scripts/dollma_make_diverse_train.py`
+ `scripts/train_bitnet.py --train-file` + `scripts/dollma_b_seedsweep{,_analyze}.py`。
**本番重み/golden/既存 val は無改変・別名出力のみ**。

### 14.1 B-0 — 既存 D2 蒸留重み (`bitnet_dense_distill`) を diverse-val で再採点

施策 C の物差しは過去の重みにも遡及できる。**D2** (Qwen2-1.5B 蒸留 hard CE 混合・旧 recall
で「過学習悪化」として §10 で不採用) を eval-only で diverse-val 採点した:

| 指標 | D2 (`bitnet_dense_distill`) | #1 (本線) |
|---|---|---|
| legacy TF recall@10 (テンプレ val 500) | 0.7402 | 0.7767 |
| pairs.val 生成 macro F1 (in-dist) | 0.4387 | 0.4715 |
| diverse_a 生成 macro F1 | **0.2701** | 0.1800 |
| diverse_b 生成 macro F1 | **0.3134** | 0.1921 |

- **旧 recall が D2 の実力を隠していた**: D2 は legacy recall (0.7402 < 0.7767) でも in-dist
  F1 (0.4387 < 0.4715) でも #1 に劣るが、**テンプレ外 diverse 生成 F1 では #1 を大幅に上回る**
  (a +0.090・b +0.121)。施策 C の中核仮説 (§13.4・roadmap「proxy 由来の見かけ」) を
  D2 の遡及採点で再確認 — D2 の「過学習悪化」は proxy 由来だった。
- **著者分布交絡の否定 (B-0 が偶然の対照群)**: D2 入力は **Qwen2 著述**・B-pilot 入力 (§14.2) は
  **Claude 著述**。両者が同等の diverse 改善を示す以上、「Claude train が Claude test (diverse-val
  も Claude 著述) に似て上がっただけ」という著者分布交絡では説明できない。多様化そのものの効果。
- 出力: `data/bitnet/eval_report_distill.json` (本番無改変・読むだけ)。

### 14.2 B-1 構成 (Claude 著述 Replace 500 パイロット)

- **生成器** = このセッションの **main Claude が会話内で散文著述** (外部 API / Qwen2 / ネット
  不使用)。**混合方式 = Replace** (総件数 4,500 を維持: 4,500 中 500 を著述文へ置換・残 4,000 は
  synthetic テンプレ)。**タグは実 danbooru のまま不変** (tags-stay-real)。**規模 500 件パイロット**。
- **データ生成** (`scripts/dollma_make_diverse_train.py`, dataset-spec §15): 段a `--emit-prompts`
  で seed 20260620 決定的に 500 件抽出 → 著述プロンプト出力 (3 assert: 抽出⊆train / val 非交差 /
  **diverse-val 非交差** / gold⊆vocab)。段c `--ingest` で著述 texts と突合 (tags バイト不変・
  post_id 非漏出・件数 4,500・synthetic 残 4,000・重複 0)。出力
  `data/bitnet/pairs.train.diverse_b.jsonl` (著述 500 = `source:"llm_distill"` + `meta.gen:"claude"`
  + `tmpl:-1` / synthetic 4,000) + `stats.diverse_b.json`。`test_dollma_make_diverse_train.py` 6/6 緑。
- **訓練**: `train_bitnet.py` に **`--train-file` オプションを追加** (既定 `pairs.train.jsonl` で
  bitwise 非回帰確認済 = 既存 #1 経路を一切変えない)。`bitnet_dense_diverse_b{,_fp32}.safetensors`
  / `train_stats_diverse_b.json` を**別名出力**。`llm_distill` 行は build_sequence で synthetic
  と同じ `1-<sep>` 経路 (D2 先例)。

### 14.3 B-1-c 採点 (seed 20260620・6ep・diverse-val 生成 macro)

| 指標 | #1 | B-pilot (Replace 500) | 差 |
|---|---|---|---|
| diverse_a F1 | 0.1800 | **0.2675** | +0.0875 |
| diverse_a precision | 0.2515 | 0.2819 | +0.030 |
| diverse_b F1 | 0.1921 | **0.3039** | +0.1118 |
| diverse_b precision | 0.2644 | 0.3160 | +0.052 |
| pairs.val (in-dist) F1 | 0.4715 | 0.4625 | −0.009 (退行なし) |
| legacy recall@10 | 0.7767 | 0.774 | ≈同値 (タグ集合同一) |

- train_loss 1.439 / val_loss 2.432 (3.28→2.41 単調収束・#1 と同オーダー)。訓練 47.8s・
  seed 20260620・FP32・GTX1080Ti。
- in-dist (pairs.val) は −0.009 で**退行なし**・legacy recall は≈同値 (タグ集合が同一のため)。
  改善は**テンプレ外の自由文でのみ**開く — §13.4 の D5 と同じ構図だが、edge が桁違いに大きい。

### 14.4 B-1-d seed 頑健性 sweep (4 seed・6ep paired・diverse)

`scripts/dollma_b_seedsweep{,_analyze}.py` で §13.3 (C-4) と同手続き。4 seed
(20260620 / 20260621 / 42 / 7)・各 arm 6ep FP32 訓練 → diverse で per-sample paired 採点
(n≈1500/seed・scratch `data/bitnet/_seedsweep_b/`・本番非汚染確認済)。

per-seed F1 (base #1 / B-pilot):

| seed | diverse_a (#1 / B) | diverse_b (#1 / B) |
|---|---|---|
| 20260620 | 0.1800 / 0.2675 | 0.1921 / 0.3039 |
| 20260621 | 0.1814 / 0.2698 | 0.1908 / 0.3101 |
| 42 | 0.1390 / 0.2622 | 0.1494 / 0.3075 (seed42 は #1 が落ちる外れ値) |
| 7 | 0.1870 / 0.2707 | 0.1980 / 0.3140 |

across-seed delta (B − #1) と判定軸:

| set / metric | delta 平均±sd | (a) 符号一貫 | (b) > #1 分散帯 sd | (c) 各 seed CI 0 除外 |
|---|---|---|---|---|
| diverse_a / F1 | **+0.0957 ± 0.0184** | 4/4 正 | **True** (band sd 0.0221・約 4–6x) | **4/4** |
| diverse_a / Jaccard | **+0.0647 ± 0.0102** | 4/4 正 | **True** (0.0124) | **4/4** |
| diverse_b / F1 | **+0.1263 ± 0.0214** | 4/4 正 | **True** (0.0223) | **4/4** |
| diverse_b / Jaccard | **+0.0877 ± 0.0125** | 4/4 正 | **True** (0.0125) | **4/4** |

paired t は全 16 セルで p < 8e-142。precision も全 seed で改善。

### 14.5 判定 — B-pilot の edge は大幅かつ全判定軸で頑健 (= 本物)

- **D5/D6 との決定的対比**: 施策 C の sweep で D5 (§13.4) は (a)+(c) 成立だが (b) 不成立
  (delta +0.009〜+0.012 = #1 seed 分散帯以下)、D6 (§12.5) は符号反転 seed ノイズだった。
  **B-pilot は判定 (a)(b)(c) すべて成立** (delta +0.06〜+0.13・分散帯の 4–6 倍) =
  **小幅でなく桁違いに大きく、かつ頑健な実効果**。D5 で消化しきれなかった「多様化の本命」を
  入力多様化が実現した。
- **C の物差しなしには見えなかった**: 旧 proxy なら D2 同様「テンプレ val 上は悪化〜同値」に
  しか見えず却下されたはず。**施策 C (§13) で作った diverse-val + 生成 set-metrics 物差しの上で
  測ったからこそ B の効果が可視化された** (roadmap 依存連鎖 C→B の実証)。
- **未決・本線昇格は別途決裁**: 本番重みは **#1 据え置き・無改変** (B-pilot 重みは別名
  `bitnet_dense_diverse_b`)。本線昇格は project-leader / ユーザー決裁 (D5 と束ねた再評価方針)。
  絶対値はなお diverse F1 ~0.26–0.31 と低帯域 → **B 著述件数拡大 (500→数千) / A 実ペア増 /
  D 容量増と束ねて再評価**が妥当 (roadmap)。

### 14.6 検証・非回帰 (ルール#4・Python のため C++ meson test 対象外)

- `scripts/test_dollma_make_diverse_train.py` 6/6 緑: 段a 抽出決定性・3 assert (抽出⊆train /
  val 非交差 / diverse-val 非交差) / gold⊆vocab・段c tags バイト不変・post_id 非漏出・件数
  4,500・synthetic 残 4,000・重複 0。
- `--train-file` 既定値 (`pairs.train.jsonl`) で **#1 経路と bitwise 非回帰**を確認。本番重み/
  golden/既存 val・既存数値はすべて無改変 (sweep は scratch `_seedsweep_b/`・本番非汚染
  スナップショット diff で担保)。
- C++ meson test 対象外 (§13.5 と同じく訓練側 Python のため・`py -3.12` で担保)。

### 14.7 再現手順

```sh
# B-0: 既存 D2 蒸留重みを diverse-val で再採点 (本番無改変・読むだけ)
py -3.12 scripts/train_bitnet.py --eval-only --weights data/bitnet/bitnet_dense_distill_fp32.safetensors
# B-1 段a: seed 20260620 で 500 件抽出 → 著述プロンプト出力 (3 assert)
py -3.12 scripts/dollma_make_diverse_train.py --emit-prompts --n 500 --seed 20260620
#   (段b: main Claude が著述 texts を埋める)
# B-1 段c: 著述 texts を突合・凍結 (tags バイト不変・件数 4500)
py -3.12 scripts/dollma_make_diverse_train.py --ingest
# B-1-c 訓練 (別名重み出力・seed 20260620・6ep)
py -3.12 scripts/train_bitnet.py --train-file data/bitnet/pairs.train.diverse_b.jsonl --seed 20260620
# B-1-d seed 頑健性 sweep (4 seed・本番非破壊) → 集計・判定
py -3.12 scripts/dollma_b_seedsweep.py
py -3.12 scripts/dollma_b_seedsweep_analyze.py
```

### 14.8 B-2 件数拡大 (Claude 著述 Replace 500 → 2,000)

§14.5 で据えた「絶対値は低帯域 → 著述件数拡大 (500→数千) と束ねて再評価」を回収する。
**§14.1〜14.7 のパイロット (Replace 500) は記録として残し**、本節は同方式・同物差し
(diverse-val 生成 set-F1)・同 sweep 手続きで**著述件数だけを 2,000 に増やした**結果を追記する。
構成は **Replace のまま総件数 4,500 を維持** (著述 2,000 + synthetic 2,500)・**tags-stay-real**・
本番重み/golden/凍結 val は無改変。著述は既存 500 (part01–05) に新規 1,500 (part06–20) を
積み増した。出力 `data/bitnet/pairs.train.diverse_b2000.jsonl` + `stats.diverse_b2000.json`
(dataset-spec §15.6)・別名重み `bitnet_dense_diverse_b2000{,_fp32}.safetensors`。

#### 14.8.1 採点 (seed 20260620・6ep・diverse-val 生成 macro)

500版・#1 と並べて 500→2,000 の伸びを示す:

| 指標 | #1 | B-pilot (500) | **B-2 (2,000)** |
|---|---|---|---|
| diverse_a F1 | 0.1800 | 0.2675 | **0.3212** |
| diverse_a precision | 0.2515 | 0.2819 | **0.3362** |
| diverse_b F1 | 0.1921 | 0.3039 | **0.3670** |
| diverse_b precision | 0.2644 | 0.3160 | **0.3772** |
| pairs.val (in-dist) F1 | 0.4715 | 0.4625 | 0.4539 (−0.009 vs 500・退行なし) |
| legacy recall@10 | 0.7767 | 0.774 | 0.7702 (#1 0.777 と同帯) |

- params 32,976,896 一致・val_loss 単調収束 (ep4 底 2.456)・訓練 116.7s
  (seed 20260620・FP32・GTX1080Ti)。
- **out-of-template だけが伸びる**: in-dist (pairs.val) は 500版比 −0.009 で誤差内据え置き・
  legacy recall も同帯。**改善はテンプレ外の自由文でのみ開き、件数増でさらに拡大**した
  (diverse_a +0.054 / diverse_b +0.063 over 500版)。

#### 14.8.2 seed 頑健性 sweep (4 seed・6ep paired・diverse)

`scripts/dollma_b_seedsweep{,_analyze}.py` で §14.4 と同手続き。4 seed
(20260620 / 20260621 / 42 / 7)・各 arm 6ep FP32 → per-sample paired 採点 (n≈1,500/seed・
scratch `data/bitnet/_seedsweep_b2000/`・本番非汚染確認済)。

across-seed delta (B-2 − #1) と判定軸:

| set / metric | delta 平均±sd | (a) 符号一貫 | (b) > #1 分散帯 sd | (c) 各 seed CI 0 除外 |
|---|---|---|---|---|
| diverse_a / F1 | **+0.1472 ± 0.0102** | 4/4 正 | **True** (約 6.7–8x) | **4/4** |
| diverse_a / Jaccard | **+0.1047 ± 0.0044** | 4/4 正 | **True** | **4/4** |
| diverse_b / F1 | **+0.1788 ± 0.0029** | 4/4 正 | **True** | **4/4** |
| diverse_b / Jaccard | **+0.1314 ± 0.0034** | 4/4 正 | **True** | **4/4** |

paired t ≈ 35–47・全セルで p < 1e-269。

- **500版 sweep との対比** (§14.4: diverse_a +0.0957±0.0184 / diverse_b +0.1263±0.0214):
  件数 500→2,000 で **delta が ~1.4–1.5x 拡大**・かつ **seed sd は逆に縮小**
  (diverse_b F1 sd 0.0214 → 0.0029)。**効果が強まりつつ頑健性も増加**した。

#### 14.8.3 判定 — 入力多様化はスケール則 (件数増で単調に強まり頭打ちなし)

- **スケール則**: 入力多様化の効果は著述件数増で**単調に強まり頭打ちが見えない** —
  in-dist は誤差内据え置きで out-of-template (汎化方向) だけが伸びる。D5 (§13.4・小幅で
  (b) 不成立) / D6 (§12.5・符号反転 seed ノイズ) と桁違いに対照的。
  - **※ 訂正 (§14.9, B-3 で更新)**: 上記「頭打ちが見えない」は 500→2,000 の **2 点からの外挿**。
    §14.9 で 3 点目 (10,000) を取ったところ **~2,000 件で飽和**することが判明した。
    スケール則の表現は **§14.9 を正**とする (効果が seed 頑健に正である点・本線昇格決裁は不変)。
- **本線昇格 決裁済 (2026-06-24・ユーザー) = レシピ既定化を確定**: 今後の訓練 (A 実ペア増 /
  D 容量増 / F 品質ループ) は多様化入力 (tags-stay-real) を**既定レシピ**とする。B は A/D と直交
  (依存連鎖 C→{B,A}→D→F の並列枝) ゆえ、劣るレシピ (#1 系) の上に A/D が積まれる事故を防ぐ。
  **正典重み `bitnet_dense{,_fp32}.safetensors` と C++ 推論 golden の差し替えは A 実ペアと束ねる
  次の出荷リトレインで1回** (golden チャーンを集約)。当面 #1 重みは据え置き。`--train-file` の運用は
  A 出荷リトレインで多様化ファイル (`pairs.train.diverse_b2000.jsonl` 等) を既定指定する契約とし、
  コード default=None は A 時まで意図的に据え置く (今書き換えると別名出力分岐に落ち、bitwise
  非回帰アンカー・golden 据え置きの遅延条項を破るため)。遅延条項の詳細は roadmap `[B-merge-at-A]`。
- **絶対値はなお低帯域**: diverse_a ~0.32 / diverse_b ~0.37。これは本線の良し悪しではなく
  **A 実ペア増 / D 容量増で取りに行く残課題** (本線の比較では #1 がさらに低い)。
- 本番重みは当面 **#1 据え置き・無改変**・別名 `bitnet_dense_diverse_b2000` 出力・凍結アンカー無改変
  (正典差し替えは上記のとおり A 出荷リトレインで実施)。

#### 14.8.4 再現手順 (件数 2,000 版)

```sh
# 段a: seed 20260620 で 2,000 件抽出 → 著述プロンプト出力 (3 assert)
py -3.12 scripts/dollma_make_diverse_train.py --emit-prompts --n 2000 --seed 20260620
#   (段b: main Claude が著述 texts part06–20 を埋める)
# 段c: 著述 texts を突合・凍結 (tags バイト不変・件数 4500・著述 2000 / synthetic 2500)
py -3.12 scripts/dollma_make_diverse_train.py --ingest
# 訓練 (別名重み出力・seed 20260620・6ep)
py -3.12 scripts/train_bitnet.py --train-file data/bitnet/pairs.train.diverse_b2000.jsonl --seed 20260620
# seed 頑健性 sweep (4 seed・本番非破壊) → 集計・判定
py -3.12 scripts/dollma_b_seedsweep.py
py -3.12 scripts/dollma_b_seedsweep_analyze.py
```

### 14.9 B-3 件数拡大 (Claude 著述 Replace 2,000 → 10,000)・スケール則は ~2,000 で飽和

§14.8.3 で据えた「件数拡大と束ねて再評価」と「頭打ちが見えない (2 点外挿)」を回収する。
本節は同方式・同物差し (diverse-val 生成 set-F1)・同 sweep 手続きで**著述件数を 10,000 に増やした**
3 点目の結果を追記する。**結論: 効果は seed 頑健に健在だが、~2,000 件で飽和する** (2,000→10,000 の
5 倍増で追加利得ゼロ)。

**構成**: P=2,500 unique post × k=4 variant = 著述 10,000 本。Replace 後の総 train 12,000
(著述 10,000 + synthetic 2,000)・**tags-stay-real**・本番重み/golden/凍結 val 無改変。B-2 の 2,000 を
内包する**スーパーセット** (variant 0 = B-2 著述 part01–20 を再利用、新規 part21–36 に各 500)。
`scripts/dollma_make_diverse_train.py` を **k-per-post 一般化** (`--n-posts`/`--k-per-post`・
variant_idx/style_hint・k=1 で B-1/B-2 と bitwise 非回帰・test 14/14 緑)。出力
`data/bitnet/pairs.train.diverse_b10k.jsonl` + `stats.diverse_b10k.json` (dataset-spec §15.7)・
別名重み `bitnet_dense_diverse_b10k{,_fp32}.safetensors`。

#### 14.9.1 seed 頑健性 sweep (4 seed・6ep paired・diverse) と 500→2,000→10,000 スケール則

`scripts/dollma_b10k_seedsweep{,_analyze}.py` で §14.8.2 と同手続き・同 seed・同ハイパラ。
唯一の差分は B arm の train ファイルと出力先 (scratch `data/bitnet/_seedsweep_b10k/`)。凍結物差し
(pairs.eval_diverse_a/b) は b2000 sweep と **byte 一致**。per-sample paired delta = B10k − #1
(n≈1,500/seed)。**3 sweep (500/2,000/10,000) の生 npz を横並びで再集計**:

| set / metric | B500 Δ | B2000 Δ | **B10k Δ** | b 絶対値 500/2k/10k |
|---|---|---|---|---|
| diverse_a / F1 | +0.0957±0.0184 | +0.1472±0.0102 | **+0.1411±0.0206** | 0.268 / 0.319 / 0.313 |
| diverse_a / Jaccard | +0.0647±0.0102 | +0.1047±0.0044 | **+0.1024±0.0111** | 0.165 / 0.205 / 0.202 |
| diverse_b / F1 | +0.1263±0.0214 | +0.1788±0.0029 | **+0.1761±0.0199** | 0.309 / 0.361 / 0.359 |
| diverse_b / Jaccard | +0.0877±0.0125 | +0.1314±0.0034 | **+0.1326±0.0119** | 0.195 / 0.239 / 0.240 |

判定軸 (§14.4): **全 set/metric で (a) 4/4 正・(b) #1 分散帯 sd 超え・(c) 各 seed paired CI が 0 除外
= YES (seed 頑健・本物)**。per-seed paired t ≈ 28–46・全セル p < 1e-170。**効果自体は B-2 同様に健在**。

#### 14.9.2 判定 — スケール則は ~2,000 件で飽和 (頭打ち)

- **飽和**: 500→2,000 は大幅増 (diverse_a F1 +0.0957→+0.1472 / diverse_b +0.1263→+0.1788) だが、
  **2,000→10,000 は平坦** — delta は +0.1472→+0.1411 (a) / +0.1788→+0.1761 (b) と **seed 分散内で
  むしろ微減**、b 絶対値も 0.319→0.313 (a) / 0.361→0.359 (b) と頭打ち。**5 倍の著述 (2,000→10,000)
  で追加利得は実質ゼロ**。§14.8.3 の「単調・頭打ちなし」は 2 点外挿の誤りで、3 点目で飽和が見えた。
- **含意 (運用知見)**: 入力多様化**単体**での伸びしろは ~2,000 件で尽きる。残る低帯域
  (diverse_a ~0.31 / diverse_b ~0.36) は **B の件数ではなく A 実ペア増 / D 容量増**で取りに行く課題。
  **今後 B 著述を 2,000 超に積む価値は薄い** (本線レシピの多様化件数は ~2,000 で足りる・10,000 は不要)。
- **本線昇格決裁は不変** (§14.8.3・2026-06-24 ユーザー決裁): 多様化入力 (tags-stay-real) を既定レシピ化。
  B-3 はその件数下限を「~2,000 で飽和」と確定しただけで、決裁・正典差し替えの遅延条項 (A 出荷リトレインで
  1 回) は変えない。出荷時の既定多様化ファイルは **b2000 で足りる** (b10k を作る必要はない)。
- 本番重みは **#1 据え置き・無改変**・別名 `bitnet_dense_diverse_b10k` 出力・凍結アンカー無改変。

#### 14.9.3 再現手順 (件数 10,000 版) と PC ハング耐性

```sh
# 段a: seed 20260620 で 2,500 post × 4 variant = 10,000 著述プロンプト出力 (スーパーセット/3リーク assert)
py -3.12 scripts/dollma_make_diverse_train.py --emit-prompts --n-posts 2500 --k-per-post 4 --seed 20260620
#   (段b: main Claude / claude サブエージェントが新規 texts part21–36 を各 500 著述。
#    禁止: ジェネレータ機械生成/固定ラッパー/カンマ列挙/no_humans 矛盾)
# 段c: 著述 texts を突合・凍結 (tags バイト不変・総 12,000・著述 10,000 / synthetic 2,000)
py -3.12 scripts/dollma_make_diverse_train.py --ingest
# seed 頑健性 sweep (4 seed・本番非破壊・scratch _seedsweep_b10k/) → 集計・判定
py -3.12 scripts/dollma_b10k_seedsweep.py
py -3.12 scripts/dollma_b10k_seedsweep_analyze.py
```

- **ハング耐性**: sweep は各 seed-arm の完了を `_seedsweep_b10k/_results/eval_persample_{arm}_{seed}.npz`
  存在で判定し、ある arm は skip して冪等再開する。本 sweep は **2026-06-25 に PC ハングで中断** (seed
  20260621 の b-arm eval 中) したが、同コマンド再投入で完了済み arm を捨てずに続きから完走した。
  eval が ~895s/本 で律速・GTX1080Ti で全 8 arm ≈ 1.5h。

## 15. INT8 dense 推論 (量子化圧縮実験)

ternary (b1.58) は目的ではなく**圧縮の研究軸**という CLAUDE.md 方針に沿い、まず素直な
dense INT8 で 33M dense LM の量子化耐性を測った。ternary GEMM (#5) や GPU/INT4 への
拡張ではなく、CPU 純ホストで「重みのみ INT8 にしたら品質がどれだけ落ちるか」を切り分ける。

### 15.1 機構

- **重みのみ per-output-row 対称 INT8** (absmax/127)・**ロード時量子化**。FP32 重み
  (`bitnet_dense_fp32.safetensors`) を読み込んだ直後に射影 Linear **8 本のみ**量子化する
  (`src/models/bitnet.hpp` の `quantize_weight_int8_perrow()` / `Int8RowQuant`)。
- **embed / lm_head (tied) / RMSNorm / RoPE / attention / softmax は FP32 据え置き** —
  精度に効く層・自己回帰の数値安定に関わる層は量子化しない。
- 活性は既存の `quantize_activation_int8` を流用し、int8×int8 内積を **int64 蓄積** →
  `w_scale[o] · x_scale · acc` で復元。`src/infer/bitnet_int8.hpp` の `BitNetInt8Infer` が
  推論経路を持つ。dense FP32 経路 (`src/infer/bitnet.hpp` `BitNetDenseInfer`) は**無改変**。

### 15.2 実測 (GTX1080Ti/i7-10700 CPU 純ホスト・実重み + golden 流用・実走)

- **INT8 logits vs FP32 golden**: seq8 max_abs_err 0.0493 / corr **0.999964**、
  seq32 0.2126 / 0.999944、seq63 0.4608 / **0.999873**。ハードゲート corr≥0.99 を**3桁上回る**。
- **greedy 生成 vs gen_golden**: 全 5 ケース完全一致 (16/16・8/8・14/14・9/9・8/8) =
  トークン一致率 **1.0 (55/55)・EXACT 5/5** (語彙外日本語含む)。
- **フットプリント**: 射影 8 本 FP32 121,634,816 B → INT8 量子化重み + per-row scale
  30,605,312 B = **削減 91,029,504 B (74.84%減・残存比 0.2516 ≈ 1/4)**。
- **seq8 forward レイテンシ**: INT8 **~152.9ms** / FP32 ~393.3ms = **0.389x** (INT8 が速い)。
  int8 内積の int64 蓄積が FP32 dense の double 蓄積 dot より軽いため。lm_head は両者 FP32 同条件。
- test_bitnet_int8 全サブテスト緑・全 22 テスト緑・Skipped 0・dense #6/A3 golden 非回帰確認済。

### 15.3 知見

- **33M dense は per-row 重み INT8 で greedy 完全一致 = 量子化耐性が高い**。corr が
  ハードゲート 0.99 を 3 桁上回り、生成トークンは 1 ビットもずれない。
- **74.84% 圧縮 (射影層 1/4) かつ CPU で 2.6x 高速化を損失ほぼ無しで両取り**できた
  (品質劣化が見えない範囲での圧縮 + 速度向上)。
- 圧縮の研究軸としての位置づけは変わらず — 本線の品質基準は dense FP32 (#6/A3) のまま。

### 15.4 スコープ外

- **GPU INT8** (device 上の int8 GEMM) は別タスク。
- **ternary GEMM (#5・`src/kernels/ternary_gemm.cu`)** は乗算削減の別実験軸。
- **INT4** (4bit 重み) も別タスク。本節は CPU・重みのみ INT8 dense に限定する。


## 16. 施策 D — 容量増 (33M → 80M) seed sweep (D クローズ・陰性確定)

施策 B (入力多様化) が ~2,000 件で飽和 (§14.9)・施策 A (実ペア増) が diverse-val F1 に
seed ノイズで非寄与 (§9.10) と判明した後、残る diverse-val F1 の伸びしろを **容量** に賭けて
測った。`src/models/bitnet_config.hpp` の `DOLLAMA_BITNET_ARCH=d80m` (N_LAYERS 8→16 /
FFN_DIM 1792→2464・D_MODEL/heads/HEAD_DIM/vocab/max_seq 据え置き = **79,908,864 params**)
ビルド時切替と `train_bitnet.py --arch d80m` は配線済 (改修ゼロで焼ける)。

**設計 (A/B/D5/D6 と同一作法)**: 2 アーム **c33(33M) / d80(80M)** とも完全同一レシピ =
b2000 多様化 train (`--train-file`) ∧ a12k identity (`--identity`)。唯一の差分は `--arch d80m`
の有無 (= 容量のみ)。4 seed (20260620 / 20260621 / 42 / 7)・6ep paired を
`scripts/dollma_d_seedsweep.py` (`--only-seed` 小出し・`_results/*.npz` 冪等 skip) で回し、
`dollma_d_seedsweep_analyze.py` で 3 軸判定。出力は `data/bitnet/_seedsweep_d80m/` 配下のみ
(本番 `bitnet_dense*`/golden・#1 本線 pairs・凍結 eval を一切無改変)。delta = d80 − c33・
band は **c33 の seed 分散** (A の base band と同じ役割)。

**着手前に明文化した打ち切り基準**: 主指標 = diverse-val 生成 set-F1 を 3 軸
((a)全 seed 符号一貫 (b)delta 平均が c33 seed 分散帯 sd 超え (c)各 seed paired 95%CI が 0 除外) で
測り、**不成立なら陰性確定 (データ律速で容量効かず)・80M は出荷しない**。retention(≥0.975 床) /
in-dist pairs.val F1 はガードレール (非退行の床であって D の成否ではない)。

### 16.1 主指標 diverse-val 生成 F1/Jaccard = seed ノイズ (全 4 set/metric 判定 NO)

delta = d80(80M) − c33(33M)・per-seed paired (n=1500/seed)。

| set / metric | per-seed delta (20260620 / 20260621 / 42 / 7) | across 平均 ± sd | c33 band sd | (a)符号 | (b)>band | (c)CI除外 | 頑健 |
|---|---|---|---|---|---|---|---|
| diverse_a / F1 | −0.0240 / −0.0047 / −0.0008 / **+0.0133** | −0.0040 ± 0.0154 | 0.0114 | NO (seed7 反転) | NO | Y/N/N/Y | **NO** |
| diverse_a / Jaccard | −0.0169 / −0.0041 / −0.0007 / **+0.0091** | −0.0032 ± 0.0107 | 0.0079 | NO | NO | Y/Y/N/Y | **NO** |
| diverse_b / F1 | −0.0264 / −0.0042 / **+0.0064** / **+0.0156** | −0.0021 ± 0.0181 | 0.0131 | NO | NO | Y/N/Y/Y | **NO** |
| diverse_b / Jaccard | −0.0207 / −0.0046 / **+0.0048** / **+0.0115** | −0.0023 ± 0.0140 | 0.0098 | NO | NO | Y/Y/Y/Y | **NO** |

全 4 set/metric で判定 NO。**seed 20260620 で大きく負 (−0.024〜−0.026)・seed 7 で正
(+0.013〜+0.016) と符号が反転**し、across 平均は −0.002〜−0.004 (むしろわずかに負) で c33
自身の seed 分散帯 sd 以下に埋もれる。各 seed 内では paired CI が 0 を除外することがある
(per-sample n=1500・|t| 大) が seed を跨ぐと安定しない = **A12k (§9.10)・D6 (§12.5) と同型の
seed ノイズ**。施策 B の ~2,000 飽和 (§14.9) と整合し、**diverse-val F1 の頭打ちはデータ律速で、
容量 (33M→80M) では取れない**ことが確定した。

### 16.2 ガードレール — 80M はむしろ床割れ気味

| seed | c33 retention | d80 retention | c33 in-dist F1 | d80 in-dist F1 |
|---|---|---|---|---|
| 20260620 | 0.9807 | 0.9744 | 0.4552 | 0.4516 |
| 20260621 | 0.9800 | 0.9739 | 0.4629 | 0.4570 |
| 42 | 0.9752 | 0.9771 | 0.4605 | 0.4609 |
| 7 | 0.9755 | 0.9711 | 0.4612 | 0.4559 |
| **across** | **0.9778 ± 0.0029 (全 seed ≥0.975 ✅)** | **0.9741 ± 0.0025 (3/4 seed 床割れ ❌)** | 0.4599 | 0.4564 (微退行) |

c33 は retention 床 0.975 を全 seed クリア・in-dist も健全。**d80 は retention が 4 seed 中 3 seed
(0.9744/0.9739/0.9711) で床割れ**し、in-dist pairs.val F1 もわずかに退行。80M を採る理由はガード
レール側にも無い。

### 16.3 判定 — D 陰性確定 (80M 不採用・勝者 = c33 33M)

着手前の打ち切り基準にそのまま該当 (3 軸不成立)。**容量倍増 (33M→80M) で diverse-val F1 は
seed ノイズ内 (平均わずかに負)・retention 床割れ・in-dist 微退行** → **80M は出荷しない**。
80M は CPU/GPU forward が ~2x (matmul 律速) になるが、その対価を払うロバストな F1 実利が無い。

**勝者 = c33 (33M・b2000 多様化 ∧ a12k identity) = #1 超え出荷候補**。施策 A/B/D5/D6/D の
探索を通じ、**diverse-val F1 を頑健に押し上げたのは入力多様化 (B・~2,000 で飽和) のみ**で、
蒸留 (D5/D6)・実ペア増 (A)・容量 (D) はいずれも非寄与か seed ノイズと確定した。残る低帯域
(diverse_a ~0.31 / diverse_b ~0.36) を取りに行く次のフロンティアは、容量でもデータ件数でもない
別軸 (データ多様性の質・アーキ・損失設計など) に求める必要がある。

正典差し替え (`bitnet_dense.safetensors` 置換 + C++ golden 再生成) は `[B-merge-at-A]` 遅延条項
どおり**勝者 33M (b2000 ∧ identity) で 1 回だけ**まとめ焼きする (別途プランモードで設計)。

### 16.4 検証・非回帰 (ルール#4・Python のため C++ meson test 対象外)

- `scripts/test_dollma_d_seedsweep.py` 構造テスト **8/8 緑** (2 アーム c33/d80・両 arm 同一
  レシピ・d80 のみ `--arch d80m` が train/eval 両方に連動・SEEDS/COMMON/setup コピー対象)。
- sweep 出力は `data/bitnet/_seedsweep_d80m/` 配下のみ (gitignore)。本番重み/golden/凍結 eval/
  #1 本線 train/val を一切無改変。`--identity` 駆動の重みは arch 非依存固定名
  `bitnet_dense_identity*` に出るため、各 arm は train 直後に即 eval→`_results/` 退避で衝突回避。

### 16.5 再現手順

```
# 全 4 seed (~3.8h・GTX1080Ti FP32・eval-only 律速)
py -3.12 scripts/dollma_d_seedsweep.py
# seed 小出し (冪等再開可)
py -3.12 scripts/dollma_d_seedsweep.py --only-seed 20260620
# 集計・3 軸判定
py -3.12 scripts/dollma_d_seedsweep_analyze.py
```

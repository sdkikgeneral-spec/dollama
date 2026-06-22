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

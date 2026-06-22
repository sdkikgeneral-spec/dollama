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

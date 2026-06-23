# dollama タグ生成 LM 訓練データセット仕様 (Phase 4 #1)

自作タグ生成 LM (`src/models/bitnet.hpp`) が学習する **「user text (自然文) → danbooru タグ列」** ペアの
データセット仕様。`scripts/train_bitnet.py` (model-trainer) と
`src/io/tokenizer.hpp` (cpp-implementer) が消費する。

**現行最終版 (1e/1f スケール後): 総 5,000 ペア (train 4,500 / val 500, 9:1)・
ja/en ≈ 50:50・rating g 100%・全件検証通過 (OOV/負語/順序/リーク/重複 すべて 0)・
ユニーク text 比率 1.0・タグ語彙 4,994。** 詳細は §11 を参照。

**Phase 4 A (A1): 同一性条件付きペア 5,000 (train 4,500 / val 500) を別ファイル
`pairs.identity.{train,val}.jsonl` に追加 (源は #1 と同じ共起キャッシュ・vocab 不変・
後方互換)。`<sep>` 2 回流用で identity/scene/target を区切る。全件検証 0・tokenizer 往復
UNK 0。詳細は §13。**

## 0. 設計趣旨 (ユーザー確定方針)

- **正解 = 実 danbooru のタグ共起** (人手アノテのタグ列)。LLM にタグを「推測」させない。
  自然文側をタグ列から**逆生成**する。
- **自然文側 = 合成テンプレート主軸**。汎用 LLM 多様化は将来の補強 (§3.1 `source:"llm_distill"`)
  で、本段では呼ばない (テンプレのみで 5,000 を達成)。
- **2D に強く・画像不要・法務クリーン・非研究機 PC で回る**。

## 1. データソース (タグのみ・画像ピクセル非取得)

### 1.1 タグ語彙の土台 — WD14 `selected_tags.csv`

- 取得元: `https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3/resolve/main/selected_tags.csv`
- 形式: `tag_id,name,category,count` の 10861 行 (header 除く)。
- category コード: **`0`=general / `4`=character / `9`=rating** (WD14 v3 実測)。
  内訳: general 8106 / character 2751 / rating 4 (`general/sensitive/questionable/explicit`)。
- これは推論モデルの分類ヘッド語彙であり、`src/infer/wd14.hpp` の `N_TAGS=10861` と一致。
  **画像は一切含まない**。タグ名・カテゴリ・出現頻度 (count) のみのメタデータ。

### 1.2 タグ共起の正解源 — danbooru 公式 API `posts.json`

- 取得元: `https://danbooru.donmai.us/posts.json?limit=200&tags=rating:general&page=b<id>`
- **タグ文字列フィールドのみ保持**し、画像 URL (`file_url` 等) ・プレビュー・
  ピクセルは取得も保持もしない:
  - `tag_string_general`   — 一般タグ (本データセットの target 主体)
  - `tag_string_character` — キャラ名 (target には**展開しない**。§4 参照。件数のみ `n_character` に記録)
  - `rating`               — `g/s/q/e` (NSFW フラグ分離に使用)
- **ページング**: keyset 方式 `page=b<最小id>` (前ページの最小 id) で過去方向へ辿る。
  offset 方式 (`page=N`) は深いページで失敗するため使わない。
- **キャッシュ延伸 (1e で追加)**: `fetch_posts` は既存キャッシュを**捨てずに**読み、
  不足分だけ**キャッシュ最古 id を起点に過去方向へ追加取得**する (post_id で重複除去・
  降順で再保存)。1c の 1,000 posts を土台に 1e で 8,200 posts まで延伸した。
- **レート制限**: 無認証は控えめに。**1 リクエスト 200 件・リクエスト間 1.0s sleep**。
  1e の 8,200 posts 取得は約 36 リクエスト ≈ 実測 50〜90 秒程度。
- **SSL**: この PC の Python は danbooru の証明書チェーンを既定 CA で検証できない場合がある。
  `certifi.where()` の CA バンドルを `ssl.create_default_context(cafile=...)` に渡す。

### 1.3 法務 (タグはメタデータ・画像非取得)

- danbooru のタグは**事実ラベル (メタデータ)** であり画像著作物ではない。画像ピクセルを
  収集・保持しないことでリスクを最小化する。
- 日本では ML 学習目的のメタデータ解析は著作権法 30条の4 が広く許容するが、
  **30条の4 は ToS を上書きしない**。danbooru API の利用は控えめなレート・
  研究目的の範囲に留める。
- **規模拡大 (1e の 5,000 超) / 商用化前には project-leader 経由で専門家確認**。
- 取得した生キャッシュ (`data/bitnet/cache/danbooru_posts.jsonl`) も **タグ文字列と
  rating のみ** を保存し、画像参照を含めない。

## 2. ファイル構成

```
data/bitnet/
  vocab.json                      — タグ語彙表 (tokenizer.hpp が読む)
  pairs.train.jsonl               — 訓練ペア (#1・4,500 行・source:synthetic)
  pairs.val.jsonl                 — 検証ペア (#1・500 行)
  stats.json                      — #1 の件数・分布・出典・seed の記録
  pairs.identity.train.jsonl      — 同一性条件付き訓練ペア (A1・4,500 行・source:identity_cond, §13)
  pairs.identity.val.jsonl        — 同一性条件付き検証ペア (A1・500 行)
  stats.identity.json             — A1 の件数・identity 分布・重複率の記録
  cache/danbooru_posts.jsonl      — 生タグ共起キャッシュ (タグ文字列 + rating のみ・8,200 posts)
  cache/selected_tags.csv         — WD14 タグ csv キャッシュ
docs/dataset-spec.md              — 本ファイル
scripts/dollma_build_vocab.py     — vocab.json 生成
scripts/dollma_make_pairs.py      — #1: 共起取得 → 射影 → 自然文逆生成 → pairs/stats 出力
scripts/dollma_make_identity_pairs.py — A1: identity/scene 分離 → 同一性条件付きペア (§13)
```

## 3. スキーマ

### 3.1 ペア JSONL (`pairs.{train,val}.jsonl`)

1 行 1 JSON オブジェクト (UTF-8, `ensure_ascii=false`):

```json
{
  "text": "long hair・blue eyes・gloves・dressの女の子が一人。full body・grey background。",
  "tags": ["1girl", "solo", "long hair", "blue eyes", "gloves", "dress", "full body", "grey background"],
  "lang": "ja",
  "source": "synthetic",
  "meta": {"rating": "g", "post_id": 11629810, "n_tags": 8, "tmpl": 1}
}
```

| フィールド | 型 | 説明 |
|---|---|---|
| `text`   | string | 自然文 (合成テンプレ逆生成)。`lang` の言語。 |
| `tags`   | string[] | **正準順序** (§5) の target タグ列。区切り正規化済み (§6・**スペース区切り**)。全要素が vocab 内。 |
| `lang`   | `"ja"`\|`"en"` | 自然文の言語。 |
| `source` | string | `"synthetic"` (本段)。**将来拡張**: `"llm_distill"` (§12)。 |
| `meta`   | object | `rating` (g/s/q/e)・`post_id` (出典 danbooru post id・再現用)・`n_tags`・`tmpl` (使用テンプレ variant 0..2)。NSFW フラグは `meta.rating` で分離。 |

- `text` は学習入力、`tags` が target。タグ生成 LM は text を読んで tags 列を自己回帰生成する (GPU 主・CUDA カーネル流用 / CPU 可・NPU 不可)。
- `post_id` はリーク防止チェック (同一投稿が train/val に跨らない) と再現に使う。
- `tmpl` は §3.3 のテンプレ多様化 variant。`post_id % 3` で決定的に選ばれる (再現的)。

### 3.2 語彙表 (`vocab.json`) — tokenizer.hpp が読む形式

```json
{
  "version": 1,
  "separator": "space",
  "specials": ["<pad>", "<bos>", "<eos>", "<sep>", "<unk>"],
  "tags": [
    {"id": 5, "tag": "1girl",   "category": 0, "freq": 5113288},
    {"id": 6, "tag": "solo",    "category": 0, "freq": 4253220},
    {"id": 7, "tag": "long hair","category": 0, "freq": 3645082}
  ]
}
```

| フィールド | 説明 |
|---|---|
| `version`   | スキーマ版 (現 1)。 |
| `separator` | タグ内単語の区切り規約。**`"space"`** = danbooru の `_` をスペースに正規化 (§6)。 |
| `specials`  | 特殊トークン。**id 0..4 を固定予約** (`<pad>`=0, `<bos>`=1, `<eos>`=2, `<sep>`=3, `<unk>`=4)。 |
| `tags`      | タグ語彙配列。**id は specials の続き (5 から) で頻度降順に連番**。各要素 `{id, tag, category, freq}`。 |

- `category`: WD14 コードを踏襲し拡張。**`0`=general / `4`=character / `9`=rating /
  `100`=color_mode 制御タグ** (本データセット拡張・§7-b)。
- `tag` 文字列は区切り正規化済み (スペース区切り)。tokenizer は `tag` 文字列で完全一致引きする。
- **id 連続性保証**: `specials` の数 = 5、`tags[i].id == 5 + i`。tokenizer は配列順 = id 順を仮定してよい。
- **現行語彙: 4,994 タグ** (min_count=2518 足切り・負語/rating 除外後)。全 target タグが
  この語彙に収まり、tokenizer 往復 UNK 0 を確認済み (§11)。

### 3.3 自然文テンプレ (合成逆生成・多様化)

- 自然文は **タグ列 → 文** の決定的逆生成。`scripts/dollma_make_pairs.py` の
  `JA_TEMPLATES` / `EN_TEMPLATES` に **各言語 3 文型 (variant 0..2)** を持つ。
- variant は `post_id % 3` で決定的に選ばれる (同一タグ集合でも文面が散り、ユニーク text
  比率を底上げする。乱数に依存しないので再現的)。
- 文型は「特徴列挙 + subj」「subj。特徴は…」「…を描いてください」系など、語順・接続詞・
  助詞が異なる。実測でテンプレ別件数はほぼ均等 (variant 0/1/2 ≈ 1685/1663/1652)、
  **ユニーク text 比率 1.0** (5,000 件すべて文面が一意)。

### 3.4 統計 (`stats.json`)

```json
{
  "version": 1,
  "seed": 20260620,
  "created_utc": "2026-06-20T...",
  "counts": {"total": 5000, "train": 4500, "val": 500},
  "lang_ratio": {"ja": 0.502, "en": 0.498},
  "source_ratio": {"synthetic": 1.0},
  "rating_ratio": {"g": 1.0},
  "split_ratio": {"train": 0.9, "val": 0.1},
  "unique_text_ratio": 1.0,
  "template_dist": {"0": 1685, "1": 1663, "2": 1652},
  "vocab": {"total_tags": 4994, "cutoff_min_count": 2518, "raw_tags": 10861},
  "tag_freq_top": [["1girl", 2784], ["solo", 2517]],
  "tags_per_pair": {"min": 4, "max": 16, "mean": 15.44},
  "sources": {
    "wd14_csv": "SmilingWolf/wd-swinv2-tagger-v3/selected_tags.csv",
    "danbooru_api": "danbooru.donmai.us/posts.json (tags-only, no pixels)"
  }
}
```

## 4. キャラ名 (character タグ) の扱い

- danbooru の `tag_string_character` (例 `elysia_(honkai_impact)`) は **target tags に展開しない**。
  CharacterBible の `name` は「拡散へ渡す文字列ではなく台帳の主キー」(spec §1) であり、
  タグ生成 LM が固有キャラ名を生成する設計ではないため。
- vocab には character カテゴリ (cat 4) を**含めてよい** (tokenizer 語彙の網羅性のため) が、
  本データセットの target からは除外する。将来キャラ条件付き生成を扱う場合に再検討する。
- これにより「同一キャラの train/val リーク」も target レベルでは発生しない
  (リーク防止は `post_id` でも担保・§8)。

## 5. タグ順序の正準化 (compose_prompt 順)

target `tags` は `src/core/character.hpp::compose_prompt` の positive 順に並べる:

```
canonical  →  color_mode  →  pose  →  expression  →  composition  →  isolation
```

danbooru の生タグは順不同なので、各タグを以下のバケットに分類して並べ替える
(`scripts/dollma_make_pairs.py` の `classify_bucket`):

| バケット | 該当タグ (例) | 由来 |
|---|---|---|
| 1 canonical | 主体数 (`1girl`/`solo`) ・髪/目/体型/服 など外見一般タグ | compose_prompt 先頭 |
| 2 color_mode | `monochrome`/`greyscale`/`lineart`/`sketch` | `color_mode_tags()` |
| 3 pose | `sitting`/`standing`/`waving`/`arms up` 等の動作語 | `SceneSpec.pose_tags` |
| 4 expression | `smile`/`blush`/`open mouth` 等の表情語 | `SceneSpec.expression_tags` |
| 5 composition | `upper body`/`full body`/`cowboy shot`/`portrait` | `SceneSpec.composition` |
| 6 isolation | `simple background`/`white background`/`grey background` | `OutputSpec.isolation_tag` |

- バケット内は danbooru count 降順 (vocab freq 順) で安定ソート。
- 分類は語彙リスト + ヒューリスティック (`POSE_TAGS`/`EXPRESSION_TAGS`/`COMPOSITION_TAGS`/
  `ISOLATION_TAGS`/`COLOR_MODE_TAGS` の集合)。未分類は canonical (バケット1) に落とす
  (compose_prompt も canonical を先頭に置くため整合)。
- タグ数は **16 で打ち切り** (バケット順を保ったまま先頭 16)。

## 6. 区切り正規化 (決定)

- **danbooru / WD14 はアンダースコア区切り** (`long_hair`, `looking_at_viewer`)。
- **dollama は character.hpp / compose_prompt がスペース区切り** (`long hair`, `silver hair`)。
- **決定: データセットは全てスペース区切りに正規化する** (`vocab.separator = "space"`)。
  - 変換: タグ名中の「英数字に挟まれた `_`」のみ ` ` (半角スペース) に置換。
  - 例外: 顔文字系タグ (`^_^`, `>_<` 等) は記号に隣接する `_` を保持 (置換しない)。
- tokenizer.hpp はこの**スペース区切り正規形**でタグを完全一致引きする。
  compose_prompt 出力 (スペース区切り) がそのままタグ生成 LM の入力語彙と一致する。

## 7. 必達制約 (CharacterBible 整合)

- **(a) 負語の分離**: `default_quality_negatives()` の語と指本数を仮定する語は
  **positive 語彙 (vocab tags) から除外**する (`dollma_build_vocab.py` の
  `NEGATIVE_BLOCKLIST`)。target tags にも混入させない (`dollma_make_pairs.py` の
  `validate_pairs` が全件検査・1e で 0 件確認)。
- **(b) color_mode 制御タグ**: `monochrome`/`greyscale`/`lineart`/`sketch` は
  vocab に **category=100 (color_mode)** で含める。バケット2 で正準順序化する。
- **(c) 区切り正規化**: §6 のとおりスペース区切りに統一・spec 明記。
- **(d) name 非展開**: §4 のとおり character タグを target に展開しない。

## 8. 分割 (seed 固定・リーク防止)

- `split_ratio` 確定 **train 0.9 / val 0.1** (test は将来導入検討)。
- **seed = 20260620 固定** (再現可能)。`random.Random(seed)` でシャッフル後に分割。
- **リーク防止 (全件アサート・1e で 0 件確認)**:
  - 同一 `post_id` が train/val に跨らない (post 単位で分割)。
  - 同一 `text` (完全一致) が train/val に跨らない (重複除去 → 分割)。
  - **重複ペア除去**: 同一 `(text, tags)` および同一 `text` の重複は生成段で除去し、
    件数に数えない (`seen_pairkey` / `seen_text`)。
  - character タグは target 非展開なので「同一キャラのタグ列リーク」は target に出ない。

## 9. 再現手順

```sh
# 1. 語彙構築 (WD14 csv DL → 正規化 → 足切り → id 割当)
python scripts/dollma_build_vocab.py --min-count 2518 --out data/bitnet/vocab.json
# 2. 共起取得 + ペア生成 (danbooru API → 射影 → 逆生成 → 分割)
#    キャッシュがあれば再利用し不足分のみ追加取得 (post_id 重複除去)。
python scripts/dollma_make_pairs.py --n 4500 --val 500 --seed 20260620 \
    --vocab data/bitnet/vocab.json --out-dir data/bitnet
```

- 両スクリプトとも seed 固定・パラメータは CLI 引数。`stats.json` に実行時の
  seed・件数・出典・分布を記録する。
- ネットワーク取得は `certifi` の CA バンドルを使用。取得失敗時はキャッシュ
  (`data/bitnet/cache/danbooru_posts.jsonl`) があればそれを再利用する。
- 歩留り (1e 実測): 投稿 → ユニークペア ≈ 79%。`--fetch-factor` (既定 1.6) で
  射影/重複落ちの余裕を持って取得する。不足時は警告を出すので factor を上げる。

## 10. 引き渡し

- **model-trainer** (`scripts/train_bitnet.py`): `pairs.{train,val}.jsonl` を読む。
  text=入力 / tags=target。tokenizer は §3.2 vocab を使う。dtype/系列長は別途確定。
- **cpp-implementer** (`src/io/tokenizer.hpp`): `vocab.json` を読む。`specials` (id 0..4 固定)
  + `tags[i].id == 5+i` 連番・スペース区切り完全一致引き。全 target タグで UNK 0 を確認済み。
- **prompt-engineer**: §6 区切り正規化・§4 name 非展開・§7-a 負語分離の規約を共有。

## 11. 最終データセット実測 (1e/1f・#1 完了ライン)

- 取得: danbooru `posts.json` `rating:g`、keyset 8,200 posts (キャッシュ延伸)。WD14 csv 10,861 行。
- 生成: 8,200 posts → ユニークペア 6,400+ (歩留り ≈ 79%) → 上限 5,000 で打ち切り。
- **総件数 5,000 (train 4,500 / val 500, split 0.9 / 0.1)**。
- **lang: ja 0.502 / en 0.498**。**source: synthetic 1.0**。**rating: g 1.0**。
- **テンプレ variant 分布: 0=1685 / 1=1663 / 2=1652** (ほぼ均等)。
- **ユニーク text 比率 1.0** (5,000 件すべて文面が一意・大量重複なし)。
- **tags/pair: min 4 / max 16 / mean 15.44**。target タグ総数 77,195。
- **タグ頻度上位**: 1girl 2784 / solo 2517 / long hair 1991 / looking at viewer 1677 /
  shirt 1613 / short hair 1401 / long sleeves 1184 / dress 1020 / black hair 973 / jacket 955 …
- **全件検証 (train+val 5,000)**: OOV 0 / 負語混入 0 / 非正準順序 0 / post_id リーク 0 /
  text リーク 0 / 重複ペア 0。
- **tokenizer 往復**: vocab.json で全 target タグを encode→decode、UNK 0・完全一致復元。
- vocab: 4,994 タグ (raw 10,861・min_count 2518)。

## 12. 将来拡張 — LLM 多様化 (`source:"llm_distill"`)

本段はテンプレのみで 5,000 を満たした。汎用 LLM (Qwen2-1.5B 蒸留・CPU 推論) による
自然文多様化は将来の補強として、**既存スキーマを変えずに追加できる**設計を維持する:

- 同じ実 danbooru タグ集合 (正準順序済み target) を入力に、LLM へ「このタグを表す
  自然な日本語/英語の依頼文を書け」と指示して text を生成 → `source:"llm_distill"` で
  ペアを追加 (`tags` は実共起のまま・LLM にタグを推測させない方針は不変)。
- `meta.post_id` は流用元 danbooru post を保持し、§8 のリーク防止 (post_id/text) を
  そのまま適用する。`meta.tmpl` は LLM 由来では省略または `-1` とする。
- 検証 (`validate_pairs`) は source 非依存で全件に適用。OOV/負語/順序は同基準。
- CPU 推論 64-71 tok/s (CLAUDE.md probe7) を前提に、バッチ時間を見積もって段階投入する。

### 12.1 実測 (D2 — Qwen2-1.5B 蒸留 3,000 ペア生成)

`data/bitnet/pairs.distill.train.jsonl` (3,000 件・`stats.distill.json`)。

- 教師 = `Qwen/Qwen2-1.5B-Instruct` (CPU)。**系列レベル蒸留・hard CE**: 教師には実 danbooru
  共起タグを表す自然文 (依頼文) のみ生成させ、`tags` は実共起のまま無改変 (LLM にタグを
  推測させない方針・§12 不変)。系列は synthetic と同じ 1-`<sep>` (`<bos> text <sep> tags <eos>`)。
- ja/en = 0.24/0.76 (en_target 0.75)・accept率 0.6637 (reject_total 1520・主因は hair/subject
  矛盾の自動棄却)・unique_text 1.0・生成 18.5 tok/s (10,662s)。
- 検証全緑: `validate_pairs` 0 / val post リーク 0 / val text リーク 0 / tokenizer UNK 0 /
  tags 改変 0。スキーマは synthetic 行と完全互換 (`meta.tmpl=-1`)。

### 12.2 実測 (D4 — 蒸留混合の効果: 過学習は緩和されず)

synthetic 4500 + distill 3000 = 7500 を train 混合 (val は #1 と同一 500・不変) で訓練した
結果、**過学習は緩和しなかった** (training-spec §10 に A/B 詳細)。要点のみ:

- val_loss 反転が #1 ep4 → 蒸留 ep2 へ**前進**、同 epoch の train-val gap は**拡大**、最良
  val_loss/recall も微減 (6ep: 2.41/0.777 → 2.60/0.762)。
- 唯一の利得は**生成多様性**: 蒸留版は val prompt から約 2 倍の unique tag を生成し、
  系列内反復が減る (per-seq unique 0.945 → 0.978〜0.991)。teacher-forced は悪いが自由生成は広い。
- 原因: train が 1.67x 増で同 epoch の勾配ステップ過多 + Qwen2 自由文が synthetic val 分布と乖離。
  hard CE 混合だけでは正則化にならない。次は soft-label KL 蒸留 (温度付き教師 logits) を検討。

### 12.3 D5 — 案A 共起 soft teacher (新ペアではない・D2/D4 hard 混合とは別物)

D5 (soft-label KL 蒸留・training-spec §11) の案A teacher は、**新しいペアファイルを一切作らない**。
既存 `cache/danbooru_posts.jsonl` (8,200 posts・タグ + rating のみ・画像非取得・§1.2/§1.3) の
**タグ共起経験分布**から、各 target 位置の**正解分布を軟化**する (prefix 条件付き soft label =
意味づけされたラベルスムージング)。`pairs.*.jsonl` も `tags` (実共起の正準順序 target) も**不変**で、
学習時に target one-hot を共起ベースの soft 分布へ置き換えるだけ。

- **D2/D4 (§12.1/§12.2) との違い**: D2/D4 は Qwen2 で自然文を多様化した**新規ペア**
  (`pairs.distill.train.jsonl` 3,000 行) を train に**混合** (入力側データ拡張・hard CE のまま)。
  D5 は**ペアを増やさず**既存 corpus の共起で**ラベル側を軟化**する (出力側 soft label)。
  別軸の手法であり、D5 は新しい JSONL を生成しない。
- 共起テーブルは `cache/danbooru_posts.jsonl.cooc.npz` (COO int32・pickle なし) にキャッシュして
  再利用する (再現的)。dataset としての新ファイルは増えない。
- A/B 結果・採否は training-spec §11 (採用せず・ただし過学習/gap は明確に改善する正の機構)。

### 12.4 D6 — 案c 外部教師 soft target (TIPO-200M・新ペアではない)

D6 (soft-label KL 蒸留・training-spec §12) の案c teacher も、**新しいペアファイルを一切作らない**。
外部教師 **TIPO-200M** (KBlueLeaf/TIPO-200M・apache-2.0) に各サンプルのタグ補完を生成させ、
出力**タグ文字列**を自作 4999 vocab に**完全一致**写像 (vocab 外は drop + in-vocab 質量で再正規化)
した「条件付きタグ集合」を、各 target 位置の soft 分布に注入する (案b-tagset)。`pairs.*.jsonl` も
`tags` も**不変**で、学習時に target one-hot を TIPO 生成由来の soft 分布へ置き換えるだけ。

- **D5 (§12.3) との違い**: D5 案A は既存 corpus の**タグ共起**で soft 化 (外部モデル不使用)。
  D6 案c は**外部 LM (TIPO) の生成タグ頻度**で soft 化 (真の知識転移)。どちらも新ペアを作らない
  出力側 soft label だが、teacher 信号の出所が共起統計か外部 LM かで別物。
- **教師キャッシュ (新ファイル・本番非破壊)**: `dollma_d6_teacher_cache.py` が position 軸付き
  COO npz `cache/d6_teacher_soft.{train,val}.npz` (rows/poss/cols/probs・pickle なし) と
  `cache/d6_teacher_stats.json` (OOV 保持率 train 0.791/val 0.790・平均エントロピー 0.85 nats)
  を出力。生成は train 4,500 件 ~1.5h・val 500 件 574s (TIPO 本体は生成時のみロード・訓練時は npz)。
- **写像規則 (絶対制約)**: カンマ split (空白では割らない = `long hair` を壊さない) →
  `Tokenizer.normalize` 再利用 → `vocab.json` 完全一致 → OOV drop + 再正規化。next-subword logit
  の直写像 (案b-logit) は却下 (TIPO BPE は 1 タグ = 複数 subword のため)。
- A/B 結果・採否は training-spec §12 (**不採用** — 単一 seed の recall 上振れは seed 頑健性
  sweep で再現せず seed ノイズと確定・#1 本線維持。再現する効果は過学習抑制のみ)。dataset
  としての新ファイルは npz キャッシュのみで `pairs.*.jsonl` は不変。

## 13. 同一性条件付きペア (Phase 4 A・A1 確定版)

Phase 4 A (同一性条件付きタグ生成) 用のデータ。#1 (§1-§12) は「user text → tags」のみで、
§4 のとおりキャラ同一性を target から扱わない。A は **キャラ同一性を条件入力** にする
ため、各 danbooru post のタグを **identity / scene に分離** した別形式ペアを持つ。
**生成: `scripts/dollma_make_identity_pairs.py`** (#1 の共起取得・§5 正準順序・§6 正規化・
テンプレを再利用)。実測は §13.6。

### 13.1 条件付け機構 (承認済み・厳守)

**(a-1) prompt prefix + `<sep>`(id=3) の 2 回流用**。訓練系列 (構築は A2 / `train_bitnet.py`):

```
<bos> [identity tags] <sep> [scene text のタグ] <sep> [target tags] <eos>
```

- **vocab.json / VOCAB_SIZE / `specials` / tokenizer は一切変更しない**。`<sep>` を 2 回
  使うだけで、各区間の意味は **位置** で決まる (id=3 が 2 個出るのは仕様)。
- 既存 #1 系列は `<bos> text <sep> tags <eos>` (`<sep>` 1 回)。A は prefix に identity を
  足し `<sep>` を 1 個増やすだけ。tokenizer.hpp / `decode` は `<sep>` を構造トークンとして
  読み飛ばすため (§3.2)、`<sep>` の回数に依存せず後方互換。

### 13.2 identity / scene 分離規則 (典拠: character-bible-spec §1/§2)

- **identity_tags** = `CharacterIdentity.canonical_tags` 相当 = コマ間で **不変** の外見属性。
  §5 バケット 1 (canonical) の **部分集合** として定義する (バケット 2..6 は定義上すべて scene):
  - 主体数/性別 (`1girl`/`1boy`/`multiple girls`/`futanari`/`otoko no ko` …)
  - 髪 色/長さ/型 (`long hair`/`silver hair`/`twintails`/`ahoge` …)
  - 目 色/瞳/特徴 (`blue eyes`/`heterochromia`/`slit pupils` …)
  - 肌 (`dark skin`/`pale skin`/`colored skin` …)
  - 体型/外見年齢 (`large breasts`/`mature female`/`loli`/`muscular`/`petite` …)
  - 種族形質 (`animal ears`/`tail`/`horns`/`wings`/`elf`/`pointy ears` …)
- **scene_tags** = それ以外すべて (pose/expression/composition/isolation/color_mode +
  **服飾・小物・状態・背景単色**)。
- 判定は `dollma_make_identity_pairs.py` の `is_identity_tag` (パターン群 `_IDENTITY_PATTERNS`
  + 除外 `_IDENTITY_EXCLUDE_EXACT` / 服飾 `_CLOTHING_RE`)。「`hair ornament`/`hair ribbon`」
  「`closed eyes`/`one eye closed`」「`breasts out`/`cleavage`」等の **装飾・状態・行為** は
  identity から除外する (不変外見ではない)。

**服装の扱い (論点・保守判断 = scene)**: canonical 思想 (bible §1) では服も同一性になり得る
(制服キャラ等) が、danbooru の 1 post = 1 イラストでは服は post ごとに変わる可変要素。
服を identity に含めると ① 同一性集合が衣装違いで膨れ「同一キャラの汎化」を測れない
② identity 重複率が人工的に下がる ③ retention 教師が「服も毎回同じ」を強制する。よって
**服飾・小物 (`shirt`/`dress`/`gloves`/`hat`/`ribbon`/`necklace`/`glasses` …) は scene 側に寄せる**。
キャラ固有衣装の固定は運用側が `CharacterIdentity.canonical_tags` に明示する設計
(compose_prompt が canonical を先頭に置く) で担保され、データ側で服を identity と決め打つ
必要はない。

### 13.3 ペア生成

- **target tags** = identity_tags ∪ scene_tags を **§5 正準バケット順**で並べる
  (#1 の `make_pair` と同一の分類 → freq 降順安定ソート → バケット順 → **16 件打ち切り**)。
  identity/scene 分離は **打ち切り後** の正準列に対して行うため、`identity_tags + scene_tags`
  は target の完全な分割 (集合一致・順序は target の部分列)。
- **identity_tags は必ず target に含める** (identity retention の教師信号の核 =
  定義上 retention 100% の正解)。identity が空になる post (同一性の核なし) は教師にならず
  **スキップ**する。
- **text** = identity を保持した状況記述 (自然文)。#1 テンプレ (`JA/EN_TEMPLATES`) を流用し、
  identity 由来語を冒頭に明示する。これにより A2 の text 側 greedy 最長一致が identity を拾える。

### 13.4 スキーマ (後方互換必須)

既存 #1 行 (`source:"synthetic"`) は **無改修で読める** まま。新形式は `source:"identity_cond"`
+ `meta` 拡張を足すのみ。`text`/`tags` の意味は #1 と同じ枠 (`tags` = target)。

```json
{
  "text": "long hair・blue eyes・animal earsの女の子が一人を描いてください。",
  "tags": ["1girl", "long hair", "blue eyes", "animal ears", "solo", "looking at viewer", "shirt", "skirt"],
  "lang": "ja",
  "source": "identity_cond",
  "meta": {
    "rating": "g", "post_id": 11620731, "n_tags": 8, "tmpl": 2,
    "identity_tags": ["1girl", "long hair", "blue eyes", "animal ears"],
    "scene_tags": ["solo", "looking at viewer", "shirt", "skirt"]
  }
}
```

| フィールド | 説明 |
|---|---|
| `source` | `"identity_cond"`。#1 は `"synthetic"`・将来 `"llm_distill"` (§12)。 |
| `meta.identity_tags` | 同一性タグ列 (target の正準順序部分列・バケット 1 のみ・必ず非空)。 |
| `meta.scene_tags` | シーンタグ列 (target の残り・正準順序部分列)。`identity ∪ scene == tags`。 |

- A2 は `meta.identity_tags` / `meta.scene_tags` から §13.1 の 2-`<sep>` 系列を組む。
  #1 の `load_pairs` / `build_sequence` でこの行を読んでも **追加 meta キーは無視され壊れない**
  (後方互換を実測確認・§13.6)。

### 13.5 検証・リーク防止 (source 非依存で全件)

`validate_identity` = #1 の `validate_pairs` を全件適用 + identity 固有検査:

- OOV 0 (target/identity/scene の全タグが vocab 内)・負語 0 (`NEGATIVE_BLOCKLIST`)・
  タグ順序 §5 正準順。
- `identity ∪ scene == target` (集合一致)・identity ∩ scene == ∅・identity ⊆ target・
  identity は全要素がバケット 1。
- **リーク防止 (§8 踏襲)**: 同一 `post_id` / 同一 `text` が train/val に跨らない (0 件)。
- **同一性汎化チェック (新)**: identity_tags の組の train/val 重複を測り stats に記録
  (`train_val_identity_overlap_rate`)。重複しすぎると同一性の汎化を測れない。
- **tokenizer 往復**: `dollma_make_identity_pairs.py` の `_Tok` が tokenizer.hpp /
  `train_bitnet.Tokenizer` と同一の normalize / specials id / `tags[i].id==5+i` を再現し、
  target/identity/scene を encode→decode して **UNK 0・完全一致** を検査する。

### 13.6 A1 実測 (確定ライン)

- **出力ファイル**: `data/bitnet/pairs.identity.{train,val}.jsonl` + `stats.identity.json`。
  **#1 (`pairs.{train,val}.jsonl`) とは別ファイル** にして #1 を汚さず、A2 が両方読んで混合する。
- 元データ: #1 と同じ `cache/danbooru_posts.jsonl` (8,200 posts・タグのみ・画像非取得)。
- **総 5,000 (train 4,500 / val 500, split 0.9/0.1)・seed 20260620**。
  lang ja 0.498 / en 0.502・source identity_cond 1.0・rating g 1.0・ユニーク text 比率 1.0。
- **分離分布**: identity/pair mean 5.81 (min 1 / max 16)・scene/pair mean 9.79 (min 0 / max 15)・
  target/pair mean 15.6。distinct identity 語彙 265 / distinct scene 語彙 1920。
- **同一性汎化**: unique identity sets 4,155/5,000 (0.831)・**train/val identity 重複率 0.2531**
  (val 識別子の ~75% が train 未出 = 汎化を測れる)。
- identity 頻出: 1girl 3002 / long hair 2266 / short hair 1396 / black hair 984 /
  blue eyes 952 / multiple girls 923 / 1boy 909 …
- scene 頻出: solo 2697 / looking at viewer 2063 / shirt 1934 / long sleeves 1409 /
  dress 1058 / white shirt 1013 / bow 1011 …(服飾・状態は scene に寄っている)。
- **全件検証 0**: OOV 0 / 負語 0 / 非正準順序 0 / post_id リーク 0 / text リーク 0 /
  identity∪scene≠target 0 / identity-not-in-target 0。**tokenizer 往復 UNK 0・mismatch 0**。
- **後方互換実測**: 既存 `pairs.train.jsonl` (synthetic 4,500) を `train_bitnet.load_pairs` +
  `build_sequence` で無改修ロード OK。新 `pairs.identity.train.jsonl` も同経路でロード OK
  (追加 meta キーは #1 リーダに無視される)。

### 13.7 再現手順

```sh
py -3.12 scripts/dollma_make_identity_pairs.py --n 4500 --val 500 --seed 20260620     --vocab data/bitnet/vocab.json --out-dir data/bitnet
```

キャッシュ (`cache/danbooru_posts.jsonl`) があれば再利用し、無ければ #1 と同じ keyset
ページングで取得する (タグ + rating のみ・画像非取得・§1.2/§1.3)。

### 13.8 引き渡し

- **model-trainer** (`train_bitnet.py` / A2): `pairs.identity.{train,val}.jsonl` を #1 と
  **混合**して読む。identity_cond 行は `meta.identity_tags`/`scene_tags` から §13.1 の
  2-`<sep>` 系列を組む。synthetic 行は従来どおり 1-`<sep>`。tokenizer / vocab は不変。
- **cpp-implementer** (`tokenizer.hpp`): **変更不要**。`<sep>` を 2 回受けても decode は
  構造トークンとして読み飛ばす。vocab.json も不変。
- **prompt-engineer**: §13.2 の identity/scene 分離規則 (特に「服=scene」) を共有し、
  compose_prompt の canonical/scene 配置と矛盾しないことを確認済み (canonical=identity・
  scene 各バケット=scene に対応)。

### 13.9 B (品質スコアラ) の扱い

B (アニメ品質スコアラ・§11 蒸留) の正解ラベルは **別仕様**とする (本 A1 では扱わない)。
§11 の合格/不合格蓄積を teacher にする設計時に、本 §13 とは独立の dataset として起こす。

## 14. diverse-val 評価データセット (施策 C・training-spec §13)

施策 C (評価作り直し) 用の **評価専用**データセット。訓練には一切使わない。従来 val が
テンプレ 3 種 (§3.3) に偏り「テンプレに合うか」しか測れなかったのを、**自然文側だけを多様化**
した out-of-template の val で「実ユーザーの自由文への汎化」を測る。構築は
`scripts/dollma_make_eval_diverse.py`。出力 `pairs.eval_diverse_a.jsonl` / `_b.jsonl` 各
**1,500 行** (post×3 variant) は**凍結** = 再現性アンカー (既存は --force でのみ再生成)。

### 14.1 不変条件 (厳守)

- **tags-stay-real (絶対方針)**: gold タグ = post の**実 danbooru タグ**で固定。**生成散文から
  タグを抽出/推測しない** (LLM にタグを作らせない — #1/D5/D6 と同じ「タグは実 danbooru」原則)。
  多様化するのは**自然文 (入力) のみ**。
- **既存 val バイト不変**: `pairs.val.jsonl` は読むだけ。新ファイルは**加算のみ**。
- **gold⊆vocab**: 全 gold タグが vocab.json 内 (UNK 0)。Pool B は vocab 射影 + 正準順序
  (§5) で整形 (`dollma_make_pairs` のタグ整形部のみ流用・二重実装禁止)。
- **リーク 0**: Pool B の post_id は train (`pairs.train` ∪ `pairs.identity.train`) とも val とも
  非交差 (段a/段c 双方で assert)。

### 14.2 Pool 定義

- **Pool A** = `pairs.val.jsonl` 由来 (in-distribution の gold)。gold タグはその val 行の `tags`
  を**バイト一致コピー**。「既知 val と同じ gold・散文だけ多様化」した対照群。
- **Pool B** = `cache/danbooru_posts.jsonl` から train∪val 非交差の未使用 post を抽出し正準順序
  整形 (タグ 4 件未満は除外)。「val にすら無い新規 post」への汎化を測る本命群。

### 14.3 構築 3 段 (人手散文を凍結に取り込む)

散文は LLM/テンプレでなく **このセッションの main Claude が著述**する (多様で自然な日本語/英語)。

1. **段a `--emit-prompts`**: gold タグ→「散文生成プロンプトのバッチ」(`eval_diverse_prompts.jsonl`)
   を出力。**散文は一切生成せず** gold タグの列挙と lang_hint (post_id で決定的に ja/en) のみ。
   ここでリーク検査・gold⊆vocab を assert。
2. **段b (main Claude)**: 各 (post_id, subset, variant) に対し gold タグを表す自然文を著述し
   `eval_diverse_texts.jsonl` へ。post_id を本文に漏らさない。
3. **段c `--ingest`**: prompts (gold の唯一のソース) と texts を突合・検証 (text 非空 / post_id 非漏出
   / lang∈{ja,en} / gold⊆vocab / リーク0) し `pairs.eval_diverse_{a,b}.jsonl` に**凍結**。
   スキーマ `source:"eval_diverse"`・`meta.{post_id,subset,variant,gen:"claude",rating,n_tags}`。

> 中間ファイル (`eval_diverse_prompts*.jsonl`・`eval_diverse_texts*.jsonl` / `_partNN`) は再生成可能。
> 凍結アンカーは `pairs.eval_diverse_a.jsonl` / `_b.jsonl` の 2 本のみ。

### 14.4 用途

`train_bitnet.py --eval-only` が生成 set-metrics (training-spec §13.2) で採点する本命 val。
施策 C 以降の B/A/D/F は**この diverse-val 上の生成 F1** を主要オフライン指標とする
(テンプレ teacher-forcing recall は非回帰アンカーとして残すのみ)。

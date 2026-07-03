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

## 15. diverse-train (Claude 著述 Replace パイロット・施策 B・training-spec §14)

施策 B (入力多様化) 用の**訓練データ**。§14 diverse-val (評価専用) と**対の構成**で、
評価で「自然文だけ多様化」した out-of-template を測れるようにしたのと同じ方針を**訓練側**に
適用する。テンプレ 3 種 (§3.3) の偏りを実世界の自由文で置き換える。構築は
`scripts/dollma_make_diverse_train.py`。出力 `data/bitnet/pairs.train.diverse_b.jsonl`
(著述 500 + synthetic 4,000 = **総 4,500 件**) + `stats.diverse_b.json`。**本パイロットの
訓練重みは別名 `bitnet_dense_diverse_b` で本番 (`bitnet_dense`) は無改変**。

### 15.1 不変条件 (厳守)

- **tags-stay-real (絶対方針)**: 各ペアの gold タグ = post の**実 danbooru タグ**で固定。
  著述する自然文 (入力) からタグを抽出/推測しない (§14 diverse-val・#1/D5/D6 と同じ原則)。
  多様化するのは**自然文 (入力) のみ**・tags は**バイト不変**。
- **Replace (件数維持)**: 総件数 4,500 を維持 = 既存 train の 4,500 件のうち seed 決定的に
  選んだ 500 件の**自然文だけを著述文へ置換**、残 4,000 件は synthetic テンプレのまま。
  件数を増やさないので「件数増による改善」と「多様化による改善」を分離できる。
- **既存 train バイト不変**: `pairs.train.jsonl` は読むだけ・新ファイルは加算のみ。
- **gold⊆vocab**: 全 gold タグが vocab.json 内 (UNK 0)。

### 15.2 構築 3 段 (人手散文を訓練に取り込む)

散文は LLM/テンプレでなく **このセッションの main Claude が著述** (外部 API/Qwen2/ネット不使用)。

1. **段a `--emit-prompts`**: 既存 train から seed 20260620 で**決定的に 500 件抽出** → 著述
   プロンプトを出力。ここで **3 assert**: ① 抽出⊆train (選んだ post が train に存在)
   ② **val 非交差** (`pairs.val` と post_id 重複なし) ③ **diverse-val 非交差**
   (`pairs.eval_diverse_{a,b}` の post と重複なし = 評価リーク厳禁)。加えて gold⊆vocab を assert。
2. **段b (main Claude)**: 各抽出 post の gold タグを表す自然文を著述。post_id を本文に漏らさない。
3. **段c `--ingest`**: 著述 texts を抽出 prompts と突合・検証し凍結。**tags バイト不変**・
   post_id 非漏出・**件数 4,500**・synthetic 残 4,000・**重複 0** を assert。

### 15.3 スキーマ (後方互換)

- 著述 500 行: `source:"llm_distill"` + `meta.gen:"claude"` + `meta.tmpl:-1` (テンプレ非由来の印)。
  D2 (§12) の `source:"llm_distill"` スキーマを流用。`tags` は元 post のバイト一致コピー。
- synthetic 4,000 行: 既存 `pairs.train` 由来のテンプレ生成行 (`source:"synthetic"`)。
- 訓練の build_sequence では `llm_distill` 行も synthetic と同じ `1-<sep>` 経路を通る (D2 先例)。

### 15.4 用途・統計

`train_bitnet.py --train-file data/bitnet/pairs.train.diverse_b.jsonl` で訓練し、§14 diverse-val
上の生成 F1 で採点 (training-spec §14.3/§14.4)。統計は `stats.diverse_b.json` に記録
(著述/synthetic 件数・重複 0・gold⊆vocab・リーク検査結果)。検証 `test_dollma_make_diverse_train.py`
6/6 緑 (段a 抽出決定性・3 assert・段c tags バイト不変・件数)。

### 15.5 再現手順

```sh
py -3.12 scripts/dollma_make_diverse_train.py --emit-prompts --n 500 --seed 20260620
#   (段b: main Claude が data/bitnet の著述 texts を埋める)
py -3.12 scripts/dollma_make_diverse_train.py --ingest
```

### 15.6 B-2 件数拡大 (Replace 500 → 2,000・training-spec §14.8)

§15.1〜15.5 のパイロット (Replace 500) は記録として残し、本節は**著述件数を 2,000 に拡大**した
データセットを追記する。**不変条件 (§15.1) はそのまま**: tags-stay-real・Replace で総件数 4,500
維持 (著述 **2,000** + synthetic **2,500**)・既存 train バイト不変・gold⊆vocab。著述は既存 500
(`diverse_train_texts_part01-05.jsonl`) に**新規 1,500** (`part06-20.jsonl`) を積み増し。

- **構築**: §15.2 と同 3 段。段a は `--emit-prompts --n 2000 --seed 20260620` で決定的に 2,000 件
  抽出 (3 assert: 抽出⊆train / val 非交差 / **diverse-val 非交差** / gold⊆vocab)。段b で main Claude
  が新規 1,500 を著述 (既存 500 は流用)。段c `--ingest` で突合・凍結 (tags バイト不変・post_id
  非漏出・件数 4,500・synthetic 残 2,500・重複 0)。
- **出力** (本番無改変・別名のみ・gitignore・再生成可):
  - `data/bitnet/pairs.train.diverse_b2000.jsonl` (著述 2,000 = `source:"llm_distill"` +
    `meta.gen:"claude"` + `meta.tmpl:-1` / synthetic 2,500)
  - `data/bitnet/stats.diverse_b2000.json` (著述/synthetic 件数・重複 0・gold⊆vocab・リーク検査)
  - 中間: `diverse_train_prompts_b2000.jsonl` / `diverse_train_todo_b2000.jsonl` /
    `diverse_train_texts_part06-20.jsonl`
  - 訓練重み (別名) `bitnet_dense_diverse_b2000{,_fp32}.safetensors` / `train_stats_diverse_b2000.json`
  - 採点レポート `eval_report_diverse_b2000.json`・sweep scratch `data/bitnet/_seedsweep_b2000/`
- **結果** (training-spec §14.8): 件数 500→2,000 で diverse 生成 macro F1 が単調に伸び
  (diverse_a 0.2675→**0.3212** / diverse_b 0.3039→**0.3670**)・in-dist は誤差内据え置き
  (汎化方向)・sweep 4 seed で delta が ~1.4–1.5x 拡大しつつ seed sd は縮小 (全判定軸成立)。
  **入力多様化のスケール則**を確認。
- **決裁 (2026-06-24・ユーザー)**: **レシピ既定化を確定** — 今後の訓練 (A/D/F) は多様化入力
  (tags-stay-real) を既定レシピとする。**正典重み `bitnet_dense{,_fp32}.safetensors` と C++ 推論
  golden の差し替えは A 実ペアと束ねる次の出荷リトレインで1回** (golden チャーンを集約)・当面 #1
  重みは据え置き。`data/bitnet/pairs.train.diverse_b2000.jsonl` は実験出力に留まらず、**A 出荷
  リトレインの既定 train ソース**として位置づける (A の新規実ペアも同じ tags-stay-real 機構で
  diverse train へ合流)。遅延条項の詳細は roadmap `[B-merge-at-A]` / training-spec §14.8.3。

### 15.7 再現手順 (件数 2,000 版)

```sh
py -3.12 scripts/dollma_make_diverse_train.py --emit-prompts --n 2000 --seed 20260620
#   (段b: main Claude が data/bitnet の著述 texts part06–20 を埋める)
py -3.12 scripts/dollma_make_diverse_train.py --ingest
```

### 15.8 B-3 件数拡大 (Replace 2,000 → 10,000・training-spec §14.9)

§15.6 (Replace 2,000) を記録として残し、本節は**著述件数を 10,000 に拡大**したデータセットを追記する。
**不変条件 (§15.1) はそのまま**: tags-stay-real・Replace 構成・既存 train バイト不変・gold⊆vocab。

- **構築 (k-per-post 一般化)**: `dollma_make_diverse_train.py` を `--n-posts`/`--k-per-post` に一般化し、
  **P=2,500 unique post × k=4 variant = 著述 10,000** を生成 (variant_idx/style_hint 付き・k=1 で
  B-1/B-2 と bitwise 非回帰・test 14/14 緑)。**B-2 のスーパーセット** (variant 0 = B-2 著述 part01–20 を
  再利用・新規 part21–36 に各 500 = +8,000)。Replace 後の総 train **12,000** (著述 **10,000** +
  synthetic **2,000**)。段a で 3 リーク assert (抽出⊆train / val 非交差 / diverse-val 非交差・
  スーパーセット assert)・段c `--ingest` で突合・凍結 (tags バイト不変・post_id 非漏出・ja/en 5,000/5,000・
  uniq テキスト 9,997・リーク 0)。
- **出力** (本番無改変・別名のみ・gitignore・再生成可):
  - `data/bitnet/pairs.train.diverse_b10k.jsonl` (12,000 行) / `data/bitnet/stats.diverse_b10k.json`
  - 中間: `diverse_train_prompts_b10k.jsonl` / `diverse_train_todo_b10k.jsonl` /
    `diverse_train_texts_part01–36.jsonl` (著述スキーマ `{post_id,variant_idx,lang,text}`・UTF-8)
  - 訓練重み (別名) `bitnet_dense_diverse_b10k{,_fp32}.safetensors`・sweep scratch
    `data/bitnet/_seedsweep_b10k/`
- **結果 (training-spec §14.9)**: sweep 4 seed で delta(B10k−#1) は全 set/metric で seed 頑健 (判定 YES)
  だが、**~2,000 件で飽和** — 2,000→10,000 の 5 倍増で delta は平坦 (diverse_a F1 +0.1472→+0.1411 /
  diverse_b +0.1788→+0.1761・seed 分散内)・b 絶対値も頭打ち (0.319→0.313 / 0.361→0.359)。
  §15.6 で確認した「スケール則」は **~2,000 で頭打ち**と訂正される (効果が seed 頑健に正である点は不変)。
- **運用知見**: 出荷リトレインの既定多様化ファイルは **b2000 で足りる** (b10k を作る必要はない)。残る
  低帯域は B 件数ではなく **A 実ペア増 / D 容量増**で取りに行く。**B 著述を 2,000 超に積む価値は薄い**。

## 16. 品質スコアラ教師データ (Phase 4 Model B・B-3a)

Model B (アニメ品質スコアラ・§13.9 で「別仕様」と予告) の **ScorerNet 蒸留教師データ**。
#1〜§15 の「user text → タグ列」ペアとは**全く別のデータセット**で、`pairs.*.jsonl` も
vocab も**一切共有しない**。ScorerNet (純 conv backbone・入力 `[1,3,512,512]`・出力 `[1,1+8]`)
を蒸留訓練するための **(画像参照, 品質スカラ, 8 軸 soft target)** を持つ。

**重要 (本節の現況)**: 本節は **配管のみ確定・実走の数値は研究機待ち (未実走)**。SDXL 生成と
WD14 実推論は研究機 (RTX5080+OV) 前提で、非研究機ではスクリプトを乾式検証したのみ。

### 16.1 教師は 2 系統 (plan 確定)

- **(A) 解剖 8 軸 soft target** — `src/infer/quality_gate.hpp::catalog()` / `enum AnomalyAxis`
  を流用。WD14 sigmoid スコアベクタ `[N_TAGS]` を **「軸ごとに該当タグ sigmoid の最大値」**
  (max-sigmoid 集約) で連続値化する。**QualityGate (C++ Stage1) は閾値 hit だが、蒸留教師は
  閾値なしの連続 soft 値**にする (ここが Stage1 との差)。軸順 = `AnomalyAxis` に 1:1:
  `0 Hands / 1 Limbs / 2 Head / 3 Eyes / 4 Ears / 5 Mouth / 6 Digits / 7 GlobalAnatomy`。
  catalog 18 エントリは実 WD14 `selected_tags.csv` で **18/18 解決**を確認。
- **(B) 品質スカラ** — 外部美的モデルで採点 (§16.9 選定)。`QualityProvider` でプラガブル:
  - `passthrough` — 入力レコードに `quality` があれば通す・無ければ `None` (後埋め境界)。
  - `waifu_scorer_v4` — Eugeoter/waifu-scorer-v4-beta (**apache-2.0**)・CLIP ViT-L/14 image
    embed[768]→MLP→生スコア 0..10→/10 で [0,1] (§16.9 候補)。
  - `anime_aesthetic_deepghs` — deepghs/anime_aesthetic (**openrail**)・ONNX swinv2pv3 448→
    7 段順序クラス確率→**期待順序スコア** [0,1] (§16.9 候補・写像は §16.9)。
  - `anatomy_proxy` — 縮退案 (`1 - max(8 軸)`)。**ユーザー方針 = 縮退保留** につき既定で使わず、
    許諾的アニメ美的モデルが使えない場合は quality=`null` で止める (provenance に proxy 印)。
  - **本機では美的モデルの重み DL・推論はしない** (純計算の正規化/期待値写像のみ確定)。重みは
    研究機で配置・採点 (matting ISNet=Apache と同立ち位置)。

### 16.2 ファイル構成

```
data/scorer/
  scorer_wd14.jsonl              — WD14 スコアベクタ入力 (研究機で生成・gitignore)
  scorer.train.jsonl            — 訓練サンプル (gitignore・再生成可)
  scorer.val.jsonl              — 検証サンプル (gitignore)
  scorer_stats.json             — provenance + 軸分布 (gitignore)
  scorer.train.example.jsonl    — スキーマ例 (追跡・配管確認用)
  scorer_stats.example.json     — provenance スキーマ例 (追跡)
scripts/dollma_make_scorer_labels.py    — WD14 スコア → 8 軸 soft + 品質スカラ (OV/SDXL 非依存)
scripts/dollma_gen_scorer_corpus.py     — SDXL 生成 + WD14 タグ付け driver (研究機専用・既定 dry-run)
scripts/tests/test_dollma_make_scorer_labels.py — 乾式テスト (torch/OV 不要・5/5 緑)
```

### 16.3 スキーマ (1 サンプル)

```json
{
  "image": "data/scorer/img/000001.png",
  "quality": 0.41,
  "axis": [0.63, 0.48, 0.0, 0.0, 0.0, 0.0, 0.22, 0.71],
  "meta": {"prompt": "2girls, complex pose, ...", "rating": "g", "n_tags_in": 10861}
}
```

| フィールド | 型 | 説明 |
|---|---|---|
| `image`   | string | 画像参照パス (ピクセルは別保持・jsonl にピクセルを埋めない)。 |
| `quality` | float\|null | 品質スカラ (B 系統)。外部美的モデル未確定の間は provider 由来 or `null`。 |
| `axis`    | float[8] | 8 軸 soft target (A 系統・`AnomalyAxis` 順・各 [0,1])。 |
| `meta`    | object | `prompt` (生成プロンプト)・`rating`・`n_tags_in` (WD14 出力長)。 |

ScorerNet 出力 `[1,1+8]` の `index0=quality` / `index1..8=axis[0..7]` に 1:1 対応。

### 16.4 WD14 スコア入力境界 (研究機と非研究機の分業)

`dollma_make_scorer_labels.py` は **WD14 実推論を呼ばない**。入力は研究機で WD14 を回した
結果の `scorer_wd14.jsonl` (1 行 = `{image, wd14_scores:[N_TAGS], prompt?, rating?, quality?}`)。
生成・推論は研究機 (`dollma_gen_scorer_corpus.py --run`)、ラベル化は非研究機でも可、の分業。

### 16.5 分割・provenance・再現性 (§8 踏襲)

- **seed 固定** (既定 20260620)・`random.Random(seed)` でシャッフル後に train/val 分割
  (既定 val_ratio 0.1)。同 seed でバイト一致を乾式テストで確認。
- **image リーク防止**: 同一 `image` が train/val に跨らないことを assert。
- `scorer_stats.json` に provenance を残す: seed・教師モデル名/版 (anatomy=wd14+catalog /
  quality=provider)・出典・catalog 解決数・軸分布・`status` (未実走印)。

### 16.6 乾式検証結果 (このセッション・非研究機)

- catalog/AnomalyAxis 写しが `quality_gate.hpp` と件数一致 (18 エントリ・8 軸)。
  実 WD14 `selected_tags.csv` で **18/18 解決**。
- max-sigmoid 軸集約が合成スコアで正しい (軸ごと最大・範囲外 index 安全 skip)。
- schema 往復 (jsonl 書き→読み等価)・seed 決定性 (同 seed バイト一致)・image リーク 0・
  provider プラガブル (passthrough/anatomy_proxy)。`test_dollma_make_scorer_labels.py` 5/5 緑。
- driver は既定 dry-run で生成計画のみ出力。`--run` は非研究機で `NotImplementedError`
  (実生成・実推論しない安全ガード)。

### 16.7 研究機で実走するために残る手順

1. `dollma_gen_scorer_corpus.py --run` の `run_on_research_machine` を結線
   (SDXL 生成 = `Txt2ImgGenerator`/`dollama --prompt`・WD14 タグ付け = OV `[1,448,448,3]` f32)。
   良/悪を散らすジョブ設計 (品質ネガ ON/OFF) は実装済。まず数百枚 (配管優先・後段で拡大)。
2. 外部美的モデルを **ライセンス確認後**に `QualityProvider` 実装として差す
   (不可なら `anatomy_proxy` で縮退・PL 判断)。
3. `scorer_wd14.jsonl` → `dollma_make_scorer_labels.py` で `scorer.{train,val}.jsonl` 生成。
4. B-3b (`scripts/train_scorer.py`・model-trainer) が本データを消費して ScorerNet を蒸留訓練。

### 16.8 引き渡し

- **model-trainer** (B-3b): `data/scorer/scorer.{train,val}.jsonl` を読む。`axis`=8 軸 BCE/MSE
  target・`quality`=スカラ MSE/Huber target (null の間は B head を縮退 or 凍結)。seed 固定。
- **cpp-implementer** (B-3d): `QualityScorer` の `axis_scores[8]` は `AnomalyAxis` に 1:1
  対応させ、`QualityGate` と同じ軸語彙で B-5 FB ループに渡す (本節の軸順と厳密一致)。

### 16.9 美的モデル選定 (品質スカラ B 系統)

品質スカラ (B) の教師となるアニメ美的モデルを **一次情報 (HF API metadata + README/meta.json)**
で調査・比較した。採否基準は matting ISNet=Apache と同じ「許諾的ライセンス + アニメ特化」。

#### 候補比較 (一次情報)

| モデル | ライセンス | 形態 | アニメ特化 | 入力前処理 | 出力レンジ | 重み入手元 |
|---|---|---|---|---|---|---|
| **deepghs/anime_aesthetic** | **openrail** | ONNX (swinv2pv3 448 / caformer_s36) | ○ | img 0-255 NHWC 448²・正規化なし (WD14 系) | 7 段順序クラス確率 `[masterpiece..worst]` | HF `deepghs/anime_aesthetic/<bb>/model.onnx` |
| Eugeoter/waifu-scorer-v4-beta | **apache-2.0** | torch CLIP+MLP (768→…→1) | ○ | CLIP ViT-L/14 image 前処理 | 生スカラ 0〜10 | HF `…/model.safetensors` |
| Eugeoter/waifu-scorer-v3 | **表記矛盾** (metadata `openrail` / README `Apache 2.0`) | torch CLIP+MLP | ○ | CLIP ViT-L/14 | 0〜10 | HF |
| cafeai/cafe_aesthetic | agpl-3.0 (コピーレフト強) | torch (ViT 分類) | △ | ViT 前処理 | クラス確率 | HF |
| shadowlilac/aesthetic-shadow | unknown (明記なし) | torch (ViT) | ○ | ViT | クラス確率 | HF |
| skytnt/anime-aesthetic | **ライセンス記載なし** | ONNX | ○ | — | スカラ | HF |
| LAION improved-aesthetic | apache (汎用) | CLIP+MLP | ✕ (汎用・実写寄り) | CLIP | 0〜10 | GitHub |

#### 選定状況 (要ユーザー確認)

- **実装済みで利用可能な provider は 2 つ**: `waifu_scorer_v4` (apache-2.0・明確に許諾的) と
  `anime_aesthetic_deepghs` (openrail)。両方を `QualityProvider` として実装し、純計算部
  (正規化 / 期待値写像) を乾式テストで検証済 (test 9/9 緑)。
- **ライセンス判断の論点**: openrail は Apache/MIT のような無条件許諾とは別系統で、
  付属の **使用行動制限条項** (Use Restrictions) を伴う。本用途
  (教師でラベル生成 → 自作 ScorerNet に蒸留・教師自体は再配布しない) は再配布を伴わず
  抵触しにくいと解釈できるが、**openrail の採否は最終的にユーザー確認事項**。
- **本データ構築エージェントの一次調査結論は apache-2.0 の `waifu_scorer_v4`**
  (ライセンスが metadata/README とも一致しクリーン・「許諾的を優先」の基準に最も合致)。
  既定 provider は当面 `waifu_scorer_v4` のままとし、`anime_aesthetic_deepghs` は
  ユーザーが openrail を承認した場合に既定へ昇格する (どちらも実装済・切替は CLI 一語)。
  ※ skytnt はライセンス記載なし・waifu-scorer-v3 は表記矛盾のため見送り。cafe_aesthetic は
    AGPL でコピーレフトが強く回避。LAION は Apache だが汎用 (アニメ非特化) で見送り。

#### deepghs 期待順序スコア写像 (`anime_aesthetic_deepghs`)

7 段順序クラス `[masterpiece(0), best(1), great(2), good(3), normal(4), low(5), worst(6)]`
(meta.json labels) を連続 quality [0,1] に写像する:

```
各クラス index k に均等値  v_k = (6 - k) / 6   (masterpiece=1.0 … worst=0.0・線形等間隔)
quality = Σ_k p_k · v_k     (softmax 確率との期待値・p が確率分布なら結果は必ず [0,1])
```

`AnimeAestheticDeepghsProvider.expected_score` が純計算で実装 (本機実走可)。one-hot
masterpiece→1.0 / worst→0.0 / uniform→0.5 を乾式テストで検証。確率が非正規化でも内部で
正規化し [0,1] にクランプ。7 クラス確率は研究機 ONNX 推論で採り `aesthetic_probs` として
入力行に積む (推論本体は `dollma_gen_scorer_corpus.py --run` の研究機実走部)。

#### waifu_scorer_v4 写像 (`waifu_scorer_v4`)

CLIP ViT-L/14 image embed `[768]` → MLP `768→2048→512→256→128→32→1` (BatchNorm 入り・
safetensors header で確認) → 生スコア 0〜10 → `normalize_score` で `/10` クランプ [0,1]。
CLIP image embedding (`clip_image_embed`[768]) と MLP 重みは研究機で配置・採点。

#### 入力境界 (研究機との分業・WD14 と同方針)

ラベル化スクリプト (`dollma_make_scorer_labels.py`) は **美的モデル推論を呼ばない**。研究機で
推論した結果を入力行に積む境界:
- `anime_aesthetic_deepghs` → `aesthetic_probs`:[7] (softmax 済み 7 クラス確率)。
- `waifu_scorer_v4` → `clip_image_embed`:[768] (+ `--quality-weights` に MLP safetensors)。
これらが無い本機では quality=`null` のまま (縮退に落とさない=ユーザー保留方針)。

#### 乾式検証 (このセッション・追加分)

- 候補ライセンス/形態を HF API + meta.json + safetensors header で一次確認。
- `waifu_scorer_v4`: `normalize_score` 0/10/7.5→0/1/0.75・範囲外クランプ・重み/embed 未配置で
  `None`・既定 provider・provenance (apache-2.0/repo) を検証。
- `anime_aesthetic_deepghs`: `class_values` (1.0…0.0)・`expected_score` (one-hot/uniform/
  非正規化/全ゼロ→None/長さ不一致→ValueError)・確率未配置で `None`・provenance
  (openrail/labels/期待値写像) を検証。
- `test_dollma_make_scorer_labels.py` **9/9 緑** (torch/OV/SDXL 不要)。

## 17. Phase 4-A 実ペア増 (12k / 25k) — 実ペア増 Phase 1

§13 (同一性条件付きペア) の **実ペア増**。A 単体の伸びしろ ([input-diversification]
スケール則が ~2,000 で飽和 → 残低帯域は「A 実ペア増 / D 容量増」で取る) を測るため、
§13 と同一機構・同一スクリプト (`scripts/dollma_make_identity_pairs.py`・本体無改修) で
A1 5k → **12k / 25k** に増やした **別名出力** を生成する。施策 B 多様化は被せない
(A 単体で清潔に測る)。本番重み (`bitnet_dense{,_fp32}.safetensors`)・全 golden・A1 5k
(無印 `pairs.identity.{train,val}.jsonl` / `stats.identity.json`)・凍結 eval
(`pairs.eval_diverse_{a,b}.jsonl` / `eval_frozen_post_ids.json`)・#1 本線
(`pairs.{train,val}.jsonl`) はすべて無改変。

### 17.1 スクリプト引数 (本体無改修・§13 のまま)

`dollma_make_identity_pairs.py` の Phase 4-A 既設引数 (§13 実装済) のみ使用:

- `--out-tag <tag>`: 出力ファイル名サフィックス (例 `a12k` → `pairs.identity.{train,val}.a12k.jsonl`
  / `stats.identity.a12k.json`)。空で従来名 (A1 5k と bitwise 非回帰)。
- `--exclude-post-ids <path>`: 凍結 eval 等の post_id 集合 JSON。A train から恒久除外
  (生成ループで skip + 末尾 assert で 2 重保証)。

### 17.2 件数・実測 (seed 20260620・rating g・B 多様化非適用)

| 指標 | A1 5k (§13.6) | **a12k** | **a25k** |
|---|---|---|---|
| train | 4,500 | **10,800** | **22,500** |
| val | 500 | **1,200** | **2,500** |
| total | 5,000 | 12,000 | 25,000 |
| teacher_retention | 1.0 | **1.0** | **1.0** |
| vocab retention (target tags in vocab) | 1.0 | **1.0** (186,332/186,332) | **1.0** (386,744/386,744) |
| tokenizer 往復 | UNK 0・完全一致 | **UNK 0・完全一致** | **UNK 0・完全一致** |
| identity 重複率 (val 基準) | 0.253 | **0.2871** (329/1,200) | **0.3003** (698/2,500) |
| unique_identity_ratio | — | 0.7863 | 0.7361 |
| id/pair mean · scene/pair mean | — | 5.65 · 9.88 | 5.59 · 9.88 |
| lang ja:en | — | 0.501:0.499 | 0.502:0.498 |
| uniq_text | — | 1.0 | 1.0 |

- **identity 重複率は件数増で漸増** (5k 0.253 → 12k 0.2871 → 25k 0.3003)。同一性集合の
  被覆が広がる中でも、val 側の約 70% は train 未見の identity 集合 (汎化を測れる領域が大きい)。
- target タグの **vocab retention 両スケール 1.0** = OOV ゼロ (target は vocab 内タグのみを
  射影する設計どおり)。

### 17.3 リーク 0 証跡 (stats JSON `phase4a_exclusion`・両スケール)

両 `stats.identity.{a12k,a25k}.json` で以下が全て 0 / disjoint:

| 項目 | a12k | a25k |
|---|---|---|
| `excluded_count_loaded` (frozen 集合サイズ) | 1,000 | 1,000 |
| `excluded_skipped_in_gen` (生成で skip した post) | 428 | 670 |
| `frozen_eval_pids_read_direct` (eval pairs 直読の post_id) | 1,000 | 1,000 |
| `leak_train_x_frozen_eval` | **0** | **0** |
| `leak_all_x_frozen_eval` | **0** | **0** |
| `leak_train_x_excluded` | **0** | **0** |
| `leak_all_x_excluded` | **0** | **0** |
| `post_id_train_val_disjoint` | true | true |
| `text_train_val_disjoint` | true | true |

`leak_*_x_frozen_eval` は `eval_frozen_post_ids.json` の集合経由でなく、凍結 eval の
`pairs.eval_diverse_{a,b}.jsonl` を **直読** した 1,000 post_id との交差 (抽出ロジック非依存の
独立再検証)。検証ログも両スケールで `検証 OK` (validate_identity 0 件)・`tokenizer 往復 OK`。

### 17.4 fetch-factor・cache 取り扱い

`fetch_posts` は cache (`data/bitnet/cache/danbooru_posts.jsonl`) を再利用し、不足分のみ
**最古 id から過去方向に延伸** (既存を捨てない)。

- **a12k**: `--fetch-factor 1.8` 既定。必要 `(10,800+1,200)*1.8+50 ≈ 21,650` posts <
  cache 38,000 → **API 取得なし・cache 無改変**。
- **a25k**: `--fetch-factor 2.2`。必要 `(22,500+2,500)*2.2+50 ≈ 55,050` posts > cache 38,000
  → cache を **38,000 → 55,200 posts へ過去方向延伸** (約 86 バッチ・`--sleep 1.0`)。
- 両スケールとも **歩留り不足なし** (full target 達成・fetch-factor を上げる再実行は不要)。

**cache 退避手順 (A 専用に本番 cache を汚さない)**: A 着手前に現 cache を
`danbooru_posts.jsonl.preA` へ退避 (過去方向延伸で a25k が cache を書き換えるため)。

```sh
# A 着手前 (1 回)
cp data/bitnet/cache/danbooru_posts.jsonl data/bitnet/cache/danbooru_posts.jsonl.preA
```

### 17.5 再現手順 (seed 20260620・B 多様化非適用)

```sh
# 0. cache 退避 (§17.4)
cp data/bitnet/cache/danbooru_posts.jsonl data/bitnet/cache/danbooru_posts.jsonl.preA

# 1. 12k (cache 充足で API 取得なし)
py -3.12 scripts/dollma_make_identity_pairs.py --n 10800 --val 1200 \
    --seed 20260620 --out-tag a12k --sleep 1.0 \
    --exclude-post-ids data/bitnet/eval_frozen_post_ids.json

# 2. 25k (cache を 38,000 → 55,200 posts へ過去方向延伸)
py -3.12 scripts/dollma_make_identity_pairs.py --n 22500 --val 2500 \
    --seed 20260620 --out-tag a25k --sleep 1.0 --fetch-factor 2.2 \
    --exclude-post-ids data/bitnet/eval_frozen_post_ids.json
```

歩留り不足警告 (`[id-pairs] 警告: 歩留り不足`) が出たら `--fetch-factor` を上げて再実行
(12k は 1.8→2.2→2.6、25k は衝突・歩留り低下が効くので 2.2 始まり)。今回は両スケールとも
不足なし。

### 17.6 成果物 (すべて gitignore・再生成可)

- `data/bitnet/pairs.identity.{train,val}.a12k.jsonl` (10,800 / 1,200) + `stats.identity.a12k.json`
- `data/bitnet/pairs.identity.{train,val}.a25k.jsonl` (22,500 / 2,500) + `stats.identity.a25k.json`
- cache バックアップ `data/bitnet/cache/danbooru_posts.jsonl.preA` (38,000 posts・A 着手前状態)

スキーマは §13 (同一性条件付きペア) と完全同一 (`source:"identity_cond"` + `meta`)。Phase 2
(再訓練 + diverse-val eval・§14) は本番重み #1 据え置きのまま別名出力で評価する。

### 17.7 a12k seed sweep 結論と a25k の扱い (A クローズ・training-spec §9.10)

Phase 2 (再訓練 + eval) は **a12k のみ** を 4 seed sweep で評価し、A をクローズした (training-spec
§9.10)。結論: A 実ペア増の効果は diverse-val 生成 F1 ではなく **identity retention (across-seed
0.9748 ± 0.0010・全 seed 頑健)**。diverse-val F1/Jaccard は 4 set/metric とも seed ノイズ
(seed 42 のみ符号反転・#1 分散帯 sd 以下・D6 と同型) で施策 B ~2,000 飽和とも整合。

**a25k (train 22,500 / val 2,500) は回さず未使用のまま保持する** (削除しない)。理由: 12k で
diverse-val が seed ノイズである以上、25k で符号反転が安定化する見込みは薄く、計算コストに
見合わない。生成済 `pairs.identity.{train,val}.a25k.jsonl` / `stats.identity.a25k.json` は
将来 (A/D 容量増と束ねた再評価・retention のスケール検証等) のため§17.6 の成果物としてそのまま残す。

## 18. Phase 4-D 容量増 sweep のデータ (新規データ無し・既存流用)

施策 D (容量増 33M→80M・training-spec §16) は**新規データセットを一切作らない**。c33/d80 両
アームとも完全同一レシピ = b2000 多様化 train (§15.6 `pairs.train.diverse_b2000.jsonl`) ∧ a12k
identity (§17.6 `pairs.identity.{train,val}.a12k.jsonl`)・凍結 diverse-val (§14) で、唯一の差分は
モデル容量 (`--arch d80m`) のみ。`scripts/dollma_d_seedsweep.py` の `setup_sweep_dir()` が上記
正準ファイルを `data/bitnet/_seedsweep_d80m/` へ sha 一致確認付きコピーするだけ (A/B sweep と同手)。

**結論 (D クローズ・陰性確定・training-spec §16)**: 容量倍増では diverse-val F1 が seed ノイズ内
(across 平均わずかに負・c33 分散帯 sd 以下)・retention 3/4 seed 床割れ・in-dist 微退行 →
**80M 不採用・勝者 = 33M (b2000 ∧ a12k identity)**。diverse-val F1 を頑健に押し上げたのは入力
多様化 (B・~2,000 飽和) のみで、蒸留 (D5/D6)・実ペア増 (A)・容量 (D) は非寄与/seed ノイズと確定。
sweep 成果物は `_seedsweep_d80m/` 配下のみ (gitignore)・本番データ無改変。

## 19. 正典化まとめ焼き `[B-merge-at-A]` のデータ (新規データ無し・既存流用・2026-07-03)

正典化まとめ焼き (training-spec §17) は **新規データセットを一切作らない**。既存の正準ファイルを
`data/bitnet/_merge_ba/` へ正準名でステージし、B(多様化) train を `--train-file` で足すだけ。

### 19.1 _merge_ba ステージング (正準名コピー・ソース read-only)

`scripts/dollma_a_seedsweep.py` の `_copy_canonical` 作法を踏襲し、以下を正準名でコピー (行数照合済)。

| dst (正準名・_merge_ba) | src | 行数 |
|---|---|---|
| vocab.json | data/bitnet/vocab.json | 29985 |
| pairs.val.jsonl | data/bitnet/pairs.val.jsonl (不変アンカー) | 500 |
| pairs.identity.train.jsonl | data/bitnet/pairs.identity.train.**a12k**.jsonl (§17) | 10800 |
| pairs.identity.val.jsonl | data/bitnet/pairs.identity.val.**a12k**.jsonl (§17) | 1200 |
| pairs.eval_diverse_a.jsonl | data/bitnet/pairs.eval_diverse_a.jsonl (§14 凍結物差し) | 1500 |
| pairs.eval_diverse_b.jsonl | data/bitnet/pairs.eval_diverse_b.jsonl (§14 凍結物差し) | 1500 |

B の train (`pairs.train.diverse_b2000.jsonl`・§15.6・4500 行) は dir に入れず、まとめ焼き時に
`--train-file` へ絶対パスで渡す (train_bitnet が syn を diverse_b2000 に差し替え、`--identity` が
identity_cond を足す既存挙動 = diverse_b2000 ∪ identity)。

### 19.2 混合実態 (訓練時ログ)

- MIXED train=**15300** = diverse_b2000 4500 (synthetic 2500 + Claude 著述 2000) ∪ a12k identity 10800
- val=**1700** = synthetic 500 (pairs.val) + identity_cond 1200 (a12k identity val)

### 19.3 結論

新規データ無し・既存データ (§14 凍結 diverse-val / §15.6 b2000 / §17 a12k identity) をそのまま流用。
まとめ焼き成果物は `data/bitnet/_merge_ba/` 配下 (gitignore・再生成可)。正典昇格後の重み/golden は
merged 基準 (training-spec §17.3)・凍結 eval セットと本番 train/val ソースは無改変。a25k は未使用保持。

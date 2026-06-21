# dollama タグ生成 LM 訓練データセット仕様 (Phase 4 #1)

自作タグ生成 LM (`src/models/bitnet.hpp`) が学習する **「user text (自然文) → danbooru タグ列」** ペアの
データセット仕様。`scripts/train_bitnet.py` (model-trainer) と
`src/io/tokenizer.hpp` (cpp-implementer) が消費する。

**現行最終版 (1e/1f スケール後): 総 5,000 ペア (train 4,500 / val 500, 9:1)・
ja/en ≈ 50:50・rating g 100%・全件検証通過 (OOV/負語/順序/リーク/重複 すべて 0)・
ユニーク text 比率 1.0・タグ語彙 4,994。** 詳細は §11 を参照。

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
  pairs.train.jsonl               — 訓練ペア (4,500 行)
  pairs.val.jsonl                 — 検証ペア (500 行)
  stats.json                      — 件数・分布・出典・seed の記録
  cache/danbooru_posts.jsonl      — 生タグ共起キャッシュ (タグ文字列 + rating のみ・8,200 posts)
  cache/selected_tags.csv         — WD14 タグ csv キャッシュ
docs/dataset-spec.md              — 本ファイル
scripts/dollma_build_vocab.py     — vocab.json 生成
scripts/dollma_make_pairs.py      — 共起取得 → 射影 → 自然文逆生成 → pairs/stats 出力
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

## 13. 将来拡張 — 同一性条件付きペア (Phase 4 A)

Phase 4 A (同一性条件付きタグ生成) 用のデータ。本段 (#1) は「user text → tags」のみで、
§4 のとおりキャラ同一性タグ (`tag_string_character`) を target から除外している。A は
キャラ同一性を**条件入力**にするため、別形式のペアが要る:

- **入力**: `CharacterIdentity` (同一性層。bible の主キーで参照) + scene 記述 (自然文)。
- **target**: 同一性を保持した danbooru タグ列 (同一性由来タグ + シーン由来タグ)。
- 既存 #1 ペアに character-bible の同一性条件を付与する形 (例: `meta.identity` 等で拡張) を想定し、
  **既存スキーマ・検証 (`validate_pairs`) を壊さない後方互換**で追加する。
- リーク防止は §8 を踏襲 (同一キャラが train/val に跨らない・`post_id`/`text` チェック)。
- B (アニメ品質スコアラ) の正解ラベルもここで扱うか別仕様にするかは設計時に決める
  (§11 の合格/不合格蓄積を教師に蒸留する案)。
- 設計は別タスク (dataset-curator)。確定時に §3 スキーマへ正式追加する。

これは私が全年齢分の著述に使っているのと同じ方法論・データ契約の骨組みで、{{ }}
の箇所をご自身で埋めて外部ツールに流す形です。コンテンツ方針/トーンの中身 (どんな描写にするか) は私からは書きません —
そこは「ご自身で変更する」前提の空欄にしてあります。骨組みと検証規約までが私の提供範囲です。

  # 役割
  あなたは「danbooru タグ集合 → ユーザーが打ちそうな自然文の作画依頼/描写」を著述するワーカー。

  # 入力
  スライスファイル {{SLICE_PATH}} を読む。各行 JSON:
    {"post_id": int, "variant_idx": int, "gold_tags": [str,...],
     "lang_hint": "ja"|"en", "style_hint": "plain"|"descriptive"|"terse"|"conversational",
     "rating": "{{RATING}}"}

  # タスク
  各行につき自然文 text を 1 本書き、次の JSONL を {{OUT_PATH}} に出力 (1 行 1 JSON):
    {"post_id": <入力と同じ>, "variant_idx": <入力と同じ>,
     "lang": "<lang_hint と同じ>", "text": "<著述>"}

  # 著述ルール (厳守)
  1. 言語: lang_hint が "ja" なら日本語、"en" なら英語で書く (text の言語 = lang)。
  2. tags-stay-real: gold_tags は触らない・出力に含めない・並べ替えただけのカンマ列挙にしない。
     主要な被写体/属性 (人数・髪・目・服・ポーズ・場所など) を自然文に溶かす。言い換え・順序入替・一部省略可。
  3. 文体 (style_hint):
     - plain: 普通の作画依頼文
     - descriptive: 情景を豊かに描写
     - terse: 短く端的だが自然な短文 (タグのカンマ列挙にしない)
     - conversational: 口語・くだけた話し言葉
  4. 機械生成スクリプト/RNG スロット埋め/固定ラッパー文の使い回しは禁止。1 行ずつ意味を考え、行ごとに言い回し・構造を変える。
  5. 意味整合: gold に no humans / no_humans があれば人物属性を書かず無人情景に。重複概念は統合し矛盾を出さない。
  6. post_id を text に絶対に書かない (数字の漏出は検証で弾かれる)。

  # ★コンテンツ方針・トーン (ここはご自身で記述してください)
  {{CONTENT_POLICY_AND_TONE — 例: レーティング/表現範囲/語彙の方針などをここに記述}}

  # 出力後の自己検証 (必須)
  - 出力行数 = 入力スライス行数
  - 各行 valid JSON・キーは {post_id, variant_idx, lang, text} のみ
  - (post_id, variant_idx) が入力と過不足なく一致 (欠落0/余分0)・複合キー重複0
  - lang == lang_hint・text 非空・text に post_id 数字を含まない
  - UTF-8 (ensure_ascii=False) で出力

  使い方:
  1. 成人向け todo (タグ手がかりのみ) は、ご希望なら私が q/e を含めて抽出してスライス化します (タグはラベルなので配布可)。
  2. 上の {{CONTENT_POLICY_AND_TONE}} をご自身で記入し、外部ツールに流して text を生成。
  3. 出力を diverse_train_texts_part<NN>.jsonl 形式で data/bitnet/ に置けば、私が組んだ取り込み (--ingest)・検証・訓練・評価の配管にそのまま乗ります
  (今回の全年齢 10,000 件と同じ経路)。

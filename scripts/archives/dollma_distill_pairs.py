# -*- coding: utf-8 -*-
"""dollama Phase 4 D2: Qwen2-1.5B によるタグ列 → 自然文「text 多様化」蒸留 (方式 A)

dataset-spec §12 (LLM 多様化 source:"llm_distill") の実装。

狙い:
  既存 data/bitnet/pairs.train.jsonl の text はテンプレ生成 (dollma_make_pairs.py の
  JA/EN_TEMPLATES)。これを Qwen2 が書く自然で多様な依頼文に置き換えた蒸留ペアを
  新規生成し、「自然文 → タグ」の入力分布を広げて過学習を緩和する。

鉄則 (dataset-spec §0):
  教師には **text だけ書かせる**。タグ (target) は実 danbooru 共起のまま固定・改変させない。
  tags は入力 (train 側) の実共起をそのままコピーする。

リーク防止:
  - 入力は train 側 (pairs.train.jsonl) の post_id/tags のみ。val (pairs.val.jsonl) は一切使わない。
  - 生成 text が既存 val の text と完全一致しないことを検証する。

出力:
  data/bitnet/pairs.distill.train.jsonl
  各行: 既存スキーマ互換 + source:"llm_distill", meta.tmpl=-1, meta.post_id (流用元), meta.lang。

冪等/再開:
  出力ファイルに既にある (post_id, lang) はスキップ。逐次 append で中断耐性。

使い方:
  py -3.12 scripts/dollma_distill_pairs.py --n 3000 --seed 20260620 \
      --vocab data/bitnet/vocab.json --out-dir data/bitnet --en-ratio 0.75
"""
import argparse
import io
import json
import os
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# dollma_make_pairs.py / train_bitnet.py を流用 (scripts/ を import path へ)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dollma_make_pairs import (  # noqa: E402
    classify_bucket,
    normalize_separator,
    validate_pairs,
    NEGATIVE_BLOCKLIST,
)
import train_bitnet  # noqa: E402  (Tokenizer / encode_text_greedy 流用)

MODEL_ID = "Qwen/Qwen2-1.5B-Instruct"


# ==============================================================
# few-shot プロンプト
#   タグの意味を変えない・タグに無い要素を足さない を強制する (JA ドリフト対策)。
#   タグは改変禁止。教師には自然文 1 文だけ書かせる。
# ==============================================================
# 例のタグ列は実データに寄せた (主体数 + 髪/目/服/状況)。意訳の手本を見せる。
_FEWSHOT_EN = [
    ("1girl, solo, long hair, blonde hair, blue eyes, school uniform, smile, classroom",
     "A smiling schoolgirl with long blonde hair and blue eyes, standing in a classroom in her uniform."),
    ("1boy, white hair, red eyes, hoodie, hands in pockets, city street, serious",
     "A serious young man with white hair and red eyes wearing a hoodie, hands in his pockets on a city street."),
    ("2girls, maid, cat ears, indoors, sitting, looking at viewer",
     "Two cat-eared maids sitting indoors and looking toward the viewer."),
]
_FEWSHOT_JA = [
    ("1girl, solo, long hair, blonde hair, blue eyes, school uniform, smile, classroom",
     "金髪ロングで青い目の女子学生が、制服姿で教室に立って微笑んでいる絵をお願いします。"),
    ("1boy, white hair, red eyes, hoodie, hands in pockets, city street, serious",
     "白髪で赤い目の少年が、パーカーを着て両手をポケットに入れ、街角で真剣な表情をしている絵。"),
    ("2girls, maid, cat ears, indoors, sitting, looking at viewer",
     "猫耳のメイドの女の子が二人、室内で座ってこちらを見ている様子を描いてください。"),
]

_SYS_EN = (
    "You write image-generation requests. Given a list of danbooru tags, write exactly ONE "
    "natural English sentence requesting that picture, as a person would ask an artist. "
    "Rules: (1) Do NOT change the meaning of any tag. If a tag says 'blonde hair' the hair is "
    "blonde, never another color. (2) Do NOT add any element that is not in the tags. "
    "(3) Do NOT list the tags verbatim; paraphrase them into one fluent sentence. "
    "(4) Output ONLY the sentence, no quotes, no explanation, no tag list."
)
_SYS_JA = (
    "あなたは画像生成の依頼文を書くアシスタントです。与えられた danbooru タグ列が表す絵を、"
    "人が絵描きに頼むような自然な日本語の依頼文ちょうど 1 文で書いてください。"
    "厳守事項: (1) タグの意味を絶対に変えないこと。例えば 'blonde hair'(金髪) を別の髪色に"
    "言い換えてはいけません。(2) タグに無い要素を足さないこと。(3) タグを英単語のまま並べず、"
    "自然な日本語 1 文に意訳すること。(4) 依頼文のみを出力し、引用符・説明・タグ列は付けないこと。"
)


def build_messages(tags_str, lang):
    """few-shot 付きチャットメッセージを組む。tags_str はカンマ区切り (英タグ名のまま)。"""
    if lang == "ja":
        sys_p, shots = _SYS_JA, _FEWSHOT_JA
        usr = lambda t: f"タグ列: {t}"
    else:
        sys_p, shots = _SYS_EN, _FEWSHOT_EN
        usr = lambda t: f"Tags: {t}"
    msgs = [{"role": "system", "content": sys_p}]
    for ex_tags, ex_out in shots:
        msgs.append({"role": "user", "content": usr(ex_tags)})
        msgs.append({"role": "assistant", "content": ex_out})
    msgs.append({"role": "user", "content": usr(tags_str)})
    return msgs


# ==============================================================
# 品質ゲート (後処理フィルタ)
#   空/極端に短い・制御文字や文字化け混入・主要タグとの明白な矛盾 を棄却。
# ==============================================================
# 髪色タグ → text に現れてはならない「別の髪色」語 (明白な矛盾検出用・最小限)
_HAIR_COLORS_EN = {
    "blonde": ["blonde", "blond", "golden hair", "yellow hair"],
    "black": ["black hair", "dark hair", "raven"],
    "brown": ["brown hair", "brunette"],
    "white": ["white hair", "silver hair", "platinum"],
    "blue": ["blue hair"],
    "red": ["red hair", "redhead", "crimson hair", "scarlet hair"],
    "pink": ["pink hair"],
    "green": ["green hair"],
    "purple": ["purple hair", "violet hair"],
    "grey": ["grey hair", "gray hair"],
    "silver": ["silver hair", "platinum"],
    "orange": ["orange hair", "ginger"],
}
# ja での髪色語 (タグ → 許容語 / 矛盾語)。最小限の代表色のみチェック。
_HAIR_COLORS_JA = {
    "blonde": (["金髪", "ブロンド", "金色"], ["黒髪", "茶髪", "赤い髪", "青い髪", "白髪", "銀髪", "緑髪", "ピンク", "紫"]),
    "black": (["黒髪", "黒い髪"], ["金髪", "ブロンド", "茶髪", "赤い髪", "青い髪", "白髪", "銀髪", "緑髪", "ピンク"]),
    "brown": (["茶髪", "茶色", "ブラウン"], ["金髪", "黒髪", "青い髪", "白髪", "銀髪", "緑髪", "ピンク"]),
    "white": (["白髪", "白い髪"], ["金髪", "黒髪", "茶髪", "青い髪", "赤い髪", "緑髪", "ピンク"]),
    "blue": (["青い髪", "青髪", "ブルー"], ["金髪", "黒髪", "茶髪", "白髪", "赤い髪", "緑髪", "ピンク"]),
    "red": (["赤い髪", "赤髪", "レッド"], ["金髪", "黒髪", "茶髪", "白髪", "青い髪", "緑髪", "ピンク"]),
    "pink": (["ピンク", "桃色"], ["金髪", "黒髪", "茶髪", "白髪", "青い髪", "赤い髪", "緑髪"]),
    "silver": (["銀髪", "シルバー"], ["金髪", "黒髪", "茶髪", "青い髪", "赤い髪", "緑髪", "ピンク"]),
}

# 主体数タグ → ja で矛盾する語 (1girl なのに「二人」「男の子」等)
_SUBJ_JA_CONTRA = {
    "1girl": ["二人", "三人", "複数", "男の子", "少年", "男性"],
    "1boy": ["二人", "三人", "複数", "女の子", "少女", "女性"],
}

_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
# CJK / ひらがな / カタカナ / 基本ラテン / 一般記号以外の混入 (文字化け検出の補助)
_ALLOWED_RE = re.compile(
    "[^"
    " -ɏ"      # 基本ラテン + ラテン拡張
    " -⁯"      # 一般句読点
    "　-ヿ"      # CJK 記号・ひらがな・カタカナ
    "㐀-鿿"      # CJK 統合漢字
    "＀-￯"      # 全角英数・半角カナ
    "]"
)


def _hair_color_tags(tags):
    """tags 中の髪色キー (en 基準) を返す。"""
    out = []
    for t in tags:
        for key in _HAIR_COLORS_EN:
            if t == f"{key} hair" or (key == "blonde" and t == "blonde hair"):
                out.append(key)
    return out


def quality_gate(text, tags, lang):
    """生成 text の品質を判定。OK なら "", NG なら棄却理由文字列を返す。"""
    if not text or not text.strip():
        return "empty"
    s = text.strip()
    # 制御文字
    if _CTRL_RE.search(s):
        return "control_char"
    # 長さ下限 (極端に短い断片を棄却)
    if lang == "ja":
        if len(s) < 12:
            return "too_short_ja"
    else:
        if len(re.findall(r"[A-Za-z]+", s)) < 4:
            return "too_short_en"
    # 上限 (暴走・複数文の列挙を棄却。1 文想定)
    if len(s) > 400:
        return "too_long"
    # 文字化け / 想定外スクリプト混入
    bad = _ALLOWED_RE.findall(s)
    if len(bad) > 0:
        # 少数のレアな記号は許すが、3 文字以上の未知文字は文字化けとみなす
        if len(bad) >= 3:
            return f"garbled({''.join(bad[:6])!r})"
    # タグをそのままカンマ列挙しただけ (意訳していない) を棄却
    if s.count(",") >= 6 and not any(p in s for p in ("。", ".", "?", "!", "、")):
        return "tag_dump"
    low = s.lower()
    # 髪色矛盾チェック (en)
    hair_keys = _hair_color_tags(tags)
    if lang == "en":
        for key in hair_keys:
            allowed = _HAIR_COLORS_EN.get(key, [])
            # 別の髪色語が出ていて、正しい色語が一切出ていない → 矛盾
            other_hit = False
            for okey, words in _HAIR_COLORS_EN.items():
                if okey == key:
                    continue
                for w in words:
                    if w in low and "hair" in low:
                        other_hit = True
            self_hit = any(w in low for w in allowed)
            if other_hit and not self_hit:
                return f"hair_contradiction_en({key})"
    else:  # ja
        for key in hair_keys:
            if key not in _HAIR_COLORS_JA:
                continue
            ok_words, bad_words = _HAIR_COLORS_JA[key]
            other_hit = any(w in s for w in bad_words)
            self_hit = any(w in s for w in ok_words)
            if other_hit and not self_hit:
                return f"hair_contradiction_ja({key})"
        # 主体数矛盾 (ja)
        for st, contra in _SUBJ_JA_CONTRA.items():
            if st in tags:
                if any(w in s for w in contra):
                    return f"subject_contradiction_ja({st})"
    return ""


def clean_text(text):
    """前処理: 引用符剥がし・改行潰し・Unicode 正規化。"""
    s = unicodedata.normalize("NFC", text).strip()
    # 行頭の見出し/引用符を剥がす
    s = re.sub(r"^[\s>*\-【「\"'「『（(]+", "", s)
    s = re.sub(r"[」』\"'」』]+$", "", s).strip()
    # 複数行 → 1 行目を採用 (1 文想定。説明が続く場合は捨てる)
    first = s.splitlines()[0].strip() if s else s
    # "依頼文:" 等のラベル除去
    first = re.sub(r"^(依頼文|出力|文|Request|Sentence)\s*[:：]\s*", "", first, flags=re.I)
    return first.strip()


# ==============================================================
# train 側ペアの読み込み (tags + post_id)。val は一切読まない。
# ==============================================================
def load_train_rows(train_path):
    rows = []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_val_texts(val_path):
    texts = set()
    if os.path.exists(val_path):
        with open(val_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.add(json.loads(line)["text"])
    return texts


def load_existing(out_path):
    """再開用: 既存出力の (post_id, lang) キーと行を読む。"""
    done = set()
    rows = []
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                done.add((r["meta"]["post_id"], r["meta"].get("lang", r["lang"])))
                rows.append(r)
    return done, rows


def make_distill_pair(text, tags, lang, post_id, rating):
    """蒸留ペアを既存スキーマ互換で組む。tags は入力の実共起をそのままコピー。"""
    return {
        "text": text,
        "tags": list(tags),
        "lang": lang,
        "source": "llm_distill",
        "meta": {
            "rating": rating,
            "post_id": post_id,
            "n_tags": len(tags),
            "tmpl": -1,          # §12: LLM 由来は -1
            "lang": lang,        # 指示どおり meta にも lang を持たせる
        },
    }


def main():
    ap = argparse.ArgumentParser(description="D2: Qwen2 で蒸留ペア生成")
    ap.add_argument("--n", type=int, default=3000, help="生成ペア数の上限 (train 4500 に対し)")
    ap.add_argument("--seed", type=int, default=20260620)
    ap.add_argument("--vocab", default="data/bitnet/vocab.json")
    ap.add_argument("--out-dir", default="data/bitnet")
    ap.add_argument("--en-ratio", type=float, default=0.75,
                    help="en の比率 (JA ドリフト対策で EN 偏重・残りが ja)")
    ap.add_argument("--temp-en", type=float, default=0.5)
    ap.add_argument("--temp-ja", type=float, default=0.3)
    ap.add_argument("--max-new", type=int, default=80)
    ap.add_argument("--max-retry", type=int, default=2, help="棄却時の再生成回数")
    ap.add_argument("--limit-posts", type=int, default=0,
                    help="使う train post 数の上限 (0=全件・デバッグ用)")
    args = ap.parse_args()

    import random
    rng = random.Random(args.seed)

    train_path = os.path.join(args.out_dir, "pairs.train.jsonl")
    val_path = os.path.join(args.out_dir, "pairs.val.jsonl")
    out_path = os.path.join(args.out_dir, "pairs.distill.train.jsonl")

    # vocab
    with open(args.vocab, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    vocab_set = {t["tag"] for t in vocab["tags"]}

    # tokenizer (往復検証用)
    tok = train_bitnet.Tokenizer(args.vocab)

    train_rows = load_train_rows(train_path)
    val_texts = load_val_texts(val_path)
    train_pids = {r["meta"]["post_id"] for r in train_rows}
    print(f"[distill] train rows={len(train_rows)} train_pids={len(train_pids)} "
          f"val_texts={len(val_texts)}", file=sys.stderr)

    # 入力 post をシャッフルし、en/ja を en-ratio で割当 (決定的)
    rng.shuffle(train_rows)
    if args.limit_posts > 0:
        train_rows = train_rows[:args.limit_posts]

    done, existing = load_existing(out_path)
    print(f"[distill] 既存出力 {len(existing)} 行を再利用 (skip 済キー {len(done)})",
          file=sys.stderr)

    # --- Qwen2 ロード ---
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"[distill] loading {MODEL_ID} device={device} dtype={dtype}", file=sys.stderr)
    t0 = time.time()
    qtok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device)
    model.eval()
    print(f"[distill] loaded in {time.time()-t0:.1f}s", file=sys.stderr)

    def gen_once(messages, temperature):
        chat = qtok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = qtok([chat], return_tensors="pt").to(device)
        n_in = inputs.input_ids.shape[1]
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=args.max_new,
                do_sample=temperature > 0, temperature=max(temperature, 1e-4),
                top_p=0.9, pad_token_id=qtok.eos_token_id,
            )
        gen_ids = out[0][n_in:]
        return qtok.decode(gen_ids, skip_special_tokens=True).strip(), gen_ids.shape[0]

    # --- 生成ループ ---
    out_f = open(out_path, "a", encoding="utf-8")
    rej_counts = {}
    n_new = 0
    n_total_target = args.n
    n_have = len(existing)
    gen_tok, gen_t = 0, 0.0
    seen_text = {r["text"] for r in existing}

    try:
        for i, r in enumerate(train_rows):
            if n_have + n_new >= n_total_target:
                break
            pid = r["meta"]["post_id"]
            tags = r["tags"]
            rating = r["meta"]["rating"]
            # en/ja 割当: rng で en-ratio に従う (決定的)
            lang = "en" if rng.random() < args.en_ratio else "ja"
            if (pid, lang) in done:
                continue
            tags_str = ", ".join(tags)
            temperature = args.temp_en if lang == "en" else args.temp_ja

            accepted = None
            for attempt in range(args.max_retry + 1):
                t1 = time.time()
                raw, ntok = gen_once(build_messages(tags_str, lang),
                                     temperature if attempt == 0 else max(temperature - 0.1, 0.0))
                gen_t += time.time() - t1
                gen_tok += ntok
                text = clean_text(raw)
                reason = quality_gate(text, tags, lang)
                if reason == "":
                    # text リーク (val と完全一致) / 出力内重複 を弾く
                    if text in val_texts:
                        reason = "val_text_leak"
                    elif text in seen_text:
                        reason = "dup_text"
                if reason == "":
                    accepted = text
                    break
                rej_counts[reason] = rej_counts.get(reason, 0) + 1

            if accepted is None:
                continue

            pair = make_distill_pair(accepted, tags, lang, pid, rating)
            out_f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            out_f.flush()
            done.add((pid, lang))
            seen_text.add(accepted)
            n_new += 1

            if (n_new) % 25 == 0:
                tps = gen_tok / gen_t if gen_t > 0 else 0
                print(f"[distill] +{n_new} (have={n_have+n_new}/{n_total_target}) "
                      f"{tps:.1f} tok/s rej={sum(rej_counts.values())}", file=sys.stderr)
    finally:
        out_f.close()

    print(f"[distill] 生成完了: 新規 {n_new} 件 (累計 {n_have+n_new})", file=sys.stderr)
    print(f"[distill] 生成時間 {gen_t:.1f}s / {gen_tok} tok = "
          f"{gen_tok/gen_t if gen_t>0 else 0:.1f} tok/s", file=sys.stderr)
    print(f"[distill] 棄却内訳: {json.dumps(rej_counts, ensure_ascii=False)}", file=sys.stderr)

    # ==========================================================
    # 検証 (validate_pairs を source 非依存で全件適用 + 蒸留固有検査)
    # ==========================================================
    _, all_rows = load_existing(out_path)
    print(f"\n[distill] === 検証 (全 {len(all_rows)} 行) ===", file=sys.stderr)

    errors = validate_pairs(all_rows, vocab_set)  # OOV/負語/正準順序

    leak_pid = leak_text = unk_total = mismatch = 0
    bad_source = bad_tmpl = 0
    out_texts = set()
    dup_text = 0
    for r in all_rows:
        if r.get("source") != "llm_distill":
            bad_source += 1
        if r["meta"].get("tmpl") != -1:
            bad_tmpl += 1
        # post_id が train 内 (val リーク 0)
        if r["meta"]["post_id"] not in train_pids:
            leak_pid += 1
        # text が val と完全一致しない
        if r["text"] in val_texts:
            leak_text += 1
        if r["text"] in out_texts:
            dup_text += 1
        out_texts.add(r["text"])
        # tags が入力の実共起と一致 (改変されていない) を確認
        # tokenizer 往復 UNK 0 + 完全一致
        ids = [tok.tag_to_id_lookup(t) for t in r["tags"]]
        if train_bitnet.TOK_UNK in ids:
            unk_total += 1
        decoded = [tok.tags[i - 5] for i in ids if i != train_bitnet.TOK_UNK]
        if decoded != r["tags"]:
            mismatch += 1

    print(f"  validate_pairs errors: {len(errors)}", file=sys.stderr)
    for e in errors[:10]:
        print("    ", e, file=sys.stderr)
    print(f"  source!=llm_distill: {bad_source}", file=sys.stderr)
    print(f"  meta.tmpl!=-1: {bad_tmpl}", file=sys.stderr)
    print(f"  post_id not in train (val leak): {leak_pid}", file=sys.stderr)
    print(f"  text == val text (text leak): {leak_text}", file=sys.stderr)
    print(f"  dup text within distill: {dup_text}", file=sys.stderr)
    print(f"  tokenizer UNK rows: {unk_total}", file=sys.stderr)
    print(f"  tokenizer roundtrip mismatch: {mismatch}", file=sys.stderr)

    # tags が元 train post の実共起と一致するか (改変禁止の最終確認)
    pid_tags = {r["meta"]["post_id"]: r["tags"] for r in train_rows}
    tag_altered = 0
    for r in all_rows:
        src_tags = pid_tags.get(r["meta"]["post_id"])
        if src_tags is not None and r["tags"] != src_tags:
            tag_altered += 1
    print(f"  tags altered vs source train post: {tag_altered}", file=sys.stderr)

    ok = (len(errors) == 0 and leak_pid == 0 and leak_text == 0 and unk_total == 0
          and mismatch == 0 and bad_source == 0 and bad_tmpl == 0 and tag_altered == 0)
    print(f"\n[distill] 検証 {'OK' if ok else 'NG'}", file=sys.stderr)

    # stats
    lang_c = {"ja": 0, "en": 0}
    for r in all_rows:
        lang_c[r["lang"]] += 1
    n = len(all_rows)
    stats = {
        "version": 1,
        "seed": args.seed,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_ID,
        "counts": {"total": n},
        "lang_ratio": {k: round(v / n, 4) for k, v in lang_c.items()} if n else {},
        "en_ratio_target": args.en_ratio,
        "temp": {"en": args.temp_en, "ja": args.temp_ja},
        "reject_counts": rej_counts,
        "reject_total": sum(rej_counts.values()),
        "accept_rate": round(n_new / (n_new + sum(rej_counts.values())), 4)
        if (n_new + sum(rej_counts.values())) else None,
        "unique_text_ratio": round(len(out_texts) / n, 4) if n else None,
        "validation": {
            "validate_pairs_errors": len(errors),
            "val_post_leak": leak_pid,
            "val_text_leak": leak_text,
            "tokenizer_unk_rows": unk_total,
            "tokenizer_mismatch": mismatch,
            "tags_altered": tag_altered,
            "ok": ok,
        },
        "gen_seconds": round(gen_t, 1),
        "gen_tok_per_s": round(gen_tok / gen_t, 1) if gen_t > 0 else None,
    }
    stats_path = os.path.join(args.out_dir, "stats.distill.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    print(f"[distill] stats: {stats_path}", file=sys.stderr)
    print(f"[distill] lang_ratio={stats['lang_ratio']} "
          f"accept_rate={stats['accept_rate']}", file=sys.stderr)

    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

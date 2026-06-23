#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dollama 評価作り直し (施策 C) — diverse-val 構築スクリプト

施策 C のゴール: 物差しを proxy (テンプレ 3 種・teacher-forcing recall) から実品質寄りへ
作り直すための diverse-val 評価データを構築する。本スクリプトは 2 モードを持つ:

  --emit-prompts : gold タグ→「散文生成プロンプトのバッチ」を出力 (段a)。
                   **散文は一切生成しない。** gold タグの列挙のみ。
                   散文 (段b) はこのセッション内の main Claude が著述する。
  --ingest       : 段a の prompts.jsonl (gold 源) + main Claude が書いた texts.jsonl を
                   突合・検証して凍結 jsonl を出力する (段c)。

不変条件 (厳守・プラン §不変条件):
  - data/bitnet/pairs.val.jsonl はバイト不変 (読むだけ)。新ファイルは加算のみ。
  - tags-stay-real: gold = post の実 danbooru タグ固定。生成文からタグを抽出しない。
  - 凍結: 出力 jsonl は再現性アンカー。既存があればスキップ (--force でのみ再生成)。

Pool 定義:
  Pool A = pairs.val.jsonl の post_id と gold タグ (その行の tags をバイト一致コピー)。
  Pool B = cache/danbooru_posts.jsonl から post_id ∉ (train ∪ val) を抽出。
           タグ整形は dollma_make_pairs.py の正準順序規約に合わせる。

使い方:
  # 段a (~30 post の小規模スモーク)
  python scripts/dollma_make_eval_diverse.py --emit-prompts \
      --n-a 30 --n-b 30 --n-variants 3 --seed 20260620
  # 段c (main Claude が texts.jsonl を書いた後)
  python scripts/dollma_make_eval_diverse.py --ingest
"""
import argparse
import json
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# 正準順序規約は dollma_make_pairs.py を唯一のソースとして流用 (二重実装禁止)。
import dollma_make_pairs as mp

# specials id (tokenizer.hpp TokenId / train_bitnet と一致)
TOK_UNK = 4


class EvalTokenizer:
    """vocab.json 駆動の軽量トークナイザ (gold⊆vocab 判定用)。

    train_bitnet.Tokenizer の normalize / tag_to_id_lookup 規約を再現する。
    train_bitnet は torch を import するため、データ構築では torch 非依存の本クラスを
    使う (test_dollma_eval_diverse.py が tb.Tokenizer とのバイト等価性を assert)。
    """

    def __init__(self, vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            vj = json.load(f)
        specials = vj["specials"]
        if specials != ["<pad>", "<bos>", "<eos>", "<sep>", "<unk>"]:
            raise ValueError(f"specials mismatch: {specials}")
        self.tags = []
        for i, t in enumerate(vj["tags"]):
            if t["id"] != 5 + i:
                raise ValueError(f"tags[{i}].id={t['id']} expected {5 + i}")
            self.tags.append(t["tag"])
        self.tag_to_id = {}
        for i, t in enumerate(self.tags):
            self.tag_to_id[self.normalize(t)] = 5 + i

    @staticmethod
    def normalize(s):
        """英数字に挟まれた '_' のみスペースへ (顔文字 '_' は保持)。tokenizer.hpp 同一。"""
        out = []
        n = len(s)
        for i, c in enumerate(s):
            if c == "_":
                prev_alnum = i > 0 and s[i - 1].isalnum()
                next_alnum = i + 1 < n and s[i + 1].isalnum()
                if prev_alnum and next_alnum:
                    out.append(" ")
                    continue
            out.append(c)
        return "".join(out)

    def tag_to_id_lookup(self, tag):
        return self.tag_to_id.get(self.normalize(tag), TOK_UNK)

# 既定パス
DEF_DATA_DIR = os.path.join(_HERE, "..", "data", "bitnet")
DEF_VOCAB = os.path.join(DEF_DATA_DIR, "vocab.json")
DEF_VAL = os.path.join(DEF_DATA_DIR, "pairs.val.jsonl")
DEF_TRAIN = os.path.join(DEF_DATA_DIR, "pairs.train.jsonl")
DEF_IDENTITY_TRAIN = os.path.join(DEF_DATA_DIR, "pairs.identity.train.jsonl")
DEF_CACHE = os.path.join(DEF_DATA_DIR, "cache", "danbooru_posts.jsonl")
DEF_PROMPTS = os.path.join(DEF_DATA_DIR, "eval_diverse_prompts.jsonl")
DEF_TEXTS = os.path.join(DEF_DATA_DIR, "eval_diverse_texts.jsonl")
DEF_OUT_A = os.path.join(DEF_DATA_DIR, "pairs.eval_diverse_a.jsonl")
DEF_OUT_B = os.path.join(DEF_DATA_DIR, "pairs.eval_diverse_b.jsonl")


# ------------------------------------------------------------
# 共通読み込み
# ------------------------------------------------------------
def _read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_vocab_set(vocab_path):
    """vocab.json から正規化済みタグ集合を作る (gold⊆vocab 判定用)。

    train_bitnet.Tokenizer と同じ normalize を通すことで二重実装を避ける。
    """
    with open(vocab_path, "r", encoding="utf-8") as f:
        vj = json.load(f)
    norm = EvalTokenizer.normalize
    vocab_set = {norm(t["tag"]) for t in vj["tags"]}
    freq = {norm(t["tag"]): t["freq"] for t in vj["tags"]}
    return vocab_set, freq


def train_post_ids():
    """train 側 (pairs.train + pairs.identity.train) の post_id 集合を抽出する。"""
    pids = set()
    for path in (DEF_TRAIN, DEF_IDENTITY_TRAIN):
        if os.path.exists(path):
            for r in _read_jsonl(path):
                pid = r.get("meta", {}).get("post_id")
                if pid is not None:
                    pids.add(pid)
    return pids


def val_post_ids_and_rows():
    """val 500 の post_id 集合と行 (gold タグ源) を返す。"""
    rows = _read_jsonl(DEF_VAL)
    pids = set()
    for r in rows:
        pid = r.get("meta", {}).get("post_id")
        if pid is not None:
            pids.add(pid)
    return pids, rows


# ------------------------------------------------------------
# Pool 構築
# ------------------------------------------------------------
def build_pool_a(val_rows, n_a, rng):
    """Pool A = val 500 の post_id と gold タグ (その行の tags をバイト一致コピー)。

    返り値: [{post_id, gold_tags, rating}] (val 行から複製・順序不変)。
    """
    eligible = [r for r in val_rows if r.get("meta", {}).get("post_id") is not None]
    rng.shuffle(eligible)
    sel = eligible[:n_a] if n_a > 0 else eligible
    pool = []
    for r in sel:
        pool.append({
            "post_id": r["meta"]["post_id"],
            # gold は val 行の tags をそのままコピー (バイト一致・整形しない)。
            "gold_tags": list(r["tags"]),
            "rating": r["meta"].get("rating", "g"),
        })
    return pool


def build_pool_b(vocab_set, freq, train_pids, val_pids, n_b, rng):
    """Pool B = cache から post_id ∉ (train ∪ val) を抽出し正準順序整形。

    タグ整形は dollma_make_pairs.make_pair と同じ正準順序規約に合わせる
    (vocab 射影 → バケット分類 → freq 降順 → バケット順 → 先頭 16)。
    """
    posts = _read_jsonl(DEF_CACHE)
    held = [p for p in posts
            if p["post_id"] not in train_pids and p["post_id"] not in val_pids]
    rng.shuffle(held)
    pool = []
    for p in held:
        ordered = _canonical_tags_from_post(p, vocab_set, freq)
        if ordered is None:
            continue  # タグが少なすぎる投稿はスキップ (make_pair と同基準)
        pool.append({
            "post_id": p["post_id"],
            "gold_tags": ordered,
            "rating": p.get("rating", "g"),
        })
        if n_b > 0 and len(pool) >= n_b:
            break
    return pool


def _canonical_tags_from_post(post, vocab_set, freq):
    """cache post → 正準順序タグ列。make_pair のタグ整形部分のみを流用 (text は作らない)。"""
    raw = post["tags_general"].split()
    proj = []
    seen = set()
    for t in raw:
        n = mp.normalize_separator(t)
        if n in vocab_set and n not in seen:
            seen.add(n)
            proj.append(n)
    if len(proj) < 4:
        return None
    buckets = {}
    for t in proj:
        buckets.setdefault(mp.classify_bucket(t), []).append(t)
    for b in buckets:
        buckets[b].sort(key=lambda x: (-freq.get(x, 0), x))
    ordered = []
    for b in (1, 2, 3, 4, 5, 6):
        ordered.extend(buckets.get(b, []))
    if len(ordered) > 16:
        ordered = ordered[:16]
    return ordered


# ------------------------------------------------------------
# 段a: プロンプトバッチ出力
# ------------------------------------------------------------
def emit_prompts(args, rng):
    vocab_set, freq = load_vocab_set(args.vocab)
    train_pids = train_post_ids()
    val_pids, val_rows = val_post_ids_and_rows()

    # B の train・val 非交差を assert (リーク 0)。
    pool_a = build_pool_a(val_rows, args.n_a, rng)
    pool_b = build_pool_b(vocab_set, freq, train_pids, val_pids, args.n_b, rng)
    b_pids = {e["post_id"] for e in pool_b}
    assert not (b_pids & train_pids), "Pool B が train と交差 (リーク)"
    assert not (b_pids & val_pids), "Pool B が val と交差 (リーク)"

    # gold⊆vocab を段a でも確認 (B は射影済み・A は val 由来で in-vocab のはず)。
    tok = EvalTokenizer(args.vocab)
    for tag_pool, name in ((pool_a, "A"), (pool_b, "B")):
        for e in tag_pool:
            for t in e["gold_tags"]:
                if tok.tag_to_id_lookup(t) == TOK_UNK:
                    raise AssertionError(
                        f"Pool {name} post {e['post_id']} に vocab 外 gold '{t}'")

    out_path = args.prompts
    if os.path.exists(out_path) and not args.force:
        print(f"[diverse] {out_path} 既存 → スキップ (--force で再生成)",
              file=sys.stderr)
        return 0

    # 言語ヒント: post_id を seed に決定的に振る (再現的・JP/EN を散らす)。
    lines = []
    for subset, pool in (("A", pool_a), ("B", pool_b)):
        for e in pool:
            lang_hint = "ja" if (e["post_id"] % 2 == 0) else "en"
            lines.append({
                "post_id": e["post_id"],
                "subset": subset,
                "gold_tags": e["gold_tags"],
                "n_variants": args.n_variants,
                "lang_hint": lang_hint,
                "rating": e["rating"],
            })

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(json.dumps(ln, ensure_ascii=False) + "\n")

    n_a = sum(1 for ln in lines if ln["subset"] == "A")
    n_b = sum(1 for ln in lines if ln["subset"] == "B")
    print(f"[diverse] prompts 出力: {out_path}", file=sys.stderr)
    print(f"[diverse]   Pool A={n_a} Pool B={n_b} "
          f"n_variants={args.n_variants} "
          f"(期待散文件数 = (A+B)×variants = {(n_a + n_b) * args.n_variants})",
          file=sys.stderr)
    print(f"[diverse]   リーク検査 OK (B ∩ train=0, B ∩ val=0), gold⊆vocab OK",
          file=sys.stderr)
    return 0


# ------------------------------------------------------------
# 段c: 取り込み・検証・凍結
# ------------------------------------------------------------
def ingest(args, rng):
    vocab_set, _ = load_vocab_set(args.vocab)
    tok = EvalTokenizer(args.vocab)
    train_pids = train_post_ids()
    val_pids, val_rows = val_post_ids_and_rows()
    val_by_pid = {}
    for r in val_rows:
        pid = r.get("meta", {}).get("post_id")
        if pid is not None:
            val_by_pid[pid] = r

    if not os.path.exists(args.prompts):
        print(f"[diverse] prompts なし: {args.prompts} (先に --emit-prompts)",
              file=sys.stderr)
        return 1
    if not os.path.exists(args.texts):
        print(f"[diverse] texts なし: {args.texts} "
              f"(main Claude が散文を書く段b が未完)", file=sys.stderr)
        return 1

    prompts = _read_jsonl(args.prompts)
    texts = _read_jsonl(args.texts)

    # prompts を (post_id, subset) で索引化 = gold の唯一のソース。
    gold_by_key = {}
    for pr in prompts:
        gold_by_key[(pr["post_id"], pr["subset"])] = pr

    out_a, out_b = [], []
    errors = []
    for i, tx in enumerate(texts):
        pid = tx.get("post_id")
        subset = tx.get("subset")
        variant = tx.get("variant")
        text = tx.get("text", "")
        lang = tx.get("lang")
        key = (pid, subset)
        if key not in gold_by_key:
            errors.append(f"texts#{i}: prompts に無い (post_id={pid}, subset={subset})")
            continue
        pr = gold_by_key[key]

        # tags-stay-real: gold は段a の固定タグ。texts からは抽出しない。
        if subset == "A":
            # Pool A の tags は pairs.val.jsonl 同 post_id 行をバイト一致コピー。
            if pid not in val_by_pid:
                errors.append(f"texts#{i}: Pool A post {pid} が val に無い")
                continue
            gold = list(val_by_pid[pid]["tags"])
        else:
            gold = list(pr["gold_tags"])

        # 検証: text 非空
        if not text or not text.strip():
            errors.append(f"texts#{i}: text 空 (post_id={pid})")
        # 検証: post_id が text に漏れない
        if str(pid) in text:
            errors.append(f"texts#{i}: post_id {pid} が text に漏出")
        # 検証: lang 付与
        if lang not in ("ja", "en"):
            errors.append(f"texts#{i}: lang 不正 '{lang}' (post_id={pid})")
        # 検証: gold⊆vocab
        for t in gold:
            if tok.tag_to_id_lookup(t) == TOK_UNK:
                errors.append(f"texts#{i}: vocab 外 gold '{t}' (post_id={pid})")

        row = {
            "text": text,
            "tags": gold,
            "lang": lang,
            "source": "eval_diverse",
            "meta": {
                "post_id": pid,
                "subset": subset,
                "variant": variant,
                "gen": "claude",
                "rating": pr.get("rating", "g"),
                "n_tags": len(gold),
            },
        }
        (out_a if subset == "A" else out_b).append(row)

    # B の train・val 非交差 (リーク 0)。
    b_pids = {r["meta"]["post_id"] for r in out_b}
    if b_pids & train_pids:
        errors.append(f"Pool B が train と交差: {sorted(b_pids & train_pids)[:5]}")
    if b_pids & val_pids:
        errors.append(f"Pool B が val と交差: {sorted(b_pids & val_pids)[:5]}")

    if errors:
        print(f"[diverse] 検証エラー {len(errors)} 件 (先頭10):", file=sys.stderr)
        for e in errors[:10]:
            print("   ", e, file=sys.stderr)
        return 2

    # 凍結出力 (既存はスキップ・--force で再生成)。
    for path, rows, name in ((args.out_a, out_a, "A"), (args.out_b, out_b, "B")):
        if os.path.exists(path) and not args.force:
            print(f"[diverse] {path} 既存 → スキップ (--force で再生成)",
                  file=sys.stderr)
            continue
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[diverse] 凍結出力 Pool {name}: {path} ({len(rows)} 行)",
              file=sys.stderr)

    print(f"[diverse] 取り込み OK: A={len(out_a)} B={len(out_b)} "
          f"(tags-stay-real / gold⊆vocab / リーク0 / text非空 / lang付与 検証済)",
          file=sys.stderr)
    return 0


def main():
    ap = argparse.ArgumentParser(description="dollama 施策C diverse-val 構築")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--emit-prompts", action="store_true",
                      help="段a: gold タグ→プロンプトバッチ出力 (散文は生成しない)")
    mode.add_argument("--ingest", action="store_true",
                      help="段c: texts.jsonl を取り込み検証・凍結")
    ap.add_argument("--seed", type=int, default=20260620)
    ap.add_argument("--n-a", type=int, default=0,
                    help="Pool A post 数 (0=val 全 500)")
    ap.add_argument("--n-b", type=int, default=0,
                    help="Pool B post 数 (0=取得可能な全件)")
    ap.add_argument("--n-variants", type=int, default=3,
                    help="post あたり散文バリアント数")
    ap.add_argument("--vocab", default=DEF_VOCAB)
    ap.add_argument("--prompts", default=DEF_PROMPTS)
    ap.add_argument("--texts", default=DEF_TEXTS)
    ap.add_argument("--out-a", default=DEF_OUT_A)
    ap.add_argument("--out-b", default=DEF_OUT_B)
    ap.add_argument("--force", action="store_true",
                    help="既存の凍結出力を上書き再生成")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    if args.emit_prompts:
        return emit_prompts(args, rng)
    else:
        return ingest(args, rng)


if __name__ == "__main__":
    raise SystemExit(main())

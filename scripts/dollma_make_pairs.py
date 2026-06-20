#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dollama BitNet 訓練ペア生成スクリプト (Phase 4 #1 / 1c→1e/1f スケール)

実 danbooru タグ共起 (posts.json・タグのみ・画像非取得) をサンプリング元にして、
  実在投稿のタグ集合 → vocab 射影 → 正準順序化 (compose_prompt 順)
  → 日英テンプレで自然文を逆生成 → ペア JSONL
を出力する。スキーマは docs/dataset-spec.md §3 / §5 / §8 に従う。

使い方:
    python scripts/dollma_make_pairs.py --n 4500 --val 500 --seed 20260620 \
        --vocab data/bitnet/vocab.json --out-dir data/bitnet
"""
import argparse
import json
import os
import random
import re
import ssl
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

try:
    import certifi
    _CAFILE = certifi.where()
except Exception:
    _CAFILE = None

DANBOORU_BASE = "https://danbooru.donmai.us/posts.json"

# ------------------------------------------------------------
# §5 正準順序バケット分類 (compose_prompt 順)
#   1 canonical → 2 color_mode → 3 pose → 4 expression → 5 composition → 6 isolation
# 区切り正規化後 (スペース区切り) の名前で判定する。
# ------------------------------------------------------------
COLOR_MODE_TAGS = {"monochrome", "greyscale", "lineart", "sketch"}

ISOLATION_TAGS = {
    "simple background", "white background", "grey background",
    "transparent background",
}

COMPOSITION_TAGS = {
    "upper body", "full body", "cowboy shot", "portrait", "close-up",
    "lower body", "feet out of frame", "wide shot", "from above",
    "from below", "from side", "from behind", "dutch angle",
}

# pose: 動作・姿勢語 (代表的な danbooru タグ。完全一致集合 + 接尾辞ヒューリスティック)
POSE_TAGS = {
    "sitting", "standing", "lying", "kneeling", "squatting", "crouching",
    "walking", "running", "jumping", "waving", "arms up", "arm up",
    "hand up", "hands up", "arms behind back", "hand on hip", "hands on hips",
    "crossed arms", "outstretched arms", "spread legs", "crossed legs",
    "leaning forward", "bent over", "all fours", "on back", "on side",
    "stretching", "pointing", "salute", "kneeling", "wariza", "seiza",
    "arm support", "knees up", "fetal position", "looking back",
    "head tilt", "hugging own legs", "holding", "carrying", "dancing",
}

# expression: 表情語
EXPRESSION_TAGS = {
    "smile", "blush", "open mouth", "closed mouth", "closed eyes",
    "grin", "laughing", "crying", "tears", "angry", "pout", "frown",
    "surprised", "embarrassed", "sad", "happy", "serious", "smirk",
    "wink", "one eye closed", "expressionless", "ahegao", "nervous",
    "sweatdrop", "scared", "annoyed", "light smile", "parted lips",
    "teeth", "fang", "tongue out", "blush stickers", "half-closed eyes",
}


def classify_bucket(tag: str) -> int:
    """タグを正準順序バケット番号 (1..6) に分類する。未分類は 1 (canonical)。"""
    if tag in COLOR_MODE_TAGS:
        return 2
    if tag in ISOLATION_TAGS:
        return 6
    if tag in COMPOSITION_TAGS:
        return 5
    if tag in EXPRESSION_TAGS:
        return 4
    if tag in POSE_TAGS:
        return 3
    return 1  # canonical (外見一般タグ・未分類はここ。compose_prompt も canonical 先頭)


def normalize_separator(name: str) -> str:
    """区切り正規化 (dollma_build_vocab.py と同一ロジック)。"""
    return re.sub(r"(?<=[0-9A-Za-z])_(?=[0-9A-Za-z])", " ", name)


# ------------------------------------------------------------
# danbooru 取得 (タグ文字列 + rating のみ保持・画像は取得しない)
# ------------------------------------------------------------
def fetch_posts(target_posts: int, ratings: list[str], sleep_s: float,
                cache_path: str) -> list[dict]:
    """danbooru posts.json から実在投稿のタグ集合を取得する。

    保持するのは general タグ列と rating のみ。画像 URL・ピクセルは取得しない。
    keyset ページング (page=b<最小id>)。取得済みキャッシュがあれば再利用し、
    不足分は **キャッシュ最古 id から過去方向に追加取得して延伸** する
    (既存キャッシュを捨てて取り直さない: API に優しく・再現的)。
    """
    posts: list[dict] = []
    seen_pid: set[int] = set()
    cursor = None
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    p = json.loads(line)
                    if p["post_id"] not in seen_pid:
                        seen_pid.add(p["post_id"])
                        posts.append(p)
        if len(posts) >= target_posts:
            print(f"[pairs] キャッシュ再利用: {cache_path} ({len(posts)} posts)",
                  file=sys.stderr)
            return posts
        # 不足: 最古 id を起点に過去方向へ追加取得
        cursor = min(p["post_id"] for p in posts)
        print(f"[pairs] キャッシュ {len(posts)} posts → {target_posts} まで延伸 "
              f"(cursor=b{cursor})", file=sys.stderr)

    ctx = ssl.create_default_context(cafile=_CAFILE)
    ua = {"User-Agent": "dollama-dataset/0.1 (research; tags-metadata-only)"}
    # 複数 rating は OR 条件 (danbooru は ~rating:g ~rating:s 形式)
    if len(ratings) == 1:
        tag_query = f"rating:{ratings[0]}"
    else:
        tag_query = " ".join(f"~rating:{r}" for r in ratings)

    while len(posts) < target_posts:
        params = {"limit": 200, "tags": tag_query}
        if cursor is not None:
            params["page"] = f"b{cursor}"
        url = DANBOORU_BASE + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers=ua)
        try:
            with urllib.request.urlopen(req, timeout=30, context=ctx) as r:
                batch = json.loads(r.read())
        except Exception as e:
            print(f"[pairs] 取得失敗 (打ち切り): {repr(e)[:160]}", file=sys.stderr)
            break
        if not batch:
            break
        for p in batch:
            pid = p.get("id")
            if pid is None or pid in seen_pid:
                continue
            seen_pid.add(pid)
            # タグ文字列 + rating のみ保持 (画像 URL は捨てる)
            posts.append({
                "post_id": pid,
                "rating": p.get("rating", "g"),
                "tags_general": p.get("tag_string_general", ""),
                # character/copyright は記録のみ (target 非展開: §4)
                "n_character": len(p.get("tag_string_character", "").split()),
            })
        cursor = min(p["id"] for p in batch)
        print(f"[pairs] 取得 {len(posts)}/{target_posts} posts "
              f"(cursor=b{cursor})", file=sys.stderr)
        time.sleep(sleep_s)

    # キャッシュ書き出し (post_id 降順で安定保存)
    posts.sort(key=lambda p: -p["post_id"])
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        for p in posts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"[pairs] キャッシュ書き出し: {cache_path} ({len(posts)} posts)",
          file=sys.stderr)
    return posts


# ------------------------------------------------------------
# 自然文逆生成テンプレート (タグ列 → 日本語/英語)
# ------------------------------------------------------------
# 主体数タグ → 訳語
SUBJECT_JA = {
    "1girl": "女の子が一人", "2girls": "女の子が二人", "3girls": "女の子が三人",
    "1boy": "男の子が一人", "2boys": "男の子が二人",
    "multiple girls": "複数の女の子", "multiple boys": "複数の男の子",
}
SUBJECT_EN = {
    "1girl": "a girl", "2girls": "two girls", "3girls": "three girls",
    "1boy": "a boy", "2boys": "two boys",
    "multiple girls": "several girls", "multiple boys": "several boys",
}

# テンプレ多様化用の文型バリアント (post_id を seed にして決定的に選ぶ →
# 同一タグ集合でも文面が散る = ユニーク text 比率を底上げ。再現性は post_id 依存で担保)。
JA_TEMPLATES = [
    # 0: A の subj、(color) (pose) (expr) (comp/iso)
    lambda subj, ap, color, pose, expr, comp, iso:
        _ja_join(f"{ap}の{subj}" if ap else subj, color, pose, expr, comp, iso),
    # 1: subj。A。(pose で) (expr)
    lambda subj, ap, color, pose, expr, comp, iso:
        _ja_join2(subj, ap, color, pose, expr, comp, iso),
    # 2: 「～を描いて」調
    lambda subj, ap, color, pose, expr, comp, iso:
        _ja_join3(subj, ap, color, pose, expr, comp, iso),
]
EN_TEMPLATES = [
    # 0: subj with A, (color), (pose), (expr), comp, iso.
    lambda subj, ap, color, pose, expr, comp, iso:
        _en_join(subj, ap, color, pose, expr, comp, iso),
    # 1: A subj who is (pose), (expr).
    lambda subj, ap, color, pose, expr, comp, iso:
        _en_join2(subj, ap, color, pose, expr, comp, iso),
    # 2: draw/illustrate phrasing
    lambda subj, ap, color, pose, expr, comp, iso:
        _en_join3(subj, ap, color, pose, expr, comp, iso),
]


def _ja_join(head, color, pose, expr, comp, iso):
    parts = [head]
    if color:
        parts.append("・".join(color) + " 調で")
    if pose:
        parts.append("・".join(pose[:2]))
    if expr:
        parts.append("・".join(expr[:2]) + "の表情")
    tail = []
    if comp:
        tail.append(comp[0])
    if iso:
        tail.append(iso[0])
    s = "、".join(parts)
    return s + ("。" + "・".join(tail) + "。" if tail else "。")


def _ja_join2(subj, ap, color, pose, expr, comp, iso):
    s = subj + "。"
    if ap:
        s += "特徴は" + "・".join(ap[:4]) + "。"
    mid = []
    if color:
        mid.append("・".join(color) + " 調")
    if pose:
        mid += pose[:2]
    if expr:
        mid.append("・".join(expr[:2]) + "の表情")
    if mid:
        s += "、".join(mid) + "。"
    tail = []
    if comp:
        tail.append(comp[0])
    if iso:
        tail.append(iso[0])
    if tail:
        s += "・".join(tail) + "。"
    return s


def _ja_join3(subj, ap, color, pose, expr, comp, iso):
    desc = []
    if ap:
        desc.append("・".join(ap[:4]) + "の")
    head = ("".join(desc)) + subj
    extra = []
    if color:
        extra.append("・".join(color) + " 調")
    if pose:
        extra += pose[:2]
    if expr:
        extra.append("・".join(expr[:2]))
    if comp:
        extra.append(comp[0])
    if iso:
        extra.append(iso[0])
    s = head
    if extra:
        s += "を、" + "・".join(extra) + "で"
    return s + "描いてください。"


def _en_join(subj, ap, color, pose, expr, comp, iso):
    clause = subj
    if ap:
        clause += " with " + ", ".join(ap[:4])
    bits = [clause]
    if color:
        bits.append("in " + " ".join(color) + " style")
    if pose:
        bits.append(", ".join(pose[:2]))
    if expr:
        bits.append(", ".join(expr[:2]))
    if comp:
        bits.append(comp[0])
    if iso:
        bits.append(iso[0])
    return ", ".join(bits) + "."


def _en_join2(subj, ap, color, pose, expr, comp, iso):
    clause = subj
    if ap:
        clause += " having " + ", ".join(ap[:4])
    tail = []
    if pose:
        tail += pose[:2]
    if expr:
        tail += expr[:2]
    if tail:
        clause += " who is " + ", ".join(tail)
    bits = [clause]
    if color:
        bits.append("rendered in " + " ".join(color) + " style")
    if comp:
        bits.append(comp[0])
    if iso:
        bits.append(iso[0])
    return ", ".join(bits) + "."


def _en_join3(subj, ap, color, pose, expr, comp, iso):
    desc = []
    if ap:
        desc.append(", ".join(ap[:4]))
    if color:
        desc.append("in " + " ".join(color) + " style")
    if pose:
        desc.append(", ".join(pose[:2]))
    if expr:
        desc.append(", ".join(expr[:2]))
    if comp:
        desc.append(comp[0])
    if iso:
        desc.append(iso[0])
    s = "Please draw " + subj
    if desc:
        s += ": " + ", ".join(desc)
    return s + "."


def build_text(buckets: dict[int, list[str]], lang: str, variant: int) -> str:
    """バケット分類済みタグから自然文を逆生成する。

    variant (0..2) は post_id 由来で決定的に選ばれ、同一タグ集合でも文面が散る。
    """
    canon = buckets.get(1, [])
    color = buckets.get(2, [])
    pose = buckets.get(3, [])
    expr = buckets.get(4, [])
    comp = buckets.get(5, [])
    iso = buckets.get(6, [])

    subject_tags = [t for t in canon if t in SUBJECT_JA]
    appearance = [t for t in canon if t not in SUBJECT_JA and t != "solo"]

    if lang == "ja":
        subj = SUBJECT_JA.get(subject_tags[0], "キャラクター") if subject_tags else "キャラクター"
        ap = "・".join(appearance[:4]) if appearance else ""
        # variant 0 は head 文字列、1/2 は appearance リストを受ける形 → 統一して渡す
        tmpl = JA_TEMPLATES[variant % len(JA_TEMPLATES)]
        if variant % len(JA_TEMPLATES) == 0:
            return tmpl(subj, ap, color, pose, expr, comp, iso)
        return tmpl(subj, appearance, color, pose, expr, comp, iso)
    else:  # en
        subj = SUBJECT_EN.get(subject_tags[0], "a character") if subject_tags else "a character"
        tmpl = EN_TEMPLATES[variant % len(EN_TEMPLATES)]
        return tmpl(subj, appearance, color, pose, expr, comp, iso)


def make_pair(post: dict, vocab_set: set[str], freq: dict[str, int],
              lang: str) -> dict | None:
    """1 投稿のタグ集合 → 正準順序ペアを作る。vocab 外タグは落とす。"""
    raw = post["tags_general"].split()
    # 区切り正規化 → vocab 射影
    proj = []
    seen = set()
    for t in raw:
        n = normalize_separator(t)
        if n in vocab_set and n not in seen:
            seen.add(n)
            proj.append(n)
    if len(proj) < 4:  # タグが少なすぎる投稿はスキップ
        return None

    # バケット分類 → バケット内 freq 降順安定ソート → バケット順連結
    buckets: dict[int, list[str]] = {}
    for t in proj:
        buckets.setdefault(classify_bucket(t), []).append(t)
    for b in buckets:
        buckets[b].sort(key=lambda x: (-freq.get(x, 0), x))

    ordered = []
    for b in (1, 2, 3, 4, 5, 6):
        ordered.extend(buckets.get(b, []))

    # タグ数を 16 に抑える (バケット順を保ったまま先頭 16 件)
    if len(ordered) > 16:
        ordered = ordered[:16]
        buckets = {}
        for t in ordered:
            buckets.setdefault(classify_bucket(t), []).append(t)
        for b in buckets:
            buckets[b].sort(key=lambda x: (-freq.get(x, 0), x))

    # テンプレ多様化: post_id を seed に決定的に variant を選ぶ (再現的)
    variant = post["post_id"] % 3
    text = build_text(buckets, lang, variant)
    return {
        "text": text,
        "tags": ordered,
        "lang": lang,
        "source": "synthetic",
        "meta": {
            "rating": post["rating"],
            "post_id": post["post_id"],
            "n_tags": len(ordered),
            "tmpl": variant,
        },
    }


# ------------------------------------------------------------
# 検証 (スキーマ妥当・vocab 内・順序正準・負語混入なし)
# ------------------------------------------------------------
NEGATIVE_BLOCKLIST = {
    "bad hands", "malformed hands", "mutated hands", "fused fingers",
    "bad anatomy", "deformed", "lowres", "worst quality", "jpeg artifacts",
    "extra fingers", "extra digits", "fewer digits", "missing fingers",
}


def validate_pairs(pairs: list[dict], vocab_set: set[str]) -> list[str]:
    """ペア集合を検証し、エラー文字列のリストを返す (空なら合格)。"""
    errors = []
    for i, p in enumerate(pairs):
        if set(p.keys()) < {"text", "tags", "lang", "source", "meta"}:
            errors.append(f"#{i}: キー不足 {p.keys()}")
        if p["lang"] not in ("ja", "en"):
            errors.append(f"#{i}: lang 不正 {p['lang']}")
        if not p["text"]:
            errors.append(f"#{i}: text 空")
        # 全 tags が vocab 内
        for t in p["tags"]:
            if t not in vocab_set:
                errors.append(f"#{i}: vocab 外タグ '{t}'")
            if t in NEGATIVE_BLOCKLIST:
                errors.append(f"#{i}: 負語混入 '{t}'")
        # 順序正準: バケット番号が非減少
        bks = [classify_bucket(t) for t in p["tags"]]
        if bks != sorted(bks):
            errors.append(f"#{i}: 順序非正準 buckets={bks} tags={p['tags']}")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description="dollama BitNet 訓練ペア生成")
    ap.add_argument("--n", type=int, default=500, help="train ペア数")
    ap.add_argument("--val", type=int, default=25, help="val ペア数")
    ap.add_argument("--seed", type=int, default=20260620)
    ap.add_argument("--vocab", default="data/bitnet/vocab.json")
    ap.add_argument("--out-dir", default="data/bitnet")
    ap.add_argument("--ratings", default="g", help="取得 rating (カンマ区切り: g,s,q,e)")
    ap.add_argument("--sleep", type=float, default=1.0, help="リクエスト間 sleep 秒")
    ap.add_argument("--fetch-factor", type=float, default=1.6,
                    help="目標件数に対する取得倍率 (射影/重複で落ちる分の余裕)")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # vocab 読み込み
    with open(args.vocab, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    vocab_set = {t["tag"] for t in vocab["tags"]}
    freq = {t["tag"]: t["freq"] for t in vocab["tags"]}

    ratings = [r.strip() for r in args.ratings.split(",") if r.strip()]
    total_needed = args.n + args.val
    # 射影/重複で落ちる投稿があるので多めに取得
    cache_path = os.path.join(args.out_dir, "cache", "danbooru_posts.jsonl")
    posts = fetch_posts(target_posts=int(total_needed * args.fetch_factor) + 50,
                        ratings=ratings, sleep_s=args.sleep,
                        cache_path=cache_path)

    if not posts:
        print("[pairs] 投稿を取得できませんでした。中止。", file=sys.stderr)
        return 1

    # 投稿をシャッフルし、各投稿に ja/en を交互割当 (言語比 ~0.5)
    rng.shuffle(posts)
    pairs: list[dict] = []
    seen_pid = set()
    seen_text = set()
    seen_pairkey = set()  # (text, tags) 完全重複の除去
    for idx, post in enumerate(posts):
        if post["post_id"] in seen_pid:
            continue
        lang = "ja" if (idx % 2 == 0) else "en"
        pair = make_pair(post, vocab_set, freq, lang)
        if pair is None:
            continue
        pairkey = (pair["text"], tuple(pair["tags"]))
        if pairkey in seen_pairkey:  # 同一 text かつ同一 tags の重複は除去 (件数に数えない)
            continue
        if pair["text"] in seen_text:  # 同一 text の重複も除去 (text リーク防止の前提)
            continue
        seen_pid.add(post["post_id"])
        seen_text.add(pair["text"])
        seen_pairkey.add(pairkey)
        pairs.append(pair)
        if len(pairs) >= total_needed:
            break

    if len(pairs) < total_needed:
        print(f"[pairs] 警告: 歩留り不足 {len(pairs)}/{total_needed}。"
              f"--fetch-factor を上げて再取得してください。", file=sys.stderr)

    # 検証
    errors = validate_pairs(pairs, vocab_set)
    if errors:
        print(f"[pairs] 検証エラー {len(errors)} 件 (先頭10):", file=sys.stderr)
        for e in errors[:10]:
            print("   ", e, file=sys.stderr)
        return 2
    print(f"[pairs] 検証 OK ({len(pairs)} ペア)", file=sys.stderr)

    # 分割 (post 単位・text 単位で既に重複除去済み → そのまま split)
    rng.shuffle(pairs)
    val = pairs[:args.val]
    train = pairs[args.val:args.val + args.n]

    # リーク防止アサート
    train_pid = {p["meta"]["post_id"] for p in train}
    val_pid = {p["meta"]["post_id"] for p in val}
    assert not (train_pid & val_pid), "post_id リーク検出"
    train_txt = {p["text"] for p in train}
    val_txt = {p["text"] for p in val}
    assert not (train_txt & val_txt), "text リーク検出"

    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, "pairs.train.jsonl")
    val_path = os.path.join(args.out_dir, "pairs.val.jsonl")
    with open(train_path, "w", encoding="utf-8") as f:
        for p in train:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(val_path, "w", encoding="utf-8") as f:
        for p in val:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # stats.json
    allp = train + val
    lang_c = {"ja": 0, "en": 0}
    rating_c: dict[str, int] = {}
    src_c: dict[str, int] = {}
    tag_freq: dict[str, int] = {}
    tmpl_c: dict[int, int] = {}
    ntags = []
    texts = []
    for p in allp:
        lang_c[p["lang"]] += 1
        rating_c[p["meta"]["rating"]] = rating_c.get(p["meta"]["rating"], 0) + 1
        src_c[p["source"]] = src_c.get(p["source"], 0) + 1
        tmpl_c[p["meta"]["tmpl"]] = tmpl_c.get(p["meta"]["tmpl"], 0) + 1
        ntags.append(len(p["tags"]))
        texts.append(p["text"])
        for t in p["tags"]:
            tag_freq[t] = tag_freq.get(t, 0) + 1
    n = len(allp)
    uniq_text = len(set(texts))
    stats = {
        "version": 1,
        "seed": args.seed,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "counts": {"total": n, "train": len(train), "val": len(val)},
        "lang_ratio": {k: round(v / n, 4) for k, v in lang_c.items()},
        "source_ratio": {k: round(v / n, 4) for k, v in src_c.items()},
        "rating_ratio": {k: round(v / n, 4) for k, v in rating_c.items()},
        "split_ratio": {"train": round(len(train) / n, 4),
                        "val": round(len(val) / n, 4)},
        "unique_text_ratio": round(uniq_text / n, 4),
        "template_dist": {str(k): v for k, v in sorted(tmpl_c.items())},
        "vocab": {
            "total_tags": len(vocab["tags"]),
            "cutoff_min_count": vocab.get("_build", {}).get("min_count"),
            "raw_tags": vocab.get("_build", {}).get("raw_tags"),
        },
        "tag_freq_top": sorted(tag_freq.items(), key=lambda x: -x[1])[:20],
        "tags_per_pair": {
            "min": min(ntags), "max": max(ntags),
            "mean": round(sum(ntags) / len(ntags), 2),
        },
        "sources": {
            "wd14_csv": "SmilingWolf/wd-swinv2-tagger-v3/selected_tags.csv",
            "danbooru_api": "danbooru.donmai.us/posts.json (tags-only, no pixels)",
        },
    }
    stats_path = os.path.join(args.out_dir, "stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)

    print(f"[pairs] train={len(train)} val={len(val)} → {train_path}, {val_path}",
          file=sys.stderr)
    print(f"[pairs] stats: {stats_path}", file=sys.stderr)
    print(f"[pairs] lang_ratio={stats['lang_ratio']} "
          f"unique_text_ratio={stats['unique_text_ratio']} "
          f"tags/pair mean={stats['tags_per_pair']['mean']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

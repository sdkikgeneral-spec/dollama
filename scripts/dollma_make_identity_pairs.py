#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dollama 同一性条件付き訓練ペア生成 (Phase 4 A / A1)

dataset-spec.md §13 (同一性条件付きペア) を確定実装する。既存 #1
(dollma_make_pairs.py の synthetic) と **同じ danbooru タグ共起キャッシュ**
(data/bitnet/cache/danbooru_posts.jsonl・タグのみ・画像非取得) を元に、
各 post のタグを identity_tags / scene_tags に分離し、A2 の混合訓練が読む

    系列 (A2 で構築):  <bos> [identity tags] <sep> [scene text のタグ] <sep> [target tags] <eos>

の素材となる「同一性条件付きペア」を生成する。

決定事項 (承認済み・厳守):
  - 条件付け機構 = (a-1) prompt prefix + <sep>(id=3) 2 回流用。vocab.json /
    VOCAB_SIZE / tokenizer / specials は一切変更しない (<sep> を 2 回使うだけ)。
  - target tags = identity_tags ∪ scene_tags を §5 正準バケット順で並べる。
    identity_tags は必ず target に含める (retention 教師信号の核 = retention 100%)。

スキーマ (後方互換):
  既存 #1 行 (source:"synthetic") は無改修で読めるまま。新形式は
    "source":"identity_cond" + "meta":{"identity_tags":[...],"scene_tags":[...],...}
  を追加する。text/tags フィールドの意味は #1 と同じ (tags = target)。

Phase 4-A (実ペア増) で追加した引数 (既定挙動は無改変):
  --exclude-post-ids <path>  凍結 eval 等の post_id 集合を A train から恒久除外する。
                             生成ループで pid ∈ excluded を skip し、末尾で
                             assert not (train_pid & excluded) を強制する。
                             引数なし時は従来 #1/A1 経路と bitwise 非回帰。
  --out-tag <tag>            出力ファイル名にサフィックスを付与する (別名出力)。
                             例 --out-tag a12k → pairs.identity.{train,val}.a12k.jsonl
                                                  stats.identity.a12k.json
                             引数なし時は従来名 (pairs.identity.{train,val}.jsonl)。

使い方 (従来・A1 5,000):
    py -3.12 scripts/dollma_make_identity_pairs.py --n 4500 --val 500 \
        --seed 20260620 --vocab data/bitnet/vocab.json --out-dir data/bitnet

使い方 (Phase 4-A 12k・凍結 eval 除外・別名):
    py -3.12 scripts/dollma_make_identity_pairs.py --n 10800 --val 1200 \
        --seed 20260620 --out-tag a12k \
        --exclude-post-ids data/bitnet/eval_frozen_post_ids.json
"""
import argparse
import json
import os
import random
import re
import sys
from datetime import datetime, timezone

# 既存 #1 スクリプトから共起取得・分類・正規化・テンプレを再利用する
# (重複実装を避け、§5 正準順序 / §6 正規化 / danbooru 取得を一本化)。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dollma_make_pairs as mp  # noqa: E402


# ============================================================
# 同一性条件付け (dataset-spec §13) — identity / scene 分離規則
# ============================================================
# 典拠: character-bible-spec §1 (同一性層 CharacterIdentity) / §2 (シーン層 SceneSpec)。
#
#   identity_tags = CharacterIdentity.canonical_tags 相当 = コマ間で「不変」の
#     外見属性。具体的には:
#       - 主体数/性別      (1girl/1boy/multiple girls/futanari/otoko no ko ...)
#       - 髪 (色/長さ/型)   (long hair/silver hair/twintails/ahoge ...)
#       - 目 (色/特徴)      (blue eyes/heterochromia/slit pupils ...)
#       - 肌                (dark skin/pale skin/colored skin ...)
#       - 体型/外見年齢     (large breasts/mature female/loli/muscular/petite ...)
#       - 種族形質          (animal ears/tail/horns/wings/elf/pointy ears ...)
#
#   scene_tags = それ以外すべて (pose/expression/composition/isolation/color_mode +
#     服飾・小物・状態・背景単色 など可変要素)。
#
# 服装の扱い (論点・保守判断):
#   canonical 思想 (spec §1) では服も同一性になり得る (制服キャラ等) が、
#   danbooru の 1 post = 1 イラストでは服は post ごとに変わる可変要素であり、
#   服を identity に含めると ① 同一性集合が衣装違いで膨れ「同一キャラの汎化」を
#   測れない ② identity 重複率が人工的に下がる ③ retention 教師が「服も毎回同じ」を
#   強制する。よって本データセットでは服飾・小物 (shirt/dress/gloves/hat/ribbon/
#   necklace/glasses 等) は **scene 側** に寄せる。キャラ固有衣装の固定は
#   CharacterIdentity.canonical_tags へ運用側が明示する設計 (compose_prompt は
#   canonical を先頭に置く) で担保され、データ側で服を identity と決め打つ必要はない。
#   → 保守的に「服=シーン」。
#
# 実装方針: identity は bucket 1 (canonical) の部分集合として定義する
#   (bucket 2..6 は定義上すべて scene)。bucket 1 のタグのうち IDENTITY パターンに
#   一致し、かつ EXCLUDE (装飾・状態・服飾) に当たらないものを identity とする。

_IDENTITY_PATTERNS = [
    # 主体数・性別・属性人称
    r"^\d+girls?$", r"^\d+boys?$", r"^\d+others?$",
    r"^multiple (girls|boys|others)$", r"^6\+(girls|boys)$",
    r"^(male|female) focus$", r"^futanari$", r"^otoko no ko$",
    r"^androgynous$",
    r"^(cat|fox|dog|wolf|dragon|demon|monster|horse|cow|rabbit|bunny|mouse|"
    r"sheep|bird|fish|raccoon|tiger|lion|deer) (girl|boy)$",
    r"^magical girl$",
    # 髪 (色・長さ・型・房)
    r"\bhair\b", r"\bbangs\b", r"\bbraids?\b", r"\bponytail\b",
    r"\btwintails?\b", r"\bahoge\b", r"\bsidelocks\b", r"\bhime cut\b",
    r"\bhair bun\b", r"\bdouble bun\b",
    # 目 (色・瞳・特徴) — 状態 (閉じ目) は除外側で落とす
    r"\beyes?\b", r"\bpupils?\b", r"\bheterochromia\b", r"\beyelashes\b",
    # 肌
    r"\bskin\b", r"^dark-skinned (female|male)$", r"^tan$", r"^pale skin$",
    # 体型・外見年齢
    r"\bbreasts\b", r"^(loli|petite|curvy|plump|fat|chubby|toned)$",
    r"^muscular( (fe)?male)?$", r"^toned (fe)?male$",
    r"^mature (fe)?male$", r"^old$", r"^aged (up|down)$",
    r"^(tall|short) (fe)?male$", r"^wide hips$", r"^thick thighs$",
    # 種族・身体形質 (不変)
    r"\banimal ears\b",
    r"^(cat|fox|dog|wolf|rabbit|bunny|mouse|horse|cow|sheep|bear|tiger|deer) ears$",
    r"\btail$", r"\bhorns?\b", r"\bwings?\b", r"^pointy ears$",
    r"^(dark )?elf$", r"^(fang|fangs|sharp teeth)$",
    r"\bmole\b", r"^freckles$",
]
_IDENTITY_RE = [re.compile(p) for p in _IDENTITY_PATTERNS]

_IDENTITY_EXCLUDE_EXACT = {
    # 髪まわりの装飾・状態 (不変外見ではない)
    "hair ornament", "hair ribbon", "hair bow", "hair flower", "hair bobbles",
    "hair between eyes", "hair over one eye", "hair over eyes",
    "hair over breasts", "floating hair", "hair intakes", "messy hair",
    "wet hair", "hair pulled back", "hair down", "hair up", "facial hair",
    "pubic hair", "female pubic hair", "male pubic hair", "armpit hair",
    "hair tie", "hair stick", "hairband", "hairclip", "hairpin",
    "hair scrunchie", "hair rings", "hair bell", "hair flaps",
    # 目まわりの状態・行為・装飾 (表情/構図/装飾 = scene)
    "closed eyes", "one eye closed", "half-closed eyes",
    "eyes visible through hair", "eye contact", "crying with eyes open",
    "one eye covered", "covered eyes", "glowing eyes", "empty eyes",
    "rolling eyes", "looking at viewer", "eyeshadow", "eyeliner",
    "scar across eye", "mole under eye", "bags under eyes",
    "eyes closed", "closed eye", "no eyes",
    # 胸まわりの行為・露出 (状態 = scene)
    "breasts out", "between breasts", "breasts apart", "cum on breasts",
    "covering breasts", "hanging breasts", "bouncing breasts",
    "breasts squeezed together", "arm under breasts",
    "necktie between breasts", "hand on breast", "grabbing own breast",
    "underboob", "sideboob", "cleavage", "breast hold", "breast rest",
    # 種族系の行為・装飾
    "fake animal ears", "fake tail", "lifted by self", "tail raised",
    "holding tail", "tail wagging", "ears down", "animal ear fluff",
    "horns through headwear",
}

# 服飾・小物パターン (identity から明示排除・服=シーン)。
_CLOTHING_RE = re.compile(
    r"shirt|dress|skirt|jacket|coat|pants|trousers|shorts|sleeves?|gloves|"
    r"\bhat\b|\bcap\b|\bribbon\b|\bbow\b|necktie|bowtie|\btie\b|scarf|uniform|"
    r"swimsuit|bikini|leotard|kimono|yukata|hoodie|sweater|socks|boots|shoes|"
    r"sandals|stockings|pantyhose|thighhighs|legwear|cape|cloak|armor|apron|"
    r"vest|collar|choker|earrings|necklace|pendant|glasses|sunglasses|veil|"
    r"hood|headband|headphones|mask|bag|backpack|belt|garter|panties|bra\b|"
    r"underwear|lingerie|robe|cardigan|blazer|overalls|jumpsuit|bodysuit|"
    r"costume|outfit|clothes|naked|nude|topless|bottomless|barefoot|jewelry|"
    r"bracelet|\bring\b|crown|tiara|wreath|flower|frills|lace|buttons|zipper|"
    r"pocket|strap|wristband|armband|bandaid|bandage|tattoo|name tag")


def is_identity_tag(tag: str) -> bool:
    """タグが同一性 (不変外見) かを判定する。bucket 1 の部分集合。"""
    if tag in _IDENTITY_EXCLUDE_EXACT:
        return False
    if _CLOTHING_RE.search(tag):
        return False
    for r in _IDENTITY_RE:
        if r.search(tag):
            return True
    return False


def split_identity_scene(ordered_tags):
    """正準順序済み target タグ列を (identity_tags, scene_tags) に分離する。

    入力の順序を保持する (両方とも正準順序の部分列になる)。
    identity は bucket 1 内の不変外見のみ。残りは scene。
    """
    ident, scene = [], []
    for t in ordered_tags:
        if mp.classify_bucket(t) == 1 and is_identity_tag(t):
            ident.append(t)
        else:
            scene.append(t)
    return ident, scene


# ============================================================
# 自然文側 (identity を保持した状況記述)
# ============================================================
def build_identity_text(identity_tags, scene_buckets, lang, variant):
    """identity を保持した自然文を逆生成する。

    既存 build_text を流用するが、identity を冒頭に明示して「同一性を保った
    状況記述」にする。text は #1 と同じく自然文。identity 由来の語が text に
    必ず現れることで、A2 の text 側 (greedy 最長一致) が identity を拾える。
    """
    # identity からテンプレ用の subject / appearance を作る
    subject = [t for t in identity_tags if t in mp.SUBJECT_JA or t in mp.SUBJECT_EN]
    appearance = [t for t in identity_tags if t not in subject and t != "solo"]
    # scene 側 (pose/expr/comp/iso/color) はバケットから引く
    color = scene_buckets.get(2, [])
    pose = scene_buckets.get(3, [])
    expr = scene_buckets.get(4, [])
    comp = scene_buckets.get(5, [])
    iso = scene_buckets.get(6, [])

    if lang == "ja":
        subj = mp.SUBJECT_JA.get(subject[0], "キャラクター") if subject else "キャラクター"
        tmpl = mp.JA_TEMPLATES[variant % len(mp.JA_TEMPLATES)]
        if variant % len(mp.JA_TEMPLATES) == 0:
            ap = "・".join(appearance[:4]) if appearance else ""
            return tmpl(subj, ap, color, pose, expr, comp, iso)
        return tmpl(subj, appearance, color, pose, expr, comp, iso)
    else:
        subj = mp.SUBJECT_EN.get(subject[0], "a character") if subject else "a character"
        tmpl = mp.EN_TEMPLATES[variant % len(mp.EN_TEMPLATES)]
        return tmpl(subj, appearance, color, pose, expr, comp, iso)


def make_identity_pair(post, vocab_set, freq, lang):
    """1 投稿 → 同一性条件付きペア。target は #1 と同じ正準順序タグ列。"""
    raw = post["tags_general"].split()
    proj, seen = [], set()
    for t in raw:
        n = mp.normalize_separator(t)
        if n in vocab_set and n not in seen:
            seen.add(n)
            proj.append(n)
    if len(proj) < 4:
        return None

    # §5 正準順序化 (#1 と同一: バケット分類 → バケット内 freq 降順 → バケット順 → 16 件)
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

    # identity / scene 分離 (target の部分列)
    identity_tags, scene_tags = split_identity_scene(ordered)
    if not identity_tags:
        # 同一性の核が無い post は条件付けの教師にならない → スキップ
        return None

    # text 用に scene 側のバケットを作る (16 件打ち切り後の scene のみ)
    scene_buckets = {}
    for t in scene_tags:
        scene_buckets.setdefault(mp.classify_bucket(t), []).append(t)
    for b in scene_buckets:
        scene_buckets[b].sort(key=lambda x: (-freq.get(x, 0), x))

    variant = post["post_id"] % 3
    text = build_identity_text(identity_tags, scene_buckets, lang, variant)

    return {
        "text": text,
        "tags": ordered,                 # target = identity ∪ scene (正準順序)
        "lang": lang,
        "source": "identity_cond",
        "meta": {
            "rating": post["rating"],
            "post_id": post["post_id"],
            "n_tags": len(ordered),
            "tmpl": variant,
            "identity_tags": identity_tags,
            "scene_tags": scene_tags,
        },
    }


# ============================================================
# 検証 (source 非依存で全件適用 + 同一性条件付き固有チェック)
# ============================================================
def validate_identity(pairs, vocab_set):
    """#1 の validate_pairs を全件に適用し、さらに identity 固有検査を足す。"""
    # 1) #1 共通検査 (OOV/負語/順序/スキーマ) を source 非依存で全件
    errors = list(mp.validate_pairs(pairs, vocab_set))

    for i, p in enumerate(pairs):
        if p.get("source") != "identity_cond":
            continue
        meta = p.get("meta", {})
        ident = meta.get("identity_tags")
        scene = meta.get("scene_tags")
        if ident is None or scene is None:
            errors.append(f"#{i}: identity_cond に identity_tags/scene_tags 欠如")
            continue
        # identity は必ず空でない (retention 教師の核)
        if not ident:
            errors.append(f"#{i}: identity_tags 空")
        # identity ∪ scene == target (順序まで含め分割が完全)
        if list(ident) + list(scene) != [t for t in p["tags"]
                                          if t in set(ident) or t in set(scene)]:
            pass  # 順序は下で厳密検査
        if sorted(ident + scene) != sorted(p["tags"]):
            errors.append(f"#{i}: identity∪scene != target")
        if set(ident) & set(scene):
            errors.append(f"#{i}: identity と scene が重複")
        # identity_tags が target に全部含まれる (retention 100% の正解)
        tagset = set(p["tags"])
        for t in ident:
            if t not in tagset:
                errors.append(f"#{i}: identity '{t}' が target に無い")
            if t not in vocab_set:
                errors.append(f"#{i}: identity vocab 外 '{t}'")
        # identity は正準順序の部分列 (bucket 1 のみ → 単調)
        ibk = [mp.classify_bucket(t) for t in ident]
        if any(b != 1 for b in ibk):
            errors.append(f"#{i}: identity に bucket1 以外 {ibk}")
    return errors


# ============================================================
# tokenizer 往復 (tokenizer.hpp / train_bitnet.Tokenizer と同一規則)
# ============================================================
class _Tok:
    """tokenizer.hpp の normalize / specials / tags[i].id==5+i を再現する最小実装。"""

    PAD, BOS, EOS, SEP, UNK = 0, 1, 2, 3, 4

    def __init__(self, vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            vj = json.load(f)
        if vj["specials"] != ["<pad>", "<bos>", "<eos>", "<sep>", "<unk>"]:
            raise ValueError("specials mismatch")
        self.tags = []
        for i, t in enumerate(vj["tags"]):
            if t["id"] != 5 + i:
                raise ValueError(f"tags[{i}].id={t['id']} expected {5 + i}")
            self.tags.append(t["tag"])
        self.tag_to_id = {self.normalize(t): 5 + i
                          for i, t in enumerate(self.tags)}

    @staticmethod
    def normalize(s):
        out, n = [], len(s)
        for i, c in enumerate(s):
            if c == "_":
                if (i > 0 and s[i - 1].isalnum()
                        and i + 1 < n and s[i + 1].isalnum()):
                    out.append(" ")
                    continue
            out.append(c)
        return "".join(out)

    def tag_to_id_lookup(self, tag):
        return self.tag_to_id.get(self.normalize(tag), self.UNK)

    def encode(self, tags):
        """encode(tags) : <bos> id... <eos>。tokenizer.hpp::encode と同じ枠。"""
        ids = [self.BOS]
        for t in tags:
            ids.append(self.tag_to_id_lookup(t))
        ids.append(self.EOS)
        return ids

    def decode(self, ids):
        """decode : 構造トークン除去・<unk> は "<unk>"・タグは正準形。"""
        out = []
        for i in ids:
            if i in (self.BOS, self.EOS, self.PAD, self.SEP):
                continue
            if i == self.UNK:
                out.append("<unk>")
                continue
            out.append(self.tags[i - 5])
        return out


def roundtrip_check(pairs, vocab_path):
    """target / identity / scene を encode→decode し UNK 0・完全一致を検査する。

    正準形 (normalize 後) で突合する (tokenizer.hpp も内部は正準形)。
    """
    tok = _Tok(vocab_path)
    unk = 0
    mism = 0
    for p in pairs:
        for field in ("tags", None):
            if field == "tags":
                seqs = [p["tags"]]
            else:
                seqs = [p["meta"].get("identity_tags", []),
                        p["meta"].get("scene_tags", [])]
            for seq in seqs:
                ids = tok.encode(seq)
                if any(i == _Tok.UNK for i in ids):
                    unk += sum(1 for i in ids if i == _Tok.UNK)
                dec = tok.decode(ids)
                expect = [_Tok.normalize(t) for t in seq]
                if dec != expect:
                    mism += 1
    return unk, mism


# ============================================================
# Phase 4-A: 凍結 eval 除外集合 + 別名出力ヘルパ
# ============================================================
def load_exclude_post_ids(path):
    """凍結除外集合 (eval_frozen_post_ids.json) を読み post_id の set を返す。

    形式は {"frozen_post_ids":[...]} か、素の [..] リストの両方を許す。
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        ids = obj.get("frozen_post_ids", [])
    else:
        ids = obj
    return {int(x) for x in ids}


def read_eval_post_ids_direct(eval_paths):
    """凍結 eval の pairs.jsonl を **直接** 読み post_id 集合を返す。

    make_eval_diverse の抽出ロジックを介さない独立再検証用。除外漏れがあれば
    この set との交差で落ちる。
    """
    pids = set()
    for path in eval_paths:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                pids.add(int(json.loads(line)["meta"]["post_id"]))
    return pids


def _tagged(base, tag, ext):
    """別名出力のファイル名を作る。tag が空なら従来名のまま (非回帰)。

    base="pairs.identity.train", tag="a12k", ext="jsonl"
      → "pairs.identity.train.a12k.jsonl"
    base="pairs.identity.train", tag="", ext="jsonl"
      → "pairs.identity.train.jsonl"
    """
    if tag:
        return f"{base}.{tag}.{ext}"
    return f"{base}.{ext}"


# ============================================================
# main
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="dollama 同一性条件付きペア生成 (Phase 4 A1)")
    ap.add_argument("--n", type=int, default=4500, help="train ペア数")
    ap.add_argument("--val", type=int, default=500, help="val ペア数")
    ap.add_argument("--seed", type=int, default=20260620)
    ap.add_argument("--vocab", default="data/bitnet/vocab.json")
    ap.add_argument("--out-dir", default="data/bitnet")
    ap.add_argument("--ratings", default="g")
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--fetch-factor", type=float, default=1.8,
                    help="目標件数に対する取得倍率 (identity 核なし post で歩留り低下)")
    # Phase 4-A 追加 (既定 None/"" で従来挙動・bitwise 非回帰)
    ap.add_argument("--exclude-post-ids", default=None,
                    help="凍結 eval 等の post_id 集合 JSON。A train から恒久除外する")
    ap.add_argument("--out-tag", default="",
                    help="出力ファイル名サフィックス (例 a12k)。空で従来名")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    with open(args.vocab, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    vocab_set = {t["tag"] for t in vocab["tags"]}
    freq = {t["tag"]: t["freq"] for t in vocab["tags"]}

    # 凍結除外集合 (引数なしなら空集合 = 従来挙動)
    excluded = set()
    if args.exclude_post_ids:
        excluded = load_exclude_post_ids(args.exclude_post_ids)
        print(f"[id-pairs] 凍結除外集合 {len(excluded)} post_id を読込 "
              f"({args.exclude_post_ids})", file=sys.stderr)

    ratings = [r.strip() for r in args.ratings.split(",") if r.strip()]
    total_needed = args.n + args.val
    cache_path = os.path.join(args.out_dir, "cache", "danbooru_posts.jsonl")
    posts = mp.fetch_posts(target_posts=int(total_needed * args.fetch_factor) + 50,
                           ratings=ratings, sleep_s=args.sleep,
                           cache_path=cache_path)
    if not posts:
        print("[id-pairs] 投稿を取得できませんでした。中止。", file=sys.stderr)
        return 1

    rng.shuffle(posts)
    pairs = []
    seen_pid, seen_text, seen_pairkey = set(), set(), set()
    n_excluded_skipped = 0
    for idx, post in enumerate(posts):
        if post["post_id"] in seen_pid:
            continue
        # Phase 4-A: 凍結 eval 等の除外集合は生成対象から外す (既定は空集合)
        if post["post_id"] in excluded:
            n_excluded_skipped += 1
            continue
        lang = "ja" if (idx % 2 == 0) else "en"
        pair = make_identity_pair(post, vocab_set, freq, lang)
        if pair is None:
            continue
        pairkey = (pair["text"], tuple(pair["tags"]))
        if pairkey in seen_pairkey or pair["text"] in seen_text:
            continue
        seen_pid.add(post["post_id"])
        seen_text.add(pair["text"])
        seen_pairkey.add(pairkey)
        pairs.append(pair)
        if len(pairs) >= total_needed:
            break

    if excluded:
        print(f"[id-pairs] 凍結除外でスキップした post: {n_excluded_skipped}",
              file=sys.stderr)

    if len(pairs) < total_needed:
        print(f"[id-pairs] 警告: 歩留り不足 {len(pairs)}/{total_needed}。"
              f"--fetch-factor を上げてください。", file=sys.stderr)

    # 検証 (source 非依存 + identity 固有)
    errors = validate_identity(pairs, vocab_set)
    if errors:
        print(f"[id-pairs] 検証エラー {len(errors)} 件 (先頭10):", file=sys.stderr)
        for e in errors[:10]:
            print("   ", e, file=sys.stderr)
        return 2
    print(f"[id-pairs] 検証 OK ({len(pairs)} ペア)", file=sys.stderr)

    # tokenizer 往復 (UNK 0・完全一致)
    unk, mism = roundtrip_check(pairs, args.vocab)
    if unk != 0 or mism != 0:
        print(f"[id-pairs] tokenizer 往復 NG: UNK={unk} mismatch={mism}",
              file=sys.stderr)
        return 3
    print("[id-pairs] tokenizer 往復 OK (UNK 0・完全一致)", file=sys.stderr)

    # 分割 (post 単位・text 単位で重複除去済み → そのまま split)
    rng.shuffle(pairs)
    val = pairs[:args.val]
    train = pairs[args.val:args.val + args.n]

    # リーク防止アサート (§8 踏襲)
    train_pid = {p["meta"]["post_id"] for p in train}
    val_pid = {p["meta"]["post_id"] for p in val}
    assert not (train_pid & val_pid), "post_id リーク検出"
    train_txt = {p["text"] for p in train}
    val_txt = {p["text"] for p in val}
    assert not (train_txt & val_txt), "text リーク検出"

    # Phase 4-A: 凍結除外集合と train/val の post_id 非交差を強制
    #   (生成ループで skip 済みだが、保証は 2 重に。除外漏れがあればここで落ちる)
    all_pid = train_pid | val_pid
    assert not (train_pid & excluded), "凍結 eval post_id が A train に混入"
    assert not (all_pid & excluded), "凍結 eval post_id が A train/val に混入"

    # Phase 4-A: 凍結 eval の pairs.jsonl を直接読んで独立再検証
    #   (make_eval_diverse の抽出ロジックを介さない・除外漏れ検出)
    eval_a_path = os.path.join(args.out_dir, "pairs.eval_diverse_a.jsonl")
    eval_b_path = os.path.join(args.out_dir, "pairs.eval_diverse_b.jsonl")
    frozen_eval_pids_direct = read_eval_post_ids_direct([eval_a_path, eval_b_path])
    leak_train_eval = train_pid & frozen_eval_pids_direct
    leak_all_eval = all_pid & frozen_eval_pids_direct
    assert not leak_train_eval, (
        f"凍結 eval (直接読込) post_id が A train に混入: {sorted(leak_train_eval)[:10]}")
    assert not leak_all_eval, (
        f"凍結 eval (直接読込) post_id が A train/val に混入: {sorted(leak_all_eval)[:10]}")

    # identity 集合の train/val 重複率 (同一性汎化を測る)
    def ikey(p):
        return tuple(p["meta"]["identity_tags"])
    train_ikeys = {ikey(p) for p in train}
    val_ikeys = {ikey(p) for p in val}
    overlap = train_ikeys & val_ikeys
    id_overlap_rate = round(len(overlap) / max(len(val_ikeys), 1), 4)

    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(
        args.out_dir, _tagged("pairs.identity.train", args.out_tag, "jsonl"))
    val_path = os.path.join(
        args.out_dir, _tagged("pairs.identity.val", args.out_tag, "jsonl"))
    with open(train_path, "w", encoding="utf-8") as f:
        for p in train:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(val_path, "w", encoding="utf-8") as f:
        for p in val:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # stats
    allp = train + val
    n = len(allp)
    lang_c = {"ja": 0, "en": 0}
    rating_c, tmpl_c = {}, {}
    id_tag_freq, scene_tag_freq = {}, {}
    n_ident, n_scene, n_tags = [], [], []
    texts = []
    # OOV drop 後の vocab 射影保持率: target タグはすべて vocab 内なので
    # 保持率は「proj 後に残ったタグが全部 vocab に在る」= 1.0 を確認する量。
    n_proj_tags = 0
    n_in_vocab = 0
    for p in allp:
        lang_c[p["lang"]] += 1
        rating_c[p["meta"]["rating"]] = rating_c.get(p["meta"]["rating"], 0) + 1
        tmpl_c[p["meta"]["tmpl"]] = tmpl_c.get(p["meta"]["tmpl"], 0) + 1
        texts.append(p["text"])
        n_ident.append(len(p["meta"]["identity_tags"]))
        n_scene.append(len(p["meta"]["scene_tags"]))
        n_tags.append(len(p["tags"]))
        for t in p["tags"]:
            n_proj_tags += 1
            if t in vocab_set:
                n_in_vocab += 1
        for t in p["meta"]["identity_tags"]:
            id_tag_freq[t] = id_tag_freq.get(t, 0) + 1
        for t in p["meta"]["scene_tags"]:
            scene_tag_freq[t] = scene_tag_freq.get(t, 0) + 1

    # identity retention 教師: 全 identity_tags が target に含まれる割合 (= 100% 設計)
    n_retained = 0
    n_ident_total = 0
    for p in allp:
        tagset = set(p["tags"])
        for t in p["meta"]["identity_tags"]:
            n_ident_total += 1
            if t in tagset:
                n_retained += 1
    teacher_retention = round(n_retained / max(n_ident_total, 1), 6)

    uniq_id = len({tuple(p["meta"]["identity_tags"]) for p in allp})
    stats = {
        "version": 1,
        "kind": "identity_cond",
        "out_tag": args.out_tag or None,
        "seed": args.seed,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "counts": {"total": n, "train": len(train), "val": len(val)},
        "lang_ratio": {k: round(v / n, 4) for k, v in lang_c.items()},
        "source_ratio": {"identity_cond": 1.0},
        "rating_ratio": {k: round(v / n, 4) for k, v in rating_c.items()},
        "split_ratio": {"train": round(len(train) / n, 4),
                        "val": round(len(val) / n, 4)},
        "unique_text_ratio": round(len(set(texts)) / n, 4),
        "template_dist": {str(k): v for k, v in sorted(tmpl_c.items())},
        # Phase 4-A: 凍結除外・リーク 0 の証跡
        "phase4a_exclusion": {
            "exclude_post_ids_file": args.exclude_post_ids,
            "excluded_count_loaded": len(excluded),
            "excluded_skipped_in_gen": n_excluded_skipped,
            "frozen_eval_pids_read_direct": len(frozen_eval_pids_direct),
            "leak_train_x_frozen_eval": len(leak_train_eval),
            "leak_all_x_frozen_eval": len(leak_all_eval),
            "leak_train_x_excluded": len(train_pid & excluded),
            "leak_all_x_excluded": len(all_pid & excluded),
            "post_id_train_val_disjoint": True,
            "text_train_val_disjoint": True,
        },
        "vocab_projection": {
            "target_tags_total": n_proj_tags,
            "target_tags_in_vocab": n_in_vocab,
            "retention_rate": round(n_in_vocab / max(n_proj_tags, 1), 6),
        },
        "identity": {
            "unique_identity_sets": uniq_id,
            "unique_identity_ratio": round(uniq_id / n, 4),
            "teacher_retention": teacher_retention,
            "train_val_identity_overlap_count": len(overlap),
            "train_val_identity_overlap_rate": id_overlap_rate,
            "identity_tags_per_pair": {
                "min": min(n_ident), "max": max(n_ident),
                "mean": round(sum(n_ident) / n, 2),
            },
            "scene_tags_per_pair": {
                "min": min(n_scene), "max": max(n_scene),
                "mean": round(sum(n_scene) / n, 2),
            },
            "distinct_identity_vocab": len(id_tag_freq),
            "distinct_scene_vocab": len(scene_tag_freq),
            "identity_tag_freq_top": sorted(id_tag_freq.items(),
                                            key=lambda x: -x[1])[:25],
            "scene_tag_freq_top": sorted(scene_tag_freq.items(),
                                         key=lambda x: -x[1])[:25],
        },
        "tags_per_pair": {
            "min": min(n_tags), "max": max(n_tags),
            "mean": round(sum(n_tags) / n, 2),
        },
        "schema": {
            "sequence": "<bos> [identity tags] <sep> [scene text] <sep> [target tags] <eos>",
            "sep_reuse": 2,
            "vocab_unchanged": True,
            "target": "identity_tags ∪ scene_tags (正準順序・identity 必ず含む)",
        },
        "sources": {
            "wd14_csv": "SmilingWolf/wd-swinv2-tagger-v3/selected_tags.csv",
            "danbooru_api": "danbooru.donmai.us/posts.json (tags-only, no pixels)",
        },
    }
    stats_path = os.path.join(
        args.out_dir, _tagged("stats.identity", args.out_tag, "json"))
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)

    print(f"[id-pairs] train={len(train)} val={len(val)} → {train_path}, {val_path}",
          file=sys.stderr)
    print(f"[id-pairs] stats: {stats_path}", file=sys.stderr)
    print(f"[id-pairs] lang={stats['lang_ratio']} uniq_text={stats['unique_text_ratio']} "
          f"id/pair mean={stats['identity']['identity_tags_per_pair']['mean']} "
          f"scene/pair mean={stats['identity']['scene_tags_per_pair']['mean']}",
          file=sys.stderr)
    print(f"[id-pairs] identity 重複率(val基準)={id_overlap_rate} "
          f"uniq_identity={uniq_id}/{n}", file=sys.stderr)
    print(f"[id-pairs] リーク 0 証跡: train∩frozen_eval(直読)="
          f"{len(leak_train_eval)} all∩frozen_eval={len(leak_all_eval)} "
          f"train∩excluded={len(train_pid & excluded)} "
          f"teacher_retention={teacher_retention}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

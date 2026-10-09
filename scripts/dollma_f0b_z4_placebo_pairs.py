# -*- coding: utf-8 -*-
"""dollma_f0b_z4_placebo_pairs.py — F-0b 再検証 Z-4 プラセボ勝者ペアの生成。

各 post の候補 (candidates_bestofn.jsonl) から **非 empty 候補を一様に 1 本** 抽選し、
勝者 (sft_bestofn.jsonl) と同 schema・同 post 順・同 lang の SFT ペアを出す。
reward を一切見ない (勝者も抽選対象)。抽選乱数は訓練シャッフルと別系列
random.Random(f"z4sel-{seed}") (str seed は sha512 経由で決定的。hash() は使わない)。
"""
import argparse
import json
import os
import random
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_CAND = os.path.join(ROOT, "data", "rollouts", "candidates_bestofn.jsonl")
DEFAULT_WIN = os.path.join(ROOT, "data", "rollouts", "sft_bestofn.jsonl")
N_EXPECTED = 400


def _read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_placebo(cand_rows, win_rows, seed, n_expected=N_EXPECTED):
    """cand_rows: 候補行 (post x N)。win_rows: 勝者行 (post 順の基準)。返り値 = プラセボ行リスト。"""
    by_post = {}
    for r in cand_rows:
        by_post.setdefault(int(r["post_id"]), []).append(r)
    for w in win_rows:
        assert len(w["tags"]) > 0, f"勝者に empty: post {w['meta']['post_id']}"
    rng = random.Random(f"z4sel-{seed}")
    out = []
    for w in win_rows:
        pid = int(w["meta"]["post_id"])
        cands = sorted(by_post[pid], key=lambda c: int(c["candidate"]))  # 入力順に依存させない
        pool = [c for c in cands if not c.get("empty") and len(c["tags"]) > 0]
        assert pool, f"post {pid} に非 empty 候補なし"
        pick = pool[rng.randrange(len(pool))]
        assert pick["input_text"] == w["text"], f"post {pid}: input_text が勝者 text と不一致"
        meta = dict(w["meta"])
        # reward 由来の勝者メタは持ち越さない (プラセボは reward 非依存)
        for k in ("reward", "winner_candidate", "winner_lm_seed"):
            meta.pop(k, None)
        meta.update({
            "selection": "random_nonempty_uniform",
            "selection_seed": seed,
            "picked_candidate": int(pick["candidate"]),
            "picked_lm_seed": pick.get("lm_seed"),
            "coincides_with_winner": int(pick["candidate"]) == int(w["meta"]["winner_candidate"]),
            "gen": "z4_placebo",
        })
        out.append({"text": w["text"], "tags": list(pick["tags"]), "lang": w["lang"],
                    "source": w["source"], "meta": meta})
    assert len(out) == len(win_rows) == n_expected, f"件数 {len(out)} != {n_expected}"
    for o, w in zip(out, win_rows):
        assert o["meta"]["post_id"] == w["meta"]["post_id"], "post 順不一致"
        assert o["lang"] == w["lang"], "lang 不一致"
        assert o["source"] == w["source"] == "rejection_sft", "source 不一致"
        assert set(o.keys()) == set(w.keys()), "schema(top keys) 不一致"
        assert len(o["tags"]) > 0, "empty タグ"
        if o["meta"]["coincides_with_winner"]:
            assert o["tags"] == w["tags"], f"post {o['meta']['post_id']}: 勝者一致行なのに tags が勝者と不一致"
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", default=DEFAULT_CAND)
    ap.add_argument("--winners", default=DEFAULT_WIN)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if args.seed is None or not args.out:
        ap.error("--seed と --out は必須")
    out = build_placebo(_read_jsonl(args.candidates), _read_jsonl(args.winners), args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    n_same = sum(o["meta"]["coincides_with_winner"] for o in out)
    print(f"[placebo] seed={args.seed} n={len(out)} coincides_with_winner={n_same} -> {args.out}")


def _synth(n_post=40, n_cand=4):
    cands, wins = [], []
    for p in range(n_post):
        wc = p % n_cand
        for c in range(n_cand):
            empty = (p == 3 and c == 0)
            cands.append({"input_text": f"t{p}", "post_id": 1000 + p, "candidate": c, "lm_seed": c,
                          "tags": [] if empty else [f"tag{p}_{c}", "x"], "empty": empty,
                          "reward": -0.1 * c, "is_winner": c == wc})
        wins.append({"text": f"t{p}", "tags": [f"tag{p}_{wc}", "x"], "lang": "ja" if p % 2 else "en",
                     "source": "rejection_sft",
                     "meta": {"post_id": 1000 + p, "reward": -0.1, "winner_candidate": wc,
                              "winner_lm_seed": wc}})
    return cands, wins


def selftest():
    ok = True
    n = 40
    cands, wins = _synth(n)
    a = build_placebo(cands, wins, 7, n_expected=n)
    b = build_placebo(cands, wins, 7, n_expected=n)
    c = build_placebo(cands, wins, 8, n_expected=n)
    if a != b:
        print("[selftest FAIL] seed 再現性")
        ok = False
    if a == c:
        print("[selftest FAIL] seed 違いで同一")
        ok = False
    if any(o["meta"]["post_id"] == 1003 and o["meta"]["picked_candidate"] == 0 for o in a):
        print("[selftest FAIL] empty 候補を抽選")
        ok = False
    if [o["meta"]["post_id"] for o in a] != [w["meta"]["post_id"] for w in wins]:
        print("[selftest FAIL] post 順")
        ok = False
    if [o["lang"] for o in a] != [w["lang"] for w in wins]:
        print("[selftest FAIL] lang")
        ok = False
    if any("reward" in o["meta"] for o in a):
        print("[selftest FAIL] reward メタ持ち越し")
        ok = False
    d = build_placebo(list(reversed(cands)), wins, 7, n_expected=n)
    if d != a:
        print("[selftest FAIL] 候補入力順依存")
        ok = False
    try:
        build_placebo(cands, wins[:n - 1], 7, n_expected=n)
        print("[selftest FAIL] 件数 assert 不発")
        ok = False
    except AssertionError:
        pass
    # 勝者一致行の tags 一致 assert が効くこと (勝者 tags を改竄 -> 一致行で不一致)
    tam = json.loads(json.dumps(wins))
    for w_ in tam:
        w_["tags"] = ["tampered"]
    try:
        build_placebo(cands, tam, 7, n_expected=n)
        print("[selftest FAIL] 勝者一致 tags assert 不発")
        ok = False
    except AssertionError:
        pass
    bad = json.loads(json.dumps(wins))
    bad[5]["tags"] = []
    try:
        build_placebo(cands, bad, 7, n_expected=n)
        print("[selftest FAIL] 勝者 empty assert 不発")
        ok = False
    except AssertionError:
        pass
    if ok:
        print("[selftest] OK (placebo pairs)")
    return ok


if __name__ == "__main__":
    main()

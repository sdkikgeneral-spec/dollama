# -*- coding: utf-8 -*-
"""dollama 施策C diverse-val 構築 単体テスト (docs/testing.md 準拠)。

torch 不要 (dollma_make_eval_diverse は EvalTokenizer で torch 非依存)。
スキーマ / リーク0 / tags-stay-real / gold⊆vocab / B の train・val 非交差を検証する。
LLM 呼び出し・散文生成は一切しない (段b は main Claude の責務)。

実行:
  python scripts/test_dollma_eval_diverse.py
  あるいは: python -m pytest scripts/test_dollma_eval_diverse.py -q
"""
import json
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import dollma_make_eval_diverse as ed

_VOCAB = os.path.join(_HERE, "..", "data", "bitnet", "vocab.json")
_PROMPTS = os.path.join(_HERE, "..", "data", "bitnet", "eval_diverse_prompts.jsonl")
_VAL = os.path.join(_HERE, "..", "data", "bitnet", "pairs.val.jsonl")


def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_eval_tokenizer_matches_train_bitnet():
    """EvalTokenizer の normalize / tag_to_id_lookup が train_bitnet.Tokenizer と
    バイト等価であること (二重実装の同期保証)。torch 不在環境では SKIP。"""
    try:
        import train_bitnet as tb  # torch を引く → py -3.12 のみ
    except Exception as e:
        print(f"[skip] train_bitnet import 不可 ({type(e).__name__}) → 等価性検証は省略")
        return
    et = ed.EvalTokenizer(_VOCAB)
    tt = tb.Tokenizer(_VOCAB)
    # normalize 等価
    for c in ["long_hair", "^_^", "short_hair", "blue_eyes", ":d", "+_+", "1girl"]:
        assert ed.EvalTokenizer.normalize(c) == tb.Tokenizer.normalize(c)
    # tag_to_id 完全一致
    assert et.tag_to_id == tt.tag_to_id
    assert ed.TOK_UNK == tb.TOK_UNK
    # lookup 等価 (既知 + OOV)
    for c in ["long hair", "1girl", "this_is_not_real_xyz", "smile"]:
        assert et.tag_to_id_lookup(c) == tt.tag_to_id_lookup(c)
    print("[ok] EvalTokenizer ≡ train_bitnet.Tokenizer (normalize/lookup/UNK)")


def test_emit_prompts_schema_and_leakage():
    """段a: prompts.jsonl のスキーマ・リーク0・gold⊆vocab・A gold バイト一致。

    既存 prompts を上書きしないよう tmp ディレクトリへ出力する。"""
    import random
    et = ed.EvalTokenizer(_VOCAB)
    vocab_set, freq = ed.load_vocab_set(_VOCAB)
    train_pids = ed.train_post_ids()
    val_pids, val_rows = ed.val_post_ids_and_rows()

    rng = random.Random(20260620)
    pool_a = ed.build_pool_a(val_rows, 20, rng)
    pool_b = ed.build_pool_b(vocab_set, freq, train_pids, val_pids, 20, rng)

    # リーク0: B ∩ (train ∪ val) == ∅
    b_pids = {e["post_id"] for e in pool_b}
    assert not (b_pids & train_pids), "Pool B が train と交差"
    assert not (b_pids & val_pids), "Pool B が val と交差"

    # gold⊆vocab
    for pool, name in ((pool_a, "A"), (pool_b, "B")):
        for e in pool:
            assert e["gold_tags"], f"Pool {name} post {e['post_id']} gold 空"
            for t in e["gold_tags"]:
                assert et.tag_to_id_lookup(t) != ed.TOK_UNK, \
                    f"Pool {name} post {e['post_id']} vocab 外 gold '{t}'"

    # A gold は val 行とバイト一致
    val_by_pid = {r["meta"]["post_id"]: r["tags"] for r in val_rows}
    for e in pool_a:
        assert e["gold_tags"] == val_by_pid[e["post_id"]], \
            f"Pool A post {e['post_id']} gold が val と不一致 (tags-stay-real 違反)"

    # B は正準順序 (バケット番号が非減少) — make_pairs 規約に整合
    for e in pool_b:
        bks = [ed.mp.classify_bucket(t) for t in e["gold_tags"]]
        assert bks == sorted(bks), f"Pool B post {e['post_id']} 順序非正準 {bks}"

    print(f"[ok] 段a スキーマ/リーク0/gold⊆vocab/A バイト一致 "
          f"(A={len(pool_a)} B={len(pool_b)})")


def test_emit_prompts_file_format():
    """既出力 prompts.jsonl があれば各行キー・値域を検証 (無ければ生成して検証)。"""
    import random
    # 一時ディレクトリへ出力して検証 (本物の凍結ファイルは触らない)。
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "prompts.jsonl")
        args = type("A", (), {})()
        args.vocab = _VOCAB
        args.prompts = out
        args.n_a = 15
        args.n_b = 15
        args.n_variants = 3
        args.force = True
        rc = ed.emit_prompts(args, random.Random(20260620))
        assert rc == 0
        rows = _read_jsonl(out)
        assert len(rows) == 30, f"期待 30 行, 実際 {len(rows)}"
        for r in rows:
            assert set(r.keys()) >= {"post_id", "subset", "gold_tags",
                                     "n_variants", "lang_hint"}
            assert r["subset"] in ("A", "B")
            assert r["lang_hint"] in ("ja", "en")
            assert isinstance(r["gold_tags"], list) and r["gold_tags"]
            assert r["n_variants"] == 3
    print("[ok] 段a 出力フォーマット (キー/値域/件数)")


def test_ingest_validation_and_freeze():
    """段c: 合成 texts を取り込み、スキーマ・tags-stay-real・リーク0 を検証して凍結。

    違反入力 (post_id 漏出 / vocab 外 gold は段a 固定なので起こらない / text 空) を
    弾くことも確認する。"""
    import random
    with tempfile.TemporaryDirectory() as td:
        prompts = os.path.join(td, "prompts.jsonl")
        texts = os.path.join(td, "texts.jsonl")
        out_a = os.path.join(td, "a.jsonl")
        out_b = os.path.join(td, "b.jsonl")

        # 段a を tmp に生成
        ea = type("A", (), {})()
        ea.vocab = _VOCAB
        ea.prompts = prompts
        ea.n_a = 5
        ea.n_b = 5
        ea.n_variants = 2
        ea.force = True
        assert ed.emit_prompts(ea, random.Random(20260620)) == 0
        prows = _read_jsonl(prompts)

        # 合成 texts: 各 prompt × variant に無害な散文を置く (タグは書かない)。
        tx = []
        for pr in prows:
            for v in range(pr["n_variants"]):
                tx.append({
                    "post_id": pr["post_id"],
                    "subset": pr["subset"],
                    "variant": v,
                    "lang": pr["lang_hint"],
                    "text": ("かわいいキャラを描いてほしいです。"
                             if pr["lang_hint"] == "ja"
                             else "Please draw a cute character for me."),
                })
        with open(texts, "w", encoding="utf-8") as f:
            for t in tx:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")

        ig = type("A", (), {})()
        ig.vocab = _VOCAB
        ig.prompts = prompts
        ig.texts = texts
        ig.out_a = out_a
        ig.out_b = out_b
        ig.force = True
        assert ed.ingest(ig, random.Random(20260620)) == 0

        arows = _read_jsonl(out_a)
        brows = _read_jsonl(out_b)
        assert arows and brows
        # スキーマ = 既存 val 同形 + source/meta
        for r in arows + brows:
            assert set(r.keys()) == {"text", "tags", "lang", "source", "meta"}
            assert r["source"] == "eval_diverse"
            m = r["meta"]
            assert set(m.keys()) >= {"post_id", "subset", "variant", "gen",
                                     "rating", "n_tags"}
            assert m["gen"] == "claude"
            assert r["text"].strip()
            assert str(m["post_id"]) not in r["text"]  # post_id 漏出なし
            assert r["lang"] in ("ja", "en")
            assert m["n_tags"] == len(r["tags"])

        # tags-stay-real: A の tags は val 行とバイト一致
        val_by_pid = {x["meta"]["post_id"]: x["tags"] for x in _read_jsonl(_VAL)}
        for r in arows:
            assert r["tags"] == val_by_pid[r["meta"]["post_id"]]

        # B の train・val 非交差
        train_pids = ed.train_post_ids()
        val_pids, _ = ed.val_post_ids_and_rows()
        b_pids = {r["meta"]["post_id"] for r in brows}
        assert not (b_pids & train_pids)
        assert not (b_pids & val_pids)

        # 違反検出: post_id を text に混ぜると弾く
        bad = os.path.join(td, "bad.jsonl")
        bad_rows = list(tx)
        bad_rows[0]["text"] = f"post {bad_rows[0]['post_id']} を描いて"
        with open(bad, "w", encoding="utf-8") as f:
            for t in bad_rows:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        ig2 = type("A", (), {})()
        ig2.vocab = _VOCAB
        ig2.prompts = prompts
        ig2.texts = bad
        ig2.out_a = os.path.join(td, "a2.jsonl")
        ig2.out_b = os.path.join(td, "b2.jsonl")
        ig2.force = True
        assert ed.ingest(ig2, random.Random(20260620)) == 2, "post_id 漏出を弾けず"

    print("[ok] 段c 取り込み/検証/凍結 + post_id 漏出弾き")


def _run_all():
    fns = [test_eval_tokenizer_matches_train_bitnet,
           test_emit_prompts_schema_and_leakage,
           test_emit_prompts_file_format,
           test_ingest_validation_and_freeze]
    for fn in fns:
        fn()
    print("\n=== ALL EVAL-DIVERSE TESTS PASSED ===")


if __name__ == "__main__":
    _run_all()

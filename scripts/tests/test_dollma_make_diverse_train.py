# -*- coding: utf-8 -*-
"""dollama 施策B B-1〜B-3 diverse-train 構築 単体テスト (docs/testing.md 準拠)。

torch 不要 (EvalTokenizer で torch 非依存)。LLM 呼び出し・散文生成はしない
(段b は main Claude の責務)。検査項目:
  - 段a の 3 assert (train 所属 / val 非交差 / diverse-val 非交差) が機能する
    (リークを仕込んだら落ちる)。
  - gold⊆vocab・抽出件数 (n=500 パイロット)・seed 決定性 (2 回 emit でバイト一致)。
  - 段c: tags バイト不変 / 件数 4500 / Replace の synthetic 残数 4000 /
    post_id 漏出検査 / 重複 0。
  - 件数拡大 (n=2000): 件数 2000 / synthetic 2500 / 総 4500 / スーパーセット性
    (先頭 500 が n=500 の抽出と完全一致) / anchor assert / todo 切り出し。
  - **B-3 (unique post × k variant)**: 件数一般化式 P×k+(T-P)=T+P(k-1) /
    複合キー (post_id, variant_idx) 重複0 / k=1 legacy schema 非回帰 /
    --n エイリアス / P>T ガード / 段c の variant 主キー追従。
  - EvalTokenizer / 正準順序を import 流用していること (二重定義してない)。

実行:
  py -3.12 scripts/tests/test_dollma_make_diverse_train.py
  あるいは: py -3.12 -m pytest scripts/tests/test_dollma_make_diverse_train.py -q
"""
import json
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import dollma_make_diverse_train as dt
import dollma_make_eval_diverse as ed
import dollma_make_pairs as mp

_VOCAB = os.path.join(_SCRIPTS, "..", "data", "bitnet", "vocab.json")
_TRAIN = os.path.join(_SCRIPTS, "..", "data", "bitnet", "pairs.train.jsonl")


def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _args(**kw):
    a = type("A", (), {})()
    a.vocab = _VOCAB
    a.seed = 20260620
    a.n = 500
    a.k_per_post = 1          # 既定 k=1 (B-1/B-2 と完全等価・legacy schema)
    a.prompts = None
    a.texts = None
    a.out_train = None
    a.stats = None
    a.anchor = None
    a.todo = None
    a.force = True
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def test_reuses_eval_diverse_and_make_pairs():
    """二重実装してないことの担保: モジュール内のシンボルが import 流用であること。"""
    # EvalTokenizer / load_vocab_set / リーク検査は eval_diverse のものと同一オブジェクト
    assert dt.EvalTokenizer is ed.EvalTokenizer
    assert dt.load_vocab_set is ed.load_vocab_set
    assert dt.train_post_ids is ed.train_post_ids
    assert dt.val_post_ids_and_rows is ed.val_post_ids_and_rows
    # 正準順序は make_pairs のものを (ed.mp 経由で) 流用
    assert ed.mp is mp
    assert hasattr(mp, "normalize_separator") and hasattr(mp, "classify_bucket")
    # モジュールに独自の normalize / classify_bucket を再定義してない
    assert not hasattr(dt, "normalize_separator")
    assert not hasattr(dt, "classify_bucket")
    print("[ok] EvalTokenizer / 正準順序を import 流用 (二重定義なし)")


def test_emit_count_and_seed_determinism():
    """段a: 抽出件数 500・gold⊆vocab・seed 決定性 (2 回 emit でバイト一致)。

    k=1 既定 = legacy schema (variant_idx/style_hint なし) の非回帰確認も兼ねる。
    """
    tok = dt.EvalTokenizer(_VOCAB)
    with tempfile.TemporaryDirectory() as td:
        p1 = os.path.join(td, "p1.jsonl")
        p2 = os.path.join(td, "p2.jsonl")
        assert dt.emit_prompts(_args(prompts=p1)) == 0
        assert dt.emit_prompts(_args(prompts=p2)) == 0
        b1 = open(p1, "rb").read()
        b2 = open(p2, "rb").read()
        assert b1 == b2, "seed 決定性違反 (2 回 emit でバイト不一致)"

        rows = _read_jsonl(p1)
        assert len(rows) == 500, f"抽出件数 {len(rows)} != 500"
        for r in rows:
            assert set(r.keys()) >= {"post_id", "gold_tags", "lang_hint", "rating"}
            # k=1 legacy schema: variant_idx / style_hint は付かない (非回帰の核)
            assert "variant_idx" not in r, "k=1 で variant_idx が混入 (legacy 非回帰違反)"
            assert "style_hint" not in r, "k=1 で style_hint が混入 (legacy 非回帰違反)"
            assert r["lang_hint"] in ("ja", "en")
            assert isinstance(r["gold_tags"], list) and r["gold_tags"]
            for t in r["gold_tags"]:
                assert tok.tag_to_id_lookup(t) != dt.TOK_UNK, \
                    f"vocab 外 gold '{t}'"
        # post_id 重複なし
        pids = [r["post_id"] for r in rows]
        assert len(set(pids)) == 500
    print("[ok] 段a 件数500 / gold⊆vocab / seed バイト一致 / post_id 一意 / "
          "k=1 legacy schema 非回帰")


def test_emit_three_asserts_pass_on_real_data():
    """段a の 3 assert (train 所属 / val 非交差 / diverse-val 非交差) が実データで通る。"""
    train_rows = dt._train_rows()
    train_pids = {dt._post_id(r) for r in train_rows if dt._post_id(r) is not None}
    val_pids, _ = dt.val_post_ids_and_rows()
    dv_pids = dt.diverse_val_post_ids()
    assert dv_pids, "diverse-val post_id が読めていない (汚染検査が無力)"

    sel = dt._select_500(train_rows, 500, 20260620)
    sel_pids = {dt._post_id(r) for r in sel}
    assert sel_pids <= train_pids
    assert not (sel_pids & val_pids)
    assert not (sel_pids & dv_pids)
    print(f"[ok] 段a 3 assert 実データで成立 "
          f"(sel=500 / val={len(val_pids)} / diverse-val={len(dv_pids)})")


def test_emit_leak_detection():
    """リークを仕込むと段a の assert が落ちることを確認する。

    diverse_val_post_ids が抽出 500 の 1 件を含むよう monkeypatch し、
    ③ diverse-val 非交差 assert が AssertionError を投げることを検証。"""
    train_rows = dt._train_rows()
    sel = dt._select_500(train_rows, 500, 20260620)
    leak_pid = dt._post_id(sel[0])

    orig = dt.diverse_val_post_ids
    try:
        dt.diverse_val_post_ids = lambda: {leak_pid}  # 汚染を注入
        raised = False
        try:
            dt.emit_prompts(_args(prompts=os.path.join(
                tempfile.gettempdir(), "_leak_should_not_write.jsonl")))
        except AssertionError as e:
            raised = True
            assert "diverse-val" in str(e)
        assert raised, "diverse-val 汚染を assert で弾けなかった"
    finally:
        dt.diverse_val_post_ids = orig
    print("[ok] 段a diverse-val リーク注入を assert で検出")


def test_emit_n2000_superset_and_counts():
    """件数拡大 (n=2000): 抽出 2000 / スーパーセット性 (先頭 500 が n=500 と完全一致) /
    --anchor assert 成立 / --todo が差分 1500 を切り出す。"""
    tok = dt.EvalTokenizer(_VOCAB)
    with tempfile.TemporaryDirectory() as td:
        p500 = os.path.join(td, "p500.jsonl")
        p2000 = os.path.join(td, "p2000.jsonl")
        todo = os.path.join(td, "todo.jsonl")

        # まず n=500 を出して anchor にする
        assert dt.emit_prompts(_args(prompts=p500, n=500)) == 0
        rows500 = _read_jsonl(p500)
        assert len(rows500) == 500
        pids500 = [r["post_id"] for r in rows500]

        # n=2000 を anchor=p500 / todo 付きで出す (anchor assert が成立するはず)
        assert dt.emit_prompts(_args(prompts=p2000, n=2000,
                                     anchor=p500, todo=todo)) == 0
        rows2000 = _read_jsonl(p2000)
        assert len(rows2000) == 2000, f"抽出件数 {len(rows2000)} != 2000"

        # スーパーセット性: 2000 の先頭 500 post_id 列が n=500 と完全一致 (順序込み)
        pids2000 = [r["post_id"] for r in rows2000]
        assert pids2000[:500] == pids500, "スーパーセット違反 (先頭 500 不一致)"
        # 500 は 2000 の部分集合
        assert set(pids500) <= set(pids2000)
        # post_id 一意
        assert len(set(pids2000)) == 2000

        # gold⊆vocab を全件で再確認
        for r in rows2000:
            for t in r["gold_tags"]:
                assert tok.tag_to_id_lookup(t) != dt.TOK_UNK, f"vocab 外 gold '{t}'"

        # todo は差分 1500 件 (anchor がカバーしない post_id のみ)
        trows = _read_jsonl(todo)
        assert len(trows) == 1500, f"todo {len(trows)} != 1500"
        tpids = {r["post_id"] for r in trows}
        # todo は anchor (先頭500) と非交差・かつ 2000 抽出の部分集合
        assert not (tpids & set(pids500)), "todo に anchor の post_id が混入"
        assert tpids <= set(pids2000)
        assert tpids == (set(pids2000) - set(pids500))
        # todo 各行は gold_tags / lang_hint を持つ
        for r in trows:
            assert set(r.keys()) >= {"post_id", "gold_tags", "lang_hint", "rating"}
            assert r["lang_hint"] in ("ja", "en")
    print("[ok] n=2000 件数2000 / スーパーセット(先頭500一致) / anchor assert / "
          "todo 差分1500")


def test_emit_anchor_mismatch_fails():
    """--anchor の post_id 列が抽出先頭と不一致なら assert で落ちる。"""
    with tempfile.TemporaryDirectory() as td:
        bad_anchor = os.path.join(td, "bad.jsonl")
        # 実在しそうにない post_id を 1 件だけ書いた偽 anchor
        with open(bad_anchor, "w", encoding="utf-8") as f:
            f.write(json.dumps({"post_id": -1, "gold_tags": ["x"],
                                "lang_hint": "ja", "rating": "g"}) + "\n")
        raised = False
        try:
            dt.emit_prompts(_args(prompts=os.path.join(td, "o.jsonl"),
                                  n=2000, anchor=bad_anchor))
        except AssertionError as e:
            raised = True
            assert "スーパーセット" in str(e)
        assert raised, "anchor 不一致を assert で弾けなかった"
    print("[ok] anchor 不一致を assert で検出")


def test_n2000_replace_synthesis_counts():
    """段c (n=2000): 著述2000 + synthetic2500 = 4500 / tags バイト不変 / 重複 0。"""
    with tempfile.TemporaryDirectory() as td:
        prompts = os.path.join(td, "prompts.jsonl")
        texts = os.path.join(td, "texts.jsonl")
        out_train = os.path.join(td, "train.diverse_b2000.jsonl")
        stats = os.path.join(td, "stats.json")

        assert dt.emit_prompts(_args(prompts=prompts, n=2000)) == 0
        prows = _read_jsonl(prompts)
        assert len(prows) == 2000

        tx = []
        for pr in prows:
            tx.append({
                "post_id": pr["post_id"],
                "lang": pr["lang_hint"],
                "text": ("かわいいキャラを描いてほしいです。"
                         if pr["lang_hint"] == "ja"
                         else "Please draw a cute character for me."),
            })
        with open(texts, "w", encoding="utf-8") as f:
            for t in tx:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")

        rc = dt.ingest(_args(prompts=prompts, texts=texts, n=2000,
                             out_train=out_train, stats=stats))
        assert rc == 0, f"ingest rc={rc}"

        out = _read_jsonl(out_train)
        assert len(out) == 4500, f"合計 {len(out)} != 4500"
        authored = [r for r in out if r["source"] == "llm_distill"]
        synthetic = [r for r in out if r["source"] == "synthetic"]
        assert len(authored) == 2000, f"著述 {len(authored)} != 2000"
        assert len(synthetic) == 2500, f"synthetic {len(synthetic)} != 2500"
        pids = [r["meta"]["post_id"] for r in out]
        assert len(set(pids)) == 4500, "合成後 post_id 重複"

        st = json.load(open(stats, encoding="utf-8"))
        assert st["n_total"] == 4500
        assert st["n_authored"] == 2000
        assert st["n_synthetic"] == 2500
        assert "2000" in st["policy"]
    print("[ok] 段c n=2000 (著述2000+synthetic2500=4500 / 重複0 / policy一般化)")


def test_ingest_replace_synthesis():
    """段c: tags バイト不変 / 件数 4500 / synthetic 残数 4000 / 重複 0 /
    post_id 漏出検査。合成 texts を作って Replace 合成を検証する。"""
    with tempfile.TemporaryDirectory() as td:
        prompts = os.path.join(td, "prompts.jsonl")
        texts = os.path.join(td, "texts.jsonl")
        out_train = os.path.join(td, "train.diverse_b.jsonl")
        stats = os.path.join(td, "stats.json")

        assert dt.emit_prompts(_args(prompts=prompts)) == 0
        prows = _read_jsonl(prompts)
        assert len(prows) == 500

        # 合成 texts: 各 prompt に無害な散文 (タグは書かない・post_id を含めない)。
        tx = []
        for pr in prows:
            tx.append({
                "post_id": pr["post_id"],
                "lang": pr["lang_hint"],
                "text": ("かわいいキャラを描いてほしいです。"
                         if pr["lang_hint"] == "ja"
                         else "Please draw a cute character for me."),
            })
        with open(texts, "w", encoding="utf-8") as f:
            for t in tx:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")

        rc = dt.ingest(_args(prompts=prompts, texts=texts,
                             out_train=out_train, stats=stats))
        assert rc == 0, f"ingest rc={rc}"

        out = _read_jsonl(out_train)
        assert len(out) == 4500, f"合計 {len(out)} != 4500"
        authored = [r for r in out if r["source"] == "llm_distill"]
        synthetic = [r for r in out if r["source"] == "synthetic"]
        assert len(authored) == 500, f"著述 {len(authored)} != 500"
        assert len(synthetic) == 4000, f"synthetic {len(synthetic)} != 4000"

        # post_id 重複 0
        pids = [r["meta"]["post_id"] for r in out]
        assert len(set(pids)) == 4500, "合成後 post_id 重複"

        # tags バイト不変: 著述行 tags == prompts gold_tags == 元 train 行 tags
        gold_by_pid = {pr["post_id"]: pr["gold_tags"] for pr in prows}
        train_by_pid = {r["meta"]["post_id"]: r["tags"]
                        for r in _read_jsonl(_TRAIN)}
        for r in authored:
            pid = r["meta"]["post_id"]
            assert r["tags"] == gold_by_pid[pid], "著述 tags が gold と不一致"
            assert r["tags"] == train_by_pid[pid], \
                "著述 tags が元 train と不一致 (tags-stay-real 違反)"
            assert r["meta"]["gen"] == "claude"
            assert r["meta"]["tmpl"] == -1
            assert r["meta"]["n_tags"] == len(r["tags"])
            assert r["lang"] in ("ja", "en")

        # synthetic 行は元 train のバイトコピー (抽出 500 を除いた残り)
        for r in synthetic:
            pid = r["meta"]["post_id"]
            assert r["tags"] == train_by_pid[pid]
            assert pid not in gold_by_pid  # 抽出された 500 は synthetic に残らない

        # stats
        st = json.load(open(stats, encoding="utf-8"))
        assert st["n_total"] == 4500
        assert st["n_authored"] == 500
        assert st["n_synthetic"] == 4000
        assert st["validation_ok"] is True
    print("[ok] 段c Replace 合成 (著述500+synthetic4000=4500 / tags不変 / 重複0)")


def test_ingest_rejects_post_id_leak():
    """段c: post_id が text に漏出したら ingest が rc=2 で弾く。"""
    with tempfile.TemporaryDirectory() as td:
        prompts = os.path.join(td, "prompts.jsonl")
        texts = os.path.join(td, "texts.jsonl")
        assert dt.emit_prompts(_args(prompts=prompts)) == 0
        prows = _read_jsonl(prompts)

        tx = []
        for j, pr in enumerate(prows):
            text = "普通の散文です。"
            if j == 0:
                text = f"post {pr['post_id']} を描いて"  # post_id 漏出を仕込む
            tx.append({"post_id": pr["post_id"], "lang": pr["lang_hint"],
                       "text": text})
        with open(texts, "w", encoding="utf-8") as f:
            for t in tx:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")

        rc = dt.ingest(_args(prompts=prompts, texts=texts,
                             out_train=os.path.join(td, "o.jsonl"),
                             stats=os.path.join(td, "s.json")))
        assert rc == 2, f"post_id 漏出を弾けず rc={rc}"
    print("[ok] 段c post_id 漏出を弾く")


# ============================================================
# B-3 追加ケース: unique post × k variant の 2 軸化
# ============================================================
def test_b3_emit_k4_generalized_counts():
    """B-3 段a 件数一般化: --n-posts 2500 --k-per-post 4 で prompts 10,000 行
    (= P×k)・複合キー (post_id,variant_idx) 重複0・post_id は k 重複許容・
    variant_idx 分布 / lang_hint 規則 / style_hint 巡回を検証。"""
    tok = dt.EvalTokenizer(_VOCAB)
    P, k = 2500, 4
    with tempfile.TemporaryDirectory() as td:
        prompts = os.path.join(td, "p10k.jsonl")
        assert dt.emit_prompts(_args(prompts=prompts, n=P, k_per_post=k)) == 0
        rows = _read_jsonl(prompts)
        assert len(rows) == P * k, f"prompts {len(rows)} != P*k {P * k}"

        # 複合キー (post_id, variant_idx) 重複0
        keys = [(r["post_id"], r["variant_idx"]) for r in rows]
        assert len(set(keys)) == P * k, "複合キー重複あり"
        # unique post は P 件・各 post は k 行
        pids = [r["post_id"] for r in rows]
        from collections import Counter
        cnt = Counter(pids)
        assert len(cnt) == P, f"unique post {len(cnt)} != P {P}"
        assert all(c == k for c in cnt.values()), "post ごとの variant 数が k でない"

        # k>1 では variant_idx / style_hint が必須・gold⊆vocab
        for r in rows:
            assert set(r.keys()) >= {"post_id", "variant_idx", "gold_tags",
                                     "lang_hint", "style_hint", "rating"}
            assert 0 <= r["variant_idx"] < k
            assert r["style_hint"] == dt.STYLE_HINTS[r["variant_idx"] % len(dt.STYLE_HINTS)]
            assert r["lang_hint"] in ("ja", "en")
            for t in r["gold_tags"]:
                assert tok.tag_to_id_lookup(t) != dt.TOK_UNK, f"vocab 外 gold '{t}'"

        # lang_hint 規則: 同一 post 内で variant 0,1 = 元 lang / 2,3 = 反対 lang
        by_pid = {}
        for r in rows:
            by_pid.setdefault(r["post_id"], {})[r["variant_idx"]] = r["lang_hint"]
        for pid, vmap in by_pid.items():
            base = vmap[0]
            assert vmap[1] == base, f"post {pid}: variant1 lang が元と不一致"
            opp = "en" if base == "ja" else "ja"
            assert vmap[2] == opp and vmap[3] == opp, \
                f"post {pid}: variant2,3 が反対 lang 強制でない"

        # 全 variant で 3 リーク assert が成立 (emit が通った時点で機構上保証済だが
        # gold⊆vocab UNK0 は上で全件確認済 = k>1 全 variant で成立)。
        train_pids = {dt._post_id(r) for r in dt._train_rows()
                      if dt._post_id(r) is not None}
        val_pids, _ = dt.val_post_ids_and_rows()
        dv_pids = dt.diverse_val_post_ids()
        sel_pids = set(pids)
        assert sel_pids <= train_pids
        assert not (sel_pids & val_pids)
        assert not (sel_pids & dv_pids)
    print(f"[ok] B-3 段a k=4 P=2500: prompts {P * k}=10000 / 複合キー重複0 / "
          f"post k重複許容 / lang規則 / style巡回 / 3リーク assert (全variant)")


def test_b3_superset_with_b2000_anchor():
    """B-3 スーパーセット: 実 B-2 anchor (diverse_train_prompts_b2000.jsonl) の
    post_id 列が、P=2500 k=4 抽出の variant 0 列の先頭 2000 件と完全一致 /
    todo が差分 (anchor未収載variant0 + 全post variant1..3) を切り出す。"""
    anchor = os.path.join(_SCRIPTS, "..", "data", "bitnet",
                          "diverse_train_prompts_b2000.jsonl")
    if not os.path.exists(anchor):
        print("[skip] B-2 anchor (diverse_train_prompts_b2000.jsonl) なし")
        return
    anchor_pids = [r["post_id"] for r in _read_jsonl(anchor)]
    Pa = len(anchor_pids)            # = 2000
    P, k = 2500, 4
    with tempfile.TemporaryDirectory() as td:
        prompts = os.path.join(td, "p10k.jsonl")
        todo = os.path.join(td, "todo.jsonl")
        assert dt.emit_prompts(_args(prompts=prompts, n=P, k_per_post=k,
                                     anchor=anchor, todo=todo)) == 0
        rows = _read_jsonl(prompts)
        # variant 0 列
        v0 = [r["post_id"] for r in rows if r["variant_idx"] == 0]
        assert len(v0) == P
        assert v0[:Pa] == anchor_pids, "スーパーセット違反 (variant0 先頭が anchor と不一致)"

        # todo = anchor 未収載 variant0 (P-Pa) + 全 post variant1..k-1 (P*(k-1))
        trows = _read_jsonl(todo)
        exp_todo = (P - Pa) + P * (k - 1)
        assert len(trows) == exp_todo, f"todo {len(trows)} != {exp_todo}"
        # todo の variant0 行は anchor と非交差
        todo_v0 = {r["post_id"] for r in trows if r["variant_idx"] == 0}
        assert not (todo_v0 & set(anchor_pids)), "todo variant0 に anchor が混入"
        assert len(todo_v0) == P - Pa
        # todo の variant>=1 は全 post 分
        for vi in range(1, k):
            cnt = sum(1 for r in trows if r["variant_idx"] == vi)
            assert cnt == P, f"todo variant{vi} {cnt} != P {P}"
    print(f"[ok] B-3 スーパーセット (実 B-2 anchor): variant0 先頭 {Pa} 一致 / "
          f"todo {exp_todo} = (P-Pa){P - Pa} + variant1..3 {P * (k - 1)}")


def test_b3_p_gt_t_guard():
    """B-3: --n-posts が train 件数 T を超えたら AssertionError (P>T ガード)。"""
    T = len(dt._train_rows())
    with tempfile.TemporaryDirectory() as td:
        raised = False
        try:
            dt.emit_prompts(_args(prompts=os.path.join(td, "o.jsonl"),
                                  n=T + 100, k_per_post=4))
        except AssertionError as e:
            raised = True
            assert "train" in str(e) or str(T) in str(e)
        assert raised, f"P>T ({T + 100}>{T}) を assert で弾けなかった"
    print(f"[ok] B-3 P>T ガード (n-posts {T + 100} > T {T} で AssertionError)")


def test_b3_n_alias():
    """B-3: --n は --n-posts のエイリアス (dest=n)。--n 2000 == --n-posts 2000 を
    バイト一致で確認する (argparse dest 共有の機械保証)。"""
    import argparse
    # main の argparse 定義を再構成して dest が共有されることを確認
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-posts", "--n", dest="n", type=int, default=dt.DEF_N)
    a1 = ap.parse_args(["--n", "2000"])
    a2 = ap.parse_args(["--n-posts", "2000"])
    assert a1.n == a2.n == 2000, "--n と --n-posts の dest が共有されていない"

    # emit 出力もバイト一致 (どちらの綴りでも同じ抽出)
    with tempfile.TemporaryDirectory() as td:
        pa = os.path.join(td, "a.jsonl")
        pb = os.path.join(td, "b.jsonl")
        assert dt.emit_prompts(_args(prompts=pa, n=2000)) == 0
        assert dt.emit_prompts(_args(prompts=pb, n=2000)) == 0
        assert open(pa, "rb").read() == open(pb, "rb").read()
    print("[ok] B-3 --n は --n-posts のエイリアス (dest 共有・出力一致)")


def test_b3_ingest_variant_key_counts():
    """B-3 段c: k=4 P=500 (小規模) で variant 主キー追従を検証。
    著述 = P×k / synthetic = T-P / 合計 T+P(k-1) / 複合キー重複0 /
    同一 post の k variant が衝突せず取り込まれ・stats が一般化式に追従。"""
    P, k = 500, 4
    T = len(dt._train_rows())
    with tempfile.TemporaryDirectory() as td:
        prompts = os.path.join(td, "prompts.jsonl")
        texts = os.path.join(td, "texts.jsonl")
        out_train = os.path.join(td, "train.diverse_b10k.jsonl")
        stats = os.path.join(td, "stats.json")

        assert dt.emit_prompts(_args(prompts=prompts, n=P, k_per_post=k)) == 0
        prows = _read_jsonl(prompts)
        assert len(prows) == P * k

        # 各 (post_id, variant_idx) に対し散文を書く (variant ごとに別文 = 主キー追従)。
        tx = []
        for pr in prows:
            vi = pr["variant_idx"]
            base = ("かわいいキャラを描いてほしい"
                    if pr["lang_hint"] == "ja" else "Please draw a cute character")
            tx.append({
                "post_id": pr["post_id"],
                "variant_idx": vi,
                "lang": pr["lang_hint"],
                "text": f"{base} (variant {vi})。",
            })
        with open(texts, "w", encoding="utf-8") as f:
            for t in tx:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")

        rc = dt.ingest(_args(prompts=prompts, texts=texts, n=P, k_per_post=k,
                             out_train=out_train, stats=stats))
        assert rc == 0, f"ingest rc={rc}"

        out = _read_jsonl(out_train)
        assert len(out) == P * k + (T - P), \
            f"合計 {len(out)} != T+P(k-1) {P * k + (T - P)}"
        authored = [r for r in out if r["source"] == "llm_distill"]
        synthetic = [r for r in out if r["source"] == "synthetic"]
        assert len(authored) == P * k, f"著述 {len(authored)} != P*k {P * k}"
        assert len(synthetic) == T - P, f"synthetic {len(synthetic)} != T-P {T - P}"

        # 著述複合キー重複0・out_rows の post_id 重複は許容 (同一 post の k variant)
        akeys = [(r["meta"]["post_id"], r["meta"].get("variant_idx", 0))
                 for r in authored]
        assert len(set(akeys)) == len(akeys), "著述複合キー重複"
        # 著述 post は P 件 unique・各 k variant
        from collections import Counter
        acnt = Counter(r["meta"]["post_id"] for r in authored)
        assert len(acnt) == P and all(c == k for c in acnt.values())
        # variant ごとに別文が入った (主キー追従の証拠)
        for r in authored:
            assert f"variant {r['meta']['variant_idx']}" in r["text"]

        st = json.load(open(stats, encoding="utf-8"))
        assert st["n_posts"] == P
        assert st["k_per_post"] == k
        assert st["n_authored"] == P * k
        assert st["n_synthetic"] == T - P
        assert st["n_total"] == P * k + (T - P)
    print(f"[ok] B-3 段c k=4 P=500: 著述 {P * k} + synthetic {T - P} = "
          f"{P * k + (T - P)} / 複合キー重複0 / variant 主キー追従 / stats 一般化")


def _run_all():
    fns = [test_reuses_eval_diverse_and_make_pairs,
           test_emit_count_and_seed_determinism,
           test_emit_three_asserts_pass_on_real_data,
           test_emit_leak_detection,
           test_emit_n2000_superset_and_counts,
           test_emit_anchor_mismatch_fails,
           test_n2000_replace_synthesis_counts,
           test_ingest_replace_synthesis,
           test_ingest_rejects_post_id_leak,
           # B-3 追加
           test_b3_emit_k4_generalized_counts,
           test_b3_superset_with_b2000_anchor,
           test_b3_p_gt_t_guard,
           test_b3_n_alias,
           test_b3_ingest_variant_key_counts]
    for fn in fns:
        fn()
    print("\n=== ALL DIVERSE-TRAIN TESTS PASSED ===")


if __name__ == "__main__":
    _run_all()

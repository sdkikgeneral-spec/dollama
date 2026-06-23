# -*- coding: utf-8 -*-
"""dollama C-2 生成ベース set-metrics 単体スモークテスト (docs/testing.md 準拠)。

手置きの gen / gold id 集合で precision / recall / F1 / Jaccard / recall@k の
算術一致を検証する (greedy/encode を重複実装せず、module の _greedy_generate /
encode_text_greedy をモンキーパッチして gen を制御する)。

実行:
  py -3.12 scripts/test_dollma_eval_setmetrics.py
  py -3.12 -m pytest scripts/test_dollma_eval_setmetrics.py -q
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import train_bitnet as tb


class _FakeModel:
    """set-metrics は model.eval() を呼ぶだけ (生成は _greedy_generate をパッチで制御)。"""

    def eval(self):
        return self


class _FakeTok:
    """tag 文字列をそのまま id (int) へ写す最小 tokenizer (set-metrics の I/F のみ実装)。

    本体タグは id>=5。row["tags"] は ["t5", "t6", ...] のような "t<id>" 文字列で渡す。
    "unk" は TOK_UNK にマップ (gold から除外される検証用)。
    """

    def tag_to_id_lookup(self, tag):
        if tag == "unk":
            return tb.TOK_UNK
        # "t5" -> 5
        return int(tag[1:])


def _approx(a, b, eps=1e-9):
    return abs(a - b) <= eps


def test_setmetrics_arithmetic():
    """1 サンプル・既知の gen/gold で各指標の手計算一致を検証。

    gen 本体タグ (順序) = [5, 6, 7, 8]   (重複なし)
    gold タグ          = {6, 7, 8, 9}
    交差 = {6,7,8} -> |交|=3、|gen|=4、|gold|=4、|和|=5。
      precision = 3/4 = 0.75
      recall    = 3/4 = 0.75
      f1        = 0.75
      jaccard   = 3/5 = 0.6
      recall@5  = (gen 先頭5={5,6,7,8} 交 gold=3) / 4 = 0.75
      recall@2  = (gen 先頭2={5,6}    交 gold=1) / 4 = 0.25
    """
    tok = _FakeTok()
    # gen を固定 (prompt は overflow しない短文・encode は中身に依存しないようパッチ)。
    tb_orig_gen = tb._greedy_generate
    tb_orig_enc = tb.encode_text_greedy
    try:
        # _greedy_generate は本体タグ + 末尾に specials を混ぜても除外されることを確認。
        tb._greedy_generate = lambda model, prompt, device, max_len: [5, 6, 7, 8, 3, 6]
        tb.encode_text_greedy = lambda tok, text: [5]  # prompt 長 = 1+1+1 = 3 < max_len
        rows = [{"text": "x", "tags": ["t6", "t7", "t8", "t9"]}]
        rep = tb.eval_generation_setmetrics(
            _FakeModel(), tok, rows, device="cpu", ks=(2, 5), max_len=tb.MAX_SEQ_LEN)
    finally:
        tb._greedy_generate = tb_orig_gen
        tb.encode_text_greedy = tb_orig_enc

    m = rep["macro"]
    assert _approx(m["precision"], 0.75), m["precision"]
    assert _approx(m["recall"], 0.75), m["recall"]
    assert _approx(m["f1"], 0.75), m["f1"]
    assert _approx(m["jaccard"], 0.6), m["jaccard"]
    assert _approx(m["recall@5"], 0.75), m["recall@5"]
    assert _approx(m["recall@2"], 0.25), m["recall@2"]
    # micro は 1 サンプルなので macro と一致 (precision/recall/jaccard)。
    mi = rep["micro"]
    assert _approx(mi["precision"], 0.75)
    assert _approx(mi["recall"], 0.75)
    assert _approx(mi["jaccard"], 0.6)
    assert rep["sum_inter"] == 3 and rep["sum_gen"] == 4 and rep["sum_gold"] == 4
    assert rep["sum_union"] == 5
    assert rep["n_cases"] == 1 and rep["n_empty_gold"] == 0
    print("[ok] set-metrics 単一サンプル算術一致")


def test_setmetrics_gold_unk_excluded_and_dedup():
    """gold の <unk> 除外、gen の specials 除外・重複除去を検証。

    gen = [5, 5, 6, 4(unk), 1(bos)]  -> 本体・重複除去後 [5, 6]
    gold tags = ["t5", "unk", "t6"]  -> {5, 6} (unk 除外)
    交 = {5,6} |交|=2 |gen|=2 |gold|=2 |和|=2 -> P=R=F1=Jaccard=1.0
    """
    tok = _FakeTok()
    tb_orig_gen = tb._greedy_generate
    tb_orig_enc = tb.encode_text_greedy
    try:
        tb._greedy_generate = lambda model, prompt, device, max_len: [5, 5, 6, 4, 1]
        tb.encode_text_greedy = lambda tok, text: [5]
        rows = [{"text": "x", "tags": ["t5", "unk", "t6"]}]
        rep = tb.eval_generation_setmetrics(
            _FakeModel(), tok, rows, device="cpu", ks=(5,), max_len=tb.MAX_SEQ_LEN)
    finally:
        tb._greedy_generate = tb_orig_gen
        tb.encode_text_greedy = tb_orig_enc
    m = rep["macro"]
    assert _approx(m["precision"], 1.0) and _approx(m["recall"], 1.0)
    assert _approx(m["f1"], 1.0) and _approx(m["jaccard"], 1.0)
    assert rep["sum_gen"] == 2 and rep["sum_gold"] == 2 and rep["sum_inter"] == 2
    print("[ok] gold unk 除外 + gen specials/重複除去")


def test_setmetrics_empty_guards():
    """空 gen / 空 gold の 0 除算ガードを検証 (例外を出さず母数から除外)。

    sample0: gen 空 (specials のみ) ・gold={5}  -> precision 母数外・recall=0
    sample1: gen={5}              ・gold 空      -> recall 母数外・precision=0
    macro precision は sample1 のみ (=0)・macro recall は sample0 のみ (=0)。
    """
    tok = _FakeTok()
    tb_orig_gen = tb._greedy_generate
    tb_orig_enc = tb.encode_text_greedy
    calls = {"n": 0}

    def fake_gen(model, prompt, device, max_len):
        calls["n"] += 1
        return [3, 2] if calls["n"] == 1 else [5]  # 1回目=specials のみ, 2回目=[5]

    try:
        tb._greedy_generate = fake_gen
        tb.encode_text_greedy = lambda tok, text: [5]
        rows = [
            {"text": "a", "tags": ["t5"]},   # gen 空
            {"text": "b", "tags": ["unk"]},  # gold 空 (unk のみ)
        ]
        rep = tb.eval_generation_setmetrics(
            _FakeModel(), tok, rows, device="cpu", ks=(5,), max_len=tb.MAX_SEQ_LEN)
    finally:
        tb._greedy_generate = tb_orig_gen
        tb.encode_text_greedy = tb_orig_enc
    m = rep["macro"]
    assert _approx(m["precision"], 0.0), m["precision"]
    assert _approx(m["recall"], 0.0), m["recall"]
    assert rep["n_cases"] == 2 and rep["n_empty_gold"] == 1
    # micro: sum_inter=0, sum_gen=1 (sample1), sum_gold=1 (sample0) -> P=R=0
    assert rep["sum_inter"] == 0 and rep["sum_gen"] == 1 and rep["sum_gold"] == 1
    print("[ok] 空 gen / 空 gold 0 除算ガード")




def test_setmetrics_per_sample_arrays():
    """collect_per_sample=True の per-sample 配列を検証 (施策C seed sweep の paired 用)。

    3 行: skip(prompt overflow) / 通常 / gen 空 を混ぜ、
      - per_sample 各配列長 == len(rows)
      - skip 行は NaN (alignment 保持)
      - 通常行は f1/jaccard が macro 定義と一致
      - gen 空行は precision/f1 が NaN、recall は 0、jaccard は 0
    既定 (collect_per_sample 省略) では per_sample キーが無い (非回帰)。
    """
    import math
    tok = _FakeTok()
    tb_orig_gen = tb._greedy_generate
    tb_orig_enc = tb.encode_text_greedy
    calls = {"n": 0}

    def fake_enc(tok, text):
        # row0 は max_len 超過させる長い text、それ以外は短い。
        if text == "long":
            return list(range(tb.MAX_SEQ_LEN + 5))
        return [5]

    def fake_gen(model, prompt, device, max_len):
        calls["n"] += 1
        # 1 回目の生成呼び出し = row1 (通常)、2 回目 = row2 (gen 空)。
        return [5, 6, 7, 8] if calls["n"] == 1 else [3, 2]

    try:
        tb._greedy_generate = fake_gen
        tb.encode_text_greedy = fake_enc
        rows = [
            {"text": "long", "tags": ["t6"]},               # skip (overflow)
            {"text": "x", "tags": ["t6", "t7", "t8", "t9"]},  # 通常
            {"text": "y", "tags": ["t5"]},                    # gen 空
        ]
        rep = tb.eval_generation_setmetrics(
            _FakeModel(), tok, rows, device="cpu", ks=(5,),
            max_len=tb.MAX_SEQ_LEN, collect_per_sample=True)
    finally:
        tb._greedy_generate = tb_orig_gen
        tb.encode_text_greedy = tb_orig_enc

    ps = rep["per_sample"]
    assert ps["n_rows"] == 3
    for key in ("f1", "jaccard", "precision", "recall"):
        assert len(ps[key]) == 3, (key, len(ps[key]))
    # row0 = skip -> 全 NaN
    assert all(math.isnan(ps[k][0]) for k in ("f1", "jaccard", "precision", "recall"))
    # row1 = 通常: gen={5,6,7,8} gold={6,7,8,9} 交=3 P=R=F1=0.75 J=3/5=0.6
    assert _approx(ps["precision"][1], 0.75) and _approx(ps["recall"][1], 0.75)
    assert _approx(ps["f1"][1], 0.75) and _approx(ps["jaccard"][1], 0.6)
    # row2 = gen 空 (specials のみ): precision/f1 NaN・recall 0・jaccard 0 (union={5})
    assert math.isnan(ps["precision"][2]) and math.isnan(ps["f1"][2])
    assert _approx(ps["recall"][2], 0.0) and _approx(ps["jaccard"][2], 0.0)
    print("[ok] per-sample 配列 (alignment/NaN/値一致)")


def test_setmetrics_per_sample_off_by_default():
    """collect_per_sample 省略時は per_sample キーが無い (既存呼び出し非回帰)。"""
    tok = _FakeTok()
    tb_orig_gen = tb._greedy_generate
    tb_orig_enc = tb.encode_text_greedy
    try:
        tb._greedy_generate = lambda m, p, d, ml: [5, 6]
        tb.encode_text_greedy = lambda tok, text: [5]
        rep = tb.eval_generation_setmetrics(
            _FakeModel(), tok, [{"text": "x", "tags": ["t5"]}],
            device="cpu", ks=(5,), max_len=tb.MAX_SEQ_LEN)
    finally:
        tb._greedy_generate = tb_orig_gen
        tb.encode_text_greedy = tb_orig_enc
    assert "per_sample" not in rep
    print("[ok] per_sample 既定オフ (非回帰)")


if __name__ == "__main__":
    test_setmetrics_arithmetic()
    test_setmetrics_gold_unk_excluded_and_dedup()
    test_setmetrics_empty_guards()
    test_setmetrics_per_sample_arrays()
    test_setmetrics_per_sample_off_by_default()
    print("ALL OK")

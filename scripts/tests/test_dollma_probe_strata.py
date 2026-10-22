# -*- coding: utf-8 -*-
"""dollma_probe_strata.py の純関数テスト (Strata 本体・ネットワーク不要)。

set-F1 配線テストのみ train_bitnet (torch) を要する (無ければ skip)。
実行:
  python -m pytest scripts/tests/test_dollma_probe_strata.py -q
  py -3.12 -m pytest scripts/tests/test_dollma_probe_strata.py -q   (配線テストも走る)
"""
import os
import sys

import pytest

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import dollma_probe_strata as ps  # noqa: E402


# ---------- タグ応答パーサ ----------
def test_parse_tags_json_object():
    t, p = ps.parse_tags('{"tags": ["1girl", "Long_Hair", "1girl", " solo "]}')
    assert (t, p) == (["1girl", "long hair", "solo"], "json")


def test_parse_tags_json_in_fence_and_prose():
    t, p = ps.parse_tags('Here you go:\n```json\n{"tags": ["a", "b"]}\n```\nthanks')
    assert (t, p) == (["a", "b"], "json")
    t, p = ps.parse_tags('Sure! {"tags": ["x y", "z"]} done')
    assert (t, p) == (["x y", "z"], "json")


def test_parse_tags_json_bare_list_and_string_value():
    assert ps.parse_tags('["a", "B"]') == (["a", "b"], "json")
    assert ps.parse_tags('{"tags": "a, b_c; d"}') == (["a", "b c", "d"], "json")


def test_parse_tags_freetext_fallback():
    t, p = ps.parse_tags("tags: 1girl, blue_eyes,\nsolo\n- smile\n2. long hair")
    assert p == "freetext"
    assert t == ["1girl", "blue eyes", "solo", "smile", "long hair"]


def test_parse_tags_broken_json_falls_back():
    t, p = ps.parse_tags('{"tags": ["a", "b"')  # 閉じていない
    assert p == "freetext" and "a" in t[0]


def test_parse_tags_empty():
    assert ps.parse_tags("") == ([], "empty")
    assert ps.parse_tags(None) == ([], "empty")


def test_extract_json_deterministic_and_brace_in_string():
    s = 'noise {"a": "}{", "b": [1, 2]} tail {"c": 1}'
    assert ps.extract_json(s) == {"a": "}{", "b": [1, 2]}
    assert ps.extract_json(s) == ps.extract_json(s)
    assert ps.extract_json("no json here") is None


# ---------- 判定パーサ ----------
def test_judgement_json():
    j = ps.parse_judgement('{"broken": true, "parts": ["hands", "legs"], "defect_score": 8}')
    assert j["broken"] is True and j["parts"] == ["hands", "legs"]
    assert j["score"] == pytest.approx(0.8) and j["path"] == "json"


def test_judgement_json_bool_only_fills_score():
    assert ps.parse_judgement('{"broken": false}')["score"] == 0.0
    assert ps.parse_judgement('{"broken": "yes"}')["score"] == 1.0


def test_judgement_json_score_only_fills_broken():
    j = ps.parse_judgement('{"defect_score": 3}')
    assert j["broken"] is False and j["score"] == pytest.approx(0.3)


def test_judgement_freetext():
    j = ps.parse_judgement("Yes, the hands look broken. score: 7/10")
    assert j["broken"] is True and j["score"] == pytest.approx(0.7) and j["path"] == "freetext"
    j = ps.parse_judgement("No, the image looks clean.")
    assert j["broken"] is False and j["score"] == 0.0
    j = ps.parse_judgement("はい、手が破綻しています")
    assert j["broken"] is True
    j = ps.parse_judgement("defect_score: 85")
    assert j["score"] == pytest.approx(0.85)


def test_judgement_unparseable():
    j = ps.parse_judgement("hmm hard to say")
    assert j["score"] is None and j["path"] == "none"


# ---------- AUC / ラベル ----------
def test_auc_perfect_and_inverse():
    assert ps.auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert ps.auc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0


def test_auc_ties_count_half():
    assert ps.auc([0.5, 0.5], [1, 0]) == 0.5
    # pos {0.7, 0.5}, neg {0.5, 0.2}: 0.7>0.5 1, 0.7>0.2 1, 0.5==0.5 .5, 0.5>0.2 1 -> 3.5/4
    assert ps.auc([0.7, 0.5, 0.5, 0.2], [1, 1, 0, 0]) == pytest.approx(0.875)


def test_auc_single_class_and_none_skipped():
    assert ps.auc([0.1, 0.2], [1, 1]) is None
    assert ps.auc([None, 0.9, 0.1], [1, 1, 0]) == 1.0


def test_parse_label():
    assert [ps.parse_label(x) for x in ("bad", "悪", "1", "good", "良", "0")] == [1, 1, 1, 0, 0, 0]
    assert ps.parse_label("label") is None


# ---------- 決定的抽出 ----------
def _rows(n=40):
    return [{"lang": "ja" if i % 2 == 0 else "en", "meta": {"post_id": i, "variant": 0}}
            for i in range(n)]


def test_select_subset_deterministic_sorted_lang():
    r = _rows()
    a = ps.select_subset(r, 5, 123, "ja")
    assert a == ps.select_subset(r, 5, 123, "ja")
    assert a == sorted(a) and len(a) == 5 and all(r[i]["lang"] == "ja" for i in a)
    assert a != ps.select_subset(r, 5, 124, "ja")
    assert len(ps.select_subset(r, 1000, 1, "all")) == 40


# ---------- set-F1 配線 (train_bitnet の関数を実際に呼ぶ) ----------
def test_score_setf1_wires_to_eval_generation_setmetrics():
    pytest.importorskip("torch")
    import train_bitnet as tb

    class Tok:
        def tag_to_id_lookup(self, tag):
            return {"a": 5, "b": 6, "c": 7}.get(tag, tb.TOK_UNK)

    rows = [{"text": "x", "tags": ["a", "b"]}, {"text": "y", "tags": ["a", "b", "c"]}]
    orig = (tb.encode_text_greedy, tb._greedy_generate)
    rep = ps.score_setf1(rows, [["a", "b"], ["a", "zzz"]], Tok())
    # 置換が元に戻っている
    assert (tb.encode_text_greedy, tb._greedy_generate) == orig
    ps_f1 = rep["per_sample"]["f1"]
    assert ps_f1[0] == pytest.approx(1.0)
    # 2 件目: gen={a,OOV} gold={a,b,c} -> P=.5 R=1/3 F1=0.4 (語彙外は gen に残る)
    assert ps_f1[1] == pytest.approx(0.4)
    assert rep["macro"]["f1"] == pytest.approx(0.7)
    assert rep["n_oov_tags"] == 1 and rep["n_pred_tags"] == 4

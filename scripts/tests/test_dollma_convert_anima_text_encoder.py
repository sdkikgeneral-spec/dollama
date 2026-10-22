# -*- coding: utf-8 -*-
"""dollma_convert_anima_text_encoder.py の純関数テスト (重み不要)。

OpenVINO があれば極小ランダム Qwen3 (2 層・hidden 64) の変換 → CPU(f32) 突合まで走る (無ければ skip)。

実行 (torch 依存のため 3.12):
  py -3.12 -m pytest scripts/tests/test_dollma_convert_anima_text_encoder.py -q
"""
import os
import sys

import numpy as np
import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import dollma_convert_anima_text_encoder as c


def test_config_matches_reference_constants():
    cfg = c.QWEN3_06B_CONFIG
    assert cfg["hidden_size"] == c.HIDDEN == 1024
    assert cfg["num_hidden_layers"] == 28 and cfg["num_attention_heads"] == 16 and cfg["num_key_value_heads"] == 8
    assert cfg["rope_theta"] == 1.0e6 and cfg["vocab_size"] == 151936


def test_io_names_and_pad_constants():
    assert c.INPUT_NAMES == ["input_ids", "attention_mask"]
    assert c.OUTPUT_NAMES == ["last_hidden_state"]
    assert c.QWEN_PAD_ID == 151643 and c.T5_EOS_ID == 1


def test_pad_ids_i64_and_right_padding():
    ids, mask, n = c.pad_ids([10, 20], 6)
    assert ids.dtype == np.int64 and mask.dtype == np.int64     # C++ へ i64 を申し送る根拠
    assert ids.shape == mask.shape == (1, 6) and n == 2
    assert ids[0].tolist() == [10, 20] + [c.QWEN_PAD_ID] * 4
    assert mask[0].tolist() == [1, 1, 0, 0, 0, 0]


@pytest.mark.parametrize("bad", [[], list(range(7))])
def test_pad_ids_rejects(bad):
    with pytest.raises(ValueError):
        c.pad_ids(bad, 6)


def test_suggest_static_len_is_multiple_of_16_and_covers_margin():
    for n in (1, 12, 42, 100, 127):
        L = c.suggest_static_len(n)
        assert L % 16 == 0 and L >= n * 1.25


def test_explicit_mask_semantics():
    am = torch.tensor([[1, 1, 0, 0]])
    em = c.explicit_attention_mask(am)
    allowed = em[0, 0] == 0
    expect = torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0]], dtype=torch.bool)
    assert torch.equal(allowed, expect)
    assert float(em.min()) == c.NEG_MASK and abs(c.NEG_MASK) < 65504


def test_explicit_mask_batch():
    am = torch.tensor([[1, 1, 1], [1, 0, 0]])
    em = c.explicit_attention_mask(am)
    assert em.shape == (2, 1, 3, 3)
    assert bool((em[1, 0, :, 1:] == c.NEG_MASK).all())


def test_compare_hidden_metrics():
    r = np.random.default_rng(0).standard_normal((1, 5, 8))
    m = c.compare_hidden(r, r + 2e-3, n=3)
    assert abs(m["max_abs_err"] - 2e-3) < 1e-9 and m["cos_min"] > 0.999
    assert c.compare_hidden(r, np.full_like(r, np.nan))["nonfinite"]


def test_latency_stats():
    s = c.latency_stats([float(i) for i in range(1, 101)])
    assert s["n"] == 100 and s["median_ms"] == 50.5 and s["p95_ms"] == 95.0 and s["min_ms"] == 1.0


def test_pipeline_id_rules():
    class Fake:
        def __init__(self, ids):
            self.ids = ids

        def __call__(self, texts, add_special_tokens=False, truncation=False):
            return {"input_ids": [list(self.ids)]}
    assert c.pipeline_qwen_ids(Fake([]), "") == [c.QWEN_PAD_ID]
    assert c.pipeline_t5_ids(Fake([5]), "x") == [5, 1]
    assert c.pipeline_t5_ids(Fake([5, 1]), "x") == [5, 1]
    assert c.pipeline_t5_ids(Fake([]), "") == [1]


def test_verify_cases_shapes():
    cases = c.verify_cases(16, tok_ids=[3, 4, 5])
    assert [x[0].split("(")[0] for x in cases] == ["real", "full", "n=1", "half"]
    for _, ids, mask in cases:
        assert ids.shape == mask.shape == (1, 16)


def test_strip_model_prefix():
    assert set(c.strip_model_prefix({"model.x": 1})) == {"x"}
    assert set(c.strip_model_prefix({"model.x": 1, "y": 2})) == {"model.x", "y"}


# ---- 極小 Qwen3 を実際に OV 変換 (openvino + transformers があれば) ----
@pytest.mark.parametrize("mask_mode,attn", [("explicit", "sdpa"), ("explicit", "eager")])
def test_tiny_qwen3_ov_conversion_matches_torch(mask_mode, attn):
    pytest.importorskip("openvino")
    pytest.importorskip("transformers")
    assert c.tiny_conversion_test(mask_mode, attn, seq_len=16, tol=1e-4)


def test_tiny_qwen3_io_signature_is_i64_static():
    pytest.importorskip("openvino")
    pytest.importorskip("transformers")
    te = c.make_qwen3(c.TINY_CFG, "sdpa")
    ov_model, info = c.convert_te(te, 16, None, "explicit")
    ins = {i["name"]: i for i in info["inputs"]}
    assert ins["input_ids"]["element_type"] == "i64" and ins["input_ids"]["shape"] == [1, 16]
    assert ins["attention_mask"]["element_type"] == "i64"
    out = info["outputs"][0]
    assert out["index"] == 0 and out["name"] == "last_hidden_state" and out["shape"] == [1, 16, 64]


def test_padded_ov_output_equals_unpadded_torch_on_valid_tokens():
    # 右 pad + causal → 有効 token の出力は pad 無し torch と一致する (本スクリプトの根幹)
    pytest.importorskip("openvino")
    pytest.importorskip("transformers")
    torch.manual_seed(1)
    te = c.make_qwen3(c.TINY_CFG, "sdpa")
    ov_model, _ = c.convert_te(te, 16, None, "explicit")
    comp = c.compile_cpu_f32(ov_model)
    ids, mask, n = c.pad_ids([11, 22, 33, 44, 55], 16)
    got = c.run_ov(comp, ids, mask)[:, :n]
    ref = c.torch_reference(te, ids, mask)
    assert c.compare_hidden(ref, got)["max_abs_err"] < 1e-4
    # 別 mask (n=9) を同じ compiled に流しても正しい = mask が baked でない
    ids2, mask2, n2 = c.pad_ids(list(range(100, 109)), 16)
    got2 = c.run_ov(comp, ids2, mask2)[:, :n2]
    assert c.compare_hidden(c.torch_reference(te, ids2, mask2), got2)["max_abs_err"] < 1e-4


def test_selftest_passes():
    assert c.selftest() is True


# ---- レビュー反映分 ----
def test_compare_hidden_token0_rest_split():
    r = np.random.default_rng(0).standard_normal((1, 6, 8))
    r[0, 0] *= 1000.0                                  # massive activation 想定
    g = r.copy()
    g[0, 0] += 1.0
    m = c.compare_hidden(r, g)
    assert m["token0_rel_err"] > 0 and m["rest_rel_err_max"] == 0.0
    assert m["token0_max_abs"] > 100 * m["rest_max_abs"]
    assert c.compare_hidden(r[:, :1], g[:, :1])["rest_rel_err_max"] is None


def test_compare_hidden_nonfinite_is_true_not_tautology():
    r = np.ones((1, 2, 4))
    bad = r.copy()
    bad[0, 1, 2] = np.inf
    assert c.compare_hidden(r, bad)["nonfinite"] is True
    assert c.compare_hidden(r, r)["nonfinite"] is False


def test_tokenizer_gate():
    rows = [{"gated": True, "qwen_match": True, "t5_match": True},
            {"gated": False, "qwen_match": True, "t5_match": False},      # extras は情報扱い
            {"gated": False, "qwen_match": None, "t5_match": None}]
    assert c.tokenizer_gate_ok(rows)
    rows.append({"gated": True, "qwen_match": True, "t5_match": False})
    assert not c.tokenizer_gate_ok(rows)


def test_empty_prompt_rule_documented():
    assert "pad" in c.EMPTY_PROMPT_RULE and "eos" in c.EMPTY_PROMPT_RULE

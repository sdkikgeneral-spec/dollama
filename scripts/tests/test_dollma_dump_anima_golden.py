# -*- coding: utf-8 -*-
"""dollma_dump_anima_golden.py の純関数テスト (重み不要・diffusers 不要)。

実行 (torch 依存のため 3.12):
  py -3.12 -m pytest scripts/tests/test_dollma_dump_anima_golden.py -q
"""
import json
import os
import sys

import pytest
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import dollma_dump_anima_golden as g


def test_selftest_all_pass():
    # スクリプト内蔵 selftest (活性値統計・crop・sigmas・Euler・conv3d→2d・bit 比較) が全緑
    assert g.selftest() is True


def test_sigmas_known_values():
    # shift=3, steps=30: stride=33.33 → int(33.33)=33 → sigmas[1] = shift(base[-34]) = shift(0.967)
    s = g.simple_sigmas(3.0, 30, 1000)
    t = 967.0 / 1000.0
    expect = 3.0 * t / (1.0 + 2.0 * t)
    assert abs(float(s[1]) - expect) < 1e-6
    assert s.shape == (31,) and float(s[0]) == 1.0 and float(s[-1]) == 0.0


def test_sigmas_one_step():
    s = g.simple_sigmas(3.0, 1, 1000)
    assert s.tolist() == [1.0, 0.0]


def test_euler_full_trajectory_recovers_x0():
    # 理想の速度場 (pred = noise - x0) なら、何 step でも最終 latent = x0
    x0 = torch.randn(1, 16, 1, 4, 4, generator=torch.Generator().manual_seed(1))
    noise = g.make_noise(0, x0.shape)
    sig = g.simple_sigmas(3.0, 10)
    x = noise * sig[0]
    for i in range(len(sig) - 1):
        pred = noise - x0
        den = g.denoised_from_pred(x, sig[i], pred)
        x = g.euler_step(x, den, sig[i], sig[i + 1])
    assert torch.allclose(x, x0, atol=1e-5)


def test_save_dict_roundtrip(tmp_path):
    from safetensors.torch import load_file, save_file
    t = {"a": torch.randn(1, 3, 4, 4), "ids": torch.tensor([[1, 2, 3]]), "big": torch.randn(1, 2, 70, 70) * 1e6}
    sd, meta, skipped, cropped = g.build_save_dict(t, never_crop=set(), budget=1000, crop=8)
    p = str(tmp_path / "x.safetensors")
    save_file(sd, p, metadata={k: str(v) for k, v in meta.items()})
    back = load_file(p)
    assert set(back) == set(sd)
    assert "ids_i64" in back and back["ids_i64"].dtype == torch.int64
    # a は 192B <= budget=1000 → full / big は 39200B > 1000 → crop + f16 省略
    assert "a_f32" in back
    assert "big_crop_f32" in back and "big_crop_f16" not in back and skipped == ["big_crop"]
    assert cropped == [("big", [1, 2, 70, 70], [1, 2, 8, 8])]


def test_save_dict_f16_matches_cast():
    x = torch.randn(1, 8) * 100
    sd, _, _, _ = g.build_save_dict({"x": x})
    assert torch.equal(sd["x_f16"], sd["x_f32"].to(torch.float16))


def test_act_stats_zero_tensor():
    s = g.act_stats(torch.zeros(1 << 20))
    assert s["max_abs"] == 0.0 and s["fp16_headroom"] == float("inf")
    assert json.loads(json.dumps(s))["max_abs"] == 0.0   # JSON サイドカーに書ける


# ---- Recorder (hook 配管) ----
class _Tuple(torch.nn.Module):
    def forward(self, x):
        return x * 2, x * 3


def test_recorder_stats_gated_by_phase_and_capture_gated_by_flag():
    m = _Tuple()
    rec = g.Recorder()
    rec.hook_stats(m, "t")
    rec.hook_capture(m, ["o0", "o1"])
    m(torch.ones(2))                       # phase None / capture False → 何も記録しない
    assert not rec.stats and not rec.tensors
    rec.phase = "p"
    rec.capture = True
    m(torch.ones(2))
    assert "p/t" in rec.stats and rec.stats["p/t"]["max_abs"] == 2.0   # tuple の先頭を統計
    assert torch.equal(rec.tensors["o1"], torch.full((2,), 3.0))
    rec.remove()
    rec.stats.clear()
    m(torch.ones(2))
    assert not rec.stats


def test_recorder_pre_arg_capture():
    m = torch.nn.Linear(3, 3)
    rec = g.Recorder()
    rec.hook_capture(m, "inp", pre_arg=0)
    rec.capture = True
    x = torch.randn(1, 3)
    m(x)
    assert torch.equal(rec.tensors["inp"], x)


# ---- VAE: causal conv3d → T=1 conv2d 置換 (本物 QwenImageCausalConv3d と同名・同ロジックの写しで検証) ----
class QwenImageCausalConv3d(torch.nn.Conv3d):
    def __init__(self, ci, co, k, stride=1, padding=0):
        super().__init__(ci, co, k, stride=stride, padding=padding)
        self._padding = (self.padding[2], self.padding[2], self.padding[1], self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        x = F.pad(x, list(self._padding))
        return super().forward(x)


class _TinyDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = QwenImageCausalConv3d(4, 6, 3, padding=1)
        self.b = QwenImageCausalConv3d(6, 6, (3, 1, 1), padding=(1, 0, 0))
        self.c = QwenImageCausalConv3d(6, 3, 1)

    def forward(self, x):
        return self.c(F.silu(self.b(F.silu(self.a(x)))))


def test_vae_patch_equivalence_and_restore():
    torch.manual_seed(0)
    net = _TinyDecoder().eval()
    x = torch.randn(1, 4, 1, 12, 10)
    ref = net(x)
    with g.vae_t1_conv2d_patch(net):
        assert "forward" in net.a.__dict__
        got = net(x)
    assert ref.shape == got.shape
    assert float((ref - got).abs().max()) < 1e-5
    # 復元: インスタンス属性が消えている (= クラスの forward に戻る)
    assert all("forward" not in m.__dict__ for m in (net.a, net.b, net.c))
    assert torch.equal(net(x), ref)


def test_vae_patch_stride():
    c = QwenImageCausalConv3d(3, 5, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)).eval()
    x = torch.randn(1, 3, 1, 9, 9)
    ref = c(x)
    got = g.causal_conv3d_t1_as_conv2d(x, c.weight, c.bias, (1, 1, 1), stride_hw=(2, 2))
    assert ref.shape == got.shape and float((ref - got).abs().max()) < 1e-5


@pytest.mark.parametrize("T", [2, 3])
def test_t_gt1_rejected(T):
    with pytest.raises(ValueError):
        g.causal_conv3d_t1_as_conv2d(torch.randn(1, 2, T, 4, 4), torch.randn(2, 2, 3, 3, 3), None, (1, 1, 1))


def test_compare_tensor_dicts_detects_single_bit():
    a = {"x": torch.arange(8, dtype=torch.float32)}
    b = {"x": a["x"].clone()}
    assert g.compare_tensor_dicts(a, b)[0]
    b["x"][3] = torch.nextafter(b["x"][3], torch.tensor(100.0))
    eq, diff, _ = g.compare_tensor_dicts(a, b)
    assert not eq and diff == ["x"]


def test_main_requires_out_dir(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x"])
    with pytest.raises(SystemExit):
        g.main()


# ---- レビュー反映分 ----
def test_save_dict_croppable_restricts_crop():
    big = torch.randn(1, 4, 64, 64)
    sd, _, _, cr = g.build_save_dict({"a": big}, never_crop=set(), budget=100, crop=8, croppable=frozenset())
    assert "a_f32" in sd and not cr                  # croppable 空 → 巨大でも full
    sd, _, _, cr = g.build_save_dict({"a": big}, never_crop=set(), budget=100, crop=8, croppable={"a"})
    assert "a_crop_f32" in sd and len(cr) == 1


def test_group_of_and_grouped_summary():
    assert g.group_of("te_pos/te/layer00") == "te"
    assert g.group_of("step0_cond/dit/block00_out") == "dit"
    assert g.group_of("allsteps/dit/block03_ff") == "dit"
    assert g.group_of("decode/vae/mid_block") == "vae"
    acts = {"a/te/x": {"max_abs": 1.0, "fp16_headroom": 6e4, "fp16_overflow": False, "nonfinite": False},
            "a/dit/y": {"max_abs": 7e4, "fp16_headroom": 0.9, "fp16_overflow": True, "nonfinite": False}}
    gs = g.fp16_summary_grouped(acts)
    assert set(gs) == {"te", "dit"} and gs["dit"]["fp16_overflow_or_nonfinite"] == ["a/dit/y"]
    assert gs["te"]["fp16_overflow_or_nonfinite"] == []


def test_running_to_stats():
    st = g.running_to_stats({"dit/b": 100.0, "dit/c": 7e4})
    assert st["allsteps/dit/c"]["fp16_overflow"] and not st["allsteps/dit/b"]["fp16_overflow"]


def test_recorder_running_max_across_calls():
    m = torch.nn.Identity()
    rec = g.Recorder()
    rec.hook_stats(m, "id")
    rec.running_on = True
    m(torch.tensor([1.0, -3.0]))
    m(torch.tensor([2.0]))
    assert float(rec.running["id"]) == 3.0
    assert not rec.stats                              # phase None → 表には出さない


def test_recorder_stats_in_and_index():
    lin = torch.nn.Linear(2, 2)
    rec = g.Recorder()
    rec.hook_stats_in(lin, "lin_in")
    rec.phase = "p"
    lin(torch.tensor([[5.0, -1.0]]))
    assert rec.stats["p/lin_in"]["max_abs"] == 5.0
    tup = _Tuple()
    rec.hook_stats(tup, "gate", index=1)
    tup(torch.ones(1))
    assert rec.stats["p/gate"]["max_abs"] == 3.0


def test_evaluate_gates():
    ok_vae = {"max_abs_diff": 1e-6, "time_conv_total_calls_in_t1_decode": 0}
    gates = g.evaluate_gates(0.0, ok_vae, True, 0.0)
    assert all(v["pass"] for v in gates.values())
    assert not g.evaluate_gates(1e-3, ok_vae, True, 0.0)["sigma_selfcheck"]["pass"]
    assert not g.evaluate_gates(0.0, {"max_abs_diff": 1e-2, "time_conv_total_calls_in_t1_decode": 0}, True, 0.0)[
        "vae_conv2d_max_diff"]["pass"]
    assert not g.evaluate_gates(0.0, {"max_abs_diff": 0.0, "time_conv_total_calls_in_t1_decode": 2}, True, 0.0)[
        "vae_time_conv_calls"]["pass"]
    assert not g.evaluate_gates(0.0, ok_vae, False, 0.0)["b0_ctx_equals_pos_adapter_out"]["pass"]
    assert "vae_conv2d_max_diff" not in g.evaluate_gates(None, ok_vae, True, None, vae_checked=False)


def test_set_fp32_ieee_returns_mode():
    assert g.set_fp32_ieee() in ("fp32_precision=ieee", "allow_tf32=False")


def test_determinism_on_cpu_is_rejected(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x", "--out-dir", "o", "--device", "cpu", "--determinism"])
    with pytest.raises(SystemExit) as e:
        g.main()
    assert e.value.code == 2                           # argparse error

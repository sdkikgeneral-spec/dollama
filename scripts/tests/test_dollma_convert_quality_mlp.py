# -*- coding: utf-8 -*-
"""dollama Phase 4 Model B / Package D-1 — QualityMLP OV 変換の単体テスト。

torch は必須。OpenVINO があれば in-memory 変換 + CPU 突合まで、無ければ
torch のみの配管検証 (Wrap 出力形状 [1,1]・param・決定性) にフォールバックする
(開発機に OV が無くても緑・研究機では OV 経路まで緑)。

実行:
  py -3.14 scripts/tests/test_dollma_convert_quality_mlp.py
"""
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import dollma_convert_quality_mlp as cv
from dollma_train_quality_mlp import QualityMLP, EMBED_DIM


def _wrap():
    m = QualityMLP(hidden=cv.ARCH_HIDDEN, dropout=0.3).eval()

    class Wrap(torch.nn.Module):
        def __init__(self, mm):
            super().__init__()
            self.m = mm

        def forward(self, x):
            return self.m.net(x)  # [1,1] logit

    return m, Wrap(m).eval()


def test_wrap_output_shape():
    # OV 変換で [1,1] を維持するラッパ (QualityMLP.forward は [B] に squeeze する)。
    _, w = _wrap()
    y = w(torch.zeros(1, EMBED_DIM))
    assert y.shape == (1, 1), y.shape


def test_param_count_expected():
    m, _ = _wrap()
    n = sum(p.numel() for p in m.parameters())
    assert n == cv.PARAM_COUNT_EXPECTED == 49281


def test_static_shape_constants():
    assert cv.STATIC_SHAPE == [1, EMBED_DIM] == [1, 768]


def test_determinism():
    _, w = _wrap()
    x = torch.rand(1, EMBED_DIM)
    x = x / x.norm(dim=-1, keepdim=True)
    with torch.no_grad():
        assert torch.equal(w(x), w(x))


def test_ov_roundtrip_if_available():
    # OV があれば in-memory 変換 → CPU 出力が PyTorch と一致 (gate <= 1e-4)。
    try:
        import openvino as ov
        import numpy as np
    except Exception:
        print("  (openvino 無し → OV 経路スキップ・torch 配管のみ検証)")
        return
    from safetensors.torch import load_file
    weights = os.path.join(_SCRIPTS, "..", "data", "scorer", "quality_mlp_fp32.safetensors")
    if not os.path.isfile(weights):
        print("  (quality_mlp_fp32.safetensors 無し → スキップ)")
        return
    m, w = _wrap()
    m.load_state_dict(load_file(weights), strict=True)
    example = torch.zeros(1, EMBED_DIM)
    ov_model = ov.convert_model(w, example_input=example)
    ov_model.reshape({ov_model.input(0): cv.STATIC_SHAPE})
    assert list(ov_model.output(0).get_partial_shape().to_shape()) == [1, 1]

    x = torch.rand(1, EMBED_DIM)
    x = x / x.norm(dim=-1, keepdim=True)
    x_np = np.ascontiguousarray(x.numpy(), dtype=np.float32)
    with torch.no_grad():
        y_torch = w(x).numpy()
    req = ov.Core().compile_model(ov_model, "CPU").create_infer_request()
    req.infer({0: x_np})
    y_ov = req.get_output_tensor(0).data.copy()
    err = float(abs(y_torch - y_ov).max())
    assert err <= 1e-4, f"OV CPU 突合 {err} > 1e-4"
    print(f"  (OV CPU 突合 max_abs {err:.2e} <= 1e-4)")


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok: {fn.__name__}")
    print(f"all {len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()

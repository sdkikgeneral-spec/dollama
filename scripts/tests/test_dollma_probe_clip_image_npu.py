# -*- coding: utf-8 -*-
"""dollama Phase 4 Model B / Package D-2 — CLIP image encoder OV probe の単体テスト。

304M ViT-L の OV 再変換 (2.8s + 1.2GB save) はテスト毎に回さない。
  - torch のみ: ImageTower ラッパ (encode_image) の配管を軽量に検証。
  - OV があり IR が既存: 生成済 IR の静的形状 [1,3,224,224]→[1,768] を read_model で検証。
開発機 (OV/IR 無し) でも torch フォールバックで緑・研究機では IR 検証まで緑。

実行:
  py -3.14 scripts/tests/test_dollma_probe_clip_image_npu.py
"""
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import dollma_probe_clip_image_npu as pr


class _StubClip(torch.nn.Module):
    """encode_image が [1,768] を返す軽量スタブ (実 CLIP をロードしない)。"""

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Conv2d(3, pr.EMBED_DIM, kernel_size=224)

    def encode_image(self, x):
        return self.proj(x).flatten(1)  # [1,768]


def test_constants():
    assert pr.STATIC_SHAPE == [1, 3, 224, 224]
    assert pr.EMBED_DIM == 768


def test_image_tower_wraps_encode_image():
    tower = pr.ImageTower(_StubClip()).eval()
    y = tower(torch.zeros(1, 3, 224, 224))
    assert y.shape == (1, pr.EMBED_DIM), y.shape


def test_corr_and_l2_helpers():
    import numpy as np
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert abs(pr.corr(a, a) - 1.0) < 1e-9
    assert abs(pr.corr(a, -a) + 1.0) < 1e-9
    # L2: 正規化後ノルム 1。
    v = pr.l2([3.0, 4.0])
    assert abs((v ** 2).sum() ** 0.5 - 1.0) < 1e-9


def test_determinism():
    tower = pr.ImageTower(_StubClip()).eval()
    x = torch.rand(1, 3, 224, 224)
    with torch.no_grad():
        assert torch.equal(tower(x), tower(x))


def test_generated_ir_static_shape_if_present():
    # 生成済 IR があれば静的形状を検証 (再変換はしない)。
    try:
        import openvino as ov
    except Exception:
        print("  (openvino 無し → IR 検証スキップ)")
        return
    xml = os.path.join(_SCRIPTS, "..", "models", "clip-image", "model_ov_fp32.xml")
    if not os.path.isfile(xml):
        print("  (model_ov_fp32.xml 未生成 → スキップ)")
        return
    m = ov.Core().read_model(xml)
    assert list(m.input(0).get_partial_shape().to_shape()) == [1, 3, 224, 224]
    assert list(m.output(0).get_partial_shape().to_shape()) == [1, 768]
    print("  (生成済 IR 静的形状 [1,3,224,224]->[1,768] OK)")


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok: {fn.__name__}")
    print(f"all {len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()

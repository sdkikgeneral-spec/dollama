# -*- coding: utf-8 -*-
"""dollama Phase 4 Model B / Package C — 自作 quality MLP 単体テスト
(torch 依存・実データ非依存・小合成データで回る)。

検査項目:
  1. forward 形状: QualityMLP(768→...→1) は [B] を返す。
  2. safetensors 往復: FP32/FP16 保存→読み戻しで key 一致・NaN/Inf なし。
  3. 決定性: 同 seed で重み bitwise 一致。
  4. 学習 sanity: 線形分離可能な小合成データで val corr が学習前より明確に上がる
     (CLIP 空間蒸留が回る配管の担保。実データ corr は実走で確認)。

実行:
  py -3.14 scripts/tests/test_dollma_train_quality_mlp.py
"""
import os
import sys
import tempfile

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import dollma_train_quality_mlp as qm


def test_forward_shape():
    m = qm.QualityMLP(hidden=(256, 64))
    x = torch.zeros(5, qm.EMBED_DIM)
    y = m(x)
    assert y.shape == (5,), y.shape
    # 1 サンプルでも [1] を返す (squeeze で 0 次元にしない)。
    assert m(torch.zeros(1, qm.EMBED_DIM)).shape == (1,)


def test_param_count_grows_with_capacity():
    small = qm.param_count(qm.QualityMLP(hidden=(256, 64)))
    big = qm.param_count(qm.QualityMLP(hidden=(2048, 512, 256, 128, 32)))
    assert big > small > 0


def test_safetensors_roundtrip():
    m = qm.QualityMLP(hidden=(128, 32))
    with tempfile.TemporaryDirectory() as d:
        for dtype, name in ((torch.float32, "q_fp32.safetensors"),
                            (torch.float16, "q_fp16.safetensors")):
            p = os.path.join(d, name)
            qm.export_safetensors(m, p, dtype)
            problems = qm.sanity_reload(p, m)
            assert problems == [], problems


def test_determinism():
    qm.set_deterministic(123)
    a = qm.QualityMLP(hidden=(64,))
    qm.set_deterministic(123)
    b = qm.QualityMLP(hidden=(64,))
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)


def test_pearson_known_values():
    x = torch.tensor([0.0, 1.0, 2.0, 3.0])
    assert abs(qm.pearson(x, x) - 1.0) < 1e-6           # 完全正相関
    assert abs(qm.pearson(x, -x) + 1.0) < 1e-6          # 完全負相関
    assert qm.pearson(x, torch.ones(4)) == 0.0          # 定数 → denom 0 → 0


def test_learns_on_synthetic():
    # 合成: quality = sigmoid(w·embed) の教師を線形方向 w で作り、MLP が
    # その方向を学べば val corr が上がる (配管が学習する担保)。
    qm.set_deterministic(7)
    g = torch.Generator().manual_seed(7)
    # n >> in_dim(768) にして線形教師を回復可能にする (少標本高次元は実問題側の
    # 課題であり sanity ではない。ここは配管が学習で corr を上げる担保のみ)。
    n_tr, n_va = 4000, 512
    w = torch.randn(qm.EMBED_DIM, generator=g)
    w = w / w.norm()

    def make(n, seed):
        gg = torch.Generator().manual_seed(seed)
        x = torch.randn(n, qm.EMBED_DIM, generator=gg)
        x = x / x.norm(dim=-1, keepdim=True)   # L2 正規化 (CLIP embed 相当)
        y = torch.sigmoid((x @ w) * 5.0)       # 単調な教師
        return x, y

    xtr, ytr = make(n_tr, 1)
    xva, yva = make(n_va, 2)

    m = qm.QualityMLP(hidden=(256, 64), dropout=0.0)
    corr0 = qm.evaluate(m, xva, yva)["val_corr"]  # 学習前
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=1e-5)
    for _ in range(1500):
        m.train()
        pred = torch.sigmoid(m(xtr))
        loss = torch.nn.functional.mse_loss(pred, ytr)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    corr1 = qm.evaluate(m, xva, yva)["val_corr"]
    # 学習で明確に正相関へ (配管が CLIP 空間で quality を学べる)。
    assert corr1 > 0.5, f"学習後 val_corr {corr1} <= 0.5"
    assert corr1 > corr0 + 0.2, f"学習で伸びていない {corr0}->{corr1}"


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok: {fn.__name__}")
    print(f"all {len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()

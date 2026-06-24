# -*- coding: utf-8 -*-
"""dollama Phase 4 Model B / B-3b — ScorerNet 蒸留訓練の乾式テスト (docs/testing.md 準拠)。

検査項目:
  - param_count == probe の 11.18M (n_attr=8・出力 [1,1+8]=9)。
  - 8 軸 head + quality head の出力形状。
  - 軸順 AXIS_ORDER が quality_gate.hpp / make_scorer_labels.py と 1:1。
  - 小さな合成 jsonl で 1〜数 step 訓練が loss が下がる方向に回る。
  - seed 決定性 (同 seed 再実行で重み bitwise 一致)。
  - quality が全 null 時に B (品質) head が凍結される (品質損失 = 0・index0 へ勾配ゼロ)。
  - quality 充足時は B head が学習される (品質損失 > 0)。

torch 必須部は torch 不在環境 (CI) で SKIP できる (train_bitnet テストと同方針)。

実行:
  py -3.12 scripts/test_dollma_train_scorer.py
  py -3.12 -m pytest scripts/test_dollma_train_scorer.py -q
"""

import json
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# torch 必須: 不在環境では全テスト SKIP (train_bitnet テストと同方針)。
try:
    import torch  # noqa: F401
    import train_scorer as ts
    _HAS = True
except Exception as _e:  # ImportError 等
    print(f"[skip] torch/train_scorer import 不可 ({type(_e).__name__}) → 全テスト SKIP")
    _HAS = False


# ------------------------------------------------------------
# 合成 jsonl 生成 (dataset-spec §16.3 スキーマ・画像は実体不要=決定的合成される)
# ------------------------------------------------------------
def _write_jsonl(path, n, with_quality):
    """n 件の合成 scorer サンプルを書く。

    with_quality=False → quality は全 null (B head 凍結検証用)。
    axis は image index から決定的に振る (学習で下がる構造を持たせる)。
    """
    rows = []
    for i in range(n):
        ax = [round(((i * 7 + k * 3) % 10) / 10.0, 3) for k in range(8)]
        q = None if not with_quality else round((i % 10) / 10.0, 3)
        rows.append({
            "image": f"data/scorer/img/{i:06d}.png",
            "quality": q,
            "axis": ax,
            "meta": {"prompt": f"prompt {i}", "rating": "g", "n_tags_in": 10861},
        })
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _args(**kw):
    """train_scorer の argparser 既定を取り、kw で上書きした args を返す。"""
    ns = ts.build_argparser().parse_args([])
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


# ------------------------------------------------------------
# 1. param_count / 出力形状 / 軸順
# ------------------------------------------------------------
def test_param_count_matches_probe():
    if not _HAS:
        return
    model = ts.ScorerNet(n_attr=ts.N_AXIS)
    n = ts.assert_param_count(model)  # 11.18M でなければ AssertionError
    assert round(n / 1e6, 2) == 11.18
    # 出力次元 = 1 + 8 = 9
    assert model.head.out_features == ts.N_OUT == 9
    print(f"[ok] param_count {n} ({n/1e6:.5f}M) == probe 11.18M / out {ts.N_OUT}")


def test_output_shape():
    if not _HAS:
        return
    model = ts.ScorerNet().eval()
    # 小解像度で前向き形状確認 (BatchNorm のため batch>=2)。
    x = torch.zeros(2, 3, 64, 64)
    with torch.no_grad():
        y = model(x)
    assert tuple(y.shape) == (2, 9), y.shape
    # index0=quality / 1..8=axis(8)
    assert y[:, 1:].shape[1] == ts.N_AXIS == 8
    print("[ok] output shape [2,9] = quality + 8 axis")


def test_axis_order_matches_cpp():
    if not _HAS:
        return
    assert ts.AXIS_ORDER == ["Hands", "Limbs", "Head", "Eyes", "Ears",
                             "Mouth", "Digits", "GlobalAnatomy"]
    assert ts.N_AXIS == 8
    print("[ok] AXIS_ORDER == quality_gate.hpp AnomalyAxis 順")


# ------------------------------------------------------------
# 2. loss が下がる方向に回る
# ------------------------------------------------------------
def test_training_loss_decreases():
    if not _HAS:
        return
    with tempfile.TemporaryDirectory() as d:
        tr = os.path.join(d, "scorer.train.jsonl")
        _write_jsonl(tr, 16, with_quality=True)
        args = _args(train_file=tr, val_file="", out_dir="", epochs=4,
                     batch_size=4, img_size=64, lr=2e-3, seed=20260620)
        _model, hist = ts.train(args)
        assert len(hist) == 4
        first = hist[0]["train_loss"]
        last = hist[-1]["train_loss"]
        assert last < first, f"loss が下がっていない: {first} -> {last}"
        print(f"[ok] train_loss {first:.5f} -> {last:.5f} (収束方向)")


# ------------------------------------------------------------
# 3. seed 決定性 (同 seed で重み bitwise 一致)
# ------------------------------------------------------------
def test_seed_determinism_bitwise():
    if not _HAS:
        return
    from safetensors import safe_open
    with tempfile.TemporaryDirectory() as d:
        tr = os.path.join(d, "scorer.train.jsonl")
        _write_jsonl(tr, 16, with_quality=True)

        def run(out):
            os.makedirs(out, exist_ok=True)
            args = _args(train_file=tr, val_file="", out_dir=out, epochs=2,
                         batch_size=4, img_size=64, seed=20260620)
            ts.train(args)
            return os.path.join(out, "scorer_net_fp32.safetensors")

        p1 = run(os.path.join(d, "o1"))
        p2 = run(os.path.join(d, "o2"))
        # 全テンソル bitwise 一致
        with safe_open(p1, framework="pt") as f1, safe_open(p2, framework="pt") as f2:
            assert set(f1.keys()) == set(f2.keys())
            for k in f1.keys():
                t1 = f1.get_tensor(k)
                t2 = f2.get_tensor(k)
                assert torch.equal(t1, t2), f"{k} が seed 再現しない"
        # バイト一致 (safetensors ファイル全体)
        with open(p1, "rb") as a, open(p2, "rb") as b:
            assert a.read() == b.read(), "safetensors バイト不一致"
        print("[ok] seed 決定性: 重み bitwise + ファイルバイト一致")


# ------------------------------------------------------------
# 4. quality 全 null → B head 凍結 (品質損失 0・index0 勾配ゼロ)
# ------------------------------------------------------------
def test_b_head_frozen_when_quality_all_null():
    if not _HAS:
        return
    with tempfile.TemporaryDirectory() as d:
        tr = os.path.join(d, "scorer.train.jsonl")
        _write_jsonl(tr, 16, with_quality=False)  # quality 全 null
        args = _args(train_file=tr, val_file="", out_dir="", epochs=2,
                     batch_size=4, img_size=64, seed=20260620, freeze_b="auto")
        # quality_count == 0 → auto で凍結されるはず
        ds = ts.ScorerDataset(tr, img_size=64, synth_seed=args.seed)
        assert ds.quality_count() == 0
        model, hist = ts.train(args)
        # B head 凍結時は品質損失が全 epoch で 0
        for h in hist:
            assert h["quality_loss"] == 0.0, f"凍結なのに quality_loss={h['quality_loss']}"
        assert getattr(model, "_b_frozen", False) is True
        print("[ok] quality 全 null → B head 凍結 (quality_loss=0)")


def test_b_head_index0_no_grad_when_frozen():
    """凍結時、損失逆伝播で head の index0 (品質行) に勾配が流れないことを直接検証。

    axis loss は head 全体に勾配を流すが、品質項を切れば index0 行の勾配寄与は
    sigmoid(pred[:,0]) を経由しないため 0 になる (head.weight[0], head.bias[0])。
    """
    if not _HAS:
        return
    import torch.nn.functional as F  # noqa: F401
    model = ts.ScorerNet().train()
    x = torch.rand(4, 3, 64, 64)
    axis_tgt = torch.rand(4, 8)
    q_tgt = torch.zeros(4)
    q_mask = torch.zeros(4)  # 全 null
    pred = model(x)
    loss, _, ql = ts.scorer_loss(pred, axis_tgt, q_tgt, q_mask, freeze_b=True)
    assert float(ql) == 0.0
    model.zero_grad(set_to_none=True)
    loss.backward()
    g = model.head.weight.grad
    gb = model.head.bias.grad
    # index0 (品質行) の勾配は 0、index1.. (軸行) は非ゼロのはず
    assert torch.count_nonzero(g[0]) == 0, "凍結なのに品質行 head.weight[0] に勾配"
    assert float(gb[0]) == 0.0, "凍結なのに品質 head.bias[0] に勾配"
    assert torch.count_nonzero(g[1:]) > 0, "軸 head に勾配が流れていない"
    print("[ok] 凍結時 head index0 勾配ゼロ / index1.. 勾配あり")


def test_b_head_learns_when_quality_present():
    """quality 充足 + freeze_b=off → 品質損失が立つ (B head が学習に寄与)。"""
    if not _HAS:
        return
    with tempfile.TemporaryDirectory() as d:
        tr = os.path.join(d, "scorer.train.jsonl")
        _write_jsonl(tr, 16, with_quality=True)
        args = _args(train_file=tr, val_file="", out_dir="", epochs=2,
                     batch_size=4, img_size=64, seed=20260620, freeze_b="off")
        _model, hist = ts.train(args)
        assert any(h["quality_loss"] > 0.0 for h in hist), "品質損失が立たない"
        print("[ok] quality 充足 + freeze off → 品質損失 > 0 (B head 学習)")


# ------------------------------------------------------------
# 5. val ファイルありで val_loss が出る
# ------------------------------------------------------------
def test_val_loss_present():
    if not _HAS:
        return
    with tempfile.TemporaryDirectory() as d:
        tr = os.path.join(d, "scorer.train.jsonl")
        va = os.path.join(d, "scorer.val.jsonl")
        _write_jsonl(tr, 16, with_quality=True)
        _write_jsonl(va, 6, with_quality=True)
        args = _args(train_file=tr, val_file=va, out_dir="", epochs=2,
                     batch_size=4, img_size=64, seed=20260620)
        _model, hist = ts.train(args)
        assert all("val_loss" in h for h in hist)
        print("[ok] val_loss が history に出る")


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
    if _HAS:
        print(f"all {len(fns)} tests passed")
    else:
        print(f"all {len(fns)} tests SKIPPED (torch 不在)")


if __name__ == "__main__":
    _run_all()

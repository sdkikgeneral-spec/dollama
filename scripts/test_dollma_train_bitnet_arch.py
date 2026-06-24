# -*- coding: utf-8 -*-
"""dollama Phase 4 施策 D — train_bitnet.py アーキ・パラメタ化の乾式テスト。

検査項目 (権威スペック・C++ src/models/bitnet.hpp と完全一致):
  - base numel == 32,976,896 (現行 param_count())。
  - d80m numel == 79,908,864 (施策 D 容量増・C++ param_count() 同式同値)。
  - compute_param_count() が実モデルの sum(numel) と base/d80m で厳密一致。
  - d80m 構成で 1step 学習が回り loss が有限 (NaN/Inf なし)。
  - seed 固定で決定性 (同 seed 再構築で重み bitwise 一致・1step 後 loss 一致)。
  - **引数なし default が現行 base アーキと一致** (非回帰): apply_arch(None,...) 後の
    定数が base と一致し、numel も 32,976,896。
  - d80m は base から伸ばすのが n_layers / ffn のみ
    (vocab / d_model / n_heads / max_seq は据え置き)。

torch 必須部は torch 不在環境 (CI) で SKIP できる (train_bitnet テストと同方針)。

実行:
  py -3.12 scripts/test_dollma_train_bitnet_arch.py
  py -3.12 -m pytest scripts/test_dollma_train_bitnet_arch.py -q
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# 権威スペック (このテストが守らせる固定値)。
BASE_NUMEL = 32976896
D80M_NUMEL = 79908864

# torch 必須: 不在環境では全テスト SKIP (train_bitnet テストと同方針)。
try:
    import torch  # noqa: F401
    import train_bitnet as tb
    _HAS = True
except Exception as _e:  # ImportError 等
    print(f"[skip] torch/train_bitnet import 不可 ({type(_e).__name__}) → 全テスト SKIP")
    _HAS = False


def _reset_base():
    """テスト間でモジュール定数を base へ戻す (apply_arch は global を書き換えるため)。"""
    tb.apply_arch(arch="base")


def _build_model_with(arch):
    """arch を適用してモデルを構築し、(numel, model) を返す。"""
    tb.apply_arch(arch=arch)
    model = tb.BitNetDense()
    numel = sum(p.numel() for p in model.parameters())
    return numel, model


def test_compute_param_count_formula():
    """compute_param_count() が base/d80m で権威値と厳密一致 (モデル非構築・純算術)。"""
    if not _HAS:
        return
    _reset_base()
    assert tb.compute_param_count() == BASE_NUMEL, tb.compute_param_count()
    # d80m を直接式で (apply_arch を経ず引数で) 検証。
    pc_d80m = tb.compute_param_count(vocab=4999, d_model=512, n_layers=16, ffn=2464)
    assert pc_d80m == D80M_NUMEL, pc_d80m
    _reset_base()
    print("[ok] compute_param_count 算術一致 (base/d80m)")


def test_default_is_base_non_regression():
    """引数なし default (apply_arch(None,...)) が現行 base アーキと一致 (非回帰)。"""
    if not _HAS:
        return
    # まず d80m を適用して global を汚し、その後 default 適用で base に戻ることを確認。
    tb.apply_arch(arch="d80m")
    info = tb.apply_arch()  # 引数なし = 現状定数を土台 → このとき d80m が残らないことを担保
    # 引数なし apply_arch は「現状定数を土台」にする仕様。直前 d80m なら d80m が残るので、
    # 「新規 import 直後相当の base」を別途確認する (起動順序の非回帰 = main 冒頭で base)。
    # → ここでは d80m が残ることを確認した上で base へ戻す。
    assert info["n_layers"] == 16 and info["ffn"] == 2464, info
    _reset_base()
    # base へ戻したら現行値そのもの。
    binfo = tb.apply_arch()  # base 状態で引数なし → base 維持
    assert binfo["vocab"] == 4999
    assert binfo["d_model"] == 512
    assert binfo["n_layers"] == 8
    assert binfo["n_heads"] == 8
    assert binfo["head_dim"] == 64
    assert binfo["ffn"] == 1792
    assert binfo["max_seq"] == 64
    assert binfo["param_count"] == BASE_NUMEL
    # モジュール定数自体も base 値。
    assert (tb.VOCAB_SIZE, tb.D_MODEL, tb.N_LAYERS, tb.N_HEADS,
            tb.HEAD_DIM, tb.FFN_DIM, tb.MAX_SEQ_LEN) == \
        (4999, 512, 8, 8, 64, 1792, 64)
    print("[ok] 引数なし default = base (非回帰)")


def test_base_numel_matches_cpp():
    """base 構成の実モデル numel == 32,976,896 (C++ param_count 一致)。"""
    if not _HAS:
        return
    _reset_base()
    numel, _ = _build_model_with("base")
    assert numel == BASE_NUMEL, numel
    assert numel == tb.compute_param_count()
    _reset_base()
    print(f"[ok] base numel == {BASE_NUMEL:,}")


def test_d80m_numel_matches_cpp():
    """d80m 構成の実モデル numel == 79,908,864 (C++ param_count 同式同値)。

    伸ばすのは n_layers (8->16) と ffn (1792->2464) のみで、
    vocab / d_model / n_heads / max_seq は据え置きであることも確認。
    """
    if not _HAS:
        return
    _reset_base()
    numel, _ = _build_model_with("d80m")
    assert numel == D80M_NUMEL, numel
    assert numel == tb.compute_param_count()
    # 据え置き軸の確認。
    assert tb.VOCAB_SIZE == 4999
    assert tb.D_MODEL == 512
    assert tb.N_HEADS == 8
    assert tb.HEAD_DIM == 64
    assert tb.MAX_SEQ_LEN == 64
    # 伸ばした軸の確認。
    assert tb.N_LAYERS == 16
    assert tb.FFN_DIM == 2464
    _reset_base()
    print(f"[ok] d80m numel == {D80M_NUMEL:,} (n_layers/ffn のみ拡張)")


def _one_step_loss(arch, seed):
    """arch を適用しモデル構築 → ごく短い合成データで 1step 学習 → loss を返す。

    合成データは vocab/max_seq 内の決定的ランダム token 列。tags 区間 loss を簡単化し、
    全位置 CE (loss_mode="all" 相当) を直接計算して 1step だけ AdamW を回す。
    """
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    tb.apply_arch(arch=arch)
    model = tb.BitNetDense().to("cpu")
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # 合成バッチ [B,S] (S < MAX_SEQ_LEN)。token は本体タグ域 [5, VOCAB) からサンプル。
    B, S = 2, 8
    g = torch.Generator().manual_seed(seed)
    toks = torch.randint(5, tb.VOCAB_SIZE, (B, S + 1), generator=g)
    inp = toks[:, :-1]   # [B,S]
    tgt = toks[:, 1:]    # [B,S]
    import torch.nn.functional as F
    logits = model(inp)  # [B,S,V]
    loss0 = F.cross_entropy(logits.reshape(-1, tb.VOCAB_SIZE), tgt.reshape(-1))
    opt.zero_grad(set_to_none=True)
    loss0.backward()
    opt.step()
    # 1step 後の loss も取る (有限性確認用)。
    with torch.no_grad():
        logits2 = model(inp)
        loss1 = F.cross_entropy(logits2.reshape(-1, tb.VOCAB_SIZE), tgt.reshape(-1))
    return float(loss0.item()), float(loss1.item())


def test_d80m_one_step_smoke():
    """d80m 構成で 1step 学習が回り loss が有限 (NaN/Inf なし)。"""
    if not _HAS:
        return
    import math
    _reset_base()
    loss0, loss1 = _one_step_loss("d80m", seed=20260620)
    assert math.isfinite(loss0), loss0
    assert math.isfinite(loss1), loss1
    # 初期 CE は log(vocab)=~8.52 近傍 (embed std 0.02 初期化のため)。極端でないこと。
    assert 0.0 < loss0 < 50.0, loss0
    _reset_base()
    print(f"[ok] d80m 1step 疎通 loss {loss0:.4f} -> {loss1:.4f} (有限)")


def test_d80m_seed_determinism():
    """同 seed で d80m を 2 回構築+1step すると loss が完全一致 (決定性)。"""
    if not _HAS:
        return
    _reset_base()
    a0, a1 = _one_step_loss("d80m", seed=12345)
    b0, b1 = _one_step_loss("d80m", seed=12345)
    assert a0 == b0, (a0, b0)
    assert a1 == b1, (a1, b1)
    # 念のため重み bitwise 一致も確認 (構築直後 = init 決定性)。
    import random
    random.seed(777)
    torch.manual_seed(777)
    tb.apply_arch(arch="d80m")
    m1 = tb.BitNetDense()
    random.seed(777)
    torch.manual_seed(777)
    tb.apply_arch(arch="d80m")
    m2 = tb.BitNetDense()
    for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        assert n1 == n2
        assert torch.equal(p1, p2), f"{n1} が seed 再現しない"
    _reset_base()
    print(f"[ok] d80m seed 決定性 loss {a0:.6f} 一致・重み bitwise 一致")


def test_individual_overrides():
    """個別上書き --d-model/--n-layers/--ffn が arch を土台に被さる (直交)。"""
    if not _HAS:
        return
    _reset_base()
    # base を土台に n_layers と ffn を d80m 値へ上書き → d80m と同 numel。
    info = tb.apply_arch(arch="base", n_layers=16, ffn=2464)
    assert info["param_count"] == D80M_NUMEL, info
    assert tb.compute_param_count() == D80M_NUMEL
    # 割り切れない d_model はエラー。
    try:
        tb.apply_arch(arch="base", d_model=500)  # 500 % 8 != 0
        raise AssertionError("割り切れない d_model がエラーにならない")
    except ValueError:
        pass
    _reset_base()
    print("[ok] 個別上書き (直交) + d_model 制約チェック")


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

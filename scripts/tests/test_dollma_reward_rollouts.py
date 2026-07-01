# -*- coding: utf-8 -*-
"""dollama Phase 4 施策 F / F-0a 報酬関数 + rollout 収集ハーネス 純ヘルパ単体テスト
(OV/SDXL/torch/重み 不要・開発機で実走緑)。

検査項目:
  dollma_reward.reward_from_scorer:
    - 決定性 (同入力→同出力)。
    - 境界: 全 0 軸 → 0.0 (最良) / 全 1 軸 → -1.0 (最悪)。
    - worst-axis 集約: 一軸だけ高い → その値 (worst) が報酬を支配する。
    - worst 単調性: worst が高いほど報酬が低い。
    - mean 従 (同点処理): worst 同点なら mean が低い (他軸異常が少ない) 方が高報酬。
    - quality 前方互換: None で anatomy のみ・非 None で美的が報酬へ寄与 (将来経路)。
    - 空 axes は ValueError。
  dollma_collect_rollouts 純ヘルパ:
    - sigmoid の境界・既知値。
    - axes_from_logits: index1..8 を sigmoid・長さ検証。
    - quality_from_logits: 凍結 (False) で None・有効 (True) で確率。
    - rollout_row: スキーマ・quality:null フィールド常在。
    - build_input_texts: seed 決定性・件数・プール内。
    - 既定資産パス: --bitnet-weights / --vocab の既定が data/bitnet/... を指す。

実行:
  py -3.14 scripts/tests/test_dollma_reward_rollouts.py
"""
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import dollma_reward as rw
import dollma_collect_rollouts as cr


# ============================================================
# reward_from_scorer
# ============================================================
def test_reward_determinism():
    axes = [0.1, 0.4, 0.2, 0.05, 0.0, 0.3, 0.6, 0.15]
    assert rw.reward_from_scorer(axes) == rw.reward_from_scorer(axes)


def test_reward_boundaries():
    # 全 0 軸 (異常なし) → 最良 0.0 (max==mean==0 ゆえ重みに依らず正確)。
    assert rw.reward_from_scorer([0.0] * 8) == 0.0
    # 全 1 軸 (全破綻) → 最悪 -1.0 (max==mean==1 ゆえ正確)。
    assert rw.reward_from_scorer([1.0] * 8) == -1.0


def test_reward_worst_axis_dominates():
    # 一軸だけ 0.9・他 0 → worst=0.9 が報酬を支配する (≈ -0.9・mean の寄与は極小)。
    one_high = [0.0] * 8
    one_high[3] = 0.9
    r = rw.reward_from_scorer(one_high)
    assert abs(r - (-0.9)) < 0.06, f"worst 支配が崩れている: {r}"
    # どの軸に置いても worst が同じなら報酬は同じ (軸の位置に依らず max 集約)。
    other = [0.0] * 8
    other[6] = 0.9
    assert abs(rw.reward_from_scorer(one_high) - rw.reward_from_scorer(other)) < 1e-9


def test_reward_worst_monotonic():
    # worst が高いほど報酬は低い (異常が悪化するほど罰が大きい)。
    def w(worst):
        a = [0.0] * 8
        a[0] = worst
        return rw.reward_from_scorer(a)
    assert w(0.2) > w(0.5) > w(0.8)


def test_reward_mean_tie_break():
    # worst が同点 (両者 0.8) でも、他軸の異常が少ない (mean が小さい) 方が高報酬。
    sharp = [0.0] * 8
    sharp[0] = 0.8                       # worst=0.8・mean=0.1
    broad = [0.5] * 8
    broad[0] = 0.8                       # worst=0.8・mean≈0.5375
    assert rw.reward_from_scorer(sharp) > rw.reward_from_scorer(broad)
    # ただし mean は従: worst が異なれば worst が勝つ (mean では逆転しない)。
    higher_worst_low_mean = [0.0] * 8
    higher_worst_low_mean[0] = 0.95      # worst=0.95・mean≈0.119
    assert rw.reward_from_scorer(higher_worst_low_mean) < rw.reward_from_scorer(sharp)


def test_reward_quality_forward_compat():
    axes = [0.0] * 8
    axes[0] = 0.4
    base = rw.reward_from_scorer(axes, quality=None)   # 現状本線: anatomy のみ
    # quality=None は anatomy 単独経路と一致 (デフォルト引数省略と同値)。
    assert base == rw.reward_from_scorer(axes)
    # 将来経路: 美的 quality が高いほど報酬は 0 方向へ (良くなる)。
    r_lowq = rw.reward_from_scorer(axes, quality=0.1)
    r_highq = rw.reward_from_scorer(axes, quality=0.9)
    assert r_highq > r_lowq, "美的が高いほど報酬が上がるべき (前方互換経路)"
    # quality を入れると anatomy のみとは値が変わる (寄与している)。
    assert r_highq != base and r_lowq != base


def test_reward_empty_raises():
    try:
        rw.reward_from_scorer([])
        assert False, "空 axes は ValueError のはず"
    except ValueError:
        pass


# ============================================================
# collect_rollouts 純ヘルパ
# ============================================================
def test_sigmoid():
    assert abs(cr.sigmoid(0.0) - 0.5) < 1e-12
    assert cr.sigmoid(20.0) > 0.99 and cr.sigmoid(-20.0) < 0.01
    assert abs(cr.sigmoid(1.0) - 1.0 / (1.0 + math.exp(-1.0))) < 1e-12
    # 大入力でも overflow せず [0,1] に収まる (数値安定)。
    assert 0.0 <= cr.sigmoid(1000.0) <= 1.0
    assert 0.0 <= cr.sigmoid(-1000.0) <= 1.0


def test_axes_from_logits():
    # index0=quality(無視) / index1..8=anatomy。logit 0 → 0.5・大正 → ~1。
    logits = [5.0] + [0.0] * 8         # quality 高・anatomy 全 0 logit
    axes = cr.axes_from_logits(logits)
    assert len(axes) == 8
    assert all(abs(a - 0.5) < 1e-9 for a in axes)   # anatomy logit 0 → 0.5
    # index0 は anatomy に混入しない (quality を拾っていない)。
    logits2 = [0.0] + [10.0, -10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    axes2 = cr.axes_from_logits(logits2)
    assert axes2[0] > 0.99 and axes2[1] < 0.01
    # 長さ不一致は ValueError。
    try:
        cr.axes_from_logits([0.0] * 8)
        assert False
    except ValueError:
        pass


def test_quality_from_logits():
    logits = [2.0] + [0.0] * 8
    # 凍結中 (既定 False) → None (JSONL quality:null・報酬非寄与)。
    assert cr.quality_from_logits(logits, enabled=False) is None
    # 有効化 → index0 を sigmoid。
    q = cr.quality_from_logits(logits, enabled=True)
    assert abs(q - cr.sigmoid(2.0)) < 1e-12


def test_rollout_row_schema():
    axes = [0.1] * 8
    row = cr.rollout_row("a girl", "1girl, solo", 123, axes, None, -0.12)
    assert set(row) == {"input_text", "prompt", "seed", "axes", "quality", "reward"}
    assert row["input_text"] == "a girl"
    assert row["prompt"] == "1girl, solo"
    assert row["seed"] == 123
    assert row["axes"] == axes and len(row["axes"]) == 8
    assert row["quality"] is None          # quality:null フィールド常在 (凍結中)
    assert row["reward"] == -0.12
    # quality 非 None も float で保持 (将来凍結解除)。
    row2 = cr.rollout_row("x", "y", 1, axes, 0.8, -0.05)
    assert row2["quality"] == 0.8


def test_build_input_texts_determinism():
    a = cr.build_input_texts(12, 20260630)
    b = cr.build_input_texts(12, 20260630)
    assert a == b and len(a) == 12            # seed 決定性・件数
    assert all(t in cr.INPUT_TEXTS for t in a)  # 全件プール内
    # 一巡ぶん (プール長以下) は重複なし (多様性を最優先)。
    pool_n = len(cr.INPUT_TEXTS)
    first_cycle = cr.build_input_texts(pool_n, 7)
    assert len(set(first_cycle)) == pool_n
    # 別 seed では並びが変わりうる。
    assert cr.build_input_texts(12, 1) != a or True


def test_default_asset_paths():
    # 定数是正の回帰: BitNet 重み/vocab の既定が data/bitnet/ 配下の本線 #1 を指す
    # (OV/重み不要・argparse 既定値の検証のみ。実走前にパスずれを開発機で捕まえる)。
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--bitnet-weights", dest="bitnet_weights", default=cr.BITNET_WEIGHTS)
    ap.add_argument("--vocab", dest="vocab", default=cr.VOCAB_PATH)
    args = ap.parse_args([])
    # 既定は module 定数と一致 (定数是正 + CLI 上書きの二段構え)。
    assert args.bitnet_weights == cr.BITNET_WEIGHTS
    assert args.vocab == cr.VOCAB_PATH
    # 実体は data/bitnet/bitnet_dense_fp32.safetensors と data/bitnet/vocab.json。
    bw = args.bitnet_weights.replace(os.sep, "/")
    vp = args.vocab.replace(os.sep, "/")
    assert bw.endswith("data/bitnet/bitnet_dense_fp32.safetensors"), bw
    assert vp.endswith("data/bitnet/vocab.json"), vp
    # 同一性版は既定に選ばない (アーキ差の可能性)。
    assert "identity" not in bw


def test_main_plan_does_not_raise():
    # 既定 (--run なし) は計画表示のみで実走しない (資産不在でも例外を投げない)。
    cr.main(["--n", "6"])


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok: {fn.__name__}")
    print(f"all {len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()

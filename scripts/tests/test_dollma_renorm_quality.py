# -*- coding: utf-8 -*-
"""dollama Phase 4 施策 F / Q-2 Step① quality 再正規化 純関数単体テスト
(実データ非依存・stdlib のみ・開発機で実走緑)。

検査項目 (dollma_renorm_quality.renorm_quality = z-score→sigmoid 写像):
  1. 単調性: raw が大きいほど quality が大きい (単調増加)。
  2. 境界: 全出力が [0,1]・raw=mean で quality=0.5。
  3. 分散: 現行 raw_waifu 分布 (std0.81) を通すと再正規化後 std>=0.15
     (旧 [0,0.18] 潰れ std0.0753 から明確に改善)。
  4. 冪等: 同一入力で同一出力 (決定的)。

実行:
  py -3.14 scripts/tests/test_dollma_renorm_quality.py
"""
import math
import os
import random
import statistics
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import dollma_renorm_quality as rq


# ============================================================
# 1. 単調性
# ============================================================
def test_monotonic_increasing():
    mean, std, k = 0.74, 0.81, 1.5
    xs = [-3.0, -1.71, -0.5, 0.0, 0.74, 1.0, 1.78, 3.0]
    ys = [rq.renorm_quality(x, mean, std, k) for x in xs]
    for a, b in zip(ys, ys[1:]):
        assert b > a, f"単調増加でない: {a} -> {b}"


# ============================================================
# 2. 境界 ([0,1] と raw=mean→0.5)
# ============================================================
def test_bounds_and_center():
    mean, std, k = 0.74, 0.81, 1.5
    # raw=mean で厳密に 0.5。
    assert abs(rq.renorm_quality(mean, mean, std, k) - 0.5) < 1e-12
    # 広い raw レンジで全出力が (0,1) ⊂ [0,1]。
    for x in [-100.0, -5.0, -0.095856, 0.74, 1.78, 5.0, 100.0]:
        q = rq.renorm_quality(x, mean, std, k)
        assert 0.0 <= q <= 1.0, f"範囲外: {q}"
    # sigmoid の対称性: mean±d は 0.5 を挟んで対称 (和=1)。
    for d in [0.3, 0.81, 2.0]:
        lo = rq.renorm_quality(mean - d, mean, std, k)
        hi = rq.renorm_quality(mean + d, mean, std, k)
        assert abs((lo + hi) - 1.0) < 1e-9


def test_std_zero_guard():
    # std<=0 は 0 除算回避で z=0 → quality=0.5 (退化入力でも例外なし)。
    assert abs(rq.renorm_quality(1.23, 0.5, 0.0, 1.5) - 0.5) < 1e-12


# ============================================================
# 3. 分散 (現行 raw_waifu 分布を通すと std>=0.15)
# ============================================================
def test_variance_expands():
    # 実測 raw_waifu 分布 (mean0.74 / std0.81 / range[-1.71,1.78]) を正規近似で再現。
    mean, std, k = 0.74, 0.81, 1.5
    rng = random.Random(20260704)
    raws = [max(-1.71, min(1.78, rng.gauss(mean, std))) for _ in range(2000)]
    qs = [rq.renorm_quality(r, mean, std, k) for r in raws]
    std_q = statistics.pstdev(qs)
    # 旧 [0,0.18] 潰れ std0.0753 から明確に改善 (>=0.15)。
    assert std_q >= 0.15, f"再正規化後 std={std_q:.4f} < 0.15"
    # [0,1] 全域を使う (min<0.5<max・両側に振れる)。
    assert min(qs) < 0.5 < max(qs)
    assert max(qs) - min(qs) > 0.5


# ============================================================
# 4. 冪等 (決定的)
# ============================================================
def test_deterministic():
    mean, std, k = 0.74, 0.81, 1.5
    for x in [-1.5, 0.0, 0.74, 1.2, 1.78]:
        assert rq.renorm_quality(x, mean, std, k) == rq.renorm_quality(x, mean, std, k)


# ============================================================
# 補助関数 (histogram)
# ============================================================
def test_histogram_shape():
    h = rq.histogram([0.0, 0.05, 0.5, 0.95, 1.0], nbins=10, lo=0.0, hi=1.0)
    assert len(h) == 10
    assert sum(c for _, _, c in h) == 5     # 総件数保存
    assert h[0][2] == 2 and h[9][2] == 2    # 端 bin (1.0 は最終 bin にクランプ)


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok: {fn.__name__}")
    print(f"all {len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()

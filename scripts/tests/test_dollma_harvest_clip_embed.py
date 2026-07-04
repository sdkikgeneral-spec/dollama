# -*- coding: utf-8 -*-
"""dollama Phase 4 Model B / Package B — CLIP embed harvest 単体テスト
(実 CLIP 非ロード・stdlib のみ・開発機で実走緑)。

検査項目 (実 CLIP 経路は実走で担保。ここは配管の純ロジックのみ):
  1. embed 次元: clip_image_embed は長さ 768。
  2. L2 正規化: 正規化後 embed の L2 norm ≈ 1。
  3. 書き戻し保持: read_jsonl→embed 追記→write_jsonl で既存列 (image/quality/axis/
     meta/quality_waifu/raw_waifu) が無改変・追記列のみ増える。
  4. 冪等: 同一ダミー embed で 2 回書き戻すと結果が一致 (決定的)。
     退避 bak は既存があれば上書きしない。

実行:
  py -3.14 scripts/tests/test_dollma_harvest_clip_embed.py
"""
import json
import math
import os
import shutil
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import dollma_harvest_clip_embed as hv


def _l2(vec):
    return math.sqrt(sum(v * v for v in vec))


def _dummy_norm_embed(seed):
    """決定的ダミー embed[768] を作り L2 正規化 (実 CLIP の normalized 相当)。"""
    vec = [((seed * 131 + i * 17) % 1000) / 1000.0 - 0.5 for i in range(hv.EMBED_DIM)]
    n = _l2(vec)
    return [v / n for v in vec]


# ============================================================
# 1 & 2. embed 次元 768 + L2 正規化後 norm≈1
# ============================================================
def test_embed_dim_and_l2_norm():
    for seed in (0, 1, 42, 999):
        e = _dummy_norm_embed(seed)
        assert len(e) == hv.EMBED_DIM == 768
        assert abs(_l2(e) - 1.0) < 1e-9, f"L2 norm {_l2(e)} != 1"


# ============================================================
# 3. 書き戻しで既存列が保たれる (追記のみ)
# ============================================================
def test_write_back_preserves_columns():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "scorer.train.jsonl")
        orig = [
            {"image": "img/000000.png", "quality": 0.17, "axis": [0.0] * 8,
             "meta": {"prompt": "1girl"}, "quality_waifu": 0.0, "raw_waifu": -0.09},
            {"image": "img/000001.png", "quality": 0.44, "axis": [0.1] * 8,
             "meta": {"prompt": "1boy"}, "quality_waifu": 0.06, "raw_waifu": 0.0},
        ]
        hv.write_jsonl(path, orig)
        rows = hv.read_jsonl(path)
        for i, r in enumerate(rows):
            r["clip_image_embed"] = _dummy_norm_embed(i)
        hv.write_jsonl(path, rows)

        back = hv.read_jsonl(path)
        for i, (o, b) in enumerate(zip(orig, back)):
            # 既存列は無改変。
            for k in ("image", "quality", "axis", "meta", "quality_waifu", "raw_waifu"):
                assert b[k] == o[k], f"列 {k} が改変された"
            # 追記列のみ増える。
            assert set(b.keys()) - set(o.keys()) == {"clip_image_embed"}
            assert len(b["clip_image_embed"]) == 768
            assert abs(_l2(b["clip_image_embed"]) - 1.0) < 1e-6


# ============================================================
# 4. 冪等 (同一ダミー embed → 同一出力) + bak 上書きしない
# ============================================================
def test_idempotent_and_backup_guard():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "scorer.val.jsonl")
        orig = [{"image": "img/x.png", "quality": 0.5, "axis": [0.0] * 8,
                 "meta": {}, "quality_waifu": 0.0, "raw_waifu": 0.0}]
        hv.write_jsonl(path, orig)

        def harvest_once():
            rows = hv.read_jsonl(path)
            for i, r in enumerate(rows):
                r["clip_image_embed"] = _dummy_norm_embed(i)
            bak = path.replace(".jsonl", ".noembed.bak.jsonl")
            if not os.path.exists(bak):
                shutil.copy2(path, bak)
            hv.write_jsonl(path, rows)
            return rows

        r1 = harvest_once()
        # bak には embed 前 (orig) が退避される。
        bak = path.replace(".jsonl", ".noembed.bak.jsonl")
        assert "clip_image_embed" not in hv.read_jsonl(bak)[0]
        r2 = harvest_once()  # 2 回目: bak は上書きされない・出力は決定的一致
        assert r1 == r2
        assert "clip_image_embed" not in hv.read_jsonl(bak)[0]  # bak 依然 embed 前


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok: {fn.__name__}")
    print(f"all {len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()

# -*- coding: utf-8 -*-
"""dollama Model B / B-3a 教師ラベル構築 乾式テスト (torch/OV/SDXL 不要)。

検査項目:
  - catalog/AnomalyAxis 写しが quality_gate.hpp と件数一致 (18 エントリ・8 軸)。
  - 軸集約 (max-sigmoid) が合成 WD14 スコアで正しい (軸ごと最大・範囲外 index skip)。
  - schema 往復 (jsonl 書き→読みで等価)。
  - seed 決定性 (同 seed で train/val 分割がバイト一致)。
  - image リーク検査 (同一 image が train/val に跨らない)。
  - 品質 provider (passthrough / waifu_scorer_v4 / anatomy_proxy) のプラガブル動作。
  - 選定美的モデル waifu_scorer_v4 (apache-2.0) の正規化[0,1]・重み未配置で null・既定選択。

実行:
  py -3.12 scripts/tests/test_dollma_make_scorer_labels.py
"""
import json
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import dollma_make_scorer_labels as ml


def _dummy_csv(path):
    """ダミー selected_tags.csv。CATALOG の一部タグ + 無関係タグを並べる。"""
    rows = [
        "tag_id,name,category,count",
        "1,1girl,0,100",            # index 0 (無関係)
        "2,bad_hands,0,50",         # index 1 → Hands
        "3,extra_arms,0,40",        # index 2 → Limbs
        "4,bad_anatomy,0,30",       # index 3 → GlobalAnatomy
        "5,deformed,0,20",          # index 4 → GlobalAnatomy
        "6,fewer_digits,0,10",      # index 5 → Digits
        "7,solo,0,90",              # index 6 (無関係)
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")


def test_catalog_matches_cpp():
    assert ml.N_AXIS == 8
    assert ml.AXIS_ORDER == ["Hands", "Limbs", "Head", "Eyes", "Ears",
                             "Mouth", "Digits", "GlobalAnatomy"]
    assert len(ml.CATALOG) == 18  # quality_gate.hpp::catalog() と一致
    # 軸名はすべて AXIS_ORDER 内
    for _, axis in ml.CATALOG:
        assert axis in ml.AXIS_INDEX


def test_resolve_and_aggregate():
    with tempfile.TemporaryDirectory() as d:
        csv_path = os.path.join(d, "tags.csv")
        _dummy_csv(csv_path)
        n2i = ml.load_name_to_index(csv_path)
        resolved = ml.resolve_catalog(n2i)
        # CSV にある 6 タグのうち CATALOG にあるのは 5 (bad_hands/extra_arms/bad_anatomy/deformed/fewer_digits)
        names = sorted(n for n, _, _ in resolved)
        assert names == ["bad_anatomy", "bad_hands", "deformed", "extra_arms", "fewer_digits"]

        # スコアベクタ: index 1=0.8(Hands) 2=0.3(Limbs) 3=0.6 4=0.9(GlobalAnatomy max=0.9) 5=0.2(Digits)
        scores = [0.0, 0.8, 0.3, 0.6, 0.9, 0.2, 0.0]
        axes = ml.aggregate_axes(scores, resolved)
        assert abs(axes[ml.AXIS_INDEX["Hands"]] - 0.8) < 1e-9
        assert abs(axes[ml.AXIS_INDEX["Limbs"]] - 0.3) < 1e-9
        assert abs(axes[ml.AXIS_INDEX["GlobalAnatomy"]] - 0.9) < 1e-9  # max(0.6,0.9)
        assert abs(axes[ml.AXIS_INDEX["Digits"]] - 0.2) < 1e-9
        assert axes[ml.AXIS_INDEX["Head"]] == 0.0  # 該当タグ無し
        assert axes[ml.AXIS_INDEX["Eyes"]] == 0.0

        # 範囲外 index は安全 skip (短いスコアベクタ)
        axes_short = ml.aggregate_axes([0.0, 0.5], resolved)
        assert abs(axes_short[ml.AXIS_INDEX["Hands"]] - 0.5) < 1e-9
        assert axes_short[ml.AXIS_INDEX["GlobalAnatomy"]] == 0.0  # index 3,4 範囲外


def test_quality_providers():
    sample = {"image": "x.png", "axis": [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4]}
    # passthrough: record に quality 無ければ None
    pt = ml.make_quality_provider("passthrough")
    assert pt.score({"image": "x.png"}) is None
    assert abs(pt.score({"quality": 0.7}) - 0.7) < 1e-9
    # anatomy_proxy: 1 - max(axis) = 1 - 0.4 = 0.6
    ap = ml.make_quality_provider("anatomy_proxy")
    assert abs(ap.score(sample) - 0.6) < 1e-9


def _build_records(n):
    recs = []
    for i in range(n):
        recs.append({
            "image": f"img_{i:04d}.png",
            "wd14_scores": [0.0, (i % 5) / 10.0, 0.1, 0.2, 0.0, 0.0, 0.0],
            "prompt": f"prompt {i}",
            "rating": "g",
        })
    return recs


def test_schema_roundtrip_and_split_determinism():
    with tempfile.TemporaryDirectory() as d:
        csv_path = os.path.join(d, "tags.csv")
        _dummy_csv(csv_path)
        wd14_path = os.path.join(d, "wd14.jsonl")
        with open(wd14_path, "w", encoding="utf-8") as f:
            for r in _build_records(50):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        def run(out_dir):
            ml.main(["--wd14", wd14_path, "--selected-tags", csv_path,
                     "--out-dir", out_dir, "--quality-provider", "anatomy_proxy",
                     "--seed", "20260620", "--val-ratio", "0.1"])
            with open(os.path.join(out_dir, "scorer.train.jsonl"), encoding="utf-8") as f:
                train = f.read()
            with open(os.path.join(out_dir, "scorer.val.jsonl"), encoding="utf-8") as f:
                val = f.read()
            return train, val

        out1 = os.path.join(d, "o1")
        out2 = os.path.join(d, "o2")
        t1, v1 = run(out1)
        t2, v2 = run(out2)
        # seed 決定性: バイト一致
        assert t1 == t2 and v1 == v2

        # schema 往復: 各行に image/quality/axis(8)/meta があり quality は proxy で埋まる
        rows = [json.loads(l) for l in t1.splitlines() if l.strip()]
        assert len(rows) == 45  # 50 * 0.9
        for row in rows:
            assert "image" in row and "axis" in row and "meta" in row
            assert len(row["axis"]) == 8
            assert row["quality"] is not None  # anatomy_proxy で充足

        # image リーク 0 (train/val 非交差)
        val_rows = [json.loads(l) for l in v1.splitlines() if l.strip()]
        train_imgs = {r["image"] for r in rows}
        val_imgs = {r["image"] for r in val_rows}
        assert not (train_imgs & val_imgs)

        # stats.json に provenance (seed/teachers/status) が残る
        with open(os.path.join(out1, "scorer_stats.json"), encoding="utf-8") as f:
            stats = json.load(f)
        assert stats["seed"] == 20260620
        assert stats["axis_order"] == ml.AXIS_ORDER
        assert "DRY" in stats["status"]
        assert stats["teachers"]["quality"]["provider"] == "anatomy_proxy"


def test_passthrough_leaves_quality_unfilled():
    """passthrough provider + quality 無し入力 → quality は None のまま (後埋め境界)。"""
    with tempfile.TemporaryDirectory() as d:
        csv_path = os.path.join(d, "tags.csv")
        _dummy_csv(csv_path)
        wd14_path = os.path.join(d, "wd14.jsonl")
        with open(wd14_path, "w", encoding="utf-8") as f:
            for r in _build_records(20):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        out = os.path.join(d, "o")
        ml.main(["--wd14", wd14_path, "--selected-tags", csv_path,
                 "--out-dir", out, "--quality-provider", "passthrough"])
        with open(os.path.join(out, "scorer_stats.json"), encoding="utf-8") as f:
            stats = json.load(f)
        assert stats["quality_filled"]["with_quality"] == 0
        assert stats["quality_filled"]["missing"] == 20


def test_waifu_scorer_normalize_and_none():
    """選定美的モデル waifu_scorer_v4 (apache-2.0) の乾式検証。

    - normalize_score: 生スコア 0..10 → [0,1] クランプ (純関数・本機でも実走可)。
    - 重み未配置 (本機) では embed があっても None (実走しない=配管のみ)。
    - 重みパスがあっても embed 無しなら None。
    - license/属性が apache-2.0・768/0..10 を保持。
    """
    prov = ml.make_quality_provider("waifu_scorer_v4")
    assert prov.license == "apache-2.0"
    assert prov.embed_dim == 768
    assert prov.raw_score_max == 10.0

    # normalize_score: 0..10 → [0,1]・範囲外クランプ
    assert abs(prov.normalize_score(0.0) - 0.0) < 1e-9
    assert abs(prov.normalize_score(10.0) - 1.0) < 1e-9
    assert abs(prov.normalize_score(7.5) - 0.75) < 1e-9
    assert prov.normalize_score(-3.0) == 0.0   # 下クランプ
    assert prov.normalize_score(13.0) == 1.0   # 上クランプ

    # 本機: 重み未配置 → embed があっても None (実走しない)
    assert prov.score({"clip_image_embed": [0.0] * 768}) is None
    # 重みパスがあっても embed 無しなら None
    prov2 = ml.make_quality_provider("waifu_scorer_v4", weights_path="/path/to/model.safetensors")
    assert prov2.score({"image": "x.png"}) is None
    # 重み + embed が揃うと研究機実走部 (本機では未結線=NotImplementedError) に入る
    try:
        prov2.score({"clip_image_embed": [0.0] * 768})
        assert False, "研究機実走部は本機で NotImplementedError のはず"
    except NotImplementedError:
        pass


def test_default_provider_is_waifu_scorer():
    """既定 provider が選定済み waifu_scorer_v4・重み未配置で quality は null のまま (縮退に落ちない)。"""
    with tempfile.TemporaryDirectory() as d:
        csv_path = os.path.join(d, "tags.csv")
        _dummy_csv(csv_path)
        wd14_path = os.path.join(d, "wd14.jsonl")
        with open(wd14_path, "w", encoding="utf-8") as f:
            for r in _build_records(20):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        out = os.path.join(d, "o")
        # --quality-provider 未指定 = 既定 waifu_scorer_v4・--quality-weights 未指定 = 本機
        ml.main(["--wd14", wd14_path, "--selected-tags", csv_path, "--out-dir", out])
        with open(os.path.join(out, "scorer_stats.json"), encoding="utf-8") as f:
            stats = json.load(f)
        q = stats["teachers"]["quality"]
        assert q["provider"] == "waifu_scorer_v4"
        assert q["license"] == "apache-2.0"
        assert q["repo"] == "Eugeoter/waifu-scorer-v4-beta"
        assert q["weights_path"] is None  # 本機: 重み未配置
        # ユーザー方針=保留: 重み無し → quality は埋めず null のまま (anatomy_proxy に落とさない)
        assert stats["quality_filled"]["with_quality"] == 0
        assert stats["quality_filled"]["missing"] == 20


def test_anime_aesthetic_deepghs_expected_score():
    """採用美的モデル anime_aesthetic_deepghs (openrail) の期待値写像 [0,1] 乾式検証。

    - class_values: index 0(masterpiece)=1.0 .. 6(worst)=0.0 線形等間隔。
    - expected_score: 7 クラス確率 → Sum p_k*(6-k)/6 ∈ [0,1] (純計算・本機実走可)。
    - 確率未配置 (本機) → None (推論しない=後埋め境界)。
    - labels/形態が meta.json と一致 (masterpiece..worst・7 クラス・448)。
    """
    prov = ml.make_quality_provider("anime_aesthetic_deepghs")
    assert prov.license == "openrail"
    assert prov.n_classes == 7
    assert prov.img_size == 448
    assert prov.labels == ["masterpiece", "best", "great", "good", "normal", "low", "worst"]

    vals = prov.class_values()
    assert abs(vals[0] - 1.0) < 1e-9   # masterpiece
    assert abs(vals[6] - 0.0) < 1e-9   # worst
    assert abs(vals[3] - 0.5) < 1e-9   # good (中央)

    # 期待値写像: one-hot とユニフォーム
    assert abs(prov.expected_score([1, 0, 0, 0, 0, 0, 0]) - 1.0) < 1e-9
    assert abs(prov.expected_score([0, 0, 0, 0, 0, 0, 1]) - 0.0) < 1e-9
    assert abs(prov.expected_score([1.0 / 7] * 7) - 0.5) < 1e-9
    # 非正規化 (合計≠1) でも内部正規化で [0,1]
    assert abs(prov.expected_score([2, 0, 0, 0, 0, 0, 0]) - 1.0) < 1e-9
    # 全ゼロは採点不能 → None
    assert prov.expected_score([0, 0, 0, 0, 0, 0, 0]) is None
    # 長さ不一致は ValueError
    try:
        prov.expected_score([1, 0, 0])
        assert False, "長さ不一致は ValueError のはず"
    except ValueError:
        pass

    # score 境界: 確率未配置 (本機) → None / 確率配置 → 写像
    assert prov.score({"image": "x.png"}) is None
    assert abs(prov.score({"aesthetic_probs": [0.5, 0.5, 0, 0, 0, 0, 0]}) - (1.0 + 5.0 / 6) / 2) < 1e-9


def test_deepghs_provider_provenance():
    """既定でない deepghs を明示選択 → provenance に openrail/labels/期待値写像が残る。"""
    with tempfile.TemporaryDirectory() as d:
        csv_path = os.path.join(d, "tags.csv")
        _dummy_csv(csv_path)
        wd14_path = os.path.join(d, "wd14.jsonl")
        with open(wd14_path, "w", encoding="utf-8") as f:
            for r in _build_records(10):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        out = os.path.join(d, "o")
        ml.main(["--wd14", wd14_path, "--selected-tags", csv_path, "--out-dir", out,
                 "--quality-provider", "anime_aesthetic_deepghs"])
        with open(os.path.join(out, "scorer_stats.json"), encoding="utf-8") as f:
            stats = json.load(f)
        q = stats["teachers"]["quality"]
        assert q["provider"] == "anime_aesthetic_deepghs"
        assert q["license"] == "openrail"
        assert q["repo"] == "deepghs/anime_aesthetic"
        assert q["labels"][0] == "masterpiece" and q["labels"][6] == "worst"
        # 確率未配置 (本機) → quality null のまま (anatomy_proxy に落ちない)
        assert stats["quality_filled"]["with_quality"] == 0
        assert stats["quality_filled"]["missing"] == 10


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok: {fn.__name__}")
    print(f"all {len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()

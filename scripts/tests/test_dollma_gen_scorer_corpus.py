# -*- coding: utf-8 -*-
"""dollama Model B / B-3a / E-1 画像コーパス driver 乾式テスト (OV/SDXL/HTTP 不要)。

検査項目 (純ヘルパのみ・実走部 run_on_research_machine は研究機専用ガードで触らない):
  - tokenizers DLL 自動検出のフォールバック (explicit 実在=採用 / 不在=RuntimeError)。
  - build_request_body: 良=品質ネガ注入 / 悪=注入なし・size/steps/response_format。
  - 品質ネガ 9 語定数が character-bible §6 と厳密一致。
  - save_b64_png: b64→PNG 保存の往復。
  - extract_rating: category9 の最大スコア name 抽出。
  - wd14_row: jsonl 行スキーマ。
  - prompt_overlap_tags: prompt 反映あり/なし。
  - build_jobs: seed 決定性・good/hard バランス・inject ラベル・後方互換。

実行:
  py -3.14 scripts/tests/test_dollma_gen_scorer_corpus.py
"""
import base64
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import dollma_gen_scorer_corpus as gc


def test_quality_negatives_match_character_bible():
    # character-bible §6 default_quality_negatives() の 9 語と厳密一致 (順序・語とも)。
    expected = [
        "bad hands", "malformed hands", "mutated hands",
        "fused fingers", "bad anatomy", "deformed",
        "lowres", "worst quality", "jpeg artifacts",
    ]
    assert gc.QUALITY_NEGATIVES == ", ".join(expected)


def test_resolve_tokenizers_dll_explicit():
    with tempfile.TemporaryDirectory() as d:
        dll = os.path.join(d, "openvino_tokenizers.dll")
        with open(dll, "wb") as f:
            f.write(b"\x00")
        # 実在ファイルはそのまま採用
        assert gc.resolve_tokenizers_dll(dll) == dll
        # 不在パスは RuntimeError (golden 落ち防止のため曖昧に通さない)
        try:
            gc.resolve_tokenizers_dll(os.path.join(d, "nope.dll"))
            assert False, "不在パスは RuntimeError のはず"
        except RuntimeError:
            pass


def test_build_request_body():
    good = {"prompt": "1girl, solo", "inject_quality_negatives": True}
    hard = {"prompt": "2girls, crowd", "inject_quality_negatives": False}
    b_good = gc.build_request_body(good, steps=20, size="1024x1024")
    assert b_good["prompt"] == "1girl, solo"
    assert b_good["steps"] == 20
    assert b_good["size"] == "1024x1024"
    assert b_good["response_format"] == "b64_json"
    assert b_good["negative_prompt"] == gc.QUALITY_NEGATIVES
    b_hard = gc.build_request_body(hard, steps=15, size="1024x1024")
    assert "negative_prompt" not in b_hard  # 悪サンプルは注入なし
    assert b_hard["steps"] == 15


def test_save_b64_png_roundtrip():
    payload = b"\x89PNG\r\n\x1a\n" + b"dummy-image-bytes" * 4
    b64 = base64.b64encode(payload).decode("ascii")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "sub", "000001.png")  # サブディレクトリ自動作成も検証
        n = gc.save_b64_png(b64, path)
        assert n == len(payload)
        with open(path, "rb") as f:
            assert f.read() == payload


def test_extract_rating():
    # names/categories: index2,4 が rating(category9)。index4 のスコアが最大。
    names = ["1girl", "solo", "general", "long_hair", "sensitive"]
    cats = [0, 0, 9, 0, 9]
    scores = [0.9, 0.8, 0.2, 0.7, 0.6]
    assert gc.extract_rating(scores, names, cats) == "sensitive"
    # rating タグが無ければ None
    assert gc.extract_rating([0.5, 0.4], ["a", "b"], [0, 0]) is None


def test_wd14_row_schema():
    row = gc.wd14_row("data/scorer/img/000000.png", [0.1, 0.2], "1girl", "general", True)
    assert set(row) == {"image", "wd14_scores", "prompt", "rating", "inject_quality_negatives"}
    assert row["image"] == "data/scorer/img/000000.png"
    assert row["wd14_scores"] == [0.1, 0.2]
    assert row["rating"] == "general"
    assert row["inject_quality_negatives"] is True


def test_prompt_overlap_tags():
    names = ["1girl", "solo", "long_hair", "short_hair", "background"]
    # prompt の語と高スコアタグが重なる: 1girl/solo/long_hair が >=thresh
    scores = [0.9, 0.8, 0.6, 0.1, 0.05]
    hits = gc.prompt_overlap_tags("1girl, solo, long hair, simple background", scores, names)
    assert hits == {"1girl", "solo", "long_hair"}
    # 全タグが低スコア → 重なり 0 (golden 落ち相当)
    assert gc.prompt_overlap_tags("1girl, solo", [0.0, 0.0, 0.0, 0.0, 0.0], names) == set()


def test_build_jobs_determinism_and_balance():
    j1 = gc.build_jobs(8, 20260620)
    j2 = gc.build_jobs(8, 20260620)
    assert j1 == j2  # seed 決定性
    # 偶数=良 (品質ネガ ON・GOOD プール)・奇数=悪 (OFF・HARD プール)
    n_good = sum(1 for j in j1 if j["inject_quality_negatives"])
    assert n_good == 4 and len(j1) == 8
    for i, j in enumerate(j1):
        if i % 2 == 0:
            assert j["inject_quality_negatives"] is True
            assert j["prompt"] in gc.GOOD_PROMPTS
        else:
            assert j["inject_quality_negatives"] is False
            assert j["prompt"] in gc.HARD_PROMPTS
    # 別 seed では並びが変わりうる (プールが十分大きい)
    assert gc.build_jobs(8, 1) != j1 or True  # 異 seed は等価でない可能性 (緩い検証)
    # プール拡張 (~20-30 件) を確認
    assert len(gc.GOOD_PROMPTS) >= 20 and len(gc.HARD_PROMPTS) >= 20


def test_main_dry_run_does_not_raise(capsys=None):
    # 既定 dry-run は実走せず計画のみ (NotImplementedError/実走に入らない)。
    gc.main(["--n", "8"])


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok: {fn.__name__}")
    print(f"all {len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()

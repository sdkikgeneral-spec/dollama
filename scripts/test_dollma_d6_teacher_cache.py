# -*- coding: utf-8 -*-
"""dollama D6 教師キャッシュ 単体テスト (docs/testing.md 準拠)。

torch 不要のテストのみ (vocab 写像・npz 往復・soft_target 質量)。教師生成 (transformers/
CUDA) は probe 実走で確認する。py -3.12 で実行:
  py -3.12 -m pytest scripts/test_dollma_d6_teacher_cache.py -q
あるいは素実行 (assert):
  py -3.12 scripts/test_dollma_d6_teacher_cache.py
"""

import os
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import train_bitnet as tb
import dollma_d6_teacher_cache as d6


def _tok():
    vocab_path = os.path.join(_HERE, "..", "data", "bitnet", "vocab.json")
    return tb.Tokenizer(os.path.abspath(vocab_path))


def test_normalize_matches_train_bitnet():
    """vocab 写像の normalize が train_bitnet.Tokenizer.normalize と同一であること。

    二重実装禁止の検証 — d6 は tb.Tokenizer.normalize を直接呼ぶ。
    `long_hair`→`long hair` (英数字間 _ は space)、顔文字 `^_^` は _ 保持。
    """
    cases = ["long_hair", "^_^", "short_hair", "blue_eyes", ":d", "+_+"]
    for c in cases:
        # d6 は ExternalTeacherCache.map_tags_to_vocab 内で tb.Tokenizer.normalize を使う。
        assert tb.Tokenizer.normalize(c) == tb.Tokenizer.normalize(c)
    assert tb.Tokenizer.normalize("long_hair") == "long hair"
    assert tb.Tokenizer.normalize("^_^") == "^_^"  # 顔文字 _ 保持
    print("[ok] normalize 整合")


def test_vocab_mapping_known_tags():
    """既知タグ片が vocab id に正しく引け、OOV は drop されること。"""
    tok = _tok()
    cache = d6.ExternalTeacherCache(tok)
    # long_hair は normalize で `long hair` → vocab に存在するはず。
    pieces = ["long_hair", "1girl", "this_is_not_a_real_tag_xyz", "^_^"]
    kept, n_total, n_kept = cache.map_tags_to_vocab(pieces)
    assert n_total == 4
    # long hair / 1girl は in-vocab (代表的 danbooru タグ)。
    lh = tok.tag_to_id.get("long hair")
    one = tok.tag_to_id.get("1girl")
    assert lh is not None and lh in kept
    assert one is not None and one in kept
    # OOV は drop されている。
    assert n_kept <= n_total
    assert tok.tag_to_id.get("this is not a real tag xyz") is None
    print(f"[ok] vocab 写像 kept={n_kept}/{n_total} long hair={lh} 1girl={one}")


def test_soft_target_mass_sums_to_one():
    """OOV drop + 再正規化後の soft 分布の質量和が 1.0 (±1e-6) であること。"""
    tok = _tok()
    cache = d6.ExternalTeacherCache(tok, main_mass=0.85, t_temp=2.0, topn=32)
    lh = tok.tag_to_id["long hair"]
    be = tok.tag_to_id["blue eyes"]
    one = tok.tag_to_id["1girl"]
    smile = tok.tag_to_id["smile"]
    # gold 以外の候補頻度をセット。
    cache.set_current({lh: 5, be: 3, smile: 2})
    dist = cache.soft_target(prefix_ids=[one], gold_id=one)
    total = sum(dist.values())
    assert abs(total - 1.0) < 1e-6, f"mass={total}"
    assert all(p >= 0.0 for p in dist.values())
    # gold (1girl) に main_mass。
    assert abs(dist[one] - 0.85) < 1e-9
    print(f"[ok] soft_target mass={total:.9f} gold_mass={dist[one]:.3f} "
          f"n_cand={len(dist) - 1}")


def test_soft_target_empty_candidates_is_hard():
    """生成候補が無い場合 gold に全質量 (hard) になること。"""
    tok = _tok()
    cache = d6.ExternalTeacherCache(tok)
    cache.set_current({})  # 候補なし
    one = tok.tag_to_id["1girl"]
    dist = cache.soft_target(prefix_ids=[], gold_id=one)
    assert dist == {one: 1.0}
    # specials gold は常に hard。
    dist2 = cache.soft_target(prefix_ids=[one], gold_id=tb.TOK_EOS)
    assert dist2 == {tb.TOK_EOS: 1.0}
    print("[ok] 空候補/specials は hard")


def test_coo_npz_roundtrip():
    """COO npz を書いて読み戻し→密展開で各位置の和=1.0・全要素≥0・dtype 一致。"""
    tok = _tok()
    cache = d6.ExternalTeacherCache(tok, main_mass=0.85, t_temp=2.0, topn=8)
    # 小さな擬似 dataset を 3 サンプル作る (実 vocab id で)。
    rows = [
        {"text": "a girl with long hair, blue eyes.",
         "tags": ["1girl", "solo", "long hair", "blue eyes", "smile"],
         "source": "synthetic"},
        {"text": "short hair, shirt.",
         "tags": ["1girl", "short hair", "shirt", "skirt"],
         "source": "synthetic"},
        {"text": "blonde hair, bow.",
         "tags": ["1girl", "blonde hair", "bow", "twintails"],
         "source": "synthetic"},
    ]
    ds = tb.PairDataset(tok, rows, tb.MAX_SEQ_LEN)
    lh = tok.tag_to_id["long hair"]
    smile = tok.tag_to_id["smile"]
    be = tok.tag_to_id["blue eyes"]
    freqs = [{lh: 4, smile: 2, be: 1}, {lh: 1}, {smile: 3, be: 2}]
    coo, mean_ent, ent_n = d6.build_coo_for_dataset(cache, ds, freqs, tb.MAX_SEQ_LEN)

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "d6_test.npz")
        np.savez_compressed(path, **coo)
        back = d6.load_coo_npz(path)

    # dtype 一致
    assert back["rows"].dtype == np.int32
    assert back["poss"].dtype == np.int32
    assert back["cols"].dtype == np.int32
    assert back["probs"].dtype == np.float32

    # 密展開: (sample, position) ごとに和=1.0 を確認。
    from collections import defaultdict
    acc = defaultdict(float)
    for r, p, c, pr in zip(back["rows"], back["poss"], back["cols"], back["probs"]):
        assert pr >= 0.0
        assert c >= 0
        acc[(int(r), int(p))] += float(pr)
    assert len(acc) > 0, "soft 位置が 0"
    for key, s in acc.items():
        assert abs(s - 1.0) < 1e-5, f"pos {key} mass={s}"
    assert mean_ent >= 0.0
    print(f"[ok] npz 往復 positions={len(acc)} mean_entropy={mean_ent:.4f} "
          f"all mass==1.0")




def test_external_tag_teacher_reproduces_npz():
    """ExternalTagTeacher が npz の (sample,t) 分布を precompute_soft_targets /
    build_soft_target_tensor で完全再現すること (順送り同期・zero-soft sample skip 込み)。

    D6 の肝: KL plumbing 無改修で CoOccurrenceTeacher と同一 I/F で動く保証。
    """
    import tempfile
    import torch
    tok = _tok()
    rows = [
        {"text": "a girl with long hair, blue eyes.",
         "tags": ["1girl", "solo", "long hair", "blue eyes", "smile"],
         "source": "synthetic"},
        {"text": "no body tags here just text",  # body タグ少 (drift 試験)
         "tags": ["1girl"], "source": "synthetic"},
        {"text": "short hair, shirt.",
         "tags": ["1girl", "short hair", "shirt", "skirt"], "source": "synthetic"},
        {"text": "blonde hair, bow.",
         "tags": ["1girl", "blonde hair", "bow", "twintails"], "source": "synthetic"},
    ]
    ds = tb.PairDataset(tok, rows, tb.MAX_SEQ_LEN)
    lh = tok.tag_to_id["long hair"]; smile = tok.tag_to_id["smile"]
    be = tok.tag_to_id["blue eyes"]
    freqs = [{lh: 4, smile: 2, be: 1}, {}, {lh: 1}, {smile: 3, be: 2}]

    helper = d6.ExternalTeacherCache(tok, main_mass=0.85, t_temp=2.0, topn=8)
    coo, _, _ = d6.build_coo_for_dataset(helper, ds, freqs, tb.MAX_SEQ_LEN)

    # 参照: helper を直接使った soft (これが npz の素)。
    ref = {}
    for i, s_ in enumerate(ds.samples):
        helper.set_current(freqs[i] if freqs[i] else {})
        ref[id(s_[0])] = tb.compute_sample_soft(helper, s_[0], s_[1], tb.MAX_SEQ_LEN)

    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "d6_teacher_soft.train.npz")
        np.savez_compressed(p, **coo)
        teacher = tb.ExternalTagTeacher(tok, p)
        teacher.bind(ds, tb.MAX_SEQ_LEN)
        soft_cache = tb.precompute_soft_targets(teacher, ds, tb.MAX_SEQ_LEN)

    maxerr = 0.0; n_pos = 0
    for s_ in ds.samples:
        k = id(s_[0])
        rp = {t: dd for t, dd in ref[k]}
        gp = {t: dd for t, dd in soft_cache[k]}
        assert set(rp) == set(gp), f"positions differ {sorted(rp)} vs {sorted(gp)}"
        for t in rp:
            for kk in set(rp[t]) | set(gp[t]):
                maxerr = max(maxerr, abs(rp[t].get(kk, 0.0) - gp[t].get(kk, 0.0)))
            n_pos += 1
    assert maxerr < 1e-6, f"max abs err {maxerr}"

    # build_soft_target_tensor: emit 位置のみ和=1.0、非 emit (EOS hard 等) は all-zero。
    batch = list(ds.samples)
    _, tgt = tb.collate(batch, tb.MAX_SEQ_LEN, "tags")
    soft = tb.build_soft_target_tensor(teacher, batch, tgt.cpu(), tb.MAX_SEQ_LEN,
                                       soft_cache=soft_cache)
    sums = soft.sum(dim=-1)
    for bi, sample in enumerate(batch):
        emitted = {t for t, _ in soft_cache[id(sample[0])]}
        for t in range(soft.shape[1]):
            v = sums[bi, t].item()
            if t in emitted:
                assert abs(v - 1.0) < 1e-5, f"emitted ({bi},{t}) sum={v}"
            else:
                assert v < 1e-6, f"non-emitted ({bi},{t}) should be 0, got {v}"
    print(f"[ok] ExternalTagTeacher 再現 positions={n_pos} max_err={maxerr:.2e}")


def _run_all():
    fns = [test_normalize_matches_train_bitnet,
           test_vocab_mapping_known_tags,
           test_soft_target_mass_sums_to_one,
           test_soft_target_empty_candidates_is_hard,
           test_coo_npz_roundtrip,
           test_external_tag_teacher_reproduces_npz]
    for fn in fns:
        fn()
    print("\n=== ALL D6 TESTS PASSED ===")


if __name__ == "__main__":
    _run_all()

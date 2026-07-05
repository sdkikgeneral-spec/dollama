# -*- coding: utf-8 -*-
"""dollama Phase 4 F-0b (G-2a) — rejection-sampling SFT 経路 単体テスト。

検証項目 (docs/testing.md 準拠・torch 依存):
  1. schema 読込: source=="rejection_sft" 行が build_sequence の synthetic 分岐 (1-<sep>) で
     読め、loss 対象区間 (tags_start) が空でないこと。
  2. 正典からの層状 warm-start: _load_export_into_model が正典 FP32 を strict ロードし、
     ロード後の forward が _load_dense_from_export (別経路ロード) と bitwise 一致すること
     = SFT が「乱数初期化でなく正典から始まる」ことの直接検証。
  3. 決定性: 同一 seed・CPU で --sft-rejection --smoke を 2 回走らせ、出力重み
     (bitnet_dense_sft_smoke_fp32.safetensors) が sha256 一致すること。かつログに
     層状 warm-start 行が出ること (正典から始まった証跡)。

torch を import するため py -3.14 (cu128) 環境で実行する。SAC は Python に非該当。
実行:
  python scripts/test_dollma_sft_rejection.py
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_DATA = os.path.join(_HERE, "..", "data", "bitnet")
_VOCAB = os.path.join(_DATA, "vocab.json")
_CANON_FP32 = os.path.join(_DATA, "bitnet_dense_fp32.safetensors")
_SFT_DATA = os.path.join(_HERE, "..", "data", "rollouts", "sft_bestofn.jsonl")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def test_sft_schema_read():
    """source=rejection_sft が synthetic 分岐で読め、loss 区間が非空であること。"""
    import train_bitnet as tb
    tok = tb.Tokenizer(_VOCAB)
    # 実 SFT データ先頭数行で検証。
    rows = tb.load_pairs(_SFT_DATA)[:5]
    assert rows, "SFT データが空"
    for r in rows:
        assert r.get("source") == "rejection_sft", r.get("source")
        ids, tags_start = tb.build_sequence(tok, r, tb.MAX_SEQ_LEN)
        # synthetic 分岐: <bos> text <sep> tags <eos>。<sep> は 1 個。
        assert ids[0] == tb.TOK_BOS
        assert ids.count(tb.TOK_SEP) == 1, f"rejection_sft は 1-<sep> のはず: {ids.count(tb.TOK_SEP)}"
        assert ids[-1] == tb.TOK_EOS
        # loss 対象 (tags_start 以降) が最低 1 トークンある。
        assert tags_start < len(ids), (tags_start, len(ids))
    print(f"[ok] schema: rejection_sft {len(rows)} 行が synthetic 1-<sep> で読める・loss 区間非空")


def test_layered_warmstart_matches_canonical():
    """_load_export_into_model が正典を strict ロードし、別経路ロード
    (_load_dense_from_export) と forward bitwise 一致 = 正典から始まる証明。"""
    import torch
    import train_bitnet as tb
    if not os.path.exists(_CANON_FP32):
        print(f"[skip] 正典 {_CANON_FP32} が無い")
        return
    tb.apply_arch()  # base 定数を確定
    # 経路1: 新規モデルへ層状ロード (SFT が使う関数)。
    m1 = tb.BitNetDense().to("cpu")
    n = tb._load_export_into_model(m1, _CANON_FP32)
    assert n == len(m1.state_dict()), (n, len(m1.state_dict()))
    m1.eval()
    # 経路2: 参照ロード。
    m2 = tb._load_dense_from_export(_CANON_FP32, "cpu")
    # state_dict bitwise 一致。
    sd1, sd2 = m1.state_dict(), m2.state_dict()
    assert set(sd1) == set(sd2)
    for k in sd1:
        assert torch.equal(sd1[k], sd2[k]), f"{k} 不一致"
    # forward bitwise 一致 (固定入力)。
    ids = [tb.TOK_BOS, 5, 6, 7, tb.TOK_SEP, 8, 9]
    inp = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        d = (m1(inp) - m2(inp)).abs().max().item()
    assert d == 0.0, f"forward 不一致 max_abs={d}"
    print(f"[ok] warm-start: 正典 {n} テンソル strict ロード・参照経路と forward bitwise 一致")


def test_sft_smoke_determinism_and_canonical_init():
    """CPU・同一 seed で --sft-rejection --smoke を 2 回走らせ重み sha256 一致、
    かつ層状 warm-start ログが出ること (正典から始まった証跡)。"""
    if not os.path.exists(_CANON_FP32):
        print(f"[skip] 正典 {_CANON_FP32} が無い")
        return
    out = os.path.join(_DATA, "bitnet_dense_sft_smoke_fp32.safetensors")
    script = os.path.join(_HERE, "train_bitnet.py")
    shas = []
    warm_seen = 0
    for _ in range(2):
        if os.path.exists(out):
            os.remove(out)
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        r = subprocess.run(
            [sys.executable, script, "--sft-rejection", "--smoke",
             "--device", "cpu", "--seed", "20260620", "--warmup", "5",
             "--lr", "2e-5"],
            cwd=os.path.join(_HERE, ".."), capture_output=True, text=True,
            encoding="utf-8", env=env)
        assert r.returncode == 0, f"smoke 失敗:\n{r.stdout}\n{r.stderr}"
        assert "warm-start" in r.stdout, "層状 warm-start ログが無い (正典から始まっていない?)"
        warm_seen += 1
        assert os.path.exists(out), "smoke 重み未出力"
        shas.append(_sha256(out))
    assert shas[0] == shas[1], f"決定性違反 sha {shas[0][:12]} != {shas[1][:12]}"
    # 後始末 (smoke 別名は本番非干渉だが残さない)。
    for suf in ("_smoke.safetensors", "_smoke_fp32.safetensors"):
        p = os.path.join(_DATA, "bitnet_dense_sft" + suf)
        if os.path.exists(p):
            os.remove(p)
    ss = os.path.join(_DATA, "train_stats_sft_smoke.json")
    if os.path.exists(ss):
        os.remove(ss)
    print(f"[ok] 決定性: 2 回とも warm-start 経由・重み sha256 一致 ({shas[0][:12]})")


def _run_all():
    for fn in (test_sft_schema_read,
               test_layered_warmstart_matches_canonical,
               test_sft_smoke_determinism_and_canonical_init):
        fn()
    print("\n=== ALL SFT-REJECTION TESTS PASSED ===")


if __name__ == "__main__":
    _run_all()

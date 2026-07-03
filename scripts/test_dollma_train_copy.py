# -*- coding: utf-8 -*-
"""dollama Phase 4 — train_bitnet.py の正典アセット搬送 (--copy / --publish) 乾式テスト。

検査項目:
  - --copy: src(リモート base) -> ローカル --data-dir へ exact コピー・sha256/size 一致。
  - 冪等: 2 回目は SKIP_SAME (再コピーしない)。
  - 往復: publish で書き戻し → 再 copy で戻して sha 完全一致。
  - 上書きログ: 差分ありの --publish は old(sha,size)->new(sha,size) を必ずログ (silent 禁止)。
  - source 不在の対象は WARN skip で継続 (exit 0)。
  - --copy と --publish 同時指定はエラー (nonzero exit)。
  - **torch 非依存**: 偽 torch(import で必ず失敗) を PYTHONPATH 先頭に置いても --copy は成功する
    = 搬送処理が `import torch` より前で走り return する証拠。対照: 搬送フラグ無し実行は
    偽 torch で失敗する (= 偽 torch shim が実際に効いており、--copy だけが torch を回避)。

本テストは **本物の重みを使わず小ダミーファイル** で回す (軽量・torch 不要)。
サブプロセスで実スクリプトを起動して実挙動を検証する。

実行:
  py -3.12 scripts/test_dollma_train_copy.py
  python  scripts/test_dollma_train_copy.py      # torch 不要 (サブプロセス起動のみ)
  py -3.12 -m pytest scripts/test_dollma_train_copy.py -q
"""

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPT = os.path.join(_HERE, "train_bitnet.py")
_PY = sys.executable


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def _run(args, env=None):
    """train_bitnet.py をサブプロセスで起動し (rc, stdout+stderr) を返す。"""
    cp = subprocess.run([_PY, _SCRIPT] + args, cwd=_HERE,
                        capture_output=True, text=True, env=env)
    return cp.returncode, (cp.stdout or "") + (cp.stderr or "")


def _make_remote(base):
    """小ダミーの正典一式を作る (重み2本 + a12k + golden 2ファイル・identity 重みは意図的に欠損)。"""
    files = {
        "bitnet_dense.safetensors": b"DUMMY-W-FP16-0001",
        "bitnet_dense_fp32.safetensors": b"DUMMY-W-FP32-0001",
        "pairs.identity.train.a12k.jsonl": b'{"a":1}\n{"a":2}\n',
        "pairs.identity.val.a12k.jsonl": b'{"v":1}\n',
        os.path.join("golden", "logits_golden.safetensors"): b"GOLDEN-LOGITS",
        os.path.join("golden", "gen_golden.safetensors"): b"GOLDEN-GEN",
        os.path.join("golden", "manifest.json"): b'{"m":true}',
        os.path.join("golden", "manifest_identity.json"): b'{"mi":true}',
    }
    for rel, data in files.items():
        _write(os.path.join(base, rel), data)
    return files


def _tmp():
    return tempfile.mkdtemp(prefix="dollama_transport_")


def test_copy_sha_and_idempotent():
    root = _tmp()
    try:
        remote = os.path.join(root, "remote")
        local = os.path.join(root, "local")
        files = _make_remote(remote)
        rc, out = _run(["--copy", remote, "--data-dir", local])
        assert rc == 0, f"--copy rc={rc}\n{out}"
        # 実在対象は sha 一致でコピーされる。
        for rel in files:
            s, d = os.path.join(remote, rel), os.path.join(local, rel)
            assert os.path.exists(d), f"未コピー: {rel}\n{out}"
            assert _sha(s) == _sha(d), f"sha 不一致: {rel}"
        assert "verify_fail=0" in out, out
        # 2 回目は冪等 (SKIP_SAME・copied=0)。
        rc2, out2 = _run(["--copy", remote, "--data-dir", local])
        assert rc2 == 0, out2
        assert "copied=0" in out2 and "SKIP_SAME" in out2, out2
        print("[ok] test_copy_sha_and_idempotent")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_publish_roundtrip():
    root = _tmp()
    try:
        remote = os.path.join(root, "remote")
        local = os.path.join(root, "local")
        remote2 = os.path.join(root, "remote2")
        _make_remote(remote)
        assert _run(["--copy", remote, "--data-dir", local])[0] == 0
        # publish: local -> remote2。
        rc, out = _run(["--publish", remote2, "--data-dir", local])
        assert rc == 0, out
        assert "verify_fail=0" in out, out
        # 往復整合: remote2 の全対象が local と sha 一致。
        for dirpath, _dn, fns in os.walk(local):
            for fn in fns:
                lf = os.path.join(dirpath, fn)
                rel = os.path.relpath(lf, local)
                rf = os.path.join(remote2, rel)
                assert os.path.exists(rf), f"publish 漏れ: {rel}"
                assert _sha(lf) == _sha(rf), f"往復 sha 不一致: {rel}"
        print("[ok] test_publish_roundtrip")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_overwrite_logs_old_and_new():
    root = _tmp()
    try:
        remote = _tmp()  # unused base pattern; keep local/dst here
        local = os.path.join(root, "local")
        dst = os.path.join(root, "dst")
        _make_remote(local)               # local を source_of_truth に
        assert _run(["--publish", dst, "--data-dir", local])[0] == 0
        # dst 側の 1 ファイルを改変 → 再 publish で OVERWRITE ログ (old->new)。
        _write(os.path.join(dst, "bitnet_dense.safetensors"), b"STALE-DIFFERENT-BYTES")
        rc, out = _run(["--publish", dst, "--data-dir", local])
        assert rc == 0, out
        assert "OVERWRITE" in out and "old(" in out and "new(" in out, \
            f"上書きログが不十分 (silent 上書き禁止):\n{out}"
        assert _sha(os.path.join(local, "bitnet_dense.safetensors")) == \
               _sha(os.path.join(dst, "bitnet_dense.safetensors"))
        shutil.rmtree(remote, ignore_errors=True)
        print("[ok] test_overwrite_logs_old_and_new")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_missing_targets_warn_continue():
    root = _tmp()
    try:
        remote = os.path.join(root, "remote")
        local = os.path.join(root, "local")
        # identity 重み・a12k val を欠いた source (_make_remote は identity 重みを含めない)。
        _make_remote(remote)
        rc, out = _run(["--copy", remote, "--data-dir", local])
        assert rc == 0, out
        assert "WARN skip" in out and "bitnet_dense_identity.safetensors" in out, out
        assert "missing=" in out, out
        print("[ok] test_missing_targets_warn_continue")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_mutual_exclusive():
    root = _tmp()
    try:
        remote = os.path.join(root, "remote")
        local = os.path.join(root, "local")
        _make_remote(remote)
        rc, out = _run(["--copy", remote, "--publish", remote, "--data-dir", local])
        assert rc != 0, f"同時指定はエラーであるべき:\n{out}"
        assert "同時指定不可" in out, out
        print("[ok] test_mutual_exclusive")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_torch_free():
    """偽 torch(import で失敗) を PYTHONPATH 先頭に置いても --copy は成功する。
    = 搬送が `import torch` より前で走り return する証拠。対照: フラグ無し実行は失敗する。"""
    root = _tmp()
    try:
        remote = os.path.join(root, "remote")
        local = os.path.join(root, "local")
        shim = os.path.join(root, "shim")
        _make_remote(remote)
        # 偽 torch: import された瞬間に必ず失敗する。
        _write(os.path.join(shim, "torch.py"),
               b'raise ImportError("torch must not be imported for --copy/--publish")\n')
        env = dict(os.environ)
        env["PYTHONPATH"] = shim + os.pathsep + env.get("PYTHONPATH", "")

        # (a) --copy: 偽 torch があっても成功 (torch を import しないため)。
        rc, out = _run(["--copy", remote, "--data-dir", local], env=env)
        assert rc == 0, f"--copy は torch 非依存で成功すべき:\n{out}"
        assert os.path.exists(os.path.join(local, "bitnet_dense.safetensors")), out

        # (b) 対照: 搬送フラグ無しの実行は偽 torch で失敗する
        #     (= shim が実際に効いており、--copy だけが import torch を回避している証明)。
        rc2, out2 = _run(["--help"], env=env)
        assert rc2 != 0, ("対照(--help)は偽 torch で失敗すべき (shim が効いている証拠)。"
                          f"失敗しないなら torch 依存の検証が無意味:\n{out2}")
        assert "torch must not be imported" in out2 or "ImportError" in out2 \
            or "ModuleNotFoundError" in out2, out2
        print("[ok] test_torch_free (--copy は torch 非依存・対照は shim で失敗)")
    finally:
        shutil.rmtree(root, ignore_errors=True)


_TESTS = [
    test_copy_sha_and_idempotent,
    test_publish_roundtrip,
    test_overwrite_logs_old_and_new,
    test_missing_targets_warn_continue,
    test_mutual_exclusive,
    test_torch_free,
]


def main():
    fails = 0
    for t in _TESTS:
        try:
            t()
        except AssertionError as e:
            fails += 1
            print(f"[FAIL] {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"[ERROR] {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILED'} ({len(_TESTS)} tests)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

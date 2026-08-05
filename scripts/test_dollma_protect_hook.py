# -*- coding: utf-8 -*-
"""正典アーティファクト保護 hook の検証 (docs/testing.md 準拠)。

`.claude/settings.json` に登録された PreToolUse hook のコマンドを**そのまま**取り出して
subprocess で起動し、保護対象パスで deny が返ること・非保護パスが素通りすることを確認する。

この hook は fail-open 設計 (入力が読めない・インタプリタが起動できない → 何もせず素通り) の
ため、**インタプリタ指定が壊れると保護が静かに無効化される**。settings.json のコマンドを
読んで実行することで、その状態を失敗として検出する。

実行:
  python scripts/test_dollma_protect_hook.py
  あるいは: python -m pytest scripts/test_dollma_protect_hook.py -q
"""
import json
import os
import shlex
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SETTINGS = os.path.join(_ROOT, ".claude", "settings.json")

# 保護対象 (hook が deny を返さなければならない)
PROTECTED = [
    "data/bitnet/bitnet_dense.safetensors",
    "data/bitnet/bitnet_dense_fp32.safetensors",
    "data/bitnet/golden/logits_seq8.safetensors",
    "data/bitnet/pairs.train.jsonl",
    "data/bitnet/pairs.val.jsonl",
    "data/bitnet/pairs.eval_diverse_a.jsonl",
    "data/bitnet/pairs.eval_diverse_b.jsonl",
    # Windows 区切りの絶対パスでも効くこと (hook 内で / へ正規化している)
    "E:\\Projects\\dollama\\data\\bitnet\\bitnet_dense.safetensors",
]

# 非保護 (素通りしなければならない = 実験重み・docs・ソース)
ALLOWED = [
    "data/bitnet/bitnet_dense_kl.safetensors",
    "data/bitnet/bitnet_dense_diverse_b2000.safetensors",
    "data/bitnet/bitnet_dense_identity.safetensors",
    "data/bitnet/pairs.train.diverse_b2000.jsonl",
    "data/bitnet/pairs.identity.train.a12k.jsonl",
    "docs/roadmap.md",
    "src/main.cpp",
]


def _hook_command():
    """settings.json から PreToolUse hook のコマンドを取り出し、実行可能な argv にする。"""
    with open(_SETTINGS, encoding="utf-8") as f:
        settings = json.load(f)

    entries = settings.get("hooks", {}).get("PreToolUse", [])
    assert entries, "settings.json に PreToolUse hook が無い"

    cmd = None
    for entry in entries:
        for h in entry.get("hooks", []):
            c = h.get("command") or ""
            if "dollama_protect_artifacts.py" in c:
                cmd = c
                break
    assert cmd, "保護 hook (dollama_protect_artifacts.py) が登録されていない"

    # ${CLAUDE_PROJECT_DIR:-.} を実際のリポジトリルートへ展開する
    cmd = cmd.replace("${CLAUDE_PROJECT_DIR:-.}", _ROOT.replace("\\", "/"))
    cmd = cmd.replace("$CLAUDE_PROJECT_DIR", _ROOT.replace("\\", "/"))
    return shlex.split(cmd)


def _run_hook(file_path):
    """hook に PreToolUse 入力を渡し、(returncode, stdout) を返す。"""
    payload = json.dumps({"tool_input": {"file_path": file_path}})
    proc = subprocess.run(
        _hook_command(),
        input=payload,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return proc.returncode, (proc.stdout or "").strip(), (proc.stderr or "").strip()


def test_hook_is_runnable():
    """settings.json のインタプリタ指定で hook が実際に起動できること。

    ここが落ちるときは保護が fail-open で無効化されている (最も危険な状態)。
    """
    try:
        rc, _, err = _run_hook("docs/roadmap.md")
    except FileNotFoundError as e:
        raise AssertionError(
            "hook のインタプリタが起動できない (保護が無効化されている): %s" % e
        )
    assert rc == 0, "hook が異常終了した: rc=%d stderr=%s" % (rc, err)


def test_protected_paths_denied():
    bad = []
    for p in PROTECTED:
        rc, out, _ = _run_hook(p)
        if rc != 0:
            bad.append("%s: rc=%d" % (p, rc))
            continue
        if not out:
            bad.append("%s: deny が返らない (素通りした)" % p)
            continue
        try:
            data = json.loads(out)
        except ValueError:
            bad.append("%s: 出力が JSON でない: %r" % (p, out[:80]))
            continue
        decision = data.get("hookSpecificOutput", {}).get("permissionDecision")
        if decision != "deny":
            bad.append("%s: permissionDecision=%r" % (p, decision))
    assert not bad, "保護対象が deny されない: %s" % bad


def test_non_protected_paths_pass_through():
    bad = []
    for p in ALLOWED:
        rc, out, _ = _run_hook(p)
        if rc != 0:
            bad.append("%s: rc=%d" % (p, rc))
        elif out:
            bad.append("%s: 素通りすべきなのに出力があった: %r" % (p, out[:80]))
    assert not bad, "非保護パスがブロックされた: %s" % bad


def test_malformed_input_fails_open():
    """壊れた入力で例外終了せず素通りすること (ガードが全 Write を止めないため)。"""
    proc = subprocess.run(
        _hook_command(),
        input="not a json",
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.returncode == 0, "壊れた入力で異常終了した: rc=%d" % proc.returncode
    assert not (proc.stdout or "").strip(), "壊れた入力でブロックした (fail-open でない)"


def main():
    tests = [
        test_hook_is_runnable,
        test_protected_paths_denied,
        test_non_protected_paths_pass_through,
        test_malformed_input_fails_open,
    ]
    fails = 0
    for t in tests:
        try:
            t()
            print("[PASS] %s" % t.__name__)
        except AssertionError as e:
            fails += 1
            print("[FAIL] %s: %s" % (t.__name__, e))
    print("\n%d/%d passed" % (len(tests) - fails, len(tests)))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

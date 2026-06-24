#!/usr/bin/env python
# dollama 本番アーティファクト保護フック (PreToolUse・Write/Edit/MultiEdit)
#
# #1 本番重み / golden / 凍結 val への誤上書きをブロックし、別名で出すよう促す。
# CLAUDE.md・施策 C/D5/D6/B で繰り返した「本番は無改変・別名のみ」規律を自動強制する。
# 実験重み (bitnet_dense_kl / _d6 / _diverse_b / _identity / _distill / _smoke 等) や
# docs は対象外 (= ブロックしない)。
#
# 入力: PreToolUse の hook 入力 JSON を stdin で受ける (tool_input.file_path)。
# 出力: 保護対象なら permissionDecision=deny の JSON を返す。それ以外は無出力 (allow)。
# 失敗時は fail-open (何もしない) — ガードが誤って全 Write をブロックしないため。
import sys
import json
import re


def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        sys.exit(0)  # 入力が読めなければ素通り (fail-open)

    ti = data.get("tool_input") or {}
    fp = ti.get("file_path") or ti.get("path") or ""
    if not fp:
        sys.exit(0)

    p = fp.replace("\\", "/")

    protected = (
        # #1 本番重み (完全一致・サフィックス付き実験重みは弾かない)
        p.endswith("data/bitnet/bitnet_dense.safetensors")
        or p.endswith("data/bitnet/bitnet_dense_fp32.safetensors")
        # golden 配下すべて
        or "data/bitnet/golden/" in p
        # 凍結 val
        or p.endswith("data/bitnet/pairs.train.jsonl")
        or p.endswith("data/bitnet/pairs.val.jsonl")
        # diverse-val 凍結アンカー (_a / _b)
        or re.search(r"data/bitnet/pairs\.eval_diverse_[ab]\.jsonl$", p) is not None
    )

    if protected:
        reason = (
            "dollama 本番アーティファクト保護: このファイルは #1 本番重み / golden / "
            "凍結 val で、CLAUDE.md と施策 C/D5/D6/B の規律により『無改変・別名のみ』です。"
            "直接の上書きをブロックしました。実験は別名 "
            "(bitnet_dense_<suffix>.safetensors 等) へ出力してください。"
            f" (path={fp})"
        )
        out = {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        }
        # ensure_ascii=True (既定) で \uXXXX エスケープの純 ASCII を出力する。
        # Windows コンソールの既定コードページ (cp932) でも print が化けず、
        # Claude Code 側の JSON パーサが日本語へ正しく復元する。
        sys.stdout.write(json.dumps(out, ensure_ascii=True))
        sys.exit(0)

    sys.exit(0)


main()

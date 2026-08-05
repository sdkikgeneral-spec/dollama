# -*- coding: utf-8 -*-
"""dollama サブエージェント定義の健全性テスト (docs/testing.md 準拠)。

.claude/agents/*.md の frontmatter と本文を検証する。外部依存なし
(PyYAML 不要・frontmatter は最小自前パース)。定義に書かれたリポジトリ相対パスが
実在するかを機械的に確認し、「定義が実装から乖離する」陳腐化を回帰として検出する。

実行:
  python scripts/test_dollma_agent_defs.py
  あるいは: python -m pytest scripts/test_dollma_agent_defs.py -q
"""
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_AGENTS_DIR = os.path.join(_ROOT, ".claude", "agents")
_COMMON_REL = "docs/agent-common.md"

# 期待する 10 体 (案A: prompt-engineer 廃止 / pipeline-debugger→perf-profiler 置換)
EXPECTED_AGENTS = {
    "cpp-implementer",
    "csharp-ui-implementer",
    "cuda-kernel-dev",
    "dataset-curator",
    "gpu-benchmarker",
    "model-converter",
    "model-trainer",
    "npu-benchmarker",
    "perf-profiler",
    "project-leader",
}

# 廃止 (ファイルが残っていてはいけない)
RETIRED_AGENTS = {"prompt-engineer", "pipeline-debugger"}

# model に書いてよいのはエイリアスのみ (具体バージョンは世代交代で陳腐化する)
MODEL_ALIASES = {"opus", "sonnet", "haiku", "inherit"}

# model を明記してよい 3 体 (重い判断)。他は書かない = セッション追従
MODEL_PINNED = {"project-leader", "cpp-implementer", "cuda-kernel-dev"}

# 標準 tools 契約
TOOLS_STANDARD = {"Bash", "PowerShell", "Read", "Write", "Edit", "Glob", "Grep"}
TOOLS_BY_AGENT = {"project-leader": {"Read", "Glob", "Grep"}}

# 陳腐化・方針違反の語 (定義に残っていてはいけない)
FORBIDDEN_TERMS = {
    "LibTorch": "LibTorch は不使用が確定方針 (CLAUDE.md 実装方針)",
    "Winsock2": "HTTP は cpp-httplib + nlohmann/json に確定 (手書き Winsock2 は不採用)",
    "openvino.runtime": "openvino.runtime は廃止 API (import openvino as ov を使う)",
}

# 本文からリポジトリ相対パスを拾う (バッククォート内のみ・ワイルドカードは除外)
_TICK_RE = re.compile(r"`([^`\n]+)`")
_PATH_PREFIXES = (
    "src/", "docs/", "scripts/", "data/", "ui/", "ui.Tests/",
    ".claude/", "models/", "subprojects/",
)


def _agent_files():
    """.claude/agents/*.md の (stem, path) を返す。"""
    out = []
    for name in sorted(os.listdir(_AGENTS_DIR)):
        if name.endswith(".md"):
            out.append((name[:-3], os.path.join(_AGENTS_DIR, name)))
    return out


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def _parse_front_matter(text):
    """--- で囲まれた frontmatter を dict にする。

    tools はカンマ区切り 1 行 / YAML リスト (- Bash) の両形式を受け、set で返す。
    それ以外は文字列。frontmatter が無ければ None。
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end is None:
        return None

    fm = {}
    key = None
    list_items = []
    for raw in lines[1:end]:
        if raw.lstrip().startswith("- ") and key is not None:
            list_items.append(raw.lstrip()[2:].strip())
            continue
        if list_items and key is not None:
            fm[key] = set(list_items)
            list_items = []
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_-]*):\s*(.*)$", raw)
        if not m:
            continue
        key = m.group(1)
        val = m.group(2).strip()
        # コメント (# ...) を落とす
        val = re.sub(r"\s+#.*$", "", val).strip()
        if val == "":
            fm[key] = ""  # 後続がリストなら上書きされる
        else:
            fm[key] = val
    if list_items and key is not None:
        fm[key] = set(list_items)

    if isinstance(fm.get("tools"), str):
        fm["tools"] = {t.strip() for t in fm["tools"].split(",") if t.strip()}
    return fm


def _body(text):
    """frontmatter を除いた本文を返す。"""
    lines = text.splitlines()
    if lines and lines[0].strip() == "---":
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                return "\n".join(lines[i + 1:])
    return text


def _candidate_paths(body):
    """本文のバッククォート内から、リポジトリ相対パスに見えるものを拾う。"""
    found = []
    for tok in _TICK_RE.findall(body):
        tok = tok.strip()
        if "*" in tok or " " in tok:
            continue  # glob 表記・文章は対象外
        if "<" in tok or ">" in tok:
            continue  # test_<component>.cpp のようなプレースホルダ表記は対象外
        if not tok.startswith(_PATH_PREFIXES):
            continue
        found.append(tok)
    return found


# ----------------------------------------------------------------
# テスト
# ----------------------------------------------------------------
def test_expected_agents_present():
    stems = {s for s, _ in _agent_files()}
    missing = EXPECTED_AGENTS - stems
    assert not missing, "期待するエージェント定義が無い: %s" % sorted(missing)


def test_retired_agents_absent():
    stems = {s for s, _ in _agent_files()}
    left = RETIRED_AGENTS & stems
    assert not left, "廃止したはずの定義が残っている: %s" % sorted(left)


def test_front_matter_required_fields():
    bad = []
    for stem, path in _agent_files():
        fm = _parse_front_matter(_read(path))
        if fm is None:
            bad.append("%s: frontmatter が無い" % stem)
            continue
        for field in ("name", "description", "tools"):
            if not fm.get(field):
                bad.append("%s: %s が無い" % (stem, field))
    assert not bad, "frontmatter 不備: %s" % bad


def test_name_matches_filename():
    bad = []
    for stem, path in _agent_files():
        fm = _parse_front_matter(_read(path)) or {}
        if fm.get("name") != stem:
            bad.append("%s: name=%r" % (stem, fm.get("name")))
    assert not bad, "name とファイル名が不一致: %s" % bad


def test_tools_contract():
    bad = []
    for stem, path in _agent_files():
        fm = _parse_front_matter(_read(path)) or {}
        tools = fm.get("tools") or set()
        want = TOOLS_BY_AGENT.get(stem, TOOLS_STANDARD)
        if tools != want:
            bad.append("%s: tools=%s (期待 %s)" % (stem, sorted(tools), sorted(want)))
    assert not bad, "tools 契約違反: %s" % bad


def test_model_alias_only():
    bad = []
    for stem, path in _agent_files():
        fm = _parse_front_matter(_read(path)) or {}
        model = fm.get("model")
        if model is None or model == "":
            if stem in MODEL_PINNED:
                bad.append("%s: model 固定が要る (opus)" % stem)
            continue
        if stem not in MODEL_PINNED:
            bad.append("%s: model を書いてはいけない (inherit 運用)" % stem)
        if model not in MODEL_ALIASES:
            bad.append("%s: model=%r はエイリアスでない" % (stem, model))
    assert not bad, "model 指定違反: %s" % bad


def test_agent_common_exists_and_referenced():
    common = os.path.join(_ROOT, _COMMON_REL.replace("/", os.sep))
    assert os.path.isfile(common), "%s が無い" % _COMMON_REL
    bad = []
    for stem, path in _agent_files():
        if _COMMON_REL not in _read(path):
            bad.append(stem)
    assert not bad, "共通ルールを参照していない: %s" % bad


def test_referenced_paths_exist():
    bad = []
    for stem, path in _agent_files():
        for rel in _candidate_paths(_body(_read(path))):
            target = os.path.join(_ROOT, rel.replace("/", os.sep))
            if not (os.path.isfile(target) or os.path.isdir(target)):
                bad.append("%s: %s が実在しない" % (stem, rel))
    assert not bad, "存在しないパスを参照している: %s" % bad


def test_no_forbidden_terms():
    bad = []
    for stem, path in _agent_files():
        text = _read(path)
        for term, why in FORBIDDEN_TERMS.items():
            if term in text:
                bad.append("%s: %r (%s)" % (stem, term, why))
    assert not bad, "禁止語が残っている: %s" % bad


def main():
    tests = [
        test_expected_agents_present,
        test_retired_agents_absent,
        test_front_matter_required_fields,
        test_name_matches_filename,
        test_tools_contract,
        test_model_alias_only,
        test_agent_common_exists_and_referenced,
        test_referenced_paths_exist,
        test_no_forbidden_terms,
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

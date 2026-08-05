# サブエージェント定義 現状化 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `.claude/agents/*.md` の 11 体を現状 (Phase 2 完了・Phase 4 一巡・fast-mode 起票・二機体制) に合わせて 10 体へ改訂し、共通ルールを `docs/agent-common.md` 1 枚に集約する。

**Architecture:** 検証スクリプト `scripts/test_dollma_agent_defs.py` を先に書き、現状 11 体に対して**赤**にする (旧パス・廃止語・`Edit` 欠落・共通ルール未参照を検出)。その後 1 タスクずつ定義を改訂して緑にしていく。スクリプトは以降も回帰テストとして残り、「定義が実装から乖離する」陳腐化を機械的に検出し続ける。

**Tech Stack:** Markdown (frontmatter + 本文) / Python 3 標準ライブラリのみ (PyYAML 不使用・frontmatter は最小自前パース) / git

## Global Constraints

- 作業ブランチは **`chore/agent-defs-refresh`** (spec コミット `0b81c6d` の続き)。
- コミットは**対象ファイルのみ明示 stage**。`git add -A` は使わない。
- 定義の文章は**日本語**で書く。
- **可変値を定義に焼かない**: test 件数・データ件数・所要秒などは書かず `CLAUDE.md` / `docs/` を参照させる。書いてよい数値は**デバイス選定根拠として確定した HW 実測のみ** (例: CLIP-L NPU 7.85ms / WD14 NPU 268ms / ISNet iGPU 99.96ms)。
- `model:` に書くのは**エイリアスのみ** (`opus`)。具体バージョン (`claude-opus-5` 等) は禁止。
- `model:` を書くのは **`project-leader` / `cpp-implementer` / `cuda-kernel-dev` の 3 体だけ**。他は行ごと書かない。
- `tools:` は **`Bash, PowerShell, Read, Write, Edit, Glob, Grep`** (カンマ区切り 1 行)。**例外は `project-leader` のみ `Read, Glob, Grep`**。
- `CLAUDE.md` と重複する規約 (Allman・日本語コメント・使う/使わない表) は各定義に書かず `docs/agent-common.md` に集約し、各定義は 1 行で参照する。
- 各定義に**存在しないファイルパスを書かない** (検証スクリプトが弾く)。
- 禁止語: `LibTorch` / `Winsock2` / `openvino.runtime` (いずれも現行方針と矛盾)。
- 完了条件は毎タスク `python scripts/test_dollma_agent_defs.py` の該当項目が緑になること。最終タスクで全項目緑。

## File Structure

| ファイル | 責務 |
|---|---|
| `scripts/test_dollma_agent_defs.py` | **新規**。定義の健全性検証 (frontmatter・tools・model・パス実在・禁止語・共通ルール参照・10 体の在不在)。回帰テストとして常設 |
| `docs/agent-common.md` | **新規**。全エージェント共通の非交渉ルール 9 節。ここが唯一の保守点 |
| `.claude/agents/project-leader.md` | 改訂。具体タスク列を撤去し「現状 docs から判断」へ。10 体の一覧と二機体制の振り分け判断 |
| `.claude/agents/cpp-implementer.md` | 改訂。担当ツリー実態化・HTTP 訂正・Phase1 バグ列撤去・OV/CUDA 隔離作法 |
| `.claude/agents/cuda-kernel-dev.md` | 改訂。完成済みカーネル一覧・fast-mode G-0〜G-6k 主タスク化・cuBLAS 方針訂正・研究機専用 |
| `.claude/agents/model-trainer.md` | 改訂。訓練機 (RTX5080 主/1080Ti 併記)・BPE 撤去・Phase4 レシピ・正典無改変・NAS 搬送 |
| `.claude/agents/dataset-curator.md` | 改訂。§12〜§19 現状化・凍結ファイル明記・**日本語→タグ変換表の引き取り** |
| `.claude/agents/model-converter.md` | 改訂。変換対象実態化・落とし穴 (MHA fastpath / eval() BN / 静的形状) |
| `.claude/agents/npu-benchmarker.md` | 改訂。確定値更新・LibTorch 削除・研究機専用・純 conv vs Window Attention の切り分け |
| `.claude/agents/gpu-benchmarker.md` | 改訂。担当実態化 (SDXL 実走 / rollout / reward 採点 / 電力診断)・SAC 制約・研究機専用 |
| `.claude/agents/perf-profiler.md` | **新規**。`src/infer/profile.cuh` 基盤・occupancy/latency 律速診断・計測クローズ事実 |
| `.claude/agents/csharp-ui-implementer.md` | 改訂 (軽微)。ui/ 実態と既存挙動 (ドラフト/サムネ) 反映 |
| `.claude/agents/pipeline-debugger.md` | **削除** (perf-profiler が置換) |
| `.claude/agents/prompt-engineer.md` | **削除** (変換表は dataset-curator へ移設) |

---

### Task 1: 検証スクリプト (テスト先行・現状で赤)

**Files:**
- Create: `scripts/test_dollma_agent_defs.py`

**Interfaces:**
- Consumes: なし (最初のタスク)
- Produces: 以降の全タスクの合否判定。テスト関数名は
  `test_expected_agents_present` / `test_retired_agents_absent` /
  `test_front_matter_required_fields` / `test_name_matches_filename` /
  `test_tools_contract` / `test_model_alias_only` /
  `test_agent_common_exists_and_referenced` / `test_referenced_paths_exist` /
  `test_no_forbidden_terms`。`main()` は失敗数を stderr に出し、非ゼロ終了する。

- [ ] **Step 1: 検証スクリプトを書く**

```python
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
```

- [ ] **Step 2: 走らせて赤を確認**

Run: `python scripts/test_dollma_agent_defs.py`

Expected: 非ゼロ終了。少なくとも以下が FAIL する (現状 11 体の実態):
`test_expected_agents_present` (perf-profiler が無い) /
`test_retired_agents_absent` (prompt-engineer・pipeline-debugger がある) /
`test_tools_contract` (Edit・PowerShell が無い) /
`test_model_alias_only` (3 体の model 固定が無い) /
`test_agent_common_exists_and_referenced` (`docs/agent-common.md` が無い) /
`test_no_forbidden_terms` (LibTorch・Winsock2 が残っている)

- [ ] **Step 3: コミット**

```bash
git add scripts/test_dollma_agent_defs.py
git commit -m "test(agents): サブエージェント定義の健全性テストを追加 (現状は赤)"
```

---

### Task 2: 共通ルール 1 枚 (`docs/agent-common.md`)

**Files:**
- Create: `docs/agent-common.md`

**Interfaces:**
- Consumes: Task 1 の `test_agent_common_exists_and_referenced` (ファイル存在部分が緑になる)
- Produces: 以降の全定義が末尾で参照する唯一の共通ルール。参照文言は
  **`共通ルール (二機体制・規約・テスト必須・正典保護・搬送・SAC・docs 分担) は docs/agent-common.md を読む。`** で統一する。

- [ ] **Step 1: `docs/agent-common.md` を書く**

以下 9 節を必ず含める。文章は日本語。各節の中身は指定の事実を漏らさず書く。

1. **走る機械の判定** — 開発機 = GTX1080Ti (sm_61・FP16 native 非対応) / i7-10700 / NPU なし / nvcc なし / `build` は `with_cuda=false, with_openvino=false`。研究機 = Core Ultra 9 285 (NPU = AI Boost) + Intel Xe iGPU + RTX5080 (sm_120)。判定手段は `nvidia-smi --query-gpu=name,compute_cap --format=csv`、`nvcc --version`、`build/meson-info/intro-buildoptions.json` の 3 つ。**研究機必須のタスクを開発機で振られたら、スクリプト著述・純ホストのビルド確認までで止めて「実走は研究機」と報告する** (勝手に代替 HW で回さない)。
2. **コーディング規約** — Allman (開き波括弧は改行)・`switch` の `case` は `switch` と同じインデント・コメントは日本語 (C++/C#/Python 共通)・`dollma_` プレフィックスは `scripts/` のみ。
3. **テスト必須** — CLAUDE.md ルール4。C++ は `src/tests/test_<component>.cpp` を作り `meson test -C build` が緑。C# は `ui.Tests` で `dotnet test` が緑。Python は `scripts/test_dollma_*.py` (関数 `test_*` + `if __name__ == "__main__"` 自走・pytest でも動く形)。規約は `docs/testing.md`。
4. **正典アーティファクト保護** — `.claude/hooks/dollama_protect_artifacts.py` が Write/Edit を deny する対象: `data/bitnet/bitnet_dense.safetensors` / `data/bitnet/bitnet_dense_fp32.safetensors` / `data/bitnet/golden/` 配下 / `data/bitnet/pairs.train.jsonl` / `data/bitnet/pairs.val.jsonl` / `data/bitnet/pairs.eval_diverse_a.jsonl` / `data/bitnet/pairs.eval_diverse_b.jsonl`。実験は必ず別名 (`bitnet_dense_<suffix>.safetensors`) へ出す。正典差し替えはユーザー決裁を経た「まとめ焼き」のときだけ。
5. **重み・golden の搬送** — 正典重み / golden / a12k は gitignore。マシン間は `scripts/train_bitnet.py` の `--copy` / `--publish` で NAS 経由。**cross-GPU 再生成は bit 非一致ゆえ exact コピーが必須** (再訓練で代用しない)。
6. **研究機の SAC 制約** — 再ビルドした exe の新ハッシュがブロックされる。カーネルを直して `dollama.exe` を実走で緑にするには allow-list 更新をユーザーへ依頼する。依頼できないときは「開発機ビルド緑 + Python/OV 経由の数値検証」で回す。
7. **実装方針 (使う / 使わない)** — 使う: STL 全般 / CUDA Runtime API / 自作 Tensor・GEMM・Attention / cpp-httplib (HTTP) / nlohmann/json / OpenVINO C++ API (NPU・iGPU 推論グルー)。使わない: PyTorch・LibTorch / diffusers・stable-diffusion.cpp / llama.cpp / 重量級 HTTP フレームワーク / 手書き JSON パーサ。**cuBLAS/cuDNN は「到達困難になった重い GEMM/Conv のみフォールバック許容」** (自作版に後で置換可能な形で入れる)。
8. **docs 更新の分担** — `CLAUDE.md` = 芯の数値と確定事項のみ / `docs/measurements-log.md` = 計測の全文 / `docs/roadmap.md` = 経緯と採否 / `docs/training-spec.md`・`docs/dataset-spec.md` = 手順の詳細 / `docs/testing.md` = テスト規約。作業で新しい数値が出たら該当 docs に追記し、CLAUDE.md には芯だけ足す。
9. **報告フォーマット** — 完了時に「何を・どの機械で・どの数値で・どの test が緑か」を返す。未達・スキップがあれば明示する (緑でないものを緑と言わない)。

- [ ] **Step 2: 参照だけ緑になることを確認**

Run: `python scripts/test_dollma_agent_defs.py`

Expected: `test_agent_common_exists_and_referenced` は依然 FAIL (定義側が未参照)。ただしメッセージが「`docs/agent-common.md` が無い」から「共通ルールを参照していない: [11 体]」に変わる。

- [ ] **Step 3: コミット**

```bash
git add docs/agent-common.md
git commit -m "docs(agents): 全エージェント共通の非交渉ルールを 1 枚に集約"
```

---

### Task 3: `project-leader` 改訂 (最重要・以降の見本)

**Files:**
- Modify: `.claude/agents/project-leader.md` (全面置換)

**Interfaces:**
- Consumes: `docs/agent-common.md` (Task 2)
- Produces: 以降 7 体が踏襲する定義の書式 (frontmatter + 本文 6 節)。

**なぜ最初か:** 「次に着手すべきタスク 1〜7」が全完了済みのまま残っており、PL がそれを読むと完了済みタスクへ誘導する。害が最大。

- [ ] **Step 1: 全面置換で書く**

```markdown
---
name: project-leader
description: dollama プロジェクト全体のタスク分割・進捗管理・エージェント間調整を担当する。コーディングはせず、何を誰にやらせるかを決める。「次に何をすべきか」「どのエージェントに頼むか」「このタスクはどちらの機械で回すか」を判断するときに使う。
tools: Read, Glob, Grep
model: opus
---

あなたは dollama プロジェクトのプロジェクトリーダー (PL) です。
**コードは書かない。** タスクの分割・優先付け・エージェントへの委譲指示が役割です。

## 役割と境界

- やる: タスク分割 / 優先付け / 担当エージェントの選定 / 二機のどちらで回すかの判断 / DoD の明文化 / 進捗の突合。
- やらない: 実装・計測の実走・ドキュメントの本文著述 (すべて担当エージェントへ渡す)。

## 承認権限

ゴールが設定された場合、プランの承認は PL が行う。ユーザーへの判断依頼は **PL が迷ったときのみ**。
方針が CLAUDE.md の確定事項と矛盾しない限り、自律的に判断して先に進める。

## 現状把握のしかた (ここに具体タスクを書かない)

**「次に着手すべきタスク」を本ファイルに列挙しない。** 焼き込むと完了後に腐り、完了済みタスクへ
誘導する事故が起きる (実際に起きた)。指示を出す前に必ず次を読んで現状から判断する。

| 読むもの | 何が分かるか |
|---|---|
| `CLAUDE.md` | 芯の確定事項・計測ベースライン・「次のタスク」節 |
| `docs/roadmap.md` | Phase ごとの段・状態・採否の経緯 |
| `docs/measurements-log.md` | 計測の全文 (芯以外の数値) |
| `docs/fast-mode-plan.md` | 自作カーネル高速化の分割タスク台帳 (G-0〜G-6k) |
| `docs/f0b-rejection-sft-plan.md` / `docs/q2-quality-branch-plan.md` | Phase 4 F/Q の結論と次レバー |
| `docs/training-spec.md` / `docs/dataset-spec.md` | 訓練・データの手順と確定レシピ |
| `git log --oneline -20` | 直近で何が動いたか |

## 専門エージェントと担当領域

| エージェント | 担当 | 走る機械 |
|---|---|---|
| `cpp-implementer` | `src/` の C++ (core/io/infer/server/models/tests)・Meson | 両機 (OV/CUDA 無効ビルドは開発機可) |
| `cuda-kernel-dev` | `src/kernels/*.cu` と `src/infer/*.cu` の CUDA | **研究機のみ** (開発機に nvcc なし) |
| `csharp-ui-implementer` | `ui/` (Blazor Server) と `ui.Tests` | 両機 |
| `dataset-curator` | `data/bitnet/` のデータセット構築・語彙・分割 | 開発機で完結 |
| `model-trainer` | PyTorch 訓練・蒸留・sweep (`scripts/train_*.py`) | 研究機主・開発機も可 (FP32) |
| `model-converter` | PyTorch/ONNX → OpenVINO IR 変換・量子化 | **研究機のみ** |
| `npu-benchmarker` | NPU / iGPU / CPU の推論計測とデバイス選定 | **研究機のみ** |
| `gpu-benchmarker` | RTX5080 実走 (SDXL 生成・rollout 収集・reward 採点・GPU golden) | **研究機のみ** |
| `perf-profiler` | 拡散パイプラインの律速内訳・occupancy/電力診断 | **研究機のみ** (計装の著述は両機) |

## タスク分割の原則

- 1 タスク = 1 エージェント。複数 HW をまたぐ場合は分割する。
- 実装タスクは「どのファイル・どのクラス・どの機能」を明示して渡す。
- **どちらの機械で回すかを必ず指定する。** 研究機必須のタスクを開発機セッションで振らない
  (振る場合は「著述まで」と明示する)。
- 各タスクに DoD (何が緑なら完了か) を書く。テスト実装は必須 (CLAUDE.md ルール4)。
- 完了報告には docs への追記 (どの docs か) を含めさせる。

## 判断基準 (確定済み・再調査させない)

- ゼロコピー CUDA↔NPU は不可。CPU pinned memory 経由で確定 (再調査は却下)。
- iGPU に大規模 Conv モデルを割り当てる提案は却下 (VAE decode stub で CPU の 8 倍遅い)。
- LLM (自己回帰) を NPU に乗せる提案は却下 (KV-cache で形状が動的)。
- WD14 は CPU 採用 (NPU 268ms は Window Attention 由来)。CLIP-L は NPU 採用 (7.85ms)。
- マッティング (ISNet) は iGPU 採用 (99.96ms)。
- 純 conv は NPU フレンドリー (ScorerNet が NPU に載る)。attention head を足す提案は NPU 不利に戻す。

## 完了条件 (DoD)

指示を出す前に「担当エージェント / 機械 / 触るファイル / 何が緑なら完了か / どの docs に追記するか」の
5 点が埋まっていること。埋まらないなら情報が足りないので調べてから出す。

共通ルール (二機体制・規約・テスト必須・正典保護・搬送・SAC・docs 分担) は docs/agent-common.md を読む。
```

- [ ] **Step 2: 検証**

Run: `python scripts/test_dollma_agent_defs.py`

Expected: `test_name_matches_filename` は緑。`test_tools_contract` の FAIL メッセージから
`project-leader` が消える (他 10 体は残る)。`test_referenced_paths_exist` に
`project-leader` の行が出ないこと (本文のパスは全て実在させたので)。

- [ ] **Step 3: コミット**

```bash
git add .claude/agents/project-leader.md
git commit -m "chore(agents): project-leader を現状化 (具体タスク列を撤去し現状 docs 参照へ)"
```

---

### Task 4: `cpp-implementer` + `cuda-kernel-dev` 改訂

**Files:**
- Modify: `.claude/agents/cpp-implementer.md` (全面置換)
- Modify: `.claude/agents/cuda-kernel-dev.md` (全面置換)

**Interfaces:**
- Consumes: Task 3 の書式 (frontmatter + 6 節 + 末尾の共通ルール参照 1 行)
- Produces: なし (以降のタスクと独立)

- [ ] **Step 1: `cpp-implementer.md` を書く**

frontmatter: `tools: Bash, PowerShell, Read, Write, Edit, Glob, Grep` / `model: opus` /
description は「`src/` の C++ 実装 (core/io/infer/server/models) と Meson ビルド。CUDA `.cu` は cuda-kernel-dev、`ui/` は csharp-ui-implementer」と境界を明記。

**必ず書く:**
- 担当ツリー (実在するもののみ):
  `src/core/` = `tensor.hpp` `allocator.hpp` `queue.hpp` `character.hpp` `affinity.hpp` `cpu_topology.hpp` `multi_frame_pipeline.hpp` /
  `src/io/` = `safetensors.hpp` `tokenizer.hpp` `png_meta.hpp` /
  `src/infer/` = `clip.hpp` `clip_encoder2.hpp` `clip_tokenizer.hpp` `wd14.hpp` `text_conditioner.hpp` `scheduler.hpp` `bitnet.hpp` `bitnet_int8.hpp` `quality_gate.hpp` `quality_scorer.hpp` `matting.hpp` (`.cu`/`.cuh` は cuda-kernel-dev と共同) /
  `src/server/` = `api.cpp` `generator.hpp` `diffusion_backend.hpp` `sdxl_backend.hpp` `sd35_backend.hpp` `backend_image_generator.hpp` `cli_generate.hpp` `matting_postprocess.hpp` `scoring_postprocess.hpp` `png.hpp` `base64.hpp` ほか /
  `src/models/` = `bitnet.hpp` `bitnet_config.hpp` / `src/tests/` / `src/main.cpp` / `src/pipeline.hpp` / `src/meson.build`
- **HTTP は cpp-httplib (単一ヘッダ) + nlohmann/json**。OpenAI Images 互換は `src/server/api.cpp`。仕様は `docs/http-api-spec.md`。
- **OV/CUDA 隔離の作法 (最重要の固有知識)**: 純 cpp interface を境界に置き (`IDiffusionBackend` / `IDiffusionRunner` / `IImageGenerator` / `IMatter` / `IScorer`)、実装を `*_runner.cu` (実) と `*_runner_stub.cpp` (OV/CUDA 無効ビルド用) の二重で持つ。`nullptr` 契約でフォールバック段へ落とす。CUDA を含むヘッダは `#ifndef __CUDACC__` で nvcc 非汚染を保つ。
- **開発機では `with_cuda=false, with_openvino=false` でビルドする** (stub 経路が緑であることを確認する)。研究機のみ実経路。
- 新規ファイルは `src/meson.build` の sources / test 定義に追記する。
- 例外は CUDA/OV エラーのみ (`std::runtime_error`)。

**必ず削る:**
- 「Winsock2 OpenAI 互換 HTTP サーバー」等の Winsock2 記述 (禁止語)
- 「既知のバグ (修正待ち) 1〜6」の全列 (Phase 1 当時のもの)
- Allman / switch のコード例 (共通ルールへ集約)
- 「使う/使わない」表 (共通ルールへ集約)

- [ ] **Step 2: `cuda-kernel-dev.md` を書く**

frontmatter: 標準 tools / `model: opus` / description は「`src/kernels/*.cu` と `src/infer/*.cu` の CUDA カーネル実装・高速化。**研究機 (RTX5080 sm_120) 専用**」と明記。

**必ず書く:**
- ターゲット: RTX5080 (Blackwell / sm_120)・`-arch=sm_120`・VRAM 16GB。**開発機に nvcc は無く `.cu` はコンパイルできない** → 開発機で振られたらコード著述までで止めて報告する。
- 完成済みカーネル (実在): `src/kernels/` = `gemm.cu` `activation.cu` `groupnorm.cu` `conv2d.cu` `attention.cu` `layernorm.cu` `geglu.cu` `bias_add.cu` `elementwise.cu` `timeembed.cu` `vae_decode.cu` `utils.cuh`。`src/infer/` = `unet.cu` `diffusion.cu` `bitnet_gpu.cu`。
- **主タスクは fast-mode (`docs/fast-mode-plan.md` の G-0〜G-6k)**: default は無改変 (golden 回帰アンカー)・高速化は `--fast` / `--fast --fp8` に隔離。#1 ループ GPU 常駐 + CUDA Graphs / #2 CFG batch=2 / #4 attention multi-warp + cp.async / #5 epilogue 融合 / #3 FP8 は最内 opt-in。
- **golden 非交渉**: default 経路の UNet noise_pred SSIM と VAE SSIM のアンカーを割らない。`--fast` は再ベースライン後に SSIM ゲート、`--fp8` は golden を使わず知覚等価 + reward 順位相関で判定。
- cuBLAS/cuDNN は**到達困難になった重い GEMM/Conv のみフォールバック許容** (自作に置換可能な形で入れる)。Attention・正規化・活性化は自作を維持。CUTLASS 置換は不採用。
- ternary GEMM (`src/kernels/ternary_gemm.cu` は未実装) は**圧縮実験に降格**しており本線ではない。着手はユーザー決裁後。
- 計測は `cudaEvent_t` かつ warmup 3 / 中央値。VRAM は `cudaMemGetInfo`。
- 新規 `.cu` は `src/meson.build` に追記。CUDA API は戻り値を必ずチェック (`CUDA_CHECK`)。

**必ず削る:**
- ternary GEMM を主役に据えた記述と `unet_attention.cu` / `vae_decode.cu` の「(将来)」表記
- 「cuBLAS は使わない」の断定
- Allman / switch / `CUDA_CHECK` のコード例 (前者 2 つは共通ルール・後者は `src/kernels/utils.cuh` を参照させる)
- Meson の CUDA ビルド例 (実物は `src/meson.build`)

- [ ] **Step 3: 検証**

Run: `python scripts/test_dollma_agent_defs.py`

Expected: `test_no_forbidden_terms` から `cpp-implementer` の Winsock2 が消える。
`test_referenced_paths_exist` に両者の行が出ない。

- [ ] **Step 4: コミット**

```bash
git add .claude/agents/cpp-implementer.md .claude/agents/cuda-kernel-dev.md
git commit -m "chore(agents): cpp-implementer/cuda-kernel-dev を現状化 (HTTP 訂正・fast-mode 主タスク化)"
```

---

### Task 5: `model-trainer` + `dataset-curator` 改訂 (変換表の移設含む)

**Files:**
- Modify: `.claude/agents/model-trainer.md` (全面置換)
- Modify: `.claude/agents/dataset-curator.md` (全面置換)
- Read: `.claude/agents/prompt-engineer.md` (移設元・削除は Task 7)

**Interfaces:**
- Consumes: Task 3 の書式
- Produces: `dataset-curator` が日本語→danbooru タグ変換表の所有者になる (Task 7 で `prompt-engineer` を削除できる前提)

- [ ] **Step 1: `model-trainer.md` を書く**

frontmatter: 標準 tools / model 行なし / description は「PyTorch 訓練・蒸留・seed sweep。データセット構築は dataset-curator、OV 変換は model-converter」と境界明記。

**必ず書く:**
- **訓練機**: 主は研究機 RTX5080 (cu128・AMP/FP16 可)。**開発機 GTX1080Ti でも 33M 級は回る**が sm_61 で FP16 native 非対応ゆえ **FP32 固定** (`docs/training-spec.md` の規約)。開発機で焼いた重みを研究機で使う/その逆は **NAS 経由の exact コピー** (`--copy` / `--publish`)。cross-GPU 再生成は bit 非一致。
- 担当スクリプト (実在): `scripts/train_bitnet.py` `scripts/train_scorer.py` `scripts/dollma_train_quality_mlp.py`、sweep 系 `scripts/dollma_a_seedsweep.py` `scripts/dollma_b2000_seedsweep.py` `scripts/dollma_b10k_seedsweep.py` `scripts/dollma_d_seedsweep.py` (+ `*_analyze.py`)、評価 `scripts/dollma_make_eval_diverse.py`。
- `train_bitnet.py` の主要フラグ: `--train-file` / `--identity` / `--arch` / `--sft-rejection` / `--distill-kl` / `--distill-ext` / `--copy` / `--publish`。
- **確定レシピ**: 入力多様化 (tags-stay-real) が既定。正典は「33M で b2000 多様化 ∧ a12k identity のまとめ焼き」。identity retention は ~0.98 が床。主指標は **diverse 生成 set-F1** (recall@10 は退役)。
- **効果が確定した/しなかった軸** (再試行させないため): 施策 B (入力多様化) のみが diverse-F1 を頑健に上げ **~2,000 件で飽和**。施策 A は retention 専 (F1 非寄与)。施策 D (容量 33M→80M) は陰性で 80M 不採用。蒸留 4 路線 (D2/D4/D5/D6) は全て非寄与 (効果は過学習抑制のみ)。F-0b RAFT-SFT は不採用 (reward +0.017 に対し set-F1 が構造的に退行)。
- **正典無改変**: 実験は必ず別名。正典差し替えはユーザー決裁の「まとめ焼き」時のみ。golden 再生成は同時に行う。
- seed sweep の作法: 4 seed paired・eval が律速 (`--eval-only` の再利用)・`_results/*.npz` 存在で冪等 skip。
- 判断が要る設計 (規模・量子化方式・データ源) は `project-leader` に確認する。

**必ず削る:**
- 「訓練は RTX5080。NPU/iGPU は推論専用」だけの記述 → 上記の二機併記へ
- 「BPE トークナイザー学習」(タグ単位完全一致で確定・BPE は別タスクへ切離)
- ternary を主題に据えた節 (圧縮実験に降格)
- 教師を Qwen2 前提とする蒸留節 (4 路線が不採用で決着済み)

- [ ] **Step 2: `dataset-curator.md` を書く**

frontmatter: 標準 tools / model 行なし / description は「`data/bitnet/` のデータセット構築・語彙・分割・**タグ語彙規約 (日本語→danbooru タグ写像を含む)**。訓練は model-trainer」。

**必ず書く:**
- 走る機械: 開発機で完結 (GPU/NPU をほぼ使わない)。
- 担当スクリプト (実在): `scripts/dollma_build_vocab.py` `scripts/dollma_make_pairs.py` `scripts/dollma_make_diverse_train.py` `scripts/dollma_make_identity_pairs.py` `scripts/dollma_make_eval_diverse.py`、検証 `scripts/test_dollma_eval_diverse.py` `scripts/test_dollma_eval_setmetrics.py`。
- 現状のデータ資産 (実在): `data/bitnet/vocab.json` / `data/bitnet/pairs.train.jsonl` / `data/bitnet/pairs.val.jsonl` / `data/bitnet/pairs.train.diverse_b2000.jsonl` / `data/bitnet/pairs.identity.train.a12k.jsonl` / `data/bitnet/pairs.eval_diverse_a.jsonl` / `data/bitnet/pairs.eval_diverse_b.jsonl`。仕様は `docs/dataset-spec.md`。
- **凍結アンカー (hook が Write を deny する)**: `pairs.train.jsonl` / `pairs.val.jsonl` / `pairs.eval_diverse_a.jsonl` / `pairs.eval_diverse_b.jsonl`。加算的に新ファイルを作る。
- **tags-stay-real 原則**: 自然文だけ多様化し、タグは実 danbooru のまま (LLM にタグを推測させない)。
- 法務: タグは事実ラベル・画像ピクセルは収集しない。ToS を尊重。規模/出典に不確実性があれば PL に確認。
- **日本語→danbooru タグ写像の規約 (`prompt-engineer` から移設)**: ツインテール=`twintails` / 魔法少女=`magical girl` / 猫耳=`cat ears, nekomimi` / 獣耳=`animal ears` / 夕焼け=`sunset` / 桜=`cherry blossoms` / 水着=`swimsuit, bikini` / 制服=`school uniform, serafuku` / 着物=`kimono, japanese clothes` / 涙=`tears, crying`。写像は語彙規約の一部として `docs/dataset-spec.md` 側に持ち、`src/core/character.hpp` の `compose_prompt` が吐くタグ形式と矛盾させない。
- 既知の未解決: **日本語入力は現行トークナイザで語彙外となり空条件化する** (F-0b で 400 入力中 184 件)。日本語対応は語彙設計を伴う別タスクで、勝手に vocab を拡張しない。
- CharacterBible との整合: `docs/character-bible-spec.md` の同一性層/シーン層と `src/core/character.hpp` の出力形式に合わせる。

**必ず削る:**
- 「Phase 4 #1 を担当」という段番号への固定 (現状は §12〜§19 まで進んでいる)
- 自然文側を Qwen2-1.5B 蒸留で作る前提 (現在は多様化著述が主)

- [ ] **Step 3: 検証**

Run: `python scripts/test_dollma_agent_defs.py`

Expected: `test_referenced_paths_exist` に両者の行が出ない (挙げた全パスが実在)。

- [ ] **Step 4: コミット**

```bash
git add .claude/agents/model-trainer.md .claude/agents/dataset-curator.md
git commit -m "chore(agents): model-trainer/dataset-curator を現状化 (二機併記・確定レシピ・タグ写像移設)"
```

---

### Task 6: 研究機 3 体 (`model-converter` / `npu-benchmarker` / `gpu-benchmarker`) 改訂

**Files:**
- Modify: `.claude/agents/model-converter.md` (全面置換)
- Modify: `.claude/agents/npu-benchmarker.md` (全面置換)
- Modify: `.claude/agents/gpu-benchmarker.md` (全面置換)

**Interfaces:**
- Consumes: Task 3 の書式
- Produces: なし

- [ ] **Step 1: `model-converter.md` を書く**

frontmatter: 標準 tools / model 行なし / description に「**研究機専用**」を明記。

**必ず書く:**
- 変換対象 (実績): ISNet-anime (マッティング・`scripts/dollma_convert_matting.py`) / ScorerNet (`scripts/dollma_convert_scorer.py`) / QualityMLP (`scripts/dollma_convert_quality_mlp.py`) / SDXL text encoders CLIP-L・bigG (`scripts/dollma_convert_sdxl_text_encoders.py`) / CLIP ViT-L image encoder。
- `models/` の実構成: `models/isnet-anime` (開発機にも配置あり)、研究機には `models/clip-image` `models/quality-mlp` 等 (gitignore)。
- **NPU は静的形状のみ** → `compile_model` 前に必ず `reshape`。`ov.convert_model` / ONNX 読み込みは既定で動的形状。
- **落とし穴 (固有知識)**: ① CLIP image tower は **MHA fastpath を無効化しないと変換に失敗する** ② BatchNorm を持つモデルは `eval()` を忘れると OV 出力がずれる ③ 変換後は PyTorch↔OV の数値差を必ず出す (err 1e-5 オーダーを目安) ④ NPU / iGPU / CPU の 3 者で計測してから採用デバイスを決める。
- API は `import openvino as ov` (旧 API 名は使わない)。
- 完了時に変換時間・IR サイズ・数値差・デバイス別レイテンシを報告し、`docs/measurements-log.md` に追記案を出す。

**必ず削る:** Qwen2 を NPU 変換対象として論じる節 (自己回帰は NPU 不可で決着)・「Aesthetic scorer 検討中」・存在しない `models/` サブディレクトリ列。

- [ ] **Step 2: `npu-benchmarker.md` を書く**

frontmatter: 標準 tools / model 行なし / description に「**研究機専用**」。

**必ず書く:**
- 環境: Core Ultra 9 285 の NPU (AI Boost・DEVICE_ARCHITECTURE 3720) / Intel Xe iGPU は OpenVINO の `GPU.0` / RTX5080 は `GPU.1` / `import openvino as ov`。
- **確定したデバイス選定 (再調査させない)**: CLIP-L text = NPU **7.85ms** (CPU 20 / iGPU 14) / WD14 SwinV2 = CPU **101ms** (NPU 268ms は Window Attention 由来) / ScorerNet (純 conv) = NPU **8.32ms** で採用 / QualityMLP = NPU **0.553ms** / CLIP image tower = NPU **85.55ms** で最速 / ISNet マッティング = **iGPU 99.96ms** で採用 / VAE decode は iGPU 不適 (stub で CPU の 8 倍遅い)。
- **切り分けの結論**: 純 conv は NPU フレンドリー。Window Attention / 自己回帰は NPU 不向き。新モデルを NPU に載せる可否はこの軸で見立てる。
- 計測作法: warmup 3 / 中央値 n=20 / ms 表示 / NPU・iGPU・CPU の 3 者比較。probe は `scripts/dollma_probe*.py` 命名。
- よくあるエラー: `Missing upper bound` → 静的形状未設定。動的形状のまま compile → `reshape` を入れる。
- 結果は `docs/measurements-log.md` へ、芯だけ `CLAUDE.md` へ。

**必ず削る:** 「本実装は C++ + LibTorch」(禁止語)・「Aesthetic scorer 小型 MLP (検討中)」・「現在は調査フェーズ」。

- [ ] **Step 3: `gpu-benchmarker.md` を書く**

frontmatter: 標準 tools / model 行なし / description に「RTX5080 実走 (SDXL 生成・rollout 収集・reward 採点・GPU golden 再確認)。**研究機専用**」。

**必ず書く:**
- 環境: RTX5080 (sm_120 / CUDA 12.8+ / VRAM 16GB) / PyTorch cu128 / `dollama.exe` の実走。
- 担当の実務 (実在スクリプト): `scripts/dollma_rollout_bestofn.py` (best-of-N・resume/chunk 対応) / `scripts/dollma_collect_rollouts.py` / `scripts/dollma_reward.py` / `scripts/dollma_score_quality_v4.py` / `scripts/dollma_g2b_reward_prepost.py` / `scripts/dollma_label_image.py` / `scripts/dollma_gen_scorer_corpus.py` / golden 生成 `scripts/dollma_dump_unet_golden.py` `scripts/dollma_dump_vae_golden.py` `scripts/dollma_dump_txt2img_golden.py`。
- ベースライン: diffusers SDXL 20step 1024² = 3.80s (参照上限)。自作 `dollama.exe` は実 checkpoint + CFG で **19.5s/枚**。
- **GPU 稼働の読み方 (固有知識)**: `nvidia-smi` の `sm%` は「1 warp でも動いた時間割合」で満杯率ではない。**真の指標は消費電力** (154W/360W ≒ 43% なら SM が埋まっていない)。帯域 11% なら帯域律速でもない → occupancy/latency 律速と判定する。
- **SAC 制約**: 再ビルドした exe の新ハッシュがブロックされる。実走が要るなら allow-list 更新をユーザーへ依頼する。依頼できなければ既存 exe + Python/OV 検証で回す。
- 長時間ジョブは resume 可能な形で回し、`data/rollouts/` を truncate しない (append・完了 id skip)。
- OOM 時は attention slicing / sequential offload を提案。iGPU に大規模モデルを割り当てる提案はしない。

**必ず削る:** 「本実装は C++ + LibTorch で行う予定」(禁止語)・「現在は調査フェーズ」・iGPU VAE stub の詳細 (npu-benchmarker 側に一本化)。

- [ ] **Step 4: 検証**

Run: `python scripts/test_dollma_agent_defs.py`

Expected: `test_no_forbidden_terms` が緑になる (LibTorch が全滅)。

- [ ] **Step 5: コミット**

```bash
git add .claude/agents/model-converter.md .claude/agents/npu-benchmarker.md .claude/agents/gpu-benchmarker.md
git commit -m "chore(agents): 研究機 3 体を現状化 (確定デバイス選定・実走担当・SAC 制約)"
```

---

### Task 7: `perf-profiler` 新規 + 廃止 2 体の削除

**Files:**
- Create: `.claude/agents/perf-profiler.md`
- Delete: `.claude/agents/pipeline-debugger.md`
- Delete: `.claude/agents/prompt-engineer.md`

**Interfaces:**
- Consumes: Task 5 (`dataset-curator` が変換表を引き取り済み = `prompt-engineer` を消せる)
- Produces: `test_expected_agents_present` / `test_retired_agents_absent` が緑になる

- [ ] **Step 1: `perf-profiler.md` を書く**

frontmatter: 標準 tools / model 行なし / description は「拡散パイプラインの律速内訳と GPU 稼働の診断 (計装・profile 実行・ボトルネック特定)。カーネル改修は cuda-kernel-dev、実走計測は gpu-benchmarker。**profile 実行は研究機**」。

**必ず書く:**
- 計時基盤は既に実装済み: `src/infer/profile.cuh` (`profile_enabled()` / `ProfileCounters` / `ScopedSyncTimer`)。環境変数 **`DOLLAMA_PROFILE`** が立っているときだけ有効・既定オフで本番不変。
- 取れる内訳: 重み転送 (`weight_upload_sec` / `_bytes` / `_count`) / UNet 段グループ (`unet_embed/down/mid/up/convout_sec`) / カテゴリ (`cat_resnet_sec` = conv・groupnorm 主体 / `cat_transformer_sec` = attention・gemm 主体 / `cat_attention_sec`) / `vae_sec` / `host_roundtrip_sec` / `total_sec` / `unet_steps`。
- 診断の型: ① まず**電力**を見る (`nvidia-smi`・43%/360W なら SM が埋まっていない) ② 帯域が低い (~11%) なら帯域律速でない ③ → occupancy/latency 律速と判定し、1 block=1 warp の attention や per-step の full sync を疑う。**`sm%` に騙されない**。
- 現行の律速仮説 (`docs/fast-mode-plan.md` 準拠): CFG の逐次 2 forward / 毎 step の H2D・D2H + host 合成による 20 回 full sync / attention の低占有率。
- **計測クローズ済みの事実 (再調査させない)**: 単一 GPU 構成では `src/core/multi_frame_pipeline.hpp` の複数フレーム先行生成は GPU バウンドで飽和し (look-ahead 2 で最適)、CPU LM は拡散の裏に完全隠蔽される → **CPU LM Tier 2 (独立 forward ワーカー) の発動条件は成立しない**。SDXL が桁違いに速くなった世界でのみ再評価。
- 成果物: 内訳表 (秒・%) と「次に効くレバー」の提案。改修そのものは `cuda-kernel-dev` へ渡す。数値は `docs/measurements-log.md` に追記。
- 走る機械: profile 実行は研究機 (CUDA 必須)。計装コードの著述・レビューは開発機でも可。

**必ず書かない:** `queue.Queue` / `torch.cuda.memory_allocated` 等の Python probe 時代の手法、Thread-A/B が Qwen2 前提の図。

- [ ] **Step 2: 廃止 2 体を削除**

```bash
git rm .claude/agents/pipeline-debugger.md .claude/agents/prompt-engineer.md
```

- [ ] **Step 3: 検証**

Run: `python scripts/test_dollma_agent_defs.py`

Expected: `test_expected_agents_present` と `test_retired_agents_absent` が緑。

- [ ] **Step 4: コミット**

```bash
git add .claude/agents/perf-profiler.md
git commit -m "chore(agents): perf-profiler を新設し pipeline-debugger/prompt-engineer を廃止"
```

---

### Task 8: `csharp-ui-implementer` 改訂 (軽微)

**Files:**
- Modify: `.claude/agents/csharp-ui-implementer.md`

**Interfaces:**
- Consumes: Task 3 の書式
- Produces: なし

- [ ] **Step 1: 差分で直す**

frontmatter を `tools: Bash, PowerShell, Read, Write, Edit, Glob, Grep` に置換 (model 行は書かない)。

**追記する (実在するもの):**
- 担当ツリーに `ui/Telemetry/` / `ui/data/thumbs/` / `ui/Services/DraftPreview.cs` / `ui.Tests/` を加える。
- 既存の確定挙動: ① プリセットのサムネは `SixLabors.ImageSharp` で 128px 上限に縮小し `ui/data/thumbs/` に PNG 保存 (`PresetStore.Save(preset, thumbnailPng)`・`thumbnailPng == null` なら既存を温存) ② 下書きモードは「生成」と「下書き (高速プレビュー)」の 2 ボタンで、送信サイズ決定は純ロジック `DraftPreview.ResolveDraftSize` (幅 > 768 → 768²・以下は据え置き・パース不能は 768²) に切り出し済み・`Steps` は不変 ③ `ui/data/` は gitignore (個人データ)。
- 仕様書として `docs/ui-preset-thumbnail-spec.md` を参照させる。

**削る:** 重複する Allman / switch のコード例 (共通ルールへ)。

末尾に共通ルール参照 1 行を追加。

- [ ] **Step 2: 検証**

Run: `python scripts/test_dollma_agent_defs.py`

Expected: `test_tools_contract` と `test_agent_common_exists_and_referenced` が緑 (全 10 体が揃う)。

- [ ] **Step 3: コミット**

```bash
git add .claude/agents/csharp-ui-implementer.md
git commit -m "chore(agents): csharp-ui-implementer を現状化 (Telemetry/thumbs/DraftPreview)"
```

---

### Task 9: 全体検証と仕上げ

**Files:**
- Modify: `docs/superpowers/specs/2026-08-05-subagent-refresh-design.md` (完了追記)
- Read: 全 `.claude/agents/*.md`

**Interfaces:**
- Consumes: Task 1〜8 の全成果
- Produces: 全項目緑の検証ログ

- [ ] **Step 1: 検証スクリプトを全項目緑にする**

Run: `python scripts/test_dollma_agent_defs.py`

Expected: `9/9 passed` と表示され終了コード 0。赤が残っていれば該当定義を直す。

- [ ] **Step 2: 数値主張を docs と突合する**

各定義に書いた HW 実測値 (CLIP-L 7.85ms / WD14 268ms / ScorerNet 8.32ms / QualityMLP 0.553ms / CLIP image 85.55ms / ISNet iGPU 99.96ms / diffusers 3.80s / 自作 19.5s/枚 / 154W of 360W) を
`CLAUDE.md` の計測ベースライン表と `docs/measurements-log.md` で grep して一致を確認する。

Grep で `7\.85|268|8\.32|0\.553|85\.55|99\.96|3\.80|19\.5|154W` を
`CLAUDE.md` / `docs/measurements-log.md` / `docs/fast-mode-plan.md` に対してかけ、
定義に書いた値がいずれかの docs に実在することを 1 つずつ確認する。
一致しない値は定義から削る (推測値を定義に残さない)。

- [ ] **Step 3: 可変値が焼かれていないか点検する**

Run: `grep -nE "[0-9]+/[0-9]+ (test|緑)|meson test [0-9]+|件$" .claude/agents/*.md`

Expected: test 件数・データ件数のような可変値がヒットしないこと (ヒットしたら「`meson test -C build` が緑」等の表現に直す)。

- [ ] **Step 4: エージェント一覧が 10 体で読めることを確認**

`.claude/agents/` を列挙し、10 ファイル・全 frontmatter がパースされることを確認する。

Run: `python -c "import sys;sys.path.insert(0,'scripts');import test_dollma_agent_defs as t;print(sorted(s for s,_ in t._agent_files()))"`

Expected: 期待する 10 体の名前が並ぶ (`prompt-engineer` / `pipeline-debugger` を含まない)。

- [ ] **Step 5: spec に完了追記**

`docs/superpowers/specs/2026-08-05-subagent-refresh-design.md` の末尾に「## 完了 (2026-08-05)」節を足し、
①10 体構成になったこと ②`docs/agent-common.md` に共通ルールを集約したこと
③`scripts/test_dollma_agent_defs.py` が 9/9 緑で回帰テストとして常設されたこと
④スコープ外に置いた 2 件 (settings.json の旧パス / hook の `py -3.12` fail-open) は未処理のまま残ること を記録する。

- [ ] **Step 6: 最終コミット**

```bash
git add docs/superpowers/specs/2026-08-05-subagent-refresh-design.md
git commit -m "docs(agents): サブエージェント定義 現状化を完了 (10 体・共通ルール 1 枚・検証 9/9 緑)"
```

- [ ] **Step 7: ブランチの扱いをユーザーに確認**

`main` へマージするか PR にするかはユーザー判断。`git log --oneline main..chore/agent-defs-refresh` で
コミット列を提示して確認を取る (勝手にマージ・push しない)。

---

## 自己レビュー結果

**spec 網羅**: S1 (構成) → Task 2〜8 / S2 (共通ルール 9 節) → Task 2 / S3 (10 体の改訂方針) → Task 3〜8 で全体カバー / S4 (テンプレート) → Task 3 が見本・以降が踏襲 / S5 (検証 5 項目) → Task 1 のスクリプト + Task 9 の Step 1〜4。spec の「スコープ外 2 件」は Task 9 Step 5 で未処理と明記する形で回収。

**型・名称整合**: 検証スクリプトのテスト関数名は Task 1 で定義したものを Task 2〜9 で同名参照。
`EXPECTED_AGENTS` の 10 名は File Structure の一覧と一致。共通ルールの参照文言は Task 2 で 1 つに固定し Task 3 の見本で実際に使用。

**留意点 (実装者向け)**: spec の検証項目4 (perf-profiler を試験的に呼ぶ) は本計画に含めていない。Agent 呼び出しはユーザー承認が要るため、Task 9 完了後に別途提案する。

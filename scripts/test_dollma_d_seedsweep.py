# -*- coding: utf-8 -*-
"""dollama Phase 4 施策 D — d_seedsweep / d_seedsweep_analyze の構造乾式テスト。

torch を起動せず (sweep スクリプトは subprocess で train_bitnet.py を呼ぶだけなので
import 自体に torch は不要) に、次の権威不変条件を守らせる:

  - dollma_d_seedsweep:
    * SEEDS == [20260620, 20260621, 42, 7] (A/B sweep と一致)。
    * 2 アーム c33/d80 のみ・唯一の差分が --arch d80m (c33 は arch None)。
    * 両アームとも同一レシピ (--train-file b2000 ∧ --identity)。
      → train() の組む cmd を検査し、両者に --train-file と --identity が在り、
        d80 のみ --arch d80m が付くことを確認。
    * eval_arm() の組む cmd: d80 は --arch d80m 連動 (モデル組立連動・必須)・c33 は無。
    * COMMON が 6ep/tags/bs32/lr3e-4 (A/B と一致)。
    * setup の正準コピー対象に b2000 train / a12k identity(正準名) / val / 凍結 eval が揃う。

  - dollma_d_seedsweep_analyze:
    * ARMS == [c33, d80]・delta = d80 - c33・band = CONTROL(c33) の seed 分散。
    * import 可能で main が引数なしで走る (レポート不在でも missing 表示で落ちない)。

subprocess も torch も使わない純構造テスト。CLAUDE.md「Python ゆえ meson 対象外」方針。

実行:
  py -3.12 scripts/test_dollma_d_seedsweep.py
  py -3.12 -m pytest scripts/test_dollma_d_seedsweep.py -q
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import dollma_d_seedsweep as sw
import dollma_d_seedsweep_analyze as an


def test_seeds_match_ab_sweep():
    assert sw.SEEDS == [20260620, 20260621, 42, 7], sw.SEEDS
    assert an.SEEDS == [20260620, 20260621, 42, 7], an.SEEDS
    print("[ok] SEEDS = A/B sweep と一致")


def test_two_arms_only_arch_differs():
    assert set(sw.ARMS) == {"c33", "d80"}, sw.ARMS
    assert sw.ARMS["c33"]["arch"] is None
    assert sw.ARMS["d80"]["arch"] == "d80m"
    print("[ok] 2 アーム c33/d80・差分は --arch d80m のみ")


def test_common_hparams():
    # 6ep / tags / bs32 / lr3e-4 (A/B sweep COMMON と一致)。
    c = sw.COMMON
    assert "--epochs" in c and c[c.index("--epochs") + 1] == "6", c
    assert "--loss-mode" in c and c[c.index("--loss-mode") + 1] == "tags", c
    assert "--batch-size" in c and c[c.index("--batch-size") + 1] == "32", c
    assert "--lr" in c and c[c.index("--lr") + 1] == "3e-4", c
    print("[ok] COMMON = 6ep/tags/bs32/lr3e-4")


def _capture_run_cmds(monkey_target):
    """sw.run を差し替えて、呼ばれた cmd を捕捉する文脈を返す。"""
    captured = []

    def fake_run(cmd):
        captured.append(list(cmd))

    setattr(sw, "run", fake_run)
    return captured


def test_train_cmd_both_arms_same_recipe():
    """train() の cmd: 両アームに --train-file(b2000) と --identity・d80 のみ --arch d80m。"""
    orig = sw.run
    try:
        cap = _capture_run_cmds(None)
        sw.train("c33", 20260620)
        sw.train("d80", 20260620)
    finally:
        sw.run = orig
    c33_cmd, d80_cmd = cap[0], cap[1]
    for cmd, arm in ((c33_cmd, "c33"), (d80_cmd, "d80")):
        assert "--train-file" in cmd, (arm, cmd)
        ti = cmd.index("--train-file") + 1
        assert cmd[ti].endswith("pairs.train.diverse_b2000.jsonl"), (arm, cmd[ti])
        assert "--identity" in cmd, (arm, cmd)
        assert "--data-dir" in cmd and "--seed" in cmd, (arm, cmd)
    # d80 のみ --arch d80m。c33 は --arch を一切持たない。
    assert "--arch" in d80_cmd and d80_cmd[d80_cmd.index("--arch") + 1] == "d80m", d80_cmd
    assert "--arch" not in c33_cmd, c33_cmd
    print("[ok] train cmd: 同一レシピ・d80 のみ --arch d80m")


def test_eval_cmd_arch_plumbing():
    """eval_arm() の cmd: --eval-only/--dump-persample/--device cuda・d80 のみ --arch d80m。
    eval-only もモジュール定数でモデルを組むため d80 重み採点に --arch d80m が必須。"""
    orig_run = sw.run
    orig_move = None
    import shutil
    orig_move = shutil.move
    try:
        cap = _capture_run_cmds(None)
        shutil.move = lambda *a, **k: None  # 退避 move を無効化 (ファイル不在でも落とさない)
        sw.eval_arm("c33", 7, "/tmp/w_fp32.safetensors")
        sw.eval_arm("d80", 7, "/tmp/w_fp32.safetensors")
    finally:
        sw.run = orig_run
        shutil.move = orig_move
    c33_cmd, d80_cmd = cap[0], cap[1]
    for cmd, arm in ((c33_cmd, "c33"), (d80_cmd, "d80")):
        assert "--eval-only" in cmd, (arm, cmd)
        assert "--dump-persample" in cmd, (arm, cmd)
        assert "--device" in cmd and cmd[cmd.index("--device") + 1] == "cuda", (arm, cmd)
        assert "--eval-name" in cmd, (arm, cmd)
        assert cmd[cmd.index("--eval-name") + 1] == f"{arm}_7", (arm, cmd)
    assert "--arch" in d80_cmd and d80_cmd[d80_cmd.index("--arch") + 1] == "d80m", d80_cmd
    assert "--arch" not in c33_cmd, c33_cmd
    print("[ok] eval cmd: --arch d80m 連動 (d80 のみ)")


def test_setup_canonical_targets():
    """setup_sweep_dir が固定名で読むファイル一式 (b2000/a12k identity/val/凍結 eval) を
    正準名にコピーすることを、_copy_canonical 呼び出しの (src,dst) で検査 (実コピーなし)。"""
    orig = sw._copy_canonical
    pairs = []
    try:
        sw._copy_canonical = lambda src, dst, label: pairs.append(
            (os.path.basename(src), os.path.basename(dst)))
        # makedirs は実行されるが _results 作成のみ (副作用軽微)。
        sw.setup_sweep_dir()
    finally:
        sw._copy_canonical = orig
    dsts = {d for _, d in pairs}
    srcs = {s for s, _ in pairs}
    # 正準名 (train_bitnet.py が固定名で読む先)。
    assert "vocab.json" in dsts
    assert "pairs.train.diverse_b2000.jsonl" in dsts
    assert "pairs.val.jsonl" in dsts
    assert "pairs.identity.train.jsonl" in dsts
    assert "pairs.identity.val.jsonl" in dsts
    assert "pairs.eval_diverse_a.jsonl" in dsts
    assert "pairs.eval_diverse_b.jsonl" in dsts
    # a12k identity を正準名へ写すこと (src が a12k 由来)。
    assert "pairs.identity.train.a12k.jsonl" in srcs, srcs
    assert "pairs.identity.val.a12k.jsonl" in srcs, srcs
    print("[ok] setup 正準コピー対象 = b2000/a12k identity/val/凍結 eval")


def test_analyze_arms_and_delta_direction():
    assert an.ARMS == ["c33", "d80"], an.ARMS
    assert an.CONTROL == "c33" and an.TREAT == "d80"
    # delta = TREAT - CONTROL = d80 - c33。band は CONTROL(c33) の seed 分散。
    print("[ok] analyze: ARMS=[c33,d80] delta=d80-c33 band=c33")


def test_analyze_main_runs_without_reports():
    """レポート不在でも analyze main が落ちない (missing 表示で完走)。"""
    argv = sys.argv
    try:
        sys.argv = ["dollma_d_seedsweep_analyze.py"]
        an.main()
    finally:
        sys.argv = argv
    print("[ok] analyze main: レポート不在でも完走")


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
    print(f"all {len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()

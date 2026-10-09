# -*- coding: utf-8 -*-
"""dollma_f0b_z4_selftest.py — Z-4 の train_bitnet.py 追加フラグの selftest (smoke 規模のみ)。

検証:
  T1 引数未指定の --smoke --sft-rejection 出力 (名前・場所・重みバイト) が HEAD の train_bitnet.py と完全一致
     (CPU・同 seed)。= 既定挙動不変。
  T2 --out-dir/--out-tag 指定時: 出力は out-dir の <base><tag>_smoke* に出て、data-dir には増えない。
     重みバイトは T1 と一致 (フラグが数値に影響しない)。
  T3 --deterministic (warn_only なし) で cuda smoke を 2 回 → 重み sha256 一致。例外の有無を報告。
  T4 --eval-only --out-dir/--out-tag/--dump-persample/--dump-retention-detail が out-dir にだけ出る
     (smoke 重みの評価であり canon/R の本評価ではない)。
データは scratch (docs/logs/f0b-z4/_scratch_data) の入力コピー・smoke は 32 件×1 epoch のみ。
"""
import hashlib
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRATCH = os.path.join(ROOT, "docs", "logs", "f0b-z4", "_scratch_data")
TRAIN = os.path.join(ROOT, "scripts", "train_bitnet.py")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def run(script, args, env_extra=None):
    env = dict(os.environ)
    env.update(env_extra or {})
    t = time.time()
    p = subprocess.run([sys.executable, script] + args, cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env)
    return p.returncode, p.stdout + p.stderr, time.time() - t


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ok = True
    t0 = time.time()
    work = os.path.join(SCRATCH, "selftest")
    shutil.rmtree(work, ignore_errors=True)
    # data-dir は入力コピーのみ (T1 の出力で汚れても selftest 専用)
    dd_new = os.path.join(work, "data_new")
    dd_old = os.path.join(work, "data_old")
    for d in (dd_new, dd_old):
        os.makedirs(d)
        for fn in ("vocab.json", "pairs.val.jsonl", "pairs.eval_diverse_a.jsonl",
                   "pairs.eval_diverse_b.jsonl", "pairs.identity.val.jsonl", "bitnet_dense_fp32.safetensors"):
            shutil.copy(os.path.join(SCRATCH, fn), d)
        # SFT 経路でも pairs.train.jsonl は読まれる (syn_train)。ダミー (val 先頭 64 件) を置く。
        with open(os.path.join(SCRATCH, "pairs.val.jsonl"), encoding="utf-8") as fi,                 open(os.path.join(d, "pairs.train.jsonl"), "w", encoding="utf-8") as fo:
            fo.writelines(list(fi)[:64])
    old_script = os.path.join(work, "train_bitnet_HEAD.py")
    src = subprocess.run(["git", "show", "HEAD:scripts/train_bitnet.py"], cwd=ROOT, capture_output=True)
    if src.returncode != 0:
        print("[selftest FAIL] git show HEAD:scripts/train_bitnet.py")
        sys.exit(1)
    with open(old_script, "wb") as f:
        f.write(src.stdout)
    common = ["--sft-rejection", "--sft-data", os.path.join(SCRATCH, "sft_bestofn.jsonl"),
              "--epochs", "3", "--lr", "2e-5", "--warmup", "5", "--smoke", "--device", "cpu"]
    # T1
    before = set(os.listdir(dd_new))
    rc1, o1, t1 = run(TRAIN, ["--data-dir", dd_new] + common)
    rc0, o0, t0_ = run(old_script, ["--data-dir", dd_old] + common)
    new_files = sorted(set(os.listdir(dd_new)) - before)
    old_files = sorted(set(os.listdir(dd_old)) - set(os.listdir(dd_new)) | set(
        f for f in os.listdir(dd_old) if f.startswith(("bitnet_dense_sft", "train_stats"))))
    names_new = sorted(f for f in os.listdir(dd_new) if f.startswith(("bitnet_dense_sft", "train_stats")))
    names_old = sorted(f for f in os.listdir(dd_old) if f.startswith(("bitnet_dense_sft", "train_stats")))
    if rc1 != 0 or rc0 != 0:
        print("[selftest FAIL] T1 smoke 実行失敗", rc1, rc0, o1[-800:], o0[-800:])
        ok = False
    else:
        same_names = names_new == names_old and len(names_new) == 3
        w_new = sha(os.path.join(dd_new, "bitnet_dense_sft_smoke_fp32.safetensors"))
        w_old = sha(os.path.join(dd_old, "bitnet_dense_sft_smoke_fp32.safetensors"))
        print(f"[T1] names={names_new} same_names={same_names} weights_equal={w_new == w_old} "
              f"({t1:.1f}s/{t0_:.1f}s)")
        ok &= same_names and (w_new == w_old)
    # T2
    od = os.path.join(work, "out_x")
    rc2, o2, t2 = run(TRAIN, ["--data-dir", dd_new, "--out-dir", od, "--out-tag", "_zz"] + common)
    names_out = sorted(os.listdir(od)) if os.path.isdir(od) else []
    data_after = sorted(f for f in os.listdir(dd_new) if f.startswith(("bitnet_dense_sft", "train_stats")))
    exp = sorted(["bitnet_dense_sft_zz_smoke.safetensors", "bitnet_dense_sft_zz_smoke_fp32.safetensors",
                  "train_stats_sft_zz_smoke.json"])
    t2_ok = rc2 == 0 and names_out == exp and data_after == names_new
    if t2_ok:
        t2_ok = sha(os.path.join(od, "bitnet_dense_sft_zz_smoke_fp32.safetensors")) == w_new
    print(f"[T2] out_files={names_out} data_dir_unchanged={data_after == names_new} ok={t2_ok} ({t2:.1f}s)")
    ok &= t2_ok
    # T3 (cuda があれば)
    det_report = "skipped(no cuda)"
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False
    if has_cuda:
        shas = []
        for i in range(2):
            od3 = os.path.join(work, f"det{i}")
            args3 = [a for a in common if a not in ("cpu",)]
            args3 = [("cuda" if a == "--device" else a) for a in args3]
            # --device の値を cuda に置換
            args3 = []
            skip = False
            for a in common:
                if a == "--device":
                    args3 += ["--device", "cuda"]
                    skip = True
                elif skip:
                    skip = False
                else:
                    args3.append(a)
            rc3, o3, t3 = run(TRAIN, ["--data-dir", dd_new, "--out-dir", od3, "--deterministic"] + args3)
            if rc3 != 0:
                print("[selftest FAIL] T3 deterministic(warn_only なし) 失敗:\n", o3[-1500:])
                det_report = "warn_only なしで例外 -> warn_only=True が必要"
                ok = False
                break
            shas.append(sha(os.path.join(od3, "bitnet_dense_sft_smoke_fp32.safetensors")))
            print(f"[T3] run{i} rc=0 ({t3:.1f}s) sha={shas[-1][:16]}")
        if len(shas) == 2:
            eq = shas[0] == shas[1]
            det_report = f"warn_only 不要・2 回の sha256 一致={eq}"
            ok &= eq
    print(f"[T3] {det_report}")
    # T4 eval-only (smoke 重みの評価・本評価ではない)
    od4 = os.path.join(work, "eval_out")
    smoke_w = os.path.join(od, "bitnet_dense_sft_zz_smoke_fp32.safetensors")
    before_dd = sorted(os.listdir(dd_new))
    rc4, o4, t4 = run(TRAIN, ["--data-dir", dd_new, "--eval-only", "--weights", smoke_w,
                              "--eval-name", "smoke_eval", "--out-dir", od4, "--out-tag", "_zz",
                              "--dump-persample", "--dump-retention-detail"])
    names4 = sorted(os.listdir(od4)) if os.path.isdir(od4) else []
    exp4 = ["eval_persample_smoke_eval_zz.npz", "eval_report_smoke_eval_zz.json"]
    t4_ok = rc4 == 0 and names4 == exp4 and sorted(os.listdir(dd_new)) == before_dd
    if t4_ok:
        import json
        rep = json.load(open(os.path.join(od4, "eval_report_smoke_eval_zz.json"), encoding="utf-8"))
        t4_ok = "details" in rep["identity_retention"]
    print(f"[T4] eval out={names4} data_dir_unchanged={sorted(os.listdir(dd_new)) == before_dd} ok={t4_ok} ({t4:.1f}s)")
    if not t4_ok:
        print(o4[-1200:])
    ok &= t4_ok
    print(f"[selftest] {'OK' if ok else 'FAIL'} (total {time.time() - t0:.1f}s)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""施策C seed sweep: D5 (soft-label KL 案A) vs #1 (plain hard CE) の diverse
F1/Jaccard 優位が seed 頑健か確定する。

各 seed で #1 と D5 を 6 epoch FP32 訓練 (出力は data/bitnet/_seedsweep/ 配下・
本番 bitnet_dense*/bitnet_dense_kl* を絶対に上書きしない) し、
pairs.eval_diverse_a/b で per-sample F1/Jaccard 配列付きで採点する。

- 訓練/採点は train_bitnet.py を subprocess 呼び出し (単一ソース維持)。
- 各 train は _seedsweep/ の固定名 (bitnet_dense*/bitnet_dense_kl*) を上書きするので、
  train 直後に即 eval し、per-sample npz と eval_report を _results/ へ退避する。
- seed 20260620 の #1 は本番 6ep 重みを再利用 (再訓練不要)。D5 は本番が 10ep なので
  sweep 一貫性のため全 seed 6ep を新規訓練する。
"""

import os
import subprocess
import sys
import shutil
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SD = os.path.join(ROOT, "data", "bitnet", "_seedsweep")
RES = os.path.join(SD, "_results")
PROD = os.path.join(ROOT, "data", "bitnet")
TRAIN = os.path.join(HERE, "train_bitnet.py")

SEEDS = [20260620, 20260621, 42, 7]
PY = sys.executable  # 起動した python をそのまま使う

# D5 採用ハイパラ (training-spec §11.2 / §11.3 D5-XL 6ep)
D5_ARGS = [
    "--distill-kl", "--kl-alpha", "0.2", "--kl-temp", "2.0",
    "--cooc-main-mass", "0.92", "--cooc-temp", "2.0", "--cooc-topn", "24",
    "--dropout", "0.0", "--weight-decay", "0.02",
]
COMMON = ["--epochs", "6", "--loss-mode", "tags", "--batch-size", "32", "--lr", "3e-4"]
EVAL_DEVICE = "cuda"  # greedy 生成を GPU で高速化 (#1/D5 同一 device → paired delta に無バイアス)


def run(cmd):
    print("[run]", " ".join(cmd), flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT)
    print(f"[run] rc={r.returncode} ({time.time()-t0:.1f}s)", flush=True)
    if r.returncode != 0:
        raise SystemExit(f"command failed rc={r.returncode}: {' '.join(cmd)}")


def train(arm, seed):
    """arm in {'base','d5'} を seed で 6ep 訓練 (出力は _seedsweep/ の固定名)。"""
    cmd = [PY, TRAIN, "--data-dir", SD, "--seed", str(seed)] + COMMON
    if arm == "d5":
        cmd += D5_ARGS
    run(cmd)


def eval_arm(arm, seed, weights):
    """weights を diverse で採点し per-sample npz + report を _results/ へ退避。"""
    name = f"{arm}_{seed}"
    cmd = [PY, TRAIN, "--eval-only", "--data-dir", SD,
           "--weights", weights, "--eval-name", name,
           "--dump-persample", "--device", EVAL_DEVICE]
    run(cmd)
    # 退避: eval_report_<name>.json と eval_persample_<name>.npz を _results/ へ移動。
    for fn in (f"eval_report_{name}.json", f"eval_persample_{name}.npz"):
        src = os.path.join(SD, fn)
        if os.path.exists(src):
            shutil.move(src, os.path.join(RES, fn))
            print(f"[move] {fn} -> _results/", flush=True)


def main():
    os.makedirs(RES, exist_ok=True)
    for seed in SEEDS:
        # ---- #1 (base) ----
        # seed 20260620 の base eval は本番 6ep 重みで実施済み (CPU・full re-score と
        # bit 一致確認済み) → _results/eval_*_base_20260620.* を再利用しスキップ。
        base_done = os.path.exists(
            os.path.join(RES, f"eval_persample_base_{seed}.npz"))
        if base_done:
            print(f"[seed {seed}] base eval 既存 -> スキップ", flush=True)
        else:
            train("base", seed)
            base_w = os.path.join(SD, "bitnet_dense_fp32.safetensors")
            eval_arm("base", seed, base_w)

        # ---- D5 (全 seed 6ep 新規訓練・本番 10ep とは別) ----
        if os.path.exists(os.path.join(RES, f"eval_persample_d5_{seed}.npz")):
            print(f"[seed {seed}] d5 eval 既存 -> スキップ", flush=True)
        else:
            train("d5", seed)
            d5_w = os.path.join(SD, "bitnet_dense_kl_fp32.safetensors")
            eval_arm("d5", seed, d5_w)

    print("[sweep] 全 seed 完了", flush=True)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""施策B 件数拡大 (2,000版) seed sweep: B2000 (diverse_b2000 train・著述2,000+synthetic2,500
=4,500・凍結済) vs #1 (plain hard CE・通常 train) の diverse-val 生成 F1/Jaccard 優位が
seed 横断で頑健か確定する。

dollma_b_seedsweep.py (500版) と同手続き・同 seed・同ハイパラで、唯一の差分は B arm の
train ファイル (500版 = pairs.train.diverse_b.jsonl → 2000版 = pairs.train.diverse_b2000.jsonl)
と出力先 (_seedsweep_b/ → _seedsweep_b2000/)。

  - 各 seed で base(#1) と b2000(B2000) を 6 epoch FP32 cosine 訓練。
  - 出力は data/bitnet/_seedsweep_b2000/ 配下のみ (本番 bitnet_dense*/golden /
    500版 _seedsweep_b/ / C-4 _seedsweep/ を一切上書きしない)。
  - pairs.eval_diverse_a/b (500版 sweep と byte 一致の凍結物差し) で per-sample
    F1/Jaccard/precision/recall 配列付き採点。
  - train は _seedsweep_b2000/ の固定名 (base=bitnet_dense*・b2000=bitnet_dense_diverse_b2000*)
    を上書きするので、train 直後に即 eval し per-sample npz / eval_report を
    _results/ へ seed 別名で退避する。

両 arm とも plain hard CE で、唯一の差分は train ファイル (#1 = pairs.train.jsonl /
B2000 = pairs.train.diverse_b2000.jsonl)。val・eval_diverse_a/b は完全に共通なので
per-sample paired delta = B2000 − #1 に train 以外のバイアスは乗らない。
"""

import os
import subprocess
import sys
import shutil
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SD = os.path.join(ROOT, "data", "bitnet", "_seedsweep_b2000")
RES = os.path.join(SD, "_results")
TRAIN = os.path.join(HERE, "train_bitnet.py")

# 500版 sweep (dollma_b_seedsweep.py SEEDS) と一致させる。
SEEDS = [20260620, 20260621, 42, 7]
PY = sys.executable  # 起動した python をそのまま使う

# 両 arm 共通ハイパラ (500版 sweep COMMON と同設定・FP32 cosine 6ep)。
COMMON = ["--epochs", "6", "--loss-mode", "tags", "--batch-size", "32", "--lr", "3e-4"]
EVAL_DEVICE = "cuda"  # greedy 生成を GPU で高速化 (base/b 同一 device → paired delta 無バイアス)

# B2000 の train ファイル (著述 2,000+synthetic 2,500=4,500・凍結済)。_seedsweep_b2000/ 内に複製済み。
B_TRAIN_FILE = os.path.join(SD, "pairs.train.diverse_b2000.jsonl")
# train-file basename "pairs.train.diverse_b2000.jsonl" → suffix "diverse_b2000"
#   → bitnet_dense_diverse_b2000{,_fp32}.safetensors
B_ARM_WEIGHTS = os.path.join(SD, "bitnet_dense_diverse_b2000_fp32.safetensors")


def run(cmd):
    print("[run]", " ".join(cmd), flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT)
    print(f"[run] rc={r.returncode} ({time.time()-t0:.1f}s)", flush=True)
    if r.returncode != 0:
        raise SystemExit(f"command failed rc={r.returncode}: {' '.join(cmd)}")


def train(arm, seed):
    """arm in {'base','b'} を seed で 6ep 訓練 (出力は _seedsweep_b2000/ の固定名)。

      base -> bitnet_dense{,_fp32}.safetensors             (--train-file 無指定)
      b    -> bitnet_dense_diverse_b2000{,_fp32}.safetensors (--train-file diverse_b2000)
    """
    cmd = [PY, TRAIN, "--data-dir", SD, "--seed", str(seed)] + COMMON
    if arm == "b":
        cmd += ["--train-file", B_TRAIN_FILE]
    run(cmd)


def eval_arm(arm, seed, weights):
    """weights を diverse_a/b で採点し per-sample npz + report を _results/ へ退避。"""
    name = f"{arm}_{seed}"
    cmd = [PY, TRAIN, "--eval-only", "--data-dir", SD,
           "--weights", weights, "--eval-name", name,
           "--dump-persample", "--device", EVAL_DEVICE]
    run(cmd)
    for fn in (f"eval_report_{name}.json", f"eval_persample_{name}.npz"):
        src = os.path.join(SD, fn)
        if os.path.exists(src):
            shutil.move(src, os.path.join(RES, fn))
            print(f"[move] {fn} -> _results/", flush=True)


def main():
    os.makedirs(RES, exist_ok=True)
    for seed in SEEDS:
        # ---- #1 (base) ----
        if os.path.exists(os.path.join(RES, f"eval_persample_base_{seed}.npz")):
            print(f"[seed {seed}] base eval 既存 -> スキップ", flush=True)
        else:
            train("base", seed)
            base_w = os.path.join(SD, "bitnet_dense_fp32.safetensors")
            eval_arm("base", seed, base_w)

        # ---- B2000 (diverse_b2000) ----
        if os.path.exists(os.path.join(RES, f"eval_persample_b_{seed}.npz")):
            print(f"[seed {seed}] b eval 既存 -> スキップ", flush=True)
        else:
            train("b", seed)
            eval_arm("b", seed, B_ARM_WEIGHTS)

    print("[sweep] 全 seed 完了", flush=True)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""施策B seed sweep: B-pilot (diverse_b train・Replace 500 著述) vs #1 (plain hard CE・
通常 train) の diverse-val 生成 F1/Jaccard 優位が seed 横断で頑健か確定する。

C-4 (dollma_c_seedsweep.py) と同手続き:
  - 各 seed で base(#1) と b(B-pilot) を 6 epoch FP32 cosine 訓練。
  - 出力は data/bitnet/_seedsweep_b/ 配下のみ (本番 bitnet_dense*/golden /
    C-4 の _seedsweep/ を一切上書きしない)。
  - pairs.eval_diverse_a/b で per-sample F1/Jaccard/precision/recall 配列付き採点。
  - train は _seedsweep_b/ の固定名 (base=bitnet_dense*・b=bitnet_dense_diverse_b*) を
    上書きするので、train 直後に即 eval し per-sample npz / eval_report を
    _results/ へ seed 別名で退避する。

両 arm とも plain hard CE で、唯一の差分は train ファイル (#1 = pairs.train.jsonl /
B-pilot = pairs.train.diverse_b.jsonl)。val・eval_diverse_a/b は完全に共通なので
per-sample paired delta = B − #1 に train 以外のバイアスは乗らない。

C-4 の base 結果 (_seedsweep/) は別ディレクトリ・別実験なので再利用しない
(B sweep を完全自己完結・隔離するため base も _seedsweep_b/ で新規訓練する)。
"""

import os
import subprocess
import sys
import shutil
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SD = os.path.join(ROOT, "data", "bitnet", "_seedsweep_b")
RES = os.path.join(SD, "_results")
TRAIN = os.path.join(HERE, "train_bitnet.py")

# C-4 と同一 seed 集合 (dollma_c_seedsweep.py SEEDS と一致させる)。
SEEDS = [20260620, 20260621, 42, 7]
PY = sys.executable  # 起動した python をそのまま使う

# 両 arm 共通ハイパラ (C-4 COMMON と同設定・FP32 cosine 6ep)。
COMMON = ["--epochs", "6", "--loss-mode", "tags", "--batch-size", "32", "--lr", "3e-4"]
EVAL_DEVICE = "cuda"  # greedy 生成を GPU で高速化 (base/b 同一 device → paired delta 無バイアス)

# B-pilot の train ファイル (Replace 500 著述・diverse_b)。_seedsweep_b/ 内に複製済み。
B_TRAIN_FILE = os.path.join(SD, "pairs.train.diverse_b.jsonl")


def run(cmd):
    print("[run]", " ".join(cmd), flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT)
    print(f"[run] rc={r.returncode} ({time.time()-t0:.1f}s)", flush=True)
    if r.returncode != 0:
        raise SystemExit(f"command failed rc={r.returncode}: {' '.join(cmd)}")


def train(arm, seed):
    """arm in {'base','b'} を seed で 6ep 訓練 (出力は _seedsweep_b/ の固定名)。

      base -> bitnet_dense{,_fp32}.safetensors        (--train-file 無指定)
      b    -> bitnet_dense_diverse_b{,_fp32}.safetensors (--train-file diverse_b)
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
        if os.path.exists(os.path.join(RES, f"eval_persample_base_{seed}.npz")):
            print(f"[seed {seed}] base eval 既存 -> スキップ", flush=True)
        else:
            train("base", seed)
            base_w = os.path.join(SD, "bitnet_dense_fp32.safetensors")
            eval_arm("base", seed, base_w)

        # ---- B-pilot (diverse_b) ----
        if os.path.exists(os.path.join(RES, f"eval_persample_b_{seed}.npz")):
            print(f"[seed {seed}] b eval 既存 -> スキップ", flush=True)
        else:
            train("b", seed)
            b_w = os.path.join(SD, "bitnet_dense_diverse_b_fp32.safetensors")
            eval_arm("b", seed, b_w)

    print("[sweep] 全 seed 完了", flush=True)


if __name__ == "__main__":
    main()

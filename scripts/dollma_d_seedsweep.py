# -*- coding: utf-8 -*-
"""Phase 4-D 容量増 seed sweep: c33(33M) vs d80(80M)・両アーム同一レシピで
`--arch` だけ差し、diverse-val 生成 F1/Jaccard 優位が seed 横断で頑健かを確定する。

設計の核 (dollma_a_seedsweep.py の複製・改変):
  - **2 アームとも完全同一レシピ** = b2000 多様化 train (`--train-file`) ∧ a12k identity
    (`--identity`)。唯一の差分は `--arch d80m` の有無 (= 容量 33M vs 80M)。
    A-sweep の「base(plain) vs a(identity)」と違い、ここでは両アームともレシピ込み。
  - delta = d80 − c33 (容量増の純効果)。base band は #1 plain でなく **c33 の seed 分散**。
  - SEEDS (20260620/20260621/42/7)・6ep・FP32 cosine は A/B sweep と一致。
  - `--only-seed` で seed 単位分割・`_results/*.npz` 存在 skip の冪等再開 (b10k 作法踏襲)。

train_bitnet.py は **改修ゼロ**で使う:
  - `--arch d80m` (既存・param assert 79,908,864) / `--train-file <b2000>` (synthetic 差し替え) /
    `--identity` (a12k identity 結合) の 3 フラグ合成は smoke 疎通済 (Phase 0)。
  - `--identity` 駆動の重み出力は arch 非依存で固定名 `bitnet_dense_identity{,_fp32}.safetensors`。
    → c33/d80 は同一 data-dir で同名に出るため、各 arm は **train 直後に即 eval して退避**し、
      次 arm の train で上書きされる前に per-sample npz を `_results/` に逃がす (順序で衝突回避)。
  - **eval-only の arch 連動**: train_bitnet.py の eval-only はモジュール定数でモデルを組むため、
    d80 重みの採点には eval 呼び出しにも `--arch d80m` が必須 (c33 は arch 無=33M)。

固定名で読むファイル一式は setup_sweep_dir で正準名コピー (a-sweep と同手):
  vocab.json / pairs.train.diverse_b2000.jsonl / pairs.val.jsonl /
  pairs.identity.{train,val}.jsonl (a12k 版を正準名へ) / pairs.eval_diverse_{a,b}.jsonl。

出力は data/bitnet/_seedsweep_d80m/ 配下のみ (本番 bitnet_dense*/golden / 凍結 eval /
#1 本線 train/val / 既存 _seedsweep_*/ を一切上書きしない)。

使い方:
  # 1 seed だけ (配管検証・小出し):
  py -3.12 scripts/dollma_d_seedsweep.py --only-seed 20260620
  # 全 seed:
  py -3.12 scripts/dollma_d_seedsweep.py
  # smoke (1 seed・--smoke 相当の極小データで疎通だけ):
  py -3.12 scripts/dollma_d_seedsweep.py --only-seed 20260620 --smoke

冪等: _results/eval_persample_{arm}_{seed}.npz が在れば該当 arm を skip (再訓練しない)。
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "bitnet")
TRAIN = os.path.join(HERE, "train_bitnet.py")
SWEEP_DIR = os.path.join(DATA, "_seedsweep_d80m")
RES_DIR = os.path.join(SWEEP_DIR, "_results")

# A/B sweep と一致させた 4 seed・6ep paired。
SEEDS = [20260620, 20260621, 42, 7]
PY = sys.executable  # 起動した python をそのまま使う

# 両 arm 共通ハイパラ (b2000/a12k sweep と同設定・FP32 cosine 6ep)。
COMMON = ["--epochs", "6", "--loss-mode", "tags", "--batch-size", "32", "--lr", "3e-4"]
EVAL_DEVICE = "cuda"  # greedy 生成を GPU で高速化 (c33/d80 同一 device → paired delta 無バイアス)

# 両アーム共通レシピ (b2000 多様化 ∧ a12k identity)。train-file は sweep dir 内の正準コピーを指す。
TRAIN_FILE = os.path.join(SWEEP_DIR, "pairs.train.diverse_b2000.jsonl")

# arm 定義: 唯一の差分は --arch d80m の有無。
#   c33 = control 33M (arch 無)、d80 = treat 80M (--arch d80m)。
ARMS = {
    "c33": {"arch": None},
    "d80": {"arch": "d80m"},
}


def run(cmd):
    print("[run]", " ".join(cmd), flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT)
    print(f"[run] rc={r.returncode} ({time.time()-t0:.1f}s)", flush=True)
    if r.returncode != 0:
        raise SystemExit(f"command failed rc={r.returncode}: {' '.join(cmd)}")


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_canonical(src, dst, label):
    """src を dst へコピーし sha を表示。既存で sha 一致なら skip。"""
    if os.path.exists(dst) and _sha256(src) == _sha256(dst):
        print(f"[setup] {label}: 既存 (sha 一致) -> skip {os.path.basename(dst)}", flush=True)
        return
    shutil.copyfile(src, dst)
    print(f"[setup] {label}: {os.path.basename(src)} -> {os.path.basename(dst)} "
          f"sha={_sha256(dst)[:12]}", flush=True)


def setup_sweep_dir():
    """sweep dir に train_bitnet.py が固定名で読むファイル一式を正準名で配置する。

      vocab.json                                 ← 語彙
      pairs.train.diverse_b2000.jsonl            ← B-2 多様化 train (両アーム共通 --train-file)
      pairs.val.jsonl                            ← #1 本線 val
      pairs.identity.train.jsonl / .val.jsonl    ← a12k identity を正準名へ
      pairs.eval_diverse_a.jsonl / _b.jsonl      ← 凍結物差し (A/B sweep と byte 一致)
    """
    os.makedirs(RES_DIR, exist_ok=True)
    _copy_canonical(os.path.join(DATA, "vocab.json"),
                    os.path.join(SWEEP_DIR, "vocab.json"), "vocab")
    _copy_canonical(os.path.join(DATA, "pairs.train.diverse_b2000.jsonl"),
                    os.path.join(SWEEP_DIR, "pairs.train.diverse_b2000.jsonl"),
                    "b2000 train (共通 --train-file)")
    _copy_canonical(os.path.join(DATA, "pairs.val.jsonl"),
                    os.path.join(SWEEP_DIR, "pairs.val.jsonl"), "val (共通)")
    # a12k identity を固定名へ。
    _copy_canonical(os.path.join(DATA, "pairs.identity.train.a12k.jsonl"),
                    os.path.join(SWEEP_DIR, "pairs.identity.train.jsonl"),
                    "identity train (a12k)")
    _copy_canonical(os.path.join(DATA, "pairs.identity.val.a12k.jsonl"),
                    os.path.join(SWEEP_DIR, "pairs.identity.val.jsonl"),
                    "identity val (a12k)")
    # 凍結 diverse-val (A/B sweep と byte 一致の物差し)。
    _copy_canonical(os.path.join(DATA, "pairs.eval_diverse_a.jsonl"),
                    os.path.join(SWEEP_DIR, "pairs.eval_diverse_a.jsonl"), "eval_diverse_a")
    _copy_canonical(os.path.join(DATA, "pairs.eval_diverse_b.jsonl"),
                    os.path.join(SWEEP_DIR, "pairs.eval_diverse_b.jsonl"), "eval_diverse_b")


def train(arm, seed, smoke=False):
    """arm in {'c33','d80'} を seed で 6ep 訓練 (出力は sweep dir の固定名)。

    両アームとも --train-file(b2000) + --identity の同一レシピ。差分は --arch d80m のみ。
    --identity 駆動なので重みは arch 非依存で bitnet_dense_identity{,_fp32}.safetensors に出る。
    """
    cmd = [PY, TRAIN, "--data-dir", SWEEP_DIR, "--seed", str(seed),
           "--train-file", TRAIN_FILE, "--identity"] + COMMON
    arch = ARMS[arm]["arch"]
    if arch is not None:
        cmd += ["--arch", arch]
    if smoke:
        cmd += ["--smoke"]
    run(cmd)


def eval_arm(arm, seed, weights):
    """weights を diverse_a/b + identity retention で採点し per-sample npz + report を
    _results/ へ seed 別名で退避する。d80 は eval にも --arch d80m が必須 (モデル組立連動)。"""
    name = f"{arm}_{seed}"
    cmd = [PY, TRAIN, "--eval-only", "--data-dir", SWEEP_DIR,
           "--weights", weights, "--eval-name", name,
           "--dump-persample", "--device", EVAL_DEVICE]
    arch = ARMS[arm]["arch"]
    if arch is not None:
        cmd += ["--arch", arch]
    run(cmd)
    for fn in (f"eval_report_{name}.json", f"eval_persample_{name}.npz"):
        src = os.path.join(SWEEP_DIR, fn)
        if os.path.exists(src):
            shutil.move(src, os.path.join(RES_DIR, fn))
            print(f"[move] {fn} -> _results/", flush=True)


def run_arm(arm, seed, smoke=False):
    """arm を train → 即 eval。冪等: 既存 npz があれば skip。

    両アーム同名重み (bitnet_dense_identity*) を共有するため、train 直後に必ず eval して
    per-sample を退避する (次 arm の train で上書きされる前に確定させる)。
    """
    npz = os.path.join(RES_DIR, f"eval_persample_{arm}_{seed}.npz")
    if os.path.exists(npz):
        print(f"[d80m seed {seed}] {arm} eval 既存 -> スキップ", flush=True)
        return
    train(arm, seed, smoke=smoke)
    # --identity 重みは arch 非依存で固定名に出る (smoke 時は _smoke サフィックス)。
    wname = "bitnet_dense_identity_fp32.safetensors"
    if smoke:
        wname = "bitnet_dense_identity_smoke_fp32.safetensors"
    weights = os.path.join(SWEEP_DIR, wname)
    eval_arm(arm, seed, weights)


def run_seed(seed, smoke=False):
    # c33 (control 33M) → d80 (treat 80M) の順。各 arm は train 直後に即 eval で退避。
    run_arm("c33", seed, smoke=smoke)
    run_arm("d80", seed, smoke=smoke)
    print(f"[d80m seed {seed} 完了] c33/d80 eval npz を _results/ に退避済み", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-seed", type=int, default=None,
                    help="指定 seed 1 つだけ実行 (無指定で SEEDS 全件)")
    ap.add_argument("--smoke", action="store_true",
                    help="train_bitnet.py の --smoke を各 train に付与 (極小データ疎通・1ep)")
    args = ap.parse_args()

    setup_sweep_dir()

    seeds = [args.only_seed] if args.only_seed is not None else SEEDS
    if args.only_seed is not None and args.only_seed not in SEEDS:
        print(f"[warn] --only-seed {args.only_seed} は標準 SEEDS {SEEDS} に無い "
              f"(集計対象外になり得る)", flush=True)

    for seed in seeds:
        run_seed(seed, smoke=args.smoke)

    print(f"[sweep d80m] 指定 seed 完了: {seeds}", flush=True)


if __name__ == "__main__":
    main()

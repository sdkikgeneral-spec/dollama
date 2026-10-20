# -*- coding: utf-8 -*-
"""Phase 4-A 実ペア増 seed sweep: base(#1 plain hard CE) vs a(A2 identity 混合) の
diverse-val 生成 F1/Jaccard + identity retention 優位が seed 横断で頑健かを、
実ペア増スケール (a12k / a25k) ごとに確定する。

dollma_b10k_seedsweep.py (施策B 10000版) と同手続き・同 seed (20260620/20260621/42/7)・
同ハイパラ (6ep FP32 cosine) で、差分は次の 3 点のみ:
  - a arm が --train-file ではなく --identity 駆動 (A2 混合訓練・identity_cond 行を混ぜる)。
  - スケール別 sweep dir を --scale {a12k,a25k} で切替 (_seedsweep_a12k/ / _seedsweep_a25k/)。
  - base arm はスケール非依存 (pairs.train.jsonl はスケール共通) なので、seed 別 base npz は
    1 度計算したら両スケール sweep dir の _results/ に流用配置する (再訓練回数を削減)。

train_bitnet.py の --identity 混合パスは固定名 pairs.identity.{train,val}.jsonl を
--data-dir から読む (L2231 付近)。よってスケール別の Phase 1 成果物
pairs.identity.{train,val}.a{12k,25k}.jsonl を sweep dir 内で正準名にコピーして置く
(b10k sweep が pairs.train.diverse_b10k.jsonl を sweep dir に複製したのと同じ手)。
#1 の pairs.{train,val}.jsonl と凍結 eval pairs.eval_diverse_{a,b}.jsonl も sweep dir に
複製する (byte/sha 一致を確認 — 過去 b10k 結果と比較可能にするため)。

両 arm とも plain/FP32・val (pairs.val.jsonl) も eval_diverse_a/b も完全共通なので
per-sample paired delta = a − base に train 構成以外のバイアスは乗らない。
a arm のみ identity_cond 行が混ざる (= 実ペア増の効果) のが唯一の差分。

出力は data/bitnet/_seedsweep_a{12k,25k}/ 配下のみ (本番 bitnet_dense*/golden /
本番 bitnet_dense_identity* / 既存 _seedsweep_b*/ / C-4 _seedsweep/ を一切上書きしない)。
identity arm の重みは sweep dir 内 bitnet_dense_identity{,_fp32}.safetensors に出る
(--data-dir が sweep dir なので本番 data/bitnet/bitnet_dense_identity* は無傷)。

使い方:
  # 1 seed だけ走らせる (配管検証・小出し運用):
  py -3.12 scripts/dollma_a_seedsweep.py --scale a12k --only-seed 20260620
  # 全 seed (無指定):
  py -3.12 scripts/dollma_a_seedsweep.py --scale a12k

冪等: _results/eval_persample_{arm}_{seed}.npz が在れば該当 arm を skip (再訓練しない)。
base npz は両スケールで共有するので、他スケール sweep dir に base_{seed} があれば
そこからコピーしてくる (再訓練を省く)。
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

# b10k sweep と一致させた 4 seed・6ep paired。
SEEDS = [20260620, 20260621, 42, 7]
SCALES = ["a12k", "a25k"]
PY = sys.executable  # 起動した python をそのまま使う

# 両 arm 共通ハイパラ (b10k sweep COMMON と同設定・FP32 cosine 6ep)。
COMMON = ["--epochs", "6", "--loss-mode", "tags", "--batch-size", "32", "--lr", "3e-4"]
EVAL_DEVICE = "cuda"  # greedy 生成を GPU で高速化 (base/a 同一 device → paired delta 無バイアス)


def sd_for(scale):
    return os.path.join(DATA, f"_seedsweep_{scale}")


def res_for(scale):
    return os.path.join(sd_for(scale), "_results")


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
    """src を dst へコピーし sha を表示 (byte/hash 一致追跡用)。既存で sha 一致なら skip。"""
    if os.path.exists(dst) and _sha256(src) == _sha256(dst):
        print(f"[setup] {label}: 既存 (sha 一致) -> skip {os.path.basename(dst)}", flush=True)
        return
    shutil.copyfile(src, dst)
    print(f"[setup] {label}: {os.path.basename(src)} -> {os.path.basename(dst)} "
          f"sha={_sha256(dst)[:12]}", flush=True)


def setup_sweep_dir(scale):
    """sweep dir に train_bitnet.py が固定名で読むファイル一式を正準名で配置する。

      vocab.json                                 ← 語彙 (train_bitnet が data-dir/vocab.json を読む)
      pairs.train.jsonl / pairs.val.jsonl        ← #1 本線 (スケール共通)
      pairs.identity.train.jsonl / .val.jsonl    ← Phase 1 の a{scale} 版を正準名へ
      pairs.eval_diverse_a.jsonl / _b.jsonl      ← 凍結物差し (b10k と byte 一致)
    """
    sd = sd_for(scale)
    os.makedirs(res_for(scale), exist_ok=True)
    # 語彙 (train_bitnet.py は --data-dir/vocab.json を唯一ソースに読む・b10k sweep も複製済)。
    _copy_canonical(os.path.join(DATA, "vocab.json"),
                    os.path.join(sd, "vocab.json"), "vocab")
    # #1 本線 train/val (スケール共通)。
    _copy_canonical(os.path.join(DATA, "pairs.train.jsonl"),
                    os.path.join(sd, "pairs.train.jsonl"), "base train")
    _copy_canonical(os.path.join(DATA, "pairs.val.jsonl"),
                    os.path.join(sd, "pairs.val.jsonl"), "val (共通)")
    # Phase 1 の identity ペア (スケール別) を固定名にコピー。
    _copy_canonical(os.path.join(DATA, f"pairs.identity.train.{scale}.jsonl"),
                    os.path.join(sd, "pairs.identity.train.jsonl"),
                    f"identity train ({scale})")
    _copy_canonical(os.path.join(DATA, f"pairs.identity.val.{scale}.jsonl"),
                    os.path.join(sd, "pairs.identity.val.jsonl"),
                    f"identity val ({scale})")
    # 凍結 diverse-val (b10k sweep と byte 一致の物差し)。
    _copy_canonical(os.path.join(DATA, "pairs.eval_diverse_a.jsonl"),
                    os.path.join(sd, "pairs.eval_diverse_a.jsonl"), "eval_diverse_a")
    _copy_canonical(os.path.join(DATA, "pairs.eval_diverse_b.jsonl"),
                    os.path.join(sd, "pairs.eval_diverse_b.jsonl"), "eval_diverse_b")


def train(scale, arm, seed):
    """arm in {'base','a'} を seed で 6ep 訓練 (出力は sweep dir の固定名)。

      base -> bitnet_dense{,_fp32}.safetensors          (--train-file 無・--identity 無)
      a    -> bitnet_dense_identity{,_fp32}.safetensors (--identity 駆動 = A2 混合)
    """
    sd = sd_for(scale)
    cmd = [PY, TRAIN, "--data-dir", sd, "--seed", str(seed)] + COMMON
    if arm == "a":
        cmd += ["--identity"]
    run(cmd)


def eval_arm(scale, arm, seed, weights):
    """weights を diverse_a/b + identity retention で採点し per-sample npz + report を
    sweep dir の _results/ へ seed 別名で退避する。"""
    sd = sd_for(scale)
    res = res_for(scale)
    name = f"{arm}_{seed}"
    cmd = [PY, TRAIN, "--eval-only", "--data-dir", sd,
           "--weights", weights, "--eval-name", name,
           "--dump-persample", "--device", EVAL_DEVICE]
    run(cmd)
    for fn in (f"eval_report_{name}.json", f"eval_persample_{name}.npz"):
        src = os.path.join(sd, fn)
        if os.path.exists(src):
            shutil.move(src, os.path.join(res, fn))
            print(f"[move] {fn} -> _results/", flush=True)


def try_share_base(scale, seed):
    """base はスケール非依存。他スケール sweep dir の _results に base_{seed} の
    npz+report が在れば、それを当該スケールへコピーして再訓練を省く。

    返り値: True なら共有でき base 計算をスキップしてよい。
    """
    res = res_for(scale)
    want_npz = os.path.join(res, f"eval_persample_base_{seed}.npz")
    if os.path.exists(want_npz):
        return True  # 既にある
    for other in SCALES:
        if other == scale:
            continue
        ores = res_for(other)
        onpz = os.path.join(ores, f"eval_persample_base_{seed}.npz")
        orep = os.path.join(ores, f"eval_report_base_{seed}.json")
        if os.path.exists(onpz):
            os.makedirs(res, exist_ok=True)
            shutil.copyfile(onpz, want_npz)
            if os.path.exists(orep):
                shutil.copyfile(orep, os.path.join(res, f"eval_report_base_{seed}.json"))
            print(f"[base-share] seed {seed}: {other} の base npz を {scale} に流用 "
                  f"(再訓練省略)", flush=True)
            return True
    return False


def run_seed(scale, seed):
    sd = sd_for(scale)
    res = res_for(scale)

    # ---- base (#1 plain hard CE・スケール非依存) ----
    if os.path.exists(os.path.join(res, f"eval_persample_base_{seed}.npz")):
        print(f"[{scale} seed {seed}] base eval 既存 -> スキップ", flush=True)
    elif try_share_base(scale, seed):
        pass  # 他スケールから流用済み
    else:
        train(scale, "base", seed)
        base_w = os.path.join(sd, "bitnet_dense_fp32.safetensors")
        eval_arm(scale, "base", seed, base_w)

    # ---- a (A2 identity 混合・スケール依存) ----
    if os.path.exists(os.path.join(res, f"eval_persample_a_{seed}.npz")):
        print(f"[{scale} seed {seed}] a eval 既存 -> スキップ", flush=True)
    else:
        train(scale, "a", seed)
        a_w = os.path.join(sd, "bitnet_dense_identity_fp32.safetensors")
        eval_arm(scale, "a", seed, a_w)

    print(f"[{scale} seed {seed} 完了] base/a eval npz を _results/ に退避済み", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", required=True, choices=SCALES,
                    help="実ペア増スケール (sweep dir を切替)")
    ap.add_argument("--only-seed", type=int, default=None,
                    help="指定 seed 1 つだけ実行 (無指定で SEEDS 全件)")
    args = ap.parse_args()

    scale = args.scale
    setup_sweep_dir(scale)

    seeds = [args.only_seed] if args.only_seed is not None else SEEDS
    if args.only_seed is not None and args.only_seed not in SEEDS:
        print(f"[warn] --only-seed {args.only_seed} は標準 SEEDS {SEEDS} に無い "
              f"(集計対象外になり得る)", flush=True)

    for seed in seeds:
        run_seed(scale, seed)

    print(f"[sweep {scale}] 指定 seed 完了: {seeds}", flush=True)


if __name__ == "__main__":
    main()

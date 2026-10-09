# -*- coding: utf-8 -*-
"""dollma_f0b_z4_run.py — F-0b Z-4 本走ランナー (起動コマンドをここで固定する)。

サブコマンド (冪等: 出力があれば skip)。--dry-run は実行・コピーをせずコマンドだけ表示する。
  prepare  main の data/ から scratch へ入力を読み取りコピー (pairs.train.jsonl も本物) し sha256 を照合・記録
  placebo  P_s の勝者ペア 8 本を生成
  train    R'_s 8 本 + P_s 8 本 + 決定論確認 1 本 (R'_20260620 を再訓練) を訓練。
           全て --sft-data と --sft-init を明示。訓練後に train stats の来歴 sha256 を照合
  eval     canon / r_ref / R'_s / P_s の 18 本を --device cuda --seed 20260620 で評価
           (--dump-persample --dump-retention-detail)
  analyze  解析 (dollma_f0b_z4_analyze.py)
  all      上を順に

main の data/ と models/ には書かない (読むだけ)。書き込みは scratch (docs/logs/f0b-z4/_scratch_data) のみ。
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCR = os.path.join(ROOT, "docs", "logs", "f0b-z4", "_scratch_data")
TRAIN = os.path.join(ROOT, "scripts", "train_bitnet.py")
PLACEBO = os.path.join(ROOT, "scripts", "dollma_f0b_z4_placebo_pairs.py")
ANALYZE = os.path.join(ROOT, "scripts", "dollma_f0b_z4_analyze.py")
DEFAULT_MAIN_DATA = r"E:\Develop\Projects\dollama\data"
SEEDS = [20260620, 20260621, 42, 7, 20260622, 20260623, 1, 2]   # analyze.py の SEEDS と同一 (assert 済)
EVAL_SEED = 20260620

# 必須の sha256 照合 (不一致なら停止)
EXPECT = {
    "bitnet_dense_fp32.safetensors": "5043772dd13f6fd854d9909191960d8e593ea830638ebd482e8d691df9ad0b61",  # 正典
    "bitnet_dense_sft_fp32.safetensors": "3b2e2181155122ef94a4cfd8d767696728076344ff642726e4b1cebb8eab8e15",  # 既存 SFT (R)
    "sft_bestofn.jsonl": "032470f2d4bff6e42489290a6d53a28bbe7ad8436dc1884281b0517055a57b0a",
    "candidates_bestofn.jsonl": "a38fba8b9dadec8ec68e8a1d94fc3813653fe932e6cbca2623a7007f728142c5",
}
# (main 側の相対パス, scratch 側の相対パス)
COPIES = [
    ("bitnet/vocab.json", "vocab.json"),
    ("bitnet/pairs.train.jsonl", "pairs.train.jsonl"),     # SFT 経路でも syn_train として読まれる (本物をコピー)
    ("bitnet/pairs.val.jsonl", "pairs.val.jsonl"),
    ("bitnet/pairs.eval_diverse_a.jsonl", "pairs.eval_diverse_a.jsonl"),
    ("bitnet/pairs.eval_diverse_b.jsonl", "pairs.eval_diverse_b.jsonl"),
    ("bitnet/pairs.identity.val.jsonl", "pairs.identity.val.jsonl"),
    ("bitnet/bitnet_dense_fp32.safetensors", "bitnet_dense_fp32.safetensors"),
    ("bitnet/bitnet_dense_sft_fp32.safetensors", os.path.join("ref", "bitnet_dense_sft_fp32.safetensors")),
    ("rollouts/sft_bestofn.jsonl", "sft_bestofn.jsonl"),
    ("rollouts/candidates_bestofn.jsonl", "candidates_bestofn.jsonl"),
]
# 本採用レシピ (train_stats_sft.json と同一・warmup 5 明示)
RECIPE = ["--epochs", "3", "--batch-size", "32", "--lr", "2e-5", "--weight-decay", "0.01", "--warmup", "5",
          "--max-len", "64", "--loss-mode", "tags"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def sh(cmd, dry, log=None):
    print("$ " + " ".join(f'"{c}"' if " " in c else c for c in cmd), flush=True)
    if dry:
        return ""
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = p.stdout + p.stderr
    if log:
        os.makedirs(os.path.dirname(log), exist_ok=True)
        with open(log, "w", encoding="utf-8") as f:
            f.write(out)
    if p.returncode != 0:
        print(out[-3000:])
        raise SystemExit(f"[停止] rc={p.returncode}: {cmd[:3]}")
    return out


def prepare(a):
    recs = []
    for src_rel, dst_rel in COPIES:
        src = os.path.join(a.main_data, *src_rel.split("/"))
        dst = os.path.join(SCR, dst_rel)
        print(f"[prepare] {src} -> {dst}")
        if a.dry_run:
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.isfile(dst):
            shutil.copy(src, dst)
        h = sha(dst)
        key = os.path.basename(dst_rel)
        if key in EXPECT and h != EXPECT[key]:
            raise SystemExit(f"[停止] sha256 不一致 {key}: {h} != {EXPECT[key]}")
        if sha(src) != h:
            raise SystemExit(f"[停止] コピーが main と不一致: {dst_rel}")
        recs.append(f"{h} *{dst_rel}")
    if not a.dry_run:
        with open(os.path.join(SCR, "_input_sha256.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(recs) + "\n")
        print("\n".join(recs))


def placebo(a):
    for s in SEEDS:
        out = os.path.join(SCR, "placebo", f"placebo_s{s}.jsonl")
        if os.path.isfile(out) and not a.dry_run:
            continue
        sh([sys.executable, PLACEBO, "--candidates", os.path.join(SCR, "candidates_bestofn.jsonl"),
            "--winners", os.path.join(SCR, "sft_bestofn.jsonl"), "--seed", str(s), "--out", out], a.dry_run)


def train_cmd(sft_data, tag, seed):
    return [sys.executable, TRAIN, "--sft-rejection", "--sft-data", sft_data,
            "--sft-init", os.path.join(SCR, "bitnet_dense_fp32.safetensors"),
            "--data-dir", SCR, "--out-dir", os.path.join(SCR, "weights"), "--out-tag", tag,
            "--seed", str(seed), "--deterministic"] + RECIPE


def verify_train(tag, sft_data, seed, log):
    st = json.load(open(os.path.join(SCR, "weights", f"train_stats_sft{tag}.json"), encoding="utf-8"))
    pv = st["sft_provenance"]
    assert pv["sft_data"]["sha256"] == sha(sft_data), f"{tag}: sft_data sha 不一致"
    assert pv["sft_init"]["sha256"] == EXPECT["bitnet_dense_fp32.safetensors"], f"{tag}: sft_init sha 不一致"
    assert st["deterministic"]["enabled"] and not st["deterministic"]["warn_only"], f"{tag}: 決定論設定"
    hp = st["hyperparams"]
    assert (hp["epochs"], hp["batch_size"], hp["lr"], hp["weight_decay"], hp["warmup"], hp["max_len"],
            hp["loss_mode"]) == (3, 32, 2e-5, 0.01, 5, 64, "tags"), f"{tag}: hyperparams {hp}"
    assert st["seed"] == seed and st["data"]["train"] == 400, f"{tag}: seed/train"
    txt = open(log, encoding="utf-8").read()
    assert "train=400" in txt and "ja=184" in txt and "en=216" in txt, f"{tag}: train=400 (ja=184 en=216) 行なし"
    assert st["result"]["train_time_s"] > 0.1


def train(a):
    wdir = os.path.join(SCR, "weights")
    jobs = []
    for s in SEEDS:
        jobs.append((f"_r_s{s}", os.path.join(SCR, "sft_bestofn.jsonl"), s))
        jobs.append((f"_p_s{s}", os.path.join(SCR, "placebo", f"placebo_s{s}.jsonl"), s))
    jobs.append((f"_detchk_r_s{SEEDS[0]}", os.path.join(SCR, "sft_bestofn.jsonl"), SEEDS[0]))
    for tag, data, s in jobs:
        out = os.path.join(wdir, f"bitnet_dense_sft{tag}_fp32.safetensors")
        log = os.path.join(SCR, "logs", f"train{tag}.log")
        if os.path.isfile(out) and not a.dry_run:
            continue
        sh(train_cmd(data, tag, s), a.dry_run, log=log)
        if not a.dry_run:
            verify_train(tag, data, s, log)
    if not a.dry_run:
        h1 = sha(os.path.join(wdir, f"bitnet_dense_sft_r_s{SEEDS[0]}_fp32.safetensors"))
        h2 = sha(os.path.join(wdir, f"bitnet_dense_sft_detchk_r_s{SEEDS[0]}_fp32.safetensors"))
        print(f"[決定論確認] R'_{SEEDS[0]} 2 回訓練 sha256 一致 = {h1 == h2} ({h1[:16]} / {h2[:16]})")
        if h1 != h2:
            raise SystemExit("[停止] 決定論確認 不一致")


def eval_(a):
    wdir = os.path.join(SCR, "weights")
    targets = [("canon", os.path.join(SCR, "bitnet_dense_fp32.safetensors")),
               ("r_ref", os.path.join(SCR, "ref", "bitnet_dense_sft_fp32.safetensors"))]
    for arm in ("r", "p"):
        for s in SEEDS:
            targets.append((f"{arm}_s{s}", os.path.join(wdir, f"bitnet_dense_sft_{arm}_s{s}_fp32.safetensors")))
    for name, w in targets:
        od = os.path.join(SCR, "eval")
        if os.path.isfile(os.path.join(od, f"eval_persample_{name}.npz")) and not a.dry_run:
            continue
        sh([sys.executable, TRAIN, "--eval-only", "--weights", w, "--eval-name", name, "--data-dir", SCR,
            "--out-dir", od, "--dump-persample", "--dump-retention-detail",
            "--device", "cuda", "--seed", str(EVAL_SEED)], a.dry_run,
           log=os.path.join(SCR, "logs", f"eval_{name}.log"))


def analyze(a):
    sh([sys.executable, ANALYZE, "--npz-dir", os.path.join(SCR, "eval"),
        "--pairs-a", os.path.join(SCR, "pairs.eval_diverse_a.jsonl"),
        "--pairs-b", os.path.join(SCR, "pairs.eval_diverse_b.jsonl"),
        "--winners", os.path.join(SCR, "sft_bestofn.jsonl"),
        "--placebo-dir", os.path.join(SCR, "placebo"), "--vocab", os.path.join(SCR, "vocab.json"),
        "--out", os.path.join(ROOT, "docs", "logs", "f0b-z4", "z4_result.json")], a.dry_run)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["prepare", "placebo", "train", "eval", "analyze", "all"])
    ap.add_argument("--main-data", default=DEFAULT_MAIN_DATA)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    steps = {"prepare": prepare, "placebo": placebo, "train": train, "eval": eval_, "analyze": analyze}
    for st in (list(steps) if a.step == "all" else [a.step]):
        print(f"=== {st} ===")
        steps[st](a)


if __name__ == "__main__":
    main()

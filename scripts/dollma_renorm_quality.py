# -*- coding: utf-8 -*-
"""dollama Phase 4 施策 F / Q-2 Step① — quality 再正規化 (z-score → sigmoid)。

現行 quality は `dollma_score_quality_v4.py` の「raw/10 クランプ [0,1]」写像で作られ、
raw_waifu (mean0.74 / std0.81 / range[-1.71,1.78]) が [0,0.18] に潰れて std0.0753 しか
立たなかった (F-0a 信号ゲートで学習信号が弱い一因)。信号自体は raw_waifu に採取済みなので、
写像だけを z-score → sigmoid に差し替えて quality を [0,1] 全域へ広げる。

写像:
  z = (raw_waifu - mean) / std
  quality = 1 / (1 + exp(-z * k))      (k = sigmoid 温度・既定 1.5)

  - mean/std は省略時 train+val 全 raw_waifu から算出 (母集団統計)。
  - raw = mean で quality = 0.5・単調増加・出力は開区間 (0,1) ⊂ [0,1]。

入力: data/scorer/scorer.{train,val}.jsonl (各行に raw_waifu 済み)。
出力: 同ファイルの quality を再計算して上書き。quality_waifu / raw_waifu 列は保持 (消さない)。
退避: 上書き前に scorer.{split}.q018.bak.jsonl へコピー (既存 bak は上書きしない=冪等)。
provenance: data/scorer/scorer_quality_renorm.json に mean/std/k/写像式/分布/ヒストグラムを記録。

scorer_net.safetensors 等のアーティファクトは触らない (再訓練は Step② で別途)。
"""

import argparse
import json
import math
import os
import shutil
import statistics
import time


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def renorm_quality(raw, mean, std, k):
    """z-score → sigmoid 写像 (純関数・test はこれを直接叩く)。

    z = (raw - mean) / std, quality = 1/(1+exp(-z*k))。
    単調増加・raw=mean で 0.5・出力は開区間 (0,1)。std<=0 は 0 除算回避で z=0 とする。
    """
    if std <= 0.0:
        z = 0.0
    else:
        z = (raw - mean) / std
    return 1.0 / (1.0 + math.exp(-z * k))


def histogram(values, nbins=10, lo=0.0, hi=1.0):
    bins = [0] * nbins
    width = (hi - lo) / nbins
    for v in values:
        k = int((v - lo) / width)
        if k >= nbins:
            k = nbins - 1
        if k < 0:
            k = 0
        bins[k] += 1
    return [(round(lo + i * width, 3), round(lo + (i + 1) * width, 3), bins[i]) for i in range(nbins)]


def main(argv=None):
    ap = argparse.ArgumentParser(description="raw_waifu を z-score→sigmoid で quality に再正規化")
    ap.add_argument("--data-dir", default="data/scorer")
    ap.add_argument("--k", type=float, default=1.5, help="sigmoid 温度 (既定 1.5)")
    ap.add_argument("--mean", type=float, default=None,
                    help="z-score 平均 (省略時 train+val 全 raw_waifu から算出)")
    ap.add_argument("--std", type=float, default=None,
                    help="z-score 標準偏差 (省略時 train+val 全 raw_waifu から算出)")
    args = ap.parse_args(argv)

    splits = ("train", "val")
    paths = {s: os.path.join(args.data_dir, f"scorer.{s}.jsonl") for s in splits}

    # 1) 全 raw_waifu を集めて母集団統計を出す (mean/std 省略時)。
    rows_by_split = {}
    all_raw = []
    for s in splits:
        rows = read_jsonl(paths[s])
        rows_by_split[s] = rows
        for r in rows:
            if "raw_waifu" not in r:
                raise KeyError(f"{paths[s]} の行に raw_waifu が無い: {r.get('image')}")
            all_raw.append(float(r["raw_waifu"]))

    mean = args.mean if args.mean is not None else statistics.mean(all_raw)
    std = args.std if args.std is not None else statistics.pstdev(all_raw)
    k = args.k
    print(f"z-score 統計: mean={mean:.6f} std={std:.6f} k={k}  (n_raw={len(all_raw)})")

    # 2) 再正規化して quality を上書き (quality_waifu / raw_waifu は保持)。
    all_q = []
    for s in splits:
        rows = rows_by_split[s]
        for r in rows:
            r["quality"] = renorm_quality(float(r["raw_waifu"]), mean, std, k)
            all_q.append(r["quality"])
        # 退避 (冪等: 既存 bak は上書きしない)。
        bak = paths[s].replace(".jsonl", ".q018.bak.jsonl")
        if not os.path.exists(bak):
            shutil.copy2(paths[s], bak)
            print(f"{s}: 退避 {os.path.basename(bak)}")
        else:
            print(f"{s}: 退避 skip (既存 {os.path.basename(bak)})")
        write_jsonl(paths[s], rows)
        print(f"{s}: {len(rows)} 行の quality を再正規化")

    # 3) provenance + 分布レポート。
    std_q = statistics.pstdev(all_q)
    n_clamp0 = sum(1 for q in all_q if q <= 0.0)
    q_stats = {
        "min": round(min(all_q), 6), "median": round(statistics.median(all_q), 6),
        "max": round(max(all_q), 6), "mean": round(statistics.mean(all_q), 6),
        "std": round(std_q, 6), "n_clamped_to_0": n_clamp0,
    }
    report = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "step": "Q-2 Step1 quality 再正規化 (z-score -> sigmoid)",
        "source_column": "raw_waifu",
        "mapping": "quality = 1/(1+exp(-((raw_waifu - mean)/std) * k))",
        "mean": round(mean, 6),
        "std": round(std, 6),
        "k": k,
        "mean_std_source": ("引数指定" if args.mean is not None and args.std is not None
                            else "train+val 全 raw_waifu (母集団)"),
        "n_rows": len(all_q),
        "raw_waifu_stats": {
            "min": round(min(all_raw), 6), "median": round(statistics.median(all_raw), 6),
            "max": round(max(all_raw), 6), "mean": round(statistics.mean(all_raw), 6),
            "std": round(statistics.pstdev(all_raw), 6),
        },
        "quality_renorm_stats": q_stats,
        "histogram_quality": histogram(all_q, nbins=10, lo=0.0, hi=1.0),
        "backup": "scorer.{train,val}.q018.bak.jsonl (旧 [0,0.18] quality を退避)",
        "preserved_columns": ["quality_waifu", "raw_waifu"],
        "note": "scorer_net.safetensors 等の再訓練は Step2 で別途。deepghs アンサンブルは raw/quality_waifu 残置で後付け可能。",
    }
    out = os.path.join(args.data_dir, "scorer_quality_renorm.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n=== quality 再正規化後 分布 ===")
    print(json.dumps(q_stats, ensure_ascii=False))
    print("histogram (lo,hi,count):")
    for lo, hi, c in report["histogram_quality"]:
        print(f"  [{lo:.2f},{hi:.2f}) {'#'*c} {c}")

    # [0,0.18]→[0,1] へ広がったかの判定。
    spread_ok = std_q >= 0.15
    full_range = q_stats["max"] > 0.5 and q_stats["min"] < 0.5  # 中央より上下両側を使う
    verdict = ("拡張成功: std={:.4f}>=0.15 かつ [0,1] 全域を使用".format(std_q)
               if spread_ok and full_range
               else "拡張不十分: std={:.4f} (目標>=0.15)".format(std_q))
    print(f"\n判定: {verdict}")
    print(f"  0 クランプ数: {n_clamp0} (旧 [0,0.18] 潰れからの改善指標)")
    print(f"レポート: {out}")


if __name__ == "__main__":
    main()

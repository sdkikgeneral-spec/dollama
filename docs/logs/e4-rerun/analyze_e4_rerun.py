# -*- coding: utf-8 -*-
# 依存: scripts/archives/dollma_e2_t2_analyze.py の compare_arms (repo root から実行すること)。
"""E-4 再走の集計 (record-writer・2026-10-22)。repo root で `py -3 docs/logs/e4-rerun/analyze_e4_rerun.py` 。
3 軸は scripts/archives/dollma_e2_t2_analyze.py の compare_arms を無改変で使う (CI は正規近似・paired bootstrap ではない)。
ノイズ床 = prompt 別の標本 sd (n-1) の平均 (grid_summary.csv の std 列 = 母 sd とは別物)。"""
import sys, csv, io, contextlib, statistics as st
sys.path.insert(0, "scripts/archives")
from dollma_e2_t2_analyze import compare_arms, load_rows
R = "docs/logs/e4-rerun/"
A = load_rows(R + "scale_013025/grid_results.csv")
B = load_rows(R + "scale_018215/grid_results.csv")
P = ["illustrious-xl", "illustrious-xl-v2", "noobai-xl", "animagine-xl-4"]
M = ["recall", "soft_recall", "worst_anatomy"]
def f(x): return float(x)
def floor(rows, m):
    by = {}
    for r in rows: by.setdefault(r["prompt_id"], []).append(f(r[m]))
    return sum(st.stdev(v) for v in by.values()) / len(by)
print("== 健全性")
for nm, rows in (("013025", A), ("018215", B)):
    print(nm, len(rows), "exit0", sum(r["exit"] == "0" for r in rows), "log_ok", sum(r["log_ok"] == "True" for r in rows),
          "reason空", sum(r["log_reason"] == "" for r in rows))
print("== preset 平均 (16 枚) / ノイズ床 = prompt 別 標本 sd(n-1) の平均")
for m in M:
    for p in P:
        a = [r for r in A if r["preset"] == p]; b = [r for r in B if r["preset"] == p]
        print(f"{m:14s} {p:18s} 013025 mean={st.mean(f(r[m]) for r in a):.4f} floor={floor(a,m):.4f} | 018215 mean={st.mean(f(r[m]) for r in b):.4f} floor={floor(b,m):.4f}")
def run(ref, oth, m, label):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = compare_arms(ref, oth, m, label)
    return res, buf.getvalue()
def show(res):
    return (f"n={res['n']} d={res['delta_mean']:+.4f} CI=[{res['ci_lo']:+.4f},{res['ci_hi']:+.4f}] "
            f"①={res['sign_consistent']} ②={res['exceeds_noise']} (floor={res['noise_floor']:.4f}) ③={res['ci_excludes_zero']} → {res['verdict']}")
print("== scaling: 0.18215 − 0.13025 (ref=0.13025 アーム)")
for m in M:
    for p in P:
        a = [r for r in A if r["preset"] == p]; b = [r for r in B if r["preset"] == p]
        res, log = run(a, b, m, p)
        signs = log.split("signs=")[1].split(")")[0]
        print(f"{m:14s} {p:18s} {show(res)} signs={signs}")
print("== scaling: ref=0.18215 アームに替えた場合の ② (delta は符号反転のみ)")
for m in M:
    for p in P:
        a = [r for r in A if r["preset"] == p]; b = [r for r in B if r["preset"] == p]
        res, _ = run(b, a, m, p)
        print(f"{m:14s} {p:18s} ②={res['exceeds_noise']} floor={res['noise_floor']:.4f} |d|={abs(res['delta_mean']):.4f}")
print("== preset: X − illustrious-xl (同 scaling 内・prompt×seed で paired・ref=illustrious-xl)")
def relabel(rows): return [dict(r, preset="_") for r in rows]
for nm, rows in (("013025", A), ("018215", B)):
    ref = relabel([r for r in rows if r["preset"] == "illustrious-xl"])
    for m in M:
        for p in P[1:]:
            oth = relabel([r for r in rows if r["preset"] == p])
            res, log = run(ref, oth, m, p)
            signs = log.split("signs=")[1].split(")")[0]
            print(f"{nm} {m:14s} {p:18s} {show(res)} signs={signs}")

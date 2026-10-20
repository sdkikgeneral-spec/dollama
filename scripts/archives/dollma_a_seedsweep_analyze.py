# -*- coding: utf-8 -*-
"""Phase 4-A 実ペア増 seed sweep 集計・判定 (dollma_b10k_seedsweep_analyze.py と同形式
+ identity retention 列を追加)。

_seedsweep_{scale}/_results/eval_persample_{base,a}_{seed}.npz と
eval_report_{...}.json を読み、seed x {#1(base),A2(a)} x {diverse_a,diverse_b} の
macro F1/Jaccard 表、各 seed の paired delta (a − #1)、across-seed 平均±sd、
paired bootstrap CI、paired t を出す。さらに identity retention (a arm のみ・
eval_report の identity_retention.mean_retention) を seed 別に表示する。

判定軸 (b10k と同一):
  (a) 全 seed で delta の符号が + で一貫するか
  (b) delta 平均が #1 自身の seed 分散帯 (base を seed 間で比べた band) を超えるか
  (c) 各 seed の paired bootstrap 95%CI が 0 を含まないか

per-sample 配列は rows と同順・NaN=skip/未定義。paired 解析は両 arm とも非 NaN の
位置のみ採用。

使い方:
  py -3.12 scripts/dollma_a_seedsweep_analyze.py --scale a12k
"""

import argparse
import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SEEDS = [20260620, 20260621, 42, 7]
SETS = ["eval_diverse_a", "eval_diverse_b"]
METRICS = ["f1", "jaccard"]
BOOT = 10000
RNG = np.random.default_rng(20260623)


def res_for(scale):
    return os.path.join(ROOT, "data", "bitnet", f"_seedsweep_{scale}", "_results")


def load_persample(res, arm, seed):
    p = os.path.join(res, f"eval_persample_{arm}_{seed}.npz")
    if not os.path.exists(p):
        return None
    # 自作 sweep が直前に書いた信頼済みファイルのみ読む。数値配列だけ参照し object
    # (文字列 provenance) フィールドには触れないので allow_pickle=False で十分。
    return np.load(p, allow_pickle=False)


def load_report(res, arm, seed):
    p = os.path.join(res, f"eval_report_{arm}_{seed}.json")
    if not os.path.exists(p):
        return {}
    return json.load(open(p, encoding="utf-8"))


def load_macro(res, arm, seed):
    """eval_report の macro F1/Jaccard (確認用・per-sample nanmean と一致するはず)。"""
    d = load_report(res, arm, seed)
    gm = d.get("generation_setmetrics", {})
    out = {}
    for s in SETS:
        r = gm.get(s)
        if r and not r.get("skipped"):
            out[s] = {"f1": r["macro"]["f1"], "jaccard": r["macro"]["jaccard"],
                      "precision": r["macro"]["precision"],
                      "recall": r["macro"]["recall"]}
    return out


def load_retention(res, arm, seed):
    """identity retention (a arm に意味あり・base は identity_cond 0 件で n=0 想定)。"""
    d = load_report(res, arm, seed)
    rr = d.get("identity_retention")
    if not rr:
        return None
    return {"mean": rr.get("mean_retention"), "n": rr.get("n_cases")}


def paired_arrays(base_npz, a_npz, s, metric):
    """両 arm で非 NaN の位置のみ揃えた (base, a) 配列を返す。"""
    b0 = base_npz[f"{s}__{metric}"].astype(float)
    b1 = a_npz[f"{s}__{metric}"].astype(float)
    assert len(b0) == len(b1), (len(b0), len(b1))
    mask = ~np.isnan(b0) & ~np.isnan(b1)
    return b0[mask], b1[mask]


def paired_bootstrap_ci(b, d, n_boot=BOOT, alpha=0.05):
    """paired delta = mean(d - b) の bootstrap 95%CI (サンプルをペアで再標本)。"""
    diff = d - b
    n = len(diff)
    idx = RNG.integers(0, n, size=(n_boot, n))
    boot_means = diff[idx].mean(axis=1)
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(diff.mean()), float(lo), float(hi)


def paired_t(b, d):
    """paired t 統計量と両側 p。"""
    from math import sqrt
    diff = d - b
    n = len(diff)
    m = diff.mean()
    sd = diff.std(ddof=1)
    if sd == 0:
        return float("inf") if m != 0 else 0.0, 0.0
    t = m / (sd / sqrt(n))
    try:
        from scipy import stats
        p = 2 * stats.t.sf(abs(t), df=n - 1)
    except Exception:
        from math import erfc
        p = erfc(abs(t) / sqrt(2))
    return float(t), float(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", required=True, choices=["a12k", "a25k"])
    args = ap.parse_args()
    scale = args.scale
    res = res_for(scale)

    # ---- 1) macro F1/Jaccard 表 (per-sample nanmean で算出・report と突合) ----
    table = {}  # (arm,seed,set) -> {metric: nanmean}
    for arm in ("base", "a"):
        for seed in SEEDS:
            npz = load_persample(res, arm, seed)
            if npz is None:
                continue
            macro = load_macro(res, arm, seed)
            for s in SETS:
                cell = {}
                for m in METRICS + ["precision", "recall"]:
                    key = f"{s}__{m}"
                    if key in npz:
                        cell[m] = float(np.nanmean(npz[key].astype(float)))
                if s in macro:
                    for m in METRICS:
                        if abs(cell.get(m, np.nan) - macro[s][m]) > 1e-6:
                            cell[f"_mismatch_{m}"] = (cell.get(m), macro[s][m])
                table[(arm, seed, s)] = cell

    print("=" * 96)
    print(f"Phase 4-A 実ペア増 seed sweep [{scale}]: "
          f"seed x {{#1(base),A2(a)}} x {{diverse_a,diverse_b}} macro 表")
    print("=" * 96)
    hdr = f"{'seed':>9} {'arm':>5} {'set':>16} {'F1':>8} {'Jaccard':>8} {'prec':>8} {'rec':>8}"
    print(hdr)
    print("-" * len(hdr))
    for seed in SEEDS:
        for arm in ("base", "a"):
            for s in SETS:
                c = table.get((arm, seed, s))
                if not c:
                    print(f"{seed:>9} {arm:>5} {s:>16}   (missing)")
                    continue
                print(f"{seed:>9} {arm:>5} {s:>16} {c.get('f1',float('nan')):8.4f} "
                      f"{c.get('jaccard',float('nan')):8.4f} "
                      f"{c.get('precision',float('nan')):8.4f} "
                      f"{c.get('recall',float('nan')):8.4f}")

    mm = [(k, c) for k, c in table.items()
          for kk in c if str(kk).startswith("_mismatch_")]
    if mm:
        print("\n[WARN] per-sample nanmean と report macro に不一致あり:")
        for k, c in mm:
            print("  ", k, {kk: vv for kk, vv in c.items()
                            if str(kk).startswith("_mismatch_")})

    # ---- 1b) identity retention 表 (a arm 中心・base は n=0 想定) ----
    print("\n" + "=" * 96)
    print(f"identity retention [{scale}] (eval_report.identity_retention.mean_retention)")
    print("=" * 96)
    print(f"{'seed':>9} {'arm':>5} {'mean_retention':>16} {'n_cases':>9}")
    for seed in SEEDS:
        for arm in ("base", "a"):
            rr = load_retention(res, arm, seed)
            if rr is None:
                print(f"{seed:>9} {arm:>5} {'(no report)':>16}")
                continue
            mr = rr.get("mean")
            n = rr.get("n")
            mr_s = f"{mr:.4f}" if isinstance(mr, (int, float)) else str(mr)
            print(f"{seed:>9} {arm:>5} {mr_s:>16} {str(n):>9}")
    a_rets = [load_retention(res, "a", seed) for seed in SEEDS]
    a_vals = [r["mean"] for r in a_rets if r and isinstance(r.get("mean"), (int, float))]
    if len(a_vals) >= 2:
        arr = np.array(a_vals)
        print(f"  a retention across-seed: mean={arr.mean():.4f} "
              f"sd={arr.std(ddof=1):.4f} vals={[round(v,4) for v in a_vals]}")

    # ---- 2) per-seed paired delta + bootstrap CI + t ----
    print("\n" + "=" * 96)
    print(f"per-seed paired delta = A2(a) - #1(base) [{scale}] (同 seed・per-sample paired)")
    print("=" * 96)
    deltas = {(s, m): [] for s in SETS for m in METRICS}
    ci_contains_zero = {(s, m): [] for s in SETS for m in METRICS}
    for s in SETS:
        for m in METRICS:
            print(f"\n--- {s} / {m} ---")
            print(f"{'seed':>9} {'#1':>8} {'a':>8} {'delta':>9} "
                  f"{'CI_lo':>9} {'CI_hi':>9} {'t':>8} {'p':>10} {'n_pair':>7}")
            for seed in SEEDS:
                bn = load_persample(res, "base", seed)
                dn = load_persample(res, "a", seed)
                if bn is None or dn is None:
                    print(f"{seed:>9}   (missing arm)")
                    continue
                b, d = paired_arrays(bn, dn, s, m)
                dm, lo, hi = paired_bootstrap_ci(b, d)
                t, pval = paired_t(b, d)
                deltas[(s, m)].append(dm)
                ci_contains_zero[(s, m)].append(lo <= 0 <= hi)
                print(f"{seed:>9} {b.mean():8.4f} {d.mean():8.4f} {dm:9.4f} "
                      f"{lo:9.4f} {hi:9.4f} {t:8.3f} {pval:10.2e} {len(b):7d}")

    # ---- 3) across-seed 平均±sd + #1 diverse seed-variance band ----
    print("\n" + "=" * 96)
    print(f"across-seed 集計 + 判定 [{scale}]")
    print("=" * 96)
    base_band = {}
    for s in SETS:
        for m in METRICS:
            vals = [table[("base", seed, s)][m] for seed in SEEDS
                    if ("base", seed, s) in table and m in table[("base", seed, s)]]
            if len(vals) >= 2:
                arr = np.array(vals)
                base_band[(s, m)] = {
                    "mean": float(arr.mean()), "sd": float(arr.std(ddof=1)),
                    "range": float(arr.max() - arr.min()), "vals": vals}

    for s in SETS:
        for m in METRICS:
            dl = deltas[(s, m)]
            if not dl:
                continue
            arr = np.array(dl)
            mean_d = arr.mean()
            sd_d = arr.std(ddof=1) if len(arr) > 1 else float("nan")
            band = base_band.get((s, m), {})
            band_sd = band.get("sd", float("nan"))
            band_range = band.get("range", float("nan"))
            all_pos = all(x > 0 for x in dl)
            all_neg = all(x < 0 for x in dl)
            ci_excl_zero = [not c for c in ci_contains_zero[(s, m)]]
            print(f"\n### {s} / {m}")
            print(f"  per-seed delta = {[round(x,4) for x in dl]}")
            print(f"  across-seed delta 平均±sd = {mean_d:+.4f} ± "
                  f"{sd_d:.4f} (n_seed={len(dl)})")
            print(f"  #1 diverse seed band: sd={band_sd:.4f} range={band_range:.4f} "
                  f"vals={[round(v,4) for v in band.get('vals',[])]}")
            print(f"  (a) 全 seed 符号一貫: {'+一貫' if all_pos else ('-一貫' if all_neg else '不一貫')}")
            exceeds = (abs(mean_d) > band_sd) if not np.isnan(band_sd) else None
            print(f"  (b) |delta平均| > #1 band sd ?: {exceeds} "
                  f"(|{mean_d:.4f}| vs sd {band_sd:.4f})")
            print(f"  (c) 各 seed paired CI が 0 を除外: {ci_excl_zero} "
                  f"(全 seed 除外={all(ci_excl_zero)})")
            verdict = (all_pos and exceeds and all(ci_excl_zero))
            print(f"  => seed 頑健 (本物): {'YES' if verdict else 'NO/弱い'}")

    print("\n" + "=" * 96)
    print(f"結論サマリ [{scale}]: (a)+(b)+(c) を全 set/metric で満たせば "
          f"A2 実ペア増の diverse 優位は seed 頑健。retention は a arm の同一性条件付け強度。")
    print("=" * 96)


if __name__ == "__main__":
    main()

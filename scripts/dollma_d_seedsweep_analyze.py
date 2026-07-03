# -*- coding: utf-8 -*-
"""Phase 4-D 容量増 seed sweep 集計・判定 (dollma_a_seedsweep_analyze.py の複製・改変)。

_seedsweep_d80m/_results/eval_persample_{c33,d80}_{seed}.npz と
eval_report_{...}.json を読み、seed x {c33(33M),d80(80M)} x {diverse_a,diverse_b} の
macro F1/Jaccard 表、各 seed の paired delta (d80 − c33)、across-seed 平均±sd、
paired bootstrap CI、paired t を出す。

A-sweep からの差分:
  - arm 名 base/a → **c33/d80** (両アームともレシピ込み・差は --arch d80m のみ)。
  - delta = **d80 − c33** (容量増の純効果)。
  - 判定 (b) の band は #1 plain でなく **c33 の seed 分散** (容量増前 33M レシピの seed ばらつき)。
  - ガードレール列を明示集計: train-val gap (両アーム) / identity retention (両アーム・>=0.975 床) /
    in-dist pairs.val F1 (非退行)。gap/retention/in-dist は D の成否ではなく床。

判定軸 (A/B と同一):
  (a) 全 seed で delta の符号が + で一貫するか
  (b) delta 平均が c33 自身の seed 分散帯 (c33 を seed 間で比べた band) を超えるか
  (c) 各 seed の paired bootstrap 95%CI が 0 を含まないか
3 軸全成立で「80M を出荷候補に昇格」、不成立で「容量効かず・80M 不出荷 (c33 が下振れ保険)」。

per-sample 配列は rows と同順・NaN=skip/未定義。paired 解析は両 arm とも非 NaN の
位置のみ採用。

使い方:
  py -3.12 scripts/dollma_d_seedsweep_analyze.py
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
ARMS = ["c33", "d80"]            # control 33M / treat 80M
CONTROL, TREAT = "c33", "d80"    # delta = TREAT - CONTROL・band は CONTROL の seed 分散
BOOT = 10000
RNG = np.random.default_rng(20260626)

RES = os.path.join(ROOT, "data", "bitnet", "_seedsweep_d80m", "_results")


def load_persample(arm, seed):
    p = os.path.join(RES, f"eval_persample_{arm}_{seed}.npz")
    if not os.path.exists(p):
        return None
    # 自作 sweep が直前に書いた信頼済みファイルのみ読む。数値配列だけ参照し object
    # (文字列 provenance) フィールドには触れないので allow_pickle=False で十分。
    return np.load(p, allow_pickle=False)


def load_report(arm, seed):
    p = os.path.join(RES, f"eval_report_{arm}_{seed}.json")
    if not os.path.exists(p):
        return {}
    return json.load(open(p, encoding="utf-8"))


def load_macro(arm, seed):
    """eval_report の macro F1/Jaccard (per-sample nanmean と一致するはず)。"""
    d = load_report(arm, seed)
    gm = d.get("generation_setmetrics", {})
    out = {}
    for s in SETS:
        r = gm.get(s)
        if r and not r.get("skipped"):
            out[s] = {"f1": r["macro"]["f1"], "jaccard": r["macro"]["jaccard"],
                      "precision": r["macro"]["precision"],
                      "recall": r["macro"]["recall"]}
    # in-dist pairs.val (非退行ガードレール) も拾う。
    pv = gm.get("pairs_val")
    if pv and not pv.get("skipped", False):
        out["pairs_val"] = {"f1": pv["macro"]["f1"], "jaccard": pv["macro"]["jaccard"]}
    return out


def load_retention(arm, seed):
    """identity retention (両アームとも --identity 駆動なので意味がある)。"""
    d = load_report(arm, seed)
    rr = d.get("identity_retention")
    if not rr:
        return None
    return {"mean": rr.get("mean_retention"), "n": rr.get("n_cases")}


def load_gap(arm, seed):
    """train-val gap ガードレール: legacy teacher-forcing val_loss を拾う。
    train_loss は eval-only レポートに無い (訓練 stats 側) ため、ここでは val_loss を
    seed 横断比較用に表示し、gap 詳細は train_stats_identity.json (sweep dir) を補助参照。"""
    d = load_report(arm, seed)
    lt = d.get("legacy_teacher_forcing", {})
    return lt.get("val_loss")


def paired_arrays(c_npz, t_npz, s, metric):
    """両 arm で非 NaN の位置のみ揃えた (control, treat) 配列を返す。"""
    b0 = c_npz[f"{s}__{metric}"].astype(float)
    b1 = t_npz[f"{s}__{metric}"].astype(float)
    assert len(b0) == len(b1), (len(b0), len(b1))
    mask = ~np.isnan(b0) & ~np.isnan(b1)
    return b0[mask], b1[mask]


def paired_bootstrap_ci(c, t, n_boot=BOOT, alpha=0.05):
    """paired delta = mean(t - c) の bootstrap 95%CI (サンプルをペアで再標本)。"""
    diff = t - c
    n = len(diff)
    idx = RNG.integers(0, n, size=(n_boot, n))
    boot_means = diff[idx].mean(axis=1)
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(diff.mean()), float(lo), float(hi)


def paired_t(c, t):
    """paired t 統計量と両側 p。"""
    from math import sqrt
    diff = t - c
    n = len(diff)
    m = diff.mean()
    sd = diff.std(ddof=1)
    if sd == 0:
        return float("inf") if m != 0 else 0.0, 0.0
    tt = m / (sd / sqrt(n))
    try:
        from scipy import stats
        p = 2 * stats.t.sf(abs(tt), df=n - 1)
    except Exception:
        from math import erfc
        p = erfc(abs(tt) / sqrt(2))
    return float(tt), float(p)


def main():
    argparse.ArgumentParser().parse_args()  # 引数なし (sweep dir 固定) だが --help は出す

    # ---- 1) macro F1/Jaccard 表 (per-sample nanmean で算出・report と突合) ----
    table = {}  # (arm,seed,set) -> {metric: nanmean}
    for arm in ARMS:
        for seed in SEEDS:
            npz = load_persample(arm, seed)
            if npz is None:
                continue
            macro = load_macro(arm, seed)
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

    print("=" * 100)
    print("Phase 4-D 容量増 seed sweep [d80m]: "
          "seed x {c33(33M),d80(80M)} x {diverse_a,diverse_b} macro 表")
    print("=" * 100)
    hdr = f"{'seed':>9} {'arm':>5} {'set':>16} {'F1':>8} {'Jaccard':>8} {'prec':>8} {'rec':>8}"
    print(hdr)
    print("-" * len(hdr))
    for seed in SEEDS:
        for arm in ARMS:
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

    # ---- 1b) ガードレール: identity retention / in-dist pairs.val F1 / legacy val_loss ----
    print("\n" + "=" * 100)
    print("ガードレール [d80m]: identity retention (>=0.975 床) / in-dist pairs.val F1 (非退行) / "
          "legacy val_loss")
    print("=" * 100)
    print(f"{'seed':>9} {'arm':>5} {'retention':>10} {'ret_n':>7} "
          f"{'indist_F1':>10} {'val_loss':>9}")
    ret_by_arm = {a: [] for a in ARMS}
    indist_by_arm = {a: [] for a in ARMS}
    for seed in SEEDS:
        for arm in ARMS:
            rr = load_retention(arm, seed)
            macro = load_macro(arm, seed)
            indist = macro.get("pairs_val", {}).get("f1")
            vloss = load_gap(arm, seed)
            mr = rr.get("mean") if rr else None
            n = rr.get("n") if rr else None
            if isinstance(mr, (int, float)):
                ret_by_arm[arm].append(mr)
            if isinstance(indist, (int, float)):
                indist_by_arm[arm].append(indist)
            mr_s = f"{mr:.4f}" if isinstance(mr, (int, float)) else str(mr)
            id_s = f"{indist:.4f}" if isinstance(indist, (int, float)) else str(indist)
            vl_s = f"{vloss:.4f}" if isinstance(vloss, (int, float)) else str(vloss)
            print(f"{seed:>9} {arm:>5} {mr_s:>10} {str(n):>7} {id_s:>10} {vl_s:>9}")
    for arm in ARMS:
        rv = ret_by_arm[arm]
        iv = indist_by_arm[arm]
        if rv:
            arr = np.array(rv)
            floor_ok = bool((arr >= 0.975).all())
            print(f"  {arm} retention across-seed: mean={arr.mean():.4f} "
                  f"sd={arr.std(ddof=1) if len(arr)>1 else float('nan'):.4f} "
                  f"床(>=0.975)全seed満たす={floor_ok} vals={[round(v,4) for v in rv]}")
        if iv:
            arr = np.array(iv)
            print(f"  {arm} in-dist pairs.val F1 across-seed: mean={arr.mean():.4f} "
                  f"vals={[round(v,4) for v in iv]}")

    # ---- 2) per-seed paired delta + bootstrap CI + t ----
    print("\n" + "=" * 100)
    print(f"per-seed paired delta = {TREAT}(80M) - {CONTROL}(33M) [d80m] (同 seed・per-sample paired)")
    print("=" * 100)
    deltas = {(s, m): [] for s in SETS for m in METRICS}
    ci_contains_zero = {(s, m): [] for s in SETS for m in METRICS}
    for s in SETS:
        for m in METRICS:
            print(f"\n--- {s} / {m} ---")
            print(f"{'seed':>9} {'c33':>8} {'d80':>8} {'delta':>9} "
                  f"{'CI_lo':>9} {'CI_hi':>9} {'t':>8} {'p':>10} {'n_pair':>7}")
            for seed in SEEDS:
                cn = load_persample(CONTROL, seed)
                tn = load_persample(TREAT, seed)
                if cn is None or tn is None:
                    print(f"{seed:>9}   (missing arm)")
                    continue
                c, t = paired_arrays(cn, tn, s, m)
                dm, lo, hi = paired_bootstrap_ci(c, t)
                tt, pval = paired_t(c, t)
                deltas[(s, m)].append(dm)
                ci_contains_zero[(s, m)].append(lo <= 0 <= hi)
                print(f"{seed:>9} {c.mean():8.4f} {t.mean():8.4f} {dm:9.4f} "
                      f"{lo:9.4f} {hi:9.4f} {tt:8.3f} {pval:10.2e} {len(c):7d}")

    # ---- 3) across-seed 平均±sd + c33 diverse seed-variance band ----
    print("\n" + "=" * 100)
    print("across-seed 集計 + 判定 [d80m] (band = c33 の seed 分散)")
    print("=" * 100)
    control_band = {}
    for s in SETS:
        for m in METRICS:
            vals = [table[(CONTROL, seed, s)][m] for seed in SEEDS
                    if (CONTROL, seed, s) in table and m in table[(CONTROL, seed, s)]]
            if len(vals) >= 2:
                arr = np.array(vals)
                control_band[(s, m)] = {
                    "mean": float(arr.mean()), "sd": float(arr.std(ddof=1)),
                    "range": float(arr.max() - arr.min()), "vals": vals}

    overall = []
    for s in SETS:
        for m in METRICS:
            dl = deltas[(s, m)]
            if not dl:
                continue
            arr = np.array(dl)
            mean_d = arr.mean()
            sd_d = arr.std(ddof=1) if len(arr) > 1 else float("nan")
            band = control_band.get((s, m), {})
            band_sd = band.get("sd", float("nan"))
            band_range = band.get("range", float("nan"))
            all_pos = all(x > 0 for x in dl)
            all_neg = all(x < 0 for x in dl)
            ci_excl_zero = [not c for c in ci_contains_zero[(s, m)]]
            print(f"\n### {s} / {m}")
            print(f"  per-seed delta = {[round(x,4) for x in dl]}")
            print(f"  across-seed delta 平均±sd = {mean_d:+.4f} ± "
                  f"{sd_d:.4f} (n_seed={len(dl)})")
            print(f"  c33 diverse seed band: sd={band_sd:.4f} range={band_range:.4f} "
                  f"vals={[round(v,4) for v in band.get('vals',[])]}")
            print(f"  (a) 全 seed 符号一貫: {'+一貫' if all_pos else ('-一貫' if all_neg else '不一貫')}")
            exceeds = (abs(mean_d) > band_sd) if not np.isnan(band_sd) else None
            print(f"  (b) |delta平均| > c33 band sd ?: {exceeds} "
                  f"(|{mean_d:.4f}| vs sd {band_sd:.4f})")
            print(f"  (c) 各 seed paired CI が 0 を除外: {ci_excl_zero} "
                  f"(全 seed 除外={all(ci_excl_zero)})")
            verdict = (all_pos and bool(exceeds) and all(ci_excl_zero))
            overall.append(verdict)
            print(f"  => seed 頑健 (本物): {'YES' if verdict else 'NO/弱い'}")

    print("\n" + "=" * 100)
    all_yes = bool(overall) and all(overall)
    print(f"結論サマリ [d80m]: 全 set/metric (a)+(b)+(c) 成立 = {all_yes}")
    print("  YES -> 80M を出荷候補に昇格 (Phase 4 で正典化)。")
    print("  NO  -> データ律速で容量効かず・80M 不出荷 (c33=33M b2000+A が下振れ保険)。")
    print("  ※ retention(両アーム>=0.975) / in-dist 非退行 はガードレール (D の成否ではない床)。")
    print("=" * 100)


if __name__ == "__main__":
    main()

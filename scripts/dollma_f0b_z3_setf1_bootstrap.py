# -*- coding: utf-8 -*-
"""dollma_f0b_z3_setf1_bootstrap.py — F-0b 再検証 Z-3
(docs/f0b-reverification-plan.md 「### Z-3」節の入口・出口・禁止事項に従う)

やること (per-sample 配列の再解析のみ・SDXL 生成ゼロ・訓練ゼロ・重み無改変):
  入口: `scripts/train_bitnet.py --eval-only --dump-persample` を canon (`bitnet_dense_fp32`) /
    SFT (`bitnet_dense_sft_fp32`) それぞれに対して実行し、diverse_a / diverse_b の
    per-case F1 配列 (`eval_persample_<name>.npz` の `eval_diverse_{a,b}__f1`) を得る
    (実行そのものは本スクリプトの外・別コマンドで行う。本スクリプトは npz 2 本を読むだけ)。
    per-sample 配列は rows と同順・1 行 1 要素・NaN = skip (prompt が max_len 超過)。
    2 本の npz (canon/SFT) は同じ eval_diverse_{a,b} ファイル・同じ max_len で評価しているので
    skip 判定は同一集合になる (train_bitnet.py の eval_generation_setmetrics docstring
    「#1 と D5 で同じ rows・同じ skip 判定」と同型の保証)。両方 NaN でない行だけを paired サンプルとする
    (本スクリプトは実行時に両側 NaN パターンが一致するか検査し、不一致なら警告を残す)。

  出口: diverse_a / diverse_b それぞれについて、canon vs SFT の per-case F1 差
    (diff_i = f1_sft_i - f1_canon_i、paired・有効サンプルのみ) から:
    ① paired bootstrap percentile CI (95%・B>=10000・複数 seed で幅を確認)
    ② 同 bootstrap の seed 依存 (複数 seed での CI 端の幅)
    ③ 参考として paired t (対応のある t 検定) の CI
    ④ 退行 (diff<0) の割合・diff==0 の件数・差の分布要約 (mean/median/sd/quartile)
    を出す。★判定 (採否・有意の断定) はしない。

  追加 (監査指摘): pairs.eval_diverse_{a,b}.jsonl は unique post_id 500 件 × variant 0/1/2 の
    3 件組 (G=500・各クラスタ size 3) で、1500 件は iid ではない。そこで上記 iid 推定量
    (paired bootstrap / paired t) は**削除せず**、post_id 単位のクラスタ推定量を並置する:
    ⑤ cluster-mean (unweighted・G=500) の t / CI (t(G-1))
    ⑥ cluster bootstrap percentile CI (post を復元抽出・B=20000・10 seed sweep)
    ⑦ CR sandwich (CR0 / CR1・df=G-1)
    ⑧ design effect = (cluster-mean se / iid se)^2 (参考)。
    あわせて「両モデルとも F1=0」の件数 (差が 0 になる件の内訳) を出す。

  検算 (cross_check・python ループ経路 vs numpy ベクトル経路の 2 経路):
    iid: mean / sd / se / t / paired t CI 上下 / 負・正・ゼロ件数 / 中央値 / 四分位 /
         iid bootstrap CI 上下 (1 本目 seed のみ)。
    cluster: cluster-mean の mean / se / t / CI 上下 / G / CR0・CR1 の se / CR0 の t /
         design effect / cluster bootstrap CI 上下 (1 本目 seed のみ) / 両モデル F1=0 の件数。
    bootstrap は乱数を伴うため 2 経路は同一 RNG のリサンプル結果を共有し、検算対象は
    「分位点の取り出しとクラスタ平均の組み立て」に限る (リサンプルの乱数列そのものは検算していない)。
    seed sweep 10 本のうち 1 本目以外の CI 端、CR1 の t・CI、p 値は cross_check 対象外。

禁止事項:
  - 採否・有意/非有意の断定はしない (数値・限界の提示のみ)。
  - 正典 (`bitnet_dense{,_fp32}`/identity/golden) は不改変。本スクリプトは npz を読むだけで
    train_bitnet.py を呼ばない (呼び出しは別コマンド・README 参照)。
  - seed 間分散 (訓練 seed を変えた SFT の再学習) は扱わない (Z-5 の範囲)。本スクリプトは
    既存の隔離重み 1 本ずつ (canon / SFT) のみを対象にする。seed の一次証拠: 評価 seed は
    両 npz の `_seed` = 20260620。訓練 seed が確認できるのは SFT だけ
    (data/bitnet/train_stats_sft.json の "seed": 20260620)。canon の訓練 seed は未確認。
"""

import argparse
import hashlib
import json
import math
import os
import platform
import sys

import numpy as np
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

ALPHA = 0.05
N_BOOTSTRAP = 20000
BOOTSTRAP_SEED = 20260620
# Z-2 と同じ 10 seed (coordinator 指示の慣行を踏襲。既定 seed を含む)。
BOOTSTRAP_SEED_SWEEP = [20260620, 0, 1, 2, 3, 4, 5, 6, 7, 42]

# 元台帳・measurements-log 記録値 (再現確認のターゲット)。
RECORDED = {
    "diverse_a": {"canon_f1": 0.3332, "sft_f1": 0.3158, "delta": -0.0174},
    "diverse_b": {"canon_f1": 0.3804, "sft_f1": 0.3563, "delta": -0.0241},
}


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_persample_f1(npz_path, tag):
    """npz から <tag>__f1 (rows と同順・NaN=skip) を取り出す。

    allow_pickle は使わない (既定 False)。provenance の文字列は固定長 Unicode 配列
    (dtype <U*) で保存されており pickle 不要。細工 npz による任意コード実行を避ける。
    """
    with np.load(npz_path) as z:
        key = f"{tag}__f1"
        if key not in z:
            raise KeyError(f"{npz_path} に {key} が無い (keys={list(z.keys())})")
        arr = np.asarray(z[key], dtype=np.float64)
        weights_path = str(z["_weights"]) if "_weights" in z else None
        weights_sha256 = str(z["_weights_sha256"]) if "_weights_sha256" in z else None
        seed = int(z["_seed"]) if "_seed" in z else None
    return arr, {"weights": weights_path, "weights_sha256": weights_sha256, "seed": seed}


def build_paired(f1_canon, f1_sft):
    """両側 NaN でない行だけを paired サンプルとして残す。
    片側だけ NaN の行があれば異常 (skip 判定が食い違っている) なので数え上げて返す。
    """
    if len(f1_canon) != len(f1_sft):
        raise ValueError(f"行数不一致: canon={len(f1_canon)} sft={len(f1_sft)}")
    nan_c = np.isnan(f1_canon)
    nan_s = np.isnan(f1_sft)
    both_nan = nan_c & nan_s
    only_c = nan_c & ~nan_s
    only_s = ~nan_c & nan_s
    valid = ~nan_c & ~nan_s
    idx = np.nonzero(valid)[0]
    diffs = f1_sft[idx] - f1_canon[idx]
    return {
        "n_rows": len(f1_canon),
        "n_valid_paired": int(valid.sum()),
        "n_both_skip": int(both_nan.sum()),
        "n_mismatch_only_canon_nan": int(only_c.sum()),
        "n_mismatch_only_sft_nan": int(only_s.sum()),
        "valid_index": idx,
        "diffs": diffs,
        "f1_canon_valid": f1_canon[idx],
        "f1_sft_valid": f1_sft[idx],
    }


# ============================================================
# 出口①②③④: 差分要約 + bootstrap + paired t (python ループ経路)
# ============================================================
def compute_diff_summary(diffs):
    n = len(diffs)
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    sd = var ** 0.5
    se = sd / math.sqrt(n)
    t_stat = mean / se if se > 0 else float("nan")
    df = n - 1
    p_two_sided = 2.0 * (1.0 - stats.t.cdf(abs(t_stat), df)) if se > 0 else float("nan")
    t_crit = stats.t.ppf(1 - ALPHA / 2, df)
    ci_t = (mean - t_crit * se, mean + t_crit * se)

    sorted_d = sorted(diffs)
    med = sorted_d[n // 2] if n % 2 else 0.5 * (sorted_d[n // 2 - 1] + sorted_d[n // 2])

    def _quantile(vals_sorted, q):
        if len(vals_sorted) == 1:
            return vals_sorted[0]
        pos = q * (len(vals_sorted) - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return vals_sorted[lo]
        frac = pos - lo
        return vals_sorted[lo] * (1 - frac) + vals_sorted[hi] * frac

    q25 = _quantile(sorted_d, 0.25)
    q75 = _quantile(sorted_d, 0.75)

    n_negative = sum(1 for d in diffs if d < 0)
    n_positive = sum(1 for d in diffs if d > 0)
    n_zero = sum(1 for d in diffs if d == 0)

    return {
        "n": n, "mean": mean, "sd": sd, "se": se, "median": med,
        "q25": q25, "q75": q75, "min": min(diffs), "max": max(diffs),
        "t_stat": t_stat, "df": df, "p_two_sided": p_two_sided,
        "t_crit_975": t_crit, "paired_t_ci95": {"lo": ci_t[0], "hi": ci_t[1]},
        "n_negative": n_negative, "n_positive": n_positive, "n_zero": n_zero,
        "frac_negative": n_negative / n, "frac_positive": n_positive / n,
        "frac_zero": n_zero / n,
    }


def _bootstrap_percentile_ci_loop(diffs, n_boot, seed):
    """paired bootstrap percentile CI。python ループで分位点抽出を素朴実装する経路。
    リサンプル自体は numpy Generator (共有 RNG) で行う (乱数生成は 2 経路で独立にしない —
    独立乱数系列を 2 本使うと「同じ検定をやり直しただけ」になり検算にならないため。
    検算対象は「同じリサンプル結果から分位点を正しく取り出せているか」に絞る)。
    """
    rng = np.random.default_rng(seed)
    n = len(diffs)
    idx_matrix = rng.integers(0, n, size=(n_boot, n))
    diffs_np = np.asarray(diffs, dtype=np.float64)
    boot_means = diffs_np[idx_matrix].mean(axis=1)
    boot_means_list = boot_means.tolist()
    boot_means_sorted = sorted(boot_means_list)

    def _pct_loop(vals_sorted, q):
        pos = q * (len(vals_sorted) - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return vals_sorted[lo]
        frac = pos - lo
        return vals_sorted[lo] * (1 - frac) + vals_sorted[hi] * frac

    lo = _pct_loop(boot_means_sorted, 0.025)
    hi = _pct_loop(boot_means_sorted, 0.975)
    return {"lo": lo, "hi": hi, "boot_mean_of_means": sum(boot_means_list) / n_boot,
            "n_boot": n_boot, "seed": seed}, boot_means


def compute_bootstrap_ci_seed_sweep(diffs, n_boot, seeds):
    results = {}
    lo_vals, hi_vals = [], []
    for sd in seeds:
        ci, _ = _bootstrap_percentile_ci_loop(diffs, n_boot, sd)
        results[str(sd)] = ci
        lo_vals.append(ci["lo"])
        hi_vals.append(ci["hi"])
    return {
        "per_seed": results,
        "seeds": seeds,
        "lo_min": min(lo_vals), "lo_max": max(lo_vals), "lo_mean": sum(lo_vals) / len(lo_vals),
        "hi_min": min(hi_vals), "hi_max": max(hi_vals), "hi_mean": sum(hi_vals) / len(hi_vals),
        "lo_range_width": max(lo_vals) - min(lo_vals),
        "hi_range_width": max(hi_vals) - min(hi_vals),
    }


# ============================================================
# post_id クラスタ推定量 (監査指摘: 1500 件 = 500 post x 3 variant は iid でない)
# ============================================================
def load_post_ids(jsonl_path):
    ids = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(int(json.loads(line)["meta"]["post_id"]))
    return ids


def _cluster_means_loop(diffs, cids):
    sums, cnts = {}, {}
    for d, c in zip(diffs, cids):
        sums[c] = sums.get(c, 0.0) + d
        cnts[c] = cnts.get(c, 0) + 1
    keys = sorted(sums)
    return keys, [sums[k] / cnts[k] for k in keys], [sums[k] for k in keys], [cnts[k] for k in keys]


def _pct_loop(vals_sorted, q):
    pos = q * (len(vals_sorted) - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return vals_sorted[lo]
    frac = pos - lo
    return vals_sorted[lo] * (1 - frac) + vals_sorted[hi] * frac


def compute_cluster_estimators(diffs, cids, iid_se):
    """cluster-mean (unweighted)・CR0/CR1 sandwich・design effect (python ループ経路)。"""
    keys, cm, sums, cnts = _cluster_means_loop(diffs, cids)
    G = len(keys)
    n = len(diffs)
    mean_cm = sum(cm) / G
    var_cm = sum((x - mean_cm) ** 2 for x in cm) / (G - 1)
    se_cm = (var_cm / G) ** 0.5
    df = G - 1
    t_crit = stats.t.ppf(1 - ALPHA / 2, df)
    mean_all = sum(diffs) / n
    # CR sandwich: 全体平均 (n 個の平均) に対するクラスタ頑健分散
    cr0_var = sum((S - m * mean_all) ** 2 for S, m in zip(sums, cnts)) / (n ** 2)
    cr1_var = cr0_var * G / (G - 1)
    out = {
        "G": G, "n": n, "cluster_sizes_distinct": sorted(set(cnts)),
        "cluster_mean": {
            "mean": mean_cm, "se": se_cm, "t_stat": mean_cm / se_cm, "df": df,
            "p_two_sided": 2.0 * (1.0 - stats.t.cdf(abs(mean_cm / se_cm), df)),
            "ci95_t": {"lo": mean_cm - t_crit * se_cm, "hi": mean_cm + t_crit * se_cm},
            "note": "unweighted mean of cluster means。size 一定ならば全体平均と一致する。",
        },
        "cr_sandwich": {},
        "iid_se": iid_se,
        "design_effect": (se_cm / iid_se) ** 2,
    }
    for name, var in (("CR0", cr0_var), ("CR1", cr1_var)):
        se = var ** 0.5
        out["cr_sandwich"][name] = {
            "mean": mean_all, "se": se, "t_stat": mean_all / se, "df": df,
            "ci95_t": {"lo": mean_all - t_crit * se, "hi": mean_all + t_crit * se}}
    if len(set(cnts)) == 1:
        m = cnts[0]
        out["implied_icc_if_balanced"] = (out["design_effect"] - 1.0) / (m - 1) if m > 1 else None
        out["implied_icc_note"] = "deff = 1 + (m-1)*rho の逆算 (balanced 前提・参考値)"
    return out


def cluster_bootstrap_ci_loop(diffs, cids, n_boot, seed):
    """post をリサンプルする cluster bootstrap (統計量 = cluster means の平均)。"""
    _, cm, _, _ = _cluster_means_loop(diffs, cids)
    G = len(cm)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, G, size=(n_boot, G))
    boot = np.asarray(cm, dtype=np.float64)[idx].mean(axis=1).tolist()
    srt = sorted(boot)
    return {"lo": _pct_loop(srt, 0.025), "hi": _pct_loop(srt, 0.975),
            "n_boot": n_boot, "seed": seed, "G": G}


def cluster_bootstrap_seed_sweep(diffs, cids, n_boot, seeds):
    per, lo_v, hi_v = {}, [], []
    for sd in seeds:
        ci = cluster_bootstrap_ci_loop(diffs, cids, n_boot, sd)
        per[str(sd)] = ci
        lo_v.append(ci["lo"])
        hi_v.append(ci["hi"])
    return {"per_seed": per, "seeds": seeds,
            "lo_min": min(lo_v), "lo_max": max(lo_v), "hi_min": min(hi_v), "hi_max": max(hi_v),
            "lo_range_width": max(lo_v) - min(lo_v), "hi_range_width": max(hi_v) - min(hi_v)}


def cluster_cross_check(diffs, cids, cl, cboot_first, first_seed, f1_canon_valid, f1_sft_valid,
                        both_zero):
    """numpy ベクトル経路 (np.unique + bincount) でクラスタ推定量を独立に組み直して照合。"""
    checks = []
    d = np.asarray(diffs, dtype=np.float64)
    _, inv = np.unique(np.asarray(cids), return_inverse=True)
    cnt = np.bincount(inv).astype(np.float64)
    sm = np.bincount(inv, weights=d)
    cm = sm / cnt
    G = len(cm)
    n = len(d)
    mean_cm = float(cm.mean())
    se_cm = float(cm.std(ddof=1) / math.sqrt(G))
    tc = float(stats.t.ppf(1 - ALPHA / 2, G - 1))
    _check(checks, "cluster.G", G, cl["G"])
    _check(checks, "cluster_mean.mean", mean_cm, cl["cluster_mean"]["mean"])
    _check(checks, "cluster_mean.se", se_cm, cl["cluster_mean"]["se"])
    _check(checks, "cluster_mean.t", mean_cm / se_cm, cl["cluster_mean"]["t_stat"])
    _check(checks, "cluster_mean.ci95_t.lo", mean_cm - tc * se_cm, cl["cluster_mean"]["ci95_t"]["lo"])
    _check(checks, "cluster_mean.ci95_t.hi", mean_cm + tc * se_cm, cl["cluster_mean"]["ci95_t"]["hi"])
    mean_all = float(d.mean())
    cr0 = float(np.sqrt(((sm - cnt * mean_all) ** 2).sum() / n ** 2))
    cr1 = cr0 * math.sqrt(G / (G - 1))
    _check(checks, "cr0.se", cr0, cl["cr_sandwich"]["CR0"]["se"])
    _check(checks, "cr1.se", cr1, cl["cr_sandwich"]["CR1"]["se"])
    _check(checks, "cr0.t", mean_all / cr0, cl["cr_sandwich"]["CR0"]["t_stat"])
    iid_se = float(d.std(ddof=1) / math.sqrt(n))
    _check(checks, "design_effect", (se_cm / iid_se) ** 2, cl["design_effect"])
    rng = np.random.default_rng(first_seed)
    idx = rng.integers(0, G, size=(N_BOOTSTRAP, G))
    bm = cm[idx].mean(axis=1)
    _check(checks, f"cluster_bootstrap.seed={first_seed}.lo", float(np.percentile(bm, 2.5)), cboot_first["lo"])
    _check(checks, f"cluster_bootstrap.seed={first_seed}.hi", float(np.percentile(bm, 97.5)), cboot_first["hi"])
    bz = int(((np.asarray(f1_canon_valid) == 0) & (np.asarray(f1_sft_valid) == 0)).sum())
    _check(checks, "n_both_f1_zero", bz, both_zero)
    ok = all(c["ok"] for c in checks)
    return {"ok": ok, "n_checks": len(checks), "n_fail": sum(1 for c in checks if not c["ok"]),
            "messages": [c["name"] for c in checks if not c["ok"]], "checks": checks}


# ============================================================
# 検算 (numpy ベクトル経路での独立再計算)
# ============================================================
def _check(checks, name, recomputed, original, rel_tol=1e-6, abs_tol=1e-9):
    if recomputed is None and original is None:
        ok, diff = True, 0.0
    elif recomputed is None or original is None:
        ok, diff = False, None
    else:
        diff = abs(float(recomputed) - float(original))
        thresh = max(abs_tol, rel_tol * max(1.0, abs(float(original))))
        ok = diff <= thresh
    checks.append({"name": name, "recomputed_numpy": recomputed, "original": original,
                    "diff": diff, "ok": ok})
    return ok


def cross_check(diffs, summary, bootstrap_first_seed_ci, first_seed):
    checks = []
    diffs_np = np.asarray(diffs, dtype=np.float64)
    n = len(diffs_np)
    mean_np = float(diffs_np.mean())
    sd_np = float(diffs_np.std(ddof=1))
    se_np = sd_np / math.sqrt(n)
    t_np = mean_np / se_np
    df = n - 1
    t_crit_np = float(stats.t.ppf(1 - ALPHA / 2, df))
    ci_lo_np = mean_np - t_crit_np * se_np
    ci_hi_np = mean_np + t_crit_np * se_np

    _check(checks, "mean", mean_np, summary["mean"])
    _check(checks, "sd", sd_np, summary["sd"])
    _check(checks, "se", se_np, summary["se"])
    _check(checks, "t_stat", t_np, summary["t_stat"])
    _check(checks, "paired_t_ci95.lo", ci_lo_np, summary["paired_t_ci95"]["lo"])
    _check(checks, "paired_t_ci95.hi", ci_hi_np, summary["paired_t_ci95"]["hi"])
    _check(checks, "n_negative", int((diffs_np < 0).sum()), summary["n_negative"])
    _check(checks, "n_positive", int((diffs_np > 0).sum()), summary["n_positive"])
    _check(checks, "n_zero", int((diffs_np == 0).sum()), summary["n_zero"])
    _check(checks, "median", float(np.median(diffs_np)), summary["median"])
    _check(checks, "q25", float(np.quantile(diffs_np, 0.25)), summary["q25"])
    _check(checks, "q75", float(np.quantile(diffs_np, 0.75)), summary["q75"])

    # bootstrap 分位点抽出の検算 (同じリサンプル結果 boot_means を numpy percentile で取り直す)。
    rng = np.random.default_rng(first_seed)
    idx_matrix = rng.integers(0, n, size=(N_BOOTSTRAP, n))
    boot_means_np = diffs_np[idx_matrix].mean(axis=1)
    lo_np = float(np.percentile(boot_means_np, 2.5))
    hi_np = float(np.percentile(boot_means_np, 97.5))
    _check(checks, f"bootstrap_ci.seed={first_seed}.lo", lo_np, bootstrap_first_seed_ci["lo"])
    _check(checks, f"bootstrap_ci.seed={first_seed}.hi", hi_np, bootstrap_first_seed_ci["hi"])

    ok = all(c["ok"] for c in checks)
    n_fail = sum(1 for c in checks if not c["ok"])
    return {"ok": ok, "n_checks": len(checks), "n_fail": n_fail,
            "messages": [c["name"] for c in checks if not c["ok"]], "checks": checks}


# ============================================================
# 1 データセット (diverse_a / diverse_b) 分の解析まとめ
# ============================================================
def analyze_dataset(tag, f1_canon, f1_sft, recorded, post_ids=None):
    paired = build_paired(f1_canon, f1_sft)
    diffs = paired["diffs"]

    summary = compute_diff_summary(diffs.tolist())
    bootstrap_first, boot_means_first = _bootstrap_percentile_ci_loop(
        diffs.tolist(), N_BOOTSTRAP, BOOTSTRAP_SEED)
    bootstrap_sweep = compute_bootstrap_ci_seed_sweep(diffs.tolist(), N_BOOTSTRAP,
                                                       BOOTSTRAP_SEED_SWEEP)
    xc = cross_check(diffs.tolist(), summary, bootstrap_first, BOOTSTRAP_SEED)

    # 再現確認: per-case 平均から出した macro F1 (canon/sft) と記録値 (measurements-log / 元台帳) を突合。
    macro_canon = float(np.mean(paired["f1_canon_valid"]))
    macro_sft = float(np.mean(paired["f1_sft_valid"]))
    macro_delta = macro_sft - macro_canon
    reproduction = {
        "recomputed_macro_f1_canon": macro_canon,
        "recomputed_macro_f1_sft": macro_sft,
        "recomputed_delta": macro_delta,
        "recorded_macro_f1_canon": recorded["canon_f1"],
        "recorded_macro_f1_sft": recorded["sft_f1"],
        "recorded_delta": recorded["delta"],
        "diff_vs_recorded_canon_f1": macro_canon - recorded["canon_f1"],
        "diff_vs_recorded_sft_f1": macro_sft - recorded["sft_f1"],
        "diff_vs_recorded_delta": macro_delta - recorded["delta"],
        # 記録値は小数4桁への丸めなので、丸め誤差 (最大 0.00005) 以内なら再現とみなす。
        "reproduced_within_rounding": (abs(macro_canon - recorded["canon_f1"]) < 5e-5 and
                                        abs(macro_sft - recorded["sft_f1"]) < 5e-5),
    }

    cluster_block = None
    if post_ids is not None:
        if len(post_ids) != paired["n_rows"]:
            raise ValueError(f"post_id 行数不一致: {len(post_ids)} vs {paired['n_rows']}")
        cids = [post_ids[i] for i in paired["valid_index"]]
        cl = compute_cluster_estimators(diffs.tolist(), cids, summary["se"])
        cboot = cluster_bootstrap_ci_loop(diffs.tolist(), cids, N_BOOTSTRAP, BOOTSTRAP_SEED)
        csweep = cluster_bootstrap_seed_sweep(diffs.tolist(), cids, N_BOOTSTRAP, BOOTSTRAP_SEED_SWEEP)
        both_zero = int(((paired["f1_canon_valid"] == 0) & (paired["f1_sft_valid"] == 0)).sum())
        cxc = cluster_cross_check(diffs.tolist(), cids, cl, cboot, BOOTSTRAP_SEED,
                                  paired["f1_canon_valid"], paired["f1_sft_valid"], both_zero)
        cluster_block = {
            "structure": {"cluster_key": "meta.post_id", "G": cl["G"],
                          "cluster_sizes_distinct": cl["cluster_sizes_distinct"],
                          "n_variants_per_post_expected": 3,
                          "note": "unique post_id x variant 0/1/2 の 3 件組。1500 件は iid でない。"},
            "estimators": cl,
            "cluster_bootstrap_ci95": cboot,
            "cluster_bootstrap_ci95_seed_sweep": csweep,
            "regression_excludes_zero_cluster_bootstrap": bool(cboot["hi"] < 0.0),
            "regression_excludes_zero_cluster_t": bool(cl["cluster_mean"]["ci95_t"]["hi"] < 0.0),
            "regression_excludes_zero_cr1": bool(cl["cr_sandwich"]["CR1"]["ci95_t"]["hi"] < 0.0),
            "n_both_models_f1_zero": both_zero,
            "n_both_models_f1_zero_frac": both_zero / len(diffs),
            "n_diff_zero_total": summary["n_zero"],
            "cross_check": cxc,
        }

    return {
        "tag": tag,
        "cluster_by_post_id": cluster_block,
        "pairing": {k: v for k, v in paired.items()
                    if k not in ("valid_index", "diffs", "f1_canon_valid", "f1_sft_valid")},
        "reproduction_check": reproduction,
        "diff_summary": summary,
        "bootstrap_ci95": {
            "first_seed": BOOTSTRAP_SEED, "n_boot": N_BOOTSTRAP,
            "lo": bootstrap_first["lo"], "hi": bootstrap_first["hi"],
            "boot_mean_of_means": bootstrap_first["boot_mean_of_means"],
        },
        "bootstrap_ci95_seed_sweep": bootstrap_sweep,
        "paired_t_ci95_reference": summary["paired_t_ci95"],
        "cross_check": xc,
        "regression_excludes_zero_bootstrap": bool(bootstrap_first["hi"] < 0.0),
        "regression_excludes_zero_t": bool(summary["paired_t_ci95"]["hi"] < 0.0),
    }


def selftest():
    """最小の自己検査: 合成 per-sample 配列で paired 構築・bootstrap・検算の健全性を確認する。"""
    ok = True
    rng = np.random.default_rng(0)
    n = 40
    f1_canon = rng.uniform(0.2, 0.5, size=n)
    f1_sft = f1_canon - 0.02
    f1_canon[[1, 5, 9]] = float("nan")
    f1_sft[[1, 5, 9]] = float("nan")

    recorded = {"canon_f1": float(np.nanmean(f1_canon)), "sft_f1": float(np.nanmean(f1_sft)),
                "delta": float(np.nanmean(f1_sft) - np.nanmean(f1_canon))}
    post_ids = [i // 3 for i in range(n)]  # 3 件組クラスタ (最後は size 1)
    res = analyze_dataset("selftest", f1_canon, f1_sft, recorded, post_ids)
    if not res["cluster_by_post_id"]["cross_check"]["ok"]:
        print("[selftest FAIL] cluster cross_check", res["cluster_by_post_id"]["cross_check"]["messages"])
        ok = False

    if res["pairing"]["n_valid_paired"] != n - 3:
        print("[selftest FAIL] n_valid_paired", res["pairing"]["n_valid_paired"])
        ok = False
    if res["pairing"]["n_mismatch_only_canon_nan"] != 0 or res["pairing"]["n_mismatch_only_sft_nan"] != 0:
        print("[selftest FAIL] mismatch counts should be 0")
        ok = False
    if abs(res["diff_summary"]["mean"] - (-0.02)) > 1e-9:
        print("[selftest FAIL] mean diff", res["diff_summary"]["mean"])
        ok = False
    if not res["reproduction_check"]["reproduced_within_rounding"]:
        print("[selftest FAIL] reproduction check on synthetic recorded values")
        ok = False
    if not res["cross_check"]["ok"]:
        print("[selftest FAIL] cross_check", res["cross_check"]["messages"])
        ok = False
    f1_canon2 = f1_canon.copy()
    f1_sft2 = f1_sft.copy()
    f1_canon2[3] = float("nan")
    paired2 = build_paired(f1_canon2, f1_sft2)
    if paired2["n_mismatch_only_canon_nan"] != 1:
        print("[selftest FAIL] mismatch-only-canon detection", paired2["n_mismatch_only_canon_nan"])
        ok = False

    if ok:
        print("[selftest] OK")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--canon-npz", default=None,
                     help="eval_persample_<canon>.npz (train_bitnet.py --eval-only --dump-persample の出力)")
    ap.add_argument("--sft-npz", default=None,
                     help="eval_persample_<sft>.npz")
    ap.add_argument("--canon-report", default=None, help="eval_report_<canon>.json (再現確認・provenance 用)")
    ap.add_argument("--sft-report", default=None, help="eval_report_<sft>.json")
    ap.add_argument("--pairs-a", default=os.path.join(ROOT, "docs", "logs", "f0b-z3", "_scratch_data",
                                                       "pairs.eval_diverse_a.jsonl"))
    ap.add_argument("--pairs-b", default=os.path.join(ROOT, "docs", "logs", "f0b-z3", "_scratch_data",
                                                       "pairs.eval_diverse_b.jsonl"))
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "logs", "f0b-z3", "z3_result.json"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        ok = selftest()
        sys.exit(0 if ok else 1)

    for p, label in ((args.canon_npz, "--canon-npz"), (args.sft_npz, "--sft-npz")):
        if not p or not os.path.isfile(p):
            print(f"[error] {label} が見つかりません: {p}")
            sys.exit(1)

    reports = {}
    for p, label in ((args.canon_report, "canon"), (args.sft_report, "sft")):
        if p and os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                reports[label] = json.load(f)

    datasets = {}
    for tag in ("eval_diverse_a", "eval_diverse_b"):
        f1_canon, prov_c = load_persample_f1(args.canon_npz, tag)
        f1_sft, prov_s = load_persample_f1(args.sft_npz, tag)
        short = "diverse_a" if tag == "eval_diverse_a" else "diverse_b"
        pj = args.pairs_a if short == "diverse_a" else args.pairs_b
        if not os.path.isfile(pj):
            print(f"[error] pairs jsonl が見つかりません: {pj}")
            sys.exit(1)
        res = analyze_dataset(tag, f1_canon, f1_sft, RECORDED[short], load_post_ids(pj))
        res["pairs_file"] = {"path": os.path.abspath(pj), "sha256": sha256_of(pj)}
        res["npz_provenance"] = {"canon": prov_c, "sft": prov_s}
        datasets[short] = res

    result = {
        "stage": "F-0b 再検証 Z-3 (set-F1 per-case paired bootstrap CI)",
        "plan_ref": "docs/f0b-reverification-plan.md 「### Z-3」節",
        "inputs": {
            "canon_npz": {"path": os.path.abspath(args.canon_npz), "sha256": sha256_of(args.canon_npz)},
            "sft_npz": {"path": os.path.abspath(args.sft_npz), "sha256": sha256_of(args.sft_npz)},
            "canon_report": ({"path": os.path.abspath(args.canon_report),
                               "sha256": sha256_of(args.canon_report)}
                              if args.canon_report and os.path.isfile(args.canon_report) else None),
            "sft_report": ({"path": os.path.abspath(args.sft_report),
                             "sha256": sha256_of(args.sft_report)}
                            if args.sft_report and os.path.isfile(args.sft_report) else None),
        },
        "env": {"python": sys.version, "platform": platform.platform(), "cwd": os.getcwd(),
                "numpy": np.__version__, "scipy": __import__("scipy").__version__},
        "config": {"alpha": ALPHA, "n_bootstrap": N_BOOTSTRAP, "bootstrap_seed": BOOTSTRAP_SEED,
                   "bootstrap_seed_sweep": BOOTSTRAP_SEED_SWEEP,
                   "seed_scope_note": "評価 seed は canon/SFT とも 20260620 (npz の _seed)。訓練 seed が"
                                      "一次証拠 (data/bitnet/train_stats_sft.json) で確認できるのは SFT"
                                      "だけ (20260620)。canon の訓練 seed は未確認。本スクリプトの"
                                      "bootstrap seed sweep は bootstrap 乱数の再現性チェックであって、"
                                      "訓練 seed を振った seed sweep (Z-5) ではない。"},
        "datasets": datasets,
        "notes": [
            "採否・有意性の断定はしていない (数値・分位点の提示のみ)。",
            "評価 seed 20260620 固定・既存の隔離重み 1 本ずつ (SFT の訓練 seed 20260620 は train_stats_sft.json で確認、canon の訓練 seed は未確認) の中の識別力のみを見ている。訓練 seed 間分散は Z-5 の範囲。",
            "diverse_{a,b} は 500 post_id x variant 3 件の構造 (G=500)。iid 推定量 (diff_summary / bootstrap_ci95) は残し、cluster_by_post_id に post_id クラスタ推定量を並置した。",
            "per-case F1 の NaN (prompt が max_len 超過でスキップ) は canon/SFT で行 index が一致する前提。"
            "本スクリプトは build_paired で両側 NaN 一致を検査し、不一致があれば "
            "n_mismatch_only_{canon,sft}_nan に計上する (0 でなければ前提が崩れている)。",
        ],
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[done] wrote {args.out}")

    for short, res in datasets.items():
        print(f"== {short} ==")
        rc = res["reproduction_check"]
        print(f"  reproduction: recomputed delta={rc['recomputed_delta']:.6f} "
              f"(recorded {rc['recorded_delta']:.4f}) "
              f"within_rounding={rc['reproduced_within_rounding']}")
        ds = res["diff_summary"]
        print(f"  n_valid_paired={res['pairing']['n_valid_paired']} mean_diff={ds['mean']:.6f} "
              f"t={ds['t_stat']:.4f} df={ds['df']} paired_t_ci95=[{ds['paired_t_ci95']['lo']:.6f}, "
              f"{ds['paired_t_ci95']['hi']:.6f}]")
        bc = res["bootstrap_ci95"]
        print(f"  bootstrap(seed={bc['first_seed']}, B={bc['n_boot']}) ci95=[{bc['lo']:.6f}, {bc['hi']:.6f}]")
        sw = res["bootstrap_ci95_seed_sweep"]
        print(f"  bootstrap seed sweep: lo in [{sw['lo_min']:.6f}, {sw['lo_max']:.6f}] "
              f"(width {sw['lo_range_width']:.6f}) hi in [{sw['hi_min']:.6f}, {sw['hi_max']:.6f}] "
              f"(width {sw['hi_range_width']:.6f})")
        print(f"  frac_negative={ds['frac_negative']:.4f} frac_positive={ds['frac_positive']:.4f} "
              f"frac_zero={ds['frac_zero']:.4f}")
        xc = res["cross_check"]
        print(f"  cross_check ok={xc['ok']} n_checks={xc['n_checks']} n_fail={xc['n_fail']}")
        cb = res["cluster_by_post_id"]
        ce = cb["estimators"]
        sw2 = cb["cluster_bootstrap_ci95_seed_sweep"]
        print(f"  [cluster G={ce['G']}] cluster-mean t={ce['cluster_mean']['t_stat']:.4f} "
              f"ci_t=[{ce['cluster_mean']['ci95_t']['lo']:.6f}, {ce['cluster_mean']['ci95_t']['hi']:.6f}] "
              f"deff={ce['design_effect']:.4f}")
        print(f"  [cluster] CR0 t={ce['cr_sandwich']['CR0']['t_stat']:.4f} CR1 t={ce['cr_sandwich']['CR1']['t_stat']:.4f} "
              f"CR1 ci=[{ce['cr_sandwich']['CR1']['ci95_t']['lo']:.6f}, {ce['cr_sandwich']['CR1']['ci95_t']['hi']:.6f}]")
        print(f"  [cluster] boot ci=[{cb['cluster_bootstrap_ci95']['lo']:.6f}, {cb['cluster_bootstrap_ci95']['hi']:.6f}] "
              f"sweep lo[{sw2['lo_min']:.6f},{sw2['lo_max']:.6f}] hi[{sw2['hi_min']:.6f},{sw2['hi_max']:.6f}]")
        print(f"  [cluster] both_f1_zero={cb['n_both_models_f1_zero']} diff_zero={cb['n_diff_zero_total']} "
              f"cluster_cross_check ok={cb['cross_check']['ok']} n={cb['cross_check']['n_checks']} fail={cb['cross_check']['n_fail']}")


if __name__ == "__main__":
    main()

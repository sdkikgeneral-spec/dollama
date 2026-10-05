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

  検算: 主要な数値 (mean diff・paired t・se・1本目の bootstrap CI) を
    python ループ経路 (compute_diff_summary) と numpy ベクトル経路 (cross_check 内) の
    2 経路で計算し照合する。bootstrap 自体は乱数を伴うため「2 経路検算」の対象は
    percentile 計算ロジック (ソート済み配列からの分位点抽出) のみで、乱数生成そのものは
    同一 RNG (numpy Generator) を両経路で共有する (=リサンプルの中身は同一・分位点の取り方だけ
    ループ実装と numpy 実装で独立に書いて比べる)。

禁止事項:
  - 採否・有意/非有意の断定はしない (数値・限界の提示のみ)。
  - 正典 (`bitnet_dense{,_fp32}`/identity/golden) は不改変。本スクリプトは npz を読むだけで
    train_bitnet.py を呼ばない (呼び出しは別コマンド・README 参照)。
  - seed 間分散 (訓練 seed を変えた SFT の再学習) は扱わない (Z-5 の範囲)。本スクリプトは
    既存の隔離重み (訓練 seed 20260620 固定) 1 本のみを対象にする。
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
def analyze_dataset(tag, f1_canon, f1_sft, recorded):
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

    return {
        "tag": tag,
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
    res = analyze_dataset("selftest", f1_canon, f1_sft, recorded)

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
        res = analyze_dataset(tag, f1_canon, f1_sft, RECORDED[short])
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
                   "seed_scope_note": "評価は既存の凍結隔離重み (訓練 seed 20260620 固定) に対して"
                                      "行っており、本スクリプトの bootstrap seed sweep は"
                                      "『bootstrap 乱数の再現性チェック』であって『SFT 訓練 seed を"
                                      "振った seed sweep (Z-5)』ではない。"},
        "datasets": datasets,
        "notes": [
            "採否・有意性の断定はしていない (数値・分位点の提示のみ)。",
            "同一 seed (訓練 20260620・評価 20260620) 内の識別力のみを見ている。seed 間分散は Z-5 の範囲。",
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


if __name__ == "__main__":
    main()

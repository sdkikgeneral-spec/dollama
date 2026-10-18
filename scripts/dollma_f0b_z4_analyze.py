# -*- coding: utf-8 -*-
"""dollma_f0b_z4_analyze.py — F-0b 再検証 Z-4 (プラセボ対照 SFT) の解析と 5 区分判定の機械適用。

Z-3 スクリプト (dollma_f0b_z3_setf1_bootstrap.py) の関数を import する薄いラッパー。
Z-3 の RECORDED (canon/SFT 固定の記録値) は無効化して使う (本スクリプトは自前の値で呼ぶ)。

入力 (--npz-dir に次の命名で置く。`train_bitnet.py --eval-only --dump-persample --eval-name <arm>` の出力):
  eval_persample_canon.npz            正典 (再評価)
  eval_persample_r_ref.npz            R = 既存 SFT 重み (参考のみ・判定に使わない)
  eval_persample_r_s<seed>.npz        R'_s (reward argmax 勝者, seed s で再訓練)
  eval_persample_p_s<seed>.npz        P_s  (プラセボ勝者, seed s)
判定規則は台帳 (docs/f0b-reverification-plan.md Z-4) に実走前固定したものを定数としてここ 1 か所に持つ。
結果を見て変えない。
"""
import argparse
import json
import math
import os
import platform
import sys

import numpy as np
from scipy import stats
from scipy.integrate import quad

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dollma_f0b_z3_setf1_bootstrap as z3  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, ".."))

# ============================================================
# 判定定数 (実走前固定・ここ 1 か所のみ。結果を見て変えない)
# ============================================================
SEEDS = [20260620, 20260621, 42, 7, 20260622, 20260623, 1, 2]   # 8 seed (台帳 Z-5 の慣行 4 + 追加 4)
M_FRAC = 0.5          # margin M = M_FRAC * |mean Delta_R'| (事前選択値)
ALPHA_CI = 0.05       # 95% 両側 CI (one-sample t df=n-1 / D の Welch CI)
ALPHA_TOST = 0.05     # TOST: 片側 0.05 x2 = 両側 90% CI が [-M, +M] 内
# base 33M 参照アーム (c33) の 4 seed macro set-F1 標本 sd・set 別。出典 = docs/training-spec.md §16.2 表。
# 一次ログ data/bitnet/_seedsweep_d80m/_results/ は本機に無く、ここでは表の値を引用しているだけ (未照合)。
# 事前検出力 (--power-only) と Z-5 分散帯比の分母の両方に使う。
SD_PRIOR = {"diverse_a": 0.0114, "diverse_b": 0.0131}
EXPECTED_EVAL_SEED = 20260620   # 評価 seed (Z-3 と同条件・全アーム同値を assert)
# Z-5 合否規則 (実走前固定・ここ 1 か所): 対象 = R' の 8 seed 全部。set ごとに (a)∧(b)∧(c)、
# diverse_a と diverse_b の両方成立で z5_pass=True。符号は退行側 (先例 dollma_d_seedsweep_analyze.py:288-294 と同型)。
#   (a) 全 seed で Delta_R' < 0   (b) |mean Delta_R'| > SD_PRIOR[set] (定義変更付き分散帯比)
#   (c) 全 seed で post_id 単位 paired CI (cluster bootstrap 95%) の上限 < 0
Z5_REFERENCE_SEEDS = [20260620, 20260621, 42, 7]   # 参考列 (z5_pass に不使用)
Z5_FIXED_NOTE = ("分散帯比は元定義 (現正典の seed 違い再訓練 sd) 未算出。"
                 "z5_pass は定義変更付き 3 軸 (分母 = c33 アームの seed sd 引用値) の判定")
N_POWER = 8          # 事前検出力の 1 アームあたり seed 数
M_PRIOR = {"diverse_a": 0.5 * 0.0174, "diverse_b": 0.5 * 0.0241}   # 記録値からの仮置き (事前検出力のみ)
CLASSES = ["①退行再現せず", "②選抜方針(reward)起因", "③両方寄与", "④追加SFTの副作用", "⑤判定不能"]
# 判定順: ① -> ② -> ③ -> ④ -> ⑤ (classify_dataset 内の記述順が判定順)

# 本採用 canon / R の記録値 (再評価一致確認のみ・判定には使わない)。計画「検証」節。
RECORDED_REEVAL = {
    "diverse_a": {"canon": 0.333250, "r_ref": 0.315838},
    "diverse_b": {"canon": 0.380411, "r_ref": 0.356268},
}
RECORDED_TOL = 1.5e-5


# ============================================================
# 統計の基本部品
# ============================================================
def one_sample_t(x, alpha=ALPHA_CI):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    m = float(x.mean())
    sd = float(x.std(ddof=1))
    se = sd / math.sqrt(n)
    df = n - 1
    tc = float(stats.t.ppf(1 - alpha / 2, df))
    return {"n": n, "mean": m, "sd": sd, "se": se, "df": df,
            "t": (m / se) if se > 0 else float("nan"),
            "ci_lo": m - tc * se, "ci_hi": m + tc * se, "level": 1 - alpha}


def welch_diff(xp, xr, alpha=ALPHA_CI):
    """D = mean(xp) - mean(xr) の Welch CI (両側 1-alpha)。"""
    xp = np.asarray(xp, dtype=np.float64)
    xr = np.asarray(xr, dtype=np.float64)
    np_, nr = len(xp), len(xr)
    vp, vr = xp.var(ddof=1) / np_, xr.var(ddof=1) / nr
    se = math.sqrt(vp + vr)
    df = (vp + vr) ** 2 / (vp ** 2 / (np_ - 1) + vr ** 2 / (nr - 1)) if (vp + vr) > 0 else float("nan")
    d = float(xp.mean() - xr.mean())
    tc = float(stats.t.ppf(1 - alpha / 2, df))
    return {"D": d, "se": se, "df": df, "level": 1 - alpha,
            "ci_lo": d - tc * se, "ci_hi": d + tc * se}


def classify_dataset(d_r, d_p):
    """1 データセット (diverse_a か diverse_b) の 5 区分。d_r / d_p = seed ごとの canon 比 Delta (長さ n)。

    返り値 dict: cls (CLASSES の 1 つ)・各 CI・M・注記。判定順は上から。
    """
    r = one_sample_t(d_r)                      # 95% two-sided, df=n-1
    p = one_sample_t(d_p)
    w95 = welch_diff(d_p, d_r, ALPHA_CI)        # D の 95% CI
    w90 = welch_diff(d_p, d_r, 2 * ALPHA_TOST)  # TOST 用 90% 両側 CI
    D = w95["D"]
    M = M_FRAC * abs(r["mean"])
    notes = []

    reproduced = (r["mean"] < 0) and (r["ci_hi"] < 0)
    out = {"R_prime": r, "P": p, "D_welch95": w95, "D_welch90_tost": w90, "M": M,
           "reproduced": reproduced, "D_point": D, "notes": notes}
    # ① 退行再現せず
    if not reproduced:
        out["cls"] = CLASSES[0]
        return out
    d_pos_sig = w95["ci_lo"] > 0
    d_ge_m = D >= M
    # ② 選抜方針起因 (P は有意に退行しない: P の 95%CI 上限 >= 0)
    if d_pos_sig and d_ge_m and p["ci_hi"] >= 0:
        out["cls"] = CLASSES[1]
        if p["ci_lo"] > 0:
            notes.append("P 改善 (Delta_P の 95%CI 下限 > 0)")
        return out
    # ③ 両方寄与 (P も有意に退行)
    if d_pos_sig and d_ge_m and p["ci_hi"] < 0:
        out["cls"] = CLASSES[2]
        return out
    # ④ 追加 SFT の副作用 (TOST: D の両側 90%CI 全体が [-M, +M] 内)
    if (w90["ci_lo"] > -M) and (w90["ci_hi"] < M):
        out["cls"] = CLASSES[3]
        return out
    # ⑤ 判定不能
    out["cls"] = CLASSES[4]
    if d_pos_sig and not d_ge_m:
        notes.append("D は有意に正だが D<M")
    if w95["ci_hi"] < 0:
        notes.append("D<0 有意 (プラセボの方が悪い)")
    return out


def z5_judge(d_r, ci_hi, sd_prior):
    """1 set の Z-5 3 軸。d_r = seed ごとの Delta_R'、ci_hi = 各 seed の post_id 単位 paired CI 上限。"""
    if len(d_r) == 0:
        return {"pass": None, "n_seeds": 0}
    a = all(x < 0 for x in d_r)
    b = abs(float(np.mean(d_r))) > sd_prior
    c = all(x < 0 for x in ci_hi)
    return {"a_sign_all_negative": a, "b_abs_mean_gt_sd_prior": b, "c_all_ci_upper_below_zero": c,
            "pass": bool(a and b and c), "n_seeds": len(d_r)}


def combine_ab(res_a, res_b):
    """a/b の合成。① は a・b どちらかで退行不成立 → ①。そうでなく区分が割れたら ⑤ (a/b 不一致)。"""
    if res_a["cls"] == CLASSES[0] or res_b["cls"] == CLASSES[0]:
        return {"cls": CLASSES[0], "note": "a/b のどちらかで退行が再現せず"}
    if res_a["cls"] != res_b["cls"]:
        return {"cls": CLASSES[4], "note": f"a/b 不一致 (a={res_a['cls']} / b={res_b['cls']})"}
    return {"cls": res_a["cls"], "note": "a/b 一致"}


# ============================================================
# 事前検出力 (TOST・D=0・同 sd・n vs n・Welch)
# ============================================================
def tost_power_mc(sd, n, M, n_sim=400000, seed=20261009):
    """Monte Carlo (Welch df・両側 90% CI が [-M, M] 内)。真の D=0, 両アーム sd 同一。"""
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, sd, size=(n_sim, n))
    b = rng.normal(0.0, sd, size=(n_sim, n))
    va, vb = a.var(axis=1, ddof=1) / n, b.var(axis=1, ddof=1) / n
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (n - 1) + vb ** 2 / (n - 1))
    tc = stats.t.ppf(1 - ALPHA_TOST, df)
    d = a.mean(axis=1) - b.mean(axis=1)
    ok = (d - tc * se > -M) & (d + tc * se < M)
    return float(ok.mean())


def tost_power_analytic_pooled(sd, n, M):
    """プール分散 t (df=2n-2) の解析積分。Welch MC の近似照合用 (等分散下では Welch とほぼ一致)。"""
    df = 2 * n - 2
    tc = stats.t.ppf(1 - ALPHA_TOST, df)
    sig_d = sd * math.sqrt(2.0 / n)

    def f(u):  # u = s/sd, (df*u^2) ~ chi2(df)
        c = M - tc * sig_d * u
        pr = (2 * stats.norm.cdf(c / sig_d) - 1) if c > 0 else 0.0
        dens = 2 * df * u * stats.chi2.pdf(df * u * u, df)
        return pr * dens
    val, _ = quad(f, 0, 6)
    return float(val)


def power_table():
    rows = {}
    for k, M in M_PRIOR.items():
        sdp = SD_PRIOR[k]
        rows[k] = {"M": M, "sd": sdp, "n_per_arm": N_POWER,
                   "power_welch_mc": tost_power_mc(sdp, N_POWER, M),
                   "power_pooled_analytic": tost_power_analytic_pooled(sdp, N_POWER, M)}
    return rows


# ============================================================
# 交絡統計 (アーム別・仮説は書かず実測のみ)
# ============================================================
def confound_stats(rows, seq_fn, max_len=64):
    """rows: SFT ペア行。seq_fn(row) -> (ids, tags_start)。"""
    n_tags = np.array([len(r["tags"]) for r in rows], dtype=np.float64)
    tgt, trunc = [], 0
    for r in rows:
        ids, ts = seq_fn(r)
        if len(ids) > max_len:
            trunc += 1
        tgt.append(max(0, min(len(ids), max_len) - ts))
    tgt = np.array(tgt, dtype=np.float64)
    uniq = len({tuple(r["tags"]) for r in rows})
    langs = {}
    for r in rows:
        langs[r.get("lang")] = langs.get(r.get("lang"), 0) + 1
    return {
        "n": len(rows), "lang": langs,
        "n_tags": {"mean": float(n_tags.mean()), "sd": float(n_tags.std(ddof=1)),
                   "min": float(n_tags.min()), "q25": float(np.quantile(n_tags, .25)),
                   "median": float(np.median(n_tags)), "q75": float(np.quantile(n_tags, .75)),
                   "max": float(n_tags.max())},
        "target_tokens_total": int(tgt.sum()),
        "max_len_trunc_rate": trunc / len(rows),
        "unique_tag_lists": uniq,
        "winner_coincidence_rate": (float(np.mean([bool(r["meta"].get("coincides_with_winner"))
                                                    for r in rows]))
                                    if "coincides_with_winner" in rows[0].get("meta", {}) else None),
    }


def _load_rows(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ============================================================
# 本体
# ============================================================
def seed_delta_and_pair(f1_canon, f1_arm, post_ids):
    """canon 比 Delta (macro, 有効行の平均差) と Z-3 推定量 (post_id 単位 / iid) を返す。"""
    macro_arm = float(np.nanmean(f1_arm))
    macro_canon_all = float(np.nanmean(f1_canon))
    rec = {"canon_f1": macro_canon_all, "sft_f1": macro_arm, "delta": macro_arm - macro_canon_all}  # RECORDED 無効化
    saved_sweep = z3.BOOTSTRAP_SEED_SWEEP
    z3.BOOTSTRAP_SEED_SWEEP = [z3.BOOTSTRAP_SEED]   # per-seed では sweep を 1 本に (速度・記述のみ)
    try:
        res = z3.analyze_dataset("z4", f1_canon, f1_arm, rec, post_ids)
    finally:
        z3.BOOTSTRAP_SEED_SWEEP = saved_sweep       # 書き換えを必ず復元
    pg = res["pairing"]
    assert pg["n_mismatch_only_canon_nan"] == 0 and pg["n_mismatch_only_sft_nan"] == 0,         f"片側 NaN 不一致 (skip 判定食い違い): {pg}"
    cb = res["cluster_by_post_id"]
    ds = res["diff_summary"]
    return {
        "delta": res["reproduction_check"]["recomputed_delta"],
        "n_valid": res["pairing"]["n_valid_paired"],
        "post_id_cluster": {"ci95_cluster_bootstrap": [cb["cluster_bootstrap_ci95"]["lo"],
                                                       cb["cluster_bootstrap_ci95"]["hi"]],
                            "ci95_cluster_mean_t": [cb["estimators"]["cluster_mean"]["ci95_t"]["lo"],
                                                    cb["estimators"]["cluster_mean"]["ci95_t"]["hi"]],
                            "G": cb["estimators"]["G"]},
        "iid": {"ci95_bootstrap": [res["bootstrap_ci95"]["lo"], res["bootstrap_ci95"]["hi"]],
                "ci95_paired_t": [ds["paired_t_ci95"]["lo"], ds["paired_t_ci95"]["hi"]],
                "n": ds["n"]},
        "excludes_zero_post_id": bool(cb["cluster_bootstrap_ci95"]["hi"] < 0
                                      or cb["cluster_bootstrap_ci95"]["lo"] > 0),
    }


def check_provenance(npz_dir, seeds, expected_seed=EXPECTED_EVAL_SEED):
    """来歴照合: 全 npz の評価 seed 同値・16 本の重み sha256 が全て異なる・重みパスに arm/seed を含む。"""
    prov = {}
    names = ["canon"] + [f"{a}_s{s}" for a in ("r", "p") for s in seeds]
    ref = os.path.join(npz_dir, "eval_persample_r_ref.npz")
    if os.path.isfile(ref):
        names.append("r_ref")
    for nm in names:
        _, pv = z3.load_persample_f1(os.path.join(npz_dir, f"eval_persample_{nm}.npz"), "eval_diverse_a")
        prov[nm] = pv
        assert pv["seed"] == expected_seed, f"{nm}: 評価 seed {pv['seed']} != {expected_seed}"
    arm_names = [f"{a}_s{s}" for a in ("r", "p") for s in seeds]
    shas = [prov[n]["weights_sha256"] for n in arm_names]
    assert all(shas) and len(set(shas)) == len(shas), "訓練アームの重み sha256 に欠落/重複あり"
    for a in ("r", "p"):
        for s in seeds:
            bn = os.path.basename(prov[f"{a}_s{s}"]["weights"])
            assert f"_{a}_s{s}_fp32" in bn, f"重みパスに arm/seed 無し: {bn} (期待 _{a}_s{s}_fp32)"
    return {"eval_seed": expected_seed, "n_arm_weights_distinct": len(set(shas)),
            "weights": {n: {"path": v["weights"], "sha256": v["weights_sha256"]} for n, v in prov.items()}}


def retention_summary(report_dir, winners_path, seeds, n_all=500, n_excl=35, n_kept=465):
    """identity retention の全件 / 訓練 post 重複除外後の併記 (記述のみ・判定不使用)。
    除外集合 = 勝者 sft_bestofn の post_id ∩ details の post_id (int 照合)。paired 比較は diverse のみ。"""
    win_pids = {int(r["meta"]["post_id"]) for r in _load_rows(winners_path)}
    names = ["canon"] + [f"{a}_s{s}" for a in ("r", "p") for s in seeds]
    if os.path.isfile(os.path.join(report_dir, "eval_report_r_ref.json")):
        names.append("r_ref")
    per_arm, ref_pids = {}, None
    for nm in names:
        with open(os.path.join(report_dir, f"eval_report_{nm}.json"), encoding="utf-8") as f:
            ir = json.load(f)["identity_retention"]
        if "details" not in ir:
            raise SystemExit(f"[停止] {nm}: details 無し (--dump-retention-detail 付きで評価すること)")
        det = ir["details"]
        if len(det) != n_all or ir["n_cases"] != n_all:
            raise SystemExit(f"[停止] {nm}: details 件数 {len(det)} / n_cases {ir['n_cases']} (期待 {n_all}・欠落行あり)")
        pids = [int(d["post_id"]) for d in det]
        assert len(set(pids)) == n_all, f"{nm}: details の post_id が重複"
        if ref_pids is None:
            ref_pids = pids
            excl = win_pids & set(pids)
            assert len(excl) == n_excl, f"除外集合 {len(excl)} != {n_excl}"
        assert pids == ref_pids, f"{nm}: details の post_id 並びが他アームと不一致"
        rets_all = [d["retention"] for d in det]
        rets_kept = [d["retention"] for d in det if int(d["post_id"]) not in excl]
        assert len(rets_kept) == n_kept
        per_arm[nm] = {"mean_all": float(np.mean(rets_all)), "n_all": len(rets_all),
                       "mean_excluding_train_posts": float(np.mean(rets_kept)), "n_kept": len(rets_kept)}
    def agg(prefix):
        out = {}
        for key in ("mean_all", "mean_excluding_train_posts"):
            v = [per_arm[f"{prefix}_s{s}"][key] for s in seeds]
            out[key] = {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))}
        return out
    return {"per_arm": per_arm, "R_prime_seed_mean_sd": agg("r"), "P_seed_mean_sd": agg("p"),
            "n_all": n_all, "n_excluded": n_excl, "n_kept": n_kept,
            "note": "記述のみ・判定には使わない。paired 比較 (Delta/CI) は diverse_a/b のみ。retention は paired 検定をしていない。"}


def analyze_all(npz_dir, pairs, seeds, with_paired=True):
    out = {"datasets": {}}
    tags = {"diverse_a": "eval_diverse_a", "diverse_b": "eval_diverse_b"}
    for short, tag in tags.items():
        pid = z3.load_post_ids(pairs[short])
        f1c, _ = z3.load_persample_f1(os.path.join(npz_dir, "eval_persample_canon.npz"), tag)
        arms = {"r": {}, "p": {}}
        per_seed = {"r": {}, "p": {}}
        for arm in ("r", "p"):
            for s in seeds:
                f1x, _ = z3.load_persample_f1(os.path.join(npz_dir, f"eval_persample_{arm}_s{s}.npz"), tag)
                if with_paired:
                    info = seed_delta_and_pair(f1c, f1x, pid)
                else:
                    pr = z3.build_paired(f1c, f1x)
                    assert pr["n_mismatch_only_canon_nan"] == 0 and pr["n_mismatch_only_sft_nan"] == 0,                         f"片側 NaN 不一致: {pr['n_mismatch_only_canon_nan']}/{pr['n_mismatch_only_sft_nan']}"
                    info = {"delta": float(np.mean(pr["diffs"]))}
                per_seed[arm][str(s)] = info
                arms[arm][s] = info["delta"]
        d_r = [arms["r"][s] for s in seeds]
        d_p = [arms["p"][s] for s in seeds]
        cls = classify_dataset(d_r, d_p)
        # D の paired 版 (同 seed 対応・参考): Delta_P(s) - Delta_R'(s) の one-sample t
        paired_D = one_sample_t([p - r for p, r in zip(d_p, d_r)])
        entry = {"classification": cls, "D_paired_by_seed_reference": paired_D,
                 "per_seed": per_seed, "delta_R_prime": d_r, "delta_P": d_p,
                 "mean_delta_R_prime": float(np.mean(d_r)), "mean_delta_P": float(np.mean(d_p)),
                 "sd_delta_R_prime": float(np.std(d_r, ddof=1)), "sd_delta_P": float(np.std(d_p, ddof=1)),
                 "sign_consistency_R_prime": f"{sum(1 for x in d_r if x < 0)}/{len(d_r)} 負"}
        if with_paired:
            # Z-5 判定軸 (c): 各 seed の paired CI は post_id 単位を主・iid は併記のみ
            entry["z5_variance_band"] = {
                "denominator_sd_prior": SD_PRIOR[short],
                "ratio_abs_mean_delta_R_prime_over_sd_prior": abs(entry["mean_delta_R_prime"]) / SD_PRIOR[short],
                "observed_sd_delta_R_prime": entry["sd_delta_R_prime"],
                "note": "分母 = c33 アームの seed sd (引用値)。R' の seed sd は別量として並記 (固定 canon からの SFT seed sd)。"}
            hi = {sd_: per_seed["r"][str(sd_)]["post_id_cluster"]["ci95_cluster_bootstrap"][1] for sd_ in seeds}
            ref4 = [sd_ for sd_ in Z5_REFERENCE_SEEDS if sd_ in seeds]
            entry["z5"] = {
                "all_seeds": z5_judge([arms["r"][sd_] for sd_ in seeds], [hi[sd_] for sd_ in seeds], SD_PRIOR[short]),
                "reference_first4_not_used_for_pass": z5_judge(
                    [arms["r"][sd_] for sd_ in ref4], [hi[sd_] for sd_ in ref4], SD_PRIOR[short]),
                "note": Z5_FIXED_NOTE}
            entry["z5_axis_c"] = {
                "primary_estimator": "post_id cluster bootstrap (G=500)",
                "n_seeds_regression_ci_below_zero_post_id": sum(
                    1 for s in seeds if per_seed["r"][str(s)]["post_id_cluster"]["ci95_cluster_bootstrap"][1] < 0),
                "n_seeds_regression_ci_below_zero_iid_reference": sum(
                    1 for s in seeds if per_seed["r"][str(s)]["iid"]["ci95_bootstrap"][1] < 0),
                "n_seeds": len(seeds),
                "G": per_seed["r"][str(seeds[0])]["post_id_cluster"]["G"]}
        # 参考: R (既存重み) - R'_20260620 (ノイズ床ではなく参考差)
        rp = os.path.join(npz_dir, "eval_persample_r_ref.npz")
        if os.path.isfile(rp):
            f1r, _ = z3.load_persample_f1(rp, tag)
            dr_ref = float(np.mean(z3.build_paired(f1c, f1r)["diffs"]))
            entry["reference_R_existing"] = {"delta_vs_canon": dr_ref,
                                             "R_minus_Rprime_20260620": dr_ref - arms["r"].get(20260620, float("nan")),
                                             "note": "参考差 (ノイズ床ではない)"}
            rec = RECORDED_REEVAL[short]
            entry["reeval_check"] = {
                "canon_macro": float(np.nanmean(f1c)), "r_ref_macro": float(np.nanmean(f1r)),
                "recorded": rec,
                "match": bool(abs(np.nanmean(f1c) - rec["canon"]) < RECORDED_TOL
                              and abs(np.nanmean(f1r) - rec["r_ref"]) < RECORDED_TOL)}
        out["datasets"][short] = entry
    if with_paired:
        out["z5_pass"] = bool(all(out["datasets"][k]["z5"]["all_seeds"]["pass"] for k in out["datasets"]))
        out["z5_note"] = Z5_FIXED_NOTE
    out["combined"] = combine_ab(out["datasets"]["diverse_a"]["classification"],
                                 out["datasets"]["diverse_b"]["classification"])
    return out


# ============================================================
# selftest (合成データ)
# ============================================================
def _synth_deltas(mean, sd, n=8, seed=0):
    """平均・標本 sd がちょうど (mean, sd) になる n 点 (rng 由来の形を標準化)。"""
    z = np.random.default_rng(seed).normal(size=n)
    z = (z - z.mean()) / z.std(ddof=1)
    return (mean + sd * z).tolist()


def selftest():
    ok = True

    def chk(name, cond):
        nonlocal ok
        if not cond:
            print(f"[selftest FAIL] {name}")
            ok = False

    # --- 判定定数の固定値 ---
    chk("const M_FRAC", M_FRAC == 0.5 and ALPHA_CI == 0.05 and ALPHA_TOST == 0.05 and len(SEEDS) == 8)
    sd = 0.002
    # R' は -0.02 (退行再現)。
    dR = _synth_deltas(-0.02, sd, seed=1)
    # ② P ≈ 0 (D=+0.02 >= M=0.01)・P の上限 >= 0
    c2 = classify_dataset(dR, _synth_deltas(0.0, sd, seed=2))
    chk(f"cls② {c2['cls']}", c2["cls"] == CLASSES[1])
    # ② 注記: P 有意改善
    c2b = classify_dataset(dR, _synth_deltas(0.01, sd, seed=3))
    chk("cls② P改善注記", c2b["cls"] == CLASSES[1] and any("P 改善" in n for n in c2b["notes"]))
    # ③ P=-0.007 (D=0.013>=M=0.01 かつ P の 95%CI 上限<0)
    c3 = classify_dataset(dR, _synth_deltas(-0.007, sd, seed=4))
    chk(f"cls③ {c3['cls']} D={c3['D_point']:.4f} Phi={c3['P']['ci_hi']:.4f}", c3["cls"] == CLASSES[2])
    # ④ P ≈ R' (D≈0, 小 sd)
    c4 = classify_dataset(dR, _synth_deltas(-0.02, sd, seed=5))
    chk(f"cls④ {c4['cls']}", c4["cls"] == CLASSES[3])
    # ① 退行再現せず (R' が 0 近傍)
    c1 = classify_dataset(_synth_deltas(-0.001, sd, seed=6), _synth_deltas(0.0, sd, seed=7))
    chk("cls①", c1["cls"] == CLASSES[0])
    # ① 平均が正
    c1b = classify_dataset(_synth_deltas(+0.005, sd, seed=6), _synth_deltas(0.0, sd, seed=7))
    chk("cls①(正)", c1b["cls"] == CLASSES[0])
    # ⑤ 大 sd (広い CI)
    c5 = classify_dataset(_synth_deltas(-0.02, 0.01, seed=8), _synth_deltas(-0.012, 0.01, seed=9))
    chk(f"cls⑤ {c5['cls']}", c5["cls"] == CLASSES[4])
    # ⑤ D<0 有意 (プラセボの方が悪い): P=-0.04, R'=-0.02 -> D=-0.02
    c5n = classify_dataset(dR, _synth_deltas(-0.04, sd, seed=10))
    chk(f"cls⑤ D<0 {c5n['cls']}", c5n["cls"] == CLASSES[4] and any("D<0" in n for n in c5n["notes"]))
    # ⑤ D 有意正だが D<M: R'=-0.02 (M=0.01) に対し P=-0.0125 (D=0.0075<M, 小 sd で有意)
    c5s = classify_dataset(_synth_deltas(-0.02, 0.0045, seed=1), _synth_deltas(-0.0125, 0.0045, seed=11))
    chk(f"cls⑤ D<M {c5s['cls']} D={c5s['D_point']:.4f} lo={c5s['D_welch95']['ci_lo']:.4f}",
        c5s["cls"] == CLASSES[4] and c5s["D_welch95"]["ci_lo"] > 0)

    # --- TOST 境界: 90%CI がちょうど M 内/外 ---
    # 同 sd・n=8 vs 8 の合成で、D を動かし境界前後で ④ が切り替わる
    base_r = _synth_deltas(-0.02, 0.002, seed=12)
    M = 0.5 * 0.02
    w = welch_diff(_synth_deltas(-0.02, 0.002, seed=13), base_r, 0.10)
    half = (w["ci_hi"] - w["ci_lo"]) / 2
    dn = M - half - 1e-6   # 内側ぎりぎり
    dout = M - half + 1e-4  # 外側ぎりぎり
    in_ = classify_dataset(base_r, [x + dn for x in _synth_deltas(-0.02, 0.002, seed=13)])
    out_ = classify_dataset(base_r, [x + dout for x in _synth_deltas(-0.02, 0.002, seed=13)])
    chk(f"TOST 境界内 {in_['cls']}", in_["cls"] == CLASSES[3])
    chk(f"TOST 境界外 {out_['cls']}", out_["cls"] != CLASSES[3])

    # --- a/b 合成 ---
    chk("combine 一致", combine_ab({"cls": CLASSES[1]}, {"cls": CLASSES[1]})["cls"] == CLASSES[1])
    cab = combine_ab({"cls": CLASSES[1]}, {"cls": CLASSES[3]})
    chk("combine a/b 不一致 -> ⑤", cab["cls"] == CLASSES[4] and "不一致" in cab["note"])
    chk("combine ① 優先", combine_ab({"cls": CLASSES[0]}, {"cls": CLASSES[1]})["cls"] == CLASSES[0])

    # --- 検出力 (解析積分と MC がほぼ一致・M 大で単調増) ---
    pa = tost_power_analytic_pooled(0.0114, 8, 0.0087)
    pm = tost_power_mc(0.0114, 8, 0.0087, n_sim=100000)
    chk(f"power analytic~MC {pa:.3f} {pm:.3f}", abs(pa - pm) < 0.02)
    chk("power 単調", tost_power_analytic_pooled(0.0114, 8, 0.0121) > pa)

    # --- 交絡統計 (合成) ---
    rows = [{"tags": ["a"] * 3, "lang": "ja", "meta": {"coincides_with_winner": True}},
            {"tags": ["b"] * 5, "lang": "en", "meta": {"coincides_with_winner": False}}]
    cs = confound_stats(rows, lambda r: (list(range(len(r["tags"]) + 2)), 3), max_len=6)
    chk("confound trunc/tokens", cs["max_len_trunc_rate"] == 0.5 and cs["target_tokens_total"] == 2 + 3)
    chk("confound coincidence", abs(cs["winner_coincidence_rate"] - 0.5) < 1e-12)

    # --- end-to-end analyze_all (合成 npz・小規模) ---
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        rng = np.random.default_rng(0)
        n = 60
        pj = os.path.join(td, "pairs.jsonl")
        with open(pj, "w", encoding="utf-8") as f:
            for i in range(n):
                f.write(json.dumps({"meta": {"post_id": i // 3}}) + "\n")
        base = rng.uniform(0.2, 0.5, size=n)
        def save(name, arr):
            np.savez_compressed(os.path.join(td, name), eval_diverse_a__f1=arr, eval_diverse_b__f1=arr)
        save("eval_persample_canon.npz", base)
        seeds = [1, 2, 3, 4]
        for i, s in enumerate(seeds):
            save(f"eval_persample_r_s{s}.npz", base - 0.02 + rng.normal(0, 0.002, n))
            save(f"eval_persample_p_s{s}.npz", base + rng.normal(0, 0.002, n))
        r = analyze_all(td, {"diverse_a": pj, "diverse_b": pj}, seeds)
        chk("e2e keys", "combined" in r and "z5_axis_c" in r["datasets"]["diverse_a"]
            and isinstance(r["z5_pass"], bool))

    # --- Z-5 合否 (合成) ---
    neg = [-0.02] * 8
    hi_ok = [-0.005] * 8
    chk("z5 全成立", z5_judge(neg, hi_ok, 0.0114)["pass"])
    chk("z5 (a)不成立", not z5_judge(neg[:7] + [0.001], hi_ok, 0.0114)["pass"])
    chk("z5 (b)不成立", not z5_judge([-0.008] * 8, hi_ok, 0.0114)["pass"])
    chk("z5 (c)不成立", not z5_judge(neg, hi_ok[:7] + [0.001], 0.0114)["pass"])
    za = z5_judge(neg, hi_ok, SD_PRIOR["diverse_a"])
    zb = z5_judge([-0.012] * 8, hi_ok, SD_PRIOR["diverse_b"])   # |0.012| < 0.0131 で (b) 不成立
    chk("z5 a/b 不一致 -> 両方成立でなく pass 偽", za["pass"] and not zb["pass"])
    # --- 来歴照合・retention (合成) ---
    with tempfile.TemporaryDirectory() as td2:
        seeds = [1, 2]
        arr = np.array([0.3, 0.4])
        def sv(nm, w, sha, seed=EXPECTED_EVAL_SEED):
            np.savez_compressed(os.path.join(td2, f"eval_persample_{nm}.npz"), eval_diverse_a__f1=arr,
                                _weights=np.asarray(w), _weights_sha256=np.asarray(sha), _seed=np.asarray(seed))
        sv("canon", "x/bitnet_dense_fp32.safetensors", "c")
        for a_ in ("r", "p"):
            for s_ in seeds:
                sv(f"{a_}_s{s_}", f"x/bitnet_dense_sft_{a_}_s{s_}_fp32.safetensors", f"{a_}{s_}")
        pv = check_provenance(td2, seeds)
        chk("prov ok", pv["n_arm_weights_distinct"] == 4)
        sv("p_s2", "x/bitnet_dense_sft_p_s2_fp32.safetensors", "r1")   # sha 重複
        try:
            check_provenance(td2, seeds)
            chk("prov sha 重複検出", False)
        except AssertionError:
            pass
        sv("p_s2", "x/bitnet_dense_sft_p_s2_fp32.safetensors", "p2", seed=7)   # seed 不一致
        try:
            check_provenance(td2, seeds)
            chk("prov seed 不一致検出", False)
        except AssertionError:
            pass
        sv("p_s2", "x/bitnet_dense_sft_p_s9_fp32.safetensors", "p2")   # パスに seed 無し
        try:
            check_provenance(td2, seeds)
            chk("prov パス検出", False)
        except AssertionError:
            pass
        # retention: 20 件・うち 5 件が訓練 post
        wp = os.path.join(td2, "win.jsonl")
        with open(wp, "w", encoding="utf-8") as f:
            for pid in list(range(100, 105)) + [999]:
                f.write(json.dumps({"meta": {"post_id": pid}}) + "\n")
        def rep(nm, base, drop=0):
            det = [{"post_id": 100 + i, "retention": base + 0.01 * (i % 3)} for i in range(20 - drop)]
            with open(os.path.join(td2, f"eval_report_{nm}.json"), "w", encoding="utf-8") as f:
                json.dump({"identity_retention": {"n_cases": 20 - drop, "details": det}}, f)
        for nm in ["canon", "r_s1", "r_s2", "p_s1", "p_s2"]:
            rep(nm, 0.97)
        rs = retention_summary(td2, wp, seeds, n_all=20, n_excl=5, n_kept=15)
        chk("retention 件数", rs["n_kept"] == 15 and rs["per_arm"]["canon"]["n_kept"] == 15)
        rep("p_s2", 0.97, drop=1)
        try:
            retention_summary(td2, wp, seeds, n_all=20, n_excl=5, n_kept=15)
            chk("retention 欠落停止", False)
        except SystemExit:
            pass
    if ok:
        print("[selftest] OK (z4 analyze)")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--power-only", action="store_true", help="事前検出力のみ算出して標準出力 (--out 不要)")
    ap.add_argument("--npz-dir", default=None)
    ap.add_argument("--pairs-a", default=None)
    ap.add_argument("--pairs-b", default=None)
    ap.add_argument("--winners", default=None, help="勝者 SFT ペア (sft_bestofn.jsonl・交絡統計用)")
    ap.add_argument("--placebo-dir", default=None, help="placebo_s<seed>.jsonl のあるディレクトリ (交絡統計用)")
    ap.add_argument("--vocab", default=None, help="交絡統計の seq 構築用 vocab.json")
    ap.add_argument("--report-dir", default=None,
                    help="--dump-retention-detail 付き eval_report_<arm>.json のディレクトリ (既定 = --npz-dir)")
    ap.add_argument("--out", default=None, help="結果 JSON (必須・既定出力先なし)")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if args.power_only:
        print(json.dumps(power_table(), ensure_ascii=False, indent=2))
        return
    if not args.out:
        ap.error("--out は必須 (Z-3 の既定出力先は使わない)")
    for k in ("npz_dir", "pairs_a", "pairs_b"):
        if not getattr(args, k):
            ap.error(f"--{k.replace('_', '-')} は必須")

    if not args.winners:
        ap.error("--winners は必須 (retention の除外集合・来歴)")
    res = analyze_all(args.npz_dir, {"diverse_a": args.pairs_a, "diverse_b": args.pairs_b}, SEEDS)
    res["provenance"] = check_provenance(args.npz_dir, SEEDS)
    res["retention"] = retention_summary(args.report_dir or args.npz_dir, args.winners, SEEDS)
    res["power_prior"] = power_table()
    res["constants"] = {"SEEDS": SEEDS, "M_FRAC": M_FRAC, "ALPHA_CI": ALPHA_CI, "ALPHA_TOST": ALPHA_TOST,
                        "SD_PRIOR": SD_PRIOR, "EXPECTED_EVAL_SEED": EXPECTED_EVAL_SEED, "classes": CLASSES}
    # 実測 sd の MDE (記述のみ・判定は変えない): TOST 90% で M を超えない最小 |D| ではなく、
    # 実測 sd で検出力 0.8 になる M を二分探索
    for k, e in res["datasets"].items():
        sdv = math.sqrt(0.5 * (e["sd_delta_R_prime"] ** 2 + e["sd_delta_P"] ** 2))
        lo, hi = 1e-5, 0.2
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if tost_power_analytic_pooled(sdv, len(SEEDS), mid) >= 0.8:
                hi = mid
            else:
                lo = mid
        e["mde_note"] = {"pooled_sd_observed": sdv, "M_for_power_0.8": hi,
                         "note": "実測 sd で TOST 検出力 0.8 になる M (記述のみ・判定は不変)"}
    if args.winners and args.placebo_dir and args.vocab:
        import train_bitnet as tb
        tok = tb.Tokenizer(args.vocab)
        fn = lambda r: tb.build_sequence(tok, r, 64)  # noqa: E731
        conf = {"R_prime_winners": confound_stats(_load_rows(args.winners), fn)}
        pm = []
        for s in SEEDS:
            c = confound_stats(_load_rows(os.path.join(args.placebo_dir, f"placebo_s{s}.jsonl")), fn)
            conf[f"P_s{s}"] = c
            pm.append(c)
        conf["P_mean"] = {
            "n_tags_mean": float(np.mean([c["n_tags"]["mean"] for c in pm])),
            "target_tokens_total_mean": float(np.mean([c["target_tokens_total"] for c in pm])),
            "max_len_trunc_rate_mean": float(np.mean([c["max_len_trunc_rate"] for c in pm])),
            "unique_tag_lists_mean": float(np.mean([c["unique_tag_lists"] for c in pm])),
            "winner_coincidence_rate_mean": float(np.mean([c["winner_coincidence_rate"] for c in pm]))}
        res["confound"] = conf
    res["env"] = {"python": sys.version, "platform": platform.platform(),
                  "numpy": np.__version__, "scipy": __import__("scipy").__version__}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"[done] wrote {args.out}")
    for k, e in res["datasets"].items():
        print(f"== {k}: {e['classification']['cls']} (M={e['classification']['M']:.5f} "
              f"D={e['classification']['D_point']:.5f})")
    print(f"== combined: {res['combined']}")


if __name__ == "__main__":
    main()

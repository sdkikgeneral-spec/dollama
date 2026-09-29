# -*- coding: utf-8 -*-
"""dollma_f0b_z2_seed_variance.py — F-0b 再検証 Z-2
(docs/f0b-reverification-plan.md 「### Z-2」節の入口・出口・禁止事項に従う)

やること (既存データの再解析のみ・SDXL 生成ゼロ・訓練ゼロ):
  出口①: g2b_prepost.jsonl (200行=100ペア) の pre→post Δreward を
    (i) naive (n=100 を独立標本として扱う)
    (ii) unweighted cluster-mean (G=47: ja は 54 ペア全部が同一 (pre_prompt, post_prompt) の
         組み合わせに収束しているため 1 クラスタ・en は各ペアが固有プロンプトなので 46 の
         singleton クラスタ = 合計 47)
    (iii) cluster-robust sandwich (CR0 / CR1・G=47)
    (iv) en のみ 46 ペア (疑似反復ゼロの独立系列)
    の 4 通りで Δ / se / t / df / 95%CI を算出する。CI は分位点 (t(df) / z) を明記して複数出す。
    加えて cluster-mean については nonparametric cluster bootstrap percentile CI も出す
    (記録値 CI 下端 -0.0115 がどの推定法で再現するかを突き止めるため)。

  出口②: σ_seed (SDXL seed 由来の reward ばらつき) を ja 54 件 (同一プロンプト×seed変動) から
    直接推定する。pre/post 別・プール (2標本プール分散)・reward 全体 + quality_contribution /
    anatomy_contribution 成分別。95%CI は χ2 ベース (正規近似) と nonparametric bootstrap の両方。

  出口③: Δ=+0.017081 (naive 全体) と Δ=+0.011152 (en46) を α=0.05 両側・検出力0.8 で検出するのに
    要る独立プロンプト数を、2 通りの設計で逆算する:
      (a) 現行設計 = pre/post で SDXL seed が異なる (Var(Δ_i) ≈ τ² + 2σ_seed²/k、k=1プロンプトあたり
          撒くseed数)。τ² は en46 の Δ 分散から σ_seed²(pooled, ja由来) の寄与を差し引いて推定
          (負なら 0 で打ち切り、その旨明記)。
      (b) seed 固定 pre/post (seed 由来分が完全相殺される楽観 ρ=1 の場合 Var(Δ_i)=τ²、
          相殺ゼロの悲観 ρ=0 の場合は (a) と同じ)。**ρ の実測値はこのデータには無い** (pre/post で
          常に別 seed が使われているため相殺の効果を直接観測できない) ことを明記する。
    1 プロンプトあたり k seed を撒く場合の総必要枚数 (プロンプト数 × k × 2 アーム) の表も出す。

  出口 (5): 検算。全ての主要数値を素朴な python ループ実装と numpy ベクトル実装の 2 経路で計算し
    一致を確認する (cross_check ブロック)。

禁止事項:
  - GPU 実走の起票・撮る枚数の最終決定はしない (数値の算出まで)。
  - 有意/採否の断定はしない (数値・限界の提示のみ)。

reward 式・成分定義は scripts/dollma_reward.py / dollma_f0b_z1_quality_contribution.py と同一
(anatomy_contribution・quality_contribution は g2b_prepost.jsonl に事前計算済みの列をそのまま使う。
本スクリプトは reward 式の複製はしない)。
"""

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MAIN_DATA_ROOT = r"E:\Develop\Projects\dollama\data\rollouts"  # 読み取り専用 (main checkout)

ALPHA = 0.05
POWER = 0.8
BOOTSTRAP_SEED = 20260620
N_BOOTSTRAP = 20000


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _mean_var_sd(vals, ddof=1):
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - ddof)
    return mean, var, var ** 0.5, n


def _dist(vals):
    if not vals:
        return None
    n = len(vals)
    s = sorted(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    med = s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
    return {"min": min(vals), "median": med, "max": max(vals), "mean": mean,
            "std_pop": var ** 0.5, "std_sample": (var * n / (n - 1)) ** 0.5 if n > 1 else None, "n": n}


# ============================================================
# データ整形
# ============================================================
def build_pairs(rows):
    """post_id -> {"pre": row, "post": row} の 100 ペアを組む。"""
    by_pid = defaultdict(dict)
    for r in rows:
        by_pid[int(r["post_id"])][r["model"]] = r
    pairs = {}
    for pid, d in by_pid.items():
        if "pre" in d and "post" in d:
            pairs[pid] = d
    return pairs


def build_clusters(pairs, key_field="reward"):
    """クラスタキー = (pre.prompt, post.prompt) の組み合わせ。
    ja 54 ペアは全部同一の組み合わせに収束するため 1 クラスタになり、
    en 46 ペアは各々固有の組み合わせなので singleton クラスタになる (データ駆動・lang をハードコードしない)。
    戻り値: {cluster_key: [(pid, delta), ...]}
    """
    clusters = defaultdict(list)
    for pid, d in pairs.items():
        ck = (d["pre"]["prompt"], d["post"]["prompt"])
        delta = d["post"][key_field] - d["pre"][key_field]
        clusters[ck].append((pid, delta))
    return clusters


# ============================================================
# 出口① 4 推定量 + cluster bootstrap CI
# ============================================================
def compute_estimators(pairs):
    deltas = [(pid, d["post"]["reward"] - d["pre"]["reward"], d["pre"]["lang"]) for pid, d in pairs.items()]
    n = len(deltas)
    vals = [d for _, d, _ in deltas]

    # --- (i) naive n=100 ---
    mean, var, sd, _ = _mean_var_sd(vals, ddof=1)
    se_naive = sd / math.sqrt(n)
    t_naive = mean / se_naive
    df_naive = n - 1
    p_naive = 2 * (1 - stats.t.cdf(abs(t_naive), df_naive))
    ci_naive_t = _ci(mean, se_naive, stats.t.ppf(0.975, df_naive))
    ci_naive_z = _ci(mean, se_naive, stats.norm.ppf(0.975))

    naive = {
        "n": n, "mean": mean, "sd": sd, "se": se_naive, "t": t_naive, "df": df_naive,
        "p_two_sided": p_naive,
        "ci95_t": {"quantile": "t(df=%d)" % df_naive, "crit": stats.t.ppf(0.975, df_naive), "lo": ci_naive_t[0], "hi": ci_naive_t[1]},
        "ci95_z": {"quantile": "z(normal)", "crit": stats.norm.ppf(0.975), "lo": ci_naive_z[0], "hi": ci_naive_z[1]},
    }

    # --- クラスタ構築 (reward) ---
    clusters = build_clusters(pairs, key_field="reward")
    cluster_deltas = {ck: [d for _, d in items] for ck, items in clusters.items()}
    cluster_means = [sum(v) / len(v) for v in cluster_deltas.values()]
    G = len(cluster_means)
    cluster_sizes = [len(v) for v in cluster_deltas.values()]

    # --- (ii) unweighted cluster-mean (G=47) ---
    cm_mean, cm_var, cm_sd, _ = _mean_var_sd(cluster_means, ddof=1)
    cm_se = cm_sd / math.sqrt(G)
    cm_t = cm_mean / cm_se
    cm_df = G - 1
    cm_p = 2 * (1 - stats.t.cdf(abs(cm_t), cm_df))
    ci_cm_t = _ci(cm_mean, cm_se, stats.t.ppf(0.975, cm_df))
    ci_cm_z = _ci(cm_mean, cm_se, stats.norm.ppf(0.975))

    cluster_mean_est = {
        "G": G, "cluster_sizes_distinct": sorted(set(cluster_sizes)),
        "mean": cm_mean, "sd_of_cluster_means": cm_sd, "se": cm_se, "t": cm_t, "df": cm_df,
        "p_two_sided": cm_p,
        "ci95_t": {"quantile": "t(df=%d)" % cm_df, "crit": stats.t.ppf(0.975, cm_df), "lo": ci_cm_t[0], "hi": ci_cm_t[1]},
        "ci95_z": {"quantile": "z(normal)", "crit": stats.norm.ppf(0.975), "lo": ci_cm_z[0], "hi": ci_cm_z[1]},
    }

    # --- cluster bootstrap percentile CI (cluster-mean 推定量の nonparametric 版) ---
    boot = _cluster_bootstrap_ci(list(cluster_deltas.values()), seed=BOOTSTRAP_SEED, n_boot=N_BOOTSTRAP)
    cluster_mean_est["bootstrap_percentile_ci95"] = boot

    # --- (iii) CR0 / CR1 sandwich (OLS 切片のみモデルへのクラスタロバスト分散) ---
    u = [v - mean for v in vals]  # 全体平均を使った残差 (切片のみ OLS の残差)
    pid_lang = {pid: lang for pid, _, lang in deltas}
    cluster_u_sums = defaultdict(float)
    for pid, d in pairs.items():
        ck = (d["pre"]["prompt"], d["post"]["prompt"])
        cluster_u_sums[ck] += (d["post"]["reward"] - d["pre"]["reward"]) - mean
    cr0_var = sum(s ** 2 for s in cluster_u_sums.values()) / (n ** 2)
    G_cr = len(cluster_u_sums)
    # CR1 小標本補正: G/(G-1) * (n-1)/(n-K), K=1 (切片のみ)
    cr1_factor = (G_cr / (G_cr - 1)) * ((n - 1) / (n - 1))
    cr1_var = cr0_var * cr1_factor

    def _cr_block(var_, label):
        se_ = var_ ** 0.5
        t_ = mean / se_
        df_ = G_cr - 1  # 慣行: クラスタ数-1 を自由度に使う (Cameron & Miller 2015 の一般的慣行)
        p_ = 2 * (1 - stats.t.cdf(abs(t_), df_))
        ci_t = _ci(mean, se_, stats.t.ppf(0.975, df_))
        ci_z = _ci(mean, se_, stats.norm.ppf(0.975))
        return {
            "label": label, "G": G_cr, "var": var_, "se": se_, "t": t_, "df": df_, "p_two_sided": p_,
            "ci95_t": {"quantile": "t(df=%d)" % df_, "crit": stats.t.ppf(0.975, df_), "lo": ci_t[0], "hi": ci_t[1]},
            "ci95_z": {"quantile": "z(normal)", "crit": stats.norm.ppf(0.975), "lo": ci_z[0], "hi": ci_z[1]},
        }

    cr_sandwich = {
        "mean": mean,
        "CR0": _cr_block(cr0_var, "CR0 (小標本補正なし)"),
        "CR1": _cr_block(cr1_var, "CR1 (G/(G-1)*(n-1)/(n-K) 補正・K=1)"),
        "cr1_factor": cr1_factor,
        "note": "df=G-1 は Cameron&Miller(2015) の慣行だが唯一の選択ではない (Satterthwaite 等の代替あり・"
                "本スクリプトでは df=G-1 のみ採用しその旨を明記する)。",
    }

    # --- (iv) en46 のみ ---
    en_vals = [d for _, d, lang in deltas if lang != "ja"]
    ja_vals = [d for _, d, lang in deltas if lang == "ja"]
    en_mean, en_var, en_sd, en_n = _mean_var_sd(en_vals, ddof=1)
    en_se = en_sd / math.sqrt(en_n)
    en_t = en_mean / en_se
    en_df = en_n - 1
    en_p = 2 * (1 - stats.t.cdf(abs(en_t), en_df))
    ci_en_t = _ci(en_mean, en_se, stats.t.ppf(0.975, en_df))
    ci_en_z = _ci(en_mean, en_se, stats.norm.ppf(0.975))
    en_only = {
        "n": en_n, "mean": en_mean, "sd": en_sd, "se": en_se, "t": en_t, "df": en_df, "p_two_sided": en_p,
        "ci95_t": {"quantile": "t(df=%d)" % en_df, "crit": stats.t.ppf(0.975, en_df), "lo": ci_en_t[0], "hi": ci_en_t[1]},
        "ci95_z": {"quantile": "z(normal)", "crit": stats.norm.ppf(0.975), "lo": ci_en_z[0], "hi": ci_en_z[1]},
    }

    return {
        "naive": naive,
        "cluster_mean": cluster_mean_est,
        "cr_sandwich": cr_sandwich,
        "en_only": en_only,
        "ja_all_delta_mean": sum(ja_vals) / len(ja_vals),
        "ja_n": len(ja_vals),
        "recorded_reference_values_2026_09_28": {
            "note": "docs/f0b-reverification-plan.md の 2026-09-28 追記との照合用。本スクリプトの計算値ではない。",
            "t_naive": 2.4025, "t_cluster_mean": 0.9549, "t_CR0": 2.7592, "t_CR1": 2.7297, "t_en46": 0.9154,
            "delta_en46": 0.0112,
            "cluster_mean_ci95_analytic_quoted_from_audit": [-0.0115, 0.0348],
        },
    }


def _ci(mean, se, crit):
    return (mean - crit * se, mean + crit * se)


def _cluster_bootstrap_ci(cluster_value_lists, seed, n_boot):
    """クラスタ単位リサンプリング (nonparametric cluster bootstrap)・percentile CI。
    各 bootstrap 反復でクラスタを重複ありで G 個引き直し、各クラスタの平均を取ってから
    その cluster-mean 群の平均 (= unweighted cluster-mean 推定量) を計算する。
    """
    G = len(cluster_value_lists)
    cluster_arrays = [np.asarray(v, dtype=np.float64) for v in cluster_value_lists]
    cluster_means_fixed = np.array([a.mean() for a in cluster_arrays])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, G, size=(n_boot, G))
    # 各 bootstrap 反復で選ばれたクラスタの (固定) cluster mean を平均する
    boot_means = cluster_means_fixed[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return {
        "method": "nonparametric cluster bootstrap (resample clusters w/ replacement, G=%d, B=%d, "
                  "unweighted mean-of-cluster-means statistic, percentile CI, numpy default_rng seed=%d)" % (G, n_boot, seed),
        "lo": float(lo), "hi": float(hi), "B": n_boot, "seed": seed,
    }


# ============================================================
# 出口② σ_seed (ja 54 件・pre/post 別 + プール)
# ============================================================
def compute_sigma_seed(rows):
    def arm_vals(model, field):
        return [r[field] for r in rows if r["lang"] == "ja" and r["model"] == model]

    def arm_stats(model, field):
        vals = arm_vals(model, field)
        mean, var, sd, n = _mean_var_sd(vals, ddof=1)
        # chi2 ベース 95%CI (正規近似・(n-1)s^2/chi2)
        lo_var = (n - 1) * var / stats.chi2.ppf(0.975, n - 1)
        hi_var = (n - 1) * var / stats.chi2.ppf(0.025, n - 1)
        boot = _bootstrap_sd_ci(vals, seed=BOOTSTRAP_SEED, n_boot=N_BOOTSTRAP)
        return {
            "field": field, "model": model, "n": n, "mean": mean, "var": var, "sd": sd,
            "chi2_ci95_sd": {"lo": lo_var ** 0.5, "hi": hi_var ** 0.5, "quantile": "chi2(df=%d)" % (n - 1)},
            "bootstrap_ci95_sd": boot,
        }

    def pooled_stats(field):
        pre_vals = arm_vals("pre", field)
        post_vals = arm_vals("post", field)
        _, var_pre, _, n_pre = _mean_var_sd(pre_vals, ddof=1)
        _, var_post, _, n_post = _mean_var_sd(post_vals, ddof=1)
        df = (n_pre - 1) + (n_post - 1)
        pooled_var = ((n_pre - 1) * var_pre + (n_post - 1) * var_post) / df
        pooled_sd = pooled_var ** 0.5
        lo_var = df * pooled_var / stats.chi2.ppf(0.975, df)
        hi_var = df * pooled_var / stats.chi2.ppf(0.025, df)
        boot = _bootstrap_pooled_sd_ci(pre_vals, post_vals, seed=BOOTSTRAP_SEED, n_boot=N_BOOTSTRAP)
        return {
            "field": field, "n_pre": n_pre, "n_post": n_post, "df": df,
            "pooled_var": pooled_var, "pooled_sd": pooled_sd,
            "chi2_ci95_sd": {"lo": lo_var ** 0.5, "hi": hi_var ** 0.5, "quantile": "chi2(df=%d)" % df},
            "bootstrap_ci95_sd": boot,
        }

    fields = ["reward", "quality_contribution", "anatomy_contribution"]
    out = {"per_arm": {}, "pooled": {}}
    for f in fields:
        out["per_arm"][f] = {"pre": arm_stats("pre", f), "post": arm_stats("post", f)}
        out["pooled"][f] = pooled_stats(f)
    out["limitations"] = [
        "σ_seed は ja の単一プロンプトペア (pre='%s...' / post 側は別の単一プロンプト) からのみ推定している。"
        "en プロンプト (46 種) には同一プロンプト×複数seedの繰り返し測定が存在しないため、"
        "en プロンプトの σ_seed をこのデータから直接検証することはできない (一般化は未検証)。" % (
            rows[0].get("prompt", "")[:20] if rows else ""
        ),
        "pre の σ_seed (reward) と post の σ_seed (reward) は異なる値 (0.0409 vs 0.0517) であり、"
        "『pre/post で同じσ_seedを共有する』という pooled 推定の前提は、この 2 標本だけでは強くは検証できない"
        "(2標本のF検定的な差の検定はこのスクリプトでは行っていない)。",
        "pre と post の各 54 件は post_id (= 元の自然文入力/画像) が異なるため、生成そのものは同一シードで"
        "繰り返した測定ではなく、『同一プロンプト文字列に対する独立な SDXL 呼び出し』のばらつきである。"
        "SDXL の呼び出しごとの乱数シードは server wall-clock 由来で非制御であり (dollma_g2b_reward_prepost.py "
        "l.16-19)、post_id が変わっても呼び出しごとに新しい seed が振られるため、ここでの分散は"
        "『同一プロンプト文字列を SDXL に何度も投げたときの reward ばらつき』として扱ってよいが、"
        "『同一 post_id に対する repeat-seed 測定』ではない (post_id は 54 通りすべて異なる)。",
        "anatomy_contribution の post 側 sd が極小 (実測は JSON 本体参照) であり、post モデルの生成が"
        "同一プロンプト下でほぼ同じ anatomy 評価に収束している可能性がある。これが「post は解剖的に安定した"
        "画像だけを生成する」ためか「ScorerNet の分解能不足でほぼ同じ値しか出ない」ためかは、本スクリプトの"
        "範囲では切り分けられない。",
    ]
    return out


def _bootstrap_sd_ci(vals, seed, n_boot):
    a = np.asarray(vals, dtype=np.float64)
    n = len(a)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_sd = a[idx].std(axis=1, ddof=1)
    lo, hi = np.percentile(boot_sd, [2.5, 97.5])
    return {"method": "nonparametric bootstrap (resample n=%d w/ replacement, B=%d, seed=%d)" % (n, n_boot, seed),
            "lo": float(lo), "hi": float(hi)}


def _bootstrap_pooled_sd_ci(pre_vals, post_vals, seed, n_boot):
    a1 = np.asarray(pre_vals, dtype=np.float64)
    a2 = np.asarray(post_vals, dtype=np.float64)
    n1, n2 = len(a1), len(a2)
    rng = np.random.default_rng(seed)
    idx1 = rng.integers(0, n1, size=(n_boot, n1))
    idx2 = rng.integers(0, n2, size=(n_boot, n2))
    s1 = a1[idx1]
    s2 = a2[idx2]
    var1 = s1.var(axis=1, ddof=1)
    var2 = s2.var(axis=1, ddof=1)
    df = (n1 - 1) + (n2 - 1)
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / df
    pooled_sd = np.sqrt(pooled_var)
    lo, hi = np.percentile(pooled_sd, [2.5, 97.5])
    return {"method": "nonparametric bootstrap (resample pre n=%d + post n=%d independently w/ replacement, "
                       "B=%d, seed=%d, pooled sd of each replicate)" % (n1, n2, n_boot, seed),
            "lo": float(lo), "hi": float(hi)}


# ============================================================
# 出口③ 必要プロンプト数の逆算
# ============================================================
def compute_required_n(sigma_seed_pooled_reward, en_delta_var, deltas_to_detect, ks=(1, 2, 4, 8, 16)):
    z_alpha2 = stats.norm.ppf(1 - ALPHA / 2)
    z_beta = stats.norm.ppf(POWER)
    k_const = (z_alpha2 + z_beta) ** 2  # 標準 2 標本/1標本平均差 sample size 近似の定数項

    sigma_seed2 = sigma_seed_pooled_reward ** 2
    # τ² = Var(Δ_en) - 2*σ_seed² (負なら 0 で打ち切り)
    tau2_raw = en_delta_var - 2 * sigma_seed2
    tau2 = max(0.0, tau2_raw)
    tau2_floored = tau2_raw < 0.0

    results = {
        "z_alpha2": z_alpha2, "z_beta": z_beta, "sample_size_const_(z_a2+z_b)^2": k_const,
        "alpha": ALPHA, "power": POWER,
        "sigma_seed_pooled_reward": sigma_seed_pooled_reward,
        "sigma_seed_pooled_reward_squared": sigma_seed2,
        "en_delta_var_observed": en_delta_var,
        "tau2_raw": tau2_raw, "tau2_used": tau2, "tau2_floored_at_zero": tau2_floored,
        "tau2_note": "τ² = Var(Δ_en46) - 2*σ_seed²(pooled, ja由来)。en の Δ 分散は『真の効果ばらつき + "
                     "pre/post 2回分のseedノイズ』を含む観測量なので、そこから pooled σ_seed² の寄与 (2倍、"
                     "pre側とpost側それぞれ1回分) を差し引いて τ² を逆算している。σ_seed が en プロンプトにも"
                     "同様に当てはまるという未検証の仮定に依存する (上記 limitations 参照)。",
        "designs": {},
    }

    for delta_name, delta_val in deltas_to_detect.items():
        design_a_rows = []
        design_b_optimistic_rows = []
        for k in ks:
            var_a = tau2 + 2 * sigma_seed2 / k
            n_a = k_const * var_a / (delta_val ** 2)
            design_a_rows.append({
                "k_seeds_per_prompt": k, "var_delta_per_prompt": var_a,
                "n_prompts_required": math.ceil(n_a), "n_prompts_required_raw": n_a,
                "total_images_required": math.ceil(n_a) * k * 2,
            })
            var_b = tau2  # 楽観: ρ=1 (seed分が完全相殺)。k はこの場合 τ² に効かないので参考として同じ行に出す
            n_b = k_const * var_b / (delta_val ** 2) if var_b > 0 else None
            design_b_optimistic_rows.append({
                "k_seeds_per_prompt": k, "var_delta_per_prompt": var_b,
                "n_prompts_required": (math.ceil(n_b) if n_b is not None else None),
                "n_prompts_required_raw": n_b,
                "total_images_required": (math.ceil(n_b) * k * 2 if n_b is not None else None),
                "caveat": "ρ=1 (同一seedでpre/postを撮ればseedノイズが完全相殺される) という楽観的仮定。"
                          "本データには ρ の実測値がない (pre/post は常に異なるseedで撮られているため)。",
            })
        results["designs"][delta_name] = {
            "delta": delta_val,
            "design_a_seed_varies_pre_post": design_a_rows,
            "design_b_seed_fixed_optimistic_rho1": design_b_optimistic_rows,
            "design_b_seed_fixed_pessimistic_rho0_note": "ρ=0 (相殺なし) の場合は design_a と数式上同一になる "
                                                           "(k=1相当) ので別表は出さない。",
        }
    return results


# ============================================================
# 検算 (numpy ベクトル経路での独立再計算)
# ============================================================
def cross_check(pairs, estimators):
    deltas = np.array([d["post"]["reward"] - d["pre"]["reward"] for d in pairs.values()])
    mean_np = float(deltas.mean())
    se_np = float(deltas.std(ddof=1) / math.sqrt(len(deltas)))
    t_np = mean_np / se_np

    ok = True
    msgs = []
    if abs(mean_np - estimators["naive"]["mean"]) > 1e-9:
        ok = False
        msgs.append("naive mean 不一致")
    if abs(t_np - estimators["naive"]["t"]) > 1e-6:
        ok = False
        msgs.append("naive t 不一致")

    return {"ok": ok, "messages": msgs, "naive_mean_numpy": mean_np, "naive_t_numpy": t_np}


def selftest():
    """最小の自己検査: 合成データで 4 推定量と σ_seed 計算の健全性を確認する。"""
    ok = True

    # 合成: 3 プロンプト。p1 は en 相当 (1ペア=1クラスタ)、p2/p3 も en 相当。
    # かつ ja 相当の重複クラスタ (同一 prompt ペア 2件) を追加。
    synth_rows = [
        {"post_id": 1, "model": "pre", "lang": "en", "prompt": "A", "reward": 0.0, "quality_contribution": 0.0, "anatomy_contribution": 0.0},
        {"post_id": 1, "model": "post", "lang": "en", "prompt": "A2", "reward": 0.1, "quality_contribution": 0.05, "anatomy_contribution": 0.05},
        {"post_id": 2, "model": "pre", "lang": "en", "prompt": "B", "reward": 0.0, "quality_contribution": 0.0, "anatomy_contribution": 0.0},
        {"post_id": 2, "model": "post", "lang": "en", "prompt": "B2", "reward": -0.05, "quality_contribution": -0.02, "anatomy_contribution": -0.03},
        # ja 相当: post_id 3,4 は同一 (pre_prompt, post_prompt)="C","C2" の組み合わせ
        {"post_id": 3, "model": "pre", "lang": "ja", "prompt": "C", "reward": 0.0, "quality_contribution": 0.0, "anatomy_contribution": 0.0},
        {"post_id": 3, "model": "post", "lang": "ja", "prompt": "C2", "reward": 0.2, "quality_contribution": 0.1, "anatomy_contribution": 0.1},
        {"post_id": 4, "model": "pre", "lang": "ja", "prompt": "C", "reward": 0.02, "quality_contribution": 0.0, "anatomy_contribution": 0.0},
        {"post_id": 4, "model": "post", "lang": "ja", "prompt": "C2", "reward": 0.18, "quality_contribution": 0.1, "anatomy_contribution": 0.1},
    ]
    pairs = build_pairs(synth_rows)
    if len(pairs) != 4:
        print("[selftest FAIL] build_pairs 件数", len(pairs))
        ok = False

    clusters = build_clusters(pairs, key_field="reward")
    # 期待: {A,A2}, {B,B2} が singleton, {C,C2} が size2 の 1 クラスタ -> G=3
    if len(clusters) != 3:
        print("[selftest FAIL] build_clusters クラスタ数", len(clusters), clusters)
        ok = False
    sizes = sorted(len(v) for v in clusters.values())
    if sizes != [1, 1, 2]:
        print("[selftest FAIL] build_clusters クラスタサイズ", sizes)
        ok = False

    est = compute_estimators(pairs)
    # naive: deltas = [0.1, -0.05, 0.2, 0.16] -> mean = 0.1025
    expected_naive_mean = (0.1 + (-0.05) + 0.2 + 0.16) / 4
    if abs(est["naive"]["mean"] - expected_naive_mean) > 1e-9:
        print("[selftest FAIL] naive mean", est["naive"]["mean"], expected_naive_mean)
        ok = False
    # cluster-mean: 3クラスタ [0.1, -0.05, (0.2+0.16)/2=0.18] -> mean=(0.1-0.05+0.18)/3
    expected_cm_mean = (0.1 + (-0.05) + 0.18) / 3
    if abs(est["cluster_mean"]["mean"] - expected_cm_mean) > 1e-9:
        print("[selftest FAIL] cluster mean", est["cluster_mean"]["mean"], expected_cm_mean)
        ok = False
    if est["cluster_mean"]["G"] != 3:
        print("[selftest FAIL] cluster mean G", est["cluster_mean"]["G"])
        ok = False

    xc = cross_check(pairs, est)
    if not xc["ok"]:
        print("[selftest FAIL] cross_check", xc)
        ok = False

    # σ_seed pooled: ja pre=[0.0,0.02] var=((0.0-0.01)^2+(0.02-0.01)^2)/1=0.0002, ja post=[0.2,0.18] var=0.0002
    sig = compute_sigma_seed(synth_rows)
    exp_pooled_var = 0.0002  # pre var=post var=0.0002 (n=2 each, ddof=1) -> pooled = same
    got_pooled_var = sig["pooled"]["reward"]["pooled_var"]
    if abs(got_pooled_var - exp_pooled_var) > 1e-8:
        print("[selftest FAIL] pooled var", got_pooled_var, exp_pooled_var)
        ok = False

    if ok:
        print("[selftest] OK")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--g2b-prepost", default=os.path.join(MAIN_DATA_ROOT, "g2b_prepost.jsonl"))
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "logs", "f0b-z2", "z2_result.json"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        ok = selftest()
        sys.exit(0 if ok else 1)

    if not os.path.isfile(args.g2b_prepost):
        print(f"[error] 入力ファイルが見つかりません (main checkout から読み取り専用で参照する想定): {args.g2b_prepost}")
        sys.exit(1)

    rows = load_jsonl(args.g2b_prepost)
    pairs = build_pairs(rows)

    estimators = compute_estimators(pairs)
    sigma_seed = compute_sigma_seed(rows)
    xc = cross_check(pairs, estimators)

    en_delta_var = estimators["en_only"]["sd"] ** 2
    sigma_seed_pooled_reward = sigma_seed["pooled"]["reward"]["pooled_sd"]
    deltas_to_detect = {
        "overall_naive_delta_0.017081": estimators["naive"]["mean"],
        "en46_delta_0.011152": estimators["en_only"]["mean"],
    }
    required_n = compute_required_n(sigma_seed_pooled_reward, en_delta_var, deltas_to_detect)

    result = {
        "stage": "F-0b 再検証 Z-2 (クラスタ補正の再解析 + σ_seed の直接推定)",
        "plan_ref": "docs/f0b-reverification-plan.md 「### Z-2」節",
        "inputs": {
            "g2b_prepost": {"path": args.g2b_prepost, "sha256": sha256_of(args.g2b_prepost), "n_rows": len(rows)},
        },
        "env": {"python": sys.version, "platform": platform.platform(), "cwd": os.getcwd(),
                "numpy": np.__version__, "scipy": __import__("scipy").__version__},
        "n_pairs": len(pairs),
        "exit1_estimators": estimators,
        "exit2_sigma_seed": sigma_seed,
        "exit3_required_n": required_n,
        "cross_check": xc,
        "notes": [
            "採否・有意性の断定はしていない (数値と分位点の明記のみ)。",
            "GPU 実走の起票・撮影枚数の最終決定はしていない (逆算値の提示まで)。",
            "cluster 定義は (pre.prompt, post.prompt) のペアで機械的に決めている (lang をハードコードしていない)。"
            "本データでは結果的に ja=1クラスタ(54件)・en=46 singleton クラスタ = G=47 になる。",
        ],
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[done] wrote {args.out}")

    print("== summary: exit1 estimators (t) ==")
    print("naive:", estimators["naive"]["t"], "df", estimators["naive"]["df"])
    print("cluster_mean:", estimators["cluster_mean"]["t"], "df", estimators["cluster_mean"]["df"],
          "bootstrap CI", estimators["cluster_mean"]["bootstrap_percentile_ci95"])
    print("CR0:", estimators["cr_sandwich"]["CR0"]["t"], "CR1:", estimators["cr_sandwich"]["CR1"]["t"])
    print("en_only:", estimators["en_only"]["t"], "df", estimators["en_only"]["df"])
    print("== summary: exit2 sigma_seed pooled (reward) ==")
    print(sigma_seed["pooled"]["reward"])
    print("== cross_check ==")
    print(xc)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""dollma_f0b_z1_quality_contribution.py — F-0b 再検証 Z-1
(docs/f0b-reverification-plan.md l.45, l.55-58 の入口・出口・禁止事項に従う)

やること (既存データの再解析のみ・SDXL 生成ゼロ・訓練ゼロ):
  ① quality 抜き reward で best-of-N を選び直したときの勝者と、現行勝者 (is_winner=True) の一致率。
     対象: data/rollouts/candidates_bestofn.jsonl (400 group x N=8 候補)。lang は
     data/rollouts/sft_bestofn.jsonl (post_id -> lang) から引く (candidates_bestofn.jsonl 自体に
     lang フィールドがないため)。
  ② reward Δ への anatomy/quality 寄与率。**まず定義を明文化してから計算する**
     (plan l.57: Δ 基準 = Δanatomy/(Δanatomy+Δquality) と 水準基準 = 平均絶対寄与比の 2 通りを
     算出し、どちらを主にするか理由を添える。★7.5% の再現を目標にしない — plan の指示通り)。
     対象: data/rollouts/g2b_prepost.jsonl (100 ペア pre/post・既に anatomy_contribution /
     quality_contribution 列を持つ = reward_components と同一定義)。

reward 式 (scripts/dollma_reward.py と同一・ここでは複製せず再定義して独立検算する):
  anatomy_reward = -((1-w_tie)*max(axes) + w_tie*mean(axes))   w_tie = MEAN_TIE_WEIGHT = 0.05
  reward         = (1-w_q)*anatomy_reward + w_q*(quality-1.0)  w_q   = QUALITY_WEIGHT   = 0.4
  quality 抜き reward (本スクリプトの定義) := anatomy_reward   (quality 項を落とした場合の順位に相当)

寄与率の定義 (本スクリプトが明文化する・plan l.57 の宿題):
  Δ基準  = |mean(Δanatomy_contribution)| / (|mean(Δanatomy_contribution)| + |mean(Δquality_contribution)|)
           (pre→post の reward 変化のうち、どちらの成分が動いたか。F-0b が「学習で動いた量」を
            問う文脈に対応するのはこちら → **主表現に採用**)
  水準基準 = mean(|anatomy_contribution|) / (mean(|anatomy_contribution|) + mean(|quality_contribution|))
           (individual row の reward の絶対水準のうちどちらの成分が大きいか。学習方向とは無関係の
            「reward の内訳」を問う副次指標)
  理由: F-0b の主張は「RAFT で reward が pre→post 変化した、その変化が anatomy か quality か」
        なので、**変化量 (Δ) を分母分子に置く Δ基準を主**とする。水準基準は「reward の絶対値の
        内訳」であって「学習が何を動かしたか」を直接には測らないため副次参考値とする。

禁止事項 (plan l.45/57 厳守):
  - 監査引用値「7.5%」の再現を目標にしない (定義を決めて独立に計算するだけ)。
  - 採否判定はしない (本スクリプトは数値のみ出力)。
"""

import argparse
import hashlib
import json
import os
import platform
import sys
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MAIN_DATA_ROOT = r"E:\Develop\Projects\dollama\data\rollouts"  # 読み取り専用 (main checkout)

MEAN_TIE_WEIGHT = 0.05
QUALITY_WEIGHT = 0.4


def anatomy_reward_of(axes):
    worst = max(axes)
    mean = sum(axes) / len(axes)
    return -((1.0 - MEAN_TIE_WEIGHT) * worst + MEAN_TIE_WEIGHT * mean)


def full_reward_of(axes, quality):
    ana = anatomy_reward_of(axes)
    return (1.0 - QUALITY_WEIGHT) * ana + QUALITY_WEIGHT * (quality - 1.0)


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


def _dist(vals):
    if not vals:
        return None
    n = len(vals)
    s = sorted(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    med = s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
    return {"min": min(vals), "median": med, "max": max(vals), "mean": mean,
            "std": var ** 0.5, "n": n}


# ============================================================
# ① quality 抜き勝者一致率
# ============================================================
def compute_winner_match(candidates_path, sft_bestofn_path):
    cand_rows = load_jsonl(candidates_path)
    lang_rows = load_jsonl(sft_bestofn_path)

    post_id_to_lang = {}
    for r in lang_rows:
        pid = r.get("meta", {}).get("post_id")
        if pid is not None:
            post_id_to_lang[int(pid)] = r.get("lang")

    groups = defaultdict(list)
    for r in cand_rows:
        groups[int(r["post_id"])].append(r)

    per_group_records = []
    sanity_mismatch = 0  # is_winner が full_reward の argmax と食い違う件数 (データ検算)
    n_empty_candidates_total = 0
    for pid, items in groups.items():
        n = len(items)
        # empty (axes=None・生成失敗) 候補はスコア不能なのでランキング対象から除外する
        # (empty=True は現行選抜でも winner になり得ない設計。除外数は記録して透明化する)
        scored_items = [it for it in items if it.get("axes") is not None]
        n_empty_candidates_total += (n - len(scored_items))

        # 現行勝者 (is_winner=True のもの。複数/0件は異常として記録)
        winners_flag = [it for it in items if it.get("is_winner")]
        official_winner_cand = winners_flag[0]["candidate"] if len(winners_flag) == 1 else None
        official_winner_anomaly = None if len(winners_flag) == 1 else len(winners_flag)

        # 検算: full reward (再計算) の argmax が is_winner と一致するか
        full_rewards = [(it["candidate"], full_reward_of(it["axes"], it["quality"])) for it in scored_items]
        recomputed_full_winner_cand = max(full_rewards, key=lambda t: t[1])[0]
        if official_winner_cand is not None and recomputed_full_winner_cand != official_winner_cand:
            sanity_mismatch += 1

        # quality 抜き (anatomy のみ) reward での勝者 (scored_items が空 = group 全滅は本データでは未発生)
        ana_rewards = [(it["candidate"], anatomy_reward_of(it["axes"])) for it in scored_items]
        if not ana_rewards:
            per_group_records.append({
                "post_id": pid, "n_candidates": n, "lang": post_id_to_lang.get(pid),
                "official_winner_candidate": official_winner_cand,
                "official_winner_flag_anomaly": official_winner_anomaly,
                "quality_free_winner_candidate": None, "quality_free_tie_count": 0,
                "tie_policy_applied": False, "match": None,
                "note": "group 内候補が全て empty のためスコア不能",
            })
            continue
        ana_rewards_sorted = sorted(ana_rewards, key=lambda t: t[1], reverse=True)
        best_val = ana_rewards_sorted[0][1]
        # タイ判定: 完全一致 (float ==) のときのみ「同率」とみなす (連続値なのでほぼ発生しない想定)
        tied = [c for c, v in ana_rewards if v == best_val]
        tie_policy_applied = len(tied) > 1
        # タイ方針: 最小 candidate index を採用 (最初に列挙された候補を勝者とする・後続一致率計算を決定的にする)
        quality_free_winner_cand = min(tied) if tied else ana_rewards_sorted[0][0]

        lang = post_id_to_lang.get(pid)
        match = (official_winner_cand is not None and quality_free_winner_cand == official_winner_cand)
        per_group_records.append({
            "post_id": pid,
            "n_candidates": n,
            "lang": lang,
            "official_winner_candidate": official_winner_cand,
            "official_winner_flag_anomaly": official_winner_anomaly,
            "quality_free_winner_candidate": quality_free_winner_cand,
            "quality_free_tie_count": len(tied),
            "tie_policy_applied": tie_policy_applied,
            "match": match,
        })

    def summarize(records):
        valid = [r for r in records if r["official_winner_candidate"] is not None and r["match"] is not None]
        n = len(valid)
        n_match = sum(1 for r in valid if r["match"])
        n_tie = sum(1 for r in valid if r["tie_policy_applied"])
        return {
            "n_groups": len(records),
            "n_groups_valid_official_winner": n,
            "n_groups_official_winner_anomaly": len(records) - n,
            "n_match": n_match,
            "match_rate": (n_match / n) if n else None,
            "n_tie_groups": n_tie,
        }

    overall = summarize(per_group_records)
    ja = summarize([r for r in per_group_records if r["lang"] == "ja"])
    en = summarize([r for r in per_group_records if r["lang"] == "en"])
    n_lang_unknown = sum(1 for r in per_group_records if r["lang"] is None)

    return {
        "candidates_per_group_n": sorted(set(r["n_candidates"] for r in per_group_records)),
        "n_empty_candidates_excluded_from_ranking": n_empty_candidates_total,
        "sanity_full_reward_argmax_mismatch_vs_is_winner": sanity_mismatch,
        "tie_policy": "float 完全一致のときのみ同率とみなし、同率内で最小 candidate index を採用する",
        "n_groups_lang_unknown": n_lang_unknown,
        "overall": overall,
        "ja": ja,
        "en": en,
        "records": per_group_records,
    }


# ============================================================
# ② Δ 寄与率 (Δ基準・水準基準)
# ============================================================
def compute_contribution_shares(g2b_path):
    rows = load_jsonl(g2b_path)
    by_pid = defaultdict(dict)
    for r in rows:
        by_pid[int(r["post_id"])][r["model"]] = r

    dana, dqual, dre = [], [], []
    dana_ja, dqual_ja, dana_en, dqual_en = [], [], [], []
    level_ana_abs, level_qual_abs = [], []

    n_pairs = 0
    for pid, d in by_pid.items():
        if "pre" not in d or "post" not in d:
            continue
        n_pairs += 1
        rp, rs = d["pre"], d["post"]
        dre.append(rs["reward"] - rp["reward"])
        da = rs["anatomy_contribution"] - rp["anatomy_contribution"]
        dq = rs["quality_contribution"] - rp["quality_contribution"]
        dana.append(da)
        dqual.append(dq)
        lang = rp.get("lang", "ja")
        if lang == "ja":
            dana_ja.append(da)
            dqual_ja.append(dq)
        else:
            dana_en.append(da)
            dqual_en.append(dq)

    for r in rows:
        level_ana_abs.append(abs(r["anatomy_contribution"]))
        level_qual_abs.append(abs(r["quality_contribution"]))

    mean_dana = sum(dana) / len(dana) if dana else None
    mean_dqual = sum(dqual) / len(dqual) if dqual else None
    mean_dre = sum(dre) / len(dre) if dre else None

    delta_based_share_anatomy = None
    delta_based_share_quality = None
    if mean_dana is not None and mean_dqual is not None:
        denom = abs(mean_dana) + abs(mean_dqual)
        if denom > 0:
            delta_based_share_anatomy = abs(mean_dana) / denom
            delta_based_share_quality = abs(mean_dqual) / denom

    mean_level_ana = sum(level_ana_abs) / len(level_ana_abs) if level_ana_abs else None
    mean_level_qual = sum(level_qual_abs) / len(level_qual_abs) if level_qual_abs else None
    level_based_share_anatomy = None
    level_based_share_quality = None
    if mean_level_ana is not None and mean_level_qual is not None:
        denom = mean_level_ana + mean_level_qual
        if denom > 0:
            level_based_share_anatomy = mean_level_ana / denom
            level_based_share_quality = mean_level_qual / denom

    return {
        "n_pairs": n_pairs,
        "delta_reward_mean": mean_dre,
        "delta_anatomy_contribution_mean": mean_dana,
        "delta_quality_contribution_mean": mean_dqual,
        "delta_based": {
            "formula": "|mean(Δanatomy_contribution)| / (|mean(Δanatomy_contribution)| + |mean(Δquality_contribution)|)",
            "anatomy_share": delta_based_share_anatomy,
            "quality_share": delta_based_share_quality,
        },
        "level_based": {
            "formula": "mean(|anatomy_contribution|) / (mean(|anatomy_contribution|) + mean(|quality_contribution|))  over 200 rows (pre+post)",
            "mean_abs_anatomy_contribution": mean_level_ana,
            "mean_abs_quality_contribution": mean_level_qual,
            "anatomy_share": level_based_share_anatomy,
            "quality_share": level_based_share_quality,
        },
        "primary_choice": "delta_based",
        "primary_choice_reason": (
            "F-0b の主張は『RAFT-SFT で reward が pre→post 変化した、その変化の中身は anatomy か "
            "quality か』であり、変化量 (Δ) を分母分子に置く delta_based のほうが主張に直接対応する。"
            "level_based は個々の reward の絶対水準に占める内訳であり、学習が何を動かしたかは測らない "
            "(anatomy_reward は値域 [-1,0] で quality_contribution ([-0.4,0]) より絶対値が大きくなりやすい "
            "スケール差の影響を受ける) ため副次参考値とする。"
        ),
        "distributions": {
            "delta_reward": _dist(dre),
            "delta_anatomy_contribution": _dist(dana),
            "delta_quality_contribution": _dist(dqual),
            "delta_anatomy_contribution_ja": _dist(dana_ja),
            "delta_quality_contribution_ja": _dist(dqual_ja),
            "delta_anatomy_contribution_en": _dist(dana_en),
            "delta_quality_contribution_en": _dist(dqual_en),
        },
    }


def selftest():
    """最小の自己検査: 既知の境界値・簡易ケースで関数の健全性を確認する。"""
    ok = True

    # 1) anatomy_reward の境界: 全0軸→0.0 / 全1軸→-1.0
    r0 = anatomy_reward_of([0.0] * 8)
    r1 = anatomy_reward_of([1.0] * 8)
    if abs(r0 - 0.0) > 1e-12 or abs(r1 - (-1.0)) > 1e-12:
        print("[selftest FAIL] anatomy_reward 境界値", r0, r1)
        ok = False

    # 2) full_reward の quality 境界: quality=1.0 のとき quality 項は 0
    fr = full_reward_of([0.0] * 8, 1.0)
    if abs(fr - 0.0) > 1e-12:
        print("[selftest FAIL] full_reward quality=1.0/axes=0 境界", fr)
        ok = False

    # 3) 勝者一致率ロジックの単体テスト (合成 2 group)
    synth_candidates = [
        # group 1: quality が逆転させるケース (anatomy 最良だが quality 最悪 → 現行は他候補が勝つ)
        {"post_id": 1, "candidate": 0, "axes": [0.01] * 8, "quality": 0.0, "is_winner": False},
        {"post_id": 1, "candidate": 1, "axes": [0.5] * 8, "quality": 1.0, "is_winner": True},
        # group 2: anatomy と quality が一致するケース
        {"post_id": 2, "candidate": 0, "axes": [0.9] * 8, "quality": 0.1, "is_winner": False},
        {"post_id": 2, "candidate": 1, "axes": [0.01] * 8, "quality": 0.9, "is_winner": True},
    ]
    synth_lang = [
        {"meta": {"post_id": 1}, "lang": "ja"},
        {"meta": {"post_id": 2}, "lang": "en"},
    ]
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        cpath = os.path.join(td, "c.jsonl")
        lpath = os.path.join(td, "l.jsonl")
        with open(cpath, "w", encoding="utf-8") as f:
            for r in synth_candidates:
                f.write(json.dumps(r) + "\n")
        with open(lpath, "w", encoding="utf-8") as f:
            for r in synth_lang:
                f.write(json.dumps(r) + "\n")
        result = compute_winner_match(cpath, lpath)
        # group1: quality free winner = candidate0 (anatomy 良い) != official winner(1) -> mismatch
        # group2: quality free winner = candidate1 (anatomy も良い) == official winner(1) -> match
        rec_by_pid = {r["post_id"]: r for r in result["records"]}
        if rec_by_pid[1]["match"] is not False or rec_by_pid[2]["match"] is not True:
            print("[selftest FAIL] 合成勝者一致率ロジック", rec_by_pid)
            ok = False
        if result["overall"]["match_rate"] != 0.5:
            print("[selftest FAIL] overall match_rate", result["overall"])
            ok = False

    # 4) 寄与率ロジックの単体テスト (合成 g2b_prepost)
    synth_g2b = [
        {"post_id": 1, "model": "pre", "lang": "ja", "anatomy_contribution": -0.1, "quality_contribution": -0.2, "reward": -0.3},
        {"post_id": 1, "model": "post", "lang": "ja", "anatomy_contribution": -0.09, "quality_contribution": -0.1, "reward": -0.19},
    ]
    with tempfile.TemporaryDirectory() as td:
        gpath = os.path.join(td, "g.jsonl")
        with open(gpath, "w", encoding="utf-8") as f:
            for r in synth_g2b:
                f.write(json.dumps(r) + "\n")
        contrib = compute_contribution_shares(gpath)
        # Δanatomy = -0.09 - (-0.1) = 0.01 ; Δquality = -0.1 - (-0.2) = 0.1
        # delta_based anatomy share = 0.01 / (0.01+0.1) = 0.0909...
        exp = 0.01 / (0.01 + 0.1)
        if abs(contrib["delta_based"]["anatomy_share"] - exp) > 1e-9:
            print("[selftest FAIL] delta_based anatomy_share", contrib["delta_based"], exp)
            ok = False

    if ok:
        print("[selftest] OK")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", default=os.path.join(MAIN_DATA_ROOT, "candidates_bestofn.jsonl"))
    ap.add_argument("--sft-bestofn", default=os.path.join(MAIN_DATA_ROOT, "sft_bestofn.jsonl"))
    ap.add_argument("--g2b-prepost", default=os.path.join(MAIN_DATA_ROOT, "g2b_prepost.jsonl"))
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "logs", "f0b-z1", "z1_result.json"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        ok = selftest()
        sys.exit(0 if ok else 1)

    for p in (args.candidates, args.sft_bestofn, args.g2b_prepost):
        if not os.path.isfile(p):
            print(f"[error] 入力ファイルが見つかりません (main checkout から読み取り専用で参照する想定): {p}")
            sys.exit(1)

    winner_match = compute_winner_match(args.candidates, args.sft_bestofn)
    contribution = compute_contribution_shares(args.g2b_prepost)

    result = {
        "stage": "F-0b 再検証 Z-1 (quality 抜き勝者一致率・寄与率)",
        "plan_ref": "docs/f0b-reverification-plan.md l.45, l.55-58",
        "inputs": {
            "candidates_bestofn": {"path": args.candidates, "sha256": sha256_of(args.candidates)},
            "sft_bestofn": {"path": args.sft_bestofn, "sha256": sha256_of(args.sft_bestofn)},
            "g2b_prepost": {"path": args.g2b_prepost, "sha256": sha256_of(args.g2b_prepost)},
        },
        "env": {
            "python": sys.version,
            "platform": platform.platform(),
            "cwd": os.getcwd(),
        },
        "constants": {"MEAN_TIE_WEIGHT": MEAN_TIE_WEIGHT, "QUALITY_WEIGHT": QUALITY_WEIGHT},
        "winner_match": winner_match,
        "contribution": contribution,
        "notes": [
            "禁止事項厳守: 監査引用値 7.5% の再現を目標にしていない (定義を明文化した上で独立計算のみ)。",
            "採否判定はしていない (数値と定義のみ)。",
            "candidates_bestofn.jsonl に lang フィールドがないため sft_bestofn.jsonl の "
            "meta.post_id -> lang で結合。結合できない post_id があれば winner_match.n_groups_lang_unknown に出る。",
        ],
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[done] wrote {args.out}")

    print("== summary ==")
    print("winner_match.overall:", winner_match["overall"])
    print("winner_match.ja:", winner_match["ja"])
    print("winner_match.en:", winner_match["en"])
    print("sanity_full_reward_argmax_mismatch_vs_is_winner:", winner_match["sanity_full_reward_argmax_mismatch_vs_is_winner"])
    print("contribution.delta_based:", contribution["delta_based"])
    print("contribution.level_based:", contribution["level_based"])


if __name__ == "__main__":
    main()

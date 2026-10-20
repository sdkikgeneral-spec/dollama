# -*- coding: utf-8 -*-
"""dollma_e2_t2_analyze.py — E-2 T2: CFG/steps スイープの3軸判定 (研究機・分析専用)。

読み込む CSV (dollma_eval_image_grid.py --run が出す grid_results.csv 相当) を preset/
prompt_id/seed でペアリングし、参照アーム (cfg=0.0=既定7.5相当) との per-seed delta を
以下 3 軸で判定する:
  ① signed delta の符号一致 (全 prompt×seed で recall_delta / worst_anatomy_delta の符号が
     揃うか。揃わなければ「レバーとして機能していない (ノイズ支配)」)。
  ② 参照アーム自身の同条件・異 seed 分散帯を超えるか (delta の絶対値が参照アームの std を
     上回らなければ「ノイズ床の中」と判定)。
  ③ paired CI (delta の平均 ± 1.96*SEM が 0 を跨がないか)。

采点器の穴 (F-0a 実測) により、anatomy 系は Limbs 以外の 7 軸が分解能不足なので
worst_anatomy / axis 系を主指標にはしない。recall (WD14再現率) を主指標に、
worst_anatomy は参考値として併記する。
"""
import argparse
import csv
import math
import os
import sys


def load_rows(csv_path):
    with open(csv_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(v):
    if v is None or v == "" or v == "None":
        return None
    return float(v)


def index_by_seed(rows):
    """(preset, prompt_id, seed) -> row の辞書を作る。"""
    return {(r["preset"], r["prompt_id"], r["seed"]): r for r in rows}


def mean_std_sem(values):
    n = len(values)
    if n == 0:
        return None, None, None
    m = sum(values) / n
    if n < 2:
        return m, 0.0, 0.0
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    std = var ** 0.5
    sem = std / (n ** 0.5)
    return m, std, sem


def compare_arms(ref_rows, other_rows, metric, label):
    """ref_rows/other_rows は同じ (preset, prompt_id, seed) 直積の行リスト (順序対応不要・
    キーでペアリング)。3軸判定を print し、結果 dict を返す。"""
    ref_idx = index_by_seed(ref_rows)
    deltas = []
    signs = []
    for key, orow in {(r["preset"], r["prompt_id"], r["seed"]): r for r in other_rows}.items():
        rrow = ref_idx.get(key)
        if rrow is None:
            continue
        rv = to_float(rrow.get(metric))
        ov = to_float(orow.get(metric))
        if rv is None or ov is None:
            continue
        d = ov - rv
        deltas.append(d)
        signs.append(1 if d > 0 else (-1 if d < 0 else 0))

    n = len(deltas)
    if n == 0:
        print(f"[{label}] {metric}: 有効ペアなし (SKIP)")
        return {"label": label, "metric": metric, "n": 0}

    dmean, dstd, dsem = mean_std_sem(deltas)

    # 軸①: 符号一致 (全ペアで同符号 or 0)
    nonzero_signs = [s for s in signs if s != 0]
    sign_consistent = len(set(nonzero_signs)) <= 1 if nonzero_signs else True
    majority_sign = (max(set(signs), key=signs.count) if signs else 0)

    # 軸②: 参照アーム自身の分散帯 (同条件・異seedのref値のstd) を分母に、delta絶対値と比較
    #   参照アーム内 (prompt_id ごとの) recall/worst の std を求める (seed違いのみで条件同一)。
    ref_by_cond = {}
    for r in ref_rows:
        ck = (r["preset"], r["prompt_id"])
        ref_by_cond.setdefault(ck, []).append(to_float(r.get(metric)))
    noise_stds = []
    for ck, vals in ref_by_cond.items():
        vals = [v for v in vals if v is not None]
        if len(vals) >= 2:
            _, s, _ = mean_std_sem(vals)
            noise_stds.append(s)
    noise_floor = (sum(noise_stds) / len(noise_stds)) if noise_stds else None
    exceeds_noise = (noise_floor is not None and noise_floor > 0
                      and abs(dmean) > noise_floor)

    # 軸③: paired CI (95%, delta_mean ± 1.96*SEM が 0 を跨がないか)
    ci_lo = dmean - 1.96 * dsem if dsem is not None else None
    ci_hi = dmean + 1.96 * dsem if dsem is not None else None
    ci_excludes_zero = (ci_lo is not None and ci_hi is not None
                         and (ci_lo > 0 or ci_hi < 0))

    print(f"[{label}] {metric}: n={n} delta_mean={dmean:.4f} delta_std={dstd:.4f} "
          f"sem={dsem:.4f} 95%CI=[{ci_lo:.4f},{ci_hi:.4f}]"
          if ci_lo is not None else f"[{label}] {metric}: n={n} delta_mean={dmean:.4f}")
    print(f"[{label}]   ①符号一致={sign_consistent} (majority_sign={majority_sign}, "
          f"signs={signs})")
    print(f"[{label}]   ②参照アーム分散帯(std)={noise_floor} vs |delta_mean|={abs(dmean):.4f} "
          f"→ 分散帯超過={exceeds_noise}")
    print(f"[{label}]   ③paired 95%CI が0を跨がない={ci_excludes_zero}")

    verdict = "有効レバー" if (sign_consistent and exceeds_noise and ci_excludes_zero) else "頭打ち/ノイズ支配"
    print(f"[{label}]   → 判定: {verdict}")

    return {
        "label": label, "metric": metric, "n": n, "delta_mean": dmean, "delta_std": dstd,
        "sem": dsem, "ci_lo": ci_lo, "ci_hi": ci_hi,
        "sign_consistent": sign_consistent, "exceeds_noise": exceeds_noise,
        "ci_excludes_zero": ci_excludes_zero, "noise_floor": noise_floor,
        "verdict": verdict,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="E-2 T2 スイープ 3軸判定")
    ap.add_argument("--csv", required=True, nargs="+", help="grid_results.csv (複数可・結合して使う)")
    ap.add_argument("--ref-cfg", default="0.0", help="参照アームの cfg 値 (文字列比較)")
    ap.add_argument("--metrics", nargs="+", default=["recall", "worst_anatomy"])
    args = ap.parse_args(argv)

    rows = []
    for p in args.csv:
        rows.extend(load_rows(p))

    cfgs = sorted(set(r["cfg"] for r in rows), key=lambda x: float(x))
    print(f"[analyze] 読み込み行数={len(rows)} cfgs={cfgs}")

    ref_rows = [r for r in rows if r["cfg"] == args.ref_cfg]
    results = []
    for cfg in cfgs:
        if cfg == args.ref_cfg:
            continue
        other_rows = [r for r in rows if r["cfg"] == cfg]
        for metric in args.metrics:
            res = compare_arms(ref_rows, other_rows, metric, label=f"cfg={cfg}")
            results.append(res)

    return results


if __name__ == "__main__":
    main()

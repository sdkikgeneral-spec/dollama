# -*- coding: utf-8 -*-
"""dollma_e2_quality_signal.py — Phase 4 施策 F / Package E-2:
Q-2 の効果判定 (quality 直交軸を reward へ結線した効果) を **apples-to-apples** で測る。

方式 (再生成でなく再採点):
  F-0a ベースラインの 80 枚 (data/rollouts/img/000000..079.png) と、その行に保存済の
  ScorerNet axes をそのまま再利用し、**quality だけ** を新経路
  (CLIP image encoder OV → L2 → QualityMLP OV → sigmoid) で足して reward を再計算する。
  - SDXL の seed は server 内 wall-clock で非再現ゆえ、再生成すると画像分散が quality 効果に
    混入し apples-to-apples にならない。同一画像・同一 axes で quality のみ差し替えるのが、
    「quality 直交軸が信号を立てたか」を分離測定する正しい実験計画。

出力: data/rollouts/e2_quality_signal.json + 標準出力に F-0a ベースラインとの並記/ゲート判定。
無改変: rollouts.jsonl / img / IR / quality_mlp / scorer_net は読むだけ。
"""
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from dollma_reward import reward_from_scorer, QUALITY_WEIGHT  # noqa: E402
import dollma_collect_rollouts as cr  # noqa: E402

# F-0a ベースライン (docs/CLAUDE.md 記載・anatomy のみ quality=null)。
BASE_STD = 0.0377
BASE_BEST_MINUS_WORST = 0.2031
BASE_CLEAN_R = 0.007
BASE_CLUTTER_R = 0.0285

# clean/clutter 分類 (INPUT_TEXTS を単独素直題材 vs 多人数/複雑ポーズ/手数ストレスに二分)。
CLUTTER = {
    "two girls hugging in a dynamic pose",
    "a girl holding two swords in an action pose",
    "a crowd of people with many hands",
    "a girl doing a handstand, full body",
    "a boy reaching toward the camera, foreshortening",
    "a girl playing the piano with both hands",
    "a girl holding a bow and arrow at full draw",
    "two boys fighting with crossed arms",
    "a girl juggling multiple objects",
    "a girl doing the splits, hands on the floor",
}


def histogram(values, nbins=10, lo=0.0, hi=1.0):
    bins = [0] * nbins
    w = (hi - lo) / nbins
    for v in values:
        k = min(nbins - 1, max(0, int((v - lo) / w)))
        bins[k] += 1
    return [(round(lo + i * w, 2), round(lo + (i + 1) * w, 2), bins[i]) for i in range(nbins)]


def is_bimodal(values, nbins=10):
    """粗い二峰判定: ヒストで谷を挟む 2 山があるか (連続 0-bin で分離された非空クラスタ>=2)。"""
    h = [c for _, _, c in histogram(values, nbins)]
    clusters, cur = 0, False
    for c in h:
        if c > 0 and not cur:
            clusters += 1
            cur = True
        elif c == 0:
            cur = False
    return clusters >= 2, clusters


def main():
    import openvino as ov
    import open_clip

    rj = os.path.join(ROOT, "data", "rollouts", "rollouts.jsonl")
    img_dir = os.path.join(ROOT, "data", "rollouts", "img")
    rows = [json.loads(l) for l in open(rj, encoding="utf-8") if l.strip()]
    print(f"[e2] baseline rollouts {len(rows)} 行 / img_dir={img_dir}")

    # quality 経路 (CPU FP32・蒸留忠実度優先)。
    core = ov.Core()
    _cm, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14-quickgelu", pretrained="openai")
    del _cm
    clip_ov = core.compile_model(core.read_model(cr.CLIP_IMAGE_IR), "CPU")
    qmlp_ov = core.compile_model(core.read_model(cr.QUALITY_MLP_IR), "CPU")
    print("[e2] quality 経路ロード完了 (CLIP image OV → QualityMLP OV・CPU FP32)")

    base_rewards, new_rewards, quals, worsts = [], [], [], []
    clean_base, clutter_base, clean_new, clutter_new = [], [], [], []
    for i, r in enumerate(rows):
        img = os.path.join(img_dir, f"{i:06d}.png")
        axes = r["axes"]
        worst = max(axes)
        q = cr.compute_quality_clip(img, preprocess, clip_ov, qmlp_ov)
        rb = r["reward"]                                  # anatomy のみ (baseline)
        rn = reward_from_scorer(axes, q)                 # anatomy + quality (QUALITY_WEIGHT)
        base_rewards.append(rb); new_rewards.append(rn)
        quals.append(q); worsts.append(worst)
        grp_b = clutter_base if r["input_text"] in CLUTTER else clean_base
        grp_n = clutter_new if r["input_text"] in CLUTTER else clean_new
        grp_b.append(rb); grp_n.append(rn)

    def stats(xs):
        return dict(std=round(statistics.pstdev(xs), 4),
                    min=round(min(xs), 4), max=round(max(xs), 4),
                    mean=round(statistics.mean(xs), 4),
                    best_minus_worst=round(max(xs) - min(xs), 4))

    def corr(a, b):
        if len(a) < 2:
            return None
        ma, mb = statistics.mean(a), statistics.mean(b)
        num = sum((a[i] - ma) * (b[i] - mb) for i in range(len(a)))
        da = sum((x - ma) ** 2 for x in a) ** 0.5
        db = sum((x - mb) ** 2 for x in b) ** 0.5
        return round(num / (da * db), 4) if da > 0 and db > 0 else None

    # clean/clutter の |mean reward| (F-0a と同じ「|r|」= 群内平均報酬の絶対値)。
    r_clean_base = abs(statistics.mean(clean_base))
    r_clutter_base = abs(statistics.mean(clutter_base))
    r_clean_new = abs(statistics.mean(clean_new))
    r_clutter_new = abs(statistics.mean(clutter_new))

    bim, nclust = is_bimodal(quals)
    q_stats = dict(min=round(min(quals), 4), median=round(statistics.median(quals), 4),
                   max=round(max(quals), 4), mean=round(statistics.mean(quals), 4),
                   std=round(statistics.pstdev(quals), 4))

    new_std = statistics.pstdev(new_rewards)
    gate_std = new_std > 0.1
    # 分離維持/改善: clutter/clean 比が baseline(4.07x) 以上か。
    sep_base = r_clutter_base / r_clean_base if r_clean_base > 0 else None
    sep_new = r_clutter_new / r_clean_new if r_clean_new > 0 else None
    gate_sep = (sep_new is not None and sep_base is not None and sep_new >= sep_base)

    result = {
        "task": "Package E-2 quality 直交軸結線の効果判定 (apples-to-apples 再採点)",
        "method": "同一 80 画像 + 同一 ScorerNet axes・quality のみ CLIP→QualityMLP で追加",
        "n": len(rows),
        "quality_weight": QUALITY_WEIGHT,
        "reward_baseline_anatomy_only": stats(base_rewards),
        "reward_new_anatomy_plus_quality": stats(new_rewards),
        "f0a_recorded_baseline": {"std": BASE_STD, "best_minus_worst": BASE_BEST_MINUS_WORST,
                                  "clean_r": BASE_CLEAN_R, "clutter_r": BASE_CLUTTER_R},
        "quality_distribution": q_stats,
        "quality_histogram": histogram(quals),
        "quality_bimodal": {"bimodal": bim, "clusters": nclust},
        "orthogonality": {
            "corr_quality_vs_worst_anatomy": corr(quals, worsts),
            "note": "低相関ほど quality が anatomy に直交 (新情報)。",
        },
        "clean_vs_clutter": {
            "baseline_clean_absmean": round(r_clean_base, 4),
            "baseline_clutter_absmean": round(r_clutter_base, 4),
            "baseline_separation_x": round(sep_base, 2) if sep_base else None,
            "new_clean_absmean": round(r_clean_new, 4),
            "new_clutter_absmean": round(r_clutter_new, 4),
            "new_separation_x": round(sep_new, 2) if sep_new else None,
        },
        "gate": {
            "reward_std_gt_0.1": gate_std,
            "new_reward_std": round(new_std, 4),
            "separation_maintained_or_improved": gate_sep,
            "verdict": ("信号が立った (ゲート通過)" if (gate_std or gate_sep)
                        else "信号なお弱い (ゲート未達)"),
        },
    }
    out = os.path.join(ROOT, "data", "rollouts", "e2_quality_signal.json")
    json.dump(result, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print("\n=== E-2 判定 (F-0a ベースライン → 新 quality 経路) ===")
    print(f"reward std       : {BASE_STD:.4f} (anatomy) → {new_std:.4f} (anatomy+quality)  gate>0.1: {gate_std}")
    print(f"best-worst       : {BASE_BEST_MINUS_WORST:.4f} → {stats(new_rewards)['best_minus_worst']:.4f}")
    print(f"quality[0,1]     : {q_stats}  bimodal={bim}({nclust} clusters)")
    print(f"corr(q, anatomy) : {corr(quals, worsts)}  (低いほど直交=新情報)")
    print(f"clean/clutter |r|: base {r_clean_base:.4f}/{r_clutter_base:.4f} ({sep_base:.2f}x) "
          f"→ new {r_clean_new:.4f}/{r_clutter_new:.4f} ({sep_new:.2f}x)")
    print(f"\n判定: {result['gate']['verdict']}  (std_gate={gate_std} / sep_gate={gate_sep})")
    print(f"レポート: {out}")


if __name__ == "__main__":
    main()

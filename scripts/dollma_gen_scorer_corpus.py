# -*- coding: utf-8 -*-
"""dollama Phase 4 Model B / B-3a — 画像コーパス生成 + WD14 タグ付け driver (研究機専用・未実走)。

**このスクリプトは研究機 (RTX5080 + OpenVINO, 実 SDXL 重み `DOLLAMA_UNET_WEIGHTS` 配置済) でのみ走る。**
非研究機 (この開発機: i7-10700 / GTX1080Ti / OV C++ 無) では `--dry-run` で配管だけ検証し、
**実生成・実推論はしない**。実走には `--run` を明示的に渡す (既定は dry-run で安全)。

役割 (2 段):
  1. SDXL 生成: 既存 txt2img パイプライン (`dollama --prompt` / Txt2ImgGenerator) で画像コーパスを作る。
     良/悪サンプルが偏らないよう、品質ネガティブ注入 ON/OFF や崩れやすいプロンプトを混ぜて
     品質・解剖の分散を意図的に作る (まず数百枚＝配管優先・本数は後段拡大)。
  2. WD14 タグ付け: 生成画像を WD14 SwinV2 (OV) でタグ付けし sigmoid スコアベクタ [N_TAGS] を得る。
     その出力を `scorer_wd14.jsonl` (1 行 = {image, wd14_scores, prompt, rating}) に書く。

  → 出力 `scorer_wd14.jsonl` を `dollma_make_scorer_labels.py --wd14` に食わせると、
     OV/SDXL 非依存パートで 8 軸 soft target + 品質スカラ (provider) に変換される。
     生成と推論はここ (研究機)、ラベル化は非研究機でも可、という分業。

外部美的モデル (品質スカラ B 系統) は **ライセンス確認後に** ここへ差す (現状は WD14 のみ)。

注意 (CLAUDE.md):
  - SDXL は RTX5080 (3.80s/枚)・WD14 は CPU が最速 (105ms) → device 既定はそれに従う。
  - WD14 OV 入力は [1,448,448,3] f32 NHWC 0-255 (wd14.hpp の前処理規約)。
  - NPU は WD14 に不向き (Window Attention・268ms) → CPU を使う。
"""

import argparse
import json
import os
import sys


# ------------------------------------------------------------
# 崩れやすさを散らすプロンプト設計 (品質・解剖の分散を意図的に作る)
# 良サンプル: 素直な 1girl・品質ネガ ON。悪サンプル: 多人数/複雑ポーズ・品質ネガ OFF。
# ------------------------------------------------------------
GOOD_PROMPTS = [
    "1girl, solo, long hair, simple background, upper body",
    "1girl, solo, short hair, standing, white background",
    "1boy, solo, looking at viewer, portrait, simple background",
]
HARD_PROMPTS = [
    "2girls, complex pose, dynamic angle, intertwined arms",
    "1girl, holding two swords, spread fingers, action pose, foreshortening",
    "multiple girls, group, many hands, crowd",
]


def build_jobs(n, seed):
    """生成ジョブ (prompt, inject_negatives) のリストを seed 決定的に作る。

    良/悪を交互に混ぜて品質・解剖の分散を作る。実生成はしない (リストだけ)。
    """
    import random
    rng = random.Random(seed)
    jobs = []
    for i in range(n):
        good = (i % 2 == 0)
        pool = GOOD_PROMPTS if good else HARD_PROMPTS
        jobs.append({
            "prompt": rng.choice(pool),
            "inject_quality_negatives": good,   # 良サンプルは品質ネガ ON
            "rating": "g",
        })
    return jobs


# ------------------------------------------------------------
# 研究機実走部 (--run でのみ呼ばれる)。本開発機では import すら走らせない。
# ------------------------------------------------------------
def run_on_research_machine(args, jobs):
    """研究機でのみ実行: SDXL 生成 → WD14 タグ付け → scorer_wd14.jsonl 出力。

    !!! この関数は研究機 (RTX5080 + OV) 前提。非研究機では呼ばない (--run 必須)。!!!
    実走時に下記を結線する (現状は配管の骨組み・実 import はガード内):
    """
    raise NotImplementedError(
        "研究機実走部は未結線 (配管のみ)。実走時に下記を埋める:\n"
        "  1. SDXL 生成: Txt2ImgGenerator / `dollama --prompt` を job ごとに呼び画像を保存。\n"
        "     品質ネガ注入は job['inject_quality_negatives'] に従う。\n"
        "  2. WD14 タグ付け: from infer import Wd14Tagger 相当を OV でロードし\n"
        "     [1,448,448,3] f32 NHWC 0-255 で infer → sigmoid スコア [N_TAGS]。\n"
        "  3. 各画像行 {image, wd14_scores, prompt, rating} を out_jsonl に追記。\n"
        "  4. 美的モデル (品質 B 系統 = deepghs/anime_aesthetic・openrail・ONNX swinv2pv3 448):\n"
        "     画像 [1,448,448,3] f32 NHWC 0-255 で ONNX 推論 → 7 段ロジット → softmax →\n"
        "     7 クラス確率 [masterpiece..worst] を 'aesthetic_probs' として行に足す\n"
        "     (連続 [0,1] 写像はラベル化側 AnimeAestheticDeepghsProvider.expected_score が担当)。\n"
        "  (代替候補 waifu_scorer_v4=apache-2.0 を使う場合は 'clip_image_embed'[768] を足す。)"
    )


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="SDXL 画像コーパス生成 + WD14 タグ付け driver (研究機専用・既定 dry-run)")
    ap.add_argument("--n", type=int, default=8, help="生成枚数 (まず数百枚規模・既定は小さく)")
    ap.add_argument("--seed", type=int, default=20260620)
    ap.add_argument("--out", default="data/scorer/scorer_wd14.jsonl",
                    help="WD14 スコアベクタ出力 jsonl (make_scorer_labels の入力)")
    ap.add_argument("--dry-run", dest="dry_run", action="store_true", default=True,
                    help="(既定) 生成計画だけ出力し実走しない")
    ap.add_argument("--run", dest="dry_run", action="store_false",
                    help="研究機で実走する (RTX5080+OV 必須・実 SDXL 重み配置前提)")
    args = ap.parse_args(argv)

    jobs = build_jobs(args.n, args.seed)

    if args.dry_run:
        # 配管検証: ジョブ計画と良/悪バランスを表示し、実走はしない。
        n_good = sum(1 for j in jobs if j["inject_quality_negatives"])
        print(f"[DRY-RUN] 生成計画 {len(jobs)} 枚 (良 {n_good} / 悪 {len(jobs) - n_good})")
        print(f"[DRY-RUN] 出力予定: {args.out}")
        print("[DRY-RUN] 実走は研究機で --run を付けて呼ぶこと (RTX5080+OV 必須)。")
        for j in jobs[:3]:
            print("  例:", json.dumps(j, ensure_ascii=False))
        return

    # --run: 研究機実走 (非研究機では NotImplementedError で止まる = 安全)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    run_on_research_machine(args, jobs)


if __name__ == "__main__":
    main()

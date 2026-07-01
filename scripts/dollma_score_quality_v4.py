# -*- coding: utf-8 -*-
"""dollama Phase 4 Model B / Q-1 — waifu_scorer_v4 (apache-2.0) 実採点ランナー (研究機実走部)。

`dollma_make_scorer_labels.py` の `WaifuScorerV4Provider._forward_mlp` がコメントで指していた
「研究機実走部」をここに結線する。本機 (RTX5080 + torch cu128 + open_clip) で:

  1. CLIP ViT-L/14 image encoder で E-1 コーパス PNG → image embed[768] を採る。
     - 実装は open_clip `ViT-L-14-quickgelu` pretrained=openai。waifu-scorer は元々
       OpenAI `clip.load("ViT-L/14")` (= QuickGELU) を使うため、これに厳密一致させる。
     - transformers 経由は不可: CLIPModel import が sklearn を引き、研究機の SAC が
       sklearn DLL をブロックする (=新規/未署名 DLL ロード不可)。open_clip は純 torch で回避。
  2. L2 正規化 (aesthetic-predictor の `normalized` 相当・waifu-scorer 忠実)。
  3. waifu-scorer-v4-beta MLP (768→2048→512→256→128→32→1・BatchNorm 入り) で生スコア。
     - 構造は model.safetensors header で確定 (strict ロード一致)。BN は eval=running stats で batch=1 安全。
  4. 生スコア → `WaifuScorerV4Provider.normalize_score` で /10 クランプ [0,1] (正規版写像を再利用)。

入力: data/scorer/scorer.{train,val}.jsonl (axis 済み・quality=null)。
出力: 同ファイルの quality を埋め直す (元は .qnull.bak へ退避)。各行に生列
      quality_waifu / raw_waifu も残す (将来 deepghs アンサンブル再判断用)。
レポート: data/scorer/scorer_quality_report.json + 標準出力に分布/退化判定。

ライセンス境界: 使うのは apache-2.0 の Eugeoter/waifu-scorer-v4-beta のみ。
deepghs/anime_aesthetic (openrail) はユーザー本人の直接承認が取れるまで実行しない
(本ランナーには結線しない)。スキーマは quality_waifu を残すのでアンサンブルは後付け可能。
"""

import argparse
import json
import math
import os
import shutil
import statistics
import sys
import time

# 正規版の /10 クランプ写像を再利用 (provider 本体は無改変)。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dollma_make_scorer_labels import WaifuScorerV4Provider


def build_mlp():
    """waifu-scorer-v4-beta MLP (model.safetensors header で確定した構造)。"""
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(768, 2048), nn.ReLU(), nn.BatchNorm1d(2048), nn.Dropout(0.3),
        nn.Linear(2048, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(0.3),
        nn.Linear(512, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.2),
        nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.1),
        nn.Linear(128, 32), nn.ReLU(),
        nn.Linear(32, 1),
    )


def load_mlp(weights_path, device):
    import torch
    from safetensors.torch import load_file
    sd = load_file(weights_path)
    mlp = build_mlp()
    # state_dict キー "layers.N.xxx" → Sequential "N.xxx" に剥がして strict ロード。
    mlp.load_state_dict({k[len("layers."):]: v for k, v in sd.items()}, strict=True)
    mlp.eval().to(device)
    return mlp


def load_clip(device):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14-quickgelu", pretrained="openai")
    model.eval().to(device)
    return model, preprocess


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_negatives(wd14_path):
    """scorer_wd14.jsonl の inject_quality_negatives を image→bool で引く (good/bad 分離検証用)。"""
    neg = {}
    if not os.path.exists(wd14_path):
        return neg
    for line in open(wd14_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        neg[r["image"]] = bool(r.get("inject_quality_negatives", False))
    return neg


def histogram(values, nbins=10, lo=0.0, hi=1.0):
    bins = [0] * nbins
    width = (hi - lo) / nbins
    for v in values:
        k = int((v - lo) / width)
        if k >= nbins:
            k = nbins - 1
        if k < 0:
            k = 0
        bins[k] += 1
    return [(round(lo + i * width, 3), round(lo + (i + 1) * width, 3), bins[i]) for i in range(nbins)]


def main(argv=None):
    ap = argparse.ArgumentParser(description="waifu_scorer_v4 (apache-2.0) 実採点で quality を埋める")
    ap.add_argument("--data-dir", default="data/scorer")
    ap.add_argument("--weights",
                    default="data/scorer/weights/waifu_scorer_v4/model.safetensors",
                    help="waifu-scorer-v4-beta MLP safetensors")
    ap.add_argument("--device", default=None, help="cuda / cpu (既定: 利用可なら cuda)")
    args = ap.parse_args(argv)

    import torch
    from PIL import Image
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} weights={args.weights}")

    mlp = load_mlp(args.weights, device)
    clip_model, preprocess = load_clip(device)
    provider = WaifuScorerV4Provider(weights_path=args.weights)  # normalize_score 写像の出所
    neg = load_negatives(os.path.join(args.data_dir, "scorer_wd14.jsonl"))

    # image → (raw, quality) を 1 度だけ計算してキャッシュ (train/val で重複し得る image に備える)。
    cache = {}

    def score_image(path):
        if path in cache:
            return cache[path]
        img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = clip_model.encode_image(img)
            feat = feat / feat.norm(dim=-1, keepdim=True)  # L2 正規化 (waifu-scorer 忠実)
            raw = float(mlp(feat.float()).item())
        q = provider.normalize_score(raw)  # 生スコア 0..10 → /10 クランプ [0,1]
        cache[path] = (raw, q)
        return cache[path]

    all_raw, all_q = [], []
    good_q, bad_q = [], []
    t0 = time.time()
    for split in ("train", "val"):
        path = os.path.join(args.data_dir, f"scorer.{split}.jsonl")
        rows = read_jsonl(path)
        for r in rows:
            raw, q = score_image(r["image"])
            r["quality"] = q
            r["quality_waifu"] = q       # 生列 (将来 deepghs アンサンブル再判断用)
            r["raw_waifu"] = round(raw, 6)
            all_raw.append(raw)
            all_q.append(q)
            (bad_q if neg.get(r["image"]) else good_q).append(q)
        # 退避してから上書き
        bak = path.replace(".jsonl", ".qnull.bak.jsonl")
        if not os.path.exists(bak):
            shutil.copy2(path, bak)
        write_jsonl(path, rows)
        print(f"{split}: {len(rows)} 行に quality を充填 (退避 {os.path.basename(bak)})")

    elapsed = time.time() - t0
    n = len(all_q)
    n_clamp0 = sum(1 for q in all_q if q <= 0.0)
    std_q = statistics.pstdev(all_q)
    report = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provider": "waifu_scorer_v4",
        "license": "apache-2.0",
        "repo": "Eugeoter/waifu-scorer-v4-beta",
        "clip_backbone": "open_clip ViT-L-14-quickgelu pretrained=openai (L2-normalized embed)",
        "normalize": "raw/10 clamp [0,1] (WaifuScorerV4Provider.normalize_score)",
        "n_images_unique": len(cache),
        "n_rows_scored": n,
        "elapsed_sec": round(elapsed, 2),
        "raw": {
            "min": round(min(all_raw), 4), "median": round(statistics.median(all_raw), 4),
            "max": round(max(all_raw), 4), "mean": round(statistics.mean(all_raw), 4),
            "std": round(statistics.pstdev(all_raw), 4),
        },
        "quality_norm": {
            "min": round(min(all_q), 4), "median": round(statistics.median(all_q), 4),
            "max": round(max(all_q), 4), "mean": round(statistics.mean(all_q), 4),
            "std": round(std_q, 4),
            "n_clamped_to_0": n_clamp0,
        },
        "histogram_quality": histogram(all_q, nbins=10, lo=0.0, hi=1.0),
        "good_bad_sanity": {
            "good_n": len(good_q), "bad_n": len(bad_q),
            "good_mean": round(statistics.mean(good_q), 4) if good_q else None,
            "bad_mean": round(statistics.mean(bad_q), 4) if bad_q else None,
            "separation_good_minus_bad": (round(statistics.mean(good_q) - statistics.mean(bad_q), 4)
                                          if good_q and bad_q else None),
        },
        "degenerate": std_q < 1e-4,
        "note_license": ("deepghs/anime_aesthetic (openrail) はユーザー本人の直接承認待ちで未実行。"
                         "quality_waifu 列を残すのでアンサンブルは後付け可能。"),
    }
    out = os.path.join(args.data_dir, "scorer_quality_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n=== quality 分布レポート ===")
    print(json.dumps(report["quality_norm"], ensure_ascii=False))
    print("raw:", json.dumps(report["raw"], ensure_ascii=False))
    print("good/bad:", json.dumps(report["good_bad_sanity"], ensure_ascii=False))
    print("histogram (lo,hi,count):")
    for lo, hi, c in report["histogram_quality"]:
        print(f"  [{lo:.2f},{hi:.2f}) {'#'*c} {c}")
    verdict = ("退化 (variance ほぼ無) → Q-2 中止推奨" if report["degenerate"]
               else f"非退化 (std={std_q:.4f}) → Q-2 続行可")
    print(f"\n判定: {verdict}")
    print(f"レポート: {out}")


if __name__ == "__main__":
    main()

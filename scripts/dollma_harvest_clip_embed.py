# -*- coding: utf-8 -*-
"""dollama Phase 4 Model B / Package B — CLIP image embed[768] harvest。

data/scorer/scorer.{train,val}.jsonl 各行の image (data/scorer/img/*.png) を
CLIP ViT-L/14 image encoder で embed[768] にして各行に clip_image_embed:[768] を載せる。
これが Package C (自作 quality MLP を CLIP 空間で waifu 蒸留) の入力になる。

CLIP backbone は dollma_score_quality_v4.py の load_clip/score_image と **厳密一致**:
  - open_clip "ViT-L-14-quickgelu" pretrained=openai (waifu-scorer の OpenAI clip.load 相当)。
  - preprocess は create_model_and_transforms が返す val 変換をそのまま使う。
  - encode_image 後に L2 正規化 (aesthetic-predictor `normalized` 相当・waifu 蒸留忠実)。
  → 蒸留の教師 (waifu raw) と生徒 (自作 MLP) で embed 空間を完全一致させるため必須。

入力: data/scorer/scorer.{train,val}.jsonl (image/quality/axis/meta/quality_waifu/raw_waifu 済み)。
出力: 同ファイルの各行に clip_image_embed:[768] を追記して書き戻す (既存列は無改変)。
退避: 上書き前に scorer.{split}.noembed.bak.jsonl へコピー (冪等・既存 bak は上書きしない)。
provenance: data/scorer/scorer_clip_embed.json に backbone/pretrained/normalize/embed_dim/n_images。
"""

import argparse
import json
import os
import shutil
import sys
import time

# Windows cp932 コンソールでも ≈ 等を出力できるよう stdout を utf-8 に。
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

EMBED_DIM = 768  # ViT-L/14 image embed 次元


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


def load_clip(device):
    """dollma_score_quality_v4.load_clip と厳密一致 (同 backbone・同 preprocess)。"""
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14-quickgelu", pretrained="openai")
    model.eval().to(device)
    return model, preprocess


def main(argv=None):
    ap = argparse.ArgumentParser(description="CLIP ViT-L/14 image embed[768] を各行に harvest")
    ap.add_argument("--data-dir", default="data/scorer")
    ap.add_argument("--device", default=None, help="cuda / cpu (既定: 利用可なら cuda)")
    args = ap.parse_args(argv)

    import torch
    from PIL import Image
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} backbone=ViT-L-14-quickgelu pretrained=openai")

    clip_model, preprocess = load_clip(device)

    # image → embed[768] を 1 度だけ計算してキャッシュ (train/val で重複し得る image に備える)。
    # dollma_score_quality_v4.score_image と同じ経路 (encode_image → L2 正規化)。
    cache = {}

    def embed_image(path):
        if path in cache:
            return cache[path]
        img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = clip_model.encode_image(img)
            feat = feat / feat.norm(dim=-1, keepdim=True)  # L2 正規化 (waifu-scorer 忠実)
        vec = feat.squeeze(0).float().cpu().tolist()
        assert len(vec) == EMBED_DIM, f"embed dim {len(vec)} != {EMBED_DIM}"
        cache[path] = vec
        return vec

    n_rows = 0
    norms = []
    t0 = time.time()
    for split in ("train", "val"):
        path = os.path.join(args.data_dir, f"scorer.{split}.jsonl")
        rows = read_jsonl(path)
        for r in rows:
            vec = embed_image(r["image"])
            r["clip_image_embed"] = vec  # 既存列は無改変・追記のみ
            norms.append(sum(v * v for v in vec) ** 0.5)
            n_rows += 1
        # 退避してから上書き (冪等: 既存 bak は上書きしない)。
        bak = path.replace(".jsonl", ".noembed.bak.jsonl")
        if not os.path.exists(bak):
            shutil.copy2(path, bak)
            print(f"{split}: 退避 {os.path.basename(bak)}")
        else:
            print(f"{split}: 退避 skip (既存 {os.path.basename(bak)})")
        write_jsonl(path, rows)
        print(f"{split}: {len(rows)} 行に clip_image_embed[{EMBED_DIM}] を追記")

    elapsed = time.time() - t0
    norm_min = min(norms)
    norm_max = max(norms)
    norm_mean = sum(norms) / len(norms)
    report = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "step": "Package B CLIP image embed harvest",
        "backbone": "ViT-L-14-quickgelu",
        "pretrained": "openai",
        "normalize": "L2 (encode_image 後・aesthetic-predictor normalized 相当)",
        "embed_dim": EMBED_DIM,
        "n_rows": n_rows,
        "n_images_unique": len(cache),
        "elapsed_sec": round(elapsed, 2),
        "l2_norm_check": {"min": round(norm_min, 6), "max": round(norm_max, 6),
                          "mean": round(norm_mean, 6)},
        "column_added": "clip_image_embed:[768] float (既存列は無改変)",
        "backup": "scorer.{train,val}.noembed.bak.jsonl",
        "note": "backbone は dollma_score_quality_v4.load_clip/score_image と厳密一致 (waifu 蒸留の教師/生徒で embed 空間統一)。",
    }
    out = os.path.join(args.data_dir, "scorer_clip_embed.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n=== CLIP embed harvest 結果 ===")
    print(f"embed 載せた行数: {n_rows} (unique image {len(cache)})")
    print(f"embed_dim: {EMBED_DIM}")
    print(f"L2 norm: min {norm_min:.6f} / max {norm_max:.6f} / mean {norm_mean:.6f} (≈1 期待)")
    print(f"レポート: {out}")


if __name__ == "__main__":
    main()

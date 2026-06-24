#!/usr/bin/env python3
# 生成画像 → danbooru タグ ラベル化 (SDXL生成→ラベル化 ループの後段)。
#
# src/infer/wd14.hpp の Wd14Tagger と同じ OV IR
# (models/wd14-swinv2-tagger-v3/model_ov.xml) を同じ採用デバイス (CPU) で走らせ、
# 生成 PNG を WD14 SwinV2 Tagger v3 にかけて danbooru タグを抽出する。
#
# 前処理は SmilingWolf 正準 (WD14 系の標準):
#   RGBA は白背景に合成 → 正方形に白パディング → 448 にリサイズ (cv2 AREA 相当) →
#   RGB→BGR → float32 (0-255, 正規化なし)。
# ※ C++ Wd14Tagger::infer はリサイズ済み [448,448,3] float (NHWC, 0-255) を受け取る設計で、
#   リサイズ/チャネル順は上流の責務。ここでは正準前処理を Python 側で実装する。
#
# 使い方:
#   python scripts/dollma_label_image.py outputs/run1.png [--thresh 0.35] [--device CPU]

import argparse
import csv
import os

import numpy as np
import openvino as ov
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
MODEL_XML = os.path.join(ROOT, "models", "wd14-swinv2-tagger-v3", "model_ov.xml")
TAGS_CSV = os.path.join(ROOT, "models", "wd14-swinv2-tagger-v3", "selected_tags.csv")
INPUT_SIZE = 448


def load_tags(csv_path):
    # selected_tags.csv: tag_id,name,category,count
    # category 9 = rating (general/sensitive/questionable/explicit), それ以外 = 通常タグ。
    names, categories = [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            names.append(row["name"])
            categories.append(int(row["category"]))
    return names, categories


def preprocess(path):
    # RGBA (マッティング透過) は白背景に合成 → 正方形に白パディング → 448 リサイズ → BGR float32。
    img = Image.open(path)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img.convert("RGBA")).convert("RGB")
    else:
        img = img.convert("RGB")

    w, h = img.size
    side = max(w, h)
    square = Image.new("RGB", (side, side), (255, 255, 255))
    square.paste(img, ((side - w) // 2, (side - h) // 2))
    square = square.resize((INPUT_SIZE, INPUT_SIZE), Image.BICUBIC)

    arr = np.asarray(square, dtype=np.float32)  # [448,448,3] RGB 0-255
    arr = arr[:, :, ::-1]                        # RGB → BGR (SmilingWolf 規約)
    return arr[np.newaxis, ...].copy()           # [1,448,448,3] NHWC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--thresh", type=float, default=0.35)
    ap.add_argument("--device", default="CPU")  # WD14 は CPU が最速 (probe8)
    ap.add_argument("--topk", type=int, default=40)
    args = ap.parse_args()

    names, categories = load_tags(TAGS_CSV)
    core = ov.Core()
    model = core.read_model(MODEL_XML)
    compiled = core.compile_model(model, args.device)

    x = preprocess(args.image)
    out = compiled(x)[compiled.output(0)][0]  # [N_TAGS] sigmoid 済みスコア

    assert len(out) == len(names), f"tag 数不一致: out {len(out)} vs csv {len(names)}"

    # rating (category 9) と通常タグを分離。
    ratings = [(names[i], float(out[i])) for i in range(len(out)) if categories[i] == 9]
    general = [(names[i], float(out[i])) for i in range(len(out)) if categories[i] != 9]
    general.sort(key=lambda t: t[1], reverse=True)
    ratings.sort(key=lambda t: t[1], reverse=True)

    hits = [(n, s) for n, s in general if s >= args.thresh]

    print(f"[label] image={args.image} device={args.device} thresh={args.thresh}")
    print(f"[label] rating: " + ", ".join(f"{n}={s:.2f}" for n, s in ratings[:4]))
    print(f"[label] {len(hits)} tags >= {args.thresh}:")
    for n, s in hits[: args.topk]:
        print(f"  {s:.3f}  {n}")

    # danbooru タグ列 (LLM フィードバックループ入力形式)
    print("\n[label] tag line:")
    print(", ".join(n.replace("_", " ") for n, _ in hits[: args.topk]))


if __name__ == "__main__":
    main()

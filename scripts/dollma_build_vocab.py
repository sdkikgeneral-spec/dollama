#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dollama BitNet タグ語彙表 (vocab.json) 生成スクリプト (Phase 4 #1 / 1b)

WD14 `selected_tags.csv` (タグメタのみ・画像なし) を土台に、
区切り正規化・低頻度足切り・負語分離・id 割当を行い `data/bitnet/vocab.json` を吐く。
スキーマは docs/dataset-spec.md §3.2 / §7 に従う。

使い方:
    python scripts/dollma_build_vocab.py --min-count 2518 --out data/bitnet/vocab.json
"""
import argparse
import csv
import io
import json
import os
import re
import ssl
import sys
import urllib.request
from datetime import datetime, timezone

try:
    import certifi
    _CAFILE = certifi.where()
except Exception:  # certifi 無しでも続行 (既定 CA)
    _CAFILE = None

# WD14 v3 のタグ csv (タグ名・カテゴリ・頻度のみ。画像は含まない)
WD14_CSV_URL = (
    "https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3/"
    "resolve/main/selected_tags.csv"
)

# 特殊トークン (id 0..4 を固定予約: docs/dataset-spec.md §3.2)
SPECIALS = ["<pad>", "<bos>", "<eos>", "<sep>", "<unk>"]

# WD14 category コード。0=general, 4=character, 9=rating。
# color_mode 制御タグは本データセット拡張で category=100 に再分類する (§7-b)。
CAT_COLOR_MODE = 100
COLOR_MODE_TAGS = {"monochrome", "greyscale", "lineart", "sketch"}

# (a) positive 語彙から除外する負語 (default_quality_negatives + 指本数仮定語)。
# character.hpp default_quality_negatives() と spec §10 に厳密対応。
NEGATIVE_BLOCKLIST = {
    # default_quality_negatives()
    "bad hands", "malformed hands", "mutated hands",
    "fused fingers", "bad anatomy", "deformed",
    "lowres", "worst quality", "jpeg artifacts",
    # 指本数を仮定する語 (本数非依存方針: spec §10 で positive にも入れない)
    "extra fingers", "extra digits", "fewer digits",
    "missing fingers", "extra arms", "extra legs",
    "missing limb", "missing arm", "missing leg",
    "poorly drawn hands", "poorly drawn face",
}


def normalize_separator(name: str) -> str:
    """区切り正規化 (docs/dataset-spec.md §6)。

    英数字に挟まれた `_` のみ半角スペースへ置換する。顔文字タグ (^_^, >_< 等) の
    記号に隣接する `_` は保持する。
    """
    # 「英数字_英数字」の `_` だけをスペースに (記号隣接の `_` は触らない)
    return re.sub(r"(?<=[0-9A-Za-z])_(?=[0-9A-Za-z])", " ", name)


def fetch_wd14_csv(local_path: str | None) -> str:
    """WD14 selected_tags.csv の中身 (テキスト) を取得する。

    local_path があればそれを読む。無ければ HF から DL してキャッシュする。
    """
    if local_path and os.path.exists(local_path):
        with open(local_path, "r", encoding="utf-8") as f:
            return f.read()

    ctx = ssl.create_default_context(cafile=_CAFILE)
    req = urllib.request.Request(
        WD14_CSV_URL, headers={"User-Agent": "dollama-dataset/0.1"}
    )
    print(f"[vocab] WD14 csv を DL: {WD14_CSV_URL}", file=sys.stderr)
    with urllib.request.urlopen(req, timeout=60, context=ctx) as r:
        text = r.read().decode("utf-8")

    if local_path:
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        with open(local_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[vocab] csv をキャッシュ: {local_path}", file=sys.stderr)
    return text


def build_vocab(csv_text: str, min_count: int) -> dict:
    """csv テキストから vocab.json 相当の dict を構築する。"""
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    raw_total = len(rows)

    # rating カテゴリ (9) は target 制御に使わないので語彙から除外
    # (NSFW 分離は pair の meta.rating で行う: §3.1)。
    entries = []  # (norm_name, category, count)
    seen = set()
    dropped_neg = 0
    for r in rows:
        cat = int(r["category"])
        if cat == 9:  # rating タグは語彙に入れない
            continue
        count = int(r["count"])
        if count < min_count:
            continue
        name = normalize_separator(r["name"])
        # (a) 負語分離: positive 語彙から除外
        if name in NEGATIVE_BLOCKLIST:
            dropped_neg += 1
            continue
        if name in seen:
            continue
        seen.add(name)
        # (b) color_mode 制御タグは category=100 に再分類
        if name in COLOR_MODE_TAGS:
            cat = CAT_COLOR_MODE
        entries.append((name, cat, count))

    # 頻度降順で安定ソート → id を specials の続き (5) から連番割当
    entries.sort(key=lambda e: (-e[2], e[0]))
    base = len(SPECIALS)
    tags = [
        {"id": base + i, "tag": name, "category": cat, "freq": count}
        for i, (name, cat, count) in enumerate(entries)
    ]

    return {
        "version": 1,
        "separator": "space",
        "specials": SPECIALS,
        "tags": tags,
        "_build": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "min_count": min_count,
            "raw_tags": raw_total,
            "kept_tags": len(tags),
            "dropped_negatives": dropped_neg,
            "source": "SmilingWolf/wd-swinv2-tagger-v3/selected_tags.csv",
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="dollama BitNet vocab.json 生成")
    ap.add_argument("--min-count", type=int, default=2518,
                    help="低頻度足切り閾値 (danbooru count。既定 2518 ≒ 上位5000語)")
    ap.add_argument("--out", default="data/bitnet/vocab.json")
    ap.add_argument("--wd14-csv", default="data/bitnet/cache/selected_tags.csv",
                    help="WD14 csv のローカルパス (無ければ DL してここにキャッシュ)")
    args = ap.parse_args()

    csv_text = fetch_wd14_csv(args.wd14_csv)
    vocab = build_vocab(csv_text, args.min_count)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=1)

    b = vocab["_build"]
    cat_dist: dict[int, int] = {}
    for t in vocab["tags"]:
        cat_dist[t["category"]] = cat_dist.get(t["category"], 0) + 1
    print(f"[vocab] 出力: {args.out}", file=sys.stderr)
    print(f"[vocab] raw={b['raw_tags']} → kept={b['kept_tags']} "
          f"(min_count={b['min_count']}, 負語除外={b['dropped_negatives']})",
          file=sys.stderr)
    print(f"[vocab] category 分布 (0=general,4=character,100=color_mode): "
          f"{dict(sorted(cat_dist.items()))}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# -*- coding: utf-8 -*-
"""dollma_eval_image_grid.py — E-1: 画質評価ハーネス (研究機専用)。

目的:
  2-6d の preset 比較は seed 1234 の N=1・12 枚のみで比較の物差しになっていない
  (measurements-log 自身が「N=1 で優劣の根拠にはならない」と明記)。本ハーネスは
  条件 (preset / prompt 題材 / steps) × seed 複数本 の直積を CLI 生成モードで回し、
  WD14 再現率 + ScorerNet anatomy 8 軸を集計して平均 ± std で報告する。

  **Phase 5-1 (崩壊境界 probe) の先行実装を兼ねる**
  (roadmap: 固定キャラ × 画角/パースタグの直積 → QualityGate 採点 → ヒートマップ)。
  題材 (PROMPTS) は「画角/パースタグ」に差し替え可能な形で切り出してあり、
  build_grid() / run_condition() はプロンプト内容に非依存 (汎用)。

閉路:
  1. (preset, prompt_id, seed) の直積を CLI 生成モード (`dollama.exe --prompt ...`) で
     1 枚ずつ別プロセス起動 (2-6d と同じ = 全枚 cold 相当。秒は characterization のみ)。
  2. 生成ログを grep して `[warn]` / stub / フォールバック / 構築に失敗 が無いことを確認
     (無効画像は集計から除外し、CSV に log_ok=False で残す)。
  3. WD14 (scripts/dollma_label_image.py と同一 OV IR・同一前処理・同一既定閾値) で
     タグ検出 → プロンプト語彙に対する再現率を算出 (2-6d と同一算出法)。
  4. ScorerNet (dollma_collect_rollouts.py と同一前処理・同一 axes_from_logits) で
     anatomy 8 軸を採点。
  5. 条件 (preset, prompt_id) ごとに CSV 行 (生データ) + 集計行 (平均 ± std) + コンタクト
     シート PNG を出力する。

**実走は研究機 (RTX5080 + OpenVINO + 実 SDXL 重み) でのみ。** 開発機では [SKIP]。

使用例 (研究機):
  set DOLLAMA_OV_TOKENIZERS_DLL=<path>/openvino_tokenizers.dll
  py -3.14 scripts/dollma_eval_image_grid.py --run --presets illustrious-xl \
      --prompts p1 p2 p3 p4 --seeds 4 --out-dir docs/logs/e1

規律 (E-1 発注文 / CLAUDE.md 由来):
  - N=1 で優劣を書かない (集計は必ず seed>=4 本の平均 ± std)。
  - `[warn]` / stub / フォールバック / 「構築に失敗」を grep で 0 件確認してから採用する
    (大文字小文字を区別する・"missing fingers" 等の negative 語との誤爆を避けるため)。
  - 走行中に models/presets/ 等の入力データを動かさない・動かさせない
    (走行前後で sha256 を控えて突合する)。
  - 秒は preset 比較の指標にしない (別プロセス起動で全枚 cold 相当)。
"""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

DOLLAMA_EXE = os.path.join(ROOT, "build", "src", "dollama.exe")
WD14_XML = os.path.join(ROOT, "models", "wd14-swinv2-tagger-v3", "model_ov.xml")
WD14_TAGS_CSV = os.path.join(ROOT, "models", "wd14-swinv2-tagger-v3", "selected_tags.csv")
SCORER_IR = os.path.join(ROOT, "models", "scorer-net", "model_ov_fp32.xml")
WD14_INPUT_SIZE = 448

# ScorerNet AnomalyAxis 順 (src/infer/quality_gate.hpp と厳密一致・index0=quality は使わない)。
AXIS_NAMES = ["Hands", "Limbs", "Head", "Eyes", "Ears", "Mouth", "Digits", "GlobalAnatomy"]

# CLI 生成モードのログに出る失敗マーカー (大文字小文字区別)。
# "MISSING" を無視大小で引くと negative プロンプト内の "missing fingers" に当たる
# (2-6e の既知の罠) ため、必ず case-sensitive grep で判定する。
BAD_MARKERS = ["[warn]", "stub", "フォールバック", "構築に失敗"]

# ============================================================
# 題材プロンプト (p1-p3 は docs/logs/2-6d と同一文字列 = 物差しの校正用。
# p4 は 2-6d に無い複数人・新設)。
# ============================================================
NEGATIVE_COMMON = (
    "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, "
    "fewer digits, cropped, worst quality, low quality, multiple views, multiple girls"
)
# p4 のみ "multiple girls" が題材そのものと矛盾するため negative から除く。
NEGATIVE_P4 = (
    "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, "
    "fewer digits, cropped, worst quality, low quality, multiple views"
)

PROMPTS = {
    "p1": {
        "label": "単独・立ち",
        "prompt": (
            "1girl, solo, standing, full body, long silver hair, blue eyes, "
            "school uniform, looking at viewer, simple background, white background"
        ),
        "negative": NEGATIVE_COMMON,
        # danbooru タグ相当語 (2-6d README と同一・algorithm: WD14 検出タグ名 (underscore 形)
        # に語 (underscore 形へ変換) が含まれるかで再現率を数える)。
        "words": [
            "1girl", "solo", "standing", "long hair", "silver hair", "blue eyes",
            "school uniform", "looking at viewer", "simple background", "white background",
        ],
    },
    "p2": {
        "label": "手・小物",
        "prompt": (
            "1girl, solo, upper body, holding cup, both hands, smile, short brown hair, "
            "casual clothes, simple background, white background"
        ),
        "negative": NEGATIVE_COMMON,
        "words": [
            "1girl", "solo", "upper body", "cup", "smile", "short hair",
            "brown hair", "simple background", "white background",
        ],
    },
    "p3": {
        "label": "動き・全身",
        "prompt": (
            "1boy, solo, running, dynamic pose, full body, black hair, jacket, "
            "from side, simple background, white background"
        ),
        "negative": NEGATIVE_COMMON,
        "words": [
            "1boy", "solo", "running", "dynamic pose", "full body", "black hair",
            "jacket", "from side", "simple background", "white background",
        ],
    },
    # p4: 複数人 (2-6d に無い・破綻の主戦場なので新設)。
    "p4": {
        "label": "複数人",
        "prompt": (
            "2girls, multiple girls, standing together, one with long red hair, "
            "one with short blue hair, both smiling, simple background, white background"
        ),
        "negative": NEGATIVE_P4,
        "words": [
            "2girls", "multiple girls", "standing", "long hair", "red hair",
            "short hair", "blue hair", "smile", "simple background", "white background",
        ],
    },
}


# ============================================================
# 純ヘルパ (OV/subprocess 非依存・単体テスト対象)
# ============================================================
def check_log(log_text):
    """CLI 生成ログを検査する。(ok, reason) を返す。

    bad marker (case-sensitive) が 1 つでもあれば無効。成功マーカー
    (`dollama HTTP server (... backend ...— NPU)` / `— CPU)`) が無ければ無効
    (段1 SDXL backend に到達していない = pipeline/stub フォールバック)。
    """
    for m in BAD_MARKERS:
        if m in log_text:
            return False, f"bad marker '{m}' 検出"
    if " backend " not in log_text or ("— NPU)" not in log_text and "— CPU)" not in log_text):
        return False, "backend 成功マーカー不在 (段1 SDXL backend 未到達)"
    return True, ""


def word_to_tag(word):
    """参照語 (空白区切り) → WD14/danbooru タグ名 (underscore 区切り) へ変換する。"""
    return word.strip().replace(" ", "_")


def compute_recall(words, hit_tag_names):
    """words (danbooru タグ相当語) のうち hit_tag_names (検出タグ name 集合・underscore 形) に
    含まれる割合を (matched, total, recall) で返す。"""
    tagset = set(hit_tag_names)
    matched = sum(1 for w in words if word_to_tag(w) in tagset)
    total = len(words)
    return matched, total, (matched / total if total else 0.0)


def build_grid(presets, prompt_ids, seeds):
    """(preset, prompt_id, seed) の直積を決定的順序で返す。"""
    grid = []
    for preset in presets:
        for pid in prompt_ids:
            for seed in seeds:
                grid.append((preset, pid, seed))
    return grid


def mean_std(values):
    n = len(values)
    if n == 0:
        return None, None
    m = sum(values) / n
    var = sum((v - m) ** 2 for v in values) / n
    return m, var ** 0.5


# ============================================================
# 実走部 (--run でのみ呼ばれる。重い import は関数内に隔離)
# ============================================================
def _assets_status(args):
    reasons = []
    try:
        import openvino  # noqa: F401
    except Exception:
        reasons.append("openvino 不在")
    if not os.path.isfile(args.exe):
        reasons.append(f"dollama.exe 不在: {args.exe}")
    if not os.path.isfile(WD14_XML):
        reasons.append(f"WD14 IR 不在: {WD14_XML}")
    if not os.path.isfile(SCORER_IR):
        reasons.append(f"ScorerNet IR 不在: {SCORER_IR}")
    for preset in args.presets:
        if preset == "base":
            continue
        pdir = os.path.join(ROOT, "models", "presets", preset)
        if not os.path.isdir(pdir):
            reasons.append(f"preset 不在: {pdir}")
    return (len(reasons) == 0), reasons


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def preset_asset_files(preset):
    """preset ディレクトリの主要ファイル (走行中の入力データ不動確認対象)。"""
    if preset == "base":
        return []
    pdir = os.path.join(ROOT, "models", "presets", preset)
    rel = [
        "unet_weights.safetensors",
        "vae_weights.safetensors",
        os.path.join("text-encoder-l", "model_ov.xml"),
        os.path.join("text-encoder-l", "model_ov.bin"),
        os.path.join("text-encoder-g", "model_ov.xml"),
        os.path.join("text-encoder-g", "model_ov.bin"),
    ]
    return [os.path.join(pdir, r) for r in rel]


def snapshot_integrity(presets, exe):
    """exe + 使用する preset 一式の sha256 を控える (走行前後の突合用)。"""
    snap = {"exe": sha256_file(exe) if os.path.isfile(exe) else None, "presets": {}}
    for preset in presets:
        files = preset_asset_files(preset)
        snap["presets"][preset] = {
            os.path.relpath(f, ROOT).replace(os.sep, "/"):
                (sha256_file(f) if os.path.isfile(f) else None)
            for f in files
        }
    return snap


def run_one(exe, preset, pid, cond, steps, seed, out_png, log_path, tok_dll, timeout):
    """1 枚生成する (CLI 生成モード・別プロセス起動)。dict を返す。"""
    env = dict(os.environ)
    env["DOLLAMA_OV_TOKENIZERS_DLL"] = tok_dll
    env["DOLLAMA_SEED"] = str(seed)
    # 走行中の preset 入力データ移動事故 (2-6d) を踏まないよう、個別 env 上書きが
    # 残っていないことをここで明示的に消しておく (preset 解決を preset dir 一本にする)。
    for k in ("DOLLAMA_UNET_WEIGHTS", "DOLLAMA_VAE_WEIGHTS", "DOLLAMA_ENCODER_L",
              "DOLLAMA_ENCODER_G", "DOLLAMA_BACKEND_PRESET"):
        env.pop(k, None)

    cmd = [
        exe, "--prompt", cond["prompt"], "--negative", cond["negative"],
        "--steps", str(steps), "--preset", preset, "--out", out_png,
    ]
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, cwd=ROOT, env=env, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout,
        )
        sec = time.time() - t0
        log_text = (proc.stdout or "") + (proc.stderr or "")
        exit_code = proc.returncode
        timed_out = False
    except subprocess.TimeoutExpired as e:
        sec = time.time() - t0
        log_text = (e.stdout or "") + (e.stderr or "") if isinstance(e.stdout, str) else ""
        exit_code = -1
        timed_out = True

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(log_text)

    return {
        "sec": sec, "exit": exit_code, "log_text": log_text,
        "timed_out": timed_out, "png_exists": os.path.isfile(out_png),
    }


def preprocess_wd14(path):
    import numpy as np
    from PIL import Image
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
    square = square.resize((WD14_INPUT_SIZE, WD14_INPUT_SIZE), Image.BICUBIC)
    arr = np.asarray(square, dtype=np.float32)
    arr = arr[:, :, ::-1]  # RGB -> BGR (SmilingWolf 規約)
    return arr[np.newaxis, ...].copy()


def load_wd14_tags(csv_path):
    names, categories = [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            names.append(row["name"])
            categories.append(int(row["category"]))
    return names, categories


def run_on_research_machine(args):
    import numpy as np  # noqa: F401
    import openvino as ov
    from dollma_collect_rollouts import preprocess_for_scorer, axes_from_logits  # 再利用

    os.makedirs(args.out_dir, exist_ok=True)
    img_dir = os.path.join(args.out_dir, "img")
    log_dir = os.path.join(args.out_dir, "log")
    sheet_dir = os.path.join(args.out_dir, "contact_sheets")
    for d in (img_dir, log_dir, sheet_dir):
        os.makedirs(d, exist_ok=True)

    tok_dll = os.environ.get("DOLLAMA_OV_TOKENIZERS_DLL", "")
    if not tok_dll:
        print("[eval] [SKIP] DOLLAMA_OV_TOKENIZERS_DLL 未設定 → 段1 SDXL backend に到達できない")
        return

    # --- 走行前 sha256 スナップショット (preset 入力データ不動確認・規律③) ---
    snap_before = snapshot_integrity(args.presets, args.exe)
    print(f"[eval] 走行前 sha256 スナップショット: exe={snap_before['exe']}")

    # --- OV モデルをロード (grid 全体で 1 回だけ・per-image reload しない) ---
    core = ov.Core()
    wd14_names, wd14_cats = load_wd14_tags(WD14_TAGS_CSV)
    wd14_model = core.compile_model(core.read_model(WD14_XML), args.wd14_device)
    scorer_model = core.compile_model(core.read_model(SCORER_IR), args.scorer_device)
    print(f"[eval] WD14({args.wd14_device})/ScorerNet({args.scorer_device}) ロード完了")

    seeds = list(range(args.seed_base, args.seed_base + args.seeds))
    grid = build_grid(args.presets, args.prompts, seeds)
    print(f"[eval] grid: presets={args.presets} prompts={args.prompts} seeds={seeds} "
          f"→ {len(grid)} 枚")

    rows = []
    n_bad = 0
    for i, (preset, pid, seed) in enumerate(grid):
        cond = PROMPTS[pid]
        tag = f"{preset}__{pid}__seed{seed}"
        out_png = os.path.join(img_dir, tag + ".png")
        log_path = os.path.join(log_dir, tag + ".log")

        res = run_one(args.exe, preset, pid, cond, args.steps, seed, out_png, log_path,
                      tok_dll, args.gen_timeout)
        ok, reason = check_log(res["log_text"])
        row = {
            "preset": preset, "prompt_id": pid, "seed": seed,
            "sec": round(res["sec"], 2), "exit": res["exit"],
            "log_ok": ok, "log_reason": reason, "png": os.path.relpath(out_png, ROOT),
        }

        if res["exit"] != 0 or not res["png_exists"] or not ok:
            n_bad += 1
            row.update({
                "recall_matched": None, "recall_total": None, "recall": None,
                **{f"axis_{a}": None for a in AXIS_NAMES}, "argmax_axis": None,
                "worst_anatomy": None,
            })
            print(f"[eval] #{i:04d} {tag} 無効 (exit={res['exit']} png={res['png_exists']} "
                  f"log_ok={ok} reason='{reason}')")
            rows.append(row)
            continue

        # WD14 再現率
        x = preprocess_wd14(out_png)
        out = wd14_model(x)[wd14_model.output(0)][0]
        hit_names = [wd14_names[j] for j in range(len(out))
                     if wd14_cats[j] != 9 and float(out[j]) >= args.wd14_thresh]
        matched, total, recall = compute_recall(cond["words"], hit_names)

        # ScorerNet anatomy 8 軸
        sx = preprocess_for_scorer(out_png)
        logits = scorer_model(sx)[scorer_model.output(0)][0]
        axes = axes_from_logits(logits)
        worst = max(axes)
        argmax_axis = AXIS_NAMES[axes.index(worst)]

        row.update({
            "recall_matched": matched, "recall_total": total, "recall": round(recall, 4),
            **{f"axis_{a}": round(axes[k], 4) for k, a in enumerate(AXIS_NAMES)},
            "argmax_axis": argmax_axis, "worst_anatomy": round(worst, 4),
        })
        rows.append(row)

        if i < 3 or (i + 1) % 5 == 0:
            print(f"[eval] #{i:04d} {tag} recall={recall:.2f} worst={worst:.3f}"
                  f"({argmax_axis}) sec={res['sec']:.1f}")

    # --- 走行後 sha256 突合 (preset 入力データ不動確認・規律③) ---
    snap_after = snapshot_integrity(args.presets, args.exe)
    integrity_ok = (snap_before == snap_after)
    if not integrity_ok:
        print("[eval] [ALERT] 走行前後で exe/preset の sha256 が変化 (入力データが動いた疑い) "
              "→ 本走行の数値は無効の可能性")
        print(f"[eval]   before={json.dumps(snap_before, ensure_ascii=False)}")
        print(f"[eval]   after ={json.dumps(snap_after, ensure_ascii=False)}")
    else:
        print("[eval] 走行前後 sha256 一致 (exe/preset 入力データは不動)")

    # --- CSV 出力 (生データ) ---
    csv_path = os.path.join(args.out_dir, "grid_results.csv")
    fieldnames = (["preset", "prompt_id", "seed", "sec", "exit", "log_ok", "log_reason",
                    "png", "recall_matched", "recall_total", "recall"]
                  + [f"axis_{a}" for a in AXIS_NAMES] + ["argmax_axis", "worst_anatomy"])
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[eval] CSV (生データ): {csv_path}")

    # --- 集計 CSV (条件ごと 平均±std・秒は characterization のみ明記) ---
    agg_path = os.path.join(args.out_dir, "grid_summary.csv")
    agg_fields = ["preset", "prompt_id", "n", "n_valid", "recall_mean", "recall_std",
                  "worst_anatomy_mean", "worst_anatomy_std",
                  "sec_mean_characterization_only", "sec_std_characterization_only"]
    with open(agg_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=agg_fields)
        w.writeheader()
        for preset in args.presets:
            for pid in args.prompts:
                sub = [r for r in rows if r["preset"] == preset and r["prompt_id"] == pid]
                valid = [r for r in sub if r["recall"] is not None]
                rmean, rstd = mean_std([r["recall"] for r in valid])
                wmean, wstd = mean_std([r["worst_anatomy"] for r in valid])
                smean, sstd = mean_std([r["sec"] for r in sub])
                w.writerow({
                    "preset": preset, "prompt_id": pid, "n": len(sub), "n_valid": len(valid),
                    "recall_mean": round(rmean, 4) if rmean is not None else None,
                    "recall_std": round(rstd, 4) if rstd is not None else None,
                    "worst_anatomy_mean": round(wmean, 4) if wmean is not None else None,
                    "worst_anatomy_std": round(wstd, 4) if wstd is not None else None,
                    "sec_mean_characterization_only": round(smean, 2) if smean is not None else None,
                    "sec_std_characterization_only": round(sstd, 2) if sstd is not None else None,
                })
    print(f"[eval] CSV (集計 平均±std): {agg_path}")

    # --- コンタクトシート (条件ごと・seed 順) ---
    try:
        from PIL import Image, ImageDraw
        for preset in args.presets:
            for pid in args.prompts:
                sub = [r for r in rows if r["preset"] == preset and r["prompt_id"] == pid
                       and r["png"] and os.path.isfile(os.path.join(ROOT, r["png"]))]
                if not sub:
                    continue
                imgs = [(r["seed"], Image.open(os.path.join(ROOT, r["png"])).convert("RGB"))
                        for r in sub]
                thumb = 256
                cols = len(imgs)
                sheet = Image.new("RGB", (thumb * cols, thumb + 24), (32, 32, 32))
                draw = ImageDraw.Draw(sheet)
                for k, (seed, im) in enumerate(imgs):
                    im2 = im.resize((thumb, thumb))
                    sheet.paste(im2, (k * thumb, 24))
                    draw.text((k * thumb + 4, 4), f"seed={seed}", fill=(255, 255, 255))
                sheet_path = os.path.join(sheet_dir, f"{preset}__{pid}.png")
                sheet.save(sheet_path)
        print(f"[eval] コンタクトシート: {sheet_dir}")
    except Exception as e:
        print(f"[eval] [WARN] コンタクトシート生成に失敗: {e}")

    print(f"[eval] 完了: {len(rows)} 枚 (無効 {n_bad} 枚) → {args.out_dir}")
    print(f"[eval] exe_sha256={snap_before['exe']} integrity_ok={integrity_ok}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="E-1 画質評価ハーネス (条件×seed の直積・研究機専用・開発機は [SKIP])")
    ap.add_argument("--presets", nargs="+", default=["illustrious-xl"])
    ap.add_argument("--prompts", nargs="+", default=["p1", "p2", "p3", "p4"],
                    choices=list(PROMPTS.keys()))
    ap.add_argument("--seeds", type=int, default=4, help="seed 本数 (>=4 必須・信号ゲート)")
    ap.add_argument("--seed-base", dest="seed_base", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--exe", default=DOLLAMA_EXE)
    ap.add_argument("--out-dir", dest="out_dir", default=os.path.join(ROOT, "docs", "logs", "e1"))
    ap.add_argument("--wd14-thresh", dest="wd14_thresh", type=float, default=0.35)
    ap.add_argument("--wd14-device", dest="wd14_device", default="CPU")
    ap.add_argument("--scorer-device", dest="scorer_device", default="CPU")
    ap.add_argument("--gen-timeout", dest="gen_timeout", type=float, default=180.0)
    ap.add_argument("--run", action="store_true",
                    help="研究機で実走する (RTX5080+OV+dollama.exe 必須)。未指定は計画表示のみ。")
    args = ap.parse_args(argv)

    if args.seeds < 4:
        print(f"[eval] [WARN] --seeds={args.seeds} < 4 (規律①: N=1 で優劣を書かない・"
              "集計は必ず seed>=4 本の平均±std で報告すること)")

    ok, reasons = _assets_status(args)
    if not args.run:
        seeds = list(range(args.seed_base, args.seed_base + args.seeds))
        grid = build_grid(args.presets, args.prompts, seeds)
        print(f"[PLAN] presets={args.presets} prompts={args.prompts} seeds={seeds} "
              f"→ {len(grid)} 枚 / steps={args.steps} / out_dir={args.out_dir}")
        print(f"[PLAN] 実走資産: {'揃っている' if ok else '不足'}"
              + ("" if ok else f" ({'; '.join(reasons)})"))
        print("[PLAN] 研究機で実走するには --run を付ける。実走前に SAC OFF をユーザーへ依頼すること。")
        return

    if not ok:
        print(f"[SKIP] 実走資産が揃っていないためスキップ: {'; '.join(reasons)}")
        print("[SKIP] 研究機 (RTX5080 + OpenVINO + dollama.exe + presets) で実走すること。")
        return

    run_on_research_machine(args)


if __name__ == "__main__":
    main()

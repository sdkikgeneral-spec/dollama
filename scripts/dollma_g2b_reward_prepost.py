# -*- coding: utf-8 -*-
"""dollma_g2b_reward_prepost.py — Phase 4 施策 F / F-0b Package G-2b:
SFT 前 (正典 LM) vs SFT 後 (rejection-SFT LM) の **平均 reward 前後比** を、
G-1 の 400 訓練入力と disjoint な held-out 入力で**ペア**測定する。

閉路 (1 入力・2 モデル):
  1. 正典 LM (bitnet_dense_fp32) と SFT LM (bitnet_dense_sft_fp32) が同一 input_text から
     **greedy 1 本ずつ** プロンプト生成 (best-of-N でなく greedy = SFT が greedy 出力を
     どれだけ良くしたか)。LM は決定的 (argmax)。
  2. 各プロンプト → 既存 txt2img HTTP サーバ (dollama --http・品質ネガなし) で SDXL 生成。
  3. ScorerNet(OV) で anatomy 8 軸 + CLIP image→QualityMLP(OV) で quality を採点。
  4. dollma_reward.reward_from_scorer(axes, quality) でスカラ報酬 (anatomy + quality w0.4)。
  5. (post_id, model) ごとに 1 行を **即 append + flush** (中断/セッション上限に頑健・resume 可)。

ペアの扱い (重要な制約):
  現行 HTTP サーバは SDXL seed を server 内 wall-clock で決める (GenRequest に seed フィールド
  無し・再ビルドは SAC でブロック)。よって同一入力の pre/post 2 枚は **異なる SDXL seed**。
  ペアは「同一 input_text で両モデルがプロンプト生成」の**入力単位**で成立し、per-input の
  Δreward には SDXL seed ノイズが乗る。~100 ペアの平均 Δreward がこのノイズを均した推定値。

リーク防止 (必須):
  held-out 入力は diverse コーパスから採り、
    (a) diverse-val 評価セット (pairs.eval_diverse_{a,b} / pairs.val) の post_id
    (b) G-1 の 400 訓練入力 (data/rollouts/sft_bestofn.jsonl の post_id)
  の**両方と素性重複ゼロ (disjoint)** を保証する。両方を除外してから決定的サンプリング。

無改変: 正典 bitnet_dense*/golden/scorer_net/quality_mlp/IR は読むだけ。data/rollouts/ は gitignore。
"""

import argparse
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import dollma_collect_rollouts as cr                     # noqa: E402
from dollma_reward import (reward_from_scorer, QUALITY_WEIGHT,  # noqa: E402
                           MEAN_TIE_WEIGHT)

SCORER_IR = cr.SCORER_IR
CLIP_IMAGE_IR = cr.CLIP_IMAGE_IR
QUALITY_MLP_IR = cr.QUALITY_MLP_IR
VOCAB_PATH = cr.VOCAB_PATH
DOLLAMA_EXE = cr.DOLLAMA_EXE

# 正典 (SFT 前) と SFT 後 (G-2a 隔離重み)。差し替えは prompt 生成側のみ・SDXL 共通。
PRE_WEIGHTS = os.path.join(ROOT, "data", "bitnet", "bitnet_dense_fp32.safetensors")
SFT_WEIGHTS = os.path.join(ROOT, "data", "bitnet", "bitnet_dense_sft_fp32.safetensors")

INPUT_CORPUS_GLOB = os.path.join(ROOT, "data", "bitnet", "diverse_train_texts_part*.jsonl")
EVAL_LEAK_FILES = [
    os.path.join(ROOT, "data", "bitnet", "pairs.eval_diverse_a.jsonl"),
    os.path.join(ROOT, "data", "bitnet", "pairs.eval_diverse_b.jsonl"),
    os.path.join(ROOT, "data", "bitnet", "pairs.val.jsonl"),
]
# G-1 の 400 訓練入力 (これと disjoint な held-out を採る)。
G1_TRAIN_SFT = os.path.join(ROOT, "data", "rollouts", "sft_bestofn.jsonl")


# ============================================================
# 入力選定 (eval + G-1 訓練入力の両方と disjoint・決定的)
# ============================================================
def _train_post_ids(path):
    """sft_bestofn.jsonl (G-1 の 400 勝者ペア) から訓練入力 post_id 集合を返す。"""
    ids = set()
    if not os.path.isfile(path):
        return ids
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            pid = (r.get("meta", {}) or {}).get("post_id", r.get("post_id"))
            if pid is not None:
                ids.add(int(pid))
    return ids


def select_heldout(corpus_glob, eval_files, train_sft, m, seed):
    """M 件の held-out 入力を eval ∪ G-1訓練 と post_id disjoint に決定的サンプリング。

    返り値: (inputs[list], report[dict])。report に disjoint 検証結果を残す。
    """
    import random
    from dollma_rollout_bestofn import load_input_corpus, _post_ids_from_pairs

    corpus = load_input_corpus(corpus_glob)
    eval_leak = set()
    per_file = {}
    for p in eval_files:
        ids = _post_ids_from_pairs(p)
        per_file[os.path.basename(p)] = len(ids)
        eval_leak |= ids
    train_leak = _train_post_ids(train_sft)
    leak = eval_leak | train_leak

    eligible = [row for pid, row in corpus.items() if pid not in leak]
    eligible.sort(key=lambda r: r["post_id"])
    rng = random.Random(seed)
    rng.shuffle(eligible)
    if len(eligible) < m:
        raise RuntimeError(
            f"eligible {len(eligible)} 件 < 要求 M={m} (corpus {len(corpus)} / "
            f"leak {len(leak)})")
    chosen = eligible[:m]

    chosen_ids = {r["post_id"] for r in chosen}
    ov_eval = chosen_ids & eval_leak
    ov_train = chosen_ids & train_leak
    n_ja = sum(1 for r in chosen if r.get("lang", "ja") == "ja")
    n_en = len(chosen) - n_ja
    report = {
        "corpus_glob": os.path.relpath(corpus_glob, ROOT).replace(os.sep, "/"),
        "corpus_unique_post_ids": len(corpus),
        "eval_leak_post_ids_per_file": per_file,
        "eval_leak_post_ids_union": len(eval_leak),
        "g1_train_post_ids": len(train_leak),
        "combined_leak_union": len(leak),
        "eligible_after_leak_exclusion": len(eligible),
        "requested_m": m,
        "selected": len(chosen),
        "selected_overlap_with_eval": len(ov_eval),
        "selected_overlap_with_g1_train": len(ov_train),
        "disjoint_ok": (len(ov_eval) == 0 and len(ov_train) == 0),
        "selected_lang_ja": n_ja,
        "selected_lang_en": n_en,
        "seed": seed,
    }
    return chosen, report


# ============================================================
# LM ロード / greedy 生成
# ============================================================
def build_lm(weights, vocab, device):
    """BitNetDense を weights からロードして (model, tok, tb, torch) を返す。"""
    import torch
    from safetensors.torch import load_file
    import train_bitnet as tb

    tok = tb.Tokenizer(vocab)
    model = tb.BitNetDense().to(device)
    sd = load_file(weights)
    model.load_state_dict(cr._remap_bitnet_state_dict(sd, tb.N_LAYERS), strict=True)
    model.eval()
    return model, tok, tb, torch


def greedy_prompt(model, tok, tb, torch, input_text, device, max_len):
    """input_text から greedy 1 本のプロンプト (dedup 済タグ列) を返す。"""
    prompt_ids = [tb.TOK_BOS] + tb.encode_text_greedy(tok, input_text) + [tb.TOK_SEP]
    with torch.no_grad():
        ids = tb._greedy_generate(model, prompt_ids, device, max_len=max_len)
    seen = set()
    tag_ids = []
    for i in ids:
        if i >= 5 and i not in seen:
            seen.add(i)
            tag_ids.append(i)
    tags = [tok.tags[i - 5] for i in tag_ids]
    return ", ".join(tags), tags


# ============================================================
# reward 内訳 (anatomy 成分 vs quality 成分)
# ============================================================
def reward_components(axes, quality):
    """reward_from_scorer と同一定義で total/anatomy寄与/quality寄与を分解して返す。

    combined = (1-w)*anatomy_reward + w*(quality-1.0)
    anatomy_contribution = (1-w)*anatomy_reward,  quality_contribution = w*(quality-1.0)
    """
    worst = max(axes)
    mean = sum(axes) / len(axes)
    anatomy_reward = -((1.0 - MEAN_TIE_WEIGHT) * worst + MEAN_TIE_WEIGHT * mean)
    if quality is None:
        return anatomy_reward, anatomy_reward, 0.0
    ana_c = (1.0 - QUALITY_WEIGHT) * anatomy_reward
    qual_c = QUALITY_WEIGHT * (quality - 1.0)
    return ana_c + qual_c, ana_c, qual_c


# ============================================================
# 統計
# ============================================================
def _dist(vals):
    if not vals:
        return None
    n = len(vals)
    s = sorted(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    med = s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
    return {"min": round(min(vals), 5), "median": round(med, 5),
            "max": round(max(vals), 5), "mean": round(mean, 5),
            "std": round(var ** 0.5, 5), "n": n}


def _done_keys(path):
    """既存 g2b jsonl から完了済み (post_id, model) キー集合を返す (resume 用・冪等)。"""
    done = set()
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                pid = r.get("post_id")
                model = r.get("model")
                if pid is not None and model is not None:
                    done.add((int(pid), model))
    return done


def summarize(out_path, sel_report, extra=None):
    """out_path の全行から pre/post ペア統計を組んで dict を返す。"""
    rows = []
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    by_pid = {}
    for r in rows:
        by_pid.setdefault(int(r["post_id"]), {})[r["model"]] = r
    pre_r, post_r = [], []
    dre, dana, dqual = [], [], []
    dre_ja, dre_en = [], []
    pairs = 0
    for pid, d in by_pid.items():
        if "pre" not in d or "post" not in d:
            continue
        pairs += 1
        rp, rs = d["pre"], d["post"]
        pre_r.append(rp["reward"])
        post_r.append(rs["reward"])
        delta = rs["reward"] - rp["reward"]
        dre.append(delta)
        dana.append(rs["anatomy_contribution"] - rp["anatomy_contribution"])
        dqual.append(rs["quality_contribution"] - rp["quality_contribution"])
        lang = rp.get("lang", "ja")
        (dre_ja if lang == "ja" else dre_en).append(delta)
    n_pos = sum(1 for d in dre if d > 0)
    n_neg = sum(1 for d in dre if d < 0)
    n_zero = sum(1 for d in dre if d == 0)
    summary = {
        "stage": "F-0b G-2b 平均 reward 前後比 (greedy pre vs post・ペア)",
        "paired_inputs": pairs,
        "reward_pre_dist": _dist(pre_r),
        "reward_post_dist": _dist(post_r),
        "delta_reward_dist": _dist(dre),
        "delta_reward_positive_frac": round(n_pos / pairs, 4) if pairs else None,
        "delta_reward_sign_counts": {"pos": n_pos, "neg": n_neg, "zero": n_zero},
        "delta_anatomy_contribution_dist": _dist(dana),
        "delta_quality_contribution_dist": _dist(dqual),
        "delta_reward_ja_dist": _dist(dre_ja),
        "delta_reward_en_dist": _dist(dre_en),
        "quality_weight": QUALITY_WEIGHT,
        "pre_weights": os.path.relpath(PRE_WEIGHTS, ROOT).replace(os.sep, "/"),
        "sft_weights": os.path.relpath(SFT_WEIGHTS, ROOT).replace(os.sep, "/"),
        "seed_note": ("LM=greedy 決定的 / SDXL seed=server wall-clock=非制御 → per-input Δ に "
                      "seed ノイズ・~100 ペア平均で均す。判定は G-3 で。"),
        "input_selection": sel_report,
    }
    if extra:
        summary.update(extra)
    return summary


# ============================================================
# 実走
# ============================================================
def run(args, inputs, sel_report):
    import numpy as np  # noqa: F401
    import openvino as ov
    import dollma_gen_scorer_corpus as gc

    done = _done_keys(args.out)
    # 未完 (post_id, model) を列挙 (pre→post の順で並べる)。
    work = []
    for oi, row in enumerate(inputs):
        pid = row["post_id"]
        for model_tag in ("pre", "post"):
            if (pid, model_tag) not in done:
                work.append((oi, row, model_tag))
    total_units = len(inputs) * 2
    print(f"[g2b] resume: 完了 {len(done)}/{total_units} units / 未完 {len(work)} "
          f"(limit={args.limit})", file=sys.stderr)
    if args.limit and args.limit > 0:
        work = work[:args.limit]
    if not work:
        summary = summarize(args.out, sel_report, {"status": "final"})
        with open(args.summary_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print("[g2b] 未完なし → summary 更新のみ", file=sys.stderr)
        print("[g2b] SUMMARY " + json.dumps(summary, ensure_ascii=False), file=sys.stderr)
        return summary

    # --- 2 LM ロード (33M×2・CPU で決定的) ---
    lms = {
        "pre": build_lm(PRE_WEIGHTS, args.vocab, args.lm_device),
        "post": build_lm(SFT_WEIGHTS, args.vocab, args.lm_device),
    }
    print(f"[g2b] LM ロード完了 pre={PRE_WEIGHTS} sft={SFT_WEIGHTS} device={args.lm_device}",
          file=sys.stderr)

    tok_dll = gc.resolve_tokenizers_dll(os.environ.get("DOLLAMA_OV_TOKENIZERS_DLL"))
    core = ov.Core()
    scorer = core.compile_model(core.read_model(SCORER_IR), args.scorer_device)
    import open_clip
    _cm, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14-quickgelu", pretrained="openai")
    del _cm
    clip_ov = core.compile_model(core.read_model(args.clip_image_ir), args.clip_device)
    qmlp_ov = core.compile_model(core.read_model(args.quality_mlp_ir), args.quality_device)
    print(f"[g2b] scorer/quality 経路ロード完了", file=sys.stderr)

    env = dict(os.environ)
    env["DOLLAMA_OV_TOKENIZERS_DLL"] = tok_dll
    env["DOLLAMA_MATTING_WEIGHTS"] = args.no_matting_marker

    proc = None
    base_url = args.server_url
    if base_url:
        gc.wait_for_health(base_url, timeout=args.health_timeout)
    else:
        base_url = f"http://127.0.0.1:{args.port}"
        cmd = [args.exe, "--http", "--port", str(args.port), "--steps", str(args.steps)]
        print(f"[g2b] サーバ起動: {' '.join(cmd)}", file=sys.stderr)
        proc = gc.subprocess.Popen(cmd, cwd=ROOT, env=env)
        gc.wait_for_health(base_url, timeout=args.health_timeout, proc=proc)
        print("[g2b] サーバ ready", file=sys.stderr)

    img_dir = args.img_dir
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    n_gen = 0
    t0 = time.time()
    try:
        fout = open(args.out, "a", encoding="utf-8")
        for k, (oi, row, model_tag) in enumerate(work):
            input_text = row["text"]
            post_id = row["post_id"]
            lang = row.get("lang", "ja")
            model, tok, tb, torch = lms[model_tag]
            prompt, tags = greedy_prompt(model, tok, tb, torch, input_text,
                                         args.lm_device, args.max_gen_len)

            axes = None
            quality = None
            reward = None
            ana_c = qual_c = None
            empty = (not prompt.strip())
            if empty:
                # 空プロンプト (日本語で全語彙外 かつ 無条件生成も空) は測定不能 → reward None。
                # greedy は通常タグを出すため稀。記録だけ残す。
                pass
            else:
                body = gc.build_request_body({"prompt": prompt}, args.steps, "1024x1024")
                b64 = gc.post_generation(base_url, body, timeout=args.gen_timeout)
                img_path = os.path.join(img_dir, f"{post_id}_{model_tag}.png")
                gc.save_b64_png(b64, img_path)
                n_gen += 1
                x = cr.preprocess_for_scorer(img_path)
                logits = scorer(x)[scorer.output(0)][0]
                axes = cr.axes_from_logits(logits)
                quality = cr.compute_quality_clip(img_path, clip_preprocess, clip_ov, qmlp_ov)
                reward = reward_from_scorer(axes, quality)
                total, ana_c, qual_c = reward_components(axes, quality)
                if not args.keep_images:
                    try:
                        os.remove(img_path)
                    except OSError:
                        pass

            rec = {
                "post_id": post_id,
                "model": model_tag,
                "input_text": input_text,
                "lang": lang,
                "prompt": prompt,
                "tags": tags,
                "empty": empty,
                "axes": ([float(a) for a in axes] if axes is not None else None),
                "quality": (float(quality) if quality is not None else None),
                "reward": (float(reward) if reward is not None else None),
                "anatomy_contribution": (float(ana_c) if ana_c is not None else None),
                "quality_contribution": (float(qual_c) if qual_c is not None else None),
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()

            if k < 5 or (k + 1) % 10 == 0:
                el = time.time() - t0
                rate = el / max(1, n_gen)
                remain = len(work) - (k + 1)
                print(f"[g2b] {k+1}/{len(work)} pid={post_id} model={model_tag} "
                      f"reward={reward if reward is None else round(reward,4)} "
                      f"gen={n_gen} el={el/60:.1f}m eta_chunk={rate*remain/60:.1f}m "
                      f"prompt='{prompt[:40]}'", file=sys.stderr)
        fout.close()
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except Exception:
                proc.kill()

    done_now = _done_keys(args.out)
    is_final = (len(done_now) >= total_units)
    summary = summarize(args.out, sel_report,
                        {"status": "final" if is_final else "partial",
                         "chunk_generations": n_gen,
                         "chunk_elapsed_sec": round(time.time() - t0, 1),
                         "sec_per_generation": round((time.time() - t0) / max(1, n_gen), 3)})
    with open(args.summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("[g2b] SUMMARY " + json.dumps(summary, ensure_ascii=False), file=sys.stderr)
    tail = ("全 units 完走 (final)" if is_final
            else f"チャンク完了 done={len(done_now)}/{total_units} (--run --limit N で resume)")
    print(f"[g2b] {tail} → {args.out} / summary → {args.summary_out}")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="F-0b G-2b reward 前後比 (greedy pre vs post・研究機 GPU)。")
    ap.add_argument("--m", type=int, default=100, help="held-out 入力数")
    ap.add_argument("--limit", type=int, default=0,
                    help="本チャンクで処理する未完 unit 上限 (0=全部・>0=N unit で resume 終了)")
    ap.add_argument("--max-gen-len", dest="max_gen_len", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20260705, help="held-out 入力選定 seed")
    ap.add_argument("--corpus-glob", dest="corpus_glob", default=INPUT_CORPUS_GLOB)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "rollouts", "g2b_prepost.jsonl"))
    ap.add_argument("--summary-out", dest="summary_out",
                    default=os.path.join(ROOT, "data", "rollouts", "g2b_summary.json"))
    ap.add_argument("--img-dir", dest="img_dir",
                    default=os.path.join(ROOT, "data", "rollouts", "img_g2b"))
    ap.add_argument("--keep-images", dest="keep_images", action="store_true",
                    help="生成画像を残す (既定は採点後に削除)")
    ap.add_argument("--vocab", dest="vocab", default=VOCAB_PATH)
    ap.add_argument("--exe", default=DOLLAMA_EXE)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--server-url", dest="server_url", default=None)
    ap.add_argument("--lm-device", dest="lm_device", default="cpu")
    ap.add_argument("--scorer-device", dest="scorer_device", default="CPU")
    ap.add_argument("--clip-image-ir", dest="clip_image_ir", default=CLIP_IMAGE_IR)
    ap.add_argument("--quality-mlp-ir", dest="quality_mlp_ir", default=QUALITY_MLP_IR)
    ap.add_argument("--clip-device", dest="clip_device", default="CPU")
    ap.add_argument("--quality-device", dest="quality_device", default="CPU")
    ap.add_argument("--health-timeout", dest="health_timeout", type=float, default=300.0)
    ap.add_argument("--gen-timeout", dest="gen_timeout", type=float, default=180.0)
    ap.add_argument("--no-matting-marker", dest="no_matting_marker", default="__no_matting__")
    ap.add_argument("--run", action="store_true", help="研究機で実走 (未指定は計画+disjoint検証のみ)")
    args = ap.parse_args(argv)

    inputs, sel_report = select_heldout(args.corpus_glob, EVAL_LEAK_FILES,
                                        G1_TRAIN_SFT, args.m, args.seed)
    print("[g2b] INPUT_SELECTION " + json.dumps(sel_report, ensure_ascii=False))
    if not sel_report["disjoint_ok"]:
        raise SystemExit("[g2b] disjoint 検証失敗 (eval/G-1訓練 と重複)")

    # 資産チェック (SFT 重みも要る)。
    class _A:
        pass
    a = _A()
    a.bitnet_weights = PRE_WEIGHTS
    a.vocab = args.vocab
    a.clip_image_ir = args.clip_image_ir
    a.quality_mlp_ir = args.quality_mlp_ir
    ok, reasons = cr._research_assets_status(a)
    if not os.path.isfile(SFT_WEIGHTS):
        ok = False
        reasons.append(f"SFT 重み不在: {SFT_WEIGHTS}")

    if not args.run:
        print(f"[PLAN] G-2b reward 前後比: M={args.m} × 2 model = {args.m * 2} 枚生成")
        print(f"[PLAN] pre={PRE_WEIGHTS}")
        print(f"[PLAN] sft={SFT_WEIGHTS}")
        print(f"[PLAN] disjoint_ok={sel_report['disjoint_ok']} "
              f"ja={sel_report['selected_lang_ja']} en={sel_report['selected_lang_en']}")
        print(f"[PLAN] 実走資産: {'揃っている' if ok else '不足'}"
              + ("" if ok else f" ({'; '.join(reasons)})"))
        for r in inputs[:3]:
            print(f"  例 input(pid={r['post_id']}): {r['text'][:48]}")
        return

    if not ok:
        print(f"[SKIP] 実走資産が揃っていない: {'; '.join(reasons)}")
        return
    run(args, inputs, sel_report)


if __name__ == "__main__":
    main()

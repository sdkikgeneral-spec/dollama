# -*- coding: utf-8 -*-
"""dollma_rollout_bestofn.py — Phase 4 施策 F / F-0b Package G-1:
best-of-N rollout 収集ハーネス (研究機・GPU 専用)。

閉路 (1 入力あたり):
  1. LM (BitNetDense) が input_text から **N=8 候補** プロンプトを temperature/top-k
     サンプリング生成 (train_bitnet._sample_generate・(input,candidate) ごとに固定 seed)。
  2. 各候補を既存 txt2img HTTP サーバ (dollama --http) に投げて SDXL 画像生成 (品質ネガ注入なし)。
  3. ScorerNet(OV) で anatomy 8 軸 + CLIP image→QualityMLP(OV) で quality を採点。
  4. dollma_reward.reward_from_scorer(axes, quality) でスカラ報酬 (anatomy + quality w0.4)。
  5. **top-1 of N (RAFT)** で reward 最大の 1 本を勝者に選抜。
  6. SFT データ (input_text → 勝者プロンプト) × M を jsonl 化 (train_bitnet SFT 経路が読める schema)。

リーク防止 (必須・G-1 の非交渉条件):
  rollout 入力は diverse 自然文コーパス (data/bitnet/diverse_train_texts_part*.jsonl) から
  採り、diverse-val 評価セット (pairs.eval_diverse_{a,b} / pairs.val) と **post_id 素性重複ゼロ
  (disjoint)** を保証する。重複すると G-2a の set-F1 非退行ゲートが自己参照で嵩上げされる。

再現性:
  - LM サンプリングは (input_index, candidate_index) 決定の固定 seed の torch.Generator で再現的。
  - **SDXL seed は server 内 wall-clock ゆえ非再現** → 画像は再生成不可。勝者プロンプト・
    全候補の reward/axes/quality を確実に保存する (data/rollouts/ は gitignore・PNG は破棄可)。

実走前提 (研究機・gpu-benchmarker): RTX5080 + OpenVINO + 実 SDXL 重み + ScorerNet/CLIP/QualityMLP
IR + BitNet 重み + openvino_tokenizers.dll。開発機では資産不在→[SKIP]。
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

# 既存 rollout ハーネスの純ヘルパ / 実走部品を最大限流用 (配管の二重実装回避)。
import dollma_collect_rollouts as cr           # noqa: E402
from dollma_reward import reward_from_scorer, QUALITY_WEIGHT  # noqa: E402

# 既定パス (collect_rollouts と同一資産)。
SCORER_IR = cr.SCORER_IR
CLIP_IMAGE_IR = cr.CLIP_IMAGE_IR
QUALITY_MLP_IR = cr.QUALITY_MLP_IR
BITNET_WEIGHTS = cr.BITNET_WEIGHTS
VOCAB_PATH = cr.VOCAB_PATH
DOLLAMA_EXE = cr.DOLLAMA_EXE

# 入力コーパス (diverse 自然文・tags-stay-real 系)。post_id で eval と disjoint 判定する。
INPUT_CORPUS_GLOB = os.path.join(ROOT, "data", "bitnet", "diverse_train_texts_part*.jsonl")
# リーク防止で除外する評価セット (post_id 素性)。
EVAL_LEAK_FILES = [
    os.path.join(ROOT, "data", "bitnet", "pairs.eval_diverse_a.jsonl"),
    os.path.join(ROOT, "data", "bitnet", "pairs.eval_diverse_b.jsonl"),
    os.path.join(ROOT, "data", "bitnet", "pairs.val.jsonl"),
]


# ============================================================
# 入力プール構築 (post_id disjoint・決定的サンプリング)
# ============================================================
def _post_ids_from_pairs(path):
    """pairs.*.jsonl から post_id 集合を取り出す (meta.post_id 優先・top-level fallback)。"""
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


def load_input_corpus(corpus_glob):
    """diverse_train_texts_part*.jsonl 群から {post_id, text} を読む (post_id 一意化)。"""
    rows = {}
    for path in sorted(glob.glob(corpus_glob)):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                pid = r.get("post_id")
                text = r.get("text")
                if pid is None or not text:
                    continue
                rows[int(pid)] = {"post_id": int(pid), "text": text,
                                  "lang": r.get("lang", "ja")}
    return rows


def select_inputs(corpus_glob, eval_files, m, seed):
    """M 入力を eval と post_id disjoint に決定的サンプリングして返す。

    返り値: (inputs[list], report[dict])。report に disjoint 検証結果を残す。
    """
    import random
    corpus = load_input_corpus(corpus_glob)
    leak = set()
    per_file = {}
    for p in eval_files:
        ids = _post_ids_from_pairs(p)
        per_file[os.path.basename(p)] = len(ids)
        leak |= ids

    eligible = [row for pid, row in corpus.items() if pid not in leak]
    # 決定的: post_id 昇順に並べてから seed 付き shuffle → 先頭 M。
    eligible.sort(key=lambda r: r["post_id"])
    rng = random.Random(seed)
    rng.shuffle(eligible)
    if len(eligible) < m:
        raise RuntimeError(
            f"eligible 入力 {len(eligible)} 件 < 要求 M={m} (コーパス {len(corpus)} / "
            f"leak 除外 {len(corpus) - len(eligible)})")
    chosen = eligible[:m]

    chosen_ids = {r["post_id"] for r in chosen}
    overlap = chosen_ids & leak
    report = {
        "corpus_glob": os.path.relpath(corpus_glob, ROOT).replace(os.sep, "/"),
        "corpus_unique_post_ids": len(corpus),
        "eval_leak_post_ids_per_file": per_file,
        "eval_leak_post_ids_union": len(leak),
        "eligible_after_leak_exclusion": len(eligible),
        "requested_m": m,
        "selected": len(chosen),
        "selected_overlap_with_eval": len(overlap),   # 0 でなければ即失敗
        "disjoint_ok": (len(overlap) == 0),
        "seed": seed,
    }
    return chosen, report


# ============================================================
# LM 候補生成 (温度サンプリング・(input,candidate) 決定 seed)
# ============================================================
def _dedup_preserve(seq):
    """順序を保って id の重複を除く (同一 danbooru タグ重複は SDXL に無意味)。"""
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def build_lm(args):
    """BitNetDense をロードして (model, tok) を返す (collect_rollouts の remap を流用)。"""
    import torch
    from safetensors.torch import load_file
    import train_bitnet as tb

    device = args.lm_device
    tok = tb.Tokenizer(args.vocab)
    model = tb.BitNetDense().to(device)
    sd = load_file(args.bitnet_weights)
    model.load_state_dict(cr._remap_bitnet_state_dict(sd, tb.N_LAYERS), strict=True)
    model.eval()
    return model, tok, tb, torch


def sample_candidates(model, tok, tb, torch, input_text, n, base_seed, input_idx,
                      temperature, top_k, device, max_len):
    """input_text から N 候補プロンプトを温度サンプリング生成する。

    各候補 j は seed = base_seed + input_idx*10007 + j で決定的 (監査可能)。
    返り値: list[dict] = [{"prompt": str, "tag_ids": [..], "lm_seed": int}] × n。
    """
    prompt_ids = [tb.TOK_BOS] + tb.encode_text_greedy(tok, input_text) + [tb.TOK_SEP]
    cands = []
    for j in range(n):
        lm_seed = int(base_seed) + int(input_idx) * 10007 + j
        gen = torch.Generator(device="cpu").manual_seed(lm_seed)
        ids = tb._sample_generate(model, prompt_ids, device, gen,
                                  max_len=max_len, temperature=temperature, top_k=top_k)
        ids = _dedup_preserve([i for i in ids if i >= 5])   # specials 除外 + 重複除去
        tags = [tok.tags[i - 5] for i in ids]
        cands.append({"prompt": ", ".join(tags), "tag_ids": ids,
                      "tags": tags, "lm_seed": lm_seed})
    return cands


# ============================================================
# 統計ヘルパ
# ============================================================
def _dist(vals):
    if not vals:
        return None
    n = len(vals)
    s = sorted(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    med = s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
    return {"min": round(min(vals), 4), "median": round(med, 4),
            "max": round(max(vals), 4), "mean": round(mean, 4),
            "std": round(var ** 0.5, 4), "n": n}


# ============================================================
# 実走
# ============================================================
def _completed_post_ids(sft_path):
    """既存 sft_bestofn.jsonl から完了済み post_id 集合を返す (resume 用・冪等)。"""
    done = set()
    if os.path.isfile(sft_path):
        with open(sft_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                pid = (r.get("meta", {}) or {}).get("post_id")
                if pid is not None:
                    done.add(int(pid))
    return done


def _cumulative_stats(sft_path, cand_path):
    """完了済み全入力の勝者 reward 分布 (sft.meta.reward=権威値) と、candidates を
    完了済み post_id でグループ化した spread/dup を累計集計する。

    返り値: (winner_rewards, spreads, dup_rates, done_count)。
    部分グループ (sft 未確定 post_id) は除外して途中中断に頑健。
    """
    winner_rewards = []
    done_set = set()
    if os.path.isfile(sft_path):
        with open(sft_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                meta = r.get("meta", {}) or {}
                pid = meta.get("post_id")
                if pid is None:
                    continue
                done_set.add(int(pid))
                if meta.get("reward") is not None:
                    winner_rewards.append(float(meta["reward"]))
    groups = {}
    if os.path.isfile(cand_path):
        with open(cand_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                pid = r.get("post_id")
                if pid is None or int(pid) not in done_set:
                    continue
                groups.setdefault(int(pid), []).append(r)
    spreads = []
    dup_rates = []
    for pid, recs in groups.items():
        rewards = [c["reward"] for c in recs if c.get("reward") is not None]
        if len(rewards) >= 2:
            spreads.append(max(rewards) - min(rewards))
        n = len(recs)
        uniq = len({c.get("prompt", "") for c in recs})
        if n > 0:
            dup_rates.append((n - uniq) / n)
    return winner_rewards, spreads, dup_rates, len(done_set)


def _write_summary(args, sel_report, chunk_n_gen, chunk_elapsed,
                   empty_prompt_cands, n_inputs_total, partial):
    """summary を累計 (disk 由来) で組んで書き、dict を返す (チャンク resume 対応)。"""
    winner_rewards, spreads, dup_rates, done = _cumulative_stats(args.sft_out, args.cand_out)
    remaining = max(0, n_inputs_total - done)
    rate = (chunk_elapsed / chunk_n_gen) if chunk_n_gen else None   # sec/gen (本チャンク)
    eta_sec = (remaining * args.n * rate) if rate else None
    summary = {
        "stage": "F-0b G-1 best-of-N rollout 収集 (resume/chunk)",
        "status": ("partial (途中経過)" if partial else "final"),
        "n_inputs_total": n_inputs_total,
        "n_inputs_done": done,
        "n_inputs_remaining": remaining,
        "n_candidates_per_input": args.n,
        "chunk_sdxl_generations": chunk_n_gen,
        "chunk_elapsed_sec": round(chunk_elapsed, 1),
        "chunk_empty_prompt_candidates": empty_prompt_cands,
        "sec_per_generation": round(rate, 3) if rate else None,
        "eta_remaining_hours": round(eta_sec / 3600.0, 2) if eta_sec is not None else None,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "quality_weight": QUALITY_WEIGHT,
        "winner_reward_dist": _dist(winner_rewards),
        "best_minus_worst_per_input_dist": _dist(spreads),
        "candidate_dup_rate_dist": _dist(dup_rates),
        "candidate_mean_unique_of_n": (round(sum(1 - d for d in dup_rates) / len(dup_rates)
                                             * args.n, 3) if dup_rates else None),
        "input_selection": sel_report,
        "sft_out": os.path.relpath(args.sft_out, ROOT).replace(os.sep, "/"),
        "cand_out": os.path.relpath(args.cand_out, ROOT).replace(os.sep, "/"),
        "bitnet_weights": os.path.relpath(args.bitnet_weights, ROOT).replace(os.sep, "/"),
        "seed_note": "LM sampling seed 固定=再現的 / SDXL seed=server wall-clock=非再現(画像再生成不可)",
        "conditioning_note": ("入力は英/日混在。日本語は encode_text_greedy で空トークン化→"
                              "prompt=[BOS,SEP] の empty conditioning、英語は tag 語で条件付け。"
                              "候補多様性は温度サンプリング由来 (eval_diverse も同条件)。"),
    }
    with open(args.summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def run(args, inputs, sel_report):
    import numpy as np  # noqa: F401
    import openvino as ov

    import dollma_gen_scorer_corpus as gc

    # --- resume: 完了 post_id を集め、未完のみ本チャンクで処理 (models ロード前に判定) ---
    completed = _completed_post_ids(args.sft_out)
    work_all = [(oi, row) for oi, row in enumerate(inputs)
                if row["post_id"] not in completed]
    remaining_before = len(work_all)
    base_done = len(completed)
    print(f"[g1] resume: 完了 {base_done}/{len(inputs)} / 未完 {remaining_before} "
          f"(limit={args.limit})", file=sys.stderr)
    if args.limit == 0:
        # 乾式 (--limit 0): resume 認識のみ・生成しない。done を正しくスキップ対象化。
        print(f"[g1] DRY(--limit 0): done={base_done} を resume 認識・{base_done} 件スキップ・"
              f"生成なしで終了", file=sys.stderr)
        summary = _write_summary(args, sel_report, 0, 0.0, 0, len(inputs), partial=True)
        print("[g1] SUMMARY " + json.dumps(summary, ensure_ascii=False), file=sys.stderr)
        return summary
    work = work_all if args.limit < 0 else work_all[:args.limit]
    if not work:
        print("[g1] 本チャンクで処理する未完入力なし → summary 更新のみで終了", file=sys.stderr)
        summary = _write_summary(args, sel_report, 0, 0.0, 0, len(inputs), partial=False)
        print("[g1] SUMMARY " + json.dumps(summary, ensure_ascii=False), file=sys.stderr)
        return summary
    print(f"[g1] 本チャンク: {work[0][0]}..{work[-1][0]} (orig index) の {len(work)} 件を生成",
          file=sys.stderr)

    model, tok, tb, torch = build_lm(args)
    print(f"[g1] LM ロード完了 device={args.lm_device} vocab={tok.vocab_size()} "
          f"weights={args.bitnet_weights}", file=sys.stderr)

    tok_dll = gc.resolve_tokenizers_dll(os.environ.get("DOLLAMA_OV_TOKENIZERS_DLL"))
    core = ov.Core()
    scorer = core.compile_model(core.read_model(SCORER_IR), args.scorer_device)

    import open_clip
    _cm, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14-quickgelu", pretrained="openai")
    del _cm
    clip_ov = core.compile_model(core.read_model(args.clip_image_ir), args.clip_device)
    qmlp_ov = core.compile_model(core.read_model(args.quality_mlp_ir), args.quality_device)
    print(f"[g1] scorer/quality 経路ロード完了 "
          f"scorer={args.scorer_device} clip={args.clip_device} qmlp={args.quality_device}",
          file=sys.stderr)

    # --- サーバ起動 (本 txt2img 有効・matting OFF) ---
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
        print(f"[g1] サーバ起動: {' '.join(cmd)}", file=sys.stderr)
        proc = gc.subprocess.Popen(cmd, cwd=ROOT, env=env)
        gc.wait_for_health(base_url, timeout=args.health_timeout, proc=proc)
        print("[g1] サーバ ready", file=sys.stderr)

    img_dir = args.img_dir
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.cand_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.sft_out) or ".", exist_ok=True)

    winner_rewards = []
    spread_bw = []            # per-input best - worst
    dup_rates = []            # per-input (n - unique)/n
    empty_prompt_cands = 0
    n_gen = 0
    t0 = time.time()

    sft_rows = []
    processed = 0
    try:
        # resume: append モード (既存 101 件を潰さない・冪等追記)。
        fcand = open(args.cand_out, "a", encoding="utf-8")
        fsft = open(args.sft_out, "a", encoding="utf-8")
        for i, row in work:                     # i = 400 中の原インデックス (seed 再現保持)
            input_text = row["text"]
            post_id = row["post_id"]
            cands = sample_candidates(model, tok, tb, torch, input_text, args.n,
                                      args.seed, i, args.temperature, args.top_k,
                                      args.lm_device, args.max_gen_len)
            uniq = len({c["prompt"] for c in cands})
            dup_rates.append((args.n - uniq) / args.n)

            scored = []
            for j, c in enumerate(cands):
                if not c["prompt"].strip():
                    empty_prompt_cands += 1
                    # 空 prompt は最悪報酬扱い (勝者にならない)。記録は残す。
                    rec = {"input_text": input_text, "post_id": post_id,
                           "candidate": j, "lm_seed": c["lm_seed"],
                           "prompt": c["prompt"], "tags": c["tags"],
                           "axes": None, "quality": None, "reward": None,
                           "empty": True, "is_winner": False}
                    scored.append((float("-inf"), rec, None))
                    continue
                body = gc.build_request_body({"prompt": c["prompt"]}, args.steps, "1024x1024")
                b64 = gc.post_generation(base_url, body, timeout=args.gen_timeout)
                img_path = os.path.join(img_dir, f"{i:04d}_{j}.png")
                gc.save_b64_png(b64, img_path)
                n_gen += 1

                x = cr.preprocess_for_scorer(img_path)
                logits = scorer(x)[scorer.output(0)][0]
                axes = cr.axes_from_logits(logits)
                quality = cr.compute_quality_clip(img_path, clip_preprocess, clip_ov, qmlp_ov)
                reward = reward_from_scorer(axes, quality)
                rec = {"input_text": input_text, "post_id": post_id,
                       "candidate": j, "lm_seed": c["lm_seed"],
                       "prompt": c["prompt"], "tags": c["tags"],
                       "axes": [float(a) for a in axes], "quality": float(quality),
                       "reward": float(reward), "empty": False, "is_winner": False}
                scored.append((reward, rec, img_path))

                if not args.keep_all_images:
                    # 勝者確定前だが losers を後で消すため、いったん残す。
                    pass

            # --- top-1 of N 選抜 ---
            scored.sort(key=lambda t: t[0], reverse=True)
            best_r, best_rec, best_img = scored[0]
            valid = [s[0] for s in scored if s[0] != float("-inf")]
            best_rec["is_winner"] = True
            winner_rewards.append(best_r)
            if len(valid) >= 2:
                spread_bw.append(max(valid) - min(valid))

            # 全候補を candidate log に書く (winner フラグ付き)。
            for _, rec, _img in scored:
                fcand.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fcand.flush()

            # SFT 行 (train_bitnet synthetic 経路が読める schema)。
            sft_row = {
                "text": input_text,
                "tags": best_rec["tags"],
                "lang": row.get("lang", "ja"),
                "source": "rejection_sft",
                "meta": {
                    "post_id": post_id,
                    "reward": round(float(best_r), 6),
                    "n_candidates": args.n,
                    "winner_candidate": best_rec["candidate"],
                    "winner_lm_seed": best_rec["lm_seed"],
                    "temperature": args.temperature,
                    "top_k": args.top_k,
                    "quality_weight": QUALITY_WEIGHT,
                    "gen": "bestofn_rollout",
                },
            }
            sft_rows.append(sft_row)
            fsft.write(json.dumps(sft_row, ensure_ascii=False) + chr(10))
            fsft.flush()
            processed += 1
            if processed % 20 == 0:
                _write_summary(args, sel_report, n_gen, time.time() - t0,
                               empty_prompt_cands, len(inputs), partial=True)

            # 画像掃除 (winner 以外を削除・disk 節約。winner も既定は消す=scores 保持で足りる)。
            if not args.keep_all_images:
                for _, rec, img in scored:
                    if img is None:
                        continue
                    if args.keep_winner_images and rec.get("is_winner"):
                        continue
                    try:
                        os.remove(img)
                    except OSError:
                        pass

            done_total = base_done + processed
            if processed <= 5 or processed % 5 == 0:
                el = time.time() - t0
                rate = el / processed
                eta_all = rate * max(0, remaining_before - processed)
                print(f"[g1] done {done_total}/{len(inputs)} (chunk {processed}/{len(work)}) "
                      f"winner_reward={best_r:+.4f} spread={spread_bw[-1] if spread_bw else 0:.4f} "
                      f"uniq={uniq}/{args.n} gen={n_gen} elapsed={el/60:.1f}m "
                      f"eta_remaining={eta_all/60:.1f}m pid={post_id} "
                      f"prompt='{best_rec['prompt'][:40]}'", file=sys.stderr)
        fcand.close()
        fsft.close()
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except Exception:
                proc.kill()

    chunk_elapsed = time.time() - t0
    done_now = len(_completed_post_ids(args.sft_out))
    is_final = (done_now >= len(inputs))
    summary = _write_summary(args, sel_report, n_gen, chunk_elapsed,
                             empty_prompt_cands, len(inputs), partial=(not is_final))
    print("[g1] SUMMARY " + json.dumps(summary, ensure_ascii=False), file=sys.stderr)
    tail = ("全 400 完走 (final)" if is_final
            else f"チャンク完了 done={done_now}/{len(inputs)} (次回 --run --limit N で resume)")
    print(f"[g1] {tail}: SFT 累計 {done_now} ペア → {args.sft_out} / summary → {args.summary_out}")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="F-0b G-1 best-of-N rollout 収集 (研究機・GPU)。未 --run は計画表示のみ。")
    ap.add_argument("--m", type=int, default=400, help="入力数 M")
    ap.add_argument("--n", type=int, default=8, help="入力あたり候補数 N")
    ap.add_argument("--limit", type=int, default=50,
                    help="本チャンクで処理する未完入力の上限 (>0=N件で resume 終了 / "
                         "0=乾式(認識のみ生成なし) / <0=残り全部)")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-k", dest="top_k", type=int, default=50)
    ap.add_argument("--max-gen-len", dest="max_gen_len", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20260704, help="入力選定 + LM sampling 基底 seed")
    ap.add_argument("--corpus-glob", dest="corpus_glob", default=INPUT_CORPUS_GLOB)
    ap.add_argument("--sft-out", dest="sft_out",
                    default=os.path.join(ROOT, "data", "rollouts", "sft_bestofn.jsonl"))
    ap.add_argument("--cand-out", dest="cand_out",
                    default=os.path.join(ROOT, "data", "rollouts", "candidates_bestofn.jsonl"))
    ap.add_argument("--summary-out", dest="summary_out",
                    default=os.path.join(ROOT, "data", "rollouts", "summary_bestofn.json"))
    ap.add_argument("--img-dir", dest="img_dir",
                    default=os.path.join(ROOT, "data", "rollouts", "img_bestofn"))
    ap.add_argument("--keep-all-images", dest="keep_all_images", action="store_true",
                    help="全候補画像を残す (既定は winner のみ残し losers 削除)")
    ap.add_argument("--no-keep-winner-images", dest="keep_winner_images",
                    action="store_false", help="winner 画像も削除 (scores のみ保持)")
    ap.set_defaults(keep_winner_images=True)
    ap.add_argument("--bitnet-weights", dest="bitnet_weights", default=BITNET_WEIGHTS)
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

    # 入力選定 + disjoint 検証は --run 有無に関わらず実施 (安全弁の事前確認)。
    inputs, sel_report = select_inputs(args.corpus_glob, EVAL_LEAK_FILES, args.m, args.seed)
    print("[g1] INPUT_SELECTION " + json.dumps(sel_report, ensure_ascii=False))
    if not sel_report["disjoint_ok"]:
        raise SystemExit(f"[g1] disjoint 検証失敗: eval と {sel_report['selected_overlap_with_eval']} 件重複")

    if not args.run:
        ok, reasons = cr._research_assets_status(args)
        print(f"[PLAN] best-of-N rollout: M={args.m} N={args.n} = {args.m * args.n} 枚生成")
        print(f"[PLAN] temperature={args.temperature} top_k={args.top_k} seed={args.seed}")
        print(f"[PLAN] SFT out={args.sft_out}")
        print(f"[PLAN] 実走資産: {'揃っている' if ok else '不足'}"
              + ("" if ok else f" ({'; '.join(reasons)})"))
        print("[PLAN] 研究機で実走するには --run を付ける。")
        for r in inputs[:3]:
            print(f"  例 input(pid={r['post_id']}): {r['text'][:48]}")
        return

    ok, reasons = cr._research_assets_status(args)
    if not ok:
        print(f"[SKIP] 実走資産が揃っていない: {'; '.join(reasons)}")
        return
    run(args, inputs, sel_report)


if __name__ == "__main__":
    main()

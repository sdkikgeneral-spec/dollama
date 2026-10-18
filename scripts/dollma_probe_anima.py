# -*- coding: utf-8 -*-
"""dollma_probe_anima.py — E-4: Anima (CircleStone Labs, Cosmos-Predict2 2B ベース・非 SDXL) 下見 probe。

用途 (CLAUDE.md の Python 許容 3 用途のうち「新規 checkpoint の下見」):
  E-1 画質評価ハーネス (scripts/dollma_eval_image_grid.py) と同じ物差し
  (prompt 条件 p1〜p4 / seed 1000〜1003 / 1024² / WD14 再現率 + ScorerNet anatomy 8 軸) で
  Anima-Base v1.0 / Anima-Aesthetic v1.1 を採点し、E-1 と同形式の
  grid_results.csv / grid_summary.csv / contact_sheets/ を出力する。
  採点関数はハーネスから import して再利用 (再実装しない)。

実行系:
  community 実装 diffusers_anima (hdae/diffusers-anima の AnimaPipeline) を専用 venv
  (models/anima/.venv・--system-site-packages) に -e で入れて使う。ComfyUI 本体は使わない。
  diffusers_anima は重みを固定の HF repo (hdae/diffusers-anima-preview) から取る実装なので、
  公式 split_files (v1.0/v1.1 の UNet 相当・Qwen3-0.6B・Qwen-Image VAE) をロードするため
  loading モジュールの固定ソース定数をローカルパスへ差し替える (外部通信なし)。
  tokenizer だけは models/anima/tokenizers_hdae/ (hdae repo から tokenizer 3 ファイルのみ取得済み) を使う。
  公式推奨 sampler の er_sde は diffusers_anima に無いので、ComfyUI の
  comfy/k_diffusion/sampling.py::sample_er_sde (CONST=rectified flow 分岐) を忠実に移植して
  run_const_sigma_samplers を差し替える (--sampler er_sde)。--sampler euler / euler_a_rf は
  diffusers_anima 実装そのまま。

使い方 (研究機・venv の python で):
  models/anima/.venv/Scripts/python.exe scripts/dollma_probe_anima.py --selftest
  models/anima/.venv/Scripts/python.exe scripts/dollma_probe_anima.py --run --limit 1      # 疎通 1 枚
  models/anima/.venv/Scripts/python.exe scripts/dollma_probe_anima.py --run                # 全 32 枚
  既存 PNG は再生成せず採点だけやり直す (resume): 同じコマンドをそのまま再実行。

規律 (E-1 と同じ):
  - N=1 で優劣を書かない。SDXL 系と Anima は seed の意味が異なる (別モデル・別ノイズ)。
  - 秒は characterization のみ (モデルロード後の warm 生成秒。E-1 は cold 別プロセスなので比較不可)。
  - 本 probe は Anima 重みの下見であり、重みは非商用ライセンス (CircleStone NC v1.2)。
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# 重み・OV IR は gitignore 済みで worktree に複製されないため、メイン checkout の models/ を探す。
def _find_models_dir():
    env = os.environ.get("DOLLAMA_MODELS_DIR")
    cands = [env] if env else []
    cands.append(os.path.join(ROOT, "models"))
    # .claude/worktrees/<name>/ → 3 つ上がメイン checkout
    cands.append(os.path.abspath(os.path.join(ROOT, "..", "..", "..", "models")))
    for c in cands:
        if c and os.path.isdir(os.path.join(c, "wd14-swinv2-tagger-v3")):
            return c
    return os.path.join(ROOT, "models")


MODELS = _find_models_dir()
ANIMA_DIR = os.path.join(MODELS, "anima")
SPLIT = os.path.join(ANIMA_DIR, "split_files")
TE_PATH = os.path.join(SPLIT, "text_encoders", "qwen_3_06b_base.safetensors")
VAE_PATH = os.path.join(SPLIT, "vae", "qwen_image_vae.safetensors")
TOKENIZER_DIR = os.path.join(ANIMA_DIR, "tokenizers_hdae")

WD14_XML = os.path.join(MODELS, "wd14-swinv2-tagger-v3", "model_ov.xml")
WD14_TAGS_CSV = os.path.join(MODELS, "wd14-swinv2-tagger-v3", "selected_tags.csv")
SCORER_IR = os.path.join(MODELS, "scorer-net", "model_ov_fp32.xml")

# 各モデルのカード推奨設定 (README.md "Prompting" / "Generation settings")。
CARD_NEGATIVE_BASE = ("worst quality, low quality, score_1, score_2, score_3, "
                      "artist name, blurry, jpeg artifacts, chromatic aberration")
# Aesthetic は score_* を positive/negative 双方で使わない (カード推奨)。quality 語も negative は最小限。
CARD_NEGATIVE_AESTHETIC = ("worst quality, low quality, "
                           "artist name, blurry, jpeg artifacts, chromatic aberration")

MODEL_SPECS = {
    "anima-base-v1.0": {
        "ckpt": os.path.join(SPLIT, "diffusion_models", "anima-base-v1.0.safetensors"),
        "prefix": "masterpiece, best quality, score_7, safe, ",
        "negative": CARD_NEGATIVE_BASE,
    },
    "anima-aesthetic-v1.1": {
        "ckpt": os.path.join(SPLIT, "diffusion_models", "anima-aesthetic-v1.1.safetensors"),
        "prefix": "masterpiece, best quality, safe, ",
        "negative": CARD_NEGATIVE_AESTHETIC,
    },
}


def build_negative(model_key, pid, prompts):
    """カード推奨 negative に E-1 の題材ガード語 (multiple views / multiple girls) を足す。

    E-1 の NEGATIVE_COMMON は両者を含み、p4 (複数人) だけ "multiple girls" を除く。
    その題材意図だけを引き継ぎ、品質語・score 語は各モデルのカード推奨に従う。
    """
    neg = MODEL_SPECS[model_key]["negative"] + ", multiple views"
    if pid != "p4":
        neg += ", multiple girls"
    return neg


def build_prompt(model_key, pid, prompts):
    return MODEL_SPECS[model_key]["prefix"] + prompts[pid]["prompt"]


# ============================================================
# ER-SDE (ComfyUI comfy/k_diffusion/sampling.py::sample_er_sde の CONST 分岐を移植)
# ============================================================
def er_sde_sample(denoise_fn, x, sigmas, noise_fn, s_noise=1.0, max_stage=3):
    """denoise_fn(x, sigma_scalar_tensor) -> denoised。sigmas は末尾 0 付き 1-D tensor。

    ComfyUI との対応: half_log_snr = -logit(sigma) (CONST) / er_lambda = exp(-half_log_snr)
    = sigma/(1-sigma) / alpha = 1 - sigma / 先頭 sigma>=1 のとき 1-1e-4 に置換 (percent_to_sigma(1e-4))。
    noise_scaler は ComfyUI 既定 x*(exp(x^0.3)+10)。
    """
    import torch

    sigmas = sigmas.clone().float()
    if sigmas[0] >= 1:
        sigmas[0] = 1.0 - 1e-4
    half_log_snrs = sigmas.logit().neg()
    er_lambdas = half_log_snrs.neg().exp()

    def noise_scaler(v):
        return v * ((v ** 0.3).exp() + 10.0)

    num_integration_points = 200.0
    point_indice = torch.arange(0, num_integration_points, dtype=torch.float32, device=x.device)

    old_denoised = None
    old_denoised_d = None
    n = len(sigmas) - 1
    for i in range(n):
        denoised = denoise_fn(x, sigmas[i])
        stage_used = min(max_stage, i + 1)
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            er_lambda_s, er_lambda_t = er_lambdas[i], er_lambdas[i + 1]
            alpha_s = sigmas[i] / er_lambda_s
            alpha_t = sigmas[i + 1] / er_lambda_t
            r_alpha = alpha_t / alpha_s
            r = noise_scaler(er_lambda_t) / noise_scaler(er_lambda_s)

            x = r_alpha * r * x + alpha_t * (1 - r) * denoised  # Stage 1 (Euler)

            if stage_used >= 2:
                dt = er_lambda_t - er_lambda_s
                lambda_step_size = -dt / num_integration_points
                lambda_pos = er_lambda_t + point_indice * lambda_step_size
                scaled_pos = noise_scaler(lambda_pos)

                s = torch.sum(1 / scaled_pos) * lambda_step_size  # Stage 2
                denoised_d = (denoised - old_denoised) / (er_lambda_s - er_lambdas[i - 1])
                x = x + alpha_t * (dt + s * noise_scaler(er_lambda_t)) * denoised_d

                if stage_used >= 3:  # Stage 3
                    s_u = torch.sum((lambda_pos - er_lambda_s) / scaled_pos) * lambda_step_size
                    denoised_u = (denoised_d - old_denoised_d) / ((er_lambda_s - er_lambdas[i - 2]) / 2)
                    x = x + alpha_t * ((dt ** 2) / 2 + s_u * noise_scaler(er_lambda_t)) * denoised_u
                old_denoised_d = denoised_d

            if s_noise > 0:
                x = x + alpha_t * noise_fn(x) * s_noise * (
                    er_lambda_t ** 2 - er_lambda_s ** 2 * r ** 2).sqrt().nan_to_num(nan=0.0)
        old_denoised = denoised
    return x


def install_er_sde_patch():
    """diffusers_anima の run_const_sigma_samplers を差し替え、sampler='euler' 指定時に ER-SDE を走らせる。

    scheduler.set_sampling_config(sampler="euler", sigma_schedule=...) の検証を通すため
    名前は 'euler' のまま使い、この関数を呼んだ場合のみ ER-SDE に置換する (環境変数不要)。
    """
    import torch
    from diffusers_anima.pipelines.anima import pipeline_anima as pa
    from diffusers_anima.pipelines.anima import sampling as sm

    orig = pa.run_const_sigma_samplers

    def patched(transformer, pipeline, latents, *, sigmas, sampler, pos_cond, neg_cond,
                guidance_scale, eta, s_noise, generator, cfg_batch_mode, model_dtype,
                callback_on_step_end, callback_on_step_end_tensor_inputs,
                input_is_noisy_latents=False, inpaint_mask=None, init_image_latents=None,
                init_noise=None):
        if sampler != "euler" or inpaint_mask is not None:
            return orig(transformer, pipeline, latents, sigmas=sigmas, sampler=sampler,
                        pos_cond=pos_cond, neg_cond=neg_cond, guidance_scale=guidance_scale,
                        eta=eta, s_noise=s_noise, generator=generator,
                        cfg_batch_mode=cfg_batch_mode, model_dtype=model_dtype,
                        callback_on_step_end=callback_on_step_end,
                        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                        input_is_noisy_latents=input_is_noisy_latents, inpaint_mask=inpaint_mask,
                        init_image_latents=init_image_latents, init_noise=init_noise)
        if not input_is_noisy_latents:
            latents = latents * sigmas[0].to(latents.dtype)
        # 初期ノイズは sigma0=1.0 なので x=noise そのまま (ComfyUI も noise_scaling で sigma=1 → noise)
        bar = pipeline.progress_bar(total=len(sigmas) - 1)

        def denoise_fn(x, sigma):
            bar.update(1)
            return sm._predict_denoised_const(
                transformer, x, sigma=sigma, pos_cond=pos_cond, neg_cond=neg_cond,
                guidance_scale=guidance_scale, cfg_batch_mode=cfg_batch_mode,
                model_dtype=model_dtype)

        def noise_fn(x):
            return sm.randn_like(x, generator=generator)

        return er_sde_sample(denoise_fn, latents.float(), sigmas.float(), noise_fn,
                             s_noise=float(s_noise)).to(latents.dtype)

    pa.run_const_sigma_samplers = patched


# ============================================================
# ローカル split_files ロード
# ============================================================
def load_anima_pipe(ckpt, device="cuda", dtype="bfloat16"):
    import warnings
    from diffusers_anima import AnimaPipeline
    from diffusers_anima.pipelines.anima import loading as L

    # 固定 HF ソースを公式 split_files のローカルパスへ差し替え (外部通信を発生させない)。
    L._TEXT_ENCODER_WEIGHTS = TE_PATH
    L._VAE_SOURCE = VAE_PATH
    L._QWEN_TOKENIZER_SOURCE = os.path.join(TOKENIZER_DIR, "prompt_tokenizer_qwen")
    L._T5_TOKENIZER_SOURCE = os.path.join(TOKENIZER_DIR, "prompt_tokenizer_t5")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return AnimaPipeline.from_single_file(ckpt, device=device, dtype=dtype,
                                              text_encoder_dtype=dtype, local_files_only=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def image_sane(path):
    """黒画/NaN 由来の一色画像を無効にする (log_ok 相当の検査)。"""
    import numpy as np
    from PIL import Image
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    return bool(a.std() > 2.0)


# ============================================================
# 実走
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="E-4 Anima probe (E-1 ハーネス物差し・研究機専用)")
    ap.add_argument("--models", nargs="+", default=list(MODEL_SPECS.keys()), choices=list(MODEL_SPECS.keys()))
    ap.add_argument("--prompts", nargs="+", default=["p1", "p2", "p3", "p4"])
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--seed-base", dest="seed_base", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=30, help="カード推奨 30-50")
    ap.add_argument("--cfg", type=float, default=4.0, help="カード推奨 4-5")
    ap.add_argument("--sampler", default="er_sde", choices=["er_sde", "euler", "euler_a_rf"])
    ap.add_argument("--sigma-schedule", dest="sigma_schedule", default="simple",
                    choices=["simple", "beta", "normal"])
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--limit", type=int, default=0, help="先頭 N 枚だけ (疎通用)。0=全部")
    ap.add_argument("--out-dir", dest="out_dir", default=os.path.join(ROOT, "docs", "logs", "e4-anima"))
    ap.add_argument("--wd14-thresh", dest="wd14_thresh", type=float, default=0.35)
    ap.add_argument("--wd14-device", dest="wd14_device", default="CPU")
    ap.add_argument("--scorer-device", dest="scorer_device", default="CPU")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    import dollma_eval_image_grid as G

    if args.selftest:
        sys.exit(0 if selftest(G) else 1)
    if not args.run:
        print("[anima] 計画のみ (--run で実走)")
        for m in args.models:
            for pid in args.prompts:
                print(f"  {m} {pid}: {build_prompt(m, pid, G.PROMPTS)!r}\n      neg={build_negative(m, pid, G.PROMPTS)!r}")
        print(f"  steps={args.steps} cfg={args.cfg} sampler={args.sampler}/{args.sigma_schedule} "
              f"seeds={args.seed_base}..{args.seed_base + args.seeds - 1} size={args.size}")
        return

    run_all(args, G)


def selftest(G):
    import torch
    fails = []

    def check(name, cond):
        print(f"[selftest] {'PASS' if cond else 'FAIL'} {name}")
        if not cond:
            fails.append(name)

    # ER-SDE: denoiser = 真値 x0 を返す理想モデルなら、ノイズ無しで最終 x が x0 に一致する。
    x0 = torch.randn(1, 4, 8, 8)
    noise = torch.randn_like(x0)
    sig = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
    out = er_sde_sample(lambda x, s: x0.clone(), noise.clone(), sig, lambda x: torch.zeros_like(x), s_noise=0.0)
    check("er_sde 理想 denoiser → x0 一致", torch.allclose(out, x0, atol=1e-5))
    # 確率ノイズ付きでも有限
    out2 = er_sde_sample(lambda x, s: x0.clone(), noise.clone(), sig, lambda x: torch.randn_like(x), s_noise=1.0)
    check("er_sde ノイズ付き finite", bool(torch.isfinite(out2).all()))
    # 先頭 sigma>=1 を 1-1e-4 に置換しても NaN にならない (logit(1)=inf 回避)
    check("er_sde 先頭 sigma=1.0 でも finite", bool(torch.isfinite(out).all()))
    # プロンプト組み立て
    p = build_prompt("anima-base-v1.0", "p1", G.PROMPTS)
    check("base prefix", p.startswith("masterpiece, best quality, score_7, safe, 1girl"))
    pa = build_prompt("anima-aesthetic-v1.1", "p1", G.PROMPTS)
    check("aesthetic に score_ 無し", "score_" not in pa and "score_" not in build_negative("anima-aesthetic-v1.1", "p1", G.PROMPTS))
    check("p4 negative に multiple girls 無し", "multiple girls" not in build_negative("anima-base-v1.0", "p4", G.PROMPTS))
    check("p1 negative に multiple girls 有り", "multiple girls" in build_negative("anima-base-v1.0", "p1", G.PROMPTS))
    print("[selftest] ALL PASS" if not fails else f"[selftest] {len(fails)} FAIL")
    return not fails


def run_all(args, G):
    import numpy as np  # noqa: F401
    import torch
    import openvino as ov
    from dollma_collect_rollouts import preprocess_for_scorer, axes_from_logits

    out_dir = args.out_dir
    img_dir = os.path.join(out_dir, "img")
    sheet_dir = os.path.join(out_dir, "contact_sheets")
    for d in (out_dir, img_dir, sheet_dir):
        os.makedirs(d, exist_ok=True)

    for p in (TE_PATH, VAE_PATH, WD14_XML, SCORER_IR, TOKENIZER_DIR):
        if not os.path.exists(p):
            print(f"[anima] [SKIP] 必要資産が無い: {p}")
            return
    for m in args.models:
        if not os.path.isfile(MODEL_SPECS[m]["ckpt"]):
            print(f"[anima] [SKIP] checkpoint が無い: {MODEL_SPECS[m]['ckpt']}")
            return

    # 走行前後で入力データ不動確認 (E-1 規律③): 重みの (size, mtime) を控える。sha256 は初回のみ保存。
    def stat_sig():
        paths = [MODEL_SPECS[m]["ckpt"] for m in args.models] + [TE_PATH, VAE_PATH]
        return {p: (os.path.getsize(p), int(os.path.getmtime(p))) for p in paths}

    sig_before = stat_sig()
    sha_path = os.path.join(out_dir, "weights_sha256.json")
    if not os.path.isfile(sha_path):
        shas = {os.path.basename(p): sha256_file(p) for p in sig_before}
        with open(sha_path, "w", encoding="utf-8") as f:
            json.dump(shas, f, indent=1)

    if args.sampler == "er_sde":
        install_er_sde_patch()
        pipe_sampler = "euler"
    else:
        pipe_sampler = args.sampler

    seeds = list(range(args.seed_base, args.seed_base + args.seeds))
    grid = [(m, pid, s) for m in args.models for pid in args.prompts for s in seeds]
    if args.limit:
        grid = grid[:args.limit]
    print(f"[anima] grid: {len(grid)} 枚 sampler={args.sampler}/{args.sigma_schedule} "
          f"steps={args.steps} cfg={args.cfg} size={args.size}")

    # ---- 生成 (モデルごとに 1 回ロード・resume: 既存 PNG+meta はスキップ) ----
    metas = {}
    for m in args.models:
        todo = [(mm, pid, s) for (mm, pid, s) in grid if mm == m]
        if not todo:
            continue

        def tag_of(pid, s):
            return f"{m}__{pid}__seed{s}__cfg{args.cfg:g}"

        need = [(pid, s) for (_, pid, s) in todo
                if not (os.path.isfile(os.path.join(img_dir, tag_of(pid, s) + ".png"))
                        and os.path.isfile(os.path.join(img_dir, tag_of(pid, s) + ".json")))]
        pipe = None
        if need:
            t0 = time.time()
            pipe = load_anima_pipe(MODEL_SPECS[m]["ckpt"], dtype=args.dtype)
            pipe.scheduler.set_sampling_config(sampler=pipe_sampler, sigma_schedule=args.sigma_schedule)
            pipe.set_progress_bar_config(disable=True)
            print(f"[anima] {m} ロード {time.time() - t0:.1f}s (vram alloc "
                  f"{torch.cuda.memory_allocated() / 2**20:.0f}MB)")
        for (_, pid, s) in todo:
            tag = tag_of(pid, s)
            png = os.path.join(img_dir, tag + ".png")
            meta_path = os.path.join(img_dir, tag + ".json")
            if (pid, s) in need:
                prompt = build_prompt(m, pid, G.PROMPTS)
                neg = build_negative(m, pid, G.PROMPTS)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
                gen = torch.Generator(device="cpu").manual_seed(s)
                res = pipe(prompt, negative_prompt=neg, width=args.size, height=args.size,
                           num_inference_steps=args.steps, guidance_scale=args.cfg, generator=gen)
                torch.cuda.synchronize()
                sec = time.time() - t0
                res.images[0].save(png)
                meta = {"sec": round(sec, 2), "vram_peak_mb": round(torch.cuda.max_memory_allocated() / 2**20),
                        "prompt": prompt, "negative": neg, "sampler": args.sampler,
                        "sigma_schedule": args.sigma_schedule, "steps": args.steps, "cfg": args.cfg,
                        "dtype": args.dtype}
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=1)
                print(f"[anima] {tag} {sec:.1f}s vram_peak={meta['vram_peak_mb']}MB")
            with open(meta_path, "r", encoding="utf-8") as f:
                metas[tag] = json.load(f)
        if pipe is not None:
            del pipe
            torch.cuda.empty_cache()

    # ---- 採点 (E-1 ハーネスの関数を再利用) ----
    core = ov.Core()
    wd14_names, wd14_cats = G.load_wd14_tags(WD14_TAGS_CSV)
    vocab_names = set(n for n, c in zip(wd14_names, wd14_cats) if c != 9)
    wd14_model = core.compile_model(core.read_model(WD14_XML), args.wd14_device)
    scorer_model = core.compile_model(core.read_model(SCORER_IR), args.scorer_device)

    rows = []
    for (m, pid, s) in grid:
        tag = f"{m}__{pid}__seed{s}__cfg{args.cfg:g}"
        png = os.path.join(img_dir, tag + ".png")
        cond = G.PROMPTS[pid]
        ok = image_sane(png)
        row = {"preset": m, "prompt_id": pid, "seed": s, "cfg": args.cfg,
               "sec": metas[tag]["sec"], "exit": 0, "log_ok": ok,
               "log_reason": "" if ok else "一色画像 (黒画/NaN 疑い)",
               "png": os.path.relpath(png, ROOT)}
        if not ok:
            row.update({k: None for k in ["recall_matched", "recall_total", "recall", "soft_recall",
                                          "soft_recall_iv", "soft_recall_logit", "n_oov",
                                          "argmax_axis", "worst_anatomy"]})
            row["oov_words"] = ""
            row.update({f"axis_{a}": None for a in G.AXIS_NAMES})
            rows.append(row)
            continue
        x = G.preprocess_wd14(png)
        out = wd14_model(x)[wd14_model.output(0)][0]
        hit = [wd14_names[j] for j in range(len(out))
               if wd14_cats[j] != 9 and float(out[j]) >= args.wd14_thresh]
        matched, total, recall = G.compute_recall(cond["words"], hit)
        name_to_prob = {wd14_names[j]: float(out[j]) for j in range(len(out)) if wd14_cats[j] != 9}
        soft = G.soft_recall_metrics(cond["words"], name_to_prob, vocab_names)
        sx = preprocess_for_scorer(png)
        logits = scorer_model(sx)[scorer_model.output(0)][0]
        axes = axes_from_logits(logits)
        worst = max(axes)

        def r4(v):
            return round(v, 4) if v is not None else None

        row.update({
            "recall_matched": matched, "recall_total": total, "recall": round(recall, 4),
            "soft_recall": r4(soft["soft_recall"]), "soft_recall_iv": r4(soft["soft_recall_iv"]),
            "soft_recall_logit": r4(soft["soft_recall_logit"]),
            "n_oov": soft["n_oov"], "oov_words": soft["oov_words"],
            **{f"axis_{a}": round(axes[k], 4) for k, a in enumerate(G.AXIS_NAMES)},
            "argmax_axis": G.AXIS_NAMES[axes.index(worst)], "worst_anatomy": round(worst, 4),
        })
        rows.append(row)
        print(f"[anima] 採点 {tag} recall={recall:.2f} soft={row['soft_recall']} "
              f"worst={worst:.4f}({row['argmax_axis']})")

    fieldnames = (["preset", "prompt_id", "seed", "cfg", "sec", "exit", "log_ok", "log_reason",
                   "png", "recall_matched", "recall_total", "recall",
                   "soft_recall", "soft_recall_iv", "soft_recall_logit", "n_oov", "oov_words"]
                  + [f"axis_{a}" for a in G.AXIS_NAMES] + ["argmax_axis", "worst_anatomy"])
    csv_path = os.path.join(out_dir, "grid_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    agg_fields = ["preset", "prompt_id", "cfg", "n", "n_valid", "recall_mean", "recall_std",
                  "soft_recall_mean", "soft_recall_std", "soft_recall_iv_mean", "soft_recall_iv_std",
                  "soft_recall_logit_mean", "soft_recall_logit_std",
                  "worst_anatomy_mean", "worst_anatomy_std",
                  "sec_mean_characterization_only", "sec_std_characterization_only"]
    agg_path = os.path.join(out_dir, "grid_summary.csv")
    with open(agg_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=agg_fields)
        w.writeheader()
        for m in args.models:
            for pid in args.prompts:
                sub = [r for r in rows if r["preset"] == m and r["prompt_id"] == pid]
                if not sub:
                    continue
                valid = [r for r in sub if r["recall"] is not None]

                def ms(key, src=valid):
                    vals = [r[key] for r in src if r[key] is not None]
                    mean, std = G.mean_std(vals)
                    return (round(mean, 4) if mean is not None else None,
                            round(std, 4) if std is not None else None)

                rm, rs = ms("recall")
                sm_, ss = ms("soft_recall")
                im, is_ = ms("soft_recall_iv")
                lm, ls = ms("soft_recall_logit")
                wm, ws = ms("worst_anatomy")
                tm, ts = G.mean_std([r["sec"] for r in sub])
                w.writerow({"preset": m, "prompt_id": pid, "cfg": args.cfg, "n": len(sub),
                            "n_valid": len(valid), "recall_mean": rm, "recall_std": rs,
                            "soft_recall_mean": sm_, "soft_recall_std": ss,
                            "soft_recall_iv_mean": im, "soft_recall_iv_std": is_,
                            "soft_recall_logit_mean": lm, "soft_recall_logit_std": ls,
                            "worst_anatomy_mean": wm, "worst_anatomy_std": ws,
                            "sec_mean_characterization_only": round(tm, 2) if tm is not None else None,
                            "sec_std_characterization_only": round(ts, 2) if ts is not None else None})

    # コンタクトシート (E-1 と同形式: 256px サムネ・seed 順・上に seed ラベル)
    from PIL import Image, ImageDraw
    for m in args.models:
        for pid in args.prompts:
            sub = [r for r in rows if r["preset"] == m and r["prompt_id"] == pid
                   and os.path.isfile(os.path.join(ROOT, r["png"]))]
            if not sub:
                continue
            thumb = 256
            sheet = Image.new("RGB", (thumb * len(sub), thumb + 24), (32, 32, 32))
            draw = ImageDraw.Draw(sheet)
            for k, r in enumerate(sub):
                im = Image.open(os.path.join(ROOT, r["png"])).convert("RGB").resize((thumb, thumb))
                sheet.paste(im, (k * thumb, 24))
                draw.text((k * thumb + 4, 4), f"seed={r['seed']}", fill=(255, 255, 255))
            sheet.save(os.path.join(sheet_dir, f"{m}__{pid}__cfg{args.cfg:g}.png"))

    sig_after = stat_sig()
    integrity_ok = sig_before == sig_after
    run_meta = {
        "sampler": args.sampler, "sigma_schedule": args.sigma_schedule, "steps": args.steps,
        "cfg": args.cfg, "dtype": args.dtype, "size": args.size, "seeds": seeds,
        "n_images": len(rows), "n_invalid": sum(1 for r in rows if not r["log_ok"]),
        "integrity_ok": integrity_ok, "torch": torch.__version__,
        "vram_peak_mb_max": max((v["vram_peak_mb"] for v in metas.values()), default=None),
        "executor": "diffusers_anima (hdae/diffusers-anima) local split_files",
    }
    with open(os.path.join(out_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=1)
    print(f"[anima] 完了: {len(rows)} 枚 (無効 {run_meta['n_invalid']}) integrity_ok={integrity_ok} → {out_dir}")


if __name__ == "__main__":
    main()

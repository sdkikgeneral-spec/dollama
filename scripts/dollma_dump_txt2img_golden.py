#!/usr/bin/env python3
# SDXL txt2img ゴールデンデータ生成 (task 2-6b Stage A)。
#
# 目的:
#   現 unet_io.safetensors の埋め込みは torch.randn の乱数 (UNet 配線検証用)。
#   2-6b では prompt -> embeds 経路 (CLIP-L 768 ++ bigG 1280 concat / bigG pooled)
#   と CFG E2E を検証するため、固定プロンプトの「実テキスト由来」エンコーダ出力と、
#   それを使って CFG で生成した最終画像を golden 化する。
#
# レシピ厳守 (diffusers StableDiffusionXLPipeline.encode_prompt と完全一致):
#   - tokenizers/text_encoders = [CLIP-L, CLIP-bigG] の順で回す
#   - 各エンコーダを output_hidden_states=True で呼ぶ
#   - clip_skip=None なので penultimate = hidden_states[-2] (= final_layer_norm 前)
#   - pooled は最後のエンコーダ (bigG, text_encoder_2) の prompt_embeds[0] (ndim==2)
#   - concat 順: list に L -> G の順で append し torch.concat(dim=-1)
#     => encoder_hidden_states[...,:768]=CLIP-L, [...,768:2048]=bigG
#   - time_ids = original_size + crops_coords_top_left + target_size
#                = (1024,1024,0,0,1024,1024)
#
# dtype 方針 (dump_unet_golden.py のルールに倣う):
#   - エンコーダ forward は fp32 (決定論) で行い f32 を必ず保存
#   - |x|<=65504 のときのみ f16 併設 (FP16 範囲外は f16 省略)
#   - 最終画像生成は probe10 同様 fp16 パイプライン (RTX5080) で実施
#
# 出力先: src/tests/data/txt2img_io.safetensors
#   保存テンソル:
#     cond_encoder_hidden_states_f32 / _f16   [1,77,2048]
#     cond_text_embeds_f32 (/_f16)            [1,1280]   (bigG pooled)
#     cond_token_ids_l                        [1,77] int64  (CLIP-L トークン)
#     cond_token_ids_g                        [1,77] int64  (bigG トークン)
#     uncond_*  上と同様 (negative prompt 由来)
#     time_ids_f32 (/_f16)                    [1,6]
#     image_golden_u8                         [1024,1024,3] uint8 RGB (CFG 最終画像)
#     image_golden_f32                        [1,3,1024,1024] float [-1,1] (decode 直後相当)
#   metadata: prompt / negative / seed / guidance_scale / steps / width / height /
#             penultimate_index="hidden_states[-2]" / concat_order="L(768)++G(1280)" 等

import os
from collections import OrderedDict

import torch
from diffusers import StableDiffusionXLPipeline
from safetensors.torch import save_file

# ====== 固定パラメータ (Stage B / C++ と完全一致させる値) ======
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
PROMPT = "1girl, solo, long hair, looking at viewer"
NEGATIVE = "lowres, bad anatomy, worst quality"
SEED = 1234
GUIDANCE_SCALE = 7.5
STEPS = 20
WIDTH = 1024
HEIGHT = 1024

OUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src", "tests", "data")
)
IO_PATH = os.path.join(OUT_DIR, "txt2img_io.safetensors")

FP16_MAX = 65504.0


def stats(t: torch.Tensor):
    tf = t.detach().to(torch.float32)
    return (
        float(tf.min()),
        float(tf.max()),
        float(tf.mean()),
        bool(torch.isnan(tf).any()),
        bool(torch.isinf(tf).any()),
    )


def tokenize(tokenizer, text):
    """diffusers encode_prompt と同じトークナイズ (max_length=77, pad, truncate)。"""
    ti = tokenizer(
        text,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    return ti.input_ids  # [1,77] int64


def encode_one(text, tokenizers, text_encoders, device):
    """
    StableDiffusionXLPipeline.encode_prompt の cond 経路を fp32 で忠実に再現。
    返り値: (prompt_embeds [1,77,2048] fp32,
             pooled [1,1280] fp32,
             token_ids_l [1,77], token_ids_g [1,77])
    """
    embeds_list = []
    pooled = None
    token_ids = []
    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        input_ids = tokenize(tokenizer, text)
        token_ids.append(input_ids.clone())
        out = text_encoder(input_ids.to(device), output_hidden_states=True)
        # pooled は prompt_embeds[0].ndim == 2 のエンコーダ (= bigG/text_encoder_2)
        if out[0].ndim == 2:
            pooled = out[0]
        # clip_skip=None => penultimate = hidden_states[-2] (final_layer_norm 前)
        penultimate = out.hidden_states[-2]
        embeds_list.append(penultimate)
    prompt_embeds = torch.concat(embeds_list, dim=-1)  # L(768) ++ G(1280) = 2048
    return (
        prompt_embeds.detach().to(torch.float32).cpu().contiguous(),
        pooled.detach().to(torch.float32).cpu().contiguous(),
        token_ids[0].cpu().contiguous(),
        token_ids[1].cpu().contiguous(),
    )


def add_f32_f16(store, name, t, skipped):
    vc = t.to(torch.float32).clone().contiguous()
    store[f"{name}_f32"] = vc
    if float(vc.abs().max()) <= FP16_MAX:
        store[f"{name}_f16"] = vc.to(torch.float16).clone().contiguous()
    else:
        skipped.append(name)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.set_grad_enabled(False)

    print("=" * 70)
    print("SDXL txt2img ゴールデンデータ生成 (task 2-6b Stage A)")
    print("=" * 70)
    print(f"model           : {MODEL_ID}")
    print(f"prompt          : {PROMPT!r}")
    print(f"negative        : {NEGATIVE!r}")
    print(f"seed            : {SEED}")
    print(f"guidance_scale  : {GUIDANCE_SCALE}")
    print(f"steps           : {STEPS}")
    print(f"size            : {WIDTH}x{HEIGHT}")

    assert torch.cuda.is_available(), "CUDA 必須 (RTX5080)"
    props = torch.cuda.get_device_properties(0)
    print(f"GPU             : {props.name} sm_{props.major}{props.minor}  CUDA {torch.version.cuda}")

    # ---- パイプライン (fp16, RTX5080) ----
    print("\nSDXL パイプライン ロード中...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    ).to("cuda")

    tokenizers = [pipe.tokenizer, pipe.tokenizer_2]
    text_encoders = [pipe.text_encoder, pipe.text_encoder_2]

    # ========================================================
    # 1. テキストエンコード (fp32 決定論)
    #    encode_prompt と同一の取り出し方を自前で再現する。
    #    fp32 forward のため一時的にエンコーダを fp32 へ。
    # ========================================================
    print("\nテキストエンコード (fp32 決定論)...")
    enc_dtypes = [te.dtype for te in text_encoders]
    for te in text_encoders:
        te.to(torch.float32)

    cond_ehs, cond_pooled, cond_ids_l, cond_ids_g = encode_one(
        PROMPT, tokenizers, text_encoders, "cuda"
    )
    uncond_ehs, uncond_pooled, uncond_ids_l, uncond_ids_g = encode_one(
        NEGATIVE, tokenizers, text_encoders, "cuda"
    )

    # エンコーダ dtype を fp16 へ戻す (最終画像生成のため)
    for te, dt in zip(text_encoders, enc_dtypes):
        te.to(dt)

    # ---- 健全性: pipe.encode_prompt と突合 (同一値か確認) ----
    pe, npe, ppe, nppe = pipe.encode_prompt(
        prompt=PROMPT,
        prompt_2=None,
        negative_prompt=NEGATIVE,
        negative_prompt_2=None,
        device=torch.device("cuda"),
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
    )
    d_ehs = float((pe.float().cpu() - cond_ehs).abs().max())
    d_pool = float((ppe.float().cpu() - cond_pooled).abs().max())
    d_nehs = float((npe.float().cpu() - uncond_ehs).abs().max())
    print(f"  自前 fp32 vs pipe.encode_prompt(fp16) 最大差: "
          f"cond_ehs={d_ehs:.4f} cond_pool={d_pool:.4f} uncond_ehs={d_nehs:.4f} "
          f"(fp16 量子化差程度なら OK)")

    # ---- time_ids = original_size + crops_coords_top_left + target_size ----
    time_ids = torch.tensor(
        [[float(HEIGHT), float(WIDTH), 0.0, 0.0, float(HEIGHT), float(WIDTH)]],
        dtype=torch.float32,
    )

    # ========================================================
    # 2. CFG 最終画像生成 (probe10 と同じ fp16 パイプライン)
    # ========================================================
    print("\nCFG 最終画像生成中 (fp16, RTX5080)...")
    gen = torch.Generator("cuda").manual_seed(SEED)
    result = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE_SCALE,
        width=WIDTH,
        height=HEIGHT,
        original_size=(HEIGHT, WIDTH),
        crops_coords_top_left=(0, 0),
        target_size=(HEIGHT, WIDTH),
        generator=gen,
        output_type="pt",  # [1,3,H,W] float [0,1]
    )
    img = result.images[0]  # [3,H,W] float [0,1] on cuda
    img_f01 = img.detach().to(torch.float32).cpu().contiguous()  # [3,H,W]
    # u8 RGB [H,W,3]
    img_u8 = (img_f01.clamp(0, 1) * 255.0).round().to(torch.uint8)
    img_u8 = img_u8.permute(1, 2, 0).contiguous()  # [H,W,3]
    # f32 [-1,1] [1,3,H,W] (VAE decode 直後 = x のスケール。E/F の decode 出力突合用)
    img_f_m11 = (img_f01 * 2.0 - 1.0).unsqueeze(0).contiguous()  # [1,3,H,W]

    # PNG も併せて保存 (目視確認用)
    png_path = os.path.join(OUT_DIR, "txt2img_golden.png")
    try:
        from PIL import Image
        Image.fromarray(img_u8.numpy()).save(png_path)
    except Exception as e:
        png_path = None
        print(f"  (PNG 保存スキップ: {e})")

    # ========================================================
    # 3. 保存
    # ========================================================
    store = OrderedDict()
    skipped_f16 = []

    add_f32_f16(store, "cond_encoder_hidden_states", cond_ehs, skipped_f16)
    add_f32_f16(store, "cond_text_embeds", cond_pooled, skipped_f16)
    add_f32_f16(store, "uncond_encoder_hidden_states", uncond_ehs, skipped_f16)
    add_f32_f16(store, "uncond_text_embeds", uncond_pooled, skipped_f16)
    add_f32_f16(store, "time_ids", time_ids, skipped_f16)

    # token ids は整数 (突合用) — int64 のまま保存
    store["cond_token_ids_l"] = cond_ids_l.to(torch.int64).contiguous()
    store["cond_token_ids_g"] = cond_ids_g.to(torch.int64).contiguous()
    store["uncond_token_ids_l"] = uncond_ids_l.to(torch.int64).contiguous()
    store["uncond_token_ids_g"] = uncond_ids_g.to(torch.int64).contiguous()

    # 最終画像
    store["image_golden_u8"] = img_u8.contiguous()       # [H,W,3] uint8
    store["image_golden_f32"] = img_f_m11.contiguous()   # [1,3,H,W] float [-1,1]

    meta = {
        "model": MODEL_ID,
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE,
        "seed": str(SEED),
        "guidance_scale": str(GUIDANCE_SCALE),
        "steps": str(STEPS),
        "width": str(WIDTH),
        "height": str(HEIGHT),
        "penultimate_index": "hidden_states[-2] (final_layer_norm 前, clip_skip=None)",
        "concat_order": "encoder_hidden_states[...,:768]=CLIP-L penultimate, [...,768:2048]=bigG penultimate",
        "pooled_source": "text_encoder_2 (bigG) prompt_embeds[0] (ndim==2)",
        "time_ids_layout": "original_size(H,W) + crops_coords_top_left(0,0) + target_size(H,W) = (1024,1024,0,0,1024,1024)",
        "encode_dtype": "fp32 (deterministic)",
        "image_dtype": "fp16 pipeline (probe10 と同条件)",
        "image_golden_u8_layout": "[H,W,3] uint8 RGB",
        "image_golden_f32_layout": "[1,3,H,W] float [-1,1] (= x*2-1, VAE decode 直後相当)",
        "cfg_formula": "noise = uncond + guidance_scale*(cond - uncond)",
    }

    save_file(store, IO_PATH, metadata=meta)
    io_bytes = os.path.getsize(IO_PATH)

    # ================= レポート =================
    print("\n" + "=" * 70)
    print("生成完了")
    print("=" * 70)
    print(f"txt2img_io.safetensors : {IO_PATH}  ({io_bytes/1e6:.2f} MB)")
    if png_path:
        print(f"txt2img_golden.png     : {png_path}")

    print("\n--- 保存キー一覧 ---")
    print(f"{'key':<38}{'shape':<24}{'dtype':<12}")
    for k, v in store.items():
        print(f"{k:<38}{str(list(v.shape)):<24}{str(v.dtype):<12}")
    if skipped_f16:
        print("\n  ★ FP16 範囲外で f16 を省略:")
        for nm in skipped_f16:
            print(f"    - {nm}")
    else:
        print("\n  FP16 範囲外なし (全 f16 併設)")

    print("\n--- 値域 (min/max/mean/NaN/Inf) ---")
    for name, t in [
        ("cond_encoder_hidden_states", cond_ehs),
        ("uncond_encoder_hidden_states", uncond_ehs),
        ("cond_text_embeds", cond_pooled),
        ("uncond_text_embeds", uncond_pooled),
        ("time_ids", time_ids),
        ("image_golden_f32([-1,1])", img_f_m11),
    ]:
        mn, mx, mean, hn, hi = stats(t)
        flag = (" NAN" if hn else "") + (" INF" if hi else "")
        print(f"  {name:<34} min={mn:>10.4f} max={mx:>10.4f} mean={mean:>10.4f}{flag}")

    # 最終画像 u8 統計
    iu = img_u8.to(torch.float32)
    print(f"  {'image_golden_u8':<34} min={float(iu.min()):>10.1f} "
          f"max={float(iu.max()):>10.1f} mean={float(iu.mean()):>10.4f}  "
          f"std={float(iu.std()):>8.3f}")

    print("\n--- token ids (突合用 先頭16) ---")
    print("  cond  L:", cond_ids_l[0, :16].tolist())
    print("  cond  G:", cond_ids_g[0, :16].tolist())
    print("  uncond L:", uncond_ids_l[0, :16].tolist())
    print("  uncond G:", uncond_ids_g[0, :16].tolist())

    print("\npenultimate 取り出し: hidden_states[-2] (両エンコーダ)")
    print("concat 順: CLIP-L(768) が先, bigG(1280) が後 -> [1,77,2048]")
    print("pooled: bigG (text_encoder_2) のみ -> [1,1280]")
    print("=" * 70)


if __name__ == "__main__":
    main()

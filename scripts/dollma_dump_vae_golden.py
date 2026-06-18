#!/usr/bin/env python3
# SDXL VAE decoder ゴールデンデータ生成。
#
# C++ 自作 CUDA カーネルで SDXL AutoencoderKL の decoder
# (latent [1,4,128,128] -> 画像 [1,3,1024,1024]) を再実装するための
# 「正解」データを safetensors で出力する。C++ 側は src/io/safetensors.hpp で
# これをロードして突合する。
#
# 出力先 (src/tests/data/):
#   vae_weights.safetensors  ... VAE decoder の全 state_dict 重み (FP16)
#                                 decoder.* と post_quant_conv.*
#   vae_io.safetensors       ... 入力 latent / 各ブロック中間出力 / 最終画像
#                                 各テンソルを _f16 / _f32 両方で保存
#   vae_golden.png           ... 正解画像 (目視確認用)
#
# === scaling_factor の扱い (重要) ===
#   SDXL の scaling_factor = 0.13025。
#   diffusers は image = vae.decode(latent / vae.config.scaling_factor) の順で
#   内部処理する (decode() に渡る前にスケールで割る)。
#   本スクリプトでは:
#     - input_latent_raw         = 固定 seed の randn 生 latent (スケール前)
#     - input_latent_post_scale  = raw / scaling_factor (decode() に実際に渡る値)
#   C++ は input_latent_post_scale を decoder への入力として使うこと。
#   (decoder への真の入力は post_quant_conv の前段、すなわち post_scale latent)
#
# === dtype 方針 ===
#   SDXL VAE は fp16 で NaN を吐く既知問題があるため、decode は fp32 (CPU) で行う。
#   重みは fp16 版を別途 .half() で保存する。中間テンソルは fp32 を正解とし、
#   保存時に fp16 版も作る。
#
# === 実行デバイス ===
#   fp32 CPU decode (NaN 回避に最も確実、VAE decode 1回だけなので速度問題なし)。

import os
from collections import OrderedDict

import torch
from diffusers import AutoencoderKL
from safetensors.torch import save_file

SCALING_FACTOR = 0.13025  # SDXL
SEED = 1234

OUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src", "tests", "data")
)
WEIGHTS_PATH = os.path.join(OUT_DIR, "vae_weights.safetensors")
IO_PATH = os.path.join(OUT_DIR, "vae_io.safetensors")
PNG_PATH = os.path.join(OUT_DIR, "vae_golden.png")


def stats(t: torch.Tensor):
    tf = t.detach().to(torch.float32)
    return (
        float(tf.min()),
        float(tf.max()),
        bool(torch.isnan(tf).any()),
        bool(torch.isinf(tf).any()),
    )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.set_grad_enabled(False)

    print("=" * 70)
    print("SDXL VAE decoder ゴールデンデータ生成")
    print("=" * 70)
    print("device: CPU  / decode dtype: fp32 (NaN 回避)")
    print(f"scaling_factor: {SCALING_FACTOR}  seed: {SEED}")

    # ---- VAE のみロード (fp32) ----
    print("\nVAE ロード中 (初回はダウンロード数分)...")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="vae",
        torch_dtype=torch.float32,
    )
    vae = vae.to("cpu").eval()
    print("ロード完了。config.scaling_factor =", vae.config.scaling_factor)

    # decoder 構造を print (サブモジュール名確認用)
    print("\n=== vae.decoder 構造 ===")
    print(vae.decoder)

    # ---- 入力 latent (固定 seed randn) ----
    gen = torch.Generator(device="cpu").manual_seed(SEED)
    latent_raw = torch.randn([1, 4, 128, 128], generator=gen, dtype=torch.float32)
    latent_post_scale = latent_raw / SCALING_FACTOR

    # ---- forward hook で中間出力を回収 ----
    captured = OrderedDict()

    def hook(name):
        def fn(module, inp, out):
            o = out[0] if isinstance(out, (tuple, list)) else out
            captured[name] = o.detach().clone()
        return fn

    handles = []
    handles.append(vae.post_quant_conv.register_forward_hook(hook("post_quant_conv_out")))
    handles.append(vae.decoder.conv_in.register_forward_hook(hook("conv_in_out")))
    handles.append(vae.decoder.mid_block.register_forward_hook(hook("mid_block_out")))
    for i, up in enumerate(vae.decoder.up_blocks):
        handles.append(up.register_forward_hook(hook(f"up_block_{i}_out")))
    handles.append(vae.decoder.conv_norm_out.register_forward_hook(hook("conv_norm_out_out")))
    handles.append(vae.decoder.conv_out.register_forward_hook(hook("final_image")))

    # ---- decode 実行 ----
    # diffusers の decode() は内部で post_quant_conv -> decoder を呼ぶ。
    # decode() に渡すのは post_scale latent (生 latent / scaling_factor)。
    print("\ndecode 実行中 (fp32 CPU)...")
    decoded = vae.decode(latent_post_scale).sample  # [1,3,1024,1024], 値域 ~[-1,1]

    for h in handles:
        h.remove()

    # final_image hook は conv_out の出力 = decode().sample と一致するはず
    captured["final_image"] = decoded.detach().clone()

    # ---- io テンソル収集 ----
    io_tensors = OrderedDict()
    io_tensors["input_latent_raw"] = latent_raw
    io_tensors["input_latent_post_scale"] = latent_post_scale
    # 互換用エイリアス: decoder への真の入力 = post_scale
    io_tensors["input_latent"] = latent_post_scale
    for k, v in captured.items():
        io_tensors[k] = v

    # === 保存方針 (golden を git に commit 可能なサイズに収める) ===
    #   中間活性は全解像度だと up_block 段だけで ~3.7GB になり commit 不可。
    #   そこで各テンソルを以下のルールで保存する:
    #     - full サイズが FULL_BUDGET 以下、または final_image は「全解像度」で保存
    #       (キー  <name>_f32 / <name>_f16)
    #     - それを超える大物中間活性は「決定論的な左上クロップ
    #       [: , : , :CROP, :CROP] (全チャネル × CROP×CROP 空間)」で保存
    #       (キー  <name>_crop_f32 / <name>_crop_f16、属性に元 shape とクロップ法を記録)
    #   C++ 側は同じ左上クロップ領域を取り出して突合する。NCHW・row-major 前提なので
    #   左上クロップは各行先頭 CROP 要素を H 方向 CROP 行ぶん取れば再現できる。
    #
    # === 重要な発見: SDXL VAE decoder の中間活性は FP16 範囲を超える ===
    #   up_block_2_out / up_block_3_out は値域が |x| > 65504 (FP16 max) に達し、
    #   f16 に落とすと Inf が発生する (up_block_3_out で実測 759 個の Inf)。
    #   これが「SDXL VAE は fp16 で NaN/壊れる」既知問題の正体。
    #   よって C++ 自作カーネルでも up_block 段は FP32 蓄積必須。
    #   FP16 で表現不能なテンソルは f16 版を保存しない (Inf を golden に混ぜない)。
    #   safetensors はストレージ共有テンソルを拒否するため必ず clone() する。
    FP16_MAX = 65504.0
    FULL_BUDGET = 160 * 1024 * 1024  # 160MB: conv_in/mid_block(33MB)/up_block_0(134MB) は全解像度
    CROP = 32                         # 大物は左上 32x32 空間クロップ (全チャネル)
    NEVER_CROP = {"final_image"}      # final_image は主役なので必ず全解像度 (12MB)

    io_save = OrderedDict()
    meta = {"scaling_factor": str(SCALING_FACTOR), "seed": str(SEED),
            "crop": str(CROP), "crop_origin": "top-left [:,:,:CROP,:CROP]"}
    skipped_f16 = []
    cropped = []

    def add(name, t):
        vc = t.to(torch.float32).clone().contiguous()
        io_save[f"{name}_f32"] = vc
        if float(vc.abs().max()) <= FP16_MAX:
            io_save[f"{name}_f16"] = vc.to(torch.float16).clone().contiguous()
        else:
            skipped_f16.append(name)

    for k, v in io_tensors.items():
        nbytes_f32 = v.numel() * 4
        if k in NEVER_CROP or nbytes_f32 <= FULL_BUDGET:
            add(k, v)
        else:
            # 左上クロップ。空間 H,W が最後の 2 軸である NCHW 前提。
            vcrop = v[..., :CROP, :CROP].contiguous()
            add(f"{k}_crop", vcrop)
            meta[f"{k}_full_shape"] = str(list(v.shape))
            cropped.append((k, list(v.shape), list(vcrop.shape)))

    save_file(io_save, IO_PATH, metadata=meta)

    # ---- 重み保存 (fp16) ----
    # decoder.* と post_quant_conv.* のみ
    full_sd = vae.state_dict()
    weights = OrderedDict()
    for k, v in full_sd.items():
        if k.startswith("decoder.") or k.startswith("post_quant_conv."):
            weights[k] = v.to(torch.float16).contiguous()
    save_file(weights, WEIGHTS_PATH)

    # ---- PNG 保存 ([-1,1] -> [0,255]) ----
    img = decoded.to(torch.float32).clamp(-1, 1)
    img = ((img + 1.0) / 2.0 * 255.0).round().clamp(0, 255).to(torch.uint8)
    img = img[0].permute(1, 2, 0).contiguous().numpy()  # HWC
    try:
        from PIL import Image
        Image.fromarray(img).save(PNG_PATH)
        png_ok = True
    except Exception as e:
        print("PNG 保存スキップ:", e)
        png_ok = False

    # ================= レポート =================
    print("\n" + "=" * 70)
    print("生成完了")
    print("=" * 70)
    print("vae_io.safetensors    :", IO_PATH)
    print("vae_weights.safetensors:", WEIGHTS_PATH)
    if png_ok:
        print("vae_golden.png        :", PNG_PATH)

    any_nan = False
    print("\n--- io テンソル一覧 (f32 値域) ---")
    print(f"{'name':<28}{'dtype':<6}{'shape':<22}{'min':>11}{'max':>11}  flags")
    for k, v in io_tensors.items():
        mn, mx, has_nan, has_inf = stats(v)
        any_nan = any_nan or has_nan or has_inf
        flag = ""
        if has_nan:
            flag += " NAN"
        if has_inf:
            flag += " INF"
        savedkey = k if f"{k}_f32" in io_save else f"{k}_crop"
        note = ""
        if savedkey.endswith("_crop") or f"{savedkey}" == f"{k}_crop":
            pass
        if f"{k}_f32" not in io_save and f"{k}_crop_f32" in io_save:
            note += " [CROP保存 32x32]"
        if f"{savedkey}_f16" not in io_save:
            note += " [f16省略:FP16範囲外]"
        print(f"{k:<28}{'F32':<6}{str(list(v.shape)):<22}{mn:>11.4f}{mx:>11.4f} {flag}{note}")
    print("  (_f32 は全テンソル保存。_f16 は |x|<=65504 のみ。大物は左上32x32クロップ=_crop)")
    if skipped_f16:
        print("  f16 を省略 (FP16 で Inf):", skipped_f16)
    if cropped:
        print("  左上クロップ保存した大物中間活性 (元shape -> crop shape):")
        for nm, fs, cs in cropped:
            print(f"    {nm:<22} {fs} -> {cs}")
    print("  saved keys:", sorted(io_save.keys()))

    print(f"\n--- 重み一覧 (FP16, {len(weights)} テンソル) ---")
    total_params = 0
    for k, v in weights.items():
        total_params += v.numel()
        print(f"{k:<48}F16  {list(v.shape)}")
    print(f"\n重み総パラメータ数: {total_params:,} (FP16 = {total_params*2/1e6:.1f} MB)")

    print("\nNaN/Inf 検出:", "あり (要調査!)" if any_nan else "なし (OK)")
    print("=" * 70)


if __name__ == "__main__":
    main()

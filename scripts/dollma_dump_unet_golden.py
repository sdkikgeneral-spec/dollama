#!/usr/bin/env python3
# SDXL UNet ゴールデンデータ生成 (task 2-5)。
#
# C++ 自作 CUDA カーネルで SDXL UNet2DConditionModel
# (latent [1,4,128,128] + timestep + encoder_hidden_states[1,77,2048]
#  + added_cond_kwargs -> noise prediction [1,4,128,128]) を再実装するための
# 「正解」データを safetensors で出力する。C++ 側は src/io/safetensors.hpp で
# これをロードして突合する (手本: src/tests/test_vae_decode.cu)。
#
# 出力先 (src/tests/data/):
#   unet_weights.safetensors  ... UNet の全 state_dict 重み (FP16, キー命名そのまま)
#   unet_io.safetensors       ... 固定入力 / 各段中間出力 / 最終 noise pred /
#                                 Euler スケジューラの sigmas/timesteps と 1step 検証
#
# === ダンプ形式 (VAE golden と完全一致させる) ===
#   各テンソルは以下のルールで保存:
#     - full サイズが FULL_BUDGET 以下、または NEVER_CROP は「全解像度」で保存
#       (キー  <name>_f32 / <name>_f16)
#     - それを超える大物中間は「左上クロップ [:,:,:CROP,:CROP]」(全チャネル)
#       (キー  <name>_crop_f32 / <name>_crop_f16、メタに元 shape を記録)
#     - FP16 範囲外 (|x| > 65504) のテンソルは f16 版を省略 (Inf を golden に混ぜない)
#   NCHW row-major 前提。左上クロップは各行先頭 CROP 要素を H 方向 CROP 行ぶん取れば再現可。
#   ※ 中間によっては [B, seq, dim] の 3 次元 (attn 後) もあるため、
#      4 次元 (NCHW) のときのみ空間クロップし、それ以外は full 保存する。
#
# === SDXL UNet の入力仕様 ===
#   - sample (latent):        [1, 4, 128, 128] fp32
#   - timestep:               scalar (拡散ステップ。Euler の sigma に対応)
#   - encoder_hidden_states:  [1, 77, 2048]  (CLIP-L 768 + CLIP-bigG 1280 結合)
#   - added_cond_kwargs:
#       text_embeds:          [1, 1280]      (bigG pooled)
#       time_ids:             [1, 6]         (h, w, crop_top, crop_left, target_h, target_w)
#   本物のテキストは通さず固定 randn でカーネル配線検証に用いる。
#
# === dtype 方針 ===
#   forward は fp32 CPU で行い (NaN 回避・決定論)、中間/重みの fp16 版も作る。
#   FP16 範囲を超える中間があれば報告 (C++ 側 FP32 中間化の判断材料)。

import os
from collections import OrderedDict

import torch
from diffusers import UNet2DConditionModel, EulerDiscreteScheduler
from safetensors.torch import save_file

SEED = 1234
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

OUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src", "tests", "data")
)
WEIGHTS_PATH = os.path.join(OUT_DIR, "unet_weights.safetensors")
IO_PATH = os.path.join(OUT_DIR, "unet_io.safetensors")


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
    print("SDXL UNet ゴールデンデータ生成 (task 2-5)")
    print("=" * 70)
    print("device: CPU  / forward dtype: fp32 (NaN 回避・決定論)")
    print(f"model: {MODEL_ID}  seed: {SEED}")

    # ---- UNet のみロード (fp32) ----
    print("\nUNet ロード中 (初回はダウンロード数 GB)...")
    unet = UNet2DConditionModel.from_pretrained(
        MODEL_ID, subfolder="unet", torch_dtype=torch.float32
    )
    unet = unet.to("cpu").eval()
    print("ロード完了。config:")
    print("  in_channels         =", unet.config.in_channels)
    print("  out_channels        =", unet.config.out_channels)
    print("  block_out_channels  =", unet.config.block_out_channels)
    print("  down_block_types    =", unet.config.down_block_types)
    print("  up_block_types      =", unet.config.up_block_types)
    print("  cross_attention_dim =", unet.config.cross_attention_dim)
    print("  addition_time_embed_dim =",
          getattr(unet.config, "addition_time_embed_dim", None))
    print("  projection_class_embeddings_input_dim =",
          getattr(unet.config, "projection_class_embeddings_input_dim", None))

    # ---- module 構造を print ----
    print("\n" + "=" * 70)
    print("=== unet 構造 (print(unet)) ===")
    print("=" * 70)
    print(unet)

    # ---- down 経路のダウンサンプル種別 (最重要確認: stride-2 conv か avgpool か) ----
    print("\n" + "=" * 70)
    print("=== down_blocks の downsamplers 解剖 (stride-2 conv? avgpool?) ===")
    print("=" * 70)
    for i, db in enumerate(unet.down_blocks):
        ds = getattr(db, "downsamplers", None)
        if ds is None:
            print(f"down_block[{i}] ({type(db).__name__}): downsamplers = None (最終段は無し)")
            continue
        for j, d in enumerate(ds):
            print(f"down_block[{i}].downsamplers[{j}]: {type(d).__name__}")
            # Downsample2D は内部に conv (use_conv=True なら stride2 Conv2d) を持つ
            conv = getattr(d, "conv", None)
            if conv is not None:
                print(f"    conv = {conv}")
                if isinstance(conv, torch.nn.Conv2d):
                    print(f"    -> Conv2d stride={conv.stride} kernel={conv.kernel_size} "
                          f"pad={conv.padding} in={conv.in_channels} out={conv.out_channels}")
                else:
                    print(f"    -> type {type(conv).__name__} (avgpool 等?)")
            else:
                print(f"    conv 属性なし -> 全 module: {d}")

    print("\n=== up_blocks の upsamplers 解剖 (interp+conv? convtranspose?) ===")
    for i, ub in enumerate(unet.up_blocks):
        us = getattr(ub, "upsamplers", None)
        if us is None:
            print(f"up_block[{i}] ({type(ub).__name__}): upsamplers = None")
            continue
        for j, u in enumerate(us):
            print(f"up_block[{i}].upsamplers[{j}]: {type(u).__name__}")
            conv = getattr(u, "conv", None)
            if conv is not None:
                print(f"    conv = {conv}")

    # ---- 固定入力 (固定 seed randn) ----
    gen = torch.Generator(device="cpu").manual_seed(SEED)
    sample = torch.randn([1, 4, 128, 128], generator=gen, dtype=torch.float32)
    encoder_hidden_states = torch.randn([1, 77, 2048], generator=gen, dtype=torch.float32)
    text_embeds = torch.randn([1, 1280], generator=gen, dtype=torch.float32)
    # SDXL time_ids = (orig_h, orig_w, crop_top, crop_left, target_h, target_w)
    time_ids = torch.tensor([[1024.0, 1024.0, 0.0, 0.0, 1024.0, 1024.0]], dtype=torch.float32)
    added_cond_kwargs = {"text_embeds": text_embeds, "time_ids": time_ids}

    # ---- Euler スケジューラ (SDXL デフォルト) ----
    print("\n" + "=" * 70)
    print("=== EulerDiscreteScheduler (num_inference_steps=20) ===")
    print("=" * 70)
    scheduler = EulerDiscreteScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    scheduler.set_timesteps(20, device="cpu")
    sigmas = scheduler.sigmas.detach().to(torch.float32).clone()       # [21] (末尾 0)
    timesteps = scheduler.timesteps.detach().to(torch.float32).clone()  # [20]
    print("sigmas    shape", list(sigmas.shape), "=", sigmas.tolist())
    print("timesteps shape", list(timesteps.shape), "=", timesteps.tolist())

    # step 0 で実際に使う timestep を採用 (Euler の最初のステップ)。
    t_step0 = timesteps[0].clone()  # scalar (例 ~999)
    timestep_for_forward = t_step0

    # scale_model_input: Euler は latent を 1/sqrt(sigma^2+1) でスケール
    latent_scaled_step0 = scheduler.scale_model_input(sample, t_step0).detach().clone()

    # ---- forward hook で中間出力を回収 ----
    captured = OrderedDict()

    def hook(name):
        def fn(module, inp, out):
            o = out[0] if isinstance(out, (tuple, list)) else out
            if isinstance(o, torch.Tensor):
                captured[name] = o.detach().clone()
        return fn

    handles = []
    # time embedding 経路
    handles.append(unet.time_embedding.register_forward_hook(hook("time_embedding_out")))
    if getattr(unet, "add_embedding", None) is not None:
        handles.append(unet.add_embedding.register_forward_hook(hook("add_embedding_out")))
    # conv_in
    handles.append(unet.conv_in.register_forward_hook(hook("conv_in_out")))
    # down blocks (ブロック出力 + 各ブロック内 resnet / attentions の代表)
    for i, db in enumerate(unet.down_blocks):
        handles.append(db.register_forward_hook(hook(f"down_block_{i}_out")))
        resnets = getattr(db, "resnets", None)
        if resnets is not None and len(resnets) > 0:
            handles.append(resnets[0].register_forward_hook(hook(f"down_block_{i}_resnet0_out")))
        attns = getattr(db, "attentions", None)
        if attns is not None and len(attns) > 0:
            handles.append(attns[0].register_forward_hook(hook(f"down_block_{i}_attn0_out")))
        ds = getattr(db, "downsamplers", None)
        if ds is not None and len(ds) > 0:
            handles.append(ds[0].register_forward_hook(hook(f"down_block_{i}_downsample0_out")))
    # mid block (+ 内部 resnet/attn)
    handles.append(unet.mid_block.register_forward_hook(hook("mid_block_out")))
    mid_res = getattr(unet.mid_block, "resnets", None)
    if mid_res is not None and len(mid_res) > 0:
        handles.append(mid_res[0].register_forward_hook(hook("mid_block_resnet0_out")))
    mid_attn = getattr(unet.mid_block, "attentions", None)
    if mid_attn is not None and len(mid_attn) > 0:
        handles.append(mid_attn[0].register_forward_hook(hook("mid_block_attn0_out")))
    # up blocks
    for i, ub in enumerate(unet.up_blocks):
        handles.append(ub.register_forward_hook(hook(f"up_block_{i}_out")))
        resnets = getattr(ub, "resnets", None)
        if resnets is not None and len(resnets) > 0:
            handles.append(resnets[0].register_forward_hook(hook(f"up_block_{i}_resnet0_out")))
        attns = getattr(ub, "attentions", None)
        if attns is not None and len(attns) > 0:
            handles.append(attns[0].register_forward_hook(hook(f"up_block_{i}_attn0_out")))
        us = getattr(ub, "upsamplers", None)
        if us is not None and len(us) > 0:
            handles.append(us[0].register_forward_hook(hook(f"up_block_{i}_upsample0_out")))
    # 末尾
    handles.append(unet.conv_norm_out.register_forward_hook(hook("conv_norm_out_out")))
    handles.append(unet.conv_out.register_forward_hook(hook("conv_out_out")))

    # ---- forward 実行 (step 0 の scale 済み latent を入力) ----
    print("\nUNet forward 実行中 (fp32 CPU)...")
    noise_pred = unet(
        latent_scaled_step0,
        timestep_for_forward,
        encoder_hidden_states=encoder_hidden_states,
        added_cond_kwargs=added_cond_kwargs,
    ).sample  # [1,4,128,128]

    for h in handles:
        h.remove()

    captured["noise_pred"] = noise_pred.detach().clone()

    # ---- Euler step 0 を実行 (S4 スケジューラ突合用) ----
    # step() は sample (scale 前の生 latent) と model_output (noise_pred) を取る。
    latent_after_step0 = scheduler.step(
        noise_pred, t_step0, sample
    ).prev_sample.detach().clone()

    # ---- io テンソル収集 ----
    io_tensors = OrderedDict()
    io_tensors["input_sample"] = sample                      # 生 latent [1,4,128,128]
    io_tensors["input_sample_scaled_step0"] = latent_scaled_step0  # forward への実入力
    io_tensors["input_encoder_hidden_states"] = encoder_hidden_states
    io_tensors["input_text_embeds"] = text_embeds
    io_tensors["input_time_ids"] = time_ids
    io_tensors["input_timestep"] = t_step0.reshape(1).to(torch.float32)
    # スケジューラ配列
    io_tensors["sched_sigmas"] = sigmas
    io_tensors["sched_timesteps"] = timesteps
    io_tensors["sched_latent_after_step0"] = latent_after_step0
    for k, v in captured.items():
        io_tensors[k] = v

    # === 保存 (VAE golden と同じルール) ===
    FP16_MAX = 65504.0
    FULL_BUDGET = 160 * 1024 * 1024  # 160MB
    CROP = 32
    NEVER_CROP = {
        "input_sample", "input_sample_scaled_step0", "noise_pred",
        "conv_out_out", "sched_sigmas", "sched_timesteps",
        "input_timestep", "input_time_ids", "input_text_embeds",
        "sched_latent_after_step0",
    }

    io_save = OrderedDict()
    meta = {"seed": str(SEED), "crop": str(CROP),
            "crop_origin": "top-left [:,:,:CROP,:CROP] (NCHW only)",
            "model": MODEL_ID}
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
        # 空間クロップは 4 次元 (NCHW) のときのみ。3 次元 (B,seq,dim) 等は full。
        if k in NEVER_CROP or nbytes_f32 <= FULL_BUDGET or v.dim() != 4:
            add(k, v)
        else:
            vcrop = v[..., :CROP, :CROP].contiguous()
            add(f"{k}_crop", vcrop)
            meta[f"{k}_full_shape"] = str(list(v.shape))
            cropped.append((k, list(v.shape), list(vcrop.shape)))

    save_file(io_save, IO_PATH, metadata=meta)

    # ---- 重み保存 (fp16, キー命名そのまま) ----
    full_sd = unet.state_dict()
    weights = OrderedDict()
    for k, v in full_sd.items():
        weights[k] = v.to(torch.float16).contiguous()
    save_file(weights, WEIGHTS_PATH)

    weights_bytes = os.path.getsize(WEIGHTS_PATH)
    io_bytes = os.path.getsize(IO_PATH)

    # ================= レポート =================
    print("\n" + "=" * 70)
    print("生成完了")
    print("=" * 70)
    print(f"unet_io.safetensors     : {IO_PATH}  ({io_bytes/1e6:.1f} MB)")
    print(f"unet_weights.safetensors: {WEIGHTS_PATH}  ({weights_bytes/1e6:.1f} MB)")

    any_nan = False
    print("\n--- io テンソル一覧 (f32 値域) ---")
    print(f"{'name':<34}{'shape':<24}{'min':>12}{'max':>12}  flags")
    for k, v in io_tensors.items():
        mn, mx, has_nan, has_inf = stats(v)
        any_nan = any_nan or has_nan or has_inf
        flag = ""
        if has_nan:
            flag += " NAN"
        if has_inf:
            flag += " INF"
        note = ""
        if f"{k}_f32" not in io_save and f"{k}_crop_f32" in io_save:
            note += " [CROP保存 32x32]"
        if k in skipped_f16 or (f"{k}_crop" in skipped_f16):
            note += " [f16省略:FP16範囲外]"
        print(f"{k:<34}{str(list(v.shape)):<24}{mn:>12.4f}{mx:>12.4f} {flag}{note}")
    print("  (_f32 は全テンソル保存。_f16 は |x|<=65504 のみ。大物 4D は左上32x32クロップ=_crop)")
    if skipped_f16:
        print("\n  ★ FP16 範囲外で f16 を省略したテンソル (C++ は FP32 中間にすること):")
        for nm in skipped_f16:
            print(f"    - {nm}")
    else:
        print("\n  FP16 範囲外の中間テンソルなし (全テンソル f16 表現可能)")
    if cropped:
        print("\n  左上クロップ保存した大物中間活性 (元shape -> crop shape):")
        for nm, fs, cs in cropped:
            print(f"    {nm:<28} {fs} -> {cs}")

    # ---- 重みキー命名規則レポート ----
    print(f"\n--- 重み一覧 (FP16, {len(weights)} テンソル) ---")
    total_params = 0
    for k, v in weights.items():
        total_params += v.numel()
    print(f"重み総パラメータ数: {total_params:,} (FP16 = {total_params*2/1e6:.1f} MB)")
    # キー命名規則がわかるよう代表キーを列挙 (全列挙は数百行になるため接頭辞で要約)。
    print("\n  キー接頭辞ごとのテンソル数 (命名規則確認用):")
    prefix_count = OrderedDict()
    for k in weights.keys():
        # 先頭 2 トークンを接頭辞とする (例 down_blocks.0)
        toks = k.split(".")
        pref = ".".join(toks[:2]) if len(toks) >= 2 else k
        prefix_count[pref] = prefix_count.get(pref, 0) + 1
    for pref, cnt in prefix_count.items():
        print(f"    {pref:<28} : {cnt} テンソル")
    print("\n  主要重みキーと shape (代表サンプル):")
    sample_keys = [
        "conv_in.weight", "conv_in.bias",
        "time_embedding.linear_1.weight", "time_embedding.linear_2.weight",
        "add_embedding.linear_1.weight", "add_embedding.linear_2.weight",
        "down_blocks.0.resnets.0.conv1.weight",
        "down_blocks.0.resnets.0.time_emb_proj.weight",
        "down_blocks.0.resnets.0.norm1.weight",
        "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q.weight",
        "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_q.weight",
        "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.weight",
        "down_blocks.1.attentions.0.transformer_blocks.0.ff.net.0.proj.weight",
        "down_blocks.1.attentions.0.proj_in.weight",
        "down_blocks.1.attentions.0.proj_out.weight",
        "down_blocks.0.downsamplers.0.conv.weight",
        "mid_block.resnets.0.conv1.weight",
        "mid_block.attentions.0.transformer_blocks.0.attn1.to_q.weight",
        "up_blocks.0.resnets.0.conv1.weight",
        "up_blocks.0.upsamplers.0.conv.weight",
        "up_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight",
        "conv_norm_out.weight", "conv_out.weight", "conv_out.bias",
    ]
    for sk in sample_keys:
        if sk in weights:
            print(f"    {sk:<70} {list(weights[sk].shape)}")
        else:
            print(f"    {sk:<70} (キー不在)")

    print("\nNaN/Inf 検出:", "あり (要調査!)" if any_nan else "なし (OK)")
    print("=" * 70)


if __name__ == "__main__":
    main()

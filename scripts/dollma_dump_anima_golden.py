#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dollma_dump_anima_golden.py — Anima 自作実装トラック A0: golden ダンプ (FP32)。

目的 (計画: Anima 自作実装トラック A0):
  Anima (Cosmos-Predict2 2B DiT + Qwen3 0.6B TE + Qwen-Image VAE) を C++/CUDA/OV で自作する際の
  「正解」を、固定 prompt / 固定 seed / 固定初期 noise で FP32 ダンプする。
  形式は scripts/dollma_dump_unet_golden.py に合わせる:
    <name>_f32 / <name>_f16 (f16 は |x|<=65504 のみ)・大きい 4D 以上は左上クロップ <name>_crop_f32。
    token id は <name>_i64 (整数・クロップ/f16 無し)。
  追加で JSON サイドカー (anima_golden_meta.json) に prompt・seed・sigmas・活性値表・VAE 検証を出す。

ダンプ内容 (anima_io.safetensors):
  tokens : pos/neg の Qwen3 token ids・T5 token ids・T5 weights
  TE     : last_hidden_state (pos/neg)・layer0 出力・最終層(final norm 前)出力
  adapter: llm_adapter 出力 (pad 前 / 512 pad 後)
  DiT    : patch_embed 入力(17ch)/出力・time_proj・temb・embedded_timestep・rope cos/sin・
           block0 の入出力と内部 (norm1/2/3 の modulation 後と gate・attn1・attn2・ff)・norm_out・proj_out・
           step0 / 最終 step の model 出力 (cond/uncond/CFG 後)
  sampler: sigmas (simple schedule + Euler)・x_step0・x_after_step0・最終 latent
  VAE    : decode 入力 z (latents_std/mean で逆正規化後)・decode 出力 [-1,1]・PNG

活性値表 (anima_golden_meta.json の "activations"):
  各段の max|x|・p99.9|x|・rms・FP16 余裕 (65504/max)。Qwen3 は先頭 token に massive activation が出るため
  TE 層は先頭 token とそれ以外を分けて記録する。FP16 で足りるか (A2 の判断材料) の一覧は "fp16_summary"。

VAE 検証 ("vae_check"):
  causal conv3d を T=1 で「時間核の最終スライス (index = 2*pad_t) だけの conv2d」に置換した版と
  元の decode (feat_cache 経路) の max / MAE。upsample3d の time_conv が T=1 では呼ばれないことも確認。

決定性:
  --determinism = 同一プロセス内で 2 回ダンプして bit 一致を確認。
  --compare A.safetensors B.safetensors = 既存 2 ファイルの bit 一致確認 (プロセス跨ぎ用・重み不要)。

使い方 (研究機・anima venv の python で。重みパスは dollma_probe_anima.py の定数を流用):
  models/anima/.venv/Scripts/python.exe scripts/dollma_dump_anima_golden.py --selftest
  models/anima/.venv/Scripts/python.exe scripts/dollma_dump_anima_golden.py ^
      --out-dir models/anima/golden --device cpu --denoise first            # 軽量 (最初の 1 step だけ)
  models/anima/.venv/Scripts/python.exe scripts/dollma_dump_anima_golden.py ^
      --out-dir models/anima/golden --device cuda --denoise full --determinism   # 本番

比較可能性 (重要): 本 golden は E-4 probe と「steps=30 / simple schedule / cfg=4.0」だけ揃えた。
E-4 は sampler=er_sde・seed=1000-1003・bf16 であり、本 golden (Euler・seed=1234・fp32) とは画像を直接比較できない。
あくまで自作実装 (C++/CUDA) の数値突合用の基準で、E-4 の画質評価とは別物。

メモリ: --device cuda では DiT FP32 重み ~8.4GB + TE 2.4GB に活性 ~8GB で 16GB を超えうるため、
encode 後に TE を CPU へ、denoise 後に DiT を CPU へ退避して VAE decode する。--device cpu で --determinism
(同一プロセス 2 回) は RAM ~30GB になるので禁止 (プロセス跨ぎ --compare を使う)。

仕様の出典 (推測でない部分): diffusers_anima の pipeline_anima.py / sampling.py / sigma_schedules.py /
text_encoding.py / image_processing.py (研究機 models/anima/diffusers-anima-src)。
"""

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

FP16_MAX = 65504.0
SEED = 1234                      # 固定 seed (golden 専用。E-4 の 1000..1003 とは別)
PROMPT_ID = "p2"                 # E-1 の p2 (手・小物)
MODEL_KEY = "anima-base-v1.0"    # dollma_probe_anima.MODEL_SPECS のキー
DEFAULT_STEPS = 30               # steps/schedule/cfg のみ E-4 に揃えた (sampler・seed・dtype は別)。pipeline 既定は 32
DEFAULT_CFG = 4.0                # probe 既定 (カード推奨 4-5)
DEFAULT_SIZE = 1024
LATENT_CH = 16
VAE_SCALE = 8
SHIFT = 3.0                      # build_anima_pipeline の既定 scheduler (shift=3.0・num_train_timesteps=1000)
NUM_TRAIN_TIMESTEPS = 1000
CROP = 32
FULL_BUDGET = 160 * 1024 * 1024  # UNet golden と同じ 160MB
NEVER_CROP = {"vae_out", "latent_final", "vae_in_z"}


# ============================================================
# 純関数 (重み不要・selftest / pytest 対象)
# ============================================================
def _np_p999(abs_flat, q=99.9):
    """abs_flat: 1-D float32 numpy。2^25 要素を超える場合は等間隔間引きで近似 (sampled=True)。"""
    sampled = False
    if abs_flat.size > (1 << 25):
        step = int(np.ceil(abs_flat.size / float(1 << 25)))
        abs_flat = abs_flat[::step]
        sampled = True
    if abs_flat.size == 0:
        return 0.0, sampled
    k = int(np.ceil(abs_flat.size * q / 100.0)) - 1
    k = min(max(k, 0), abs_flat.size - 1)
    return float(np.partition(abs_flat, k)[k]), sampled


def act_stats(t, split_first_token=False):
    """活性値統計。max|x|・p99.9|x|・rms・FP16 余裕。NaN/Inf があれば nonfinite を立てる。

    split_first_token=True: [B, T, D] の T=0 とそれ以外を別記録 (Qwen3 の massive activation 対策)。
    """
    tf = t.detach().to(torch.float32)
    finite = bool(torch.isfinite(tf).all())
    a = tf.abs().reshape(-1)
    a_np = a.cpu().numpy()
    mx = float(a_np.max()) if a_np.size else 0.0
    p, sampled = _np_p999(a_np)
    out = {
        "shape": list(tf.shape),
        "max_abs": mx,
        "p999_abs": p,
        "rms": float(torch.sqrt((tf.double() ** 2).mean())) if tf.numel() else 0.0,
        "fp16_headroom": (FP16_MAX / mx) if mx > 0 else float("inf"),
        "fp16_overflow": bool(mx > FP16_MAX),
        "nonfinite": (not finite),
    }
    if sampled:
        out["p999_sampled"] = True
    if split_first_token and tf.dim() == 3 and tf.shape[1] >= 2:
        out["max_abs_pos0"] = float(tf[:, 0].abs().max())
        out["max_abs_rest"] = float(tf[:, 1:].abs().max())
    return out


def time_snr_shift(alpha, t):
    """sigma_schedules._time_snr_shift と同式 (alpha=1 は恒等)。"""
    if alpha == 1.0:
        return t
    return (t * alpha) / (1.0 + t * (alpha - 1.0))


def simple_sigmas(shift=SHIFT, steps=DEFAULT_STEPS, n_train=NUM_TRAIN_TIMESTEPS):
    """simple schedule の sigmas (長さ steps+1・末尾 0.0)。sigma_schedules.build_simple_sigmas 相当。

    base_sigmas = shift_fn(arange(1..n_train)/n_train) (昇順)。stride = n_train/steps として
    picked[i] = base_sigmas[-(1 + int(i*stride))] (降順に拾う)。
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    t = torch.arange(1, n_train + 1, dtype=torch.float32) / float(n_train)
    base = time_snr_shift(shift, t)
    stride = len(base) / float(steps)
    picked = [float(base[-(1 + int(i * stride))].item()) for i in range(steps)]
    picked.append(0.0)
    return torch.tensor(picked, dtype=torch.float32)


def cfg_combine(uncond, cond, g):
    """CFG: uncond + g * (cond - uncond) (sampling._predict_noise_cfg と同式)。"""
    return uncond + g * (cond - uncond)


def denoised_from_pred(x, sigma, noise_pred):
    """CONST (rectified flow) の x0 予測: x - sigma * pred (sampling._predict_denoised_const)。"""
    return x - sigma.float() * noise_pred


def euler_step(x, denoised, sigma, sigma_next):
    """sample_euler の 1 step。sigma_next == 0 なら denoised をそのまま返す。"""
    if float(sigma_next) == 0.0:
        return denoised
    d = (x - denoised) / sigma.to(x.dtype)
    dt = (sigma_next - sigma).to(x.dtype)
    return x + d * dt


def make_noise(seed, shape):
    """pipeline の初期 noise (CPU generator・fp32・randn)。"""
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn(tuple(shape), generator=g, dtype=torch.float32)


def denormalize_latents(x, mean, std):
    """decode_latents と同じ: latents * std + mean (5D [B,16,T,H,W])。"""
    m = torch.tensor(mean, dtype=x.dtype, device=x.device).view(1, -1, 1, 1, 1)
    s = torch.tensor(std, dtype=x.dtype, device=x.device).view(1, -1, 1, 1, 1)
    return x * s + m


def crop_topleft(t, crop=CROP):
    """4D 以上の [..., H, W] を左上 crop x crop に切る。"""
    return t[..., :crop, :crop].contiguous()


def build_save_dict(tensors, never_crop=NEVER_CROP, budget=FULL_BUDGET, crop=CROP, croppable=None):
    """UNet golden と同じ保存規則。

    - 整数 tensor は <name>_i64 のみ。
    - float: f32 全保存。4D 以上で budget 超 かつ never_crop 外は左上クロップ (<name>_crop_*) + meta に元 shape。
      croppable (set) を渡すとその名前だけがクロップ対象 (末尾 2 軸が H,W の NCHW/NCTHW 専用・
      [B,heads,seq,dim] 等 hidden 軸を切る tensor を誤って切らない)。None なら従来どおり dim>=4 全部。
    - f16 版は |x|<=65504 のときだけ作る (超過は skipped_f16 に記録)。
    戻り値: (io_save, meta, skipped_f16, cropped)
    """
    io_save = OrderedDict()
    meta = {"crop": str(crop), "crop_origin": "top-left [..., :CROP, :CROP] (dim>=4 のみ)"}
    skipped_f16 = []
    cropped = []

    def add(name, t):
        vc = t.to(torch.float32).clone().contiguous()
        io_save[f"{name}_f32"] = vc
        if float(vc.abs().max()) <= FP16_MAX:
            io_save[f"{name}_f16"] = vc.to(torch.float16).clone().contiguous()
        else:
            skipped_f16.append(name)

    for k, v in tensors.items():
        if v.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool):
            io_save[f"{k}_i64"] = v.to(torch.int64).clone().contiguous()
            continue
        nbytes = v.numel() * 4
        if (k in never_crop or nbytes <= budget or v.dim() < 4
                or (croppable is not None and k not in croppable)):
            add(k, v)
        else:
            vc = crop_topleft(v, crop)
            add(f"{k}_crop", vc)
            meta[f"{k}_full_shape"] = str(list(v.shape))
            cropped.append((k, list(v.shape), list(vc.shape)))
    return io_save, meta, skipped_f16, cropped


def causal_conv3d_slice_index(pad_t2):
    """T=1 のとき効く時間核スライス index。左 zero pad = 2*pad_t なので index = 2*pad_t (= _padding[4])。"""
    return int(pad_t2)


def conv3d_to_conv2d_weight(weight5d, pad_t2):
    """conv3d weight [O,I,Kt,Kh,Kw] → conv2d weight [O,I,Kh,Kw] (T=1 の causal conv 用)。"""
    idx = causal_conv3d_slice_index(pad_t2)
    if idx >= weight5d.shape[2]:
        raise ValueError(f"時間核 index {idx} が Kt={weight5d.shape[2]} を超える")
    return weight5d[:, :, idx].contiguous()


def causal_conv3d_reference(x5, weight5d, bias, pad_thw):
    """参照: QwenImageCausalConv3d と同じ「時間は左のみ 2*pad_t の zero pad・空間は対称 pad」の conv3d。"""
    pt, ph, pw = pad_thw
    xp = F.pad(x5, (pw, pw, ph, ph, 2 * pt, 0))
    return F.conv3d(xp, weight5d, bias)


def causal_conv3d_t1_as_conv2d(x5, weight5d, bias, pad_thw, stride_hw=(1, 1)):
    """T=1 の x5 [B,C,1,H,W] に対する conv2d 版 (時間核の最終有効スライスのみ)。出力 [B,O,1,H',W']。"""
    if x5.shape[2] != 1:
        raise ValueError("T=1 専用")
    pt, ph, pw = pad_thw
    w2 = conv3d_to_conv2d_weight(weight5d, 2 * pt)
    x2 = F.pad(x5[:, :, 0], (pw, pw, ph, ph))
    return F.conv2d(x2, w2, bias, stride=stride_hw).unsqueeze(2)


def bitwise_equal(a, b):
    """dtype・shape・全バイト一致。NaN も bit で比べる。"""
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    return bool(torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))


def compare_tensor_dicts(da, db):
    """2 つの tensor dict の bit 一致。戻り: (all_equal, [不一致 key], [片方にしか無い key])。"""
    ka, kb = set(da), set(db)
    only = sorted(ka ^ kb)
    diff = [k for k in sorted(ka & kb) if not bitwise_equal(da[k], db[k])]
    return (not diff and not only), diff, only


def fp16_summary(activations):
    """活性値表から FP16 判定の要約 (max 上位・超過数・余裕 < 4 倍の段)。"""
    rows = [(k, v["max_abs"], v.get("max_abs_rest")) for k, v in activations.items()]
    rows.sort(key=lambda r: -r[1])
    over = [k for k, v in activations.items() if v["fp16_overflow"] or v["nonfinite"]]
    tight = [k for k, v in activations.items()
             if (not v["fp16_overflow"]) and v["fp16_headroom"] < 4.0]
    return {
        "top10_max_abs": [{"name": k, "max_abs": m, "max_abs_rest": r} for k, m, r in rows[:10]],
        "n_entries": len(rows),
        "fp16_overflow_or_nonfinite": over,
        "headroom_lt_4x": tight,
        "verdict_hint": ("FP16 不可の段あり (BF16/FP32 中間が必要)" if over
                         else ("FP16 可だが余裕 4 倍未満の段あり (要注意)" if tight else "全段 FP16 余裕 4 倍以上")),
    }


def set_fp32_ieee():
    """TF32 を無効化 (torch 2.11 は fp32_precision API・旧 allow_tf32 は非推奨 → フォールバック)。戻り: 使った方式。"""
    try:
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.fp32_precision = "ieee"
        try:
            torch.backends.cudnn.conv.fp32_precision = "ieee"
        except Exception:  # noqa: BLE001
            pass
        return "fp32_precision=ieee"
    except Exception:  # noqa: BLE001
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        return "allow_tf32=False"


def group_of(key):
    """活性値表のキー ('phase/label...') → te / adapter / dit / vae。"""
    for part in key.split("/"):
        if part in ("te", "adapter", "dit", "vae"):
            return part
    return "other"


def fp16_summary_grouped(activations):
    groups = {}
    for k, v in activations.items():
        groups.setdefault(group_of(k), {})[k] = v
    return {g: fp16_summary(d) for g, d in groups.items()}


def running_to_stats(running):
    """running max (label → float) を活性値表の形式にする。"""
    out = {}
    for k, mx in running.items():
        out[f"allsteps/{k}"] = {"max_abs": mx, "fp16_headroom": (FP16_MAX / mx) if mx > 0 else float("inf"),
                                "fp16_overflow": bool(mx > FP16_MAX), "nonfinite": bool(mx != mx or mx == float("inf"))}
    return out


def evaluate_gates(sigma_check, vae_check, ctx_equal, denoised_ref_diff, vae_checked=True):
    """FAIL 条件 (exit≠0)。戻り: {name: {value, limit, pass}}。"""
    gates = {}
    if sigma_check is not None:
        gates["sigma_selfcheck"] = {"value": sigma_check, "limit": 1e-6, "pass": bool(sigma_check < 1e-6)}
    if vae_checked:
        gates["vae_conv2d_max_diff"] = {"value": vae_check["max_abs_diff"], "limit": 1e-4,
                                        "pass": bool(vae_check["max_abs_diff"] < 1e-4)}
        gates["vae_time_conv_calls"] = {"value": vae_check["time_conv_total_calls_in_t1_decode"], "limit": 0,
                                        "pass": bool(vae_check["time_conv_total_calls_in_t1_decode"] == 0)}
    gates["b0_ctx_equals_pos_adapter_out"] = {"value": bool(ctx_equal), "limit": True, "pass": bool(ctx_equal)}
    if denoised_ref_diff is not None:
        gates["denoised_step0_vs_diffusers_anima"] = {"value": denoised_ref_diff, "limit": 1e-5,
                                                      "pass": bool(denoised_ref_diff < 1e-5)}
    return gates


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


# ============================================================
# hook 記録器
# ============================================================
def _first_tensor(out):
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor):
        return out[0]
    if hasattr(out, "sample"):
        return out.sample
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state
    return None


class Recorder:
    """forward hook で (a) 活性値統計 (phase 指定時のみ) と (b) テンソル捕捉 (capture 時のみ) を行う。"""

    def __init__(self):
        self.running_on = False
        self.running = {}            # label -> device scalar tensor (全 step の running max|x|)
        self.phase = None
        self.capture = False
        self.tensors = OrderedDict()
        self.stats = OrderedDict()
        self.handles = []
        self.call_counts = {}

    def _record(self, label, t, split_first):
        if t is None or not t.is_floating_point():
            return
        if self.running_on:
            mx = t.detach().abs().max().float()
            prev = self.running.get(label)
            self.running[label] = mx if prev is None else torch.maximum(prev, mx)
        if self.phase is not None:
            self.stats[f"{self.phase}/{label}"] = act_stats(t, split_first_token=split_first)

    def hook_stats(self, module, label, split_first=False, index=0):
        """forward 出力 (tuple なら index 番目) の統計。"""
        def fn(mod, inp, out):
            if self.phase is None and not self.running_on:
                return
            if isinstance(out, (tuple, list)):
                t = out[index] if len(out) > index and isinstance(out[index], torch.Tensor) else None
            else:
                t = _first_tensor(out) if index == 0 else None
            self._record(label, t, split_first)
        self.handles.append(module.register_forward_hook(fn))

    def hook_stats_in(self, module, label, split_first=False):
        """forward 入力 (第 1 位置引数) の統計 (GELU 後 8192 幅・MLP down_proj 入力など)。"""
        def pre(mod, args):
            if self.phase is None and not self.running_on:
                return
            self._record(label, args[0] if args else None, split_first)
        self.handles.append(module.register_forward_pre_hook(pre))

    def hook_adaln_stats(self, module, label):
        """CosmosAdaLayerNormZero の shift/scale/gate を入力 (hidden, embedded_timestep, temb) から再計算して統計。"""
        def pre(mod, args):
            if self.phase is None and not self.running_on:
                return
            emb, temb = args[1], (args[2] if len(args) > 2 else None)
            e = mod.linear_2(mod.linear_1(mod.activation(emb)))
            if temb is not None:
                e = e + temb
            shift, scale, gate = e.chunk(3, dim=-1)
            self._record(f"{label}_shift", shift, False)
            self._record(f"{label}_scale", scale, False)
            self._record(f"{label}_gate", gate, False)
        self.handles.append(module.register_forward_pre_hook(pre))

    def hook_capture(self, module, names, pre_arg=None):
        """names: 出力 tensor 名 (tuple 出力なら複数)。pre_arg を与えると forward_pre_hook で位置引数を捕捉。"""
        if isinstance(names, str):
            names = [names]

        if pre_arg is not None:
            def pre(mod, args):
                if self.capture:
                    self.tensors[names[0]] = args[pre_arg].detach().clone().float().cpu()
            self.handles.append(module.register_forward_pre_hook(pre))
            return

        def fn(mod, inp, out):
            if not self.capture:
                return
            outs = out if isinstance(out, (tuple, list)) else (out,)
            for n, o in zip(names, outs):
                if isinstance(o, torch.Tensor):
                    self.tensors[n] = o.detach().clone().float().cpu()
        self.handles.append(module.register_forward_hook(fn))

    def hook_count(self, module, label):
        self.call_counts.setdefault(label, 0)

        def pre(mod, args, kwargs=None):
            self.call_counts[label] += 1
        self.handles.append(module.register_forward_pre_hook(pre))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


# ============================================================
# VAE: causal conv3d → T=1 conv2d 置換
# ============================================================
class vae_t1_conv2d_patch:
    """with 文の間、vae 内の全 QwenImageCausalConv3d.forward を T=1 conv2d 版に差し替える。"""

    def __init__(self, vae):
        self.vae = vae
        self.saved = []

    def __enter__(self):
        for m in self.vae.modules():
            if type(m).__name__ != "QwenImageCausalConv3d":
                continue
            pad = m._padding  # (w_l, w_r, h_t, h_b, 2*pad_t, 0)
            pad_thw = (pad[4] // 2, pad[2], pad[0])
            stride_hw = (m.stride[1], m.stride[2])

            def make(mod, pad_thw=pad_thw, stride_hw=stride_hw):
                def fwd(x, cache_x=None):
                    return causal_conv3d_t1_as_conv2d(x, mod.weight, mod.bias, pad_thw, stride_hw)
                return fwd
            self.saved.append((m, m.__dict__.get("forward")))
            m.forward = make(m)
        return self

    def __exit__(self, *exc):
        for m, old in self.saved:
            if old is None:
                m.__dict__.pop("forward", None)
            else:
                m.forward = old
        self.saved = []


# ============================================================
# selftest (重み不要)
# ============================================================
def selftest():
    fails = []

    def check(name, cond):
        print(f"[selftest] {'PASS' if cond else 'FAIL'} {name}")
        if not cond:
            fails.append(name)

    g = torch.Generator().manual_seed(0)

    # --- 活性値統計 ---
    x = torch.randn(1, 8, 16, generator=g)
    x[0, 0, 0] = 1000.0
    s = act_stats(x, split_first_token=True)
    check("act_stats max_abs", abs(s["max_abs"] - 1000.0) < 1e-3)
    check("act_stats pos0/rest 分離", s["max_abs_pos0"] == 1000.0 and s["max_abs_rest"] < 10.0)
    check("act_stats p999 <= max", s["p999_abs"] <= s["max_abs"])
    check("act_stats fp16_headroom", abs(s["fp16_headroom"] - FP16_MAX / 1000.0) < 1e-3)
    big = torch.zeros(4)
    big[0] = 70000.0
    check("act_stats fp16_overflow", act_stats(big)["fp16_overflow"])
    nf = torch.tensor([1.0, float("nan")])
    check("act_stats nonfinite", act_stats(nf)["nonfinite"])
    # p99.9: 1..1000 の abs → ceil(1000*.999)-1 = 998 番目 → 999.0
    check("p999 index", _np_p999(np.arange(1, 1001, dtype=np.float32))[0] == 999.0)

    # --- crop / save dict ---
    big4 = torch.randn(1, 2, 64, 64, generator=g)
    sd, meta, sk, cr = build_save_dict({"a": big4, "b": big4}, never_crop={"b"}, budget=1024, crop=8)
    check("crop 対象は _crop_f32", "a_crop_f32" in sd and sd["a_crop_f32"].shape == (1, 2, 8, 8))
    check("crop meta に元 shape", meta.get("a_full_shape") == "[1, 2, 64, 64]")
    check("never_crop は full", "b_f32" in sd and sd["b_f32"].shape == (1, 2, 64, 64))
    check("左上クロップの値一致", torch.equal(sd["a_crop_f32"], big4[..., :8, :8]))
    v3 = torch.randn(1, 4, 4, generator=g) * 1e6
    sd3, _, sk3, _ = build_save_dict({"c": v3})
    check("FP16 範囲外は f16 省略", "c_f32" in sd3 and "c_f16" not in sd3 and sk3 == ["c"])
    ids = torch.tensor([[1, 2, 3]], dtype=torch.int64)
    sdi, _, _, _ = build_save_dict({"ids": ids})
    check("整数は _i64 のみ", list(sdi.keys()) == ["ids_i64"])

    # --- sigmas ---
    sg = simple_sigmas(3.0, 30)
    check("sigmas 長さ 31・末尾 0", sg.shape == (31,) and float(sg[-1]) == 0.0)
    check("sigmas[0]=1.0 (t=1 は shift 不変)", abs(float(sg[0]) - 1.0) < 1e-6)
    check("sigmas 単調減少", bool((sg[1:] < sg[:-1]).all()))
    check("shift=1 は一様", abs(float(simple_sigmas(1.0, 4)[1]) - 0.75) < 1e-6)
    # shift 式: t=0.5, alpha=3 → 1.5/(1+1.0)=0.75
    check("time_snr_shift 値", abs(float(time_snr_shift(3.0, torch.tensor(0.5))) - 0.75) < 1e-7)

    # --- Euler / CFG ---
    xx = torch.randn(1, 16, 1, 4, 4, generator=g)
    pred = torch.randn_like(xx)
    s0, s1 = torch.tensor(0.8), torch.tensor(0.6)
    den = denoised_from_pred(xx, s0, pred)
    x1 = euler_step(xx, den, s0, s1)
    # 理想: x1 = x - (s0 - s1) * pred
    check("euler_step = x - (s0-s1)*pred", torch.allclose(x1, xx - (s0 - s1) * pred, atol=1e-6))
    check("最終 step は denoised", torch.equal(euler_step(xx, den, s0, torch.tensor(0.0)), den))
    check("cfg g=1 は cond", torch.allclose(cfg_combine(xx, pred, 1.0), pred, atol=1e-6))
    check("cfg g=0 は uncond", torch.allclose(cfg_combine(xx, pred, 0.0), xx, atol=1e-6))

    # --- noise 決定性 ---
    n1 = make_noise(SEED, (1, 16, 1, 8, 8))
    n2 = make_noise(SEED, (1, 16, 1, 8, 8))
    check("noise 同 seed bit 一致", bitwise_equal(n1, n2))
    check("noise 別 seed 不一致", not bitwise_equal(n1, make_noise(SEED + 1, (1, 16, 1, 8, 8))))

    # --- 逆正規化 ---
    z = torch.zeros(1, 2, 1, 2, 2)
    dz = denormalize_latents(z, [1.0, 2.0], [3.0, 4.0])
    check("denormalize: 0*std+mean", torch.equal(dz[0, :, 0, 0, 0], torch.tensor([1.0, 2.0])))

    # --- conv3d → conv2d (T=1) ---
    ok_all = True
    maxerr = 0.0
    for (ci, co, k, p) in [(4, 6, (3, 3, 3), (1, 1, 1)), (4, 5, (1, 1, 1), (0, 0, 0)), (3, 3, (3, 1, 1), (1, 0, 0))]:
        conv = torch.nn.Conv3d(ci, co, k)
        conv.weight.data = torch.randn(conv.weight.shape, generator=g)
        conv.bias.data = torch.randn(conv.bias.shape, generator=g)
        xin = torch.randn(1, ci, 1, 9, 7, generator=g)
        ref = causal_conv3d_reference(xin, conv.weight, conv.bias, p)
        got = causal_conv3d_t1_as_conv2d(xin, conv.weight, conv.bias, p)
        e = float((ref - got).abs().max())
        maxerr = max(maxerr, e)
        ok_all = ok_all and ref.shape == got.shape and e < 1e-5
    check(f"causal conv3d(T=1) == conv2d 最終スライス (max err {maxerr:.2e})", ok_all)
    check("スライス index は 2*pad_t", causal_conv3d_slice_index(2) == 2)
    wt = torch.randn(2, 3, 3, 3, 3, generator=g)
    check("conv3d_to_conv2d_weight は最終スライス", torch.equal(conv3d_to_conv2d_weight(wt, 2), wt[:, :, 2]))
    try:
        conv3d_to_conv2d_weight(torch.randn(2, 3, 1, 3, 3), 2)
        check("核が小さすぎる時は ValueError", False)
    except ValueError:
        check("核が小さすぎる時は ValueError", True)
    # T=1 以外は拒否
    try:
        causal_conv3d_t1_as_conv2d(torch.randn(1, 3, 2, 4, 4), wt, None, (1, 1, 1))
        check("T!=1 は ValueError", False)
    except ValueError:
        check("T!=1 は ValueError", True)

    # --- bit 比較 ---
    a = torch.randn(10, generator=g)
    b = a.clone()
    check("bitwise_equal 同一", bitwise_equal(a, b))
    b[3] = torch.nextafter(b[3], torch.tensor(1e9))
    check("bitwise_equal 1ulp 差を検出", not bitwise_equal(a, b))
    nan1 = torch.tensor([float("nan")])
    check("bitwise_equal NaN 同 bit は一致", bitwise_equal(nan1, nan1.clone()))
    eq, diff, only = compare_tensor_dicts({"x": a, "y": a}, {"x": a.clone(), "z": a})
    check("compare_tensor_dicts: 片側のみ検出", (not eq) and only == ["y", "z"] and diff == [])

    # --- fp16_summary ---
    act = {"a": {"max_abs": 100.0, "fp16_headroom": 655.04, "fp16_overflow": False, "nonfinite": False},
           "b": {"max_abs": 30000.0, "fp16_headroom": 2.18, "fp16_overflow": False, "nonfinite": False}}
    fs = fp16_summary(act)
    check("fp16_summary: 余裕 4 倍未満を列挙", fs["headroom_lt_4x"] == ["b"] and fs["top10_max_abs"][0]["name"] == "b")
    act["c"] = {"max_abs": 7e4, "fp16_headroom": 0.9, "fp16_overflow": True, "nonfinite": False}
    check("fp16_summary: 超過を列挙", fp16_summary(act)["fp16_overflow_or_nonfinite"] == ["c"])

    print("[selftest] ALL PASS" if not fails else f"[selftest] {len(fails)} FAIL: {fails}")
    return not fails


# ============================================================
# 実走 (研究機・anima venv)
# ============================================================
def _fmt(x):
    return f"{x:.4g}"


def run_dump(args, out_io_path):
    """1 回分のダンプ。戻り: (tensors dict (保存前の生), meta dict)。
    終了時に pipe を確実に解放 (gc + empty_cache)。--determinism の 2 回目ロードが OOM しないように。"""
    try:
        return _run_dump_impl(args, out_io_path)
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _offload(module, device):
    """CUDA 時のみ module を CPU へ退避して VRAM を空ける。"""
    if device.startswith("cuda"):
        module.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()


def _run_dump_impl(args, out_io_path):
    import dollma_probe_anima as P
    import dollma_eval_image_grid as G
    from diffusers_anima.pipelines.anima import text_encoding as TE
    from diffusers_anima.pipelines.anima.sigma_schedules import build_sampling_sigmas
    from diffusers_anima.pipelines.anima.sampling import _predict_denoised_const

    torch.set_grad_enabled(False)
    device = args.device
    if device.startswith("cuda"):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        tf32_mode = set_fp32_ieee()
    else:
        tf32_mode = "cpu"
    torch.backends.cudnn.benchmark = False

    prompt = P.build_prompt(MODEL_KEY, args.prompt_id, G.PROMPTS)
    negative = P.build_negative(MODEL_KEY, args.prompt_id, G.PROMPTS)
    ckpt = P.MODEL_SPECS[MODEL_KEY]["ckpt"]
    for p in (ckpt, P.TE_PATH, P.VAE_PATH, P.TOKENIZER_DIR):
        if not os.path.exists(p):
            raise FileNotFoundError(f"必要資産が無い: {p}")

    t0 = time.time()
    pipe = P.load_anima_pipe(ckpt, device=device, dtype="float32")
    tr, te, vae, tok = pipe.transformer, pipe.text_encoder, pipe.vae, pipe.prompt_tokenizer
    for m in (tr, te, vae):
        m.eval()
    load_sec = time.time() - t0
    print(f"[golden] ロード {load_sec:.1f}s device={device} dtype=float32")

    rec = Recorder()
    tensors = OrderedDict()
    core = tr.core
    blocks = core.transformer_blocks

    # ---- 統計 hook ----
    for i, layer in enumerate(te.layers):
        rec.hook_stats(layer, f"te/layer{i:02d}", split_first=True)
        rec.hook_stats_in(layer.mlp.down_proj, f"te/layer{i:02d}_mlp_down_in", split_first=True)
    rec.hook_stats(te.norm, "te/final_norm", split_first=True)
    ad = tr.llm_adapter
    for i, b in enumerate(ad.blocks):
        rec.hook_stats(b, f"adapter/block{i}")
        rec.hook_stats_in(b.mlp[2], f"adapter/block{i}_mlp2_in")
    rec.hook_stats(ad, "adapter/out")
    for i, b in enumerate(blocks):
        rec.hook_stats(b, f"dit/block{i:02d}_out")
        rec.hook_stats(b.attn1, f"dit/block{i:02d}_attn1")
        rec.hook_stats(b.attn2, f"dit/block{i:02d}_attn2")
        rec.hook_stats(b.ff, f"dit/block{i:02d}_ff")
        rec.hook_stats_in(b.ff.net[2], f"dit/block{i:02d}_ff_down_in")      # GELU 後 8192 幅
        for nm in ("norm1", "norm2", "norm3"):
            nmod = getattr(b, nm)
            rec.hook_stats(nmod, f"dit/block{i:02d}_{nm}_mod", index=0)     # 変調後の入力
            rec.hook_adaln_stats(nmod, f"dit/block{i:02d}_{nm}")            # shift/scale/gate
        for an in ("attn1", "attn2"):
            amod = getattr(b, an)
            rec.hook_stats(amod.norm_q, f"dit/block{i:02d}_{an}_q_normed")
            rec.hook_stats(amod.norm_k, f"dit/block{i:02d}_{an}_k_normed")
    rec.hook_stats(core.time_embed, "dit/temb", index=0)
    rec.hook_stats(core.time_embed, "dit/embedded_timestep", index=1)
    rec.hook_stats(core.patch_embed, "dit/patch_embed")
    rec.hook_stats(core.norm_out, "dit/norm_out")
    rec.hook_stats(core.proj_out, "dit/proj_out")
    rec.hook_stats(vae.decoder.mid_block, "vae/mid_block")
    for name, m in vae.decoder.named_modules():
        tn = type(m).__name__
        if tn == "QwenImageResidualBlock" or (name.endswith((".conv1", ".conv2")) and "resnets" in name) \
                or (name.endswith((".conv1", ".conv2")) and "mid_block" in name):
            rec.hook_stats(m, f"vae/{name}")
    for i, ub in enumerate(vae.decoder.up_blocks):
        rec.hook_stats(ub, f"vae/up_block{i}")
    rec.hook_stats(vae.decoder.conv_out, "vae/conv_out")

    # ---- 捕捉 hook (step0 cond のときだけ capture=True) ----
    b0 = blocks[0]
    rec.hook_capture(core.patch_embed, "dit_patch_embed_in", pre_arg=0)
    rec.hook_capture(core.patch_embed, "dit_patch_embed_out")
    rec.hook_capture(core.rope, ["dit_rope_cos", "dit_rope_sin"])
    rec.hook_capture(core.time_embed.time_proj, "dit_time_proj")
    rec.hook_capture(core.time_embed, ["dit_temb", "dit_embedded_timestep"])
    rec.hook_capture(b0, "dit_b0_in_hidden", pre_arg=0)
    rec.hook_capture(b0, "dit_b0_in_ctx", pre_arg=1)
    rec.hook_capture(b0.norm1, ["dit_b0_norm1_mod", "dit_b0_norm1_gate"])
    rec.hook_capture(b0.attn1, "dit_b0_attn1_out")
    rec.hook_capture(b0.norm2, ["dit_b0_norm2_mod", "dit_b0_norm2_gate"])
    rec.hook_capture(b0.attn2, "dit_b0_attn2_out")
    rec.hook_capture(b0.norm3, ["dit_b0_norm3_mod", "dit_b0_norm3_gate"])
    rec.hook_capture(b0.ff, "dit_b0_ff_out")
    rec.hook_capture(b0, "dit_b0_out")
    rec.hook_capture(core.norm_out, "dit_norm_out_out")
    rec.hook_capture(core.proj_out, "dit_proj_out_out")

    # ---- 1. tokens / TE / adapter ----
    def encode(text, tag, capture):
        tk = tok.tokenize_with_weights(text)
        qwen_ids = [t for t, _ in tk["qwen3_06b"][0]]
        t5_ids_l = [t for t, _ in tk["t5xxl"][0]]
        tensors[f"{tag}_qwen_ids"] = torch.tensor(qwen_ids, dtype=torch.int64).unsqueeze(0)
        tensors[f"{tag}_t5_ids"] = torch.tensor(t5_ids_l, dtype=torch.int64).unsqueeze(0)
        rec.phase = f"te_{tag}"
        rec.capture = capture
        h, t5_ids, t5_w = TE.prepare_condition_inputs(
            tok, te, [text], execution_device=device, model_dtype=torch.float32)
        tensors[f"{tag}_te_last_hidden"] = h.float().cpu()
        tensors[f"{tag}_t5_weights"] = t5_w.float().cpu()
        rec.phase = f"adapter_{tag}"
        cond = TE.build_condition(tr, qwen_hidden=h, t5_ids=t5_ids, t5_weights=t5_w)
        tensors[f"{tag}_adapter_out_512"] = cond.float().cpu()
        # pad 前の adapter 出力 (T5 長): 先頭 n 行 (n = T5 token 数) を明示保存
        n = t5_ids.shape[1]
        tensors[f"{tag}_adapter_out_unpadded"] = cond[:, :n].float().cpu()
        rec.phase = None
        rec.capture = False
        return cond, len(qwen_ids), len(t5_ids_l)

    # TE の layer0 / 最終層出力 (pos のみ)
    rec.hook_capture(te.layers[0], "te_pos_layer0_out")
    rec.hook_capture(te.layers[len(te.layers) - 1], "te_pos_layer_last_out")
    rec.hook_capture(ad, "adapter_pos_out_raw")
    pos_cond, n_qwen_pos, n_t5_pos = encode(prompt, "pos", True)
    for k in ("te_pos_layer0_out", "te_pos_layer_last_out", "adapter_pos_out_raw"):
        if k in rec.tensors:
            tensors[k] = rec.tensors.pop(k)
    neg_cond, n_qwen_neg, n_t5_neg = encode(negative, "neg", False)
    print(f"[golden] tokens: pos qwen={n_qwen_pos} t5={n_t5_pos} / neg qwen={n_qwen_neg} t5={n_t5_neg}")
    _offload(te, device)    # TE は以後不要 (VRAM 退避)
    ctx_equal = None        # step0 の dit_b0_in_ctx == pos_adapter_out_512 を後で判定

    # ---- 2. sigmas / noise ----
    height = width = args.size
    lh, lw = height // VAE_SCALE, width // VAE_SCALE
    sigmas = build_sampling_sigmas(pipe.scheduler, num_inference_steps=args.steps,
                                   sigma_schedule=args.sigma_schedule, device="cpu").float()
    sigma_check = None
    if args.sigma_schedule == "simple":
        mine = simple_sigmas(float(pipe.scheduler.config.shift), args.steps,
                             int(pipe.scheduler.config.num_train_timesteps))
        sigma_check = float((mine - sigmas).abs().max())
        print(f"[golden] sigmas: diffusers_anima と自前 simple_sigmas の max差 = {sigma_check:.2e}")
    tensors["sched_sigmas"] = sigmas
    noise = make_noise(args.seed, (1, LATENT_CH, 1, lh, lw))
    tensors["input_noise"] = noise
    x = (noise * sigmas[0]).to(device)
    n_steps = args.steps if args.denoise == "full" else 1
    last_idx = n_steps - 1

    # ---- 3. denoise loop ----
    def dit(xin, sigma, cond, phase, capture):
        rec.phase = phase
        rec.capture = capture
        ts = sigma.to(device=device, dtype=torch.float32).reshape(1)
        out = tr(xin.to(device), ts, encoder_hidden_states=cond.to(device), return_dict=False)[0]
        rec.phase = None
        rec.capture = False
        return out.float()

    t_loop = time.time()
    sec_per_step = []
    denoised = None
    denoised_ref_diff = None
    rec.running_on = True          # 全 step の running max (device 上で torch.maximum)
    for i in range(n_steps):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        ts0 = time.time()
        is_first, is_last = (i == 0), (i == last_idx)
        tensors[f"traj_x_{i:03d}"] = x.float().cpu()     # 各 step の入力 x (A3 の発散 step 特定用)
        if is_first:
            tensors["x_step0"] = x.float().cpu()
        pc = dit(x, sigma, pos_cond, "step0_cond" if is_first else ("last_cond" if is_last else None), is_first)
        if is_first:
            for k, v in rec.tensors.items():
                tensors[k] = v
            rec.tensors.clear()
        pu = dit(x, sigma, neg_cond, "step0_uncond" if is_first else ("last_uncond" if is_last else None), False)
        pred = cfg_combine(pu, pc, args.cfg)
        denoised = denoised_from_pred(x.float(), sigma.to(device), pred)
        if is_first:
            # 手書き denoised と diffusers_anima の _predict_denoised_const の突合 (統計 hook を汚さないよう running を止める)
            rec.running_on = False
            ref_den = _predict_denoised_const(
                tr, x, sigma=sigma.to(device), pos_cond=pos_cond.to(device), neg_cond=neg_cond.to(device),
                guidance_scale=args.cfg, cfg_batch_mode="split", model_dtype=torch.float32)
            rec.running_on = True
            denoised_ref_diff = float((ref_den.float() - denoised).abs().max())
            del ref_den
            print(f"[golden] denoised_step0: 手書き vs diffusers_anima max差 = {denoised_ref_diff:.3e}")
            ctx_equal = bool(torch.equal(tensors["dit_b0_in_ctx"], tensors["pos_adapter_out_512"]))
            if not ctx_equal:
                print("[golden] FAIL: block0 の cross-attn 入力が pos adapter 出力と不一致")
        for tag, cond_flag in (("step0", is_first), ("last", is_last)):
            if cond_flag:
                tensors[f"dit_out_{tag}_cond"] = pc.cpu()
                tensors[f"dit_out_{tag}_uncond"] = pu.cpu()
                tensors[f"dit_out_{tag}_cfg"] = pred.cpu()
                tensors[f"x_{tag}"] = x.float().cpu()
                tensors[f"sigma_{tag}"] = sigma.reshape(1).float()
                tensors[f"denoised_{tag}"] = denoised.cpu()
        x = euler_step(x.float(), denoised, sigma.to(device), sigma_next.to(device))
        if is_first:
            tensors["x_after_step0"] = x.cpu()
        sec_per_step.append(time.time() - ts0)
        print(f"[golden] step {i + 1}/{n_steps} sigma={float(sigma):.4f} {sec_per_step[-1]:.1f}s "
              f"|pred|max={float(pred.abs().max()):.3f}")
    rec.running_on = False
    loop_sec = time.time() - t_loop
    running_max = {k: float(v) for k, v in rec.running.items()}
    del pos_cond, neg_cond
    _offload(tr, device)    # DiT を CPU へ退避してから VAE decode (FP32 の VRAM 超過回避)
    # first モード: 1 step だけなので「最終 latent」は step0 の x0 予測 (denoised) を使う
    latent_final = x if args.denoise == "full" else denoised
    tensors["latent_final"] = latent_final.float().cpu()
    tensors[f"traj_x_{n_steps:03d}"] = latent_final.float().cpu()

    # ---- 4. VAE decode (元 / conv2d 置換) ----
    z = denormalize_latents(latent_final.float(), vae.config.latents_mean, vae.config.latents_std)
    z = z.to(device=device, dtype=torch.float32)
    tensors["vae_in_z"] = z.cpu()
    rec.phase = "decode"
    for name, m in vae.named_modules():
        if name.startswith("decoder.") and name.endswith("time_conv"):
            rec.hook_count(m, f"time_conv:{name}")
    t_vae = time.time()
    out_orig = vae.decode(z, return_dict=False)[0]
    vae_sec = time.time() - t_vae
    rec.phase = None
    out_orig = out_orig.float()
    tensors["vae_out"] = out_orig.cpu()
    t_vae2 = time.time()
    if args.no_vae_check:
        out_c2, vae2_sec = out_orig, 0.0
    else:
        with vae_t1_conv2d_patch(vae):
            out_c2 = vae.decode(z, return_dict=False)[0].float()
        vae2_sec = time.time() - t_vae2
    d = (out_orig - out_c2).abs()
    # uint8 化後の不一致画素率 (decode_latents と同じ丸め)
    def to_u8(o):
        return (((o[:, :, 0].clamp(-1, 1) + 1.0) / 2.0).clamp(0, 1) * 255).round().to(torch.uint8)
    u8_diff = float((to_u8(out_orig) != to_u8(out_c2)).float().mean())
    time_conv_calls = {k: v for k, v in rec.call_counts.items() if k.startswith("time_conv:")}
    vae_check = {
        "max_abs_diff": float(d.max()), "mae": float(d.mean()),
        "uint8_mismatch_ratio": u8_diff,
        "time_conv_total_calls_in_t1_decode": int(sum(time_conv_calls.values())),
        "n_time_conv_modules": len(time_conv_calls),
        "decode_sec_orig": round(vae_sec, 2), "decode_sec_conv2d": round(vae2_sec, 2),
        "note": "conv2d 版 = 全 QwenImageCausalConv3d を weight[:,:,2*pad_t] の conv2d に置換。"
                "time_conv 呼出 0 = upsample3d が T=1 初回チャンクで 'Rep' 扱い (時間方向 2 倍化なし)",
    }
    print(f"[golden] VAE check: max={vae_check['max_abs_diff']:.3e} MAE={vae_check['mae']:.3e} "
          f"u8不一致={u8_diff:.2e} time_conv呼出={vae_check['time_conv_total_calls_in_t1_decode']}")

    # PNG (golden 画像)
    from PIL import Image
    img_np = to_u8(out_orig)[0].permute(1, 2, 0).cpu().numpy()
    png_path = os.path.join(args.out_dir, "anima_golden_decode.png")
    Image.fromarray(img_np).save(png_path)

    rec.remove()

    gates = evaluate_gates(sigma_check, vae_check, ctx_equal, denoised_ref_diff,
                           vae_checked=not args.no_vae_check)
    for gname, gv in gates.items():
        print(f"[golden] GATE {'PASS' if gv['pass'] else 'FAIL'} {gname}: {gv['value']} (limit {gv['limit']})")

    meta = OrderedDict()
    meta["prompt"] = prompt
    meta["negative"] = negative
    meta["prompt_id"] = args.prompt_id
    meta["model"] = MODEL_KEY
    meta["seed"] = args.seed
    meta["size"] = args.size
    meta["latent_shape"] = [1, LATENT_CH, 1, lh, lw]
    meta["steps"] = args.steps
    meta["steps_run"] = n_steps
    meta["denoise_mode"] = args.denoise
    meta["sampler"] = "euler (CONST / rectified flow・split CFG)"
    meta["sigma_schedule"] = args.sigma_schedule
    meta["cfg"] = args.cfg
    meta["scheduler_shift"] = float(pipe.scheduler.config.shift)
    meta["sigmas"] = [float(s) for s in sigmas]
    meta["sigma_selfcheck_max_diff"] = sigma_check
    meta["token_counts"] = {"pos_qwen": n_qwen_pos, "pos_t5": n_t5_pos, "neg_qwen": n_qwen_neg, "neg_t5": n_t5_neg}
    meta["vae_check"] = vae_check
    meta["vae_latents_mean"] = list(vae.config.latents_mean)
    meta["vae_latents_std"] = list(vae.config.latents_std)
    meta["timing_sec"] = {"load": round(load_sec, 1), "loop": round(loop_sec, 1),
                          "per_step": [round(s, 2) for s in sec_per_step], "vae_decode": round(vae_sec, 1)}
    meta["comparability_note"] = ("steps/simple/cfg のみ E-4 に揃えた。E-4 は er_sde・seed 1000-1003・bf16 で、"
                                  "本 golden (Euler・seed 1234・fp32) とは画像を直接比較できない。")
    meta["denoised_step0_ref_diff"] = denoised_ref_diff
    meta["gates"] = gates
    meta["env"] = {"device": device, "dtype": "float32", "torch": torch.__version__, "fp32_mode": tf32_mode,
                   "threads": torch.get_num_threads()}
    try:
        import transformers
        import diffusers
        meta["env"]["transformers"] = transformers.__version__
        meta["env"]["diffusers"] = diffusers.__version__
    except Exception:
        pass
    acts = dict(rec.stats)
    acts.update(running_to_stats(running_max))
    meta["activations"] = acts
    meta["fp16_summary"] = fp16_summary(acts)
    meta["fp16_summary_by_group"] = fp16_summary_grouped(acts)
    return tensors, meta


def save_outputs(args, tensors, meta, io_path):
    io_save, smeta, skipped, cropped = build_save_dict(tensors, croppable=frozenset())
    from safetensors.torch import save_file
    smeta = {**{k: str(v) for k, v in smeta.items()},
             "seed": str(args.seed), "prompt": meta["prompt"], "model": MODEL_KEY}
    save_file(io_save, io_path, metadata=smeta)
    meta["saved_keys"] = sorted(io_save.keys())
    meta["f16_skipped"] = skipped
    meta["cropped"] = [{"name": n, "full": f, "crop": c} for n, f, c in cropped]
    return io_save


def cmd_compare(a_path, b_path):
    from safetensors.torch import load_file
    da, db = load_file(a_path), load_file(b_path)
    eq, diff, only = compare_tensor_dicts(da, db)
    print(f"[compare] {a_path} vs {b_path}: {len(da)} / {len(db)} tensors")
    if eq:
        print("[compare] BIT-EXACT (全 tensor 一致)")
    else:
        print(f"[compare] 不一致: diff={diff[:20]} 片側のみ={only[:20]}")
    return eq


def main():
    ap = argparse.ArgumentParser(description="Anima A0 golden ダンプ (FP32・研究機 anima venv)")
    ap.add_argument("--selftest", action="store_true", help="重み不要の純関数テスト")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), help="2 つの safetensors の bit 一致確認")
    ap.add_argument("--out-dir", dest="out_dir", default=None,
                    help="出力先 (研究機 models/anima/golden/・gitignore 下)")
    ap.add_argument("--device", default="cpu", help="cpu / cuda (cuda でも fp32・TF32 無効)")
    ap.add_argument("--denoise", choices=["full", "first"], default="full",
                    help="full=全 step 実行 / first=最初の 1 step のみ (軽量・最終 latent は step0 の x0 予測)")
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--cfg", type=float, default=DEFAULT_CFG)
    ap.add_argument("--sigma-schedule", dest="sigma_schedule", default="simple",
                    choices=["simple", "beta", "normal", "uniform"])
    ap.add_argument("--size", type=int, default=DEFAULT_SIZE)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--prompt-id", dest="prompt_id", default=PROMPT_ID)
    ap.add_argument("--no-vae-check", dest="no_vae_check", action="store_true",
                    help="conv2d 置換 decode の比較を省略 (時間短縮)")
    ap.add_argument("--determinism", action="store_true", help="同一プロセスで 2 回ダンプして bit 一致を確認")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if args.compare:
        sys.exit(0 if cmd_compare(*args.compare) else 1)
    if not args.out_dir:
        ap.error("--out-dir が必要 (--selftest / --compare を除く)")

    if args.determinism and not args.device.startswith("cuda"):
        ap.error("--determinism (同一プロセス 2 回) は CPU だと RAM ~30GB になるため禁止。"
                 "プロセスを分けて 2 回実行し --compare A B を使うこと")
    os.makedirs(args.out_dir, exist_ok=True)
    io_path = os.path.join(args.out_dir, "anima_io.safetensors")
    t0 = time.time()
    tensors, meta = run_dump(args, io_path)
    io_save = save_outputs(args, tensors, meta, io_path)
    meta["io_safetensors_bytes"] = os.path.getsize(io_path)
    meta["io_sha256"] = sha256_file(io_path)

    if args.determinism:
        print("[golden] 決定性確認: 2 回目のダンプ ...")
        tensors2, _ = run_dump(args, io_path)
        io_save2, _, _, _ = build_save_dict(tensors2, croppable=frozenset())
        eq, diff, only = compare_tensor_dicts(io_save, io_save2)
        meta["determinism"] = {"mode": "same-process-2runs", "bit_exact": bool(eq),
                               "diff_keys": diff[:50], "only_one_side": only[:50]}
        print(f"[golden] 決定性: {'BIT-EXACT' if eq else 'NOT bit-exact: ' + str(diff[:10])}")

    meta["total_sec"] = round(time.time() - t0, 1)
    json_path = os.path.join(args.out_dir, "anima_golden_meta.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)

    fs = meta["fp16_summary"]
    failed = [k for k, v in meta["gates"].items() if not v["pass"]]
    if meta.get("determinism") and not meta["determinism"]["bit_exact"]:
        failed.append("determinism")
    print("\n" + "=" * 70)
    print(f"io     : {io_path} ({meta['io_safetensors_bytes'] / 1e6:.1f} MB) sha256={meta['io_sha256'][:16]}...")
    print(f"meta   : {json_path}")
    print(f"FP16   : {fs['verdict_hint']}")
    for r in fs["top10_max_abs"][:5]:
        print(f"   max|x| {_fmt(r['max_abs']):>10}  {r['name']}")
    print(f"VAE    : conv3d->conv2d(T=1) max={meta['vae_check']['max_abs_diff']:.3e} "
          f"MAE={meta['vae_check']['mae']:.3e}")
    for grp, gs in meta["fp16_summary_by_group"].items():
        print(f"FP16[{grp}]: {gs['verdict_hint']}")
    print("GATES  :", "ALL PASS" if not failed else f"FAIL {failed}")
    print("=" * 70)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

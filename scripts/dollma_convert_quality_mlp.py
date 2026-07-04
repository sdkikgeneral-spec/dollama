#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""dollma_convert_quality_mlp.py — Phase 4 Model B / Package D-1:
自作 QualityMLP (CLIP embed[768] → quality logit・768→64→1) を OpenVINO IR へ変換 +
全 HW latency 実測 + PyTorch↔OV 突合。

背景 (Package C):
  生ピクセル ScorerNet の quality 蒸留は逆相関/不安定 (val corr -0.25〜+0.40) だったが、
  CLIP 空間へ移した自作 QualityMLP は OOF corr +0.5253 で立て直した。
  出荷推論経路: 生成 PNG → CLIP ViT-L image encoder → QualityMLP → sigmoid → renorm。
  本スクリプトはその QualityMLP 部を NPU に載せるための OV 変換。

雛形は dollma_convert_scorer.py (B-3c)。静的形状 [1,768] で reshape してから compile。
入力は CLIP image embed[768] (L2 正規化済) を渡す前提。出力は quality logit [1,1] (sigmoid 前)。

やること:
  1. QualityMLP ロード + state_dict load + eval() + param_count assert (49281)。
  2. ov.convert_model -> 静的 [1,768] reshape -> save FP32/FP16。
  3. 一致確認: PyTorch eval vs OV CPU FP32/FP16 (同一固定入力・gate <= 1e-4)。
  4. 全 HW latency: CPU/iGPU(GPU.0)/RTX5080(GPU.1)/NPU 中央値。
  5. meta 記録 (models/quality-mlp/convert_meta.json)。

出力:
  models/quality-mlp/model_ov_fp32.xml / .bin   (静的 [1,768])
  models/quality-mlp/model_ov_fp16.xml / .bin
  models/quality-mlp/convert_meta.json
"""
import json, os, sys, time
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np
import torch
import openvino as ov

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from dollma_train_quality_mlp import QualityMLP, EMBED_DIM  # noqa: E402

OUTDIR = os.path.join(ROOT, "models", "quality-mlp")
WEIGHTS = os.path.join(ROOT, "data", "scorer", "quality_mlp_fp32.safetensors")
os.makedirs(OUTDIR, exist_ok=True)

STATIC_SHAPE = [1, EMBED_DIM]  # [1,768]
PARAM_COUNT_EXPECTED = 49281
ARCH_HIDDEN = (64,)
GOLDEN_SEED = 20260628


def load_quality_mlp():
    from safetensors.torch import load_file
    model = QualityMLP(hidden=ARCH_HIDDEN, dropout=0.3)
    sd = load_file(WEIGHTS)
    model.load_state_dict(sd, strict=True)
    model.eval()  # dropout 無効化 (推論)
    n = sum(p.numel() for p in model.parameters())
    assert n == PARAM_COUNT_EXPECTED, f"param_count {n} != {PARAM_COUNT_EXPECTED}"
    print(f"[qmlp] QualityMLP params = {n} (== {PARAM_COUNT_EXPECTED}) eval()=ON hidden={ARCH_HIDDEN}")
    return model


def bench_device(core, xml, device, x_np, n=15):
    try:
        cm = core.compile_model(xml, device)
        req = cm.create_infer_request()
        for _ in range(5):
            req.infer({0: x_np})
        ts = []
        for _ in range(n):
            t = time.perf_counter()
            req.infer({0: x_np})
            ts.append((time.perf_counter() - t) * 1000.0)
        out = req.get_output_tensor(0).data.copy()
        return float(np.median(ts)), out, None
    except Exception as e:
        return None, None, f"{type(e).__name__}: {e}"


def main():
    print(f"[src] weights {WEIGHTS}  ({os.path.getsize(WEIGHTS)/1e3:.1f} KB)")
    model = load_quality_mlp()

    # QualityMLP.forward は [B] を返す (squeeze(-1))。OV 変換では [1,1] 維持のため
    # squeeze しないラッパで包む (出力形状を静的 [1,1] に固定)。
    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m.net(x)  # [1,1] logit (squeeze しない)

    wrapped = Wrap(model).eval()

    example = torch.zeros(1, EMBED_DIM)
    t0 = time.perf_counter()
    ov_model = ov.convert_model(wrapped, example_input=example)
    ov_model.reshape({ov_model.input(0): STATIC_SHAPE})
    convert_ms = (time.perf_counter() - t0) * 1000.0

    in_port, out_port = ov_model.input(0), ov_model.output(0)
    in_shape = list(in_port.get_partial_shape().to_shape())
    out_shape = list(out_port.get_partial_shape().to_shape())
    in_et = in_port.get_element_type().get_type_name()
    out_et = out_port.get_element_type().get_type_name()
    print(f"[IR] input  {in_et} {in_shape}  /  output {out_et} {out_shape}")
    assert in_shape == STATIC_SHAPE, f"input {in_shape} != {STATIC_SHAPE}"
    assert out_shape == [1, 1], f"output {out_shape} != [1,1]"
    assert in_et == "f32" and out_et == "f32"

    fp32_xml = os.path.join(OUTDIR, "model_ov_fp32.xml")
    fp16_xml = os.path.join(OUTDIR, "model_ov_fp16.xml")
    ov.save_model(ov_model, fp32_xml, compress_to_fp16=False)
    ov.save_model(ov_model, fp16_xml, compress_to_fp16=True)
    print(f"[save] FP32 {fp32_xml} / FP16 {fp16_xml}  (convert {convert_ms:.0f}ms)")

    # 固定入力 (L2 正規化した [1,768]・実 CLIP embed を模す)。
    g = torch.Generator()
    g.manual_seed(GOLDEN_SEED)
    x = torch.rand(1, EMBED_DIM, generator=g, dtype=torch.float32)
    x = x / x.norm(dim=-1, keepdim=True)
    x_np = np.ascontiguousarray(x.numpy(), dtype=np.float32)
    with torch.no_grad():
        y_torch = wrapped(x).numpy().astype(np.float32)  # [1,1]

    core = ov.Core()
    r32 = core.compile_model(fp32_xml, "CPU").create_infer_request()
    r32.infer({0: x_np})
    y_ov32 = r32.get_output_tensor(0).data.copy().astype(np.float32)
    r16 = core.compile_model(fp16_xml, "CPU").create_infer_request()
    r16.infer({0: x_np})
    y_ov16 = r16.get_output_tensor(0).data.copy().astype(np.float32)

    e32 = float(np.abs(y_torch - y_ov32).max())
    e16 = float(np.abs(y_torch - y_ov16).max())
    print(f"[acc] |PyTorch-OVfp32| = {e32:.3e} (gate <= 1e-4)")
    print(f"[acc] |PyTorch-OVfp16| = {e16:.3e} (記録のみ)")
    assert e32 <= 1e-4, f"FP32 一致ゲート違反 {e32:.3e} > 1e-4"

    # 全 HW latency。
    bench_plan = [("CPU", fp32_xml), ("GPU.0", fp16_xml), ("GPU.1", fp32_xml), ("NPU", fp16_xml)]
    latency = {}
    avail = core.available_devices
    for dev, xml in bench_plan:
        if dev not in avail:
            latency[dev] = {"median_ms": None, "error": "not available"}
            print(f"[lat] {dev:6s} NOT AVAILABLE")
            continue
        ms, out, err = bench_device(core, xml, dev, x_np)
        if ms is None:
            latency[dev] = {"median_ms": None, "error": err}
            print(f"[lat] {dev:6s} FAILED: {err}")
        else:
            dvs = float(np.abs(out - y_ov32).max()) if out is not None else None
            latency[dev] = {"median_ms": round(ms, 3), "ir": os.path.basename(xml),
                            "max_abs_vs_ovfp32": dvs}
            print(f"[lat] {dev:6s} {ms:7.3f} ms (vs OVfp32 |Δ|={dvs:.2e})")

    meta = {
        "model": "QualityMLP (Package C・CLIP embed[768] → quality logit・768→64→1)",
        "task": "Package D-1 (PyTorch -> OV IR + 全 HW latency)",
        "weights_src": os.path.relpath(WEIGHTS, ROOT).replace("\\", "/"),
        "param_count": PARAM_COUNT_EXPECTED,
        "static_shape_in": STATIC_SHAPE, "static_shape_out": [1, 1],
        "input": {"element_type": in_et, "shape": in_shape,
                  "semantics": "CLIP ViT-L image embed[768] (L2 正規化済)"},
        "output": {"element_type": out_et, "shape": out_shape,
                   "semantics": "quality logit (sigmoid 前)。推論は sigmoid で [0,1]"},
        "convert_ms": round(convert_ms, 0),
        "accuracy": {"max_abs_pytorch_vs_ovfp32": e32,
                     "max_abs_pytorch_vs_ovfp16": e16,
                     "gate_pytorch_vs_ovfp32": "<= 1e-4 (PASS)"},
        "latency_median_ms": latency,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": "出荷経路: CLIP image embed → 本 IR → sigmoid → Step① z-sigmoid renorm。",
    }
    json.dump(meta, open(os.path.join(OUTDIR, "convert_meta.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print(f"\n[meta] {os.path.join(OUTDIR, 'convert_meta.json')}")


if __name__ == "__main__":
    main()

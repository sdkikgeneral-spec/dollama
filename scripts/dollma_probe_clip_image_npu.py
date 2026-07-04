#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""dollma_probe_clip_image_npu.py — Phase 4 Model B / Package D-2:
CLIP ViT-L/14 image encoder (open_clip ViT-L-14-quickgelu pretrained=openai) を
OpenVINO IR へ変換 + NPU/iGPU/GPU/CPU latency probe + PyTorch↔OV embed 突合。

背景:
  出荷経路「生成 PNG → CLIP ViT-L image encoder → QualityMLP → sigmoid」の image encoder 部。
  QualityMLP は open_clip encode_image + L2 正規化した embed[768] 空間で蒸留 (Package C) した
  ので、OV image encoder の出力が PyTorch encode_image と厳密一致しないと蒸留が無効になる。
  → corr/max_abs を厳密に突合する。

本命の研究価値:
  現行 CLIP **text** encoder は NPU 7.85ms で採用済 (123M・固定 77tok)。
  image encoder (ViT・Attention 主体) が NPU に載るか (メモリ収支・Attention 対応) は未知。
  probe10/CLIP text と同方式で全 HW latency を採り採用デバイスを判定する。

入力: preprocess 済み画像テンソル [1,3,224,224] (open_clip preprocess 出力に厳密一致)。
出力: image embed[768] (L2 正規化は OV グラフ外)。

やること:
  1. open_clip image tower を encode_image ラッパで OV 変換 -> 静的 [1,3,224,224] -> save。
  2. 一致確認: PyTorch encode_image vs OV CPU (corr/max_abs)。
  3. 全 HW latency: CPU/GPU.0(iGPU)/GPU.1(RTX5080)/NPU 中央値。NPU 失敗時は代替判定。
  4. text encoder (現行 NPU 7.85ms) とのメモリ同居に関する一言メモ。

出力:
  models/clip-image/model_ov_fp32.xml / .bin  (静的 [1,3,224,224])
  models/clip-image/model_ov_fp16.xml / .bin
  models/clip-image/probe_result.json
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
OUTDIR = os.path.join(ROOT, "models", "clip-image")
os.makedirs(OUTDIR, exist_ok=True)

STATIC_SHAPE = [1, 3, 224, 224]
EMBED_DIM = 768
GOLDEN_SEED = 20260628


class ImageTower(torch.nn.Module):
    """open_clip model を encode_image だけ通すラッパ (OV 変換対象)。

    出力は L2 正規化前の image embed[768] (正規化は OV グラフ外で行う=蒸留の教師/生徒とも
    正規化を外に出す設計。Package B harvest も encode_image 後に L2 した)。
    """

    def __init__(self, clip_model):
        super().__init__()
        self.clip = clip_model

    def forward(self, x):
        return self.clip.encode_image(x)  # [1,768]


def bench_device(core, xml, device, x_np, n=11):
    try:
        cm = core.compile_model(xml, device)
        req = cm.create_infer_request()
        for _ in range(3):
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


def l2(v):
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    return v / (np.linalg.norm(v) + 1e-12)


def corr(a, b):
    a = np.asarray(a, np.float64).reshape(-1)
    b = np.asarray(b, np.float64).reshape(-1)
    a = a - a.mean(); b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a @ b) / d) if d else 0.0


def main():
    import open_clip
    print("[clip] loading open_clip ViT-L-14-quickgelu pretrained=openai ...")
    model, _, _ = open_clip.create_model_and_transforms("ViT-L-14-quickgelu", pretrained="openai")
    model.eval()
    tower = ImageTower(model).eval()
    n_params = sum(p.numel() for p in model.visual.parameters())
    print(f"[clip] image tower params ~= {n_params/1e6:.1f}M")

    # nn.MultiheadAttention の fused fastpath (_native_multi_head_attention) は
    # トレース sanity 失敗/ONNX 未対応の元凶。無効化すると素の matmul 系に展開され
    # ov.convert_model の直トレースが通る (ViT-L・probe9 CLIP text と同系の対処)。
    try:
        torch.backends.mha.set_fastpath_enabled(False)
    except Exception:
        pass

    example = torch.zeros(*STATIC_SHAPE)
    t0 = time.perf_counter()
    ov_model = ov.convert_model(tower, example_input=example)
    ov_model.reshape({ov_model.input(0): STATIC_SHAPE})
    convert_ms = (time.perf_counter() - t0) * 1000.0

    in_port, out_port = ov_model.input(0), ov_model.output(0)
    in_shape = list(in_port.get_partial_shape().to_shape())
    out_shape = list(out_port.get_partial_shape().to_shape())
    in_et = in_port.get_element_type().get_type_name()
    out_et = out_port.get_element_type().get_type_name()
    print(f"[IR] input {in_et} {in_shape} / output {out_et} {out_shape}  (convert {convert_ms:.0f}ms)")
    assert in_shape == STATIC_SHAPE, f"input {in_shape} != {STATIC_SHAPE}"
    assert out_shape == [1, EMBED_DIM], f"output {out_shape} != [1,{EMBED_DIM}]"

    fp32_xml = os.path.join(OUTDIR, "model_ov_fp32.xml")
    fp16_xml = os.path.join(OUTDIR, "model_ov_fp16.xml")
    ov.save_model(ov_model, fp32_xml, compress_to_fp16=False)
    ov.save_model(ov_model, fp16_xml, compress_to_fp16=True)
    print(f"[save] FP32 {os.path.getsize(fp32_xml.replace('.xml','.bin'))/1e6:.0f}MB / "
          f"FP16 {os.path.getsize(fp16_xml.replace('.xml','.bin'))/1e6:.0f}MB bin")

    # 固定入力 [1,3,224,224] (前処理済みを模す乱数)。
    g = torch.Generator()
    g.manual_seed(GOLDEN_SEED)
    x = torch.rand(*STATIC_SHAPE, generator=g, dtype=torch.float32)
    x_np = np.ascontiguousarray(x.numpy(), dtype=np.float32)
    with torch.no_grad():
        y_torch = tower(x).numpy().astype(np.float32)  # [1,768]

    core = ov.Core()
    r32 = core.compile_model(fp32_xml, "CPU").create_infer_request()
    r32.infer({0: x_np})
    y_ov32 = r32.get_output_tensor(0).data.copy().astype(np.float32)

    e32 = float(np.abs(y_torch - y_ov32).max())
    c32 = corr(y_torch, y_ov32)
    # L2 正規化後の embed の一致 (蒸留で実際に使うのは正規化後)。
    e32_l2 = float(np.abs(l2(y_torch) - l2(y_ov32)).max())
    print(f"[acc] |PyTorch-OVfp32| raw = {e32:.3e}  corr = {c32:.6f}  L2後|Δ|max = {e32_l2:.3e}")
    # embed 空間の一致ゲート: 蒸留が有効であるには corr≈1 かつ raw max_abs 小。
    assert c32 > 0.9999, f"embed corr {c32} <= 0.9999 (蒸留無効化リスク)"

    # 全 HW latency + 各デバイス embed の PyTorch との corr。
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
            latency[dev] = {"median_ms": None, "error": err, "ir": os.path.basename(xml)}
            sev = "  !!! image encoder が NPU に載らない" if dev == "NPU" else ""
            print(f"[lat] {dev:6s} FAILED: {err}{sev}")
        else:
            dcorr = corr(out, y_torch) if out is not None else None
            latency[dev] = {"median_ms": round(ms, 2), "ir": os.path.basename(xml),
                            "corr_vs_pytorch": round(dcorr, 6) if dcorr is not None else None}
            print(f"[lat] {dev:6s} {ms:8.2f} ms  (corr vs PyTorch {dcorr:.5f})")

    # 採用デバイス判定 (probe10/CLIP text と同方式: 最小 latency・ただし embed corr>0.999 を満たすもの)。
    cand = {d: v["median_ms"] for d, v in latency.items()
            if v.get("median_ms") is not None and (v.get("corr_vs_pytorch") or 1.0) > 0.999}
    chosen = min(cand, key=cand.get) if cand else None
    npu = latency.get("NPU", {})
    npu_ok = npu.get("median_ms") is not None
    print(f"\n[verdict] 採用デバイス = {chosen} ({cand.get(chosen)}ms)  "
          f"| NPU {'OK '+str(npu.get('median_ms'))+'ms' if npu_ok else 'NG: '+str(npu.get('error'))}")
    print(f"[verdict] text encoder NPU 7.85ms との対比: image encoder NPU = "
          f"{npu.get('median_ms') if npu_ok else '不可'}")

    meta = {
        "model": "CLIP ViT-L/14 image encoder (open_clip ViT-L-14-quickgelu pretrained=openai)",
        "task": "Package D-2 (image tower -> OV IR + NPU/HW latency probe)",
        "image_tower_params_M": round(n_params / 1e6, 1),
        "static_shape_in": STATIC_SHAPE, "static_shape_out": [1, EMBED_DIM],
        "input": {"element_type": in_et, "shape": in_shape,
                  "note": "open_clip preprocess 出力 [1,3,224,224]。L2 正規化は OV グラフ外。"},
        "output": {"element_type": out_et, "shape": out_shape,
                   "note": "image embed[768] (L2 正規化前)。QualityMLP 蒸留と同空間。"},
        "convert_ms": round(convert_ms, 0),
        "accuracy": {"max_abs_pytorch_vs_ovfp32_raw": e32,
                     "corr_pytorch_vs_ovfp32": c32,
                     "max_abs_l2normalized": e32_l2,
                     "gate_corr": "> 0.9999 (PASS・蒸留空間一致)"},
        "latency_median_ms": latency,
        "verdict": {
            "chosen_device": chosen,
            "chosen_ms": cand.get(chosen),
            "npu_ok": npu_ok,
            "npu_ms": npu.get("median_ms"),
            "npu_error": npu.get("error"),
            "vs_text_encoder": "text encoder は NPU 7.85ms・123M で採用済 (固定 77tok)。",
            "method": "probe10/CLIP text と同方式: embed corr>0.999 を満たすうち最小 latency。",
        },
        "memory_note": (
            "text/image encoder は別々 load でも可 (生成後の遊休窓で image を回す設計・同時常駐必須でない)。"
            " NPU オンチップメモリに ViT-L image tower (~304M) が載るかは本 probe の NPU compile 可否で判定。"
        ),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    json.dump(meta, open(os.path.join(OUTDIR, "probe_result.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print(f"[meta] {os.path.join(OUTDIR, 'probe_result.json')}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""dollma_convert_scorer.py — Phase 4 Model B / B-3c:
ScorerNet (PyTorch 純 conv ResNet-18 級 11.18M) -> OpenVINO IR 変換 + 全 HW レイテンシ実測 + golden 確定。

背景 (B-3b):
  ScorerNet 出力 [1, 1+8]=9: index0=quality logit / index1..8=解剖 8 軸 logit
  (AnomalyAxis Hands/Limbs/Head/Eyes/Ears/Mouth/Digits/GlobalAnatomy に 1:1)。
  重みは data/scorer/scorer_net_fp32.safetensors (B-3b 訓練済)。

この機 = 研究機 (OV2026.2・CPU/GPU.0=iGPU Xe/GPU.1=RTX5080/NPU 全可視)。
B の芯 = 純 conv スコアラが NPU に静的形状で載ること
  (probe 4-B 予測: NPU 512² 6.14ms・NPU/CPU 0.55x)。

雛形は dollma_convert_matting.py (M-1)。ただし matting は ONNX ソース、本件は
PyTorch モデルなので ov.convert_model(model, example_input=...) 経路。

最大の地雷: BN running stats。model.eval() を必ず呼ぶ (漏れると IR が訓練時統計で
焼かれ golden がズレる)。

やること:
  1. ScorerNet ロード + state_dict load + model.eval() + param_count assert。
  2. ov.convert_model -> 静的 [1,3,512,512] reshape -> save_model FP32/FP16。
  3. 入出力メタ記録 (出力 [1,9] f32 assert・i32/i64 教訓警告)。
  4. 一致確認: PyTorch eval vs OV CPU FP32/FP16 (同一生テンソルを両経路へ)。
  5. 全 HW レイテンシ: CPU/iGPU(GPU.0)/RTX5080(GPU.1)/NPU 中央値 (N>=7・512²)。
  6. golden 確定: 固定 seed 乱数入力 -> OV IR(FP32) CPU 出力を正として safetensors+meta。

出力:
  models/scorer-net/model_ov_fp32.xml / .bin   (静的 [1,3,512,512])
  models/scorer-net/model_ov_fp16.xml / .bin   (NPU/iGPU 向け)
  src/tests/data/scorer/golden_scorernet.safetensors   (input + logits)
  src/tests/data/scorer/golden_scorernet_meta.json     (形状/型/誤差/全HWレイテンシ/8軸表/警告)
"""
import json, os, struct, sys, time
# Windows cp932 コンソールでも ² 等を出力できるよう stdout を utf-8 に。
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import numpy as np
import torch
import openvino as ov

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from train_scorer import ScorerNet, N_OUT, IMG_SIZE, AXIS_ORDER  # noqa: E402

OUTDIR = os.path.join(ROOT, "models", "scorer-net")
DATA = os.path.join(ROOT, "src", "tests", "data", "scorer")
WEIGHTS = os.path.join(ROOT, "data", "scorer", "scorer_net_fp32.safetensors")
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(DATA, exist_ok=True)

STATIC_SHAPE = [1, 3, IMG_SIZE, IMG_SIZE]
PARAM_COUNT_EXPECTED = 11_181_129

# golden 固定入力 seed (決定的)。
GOLDEN_SEED = 20260628


# ----------------------------------------------------------------------------
# safetensors writer (最小・C++ src/io/safetensors.hpp が読める形式)
#   [0..8) uint64 LE = JSON ヘッダ長 N / [8..8+N) UTF-8 JSON / [8+N..) raw data
#   (dollma_convert_matting.py から流用・F32 のみ)
# ----------------------------------------------------------------------------
def save_safetensors(path, tensors):
    header = {}
    blob = bytearray()
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr)
        assert arr.dtype == np.float32, f"{name} must be f32"
        begin = len(blob)
        blob += arr.tobytes()
        end = len(blob)
        header[name] = {"dtype": "F32", "shape": list(arr.shape),
                        "data_offsets": [begin, end]}
    hjson = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        f.write(blob)


def load_scorer():
    """ScorerNet を生成し B-3b 重みをロード。model.eval() 必須・param_count assert。"""
    from safetensors.torch import load_file

    model = ScorerNet()
    sd = load_file(WEIGHTS)
    missing, unexpected = model.load_state_dict(sd, strict=True)
    # strict=True なので例外で弾かれるが念のため。
    assert not missing and not unexpected, f"state_dict mismatch: {missing} {unexpected}"

    # ★ 最大の地雷: BN running stats。eval() を必ず呼ぶ。
    model.eval()

    n = sum(p.numel() for p in model.parameters())
    assert n == PARAM_COUNT_EXPECTED, \
        f"param_count {n} != expected {PARAM_COUNT_EXPECTED}"
    print(f"[scorer] ScorerNet params = {n} (== {PARAM_COUNT_EXPECTED}) eval()=ON")
    return model


def bench_device(core, xml, device, x_np, n=11):
    """compile_model + warmup + 中央値レイテンシ。失敗時は (None, err文字列)。"""
    try:
        cm = core.compile_model(xml, device)
        req = cm.create_infer_request()
        # warmup
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


def main():
    print(f"[src] weights {WEIGHTS}  ({os.path.getsize(WEIGHTS)/1e6:.1f} MB)")
    model = load_scorer()

    # ------------------------------------------------------------
    # 1. PyTorch -> OV IR (静的 [1,3,512,512] を明示)
    # ------------------------------------------------------------
    example = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
    t0 = time.perf_counter()
    ov_model = ov.convert_model(model, example_input=example)
    ov_model.reshape({ov_model.input(0): STATIC_SHAPE})
    convert_ms = (time.perf_counter() - t0) * 1000.0

    in_port = ov_model.input(0)
    out_port = ov_model.output(0)
    def safe_name(port, fallback):
        try:
            return port.get_any_name()
        except Exception:
            # PyTorch 変換モデルは出力に名前が無いことがある (index でアクセス)。
            return fallback

    in_name = safe_name(in_port, "input.0")
    out_name = safe_name(out_port, "output.0")
    in_shape = list(in_port.get_partial_shape().to_shape())
    out_shape = list(out_port.get_partial_shape().to_shape())
    in_et = in_port.get_element_type().get_type_name()
    out_et = out_port.get_element_type().get_type_name()
    print(f"[IR] input  '{in_name}' {in_et} {in_shape}")
    print(f"[IR] output '{out_name}' {out_et} {out_shape}")

    # 検証ゲート: 出力 [1,9] / f32。
    assert in_shape == STATIC_SHAPE, f"input shape {in_shape} != {STATIC_SHAPE}"
    assert out_shape == [1, N_OUT], f"output shape {out_shape} != [1,{N_OUT}]"
    assert in_et == "f32", f"input et {in_et} != f32"
    assert out_et == "f32", f"output et {out_et} != f32"

    fp32_xml = os.path.join(OUTDIR, "model_ov_fp32.xml")
    fp16_xml = os.path.join(OUTDIR, "model_ov_fp16.xml")
    ov.save_model(ov_model, fp32_xml, compress_to_fp16=False)
    ov.save_model(ov_model, fp16_xml, compress_to_fp16=True)
    print(f"[save] FP32 {fp32_xml}  ({os.path.getsize(fp32_xml.replace('.xml','.bin'))/1e6:.1f} MB bin)")
    print(f"[save] FP16 {fp16_xml}  ({os.path.getsize(fp16_xml.replace('.xml','.bin'))/1e6:.1f} MB bin)")
    print(f"[convert] {convert_ms:.0f} ms")

    # ------------------------------------------------------------
    # 2. 一致確認用の決定的固定入力 (golden 入力も兼ねる)。
    #    固定 seed の torch.Generator で [1,3,512,512] [0,1] 乱数を生成し、
    #    同一の生テンソルを PyTorch / OV 両経路へ渡す (前処理は挟まない)。
    # ------------------------------------------------------------
    g = torch.Generator()
    g.manual_seed(GOLDEN_SEED)
    x_torch = torch.rand(1, 3, IMG_SIZE, IMG_SIZE, generator=g, dtype=torch.float32)
    x_np = np.ascontiguousarray(x_torch.numpy(), dtype=np.float32)

    with torch.no_grad():
        y_torch = model(x_torch).numpy().astype(np.float32)  # [1,9]

    core = ov.Core()

    # OV CPU FP32 (= golden の正) / FP16。
    cm_fp32 = core.compile_model(fp32_xml, "CPU")
    cm_fp16 = core.compile_model(fp16_xml, "CPU")
    r32 = cm_fp32.create_infer_request()
    r32.infer({0: x_np})
    y_ov32 = r32.get_output_tensor(0).data.copy().astype(np.float32)
    r16 = cm_fp16.create_infer_request()
    r16.infer({0: x_np})
    y_ov16 = r16.get_output_tensor(0).data.copy().astype(np.float32)

    e_torch_ov32 = float(np.abs(y_torch - y_ov32).max())
    e_ov32_ov16 = float(np.abs(y_ov32 - y_ov16).max())
    e_torch_ov16 = float(np.abs(y_torch - y_ov16).max())
    print(f"[acc] |PyTorch-OVfp32| = {e_torch_ov32:.3e}  (gate <= 1e-4)")
    print(f"[acc] |OVfp32-OVfp16|  = {e_ov32_ov16:.3e}  (gate 外・記録のみ)")
    print(f"[acc] |PyTorch-OVfp16| = {e_torch_ov16:.3e}")

    # 検証ゲート: PyTorch eval vs OV CPU FP32 max abs err <= 1e-4。
    assert e_torch_ov32 <= 1e-4, \
        f"FP32 一致ゲート違反 {e_torch_ov32:.3e} > 1e-4 (eval() 漏れ/前処理不一致を疑え)"

    # ------------------------------------------------------------
    # 3. 全 HW レイテンシ実測 (CPU / GPU.0=iGPU Xe / GPU.1=RTX5080 / NPU)。
    #    NPU compile 成功 = B の芯の実証。FP16 IR を NPU/iGPU へ載せる。
    # ------------------------------------------------------------
    # CPU は FP32、iGPU/NPU/GPU.1 は FP16 IR を使う (NPU/iGPU は FP16 ネイティブ)。
    bench_plan = [
        ("CPU", fp32_xml),
        ("GPU.0", fp16_xml),  # iGPU Xe
        ("GPU.1", fp32_xml),  # RTX5080 (FP16 IR は intel_gpu プラグインが kernel 選択失敗→FP32)
        ("NPU", fp16_xml),
    ]
    latency = {}
    avail = core.available_devices
    for dev, xml in bench_plan:
        if dev not in avail:
            latency[dev] = {"median_ms": None, "error": "device not available",
                            "ir": os.path.basename(xml)}
            print(f"[lat] {dev:6s}  NOT AVAILABLE")
            continue
        ms, out, err = bench_device(core, xml, dev, x_np)
        if ms is None:
            latency[dev] = {"median_ms": None, "error": err, "ir": os.path.basename(xml)}
            sev = " !!! 重大 (B の芯)" if dev == "NPU" else ""
            print(f"[lat] {dev:6s}  FAILED: {err}{sev}")
        else:
            # FP16 デバイス出力と FP32 golden の差も記録 (参考)。
            dvs_g = float(np.abs(out - y_ov32).max()) if out is not None else None
            latency[dev] = {"median_ms": round(ms, 2),
                            "ir": os.path.basename(xml),
                            "max_abs_vs_ovfp32": dvs_g}
            print(f"[lat] {dev:6s}  {ms:7.2f} ms  (vs OVfp32 |Δ|={dvs_g:.2e})")

    # probe 4-B 予測との突合コメント。
    npu_ms = latency.get("NPU", {}).get("median_ms")
    cpu_ms = latency.get("CPU", {}).get("median_ms")
    probe_note = "probe 4-B 予測: NPU 512² 6.14ms / NPU/CPU 0.55x"
    if npu_ms and cpu_ms:
        ratio = npu_ms / cpu_ms
        probe_note += f" | 実測 NPU {npu_ms}ms / CPU {cpu_ms}ms / NPU/CPU {ratio:.2f}x"
    print(f"[probe] {probe_note}")

    # ------------------------------------------------------------
    # 4. golden 確定 (safetensors: C++ src/io/safetensors.hpp が読める)。
    #    input  : [1,3,512,512] f32 ([0,1]) — C++ B-3e がそのまま IR へ渡す。
    #    logits : [1,9] f32 — OV IR(FP32) CPU 出力 = 期待値。
    # ------------------------------------------------------------
    gpath = os.path.join(DATA, "golden_scorernet.safetensors")
    save_safetensors(gpath, {
        "input": x_np.astype(np.float32),
        "logits": np.ascontiguousarray(y_ov32, np.float32),
    })

    axis_map = {str(i + 1): AXIS_ORDER[i] for i in range(len(AXIS_ORDER))}
    meta = {
        "model": "ScorerNet (B-3b 純 conv ResNet-18 級・品質+解剖8軸)",
        "task": "Phase 4 Model B / B-3c (PyTorch -> OV IR 変換 + 全 HW 実測 + golden)",
        "weights_src": os.path.relpath(WEIGHTS, ROOT).replace("\\", "/"),
        "param_count": PARAM_COUNT_EXPECTED,
        "eval_mode": "model.eval() 適用済 (BN running stats 使用・訓練統計焼き込み回避)",
        "ir_fp32": os.path.relpath(fp32_xml, ROOT).replace("\\", "/"),
        "ir_fp16": os.path.relpath(fp16_xml, ROOT).replace("\\", "/"),
        "input": {"name": in_name, "element_type": in_et, "shape": in_shape,
                  "layout": "NCHW", "range": "[0,1] (train_scorer は img/255 のみ・追加正規化なし)"},
        "output": {"name": out_name, "element_type": out_et, "shape": out_shape,
                   "semantics": "index0=quality logit (sigmoid 前) / index1..8=解剖8軸 logit",
                   "axis_map": axis_map},
        "convert_ms": round(convert_ms, 0),
        "accuracy": {
            "max_abs_pytorch_vs_ovfp32": e_torch_ov32,
            "max_abs_ovfp32_vs_ovfp16": e_ov32_ov16,
            "max_abs_pytorch_vs_ovfp16": e_torch_ov16,
            "gate_pytorch_vs_ovfp32": "<= 1e-4 (PASS)",
        },
        "latency_median_ms": latency,
        "probe_4b": {
            "prediction": "NPU 512² 6.14ms / NPU/CPU 0.55x",
            "measured_npu_ms": npu_ms,
            "measured_cpu_ms": cpu_ms,
            "npu_over_cpu": round(npu_ms / cpu_ms, 3) if (npu_ms and cpu_ms) else None,
        },
        "golden": {
            "file": "golden_scorernet.safetensors",
            "format": "safetensors (F32). src/io/safetensors.hpp で読める。",
            "seed": GOLDEN_SEED,
            "input_gen": f"torch.Generator(seed={GOLDEN_SEED}) rand [1,3,512,512] [0,1]",
            "tensors": {
                "input": {"dtype": "F32", "shape": list(x_np.shape),
                          "note": "決定的固定入力。C++ B-3e はこれを IR 入力へコピーして渡す。"},
                "logits": {"dtype": "F32", "shape": list(y_ov32.shape),
                           "note": "OV IR(FP32) CPU 出力 = 期待 logits。C++ は max abs err で突合。"},
            },
            "note": "golden は OV IR(FP32) CPU 出力を正とする (C++ も同じ IR を使うので一致するはず)。",
        },
        "io_warning_for_B3e": (
            "入力 element_type は f32 ([1,3,512,512] NCHW)。C++ では ov::element::f32 で "
            "テンソルを作り IR と厳密一致させること (CLIP の i32/i64 0xC0000409 教訓)。"
            "出力は f32 [1,9]、get_output_tensor(0)。index0=quality / index1..8=8軸 logit。"
        ),
    }
    json.dump(meta, open(os.path.join(DATA, "golden_scorernet_meta.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    print(f"\n[golden] {gpath}  ({os.path.getsize(gpath)/1e6:.3f} MB)")
    print(f"[meta]   {os.path.join(DATA, 'golden_scorernet_meta.json')}")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

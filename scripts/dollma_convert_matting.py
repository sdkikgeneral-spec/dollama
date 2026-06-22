#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dollma_convert_matting.py — マッティング M-1: ISNet-anime ONNX -> OpenVINO IR 変換 + golden 確定

採用モデル (M0-2 probe / ユーザー確定):
  ISNet-anime = skytnt/anime-seg の isnetis.onnx (176MB, Apache-2.0)
  入出力 (probe 実確認・静的形状):
    入力  : [1,3,1024,1024] f32  ([0,1] レンジ, img/255 のみ。正規化なし)
    出力  : [1,1,1024,1024] f32  (sigmoid 済み soft マスク [0,1])

この PC: GTX1080Ti / i7-10700 / NPU なし / nvcc なし。OV 変換・推論は CPU で行う。
  Python 3.14 に onnxruntime 1.26.0 + openvino 2026.2.1 を導入済み。

やること:
  1. ONNX -> OV IR (FP32 と FP16 両方)。入力静的形状 [1,3,1024,1024] を明示。
     (CLIP の教訓: NPU は静的形状必須。研究機で NPU/iGPU に載せる前提で静的化必須)
  2. OV CPU でロード&推論。ONNX Runtime 出力との一致 (max abs err) を確認。
     入出力 element_type/shape を記録 (C++ M-4 が厳密一致で渡すため・i32/i64 教訓)。
  3. golden 確定: 固定サンプル画像 -> 期待 soft マスクを safetensors で保存。
     C++ M-4 が src/io/safetensors.hpp でそのまま読める形式。golden は OV IR(FP32) の出力を正とする。

出力:
  models/isnet-anime/model_ov_fp32.xml / .bin   (静的 [1,3,1024,1024])
  models/isnet-anime/model_ov_fp16.xml / .bin   (研究機 NPU/iGPU 向け)
  src/tests/data/matting/golden_isnet.safetensors  (input + 期待マスク)
  src/tests/data/matting/golden_isnet_meta.json    (形式・形状・型・誤差メタ)
"""
import json, os, struct, time
import numpy as np
from PIL import Image
import onnxruntime as ort
import openvino as ov

HERE   = os.path.dirname(os.path.abspath(__file__))
ROOT   = os.path.abspath(os.path.join(HERE, ".."))
OUTDIR = os.path.join(ROOT, "models", "isnet-anime")
DATA   = os.path.join(ROOT, "src", "tests", "data", "matting")
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(DATA, exist_ok=True)

H = W = 1024
STATIC_SHAPE = [1, 3, H, W]


def pre_isnet(img):
    # probe / 公式 inference.py と同一: img/255 のみ ([0,1]), NCHW
    a = np.asarray(img.resize((W, H), Image.BILINEAR), np.float32) / 255.0
    return np.ascontiguousarray(a.transpose(2, 0, 1)[None], np.float32)


# ----------------------------------------------------------------------------
# safetensors writer (最小・C++ src/io/safetensors.hpp が読める形式)
#   [0..8) uint64 LE = JSON ヘッダ長 N / [8..8+N) UTF-8 JSON / [8+N..) raw data
# ----------------------------------------------------------------------------
def save_safetensors(path, tensors):
    # tensors: dict name -> np.ndarray (F32)
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
    # 8 バイト境界に合わせる必要はないが安全のためパディング不要 (仕様上任意)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        f.write(blob)


def main():
    paths = json.load(open(os.path.join(HERE, "_matting_paths.json")))
    onnx_path = paths["isnet"]
    print(f"[src] {onnx_path}  ({os.path.getsize(onnx_path)/1e6:.1f} MB)")

    # ------------------------------------------------------------
    # 1. ONNX -> OV IR (静的 [1,3,1024,1024] を明示)
    # ------------------------------------------------------------
    t0 = time.perf_counter()
    ov_model = ov.convert_model(onnx_path, input=[("img", STATIC_SHAPE)])
    # input 名が "img" でない場合に備えフォールバック: 名前指定なしで reshape
    ov_model.reshape({ov_model.input(0): STATIC_SHAPE})
    convert_ms = (time.perf_counter() - t0) * 1000

    in_port = ov_model.input(0)
    out_port = ov_model.output(0)
    in_name = in_port.get_any_name()
    out_name = out_port.get_any_name()
    in_shape = list(in_port.get_partial_shape().to_shape())
    out_shape = list(out_port.get_partial_shape().to_shape())
    in_et = in_port.get_element_type().get_type_name()
    out_et = out_port.get_element_type().get_type_name()
    print(f"[IR] input  '{in_name}' {in_et} {in_shape}")
    print(f"[IR] output '{out_name}' {out_et} {out_shape}")

    fp32_xml = os.path.join(OUTDIR, "model_ov_fp32.xml")
    fp16_xml = os.path.join(OUTDIR, "model_ov_fp16.xml")
    ov.save_model(ov_model, fp32_xml, compress_to_fp16=False)
    ov.save_model(ov_model, fp16_xml, compress_to_fp16=True)
    print(f"[save] FP32 {fp32_xml}  ({os.path.getsize(fp32_xml.replace('.xml','.bin'))/1e6:.1f} MB bin)")
    print(f"[save] FP16 {fp16_xml}  ({os.path.getsize(fp16_xml.replace('.xml','.bin'))/1e6:.1f} MB bin)")
    print(f"[convert] {convert_ms:.0f} ms")

    # ------------------------------------------------------------
    # 2. 推論一致確認: ONNX Runtime vs OV CPU (FP32 / FP16)
    #    固定サンプル: probe が生成した input_real_anime.png (実アニメ・golden に最適)
    #    + 合成 input_simple.png (前景被覆あり) もチェック
    # ------------------------------------------------------------
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_in = sess.get_inputs()[0].name

    core = ov.Core()
    cm_fp32 = core.compile_model(fp32_xml, "CPU")
    cm_fp16 = core.compile_model(fp16_xml, "CPU")

    samples = ["input_real_anime.png", "input_simple.png", "input_white.png", "input_grey.png"]
    samples = [s for s in samples if os.path.exists(os.path.join(DATA, s))]

    errs = {}
    golden_name = "input_real_anime.png" if "input_real_anime.png" in samples else samples[0]
    golden_x = None
    golden_ov_mask = None

    for s in samples:
        img = Image.open(os.path.join(DATA, s)).convert("RGB")
        x = pre_isnet(img)

        # ONNX Runtime (正の比較基準)
        onnx_out = sess.run(None, {onnx_in: x})[0]

        # OV CPU FP32 (= golden の正)
        r32 = cm_fp32.create_infer_request()
        # warmup + 計測
        r32.infer({0: x})
        ts = []
        for _ in range(5):
            t = time.perf_counter(); r32.infer({0: x}); ts.append((time.perf_counter()-t)*1000)
        ov32_out = r32.get_output_tensor(0).data.copy()
        ov32_ms = float(np.median(ts))

        # OV CPU FP16
        r16 = cm_fp16.create_infer_request()
        r16.infer({0: x})
        ov16_out = r16.get_output_tensor(0).data.copy()

        e_onnx_ov32 = float(np.abs(onnx_out - ov32_out).max())
        e_ov32_ov16 = float(np.abs(ov32_out - ov16_out).max())
        e_onnx_ov16 = float(np.abs(onnx_out - ov16_out).max())
        errs[s] = dict(max_abs_onnx_vs_ovfp32=e_onnx_ov32,
                       max_abs_ovfp32_vs_ovfp16=e_ov32_ov16,
                       max_abs_onnx_vs_ovfp16=e_onnx_ov16,
                       ov_fp32_cpu_ms=round(ov32_ms, 1))
        print(f"[{s:22s}] OVfp32 {ov32_ms:6.1f}ms | "
              f"|ONNX-OVfp32|={e_onnx_ov32:.2e} |OVfp32-OVfp16|={e_ov32_ov16:.2e}")

        if s == golden_name:
            golden_x = x
            golden_ov_mask = ov32_out  # golden = OV IR(FP32) 出力を正とする

    # ------------------------------------------------------------
    # 3. golden 確定 (safetensors: C++ src/io/safetensors.hpp が読める)
    #    input  : [1,3,1024,1024] f32 ([0,1], NCHW) — C++ M-4 がそのまま IR へ渡す
    #    mask   : [1,1,1024,1024] f32 ([0,1] soft) — OV IR(FP32) 出力 = 期待値
    # ------------------------------------------------------------
    gpath = os.path.join(DATA, "golden_isnet.safetensors")
    save_safetensors(gpath, {
        "input": golden_x.astype(np.float32),
        "mask":  np.ascontiguousarray(golden_ov_mask, np.float32),
    })

    meta = {
        "model": "ISNet-anime (skytnt/anime-seg isnetis.onnx)",
        "license": "Apache-2.0",
        "onnx_src": onnx_path,
        "ir_fp32": os.path.relpath(fp32_xml, ROOT).replace("\\", "/"),
        "ir_fp16": os.path.relpath(fp16_xml, ROOT).replace("\\", "/"),
        "input":  {"name": in_name,  "element_type": in_et,  "shape": in_shape,
                   "layout": "NCHW", "range": "[0,1] = img/255 (正規化なし)"},
        "output": {"name": out_name, "element_type": out_et, "shape": out_shape,
                   "semantics": "sigmoid 済み soft マスク [0,1]"},
        "golden": {
            "file": "golden_isnet.safetensors",
            "format": "safetensors (F32). src/io/safetensors.hpp で読める。",
            "source_image": golden_name,
            "tensors": {
                "input": {"dtype": "F32", "shape": list(golden_x.shape),
                          "note": "前処理済み NCHW [0,1]。C++ M-4 はこれを IR 入力にコピーして渡す。"},
                "mask":  {"dtype": "F32", "shape": list(golden_ov_mask.shape),
                          "note": "OV IR(FP32) CPU 出力 = 期待 soft マスク。C++ は IoU/MAE で突合。"},
            },
            "note": "golden は OV IR(FP32) の出力を正とする (C++ も同じ IR を使うので一致するはず)。",
        },
        "convert_ms": round(convert_ms, 0),
        "accuracy": errs,
        "io_warning_for_M4": (
            "入力 element_type は f32 ([1,3,1024,1024] NCHW)。C++ では ov::element::f32 で "
            "テンソルを作り IR と厳密一致させること (CLIP の i32/i64 0xC0000409 教訓)。"
            "出力は f32 [1,1,1024,1024]、get_output_tensor(0)。"
        ),
    }
    json.dump(meta, open(os.path.join(DATA, "golden_isnet_meta.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    print(f"\n[golden] {gpath}  ({os.path.getsize(gpath)/1e6:.1f} MB)")
    print(f"[meta]   {os.path.join(DATA, 'golden_isnet_meta.json')}")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

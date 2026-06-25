#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dollma_probe_matting_device.py — マッティング M-5: ISNet-anime の HW デバイス比較 probe

roadmap.md マッティング段 / M-5「NPU/iGPU HW 比較 (matting_device 確定)」を埋める probe。
この研究機 (Intel Core Ultra 9 285K + NPU AI Boost + RTX5080) でしか測れない。

問い:
  透過 PNG 切り抜きモデル ISNet-anime (skytnt/anime-seg isnetis.onnx, 176MB, Apache-2.0,
  入力 [1,3,1024,1024] f32 [0,1]=img/255 → 出力 [1,1,1024,1024] f32 sigmoid soft マスク) を
  CPU / iGPU(Xe=GPU.0) / RTX5080(GPU.1) / NPU(AI Boost) のどれで推論すべきか?
  → OutputSpec::matting_device (src/core/character.hpp:119) の確定値を出す。

背景 (前科):
  - M-1 (別開発機・NPU 無し) で OV CPU 583ms と確認済み。
  - VAE decode で iGPU は大型 conv に弱かった (probe4: iGPU 995ms vs CPU 126ms)。
  - WD14 (Window Attention) は NPU 268ms と遅かったが、純 conv なら NPU 最速の前科
    (4-B quality scorer probe: 448² で NPU 4.62ms < iGPU < CPU)。
  - ISNet は基本 conv-UNet なので NPU が速い可能性 — だが 1024² の大型モデルで
    NPU がオンチップメモリ超過/非対応 op で compile できるかが鍵。落ちたらそれも M-5 の結論。

手法 (CLAUDE.md 準拠):
  - ISNet ONNX を ov.convert_model(onnx, input=[("img",[1,3,1024,1024])]) で OV IR 化 →
    reshape([1,3,1024,1024]) で静的化 (NPU 必須)。
  - 入力名は IR から取得 (img のはず)。入力は f32 [1,3,1024,1024] 乱数
    (レイテンシは重み値・入力値非依存)。
  - ["CPU","GPU.0","GPU.1","NPU"] の available を try/except で個別 compile + bench。
    NPU が compile/推論で落ちても例外を記録して probe 自体は完走させる。
  - 1024² は重いので warmup 10 / iters 20。compile 時間も記録。
  - レポートを scripts/_matting_device_report.json に保存し判定サマリを print。

注意 (CLAUDE.md):
  - NPU は静的形状のみ → compile 前に reshape 必須。
  - 入力 element_type は IR に厳密一致 (画像入力は f32)。i32/i64/u8 を渡さない
    (CLIP 0xC0000409 教訓)。
"""
import json, os, time
import numpy as np
import openvino as ov
from huggingface_hub import hf_hub_download

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, "_matting_device_report.json")

H = W = 1024
STATIC_SHAPE = [1, 3, H, W]


def bench(compiled, x_np, inname, warmup=10, iters=20):
    for _ in range(warmup):
        compiled({inname: x_np})
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        compiled({inname: x_np})
        ts.append((time.perf_counter() - t) * 1000.0)
    ts = np.array(ts)
    return dict(median=round(float(np.median(ts)), 2),
                min=round(float(ts.min()), 2),
                max=round(float(ts.max()), 2))


def main():
    # ------------------------------------------------------------
    # 1. ISNet ONNX 取得 (この研究機の HF キャッシュへ)
    # ------------------------------------------------------------
    onnx_path = hf_hub_download(repo_id="skytnt/anime-seg", filename="isnetis.onnx")
    print(f"[src] {onnx_path}  ({os.path.getsize(onnx_path)/1e6:.1f} MB)")

    # ------------------------------------------------------------
    # 2. ONNX -> OV IR (静的 [1,3,1024,1024] を明示・NPU 必須)
    # ------------------------------------------------------------
    t0 = time.perf_counter()
    ov_model = ov.convert_model(onnx_path, input=[("img", STATIC_SHAPE)])
    ov_model.reshape({ov_model.input(0): STATIC_SHAPE})  # 静的化 (NPU 必須)
    convert_ms = (time.perf_counter() - t0) * 1000.0

    in_port = ov_model.input(0)
    out_port = ov_model.output(0)
    inname = in_port.get_any_name()
    in_et  = in_port.get_element_type().get_type_name()
    in_shape  = list(in_port.get_partial_shape().to_shape())
    out_shape = list(out_port.get_partial_shape().to_shape())
    print(f"[IR] input  '{inname}' {in_et} {in_shape}")
    print(f"[IR] output '{out_port.get_any_name()}' "
          f"{out_port.get_element_type().get_type_name()} {out_shape}")
    print(f"[convert] {convert_ms:.0f} ms")
    assert in_et == "f32", f"想定外の入力型 {in_et} (CLIP 0xC0000409 教訓: f32 厳守)"

    core = ov.Core()
    devs = core.available_devices
    print("OV devices:", devs)
    dev_label = {"CPU": "CPU", "GPU.0": "iGPU(Xe)", "GPU.1": "RTX5080", "NPU": "NPU"}
    target_devs = [d for d in ["CPU", "GPU.0", "GPU.1", "NPU"] if d in devs]

    # 入力は f32 乱数 ([0,1] レンジ・レイテンシは値非依存)
    x_np = np.random.rand(1, 3, H, W).astype(np.float32)

    report = {
        "_model": "ISNet-anime (skytnt/anime-seg isnetis.onnx, Apache-2.0)",
        "_onnx": onnx_path,
        "_input": {"name": inname, "element_type": in_et, "shape": in_shape},
        "_output_shape": out_shape,
        "_devices": devs,
        "_convert_ms": round(convert_ms, 0),
        "runs": {},
    }

    # ------------------------------------------------------------
    # 3. 各 device で compile + bench (NPU 失敗も記録して完走)
    # ------------------------------------------------------------
    print(f"\n=== ISNet 1024x1024 (input '{inname}') ===")
    for d in target_devs:
        lbl = dev_label.get(d, d)
        try:
            t0 = time.perf_counter()
            compiled = core.compile_model(ov_model, d)
            compile_ms = (time.perf_counter() - t0) * 1000.0
            r = bench(compiled, x_np, inname)
            r["compile_ms"] = round(compile_ms, 0)
            report["runs"][d] = r
            print(f"  [{lbl:9s}] median {r['median']:8.2f}ms  "
                  f"(min {r['min']:.2f} / max {r['max']:.2f})  compile {r['compile_ms']:.0f}ms")
        except Exception as e:
            msg = f"{type(e).__name__}: {str(e)[:300]}"
            report["runs"][d] = {"error": msg}
            print(f"  [{lbl:9s}] FAILED  {msg}")

    json.dump(report, open(OUT, "w"), indent=2)
    print("\nsaved:", OUT)

    # ------------------------------------------------------------
    # 4. 判定サマリ: matting_device 推奨
    # ------------------------------------------------------------
    print("\n--- 判定サマリ (matting_device 材料) ---")
    def g(d):
        v = report["runs"].get(d, {})
        return v.get("median") if "median" in v else None
    cpu, igpu, dgpu, npu = g("CPU"), g("GPU.0"), g("GPU.1"), g("NPU")
    for name, v in [("CPU", cpu), ("iGPU(Xe)", igpu), ("RTX5080", dgpu), ("NPU", npu)]:
        print(f"  {name:9s} = {v}ms" if v is not None else f"  {name:9s} = N/A (失敗)")
    if npu is not None and cpu is not None:
        print(f"  NPU/CPU  = {npu/cpu:.2f}x")
    if igpu is not None and cpu is not None:
        print(f"  iGPU/CPU = {igpu/cpu:.2f}x")

    # 最速 device を選ぶ。NPU が落ちたら CPU/iGPU/RTX5080 から選ぶ。
    cand = {("NPU" if d == "NPU" else
             "iGPU" if d == "GPU.0" else
             "CPU" if d == "CPU" else "RTX5080"): g(d)
            for d in target_devs if g(d) is not None}
    if cand:
        best_label = min(cand, key=cand.get)
        # matting_device の許容値は "iGPU"/"NPU"/"CPU"。RTX5080 は拡散専任なので
        # マッティングには通常割り当てない方針 → CPU/iGPU/NPU の中の最速を推奨に採る。
        allowed = {k: v for k, v in cand.items() if k in ("CPU", "iGPU", "NPU")}
        rec = min(allowed, key=allowed.get) if allowed else best_label
        print(f"\n  全 device 最速      : {best_label} = {cand[best_label]}ms")
        print(f"  matting_device 推奨 : \"{rec}\" = {allowed.get(rec)}ms "
              f"(RTX5080 は拡散専任のため除外)")
        report["_fastest_all"] = {"device": best_label, "median_ms": cand[best_label]}
        report["_matting_device_recommend"] = {"device": rec, "median_ms": allowed.get(rec)}
        json.dump(report, open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main()

"""dollama probe: SDXL CLIP-bigG (text encoder G) のデバイス別レイテンシ計測。

Phase 2-6b Stage C。models/sdxl-text-encoder-g/model_ov.{xml,bin} を
NPU / CPU / GPU.0 (Intel Xe iGPU) でコンパイル・推論し、中央値レイテンシを比較。

- 入力 input_ids i64 [1,77] 静的 (IR が既に静的なので reshape は冪等に呼ぶ)
- 出力 0 = penultimate [1,77,1280] f32, 出力 1 = pooled [1,1280] f32
- bigG は 694M params / bin 1.32GB。NPU オンチップ超過の可能性あり (probe6 Phi-3 前例)。
  コンパイル/推論失敗は例外を捕まえて理由を記録する。

ウォームアップ 3 回を除いた中央値 (N=100) を ms で報告する。
"""

import statistics
import time
import traceback

import numpy as np
import openvino as ov

MODEL_XML = "models/sdxl-text-encoder-g/model_ov.xml"
WARMUP = 3
N = 100

# CLIP bigG の固定 token ids ダミー。BOS=49406, EOS=49407, PAD=0 を模す。
# (Stage A の cond_token_ids_g 相当。値そのものはレイテンシに影響しないのでダミーで可)
tok = np.zeros((1, 77), dtype=np.int64)
tok[0, 0] = 49406  # BOS
tok[0, 1:5] = [320, 1125, 539, 786]  # 適当なトークン
tok[0, 5] = 49407  # EOS
# 残りは PAD(0)


def bench(core, device):
    """device 上でコンパイル→推論し、(中央値ms, min, max, 出力情報) を返す。失敗時は例外送出。"""
    model = core.read_model(MODEL_XML)
    # 静的形状を明示 (IR が既に [1,77] でも冪等)
    model.reshape({"input_ids": [1, 77]})

    t0 = time.perf_counter()
    compiled = core.compile_model(model, device)
    compile_s = time.perf_counter() - t0

    req = compiled.create_infer_request()
    in_tensor = ov.Tensor(tok)

    # ウォームアップ
    for _ in range(WARMUP):
        req.infer({0: in_tensor})

    lat = []
    last = None
    for _ in range(N):
        s = time.perf_counter()
        last = req.infer({0: in_tensor})
        lat.append((time.perf_counter() - s) * 1000.0)

    out_info = {}
    for i, port in enumerate(compiled.outputs):
        t = req.get_output_tensor(i)
        arr = np.array(t.data)
        out_info[port.get_any_name()] = (
            list(arr.shape),
            str(t.element_type),
            float(arr.min()),
            float(arr.max()),
        )

    return {
        "median": statistics.median(lat),
        "min": min(lat),
        "max": max(lat),
        "compile_s": compile_s,
        "outputs": out_info,
    }


def main():
    core = ov.Core()
    print("available devices:", core.available_devices)
    for d in core.available_devices:
        try:
            full = core.get_property(d, "FULL_DEVICE_NAME")
        except Exception:
            full = "?"
        print(f"  {d}: {full}")
    print()

    results = {}
    for device in ["NPU", "CPU", "GPU.0"]:
        if device not in core.available_devices:
            print(f"[{device}] not available, skip")
            results[device] = None
            continue
        print(f"[{device}] compiling + benchmarking ...")
        try:
            r = bench(core, device)
            results[device] = r
            print(
                f"  -> median {r['median']:.3f} ms "
                f"(min {r['min']:.3f} / max {r['max']:.3f}), "
                f"compile {r['compile_s']:.2f} s"
            )
            for name, (shape, et, mn, mx) in r["outputs"].items():
                print(f"     out {name}: {shape} {et} range [{mn:.2f}, {mx:.2f}]")
        except Exception as e:
            results[device] = {"error": f"{type(e).__name__}: {e}"}
            print(f"  -> FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
        print()

    # サマリ
    print("=" * 60)
    print("SUMMARY (CLIP-bigG / SDXL text encoder G, [1,77] static)")
    print("=" * 60)
    base = None
    for d in ["NPU", "CPU", "GPU.0"]:
        r = results.get(d)
        if r is None:
            print(f"  {d:6s}: N/A (device absent)")
        elif "error" in r:
            print(f"  {d:6s}: FAILED ({r['error']})")
        else:
            print(f"  {d:6s}: {r['median']:8.3f} ms")
            if d == "CPU":
                base = r["median"]
    if base:
        print()
        for d in ["NPU", "GPU.0"]:
            r = results.get(d)
            if r and "error" not in r:
                ratio = base / r["median"]
                print(f"  {d} vs CPU: {ratio:.2f}x faster" if ratio >= 1
                      else f"  {d} vs CPU: {1/ratio:.2f}x slower")


if __name__ == "__main__":
    main()

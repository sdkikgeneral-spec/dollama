"""
dollma_probe9_clip.py - CLIP text encoder を NPU/iGPU/CPU で比較
================================================================

SD パイプラインのテキスト条件付けに使う CLIP-L text encoder。
入力: 77 トークン固定 → NPU に最適な形状。

確認項目:
  STEP 1 : CLIP-L text encoder を ONNX エクスポート + OV IR 変換
  STEP 2 : NPU / iGPU / CPU 推論速度比較
  STEP 3 : 実テキストでの出力確認 (embedding の L2 norm 等)

モデル: openai/clip-vit-large-patch14
  - text encoder のみ抽出 (image encoder は除外)
  - 入力: [1, 77] int32 (token ids)
  - 出力: [1, 77, 768] float32 (hidden states) + [1, 768] (pooled)
"""

import time
from pathlib import Path

import numpy as np

MODEL_DIR   = Path("models/clip-l-text-encoder")
ONNX_PATH   = MODEL_DIR / "model.onnx"
OV_XML_PATH = MODEL_DIR / "model_ov.xml"

HF_MODEL_ID = "openai/clip-vit-large-patch14"
SEQ_LEN = 77  # CLIP 固定


# ============================================================
# STEP 1: CLIP text encoder エクスポート + OV 変換
# ============================================================
def export_and_convert():
    print("=" * 60)
    print("STEP 1: CLIP text encoder エクスポート + OV IR 変換")
    print("=" * 60)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not OV_XML_PATH.exists():
        print(f"CLIP-L text encoder を OV IR に変換中: {HF_MODEL_ID}")
        import torch
        import openvino as ov
        from transformers import CLIPModel, CLIPTokenizer

        print("  モデルダウンロード中...")
        clip_full = CLIPModel.from_pretrained(HF_MODEL_ID)
        text_model = clip_full.text_model
        text_model.eval()

        # ダミー入力 (77トークン固定)
        dummy_ids = torch.zeros(1, SEQ_LEN, dtype=torch.long)

        # ONNX を経由せず直接 OV IR に変換 (推奨パス)
        print("  ov.convert_model() 実行中...")
        ov_model = ov.convert_model(text_model, example_input=(dummy_ids,))

        # 静的形状に固定 (NPU 必須)
        ov_model.reshape({"input_ids": [1, SEQ_LEN]})
        print(f"  静的形状固定: [1, {SEQ_LEN}]")
        print(f"  入力: {[(i.get_any_name(), i.shape) for i in ov_model.inputs]}")
        print(f"  出力: {[(o.get_any_name(), o.shape) for o in ov_model.outputs]}")

        ov.save_model(ov_model, OV_XML_PATH)
        xml_kb = OV_XML_PATH.stat().st_size / 1024
        bin_mb = (MODEL_DIR / "model_ov.bin").stat().st_size / 1024 / 1024
        print(f"  → {OV_XML_PATH}  (xml {xml_kb:.0f} KB, bin {bin_mb:.1f} MB)")
    else:
        bin_mb = (MODEL_DIR / "model_ov.bin").stat().st_size / 1024 / 1024
        print(f"OV IR キャッシュ済み  (bin {bin_mb:.1f} MB)")

    print("✅ STEP 1 完了")


# ============================================================
# STEP 2: NPU / iGPU / CPU 推論速度比較
# ============================================================
def benchmark_devices():
    print("\n" + "=" * 60)
    print("STEP 2: デバイス別推論速度比較")
    print(f"入力: [1, {SEQ_LEN}] int32 (固定形状)")
    print("=" * 60)

    import openvino as ov
    core = ov.Core()
    ov_model = core.read_model(OV_XML_PATH)

    dummy = np.zeros((1, SEQ_LEN), dtype=np.int32)
    # 実際のトークン列に近い値 (BOS=49406, EOS=49407, PAD=0)
    dummy[0, 0] = 49406   # BOS
    dummy[0, 1] = 2368    # "anime"
    dummy[0, 2] = 49407   # EOS

    results = {}
    for device in ["CPU", "GPU.0", "NPU"]:
        try:
            print(f"\n{device} コンパイル中...", end="", flush=True)
            t0 = time.perf_counter()
            compiled = core.compile_model(ov_model, device)
            compile_ms = (time.perf_counter() - t0) * 1000
            print(f" {compile_ms:.0f}ms")

            req = compiled.create_infer_request()

            # ウォームアップ
            for _ in range(5):
                req.infer({"input_ids": dummy})

            # 計測
            N = 50
            times = []
            for _ in range(N):
                t0 = time.perf_counter()
                req.infer({"input_ids": dummy})
                times.append((time.perf_counter() - t0) * 1000)

            med = float(np.median(times))
            mn  = float(np.min(times))
            print(f"  推論: {med:.2f}ms (最速 {mn:.2f}ms)  [{N}回中央値]")
            results[device] = {"compile_ms": compile_ms, "infer_ms": med,
                               "compiled": compiled, "request": req}

        except Exception as e:
            print(f"\n  ❌ {device}: {e}")
            results[device] = None

    return results, dummy


# ============================================================
# STEP 3: 実テキストでの出力確認
# ============================================================
def check_output(results: dict, dummy: np.ndarray):
    print("\n" + "=" * 60)
    print("STEP 3: 出力テンソル確認")
    print("=" * 60)

    best_device = None
    best_ms = float("inf")
    for dev, r in results.items():
        if r and r["infer_ms"] < best_ms:
            best_ms = r["infer_ms"]
            best_device = dev

    if not best_device:
        print("❌ 利用可能なデバイスなし")
        return

    print(f"最速デバイス: {best_device} ({best_ms:.2f}ms)")

    req = results[best_device]["request"]
    out = req.infer({"input_ids": dummy})

    for i, (name, tensor) in enumerate(out.items()):
        arr = tensor
        print(f"  出力[{i}] '{name}': shape={arr.shape}, "
              f"mean={arr.mean():.4f}, std={arr.std():.4f}, "
              f"L2={np.linalg.norm(arr):.2f}")


# ============================================================
# メイン
# ============================================================
if __name__ == "__main__":
    print("dollama probe9: CLIP-L text encoder — NPU/iGPU/CPU 比較")
    print(f"入力: [1, {SEQ_LEN}] 固定形状 (SD パイプラインのテキスト条件付け)")
    print()

    export_and_convert()
    results, dummy = benchmark_devices()
    check_output(results, dummy)

    print("\n" + "=" * 60)
    print("probe9 完了")
    print("=" * 60)
    print("\n速度まとめ:")
    for dev, r in results.items():
        if r:
            print(f"  {dev:<8}: {r['infer_ms']:.2f}ms  (コンパイル {r['compile_ms']:.0f}ms)")
        else:
            print(f"  {dev:<8}: ❌")

    print("\n→ 最速デバイスを SD パイプラインの CLIP encoder に採用")

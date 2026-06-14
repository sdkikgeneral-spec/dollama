"""
dollma_probe8_wd14.py - WD14 SwinV2 Tagger を NPU/iGPU/CPU で比較
===================================================================

WD14 は 2D アニメ画像に特化した danbooru タグ分類器。
入力 448×448 固定 → NPU 向き。

確認項目:
  STEP 1 : ONNX モデルダウンロード + OV IR 変換
  STEP 2 : NPU / iGPU / CPU で推論速度比較
  STEP 3 : 実際のタグ出力確認
"""

import time
import json
from pathlib import Path

import numpy as np

MODEL_DIR   = Path("models/wd14-swinv2-tagger-v3")
ONNX_PATH   = MODEL_DIR / "model.onnx"
OV_XML_PATH = MODEL_DIR / "model_ov.xml"
TAGS_PATH   = MODEL_DIR / "selected_tags.csv"

HF_REPO = "SmilingWolf/wd-swinv2-tagger-v3"

INPUT_SIZE = 448   # 固定形状


# ============================================================
# STEP 1: モデル取得 + OV 変換
# ============================================================
def download_and_convert():
    print("=" * 60)
    print("STEP 1: モデル取得 + OpenVINO IR 変換")
    print("=" * 60)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ONNX ダウンロード
    if not ONNX_PATH.exists():
        print(f"ダウンロード中: {HF_REPO}/model.onnx ...")
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo_id=HF_REPO, filename="model.onnx",
                        local_dir=MODEL_DIR)
        print(f"  → {ONNX_PATH}  ({ONNX_PATH.stat().st_size/1024/1024:.1f} MB)")
    else:
        print(f"ONNX キャッシュ済み: {ONNX_PATH.stat().st_size/1024/1024:.1f} MB")

    # タグ CSV ダウンロード
    if not TAGS_PATH.exists():
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo_id=HF_REPO, filename="selected_tags.csv",
                        local_dir=MODEL_DIR)
        print(f"  → {TAGS_PATH}")

    # OV IR 変換
    if not OV_XML_PATH.exists():
        print("\nONNX → OpenVINO IR 変換中 ...")
        import openvino as ov
        core = ov.Core()
        ov_model = core.read_model(ONNX_PATH)

        # 入力を静的形状に固定 (NPU 必須)
        ov_model.reshape([1, INPUT_SIZE, INPUT_SIZE, 3])
        print(f"  入力形状固定: [1, {INPUT_SIZE}, {INPUT_SIZE}, 3]")

        ov.save_model(ov_model, OV_XML_PATH)
        print(f"  → {OV_XML_PATH}  ({OV_XML_PATH.stat().st_size/1024:.0f} KB)")
    else:
        print(f"\nOV IR キャッシュ済み: {OV_XML_PATH}")

    print("✅ STEP 1 完了")


# ============================================================
# STEP 2: NPU / iGPU / CPU 推論速度比較
# ============================================================
def benchmark_devices():
    print("\n" + "=" * 60)
    print("STEP 2: デバイス別推論速度比較")
    print("=" * 60)

    import openvino as ov
    core = ov.Core()

    ov_model = core.read_model(OV_XML_PATH)

    # ダミー入力 (uint8, 0-255)
    dummy = np.random.randint(0, 255, (1, INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8).astype(np.float32)

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
            for _ in range(3):
                req.infer({0: dummy})

            # 計測
            N = 20
            times = []
            for _ in range(N):
                t0 = time.perf_counter()
                req.infer({0: dummy})
                times.append((time.perf_counter() - t0) * 1000)

            med = float(np.median(times))
            mn  = float(np.min(times))
            print(f"  推論: {med:.1f}ms (最速 {mn:.1f}ms)  [{N}回中央値]")
            results[device] = {"compile_ms": compile_ms, "infer_ms": med,
                               "compiled": compiled}

        except Exception as e:
            print(f"\n  ❌ {device}: {e}")
            results[device] = None

    return results


# ============================================================
# STEP 3: 実際のタグ出力
# ============================================================
def show_tags(results: dict):
    print("\n" + "=" * 60)
    print("STEP 3: タグ出力確認 (ダミー画像)")
    print("=" * 60)

    # タグ読み込み
    import csv
    tags = []
    with open(TAGS_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tags.append(row["name"])
    print(f"タグ数: {len(tags)}")

    # 最速デバイスで推論
    best_device = None
    best_ms = float("inf")
    for dev, r in results.items():
        if r and r["infer_ms"] < best_ms:
            best_ms = r["infer_ms"]
            best_device = dev

    if not best_device:
        print("❌ 利用可能なデバイスなし")
        return

    print(f"最速デバイス: {best_device} ({best_ms:.1f}ms)")

    compiled = results[best_device]["compiled"]
    req = compiled.create_infer_request()

    # アニメ調ダミー (肌色・青空的なグラデーション)
    img = np.zeros((1, INPUT_SIZE, INPUT_SIZE, 3), dtype=np.float32)
    img[0, :, :, 0] = 200  # R
    img[0, :, :, 1] = 150  # G
    img[0, :, :, 2] = 180  # B

    result = req.infer({0: img})
    scores = result[0].flatten()

    # 上位タグ表示
    top_idx = np.argsort(scores)[::-1][:20]
    print("\n上位 20 タグ:")
    for i, idx in enumerate(top_idx):
        if idx < len(tags):
            print(f"  {i+1:2d}. {tags[idx]:<40} {scores[idx]:.3f}")


# ============================================================
# メイン
# ============================================================
if __name__ == "__main__":
    print("dollama probe8: WD14 SwinV2 Tagger — NPU/iGPU/CPU 比較")
    print("入力: 448×448 固定形状 (NPU 向き)")
    print()

    download_and_convert()
    results = benchmark_devices()
    show_tags(results)

    print("\n" + "=" * 60)
    print("probe8 完了")
    print("=" * 60)
    print("\n速度まとめ:")
    for dev, r in results.items():
        if r:
            print(f"  {dev:<8}: {r['infer_ms']:.1f}ms")
        else:
            print(f"  {dev:<8}: ❌")

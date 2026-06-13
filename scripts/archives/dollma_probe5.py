"""
dollma_probe5.py - iGPU 活用候補の実測比較
============================================

4候補を iGPU (GPU.0) / CPU で実測し、
「RTX5080 が拡散している間に終わるか」を判定する。

RTX5080 拡散時間の目安: 30ステップ × ~50ms = ~1500ms (推定)
→ iGPU タスクが 1500ms 以内なら pipeline parallelism で隠蔽できる

候補:
  A: ControlNet 前処理  (depth estimation スタブ)
  B: CLIP 画像エンコーダー (aesthetic scoring / img2img)
  C: VAE Encode         (img2img 入力側)
  D: Aesthetic Scorer MLP (CLIP embedding → score)
"""

import time
import numpy as np


def get_igpu(core):
    """GPU.0/GPU.1 から INTEGRATED (Intel iGPU) を返す。"""
    for dev in core.available_devices:
        if dev.startswith("GPU"):
            try:
                if "INTEGRATED" in str(core.get_property(dev, "DEVICE_TYPE")):
                    return dev
            except Exception:
                pass
    return None


def bench(infer_req, dummy_input, n=10, warmup=3):
    """ウォームアップ後に中央値を返す (ms)。"""
    for _ in range(warmup):
        infer_req.infer({0: dummy_input})
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        infer_req.infer({0: dummy_input})
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def compare(core, igpu, ov_model, input_shape, label, n=10):
    """iGPU と CPU の速度を比較して表示する。"""
    dummy = np.random.randn(*input_shape).astype(np.float32)
    results = {}
    for device in [igpu, "CPU"]:
        try:
            req = core.compile_model(ov_model, device).create_infer_request()
            ms = bench(req, dummy, n=n)
            results[device] = ms
        except Exception as e:
            results[device] = f"NG ({e})"

    igpu_ms = results.get(igpu)
    cpu_ms  = results.get("CPU")

    igpu_str = f"{igpu_ms:.1f}ms" if isinstance(igpu_ms, float) else str(igpu_ms)
    cpu_str  = f"{cpu_ms:.1f}ms"  if isinstance(cpu_ms,  float) else str(cpu_ms)

    ratio = ""
    verdict = ""
    if isinstance(igpu_ms, float) and isinstance(cpu_ms, float):
        ratio = f"{cpu_ms/igpu_ms:.1f}x"
        if igpu_ms < cpu_ms:
            verdict = "✅ iGPU 速い"
        elif igpu_ms < cpu_ms * 1.5:
            verdict = "△ ほぼ同等"
        else:
            verdict = "❌ CPU の方が速い"

    print(f"  iGPU : {igpu_str:<12}  CPU : {cpu_str:<12}  比 : {ratio:<6}  {verdict}")
    return igpu_ms, cpu_ms


# ============================================================
# 候補 A: ControlNet 前処理 (depth estimation スタブ)
# 入力: [1, 3, 512, 512]  出力: depth map [1, 1, 512, 512]
# ============================================================
def bench_depth_estimation(core, igpu):
    print("\n--- 候補 A: ControlNet 前処理 (depth estimation) ---")
    print("  入力: [1, 3, 512, 512] → [1, 1, 512, 512]")

    import openvino as ov
    import torch
    import torch.nn as nn

    # MiDaS-small 相当: encoder (ResNet-like) + decoder
    class DepthStub(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3),   nn.ReLU(),  # 256
                nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),  # 128
                nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(), # 64
                nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.ReLU(), # 32
            )
            self.dec = nn.Sequential(
                nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), nn.ReLU(), # 64
                nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.ReLU(), # 128
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  nn.ReLU(), # 256
                nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),    nn.Sigmoid(), # 512
            )
        def forward(self, x):
            return self.dec(self.enc(x))

    model = DepthStub().eval()
    example = torch.randn(1, 3, 512, 512)
    print("  変換中...", end=" ", flush=True)
    ov_model = ov.convert_model(model, example_input=example)
    ov_model.reshape([1, 3, 512, 512])
    print("完了")

    return compare(core, igpu, ov_model, (1, 3, 512, 512), "depth")


# ============================================================
# 候補 B: CLIP 画像エンコーダー
# 入力: [1, 3, 224, 224]  出力: embedding [1, 512]
# ※ attention 回避のため Conv + pooling で近似
# ============================================================
def bench_clip_image_encoder(core, igpu):
    print("\n--- 候補 B: CLIP 画像エンコーダー (aesthetic / img2img) ---")
    print("  入力: [1, 3, 224, 224] → [1, 512]")

    import openvino as ov
    import torch
    import torch.nn as nn

    # ViT-B/32 相当の計算量を Conv で近似
    # ViT-B/32: ~10 GFLOPs / ResNet-50: ~4 GFLOPs → 中間的な規模
    class CLIPImageStub(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3),    nn.ReLU(), # 112
                nn.Conv2d(64, 128, 3, stride=2, padding=1),  nn.ReLU(), # 56
                nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(), # 28
                nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.ReLU(), # 14
                nn.Conv2d(512, 512, 3, stride=2, padding=1), nn.ReLU(), # 7
            )
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.proj = nn.Linear(512, 512)

        def forward(self, x):
            f = self.backbone(x)
            f = self.pool(f).flatten(1)
            return self.proj(f)

    model = CLIPImageStub().eval()
    example = torch.randn(1, 3, 224, 224)
    print("  変換中...", end=" ", flush=True)
    ov_model = ov.convert_model(model, example_input=example)
    ov_model.reshape([1, 3, 224, 224])
    print("完了")

    return compare(core, igpu, ov_model, (1, 3, 224, 224), "clip")


# ============================================================
# 候補 C: VAE Encode (img2img 入力側)
# 入力: [1, 3, 1024, 1024]  出力: latent [1, 4, 128, 128]
# probe4 では decode が遅かったが encode (逆方向) はどうか
# ============================================================
def bench_vae_encode(core, igpu):
    print("\n--- 候補 C: VAE Encode (img2img 入力側) ---")
    print("  入力: [1, 3, 1024, 1024] → [1, 4, 128, 128]  (decode の逆方向)")

    import openvino as ov
    import torch
    import torch.nn as nn

    # SD-VAE encoder 相当: 8倍ダウンスケール
    class VAEEncodeStub(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),               nn.SiLU(),
                nn.Conv2d(64, 128, 4, stride=2, padding=1),   nn.SiLU(), # 512
                nn.Conv2d(128, 256, 4, stride=2, padding=1),  nn.SiLU(), # 256
                nn.Conv2d(256, 512, 4, stride=2, padding=1),  nn.SiLU(), # 128
                nn.Conv2d(512, 4, 3, padding=1),                          # latent
            )
        def forward(self, x):
            return self.layers(x)

    model = VAEEncodeStub().eval()
    example = torch.randn(1, 3, 1024, 1024)
    print("  変換中...", end=" ", flush=True)
    ov_model = ov.convert_model(model, example_input=example)
    ov_model.reshape([1, 3, 1024, 1024])
    print("完了")

    return compare(core, igpu, ov_model, (1, 3, 1024, 1024), "vae_enc", n=5)


# ============================================================
# 候補 D: Aesthetic Scorer MLP
# 入力: CLIP embedding [1, 512]  出力: score [1, 1]
# NPU 出力 (または CLIP 出力) を直接受け取ってスコアリング
# ============================================================
def bench_aesthetic_scorer(core, igpu):
    print("\n--- 候補 D: Aesthetic Scorer MLP ---")
    print("  入力: [1, 512] → [1, 1]  (CLIP embedding → 美的スコア)")

    import openvino as ov
    import torch
    import torch.nn as nn

    class AestheticScorer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(512, 256), nn.ReLU(),
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, 64),  nn.ReLU(),
                nn.Linear(64, 1),
            )
        def forward(self, x):
            return self.mlp(x)

    model = AestheticScorer().eval()
    example = torch.randn(1, 512)
    print("  変換中...", end=" ", flush=True)
    ov_model = ov.convert_model(model, example_input=example)
    ov_model.reshape([1, 512])
    print("完了")

    return compare(core, igpu, ov_model, (1, 512), "aesthetic", n=50)


# ============================================================
# メイン
# ============================================================
if __name__ == "__main__":
    print("dollama probe5: iGPU 活用候補 実測比較")
    print("環境: Intel Core Ultra 9 285 (NPU + Xe iGPU) + RTX5080")
    print()

    import openvino as ov
    core = ov.Core()
    print(f"検出デバイス: {core.available_devices}")

    igpu = get_igpu(core)
    if igpu is None:
        print("Intel iGPU が見つかりません。BIOS設定を確認してください。")
        exit(1)
    print(f"Intel iGPU: {igpu}  ({core.get_property(igpu, 'FULL_DEVICE_NAME')})")

    results = {}
    results["A_depth"]    = bench_depth_estimation(core, igpu)
    results["B_clip"]     = bench_clip_image_encoder(core, igpu)
    results["C_vae_enc"]  = bench_vae_encode(core, igpu)
    results["D_aesthetic"]= bench_aesthetic_scorer(core, igpu)

    print("\n" + "=" * 60)
    print("結果サマリー (RTX5080 拡散 ~1500ms の間に終わるか)")
    print("=" * 60)
    print(f"{'候補':<28} {'iGPU':>8}  {'CPU':>8}  {'判定'}")
    print("-" * 60)

    labels = {
        "A_depth":     "A: ControlNet depth (512px)",
        "B_clip":      "B: CLIP image encoder (224px)",
        "C_vae_enc":   "C: VAE encode (1024px)",
        "D_aesthetic": "D: Aesthetic MLP",
    }
    DIFFUSION_MS = 1500

    for key, label in labels.items():
        igpu_ms, cpu_ms = results[key]
        igpu_str = f"{igpu_ms:.1f}ms" if isinstance(igpu_ms, float) else "NG"
        cpu_str  = f"{cpu_ms:.1f}ms"  if isinstance(cpu_ms,  float) else "NG"

        fit = ""
        if isinstance(igpu_ms, float):
            if igpu_ms < DIFFUSION_MS:
                fit = f"✅ 拡散内に収まる ({igpu_ms/DIFFUSION_MS*100:.0f}%使用)"
            else:
                fit = f"❌ 拡散より遅い"

        print(f"  {label:<26} {igpu_str:>8}  {cpu_str:>8}  {fit}")

    print("""
判定の読み方:
  ✅ かつ iGPU > CPU  → iGPU に任せる価値あり
  ✅ かつ iGPU < CPU  → CPU の方が速いが、並列ならiGPUも使える
  ❌            → iGPU には向かない
""")

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dollma_probe_quality_scorer.py — Model B (アニメ品質スコアラ) の NPU 実行性 probe

roadmap.md Phase 4 / 技術的リスク表「B の NPU 実行性」を埋める probe。

問い:
  品質スコアラ (§11 蒸留 QA スコアラ・生成画像→品質スカラ + 属性ロジット) を
  NPU に載せられるか? どの HW が速いか?

背景 (前科):
  - WD14 SwinV2 は NPU 268ms と CPU 101ms より 2.6x 遅かった (probe8)。
    原因は Window Attention が NPU に不向きと結論づけた。
  - スコアラは「純 conv backbone」で設計できる (window attention 不要)。
    → 純 conv なら NPU で速いのか? を切り分けるのがこの probe の核心。

手法 (再現可能・ライセンス完全自己完結):
  - 代表 backbone = ResNet-18 級の純 conv 分類網 (~11M, attention 皆無) を
    自前実装 (timm/torchvision 不要)。head は Linear(512 -> 8):
      index0 = 品質スカラ / 1..7 = 属性ロジット (anatomy/hands/lineart/...)。
    これは §11 蒸留 QA スコアラの代表形状。重みは乱数 (レイテンシは値に非依存)。
  - ov.convert_model で OV IR 化 → 静的形状 [1,3,H,W] に reshape (NPU 必須) →
    CPU / GPU.0(iGPU Xe) / GPU.1(RTX5080) / NPU で compile し中央値レイテンシ計測。
  - 解像度 448 (WD14 比較と同条件) と 512 (スコアラ実用域) の 2 つ。

注意 (CLAUDE.md):
  - NPU は静的形状のみ → compile 前に reshape 必須。
  - 入力 element_type は IR に厳密一致 (画像入力は f32)。
  - 各 device の compile は try/except で囲み、NPU 非対応 op でも probe 自体は完走。
"""
import json, os, time, sys
import numpy as np
import torch
import torch.nn as nn
import openvino as ov

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, "_quality_scorer_report.json")

# ----------------------------------------------------------------------------
# 1. 純 conv 代表 backbone (ResNet-18 級・attention 皆無)
#    timm/torchvision 非依存。BatchNorm は eval で conv に畳まれる。
# ----------------------------------------------------------------------------
class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(cout)
        self.relu  = nn.ReLU(inplace=True)
        self.down = None
        if stride != 1 or cin != cout:
            self.down = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        idt = x if self.down is None else self.down(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + idt)


class ScorerNet(nn.Module):
    """ResNet-18 級の純 conv スコアラ。head = 品質スカラ + 属性ロジット 7。"""
    def __init__(self, n_attr=7):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False), nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), nn.MaxPool2d(3, 2, 1))
        def layer(cin, cout, stride):
            return nn.Sequential(BasicBlock(cin, cout, stride), BasicBlock(cout, cout, 1))
        self.l1 = layer(64, 64, 1)
        self.l2 = layer(64, 128, 2)
        self.l3 = layer(128, 256, 2)
        self.l4 = layer(256, 512, 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(512, 1 + n_attr)  # [品質, 属性×7]

    def forward(self, x):
        x = self.stem(x)
        x = self.l4(self.l3(self.l2(self.l1(x))))
        x = self.pool(x).flatten(1)
        return self.head(x)


# ----------------------------------------------------------------------------
# 2. 計測
# ----------------------------------------------------------------------------
def bench(compiled, x_np, inname, warmup=10, iters=50):
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
    core = ov.Core()
    devs = core.available_devices
    print("OV devices:", devs)
    dev_label = {"CPU": "CPU", "GPU.0": "iGPU(Xe)", "GPU.1": "RTX5080", "NPU": "NPU"}
    target_devs = [d for d in ["CPU", "GPU.0", "GPU.1", "NPU"] if d in devs]

    torch.manual_seed(20260623)
    model = ScorerNet().eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ScorerNet params = {n_params/1e6:.2f}M (pure conv, no attention)")

    report = {"_params_M": round(n_params / 1e6, 3), "_devices": devs, "runs": {}}

    for res in (448, 512):
        ex = torch.zeros(1, 3, res, res, dtype=torch.float32)
        with torch.no_grad():
            ov_model = ov.convert_model(model, example_input=ex)
        ov_model.reshape([1, 3, res, res])  # NPU 必須: 静的形状
        inname = ov_model.inputs[0].get_any_name()
        x_np = np.random.rand(1, 3, res, res).astype(np.float32)

        print(f"\n=== resolution {res}x{res} (input '{inname}') ===")
        report["runs"][res] = {}
        for d in target_devs:
            lbl = dev_label.get(d, d)
            try:
                t0 = time.perf_counter()
                compiled = core.compile_model(ov_model, d)
                compile_ms = (time.perf_counter() - t0) * 1000.0
                r = bench(compiled, x_np, inname)
                r["compile_ms"] = round(compile_ms, 0)
                report["runs"][res][d] = r
                print(f"  [{lbl:9s}] median {r['median']:7.2f}ms  "
                      f"(min {r['min']:.2f} / max {r['max']:.2f})  compile {r['compile_ms']:.0f}ms")
            except Exception as e:
                msg = f"{type(e).__name__}: {str(e)[:200]}"
                report["runs"][res][d] = {"error": msg}
                print(f"  [{lbl:9s}] FAILED  {msg}")

    json.dump(report, open(OUT, "w"), indent=2)
    print("\nsaved:", OUT)

    # 判定サマリ: NPU vs CPU vs iGPU
    print("\n--- 判定サマリ (scorer_device 材料) ---")
    for res in (448, 512):
        row = report["runs"].get(res, {})
        def g(d):
            v = row.get(d, {})
            return v.get("median") if "median" in v else None
        cpu, igpu, npu, dgpu = g("CPU"), g("GPU.0"), g("NPU"), g("GPU.1")
        parts = []
        for name, v in [("CPU", cpu), ("iGPU", igpu), ("RTX5080", dgpu), ("NPU", npu)]:
            parts.append(f"{name}={v}ms" if v is not None else f"{name}=N/A")
        print(f"  {res}: " + "  ".join(parts))
        if npu is not None and cpu is not None:
            ratio = npu / cpu
            verdict = "NPU 採用余地あり" if ratio <= 1.2 else \
                      ("NPU 微妙 (CPU 同等以下)" if ratio <= 2.0 else "NPU 不利 → CPU 採用")
            print(f"       NPU/CPU = {ratio:.2f}x → {verdict}")


if __name__ == "__main__":
    main()

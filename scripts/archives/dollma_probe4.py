"""
dollma_probe4.py - Intel SoC クラスター検証 (iGPU BIOS有効化後)
==============================================================

BIOS で iGPU を DXGI に表示させた結果:
  ['CPU', 'GPU.0', 'GPU.1', 'NPU'] の4デバイスが検出される

検証項目:
  STEP 1 : GPU.0 / GPU.1 を列挙し Intel iGPU と RTX5080 を識別
  STEP 2 : Intel iGPU で VAE デコーダースタブを動かして速度計測
  STEP 3 : NPU (text encoder) → iGPU ゼロコピー有効性の確認
  STEP 4 : iGPU 出力 → RTX5080 転送コスト (PCIe 1回分)
"""

import time
import numpy as np


def find_gpu_devices(core):
    """GPU.0 / GPU.1 を列挙して INTEGRATED / DISCRETE を識別する。
    戻り値: (igpu_name, dgpu_name) どちらかが None の場合は未検出。
    """
    import openvino as ov
    devices = core.available_devices
    igpu_name = None
    dgpu_name = None

    gpu_devs = [d for d in devices if d.startswith("GPU")]
    for dev in gpu_devs:
        try:
            dtype = core.get_property(dev, "DEVICE_TYPE")
            name  = core.get_property(dev, "FULL_DEVICE_NAME")
            print(f"  {dev}: {name}  [{dtype}]")
            if "INTEGRATED" in str(dtype):
                igpu_name = dev
            elif "DISCRETE" in str(dtype):
                dgpu_name = dev
        except Exception as e:
            print(f"  {dev}: プロパティ取得失敗 ({e})")

    return igpu_name, dgpu_name


# ============================================================
# STEP 1: デバイス一覧と iGPU / dGPU の識別
# ============================================================
def probe_igpu_info():
    print("=" * 60)
    print("STEP 1: デバイス識別 (GPU.0 / GPU.1 どちらが iGPU か)")
    print("=" * 60)

    try:
        import openvino as ov
        core = ov.Core()
        devices = core.available_devices
        print(f"検出デバイス: {devices}")

        igpu, dgpu = find_gpu_devices(core)

        if igpu is None:
            print("\n⚠ Intel iGPU (INTEGRATED) が見つからない")
            print("  → BIOSのiGPU設定を確認してください")
            return None, None, core

        print(f"\n✅ Intel iGPU = {igpu}")
        print(f"✅ NVIDIA dGPU = {dgpu}")

        # iGPU の詳細プロパティ
        print(f"\n--- {igpu} 詳細 ---")
        for p in ["OPTIMIZATION_CAPABILITIES", "DEVICE_GFLOPS", "RANGE_FOR_ASYNC_INFER_REQUESTS"]:
            try:
                print(f"  {p}: {core.get_property(igpu, p)}")
            except Exception:
                pass

        return igpu, dgpu, core

    except ImportError:
        print("OpenVINO 未インストール")
        return None, None, None


# ============================================================
# STEP 2: Intel iGPU で VAE デコーダースタブを実行
# SD の潜在表現 [1, 4, 128, 128] → 画像 [1, 3, 1024, 1024]
# ============================================================
def probe_igpu_vae_decode(igpu_name, core):
    print("\n" + "=" * 60)
    print(f"STEP 2: {igpu_name} (Intel iGPU) VAE デコーダー速度計測")
    print("  入力: latent [1, 4, 128, 128]  出力: image [1, 3, 1024, 1024]")
    print("=" * 60)

    try:
        import openvino as ov
        import torch
        import torch.nn as nn

        # VAE デコーダースタブ: SD-VAE に近い Conv 構造
        # latent (128x128) → image (1024x1024) の 8倍アップスケール
        class VAEDecodeStub(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.ConvTranspose2d(4, 512, 3, padding=1),
                    nn.SiLU(),
                    nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),  # 128→256
                    nn.SiLU(),
                    nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 256→512
                    nn.SiLU(),
                    nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # 512→1024
                    nn.SiLU(),
                    nn.Conv2d(64, 3, 3, padding=1),
                    nn.Tanh(),
                )

            def forward(self, x):
                return self.layers(x)

        model = VAEDecodeStub().eval()
        example = torch.randn(1, 4, 128, 128)

        print("OpenVINO に変換中...")
        ov_model = ov.convert_model(model, example_input=example)
        ov_model.reshape([1, 4, 128, 128])

        print(f"{igpu_name} にコンパイル中...")
        compiled_igpu = core.compile_model(ov_model, igpu_name)
        infer_req = compiled_igpu.create_infer_request()

        dummy_latent = np.random.randn(1, 4, 128, 128).astype(np.float32)

        for _ in range(3):
            infer_req.infer({0: dummy_latent})

        N = 10
        times = []
        for _ in range(N):
            t0 = time.perf_counter()
            infer_req.infer({0: dummy_latent})
            times.append((time.perf_counter() - t0) * 1000)

        igpu_ms = np.median(times)
        output_shape = infer_req.get_output_tensor(0).data.shape
        print(f"iGPU VAE decode (stub): {igpu_ms:.1f}ms (中央値, n={N})")
        print(f"  出力テンソル shape: {output_shape}")

        # CPU 比較
        compiled_cpu = core.compile_model(ov_model, "CPU")
        req_cpu = compiled_cpu.create_infer_request()
        for _ in range(3):
            req_cpu.infer({0: dummy_latent})
        times_cpu = []
        for _ in range(N):
            t0 = time.perf_counter()
            req_cpu.infer({0: dummy_latent})
            times_cpu.append((time.perf_counter() - t0) * 1000)
        cpu_ms = np.median(times_cpu)
        print(f"CPU VAE decode (stub):  {cpu_ms:.1f}ms")
        print(f"  iGPU/CPU 速度比: {cpu_ms/igpu_ms:.1f}x")

        return igpu_ms

    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================
# STEP 3: NPU Text Encoder → iGPU Conditioning ゼロコピー確認
# NPU出力はシステムRAM → .copy() なしで iGPU に渡せる
# ============================================================
def probe_npu_to_igpu_zero_copy(igpu_name, core):
    print("\n" + "=" * 60)
    print("STEP 3: NPU → iGPU ゼロコピーパイプライン計測")
    print(f"  NPU: Text Encoder (token → embedding) [システムRAM]")
    print(f"  iGPU ({igpu_name}): Conditioning (embedding → conditioning)")
    print("=" * 60)

    try:
        import openvino as ov
        import torch
        import torch.nn as nn

        # --- NPU: テキストエンコーダースタブ ---
        # tokens [1, 77] → embeddings [1, 77, 768]
        class TextEncoderStub(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(49408, 768)
                self.proj  = nn.Linear(768, 768)

            def forward(self, x):
                return self.proj(self.embed(x).float())

        text_model    = TextEncoderStub().eval()
        example_tok   = torch.zeros(1, 77, dtype=torch.long)
        ov_text       = ov.convert_model(text_model, example_input=example_tok)
        ov_text.reshape([1, 77])
        npu_req = core.compile_model(ov_text, "NPU").create_infer_request()

        # --- iGPU: Conditioning スタブ (Linear のみ、トレーシング安定) ---
        # embeddings [1, 77, 768] → conditioning [1, 77, 768]
        class ConditioningStub(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(768, 768)
                self.fc2 = nn.Linear(768, 768)

            def forward(self, x):
                return self.fc2(torch.relu(self.fc1(x))) + x

        cond_model  = ConditioningStub().eval()
        example_emb = torch.randn(1, 77, 768)
        ov_cond     = ov.convert_model(cond_model, example_input=example_emb)
        ov_cond.reshape([1, 77, 768])
        igpu_req = core.compile_model(ov_cond, igpu_name).create_infer_request()

        dummy_tokens = np.zeros((1, 77), dtype=np.int32)

        for _ in range(3):
            npu_req.infer({0: dummy_tokens})
            igpu_req.infer({0: npu_req.get_output_tensor(0).data})

        N = 20

        # パターンA: ゼロコピー (NPU出力 numpy をそのまま渡す)
        times_zero = []
        for _ in range(N):
            t0 = time.perf_counter()
            npu_req.infer({0: dummy_tokens})
            npu_out = npu_req.get_output_tensor(0).data      # システムRAM上の numpy
            igpu_req.infer({0: npu_out})                      # コピーなし
            times_zero.append((time.perf_counter() - t0) * 1000)

        # パターンB: 明示コピーあり
        times_copy = []
        for _ in range(N):
            t0 = time.perf_counter()
            npu_req.infer({0: dummy_tokens})
            npu_out = npu_req.get_output_tensor(0).data.copy()  # 明示コピー
            igpu_req.infer({0: npu_out})
            times_copy.append((time.perf_counter() - t0) * 1000)

        ms_zero = np.median(times_zero)
        ms_copy = np.median(times_copy)
        out_bytes = npu_req.get_output_tensor(0).data.nbytes

        print(f"\nNPU出力サイズ: {out_bytes:,} bytes ({out_bytes/1024:.1f} KB)")
        print(f"ゼロコピー (直接渡し): {ms_zero:.2f}ms")
        print(f"コピーあり:           {ms_copy:.2f}ms")
        print(f"差分:                 {ms_copy - ms_zero:.3f}ms")

        if abs(ms_zero - ms_copy) < 0.5:
            print("\n✅ コピーコストがほぼゼロ → NPU/iGPU 間はシステムRAM共有を確認")
        else:
            print(f"\n→ コピーに {ms_copy - ms_zero:.2f}ms のオーバーヘッド")

        return True

    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================
# STEP 4: iGPU (システムRAM) → RTX5080 (VRAM) 転送コスト
# ============================================================
def probe_igpu_to_dgpu_transfer():
    print("\n" + "=" * 60)
    print("STEP 4: iGPU (system RAM) → RTX5080 (VRAM) 転送コスト")
    print("=" * 60)

    try:
        import torch

        if not torch.cuda.is_available():
            print("CUDA 不可")
            return

        transfers = [
            ("text embedding  (~237KB)", 1 * 77 * 768 * 4),
            ("latent 128x128x4 (~256KB)", 1 * 4 * 128 * 128 * 4),
            ("image 1024x1024x3 (~12MB)", 1 * 3 * 1024 * 1024 * 4),
        ]

        print(f"\n{'データ種別':<28} {'サイズ':<10} {'転送時間':<12} {'帯域'}")
        print("-" * 65)

        N = 30
        for name, size_bytes in transfers:
            cpu_pinned = torch.zeros(size_bytes // 4).pin_memory()
            times = []
            for _ in range(N):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                gpu_t = cpu_pinned.cuda(non_blocking=True)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000)
            ms = np.median(times)
            bw = (size_bytes / 1e9) / (ms / 1000)
            print(f"{name:<28} {size_bytes/1024:>6.0f}KB  {ms:>7.3f}ms   {bw:.1f} GB/s")

        print("\n→ latent (256KB) ≈ 0.01ms = 拡散1ステップに比べて誤差レベル")

    except Exception as e:
        print(f"エラー: {e}")


# ============================================================
# メイン
# ============================================================
if __name__ == "__main__":
    print("dollama probe4: Intel SoC クラスター (NPU + iGPU) 検証")
    print("環境: Intel Core Ultra 9 285 (NPU + Xe iGPU) + RTX5080")
    print()

    igpu, dgpu, core = probe_igpu_info()

    if igpu is not None:
        probe_igpu_vae_decode(igpu, core)
        probe_npu_to_igpu_zero_copy(igpu, core)
        probe_igpu_to_dgpu_transfer()
    else:
        print("\niGPU が見つからないため STEP 2-4 をスキップします")

    print("\n" + "=" * 60)
    print("probe4 完了")
    print("=" * 60)
    print("""
結果の見方:
  STEP1: iGPU (INTEGRATED) と RTX5080 (DISCRETE) が識別できること
  STEP2: iGPU VAE < CPU → iGPU に処理を委ねる価値がある
  STEP3: ゼロコピー ≈ コピーあり → NPU/iGPU 間はシステムRAM共有を確認
  STEP4: latent 転送 < 0.1ms → PCIe コストは無視できる
""")

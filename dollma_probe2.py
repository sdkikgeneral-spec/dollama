"""
dollma probe2 - Win32ハンドル共有 + NPU推論パイプライン検証
============================================================

probe.py 結果:
  - Virtual Memory API: OK 粒度2MB / Win32ハンドル対応
  - NPU: OK FP16/INT8 / Intel AI Boost
  - cuMemExportToShareableHandle (cuMemAlloc経由): NG

修正内容 (probe2 v2):
  STEP 5: CUmemAllocationProp 構造体レイアウトを修正
          handle型を c_void_p -> c_uint64 に修正
  STEP 7: openvino.runtime (廃止) -> ov.convert_model に変更
"""

import time
import ctypes
import ctypes.wintypes
import numpy as np

# ============================================================
# STEP 5: cuMemCreate + Win32ハンドルの取得
# ============================================================
def probe_virtual_memory_handle():
    print("=" * 60)
    print("STEP 5: cuMemCreate + Win32ハンドル共有テスト")
    print("=" * 60)

    try:
        cuda = ctypes.CDLL("nvcuda.dll")

        cuda.cuInit(0)
        device = ctypes.c_int(0)
        cuda.cuDeviceGet(ctypes.byref(device), 0)
        ctx = ctypes.c_void_p()
        cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, device)

        # CUDAヘッダー通りの正確なネスト構造
        # CUmemLocation: {int type; int id;} = 8 bytes
        class CUmemLocation(ctypes.Structure):
            _fields_ = [
                ("type", ctypes.c_int),
                ("id",   ctypes.c_int),
            ]

        # allocFlags union の最大サイズ = 8 bytes
        class CUmemAllocFlags(ctypes.Structure):
            _fields_ = [
                ("compressionType",      ctypes.c_uint8),
                ("gpuDirectRDMACapable", ctypes.c_uint8),
                ("usage",                ctypes.c_uint16),
                ("reserved",             ctypes.c_uint8 * 4),
            ]

        # CUmemAllocationProp: 正確なレイアウト = 32 bytes
        # int(4) + int(4) + CUmemLocation(8) + ptr(8) + allocFlags(8)
        class CUmemAllocationProp(ctypes.Structure):
            _fields_ = [
                ("type",                 ctypes.c_int),
                ("requestedHandleTypes", ctypes.c_int),
                ("location",             CUmemLocation),
                ("win32HandleMetaData",  ctypes.c_void_p),
                ("allocFlags",           CUmemAllocFlags),
            ]

        # Win32ハンドル使用時は SECURITY_ATTRIBUTES が必須
        # NULL のままだと INVALID_VALUE になる (Windows ドライバーの仕様)
        class SECURITY_ATTRIBUTES(ctypes.Structure):
            _fields_ = [
                ("nLength",              ctypes.wintypes.DWORD),
                ("lpSecurityDescriptor", ctypes.c_void_p),
                ("bInheritHandle",       ctypes.wintypes.BOOL),
            ]

        sa = SECURITY_ATTRIBUTES()
        sa.nLength = ctypes.sizeof(sa)
        sa.lpSecurityDescriptor = None  # デフォルトセキュリティ
        sa.bInheritHandle = True

        prop = CUmemAllocationProp()
        prop.type = 1                   # CU_MEM_ALLOCATION_TYPE_PINNED
        prop.requestedHandleTypes = 0x2 # CU_MEM_HANDLE_TYPE_WIN32
        prop.location.type = 1          # CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = 0
        prop.win32HandleMetaData = ctypes.cast(ctypes.pointer(sa), ctypes.c_void_p)

        sz = ctypes.sizeof(prop)
        print(f"  CUmemAllocationProp サイズ: {sz} bytes (期待値: 32)")
        if sz != 32:
            print(f"  警告: サイズ不一致 → cuMemCreate は誤動作する可能性")

        GRANULARITY = 2 * 1024 * 1024  # probe1 で確認済み
        alloc_size = GRANULARITY

        # CUmemGenericAllocationHandle = unsigned long long (64-bit値)
        handle_mem = ctypes.c_uint64(0)
        result = cuda.cuMemCreate(
            ctypes.byref(handle_mem),
            ctypes.c_size_t(alloc_size),
            ctypes.byref(prop),
            ctypes.c_ulonglong(0)
        )
        print(f"cuMemCreate: {'OK' if result == 0 else f'NG エラー {result}'}")

        if result != 0:
            codes = {1: "INVALID_VALUE", 2: "OUT_OF_MEMORY", 801: "NOT_SUPPORTED"}
            print(f"  エラー詳細: {codes.get(result, '不明')}")
            return False

        # Win32ハンドルをエクスポート
        # handle_mem (c_uint64) は値渡し
        win32_handle = ctypes.c_void_p()
        CU_MEM_HANDLE_TYPE_WIN32 = 0x2
        result_exp = cuda.cuMemExportToShareableHandle(
            ctypes.byref(win32_handle),
            handle_mem,
            ctypes.c_int(CU_MEM_HANDLE_TYPE_WIN32),
            ctypes.c_ulonglong(0)
        )
        print(f"cuMemExportToShareableHandle: {'OK' if result_exp == 0 else f'NG エラー {result_exp}'}")

        if result_exp == 0:
            print(f"  Win32ハンドル: {win32_handle.value}")
            print(f"  -> DX12 OpenSharedHandle / Vulkan External Memory で参照可能")

        # VA予約 + マップ
        class CUmemAccessDesc(ctypes.Structure):
            _fields_ = [
                ("location", CUmemLocation),
                ("flags",    ctypes.c_int),
            ]

        va_ptr = ctypes.c_size_t(0)
        r_va = cuda.cuMemAddressReserve(
            ctypes.byref(va_ptr),
            ctypes.c_size_t(alloc_size),
            ctypes.c_size_t(0),
            ctypes.c_size_t(0),
            ctypes.c_ulonglong(0)
        )
        print(f"cuMemAddressReserve: {'OK' if r_va == 0 else f'NG エラー {r_va}'}")

        if r_va == 0:
            cuda.cuMemMap(
                va_ptr,
                ctypes.c_size_t(alloc_size),
                ctypes.c_size_t(0),
                handle_mem,
                ctypes.c_ulonglong(0)
            )
            access_desc = CUmemAccessDesc()
            access_desc.location.type = 1
            access_desc.location.id = 0
            access_desc.flags = 1  # READ_WRITE
            cuda.cuMemSetAccess(va_ptr, ctypes.c_size_t(alloc_size),
                                ctypes.byref(access_desc), ctypes.c_size_t(1))

            print(f"  マップ済みVA: 0x{va_ptr.value:016X}")
            print(f"  -> Win32共有ハンドル付きVRAM確保完了")

            cuda.cuMemUnmap(va_ptr, ctypes.c_size_t(alloc_size))
            cuda.cuMemAddressFree(va_ptr, ctypes.c_size_t(alloc_size))

        cuda.cuMemRelease(handle_mem)
        return result_exp == 0

    except Exception as e:
        print(f"エラー: {e}")
        return False


# ============================================================
# STEP 6: OpenVINO NPU remote_tensor サポート確認
# ============================================================
def probe_npu_remote_context():
    print("\n" + "=" * 60)
    print("STEP 6: OpenVINO NPU remote_tensor / RemoteContext 確認")
    print("=" * 60)

    try:
        import openvino as ov
        core = ov.Core()

        plugins = core.available_devices
        print(f"利用可能デバイス: {plugins}")

        if "CUDA" in plugins:
            print("OpenVINO CUDA plugin 検出 -> CUDAコンテキスト共有が可能")
        else:
            print("OpenVINO CUDA plugin 未検出")
            print("   -> NPU<->CUDA 直接ゼロコピーは不可")

        try:
            ctx = core.get_property("NPU", "AVAILABLE_DEVICES")
            print(f"NPU AVAILABLE_DEVICES: {ctx}")
        except Exception:
            pass

        if "GPU" in plugins:
            print("\n--- Intel iGPU RemoteContext ---")
            try:
                remote_ctx = core.create_context("GPU", {})
                print(f"iGPU RemoteContext 作成成功: {type(remote_ctx)}")
                print("   -> NPU->iGPU は RemoteTensor 経由でゼロコピー可能")
                print("   -> RTX5080との直接共有ではない点に注意")
            except Exception as e:
                print(f"iGPU RemoteContext: {e}")

    except ImportError:
        print("OpenVINO 未インストール")


# ============================================================
# STEP 7: NPU推論 -> CPU pinned -> GPU パイプライン実測
# openvino.runtime は廃止 -> ov.convert_model (PyTorch経由) を使用
# ============================================================
def measure_npu_to_gpu_pipeline():
    print("\n" + "=" * 60)
    print("STEP 7: NPU推論 -> CPU pinned -> GPU パイプライン計測")
    print("=" * 60)

    try:
        import openvino as ov
        import torch
        import torch.nn as nn

        if not torch.cuda.is_available():
            print("CUDA 不可")
            return

        core = ov.Core()

        # PyTorchモデルを OpenVINO に変換 (openvino.runtime 不要)
        print("ダミーNPUモデル (512dim MLP) を作成中...")

        class DummyLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(512, 512, bias=False)

            def forward(self, x):
                return self.linear(x)

        torch_model = DummyLayer().eval()
        example_input = torch.randn(1, 512)

        ov_model = ov.convert_model(torch_model, example_input=example_input)

        # NPU は静的形状のみ受け付ける → reshape で [1, 512] に固定
        ov_model.reshape([1, 512])

        print("NPU にコンパイル中...")
        compiled = core.compile_model(ov_model, "NPU")
        infer_req = compiled.create_infer_request()

        dummy_input = np.random.randn(1, 512).astype(np.float32)

        # ウォームアップ (コールドスタート除外)
        for _ in range(3):
            infer_req.infer({0: dummy_input})

        # NPU推論レイテンシ計測
        N = 20
        times_npu = []
        for _ in range(N):
            t0 = time.perf_counter()
            infer_req.infer({0: dummy_input})
            t1 = time.perf_counter()
            times_npu.append((t1 - t0) * 1000)

        npu_ms = np.median(times_npu)
        print(f"NPU推論レイテンシ: {npu_ms:.2f}ms (中央値, n={N})")

        # NPU出力 -> CPU pinned memory -> GPU 転送計測
        npu_output = infer_req.get_output_tensor(0).data
        output_bytes = npu_output.nbytes

        times_transfer = []
        for _ in range(N):
            t0 = time.perf_counter()
            pinned = torch.from_numpy(npu_output.copy()).pin_memory()
            gpu_tensor = pinned.cuda(non_blocking=True)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times_transfer.append((t1 - t0) * 1000)

        transfer_ms = np.median(times_transfer)
        total_ms = npu_ms + transfer_ms

        print(f"転送レイテンシ ({output_bytes} bytes): {transfer_ms:.3f}ms")
        print(f"\n--- パイプライン合計 ---")
        print(f"  NPU推論:    {npu_ms:.2f}ms")
        print(f"  CPU->GPU:   {transfer_ms:.3f}ms")
        print(f"  合計:       {total_ms:.2f}ms")
        print(f"  転送オーバーヘッド: {transfer_ms/total_ms*100:.1f}%")

        if transfer_ms / total_ms < 0.1:
            print("\n-> 転送オーバーヘッド 10%未満 : CPU経由パイプラインで十分")
            print("   マルチスレッドで隠蔽可能 -> 実装フェーズへ移行")
        else:
            print("\n-> 転送オーバーヘッド大 : ゼロコピー最適化を検討")

    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()


# ============================================================
# メイン
# ============================================================
if __name__ == "__main__":
    print("dollama probe2 v2: Win32ハンドル共有 + NPUパイプライン検証")
    print("環境: Intel Core Ultra 9 285 (NPU=AI Boost) + RTX5080")
    print()

    step5_ok = probe_virtual_memory_handle()
    probe_npu_remote_context()
    measure_npu_to_gpu_pipeline()

    print("\n" + "=" * 60)
    print("probe2 完了")
    print("=" * 60)
    print("""
次フェーズ判断:
  STEP5 OK + STEP7 オーバーヘッド < 10%
    -> CPU経由パイプライン + マルチスレッドで実装開始

  STEP5 OK + STEP7 オーバーヘッド >= 10%
    -> DX12/Vulkan External Memory でのゼロコピーを調査

  STEP5 NG (INVALID_VALUE 継続)
    -> Win32セキュリティ記述子が必要な可能性
    -> SECURITY_ATTRIBUTES を付けた cuMemCreate を試す
""")

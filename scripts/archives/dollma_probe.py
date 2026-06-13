"""
dollma - NPU→GPU memory bridge / 二次元特化 軽量画像生成
========================================================
probe.py : NPU→GPU メモリマッピング調査スクリプト

実行環境: Windows + Intel Core Ultra 9 285 + RTX5080

調査項目:
1. OpenVINOでNPUデバイスとメモリ情報
2. CUDAでVRAMアドレス空間
3. cuMemImportFromShareableHandle の可能性
4. 転送レイテンシの計測
"""

import time
import numpy as np

# ============================================================
# STEP 1: OpenVINO NPU デバイス調査
# ============================================================
def probe_npu():
    print("=" * 50)
    print("STEP 1: NPU デバイス調査 (OpenVINO)")
    print("=" * 50)
    
    try:
        import openvino as ov
        core = ov.Core()
        devices = core.available_devices
        print(f"検出デバイス: {devices}")
        
        if "NPU" in devices:
            print("\n✅ NPU検出!")
            
            # NPUのプロパティを全部取得
            props_to_check = [
                "FULL_DEVICE_NAME",
                "DEVICE_TYPE", 
                "OPTIMIZATION_CAPABILITIES",
                "RANGE_FOR_ASYNC_INFER_REQUESTS",
                "PERFORMANCE_HINT",
                "DEVICE_ARCHITECTURE",
            ]
            
            for prop in props_to_check:
                try:
                    val = core.get_property("NPU", prop)
                    print(f"  {prop}: {val}")
                except Exception as e:
                    print(f"  {prop}: 取得不可 ({e})")
                    
            # NPUのメモリ情報
            try:
                mem = core.get_property("NPU", "DEVICE_GFLOPS")
                print(f"  GFLOPS: {mem}")
            except:
                pass
                
        else:
            print("❌ NPU未検出 (ドライバー確認が必要)")
            print(f"  検出済みデバイス: {devices}")
            
    except ImportError:
        print("OpenVINO未インストール: pip install openvino")

# ============================================================
# STEP 2: CUDA VRAMアドレス空間調査
# ============================================================
def probe_cuda():
    print("\n" + "=" * 50)
    print("STEP 2: CUDA VRAM アドレス空間調査")
    print("=" * 50)
    
    try:
        import ctypes
        cuda = ctypes.CDLL("nvcuda.dll")  # Windows
        
        # CUDA初期化
        result = cuda.cuInit(0)
        print(f"cuInit: {'✅ OK' if result == 0 else f'❌ エラー {result}'}")
        
        if result != 0:
            return
            
        # デバイス数
        count = ctypes.c_int(0)
        cuda.cuDeviceGetCount(ctypes.byref(count))
        print(f"CUDAデバイス数: {count.value}")
        
        # デバイス情報
        device = ctypes.c_int(0)
        cuda.cuDeviceGet(ctypes.byref(device), 0)
        
        name = ctypes.create_string_buffer(256)
        cuda.cuDeviceGetName(name, 256, device)
        print(f"デバイス名: {name.value.decode()}")
        
        # コンテキスト作成
        ctx = ctypes.c_void_p()
        cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, device)
        
        # VRAMメモリ割り当てとアドレス取得
        ptr = ctypes.c_size_t(0)
        size = 1024 * 1024  # 1MB
        result = cuda.cuMemAlloc_v2(ctypes.byref(ptr), size)
        
        if result == 0:
            print(f"\n✅ VRAM割り当て成功")
            print(f"  VRAMアドレス: 0x{ptr.value:016X}")
            print(f"  サイズ: {size // 1024}KB")
            
            # ShareableHandleの取得を試みる
            # cuMemExportToShareableHandle → 外部からのマップに必要
            handle = ctypes.c_void_p()
            HANDLE_TYPE_WIN32 = 1
            result2 = cuda.cuMemExportToShareableHandle(
                ctypes.byref(handle), ptr, HANDLE_TYPE_WIN32, 0
            )
            if result2 == 0:
                print(f"\n✅ ShareableHandle取得成功!")
                print(f"  Handle: {handle.value}")
                print(f"  → NPU側からこのHandleでマップ可能な可能性あり")
            else:
                print(f"\n⚠️  ShareableHandle取得失敗 (エラー: {result2})")
                print(f"  → cuMemAllocは使えるがExportは制限あり")
                print(f"  → cuMemCreate (Virtual Memory)で再試行が必要")
                
            cuda.cuMemFree_v2(ptr)
        else:
            print(f"❌ VRAM割り当て失敗: {result}")
            
    except OSError:
        print("nvcuda.dll が見つかりません")
        print("→ NVIDIAドライバーがインストールされているか確認")

# ============================================================
# STEP 3: CUDA Virtual Memory API 調査
# (cuMemCreate → cuMemMap → 外部共有の本命)
# ============================================================
def probe_cuda_virtual_memory():
    print("\n" + "=" * 50)
    print("STEP 3: CUDA Virtual Memory API 調査")
    print("=" * 50)
    
    try:
        import ctypes
        cuda = ctypes.CDLL("nvcuda.dll")
        
        # CUmemAllocationProp構造体
        class CUmemAllocationProp(ctypes.Structure):
            _fields_ = [
                ("type", ctypes.c_int),
                ("requestedHandleTypes", ctypes.c_int),
                ("location_type", ctypes.c_int),
                ("location_id", ctypes.c_int),
                ("win32HandleMetaData", ctypes.c_void_p),
                ("allocFlags_usage", ctypes.c_uint),
                ("allocFlags_reserved", ctypes.c_uint * 4),
            ]
        
        prop = CUmemAllocationProp()
        prop.type = 1  # CU_MEM_ALLOCATION_TYPE_PINNED
        prop.requestedHandleTypes = 2  # CU_MEM_HANDLE_TYPE_WIN32 = 0x2
        prop.location_type = 1  # CU_MEM_LOCATION_TYPE_DEVICE
        prop.location_id = 0   # GPU 0
        
        # granularityを取得
        granularity = ctypes.c_size_t(0)
        result = cuda.cuMemGetAllocationGranularity(
            ctypes.byref(granularity),
            ctypes.byref(prop),
            1  # CU_MEM_ALLOC_GRANULARITY_MINIMUM
        )
        
        if result == 0:
            print(f"✅ Virtual Memory API 利用可能")
            print(f"  アロケーション粒度: {granularity.value} bytes")
            print(f"  → Win32 Shareableハンドルでの共有が可能")
            print(f"  → これがNPU→GPU直接マップの鍵になる可能性")
        else:
            print(f"❌ Virtual Memory API エラー: {result}")
            
    except Exception as e:
        print(f"エラー: {e}")

# ============================================================
# STEP 4: 転送レイテンシ計測
# (現状のCPU経由と比較ベースラインを取る)
# ============================================================
def measure_transfer_latency():
    print("\n" + "=" * 50)
    print("STEP 4: 転送レイテンシ計測 (ベースライン)")
    print("=" * 50)
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("❌ CUDA不可 (PyTorch)")
            return
            
        sizes = [
            ("1MB",   1 * 1024 * 1024),
            ("10MB",  10 * 1024 * 1024),
            ("100MB", 100 * 1024 * 1024),
        ]
        
        print(f"{'サイズ':<10} {'CPU→VRAM':<15} {'VRAM→CPU':<15} {'帯域(GB/s)':<10}")
        print("-" * 50)
        
        for name, size in sizes:
            n = size // 4  # float32
            
            # CPU (pinned memory) → VRAM
            cpu_tensor = torch.zeros(n, dtype=torch.float32).pin_memory()
            
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            gpu_tensor = cpu_tensor.cuda(non_blocking=True)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            upload_ms = (t1 - t0) * 1000
            
            # VRAM → CPU
            t0 = time.perf_counter()
            back = gpu_tensor.cpu()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            download_ms = (t1 - t0) * 1000
            
            bandwidth = (size / 1e9) / ((upload_ms / 1000))
            print(f"{name:<10} {upload_ms:.2f}ms{'':<8} {download_ms:.2f}ms{'':<8} {bandwidth:.1f}")
            
        print("\n→ NPU経由の場合はこれに NPU処理時間が加算される")
        print("→ 上記と比較してオーバーヘッドを評価する")
        
    except ImportError:
        print("PyTorch未インストール: pip install torch")

# ============================================================
# メイン
# ============================================================
if __name__ == "__main__":
    print("NPU→GPU メモリマッピング調査")
    print("環境: Intel Core Ultra 9 285 + RTX5080")
    print()
    
    probe_npu()
    probe_cuda()
    probe_cuda_virtual_memory()
    measure_transfer_latency()
    
    print("\n" + "=" * 50)
    print("調査完了")
    print("=" * 50)
    print("""
次のステップ:
1. STEP3でVirtual Memory APIが使えた場合
   → cuMemCreate + Win32ハンドルでNPUとの共有を試みる
   
2. OpenVINOでNPUのremote_tensorが使えた場合
   → NPU側からVRAMハンドルを参照できるか試す
   
3. 両方ダメな場合
   → RING0ドライバーでPCIeバスを直接叩く方向に
""")

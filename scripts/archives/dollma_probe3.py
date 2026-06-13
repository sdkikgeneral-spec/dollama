"""
dollama probe3 - D3D12 クロスアダプター経路調査
============================================================

未探索ルート: RTX5080(D3D12) → 共有ハンドル → Intel iGPU(D3D12)
              → OpenVINO RemoteContext → NPU

STEP 8:  DXGI アダプター列挙
STEP 9:  D3D12 デバイス作成 (RTX5080 + iGPU)
STEP 10: クロスアダプターヒープ + 共有ハンドル
STEP 11: OpenVINO iGPU RemoteContext に D3D12 バッファを渡す
"""

import ctypes
import ctypes.wintypes
import numpy as np
import time

HRESULT = ctypes.c_int32

# --- GUID ---
class GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_uint32),
        ("Data2", ctypes.c_uint16),
        ("Data3", ctypes.c_uint16),
        ("Data4", ctypes.c_uint8 * 8),
    ]

    @classmethod
    def from_str(cls, s):
        s = s.strip("{}")
        p = s.split("-")
        g = cls()
        g.Data1 = int(p[0], 16)
        g.Data2 = int(p[1], 16)
        g.Data3 = int(p[2], 16)
        d4 = bytes.fromhex(p[3] + p[4])
        for i, b in enumerate(d4):
            g.Data4[i] = b
        return g

# COM vtable ヘルパー (64bit Windows 専用)
def vtcall(obj, idx, restype, argtypes, *args):
    vtable = ctypes.c_size_t.from_address(obj).value
    fn     = ctypes.c_size_t.from_address(vtable + idx * 8).value
    F = ctypes.WINFUNCTYPE(restype, ctypes.c_size_t, *argtypes)
    return F(fn)(obj, *args)

def com_release(obj):
    if obj:
        vtcall(obj, 2, ctypes.c_uint32, [])

# --- IIDs ---
IID_IDXGIFactory4  = GUID.from_str("{1BC6EA02-EF36-464F-BF0C-21CA39E5168A}")
IID_IDXGIAdapter1  = GUID.from_str("{29038F61-3839-4626-91FD-086879011A05}")
IID_ID3D12Device   = GUID.from_str("{189819F1-1DB6-4B57-BE54-1821339B85F7}")
IID_ID3D12Heap     = GUID.from_str("{6B3B2502-6E51-45B3-90EE-9884265E8DF3}")
IID_ID3D12Resource = GUID.from_str("{696442BE-A72E-4059-BC79-5B5C98040FAD}")

# --- 構造体 ---
class LUID(ctypes.Structure):
    _fields_ = [("LowPart", ctypes.c_ulong), ("HighPart", ctypes.c_long)]

class DXGI_ADAPTER_DESC1(ctypes.Structure):
    _fields_ = [
        ("Description",           ctypes.c_wchar * 128),
        ("VendorId",              ctypes.c_uint),
        ("DeviceId",              ctypes.c_uint),
        ("SubSysId",              ctypes.c_uint),
        ("Revision",              ctypes.c_uint),
        ("DedicatedVideoMemory",  ctypes.c_size_t),
        ("DedicatedSystemMemory", ctypes.c_size_t),
        ("SharedSystemMemory",    ctypes.c_size_t),
        ("AdapterLuid",           LUID),
        ("Flags",                 ctypes.c_uint),
    ]

class D3D12_HEAP_PROPERTIES(ctypes.Structure):
    _fields_ = [
        ("Type",                  ctypes.c_int),
        ("CPUPageProperty",       ctypes.c_int),
        ("MemoryPoolPreference",  ctypes.c_int),
        ("CreationNodeMask",      ctypes.c_uint),
        ("VisibleNodeMask",       ctypes.c_uint),
    ]

class D3D12_HEAP_DESC(ctypes.Structure):
    _fields_ = [
        ("SizeInBytes", ctypes.c_uint64),
        ("Properties",  D3D12_HEAP_PROPERTIES),  # 20 bytes → 次フィールドに4byteパディング発生
        ("Alignment",   ctypes.c_uint64),
        ("Flags",       ctypes.c_uint),
    ]

class SECURITY_ATTRIBUTES(ctypes.Structure):
    _fields_ = [
        ("nLength",              ctypes.c_uint32),
        ("lpSecurityDescriptor", ctypes.c_void_p),
        ("bInheritHandle",       ctypes.c_int),
    ]

# --- 定数 ---
VENDOR_NVIDIA = 0x10DE
VENDOR_INTEL  = 0x8086
D3D_FEATURE_LEVEL_11_0          = 0xb000
D3D12_HEAP_TYPE_CUSTOM          = 4
D3D12_CPU_PAGE_PROPERTY_WRITE_BACK = 3
D3D12_MEMORY_POOL_L0            = 1     # system RAM
D3D12_HEAP_FLAG_SHARED          = 0x4
D3D12_HEAP_FLAG_SHARED_CROSS_ADAPTER = 0x20
D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS   = 0x40
GENERIC_ALL                     = 0x10000000
DXGI_ERROR_NOT_FOUND            = 0x887A0002


# ============================================================
# STEP 8: DXGI アダプター列挙
# ============================================================
def step8_enumerate_adapters():
    print("=" * 60)
    print("STEP 8: DXGI アダプター列挙")
    print("=" * 60)

    try:
        dxgi = ctypes.WinDLL("dxgi.dll")
    except OSError:
        print("dxgi.dll ロード失敗")
        return None, None

    factory = ctypes.c_size_t(0)
    hr = dxgi.CreateDXGIFactory2(0, ctypes.byref(IID_IDXGIFactory4), ctypes.byref(factory))
    if hr != 0:
        print(f"CreateDXGIFactory2 失敗: 0x{hr & 0xFFFFFFFF:08X}")
        return None, None

    print(f"IDXGIFactory4: OK")

    nvidia_adapter = None
    intel_adapter  = None

    # --- EnumAdapters1 (標準列挙) ---
    print("\n[EnumAdapters1]")
    for i in range(16):
        adapter = ctypes.c_size_t(0)
        hr = vtcall(factory.value, 12, HRESULT,
                    [ctypes.c_uint, ctypes.POINTER(ctypes.c_size_t)],
                    i, ctypes.byref(adapter))
        if hr & 0xFFFFFFFF == DXGI_ERROR_NOT_FOUND:
            break
        if hr != 0:
            break

        desc = DXGI_ADAPTER_DESC1()
        vtcall(adapter.value, 10, HRESULT,
               [ctypes.POINTER(DXGI_ADAPTER_DESC1)],
               ctypes.byref(desc))

        vram_gb = desc.DedicatedVideoMemory / (1024 ** 3)
        print(f"  [{i}] {desc.Description}  VendorId:0x{desc.VendorId:04X}  VRAM:{vram_gb:.1f}GB")

        if desc.VendorId == VENDOR_NVIDIA and nvidia_adapter is None:
            print(f"       -> NVIDIA GPU")
            nvidia_adapter = adapter.value
        elif desc.VendorId == VENDOR_INTEL and intel_adapter is None:
            print(f"       -> Intel iGPU")
            intel_adapter = adapter.value
        else:
            com_release(adapter.value)

    # --- IDXGIFactory6::EnumAdapterByGpuPreference で iGPU を探す ---
    # iGPU は discrete GPU 接続時に EnumAdapters1 から消えることがある
    # DXGI_GPU_PREFERENCE_MINIMUM_POWER(2) で省電力=iGPU を優先列挙
    if intel_adapter is None:
        print("\n[IDXGIFactory6::EnumAdapterByGpuPreference (MINIMUM_POWER)]")
        IID_IDXGIFactory6 = GUID.from_str("{C1B6694F-FF09-44A9-B03C-77900A0A1D17}")
        factory6 = ctypes.c_size_t(0)

        # IDXGIFactory4::QueryInterface → vtable index 0
        hr = vtcall(factory.value, 0, HRESULT,
                    [ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_size_t)],
                    ctypes.byref(IID_IDXGIFactory6), ctypes.byref(factory6))

        if hr == 0 and factory6.value:
            # IDXGIFactory6 vtable layout:
            # IUnknown(0-2) + IDXGIObject(3-6) + IDXGIFactory(7-11)
            # + IDXGIFactory1(12-13) + IDXGIFactory2(14-24) + IDXGIFactory3(25)
            # + IDXGIFactory4(26-27) + IDXGIFactory5(28) + IDXGIFactory6(29-30)
            # EnumAdapterByGpuPreference = index 29
            DXGI_GPU_PREFERENCE_MINIMUM_POWER = 1
            for i in range(16):
                adapter = ctypes.c_size_t(0)
                hr = vtcall(factory6.value, 29, HRESULT,
                            [ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_size_t)],
                            i, DXGI_GPU_PREFERENCE_MINIMUM_POWER,
                            ctypes.byref(IID_IDXGIAdapter1), ctypes.byref(adapter))
                if hr & 0xFFFFFFFF == DXGI_ERROR_NOT_FOUND:
                    break
                if hr != 0:
                    print(f"  EnumAdapterByGpuPreference エラー: 0x{hr & 0xFFFFFFFF:08X}")
                    break

                desc = DXGI_ADAPTER_DESC1()
                vtcall(adapter.value, 10, HRESULT,
                       [ctypes.POINTER(DXGI_ADAPTER_DESC1)],
                       ctypes.byref(desc))

                vram_gb = desc.DedicatedVideoMemory / (1024 ** 3)
                print(f"  [{i}] {desc.Description}  VendorId:0x{desc.VendorId:04X}  VRAM:{vram_gb:.1f}GB")

                if desc.VendorId == VENDOR_INTEL and intel_adapter is None:
                    print(f"       -> Intel iGPU (MINIMUM_POWER で発見)")
                    intel_adapter = adapter.value
                else:
                    com_release(adapter.value)
            com_release(factory6.value)
        else:
            print(f"  IDXGIFactory6 取得失敗: 0x{hr & 0xFFFFFFFF:08X}")

    if intel_adapter is None:
        print("\n  -> Intel iGPU は DXGI に非表示 (コンピュート専用モード)")
        print("  -> Level Zero / OpenCL 経由でのみアクセス可能")
        print("  -> D3D12 クロスアダプター経路は このシステム構成では不可")

    com_release(factory.value)
    return nvidia_adapter, intel_adapter


# ============================================================
# STEP 9: D3D12 デバイス作成
# ============================================================
def step9_create_devices(nvidia_adapter, intel_adapter):
    print("\n" + "=" * 60)
    print("STEP 9: D3D12 デバイス作成")
    print("=" * 60)

    try:
        d3d12 = ctypes.WinDLL("d3d12.dll")
    except OSError:
        print("d3d12.dll ロード失敗")
        return None, None

    d3d12.D3D12CreateDevice.restype = HRESULT
    d3d12.D3D12CreateDevice.argtypes = [
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.POINTER(GUID),
        ctypes.POINTER(ctypes.c_size_t),
    ]

    nv_dev = ctypes.c_size_t(0)
    hr = d3d12.D3D12CreateDevice(
        nvidia_adapter, D3D_FEATURE_LEVEL_11_0,
        ctypes.byref(IID_ID3D12Device), ctypes.byref(nv_dev))
    print(f"NVIDIA D3D12Device: {'OK' if hr == 0 else f'NG 0x{hr & 0xFFFFFFFF:08X}'}")

    ig_dev = ctypes.c_size_t(0)
    hr = d3d12.D3D12CreateDevice(
        intel_adapter, D3D_FEATURE_LEVEL_11_0,
        ctypes.byref(IID_ID3D12Device), ctypes.byref(ig_dev))
    print(f"Intel D3D12Device:  {'OK' if hr == 0 else f'NG 0x{hr & 0xFFFFFFFF:08X}'}")

    return nv_dev.value if nv_dev.value else None, ig_dev.value if ig_dev.value else None


# ============================================================
# STEP 10: クロスアダプターヒープ + 共有ハンドル
# ============================================================
def step10_cross_adapter(nv_dev, ig_dev):
    print("\n" + "=" * 60)
    print("STEP 10: D3D12 クロスアダプターヒープ + 共有ハンドル")
    print("=" * 60)

    HEAP_SIZE = 2 * 1024 * 1024  # 2MB

    # クロスアダプターヒープは system RAM (L0 pool) に配置する必要がある
    heap_desc = D3D12_HEAP_DESC()
    heap_desc.SizeInBytes                       = HEAP_SIZE
    heap_desc.Properties.Type                  = D3D12_HEAP_TYPE_CUSTOM
    heap_desc.Properties.CPUPageProperty       = D3D12_CPU_PAGE_PROPERTY_WRITE_BACK
    heap_desc.Properties.MemoryPoolPreference  = D3D12_MEMORY_POOL_L0
    heap_desc.Properties.CreationNodeMask      = 1
    heap_desc.Properties.VisibleNodeMask       = 1
    heap_desc.Alignment                         = 0
    heap_desc.Flags = (D3D12_HEAP_FLAG_SHARED |
                       D3D12_HEAP_FLAG_SHARED_CROSS_ADAPTER |
                       D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS)

    print(f"D3D12_HEAP_DESC size: {ctypes.sizeof(heap_desc)} bytes")

    # CreateHeap → ID3D12Device vtable index 28
    heap = ctypes.c_size_t(0)
    hr = vtcall(nv_dev, 28, HRESULT,
                [ctypes.POINTER(D3D12_HEAP_DESC), ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_size_t)],
                ctypes.byref(heap_desc), ctypes.byref(IID_ID3D12Heap), ctypes.byref(heap))
    print(f"CreateHeap (SHARED_CROSS_ADAPTER): {'OK' if hr == 0 else f'NG 0x{hr & 0xFFFFFFFF:08X}'}")

    if hr != 0:
        return None, None

    # CreateSharedHandle → vtable index 31
    sa = SECURITY_ATTRIBUTES()
    sa.nLength = ctypes.sizeof(sa)
    sa.bInheritHandle = True

    nt_handle = ctypes.c_void_p()
    hr = vtcall(nv_dev, 31, HRESULT,
                [ctypes.c_size_t, ctypes.c_void_p, ctypes.c_uint32,
                 ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_void_p)],
                heap.value,
                ctypes.cast(ctypes.pointer(sa), ctypes.c_void_p),
                GENERIC_ALL, None, ctypes.byref(nt_handle))
    print(f"CreateSharedHandle: {'OK handle=' + str(nt_handle.value) if hr == 0 else f'NG 0x{hr & 0xFFFFFFFF:08X}'}")

    if hr != 0:
        com_release(heap.value)
        return None, None

    # Intel iGPU で OpenSharedHandle → vtable index 32
    shared_heap = ctypes.c_size_t(0)
    hr = vtcall(ig_dev, 32, HRESULT,
                [ctypes.c_void_p, ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_size_t)],
                nt_handle, ctypes.byref(IID_ID3D12Heap), ctypes.byref(shared_heap))
    print(f"OpenSharedHandle (Intel iGPU): {'OK' if hr == 0 else f'NG 0x{hr & 0xFFFFFFFF:08X}'}")

    if hr == 0:
        print(f"  -> system RAM ヒープを両アダプターが参照")
        print(f"  -> NVIDIA DMA がこのヒープに書き込み → CPU コピー不要")
        return heap.value, shared_heap.value

    com_release(heap.value)
    return None, None


# ============================================================
# STEP 11: OpenVINO iGPU RemoteContext + D3D12 バッファ
# ============================================================
def step11_openvino_d3d12(ig_dev):
    print("\n" + "=" * 60)
    print("STEP 11: OpenVINO iGPU RemoteContext + D3D12 interop")
    print("=" * 60)

    try:
        import openvino as ov
        core = ov.Core()

        # D3D12 デバイスポインタを OpenVINO に渡す
        # OpenVINO GPU plugin は CONTEXT_TYPE: D3D12_DEVICE を受け付ける (バージョン依存)
        ctx = None

        for ctx_type in ["D3D12_DEVICE", "D3D12", "DX12"]:
            try:
                ctx = core.create_context("GPU", {
                    "CONTEXT_TYPE": ctx_type,
                    "D3D12_DEVICE": ig_dev,
                })
                print(f"OpenVINO D3D12 Context ({ctx_type}): OK -> {type(ctx)}")
                break
            except Exception as e:
                print(f"  CONTEXT_TYPE={ctx_type}: NG ({e})")

        if ctx is None:
            print("\n  -> D3D12 直接渡しは未対応")
            print("  -> 代替: OCL RemoteContext + cl_mem 経由を確認")

            # OCL RemoteContext (probe2 で確認済み) + OpenCL-D3D12 interop の可能性
            try:
                ocl_ctx = core.create_context("GPU", {})
                print(f"\nOCL RemoteContext: OK -> {type(ocl_ctx)}")

                # OpenCL の cl_context ポインタを取得
                # これを通じて clCreateFromD3D12BufferKHR が使えるか確認
                params = ocl_ctx.get_params()
                print(f"Context params: {list(params.keys())}")
                if "OCL_CONTEXT" in params:
                    cl_ctx = params["OCL_CONTEXT"]
                    print(f"  cl_context ptr: 0x{cl_ctx:016X}")
                    print(f"  -> clCreateFromD3D12BufferKHR (cl_khr_d3d12_sharing) で")
                    print(f"     D3D12 バッファ → cl_mem → OpenVINO テンソル が理論上可能")
            except Exception as e:
                print(f"OCL fallback: {e}")

    except ImportError:
        print("OpenVINO 未インストール")


# ============================================================
# メイン
# ============================================================
if __name__ == "__main__":
    print("dollama probe3: D3D12 クロスアダプター経路調査")
    print("環境: Intel Core Ultra 9 285 + RTX5080")
    print()

    nvidia_adapter, intel_adapter = step8_enumerate_adapters()

    if nvidia_adapter is None or intel_adapter is None:
        print("\nアダプター取得失敗 → 終了")
        exit(1)

    nv_dev, ig_dev = step9_create_devices(nvidia_adapter, intel_adapter)

    if nv_dev and ig_dev:
        nv_heap, ig_heap = step10_cross_adapter(nv_dev, ig_dev)
    else:
        nv_heap, ig_heap = None, None

    if ig_dev:
        step11_openvino_d3d12(ig_dev)

    print("\n" + "=" * 60)
    print("probe3 完了")
    print("=" * 60)
    print("""
判断基準:
  STEP10 OK (OpenSharedHandle 成功)
    -> system RAM ヒープを両アダプターが共有
    -> CPU コピー不要 (NVIDIA DMA が書く / Intel iGPU が読む)
    -> STEP11 の OpenCL-D3D12 interop が通れば OpenVINO への橋渡し完成

  STEP10 NG
    -> D3D12 クロスアダプター経路も詰まり
    -> CPU 経由 3.4% オーバーヘッドで確定、実装フェーズへ

  STEP11 OCL_CONTEXT 取得成功
    -> clCreateFromD3D12BufferKHR (cl_khr_d3d12_sharing 拡張) を試す価値あり
""")

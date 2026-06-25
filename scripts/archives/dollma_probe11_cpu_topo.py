"""
dollma_probe11_cpu_topo.py - Core Ultra の P/E コアトポロジ確認
================================================================

GetLogicalProcessorInformationEx で各コアの EfficiencyClass を取得し、
P-core / E-core のアフィニティマスクを算出する。

出力例:
  P-core: 8コア (logical 16) mask=0x0000FFFF
  E-core: 16コア (logical 16) mask=0xFFFF0000
  → dollama スレッド割り当て案を表示
"""

import ctypes
import ctypes.wintypes as W
from collections import defaultdict

# ============================================================
# Windows API 構造体
# ============================================================
RelationProcessorCore = 0
RelationAll = 0xFFFF

class GROUP_AFFINITY(ctypes.Structure):
    _fields_ = [
        ("Mask",     ctypes.c_ulonglong),
        ("Group",    W.WORD),
        ("Reserved", W.WORD * 3),
    ]

class PROCESSOR_RELATIONSHIP(ctypes.Structure):
    _fields_ = [
        ("Flags",           ctypes.c_ubyte),
        ("EfficiencyClass", ctypes.c_ubyte),   # P-core > E-core
        ("Reserved",        ctypes.c_ubyte * 20),
        ("GroupCount",      W.WORD),
        ("GroupMask",       GROUP_AFFINITY * 1),  # 実際は GroupCount 個
    ]

class SLPIEX(ctypes.Structure):
    """SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX"""
    _fields_ = [
        ("Relationship", ctypes.c_uint),
        ("Size",         W.DWORD),
        ("Processor",    PROCESSOR_RELATIONSHIP),
    ]

def get_logical_processor_info():
    kernel32 = ctypes.windll.kernel32
    size = W.DWORD(0)
    # 必要バッファサイズを取得
    kernel32.GetLogicalProcessorInformationEx(RelationProcessorCore, None, ctypes.byref(size))
    buf = (ctypes.c_byte * size.value)()
    ok = kernel32.GetLogicalProcessorInformationEx(
        RelationProcessorCore, ctypes.cast(buf, ctypes.POINTER(SLPIEX)), ctypes.byref(size)
    )
    if not ok:
        raise OSError(f"GetLogicalProcessorInformationEx failed: {ctypes.get_last_error()}")
    return buf, size.value

# ============================================================
# STEP 1: P/E コア分類
# ============================================================
def analyze_cores():
    print("=" * 60)
    print("STEP 1: P/E コアトポロジ解析")
    print("=" * 60)

    buf, total = get_logical_processor_info()
    offset = 0
    cores = []  # (efficiency_class, mask)

    while offset < total:
        entry = SLPIEX.from_buffer_copy(buf, offset)
        if entry.Relationship == RelationProcessorCore:
            mask = entry.Processor.GroupMask[0].Mask
            ec   = entry.Processor.EfficiencyClass
            cores.append((ec, mask))
        offset += entry.Size

    # EfficiencyClass でグループ化 (大きい方が P-core)
    by_class = defaultdict(list)
    for ec, mask in cores:
        by_class[ec].append(mask)

    classes = sorted(by_class.keys(), reverse=True)
    labels  = ["P-core"] + ["E-core"] * (len(classes) - 1)
    if len(classes) == 3:
        labels = ["P-core", "E-core", "LP E-core"]

    result = {}
    for label, ec in zip(labels, classes):
        masks = by_class[ec]
        combined = 0
        logical_count = 0
        for m in masks:
            combined |= m
            logical_count += bin(m).count('1')
        physical = len(masks)
        print(f"  {label:10s}: {physical:2d}コア (logical {logical_count:2d})  "
              f"mask=0x{combined:016X}  EfficiencyClass={ec}")
        result[label] = {"mask": combined, "physical": physical, "logical": logical_count}

    return result

# ============================================================
# STEP 2: dollama スレッド割り当て案
# ============================================================
def suggest_affinity(topo):
    print()
    print("=" * 60)
    print("STEP 2: dollama スレッド割り当て案")
    print("=" * 60)

    p_mask  = topo.get("P-core",    {}).get("mask", 0)
    e_mask  = topo.get("E-core",    {}).get("mask", 0)
    lp_mask = topo.get("LP E-core", {}).get("mask", 0)

    assignments = [
        ("LLM (Qwen2 / BitNet)",    p_mask,           "重いシングルスレッド処理"),
        ("SDXL 制御スレッド",        p_mask,           "GPU への指示・同期"),
        ("WD14 tagger",             e_mask or p_mask, "GPU 生成中に並列実行"),
        ("パイプライン制御キュー",   lp_mask or e_mask,"軽量ルーティング"),
    ]

    for name, mask, note in assignments:
        print(f"  {name:25s}  mask=0x{mask:016X}  # {note}")

    print()
    print("  C++ での設定:")
    print("  SetThreadAffinityMask(thread.native_handle(), mask);")
    print()
    print("  Linux での設定:")
    print("  cpu_set_t cs; CPU_ZERO(&cs);")
    print("  // mask の各ビットを CPU_SET(bit, &cs) でセット")
    print("  pthread_setaffinity_np(thread.native_handle(), sizeof(cs), &cs);")

# ============================================================
# STEP 3: 現在のスレッドで動作確認
# ============================================================
def verify_affinity(topo):
    print()
    print("=" * 60)
    print("STEP 3: アフィニティ設定の動作確認")
    print("=" * 60)

    import threading, time

    p_mask = topo.get("P-core", {}).get("mask", 0)
    e_mask = topo.get("E-core", {}).get("mask", 0)

    if not p_mask or not e_mask:
        print("  P/E コアが区別できないためスキップ")
        return

    kernel32 = ctypes.windll.kernel32

    def burn(label, mask, duration=0.5):
        h = kernel32.OpenThread(0x0060, False, ctypes.windll.kernel32.GetCurrentThreadId())
        kernel32.SetThreadAffinityMask(h, mask)
        kernel32.CloseHandle(h)
        t0 = time.perf_counter()
        count = 0
        while time.perf_counter() - t0 < duration:
            count += 1
        elapsed = time.perf_counter() - t0
        print(f"  [{label}] {count/elapsed/1e6:.1f}M iter/s  (mask=0x{mask:X})")

    t_p = threading.Thread(target=burn, args=("P-core", p_mask))
    t_e = threading.Thread(target=burn, args=("E-core", e_mask))

    t_p.start(); t_e.start()
    t_p.join();  t_e.join()
    print("  → 両スレッドが干渉なく並列実行できれば OK")

# ============================================================
if __name__ == "__main__":
    topo = analyze_cores()
    suggest_affinity(topo)
    verify_affinity(topo)

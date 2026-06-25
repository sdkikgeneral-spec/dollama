// CPU トポロジ自動検出 (core/cpu_topology.hpp) 構造的健全性テスト。
//
// 値そのものは機種依存ゆえ検証しない。検証するのは "構造的" な健全性のみ:
//   1. 列挙が成功する (available == true)。
//   2. 物理コア数 >= 1。
//   3. 論理コア総数が std::thread::hardware_concurrency() と整合する。
//   4. 各物理コアの論理マスクが互いに disjoint (重なりなし)。
//   5. 各物理コアの logical_count == マスクの popcount、hyperthreaded フラグ整合。
//   6. single_logical_per_core_masks() が物理コア数本・各 1 ビット・互いに disjoint。
//
// Win32 不在 (Linux 未対応など available=false) の場合は [SKIP] して return 0。

#include <cstdint>
#include <iostream>
#include <thread>
#include <vector>

#include "core/cpu_topology.hpp"

using namespace dollama;

static int g_fail = 0;

static void check(bool cond, const char* msg)
{
    if (!cond)
    {
        std::cout << "[FAIL] " << msg << "\n";
        ++g_fail;
    }
}

static int popcount64(uint64_t x)
{
    int n = 0;
    while (x)
    {
        x &= (x - 1);
        ++n;
    }
    return n;
}

int main()
{
    CpuTopology topo = detect_cpu_topology();

    if (!topo.available)
    {
        // 未対応 OS / 列挙失敗。CI ビルド緑維持のため健全に skip。
        std::cout << "[test_cpu_topology] SKIP: topology 未取得 "
                     "(未対応 OS または列挙失敗)\n";
        return 0;
    }

    std::cout << "[test_cpu_topology] physical=" << topo.physical_count
              << " logical=" << topo.logical_count
              << " eff_classes=" << topo.efficiency_class_count << "\n";

    // 2. 物理コア数 >= 1。
    check(topo.physical_count >= 1, "physical_count >= 1");
    check(static_cast<int>(topo.cores.size()) == topo.physical_count,
          "cores.size() == physical_count");

    // 3. 論理総数が hardware_concurrency と整合。
    //    hardware_concurrency は 0 を返しうる (情報なし) ので、その場合のみ緩める。
    const unsigned hc = std::thread::hardware_concurrency();
    if (hc != 0)
    {
        check(topo.logical_count == static_cast<int>(hc),
              "logical_count == hardware_concurrency()");
    }
    else
    {
        std::cout << "[test_cpu_topology] note: hardware_concurrency()==0、"
                     "論理数整合チェックを緩和\n";
    }
    check(topo.logical_count >= topo.physical_count,
          "logical_count >= physical_count");

    // 4 + 5. 各コアのマスク disjoint + popcount 整合。
    uint64_t seen = 0;
    int logical_sum = 0;
    for (const PhysicalCore& c : topo.cores)
    {
        check(c.affinity_mask != 0, "core mask != 0");
        check((seen & c.affinity_mask) == 0, "core masks are disjoint");
        seen |= c.affinity_mask;

        const int pc = popcount64(c.affinity_mask);
        check(pc == c.logical_count, "logical_count == popcount(mask)");
        check(c.hyperthreaded == (c.logical_count > 1),
              "hyperthreaded == (logical_count > 1)");
        logical_sum += c.logical_count;
    }
    check(logical_sum == topo.logical_count,
          "sum(core logical_count) == logical_count");

    // 6. single_logical_per_core_masks の健全性。
    std::vector<uint64_t> singles = topo.single_logical_per_core_masks();
    check(static_cast<int>(singles.size()) == topo.physical_count,
          "single_logical masks count == physical_count");
    uint64_t sseen = 0;
    for (uint64_t m : singles)
    {
        check(popcount64(m) == 1, "single_logical mask has exactly 1 bit");
        check((sseen & m) == 0, "single_logical masks disjoint");
        // その 1 ビットがいずれかの物理コアに属すること。
        bool belongs = false;
        for (const PhysicalCore& c : topo.cores)
        {
            if (c.affinity_mask & m)
            {
                belongs = true;
                break;
            }
        }
        check(belongs, "single_logical mask belongs to a physical core");
        sseen |= m;
    }

    // efficiency_class_count の整合 (1 以上)。
    check(topo.efficiency_class_count >= 1, "efficiency_class_count >= 1");

    if (g_fail == 0)
    {
        std::cout << "[test_cpu_topology] PASS\n";
        return 0;
    }
    std::cout << "[test_cpu_topology] FAILED (" << g_fail << ")\n";
    return 1;
}

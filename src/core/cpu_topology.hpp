#pragma once

// CPU トポロジ自動検出 (OS 依存をこのファイルだけに閉じ込める)。
//
// 目的:
//   実機の「物理コア」「論理プロセッサ (HT 兄弟)」「EfficiencyClass (P/E)」を
//   世代非依存に列挙する。pipeline.hpp の決め打ちマスク (Arrow Lake 固有) を将来
//   置き換えるための基盤 (roadmap L221「HW 環境抽象化」)。
//
//   - Windows: GetLogicalProcessorInformationEx(RelationProcessorCore) で列挙。
//     KAFFINITY マスク / 論理ビット数 / EfficiencyClass を物理コア単位で得る。
//   - Linux:   /sys/devices/system/cpu/*/topology から best-effort。難しい場合は
//              「未対応 (available=false)」を返す (開発機 Windows 優先)。
//
// このファイルは "列挙のみ" を担当する。アフィニティ "設定" は core/affinity.hpp の
// set_current_thread_affinity を流用すること (再実装しない)。
//
// C++ スタイル: Allman・日本語コメント。

#include <cstdint>
#include <vector>
#include <algorithm>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__linux__)
#include <cstdio>
#include <cstring>
#include <string>
#include <set>
#endif

namespace dollama {

// 物理コア 1 個分の情報。
struct PhysicalCore
{
    uint64_t affinity_mask = 0;   // この物理コアに属する論理プロセッサのビットマスク
    int      logical_count = 0;   // 論理プロセッサ数 (HT なら 2、HT なしなら 1)
    int      efficiency_class = 0; // EfficiencyClass (数値大きいほど高性能 = P コア)
    bool     hyperthreaded = false; // logical_count > 1
};

// 実機トポロジ全体。
struct CpuTopology
{
    bool                       available = false; // 列挙に成功したか
    std::vector<PhysicalCore>  cores;             // 物理コア列
    int                        physical_count = 0; // 物理コア総数
    int                        logical_count = 0;  // 論理プロセッサ総数
    int                        efficiency_class_count = 0; // efficiency class 種類数

    // efficiency_class の最大値を持つ物理コア群 (= P コア候補)。
    // homogeneous 機 (efficiency_class 1 種) の場合は全コアが該当する。
    int max_efficiency_class() const
    {
        int m = 0;
        for (const PhysicalCore& c : cores)
        {
            m = std::max(m, c.efficiency_class);
        }
        return m;
    }

    int min_efficiency_class() const
    {
        if (cores.empty())
        {
            return 0;
        }
        int m = cores.front().efficiency_class;
        for (const PhysicalCore& c : cores)
        {
            m = std::min(m, c.efficiency_class);
        }
        return m;
    }

    // 指定 efficiency_class を持つ最初の物理コアの affinity マスクを返す (なければ 0)。
    uint64_t first_core_mask_of_class(int eff_class) const
    {
        for (const PhysicalCore& c : cores)
        {
            if (c.efficiency_class == eff_class)
            {
                return c.affinity_mask;
            }
        }
        return 0;
    }

    // 各物理コアの "1 論理プロセッサだけ" を選んだ disjoint マスク列を返す。
    // (HT 兄弟を踏まずに物理コア N 個へ 1 本ずつピン留めするスケール計測用。)
    // 並びは efficiency_class 降順 (P コア優先) で安定ソートする。
    std::vector<uint64_t> single_logical_per_core_masks() const
    {
        // efficiency_class 降順に物理コアを並べたインデックスを作る。
        std::vector<size_t> order(cores.size());
        for (size_t i = 0; i < cores.size(); ++i)
        {
            order[i] = i;
        }
        std::stable_sort(order.begin(), order.end(),
            [this](size_t a, size_t b)
            {
                return cores[a].efficiency_class > cores[b].efficiency_class;
            });

        std::vector<uint64_t> masks;
        masks.reserve(cores.size());
        for (size_t idx : order)
        {
            const uint64_t m = cores[idx].affinity_mask;
            // 最下位ビット (= その物理コアの最初の論理プロセッサ) のみ採用。
            const uint64_t lsb = m & (~m + 1ULL);
            if (lsb != 0)
            {
                masks.push_back(lsb);
            }
        }
        return masks;
    }
};

#ifdef _WIN32

// Windows 実装: GetLogicalProcessorInformationEx(RelationProcessorCore)。
inline CpuTopology detect_cpu_topology()
{
    CpuTopology topo;

    // 必要バッファ長を 0 呼び出しで取得する。
    DWORD len = 0;
    if (GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len))
    {
        // len=0 で成功するはずがない。失敗扱い。
        return topo;
    }
    if (GetLastError() != ERROR_INSUFFICIENT_BUFFER || len == 0)
    {
        return topo;
    }

    std::vector<uint8_t> buf(static_cast<size_t>(len));
    auto* info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data());
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, info, &len))
    {
        return topo;
    }

    // 可変長レコードを順に走査する。各レコードが 1 物理コア。
    uint8_t* ptr = buf.data();
    uint8_t* end = buf.data() + len;
    while (ptr < end)
    {
        auto* rec = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(ptr);
        if (rec->Size == 0)
        {
            break; // 異常 (無限ループ防止)
        }
        if (rec->Relationship == RelationProcessorCore)
        {
            PhysicalCore core;

            // EfficiencyClass はバイトオフセット固定で読む (古い MinGW ヘッダの
            // _PROCESSOR_RELATIONSHIP には EfficiencyClass フィールドがなく Reserved[21]
            // となっているため、named field では参照できない)。実 OS のメモリ配置は
            // PROCESSOR_RELATIONSHIP { BYTE Flags; BYTE EfficiencyClass; BYTE Reserved[20];
            // WORD GroupCount; ... } で固定なので、Processor 構造体先頭 +1 バイト目を読む。
            const uint8_t* prel =
                reinterpret_cast<const uint8_t*>(&rec->Processor);
            core.efficiency_class = static_cast<int>(prel[1]);

            // GroupCount は通常 1 (64 論理以下の単一プロセッサグループ)。
            // 複数グループ環境では先頭グループのみ採用する (uint64 マスク前提)。
            uint64_t mask = 0;
            int bits = 0;
            if (rec->Processor.GroupCount >= 1)
            {
                const KAFFINITY km = rec->Processor.GroupMask[0].Mask;
                mask = static_cast<uint64_t>(km);
                for (int b = 0; b < 64; ++b)
                {
                    if (mask & (1ULL << b))
                    {
                        ++bits;
                    }
                }
            }
            core.affinity_mask = mask;
            core.logical_count = bits;
            core.hyperthreaded = (bits > 1);

            if (mask != 0)
            {
                topo.cores.push_back(core);
            }
        }
        ptr += rec->Size;
    }

    if (topo.cores.empty())
    {
        return topo; // available=false のまま
    }

    // 集計。
    topo.physical_count = static_cast<int>(topo.cores.size());
    int logical = 0;
    std::vector<int> classes;
    for (const PhysicalCore& c : topo.cores)
    {
        logical += c.logical_count;
        if (std::find(classes.begin(), classes.end(), c.efficiency_class) == classes.end())
        {
            classes.push_back(c.efficiency_class);
        }
    }
    topo.logical_count = logical;
    topo.efficiency_class_count = static_cast<int>(classes.size());
    topo.available = true;
    return topo;
}

#elif defined(__linux__)

// Linux 実装: /sys/devices/system/cpu/*/topology を best-effort で読む。
// core_id (物理コア識別) ごとに論理 CPU をまとめる。EfficiencyClass は Linux で
// 直接の等価情報がない (cpu_capacity を見る手はあるが機種依存) ため一律 0 とする。
// 読み取りに失敗した場合は available=false を返す (= 未対応扱い)。
inline CpuTopology detect_cpu_topology()
{
    CpuTopology topo;

    // 物理コアキー = (physical_package_id, core_id)。論理 CPU id を集める。
    struct Key
    {
        int pkg;
        int core;
        bool operator<(const Key& o) const
        {
            if (pkg != o.pkg) return pkg < o.pkg;
            return core < o.core;
        }
    };
    std::vector<std::pair<Key, std::vector<int>>> groups;

    auto read_int = [](const std::string& path, int& out) -> bool
    {
        FILE* f = std::fopen(path.c_str(), "r");
        if (!f)
        {
            return false;
        }
        int v = 0;
        const int n = std::fscanf(f, "%d", &v);
        std::fclose(f);
        if (n != 1)
        {
            return false;
        }
        out = v;
        return true;
    };

    // online CPU を 0..255 まで走査 (uint64 マスク前提なので有効ビットは 0..63)。
    for (int cpu = 0; cpu < 256; ++cpu)
    {
        const std::string base =
            "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
        int core_id = 0;
        int pkg_id = 0;
        if (!read_int(base + "core_id", core_id))
        {
            continue; // この cpu は存在しない or 読めない
        }
        if (!read_int(base + "physical_package_id", pkg_id))
        {
            pkg_id = 0;
        }
        Key key{pkg_id, core_id};
        bool found = false;
        for (auto& g : groups)
        {
            if (!(g.first < key) && !(key < g.first))
            {
                g.second.push_back(cpu);
                found = true;
                break;
            }
        }
        if (!found)
        {
            groups.push_back({key, {cpu}});
        }
    }

    if (groups.empty())
    {
        return topo; // available=false
    }

    for (auto& g : groups)
    {
        PhysicalCore core;
        uint64_t mask = 0;
        int bits = 0;
        for (int cpu : g.second)
        {
            if (cpu >= 0 && cpu < 64)
            {
                mask |= (1ULL << cpu);
                ++bits;
            }
        }
        if (mask == 0)
        {
            continue;
        }
        core.affinity_mask = mask;
        core.logical_count = bits;
        core.hyperthreaded = (bits > 1);
        core.efficiency_class = 0; // Linux では一律 0 (P/E 判別なし)
        topo.cores.push_back(core);
    }

    if (topo.cores.empty())
    {
        return topo;
    }

    topo.physical_count = static_cast<int>(topo.cores.size());
    int logical = 0;
    for (const PhysicalCore& c : topo.cores)
    {
        logical += c.logical_count;
    }
    topo.logical_count = logical;
    topo.efficiency_class_count = 1; // 全コア class 0
    topo.available = true;
    return topo;
}

#else

// 非対応 OS: 未対応を返す。
inline CpuTopology detect_cpu_topology()
{
    return CpuTopology{};
}

#endif

} // namespace dollama

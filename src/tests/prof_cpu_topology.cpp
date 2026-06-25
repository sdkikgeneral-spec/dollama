// CPU LM 推論 Tier 2 前提計測 (計測専用 exe・test 非登録)。
//
// 目的:
//   Tier 1 (AVX2 デュアルパス) の BitNetDenseInfer::forward_fast が「どの物理/論理コアで
//   何本相当で動くか」を実機トポロジ上で確定する。prof_bitnet.cpp は affinity 未設定で
//   per-thread クリーン基準が出ないため、ここでは cpu_topology.hpp で実機トポロジを
//   自動検出し、affinity を制御ピン留めして以下を計測する:
//
//   ① 単一論理 CPU 1 本     … クリーン per-thread 基準 (Tier 1 の真の意味)。
//   ② 物理コア 1->N スケール … 各スレッドを別物理コアの単一論理にピン留め (HT 兄弟を
//                              踏まない disjoint マスク) し、N 本同時 forward_fast の
//                              総スループット。
//   ③ HT 兄弟 2 本同居      … 同一物理コアの 2 論理に 2 スレッドを乗せ HT 係数を出す。
//   ④ P コア 1 本 vs E コア 1 本 … EfficiencyClass で判別。homogeneous 機は N/A 表示。
//
//   マスク/コア集合は全てトポロジ自動検出で決定する (決め打ちマスクをこの probe から排除)。
//
// 並列計時の注意:
//   forward_fast は const メソッドで、モデルのメンバ (embed_/final_norm_/layers_) は
//   read-only。各スレッドが自前の tokens ベクタと結果領域 (返り値 std::vector) を使う限り
//   共有 mutable 状態への書き込みはない => 並列呼び出し可能。本 probe は各スレッドに独立
//   tokens を持たせて検証する。
//
// 重み不在時は健全に skip 終了 (return 0)。HAVE_OPENVINO 不要 (LM 単独計測)。
// 引数: シナリオ選択 (all|clean|scale|ht|pe)。省略時 all。

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "core/affinity.hpp"
#include "core/cpu_topology.hpp"
#include "infer/bitnet.hpp"

#ifndef WEIGHTS_PATH
#define WEIGHTS_PATH "E:/Projects/dollama/data/bitnet/bitnet_dense_fp32.safetensors"
#endif

using namespace dollama;
using Clock = std::chrono::steady_clock;
using std::chrono::duration;

static const int kSeqs[3] = {8, 32, 63};
static const int kWarmup = 3;
static const int kIters  = 15;

static double ms_since(Clock::time_point a, Clock::time_point b)
{
    return duration<double, std::milli>(b - a).count();
}

static double median(std::vector<double> v)
{
    if (v.empty())
    {
        return 0.0;
    }
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

static bool file_exists(const std::string& path)
{
    if (path.empty())
    {
        return false;
    }
    std::ifstream f(path, std::ios::binary);
    return static_cast<bool>(f);
}

// 指定 seq の合法 token 列を生成する (prof_bitnet と同じ規則)。
static std::vector<int> make_tokens(int S)
{
    constexpr int V = BitNetDenseInfer::VOCAB_SIZE;
    std::vector<int> t(static_cast<size_t>(S));
    for (int i = 0; i < S; ++i)
    {
        t[static_cast<size_t>(i)] = (i * 37 + 5) % V;
    }
    return t;
}

// 単一スレッドで mask にピン留めし、seq の forward_fast 中央値 [ms] を計測する。
static double measure_single(const BitNetDenseInfer& model, int S, uint64_t mask)
{
    double result = 0.0;
    std::thread th([&]()
    {
        set_current_thread_affinity(mask);
        std::vector<int> tokens = make_tokens(S);

        for (int w = 0; w < kWarmup; ++w)
        {
            volatile float a = model.forward_fast(tokens, false)[0];
            (void)a;
        }
        std::vector<double> v;
        v.reserve(static_cast<size_t>(kIters));
        for (int it = 0; it < kIters; ++it)
        {
            auto t0 = Clock::now();
            volatile float sink = model.forward_fast(tokens, false)[0];
            auto t1 = Clock::now();
            (void)sink;
            v.push_back(ms_since(t0, t1));
        }
        result = median(v);
    });
    th.join();
    return result;
}

// N 本のスレッドを masks[i] へ各々ピン留めし、全スレッドが kIters 回 forward_fast を
// 並列実行する。返り値: 各スレッドの per-iter 中央値 [ms] の平均、および総スループット
// (forwards/s)。全スレッドは同一の開始バリアで足並みを揃える。
struct ParallelResult
{
    double avg_median_ms = 0.0; // スレッド毎中央値の平均
    double max_median_ms = 0.0; // スレッド毎中央値の最大 (壁時計律速)
    double throughput    = 0.0; // forwards/s (N * iters / 全体壁時計秒)
    int    threads       = 0;
};

static ParallelResult measure_parallel(const BitNetDenseInfer& model, int S,
                                        const std::vector<uint64_t>& masks)
{
    const int N = static_cast<int>(masks.size());
    ParallelResult res;
    res.threads = N;
    if (N == 0)
    {
        return res;
    }

    std::vector<double> medians(static_cast<size_t>(N), 0.0);
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(N));

    Clock::time_point wall_start;
    Clock::time_point wall_end;
    std::atomic<int> done{0};

    for (int i = 0; i < N; ++i)
    {
        workers.emplace_back([&, i]()
        {
            set_current_thread_affinity(masks[static_cast<size_t>(i)]);
            // 各スレッド独立の入力 / 出力領域 (共有 mutable 書き込みなし)。
            std::vector<int> tokens = make_tokens(S);

            for (int w = 0; w < kWarmup; ++w)
            {
                volatile float a = model.forward_fast(tokens, false)[0];
                (void)a;
            }

            ready.fetch_add(1, std::memory_order_acq_rel);
            while (!go.load(std::memory_order_acquire))
            {
                std::this_thread::yield();
            }

            std::vector<double> v;
            v.reserve(static_cast<size_t>(kIters));
            for (int it = 0; it < kIters; ++it)
            {
                auto t0 = Clock::now();
                volatile float sink = model.forward_fast(tokens, false)[0];
                auto t1 = Clock::now();
                (void)sink;
                v.push_back(ms_since(t0, t1));
            }
            medians[static_cast<size_t>(i)] = median(v);
            done.fetch_add(1, std::memory_order_acq_rel);
        });
    }

    // 全スレッドが warmup を終え ready になるまで待つ。
    while (ready.load(std::memory_order_acquire) < N)
    {
        std::this_thread::yield();
    }
    wall_start = Clock::now();
    go.store(true, std::memory_order_release);

    for (std::thread& t : workers)
    {
        t.join();
    }
    wall_end = Clock::now();

    double sum = 0.0;
    double mx = 0.0;
    for (double m : medians)
    {
        sum += m;
        mx = std::max(mx, m);
    }
    res.avg_median_ms = sum / N;
    res.max_median_ms = mx;
    const double wall_s = ms_since(wall_start, wall_end) / 1000.0;
    if (wall_s > 0.0)
    {
        res.throughput = static_cast<double>(N) * kIters / wall_s;
    }
    return res;
}

static void print_topology(const CpuTopology& topo)
{
    std::cout << "==== 検出トポロジ ====\n";
    std::cout << "physical cores : " << topo.physical_count << "\n";
    std::cout << "logical cores  : " << topo.logical_count << "\n";
    std::cout << "efficiency classes : " << topo.efficiency_class_count
              << (topo.efficiency_class_count == 1 ? " (homogeneous)" : " (hybrid)")
              << "\n";
    std::cout << "core# mask(hex) logical HT effclass\n";
    int idx = 0;
    for (const PhysicalCore& c : topo.cores)
    {
        std::cout << "  [" << idx << "] 0x" << std::hex << c.affinity_mask
                  << std::dec << "   " << c.logical_count
                  << "      " << (c.hyperthreaded ? "Y" : "N")
                  << "    " << c.efficiency_class << "\n";
        ++idx;
    }
    std::cout << "\n";
}

int main(int argc, char** argv)
{
    std::string scenario = "all";
    if (argc >= 2)
    {
        scenario = argv[1];
    }
    const bool run_clean = (scenario == "all" || scenario == "clean");
    const bool run_scale = (scenario == "all" || scenario == "scale");
    const bool run_ht    = (scenario == "all" || scenario == "ht");
    const bool run_pe    = (scenario == "all" || scenario == "pe");

    CpuTopology topo = detect_cpu_topology();
    if (!topo.available)
    {
        std::cout << "[prof_cpu_topology] SKIP: topology 未取得 (未対応 OS または列挙失敗)\n";
        return 0;
    }
    print_topology(topo);

    const std::string wpath = WEIGHTS_PATH;
    if (!file_exists(wpath))
    {
        std::cout << "[prof_cpu_topology] SKIP: weights 不在 (" << wpath << ")\n";
        return 0;
    }
    std::cout << "[prof_cpu_topology] weights: " << wpath << "\n\n";
    BitNetDenseInfer model(wpath);

    // efficiency_class 降順に並べた単一論理マスク列 (P コア優先・HT 兄弟を踏まない)。
    const std::vector<uint64_t> single_masks = topo.single_logical_per_core_masks();

    // -------------------------------------------------------------
    // ① 単一論理 CPU 1 本 (クリーン per-thread 基準)
    // -------------------------------------------------------------
    if (run_clean)
    {
        std::cout << "==== ① 単一論理 CPU 1 本 (クリーン per-thread 基準) ====\n";
        const uint64_t mask = single_masks.empty() ? 1ULL : single_masks.front();
        std::cout << "  pinned mask : 0x" << std::hex << mask << std::dec << "\n";
        std::cout << "  seq, forward_fast_ms (median)\n";
        for (int si = 0; si < 3; ++si)
        {
            const double ms = measure_single(model, kSeqs[si], mask);
            std::cout << "  " << kSeqs[si] << ", " << ms << "\n";
        }
        std::cout << "\n";
    }

    // -------------------------------------------------------------
    // ② 物理コア 1->N スケール (HT 兄弟を踏まない disjoint)
    // -------------------------------------------------------------
    if (run_scale)
    {
        std::cout << "==== ② 物理コア 1->N スケール (HT 兄弟を踏まない) ====\n";
        const int maxN = static_cast<int>(single_masks.size());
        // seq32 を代表に 1..N をスイープ。各 N の総スループット (forwards/s) を出す。
        // 1 本基準を baseline にスケール倍率も併記。
        for (int si = 0; si < 3; ++si)
        {
            const int S = kSeqs[si];
            std::cout << "  -- seq=" << S << " --\n";
            std::cout << "  N, throughput(fwd/s), scale_vs_1, avg_median_ms, max_median_ms\n";
            double base_tp = 0.0;
            for (int n = 1; n <= maxN; ++n)
            {
                std::vector<uint64_t> masks(single_masks.begin(),
                                            single_masks.begin() + n);
                ParallelResult r = measure_parallel(model, S, masks);
                if (n == 1)
                {
                    base_tp = r.throughput;
                }
                const double scale = (base_tp > 0.0) ? r.throughput / base_tp : 0.0;
                std::cout << "  " << n << ", " << r.throughput << ", " << scale
                          << ", " << r.avg_median_ms << ", " << r.max_median_ms << "\n";
            }
            std::cout << "\n";
        }
    }

    // -------------------------------------------------------------
    // ③ HT 兄弟 2 本同居 (HT 係数)
    // -------------------------------------------------------------
    if (run_ht)
    {
        std::cout << "==== ③ HT 兄弟 2 本同居 (HT 係数) ====\n";
        // HT ありの物理コアを 1 個探す。
        const PhysicalCore* ht_core = nullptr;
        for (const PhysicalCore& c : topo.cores)
        {
            if (c.hyperthreaded)
            {
                ht_core = &c;
                break;
            }
        }
        if (!ht_core)
        {
            std::cout << "  N/A (HT なし: 全物理コアが単一論理)\n\n";
        }
        else
        {
            // この物理コアの 2 つの論理ビットを取り出す。
            uint64_t m = ht_core->affinity_mask;
            std::vector<uint64_t> sib;
            for (int b = 0; b < 64 && sib.size() < 2; ++b)
            {
                if (m & (1ULL << b))
                {
                    sib.push_back(1ULL << b);
                }
            }
            // 別の物理コア (2 本) も取って「別物理 2 本」基準を出す。
            std::vector<uint64_t> two_phys;
            if (single_masks.size() >= 2)
            {
                two_phys.assign(single_masks.begin(), single_masks.begin() + 2);
            }

            std::cout << "  HT 兄弟マスク : 0x" << std::hex << sib[0]
                      << " + 0x" << sib[1] << std::dec << "\n";
            std::cout << "  seq, ht_sib_tp(fwd/s), two_phys_tp(fwd/s), ht_ratio(%)\n";
            for (int si = 0; si < 3; ++si)
            {
                const int S = kSeqs[si];
                ParallelResult ht = measure_parallel(model, S, sib);
                ParallelResult tp = measure_parallel(model, S, two_phys);
                const double ratio =
                    (tp.throughput > 0.0) ? ht.throughput / tp.throughput * 100.0 : 0.0;
                std::cout << "  " << S << ", " << ht.throughput << ", "
                          << tp.throughput << ", " << ratio << "\n";
            }
            std::cout << "\n";
        }
    }

    // -------------------------------------------------------------
    // ④ P コア 1 本 vs E コア 1 本
    // -------------------------------------------------------------
    if (run_pe)
    {
        std::cout << "==== ④ P コア 1 本 vs E コア 1 本 ====\n";
        if (topo.efficiency_class_count <= 1)
        {
            std::cout << "  N/A (single efficiency class = homogeneous CPU)\n\n";
        }
        else
        {
            const int pc = topo.max_efficiency_class();
            const int ec = topo.min_efficiency_class();
            const uint64_t pmask_full = topo.first_core_mask_of_class(pc);
            const uint64_t emask_full = topo.first_core_mask_of_class(ec);
            // 各コアの最下位ビット 1 本にピン留め。
            const uint64_t pmask = pmask_full & (~pmask_full + 1ULL);
            const uint64_t emask = emask_full & (~emask_full + 1ULL);
            std::cout << "  P core (effclass " << pc << ") mask : 0x" << std::hex
                      << pmask << std::dec << "\n";
            std::cout << "  E core (effclass " << ec << ") mask : 0x" << std::hex
                      << emask << std::dec << "\n";
            std::cout << "  seq, P_ms, E_ms, E/P(slowdown)\n";
            for (int si = 0; si < 3; ++si)
            {
                const int S = kSeqs[si];
                const double pms = measure_single(model, S, pmask);
                const double ems = measure_single(model, S, emask);
                const double ratio = (pms > 0.0) ? ems / pms : 0.0;
                std::cout << "  " << S << ", " << pms << ", " << ems << ", "
                          << ratio << "\n";
            }
            std::cout << "\n";
        }
    }

    std::cout << "[prof_cpu_topology] done\n";
    return 0;
}

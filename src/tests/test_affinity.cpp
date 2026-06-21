// set_current_thread_affinity 単体テスト (STL のみ・常時実行)
// HAVE_OPENVINO は不要。
// 検証:
//   - 有効マスク (0x1) でワーカースレッド自身に設定 → true・クラッシュなし
//   - mask=0 で false (境界)
//   - 範囲外/全ビットマスクでもクラッシュしない
// 自己ピン留め API のため、設定はワーカースレッド内で呼び、結果を親へ返す。
// testing.md 形式: if(!cond){cerr;return false;}、main 集約、ALL PASSED。
#include <atomic>
#include <cstdint>
#include <iostream>
#include <thread>

#include "core/affinity.hpp"

namespace dollama {

// 与えられたマスクをワーカースレッド内で自身に設定し、結果を返すヘルパ。
// 自己ピン留め API のため、呼び出しは必ずワーカースレッドの中で行う。
static bool run_self_affinity(uint64_t mask)
{
    std::atomic<bool> result{false};
    std::jthread t(
        [&result, mask]()
        {
            result.store(set_current_thread_affinity(mask));
        });
    t.join();
    return result.load();
}

// ----------------------------------------------------------------
// テスト1: 有効マスク (0x1 = 論理コア0) で true が返ること
// ----------------------------------------------------------------
static bool test_valid_mask()
{
    bool ok = run_self_affinity(0x1);

    if (!ok)
    {
        std::cerr << "[test_valid_mask] 有効マスク 0x1 で false が返った\n";
        return false;
    }

    std::cout << "[test_valid_mask] PASSED  (mask=0x1 → true)\n";
    return true;
}

// ----------------------------------------------------------------
// テスト2: mask=0 (境界) で false が返ること
// ----------------------------------------------------------------
static bool test_zero_mask()
{
    bool ok = run_self_affinity(0);

    if (ok)
    {
        std::cerr << "[test_zero_mask] mask=0 で true が返った (false 期待)\n";
        return false;
    }

    std::cout << "[test_zero_mask] PASSED  (mask=0 → false)\n";
    return true;
}

// ----------------------------------------------------------------
// テスト3: 実マシンのマスク (P/E core) でクラッシュしないこと
// 環境により成否は変わり得るので bool 値は問わず、no-crash のみ確認。
// すべてワーカースレッド内で自身に設定する。
// ----------------------------------------------------------------
static bool test_real_masks_no_crash()
{
    // P-core / E-core マスク (probe11 実測)
    (void)run_self_affinity(0x00C03C03ULL);
    (void)run_self_affinity(0x003FC3FCULL);
    // 全ビット (存在しない論理コアを含む) でもクラッシュしないこと
    (void)run_self_affinity(~0ULL);

    std::cout << "[test_real_masks_no_crash] PASSED  (no crash)\n";
    return true;
}

} // namespace dollama

int main()
{
    bool ok = true;
    ok = dollama::test_valid_mask()          && ok;
    ok = dollama::test_zero_mask()           && ok;
    ok = dollama::test_real_masks_no_crash() && ok;

    if (!ok)
    {
        std::cerr << "[test_affinity] FAILED\n";
        return 1;
    }
    std::cout << "[test_affinity] ALL PASSED\n";
    return 0;
}

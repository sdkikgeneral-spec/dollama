// set_thread_affinity 単体テスト (STL のみ・常時実行)
// HAVE_OPENVINO は不要。
// 検証:
//   - 有効マスク (0x1) で true・クラッシュなし
//   - mask=0 で false (境界)
//   - 範囲外/全ビットマスクでもクラッシュしない
// testing.md 形式: if(!cond){cerr;return false;}、main 集約、ALL PASSED。
#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <thread>

#include "core/affinity.hpp"

namespace dollama {

// 短命だが起動済みのスレッドを作り、native_handle が有効な状態でテストする。
// スレッドは stop_token が立つまで軽くスピンして生かしておく。
static std::jthread make_live_thread(std::atomic<bool>& started)
{
    return std::jthread(
        [&started](std::stop_token st)
        {
            started.store(true);
            while (!st.stop_requested())
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        });
}

// ----------------------------------------------------------------
// テスト1: 有効マスク (0x1 = 論理コア0) で true が返ること
// ----------------------------------------------------------------
static bool test_valid_mask()
{
    std::atomic<bool> started{false};
    std::jthread t = make_live_thread(started);
    while (!started.load())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    bool ok = set_thread_affinity(t, 0x1);
    t.request_stop();

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
    std::atomic<bool> started{false};
    std::jthread t = make_live_thread(started);
    while (!started.load())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    bool ok = set_thread_affinity(t, 0);
    t.request_stop();

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
// ----------------------------------------------------------------
static bool test_real_masks_no_crash()
{
    std::atomic<bool> started{false};
    std::jthread t = make_live_thread(started);
    while (!started.load())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // P-core / E-core マスク (probe11 実測)
    (void)set_thread_affinity(t, 0x00C03C03ULL);
    (void)set_thread_affinity(t, 0x003FC3FCULL);
    // 全ビット (存在しない論理コアを含む) でもクラッシュしないこと
    (void)set_thread_affinity(t, ~0ULL);

    t.request_stop();

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

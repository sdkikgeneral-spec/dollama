#pragma once

// CPU アフィニティ設定 (OS 依存ラッパー)
// 呼び出したスレッド自身に設定する自己ピン留め API。
// 親スレッドから子の native_handle を渡す設計は MinGW-W64 posix (winpthreads)
// で native_handle が pthread_t となり Win32 HANDLE に渡せないため採用しない。
// 各ワーカーが起動直後に自分自身へ設定する形にすることで OS/ランタイム差を吸収する。
// OS 依存はこのファイルだけに閉じ込める。
// マスクのビットパターンは docs/cpu-topology.md (probe11 実測) を参照。
//
// 戻り値 bool: 失敗を握り潰さず呼び出し側へ返す。
//   - mask == 0          → 無効指定として false
//   - OS API が失敗       → false
//   - 成功               → true
//   - 対応 OS でない場合   → no-op + false

#include <cstdint>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace dollama {

// 呼び出したスレッド自身に CPU アフィニティマスクを設定する。
// mask: 論理プロセッサのビットマスク (bit i = 論理コア i を許可)。
// 戻り値: 成功で true、失敗 (mask=0 / API 失敗 / 非対応 OS) で false。
//
// Windows: GetCurrentThread() の擬似ハンドル (実 HANDLE 不要・MSVC/MinGW 両対応)
//          を SetThreadAffinityMask に渡す。native_handle 非依存。
// Linux:   pthread_self() を pthread_setaffinity_np に渡す。
inline bool set_current_thread_affinity(uint64_t mask)
{
    // mask=0 はどの論理コアも許可しない無効指定 → false
    if (mask == 0)
    {
        return false;
    }

#ifdef _WIN32
    // GetCurrentThread() は呼び出しスレッドを指す擬似ハンドル。
    // SetThreadAffinityMask は成功で旧マスク (非0)、失敗で 0 を返す。
    DWORD_PTR prev = SetThreadAffinityMask(
        GetCurrentThread(), static_cast<DWORD_PTR>(mask));
    return prev != 0;
#elif defined(__linux__)
    cpu_set_t cs;
    CPU_ZERO(&cs);
    for (int i = 0; i < 64; ++i)
    {
        if (mask & (1ULL << i))
        {
            CPU_SET(i, &cs);
        }
    }
    // pthread_setaffinity_np は成功で 0、失敗でエラー番号 (非0) を返す
    int rc = pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
    return rc == 0;
#else
    // 非対応 OS: no-op。失敗を握り潰さないため false を返す。
    (void)mask;
    return false;
#endif
}

} // namespace dollama

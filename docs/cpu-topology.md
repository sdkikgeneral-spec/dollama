# CPU トポロジ調査 — Intel Core Ultra 9 285

probe11 (`dollma_probe11_cpu_topo.py`) による実測結果。

## コア構成

| 種別 | 物理コア数 | 論理コア数 | EfficiencyClass | アフィニティマスク |
|---|---|---|---|---|
| P-core | 8 | 8 | 1 (高) | `0x0000000000C03C03` |
| E-core | 16 | 16 | 0 (低) | `0x00000000003FC3FC` |

- Arrow Lake は P-core の HyperThreading を廃止 → P 8コア = 論理 8
- E-core も同様 → 16コア = 論理 16
- 合計: 24 論理プロセッサ

## アフィニティマスクのビットパターン

```
ビット位置: 23 22 21 20 19 18 17 16 15 14 13 12 11 10  9  8  7  6  5  4  3  2  1  0
P-core:      P  P  .  .  .  .  .  .  .  .  P  P  P  P  .  .  .  .  .  .  .  .  P  P
E-core:      .  .  E  E  E  E  E  E  E  E  .  .  .  .  E  E  E  E  E  E  E  E  .  .
```

P / E が交互にグループ配置されている。

## dollama スレッド割り当て

| スレッド | 割り当て | マスク | 理由 |
|---|---|---|---|
| LLM (Qwen2 / 将来 自作タグ生成 LM) | P-core | `0xC03C03` | 重いシングルスレッド処理 |
| SDXL 制御 (GPU への指示・同期) | P-core | `0xC03C03` | レイテンシ優先 |
| WD14 tagger | E-core | `0x3FC3FC` | GPU 生成中に並列、E で十分 |
| パイプライン制御キュー | E-core | `0x3FC3FC` | 軽量ルーティング |

→ LLM (P-core) と WD14 (E-core) が物理的に異なるコアで動作するため CPU ダブルブッキングなし。

## 動作確認結果

```
[P-core] 11.7M iter/s  (mask=0xC03C03)
[E-core] 10.3M iter/s  (mask=0x3FC3FC)
→ 両スレッドが干渉なく並列実行できることを確認 ✅
```

## C++ での設定方法

**自己ピン留め方式 (確定)**: 各ワーカースレッドが起動直後に**自分自身**へ設定する
(親から子の native_handle を触らない)。理由は下記「スレッド実装方針」参照。

```cpp
// 各ワーカーラムダの先頭で:
const uint64_t P_CORE_MASK = 0x00C03C03ULL;
const uint64_t E_CORE_MASK = 0x003FC3FCULL;
set_current_thread_affinity(P_CORE_MASK);  // llm / sdxl
set_current_thread_affinity(E_CORE_MASK);  // clip / wd14
```

> **注意**: マスク値はこのマシン固有。他環境では `GetLogicalProcessorInformationEx` で動的取得が必要。
> probe11 の `analyze_cores()` がその実装例。

## スレッド実装方針 (task 7)

- **スレッド/同期は STL**: `std::jthread` (C++20, 自動 join + `stop_token` でパイプライン停止)、
  `condition_variable`、自作 SPSC キュー。これらは標準でクロスプラットフォーム。
- **CPU アフィニティだけが OS 依存** → 薄い `#ifdef` ラッパーに集約する。
  **自己ピン留め型** (`std::jthread&` を取らず、呼んだスレッド自身に設定する):

```cpp
// src/core/affinity.hpp — OS 依存はここだけ
inline bool set_current_thread_affinity(uint64_t mask)
{
    if (mask == 0) return false;
#ifdef _WIN32
    // GetCurrentThread() は擬似ハンドル。MSVC/MinGW 両対応・native_handle 非依存。
    return SetThreadAffinityMask(GetCurrentThread(),
                                 static_cast<DWORD_PTR>(mask)) != 0;
#elif defined(__linux__)
    cpu_set_t cs; CPU_ZERO(&cs);
    for (int i = 0; i < 64; ++i)
        if (mask & (1ULL << i)) CPU_SET(i, &cs);
    return pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs) == 0;
#else
    return false;
#endif
}
```

> **なぜ自己ピン留めか (MinGW 対応)**: 当初は親が子 jthread の `native_handle()` を
> `SetThreadAffinityMask` に渡す設計だったが、`std::thread::native_handle_type` は
> **MSVC では Win32 `HANDLE`・MinGW-W64 posix では `pthread_t`** で、MinGW では
> `HANDLE` に渡せずコンパイルできない (winpthreads は `pthread_getw32threadhandle_np` /
> `pthread_setaffinity_np` を未提供)。`GetCurrentThread()` 擬似ハンドルで自スレッドに
> 設定する形なら native_handle に触れず両 toolchain で実際にピン留めできる。
> 呼び出しは各ワーカーラムダの先頭で行う (`src/pipeline.hpp`)。

- **Boost は不採用**: スレッド本体は STL で足り、肝心のアフィニティを Boost が綺麗に
  抽象化するわけでもない (Boost.Thread の affinity 対応は限定的)。結局 OS コールの
  `#ifdef` ラップが要るため、重量級依存を増やす価値がない。「単一バイナリ・重量級
  フレームワーク不使用」の方針 (CLAUDE.md「実装方針」) とも一貫。

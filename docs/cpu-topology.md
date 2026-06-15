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
| LLM (Qwen2 / 将来 BitNet) | P-core | `0xC03C03` | 重いシングルスレッド処理 |
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

```cpp
// Windows
const DWORD_PTR P_CORE_MASK = 0x00C03C03ULL;
const DWORD_PTR E_CORE_MASK = 0x003FC3FCULL;

SetThreadAffinityMask(llm_thread.native_handle(),  P_CORE_MASK);
SetThreadAffinityMask(wd14_thread.native_handle(), E_CORE_MASK);
```

```cpp
// Linux (クロスプラットフォーム対応)
auto pin_thread = [](std::thread& t, uint64_t mask) {
    cpu_set_t cs;
    CPU_ZERO(&cs);
    for (int i = 0; i < 64; ++i)
        if (mask & (1ULL << i)) CPU_SET(i, &cs);
    pthread_setaffinity_np(t.native_handle(), sizeof(cs), &cs);
};
pin_thread(llm_thread,  P_CORE_MASK);
pin_thread(wd14_thread, E_CORE_MASK);
```

> **注意**: マスク値はこのマシン固有。他環境では `GetLogicalProcessorInformationEx` で動的取得が必要。
> probe11 の `analyze_cores()` がその実装例。

## スレッド実装方針 (task 7)

- **スレッド/同期は STL**: `std::jthread` (C++20, 自動 join + `stop_token` でパイプライン停止)、
  `condition_variable`、自作 SPSC キュー。これらは標準でクロスプラットフォーム。
- **CPU アフィニティだけが OS 依存** → 薄い `#ifdef` ラッパーに集約する:

```cpp
// src/core/affinity.hpp — OS 依存はここだけ
inline void set_thread_affinity(std::jthread& t, uint64_t mask)
{
#ifdef _WIN32
    SetThreadAffinityMask(t.native_handle(), static_cast<DWORD_PTR>(mask));
#elif defined(__linux__)
    cpu_set_t cs; CPU_ZERO(&cs);
    for (int i = 0; i < 64; ++i)
        if (mask & (1ULL << i)) CPU_SET(i, &cs);
    pthread_setaffinity_np(t.native_handle(), sizeof(cs), &cs);
#endif
}
```

- **Boost は不採用**: スレッド本体は STL で足り、肝心のアフィニティを Boost が綺麗に
  抽象化するわけでもない (Boost.Thread の affinity 対応は限定的)。結局 OS コールの
  `#ifdef` ラップが要るため、重量級依存を増やす価値がない。「単一バイナリ・重量級
  フレームワーク不使用」の方針 (CLAUDE.md「実装方針」) とも一貫。

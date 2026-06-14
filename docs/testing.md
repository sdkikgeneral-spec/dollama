# テスト方針・実行ガイド — dollama

## テストの基本方針

- **単体テスト**: コアコンポーネント (`src/core/`, `src/kernels/`) を個別に検証
- **マルチスレッドテスト**: キュー・パイプラインは必ずスレッドを立てて検証
- **統合テスト**: パイプライン全体はロードマップ Phase 1 完了後に別途追加
- テストは `src/tests/` に置き、`meson test` で自動実行する

## テスト実行コマンド

```bash
# ビルド (初回 or ビルド定義変更後)
meson setup build --wipe
meson compile -C build

# テスト全件実行
meson test -C build

# 特定テストのみ実行 (verbose)
meson test -C build --verbose spsc_queue

# テスト結果ログ確認
cat build/meson-logs/testlog.txt
```

---

## テスト一覧

### `src/tests/test_queue.cpp` — SPSCQueue

`meson test` 名: `spsc_queue`

| テスト関数 | 種別 | 内容 |
|---|---|---|
| `test_single_fifo` | シングルスレッド | push/pop の FIFO 動作、満杯時 false、空時 false |
| `test_push_wait_timeout` | シングルスレッド | push_wait が満杯時に ~5ms でタイムアウトすること |
| `test_multithread_producer_consumer` | マルチスレッド | 2スレッドで 10,000 件を送受信し、欠損・順序崩れがないこと |

---

## テストファイルの追加方法

### 1. テストファイルを作成

`src/tests/test_<component>.cpp` を新規作成する。

```cpp
// コンポーネント名 単体テスト
#include <iostream>
#include "core/tensor.hpp"  // テスト対象をインクルード

namespace dollama {

static bool test_something()
{
    // テスト内容
    // 失敗時: std::cerr に出力して return false
    std::cout << "[test_something] PASSED\n";
    return true;
}

} // namespace dollama

int main()
{
    bool ok = true;
    ok = dollama::test_something() && ok;

    if (!ok)
    {
        std::cerr << "[test_<component>] FAILED\n";
        return 1;
    }
    std::cout << "[test_<component>] ALL PASSED\n";
    return 0;
}
```

### 2. `src/meson.build` に登録

既存の `test('spsc_queue', ...)` の後に追加する:

```meson
test_<component>_exe = executable('test_<component>',
  sources             : files('tests/test_<component>.cpp'),
  include_directories : src_inc,
  dependencies        : deps,
  cpp_args            : cpp_args,
)
test('<component>', test_<component>_exe)
```

---

## 各コンポーネントのテスト計画

### Phase 1 (現在)

| コンポーネント | テストファイル | 状態 | 主な検証内容 |
|---|---|---|---|
| `SPSCQueue` | `test_queue.cpp` | ✅ 完了 | FIFO、満杯、タイムアウト、マルチスレッド |
| `Tensor` | `test_tensor.cpp` | ⏳ 未着手 | 形状・要素数・デバイスガード・nbytes |
| `UniqueBuffer` (allocator) | `test_allocator.cpp` | ⏳ 未着手 | RAII 解放・ムーブ後の状態 |
| CLIP NPU 推論 | `test_clip.cpp` | ⏳ 未着手 | 出力テンソルの shape・L2 norm が probe9 と一致 |
| WD14 CPU 推論 | `test_wd14.cpp` | ⏳ 未着手 | 上位タグが probe8 と一致 |

### Phase 2 以降

| コンポーネント | テストファイル | 主な検証内容 |
|---|---|---|
| ternary GEMM | `test_ternary_gemm.cpp` | 小行列で float GEMM と結果比較 |
| Attention カーネル | `test_attention.cpp` | known input → expected output |
| VAE decode | `test_vae_decode.cpp` | latent → image が probe10 出力と SSIM ≥ 0.99 |

---

## テストの規約

- コーディング規約は本体と同じ (Allman スタイル、コメント日本語)
- `assert` より `if (!cond) { cerr; return false; }` を推奨 (失敗箇所が分かりやすい)
- マルチスレッドテストのタイムアウトは余裕を持たせる (最低 5 秒)
- ハードウェア依存テスト (NPU, CUDA) は `#ifdef HAVE_OPENVINO` / `#ifdef HAVE_CUDA` でガードする

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
| `bench_single_thread_latency` | ベンチ | 1M 回 push+pop のラウンドトリップ ns/op (実績: 1.914 ns) |

---

### `src/tests/test_tensor.cpp` — Tensor

`meson test` 名: `tensor`

| テスト関数 | 種別 | 内容 |
|---|---|---|
| `test_default_ctor` | 機能 | デフォルト構築: numel==0, ndim==0 |
| `test_shape_cpu` | 機能 | {2,3,4} の ndim/dim/numel/nbytes/device が正しい |
| `test_data_zero_init` | 機能 | CPU Tensor が 0 初期化され、書き込み後に読み戻せる |
| `test_pinned_device` | 機能 | PINNED デバイスで data() が例外を投げない |
| `test_cuda_data_throws` | エラー | Device::CUDA で data() が std::logic_error |
| `test_npu_data_throws` | エラー | Device::NPU で data() が std::logic_error |
| `test_set_data_ptr` | 機能 | set_data_ptr() 後に data_ptr() が同じポインタを返す |
| `test_data_ptr_cuda_no_ptr_throws` | エラー | CUDA Tensor で set_data_ptr() 未呼び出し → data_ptr() が logic_error |
| `test_data_ptr_cpu_returns_buf` | 機能 | CPU Tensor で data_ptr() == data() |
| `bench_tensor_create` | ベンチ | Tensor({1024,1024},CPU) × 10K 生成コスト (実績: 937 µs/create) |
| `bench_data_access` | ベンチ | 1M 要素 float RW × 100 回スループット (実績: 58.8 GB/s) |

---

### `src/tests/test_allocator.cpp` — Allocator / UniqueBuffer

`meson test` 名: `allocator`

| テスト関数 | 種別 | 内容 |
|---|---|---|
| `test_pinned_alloc_free` | 機能 | PinnedAllocator::alloc/free が非 null を返しクラッシュなし |
| `test_unique_buffer_basic` | 機能 | get() 非 null・size() が要求バイト数と一致 |
| `test_unique_buffer_move_ctor` | 機能 | ムーブ後: 新側が ptr、旧側が nullptr/size==0 |
| `test_unique_buffer_move_assign` | 機能 | ムーブ代入: 旧バッファ解放、新側が ptr |
| `test_unique_buffer_self_assign` | エラー | 自己代入でクラッシュなし |
| `test_cuda_alloc_no_cuda_throws` | エラー | HAVE_CUDA 未定義時に CudaAllocator::alloc が runtime_error |
| `test_pinned_buffer_alias` | 機能 | PinnedBuffer エイリアスが正常動作 |
| `bench_pinned_alloc_free` | ベンチ | 1MB alloc/free × 1K 回スループット (実績: 207 GB/s ※operator new) |
| `bench_unique_buffer_create` | ベンチ | UniqueBuffer 4KB/1MB 生成コスト (実績: 0.032 / 5.12 µs/op) |

---

### `src/tests/test_character.cpp` — CharacterBible / プロンプト合成

`meson test` 名: `character`

| テスト関数 | 種別 | 内容 |
|---|---|---|
| `test_bible_put_find` | 機能 | put 後 find が同じ name を返す / 未登録 name は nullptr |
| `test_bible_overwrite` | 機能 | 同名 put で上書きされ size が増えない |
| `test_compose_order` | 機能 | positive[0] が canonical_tags 先頭、pose→expression→composition→isolation の順 |
| `test_compose_negative` | 機能 | forbidden_tags が negative に入る |
| `test_compose_seed` | 機能 | identity.seed!=0 はそれを採用、0 なら scene.scene_seed |
| `test_isolation_tag` | 機能 | isolation_tag が positive 末尾に入る (空文字なら入らない) |
| `test_compose_empty_skip` | 機能 | composition / isolation_tag が空文字のとき positive に空要素が入らない |
| `test_quality_negatives` | 機能 | quality_negatives=true で default_quality_negatives() が negative に合流 / false で入らない |
| `test_digits_default` | 機能 | digits_per_hand のデフォルトが 5、DIGITS_UNCOUNTABLE==0 |
| `bench_compose_prompt` | ベンチ | 代表 Identity/Scene/Output で compose_prompt × 1M 回 (実績: 242 ns/op) |
| `bench_bible_find` | ベンチ | 10,000 体登録し find × 1M 回ルックアップ (実績: 10.5 ns/op) |


### `src/tests/test_wd14.cpp` — Wd14Tagger

`meson test` 名: `wd14`

入力前処理は probe8 (`dollma_probe8_wd14.py`) 準拠: uint8 (0-255) → float32 キャストのみ・正規化なし。合成入力も 0-255 レンジの float を用いる。HAVE_OPENVINO 未定義・モデル不在は [SKIP]。

| テスト関数 | 種別 | 内容 |
|---|---|---|
| `test_output_shape` | 機能 | 出力 size がモデル出力実 size (= N_TAGS = 10861) と一致 |
| `test_scores_range` | 機能 | 全スコアが [0,1] (sigmoid 済み出力) |
| `test_determinism` | 機能 | 同一入力 2 回で出力が完全一致 |
| `test_input_size_guard` | エラー | 不正サイズ入力で `std::invalid_argument` が飛ぶ |
| `bench_infer_latency` | ベンチ | warmup5 + 100回中央値/min/max (実測: 中央値 105.3ms / min 99.1 / max 132.8 / N=100, CPU) |
| `info_top_tags` | 情報 | selected_tags.csv があれば top-10 タグ名を出力 (assert 外) |

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
| `Tensor` | `test_tensor.cpp` | ✅ 完了 | 形状・要素数・デバイスガード・nbytes・ベンチ |
| `UniqueBuffer` (allocator) | `test_allocator.cpp` | ✅ 完了 | RAII 解放・ムーブ後の状態・ベンチ |
| `CharacterBible` | `test_character.cpp` | ✅ 完了 | put/find・上書き・プロンプト合成・品質ネガティブ・ベンチ |
| CLIP NPU 推論 | `test_clip.cpp` | ⏳ 未着手 | 出力テンソルの shape・L2 norm が probe9 と一致 |
| WD14 CPU 推論 | `test_wd14.cpp` | ✅ 完了 | 出力 shape・スコア値域 [0,1]・決定性・サイズガード・ベンチ |

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

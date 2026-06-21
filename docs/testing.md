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


### `src/tests/test_safetensors.cpp` — SafeTensors ローダー

`meson test` 名: `safetensors`

CUDA 不要・常時実行。fixture `src/tests/data/golden.safetensors` は
`scripts/dollma_make_safetensors_fixture.py` (safetensors 0.8.0 + torch.bfloat16) が生成した
既知値ファイル。パスは meson の `cpp_args` で `-DGOLDEN_PATH=` を埋め込む (cwd 非依存)。
golden 内容: t_f32 F32[2,3]={0..5} / t_f16 F16[4] / t_i64 I64[2,2]={1,2,3,4} /
t_i8 I8[3]={-128,0,127} / t_bf16 BF16[2]={1.0,-2.5}。

| テスト関数 | 種別 | 内容 |
|---|---|---|
| `test_names` | 機能 | テンソル名 5 件・`__metadata__` がテンソルに混入しない |
| `test_dtype_shape` | 機能 | 各テンソルの dtype/shape が期待と一致 |
| `test_f32_values` | 機能 | F32[2,3] を float 値で突合 |
| `test_i64_values` | 機能 | I64[2,2] を int64 値で突合 |
| `test_i8_values` | 機能 | I8[3] を符号付き境界値 (-128/0/127) で突合 |
| `test_half_bits` | 機能 | F16/BF16 を 16bit ビットパターンで突合 (生バイト透過性) |
| `test_missing_file_throws` | エラー | 不在ファイルで例外 |
| `test_corrupt_header_throws` | エラー | ヘッダ長を過大に細工した一時ファイルで例外 |
| `test_unknown_tensor_throws` | エラー | 未登録テンソル名アクセスで例外 |
| `bench_load` | ベンチ | golden を 10,000 回ロードして µs/op (実測: 19.0 µs/op) |


### `src/tests/test_bitnet.cpp` — 自作タグ生成 LM モデル定義 (旧 BitNet b1.58, Phase 4-2)

`meson test` 名: `bitnet`

純 C++・CUDA/OpenVINO 不要・常時実行。fixture / `-D` 埋め込み不要 (固定 seed の決定的テスト)。
対象は `src/models/bitnet.hpp` (アーキ定義 + 量子化関数 + ホスト参照 forward)。
参照 forward が naive で重く、ベンチ込みで ~2 分かかるため `timeout : 600`。

| テスト関数 | 種別 | 内容 |
|---|---|---|
| `test_param_count` | 機能 | 全重み要素数合計が 30M ≤ params ≤ 100M (実数を出力。確定 32.98M) |
| `test_ternary_quant` | 機能 | absmean α と {-1,0,+1} が手計算一致 (全正/全負/0近傍/全0 ゼロ除算回避) |
| `test_int8_quant` | 機能 | absmax/127 スケール・round 一致 (既知値/飽和境界±127/全0 ゼロ除算回避) |
| `test_rms_norm` | 機能 | RMSNorm 既知入力一致・weight スケール・全0 入力でゼロ除算しない |
| `test_vocab_and_tied` | 機能 | VOCAB_SIZE==4999・embed tied (embed と lm_head が同一ポインタ) |
| `test_forward_deterministic` | 機能 | 出力形状 [seq_len,4999]・2回実行で完全一致・NaN/Inf なし |
| `test_logit_sanity` | 機能 | logit 非発散 (|max|<1e4)・安定 softmax 正規化和が 1 |
| `bench_forward_latency` | ベンチ | seq=32, warmup1+N=5 中央値 ms/forward (実測: 中央値 ~18,982ms。naive 参照・速度目標なし) |


### `src/tests/test_tokenizer.cpp` — Tokenizer (Phase 4-3)

`meson test` 名: `tokenizer`

純 C++・CUDA/OpenVINO 不要・常時実行。対象は `src/io/tokenizer.hpp` (vocab.json 駆動の
タグ単位完全一致トークナイザ。旧称「BPE」だがサブワードではない)。fixture パス
(vocab.json / pairs.*.jsonl) は meson の `-DVOCAB_PATH=` / `-DPAIRS_TRAIN_PATH=` /
`-DPAIRS_VAL_PATH=` で埋め込む (cwd 非依存)。pairs 不在は real_pairs のみ [SKIP]。

| テスト関数 | 種別 | 内容 |
|---|---|---|
| `test_load` | 機能 | vocab ロード: 総語彙 4999・id 連番・specials id 0..4 固定 |
| `test_invalid_vocab` | エラー | specials 不整合・id 非連番の vocab で例外 |
| `test_encode_decode_roundtrip` | 機能 | target タグ列 encode→decode で UNK 0・完全一致復元 |
| `test_unknown_tag` | 機能 | 未知タグ → `<unk>`(4) |
| `test_normalize` | 機能 | §6 正規化: `long_hair`→`long hair` で id 7・顔文字 `^_^`/`>_<` の `_` 保持 |
| `test_encode_text` | 機能 | greedy 最長一致: 埋め込みタグ回収・接続語スキップ・"long hair" 非分割 |
| `test_framing` | 機能 | `<bos>`/`<sep>`/`<eos>`/`<pad>` の配置 |
| `test_boundaries` | 境界 | 空入力・空タグ列・max_len 打ち切り |
| `test_real_pairs` | 機能 | 実 pairs.train+val (5,000行/77,195タグ) 全 target を encode し UNK 0 を実走確認 |
| `bench` | ベンチ | encode/decode/encode_text の ns/op (実測: 365 / 168 / 2655 ns/op) |


### `src/tests/test_clip.cpp` — ClipEncoder (NPU)

`meson test` 名: `clip`

HAVE_OPENVINO 未定義・モデル不在は [SKIP]。BOS(49406)+anime(2368)+EOS(49407) の固定トークン列を入力に使う。

| テスト関数 | 種別 | 内容 |
|---|---|---|
| `test_output_shape` | 機能 | 出力サイズが SEQ_LEN*HIDDEN = 77*768 = 59136 と一致 |
| `test_output_l2_norm` | 機能 | 出力 L2 norm が 0 < norm < 10000 |
| `test_zero_input` | 機能 | 全 0 トークン入力でクラッシュなく所定 shape を返す |
| `bench_infer_latency` | ベンチ | warmup + N=100 中央値 (実測: 中央値 7.82ms / min 7.61 / max 12.15, NPU) |


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


### `src/tests/test_affinity.cpp` — set_current_thread_affinity

`meson test` 名: `affinity`

STL のみ・常時実行 (HAVE_OPENVINO 不要)。**自己ピン留め型** API (`set_current_thread_affinity(mask)`、呼んだスレッド自身に設定。MinGW 対応で native_handle 依存を撤廃)。ワーカースレッド内で自身に設定し、結果を `std::atomic<bool>` で親へ返して検証する。

| テスト関数 | 種別 | 内容 |
|---|---|---|
| `test_valid_mask` | 機能 | 有効マスク 0x1 で true が返る・クラッシュなし |
| `test_zero_mask` | 境界 | mask=0 で false が返る (無効指定) |
| `test_real_masks_no_crash` | 機能 | P/E-core マスク・全ビットマスクでクラッシュしない |


### `src/tests/test_pipeline.cpp` — Pipeline (Phase 1 マルチスレッド骨格)

`meson test` 名: `pipeline`

stub(LLM) → CLIP(NPU) → stub(SDXL) → WD14(CPU) → feedback の縦通し統合テスト。HAVE_OPENVINO 未定義・CLIP/WD14 いずれかのモデル不在は [SKIP]。run() を `std::async` で起動し、ウォッチドッグ (10+frames 秒) でデッドロックを検出する。

| テスト関数 | 種別 | 内容 |
|---|---|---|
| `test_run_frames` | 機能 | 5 フレーム実行で各フレーム非空タグ・デッドロックなし・クリーン join |
| `bench_pipeline` | ベンチ | 定常スループット frames/s と 1 フレームレイテンシ中央値 (実測: 9.13 frames/s / per_frame 109ms / 単発レイテンシ中央値 157ms。WD14 CPU ~105ms 律速) |

---

## CUDA カーネルテスト (Phase 2、`.cu`)

`with_cuda=true` (nvcc 検出) のときのみビルド・登録される。HAVE_CUDA 未定義時は各テストが `[SKIP]` で return 0。入力は FP32 乱数 → FP16 丸め → そのデコード値を CPU 参照入力にも使い (ビット一致)、カーネル誤差のみを計測する。ベンチは cudaEvent 中央値 (warmup 5 / iters 50〜100)。tol は FP16 相応 (固定 or K スケーリング)。

### `src/tests/test_cuda_smoke.cu` — CUDA 疎通 (2-0/2-1)

`meson test` 名: `cuda_smoke`

| テスト関数 | 種別 | 内容 |
|---|---|---|
| `test_ceil_div` | 機能 | ceil_div の境界 (0/1/256/257) |
| `test_vector_add` | 機能 | H2D→vector_add→D2H で a[i]+b[i] 一致 |
| `bench_vector_add` | ベンチ | N=16.7M H2D×2+D2H (実測: 中央値 14.9ms / 13.5 GB/s, pageable) |

### `src/tests/test_gemm.cu` — dense FP16 GEMM (2-2-1)

`meson test` 名: `gemm`

| テスト関数 | 種別 | 内容 |
|---|---|---|
| `test_identity` | 機能 | B=単位行列で C==A |
| `test_small_square` | 機能 | M=N=K=8、CPU FP32 参照と tol 比較 |
| `test_rectangular` | 機能 | M≠N≠K で添字検証 |
| `test_transB` | 機能 | transB=true (SDXL Linear x@W^T) |
| `test_alpha_beta` | 機能 | alpha≠1, beta≠0, C 初期値ありで alpha*AB+beta*C |
| `bench_gemm` | ベンチ | 1024³ / SDXL Linear (実測: 4730 / 4208 GFLOPS) |

### `src/tests/test_activation.cu` — SiLU / GeLU (2-2-2)

`meson test` 名: `activation`

| テスト関数 | 種別 | 内容 |
|---|---|---|
| `test_silu_known` | 機能 | x=0→0・大正→≈x・大負→≈0・±20 極値 |
| `test_silu_random` | 機能 | 端数 n を CPU FP32 SiLU 参照と比較 |
| `test_gelu_known` | 機能 | x=0→0・x=1→≈0.8413 (erf 既知値) |
| `test_gelu_random` | 機能 | CPU erf 版参照と比較 |
| `test_gelu_tanh_random` | 機能 | tanh 近似版を別の CPU 参照式と比較 |
| `test_inplace` | 機能 | d_in==d_out が out-of-place と一致 |
| `bench` | ベンチ | FFN/UNet FM の GB/s (実測: FFN SiLU 526 / GeLU 544 GB/s) |

### `src/tests/test_groupnorm.cu` — GroupNorm (2-2-3)

`meson test` 名: `groupnorm`

affine gamma/beta はチャネルごと [C] (PyTorch 仕様)。1 グループ=1 ブロック、1 パスで sum/sum-sq を FP32 蓄積。tol は K=cpg*H*W スケーリング。

| テスト関数 | 種別 | 内容 |
|---|---|---|
| `test_groupnorm_known` | 機能 | 小 shape の手計算 mean/var/出力一致 (平均≈0・分散≈1) |
| `test_groupnorm_random` | 機能 | SDXL 代表 shape を CPU FP32 参照と比較 |
| `test_groupnorm_affine` | 機能 | gamma/beta 非自明値で affine 検証 |
| `test_groupnorm_inplace` | 機能 | d_in==d_out が out-of-place と一致 |
| edge (G==C / G==1) | 機能 | InstanceNorm 相当 / LayerNorm 相当 |
| `bench` | ベンチ | UNet/VAE FM の GB/s (実測: UNet 73-75 / VAE FM 48 GB/s) |

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

### CUDA (`.cu`) テストの登録

`.cu` テストは `if cuda_enabled` ブロック内に置き、カーネル本体 `.cu` を sources に同梱し、
`cuda_args` に既存の `cuda_test_args` (`-Xcompiler /utf-8` + `-Xcompiler /std:c++14`) を渡す。
この cuda_args は CUDA 13.3 + MSVC のクラッシュ (0xC0000409 / C1070) 回避に**必須**。

```meson
if cuda_enabled
  test_<comp>_exe = executable('test_<comp>',
    sources             : files('tests/test_<comp>.cu', 'kernels/<comp>.cu'),
    include_directories : src_inc,
    dependencies        : deps,
    cpp_args            : cpp_args,
    cuda_args           : cuda_test_args,
  )
  test('<comp>', test_<comp>_exe)
endif
```

ビルド/テストは `meson setup build --wipe -Dwith_cuda=true` 後、nvcc を PATH に通して実行する。

---

## 各コンポーネントのテスト計画

### Phase 1 (現在)

| コンポーネント | テストファイル | 状態 | 主な検証内容 |
|---|---|---|---|
| `SPSCQueue` | `test_queue.cpp` | ✅ 完了 | FIFO、満杯、タイムアウト、マルチスレッド |
| `Tensor` | `test_tensor.cpp` | ✅ 完了 | 形状・要素数・デバイスガード・nbytes・ベンチ |
| `UniqueBuffer` (allocator) | `test_allocator.cpp` | ✅ 完了 | RAII 解放・ムーブ後の状態・ベンチ |
| `CharacterBible` | `test_character.cpp` | ✅ 完了 | put/find・上書き・プロンプト合成・品質ネガティブ・カラーモード注入・ベンチ |
| CLIP NPU 推論 | `test_clip.cpp` | ✅ 完了 | 出力 shape・L2 norm・全0入力・ベンチ (NPU 7.82ms) |
| WD14 CPU 推論 | `test_wd14.cpp` | ✅ 完了 | 出力 shape・スコア値域 [0,1]・決定性・サイズガード・ベンチ |
| アフィニティ | `test_affinity.cpp` | ✅ 完了 | 自己ピン留め (`set_current_thread_affinity`): 有効マスク true・mask=0 false・no-crash |
| Pipeline 骨格 | `test_pipeline.cpp` | ✅ 完了 | 縦通し・非空タグ・デッドロックなし・クリーン join・スループット/レイテンシ |

### Phase 2 (CUDA カーネル)

| 段 | コンポーネント | テストファイル | 状態 | 主な検証内容 |
|---|---|---|---|---|
| 2-0/2-1 | CUDA 疎通・基盤 | `test_cuda_smoke.cu` | ✅ 完了 | ceil_div・vector_add 一致・転送ベンチ |
| 2-2-1 | dense FP16 GEMM | `test_gemm.cu` | ✅ 完了 | identity/square/rect/transB/alpha_beta・GFLOPS |
| 2-2-2 | SiLU / GeLU | `test_activation.cu` | ✅ 完了 | known/random/tanh別参照/in-place・GB/s |
| 2-2-3 | GroupNorm | `test_groupnorm.cu` | ✅ 完了 | known/random/affine/inplace/エッジ・GB/s |
| 2-2-4 | Conv2d | `test_conv2d.cu` | ✅ 完了 | direct 畳み込みを CPU 参照と tol 比較・GFLOPS |
| 2-2-5 | Attention (self/cross) | `test_attention.cu` | ✅ 完了 | known input → expected output・GFLOPS |
| 2-2 補助 | LayerNorm/GEGLU/time embed/bias add/elementwise | `test_layernorm/geglu/timeembed/bias_add/elementwise.cu` | ✅ 完了 | CPU 参照と tol 比較・GB/s |
| 2-3 | safetensors ローダー | `test_safetensors.cpp` | ✅ 完了 | golden の dtype/shape/値・破損/不在で例外・load µs/op |
| 2-4 | VAE decode | `test_vae_decode.cu` | ✅ 完了 | latent → image を golden と SSIM(11×11一様窓)突合。実測 SSIM 0.999992 / MAE 5.45e-4 / Inf-NaN=0 / decode 中央値 7.96s。中間段は VAE_DEBUG=1 で全段ダンプ確認可。timeout 300・is_parallel:false |
| 2-5 | SDXL UNet + Euler scheduler | `test_unet.cu` / `test_scheduler.cpp` | ✅ 完了 | UNet 24段ゴールデン突合 (noise_pred SSIM 0.999998)・scheduler は diffusers golden と sigmas/timesteps/step 突合。timeout 600・is_parallel:false |
| 2-6a | フル拡散パイプライン + 生成器 | `test_diffusion.cu` / `test_pipeline_generator.cu` / `test_pipeline_factory.cpp` | ✅ 完了 | 2step smoke (NaN/Inf なし・[0,255]・非定数) 緑判定 + 20step は DOLLAMA_BENCH=1 で計測のみ (84s)。PipelineGenerator は実 PNG (シグネチャ/1024²)・非1024 reject。factory は不在パス→nullptr フォールバック。GPU 系 timeout 600・is_parallel:false |

### Phase 3 / IO

| コンポーネント | テストファイル | 状態 | 主な検証内容 |
|---|---|---|---|
| HTTP サーバー (OpenAI 互換) | `test_http.cpp` | ✅ 完了 | 自己リクエストで生成→PNG base64 往復・health/models・往復 2.11ms |
| PNG メタ往復 (character-bible §7) | `test_png_meta.cpp` | ✅ 完了 | 構造体⇔§7 JSON⇔tEXt の往復・日本語 name・enum 全網羅 (Sex×Matting×ColorMode)・破損/欠落の前方互換・ベンチ |

### Phase 4 (自作タグ生成 LM, 旧 BitNet b1.58)

| # | コンポーネント | テストファイル | 状態 | 主な検証内容 |
|---|---|---|---|---|
| 4-2 | タグ生成 LM モデル定義 (bitnet.hpp 32.98M) | `test_bitnet.cpp` | ✅ 完了 | param 範囲・ternary/int8 量子化・RMSNorm・embed tied・決定的 forward・logit 健全性・forward ベンチ |
| 4-3 | トークナイザー (タグ単位完全一致) | `test_tokenizer.cpp` | ✅ 完了 | vocab ロード検証・encode/decode 往復 UNK 0・未知→unk・§6 正規化・greedy 最長一致・実 pairs UNK 0・ベンチ |

---

## テストの規約

- コーディング規約は本体と同じ (Allman スタイル、コメント日本語)
- `assert` より `if (!cond) { cerr; return false; }` を推奨 (失敗箇所が分かりやすい)
- マルチスレッドテストのタイムアウトは余裕を持たせる (最低 5 秒)
- ハードウェア依存テスト (NPU, CUDA) は `#ifdef HAVE_OPENVINO` / `#ifdef HAVE_CUDA` でガードする

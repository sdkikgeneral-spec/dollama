deps     = []
cpp_args = []

if openvino_dep.found()
  deps     += openvino_dep
  cpp_args += '-DHAVE_OPENVINO'
endif

if cuda_dep.found()
  deps     += cuda_dep
  cpp_args += '-DHAVE_CUDA'
endif

# src/ をインクルードパスに追加 (core/*.hpp を #include "core/tensor.hpp" で参照可能)
src_inc = include_directories('.')

exe = executable('dollama',
  sources             : files('main.cpp'),
  include_directories : src_inc,
  dependencies        : deps,
  cpp_args            : cpp_args,
)

# SPSCQueue 単体テスト
test_queue_exe = executable('test_queue',
  sources             : files('tests/test_queue.cpp'),
  include_directories : src_inc,
  dependencies        : deps,
  cpp_args            : cpp_args,
)
test('spsc_queue', test_queue_exe)

# Tensor 単体テスト
test_tensor_exe = executable('test_tensor',
  sources             : files('tests/test_tensor.cpp'),
  include_directories : src_inc,
  dependencies        : deps,
  cpp_args            : cpp_args,
)
test('tensor', test_tensor_exe)

# Allocator 単体テスト
test_allocator_exe = executable('test_allocator',
  sources             : files('tests/test_allocator.cpp'),
  include_directories : src_inc,
  dependencies        : deps,
  cpp_args            : cpp_args,
)
test('allocator', test_allocator_exe)

# CLIP NPU 推論テスト
test_clip_exe = executable('test_clip',
  sources             : files('tests/test_clip.cpp'),
  include_directories : src_inc,
  dependencies        : deps,
  cpp_args            : cpp_args,
)
test('clip', test_clip_exe)

# WD14 CPU 推論テスト
test_wd14_exe = executable('test_wd14',
  sources             : files('tests/test_wd14.cpp'),
  include_directories : src_inc,
  dependencies        : deps,
  cpp_args            : cpp_args,
)
test('wd14', test_wd14_exe)

# スレッドアフィニティ単体テスト (STL のみ・常時実行)
test_affinity_exe = executable('test_affinity',
  sources             : files('tests/test_affinity.cpp'),
  include_directories : src_inc,
  dependencies        : deps,
  cpp_args            : cpp_args,
)
test('affinity', test_affinity_exe)

# Pipeline (Phase 1 マルチスレッド骨格) 統合テスト + ベンチ
test_pipeline_exe = executable('test_pipeline',
  sources             : files('tests/test_pipeline.cpp'),
  include_directories : src_inc,
  dependencies        : deps,
  cpp_args            : cpp_args,
)
test('pipeline', test_pipeline_exe, timeout : 120)

# CharacterBible / プロンプト合成テスト
test_character_exe = executable('test_character',
  sources             : files('tests/test_character.cpp'),
  include_directories : src_inc,
  dependencies        : deps,
  cpp_args            : cpp_args,
)
test('character', test_character_exe)

# safetensors 重みローダーテスト (CPU・常時実行、Phase 2 マイルストーン 2-3)
# fixture (golden.safetensors) のパスはソース相対の絶対パスをコンパイル時に埋め込む
# (テスト実行 cwd = build ディレクトリのため)。既存 cpp_args を壊さず専用引数で渡す。
golden_path = meson.project_source_root() / 'src' / 'tests' / 'data' / 'golden.safetensors'
test_safetensors_exe = executable('test_safetensors',
  sources             : files('tests/test_safetensors.cpp'),
  include_directories : src_inc,
  dependencies        : deps,
  cpp_args            : cpp_args + ['-DGOLDEN_PATH="' + golden_path + '"'],
)
test('safetensors', test_safetensors_exe)

# CUDA 疎通テスト + vector_add ベンチ (Phase 2 マイルストーン 2-0 / 2-1)
# .cu を含むため CUDA 言語が有効なときのみビルドする。
if cuda_enabled
  # cpp_args は cpp ソース用なので、.cu には cuda_args で -DHAVE_CUDA を渡す。
  cuda_test_args = ['-DHAVE_CUDA']

  # MSVC ホストコンパイラ (cl) 固有の対処:
  # - /utf-8: 日本語コメント (UTF-8) を CP932 誤読させない (未指定だと C1070)。
  if cpp.get_id() == 'msvc'
    # /utf-8: 日本語コメント (UTF-8) を CP932 誤読させない (未指定だと C1070)。
    cuda_test_args += ['-Xcompiler', '/utf-8']
    # CUDA 13.3 の cudafe++ は VS18 (MSVC 14.51) の c++17/20 標準ライブラリヘッダを
    # 複数組み合わせるとスタックオーバーラン (0xC0000409) でクラッシュする。
    # カーネル側は新しい言語機能を必要としないため、meson が注入する /std:c++20 の
    # 後ろに /std:c++14 を付与して CUDA TU のホストコンパイルだけ c++14 へ落とす
    # (後勝ちで c++14 が有効。cpp 側は c++20 のまま)。
    cuda_test_args += ['-Xcompiler', '/std:c++14']
  endif

  test_cuda_smoke_exe = executable('test_cuda_smoke',
    sources             : files('tests/test_cuda_smoke.cu'),
    include_directories : src_inc,
    dependencies        : deps,
    cpp_args            : cpp_args,
    cuda_args           : cuda_test_args,
  )
  test('cuda_smoke', test_cuda_smoke_exe)

  # dense FP16 GEMM テスト + ベンチ (Phase 2 マイルストーン 2-2-1)
  # gemm.cu をテストと一緒にビルドして launch_gemm_fp16 をリンクする。
  test_gemm_exe = executable('test_gemm',
    sources             : files('tests/test_gemm.cu', 'kernels/gemm.cu'),
    include_directories : src_inc,
    dependencies        : deps,
    cpp_args            : cpp_args,
    cuda_args           : cuda_test_args,
  )
  test('gemm', test_gemm_exe)

  # 要素ごと活性化 (SiLU / GeLU) テスト + ベンチ (Phase 2 マイルストーン 2-2-2)
  # activation.cu をテストと一緒にビルドして launch_silu / launch_gelu をリンクする。
  test_activation_exe = executable('test_activation',
    sources             : files('tests/test_activation.cu', 'kernels/activation.cu'),
    include_directories : src_inc,
    dependencies        : deps,
    cpp_args            : cpp_args,
    cuda_args           : cuda_test_args,
  )
  test('activation', test_activation_exe)

  # GroupNorm テスト + ベンチ (Phase 2 マイルストーン 2-2-3)
  # groupnorm.cu をテストと一緒にビルドして launch_group_norm をリンクする。
  test_groupnorm_exe = executable('test_groupnorm',
    sources             : files('tests/test_groupnorm.cu', 'kernels/groupnorm.cu'),
    include_directories : src_inc,
    dependencies        : deps,
    cpp_args            : cpp_args,
    cuda_args           : cuda_test_args,
  )
  test('groupnorm', test_groupnorm_exe)

  # Conv2d テスト + ベンチ (Phase 2 マイルストーン 2-2-4)
  test_conv2d_exe = executable('test_conv2d',
    sources             : files('tests/test_conv2d.cu', 'kernels/conv2d.cu'),
    include_directories : src_inc,
    dependencies        : deps,
    cpp_args            : cpp_args,
    cuda_args           : cuda_test_args,
  )
  test('conv2d', test_conv2d_exe)

  # Attention テスト + ベンチ (Phase 2 マイルストーン 2-2-5)
  test_attention_exe = executable('test_attention',
    sources             : files('tests/test_attention.cu', 'kernels/attention.cu'),
    include_directories : src_inc,
    dependencies        : deps,
    cpp_args            : cpp_args,
    cuda_args           : cuda_test_args,
  )
  test('attention', test_attention_exe)
endif

# Windows: OpenVINO DLL を build/src/ にコピーして PATH 設定なしで実行できるようにする
if ov_dll_dir != ''
  custom_target('copy_openvino_dlls',
    output           : '.openvino_dlls_copied',
    command          : [
      'powershell', '-NoProfile', '-NonInteractive', '-Command',
      'Copy-Item -Path "' + ov_dll_dir + '/*.dll" -Destination "' +
      meson.current_build_dir() + '" -Force; New-Item -Force -ItemType File "@OUTPUT@" | Out-Null',
    ],
    build_by_default : true,
    depends          : exe,
  )
endif

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

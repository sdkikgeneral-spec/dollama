project('dollama', ['cpp'],
  version : '0.1.0',
  default_options : [
    'cpp_std=c++20',
    'buildtype=release',
    'warning_level=2',
  ]
)

cpp      = meson.get_compiler('cpp')
host_os  = host_machine.system()
is_win   = host_os == 'windows'
is_linux = host_os == 'linux'

# MSVC: ソース/実行文字セットを UTF-8 に統一 (文字化け防止)
if cpp.get_id() == 'msvc'
  add_project_arguments('/utf-8', language: 'cpp')
endif

# ----------------------------------------------------------------
# OpenVINO (NPU / iGPU)
# ----------------------------------------------------------------
openvino_dep = dependency('', required: false)  # デフォルト: 無効
ov_dll_dir   = ''                               # Windows DLL コピー用 (src/meson.build で参照)

if get_option('with_openvino')
  ov_root = get_option('npu_sdk_root')
  if ov_root == ''
    ov_root = is_win ? 'C:/Program Files (x86)/Intel/openvino_2024' \
                     : '/opt/intel/openvino_2024'
  endif

  # pip パッケージ形式 (include/ + libs/) と SDK 形式 (runtime/include/ + runtime/lib/) を自動判別
  fs = import('fs')
  if fs.exists(ov_root / 'libs')
    ov_inc     = include_directories(ov_root / 'include', is_system: true)
    ov_lib_dir = ov_root / 'libs'
  else
    ov_inc     = include_directories(ov_root / 'runtime/include', is_system: true)
    ov_lib_dir = is_win ? ov_root / 'runtime/lib/intel64/Release' \
                        : ov_root / 'runtime/lib/intel64'
  endif
  ov_lib     = cpp.find_library('openvino', dirs: [ov_lib_dir], required: false)

  if ov_lib.found()
    openvino_dep = declare_dependency(
      include_directories : ov_inc,
      dependencies        : [ov_lib],
    )
    if is_win
      ov_dll_dir = ov_lib_dir  # src/meson.build で DLL を build/src/ にコピー
    endif
    message('OpenVINO: ' + ov_root)
  else
    warning('OpenVINO が見つかりません。-Dopenvino_root= で指定してください。')
  endif
endif

# ----------------------------------------------------------------
# CUDA (RTX5080 / Blackwell / sm_120)
# ----------------------------------------------------------------
# CUDA 言語は with_cuda が true のときだけ有効化する。
# project() に 'cuda' を常時含めると nvcc 不在環境で setup が壊れるため、
# add_languages() を条件付き (required: with_cuda) で呼ぶことで回避する。
cuda_dep     = dependency('', required: false)
cuda_enabled = false

if get_option('with_cuda')
  cuda_root = get_option('gpu_sdk_root')
  if cuda_root == ''
    cuda_root = is_win ? 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3' \
                       : '/usr/local/cuda'
  endif

  cuda_inc     = include_directories(cuda_root / 'include', is_system: true)
  cuda_lib_dir = is_win ? cuda_root / 'lib/x64' : cuda_root / 'lib64'
  cuda_rt      = cpp.find_library('cudart', dirs: [cuda_lib_dir], required: false)

  if cuda_rt.found()
    # cpp 側 (allocator.hpp など) が cudaMalloc 等をリンクするための依存。
    # CUDA 言語有効化前から維持する (壊さない)。
    cuda_dep = declare_dependency(
      include_directories : cuda_inc,
      dependencies        : [cuda_rt],
    )

    # .cu をコンパイルするための CUDA 言語を有効化 (nvcc 検出)。
    cuda_enabled = add_languages('cuda', required: true, native: false)
    if cuda_enabled
      add_project_arguments('-arch=sm_120', language: 'cuda')
      message('CUDA: ' + cuda_root + ' (sm_120 有効)')
    else
      warning('nvcc が見つかりません。.cu のコンパイルは無効です。')
    endif
  else
    warning('CUDA が見つかりません。-Dgpu_sdk_root= で指定してください。')
  endif
endif

subdir('src')

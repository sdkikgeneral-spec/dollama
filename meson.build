project('dollama', ['cpp'],
  version : '0.1.0',
  default_options : [
    'cpp_std=c++20',
    'buildtype=release',
    'warning_level=2',
    # S3-0: release ビルドでのみ assert を無効化 (-DNDEBUG)。
    # device/host 両方の assert と CCCL_ENABLE_ASSERTIONS を外す。
    # debug ビルドでは assert を残す。revert は 'b_ndebug=false' を指定。
    'b_ndebug=if-release',
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
# S3-0 ホスト最適化フラグ (タスク 2-6 最適化・ビルド構成のみ)
# すべて meson option で on/off 可。崩れたら 1 フラグで revert できる。
# ----------------------------------------------------------------
# host_avx2: MSVC ホストに /arch:AVX2 (safetensors ロード / WD14 グルー / HTTP /
#            JSON / base64 など host 側 SIMD。数〜十数%)。.cu のホスト部にも下で渡す。
if cpp.get_id() == 'msvc' and get_option('host_avx2')
  add_project_arguments('/arch:AVX2', language: 'cpp')
  message('host_avx2: /arch:AVX2 (cpp) 有効')
endif

# host_lto: ホスト LTO。nvcc obj と MSVC LTCG は非互換のため、CUDA をリンクする
#           exe (dollama 本体 / test_unet / test_diffusion 等) の link が壊れうる。
#           そのため既定 false。true のときだけ cpp 側に LTO フラグを足す。
#           CUDA exe で link が通るかは setup→compile で実検証する。
if get_option('host_lto')
  if cpp.get_id() == 'msvc'
    # MSVC: コンパイル時 /GL + リンク時 /LTCG。
    add_project_arguments('/GL', language: 'cpp')
    add_project_link_arguments('/LTCG', language: 'cpp')
  else
    add_project_arguments('-flto', language: 'cpp')
    add_project_link_arguments('-flto', language: 'cpp')
  endif
  message('host_lto: ホスト LTO 有効 (CUDA exe の link 互換性に注意)')
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
      # S3-0 fast_math: device 側 fast-math (数%)。revert は -Dfast_math=false。
      if get_option('fast_math')
        add_project_arguments('--use_fast_math', language: 'cuda')
        message('fast_math: --use_fast_math (cuda) 有効')
      endif
      message('CUDA: ' + cuda_root + ' (sm_120 有効)')
    else
      warning('nvcc が見つかりません。.cu のコンパイルは無効です。')
    endif
  else
    warning('CUDA が見つかりません。-Dgpu_sdk_root= で指定してください。')
  endif
endif

# ----------------------------------------------------------------
# nlohmann/json (ヘッダオンリー・常時解決)
# ----------------------------------------------------------------
# PNG メタ往復 (src/io/png_meta.hpp) など HTTP 以外でも JSON を使うため、
# with_http に依存せず常に解決しておく (ヘッダオンリーなのでコスト極小)。
# wrap が無い環境では無効になり、JSON を要するテストは登録をスキップする。
json_dep = dependency('nlohmann_json', fallback: ['nlohmann_json', 'nlohmann_json_dep'], required: false)

# ----------------------------------------------------------------
# HTTP サーバー (cpp-httplib + nlohmann/json, ヘッダオンリー)
# ----------------------------------------------------------------
# 配管は自作せず subproject (wrap) で取り込む。両者ともヘッダオンリーなので
# include_directories を渡すだけ。HTTP サーバ自体は CUDA/OpenVINO 非依存の純 C++。
http_dep     = dependency('', required: false)  # デフォルト: 無効
http_enabled = false

if get_option('with_http')
  httplib_dep = dependency('cpp-httplib', required: true)
  if not json_dep.found()
    json_dep = dependency('nlohmann_json', fallback: ['nlohmann_json', 'nlohmann_json_dep'], required: true)
  endif

  http_extra = []
  if is_win
    # cpp-httplib は Winsock2 を使う。MSVC では ws2_32 を明示リンクする。
    http_extra += cpp.find_library('ws2_32', required: true)
  endif

  http_dep = declare_dependency(
    dependencies : [json_dep, httplib_dep] + http_extra,
  )
  http_enabled = true
  message('HTTP: cpp-httplib + nlohmann/json (有効)')
endif

subdir('src')

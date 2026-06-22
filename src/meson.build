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

# HTTP サーバー (cpp-httplib + nlohmann/json)。純 C++・CUDA/OpenVINO 非依存。
# exe / test_http にのみ追加する (推論テスト群には不要)。
http_deps     = []
http_cpp_args = []
if http_enabled
  http_deps     += http_dep
  http_cpp_args += '-DHAVE_HTTP'
endif

# src/ をインクルードパスに追加 (core/*.hpp を #include "core/tensor.hpp" で参照可能)
src_inc = include_directories('.')

# .cu 用 CUDA 引数 (CUDA 有効時のみ参照)。
# - -DHAVE_CUDA: pipeline_generator.hpp 等のガードを開く。
# - /utf-8: 日本語コメント (UTF-8) を CP932 誤読させない (MSVC ホスト)。
# - /std:c++14: CUDA 13.3 + MSVC で c++17/20 ヘッダ組合せの 0xC0000409 を回避。
cuda_args = []
if cuda_enabled
  cuda_args = ['-DHAVE_CUDA']
  if cpp.get_id() == 'msvc'
    cuda_args += ['-Xcompiler', '/utf-8']
    cuda_args += ['-Xcompiler', '/std:c++14']
  endif
endif

# DiffusionPipeline を構成する CUDA カーネル群 (test_diffusion と同一)。
# main 実行ファイルが PipelineGenerator (= DiffusionPipeline) をリンクするため必要。
diffusion_cu_sources = [
  'infer/diffusion.cu',
  'infer/unet.cu',
  'kernels/vae_decode.cu',
  'kernels/conv2d.cu',
  'kernels/groupnorm.cu',
  'kernels/layernorm.cu',
  'kernels/activation.cu',
  'kernels/geglu.cu',
  'kernels/attention.cu',
  'kernels/gemm.cu',
  'kernels/elementwise.cu',
  'kernels/bias_add.cu',
  'kernels/timeembed.cu',
]

# ----------------------------------------------------------------
# dollama 実行ファイル (main)
# ----------------------------------------------------------------
# DI 設計 (ST-5):
#   main.cpp は CUDA 非依存の cpp TU。PipelineGenerator の構築は
#   make_pipeline_generator() ファクトリ越し。実装を CUDA 有無で切り替える:
#     - CUDA 有効: pipeline_generator_factory.cu + DiffusionPipeline 全 .cu をリンク。
#     - CUDA 無効: pipeline_generator_factory_stub.cpp (nullptr 返し) をリンク。
#   これにより CUDA 無効ビルドでも main がリンクでき、重み不在でも HTTP が起動する。
exe_sources = ['main.cpp']
exe_cuda_sources = []
if http_enabled
  exe_sources += 'server/api.cpp'
  if cuda_enabled
    # CUDA 有効: 本 txt2img を DI 可能にする。factory.cu と全カーネルをリンク。
    exe_cuda_sources += 'server/pipeline_generator_factory.cu'
    exe_cuda_sources += diffusion_cu_sources
  else
    # CUDA 無効: ファクトリは nullptr を返す cpp スタブ。main は Stub にフォールバック。
    exe_sources += 'server/pipeline_generator_factory_stub.cpp'
  endif
endif

exe = executable('dollama',
  sources             : files(exe_sources + exe_cuda_sources),
  include_directories : src_inc,
  dependencies        : deps + http_deps,
  cpp_args            : cpp_args + http_cpp_args,
  cuda_args           : cuda_args,
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
# OV モデルロードが重く既定 30s を超えうる (並列実行時の CPU 競合で顕著)。
test('wd14', test_wd14_exe, timeout : 120)

# マッティング (ISNet-anime) CPU 推論テスト (M-4)
# matting.hpp は OV ガード越し (HAVE_OPENVINO 未定義時は推論 [SKIP]・compose_rgba は常時)。
# モデル (model_ov_fp32.xml) は .gitignore 済・再生成可。golden は src/tests/data/matting/。
# パスは meson の cpp_args で -D 埋め込み (cwd 非依存)。OV/モデル/golden 不在は test 側で [SKIP]。
matting_model_xml   = meson.project_source_root() / 'models' / 'isnet-anime' / 'model_ov_fp32.xml'
matting_golden_path = meson.project_source_root() / 'src' / 'tests' / 'data' / 'matting' / 'golden_isnet.safetensors'
test_matting_exe = executable('test_matting',
  sources             : files('tests/test_matting.cpp'),
  include_directories : src_inc,
  dependencies        : deps,
  cpp_args            : cpp_args + [
    '-DMATTING_MODEL_XML="' + matting_model_xml + '"',
    '-DMATTING_GOLDEN_PATH="' + matting_golden_path + '"',
  ],
)
# ISNet-anime は重い (CPU ~583ms/推論 x ベンチ N=7 + ロード)。既定 30s を超えうるため延伸。
test('matting', test_matting_exe, timeout : 180)

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

# make_pipeline_generator フォールバック判定テスト (純 cpp・CUDA 不要、ST-5)
# 不在パス → nullptr (= StubGenerator フォールバック経路) を保証する。
# ファクトリ実装は exe と同じく CUDA 有無で切り替えてリンクする。
factory_test_cuda_sources = []
if cuda_enabled
  factory_test_cuda_sources += 'server/pipeline_generator_factory.cu'
  factory_test_cuda_sources += diffusion_cu_sources
  test_pipeline_factory_exe = executable('test_pipeline_factory',
    sources             : files(['tests/test_pipeline_factory.cpp'] + factory_test_cuda_sources),
    include_directories : src_inc,
    dependencies        : deps,
    cpp_args            : cpp_args,
    cuda_args           : cuda_args,
  )
else
  test_pipeline_factory_exe = executable('test_pipeline_factory',
    sources             : files('tests/test_pipeline_factory.cpp',
                                'server/pipeline_generator_factory_stub.cpp'),
    include_directories : src_inc,
    dependencies        : deps,
    cpp_args            : cpp_args,
  )
endif
test('pipeline_factory', test_pipeline_factory_exe)

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

# PNG エンコーダ (RGB8 / RGBA8) テスト (M-3 RGBA 透過 PNG 追加)
# 純 C++ (CUDA/OpenVINO/JSON 非依存)・常時実行。
test_png_exe = executable('test_png',
  sources             : files('tests/test_png.cpp'),
  include_directories : src_inc,
  dependencies        : deps,
  cpp_args            : cpp_args,
)
test('png', test_png_exe)

# PNG メタ往復 (character-bible-spec §7 tEXt 焼き込み) テスト + ベンチ
# 純 C++ (CUDA/OpenVINO 非依存)・常時実行。JSON は nlohmann/json (json_dep)。
# json_dep が解決できた場合のみ登録する (wrap 不在環境では自動スキップ)。
if json_dep.found()
  test_png_meta_exe = executable('test_png_meta',
    sources             : files('tests/test_png_meta.cpp'),
    include_directories : src_inc,
    dependencies        : deps + [json_dep],
    cpp_args            : cpp_args,
  )
  test('png_meta', test_png_meta_exe)
endif

# EulerDiscreteScheduler テスト (純 C++・CUDA 不要、Phase 2 マイルストーン 2-5)
# golden (unet_io.safetensors) のパスをコンパイル時に -D 埋め込み (cwd 非依存)。
# golden 不在時は [SKIP] return 0 (test_scheduler 側で判定)。
unet_io_path = meson.project_source_root() / 'src' / 'tests' / 'data' / 'unet_io.safetensors'
test_scheduler_exe = executable('test_scheduler',
  sources             : files('tests/test_scheduler.cpp'),
  include_directories : src_inc,
  dependencies        : deps,
  cpp_args            : cpp_args + ['-DUNET_IO_PATH="' + unet_io_path + '"'],
)
test('scheduler', test_scheduler_exe)

# BitNet b1.58 モデル定義テスト + forward ベンチ (純 C++・CUDA 不要、Phase 4 #2)
# bitnet.hpp は STL のみで自己完結 (量子化 / RMSNorm / 参照 forward)。
# fixture / -D 埋め込み不要 (固定 seed の決定的テスト)。
test_bitnet_exe = executable('test_bitnet',
  sources             : files('tests/test_bitnet.cpp'),
  include_directories : src_inc,
  dependencies        : deps,
  cpp_args            : cpp_args,
)
# 参照 forward は naive (重み毎回 re-quantize + lm_head 4999x512) で重く、
# ベンチ (5 回 forward x ~19s) を含めると ~2 分かかる。既定 30s を大きく超えるため延伸。
test('bitnet', test_bitnet_exe, timeout : 600)

# タグトークナイザテスト (純 C++・CUDA/OpenVINO 非依存、Phase 4 dense #3)
# vocab.json は data/bitnet/ にリポジトリ追跡済 (必須・不在なら fail)。
# pairs.*.jsonl は実回帰用 (不在時は test 側で [SKIP])。パスは -D 埋め込み (cwd 非依存)。
tok_vocab_path = meson.project_source_root() / 'data' / 'bitnet' / 'vocab.json'
tok_train_path = meson.project_source_root() / 'data' / 'bitnet' / 'pairs.train.jsonl'
tok_val_path   = meson.project_source_root() / 'data' / 'bitnet' / 'pairs.val.jsonl'
test_tokenizer_exe = executable('test_tokenizer',
  sources             : files('tests/test_tokenizer.cpp'),
  include_directories : src_inc,
  dependencies        : deps,
  cpp_args            : cpp_args + [
    '-DVOCAB_PATH="' + tok_vocab_path + '"',
    '-DPAIRS_TRAIN_PATH="' + tok_train_path + '"',
    '-DPAIRS_VAL_PATH="' + tok_val_path + '"',
  ],
)
test('tokenizer', test_tokenizer_exe, timeout : 120)

# BitNet dense (FP32) 推論テスト (純 C++・CUDA/OpenVINO 非依存、Phase 4 dense #6)
# dense forward + greedy デコードを PyTorch golden と数値突合する。
# 重み (bitnet_dense_fp32.safetensors) / golden (golden/*.safetensors) は gitignore 対象。
# パスは -D 埋め込み (cwd 非依存)。いずれか不在なら test 側で [SKIP] (CI でビルド緑維持)。
# naive 参照 forward を greedy で複数ステップ回すため重い (既定 30s 超) -> timeout 延伸。
bn_weights_path = meson.project_source_root() / 'data' / 'bitnet' / 'bitnet_dense_fp32.safetensors'
bn_logits_path  = meson.project_source_root() / 'data' / 'bitnet' / 'golden' / 'logits_golden.safetensors'
bn_gen_path     = meson.project_source_root() / 'data' / 'bitnet' / 'golden' / 'gen_golden.safetensors'
# Phase 4 A (A3) 同一性条件付き版: 別重み + identity_ 別 golden + vocab.json。
# vocab は generate_with_identity の prompt 組み立て (tag_to_id + scene greedy) に必要。
bn_id_weights_path = meson.project_source_root() / 'data' / 'bitnet' / 'bitnet_dense_identity_fp32.safetensors'
bn_id_logits_path  = meson.project_source_root() / 'data' / 'bitnet' / 'golden' / 'logits_golden_identity.safetensors'
bn_id_gen_path     = meson.project_source_root() / 'data' / 'bitnet' / 'golden' / 'gen_golden_identity.safetensors'
bn_vocab_path      = meson.project_source_root() / 'data' / 'bitnet' / 'vocab.json'
test_bitnet_infer_exe = executable('test_bitnet_infer',
  sources             : files('tests/test_bitnet_infer.cpp'),
  include_directories : src_inc,
  dependencies        : deps,
  cpp_args            : cpp_args + [
    '-DWEIGHTS_PATH="' + bn_weights_path + '"',
    '-DLOGITS_GOLDEN_PATH="' + bn_logits_path + '"',
    '-DGEN_GOLDEN_PATH="' + bn_gen_path + '"',
    '-DIDENTITY_WEIGHTS_PATH="' + bn_id_weights_path + '"',
    '-DIDENTITY_LOGITS_GOLDEN_PATH="' + bn_id_logits_path + '"',
    '-DIDENTITY_GEN_GOLDEN_PATH="' + bn_id_gen_path + '"',
    '-DVOCAB_PATH="' + bn_vocab_path + '"',
  ],
)
test('bitnet_infer', test_bitnet_infer_exe, timeout : 600)

# CUDA 疎通テスト + vector_add ベンチ (Phase 2 マイルストーン 2-0 / 2-1)
# .cu を含むため CUDA 言語が有効なときのみビルドする。
if cuda_enabled
  # cpp_args は cpp ソース用なので、.cu には cuda_args で -DHAVE_CUDA を渡す。
  # (cuda_args はファイル冒頭で定義済み。テスト群でもそれを使う。)
  cuda_test_args = cuda_args

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
    sources             : files('tests/test_conv2d.cu', 'kernels/conv2d.cu', 'kernels/gemm.cu'),
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
  test('attention', test_attention_exe, timeout : 120)

  # Elementwise (residual add / nearest upsample 2x) テスト + ベンチ (Phase 2 マイルストーン 2-4)
  test_elementwise_exe = executable('test_elementwise',
    sources             : files('tests/test_elementwise.cu', 'kernels/elementwise.cu'),
    include_directories : src_inc,
    dependencies        : deps,
    cpp_args            : cpp_args,
    cuda_args           : cuda_test_args,
  )
  test('elementwise', test_elementwise_exe)

  # LayerNorm テスト + ベンチ (Phase 2 マイルストーン 2-5 / UNet Transformer2D)
  test_layernorm_exe = executable('test_layernorm',
    sources             : files('tests/test_layernorm.cu', 'kernels/layernorm.cu'),
    include_directories : src_inc,
    dependencies        : deps,
    cpp_args            : cpp_args,
    cuda_args           : cuda_test_args,
  )
  test('layernorm', test_layernorm_exe)

  # GEGLU テスト + ベンチ (Phase 2 マイルストーン 2-5 / UNet FeedForward)
  test_geglu_exe = executable('test_geglu',
    sources             : files('tests/test_geglu.cu', 'kernels/geglu.cu'),
    include_directories : src_inc,
    dependencies        : deps,
    cpp_args            : cpp_args,
    cuda_args           : cuda_test_args,
  )
  test('geglu', test_geglu_exe)

  # sinusoidal time embedding テスト + ベンチ (Phase 2 マイルストーン 2-5 / UNet)
  test_timeembed_exe = executable('test_timeembed',
    sources             : files('tests/test_timeembed.cu', 'kernels/timeembed.cu'),
    include_directories : src_inc,
    dependencies        : deps,
    cpp_args            : cpp_args,
    cuda_args           : cuda_test_args,
  )
  test('timeembed', test_timeembed_exe)

  # broadcast bias add テスト + ベンチ (Phase 2 マイルストーン 2-5 / UNet)
  test_bias_add_exe = executable('test_bias_add',
    sources             : files('tests/test_bias_add.cu', 'kernels/bias_add.cu'),
    include_directories : src_inc,
    dependencies        : deps,
    cpp_args            : cpp_args,
    cuda_args           : cuda_test_args,
  )
  test('bias_add', test_bias_add_exe)

  # VAE decoder ゴールデン突合テスト + レイテンシ計測 (Phase 2 マイルストーン 2-4 本体)
  # vae_decode.cu と全プリミティブ .cu を一緒にビルドしてリンクする。
  # ゴールデンパスはソース相対の絶対パスをコンパイル時に -D 埋め込み (cwd 非依存)。
  vae_weights_path = meson.project_source_root() / 'src' / 'tests' / 'data' / 'vae_weights.safetensors'
  vae_io_path      = meson.project_source_root() / 'src' / 'tests' / 'data' / 'vae_io.safetensors'
  test_vae_decode_exe = executable('test_vae_decode',
    sources             : files(
      'tests/test_vae_decode.cu',
      'kernels/vae_decode.cu',
      'kernels/conv2d.cu',
      'kernels/groupnorm.cu',
      'kernels/activation.cu',
      'kernels/attention.cu',
      'kernels/gemm.cu',
      'kernels/elementwise.cu',
    ),
    include_directories : src_inc,
    dependencies        : deps,
    cpp_args            : cpp_args,
    cuda_args           : cuda_test_args + [
      '-DVAE_WEIGHTS_PATH="' + vae_weights_path + '"',
      '-DVAE_IO_PATH="' + vae_io_path + '"',
    ],
  )
  # 重い GPU テスト (UNet 重み 5.1GB 常駐) は is_parallel:false で単独実行。
  # 複数の 5GB テストが並列に走ると VRAM (16GB) を食い潰し OOM/タイムアウトするため。
  test('vae_decode', test_vae_decode_exe, timeout : 300, is_parallel : false)

  # UNet2DConditionModel ゴールデン突合テスト + 1step レイテンシ計測 (Phase 2 マイルストーン 2-5)
  # unet.cu と全プリミティブ .cu を一緒にビルドしてリンクする。
  # ゴールデンパスはソース相対の絶対パスをコンパイル時に -D 埋め込み (cwd 非依存)。
  unet_weights_path = meson.project_source_root() / 'src' / 'tests' / 'data' / 'unet_weights.safetensors'
  unet_io_path2     = meson.project_source_root() / 'src' / 'tests' / 'data' / 'unet_io.safetensors'
  test_unet_exe = executable('test_unet',
    sources             : files(
      'tests/test_unet.cu',
      'infer/unet.cu',
      'kernels/conv2d.cu',
      'kernels/groupnorm.cu',
      'kernels/layernorm.cu',
      'kernels/activation.cu',
      'kernels/geglu.cu',
      'kernels/attention.cu',
      'kernels/gemm.cu',
      'kernels/elementwise.cu',
      'kernels/bias_add.cu',
      'kernels/timeembed.cu',
    ),
    include_directories : src_inc,
    dependencies        : deps,
    cpp_args            : cpp_args,
    cuda_args           : cuda_test_args + [
      '-DUNET_WEIGHTS_PATH="' + unet_weights_path + '"',
      '-DUNET_IO_PATH="' + unet_io_path2 + '"',
    ],
  )
  test('unet', test_unet_exe, timeout : 600, is_parallel : false)

  # 拡散ループ結線 (DiffusionPipeline) smoke + フルベンチ (Phase 2 マイルストーン 2-6a)
  # diffusion.cu + unet.cu + vae_decode.cu + 全プリミティブ .cu を一括ビルドしてリンクする。
  # golden 埋め込みは unet_io.safetensors (UNET_IO_PATH) を使い回す。
  # 重みパス (UNET_WEIGHTS_PATH / VAE_WEIGHTS_PATH / UNET_IO_PATH) は -D 埋め込み。
  diff_unet_w  = meson.project_source_root() / 'src' / 'tests' / 'data' / 'unet_weights.safetensors'
  diff_vae_w   = meson.project_source_root() / 'src' / 'tests' / 'data' / 'vae_weights.safetensors'
  diff_unet_io = meson.project_source_root() / 'src' / 'tests' / 'data' / 'unet_io.safetensors'
  test_diffusion_exe = executable('test_diffusion',
    sources             : files(['tests/test_diffusion.cu'] + diffusion_cu_sources),
    include_directories : src_inc,
    dependencies        : deps,
    cpp_args            : cpp_args,
    cuda_args           : cuda_test_args + [
      '-DUNET_WEIGHTS_PATH="' + diff_unet_w + '"',
      '-DVAE_WEIGHTS_PATH="' + diff_vae_w + '"',
      '-DUNET_IO_PATH="' + diff_unet_io + '"',
    ],
  )
  test('diffusion', test_diffusion_exe, timeout : 600, is_parallel : false)

  # PipelineGenerator (DiffusionPipeline -> IImageGenerator アダプタ) テスト (Phase 2 / 2-6a ST-3+ST-4)
  # pipeline_generator.hpp は HAVE_CUDA ガード越しに diffusion.cuh を include するため
  # .cu (nvcc 経由) でビルドする。リンクするソースは test_diffusion と同一 (diffusion.cu +
  # unet.cu + vae_decode.cu + 全プリミティブ)。重みパスは -D 埋め込み (cwd 非依存)。
  test_pipeline_generator_exe = executable('test_pipeline_generator',
    sources             : files(['tests/test_pipeline_generator.cu'] + diffusion_cu_sources),
    include_directories : src_inc,
    dependencies        : deps,
    cpp_args            : cpp_args,
    cuda_args           : cuda_test_args + [
      '-DUNET_WEIGHTS_PATH="' + diff_unet_w + '"',
      '-DVAE_WEIGHTS_PATH="' + diff_vae_w + '"',
      '-DUNET_IO_PATH="' + diff_unet_io + '"',
    ],
  )
  test('pipeline_generator', test_pipeline_generator_exe, timeout : 600, is_parallel : false)
endif

# HTTP サーバー (OpenAI Images 互換) テスト — 純 C++・CUDA/OpenVINO 非依存・常時実行
# StubGenerator を注入し httplib::Server を bind_to_any_port で起動して自己リクエスト。
if http_enabled
  test_http_exe = executable('test_http',
    sources             : files('tests/test_http.cpp', 'server/api.cpp'),
    include_directories : src_inc,
    dependencies        : deps + http_deps,
    cpp_args            : cpp_args + http_cpp_args,
  )
  test('http', test_http_exe, timeout : 60)
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

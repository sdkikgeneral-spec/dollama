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
  # cpp-httplib は Windows 10 以降の Win32 API を要求する。MinGW GCC は
  # _WIN32_WINNT を既定で低く設定するため、httplib.h の Windows 版判定 /
  # CreateFile2 が通るよう Windows 10 (0x0A00) を明示する (MSVC では既定で充足)。
  # なお非同期名前解決 (GetAddrInfoExCancel) は MinGW の msvcrt ヘッダに無いため、
  # サブプロジェクトオプション cpp-httplib:non_blocking_getaddrinfo=disabled で無効化する。
  if is_win
    http_cpp_args += '-D_WIN32_WINNT=0x0A00'
  endif
endif

# src/ をインクルードパスに追加 (core/*.hpp を #include "core/tensor.hpp" で参照可能)
src_inc = include_directories('.')

# .cu 用 CUDA 引数 (CUDA 有効時のみ参照)。
# - -DHAVE_CUDA: pipeline_generator.hpp 等のガードを開く。
# - /utf-8: 日本語コメント (UTF-8) を CP932 誤読させない (MSVC ホスト)。
# - /std:c++14: CUDA 13.3 + MSVC で c++17/20 ヘッダ組合せの 0xC0000409 を回避。
# - /arch:AVX2: S3-0。.cu のホスト部 (カーネル起動グルー等) にも AVX2 を効かせる。
#   -Xcompiler でホストコンパイラ (cl) にのみ転送 (device codegen には無影響)。
#   revert は -Dhost_avx2=false。
cuda_args = []
if cuda_enabled
  cuda_args = ['-DHAVE_CUDA']
  if cpp.get_id() == 'msvc'
    cuda_args += ['-Xcompiler', '/utf-8']
    cuda_args += ['-Xcompiler', '/std:c++14']
    if get_option('host_avx2')
      cuda_args += ['-Xcompiler', '/arch:AVX2']
    endif
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
    # CUDA 有効: 本 txt2img を DI 可能にする。factory.cu / diffusion_runner.cu と全カーネルをリンク。
    exe_cuda_sources += 'server/pipeline_generator_factory.cu'
    exe_cuda_sources += 'server/diffusion_runner.cu'
    exe_cuda_sources += diffusion_cu_sources
  else
    # CUDA 無効: ファクトリ / runner は nullptr を返す cpp スタブ。main は Stub にフォールバック。
    exe_sources += 'server/pipeline_generator_factory_stub.cpp'
    exe_sources += 'server/diffusion_runner_stub.cpp'
  endif
  # M-6: マッティング (IMatter) の make_matter 実体。cli_generate.hpp が呼ぶ。
  #   OV 有効 → OV 実体 (matter_runner.cpp)、OV 無効 → nullptr 返しの stub。
  if openvino_dep.found()
    exe_sources += 'server/matter_runner.cpp'
  else
    exe_sources += 'server/matter_runner_stub.cpp'
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

# TextConditioner (SDXL prompt -> UNet 入力埋め込み) テスト (Phase 2-6b Stage D)
# text_conditioner.hpp / clip_tokenizer.hpp / clip_encoder2.hpp は HAVE_OPENVINO ガード越し。
# golden / モデル / openvino_tokenizers.dll のパスを -D 埋め込み (cwd 非依存)。
# openvino_tokenizers.dll は python の openvino_tokenizers パッケージから自動検出する
# (見つからなければ空文字 → test 側で [SKIP])。
txt2img_golden    = meson.project_source_root() / 'src' / 'tests' / 'data' / 'txt2img_io.safetensors'
sdxl_tok_l_xml    = meson.project_source_root() / 'models' / 'sdxl-tokenizer-l' / 'openvino_tokenizer.xml'
sdxl_tok_g_xml    = meson.project_source_root() / 'models' / 'sdxl-tokenizer-g' / 'openvino_tokenizer.xml'
sdxl_enc_l_xml    = meson.project_source_root() / 'models' / 'sdxl-text-encoder-l' / 'model_ov.xml'
sdxl_enc_g_xml    = meson.project_source_root() / 'models' / 'sdxl-text-encoder-g' / 'model_ov.xml'

ov_tok_dll = ''
find_dll = run_command('python',
  meson.project_source_root() / 'scripts' / 'dollma_find_ov_tokenizers_dll.py',
  check : false)
if find_dll.returncode() == 0
  ov_tok_dll = find_dll.stdout().strip()
endif

test_text_conditioner_exe = executable('test_text_conditioner',
  sources             : files('tests/test_text_conditioner.cpp'),
  include_directories : src_inc,
  dependencies        : deps,
  cpp_args            : cpp_args + [
    '-DTXT2IMG_GOLDEN_PATH="' + txt2img_golden + '"',
    '-DTOKENIZER_L_XML="' + sdxl_tok_l_xml + '"',
    '-DTOKENIZER_G_XML="' + sdxl_tok_g_xml + '"',
    '-DENCODER_L_XML="' + sdxl_enc_l_xml + '"',
    '-DENCODER_G_XML="' + sdxl_enc_g_xml + '"',
    '-DTOKENIZERS_DLL="' + ov_tok_dll + '"',
  ],
)
# OV モデル 3 本ロード + NPU/CPU コンパイルで重い。既定 30s を超えうるため延伸。
test('text_conditioner', test_text_conditioner_exe, timeout : 180)

# WD14 CPU 推論テスト
test_wd14_exe = executable('test_wd14',
  sources             : files('tests/test_wd14.cpp'),
  include_directories : src_inc,
  dependencies        : deps,
  cpp_args            : cpp_args,
)
# OV モデルロードが重く既定 30s を超えうる (並列実行時の CPU 競合で顕著)。
test('wd14', test_wd14_exe, timeout : 120)

# Model B Stage 1: QualityGate (WD14 異常タグ soft ゲート) テスト
# quality_gate.hpp は OV 非依存 (WD14 スコアベクタを引数で受ける)。
# 合成 CSV + 合成スコアで実走するため OV ロード不要・常時緑 (timeout 延伸不要)。
test_quality_gate_exe = executable('test_quality_gate',
  sources             : files('tests/test_quality_gate.cpp'),
  include_directories : src_inc,
  dependencies        : deps,
  cpp_args            : cpp_args,
)
test('quality_gate', test_quality_gate_exe)

# マッティング (ISNet-anime) CPU 推論テスト (M-4)
# matting.hpp は OV ガード越し (HAVE_OPENVINO 未定義時は推論 [SKIP]・compose_rgba は常時)。
# モデル (model_ov_fp32.xml) は .gitignore 済・再生成可。golden は src/tests/data/matting/。
# パスは meson の cpp_args で -D 埋め込み (cwd 非依存)。OV/モデル/golden 不在は test 側で [SKIP]。
matting_model_xml   = meson.project_source_root() / 'models' / 'isnet-anime' / 'model_ov_fp32.xml'
matting_golden_path = meson.project_source_root() / 'src' / 'tests' / 'data' / 'matting' / 'golden_isnet.safetensors'
# M-6: test_matting は encode_png_maybe_transparent (matting_postprocess.hpp) を
#   検証するため make_matter の実体が要る。OV 有無で matter_runner(_stub) を切り替える
#   (exe と同じ DI 設計・開発機 OV 無では stub をリンク → make_matter は nullptr)。
matting_sources = ['tests/test_matting.cpp']
if openvino_dep.found()
  matting_sources += 'server/matter_runner.cpp'
else
  matting_sources += 'server/matter_runner_stub.cpp'
endif
test_matting_exe = executable('test_matting',
  sources             : files(matting_sources),
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

# BitNet d80m config (容量増・施策 D) 疎通テスト (純 C++・CUDA 不要)
# -DDOLLAMA_BITNET_ARCH=1 でアーキを d80m (N_LAYERS=16 / FFN_DIM=2464) に切替えてビルドし、
# default ビルド (test_bitnet) を一切壊さず、d80m 構成の param_count 厳密検証 +
# host forward 疎通 + safetensors 往復を検証する。
# 一時 safetensors の書き出し先はビルドディレクトリ配下 (-DD80M_TMP_PATH)。
d80m_tmp_path = meson.current_build_dir() / 'test_bitnet_d80m_tmp.safetensors'
test_bitnet_d80m_exe = executable('test_bitnet_d80m',
  sources             : files('tests/test_bitnet_d80m.cpp'),
  include_directories : src_inc,
  dependencies        : deps,
  cpp_args            : cpp_args + [
    '-DDOLLAMA_BITNET_ARCH=1',
    '-DD80M_TMP_PATH="' + d80m_tmp_path + '"',
  ],
)
# d80m (16 層) の ternary 参照 forward は default の 2 倍重い。既定 30s を超えうるため延伸。
test('bitnet_d80m', test_bitnet_d80m_exe, timeout : 600)

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

# BitNet dense 推論プロファイラ (Tier 1 速度最適化・計測専用 exe・test 非登録)
# 本番 bitnet.hpp は無改変・public API (forward / forward_fast) 経由。
# double 参照経路と float32+AVX2 高速パスの seq8/32/63 中央値と倍率を出す。
# 重み (bitnet_dense_fp32.safetensors) 不在時は health に skip 終了 (return 0)。
prof_bitnet_exe = executable('prof_bitnet',
  sources             : files('tests/prof_bitnet.cpp'),
  include_directories : src_inc,
  dependencies        : deps,
  cpp_args            : cpp_args + ['-DWEIGHTS_PATH="' + bn_weights_path + '"'],
)

# ----------------------------------------------------------------
# CPU トポロジ自動検出 (core/cpu_topology.hpp)
# ----------------------------------------------------------------
# 列挙のみ担当 (Win32 GetLogicalProcessorInformationEx / Linux sysfs best-effort)。
# アフィニティ設定は core/affinity.hpp を流用。roadmap L221 HW 環境抽象化の基盤。

# 構造的健全性テスト (純 C++・常時実行。Win32 不在環境は [SKIP] return 0)。
test_cpu_topology_exe = executable('test_cpu_topology',
  sources             : files('tests/test_cpu_topology.cpp'),
  include_directories : src_inc,
  dependencies        : deps,
  cpp_args            : cpp_args,
)
test('cpu_topology', test_cpu_topology_exe)

# CPU LM 推論 Tier 2 前提計測 (計測専用 exe・test 非登録・prof_bitnet と同じ扱い)。
# forward_fast を制御ピン留め (単一論理 / 物理スケール / HT 同居 / P vs E) で計時。
# マスクは cpu_topology.hpp の自動検出で決定 (決め打ちマスク排除)。
# WEIGHTS パスは prof_bitnet と同様に -D 埋め込み。重み不在時は健全に skip。
prof_cpu_topology_exe = executable('prof_cpu_topology',
  sources             : files('tests/prof_cpu_topology.cpp'),
  include_directories : src_inc,
  dependencies        : deps,
  cpp_args            : cpp_args + ['-DWEIGHTS_PATH="' + bn_weights_path + '"'],
)

# BitNet dense INT8 推論テスト (Phase 4 圧縮実験・重みのみ INT8)
# bitnet_dense_fp32 の FP32 射影 8 本を per-row INT8 量子化し、FP32 golden と
# 突合して劣化 (logits corr / 生成一致率) を数値化する。重み/golden は dense #6 と
# 同一ファイルを流用 (-D は bn_weights_path / bn_logits_path / bn_gen_path 再利用)。
test_bitnet_int8_exe = executable('test_bitnet_int8',
  sources             : files('tests/test_bitnet_int8.cpp'),
  include_directories : src_inc,
  dependencies        : deps,
  cpp_args            : cpp_args + [
    '-DWEIGHTS_PATH="' + bn_weights_path + '"',
    '-DLOGITS_GOLDEN_PATH="' + bn_logits_path + '"',
    '-DGEN_GOLDEN_PATH="' + bn_gen_path + '"',
  ],
)
test('bitnet_int8', test_bitnet_int8_exe, timeout : 600)

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

  # BitNet dense (FP32) GPU 推論テスト (Phase 4 #6-GPU / T2)
  # class BitNetGpuInfer (bitnet_gpu.cuh) + device forward 本体 (bitnet_gpu.cu) を
  # 同一重みで CPU 版 BitNetDenseInfer と突合する (CPU を golden 代わりに使うため
  # PyTorch golden は不要)。test 本体は .cpp だが BitNetGpuInfer が cudaMalloc/cublas を
  # 呼ぶため infer/bitnet_gpu.cu を同時リンクし cuda_dep (cudart/cublas) を依存に取る。
  # 重み (bitnet_dense{,_identity}_fp32.safetensors) / vocab.json は gitignore 対象。
  # 不在 or 例外 (GPU 不在含む) なら test 側で [SKIP] (CI でビルド緑維持)。
  test_bitnet_gpu_exe = executable('test_bitnet_gpu',
    sources             : files('tests/test_bitnet_gpu.cpp', 'infer/bitnet_gpu.cu'),
    include_directories : src_inc,
    dependencies        : deps,
    cpp_args            : cpp_args + [
      '-DWEIGHTS_PATH="' + bn_weights_path + '"',
      '-DIDENTITY_WEIGHTS_PATH="' + bn_id_weights_path + '"',
      '-DVOCAB_PATH="' + bn_vocab_path + '"',
    ],
    cuda_args           : cuda_test_args,
  )
  # CPU 版を golden 代わりに greedy 生成を複数回回すため重い (CPU naive forward x ステップ)。
  test('bitnet_gpu', test_bitnet_gpu_exe, timeout : 600)
endif

# ----------------------------------------------------------------
# Txt2ImgGenerator (本物の text->image) エンドツーエンドテスト (Phase 2-6b Stage F)
# ----------------------------------------------------------------
# txt2img_generator.hpp は OV 依存 (HAVE_OPENVINO ガード越し) かつ拡散ループを
# 純 cpp interface IDiffusionRunner 越しに呼ぶ。よって test 本体は .cpp (cl) でビルドし、
# runner の実体だけ CUDA 有無で切り替えてリンクする:
#   - CUDA 有効: diffusion_runner.cu + 全カーネル .cu を一緒にリンク (nvcc 経由)。
#   - CUDA 無効: diffusion_runner_stub.cpp をリンク (test は HAVE_CUDA 未定義で [SKIP])。
# アセット (OV tokenizer/encoder L/G + tokenizers.dll + unet/vae 重み + embeds) は
# -D 埋め込み (cwd 非依存)。不在なら test 側で [SKIP] (CI ビルド緑維持)。
# tokenizers.dll は test_text_conditioner と同じ dollma_find_ov_tokenizers_dll.py 検出を流用。
txt2img_unet_w  = meson.project_source_root() / 'src' / 'tests' / 'data' / 'unet_weights.safetensors'
txt2img_vae_w   = meson.project_source_root() / 'src' / 'tests' / 'data' / 'vae_weights.safetensors'
txt2img_embeds  = meson.project_source_root() / 'src' / 'tests' / 'data' / 'unet_io.safetensors'
txt2img_defines = [
  '-DTOKENIZER_L_XML="' + sdxl_tok_l_xml + '"',
  '-DTOKENIZER_G_XML="' + sdxl_tok_g_xml + '"',
  '-DENCODER_L_XML="' + sdxl_enc_l_xml + '"',
  '-DENCODER_G_XML="' + sdxl_enc_g_xml + '"',
  '-DTOKENIZERS_DLL="' + ov_tok_dll + '"',
  '-DUNET_WEIGHTS_PATH="' + txt2img_unet_w + '"',
  '-DVAE_WEIGHTS_PATH="' + txt2img_vae_w + '"',
  '-DEMBEDS_PATH="' + txt2img_embeds + '"',
]
if cuda_enabled
  test_txt2img_exe = executable('test_txt2img',
    sources             : files(['tests/test_txt2img.cpp', 'server/diffusion_runner.cu'] + diffusion_cu_sources),
    include_directories : src_inc,
    dependencies        : deps,
    cpp_args            : cpp_args + txt2img_defines,
    cuda_args           : cuda_args,
  )
else
  test_txt2img_exe = executable('test_txt2img',
    sources             : files('tests/test_txt2img.cpp', 'server/diffusion_runner_stub.cpp'),
    include_directories : src_inc,
    dependencies        : deps,
    cpp_args            : cpp_args + txt2img_defines,
  )
endif
# 20step 実走 (UNet 5.1GB 常駐 + OV ロード) を見込み大きめ timeout。test_diffusion と同等。
test('txt2img', test_txt2img_exe, timeout : 600, is_parallel : false)

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

# ----------------------------------------------------------------
# CLI 生成モード (build_image_generator / write_png_file) テスト (純 cpp)
# ----------------------------------------------------------------
# cli_generate.hpp は make_pipeline_generator (ファクトリ) を呼び、CUDA+OV 双方有効時のみ
# Txt2ImgGenerator (→ make_diffusion_runner) を参照する。よって test 本体は .cpp (cl) で
# ビルドし、ファクトリ / runner の実体だけ CUDA 有無で切り替えてリンクする (exe と同じ DI 設計):
#   - CUDA 有効: pipeline_generator_factory.cu + diffusion_runner.cu + 全カーネル .cu をリンク。
#   - CUDA 無効: pipeline_generator_factory_stub.cpp + diffusion_runner_stub.cpp をリンク。
# 重み/OV アセットは存在しない (= 解決されない) 想定で、build_image_generator は実行時に
# 必ず StubGenerator にフォールバックする (NPU/CUDA/OV に依存せず CI 緑)。
# M-6: cli_generate.hpp (build_image_generator) が make_matter を呼ぶため、matter 実体を
#   リンクする。OV 有無で matter_runner(_stub) を切り替える (exe と同じ DI 設計)。
cli_matter_src = openvino_dep.found() ? 'server/matter_runner.cpp' : 'server/matter_runner_stub.cpp'
if cuda_enabled
  test_cli_generate_exe = executable('test_cli_generate',
    sources             : files(['tests/test_cli_generate.cpp',
                                 'server/pipeline_generator_factory.cu',
                                 'server/diffusion_runner.cu',
                                 cli_matter_src] + diffusion_cu_sources),
    include_directories : src_inc,
    dependencies        : deps,
    cpp_args            : cpp_args,
    cuda_args           : cuda_args,
  )
else
  test_cli_generate_exe = executable('test_cli_generate',
    sources             : files('tests/test_cli_generate.cpp',
                                'server/pipeline_generator_factory_stub.cpp',
                                'server/diffusion_runner_stub.cpp',
                                cli_matter_src),
    include_directories : src_inc,
    dependencies        : deps,
    cpp_args            : cpp_args,
  )
endif
test('cli_generate', test_cli_generate_exe, timeout : 120)

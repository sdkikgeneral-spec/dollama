"""
dollma_probe7.py - 小型モデル NPU コンパイル互換性検証
=======================================================

目的: Phi-3 mini (1984MB INT4) が NPU 失敗した原因がモデルサイズか
      VPUX コンパイラのバグかを切り分ける。

確認項目:
  STEP 1 : モデルディレクトリ確認
  STEP 2 : NPU コンパイル試行 (openvino_genai.LLMPipeline)
  STEP 3 : NPU 成功時: テキスト生成レイテンシ計測
  STEP 4 : CPU ベースライン計測 (比較用)

使い方:
  python dollma_probe7.py tinyllama    # TinyLlama-1.1B
  python dollma_probe7.py qwen2        # Qwen2-1.5B (デフォルト)
"""

import time
import sys
from pathlib import Path

MODEL_CHOICE = sys.argv[1] if len(sys.argv) > 1 else "qwen2"

MODEL_DIRS = {
    "tinyllama": Path("models/tinyllama-1.1b-int4-npu"),
    "qwen2":     Path("models/qwen2-1.5b-int4-npu"),
}

MODEL_NAMES = {
    "tinyllama": "TinyLlama-1.1B-Chat INT4",
    "qwen2":     "Qwen2-1.5B-Instruct INT4",
}

if MODEL_CHOICE not in MODEL_DIRS:
    print("使い方: python dollma_probe7.py [tinyllama|qwen2]")
    sys.exit(1)

MODEL_DIR  = MODEL_DIRS[MODEL_CHOICE]
MODEL_NAME = MODEL_NAMES[MODEL_CHOICE]

TEST_PROMPTS = [
    "こんにちは",
    "1girl, magical girl, twin tails, sunset",
    "Generate a Stable Diffusion prompt for an anime girl with silver hair",
]

print(f"dollama probe7: {MODEL_NAME} NPU コンパイル検証")
print(f"環境: Intel Core Ultra 9 285 (NPU) + RTX5080")
print(f"モデル: {MODEL_DIR}")
print()


# ============================================================
# STEP 1: モデルディレクトリ確認
# ============================================================
def check_model_dir() -> bool:
    print("=" * 60)
    print("STEP 1: モデルディレクトリ確認")
    print("=" * 60)

    if not MODEL_DIR.exists():
        print(f"❌ {MODEL_DIR} が存在しません")
        print(f"   先に実行: python dollma_convert2.py {MODEL_CHOICE}")
        return False

    files = list(MODEL_DIR.iterdir())
    total_mb = sum(f.stat().st_size for f in files) / 1024 / 1024
    print(f"ディレクトリ: {MODEL_DIR}  (合計 {total_mb:.0f} MB)")
    for f in sorted(files):
        mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.name:<50} {mb:>8.1f} MB")

    has_ov_model = any(f.name.startswith("openvino_model") for f in files)
    print(f"\nopenvino_model*.xml/bin: {'✅' if has_ov_model else '❌'}")

    if not has_ov_model:
        print(f"\n❌ 変換未完了。先に: python dollma_convert2.py {MODEL_CHOICE}")
        return False
    return True


# ============================================================
# STEP 2+3: LLMPipeline (NPU) コンパイル + 推論計測
# ============================================================
def bench_npu() -> bool:
    print("\n" + "=" * 60)
    print("STEP 2+3: LLMPipeline (NPU) コンパイル + 推論計測")
    print("=" * 60)

    try:
        import openvino_genai as ov_genai
    except ImportError:
        print("❌ openvino_genai が見つかりません: pip install openvino-genai")
        return False

    print("NPU にロード中 (コンパイルに数分かかる場合があります)...")
    t_load = time.perf_counter()
    try:
        pipe = ov_genai.LLMPipeline(str(MODEL_DIR), "NPU")
        load_s = time.perf_counter() - t_load
        print(f"✅ NPU コンパイル成功！ロード時間: {load_s:.1f}s")
    except Exception as e:
        load_s = time.perf_counter() - t_load
        print(f"❌ NPU コンパイル失敗 ({load_s:.1f}s)")
        print(f"   エラー: {e}")
        return False

    config = ov_genai.GenerationConfig()
    config.max_new_tokens = 60
    config.do_sample = False

    print("\n推論計測:")
    for prompt in TEST_PROMPTS:
        print(f"\nプロンプト: \"{prompt}\"")
        t0 = time.perf_counter()
        result = pipe.generate(prompt, config)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        output = result.texts[0] if hasattr(result, "texts") else str(result)
        approx_tps = len(output) // 4 / (elapsed_ms / 1000)
        print(f"  出力: {output[:80].replace(chr(10), ' ')}...")
        print(f"  時間: {elapsed_ms:.0f}ms  推定 tok/s: {approx_tps:.1f}")

    return True


# ============================================================
# STEP 4: LLMPipeline (CPU) ベースライン
# ============================================================
def bench_cpu():
    print("\n" + "=" * 60)
    print("STEP 4: LLMPipeline (CPU) ベースライン")
    print("=" * 60)

    try:
        import openvino_genai as ov_genai
    except ImportError:
        print("❌ openvino_genai が見つかりません")
        return

    print("CPU にロード中...")
    t_load = time.perf_counter()
    try:
        pipe = ov_genai.LLMPipeline(str(MODEL_DIR), "CPU")
        load_s = time.perf_counter() - t_load
        print(f"ロード時間: {load_s:.1f}s")
    except Exception as e:
        print(f"❌ CPU ロード失敗: {e}")
        return

    config = ov_genai.GenerationConfig()
    config.max_new_tokens = 60
    config.do_sample = False

    for prompt in TEST_PROMPTS:
        print(f"\nプロンプト: \"{prompt}\"")
        t0 = time.perf_counter()
        result = pipe.generate(prompt, config)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        output = result.texts[0] if hasattr(result, "texts") else str(result)
        approx_tps = len(output) // 4 / (elapsed_ms / 1000)
        print(f"  出力: {output[:80].replace(chr(10), ' ')}...")
        print(f"  時間: {elapsed_ms:.0f}ms  推定 tok/s: {approx_tps:.1f}")


# ============================================================
# メイン
# ============================================================
if __name__ == "__main__":
    ok = check_model_dir()
    if not ok:
        sys.exit(1)

    npu_ok = bench_npu()
    bench_cpu()

    print("\n" + "=" * 60)
    print("probe7 完了")
    print("=" * 60)
    if npu_ok:
        print("✅ NPU 推論成功 → このモデルサイズは NPU 対応")
        print("   次: より大きいモデルで上限を探るか、dollama パイプラインに統合")
    else:
        print("❌ NPU 推論失敗 → VPUX コンパイラのバグ or NPU メモリ不足")
        print("   次: TinyLlama (最小) も失敗するなら OpenVINO バージョン問題の可能性")
        print("   参考: https://github.com/openvinotoolkit/openvino/issues")

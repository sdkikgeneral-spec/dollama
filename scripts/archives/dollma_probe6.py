"""
dollma_probe6.py - Phi-3 mini INT4 on NPU + OV Tokenizer 検証
=============================================================

確認項目:
  STEP 1 : モデルディレクトリの構成確認
  STEP 2 : OV Tokenizer デバイスと速度確認 (openvino_tokenizer.xml がある場合)
  STEP 3 : LLMPipeline (NPU) でテキスト生成・レイテンシ計測
  STEP 4 : LLMPipeline (CPU) と速度比較
"""

import time
import os
from pathlib import Path

MODEL_DIR = Path("models/phi3-mini-int4-npu")

TEST_PROMPTS = [
    "こんにちは",
    "1girl, magical girl, twin tails, sunset",
    "Generate a Stable Diffusion prompt for an anime girl with silver hair",
]


# ============================================================
# STEP 1: モデルディレクトリ構成確認
# ============================================================
def check_model_dir():
    print("=" * 60)
    print("STEP 1: モデルディレクトリ構成確認")
    print("=" * 60)

    if not MODEL_DIR.exists():
        print(f"❌ {MODEL_DIR} が存在しません")
        return False

    files = list(MODEL_DIR.iterdir())
    print(f"ディレクトリ: {MODEL_DIR}")
    for f in sorted(files):
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.name:<50} {size_mb:>8.1f} MB")

    has_ov_model     = any(f.name.startswith("openvino_model") for f in files)
    has_ov_tokenizer = (MODEL_DIR / "openvino_tokenizer.xml").exists()
    has_hf_tokenizer = (MODEL_DIR / "tokenizer.json").exists()

    print()
    print(f"openvino_model*.xml/bin  : {'✅' if has_ov_model else '❌'}")
    print(f"openvino_tokenizer.xml   : {'✅' if has_ov_tokenizer else '❌ (HF tokenizer.json で代替可能)'}")
    print(f"tokenizer.json (HF)      : {'✅' if has_hf_tokenizer else '❌'}")

    if not has_ov_model:
        print("\n❌ openvino_model.xml が見つかりません。dollma_convert.py を実行してください。")
        return False

    # OV Tokenizer がない場合は変換を試みる
    if not has_ov_tokenizer and has_hf_tokenizer:
        print("\nOV Tokenizer を生成中...")
        try:
            from openvino_tokenizers import convert_tokenizer
            from transformers import AutoTokenizer
            import openvino as ov
            hf_tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
            ov_tok, ov_detok = convert_tokenizer(hf_tok, with_detokenizer=True)
            ov.save_model(ov_tok,   MODEL_DIR / "openvino_tokenizer.xml")
            ov.save_model(ov_detok, MODEL_DIR / "openvino_detokenizer.xml")
            print("  ✅ openvino_tokenizer.xml 生成完了")
            has_ov_tokenizer = True
        except Exception as e:
            print(f"  ⚠ 変換失敗 ({e}) → LLMPipeline は HF tokenizer で試みる")

    return has_ov_model


# ============================================================
# STEP 2: OV Tokenizer のデバイスと速度確認
# ============================================================
def check_tokenizer_device():
    print("\n" + "=" * 60)
    print("STEP 2: OV Tokenizer デバイスと速度確認")
    print("=" * 60)

    tok_path = MODEL_DIR / "openvino_tokenizer.xml"
    if not tok_path.exists():
        print("⚠ openvino_tokenizer.xml なし → STEP 2 スキップ")
        return

    try:
        import openvino as ov
        import openvino_tokenizers  # カスタム ops を core に登録
        import numpy as np
        core = ov.Core()

        tokenizer_model = core.read_model(tok_path)
        test_text = "Generate an anime girl with silver hair"
        input_data = np.array([test_text])

        print(f"テストテキスト: \"{test_text}\"")
        print()

        for device in ["CPU", "GPU.0", "NPU"]:
            try:
                compiled = core.compile_model(tokenizer_model, device)
                req = compiled.create_infer_request()

                for _ in range(3):
                    req.infer({0: input_data})

                N = 50
                times = []
                for _ in range(N):
                    t0 = time.perf_counter()
                    result = req.infer({0: input_data})
                    times.append((time.perf_counter() - t0) * 1000)

                ms = float(np.median(times))
                token_ids = list(result[0].flatten())
                print(f"  {device:<8}: {ms:.3f}ms  →  {len(token_ids)} tokens: {token_ids[:10]}...")

            except Exception as e:
                print(f"  {device:<8}: NG ({e})")

    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()


# ============================================================
# STEP 3/4: LLMPipeline でテキスト生成・レイテンシ計測
# ============================================================
def bench_llm_pipeline(device: str):
    step = "3" if device == "NPU" else "4"
    print(f"\n" + "=" * 60)
    print(f"STEP {step}: LLMPipeline ({device}) テキスト生成計測")
    print("=" * 60)

    try:
        import openvino_genai as ov_genai

        print(f"{device} にロード中 (初回は時間がかかります)...")
        t_load = time.perf_counter()
        pipe = ov_genai.LLMPipeline(str(MODEL_DIR), device)
        load_ms = (time.perf_counter() - t_load) * 1000
        print(f"ロード時間: {load_ms/1000:.1f}s")

        config = ov_genai.GenerationConfig()
        config.max_new_tokens = 60
        config.do_sample = False

        for prompt in TEST_PROMPTS:
            print(f"\nプロンプト: \"{prompt}\"")

            t0 = time.perf_counter()
            result = pipe.generate(prompt, config)
            elapsed = (time.perf_counter() - t0) * 1000

            output = result.texts[0] if hasattr(result, 'texts') else str(result)
            output_preview = output[:80].replace('\n', ' ')

            approx_tokens = len(output) // 4
            tps = approx_tokens / (elapsed / 1000) if elapsed > 0 else 0

            print(f"  出力: {output_preview}...")
            print(f"  時間: {elapsed:.0f}ms  推定 tok/s: {tps:.1f}")

        return True

    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================
# メイン
# ============================================================
if __name__ == "__main__":
    print("dollama probe6: Phi-3 mini INT4 on NPU + OV Tokenizer")
    print("環境: Intel Core Ultra 9 285 (NPU + Xe iGPU) + RTX5080")
    print()

    ok = check_model_dir()
    if not ok:
        print("\nモデルを変換してから再実行してください。")
        exit(1)

    check_tokenizer_device()
    bench_llm_pipeline("NPU")
    bench_llm_pipeline("CPU")

    print("\n" + "=" * 60)
    print("probe6 完了")
    print("=" * 60)
    print("""
見方:
  STEP2: トークナイザーが NPU/iGPU で動けば完全ローカルパイプライン
         CPU のみなら LLMPipeline 内部で自動 CPU フォールバック
  STEP3: NPU tok/s が dollama の要件 (数十 tok/s) を満たすか確認
  STEP4: CPU と比較して NPU の優位性を確認
""")

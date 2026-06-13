"""
dollma_probe7b.py - NPU で OVModelForCausalLM (genai なし) を試す
===================================================================

仮説: openvino_genai.LLMPipeline が KV-cache / paged attention を追加した結果
     VPUX コンパイラの StopLocationVerifierPass が重複名を検出している。

     LLMPipeline を使わず OVModelForCausalLM を NPU に直接ロードして
     問題がどこにあるか切り分ける。

確認項目:
  STEP 1 : OVModelForCausalLM NPU ロード (with_past なし、静的形状)
  STEP 2 : OVModelForCausalLM CPU ロード (比較)
  STEP 3 : モデル本体 openvino_model.xml だけ compile_model で試す
"""

import time
from pathlib import Path

MODEL_DIR = Path("models/qwen2-1.5b-int4-npu")

print("dollama probe7b: OVModelForCausalLM NPU 直接ロード検証")
print(f"モデル: {MODEL_DIR}")
print()


# ============================================================
# STEP 1: OVModelForCausalLM を NPU でロード
# ============================================================
print("=" * 60)
print("STEP 1: OVModelForCausalLM (NPU)")
print("=" * 60)

try:
    from optimum.intel import OVModelForCausalLM
    from transformers import AutoTokenizer

    # Python 3.14 partial パッチ
    try:
        from optimum.utils.normalized_config import NormalizedTextConfigWithGQA
        from optimum.exporters.openvino.model_configs import Phi3OpenVINOConfig, PhiMoEOpenVINOConfig
        Phi3OpenVINOConfig.NORMALIZED_CONFIG_CLASS = NormalizedTextConfigWithGQA
        PhiMoEOpenVINOConfig.NORMALIZED_CONFIG_CLASS = NormalizedTextConfigWithGQA
    except Exception:
        pass

    print("NPU にロード中 (use_cache=False で静的形状を試みる)...")
    t0 = time.perf_counter()
    model = OVModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        device="NPU",
        use_cache=False,      # KV-cache を無効化 → 動的形状を避ける
        compile=False,        # まずロードだけ確認
    )
    print(f"  モデル構築: {(time.perf_counter()-t0)*1000:.0f}ms")

    print("  compile() 実行中...")
    t0 = time.perf_counter()
    model.compile()
    print(f"  ✅ NPU compile 成功: {(time.perf_counter()-t0)*1000:.0f}ms")

    print("  テキスト生成中...")
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    inputs = tok("こんにちは", return_tensors="pt")
    t0 = time.perf_counter()
    outputs = model.generate(**inputs, max_new_tokens=20)
    elapsed = (time.perf_counter() - t0) * 1000
    text = tok.decode(outputs[0], skip_special_tokens=True)
    print(f"  出力: {text}")
    print(f"  時間: {elapsed:.0f}ms")

except Exception as e:
    print(f"  ❌ NPU ロード失敗: {e}")


# ============================================================
# STEP 2: OVModelForCausalLM を CPU でロード (比較)
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: OVModelForCausalLM (CPU, use_cache=False)")
print("=" * 60)

try:
    from optimum.intel import OVModelForCausalLM
    from transformers import AutoTokenizer

    print("CPU にロード中...")
    t0 = time.perf_counter()
    model = OVModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        device="CPU",
        use_cache=False,
    )
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"  ロード時間: {load_ms:.0f}ms")

    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    inputs = tok("こんにちは", return_tensors="pt")
    t0 = time.perf_counter()
    outputs = model.generate(**inputs, max_new_tokens=20)
    elapsed = (time.perf_counter() - t0) * 1000
    text = tok.decode(outputs[0], skip_special_tokens=True)
    print(f"  出力: {text}")
    print(f"  時間: {elapsed:.0f}ms")

except Exception as e:
    print(f"  ❌ CPU ロード失敗: {e}")


# ============================================================
# STEP 3: compile_model で openvino_model.xml を NPU に直接コンパイル
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: core.compile_model (NPU) — openvino_model.xml 直接")
print("=" * 60)

try:
    import openvino as ov
    core = ov.Core()

    xml_path = MODEL_DIR / "openvino_model.xml"
    print(f"モデル読み込み中: {xml_path}")
    t0 = time.perf_counter()
    ov_model = core.read_model(xml_path)
    read_ms = (time.perf_counter() - t0) * 1000
    print(f"  read_model: {read_ms:.0f}ms")
    print(f"  入力: {[(i.get_any_name(), i.shape) for i in ov_model.inputs]}")

    print("NPU に compile_model 中...")
    t0 = time.perf_counter()
    compiled = core.compile_model(ov_model, "NPU")
    compile_ms = (time.perf_counter() - t0) * 1000
    print(f"  ✅ compile_model 成功: {compile_ms:.0f}ms")

except Exception as e:
    print(f"  ❌ compile_model 失敗: {e}")

print("\n" + "=" * 60)
print("probe7b 完了")
print("=" * 60)

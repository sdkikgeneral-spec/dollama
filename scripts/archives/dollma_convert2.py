"""
dollma_convert2.py - 小型モデルを OpenVINO IR に変換する (NPU 互換性検証用)

目的: Phi-3 mini (1984MB) が NPU コンパイル失敗するため、
      小型モデルで VPUX コンパイラの上限を探る。

対象モデル:
  TinyLlama-1.1B-Chat  ~550MB (INT4)
  Qwen2-1.5B-Instruct  ~800MB (INT4)  ← デフォルト
"""
from pathlib import Path
import sys

# 変換するモデルをコマンドライン引数で切り替え可能
MODEL_CHOICE = sys.argv[1] if len(sys.argv) > 1 else "qwen2"

MODELS = {
    "tinyllama": {
        "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "output":   Path("models/tinyllama-1.1b-int4-npu"),
    },
    "qwen2": {
        "model_id": "Qwen/Qwen2-1.5B-Instruct",
        "output":   Path("models/qwen2-1.5b-int4-npu"),
    },
}

if MODEL_CHOICE not in MODELS:
    print(f"使い方: python dollma_convert2.py [tinyllama|qwen2]")
    sys.exit(1)

MODEL_ID = MODELS[MODEL_CHOICE]["model_id"]
OUTPUT   = MODELS[MODEL_CHOICE]["output"]

print(f"変換開始: {MODEL_ID} → {OUTPUT}")

from optimum.intel import OVModelForCausalLM, OVWeightQuantizationConfig
from transformers import AutoTokenizer

# Python 3.14 functools.partial descriptor 問題のパッチ
# (Qwen2 / TinyLlama は Phi3 config ではないが、念のため Phi3 系だけパッチ)
try:
    from optimum.utils.normalized_config import NormalizedTextConfigWithGQA
    from optimum.exporters.openvino.model_configs import Phi3OpenVINOConfig, PhiMoEOpenVINOConfig
    Phi3OpenVINOConfig.NORMALIZED_CONFIG_CLASS = NormalizedTextConfigWithGQA
    PhiMoEOpenVINOConfig.NORMALIZED_CONFIG_CLASS = NormalizedTextConfigWithGQA
except Exception as e:
    print(f"  Phi3 パッチ不要 or 失敗 (無視): {e}")

OUTPUT.mkdir(parents=True, exist_ok=True)

print("INT4 量子化設定...")
q_config = OVWeightQuantizationConfig(bits=4, ratio=1.0)

print("モデルをダウンロード・変換中 (数分かかります)...")
model = OVModelForCausalLM.from_pretrained(
    MODEL_ID,
    export=True,
    quantization_config=q_config,
)

print("保存中...")
model.save_pretrained(OUTPUT)

print("トークナイザー保存中...")
tok = AutoTokenizer.from_pretrained(MODEL_ID)
tok.save_pretrained(OUTPUT)

print("OpenVINO トークナイザー変換中...")
try:
    from openvino_tokenizers import convert_tokenizer
    import openvino as ov
    ov_tok, ov_detok = convert_tokenizer(tok, with_detokenizer=True)
    ov.save_model(ov_tok,   OUTPUT / "openvino_tokenizer.xml")
    ov.save_model(ov_detok, OUTPUT / "openvino_detokenizer.xml")
    print("  openvino_tokenizer.xml / openvino_detokenizer.xml 生成完了")
except Exception as e:
    print(f"  OV tokenizer 変換スキップ: {e}")

print(f"\n完了: {OUTPUT}")
import os
total_mb = 0
for f in sorted(OUTPUT.iterdir()):
    mb = f.stat().st_size / 1024 / 1024
    total_mb += mb
    print(f"  {f.name:<50} {mb:>8.1f} MB")
print(f"  {'合計':<50} {total_mb:>8.1f} MB")

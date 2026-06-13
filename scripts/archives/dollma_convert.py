"""
dollma_convert.py - Phi-3 mini INT4 を OpenVINO IR に変換する
"""
from pathlib import Path

MODEL_ID  = "microsoft/Phi-3-mini-4k-instruct"
OUTPUT    = Path("models/phi3-mini-int4-npu")

print(f"変換開始: {MODEL_ID} → {OUTPUT}")

from optimum.intel import OVModelForCausalLM, OVWeightQuantizationConfig
from transformers import AutoTokenizer

# Python 3.14 では functools.partial が descriptor protocol (__get__) を実装した。
# クラス属性に置くとインスタンスアクセス時に self がバインドされる。
# optimum の with_args() が返す partial がこれに該当するためパッチが必要。
from optimum.utils.normalized_config import NormalizedTextConfigWithGQA
from optimum.exporters.openvino.model_configs import Phi3OpenVINOConfig, PhiMoEOpenVINOConfig
Phi3OpenVINOConfig.NORMALIZED_CONFIG_CLASS = NormalizedTextConfigWithGQA
PhiMoEOpenVINOConfig.NORMALIZED_CONFIG_CLASS = NormalizedTextConfigWithGQA

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
for f in sorted(OUTPUT.iterdir()):
    print(f"  {f.name}  ({f.stat().st_size/1024/1024:.1f} MB)")

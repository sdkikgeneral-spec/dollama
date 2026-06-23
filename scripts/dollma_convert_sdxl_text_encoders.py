"""
dollma_convert_sdxl_text_encoders.py
====================================
タスク 2-6b Stage B: SDXL の dual text encoder + 両トークナイザを OpenVINO IR 化する。

SDXL のテキスト条件付けレシピ (diffusers pipeline_stable_diffusion_xl.py 準拠):
  - text_encoder   (CLIP-L,    openai/clip-vit-large-patch14 相当, hidden 768, 12層):
      penultimate = hidden_states[-2]  [1,77,768]   (final_layer_norm は通さない)
  - text_encoder_2 (CLIP-bigG, laion CLIP-ViT-bigG,  hidden 1280, 32層):
      penultimate = hidden_states[-2]  [1,77,1280]
      pooled      = text_embeds        [1,1280]      (EOS 位置 × text_projection)

  diffusers は「最後のエンコーダ (=text_encoder_2) の pooled output のみ」を採用する。
  CLIP-L 側の pooled は使わない (penultimate のみ)。

出力先:
  models/sdxl-text-encoder-l/model_ov.{xml,bin}   (penultimate 768 のみ・1出力)
  models/sdxl-text-encoder-g/model_ov.{xml,bin}   (penultimate 1280 + pooled 1280・2出力)
  models/sdxl-tokenizer-l/   (openvino-tokenizers, CLIP-L 仕様: pad=eos=49407)
  models/sdxl-tokenizer-g/   (openvino-tokenizers, bigG 仕様: pad=0)

実重み: stabilityai/stable-diffusion-xl-base-1.0 subfolder=text_encoder / text_encoder_2

NPU 静的形状必須: compile 前に reshape({"input_ids":[1,77]})。
本スクリプトでも convert 直後に reshape して保存する。
"""

import sys
from pathlib import Path

import numpy as np
import torch
import openvino as ov

SDXL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
SEQ_LEN = 77
PROMPT = "1girl, solo"  # 突合用の固定プロンプト

MODELS = Path("models")
OUT_L = MODELS / "sdxl-text-encoder-l"
OUT_G = MODELS / "sdxl-text-encoder-g"
TOK_L = MODELS / "sdxl-tokenizer-l"
TOK_G = MODELS / "sdxl-tokenizer-g"


# ============================================================
# ラッパー: forward が SDXL の要る出力だけを返すよう固定する
# ============================================================
class TextEncoderLWrapper(torch.nn.Module):
    """CLIP-L: penultimate hidden_states[-2] [1,77,768] のみ返す。"""

    def __init__(self, text_encoder):
        super().__init__()
        self.te = text_encoder

    def forward(self, input_ids):
        out = self.te(input_ids=input_ids, output_hidden_states=True, return_dict=True)
        # hidden_states は (embeddings, layer1, ..., layerN) の N+1 要素。
        # [-2] が penultimate (final_layer_norm 適用前の最終 transformer 層出力)。
        return out.hidden_states[-2]


class TextEncoderGWrapper(torch.nn.Module):
    """CLIP-bigG: (penultimate [1,77,1280], pooled [1,1280]) を返す。

    pooled は CLIPTextModelWithProjection の text_embeds と同一:
      EOS トークン位置の hidden を取り text_projection を掛けたもの。
    diffusers は prompt_embeds[0] (=text_embeds, ndim==2) を pooled として採用する。
    """

    def __init__(self, text_encoder_2):
        super().__init__()
        self.te = text_encoder_2

    def forward(self, input_ids):
        out = self.te(input_ids=input_ids, output_hidden_states=True, return_dict=True)
        penultimate = out.hidden_states[-2]   # [1,77,1280]
        pooled = out.text_embeds              # [1,1280]
        return penultimate, pooled


# ============================================================
# OV 変換 + 静的 reshape + 保存
# ============================================================
def convert_encoder(wrapper, out_dir, n_outputs, label, output_names):
    print(f"\n{'='*60}\n{label} を OV IR に変換\n{'='*60}")
    out_dir.mkdir(parents=True, exist_ok=True)
    wrapper.eval()

    dummy = torch.zeros(1, SEQ_LEN, dtype=torch.long)
    with torch.no_grad():
        ov_model = ov.convert_model(wrapper, example_input=(dummy,))

    # 入力名を input_ids に正規化 (convert_model はデフォルトで引数名を拾うが念のため固定)
    ov_model.inputs[0].get_tensor().set_names({"input_ids"})

    # 出力名を明示設定 (convert_model は名前を付けないことがある → C++ から参照不能になる)
    for o, nm in zip(ov_model.outputs, output_names):
        o.get_tensor().set_names({nm})

    # NPU 必須: 静的形状に固定
    ov_model.reshape({"input_ids": [1, SEQ_LEN]})

    print(f"  入力:")
    for i in ov_model.inputs:
        print(f"    name={i.get_any_name()!r} shape={i.shape} dtype={i.get_element_type()}")
    print(f"  出力:")
    for idx, o in enumerate(ov_model.outputs):
        print(f"    [{idx}] name={o.get_any_name()!r} shape={o.shape} dtype={o.get_element_type()}")
    assert len(ov_model.outputs) == n_outputs, \
        f"出力数 {len(ov_model.outputs)} != 期待 {n_outputs}"

    xml_path = out_dir / "model_ov.xml"
    ov.save_model(ov_model, xml_path)
    bin_mb = (out_dir / "model_ov.bin").stat().st_size / 1024 / 1024
    print(f"  -> {xml_path}  (bin {bin_mb:.1f} MB)")
    return ov_model


# ============================================================
# トークナイザ OV 化
# ============================================================
def convert_tokenizer(hf_tok, out_dir, label):
    from openvino_tokenizers import convert_tokenizer as ovt_convert
    print(f"\n{'='*60}\n{label} トークナイザを OV 化\n{'='*60}")
    out_dir.mkdir(parents=True, exist_ok=True)
    # CLIP は max_length=77 固定パディングが必須。
    # use_max_padding=True で max_length まで pad する (CLIP は 77 固定パディング必須)。
    # pad トークンは hf_tok.pad_token_id を継承する (CLIP-L=49407 / bigG=0)。
    ov_tok = ovt_convert(
        hf_tok,
        with_detokenizer=False,
        max_length=SEQ_LEN,
        use_max_padding=True,
    )
    xml_path = out_dir / "openvino_tokenizer.xml"
    ov.save_model(ov_tok, xml_path)
    print(f"  bos={hf_tok.bos_token_id} eos={hf_tok.eos_token_id} "
          f"pad={hf_tok.pad_token_id} model_max={hf_tok.model_max_length}")
    print(f"  -> {xml_path}")
    return xml_path


# ============================================================
# 検証: PyTorch penultimate/pooled vs OV
# ============================================================
def verify_encoder(ov_model, wrapper, input_ids_np, label, has_pooled):
    core = ov.Core()
    compiled = core.compile_model(ov_model, "CPU")
    req = compiled.create_infer_request()
    ov_out = req.infer({"input_ids": input_ids_np.astype(np.int64)})

    with torch.no_grad():
        ref = wrapper(torch.from_numpy(input_ids_np).long())

    if has_pooled:
        ref_pen, ref_pool = ref
        ov_pen = ov_out[compiled.outputs[0]]
        ov_pool = ov_out[compiled.outputs[1]]
        e_pen = float(np.max(np.abs(ref_pen.numpy() - ov_pen)))
        e_pool = float(np.max(np.abs(ref_pool.numpy() - ov_pool)))
        r_pen = ref_pen.numpy()
        rel_pen = e_pen / float(np.max(np.abs(r_pen)))
        print(f"  [{label}] penultimate max_abs_err={e_pen:.2e}  (rel {rel_pen:.2e}, "
              f"range +-{np.abs(r_pen).max():.0f})  shape={ov_pen.shape}")
        print(f"  [{label}] pooled      max_abs_err={e_pool:.2e}  shape={ov_pool.shape}")
        return e_pen, e_pool
    else:
        ov_pen = ov_out[compiled.outputs[0]]
        r = ref.numpy()
        e_pen = float(np.max(np.abs(r - ov_pen)))
        rel_pen = e_pen / float(np.max(np.abs(r)))
        print(f"  [{label}] penultimate max_abs_err={e_pen:.2e}  (rel {rel_pen:.2e}, "
              f"range +-{np.abs(r).max():.0f})  shape={ov_pen.shape}")
        return e_pen, None


def verify_tokenizer(tok_xml, hf_tok, label):
    core = ov.Core()
    compiled = core.compile_model(core.read_model(tok_xml), "CPU")
    ov_out = compiled([PROMPT])
    ov_ids = ov_out["input_ids"].astype(np.int64)
    ref = hf_tok([PROMPT], padding="max_length", max_length=SEQ_LEN,
                 truncation=True, return_tensors="np")["input_ids"].astype(np.int64)
    match = bool(np.array_equal(ov_ids, ref))
    print(f"  [{label}] token ids 一致={match}  ov_shape={ov_ids.shape}")
    if not match:
        print(f"    ref={ref[0][:12].tolist()}")
        print(f"    ov ={ov_ids[0][:12].tolist()}")
    return ov_ids, ref, match


# ============================================================
# メイン
# ============================================================
def main():
    from transformers import (
        CLIPTextModel, CLIPTextModelWithProjection, AutoTokenizer,
    )

    print("SDXL text encoder ロード中...")
    # SDXL の重みは FP16 保存だが penultimate 値域が ±850 と広く、FP16 だと
    # 絶対誤差が ~1.0 まで膨らむ。FP32 に上げて保存する (NPU は内部 FP16 駆動でも
    # IR は FP32 が無難。精度差は D/C 担当が NPU 実測で再評価)。
    te_l = CLIPTextModel.from_pretrained(SDXL_ID, subfolder="text_encoder").float()
    te_g = CLIPTextModelWithProjection.from_pretrained(SDXL_ID, subfolder="text_encoder_2").float()
    tok_l = AutoTokenizer.from_pretrained(SDXL_ID, subfolder="tokenizer")
    tok_g = AutoTokenizer.from_pretrained(SDXL_ID, subfolder="tokenizer_2")

    wrap_l = TextEncoderLWrapper(te_l)
    wrap_g = TextEncoderGWrapper(te_g)

    # --- 変換 ---
    ov_l = convert_encoder(wrap_l, OUT_L, n_outputs=1, label="CLIP-L (text_encoder)",
                           output_names=["penultimate"])
    ov_g = convert_encoder(wrap_g, OUT_G, n_outputs=2, label="CLIP-bigG (text_encoder_2)",
                           output_names=["penultimate", "pooled"])
    tok_l_xml = convert_tokenizer(tok_l, TOK_L, "CLIP-L")
    tok_g_xml = convert_tokenizer(tok_g, TOK_G, "bigG")

    # --- 検証 ---
    print(f"\n{'='*60}\n検証 (PROMPT={PROMPT!r})\n{'='*60}")
    ids_l = tok_l([PROMPT], padding="max_length", max_length=SEQ_LEN,
                  truncation=True, return_tensors="np")["input_ids"]
    ids_g = tok_g([PROMPT], padding="max_length", max_length=SEQ_LEN,
                  truncation=True, return_tensors="np")["input_ids"]

    print("\nトークナイザ突合:")
    verify_tokenizer(tok_l_xml, tok_l, "CLIP-L")
    verify_tokenizer(tok_g_xml, tok_g, "bigG")

    print("\nエンコーダ突合:")
    el, _ = verify_encoder(ov_l, wrap_l, ids_l, "CLIP-L", has_pooled=False)
    eg_pen, eg_pool = verify_encoder(ov_g, wrap_g, ids_g, "bigG", has_pooled=True)

    print(f"\n{'='*60}\nまとめ\n{'='*60}")
    print(f"CLIP-L  penultimate err = {el:.2e}")
    print(f"bigG    penultimate err = {eg_pen:.2e}")
    print(f"bigG    pooled      err = {eg_pool:.2e}")


if __name__ == "__main__":
    main()

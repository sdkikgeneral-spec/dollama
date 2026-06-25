# -*- coding: utf-8 -*-
"""
D1 調査: Qwen2-1.5B-Instruct を GTX1080Ti (torch cu121) でロード&生成実証。
蒸留での用途 (タグ列 → 自然な依頼文 ja/en) を疎通テストする。
教師には text だけ生成させる (タグは固定・改変させない)。
"""
import sys, time, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2-1.5B-Instruct"
DTYPE = torch.float16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"=== D1 Qwen2 probe ===")
print(f"device={DEVICE} torch={torch.__version__}")
if DEVICE == "cuda":
    print(f"gpu={torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability(0)}")
    print(f"vram total={torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")

t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)
model.eval()
load_s = time.time() - t0
nparams = sum(p.numel() for p in model.parameters())
print(f"loaded in {load_s:.1f}s  params={nparams/1e6:.0f}M")
if DEVICE == "cuda":
    print(f"VRAM allocated after load: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    print(f"VRAM reserved  after load: {torch.cuda.memory_reserved()/1024**3:.2f} GB")

# 蒸留プロンプト: タグ列を表す自然な画像生成依頼文を 1 文。タグは改変禁止を明示。
TAG_SAMPLES = [
    "1girl, long hair, blue eyes, school uniform, smile",
    "1girl, twintails, magical girl, frilled dress, holding wand, night sky",
    "2girls, maid, cat ears, indoors, looking at viewer, sitting",
    "1boy, white hair, hoodie, hands in pockets, city street, serious",
    "1girl, swimsuit, beach, wind, smile, long blonde hair",
]

def build_messages(tags, lang):
    if lang == "ja":
        sys_p = ("あなたは画像生成の依頼文を書くアシスタントです。"
                 "与えられた danbooru タグ列が表す情景を、自然な日本語の依頼文 1 文で書いてください。"
                 "タグを列挙したり英単語を並べたりせず、人間が画家に頼むような自然な文にしてください。"
                 "依頼文のみを出力し、説明や引用符は付けないでください。")
        usr_p = f"タグ列: {tags}"
    else:
        sys_p = ("You write image-generation requests. Given a list of danbooru tags, "
                 "write ONE natural English sentence requesting that picture, as a person would ask an artist. "
                 "Do not list the tags; paraphrase them naturally. Output only the request sentence, no quotes.")
        usr_p = f"Tags: {tags}"
    return [{"role": "system", "content": sys_p},
            {"role": "user", "content": usr_p}]

def generate(messages, max_new=64):
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok([text], return_tensors="pt").to(DEVICE)
    n_in = inputs.input_ids.shape[1]
    t = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=True,
                             temperature=0.8, top_p=0.9, pad_token_id=tok.eos_token_id)
    dt = time.time() - t
    gen_ids = out[0][n_in:]
    n_out = gen_ids.shape[0]
    txt = tok.decode(gen_ids, skip_special_tokens=True).strip()
    return txt, n_out, dt

print("\n=== 生成サンプル (text のみ・タグ固定) ===")
total_tok, total_t = 0, 0.0
for tags in TAG_SAMPLES:
    for lang in ("ja", "en"):
        txt, n_out, dt = generate(build_messages(tags, lang))
        total_tok += n_out; total_t += dt
        print(f"\n[{lang}] tags: {tags}")
        print(f"  -> {txt}")
        print(f"  ({n_out} tok / {dt:.2f}s = {n_out/dt:.1f} tok/s)")

print(f"\n=== 集計 ===")
print(f"平均生成速度: {total_tok/total_t:.1f} tok/s ({total_tok} tok / {total_t:.1f}s)")
if DEVICE == "cuda":
    print(f"VRAM peak: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB allocated / "
          f"{torch.cuda.max_memory_reserved()/1024**3:.2f} GB reserved")

# -*- coding: utf-8 -*-
"""D1: DanTagGen-delta-rev2 を実機ロード&生成。サイズ/VRAM/出力タグ vs dollama vocab 整合。"""
import sys, io, time, json, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import torch
from huggingface_hub import model_info
from transformers import AutoModelForCausalLM, AutoTokenizer

MID = "KBlueLeaf/DanTagGen-delta-rev2"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

info = model_info(MID, files_metadata=True)
for f in (info.siblings or []):
    if f.rfilename.endswith((".safetensors", ".json", ".gguf")) or "model" in f.rfilename:
        print(f"  {f.rfilename}: {(f.size or 0)/1e6:.1f} MB")

print(f"\nloading {MID} on {DEV} ...")
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float16).to(DEV).eval()
print(f"loaded {time.time()-t0:.1f}s")
n = sum(p.numel() for p in model.parameters())
print(f"params = {n/1e6:.1f}M   config arch = {model.config.architectures}")
if DEV == "cuda":
    print(f"VRAM alloc = {torch.cuda.memory_allocated()/1024**3:.2f} GB")

# DanTagGen のプロンプト形式 (delta 系) を試す
PROMPT = """quality: masterpiece
rating: safe
artist: <|empty|>
characters: <|empty|>
copyrights: <|empty|>
aspect ratio: 1.0
target: <|short|>
general: 1girl, solo, long hair<|input_end|>
"""
inputs = tok(PROMPT, return_tensors="pt").to(DEV)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=1.0,
                         top_p=0.95, pad_token_id=tok.eos_token_id)
gen = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
print("\n=== DanTagGen 生成 (general タグ拡張) ===")
print(gen)

# dollama vocab 整合: 生成タグ列のうち dollama 4994 vocab に入る割合
voc = json.load(open("data/bitnet/vocab.json", encoding="utf-8"))
vocset = {t["tag"] if isinstance(t, dict) else t for t in voc["tags"]}
# general: 行以降のタグを抽出
gentags = []
g = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
for part in (g.replace("\n", ",").split(",")):
    s = part.strip()
    if s and "<|" not in s and ":" not in s:
        gentags.append(s)
gentags = [t for t in gentags if t]
inv = [t for t in gentags if t in vocset]
print(f"\n生成タグ {len(gentags)} 個中 dollama vocab(4994) に存在: {len(inv)} "
      f"({100*len(inv)/max(1,len(gentags)):.0f}%)")
print("vocab内:", inv[:20])
print("vocab外:", [t for t in gentags if t not in vocset][:20])

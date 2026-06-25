# -*- coding: utf-8 -*-
"""D1: DanTagGen 出力タグ vs dollama vocab(4994) の重なりを複数サンプルで集計。"""
import sys, io, json, torch
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from transformers import AutoModelForCausalLM, AutoTokenizer

MID = "KBlueLeaf/DanTagGen-delta-rev2"
DEV = "cuda"
tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float16).to(DEV).eval()

voc = json.load(open("data/bitnet/vocab.json", encoding="utf-8"))
vocset = {t["tag"] if isinstance(t, dict) else t for t in voc["tags"]}

SEEDS = ["1girl, solo", "2girls", "1boy, solo", "1girl, cat ears",
         "1girl, swimsuit", "1girl, kimono", "no humans, scenery", "1girl, armor, sword"]
def gen(seed, ratings="safe", tgt="<|long|>"):
    p = (f"quality: masterpiece\nrating: {ratings}\nartist: <|empty|>\ncharacters: <|empty|>\n"
         f"copyrights: <|empty|>\naspect ratio: 1.0\ntarget: {tgt}\ngeneral: {seed}<|input_end|>\n")
    inp = tok(p, return_tensors="pt").to(DEV)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=128, do_sample=True, temperature=1.0,
                             top_p=0.95, pad_token_id=tok.eos_token_id)
    g = tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
    g = g.split("target:")[-1]  # strip leading target token if present
    tags = [s.strip() for s in g.replace("\n", ",").split(",") if s.strip() and "<|" not in s and ":" not in s]
    return tags

all_in = all_tot = 0
for s in SEEDS:
    tags = gen(s)
    inv = [t for t in tags if t in vocset]
    all_in += len(inv); all_tot += len(tags)
    print(f"[{s}] {len(inv)}/{len(tags)} in-vocab; out: {[t for t in tags if t not in vocset][:8]}")
print(f"\n総計 in-vocab: {all_in}/{all_tot} = {100*all_in/max(1,all_tot):.1f}%")
print(f"(dollama vocab=4994 は出現頻度上位 min_count=2518 で足切り済 → 低頻度タグは vocab 外で正常)")

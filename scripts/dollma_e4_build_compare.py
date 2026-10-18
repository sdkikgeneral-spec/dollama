"""E-4 比較ページ生成: Illustrious v0/v2 と Anima Base/Aesthetic を prompt×seed で並べる。"""
import csv, glob, html, os, sys
from PIL import Image

WT = sys.argv[1]            # worktree root
OUT = sys.argv[2]           # 出力ディレクトリ (index.html + img/)
os.makedirs(os.path.join(OUT, "img"), exist_ok=True)

MODELS = [
    ("illustrious-xl", "e4-illustrious-v2", "Illustrious v0 (現行既定)", "SDXL · 20 step · 2024 初期版"),
    ("illustrious-xl-v2", "e4-illustrious-v2", "Illustrious v2.0", "SDXL · 20 step · preset 追加"),
    ("anima-base-v1.0", "e4-anima", "Anima Base v1.0", "Cosmos 2B · 30 step · CFG 4 · 非商用重み"),
    ("anima-aesthetic-v1.1", "e4-anima", "Anima Aesthetic v1.1", "Cosmos 2B · 30 step · CFG 4 · 非商用重み"),
]
PROMPTS = [
    ("p1", "単独・立ち", "1girl, solo, standing, full body, long silver hair, blue eyes, school uniform, looking at viewer, simple background, white background"),
    ("p2", "手・小物", "1girl, solo, upper body, holding cup, both hands, smile, short brown hair, casual clothes, simple background, white background"),
    ("p3", "動き・全身", "1boy, solo, running, dynamic pose, full body, black hair, jacket, from side, simple background, white background"),
    ("p4", "複数人", "2girls, multiple girls, standing together, one with long red hair, one with short blue hair, both smiling, simple background, white background"),
]
SEEDS = [1000, 1001, 1002, 1003]

def load_rows(d):
    p = os.path.join(WT, "docs", "logs", d, "grid_results.csv")
    if not os.path.isfile(p):
        return {}
    with open(p, encoding="utf-8") as f:
        return {(r["preset"], r["prompt_id"], int(r["seed"])): r for r in csv.DictReader(f)}

rows = {}
for _, d, _, _ in MODELS:
    rows.update(load_rows(d))

def fnum(v, fmt):
    try:
        return format(float(v), fmt)
    except (TypeError, ValueError):
        return "–"

# モデル別平均 (recall / worst_anatomy)
def mean(vals):
    vals = [float(v) for v in vals if v not in (None, "", "None")]
    return sum(vals) / len(vals) if vals else None

summary = []
for m, d, name, sub in MODELS:
    rs = [r for k, r in rows.items() if k[0] == m]
    summary.append((name, len(rs), mean([r.get("recall") for r in rs]),
                    mean([r.get("soft_recall") for r in rs]), mean([r.get("worst_anatomy") for r in rs])))

sections = []
for pid, plabel, ptext in PROMPTS:
    trs = []
    for m, d, name, sub in MODELS:
        cells = []
        for s in SEEDS:
            hits = glob.glob(os.path.join(WT, "docs", "logs", d, "img", f"{m}__{pid}__seed{s}__*.png"))
            r = rows.get((m, pid, s), {})
            if hits:
                fn = f"{m}__{pid}__{s}.jpg"
                im = Image.open(hits[0]).convert("RGB")
                im.thumbnail((640, 640))
                im.save(os.path.join(OUT, "img", fn), quality=86)
                meta = f"recall {fnum(r.get('recall'), '.2f')} · 崩れ {fnum(r.get('worst_anatomy'), '.3f')}"
                cells.append(f'<figure><button class="shot" type="button" data-src="img/{fn}" '
                             f'data-cap="{html.escape(name)} · {pid} · seed {s}"><img src="img/{fn}" '
                             f'alt="{html.escape(name)} {pid} seed {s}" loading="lazy"></button>'
                             f'<figcaption>seed {s}<span>{meta}</span></figcaption></figure>')
            else:
                cells.append(f'<figure class="miss"><div class="hole">未生成</div><figcaption>seed {s}</figcaption></figure>')
        trs.append(f'<div class="row"><div class="who"><b>{html.escape(name)}</b><small>{html.escape(sub)}</small></div>'
                   f'<div class="cells">{"".join(cells)}</div></div>')
    sections.append(f'<section id="{pid}"><header><span class="pid">{pid}</span><h2>{plabel}</h2></header>'
                    f'<p class="prompt"><code>{html.escape(ptext)}</code></p>{"".join(trs)}</section>')

sum_rows = "".join(
    f"<tr><th>{html.escape(n)}</th><td>{c}/16</td><td>{fnum(a, '.3f')}</td><td>{fnum(b, '.3f')}</td><td>{fnum(w, '.4f')}</td></tr>"
    for n, c, a, b, w in summary)

page = open(os.path.join(os.path.dirname(__file__), "dollma_e4_compare_template.html"), encoding="utf-8").read()
page = page.replace("{{SUMMARY}}", sum_rows).replace("{{SECTIONS}}", "".join(sections))
open(os.path.join(OUT, "index.html"), "w", encoding="utf-8").write(page)
print("ok", sum(1 for _ in glob.glob(os.path.join(OUT, "img", "*.jpg"))), "images")

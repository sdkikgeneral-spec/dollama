#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dollma_probe_matting.py — マッティング (透過 PNG 切り抜き) モデル比較 probe

character-bible-spec §3 (MattingMode::Segment / soft α / emit_alpha=RGBA) の本体。
ISNet-anime (skytnt/anime-seg) と BiRefNet (onnx-community/BiRefNet-ONNX) を
CPU (ONNX Runtime) で比較する。

この PC: GTX1080Ti / i7-10700 / NPU なし / nvcc 未導入。
推論ランタイム選定理由:
  - PyTorch/OpenVINO 未導入。onnxruntime は導入済み。
  - 両モデルとも ONNX が直接配布 (export 不要)。
  - ONNX 入口は後段 M-1 の OV IR 変換 (ONNX->OV) にそのまま乗る。
  → ONNX Runtime CPUExecutionProvider を採用。

サンプル画像選定理由:
  - 手元に立ち絵なし、PyTorch 無しで生成不可、外部 anime 画像はライセンス不確実。
  - PIL で「髪の細いストランド・指・装飾・アンチエイリアス縁」を作り込んだ
    合成キャラを生成。出所完全自己完結・再現可能。spec の3背景色(white/grey/
    simple=薄色)で抜き品質(縁スピル・白系コントラスト)も同時に見る。
"""
import json, time, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, "..", "src", "tests", "data", "matting")
OUT  = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------------
# 1. 合成アニメキャラ画像生成 (RGB, 1024x1024)
#    抜き品質を試す細部: 髪の細ストランド / 指 / 髪飾り / AA 縁
# ----------------------------------------------------------------------------
def make_character(bg_rgb, seed=0):
    W = H = 1024
    img = Image.new("RGB", (W, H), bg_rgb)
    # 4x スーパーサンプルで AA 縁を作る
    S = 2
    cv = Image.new("RGB", (W*S, H*S), bg_rgb)
    d = ImageDraw.Draw(cv)
    cx = W*S//2
    skin = (250, 224, 200)
    hair = (90, 70, 140)      # 紫髪
    hair2 = (120, 95, 175)
    cloth = (60, 90, 160)
    # 胴・服 (台形)
    d.polygon([(cx-230*S, H*S), (cx+230*S, H*S), (cx+170*S, 560*S), (cx-170*S, 560*S)], fill=cloth)
    # 首
    d.rectangle([cx-55*S, 470*S, cx+55*S, 580*S], fill=skin)
    # 顔 (楕円)
    d.ellipse([cx-150*S, 230*S, cx+150*S, 520*S], fill=skin)
    # 髪 後ろ毛 (大きく広がる)
    d.polygon([(cx-210*S, 240*S),(cx+210*S,240*S),(cx+260*S,640*S),
               (cx-260*S,640*S)], fill=hair)
    # 前髪
    d.polygon([(cx-170*S,210*S),(cx+170*S,210*S),(cx+150*S,360*S),
               (cx,300*S),(cx-150*S,360*S)], fill=hair2)
    # 細い髪ストランド (アホ毛・サイド) — 抜き品質の本命
    rng = np.random.default_rng(seed)
    for i in range(40):
        x0 = cx + int((rng.random()-0.5)*520*S)
        y0 = 240*S + int(rng.random()*300*S)
        x1 = x0 + int((rng.random()-0.5)*120*S)
        y1 = y0 + int(rng.random()*260*S)
        w = max(1, int((1+rng.random()*2)*S))
        d.line([(x0,y0),(x1,y1)], fill=hair if i%2 else hair2, width=w)
    # アホ毛 (頭頂から細く跳ねる)
    d.line([(cx,210*S),(cx-30*S,90*S),(cx+40*S,60*S)], fill=hair2, width=3*S)
    # 髪飾り (装飾)
    d.ellipse([cx-185*S,300*S,cx-150*S,340*S], fill=(230,60,90))
    # 目
    for ex in (-65,65):
        d.ellipse([cx+(ex-28)*S,360*S,cx+(ex+28)*S,420*S], fill=(255,255,255))
        d.ellipse([cx+(ex-18)*S,370*S,cx+(ex+18)*S,415*S], fill=(40,120,200))
        d.ellipse([cx+(ex-9)*S,380*S,cx+(ex+9)*S,402*S], fill=(20,20,30))
    # 腕 + 手 (指 — 細部)
    for sgn in (-1, 1):
        ax = cx + sgn*175*S
        d.line([(ax,600*S),(ax+sgn*90*S,860*S)], fill=skin, width=44*S)
        hx, hy = ax+sgn*90*S, 860*S
        d.ellipse([hx-40*S,hy-40*S,hx+40*S,hy+40*S], fill=skin)  # 手のひら
        for f in range(4):  # 指4本
            fx = hx + (f-1.5)*22*S
            d.line([(fx,hy),(fx,hy+70*S)], fill=skin, width=12*S)
        d.line([(hx+sgn*38*S,hy),(hx+sgn*70*S,hy-40*S)], fill=skin, width=14*S)  # 親指
    cv = cv.resize((W, H), Image.LANCZOS)  # ダウンサンプルで AA 縁
    return cv

BACKGROUNDS = {
    "white":  (255, 255, 255),  # spec 候補 white background
    "grey":   (160, 160, 160),  # spec 候補 grey background
    "simple": (210, 222, 235),  # spec 既定 simple background (薄色)
}

# ----------------------------------------------------------------------------
# 2. 前処理 (各モデル)
# ----------------------------------------------------------------------------
def pre_isnet(img):
    # 公式 inference.py: img/255 のみ ([0,1] レンジ)。正方入力ゆえパディング不要。
    a = np.asarray(img.resize((1024,1024), Image.BILINEAR), np.float32)/255.0
    return a.transpose(2,0,1)[None].astype(np.float32)

def pre_birefnet(img):
    # ImageNet 正規化 (preprocessor_config.json)
    mean=np.array([0.485,0.456,0.406],np.float32); std=np.array([0.229,0.224,0.225],np.float32)
    a = np.asarray(img.resize((1024,1024), Image.BILINEAR), np.float32)/255.0
    a = (a-mean)/std
    return a.transpose(2,0,1)[None].astype(np.float32)

def sigmoid(x): return 1.0/(1.0+np.exp(-x))

def post(mask_raw):
    # [1,1,1024,1024] -> [1024,1024] float, レンジを [0,1] に正規化
    m = mask_raw[0,0].astype(np.float32)
    mn, mx = float(m.min()), float(m.max())
    raw_range = (mn, mx)
    # 出力が既に [0,1] 近傍ならそのまま, ロジットっぽければ sigmoid
    if mx > 1.5 or mn < -0.5:
        m = sigmoid(m)
    m = np.clip(m, 0, 1)
    return m, raw_range

# ----------------------------------------------------------------------------
# 3. メイン
# ----------------------------------------------------------------------------
def main():
    paths = json.load(open(os.path.join(HERE,"_matting_paths.json")))
    so = ort.SessionOptions()
    so.intra_op_num_threads = os.cpu_count()
    sess = {
        "isnet":    ort.InferenceSession(paths["isnet"],    providers=["CPUExecutionProvider"], sess_options=so),
        "birefnet": ort.InferenceSession(paths["birefnet"], providers=["CPUExecutionProvider"], sess_options=so),
    }
    pre = {"isnet":pre_isnet, "birefnet":pre_birefnet}
    inname = {m:sess[m].get_inputs()[0].name for m in sess}

    report = {}
    # 評価対象: 合成3背景 + 実アニメ画像(SkyTNT banner より抽出, Touhou/Reimu)
    real_path = os.path.join(OUT, "input_real_anime.png")
    cases = [(bg, make_character(rgb, seed=42)) for bg,rgb in BACKGROUNDS.items()]
    if os.path.exists(real_path):
        cases.append(("real_anime", Image.open(real_path).convert("RGB")))
    for bgname, img in cases:
        img.save(os.path.join(OUT, f"input_{bgname}.png"))
        for m in sess:
            x = pre[m](img)
            # warmup
            sess[m].run(None, {inname[m]: x})
            ts=[]
            for _ in range(5):
                t=time.perf_counter(); raw=sess[m].run(None, {inname[m]: x}); ts.append((time.perf_counter()-t)*1000)
            med = float(np.median(ts))
            mask01, raw_range = post(raw[0])
            # 連続値か (二値でないか) を測る: 中間 [0.05,0.95] にある画素割合
            soft_frac = float(((mask01>0.05)&(mask01<0.95)).mean())
            fg_cover  = float((mask01>0.5).mean())
            # フル解像でα合成 (二値化しない soft α)
            mask_full = Image.fromarray((mask01*255).astype(np.uint8)).resize(img.size, Image.BILINEAR)
            rgba = img.convert("RGBA"); rgba.putalpha(mask_full)
            rgba.save(os.path.join(OUT, f"cutout_{m}_{bgname}.png"))
            # 生マスク保存 (golden 前段)
            Image.fromarray((mask01*255).astype(np.uint8)).save(os.path.join(OUT, f"mask_{m}_{bgname}.png"))
            np.save(os.path.join(OUT, f"mask_{m}_{bgname}.npy"), mask01)
            report.setdefault(m,{})[bgname] = dict(
                ms_median=round(med,1), raw_range=[round(raw_range[0],3),round(raw_range[1],3)],
                soft_edge_frac=round(soft_frac,4), fg_cover=round(fg_cover,4),
                ms_all=[round(t,1) for t in ts])
            print(f"[{m:9s} {bgname:6s}] {med:6.1f}ms  raw{raw_range}  soft_edge={soft_frac:.4f}  fg={fg_cover:.4f}")

    # モデルサイズ
    for m in sess:
        report.setdefault(m,{})["_size_MB"] = round(os.path.getsize(paths[m])/1e6,1)
    json.dump(report, open(os.path.join(OUT,"_report.json"),"w"), indent=2)
    print("\nsaved to", OUT)
    print(json.dumps(report, indent=2))

if __name__=="__main__":
    main()

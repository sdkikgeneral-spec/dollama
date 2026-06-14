"""
dollma_probe10_sdxl.py - SDXL on RTX5080 (初の実画像生成)
==========================================================

確認項目:
  STEP 1 : RTX5080 (sm_120 / cu128) で diffusers が動くか確認
  STEP 2 : SDXL テキスト→画像生成 + レイテンシ計測
  STEP 3 : 生成画像を PNG 保存

SDXL の CLIP エンコーダー:
  - CLIP-L  (openai/clip-vit-large-patch14)   → 将来的に NPU へ
  - OpenCLIP-bigG (laion/CLIP-ViT-bigG-14...) → 将来的に NPU へ
  現 probe では全部 GPU で動かし、ベースラインを取る。
"""

import time
from pathlib import Path

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

PROMPT = (
    "1girl, anime, silver hair, magical girl, twin tails, "
    "detailed eyes, soft lighting, pastel colors, "
    "masterpiece, best quality"
)
NEGATIVE = "lowres, bad anatomy, bad hands, cropped, worst quality"

STEPS  = 20
WIDTH  = 1024
HEIGHT = 1024


# ============================================================
# STEP 1: 環境確認
# ============================================================
print("=" * 60)
print("STEP 1: 環境確認")
print("=" * 60)

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("❌ CUDA が使えません")
    exit(1)

props = torch.cuda.get_device_properties(0)
vram_gb = props.total_memory / 1024**3
print(f"GPU: {props.name}")
print(f"VRAM: {vram_gb:.1f} GB")
print(f"Compute: sm_{props.major}{props.minor}")
print(f"CUDA: {torch.version.cuda}")

# RTX5080 = sm_120 確認
assert props.major >= 12, f"Expected sm_12x (Blackwell), got sm_{props.major}{props.minor}"
print("✅ RTX5080 (Blackwell sm_120) 確認")


# ============================================================
# STEP 2: SDXL ロード + 生成
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: SDXL ロード + 画像生成")
print("=" * 60)

from diffusers import StableDiffusionXLPipeline
import torch

print("SDXL ロード中 (初回はダウンロード数分)...")
t_load = time.perf_counter()

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
pipe = pipe.to("cuda")
load_s = time.perf_counter() - t_load
print(f"ロード時間: {load_s:.1f}s")

# VRAM 確認
vram_used = torch.cuda.memory_allocated() / 1024**3
vram_reserved = torch.cuda.memory_reserved() / 1024**3
print(f"VRAM 使用: {vram_used:.2f} GB / 予約: {vram_reserved:.2f} GB")

print(f"\nプロンプト: {PROMPT[:60]}...")
print(f"サイズ: {WIDTH}×{HEIGHT}  ステップ数: {STEPS}")

# ウォームアップ (1ステップ)
print("\nウォームアップ中 (1step)...")
with torch.no_grad():
    _ = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE,
        num_inference_steps=1,
        width=WIDTH, height=HEIGHT,
        output_type="latent",
    )
torch.cuda.synchronize()

# 本計測
print(f"生成計測中 ({STEPS}steps)...")
times = []
for i in range(3):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE,
        num_inference_steps=STEPS,
        width=WIDTH, height=HEIGHT,
        generator=torch.Generator("cuda").manual_seed(42 + i),
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    times.append(elapsed)
    print(f"  run {i+1}: {elapsed:.2f}s  ({STEPS/elapsed:.1f} steps/s)")

import statistics
med_s = statistics.median(times)
print(f"\n中央値: {med_s:.2f}s  ({STEPS/med_s:.1f} steps/s)")
print(f"it/s: {1/med_s*STEPS:.2f}  |  ms/step: {med_s/STEPS*1000:.0f}ms")

vram_peak = torch.cuda.max_memory_allocated() / 1024**3
print(f"VRAM ピーク: {vram_peak:.2f} GB")


# ============================================================
# STEP 3: 画像保存
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: 画像保存")
print("=" * 60)

image = result.images[0]
out_path = OUTPUT_DIR / "probe10_sdxl_test.png"
image.save(out_path)
print(f"✅ 保存: {out_path}  ({image.size[0]}×{image.size[1]})")


print("\n" + "=" * 60)
print("probe10 完了")
print("=" * 60)
print(f"""
計測サマリー:
  ロード時間      : {load_s:.1f}s
  生成時間 (中央値): {med_s:.2f}s  ({STEPS}steps, {WIDTH}×{HEIGHT})
  スループット    : {STEPS/med_s:.1f} steps/s
  VRAM ピーク     : {vram_peak:.2f} GB
  出力            : {out_path}
""")

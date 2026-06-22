# dollama

**English** | **[日本語](README_jp.md)**

**Using every piece of hardware in the box — CPU / NPU / iGPU / RTX5080 — to research a 2D-illustration–focused image generation pipeline.**

The goal is to make each device cooperate by playing to its strengths. Not the shortest path to a working
generator, but the best *assignment* of work across heterogeneous hardware. Built from scratch in C++,
without ML frameworks.

> ⚠️ **Work in progress — research project, not a finished product.**
> The full from-scratch C++ diffusion pipeline (custom UNet + Euler scheduler + VAE decode) now
> **generates real 1024×1024 images** (task 2-6a). Two caveats remain: (1) speed is bound by the naive
> hand-written kernels — **84 s for 20 steps** (22× slower than the PyTorch/diffusers probe10 baseline;
> Tensor-Core / flash kernels are the next focus), and (2) inputs are golden embeddings — **wiring up
> arbitrary text → image (SDXL dual encoder with CLIP-G + CFG) is not done yet** (task 2-6b).
> APIs, file layout, and measurements will change. Published for transparency of the research process,
> not for turnkey use.

> **This research targets a specific machine: an Intel Core Ultra 9 285 (Intel AI Boost NPU + Intel Xe iGPU)
> combined with an NVIDIA RTX 5080.** The NPU / iGPU usage (via OpenVINO) depends on Intel-platform
> specifics, so all measurements and device choices assume this environment.

> 📖 The full design notes, pipeline diagrams, and detailed tables are maintained in Japanese:
> **[README_jp.md](README_jp.md)**. This page is a condensed English overview.

---

## Overview

dollama is a research project that makes the **CPU, NPU, iGPU, and dGPU of a single machine cooperate as one
inference pipeline** to generate 2D illustration / manga **character** images (no background, output as a
pre-cut transparent PNG). It is built **entirely from scratch in C++** — from the inference kernels to the
HTTP server — without ML frameworks like PyTorch or diffusers.

**Core idea — "temporal cooperation":** instead of wiring devices together with zero-copy interconnects,
the seconds the GPU spends on diffusion are used to run the next request's CLIP embedding and tagging on the
otherwise-idle NPU/CPU in parallel. Each device is **assigned the task it is best at**, and they fill each
other's wait time.

```
[CPU]     LLM (prompt generation) / WD14 tagging
[NPU]     CLIP text encoder (fastest on fixed shapes)
[iGPU]    VAE encode (img2img input, parallel with CPU)
[RTX5080] SDXL UNet 20 steps + VAE decode (the generator itself)
```

**Current status:**

| Phase | Scope | Status |
|---|---|---|
| Phase 1 | C++ pipeline skeleton (Tensor/Allocator/Queue/CLIP-NPU/WD14-CPU/threads) | ✅ Done (9.13 frames/s) |
| Phase 2 | CUDA kernels + safetensors + VAE decode + SDXL UNet + Euler + full diffusion wiring | ✅ Done (generates real 1024² images — **84 s / 20 steps**) |
| Phase 3 | OpenAI-compatible HTTP server (cpp-httplib / nlohmann-json) | ✅ Done (PipelineGenerator wired via DI, with fallback) |
| Phase 4 | Custom tag-generation LM (bitnet.hpp 33M) + identity conditioning / quality scorer; ternary as a compression experiment | ⏳ Dense line done #1–#4 + #6 (CPU C++ inference, text→tags, golden-matched corr 1.0); top10 tag recall 0.777 |

> **Next up:** ① speed optimization (direct conv → im2col/Tensor-Core GEMM, naive attention → flash) to bring
> 84 s down toward probe10's 3.80 s. ② arbitrary text → image (task 2-6b: SDXL dual encoder with CLIP-G + CFG + negative prompt).

> Full roadmap in [`docs/roadmap.md`](docs/roadmap.md); the rationale behind the HW roles is in
> "What we learned" below.

---

## What we learned (probe summary)

Key takeaways from 10 probes plus the Phase 2 hand-written kernels. **Conclusions first, numbers second:**

1. **Device selection is decided by "task × architecture fit" — intuition is unreliable.**
   The NPU is fastest for a fixed-shape encoder (CLIP-L, 77 tokens): 7.85 ms, 2.5× faster than CPU.
   Yet for *the same kind of inference* on WD14 SwinV2 (window attention) the NPU is the **slowest**
   (268 ms vs. 101 ms on CPU), and an autoregressive LLM (dynamic KV-cache) **won't even compile**.
   "NPU = fast" is wrong; "NPU = fast *only* for fixed-shape pure-GEMM chains" is right.
   → We measured every model before assigning it.

2. **Zero-copy device-to-device sharing is impossible on this setup; CPU pinned memory is the practical path.**
   There is no CUDA↔NPU interop API, and the iGPU is hidden from DXGI by BIOS settings (no D3D12
   cross-adapter). The remaining route — transfers through CPU pinned memory — costs only **3.4%** overhead,
   fully hideable behind GPU diffusion with multithreading. Chasing direct interconnects was unnecessary.

3. **The iGPU is useful *directionally*.** Large convolutions (VAE decode) run 8× slower than CPU and are a
   dead end, but VAE **encode** (img2img) is faster on the iGPU (**79 ms** vs. 117 ms on CPU) — and it runs
   in parallel with the CPU's ~2 s LLM step, so it costs zero wall-clock time on the img2img path.

4. **The RTX 5080 (Blackwell / sm_120) runs SDXL at 3.80 s/image** (1024², 20 steps), peak VRAM 10.49 GB
   of 16 GB — leaving headroom up to ~1536px long edge. The generator lives on the GPU while the NPU/CPU
   fill its idle time with CLIP encoding and tagging in parallel.

5. **A full diffusion pipeline of hand-written CUDA kernels (no ML framework) runs and produces real images.**
   Phase 2 implements GEMM / activation / GroupNorm / Conv2d / Attention from scratch, each validated
   against a CPU reference with golden tests (GEMM 4730 GFLOPS / Conv2d 1807 GFLOPS / Attention 1631 GFLOPS).
   On top of those, a custom VAE decode (final SSIM 0.999992) and custom SDXL UNet (noise_pred SSIM 0.999998)
   are wired to the Euler scheduler to **generate a real 1024² image in 20 steps (84 s)**. Correctness is
   there — speed (bound by direct conv + naive attention, 22× slower than probe10) is the next focus, with
   Tensor-Core / flash kernels as the main lever.

→ **"Use all the hardware" is not wishful thinking — measurements show it holds.** The key is not direct
interconnects but *temporal* cooperation: assign each device the task it's best at, and fill the GPU's
generation window with NPU/CPU work.

---

## Target environment

| Component | Detail |
|---|---|
| CPU / NPU | Intel Core Ultra 9 285 (NPU = Intel AI Boost, DEVICE_ARCHITECTURE: 3720) |
| GPU | NVIDIA GeForce RTX 5080 (Blackwell / sm_120, CUDA 12.8, 15.9 GB VRAM) |
| iGPU | Intel Xe Graphics (OpenVINO GPU.0, shares system RAM) |
| OS | Windows 11 |
| Investigation phase | Python 3.14 + OpenVINO / diffusers (probe scripts) |
| Implementation | C++ + Meson, STL + CUDA API + Winsock2 only (no ML frameworks) |

---

## Hardware role assignment (all measured)

| HW | Task | Measured |
|---|---|---|
| **NPU** | CLIP-L text encoder (fixed 77 tokens) | **7.85 ms** — 2.5× faster than CPU |
| **NPU** | WD14 SwinV2 tagger (448×448, fixed) | 268 ms (runs in parallel during GPU generation) |
| **iGPU** | VAE encode — img2img only (image → latent) | **79 ms** — faster than CPU's 117 ms |
| **CPU** | LLM prompt generation (Qwen2-1.5B for now → custom tag-generation LM bitnet.hpp later) | 64–71 tok/s |
| **RTX5080** | SDXL UNet (20 steps) + VAE decode | **3.80 s** / 1024×1024 |

### Device selection rationale

| Model | CPU | iGPU | NPU | Chosen | Reason |
|---|---|---|---|---|---|
| CLIP-L text encoder | 20 ms | 14 ms | **7.85 ms** | NPU | Pure GEMM chain |
| WD14 SwinV2 tagger | **101 ms** | 104 ms | 268 ms | CPU | Window attention is a poor NPU fit |
| VAE decode | 126 ms | 995 ms | – | RTX5080 | iGPU is 8× slower |
| VAE encode (img2img) | 117 ms | **79 ms** | – | iGPU | The encode direction favors the iGPU |

---

## Pipeline

### txt2img

```
[CPU] Qwen2-1.5B (interim) / later: custom tag-generation LM (bitnet.hpp 33M, GPU-first via custom CUDA kernels / CPU ok / NPU excluded)
  natural language → danbooru tag list (~2s / future <10ms)
    │
    ▼
[NPU] CLIP-L text encoder (7.85ms)
  text → embedding [1, 77, 768]
    │
    ▼
[RTX5080] SDXL UNet × 20 steps + VAE decode (3.80s / 1024×1024)
    │
    ├─ [CPU] WD14 SwinV2 tagger (101ms) ← runs in parallel during GPU generation
    │         generated image → danbooru tags → LLM feedback loop
    └─ output image
```

### img2img (additional path)

```
input image
    ├─→ [iGPU] VAE encode (79ms)    ─→ latent ─┐
    │                                           │
    └─→ [CPU]  LLM text generation (~2s) ───────┤ (parallel)
                                               │
                                    [NPU] CLIP (7.85ms)
                                               │
                                    [RTX5080] SDXL UNet + VAE decode (3.80s)
                                               │
                                           output image
```

The iGPU's VAE encode (79 ms) runs in parallel with the CPU's LLM generation (~2 s), so it adds zero wait.

---

## Measurement baseline (probe results)

| Metric | Value | probe |
|---|---|---|
| CPU→VRAM (100 MB) | 3.46 ms / 30.3 GB/s | probe2 |
| NPU inference (512-dim MLP) | 0.88 ms | probe2 |
| NPU output → GPU | 0.031 ms (3.4%) | probe2 |
| iGPU VAE decode stub | 995 ms (8× slower than CPU ❌) | probe4 |
| iGPU VAE encode (1024→128) | **79 ms** (faster than CPU's 117 ms ✅) | probe5 |
| Qwen2-1.5B INT4 CPU tok/s | 64–71 tok/s | probe7 |
| **WD14 SwinV2 (448×448)** | CPU 101 ms / iGPU 104 ms / **NPU 268 ms** | probe8 |
| **CLIP-L text encoder (77 tokens)** | CPU 20 ms / iGPU 14 ms / **NPU 7.85 ms** | probe9 |
| **SDXL 20 steps 1024×1024** | **3.80 s** / 5.3 it/s / peak VRAM 10.49 GB | probe10 |

### Hand-written CUDA kernels (Phase 2, RTX5080)

| Kernel | Measured |
|---|---|
| FP16 GEMM (shared-mem tiling, FP32 accum) | 1024³ **4730 GFLOPS** / SDXL Linear transB 4208 GFLOPS |
| SiLU / GeLU activation | FFN 544 GB/s |
| GroupNorm (1 group = 1 block, single-pass) | UNet 73–75 GB/s |
| Conv2d (direct, FP32 accum) | UNet C320 64² 3×3 **1807 GFLOPS** / VAE C128 512² 3×3 1667 GFLOPS |
| Attention (per-(b,h,row) block, FP32 softmax) | self 1024² Dh80 **1631 GFLOPS** / cross Sk77 1748 GFLOPS |

---

## Implementation approach (C++ from scratch)

```
src/
├── core/        — Tensor (STL-based), allocator (CPU / pinned / VRAM), CharacterBible + prompt compose + color mode
├── kernels/     — GEMM, activation, GroupNorm, Conv2d, Attention, VAE decode (custom CUDA) + ternary GEMM (compression experiment, planned)
├── infer/       — CLIP (NPU, OpenVINO), WD14 (CPU), SDXL UNet, Euler scheduler, full diffusion loop
├── io/          — safetensors loader, PNG character-metadata round-trip, tokenizer (tag-level exact-match)
└── server/      — cpp-httplib + OpenAI-compatible API, PipelineGenerator (DI with stub fallback)
```

**Used:** STL / CUDA API / Winsock2 — **Not used:** PyTorch / OpenVINO (probes only) / diffusers / llama.cpp.

For the future custom tag-generation LM (bitnet.hpp 33M; ternary {-1, 0, +1} is a later compression experiment, not the goal), the pipeline structure,
character-consistency design, and full roadmap, see **[README_jp.md](README_jp.md)** and `docs/`.

---

## Build (C++)

> Status: builds today include the core, CLIP (NPU) / WD14 (CPU) inference glue, the threaded pipeline
> skeleton, the full Phase 2 CUDA stack (kernels + VAE decode + SDXL UNet + Euler scheduler + diffusion
> loop), and the OpenAI-compatible HTTP server, all with tests. **End-to-end image generation from golden
> embeddings works** (84 s / 20 steps); generation from arbitrary text is task 2-6b.

**Prerequisites**

- [Meson](https://mesonbuild.com/) + Ninja, and a C++20 compiler (tested with MSVC on Windows 11)
- **CUDA Toolkit 13.x** (`nvcc`) for the kernels — RTX 5080 needs `sm_120` (CUDA 12.8+). Put `nvcc` on `PATH`.
- **OpenVINO 2024.x+** runtime for the CLIP/WD14 NPU paths (optional; toggle with `-Dwith_openvino`)

**Build & test**

```bash
# configure (point the options at your local toolkits)
meson setup build \
  -Dwith_cuda=true     -Dgpu_sdk_root="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3" \
  -Dwith_openvino=true -Dnpu_sdk_root="C:/Program Files (x86)/Intel/openvino_2024"

meson compile -C build
meson test -C build            # runs all unit tests + kernel golden tests/benchmarks
```

CUDA support can be disabled with `-Dwith_cuda=false` (the `.cu` tests are then skipped). See
[`docs/testing.md`](docs/testing.md) for the per-test breakdown.

---

## Models

Model weights are **not included** in this repository (`models/` is git-ignored). To run the CLIP / WD14 /
SDXL stages you must obtain and convert the models yourself:

- **CLIP-L text encoder** and **WD14 SwinV2 tagger** → OpenVINO IR (NPU / CPU), converted via the scripts in
  `scripts/` (see the probe scripts) using `optimum-intel` / OpenVINO.
- **Qwen2-1.5B (INT4)** for the interim LLM prompt stage.
- **SDXL** for the diffusion stage (the custom CUDA kernels now generate images; the text front-end is task 2-6b).

The project **code** is Apache-2.0 (below), but each model's **weights are governed by their own upstream
licenses** (SDXL, CLIP-L, WD14, Qwen2, etc.). You are responsible for complying with those terms.

---

## Setup (investigation phase / Python probes)

```bash
pip install openvino openvino-genai openvino-tokenizers
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install diffusers transformers accelerate optimum[openvino]
pip install huggingface_hub
```

---

## License

Licensed under the [Apache License 2.0](LICENSE). See the [`LICENSE`](LICENSE) file for details.

# dollama

**English** | **[日本語](README_jp.md)**

**Using every piece of hardware in the box — CPU / NPU / iGPU / RTX5080 — to research a 2D-illustration–focused image generation pipeline.**

The goal is to make each device cooperate by playing to its strengths. Not the shortest path to a working
generator, but the best *assignment* of work across heterogeneous hardware. Built from scratch in C++,
without ML frameworks.

> ⚠️ **Work in progress — research project, not a finished product.**
> The full from-scratch C++ diffusion pipeline (custom UNet + Euler scheduler + VAE decode) now
> **generates real 1024×1024 images from arbitrary text** (SDXL dual encoder with CLIP-G + CFG, task 2-6b),
> and mats the result to a **cut-out transparent PNG** (iGPU ISNet-anime). Speed has been brought down from
> an initial 84 s to **11.3 s for 20 steps** (im2col / Tensor-Core GEMM + cuBLAS fallbacks; ~3× of the
> PyTorch/diffusers probe10 baseline of 3.80 s). The remaining kernel bottleneck is UNet attention, but the
> from-scratch CUDA speed work is paused here — the research focus has moved to the **custom tag-generation
> LM (Phase 4)**. APIs, file layout, and measurements will change. Published for transparency of the
> research process, not for turnkey use.

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
| Phase 2 | CUDA kernels + safetensors + VAE decode + SDXL UNet + Euler + full diffusion wiring | ✅ Done (generates real 1024² images — optimized to **11.3 s / 20 steps**) |
| Phase 2-6b | Arbitrary text → image (SDXL dual encoder with CLIP-G + CFG + negative prompt) + matting → transparent PNG | ✅ Done (prompt → real 1024² transparent PNG end-to-end) |
| Phase 3 | OpenAI-compatible HTTP server (cpp-httplib / nlohmann-json) | ✅ Done (PipelineGenerator wired via DI, with fallback) |
| Phase 4 | Custom tag-generation LM (bitnet.hpp 33M) + identity conditioning / quality scorer; ternary as a compression experiment | ⏳ In progress — dense LM trains & infers in C++ (CPU/GPU, golden corr 1.0; GPU 87.5×, INT8 / AVX2 paths), identity conditioning (A) closed (retention 0.975), input-diversification (B) promoted to mainline, quality scorer (Model B) distilled over 8 anatomy axes |

> **Next up:** the research focus is now the **custom tag-generation LM (Phase 4)** — replacing the interim
> Qwen2 prompt stage with a 33M from-scratch model (text → danbooru tags, identity-conditioned) and closing
> the Model B anime quality-feedback loop. Swapping in a more recent anime-specialized SDXL checkpoint
> (no kernel changes needed) remains the single biggest lever on output quality.

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
   are wired to the Euler scheduler to **generate a real 1024² image from arbitrary text in 20 steps**.
   Speed was optimized from an initial 84 s down to **11.3 s** (im2col / Tensor-Core GEMM + cuBLAS
   fallbacks; ~3× of probe10); the remaining bottleneck is UNet attention, and the research focus has
   since moved on to the custom tag-generation LM (Phase 4).

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
    └─→ [iGPU] matting (ISNet-anime, ~100ms) → α extraction → transparent PNG
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
> loop), the OpenAI-compatible HTTP server, and matting → transparent PNG, all with tests.
> **End-to-end image generation from arbitrary text works** (11.3 s / 20 steps).

### One-shot installer (Windows, recommended)

On a fresh Windows box you can install everything the build/run needs (VS Build Tools / CUDA /
OpenVINO / Python deps) in one go via winget + pip:

```powershell
powershell -ExecutionPolicy Bypass -File install_windows.ps1
```

- Installs the CUDA Toolkit and torch=cu128 only when an NVIDIA dGPU is detected; otherwise it skips
  CUDA, installs CPU torch, and recommends `-Dwith_cuda=false`.
- Flags: `-DryRun` (print commands only), `-CheckOnly` (probe only), `-SkipCuda`, `-SkipSdk`,
  `-PythonExe <path>`, `-Force`.
- It prints a **recommended `meson setup` command** tailored to the detected environment at the end.

Prefer to set things up by hand? Follow the prerequisites and steps below.

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
- **SDXL** for the diffusion stage (the custom CUDA kernels generate images from text end-to-end).
- **ISNet-anime** (anime-segmentation) for matting → transparent PNG (iGPU; optional, falls back to opaque PNG).

The project **code** is Apache-2.0 (below), but each model's **weights are governed by their own upstream
licenses**. You are responsible for complying with those terms.

### Third-party model attribution

| Model | Role | Upstream license |
|---|---|---|
| SDXL | Diffusion (UNet + VAE) | Stability AI Community / CreativeML OpenRAIL-M (per checkpoint) |
| CLIP-L / CLIP-bigG | Text encoder (NPU) | MIT (OpenAI CLIP) / OpenCLIP |
| WD14 SwinV2 tagger v3 | Tagging (CPU) | Apache-2.0 |
| Qwen2-1.5B | Interim LLM prompt stage | Apache-2.0 |
| ISNet-anime (anime-segmentation) | Matting / transparent PNG (iGPU) | Apache-2.0 |
| TIPO-200M (KBlueLeaf) | Distillation teacher experiments (4-D6, not in production) | Apache-2.0 |
| **Eugeoter/waifu-scorer-v4-beta** | **Model B quality scorer — aesthetic teacher (default; label generation only)** | **Apache-2.0** |
| deepghs/anime_aesthetic | Model B quality scorer — alternative aesthetic teacher (switchable) | OpenRAIL |

> **On the Model B aesthetic teacher:** the default teacher (waifu-scorer-v4-beta) is Apache-2.0.
> A switchable alternative (deepghs/anime_aesthetic) is OpenRAIL, which permits commercial use,
> redistribution and modification — its only restrictions are the behavioral use-based prohibitions in
> the license appendix (no illegal / discriminatory / harmful uses). In either case dollama uses the model
> **only as a teacher to generate soft labels** that are distilled into our own from-scratch ScorerNet;
> the teacher weights themselves are **not redistributed** by this project.

---

## Setup (investigation phase / Python probes)

On Windows, `install_windows.ps1` (above) also installs these pip dependencies for you
(driven by `requirements.txt`; torch's index is auto-selected by GPU presence). To install by hand:


```bash
pip install openvino openvino-genai openvino-tokenizers
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install diffusers transformers accelerate optimum[openvino]
pip install huggingface_hub
```

---

## License

Licensed under the [Apache License 2.0](LICENSE). See the [`LICENSE`](LICENSE) file for details.

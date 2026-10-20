# dollama

**English** | **[日本語](README.md)**

**Research into reaching 2D illustration / manga character generation while using every piece of hardware in the box — CPU / NPU / iGPU / RTX5080.**

The point is not the shortest path to a working generator, but how to make each device play to its strengths and cooperate.
The **inference path** avoids PyTorch / diffusers / llama.cpp and is written in C++ with hand-written CUDA kernels (+ cuBLAS / the OpenVINO runtime).
Training, model conversion and golden-data generation are done in Python (PyTorch / diffusers / OpenVINO).

> ⚠️ **Research project, not a finished product.** Published to make the **research process** public, not for turnkey use.
> APIs, file layout and measurements will keep changing.
>
> - What works today: arbitrary text → a **fixed 1024×1024** image through the from-scratch C++/CUDA SDXL path (dual text encoder + CFG + UNet + Euler + VAE decode),
>   matted on the iGPU (ISNet-anime) into a **cut-out transparent PNG**. The default weights are the anime-specialized SDXL preset `illustrious-xl`.
> - Speed: the shipping path (CFG g=7.5, `--fast`, CFG batch=2) measures **10.65–10.96 s/image** (2026-09-04, research machine, UNet 20 steps + VAE decode, text encoding excluded).
>   The diffusers reference (probe10) of **3.80 s** was taken under different conditions (text encoding included); the custom path is **about 2.8× slower** than that (a rough comparison).
>   Speed work continues (G-10k closed negative on 2026-09-17; the next step, G-14k attention rewrite, is still at the planning stage).
> - Not done yet: **wiring** the custom tag-generation LM into the pipeline (prompts are passed in verbatim today), img2img (`/v1/images/edits` returns 501), WD14 tag feedback.

> **This research targets one machine: an Intel Core Ultra 9 285 (Intel AI Boost NPU + Intel Xe iGPU) combined with an NVIDIA RTX 5080.**
> NPU / iGPU usage (via OpenVINO) depends on Intel-platform specifics, so all measurements and device choices assume this environment.

> 📖 The Japanese **[README.md](README.md)** is the primary version (src tree, model placement table, file layout, full tables).
> This page is a condensed English overview.

---

## Overview and status

dollama makes the **CPU, NPU, iGPU and dGPU of a single machine cooperate as one inference pipeline** to generate 2D illustration / manga
**character** images (no background, output as a cut-out transparent PNG). Backgrounds are out of scope (composited with external tools).

**Core idea — "temporal cooperation" (a design goal):** instead of wiring devices together with zero-copy interconnects, use the seconds the GPU
spends on diffusion to run other stages (text embedding, scoring, tagging) on the otherwise-idle NPU/CPU/iGPU.
Today the shipping txt2img path processes a request sequentially; parallelism is only validated with full-size stubs so far.

| Phase | Scope | Status |
|---|---|---|
| Phase 1 | C++ pipeline skeleton (Tensor / Allocator / SPSC queue / CLIP-NPU / WD14-CPU / threads + affinity) | ✅ Done (9.13 frames/s, WD14-bound) |
| Phase 2 (2-0 to 2-6a) | CUDA kernels + safetensors + VAE decode + SDXL UNet + Euler + full diffusion loop | ✅ Done (first version 84.07 s / 20 steps) |
| 2-6 optimization | direct conv → im2col + GEMM, naive attention → flash + wmma, large GEMMs on cuBLAS | 🟡 Closed for now on 2026-06-22 (84.07 s → 11.30 s — but that is a **non-CFG, B=1, `generate()`-path value and is now stale**). Speed work continued afterwards: part of `--fast` (G-2k CFG batch2 / G-3k attention / G-4k epilogue fusion), on by default for every path (G-8k VRAM arena; kill switch `DOLLAMA_POOL=0`), env opt-in (G-10k conv true batch2, `DOLLAMA_CONV_BATCH=1`, closed negative) |
| 2-6b | Real txt2img: SDXL dual encoder (CLIP-L + CLIP-G) + CFG + negative prompt + matting → transparent PNG | ✅ Done (2026-06-23, `c3b1dac`) |
| 2-6c | Diffusion backend plugin interface (`IDiffusionBackend` registry: `"sdxl"` live / `"sd35"` stub) | ✅ Done (2026-07-02) |
| 2-6d | Three anime-specialized SDXL presets (NoobAI-XL 1.1 / Animagine XL 4.0 / Illustrious XL v0.1) — same architecture, weights swapped only | ✅ Done (2026-09-17; default `illustrious-xl`, plain SDXL via `--preset base`) |
| 2-6e | Automatic preset `prompt_prefix` / `negative_prefix` | ✅ Done (2026-09-19, `4d4a405`) |
| 2-6f | Switch between multiple dollama servers at runtime from the Blazor UI | ✅ Done (2026-09-19) |
| 2-6 remaining | Preset field in the HTTP API / preset selection in the UI | 🔲 Not started |
| Phase 3 | OpenAI Images–compatible HTTP server (cpp-httplib / nlohmann-json) | ✅ Done (generator DI with a 3-stage fallback: `BackendImageGenerator` → `PipelineGenerator` → `StubGenerator`) |
| LoRA (L-1 / L-2) | Offline merge script + runtime LoRA (HTTP `loras`, UI chips) | ✅ Done |
| Phase 4 | Custom tag-generation LM (33M) + identity conditioning (A) + anime quality scorer (Model B) + quality feedback (F) | ⏳ In progress |

Full roadmap: [`docs/roadmap.md`](docs/roadmap.md). Full measurements: [`docs/measurements-log.md`](docs/measurements-log.md) (Japanese).

---

## What we learned

1. **Device choice is decided by "task × architecture fit" — intuition is unreliable.** The NPU is fastest for a fixed-shape encoder
   (CLIP-L, 77 tokens: 7.85 ms, 2.5× CPU), but the **slowest** on WD14 SwinV2 (268 ms vs. 101 ms on CPU), and autoregressive LLMs
   (Phi-3 mini / Qwen2-1.5B INT4) **failed to compile** on it (dynamic shapes / KV-cache are outside the NPU's design).
2. **Zero-copy device-to-device sharing is not possible on this setup; CPU pinned memory is the practical path** (3.4% overhead).
3. **The iGPU is useful directionally.** VAE decode (stub) is 8× slower than CPU, but VAE encode is faster (79 ms vs. 117 ms), and matting
   (ISNet-anime) is fastest on the iGPU (99.96 ms) — so the iGPU owns matting in the shipping path.
4. **diffusers SDXL on the RTX 5080 takes 3.80 s/image** (1024², 20 steps, CFG, text encoding included), peak VRAM 10.49 GB (probe10).
   This is the reference; the custom path (10.65–10.96 s/image, text encoding excluded) does not reach it yet.
5. **A full diffusion pipeline on hand-written CUDA kernels runs and matches golden data**: custom VAE decode SSIM **0.999988**,
   custom SDXL UNet noise_pred SSIM **0.999996** (post-2-6 values; 0.999992 / 0.999998 at the time of 2-4 / 2-5).
6. **All-green numeric gates do not imply a speedup.** G-10k (true batch-2 conv2d) passed every numeric gate including bit-exact checks,
   yet the resnet bucket got +3.38% / +1.87% slower → demoted to opt-in (`DOLLAMA_CONV_BATCH=1`) and closed negative (`c3ee3ca`).
7. **Only input diversification robustly improved the tag LM.** Four distillation routes, more real pairs, and 33M→80M capacity did not move
   the main metric (diverse generation set-F1) beyond seed noise; input diversification (B) did, saturating at ~2,000 examples.

---

## Hardware assignment

### Implemented (shipping txt2img path, `src/server/cli_generate.hpp`)

| HW | Task | Notes |
|---|---|---|
| **CPU** | OV tokenizer (CLIP-L / CLIP-G tokenization) | Runs on CPU every request (`src/infer/clip_tokenizer.hpp` default device) |
| **NPU** | SDXL dual text encoder (CLIP-L + CLIP-G) | Falls back to CPU if NPU setup fails |
| **NPU** | ScorerNet quality scoring (custom 11.18M, 8 anatomy axes) | **Logged only** (no consumer yet); 8.32 ms on NPU |
| **iGPU** (`GPU.0`) | ISNet-anime matting → transparent PNG | 99.96 ms (M-5 probe); `--no-matting` turns it off |
| **RTX5080** | Custom SDXL UNet × 20 steps (CFG) + Euler + VAE decode | 10.65–10.96 s/image with `--fast` |

### Planned (not implemented)

| Stage | Intended HW | Status |
|---|---|---|
| Custom tag-generation LM (text → danbooru tags) | GPU first (LM-specific FP32 cuBLAS + its own kernels in `src/infer/bitnet_gpu.cu`; the diffusion kernels are not reused) / CPU possible / NPU excluded | Model and C++ inference exist (GPU forward seq8 2.43 ms) but it is **not wired into the pipeline**; the prompt goes straight to the text encoders (`backend_image_generator.hpp` → `sdxl_backend.hpp`) |
| WD14 tagging → feedback | CPU | WD14 glue exists in the Phase 1 skeleton (105.3 ms in C++), not wired into the shipping path |
| img2img | iGPU VAE encode (probe5: 79 ms) | Not implemented (`/v1/images/edits` returns 501) |

### Device selection (probe measurements)

| Model | CPU | iGPU | NPU | Chosen |
|---|---|---|---|---|
| CLIP-L text encoder | 20 ms | 14 ms | **7.85 ms** | NPU |
| WD14 SwinV2 tagger | **101 ms** | 104 ms | 268 ms | CPU |
| VAE decode (stub) | 126 ms | 995 ms | – | RTX5080 |
| VAE encode (img2img) | 117 ms | **79 ms** | – | iGPU |
| Matting ISNet-anime | 204.20 ms | **99.96 ms** | 142.96 ms | iGPU |

---

## Measurement baseline

| Metric | Value | Source |
|---|---|---|
| SDXL 20 steps 1024² — diffusers (CFG, text encoding included) | **3.80 s** / 5.3 it/s / peak VRAM 10.49 GB | probe10 |
| Custom path, shipping `--fast`, g=7.5 (UNet + VAE decode, text encoding excluded) | **10.65–10.96 s/image** (2 runs × 2 images) | 2026-09-04, `prof_arena_e2e`, HEAD `e17374c`; raw logs `docs/logs/g10k-baseline/e2e_run{1,2}.log` |
| Custom path, default (non-`--fast`), g=7.5 | 14.1 s (14066.9 ms) — **reference value with `DOLLAMA_PROFILE=1` instrumentation on**; `--fast` in the same run: 10799.8 ms | 2026-09-16, `test_diffusion_batch2` (DB2_BENCH 20 steps), `DOLLAMA_CONV_BATCH=0` (same per-n path as today's default), HEAD `c0f6d8f` (before E-0); raw log `docs/logs/g10k-e2e/t7_p2_convbatch0.log` |
| Custom path VRAM peak (`--fast`) | 13,397 MB (`cudaMemGetInfo` used, whole device, incl. CUDA context and VRAM arena) | same e2e logs |
| Old value (reference only) | ~~11.30 s~~ — non-CFG, B=1, `generate()` path, 2026-06-22; **stale, not the shipping path** | measurements-log |
| CLIP-L NPU / WD14 CPU (C++) | 7.82 ms / 105.3 ms (median, N=100) | test_clip / test_wd14 |
| Multi-frame parallelism (full-size stubs) | 0.258 fps = 98% of GPU bound | test_multi_frame_pipeline |

**Phase 2 first-version kernels** (the current default path has since replaced large GEMMs with cuBLAS, conv with im2col + GEMM, and attention
with flash + wmma): FP16 GEMM 1024³ 4730 GFLOPS / Conv2d direct 1807 GFLOPS / Attention self 1631 GFLOPS / SiLU-GeLU 544 GB/s /
GroupNorm 73 GB/s (the G-4k multi-block version used by the `--fast` epilogue path reaches 444.6 GB/s).

---

## Output size

**Fixed at 1024×1024 for now.** `--width` / `--height` and the HTTP `size` field are parsed, but anything other than 1024 is rejected with a
`runtime_error` (the diffusion loop is wired for a fixed `[1,4,128,128]` latent).

---

## Custom tag-generation LM (Phase 4)

A small decoder-only LLaMA-style model (`src/models/bitnet.hpp`, **32.98M** params, ~66 MB FP16) specialized for natural language → danbooru tags,
with a tag-level exact-match tokenizer (`src/io/tokenizer.hpp`).

- **Training is in PyTorch** (`scripts/train_bitnet.py`); **inference is in C++** (CPU / GPU / INT8: `src/infer/bitnet.hpp`, `bitnet_gpu.cu`,
  `bitnet_int8.hpp`). Logits corr 1.0 vs. the PyTorch golden; GPU forward is 46.8× (seq8) / 87.5× (seq63) faster than the CPU version.
- **Canonical model `[B-merge-at-A]`** (2026-07-03): input diversification (B) + identity conditioning (A) baked into the 33M model. All four promotion
  gates passed (single seed 20260620): identity retention 0.9807 / diverse_a set-F1 0.3332 / diverse_b set-F1 0.3804 / in-dist 0.4552 (A alone: retention 0.975).
- Not adopted: four distillation routes, and 33M→80M capacity (closed negative).
- **Not yet wired** into the txt2img pipeline. Ternary (b1.58) is a later compression experiment, not the goal (`ternary_gemm.cu` does not exist yet).
- **Quality scorer / feedback:** ScorerNet is wired into C++ (logging only). Q-2 split quality into a separate CLIP-embedding branch (QualityMLP,
  used by the Python-side reward, not wired into C++), which passed the signal gate (reward std 0.1038 / best−worst 0.3575).
  **F-0b (RAFT-SFT) was not adopted: no effect could be detected** — under the current reward design (7 of 8 anatomy axes barely move), a single seed,
  and insufficient power; it does not show the method is ineffective in general. Next levers: reward design, diagnosing the dead anatomy axes, etc.

---

## Implementation approach

Hand-write only the research core that drives the hardware; use standard header-only libraries for plumbing.

- **Used:** STL / CUDA Runtime + **cuBLAS** (default path for large GEMMs; `DOLLAMA_GEMM=wmma` switches to the custom wmma kernel) /
  custom CUDA kernels / **OpenVINO runtime** (NPU / iGPU / CPU inference glue) / **cpp-httplib** / **nlohmann/json**.
- **Not used (inference path):** PyTorch / LibTorch / diffusers / stable-diffusion.cpp / llama.cpp.
- `ws2_32` is linked on Windows for cpp-httplib; dollama's own code does not use Winsock2 directly.

---

## Build

On a fresh Windows box, `powershell -ExecutionPolicy Bypass -File install_windows.ps1` installs VS Build Tools / CUDA / OpenVINO / Python deps
(`-DryRun`, `-CheckOnly`, `-SkipCuda`, `-SkipSdk`, `-PythonExe <path>`, `-Force`) and prints a recommended `meson setup` command.

Manual prerequisites: Meson + Ninja, a C++20 compiler (tested with MSVC), CUDA Toolkit (requires 12.8+ for sm_120; tested with 13.3),
OpenVINO 2024.x+ runtime (**required for text-to-image**; without it the generator falls back to the golden-embedding path or the stub).

```bash
# Default SDK paths live in meson.build (CUDA v13.3 / openvino_2024); override with -D if yours differ
meson setup build \
  -Dgpu_sdk_root="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3" \
  -Dnpu_sdk_root="C:/Program Files (x86)/Intel/openvino_2024"
meson compile -C build
meson test -C build
```

---

## Run

Default weight paths are relative to the repository root, so run from there.

```powershell
build\src\dollama.exe --prompt "1girl, solo, smile" --out out.png --seed 1234 --fast   # CLI, one image
build\src\dollama.exe --http --port 8080                                              # HTTP server on 127.0.0.1
.\run-ui.ps1                                                                           # Blazor UI (http://localhost:5074)
```

- Main flags: `--prompt` / `--negative` / `--out` / `--preset <name>` (default `illustrious-xl`, `base` = plain SDXL) / `--no-preset-prefix` /
  `--fast` / `--seed` / `--cfg` / `--steps` / `--no-matting`. The real txt2img path also needs `DOLLAMA_OV_TOKENIZERS_DLL` (path to the openvino_tokenizers DLL).
- HTTP: `POST /v1/images/generations` (optional `negative_prompt`, `steps`, `size` (1024x1024 only), `seed`, `guidance_scale`, `preset_prefix`,
  `loras`), `GET /health`, `GET /v1/models`. Spec: [`docs/http-api-spec.md`](docs/http-api-spec.md). UI: [`ui/README.md`](ui/README.md).

---

## Models

Weights are **not included** (`models/` is git-ignored). The placement table and conversion scripts
(`dollma_convert_sdxl_text_encoders.py`, `dollma_export_sdxl_preset.py`, `dollma_convert_matting.py`, `dollma_convert_scorer.py`,
`dollma_dump_{unet,vae}_golden.py`) are listed in the Japanese [README.md](README.md). Scripts under `scripts/archives/` are kept as records and are not meant to be re-run.

The project **code** is Apache-2.0, but each model's **weights are governed by their own licenses**; you are responsible for complying with them.
The authoritative list is [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Main entries:

| Model | Role | License |
|---|---|---|
| Stable Diffusion XL base 1.0 | Diffusion (UNet + VAE), text encoders for `--preset base` | CreativeML Open RAIL++-M |
| **Illustrious XL v0.1** (early-release-v0) | **Default preset** `illustrious-xl` | Fair AI Public License 1.0-SD (has a copyleft clause; weights are not redistributed) |
| NoobAI-XL 1.1 | Preset `noobai-xl` | Fair AI Public License 1.0-SD |
| Animagine XL 4.0 | Preset `animagine-xl-4` | CreativeML Open RAIL++-M |
| CLIP ViT-L/14 | Text encoder / aesthetic image embed | MIT |
| WD14 SwinV2 Tagger | Tagging (CPU), anatomy label teacher | Apache-2.0 |
| ISNet-anime | Matting (iGPU) | Apache-2.0 |
| waifu-scorer-v4-beta | Quality-scorer teacher (label generation only) | Apache-2.0 |
| deepghs/anime_aesthetic | Quality-scorer teacher candidate (has use-restriction clauses) | OpenRAIL |

---

## License

Licensed under the [Apache License 2.0](LICENSE). Third-party models and libraries: [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

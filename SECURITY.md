# Security Policy

**English** | **[日本語](SECURITY_jp.md)**

## Status

dollama is an **experimental research project**, under active development and
**not intended for production use**. It is provided "as is" (see [LICENSE](LICENSE)),
with no guarantee of timely fixes or backports.

## Supported Versions

Only the latest `main` branch is supported. There are no released/versioned
artifacts yet.

## Reporting a Vulnerability

Please report security issues **privately** — do not open a public issue.

- Preferred: GitHub **"Report a vulnerability"** (Security → Advisories) on this repository.
- Include: affected file/component, reproduction steps, and impact.

Reports are handled on a best-effort basis (this is a personal research project,
so response times may vary).

## Scope

In scope — first-party code in this repository:

- C++ core, custom CUDA kernels, and inference glue (`src/`)
- The planned HTTP server (`src/server/`, Phase 3) once implemented

Out of scope:

- Third-party **model weights** (SDXL, CLIP-L, WD14, Qwen2, …) and their licenses
- Third-party runtimes/libraries (OpenVINO, CUDA, cpp-httplib, nlohmann/json, …)
- Issues that require a malicious local environment you already fully control

## Operational Notes (please read before running)

- The planned HTTP server is **unauthenticated and not hardened**. Do **not**
  expose it to untrusted networks; bind to localhost for local experiments.
- The weight/model loader (e.g. safetensors) is **not** a security boundary.
  **Only load model files you trust** — malformed/untrusted files may be unsafe.
- Responsibility for generated content and model licensing lies with the user.

# AGENTS.md

Guidance for AI coding agents working in this repository.

## Project overview

`diffuse-rocm` is a focused diffusion inference engine for ROCm (Strix Halo /
gfx1151, RDNA 3.5, native BF16/FP16, 128GB unified RAM). It loads **one model at
a time** and exposes a small, explicit surface rather than a full framework like
ComfyUI.

- **Layout**:
  - `diffuse/` — server package: `__main__.py` + `cli.py` (CLI entrypoints),
    `api.py` (FastAPI `/text2image`), `runtime.py` (single-model runtime),
    `models/` (adapters + catalog + detect), `dit/` (per-model compute),
    `vae/` (shared Qwen-Image VAE), `utils/` (safetensors, lora, attention, device).
  - `scripts/` — model download helpers.
  - `tests/` — CLI + runtime + detection tests (no torch needed).
- Invocation: `python -m diffuse serve ...` and `python -m diffuse generate ...`.

## Critical constraints

1. **Never run `uv sync`.** It would replace/break the ROCm `torch` build that the
   maintainer installs directly into the venv. Use `uv pip install` instead
   (e.g. `uv pip install -e .`). `torch` is intentionally not listed in
   `pyproject.toml`.

2. **Never run the program yourself.** You are in a containerized environment that
   cannot run the project with real models — there is no GPU and not enough RAM.
   Do not attempt to start the server, run `generate`, or load a model.

3. **When unsure on the correct way to proceed, ask the user** rather than guessing or overthinking.

## Workflow

- Verify changes with the test suite: `.venv/bin/python -m pytest tests/ -q`
  (tests are designed to run without torch or real weights).
- Do not run heavy/compute-heavy commands.

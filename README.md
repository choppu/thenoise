# diffuse-rocm

A focused diffusion inference engine for **ROCm**, targeting a Strix Halo APU
(**gfx1151**, RDNA 3.5, native BF16/FP16, 128GB unified RAM).

This is deliberately a *focused server* — it loads **one model at a time** and exposes a
small, explicit surface — not a full framework like ComfyUI.

## Scope

- **Krea 2 (K2)** — text-to-image (single-stream MMDiT + Qwen3-VL conditioner + Qwen-Image VAE)
- **Anima** (Cosmos-Predict2 2B text2image) — MiniTrainDiT + Qwen3-0.6B + LLM Adapter + Qwen-Image VAE
- BF16 inference only. No fp8 / other quant formats.
- No video support.

## Why PyTorch + an own implementation?

Diffusion DiTs are memory-bandwidth-bound, and Strix Halo's unified 128GB RAM is an
ideal target (everything fits in shared memory, no paging). PyTorch on ROCm +
`torch.compile` (inductor) gives near-optimal bandwidth-bound performance without
hand-written kernels. The "lightweight" part is the service surface, not the compute
layer — so low-level HIP/MIOpen kernels are not justified.

The model code is an **own implementation** (derived from musubi-tuner / sd-scripts)
so it can be optimized freely. It is organized so that each model is independently
optimizable, while genuinely-shared pieces (the Qwen-Image VAE, safetensors loader,
LoRA merge, attention) are implemented once.

## Layout

```
diffuse/              server package (FastAPI app, config, runtime, adapters)
  runtime.py          single-model runtime (loads one model, swaps on reload)
  api.py              generic FastAPI /text2image surface
  cli.py + server.py  CLI entrypoints: serve / generate
  models/             model adapters (DiffusionModel ABC + catalog + detect)
    base.py           detect/load/generate interface
    krea2.py / anima.py
  dit/                per-model compute (independently optimizable)
    krea2/            Krea2 MMDiT + Qwen3-VL conditioner + sampler
    anima/            Anima DiT + LLM Adapter + strategies + tokenizer configs
  vae/                shared Qwen-Image VAE (single implementation)
  utils/              shared infra: safetensors, lora, attention, device
  networks/           LoRA network types (LoHa/LoKr marker)
scripts/              download_krea2.py, download_anima.py
tests/                config + runtime + detection tests (no torch needed)
config.example.json   server knobs only (model paths come from the CLI)
```

## Setup

ROCm is installed per-venv (the maintainer handles this; `torch` is intentionally not
in `pyproject.toml`). Then:

```bash
python -m venv .venv && source .venv/bin/activate
uv pip install -e .            # installs deps, does NOT touch your ROCm torch
python scripts/download_krea2.py --out ./models/krea2
python scripts/download_anima.py --out ./models/anima --variant turbo-v1.0
```

Model checkpoints are passed on the CLI, **not** read from config. `config.json` only
holds server knobs (`host`, `port`, `device`, `dtype`).

## CLI

Serve one model over HTTP:

```bash
python -m diffuse.server serve --model krea2 \
  --dit ./models/krea2/turbo.safetensors \
  --vae ./models/krea2/split_files/vae/qwen_image_vae.safetensors \
  --text-encoder ./models/krea2/text_encoders/qwen3vl_4b_bf16.safetensors \
  -c config.json
```

Generate a single image without the server:

```bash
python -m diffuse.server generate --model anima \
  --dit ./models/anima/split_files/diffusion_models/anima-turbo-v1.0.safetensors \
  --vae ./models/anima/split_files/vae/qwen_image_vae.safetensors \
  --text-encoder ./models/anima/split_files/text_encoders/qwen_3_06b_base.safetensors \
  --prompt "a fox walking in the snow" --out /tmp/fox.png
```

Only one model is loaded at a time (loading another swaps the resident one).
`--model` is optional: when omitted, the model type is detected automatically from
the `--dit` checkpoint (each model class owns a `detect()` routine that inspects the
safetensors keys). If `--model` is given it is validated against the detected type.
The text encoder and VAE are assumed to match the detected model.

## API

- `GET /health` — status + the currently loaded model
- `POST /text2image` — `{prompt, negative_prompt, width, height, steps,
  guidance_scale, seed}`

Returns `{model, seed, image}` (a single base64 PNG). Per-model "advanced" sampler
params (`y1`/`y2`/`mu` for Krea2, `flow_shift` for Anima) are hard-coded defaults owned
by each model class and are **not** exposed. When a field is omitted, the model's own
default is used. If no model is loaded, `/text2image` returns 503.

Example:

```bash
curl -s localhost:8000/text2image \
  -H 'content-type: application/json' \
  -d '{"prompt":"a fox walking in the snow","steps":8}' \
  | python3 -c 'import sys,json,base64; d=json.load(sys.stdin);
                 open("/tmp/fox.png","wb").write(base64.b64decode(d["image"]))'
```

## Codebase notes

- **Shared, model-agnostic pieces are implemented once** (`diffuse/vae`, `diffuse/utils`):
  the Qwen-Image VAE (same for both models), the safetensors loader (with `rename_hook`
  for Anima), the LoRA merge, the attention backend (GQA-aware, serves both models),
  and device helpers.
- **Per-model compute lives in `diffuse/dit/{krea2,anima}`** so each can be tuned
  independently (e.g. Krea2's GQA attention vs Anima's strategies).
- **fp8 and block-swap are dropped** (bf16-only; 128GB unified RAM makes block swap
  pointless). Their code paths, monkey-patches and offloader implementations are removed.
- **LoRAs are kept**: standard LoRA (`lora_down`/`lora_up`) is merged at load time.
  LoHa/LoKr need the full `networks/` package (not ported from upstream) and are
  imported lazily — they raise if used.
- The Anima tokenizer configs (`qwen3_06b`, `t5_old`) are packaged as data under
  `diffuse/dit/anima/configs/` so the wheel is self-contained.

## Future work

- Validate outputs against known-good reference images (golden-image check) after the
  de-vendoring refactor
- `torch.compile` per-block tuning and benchmarking on the APU
- Keep the VAE/encoder resident on-device (Krea2's `sample()` shuttles the VAE)
- Port the full `networks/` package to enable LoHa/LoKr LoRAs

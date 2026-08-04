# TheNoise

A text-to-image diffusion inference engine. Tested on Strix Halo and Strix Point.

Loads one model at a time and generates images from text prompts. Available as a CLI tool, an HTTP API (with a simple web UI).

---

## Supported Models

### Krea 2 (K2)

Single-stream MMDiT with Qwen3-VL text encoder and Qwen-Image VAE.

| | |
|---|---|
| **Source** | `Comfy-Org/Krea-2` on Hugging Face |
| **Variants** | Turbo (default), RAW (`--include-raw`) |
| **Default steps** | 8 |
| **Default guidance** | 1.0 (CFG disabled by default) |
| **Default resolution** | 1024 × 1024 |
| **Size rounding** | Padded up to nearest multiple of 16 |

Download:

```bash
python scripts/download_krea2.py --out ./models/krea2
```

### Anima

Cosmos-Predict2 2B with Qwen3-0.6B text encoder, T5 cross-attention adapter, and Qwen-Image VAE.

| | |
|---|---|
| **Source** | `circlestone-labs/Anima` on Hugging Face |
| **Variants** | `aesthetic-v1.1` (default), `turbo-v1.0`, `base-v1.0`, `preview3-base`, and more |
| **Default steps** | 50 |
| **Default guidance** | 3.5 |
| **Default resolution** | 1024 × 1024 |
| **Size rounding** | Must be divisible by 32 |

Download:

```bash
python scripts/download_anima.py --out ./models/anima --variant aesthetic-v1.1
```

---

## Operation Modes

TheNoise can be used in three ways:

1. **CLI** — generate a single image from the command line
2. **HTTP server** — serve a model over HTTP with a JSON API
3. **Web UI** — a very basic browser interface served at `http://localhost:8000/` when running the server

The model type is **auto-detected** from the DiT checkpoint — no need to specify which model you are using.

---

## CLI

### Serve a model over HTTP

```bash
python -m thenoise serve \
  --dit ./models/krea2/diffusion_models/krea2_turbo_bf16.safetensors \
  --vae ./models/krea2/vae/qwen_image_vae.safetensors \
  --text-encoder ./models/krea2/text_encoders/qwen3vl_4b_bf16.safetensors \
  --host 127.0.0.1 --port 8000
```

### Generate a single image

```bash
python -m thenoise generate \
  --dit ./models/anima/split_files/diffusion_models/anima-turbo-v1.0.safetensors \
  --vae ./models/anima/split_files/vae/qwen_image_vae.safetensors \
  --text-encoder ./models/anima/split_files/text_encoders/qwen_3_06b_base.safetensors \
  --prompt "a fox walking in the snow" \
  --out /tmp/fox.png
```

### Load LoRAs

Place `.safetensors` LoRA files in a directory and point `--lora-dir` at it (both `serve` and `generate`). Then apply LoRAs per-request:

```bash
python -m thenoise generate \
  --dit ... --vae ... --text-encoder ... \
  --lora-dir ./models/loras \
  --prompt "a cyberpunk cityscape" \
  --lora "style-cyberpunk:0.8" \
  --lora "sub/detail-booster:0.5" \
  --out /tmp/city.png
```

LoRA format is `filename:weight` — the `.safetensors` extension is appended automatically. Omit `:weight` to use the default of `1.0`. LoRAs are switched in-memory without reloading the base model.

---

## HTTP API

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Web UI |
| `GET` | `/health` | Server status and loaded model |
| `POST` | `/text2image` | Generate an image |

### `/text2image` request body

All fields except `prompt` are optional. Omitted fields use the loaded model's defaults.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt` | `string` | *(required)* | Text prompt |
| `negative_prompt` | `string` | `""` | Negative prompt |
| `width` | `int` | model default | Output width in pixels |
| `height` | `int` | model default | Output height in pixels |
| `steps` | `int` | model default | Number of denoising steps |
| `guidance_scale` | `float` | model default | CFG scale (≤ 1.0 disables CFG) |
| `seed` | `int` | random | Random seed (`-1` for random) |
| `upscale` | `bool` | `false` | 2× latent-space upscale with refine denoise |
| `sampler` | `string` | `er_sde` | Denoising solver: `euler` or `er_sde` |
| `qwen_vae_enhance` | `bool` | `false` | Nyquist notch post-filter (removes 2px grid artifacts) |
| `film_grain` | `float` | `0.0` | Film grain strength, 0.0–10.0 |
| `sharpening` | `float` | `0.0` | RCAS sharpening strength, 0.0–1.0 |
| `lora_specs` | `string[]` | `null` | LoRA specs, e.g. `["style:0.8"]` |

### Response

Returns a PNG image directly (`Content-Type: image/png`).

### Example

```bash
curl -s localhost:8000/text2image \
  -H 'content-type: application/json' \
  -d '{"prompt":"a fox walking in the snow","steps":8}' \
  --output /tmp/fox.png
```

If no model is loaded, `/text2image` returns HTTP 503.

---

## CLI Parameters Reference

### Shared flags (`serve` and `generate`)

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--dit` | yes | — | Path to the DiT checkpoint (`.safetensors`) |
| `--vae` | yes | — | Path to the VAE checkpoint (`.safetensors`) |
| `--text-encoder` | yes | — | Path to the text encoder checkpoint (`.safetensors`) |
| `--lora-dir` | no | — | Directory containing LoRA `.safetensors` files |
| `--device` | no | `cuda` | Inference device (ROCm aliases `cuda` → `hip`) |

### `serve` only

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `127.0.0.1` | Bind host |
| `--port` | `8000` | Bind port |

### `generate` only

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--prompt` | yes | — | Text prompt |
| `--negative-prompt` | no | `""` | Negative prompt |
| `--width` | no | model default | Output width |
| `--height` | no | model default | Output height |
| `--steps` | no | model default | Denoising steps |
| `--guidance-scale` | no | model default | CFG scale |
| `--seed` | no | random | Random seed |
| `--out` | no | `out.png` | Output file path |
| `--lora` | no | — | LoRA to apply (repeatable, format: `file:weight`) |
| `--upscale` | no | off | 2× latent upscale with refine denoise |
| `--sampler` | no | `er_sde` | Solver: `euler` or `er_sde` |
| `--qwen-vae-enhance` | no | off | Nyquist notch post-filter |
| `--film-grain` | no | `0.0` | Film grain strength (0.0–10.0) |
| `--sharpening` | no | `0.0` | RCAS sharpening strength (0.0–1.0) |

---

## Model Defaults at a Glance

| Parameter | Krea 2 | Anima |
|-----------|--------|-------|
| Default steps | 8 | 50 |
| Default guidance | 1.0 (no CFG) | 3.5 |
| Default resolution | 1024 × 1024 | 1024 × 1024 |
| Default sampler | `er_sde` | `er_sde` |
| Size constraint | multiple of 16 | multiple of 32 |
| Precision | BF16 | BF16 |

---

## Setup

ROCm PyTorch must be installed in the virtual environment (the ROCm build of `torch` is not managed by `uv` — install it manually per the ROCm documentation). Then:

```bash
python -m venv .venv && source .venv/bin/activate
uv pip install -e .
```

Download one or both models using the scripts above. Everything is configured on the CLI — there is no config file.

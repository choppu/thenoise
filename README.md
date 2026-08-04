# TheNoise

A text-to-image diffusion inference engine. Tested on Strix Halo and Strix Point.

Loads one model at a time and generates images from text prompts. Available as a CLI tool, an HTTP API (with a simple web UI).

---

## Why? ComfyUI exists

Yes, and ComfyUI will always be better than this for the advanced user. This is good for the following scenarios:

1. You got a Strix Halo (congratulations!) and want to quickly start generating images
2. You don't want to care about "workflows"
3. You want to add an easy but powerful image generation endpoint for usage through other software
4. You want something targeted at your machine. Our goal is to optimize this for Strix Halo as much as possible.

---

## Setup

TheNoise ships with a bootstrap script (`thenoise.sh`) that handles everything:

1. Checks that [`uv`](https://github.com/astral-sh/uv) is installed (prints install instructions if not)
2. Creates a `.venv` with Python 3.13 (if it doesn't exist)
3. Installs the ROCm build of PyTorch
4. Installs the project in editable mode
5. Sets ROCm performance environment variables
6. Launches the project

**First run** (installs torch + project, then starts):

```bash
./thenoise.sh serve --dit ... --vae ... --text-encoder ...
```

**Subsequent runs** skip the torch install (detected via `import torch`).

By default the script targets `gfx1151` (Strix Halo). Override with the `GFX_ARCH` environment variable:

```bash
GFX_ARCH=gfx1150 ./thenoise.sh serve ...
```

---

## Performance

**First-run compilation:** the DiT model is compiled with `torch.compile` on load. The first
generation will be noticeably slower while the inductor traces and compiles kernels. 
You will also see some warnings on the console, these are normal.
All subsequent generations use the cached compiled code and run at full speed. 
Compilation is transparent — no configuration needed.

---

## Supported Models

At the moment, only Krea 2 and Anima are supported. New models will be added. PRs adding model support are welcome.

### Krea 2

Download:

```bash
python scripts/download_krea2.py --out ./models/krea2
```

### Anima


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
./thenoise.sh serve \
  --dit ./models/krea2/diffusion_models/krea2_turbo_bf16.safetensors \
  --vae ./models/krea2/vae/qwen_image_vae.safetensors \
  --text-encoder ./models/krea2/text_encoders/qwen3vl_4b_bf16.safetensors \
  --host 127.0.0.1 --port 8000
```

### Generate a single image

```bash
./thenoise.sh generate \
  --dit ./models/anima/split_files/diffusion_models/anima-turbo-v1.0.safetensors \
  --vae ./models/anima/split_files/vae/qwen_image_vae.safetensors \
  --text-encoder ./models/anima/split_files/text_encoders/qwen_3_06b_base.safetensors \
  --prompt "a fox walking in the snow" \
  --out /tmp/fox.png
```

### Load LoRAs

Place `.safetensors` LoRA files in a directory and point `--lora-dir` at it (both `serve` and `generate`). Then apply LoRAs per-request:

```bash
./thenoise.sh generate \
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

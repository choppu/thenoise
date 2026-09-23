# Qwen-Image 2.1

**One model for everything: t2i *and* edit.**

Qwen-Image 2.1 is the newest generation: a single-stream DiT that does
text-to-image **and** instruction-based editing from one checkpoint, with a full
Qwen3-VL-8B conditioner and the engine's first RGBA VAE (Wan 2.2 layout — the
pipeline composites alpha onto white at the boundaries that can't carry it).
The reference-latent KV cache is architectural here, so it is exact and on by
default.

- Text-to-image **and editing** (one DiT, both jobs)
- Reference-latent **KV cache** — exact, enabled by default
- int8-convrot variant available (~17 GB instead of ~32 GB total)
- Built-in defaults: **28 steps, guidance 1.0, euler**

> Commands below assume a dev checkout (`./thenoise.sh`, `.venv/bin/python`).
> On a [portable bundle](../setup.md) use `./bin/thenoise` and
> `./bin/python3` instead.

## Specs

| | |
|---|---|
| Architecture | Single-stream DiT (flow matching, causal text/reference prefix) |
| Download size | ~32 GB (bf16) · ~17 GB (int8-convrot) |
| VAE | Wan 2.2-layout 64-ch/16× RGBA VAE (never quantized) |
| Text encoder | Qwen3-VL-8B (LM *and* vision tower) |
| Editing | ✓ |
| KV cache | ✓ (exact, on by default) |
| Default settings | 1024×1024, 28 steps, guidance 1.0, euler |

## Performance (Strix Halo)

TBD (1024×768).

## Examples

*Generated:*

<img src="https://github.com/user-attachments/assets/TODO-qwen21-generate-1" alt="Qwen-Image 2.1 generated example 1" />

<img src="https://github.com/user-attachments/assets/TODO-qwen21-generate-2" alt="Qwen-Image 2.1 generated example 2" />

*Edited — instruction: "a fox wearing a red scarf":*

| Before | After |
|---|---|
| <img src="https://github.com/user-attachments/assets/TODO-qwen21-edit-before" alt="Qwen-Image 2.1 edit: input image" /> | <img src="https://github.com/user-attachments/assets/TODO-qwen21-edit-after" alt="Qwen-Image 2.1 edit: output image" /> |

## Download

```bash
.venv/bin/python scripts/download.py --model qwen-image-2.1
```

Fetches the bf16 DiT (~14 GB), the Qwen3-VL-8B text encoder (~17.5 GB) and the
VAE (~0.7 GB). Options:

```bash
# int8-convrot DiT AND text encoder instead (~17 GB total)
.venv/bin/python scripts/download.py --model qwen-image-2.1 --int8-convrot
```

Everything lands in `./models/qwen_image21/...`.

## Usage

Generate:

```bash
./thenoise.sh generate \
  --dit ./models/qwen_image21/diffusion_models/qwen_image_2.1_bf16.safetensors \
  --vae ./models/qwen_image21/vae/qwen_image_2.1_vae_bf16.safetensors \
  --text-encoder ./models/qwen_image21/text_encoders/qwen3vl_8b_bf16.safetensors \
  --prompt "a bioluminescent forest at night" \
  --out forest.png
```

Edit (`--image` is repeatable; the KV cache is on by default for this model):

```bash
./thenoise.sh edit \
  --dit ./models/qwen_image21/diffusion_models/qwen_image_2.1_bf16.safetensors \
  --vae ./models/qwen_image21/vae/qwen_image_2.1_vae_bf16.safetensors \
  --text-encoder ./models/qwen_image21/text_encoders/qwen3vl_8b_bf16.safetensors \
  --image fox.png \
  --prompt "a fox wearing a red scarf" \
  --out fox_edited.png
```

Serve over HTTP with the web UI (open <http://localhost:8000/>):

```bash
./thenoise.sh serve \
  --dit ./models/qwen_image21/diffusion_models/qwen_image_2.1_bf16.safetensors \
  --vae ./models/qwen_image21/vae/qwen_image_2.1_vae_bf16.safetensors \
  --text-encoder ./models/qwen_image21/text_encoders/qwen3vl_8b_bf16.safetensors \
  --host 127.0.0.1 --port 8000
```

For every flag (size, steps, seed, multi-image references, KV cache, LoRAs,
upscaling, post-processing), see the [CLI reference](../cli.md) and the
[HTTP API reference](../api.md).

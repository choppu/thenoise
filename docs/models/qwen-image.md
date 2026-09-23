# Qwen-Image / Qwen-Image-Edit

**The all-rounder of the 1.x line.**

Qwen-Image is a large dual-stream DiT with a Qwen2.5-VL-7B conditioner. The
image and edit variants are separate checkpoints that share one text encoder and
one VAE, so most people download both at once. It is the go-to when you want
strong prompt fidelity *and* instruction-based editing in one engine, and it
supports the reference-latent KV cache for faster edits.

- Text-to-image **and editing** (separate checkpoints, shared TE + VAE)
- Reference-latent **KV cache** (exact when the checkpoint carries the
  `index_timestep_zero` marker)
- Built-in defaults: **28 steps, guidance 2.5, euler** — the 4/8-step
  Lightning LoRAs in `models/loras/` exist if you want it faster

> Commands below assume a dev checkout (`./thenoise.sh`, `.venv/bin/python`).
> On a [portable bundle](../setup.md) use `./bin/thenoise` and
> `./bin/python3` instead.

## Specs

| | |
|---|---|
| Architecture | Dual-stream DiT (flow matching) |
| Download size | ~40 GB (both DiTs + shared TE + VAE, bf16) |
| VAE | Qwen-Image VAE (shared) |
| Text encoder | Qwen2.5-VL-7B (shared) |
| Editing | ✓ (Edit checkpoint) |
| KV cache | ✓ |
| Default settings | 1024×1024, 28 steps, guidance 2.5, euler |

The download script always fetches the **latest dated** checkpoints
(`qwen_image_2512` / `qwen_image_edit_2511`); older/unversioned releases are
skipped.

## Performance (Strix Halo)

Edit-2511 BF16, 1024×768:

- Generate @ 4 steps (Lightning LoRA): ~8 s
- Edit @ 4 steps (Lightning LoRA): ~19 s

## Examples

*Generated (Qwen-Image 2512):*

<img src="https://github.com/user-attachments/assets/TODO-qwen-image-generate-1" alt="Qwen-Image generated example 1" />

*Edited (Qwen-Image-Edit 2511) — instruction: "a fox wearing a red scarf":*

| Before | After |
|---|---|
| <img src="https://github.com/user-attachments/assets/TODO-qwen-image-edit-before" alt="Qwen-Image-Edit: input image" /> | <img src="https://github.com/user-attachments/assets/TODO-qwen-image-edit-after" alt="Qwen-Image-Edit: output image" /> |

## Download

```bash
# both checkpoints + shared text encoder + VAE (default)
.venv/bin/python scripts/download.py --model qwen-image

# only one of the DiTs (shared TE + VAE are still fetched)
.venv/bin/python scripts/download.py --model qwen-image --image-only
.venv/bin/python scripts/download.py --model qwen-image --edit-only

# int8-convrot DiTs instead of bf16 (swaps BOTH)
.venv/bin/python scripts/download.py --model qwen-image --int8-convrot
```

Everything lands in `./models/qwen_image/split_files/...`.

## Usage

Generate (Qwen-Image 2512):

```bash
./thenoise.sh generate \
  --dit ./models/qwen_image/split_files/diffusion_models/qwen_image_2512_bf16.safetensors \
  --vae ./models/qwen_image/split_files/vae/qwen_image_vae.safetensors \
  --text-encoder ./models/qwen_image/split_files/text_encoders/qwen_2.5_vl_7b.safetensors \
  --prompt "a paper crane flying over rice paddies at dawn" \
  --out crane.png
```

Faster generation with a Lightning LoRA (see
[LoRAs](../cli.md#loras)):

```bash
./thenoise.sh generate \
  --dit ./models/qwen_image/split_files/diffusion_models/qwen_image_2512_bf16.safetensors \
  --vae ./models/qwen_image/split_files/vae/qwen_image_vae.safetensors \
  --text-encoder ./models/qwen_image/split_files/text_encoders/qwen_2.5_vl_7b.safetensors \
  --lora-dir ./models/loras \
  --lora "Qwen-Image-2512-Lightning-8steps-V1.0-bf16:1.0" \
  --prompt "a paper crane flying over rice paddies at dawn" \
  --steps 8 \
  --out crane_fast.png
```

Edit (Qwen-Image-Edit 2511 — `--image` is repeatable):

```bash
./thenoise.sh edit \
  --dit ./models/qwen_image/split_files/diffusion_models/qwen_image_edit_2511_bf16.safetensors \
  --vae ./models/qwen_image/split_files/vae/qwen_image_vae.safetensors \
  --text-encoder ./models/qwen_image/split_files/text_encoders/qwen_2.5_vl_7b.safetensors \
  --image fox.png \
  --prompt "a fox wearing a red scarf" \
  --kv-cache \
  --out fox_edited.png
```

Serve over HTTP with the web UI (open <http://localhost:8000/>):

```bash
./thenoise.sh serve \
  --dit ./models/qwen_image/split_files/diffusion_models/qwen_image_edit_2511_bf16.safetensors \
  --vae ./models/qwen_image/split_files/vae/qwen_image_vae.safetensors \
  --text-encoder ./models/qwen_image/split_files/text_encoders/qwen_2.5_vl_7b.safetensors \
  --host 127.0.0.1 --port 8000
```

For every flag (size, steps, seed, multi-image references, KV cache, LoRAs,
upscaling, post-processing), see the [CLI reference](../cli.md) and the
[HTTP API reference](../api.md).

# TheNoise

A focused diffusion inference engine for image generation and editing on AMD
GPUs. TheNoise is built for the **Strix Halo** (gfx1151) and works on other
ROCm-capable AMD iGPUs and dGPUs (gfx1150, gfx1152). It is tuned to make good
use of the machine it runs on.

TheNoise loads one model at a time and generates images from text prompts.
Models that support it can also edit an existing image from a text instruction.
It is available as a CLI, an HTTP API, and a simple web UI.

<img width="2048" height="1066" alt="thenoise-main-screenshot" src="https://github.com/user-attachments/assets/afaf2d89-5857-4f50-995f-06fdf556a3c4" />

<details>
  <summary>Edit tab</summary>
  <img width="2048" height="1066" alt="thenoise-edit" src="https://github.com/user-attachments/assets/17efedda-b887-4f87-b0c0-c151619b19ac" />
</details>
<details>
  <summary>Upscale tab</summary> 
  <img width="2048" height="1066" alt="thenoise-upscaler" src="https://github.com/user-attachments/assets/f7ce89b7-fd25-4ad9-a3e4-d5e367530ab7" />
</details>

---

## How does it compare to ComfyUI?

ComfyUI is a general-purpose, node-based framework and remains the better
choice for advanced, customizable workflows. TheNoise is a focused engine, and
it is a good fit when:

- you are running a Strix Halo and would like to start generating images
  quickly, without learning a node graph first,
- you prefer a simple command line or a small UI over building and maintaining
  workflows,
- you want a small, stable image-generation endpoint that other software can
  call,
- you would rather have an engine optimized for your hardware than a
  general-purpose one.

## Performance

TheNoise and ComfyUI on the same Strix Halo (gfx1151, 128 GB unified). Times
are seconds per image, measured after a warmup run, and reported as
**TheNoise / ComfyUI**.

*Text to image (generation):*

| Model & settings | 768×1024 | 1024×1024 | 1536×2048 |
|---|---|---|---|
| Krea 2 Turbo · BF16 · 8 steps | TBD / TBD | TBD / TBD | TBD / TBD |
| Krea 2 Turbo · INT8-ConvRot · 8 steps | TBD / TBD | TBD / TBD | TBD / TBD |
| Anima Turbo · 8 steps | TBD / TBD | TBD / TBD | TBD / TBD |
| Z-Image Turbo · 8 steps | TBD / TBD | TBD / TBD | TBD / TBD |
| Flux.2 Klein 9B · INT8-ConvRot · 4 steps | TBD / TBD | TBD / TBD | TBD / TBD |
| Qwen-Image-Edit 2511 · BF16 · 4 steps | TBD / TBD | TBD / TBD | TBD / TBD |

*Image + instruction to edited image (editing):*

| Model & settings | 768×1024 | 1024×1024 | 1536×2048 |
|---|---|---|---|
| Flux.2 Klein 9B · INT8-ConvRot · 4 steps | TBD / TBD | TBD / TBD | TBD / TBD |
| Qwen-Image-Edit 2511 · BF16 · 4 steps | TBD / TBD | TBD / TBD | TBD / TBD |
| Qwen-Image 2.1 · BF16 · 4 steps | TBD / TBD | TBD / TBD | TBD / TBD |

<small>Exact test conditions (ComfyUI versions, settings, warmup protocol) will
be documented here once the runs are complete.</small>

## Supported models

| Model | On disk | Generate | Edit | Details |
|---|---|---|---|---|
| **Anima** — small and fast | ~5.4 GB | ✓ | — | [anima](docs/models/anima.md) |
| **Krea 2** — highest image quality | ~35 GB | ✓ | — | [krea2](docs/models/krea2.md) |
| **Z-Image / Z-Image-Turbo** — quality at 8 steps | ~21 GB | ✓ | — | [zimage](docs/models/zimage.md) |
| **Flux.2 Klein 4B / 9B** — 4 steps, with editing | 12 / 25 GB | ✓ | ✓ | [flux2-klein](docs/models/flux2-klein.md) |
| **Qwen-Image / Qwen-Image-Edit** — generation and editing | ~40 GB | ✓ | ✓ | [qwen-image](docs/models/qwen-image.md) |
| **Qwen-Image 2.1** — generation and editing in one model | ~32 GB | ✓ | ✓ | [qwen-image-2.1](docs/models/qwen-image-2.1.md) |

New models are added over time. PRs adding model support are welcome.

## Quick start

A short version of the full walkthrough in [docs/setup.md](docs/setup.md):

```bash
# 1. grab the portable bundle for your GPU from the releases page, extract it
tar -xzf thenoise-<version>-rocm<rocm>-gfx1151-x64.tar.gz
cd thenoise-<version>-rocm<rocm>-gfx1151-x64

# 2. download a model
./bin/python3 scripts/download.py --model anima

# 3. generate
./bin/thenoise generate \
  --dit ./models/anima/split_files/diffusion_models/anima-turbo-v1.0.safetensors \
  --vae ./models/anima/split_files/vae/qwen_image_vae.safetensors \
  --text-encoder ./models/anima/split_files/text_encoders/qwen_3_06b_base.safetensors \
  --prompt "a fox walking in the snow" --out fox.png
```

The portable bundle is self-contained: it needs no Python installation, build
tools, or administrator rights on the target machine.

## Using TheNoise

| If you want to… | Use |
|---|---|
| generate an image from the command line | the [CLI](docs/cli.md) — `generate`, `edit`, `upscale` |
| work through a browser | the web UI at `http://localhost:8000/` when running `serve` |
| call it from other software | the [HTTP API](docs/api.md) — `/text2image`, `/edit`, `/upscale` |

## Documentation

| | |
|---|---|
| [**Setup**](docs/setup.md) | from a released build to your first image |
| [**CLI reference**](docs/cli.md) | all `generate` / `edit` / `serve` / `upscale` flags, upscaling, LoRAs |
| [**HTTP API**](docs/api.md) | endpoints, request/response reference, curl and Python examples |
| [**Model pages**](docs/models/anima.md) | per-model presentation, examples, download options, usage |
| [**Development & Contribution**](docs/development.md) | building from source, tests, portable builds, adding models |

## Acknowledgments

This project incorporates code from:

1. [Musubi Tuner](https://github.com/kohya-ss/musubi-tuner)
2. [SD Scripts](https://github.com/kohya-ss/sd-scripts)
3. [SesquiLSR](https://github.com/LoganBooker/SesquiLSR)

plus smaller snippets from other sources or transitively inherited through the
above codebases.

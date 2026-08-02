# diffuse-rocm

A focused diffusion inference engine for **ROCm**, targeting a Strix Halo APU
(**gfx1151**, RDNA 3.5, native BF16/FP16, 128GB unified RAM).

This is deliberately a *focused server* — it loads a handful of models and exposes a
small, explicit API surface — not a full framework like ComfyUI.

## Scope

- **Krea 2 (K2)** — text-to-image (single-stream MMDiT + Qwen3-VL conditioner + Qwen-Image VAE)
- **Anima** (Cosmos-Predict2 2B text2image) — MiniTrainDiT + Qwen3-0.6B + LLM Adapter + Qwen-Image VAE
- BF16 inference only. No fp8 / other quant formats.
- No video support.

## Why PyTorch + vendored model code?

Diffusion DiTs are memory-bandwidth-bound, and Strix Halo's unified 128GB RAM is an
ideal target (everything fits in shared memory, no paging). PyTorch on ROCm +
`torch.compile` (inductor) gives near-optimal bandwidth-bound performance without
hand-written kernels. The "lightweight" part is the service surface, not the compute
layer — so low-level HIP/MIOpen kernels are not justified.

The model code is **vendored** (not installed as a dependency) so it can be optimized
freely:

- [`vendor/musubi_tuner/`](vendor/musubi_tuner/) — from
  [kohya-ss/musubi-tuner](https://github.com/kohya-ss/musubi-tuner) (Krea2)
- [`vendor/sd_scripts/`](vendor/sd_scripts/) — from
  [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) (Anima)

Both are pruned to the inference-only code paths for the models above.

## Layout

```
diffuse/              server package (FastAPI app, config, model registry, adapters)
vendor/musubi_tuner/  vendored, pruned model code (Krea2)
vendor/sd_scripts/    vendored, pruned model code (Anima)
scripts/              download_models.py, download_anima.py
tests/                config tests (no torch needed)
config.example.json   sample runtime config
```

## Setup

ROCm is installed per-venv (the maintainer handles this; `torch` is intentionally not
in `pyproject.toml`). Then:

```bash
python -m venv .venv && source .venv/bin/activate
uv pip install -e .            # installs deps, does NOT touch your ROCm torch
python scripts/download_krea2.py --out ./models/krea2
python scripts/download_anima.py --out ./models/anima --variant turbo-v1.0
cp config.example.json config.json
python -m diffuse.server -c config.json
```

## API

- `GET /health` — status + loaded models
- `POST /krea2/text2image` — `{prompt, negative_prompt, width, height, steps,
  guidance_scale, seed, y1, y2, mu, num_images}`
- `POST /anima/text2image` — `{prompt, negative_prompt, width, height, steps,
  guidance_scale, flow_shift, seed}`

Both return `{model, seed, images: [base64 PNG]}`. A model is loaded only if all three
of its paths are set in the config; otherwise the server answers `/health` and returns
503 for that model's endpoint.

Example:

```bash
curl -s localhost:8000/krea2/text2image \
  -H 'content-type: application/json' \
  -d '{"prompt":"a fox walking in the snow","steps":8}' \
  | python3 -c 'import sys,json,base64; d=json.load(sys.stdin);
                 open("/tmp/fox.png","wb").write(base64.b64decode(d["images"][0]))'
```

## Vendoring notes

**musubi_tuner (Krea2):**
- `qwen_image/qwen_image_utils.py` is pruned: the upstream imports `dataset/` and
  `flux/` for control-image bucketing and fp8 detection; this engine is bf16 text2image
  only, so those are replaced with a constant `is_fp8 = False`. The dead
  `preprocess_control_image` helper references the removed names but is never called.
- `modules/custom_offloading_utils.py` is a stub: block swap is unnecessary on 128GB
  unified RAM.

**sd_scripts (Anima):**
- Imports rewritten from `library.` to the `sd_scripts.` namespace.
- `anima_train_utils` is **not** vendored (it would drag in the whole training stack);
  the VAE is loaded directly via `qwen_image_autoencoder_kl.load_vae`.
- `strategy_base.py` / `strategy_anima.py` are pruned to the tokenize/encode strategies
  (the caching strategies and their training deps are dropped).
- `utils.py` is pruned to `setup_logging` (avoids cv2/torchvision/diffusers-scheduler
  imports).
- `lora_utils.py` has its `networks.loha`/`networks.lokr` imports made lazy (they
  transitively pull in `sdxl_original_unet`). When LoRA support lands, vendor the
  `networks/` package properly.
- `custom_offloading_utils.py` is a stub (`ModelOffloader` / `BlockSwapConfig`); block
  swap is unnecessary on 128GB unified RAM.
- The `configs/qwen3_06b/` and `configs/t5_old/` tokenizer configs are vendored (the
  text encoder's safetensors loading uses them).

## Future work

- LoRA loading (Krea2 supports it at load time; Anima needs the `networks/` package)
- `torch.compile` per-block tuning and benchmarking on the APU
- Keep the VAE/encoder resident on-device (Krea2's `sample()` shuttles the VAE)
- Validate outputs against known-good reference images

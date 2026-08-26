"""CLI generation: load one model, run one generation, save a single PNG.

Thin wrapper over the same adapter ``generate()`` methods the HTTP API uses, so
there is no logic drift between the two surfaces. The seed is resolved here (when
not given) so it can be reported for reproducibility.
"""
from __future__ import annotations

import logging
import os
import random
import sys

from .utils.paths import ensure_png_extension

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_generate(args) -> None:
    _MAX_DIM = 4096
    if args.width is not None and (args.width < 0 or args.width > _MAX_DIM):
        print(f"error: width must be between 0 and {_MAX_DIM} (got {args.width}).", file=sys.stderr)
        sys.exit(1)
    if args.height is not None and (args.height < 0 or args.height > _MAX_DIM):
        print(f"error: height must be between 0 and {_MAX_DIM} (got {args.height}).", file=sys.stderr)
        sys.exit(1)

    from .models.config import GenerateRequest
    from .runtime import Settings, ModelPaths, Runtime
    settings = Settings(device=args.device)

    runtime = Runtime(settings)

    # ``--pixel-upscaler`` is a one-shot convenience: a full path to the model.
    # Split it into ``upscaler_dir`` (server config) + ``pixel_upscaler`` (name,
    # sans suffix) so the runtime/controller only ever see the same directory +
    # name form as the ``serve``/API path.
    upscaler_dir = ""
    pixel_upscaler = None
    if args.pixel_upscaler:
        # ``or "."`` keeps a bare filename (no directory) usable: it then
        # resolves against the current working directory.
        upscaler_dir = os.path.dirname(args.pixel_upscaler) or "."
        pixel_upscaler = os.path.basename(args.pixel_upscaler)
        if pixel_upscaler.endswith(".safetensors"):
            pixel_upscaler = pixel_upscaler[: -len(".safetensors")]
        settings.upscaler_dir = upscaler_dir

    runtime.load(
        ModelPaths(
            dit_path=args.dit,
            vae_path=args.vae,
            text_encoder_path=args.text_encoder,
            lora_dir=args.lora_dir,
        ),
    )

    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    request = GenerateRequest(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=seed,
        upscale=args.upscale,
        upscale_factor=args.upscale_factor,
        upscale_type=args.upscale_type,
        sampler=args.sampler,
        qwen_vae_enhance=args.qwen_vae_enhance,
        film_grain=args.film_grain,
        sharpening=args.sharpening,
        lora_specs=args.lora or None,
        pixel_upscaler=pixel_upscaler,
    )

    # ``--image`` switches to instruction-based editing (same model, reference-
    # latent path) instead of text-to-image.
    if args.image:
        from PIL import Image

        request.image = Image.open(args.image).convert("RGB")
        image = runtime.pipeline.edit(request)
    else:
        image = runtime.pipeline.generate(request)

    # If the user omitted the output extension, PIL cannot infer a format.
    # Default to PNG so a bare --out like ``out`` (or ``dir/out``) still works.
    out_path = ensure_png_extension(args.out)

    image.save(out_path, pnginfo=getattr(image, "_pnginfo", None))
    logger.info("saved %s (model=%s, seed=%s)", out_path, runtime.model_name, seed)
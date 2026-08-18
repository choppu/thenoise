"""CLI pixel upscaling: load a pixel upscaler, upscale one image, save a PNG.

Model-free: needs no diffusion model (unlike ``generate``). Thin wrapper over the
same ``PixelUpscaleController`` the HTTP API uses, so there is no logic drift
between the two surfaces.
"""
from __future__ import annotations

import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_upscale(args) -> None:
    from PIL import Image
    from .runtime import Settings, Runtime

    # ``--pixel-upscaler`` is a one-shot convenience: a full path to the model.
    # Split it into ``upscaler_dir`` (server config) + name (sans suffix), the
    # same form the ``serve``/API path uses.
    upscaler_dir = os.path.dirname(args.pixel_upscaler)
    name = os.path.basename(args.pixel_upscaler)
    if name.endswith(".safetensors"):
        name = name[: -len(".safetensors")]

    settings = Settings(device=args.device, upscaler_dir=upscaler_dir)
    runtime = Runtime(settings)  # no load() — pixel upscaling is model-free

    image = Image.open(args.input).convert("RGB")
    out = runtime.upscaler.upscale(image, args.upscale_factor, name)

    out.save(args.out, pnginfo=getattr(out, "_pnginfo", None))
    logger.info("saved %s (upscaler=%s, factor=%s)", args.out, name, args.upscale_factor)

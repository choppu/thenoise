"""CLI generation: load one model, run one generation, save a single PNG.

Thin wrapper over the same adapter ``generate()`` methods the HTTP API uses, so
there is no logic drift between the two surfaces. The seed is resolved here (when
not given) so it can be reported for reproducibility.
"""
from __future__ import annotations

import logging
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_generate(args) -> None:
    from .runtime import Settings, ModelPaths, Runtime
    settings = Settings(device=args.device)

    runtime = Runtime(settings)
    runtime.load(
        ModelPaths(
            dit_path=args.dit,
            vae_path=args.vae,
            text_encoder_path=args.text_encoder,
            lora_weights=args.lora,
            lora_multipliers=[float(m) for m in (args.lora_multiplier or [])],
        ),
    )

    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    image = runtime.model.generate(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=seed,
        upscale=args.upscale,
    )

    image.save(args.out)
    logger.info("saved %s (model=%s, seed=%s)", args.out, runtime.model_name, seed)

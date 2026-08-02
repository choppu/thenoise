"""Krea 2 (K2) adapter.

Loads the MMDiT, Qwen-Image VAE and Qwen3-VL conditioner once and reuses them
across requests. Wraps the vendored ``musubi_tuner.krea2`` modules (Phase 5 moves
this into the own implementation).

Project decisions baked in here:
  * bf16 only (no fp8 / other formats)
  * SDPA attention (``attn_mode="torch"``) -- no flash-attn needed on ROCm
  * no block swap (everything fits in 128GB unified RAM)
  * LoRA merging at load time via ``krea2_utils.load_krea2_dit(lora_weights=...)``

The model class owns its defaults, including the "advanced" sampler parameters
(``y1``, ``y2``, ``mu``), which are hard-coded and NOT exposed to the API/CLI.

Inference is serialized with a lock: torch forward on a shared model is not
thread-safe, so concurrent requests are queued per model.
"""
from __future__ import annotations

import logging
import random
import threading
from typing import Optional

import torch
from PIL import Image
from safetensors.torch import load_file

from musubi_tuner.krea2 import krea2_utils
from musubi_tuner.krea2.krea2_sampling import encode_prompts, sample
from musubi_tuner.qwen_image.qwen_image_utils import load_vae

logger = logging.getLogger(__name__)


class Krea2Model:
    name = "krea2"

    # Model-owned defaults (incl. advanced sampler params -- not exposed to API/CLI).
    DEFAULT_STEPS = 8
    DEFAULT_GUIDANCE_SCALE = 0.0
    DEFAULT_WIDTH = 1024
    DEFAULT_HEIGHT = 1024
    DEFAULT_Y1 = 0.5
    DEFAULT_Y2 = 1.15
    DEFAULT_MU = None

    @staticmethod
    def detect(f) -> bool:
        """True if this handle is the Krea2 (single-stream MMDiT) DiT.

        Krea2 DiTs expose ``x_embedder.`` and ``txtfusion.`` and have no
        ``model.diffusion_model`` prefix (which is the Anima signature).
        """
        keys = f.keys()
        has_x_embedder = any(k.startswith("x_embedder.") for k in keys)
        has_txtfusion = any(k.startswith("txtfusion.") for k in keys)
        no_anima_prefix = not any(k.startswith("model.diffusion_model.") for k in keys)
        return has_x_embedder and has_txtfusion and no_anima_prefix

    def __init__(
        self,
        *,
        dit_path: str,
        vae_path: str,
        text_encoder_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        lora_weights: Optional[list] = None,
        lora_multipliers: Optional[list] = None,
    ):
        self.device = device
        self.dtype = dtype
        self._lock = threading.Lock()

        # LoRA state dicts are merged into the base weights at load time.
        lora_sds = [load_file(p) for p in (lora_weights or [])]
        mults = list(lora_multipliers) if lora_multipliers else [1.0] * len(lora_sds)

        logger.info("Loading Krea 2 DiT from %s", dit_path)
        self.dit = krea2_utils.load_krea2_dit(
            dit_path,
            device=device,
            dtype=dtype,
            attn_mode="torch",  # SDPA
            lora_weights=lora_sds or None,
            lora_multipliers=mults or None,
        )

        logger.info("Loading Krea 2 VAE from %s", vae_path)
        self.ae = load_vae(vae_path, input_channels=3, device="cpu", disable_mmap=True)
        self.ae = self.ae.to(dtype).eval().requires_grad_(False)

        logger.info("Loading Krea 2 text encoder from %s", text_encoder_path)
        self.encoder = krea2_utils.load_krea2_text_encoder(
            text_encoder_path, dtype=dtype, device=device
        )

        logger.info("Krea 2 model ready on %s (%s)", device, dtype)

    def generate(
        self,
        *,
        prompt: str,
        negative_prompt: str = "",
        width: Optional[int] = None,
        height: Optional[int] = None,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """Encode -> denoise -> decode. Returns a single PIL image."""
        width = width or self.DEFAULT_WIDTH
        height = height or self.DEFAULT_HEIGHT
        steps = steps or self.DEFAULT_STEPS
        guidance_scale = (
            self.DEFAULT_GUIDANCE_SCALE
            if guidance_scale is None
            else guidance_scale
        )

        with self._lock:
            cfg = guidance_scale > 1.0
            txt, txtmask, untxt, untxtmask = encode_prompts(
                self.encoder, [prompt], [negative_prompt], cfg=cfg
            )

            if seed is None:
                seed = random.randint(0, 2**32 - 1)

            images = sample(
                self.dit,
                self.ae,
                txt,
                txtmask,
                untxt=untxt,
                untxtmask=untxtmask,
                device=self.device,
                dtype=self.dtype,
                width=width,
                height=height,
                steps=steps,
                cfg_scale=guidance_scale,
                seed=seed,
                y1=self.DEFAULT_Y1,
                y2=self.DEFAULT_Y2,
                mu=self.DEFAULT_MU,
            )
            return images[0]

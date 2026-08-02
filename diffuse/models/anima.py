"""Anima (Cosmos-Predict2 2B text2image) adapter.

Loads the Anima DiT, Qwen3-0.6B text encoder + tokenizers, and Qwen-Image VAE once
and reuses them across requests. Uses the own implementation in ``diffuse.dit.anima``
and the shared VAE in ``diffuse.vae``.

Project decisions baked in here (same as Krea2):
  * bf16 only
  * SDPA attention (``attn_mode="torch"``, ``split_attn=True``)
  * no block swap / fp8 (128GB unified RAM, bf16-only)
  * text encoding via the AnimaTokenizeStrategy / AnimaTextEncodingStrategy classes
    called directly (no global strategy registry needed for a single instance)

The model class owns its defaults, including the "advanced" sampler parameter
(``flow_shift``), which is hard-coded and NOT exposed to the API/CLI.

Inference is serialized with a lock (torch forward on a shared model is not
thread-safe).
"""
from __future__ import annotations

import logging
import random
import threading
from typing import Optional

import torch
from PIL import Image
from tqdm import tqdm

from diffuse.dit.anima import utils as anima_utils
from diffuse.dit.anima import sampling as anima_sampling
from diffuse.dit.anima.strategy import AnimaTextEncodingStrategy, AnimaTokenizeStrategy
from diffuse.vae import load_vae

logger = logging.getLogger(__name__)

# Anima uses a Qwen-Image style 8x VAE, same as Krea2.
_VAE_SCALE = 8


class AnimaModel:
    name = "anima"

    # Model-owned defaults (incl. advanced sampler params -- not exposed to API/CLI).
    DEFAULT_STEPS = 50
    DEFAULT_GUIDANCE_SCALE = 3.5
    DEFAULT_WIDTH = 1024
    DEFAULT_HEIGHT = 1024
    DEFAULT_FLOW_SHIFT = 5.0

    @staticmethod
    def detect(f) -> bool:
        """True if this handle is the Anima DiT.

        Anima DiTs expose keys under the ``model.diffusion_model.`` prefix.
        """
        return any(k.startswith("model.diffusion_model.") for k in f.keys())

    def __init__(
        self,
        *,
        dit_path: str,
        vae_path: str,
        text_encoder_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.device = device
        self.dtype = dtype
        self._lock = threading.Lock()

        # DiT (bf16, SDPA attention).
        logger.info("Loading Anima DiT from %s", dit_path)
        self.dit = anima_utils.load_anima_model(
            device,
            dit_path,
            attn_mode="torch",
            split_attn=True,
            loading_device=device,
            dit_weight_dtype=dtype,
            fp8_scaled=False,
        )
        self.dit.eval().requires_grad_(False)

        # Text encoder (Qwen3-0.6B) + tokenizers.
        logger.info("Loading Anima text encoder from %s", text_encoder_path)
        self.text_encoder, self.qwen3_tokenizer = anima_utils.load_qwen3_text_encoder(
            text_encoder_path, dtype=dtype, device=device
        )
        self.text_encoder.eval().requires_grad_(False)
        self.t5_tokenizer = anima_utils.load_t5_tokenizer(None)

        # Tokenize / encode strategies (called directly, not through the global registry).
        self.tokenize_strategy = AnimaTokenizeStrategy(
            qwen3_tokenizer=self.qwen3_tokenizer,
            t5_tokenizer=self.t5_tokenizer,
            qwen3_max_length=512,
            t5_max_length=512,
        )
        self.encoding_strategy = AnimaTextEncodingStrategy()

        # VAE.
        logger.info("Loading Anima VAE from %s", vae_path)
        self.vae = load_vae(
            vae_path,
            device=device,
            disable_mmap=True,
            spatial_chunk_size=None,
            disable_cache=True,
        )
        self.vae = self.vae.to(dtype).eval().requires_grad_(False)

        logger.info("Anima model ready on %s (%s)", device, dtype)

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
        if height % 32 != 0 or width % 32 != 0:
            raise ValueError(f"height and width must be divisible by 32, got {height}x{width}")

        with self._lock:
            if seed is None:
                seed = random.randint(0, 2**32 - 1)

            cond_embed = self._encode_prompt(prompt)
            null_embed = self._encode_prompt(negative_prompt) if guidance_scale != 1.0 else cond_embed

            latent = self._denoise(cond_embed, null_embed, guidance_scale, steps, height, width, seed)
            pixels = self._decode(latent)
            return self._to_pil(pixels)

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        """Tokenize -> Qwen3 encode -> LLM-adapter cross-attention embedding (bf16)."""
        dev = torch.device(self.device)
        with torch.no_grad():
            tokens = self.tokenize_strategy.tokenize(prompt)
            # [prompt_embeds, qwen3_mask, t5_ids, t5_mask]
            embed = self.encoding_strategy.encode_tokens(self.tokenize_strategy, [self.text_encoder], tokens)
            crossattn_emb = self.dit._preprocess_text_embeds(
                source_hidden_states=embed[0].to(dev),
                target_input_ids=embed[2].to(dev),
                target_attention_mask=embed[3].to(dev),
                source_attention_mask=embed[1].to(dev),
            )
            crossattn_emb[~embed[3].bool()] = 0
            return crossattn_emb.to(torch.bfloat16)

    def _denoise(
        self,
        cond_embed: torch.Tensor,
        null_embed: torch.Tensor,
        guidance_scale: float,
        steps: int,
        height: int,
        width: int,
        seed: int,
    ) -> torch.Tensor:
        dev = torch.device(self.device)
        seed_g = torch.Generator(device=dev).manual_seed(seed)

        num_channels = self.dit.LATENT_CHANNELS
        shape = (1, num_channels, 1, height // _VAE_SCALE, width // _VAE_SCALE)
        latents = torch.randn(shape, generator=seed_g, device=dev, dtype=torch.bfloat16)
        padding_mask = torch.zeros(1, 1, height // _VAE_SCALE, width // _VAE_SCALE, dtype=torch.bfloat16, device=dev)

        timesteps, sigmas = anima_sampling.get_timesteps_sigmas(steps, self.DEFAULT_FLOW_SHIFT, dev)
        timesteps = (timesteps / 1000).to(dev, dtype=torch.bfloat16)

        do_cfg = guidance_scale != 1.0
        for i, t in tqdm(enumerate(timesteps), total=len(timesteps), desc="sampling"):
            t_expand = t.expand(latents.shape[0])
            with torch.no_grad():
                noise_pred = self.dit(latents, t_expand, cond_embed, padding_mask=padding_mask)
                if do_cfg:
                    uncond = self.dit(latents, t_expand, null_embed, padding_mask=padding_mask)
                    noise_pred = uncond + guidance_scale * (noise_pred - uncond)
            latents = anima_sampling.step(latents, noise_pred, sigmas, i).to(latents.dtype)
        return latents

    def _decode(self, latent: torch.Tensor) -> torch.Tensor:
        dev = torch.device(self.device)
        with torch.no_grad():
            pixels = self.vae.decode_to_pixels(latent.to(dev, dtype=self.vae.dtype))
        if pixels.ndim == 5:  # [B, C, 1, H, W] -> [B, C, H, W]
            pixels = pixels.squeeze(2)
        pixels = pixels.to("cpu", dtype=torch.float32)
        return pixels[0]  # [C, H, W] in [-1, 1]

    @staticmethod
    def _to_pil(sample: torch.Tensor) -> Image.Image:
        x = torch.clamp(sample, -1.0, 1.0)
        x = ((x + 1.0) * 127.5).to(torch.uint8).numpy()
        x = x.transpose(1, 2, 0)  # C, H, W -> H, W, C
        return Image.fromarray(x)

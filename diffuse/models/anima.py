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

The base ``DiffusionModel`` owns the pipeline (encode -> shared denoise loop ->
decode -> postprocess). Anima implements the model-specific kernels: its DiT
operates on a 5D latent ``[B, C, 1, H, W]`` (a frame axis of 1), so ``prepare_latent``
adds and ``finalize_latent`` removes that axis around the shared loop.

Inference is serialized with a lock (torch forward on a shared model is not
thread-safe).
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
from safetensors.torch import load_file

from diffuse.dit.anima import utils as anima_utils
from diffuse.dit.anima import sampling as anima_sampling
from diffuse.dit.anima.strategy import AnimaTextEncodingStrategy, AnimaTokenizeStrategy
from diffuse.models.base import Conditioning, DiffusionModel, Step

logger = logging.getLogger(__name__)


class AnimaModel(DiffusionModel):
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
        lora_weights: Optional[list] = None,
        lora_multipliers: Optional[list] = None,
    ):
        super().__init__(
            dit_path=dit_path,
            vae_path=vae_path,
            text_encoder_path=text_encoder_path,
            device=device,
            dtype=dtype,
            lora_weights=lora_weights,
            lora_multipliers=lora_multipliers,
        )

        # LoRA state dicts are merged into the base weights at load time.
        lora_sds = [load_file(p) for p in (lora_weights or [])]
        mults = list(lora_multipliers) if lora_multipliers else [1.0] * len(lora_sds)

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
            lora_weights_list=lora_sds or None,
            lora_multipliers=mults or None,
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

        logger.info("Anima model ready on %s (%s)", device, dtype)

    # ------------------------------------------------------------ kernels
    def encode_prompt(
        self,
        prompt: str,
        negative_prompt: str = "",
        *,
        guidance_scale: float,
    ) -> Conditioning:
        cond = self._encode_prompt(prompt)
        null = None
        if guidance_scale > 1.0:
            null = self._encode_prompt(negative_prompt)
        return Conditioning(cond=cond, null=null)

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

    def init_latents(self, height: int, width: int, seed: int) -> torch.Tensor:
        dev = torch.device(self.device)
        num_channels = self.dit.LATENT_CHANNELS
        shape = (1, num_channels, height // self._VAE_SCALE, width // self._VAE_SCALE)
        generator = torch.Generator(device=dev).manual_seed(seed)
        return torch.randn(shape, generator=generator, device=dev, dtype=self.dtype)

    def prepare_latent(
        self,
        latents: torch.Tensor,
        cond: Conditioning,
        steps: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        # The Anima DiT expects a frame axis: [B, C, H, W] -> [B, C, 1, H, W].
        return latents.unsqueeze(2)

    def schedule(self, steps: int, height: int, width: int) -> list[Step]:
        dev = torch.device(self.device)
        timesteps, sigmas = anima_sampling.get_timesteps_sigmas(steps, self.DEFAULT_FLOW_SHIFT, dev)
        timesteps = (timesteps / 1000).to(dev, dtype=self.dtype)
        sigmas = sigmas.to(dev)
        return [
            Step(t=timesteps[i], delta=sigmas[i] - sigmas[i + 1])
            for i in range(len(sigmas) - 1)
        ]

    def denoise_step(
        self,
        latents: torch.Tensor,
        t: torch.Tensor,
        cond: Conditioning,
        guidance_scale: float,
        i: int,
    ) -> torch.Tensor:
        dev = torch.device(self.device)
        t_expand = t.expand(latents.shape[0])
        padding_mask = torch.zeros(
            1, 1, latents.shape[3], latents.shape[4], dtype=torch.bfloat16, device=dev
        )
        with torch.no_grad():
            noise_pred = self.dit(latents, t_expand, cond.cond, padding_mask=padding_mask)
            if guidance_scale > 1.0 and cond.null is not None:
                uncond = self.dit(latents, t_expand, cond.null, padding_mask=padding_mask)
                noise_pred = uncond + guidance_scale * (noise_pred - uncond)
        return noise_pred

    def finalize_latent(self, latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
        # Drop the frame axis back to canonical 4D: [B, C, 1, H, W] -> [B, C, H, W].
        return latents.squeeze(2)

    def resolve_size(self, width: int, height: int) -> tuple[int, int]:
        if height % 32 != 0 or width % 32 != 0:
            raise ValueError(f"height and width must be divisible by 32, got {height}x{width}")
        return width, height

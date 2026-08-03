"""Abstract interface for a diffusion model adapter.

The base class owns the entire high-level pipeline — encode -> denoise -> decode
-> postprocess -> PIL — plus the shared Qwen-Image VAE, the inference lock, the
denoising loop, and the tensor/PIL conversions. Subclasses implement only the
model-specific kernels:

  * ``detect(f)``            — recognize this model's DiT from a safetensors handle.
  * ``encode_prompt(...)``   — text -> conditioning embeddings (cond + null).
  * ``init_latents(...)``    — seeded noise in the canonical 4D latent format.
  * ``prepare_latent(...)``  — canonical -> model-internal latent (once, pre-loop).
  * ``schedule(...)``        — the model's timestep/step-size schedule.
  * ``denoise_step(...)``    — one DiT forward + CFG, returning a velocity.
  * ``finalize_latent(...)`` — model-internal -> canonical latent (once, post-loop).
  * ``resolve_size(...)``    — per-model size rounding / validation.

Both models use the same Qwen-Image VAE (z_dim=16, spatial compression 8), so
``init_latents`` produces and ``finalize_latent`` returns the canonical latent
format ``[B, C, H, W]`` (4D). The VAE's ``decode_to_pixels`` accepts that
directly (it adds the frame axis internally). Model-internal reshaping (e.g.
Anima's frame axis, Krea2's patchify) lives in ``prepare_latent``/``finalize_latent``
and runs ONCE around the loop, so the per-step ``denoise_step`` never re-converts
the latent.

The shared loop is Euler integration of a flow ODE with CFG:

    x <- x - delta * velocity

where ``delta`` (the step size) and ``velocity`` come from the model's
``schedule`` and ``denoise_step`` respectively, and ``x`` is the model-internal
latent. Both current models reduce to exactly this update.

Post-processing runs on the decoded pixels as an fp32 GPU tensor (bf16's ~7-bit
mantissa causes banding in image filters); the tensor is only cast to uint8 and
moved to CPU inside ``_to_pil``. Metadata that must live on the final PNG is a
separate concern and is added later, after the PIL conversion.
"""
from __future__ import annotations

import random
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
from PIL import Image
from tqdm import tqdm

from diffuse.upscale import load_upscaler
from diffuse.vae import load_vae


@dataclass
class Conditioning:
    """Bundle of (un)conditional embeddings produced by ``encode_prompt``.

    ``null``/``null_mask`` are ``None`` when guidance is off (CFG disabled), so
    ``denoise_step`` can skip the unconditional forward.
    """

    cond: torch.Tensor
    cond_mask: Optional[torch.Tensor] = None
    null: Optional[torch.Tensor] = None
    null_mask: Optional[torch.Tensor] = None


@dataclass
class Step:
    """One denoising step: current timestep ``t`` and Euler step size ``delta``.

    The shared loop updates ``latents -= delta * velocity``; ``t`` is handed to
    ``denoise_step`` to build the DiT's per-step conditioning.
    """

    t: torch.Tensor
    delta: torch.Tensor


class DiffusionModel(ABC):
    """Base class for model adapters. Subclasses must set ``name``."""

    name: str = ""

    # Model-owned defaults. Advanced sampler params (y1/y2/mu, flow_shift) live
    # here too but are NOT exposed to the API/CLI.
    DEFAULT_WIDTH = 1024
    DEFAULT_HEIGHT = 1024
    DEFAULT_STEPS = 28
    DEFAULT_GUIDANCE_SCALE = 0.0

    # Canonical latent geometry (shared Qwen-Image VAE).
    LATENT_CHANNELS = 16
    _VAE_SCALE = 8

    # Optional latent upscaling + refine (SesquiLSR). ``UPSCALE_SCALE`` is the
    # latent upscale factor (2x). The refine is a single final low-strength
    # denoise step on the upscaled latent — the ``denoise=0.1`` second KSampler
    # pass in ComfyUI. Its noise level is taken from the schedule's last
    # timestep, which lands naturally around 0.1 for both models' schedules.
    UPSCALE_SCALE = 2

    @staticmethod
    @abstractmethod
    def detect(f) -> bool:
        """Return True if the open safetensors handle ``f`` is this model's DiT.

        ``f`` is a ``safetensors.safe_open(..., framework="pt")`` handle, opened
        ONCE by the runtime and passed to each class in turn — the file is never
        re-opened per class.
        """

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
        self.dit_path = dit_path
        self.vae_path = vae_path
        self.text_encoder_path = text_encoder_path
        self.lora_weights = lora_weights
        self.lora_multipliers = lora_multipliers
        self._lock = threading.Lock()

        # Lazy Sesqui latent upscaler (only loaded if ``upscale`` is requested).
        self._upscaler = None
        self._adaptor = None

        # Shared Qwen-Image VAE. Single-frame decode, so caching is disabled.
        self.vae = self._load_vae(vae_path)

    def _load_vae(self, vae_path: str):
        """Load the shared Qwen-Image VAE onto device/dtype."""
        vae = load_vae(
            vae_path,
            device=self.device,
            disable_mmap=True,
            disable_cache=True,
        )
        return vae.to(self.dtype).eval().requires_grad_(False)

    # ------------------------------------------------------------------ hooks
    @abstractmethod
    def encode_prompt(
        self,
        prompt: str,
        negative_prompt: str = "",
        *,
        guidance_scale: float,
    ) -> Conditioning:
        """Tokenize + encode prompt (and negative) into conditioning."""

    @abstractmethod
    def init_latents(self, height: int, width: int, seed: int) -> torch.Tensor:
        """Seed the canonical 4D latent ``[B, C, H//8, W//8]``."""

    def prepare_latent(
        self,
        latents: torch.Tensor,
        cond: Conditioning,
        steps: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Canonical -> model-internal latent. Runs ONCE before the loop.

        Override for reshaping (e.g. Krea2 patchify, Anima frame axis); default
        is the identity (canonical == internal).
        """
        return latents

    @abstractmethod
    def schedule(self, steps: int, height: int, width: int) -> list[Step]:
        """Build the model's denoising schedule (one ``Step`` per iteration)."""

    @abstractmethod
    def denoise_step(
        self,
        latents: torch.Tensor,
        t: torch.Tensor,
        cond: Conditioning,
        guidance_scale: float,
        i: int,
    ) -> torch.Tensor:
        """One DiT forward (+ CFG) returning the velocity in internal form."""

    def finalize_latent(
        self,
        latents: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Model-internal -> canonical 4D latent. Runs ONCE after the loop.

        Override to invert ``prepare_latent``; default is the identity.
        """
        return latents

    def resolve_size(self, width: int, height: int) -> tuple[int, int]:
        """Return the effective (width, height). Override to round/validate."""
        return width, height

    # ------------------------------------------------------------ pipeline
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
        upscale: bool = False,
    ) -> Image.Image:
        """Encode -> denoise -> decode -> postprocess. Returns a single PIL image.

        With ``upscale=True``, the canonical latent is upscaled 2x in latent
        space (SesquiLSR) after the main denoise loop and given a short
        low-strength refine denoise before decoding, doubling the output size.
        """
        width = width or self.DEFAULT_WIDTH
        height = height or self.DEFAULT_HEIGHT
        steps = steps or self.DEFAULT_STEPS
        guidance_scale = (
            self.DEFAULT_GUIDANCE_SCALE
            if guidance_scale is None
            else guidance_scale
        )
        width, height = self.resolve_size(width, height)

        with self._lock:
            if seed is None:
                seed = random.randint(0, 2**32 - 1)

            with torch.no_grad():
                cond = self.encode_prompt(prompt, negative_prompt, guidance_scale=guidance_scale)
                latents = self._denoise(cond, steps, height, width, seed, guidance_scale)
                if upscale:
                    latents = self._upscale_and_refine(
                        latents, cond, steps, height, width, seed, guidance_scale
                    )
                pixels = self.decode(latents)                 # fp32 GPU tensor [C,H,W]
                pixels = self.postprocess(pixels)             # tensor filters (hook)
                return self._to_pil(pixels)                   # final uint8 -> PIL

    def _denoise(
        self,
        cond: Conditioning,
        steps: int,
        height: int,
        width: int,
        seed: int,
        guidance_scale: float,
    ) -> torch.Tensor:
        """Shared denoising loop: Euler integration of the flow ODE with CFG.

        Iterates the model's ``schedule``, calling ``denoise_step`` for each
        velocity, and applies ``x <- x - delta * v``. Integration runs in fp32
        (precise, cheap) and is cast back to the latent dtype each step.
        """
        latents = self.init_latents(height, width, seed)
        x = self.prepare_latent(latents, cond, steps, height, width)
        dtype = x.dtype
        schedule = self.schedule(steps, height, width)
        for i, step in tqdm(enumerate(schedule), total=len(schedule), desc="sampling"):
            v = self.denoise_step(x, step.t, cond, guidance_scale, i)
            x = x.float() - step.delta * v.float()
            x = x.to(dtype)
        return self.finalize_latent(x, height, width)

    # ------------------------------------------------------------- upscaling
    def _load_upscaler(self):
        """Load the Sesqui latent upscaler (once, lazily, under the lock)."""
        if self._upscaler is None:
            self._upscaler, self._adaptor = load_upscaler(
                device=self.device, dtype=self.dtype
            )
        return self._upscaler, self._adaptor

    def _upscale_and_refine(
        self,
        latents: torch.Tensor,
        cond: Conditioning,
        steps: int,
        height: int,
        width: int,
        seed: int,
        guidance_scale: float,
    ) -> torch.Tensor:
        """Upscale the canonical latent ``UPSCALE_SCALE``x in latent space, then
        run a short low-strength refine denoise at the new size.

        Sesqui operates on raw VAE latents; the Wan21 adaptor converts the
        canonical (z-score) latent to/from that space. The refined result is the
        canonical latent at the upscaled spatial size, ready for ``decode``.
        """
        upscaler, adaptor = self._load_upscaler()
        scale = self.UPSCALE_SCALE
        z = latents.to(device=self.device, dtype=self.dtype)

        with torch.no_grad():
            # Adaptor math in fp32; the model runs in bf16.
            raw = adaptor.to_vae_latent(z).to(self.dtype)
            h, w = z.shape[-2:]
            raw_up = upscaler(raw, (scale * h, scale * w))
            z_up = adaptor.from_vae_latent(raw_up.float()).to(self.dtype)

        # One short low-strength refine denoise at the upscaled size.
        return self._refine(
            z_up, cond, steps, scale * height, scale * width, seed, guidance_scale
        )

    def _refine(
        self,
        z: torch.Tensor,
        cond: Conditioning,
        steps: int,
        height: int,
        width: int,
        seed: int,
        guidance_scale: float,
    ) -> torch.Tensor:
        """One low-strength refine denoise step on an already-clean latent.

        Mirrors the ``denoise=0.1`` second KSampler pass in ComfyUI. We run a
        single final step — the last step of the full schedule, whose timestep
        ``t`` lands naturally around 0.1 for both models' schedules. A small
        amount of noise at that level is added to the clean upscaled latent; the
        total Euler delta applied equals ``t``, so the update is self-consistent
        and removes exactly the added noise: ``x <- x - delta*v``.
        """
        full = self.schedule(steps, height, width)
        k = len(full) - 1
        sub = full[k:]  # the single last step of the schedule
        strength = float(sub[0].t)  # noise level == the step's timestep

        # Add a small amount of noise (seeded) to the clean upscaled latent.
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noise = torch.randn_like(z, generator=generator)
        noised = z + strength * noise

        x = self.prepare_latent(noised, cond, steps, height, width)
        dtype = x.dtype
        for i, step in tqdm(enumerate(sub), total=len(sub), desc="refining"):
            v = self.denoise_step(x, step.t, cond, guidance_scale, k + i)
            x = x.float() - step.delta * v.float()
            x = x.to(dtype)
        return self.finalize_latent(x, height, width)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Shared Qwen-Image VAE decode.

        Accepts the canonical 4D latent ``[B, C, H, W]`` (the VAE adds the frame
        axis internally) and returns pixels ``[C, H, W]`` in [-1, 1] as an fp32
        GPU tensor, ready for tensor post-processing.
        """
        dev = torch.device(self.device)
        with torch.no_grad():
            pixels = self.vae.decode_to_pixels(latents.to(dev, dtype=self.vae.dtype))
        if pixels.ndim == 5:  # [B, C, 1, H, W] -> [B, C, H, W]
            pixels = pixels.squeeze(2)
        pixels = pixels.to(torch.float32)
        return pixels[0]  # [C, H, W] in [-1, 1]

    def postprocess(self, pixels: torch.Tensor) -> torch.Tensor:
        """Tensor post-processing hook. Runs on the fp32 GPU pixels.

        Override or compose (e.g. color grading, sharpening, upscaling). Metadata
        that must live on the final PNG is added after ``_to_pil`` instead.
        """
        return pixels

    @staticmethod
    def _to_pil(pixels: torch.Tensor) -> Image.Image:
        x = torch.clamp(pixels, -1.0, 1.0)
        x = ((x + 1.0) * 127.5).to(torch.uint8).cpu().numpy()
        x = x.transpose(1, 2, 0)  # C, H, W -> H, W, C
        return Image.fromarray(x)

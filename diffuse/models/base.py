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

Pipeline caching
----------------
Each stage of the generate pipeline is cached (single-entry, on-device tensors).
Cache keys are computed from *resolved* parameters (after defaults are applied).
Keys are nested: the sampling key embeds the prompt key, and the decode key
embeds the sampling key. This gives automatic cascade invalidation — a change
at any stage invalidates that stage and all downstream stages.

  Stage          | Cache key depends on                    | Cached value
  ---------------+-----------------------------------------+---------------
  Prompt         | prompt, negative_prompt, guidance_scale | Conditioning
  Sampling       | prompt_key + size, steps, seed, sampler, lora_specs | latents
  Upscale+refine | (no cache — thin middle layer)          | —
  VAE decode     | sampling_key (+ upscale constants)      | pixels (fp32)
  Postprocess    | (not cached — cheap)                    | —

LoRA switching
---------------
LoRAs are applied per-request via ``switch_loras()``. The base model is loaded
without any LoRA baked in. At request time, the requested LoRA(s) are loaded
from disk and their deltas are added to the model's parameters. When the next
request asks for different LoRAs, the old deltas are subtracted (undo) before
applying the new ones. This avoids reloading the entire model from disk.
"""
from __future__ import annotations

import glob
import os
import random
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from safetensors.torch import load_file
from tqdm import tqdm

from diffuse.upscale import load_upscaler
from diffuse.utils.lora import apply_lora_to_model, undo_lora_on_model
from diffuse.vae import load_vae
from diffuse.postprocess.film_grain import film_grain
from diffuse.postprocess.nyquist import nyquist_notch
from diffuse.postprocess.rcas import rcas


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

    DEFAULT_WIDTH = 1024
    DEFAULT_HEIGHT = 1024
    DEFAULT_STEPS = 28
    DEFAULT_GUIDANCE_SCALE = 0.0

    SAMPLER = "er_sde"

    # Canonical latent geometry (shared Qwen-Image VAE).
    LATENT_CHANNELS = 16
    _VAE_SCALE = 8

    UPSCALE_SCALE = 2
    REFINE_STEPS = 1
    REFINE_DENOISE = 0.1

    @staticmethod
    @abstractmethod
    def detect(f) -> bool:
        """Return True if the open safetensors handle ``f`` is this model's DiT."""

    def __init__(
        self,
        *,
        dit_path: str,
        vae_path: str,
        text_encoder_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        lora_dir: Optional[str] = None,
    ):
        self.device = device
        self.dtype = dtype
        self.dit_path = dit_path
        self.vae_path = vae_path
        self.text_encoder_path = text_encoder_path
        self.lora_dir = lora_dir
        self._lock = threading.Lock()

        # LoRA state: undo deltas for clean switching
        self._undo_deltas: Dict[str, torch.Tensor] = {}
        self._active_lora_spec: Optional[str] = None

        # Pipeline result cache (single-entry per stage, on-device)
        self._cache_prompt_key: Optional[Tuple] = None
        self._cache_prompt_value: Optional[Conditioning] = None
        self._cache_sampling_key: Optional[Tuple] = None
        self._cache_sampling_value: Optional[torch.Tensor] = None
        self._cache_decode_key: Optional[Tuple] = None
        self._cache_decode_value: Optional[torch.Tensor] = None

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

    def _apply_loras_for_generation(
        self, lora_specs: Optional[List[str]]
    ) -> None:
        """Apply LoRAs before generation. Override in subclasses to pass the DiT.

        The base implementation is a no-op; subclasses call ``self.switch_loras(
        lora_specs, self.dit)`` (or equivalent) to target the actual model module.
        """
        pass

    # --------------------------------------------------------------- LoRA
    def _parse_lora_spec(self, spec: str) -> Tuple[str, float]:
        """Parse a 'filename:weight' spec into (filename, weight).

        Auto-appends .safetensors if the filename has no extension.
        """
        if ":" in spec:
            filename, weight_str = spec.rsplit(":", 1)
            weight = float(weight_str)
        else:
            filename = spec
            weight = 1.0

        # Auto-append .safetensors if no extension present
        if "." not in os.path.basename(filename):
            filename = filename + ".safetensors"

        return filename, weight

    def _resolve_lora_path(self, filename: str) -> str:
        """Resolve a LoRA filename to an absolute path, guarded against traversal.

        Subdirectories are allowed, but .. components that would escape lora_dir
        raise ValueError.
        """
        if not self.lora_dir:
            raise ValueError("lora_dir is not set")

        base = os.path.realpath(self.lora_dir)
        candidate = os.path.realpath(os.path.join(self.lora_dir, filename))

        if not candidate.startswith(base + os.sep) and candidate != base:
            raise ValueError(
                f"LoRA path escapes lora_dir: {filename!r} "
                f"(resolved to {candidate}, must stay under {base})"
            )

        return candidate

    def _get_lora_sd(self, filename: str) -> Dict[str, torch.Tensor]:
        """Load a LoRA state dict from disk."""
        filepath = self._resolve_lora_path(filename)

        logger = __import__("logging").getLogger(__name__)
        logger.info("Loading LoRA: %s", filepath)
        return load_file(filepath, device="cpu")

    def _make_lora_spec_hash(self, lora_specs: Optional[List[str]]) -> str:
        """Create a hash string for the current LoRA configuration."""
        if not lora_specs:
            return "__none__"
        return "|".join(sorted(lora_specs))

    def switch_loras(
        self,
        lora_specs: Optional[List[str]],
        dit: torch.nn.Module,
    ) -> None:
        """Switch active LoRAs on the DiT module (in-place, under the lock).

        Args:
            lora_specs: list of "filename:weight" strings, or None for base model.
            dit: the DiT model module whose parameters will be modified.

        Skips the switch if the requested config matches the current one.
        """
        new_spec = self._make_lora_spec_hash(lora_specs)
        if new_spec == self._active_lora_spec:
            return  # no-op: same LoRA config

        logger = __import__("logging").getLogger(__name__)

        # Undo any currently active LoRA
        if self._undo_deltas:
            logger.debug("Undoing previous LoRA (%d keys)", len(self._undo_deltas))
            undo_lora_on_model(dit, self._undo_deltas, torch.device(self.device))
            self._undo_deltas = {}

        # Apply new LoRAs
        if lora_specs and self.lora_dir is not None:
            lora_sds = []
            multipliers = []
            for spec in lora_specs:
                filename, weight = self._parse_lora_spec(spec)
                lora_sds.append(self._get_lora_sd(filename))
                multipliers.append(weight)

            self._undo_deltas = apply_lora_to_model(
                dit, lora_sds, multipliers, torch.device(self.device)
            )
            active_names = ", ".join(
                self._parse_lora_spec(s)[0] for s in lora_specs
            )
            logger.info("Applied LoRA(s): %s", active_names)
        else:
            logger.debug("Using base model (no LoRA)")

        self._active_lora_spec = new_spec

    def list_loras(self) -> List[str]:
        """List available LoRA filenames in the LoRA directory."""
        if not self.lora_dir:
            return []
        return [
            os.path.basename(p)
            for p in sorted(glob.glob(os.path.join(self.lora_dir, "*.safetensors")))
        ]

    def percent_to_sigma(self, percent: float) -> float:
        """Map a percent (0..1) to a sigma, used by the sampler's SNR offset.

        The ER-SDE solver needs ``sigma`` just below 1 (its ``sigma/(1-sigma)``
        blows up at exactly 1). Flow models override this with their shift
        the default is a linear fallback."""
        return 1.0 - percent

    # ---------------------------------------------------------- pipeline cache
    def _cache_key_prompt(
        self,
        prompt: str,
        negative_prompt: str,
        guidance_scale: float,
    ) -> Tuple:
        """Cache key for prompt conditioning."""
        return ("prompt", prompt, negative_prompt, guidance_scale)

    def _cache_key_sampling(
        self,
        prompt_key: Tuple,
        width: int,
        height: int,
        steps: int,
        seed: int,
        sampler: str,
        lora_specs: Optional[List[str]],
    ) -> Tuple:
        """Cache key for the sampling (denoise stage).

        Embeds the prompt key so any prompt/guidance change cascades.
        guidance_scale is not repeated — it is already in prompt_key.
        """
        return (
            "sampling",
            prompt_key,
            width,
            height,
            steps,
            seed,
            sampler,
            tuple(sorted(lora_specs)) if lora_specs else None,
        )

    def _cache_key_decode(
        self,
        sampling_key: Tuple,
        upscale: bool,
    ) -> Tuple:
        """Cache key for the VAE decode stage.

        Embeds the sampling key so any upstream change cascades.
        When upscale is True the upscale-and-refine pipeline produces
        different latents, so the upscale class-constants are added.
        """
        if not upscale:
            return ("decode", sampling_key)
        return (
            "decode_upscale",
            sampling_key,
            self.UPSCALE_SCALE,
            self.REFINE_STEPS,
            self.REFINE_DENOISE,
        )

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
        sampler: Optional[str] = None,
        qwen_vae_enhance: bool = False,
        film_grain: float = 0.0,
        sharpening: float = 0.0,
        lora_specs: Optional[List[str]] = None,
    ) -> Image.Image:
        """Encode -> denoise -> decode -> postprocess. Returns a single PIL image.

        Each pipeline stage is cached (single-entry). Cache keys are computed from
        *resolved* parameters so that defaults are accounted for. A change at any
        stage invalidates that stage and all downstream stages automatically via
        the nested key structure.
        """
        # --- resolve defaults (cache keys must use actual resolved values) ---
        width = width or self.DEFAULT_WIDTH
        height = height or self.DEFAULT_HEIGHT
        steps = steps or self.DEFAULT_STEPS
        guidance_scale = (
            self.DEFAULT_GUIDANCE_SCALE
            if guidance_scale is None
            else guidance_scale
        )
        width, height = self.resolve_size(width, height)
        effective_sampler = sampler or self.SAMPLER

        # --- compute cache keys (pure data, no model access) ---
        prompt_key = self._cache_key_prompt(
            prompt, negative_prompt, guidance_scale
        )
        sampling_key = self._cache_key_sampling(
            prompt_key, width, height, steps, seed,
            effective_sampler, lora_specs,
        )
        decode_key = self._cache_key_decode(
            sampling_key, upscale,
        )

        # --- locked section: cache checks + model access ---
        with self._lock:
            # Stage 1: prompt conditioning
            if self._cache_prompt_key == prompt_key:
                cond = self._cache_prompt_value
            else:
                cond = self.encode_prompt(
                    prompt, negative_prompt, guidance_scale=guidance_scale
                )
                self._cache_prompt_key = prompt_key
                self._cache_prompt_value = cond

            # Stage 2: sampling (denoise)
            if self._cache_sampling_key == sampling_key:
                latents = self._cache_sampling_value
            else:
                self._apply_loras_for_generation(lora_specs)
                if seed is None:
                    seed = random.randint(0, 2**32 - 1)
                with torch.no_grad():
                    latents = self._denoise(
                        cond, steps, height, width, seed,
                        guidance_scale, effective_sampler,
                    )
                self._cache_sampling_key = sampling_key
                self._cache_sampling_value = latents

            # Stage 3: upscale (if requested) — no cache, sits between
            # sampling and decode.  Uses cached latents from stage 2.
            if upscale:
                latents = self._upscale_and_refine(
                    latents, cond, steps, height, width, seed, guidance_scale
                )

            # Stage 4: VAE decode
            if self._cache_decode_key == decode_key:
                pixels = self._cache_decode_value
            else:
                pixels = self.decode(latents)  # fp32 GPU tensor [C,H,W]
                self._cache_decode_key = decode_key
                self._cache_decode_value = pixels

            # Stage 5: postprocess (cheap — not cached)
            pixels = self.postprocess(
                pixels,
                qwen_vae_enhance=qwen_vae_enhance,
                film_grain_strength=film_grain,
                sharpening=sharpening,
            )
            return self._to_pil(pixels)

    def _denoise(
        self,
        cond: Conditioning,
        steps: int,
        height: int,
        width: int,
        seed: int,
        guidance_scale: float,
        sampler: str,
    ) -> torch.Tensor:
        """Shared denoising loop over the model's ``schedule``.

        Dispatches to the selected solver (``euler`` or ``er_sde``); both call
        ``denoise_step`` once per schedule step. Integration runs in fp32
        (precise, cheap) and is cast back to the latent dtype each step.
        """
        sampler = sampler or self.SAMPLER
        if sampler not in ("euler", "er_sde"):
            raise ValueError(
                f"unknown sampler: {sampler!r} (choose 'euler' or 'er_sde')"
            )

        latents = self.init_latents(height, width, seed)
        x = self.prepare_latent(latents, cond, steps, height, width)
        dtype = x.dtype
        schedule = self.schedule(steps, height, width)

        if sampler == "er_sde":
            x = self._sampler_er_sde(
                x, schedule, cond, guidance_scale, seed, dtype
            )
        else:  # euler
            for i, step in tqdm(enumerate(schedule), total=len(schedule), desc="sampling"):
                v = self.denoise_step(x, step.t, cond, guidance_scale, i)
                x = x.float() - step.delta * v.float()
                x = x.to(dtype)

        return self.finalize_latent(x, height, width)

    def _sampler_er_sde(
        self,
        x: torch.Tensor,
        schedule: list[Step],
        cond: Conditioning,
        guidance_scale: float,
        seed: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """ER-SDE solver for flow (CONST) models.

        A higher-order stochastic solver, still one ``denoise_step`` per
        schedule step (same compute cost as Euler). ``denoised`` is the
        CONST-style x0 prediction ``x - sigma*v`` derived from the model's
        velocity ``v``. The sigmas are reconstructed from the schedule (t: 1->0,
        plus a trailing 0); the first sigma is nudged just below 1 via
        ``percent_to_sigma`` so ``sigma/(1-sigma)`` stays finite.
        """
        s_noise = 1.0

        def noise_scaler(t):
            return t * (torch.exp(t ** 0.3) + 10.0)

        sigmas = torch.tensor(
            [step.t for step in schedule] + [0.0],
            device=x.device,
            dtype=torch.float32,
        )
        if sigmas[0].item() >= 1.0:
            sigmas[0] = self.percent_to_sigma(1e-4)

        half_log_snrs = -torch.log(sigmas / (1.0 - sigmas))  # CONST: -logit(sigma)
        er_lambdas = (-half_log_snrs).exp()                    # sigma/(1-sigma)

        generator = torch.Generator(device=x.device).manual_seed(seed)
        num_points = 200.0
        point_indice = torch.arange(0, num_points, dtype=torch.float32, device=x.device)

        old_denoised = None
        old_denoised_d = None
        for i in tqdm(range(len(sigmas) - 1), desc="sampling"):
            sigma_i = sigmas[i]
            v = self.denoise_step(x, sigma_i.to(dtype), cond, guidance_scale, i)
            xf = x.float()
            denoised = xf - sigma_i * v.float()

            if sigmas[i + 1] == 0:
                x = denoised.to(dtype)
            else:
                er_lambda_s = er_lambdas[i]
                er_lambda_t = er_lambdas[i + 1]
                alpha_s = sigmas[i] / er_lambda_s
                alpha_t = sigmas[i + 1] / er_lambda_t
                r_alpha = alpha_t / alpha_s
                r = noise_scaler(er_lambda_t) / noise_scaler(er_lambda_s)

                # Stage 1 (Euler).
                xf = r_alpha * r * xf + alpha_t * (1.0 - r) * denoised

                stage_used = min(3, i + 1)
                if stage_used >= 2:
                    dt = er_lambda_t - er_lambda_s
                    lambda_step_size = -dt / num_points
                    lambda_pos = er_lambda_t + point_indice * lambda_step_size
                    scaled_pos = noise_scaler(lambda_pos)
                    s = torch.sum(1.0 / scaled_pos) * lambda_step_size
                    denoised_d = (denoised - old_denoised) / (
                        er_lambda_s - er_lambdas[i - 1]
                    )
                    xf = xf + alpha_t * (dt + s * noise_scaler(er_lambda_t)) * denoised_d

                    if stage_used >= 3:
                        s_u = torch.sum(
                            (lambda_pos - er_lambda_s) / scaled_pos
                        ) * lambda_step_size
                        denoised_u = (denoised_d - old_denoised_d) / (
                            (er_lambda_s - er_lambdas[i - 2]) / 2
                        )
                        xf = xf + alpha_t * (
                            (dt ** 2) / 2 + s_u * noise_scaler(er_lambda_t)
                        ) * denoised_u
                    old_denoised_d = denoised_d

                if s_noise > 0:
                    noise = torch.randn_like(xf, generator=generator)
                    noise_term = (
                        er_lambda_t ** 2 - er_lambda_s ** 2 * r ** 2
                    ).sqrt().nan_to_num(nan=0.0)
                    xf = xf + alpha_t * noise * s_noise * noise_term

                x = xf.to(dtype)

            old_denoised = denoised

        return x

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

        # One short low-strength refine denoise at the upscaled size. The refine
        # runs on an independent schedule (see ``_refine``), so the original
        # ``steps`` count is deliberately NOT forwarded.
        return self._refine(
            z_up, cond, scale * height, scale * width, seed, guidance_scale
        )

    def _refine(
        self,
        z: torch.Tensor,
        cond: Conditioning,
        height: int,
        width: int,
        seed: int,
        guidance_scale: float,
    ) -> torch.Tensor:
        """One low-strength refine denoise step on an already-clean latent."""
        refine_steps = self.REFINE_STEPS
        denoise = self.REFINE_DENOISE
        new_steps = int(refine_steps / denoise)  # int(1/0.1) = 10

        # Last ``refine_steps`` steps of an independent ``new_steps`` schedule.
        full = self.schedule(new_steps, height, width)
        sub = full[-refine_steps:]
        strength = float(sub[0].t)  # sigma_hat == the step's timestep

        # ComfyUI CONST noise scaling: x = sigma*noise + (1-sigma)*z.
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noise = torch.randn_like(z, generator=generator)
        noised = strength * noise + (1.0 - strength) * z

        x = self.prepare_latent(noised, cond, new_steps, height, width)
        dtype = x.dtype
        for i, step in tqdm(enumerate(sub), total=len(sub), desc="refining"):
            v = self.denoise_step(x, step.t, cond, guidance_scale, i)
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

    def postprocess(
        self,
        pixels: torch.Tensor,
        *,
        qwen_vae_enhance: bool = False,
        film_grain_strength: float = 0.0,
        sharpening: float = 0.0,
    ) -> torch.Tensor:
        """Tensor post-processing hook. Runs on the fp32 GPU pixels."""
        if qwen_vae_enhance:
            pixels = nyquist_notch(pixels)
        if sharpening > 0.0:
            pixels = rcas(pixels, strength=sharpening)
        if film_grain_strength > 0.0:
            pixels = film_grain(pixels, strength=film_grain_strength/10.0)
        return pixels

    @staticmethod
    def _to_pil(pixels: torch.Tensor) -> Image.Image:
        x = torch.clamp(pixels, -1.0, 1.0)
        x = ((x + 1.0) * 127.5).to(torch.uint8).cpu().numpy()
        x = x.transpose(1, 2, 0)  # C, H, W -> H, W, C
        return Image.fromarray(x)

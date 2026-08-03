"""SesquiLSR latent upscaler, vendored for diffuse-rocm.

Both models (Krea2, Anima) use the shared Qwen-Image VAE, so a single Wan21
upscaler covers the whole repo: it operates on the 16-channel, z-score-normalized
canonical latent. The weights (~6MB bf16) are committed under ``weights/``.

Usage:
    model, adaptor = load_upscaler(device="cuda", dtype=torch.bfloat16)
    raw = adaptor.to_vae_latent(latent)          # normalized -> raw VAE latent
    up  = model(raw, (2*h, 2*w))                 # 2x latent upscale
    out = adaptor.from_vae_latent(up)            # raw -> normalized
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch

from .inference_adaptors import LatentFormatAdaptor, make_wan21
from .upscaler import LatentUpscaler

logger = logging.getLogger(__name__)

_WEIGHT_FILE = Path(__file__).resolve().parent / "weights" / "upscaler_Wan21.safetensors"

# Wan21 checkpoint: 16-channel Qwen-Image / Wan / Anima latent.
_UPSCALER_CHANNELS = 16


def upscale_weight_path() -> Path:
    """Path to the committed Wan21 upscaler weights."""
    if not _WEIGHT_FILE.is_file():
        raise FileNotFoundError(
            f"upscaler weights not found at {_WEIGHT_FILE}; "
            "the package was not installed with its package-data"
        )
    return _WEIGHT_FILE


def load_upscaler(
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[LatentUpscaler, LatentFormatAdaptor]:
    """Load the Wan21 latent upscaler (bf16) and its Wan21 format adaptor.

    The state dict is shipped as bf16 to match the engine's bf16-only convention
    (the upstream README notes half-precision has no quality effect).
    """
    from safetensors.torch import load_file

    path = upscale_weight_path()
    logger.info("Loading Sesqui latent upscaler from %s", path)
    state_dict = load_file(str(path), device=str(device))

    model = LatentUpscaler(in_channels=_UPSCALER_CHANNELS)
    model.load_state_dict(state_dict)
    model.to(device=device, dtype=dtype).eval().requires_grad_(False)

    adaptor = make_wan21()
    logger.info("Latent upscaler ready on %s (%s)", device, dtype)
    return model, adaptor


__all__ = [
    "LatentUpscaler",
    "LatentFormatAdaptor",
    "make_wan21",
    "load_upscaler",
    "upscale_weight_path",
]

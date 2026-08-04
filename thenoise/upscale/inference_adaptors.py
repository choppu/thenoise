"""Latent-format adaptors for Sesqui inference — Wan21 subset.

Sesqui upscalers are trained on *raw* VAE latents. A ``LatentFormatAdaptor``
converts between a pipeline's external latent space (what the diffusion model
passes around) and the canonical raw VAE latent space.

thenoise's canonical latent is the *normalized* (per-channel z-score) latent
``(VAE_raw - mean) / std``, exactly the Qwen-Image / Wan21 / Anima latent format
that ``make_wan21`` expects.

Copied and trimmed from https://github.com/LoganBooker/SesquiLSR (MIT).
"""

# Copyright (c) 2025 LoganBooker. Licensed under the MIT License.
# Source: https://github.com/LoganBooker/SesquiLSR
from __future__ import annotations

import torch
from torch import Tensor


# Wan21 / QwenImage / Anima latents_mean / latents_std (per-channel z-score).
_DEFAULT_MEAN = [
    -0.7571, -0.7089, -0.9113,  0.1075, -0.1745,  0.9653, -0.1517,  1.5508,
     0.4134, -0.0715,  0.5517, -0.3632, -0.1922, -0.9497,  0.2503, -0.2921,
]
_DEFAULT_STD = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
]


class LatentFormatAdaptor:
    """Convert between an external pipeline's latent and VAE latent."""

    def __init__(self, external_channels: int, spatial_scale: int = 1):
        self.external_channels = external_channels
        self.spatial_scale = spatial_scale

    def to_vae_latent(self, z: Tensor) -> Tensor:
        """External/pipeline latent -> raw VAE latent."""
        return z.float()

    def from_vae_latent(self, z: Tensor) -> Tensor:
        """Raw VAE latent -> external/pipeline latent."""
        return z

    def vae_target_size(self, target_size: tuple[int, int]) -> tuple[int, int]:
        """Target size in external coords -> VAE-latent coords."""
        h, w = target_size
        s = self.spatial_scale
        return (h * s, w * s) if s != 1 else target_size


class _ZScoreAdaptor(LatentFormatAdaptor):
    """Per-channel z-score normalization (Wan21 / Anima / QwenImage).

    Pipeline latent = ``(VAE_raw - mean) / std``; Sesqui operates on ``VAE_raw``.
    """

    def __init__(self, external_channels: int, mean: Tensor, std: Tensor):
        super().__init__(external_channels=external_channels)
        self.mean = mean.clone().view(1, -1, 1, 1)
        self.std = std.clone().view(1, -1, 1, 1)

    def _cast(self, z: Tensor) -> tuple[Tensor, Tensor]:
        d = z.device
        return (
            self.mean.to(dtype=z.dtype, device=d),
            self.std.to(dtype=z.dtype, device=d),
        )

    def to_vae_latent(self, z: Tensor) -> Tensor:
        # Pipeline = (VAE_raw - mean) / std  ->  VAE_raw = z * std + mean
        m, s = self._cast(z)
        return z.float() * s + m

    def from_vae_latent(self, z: Tensor) -> Tensor:
        m, s = self._cast(z)
        return (z.float() - m) / s


def make_wan21(
    latents_mean: Tensor | None = None,
    latents_std: Tensor | None = None,
) -> LatentFormatAdaptor:
    """Wan21 / Anima / QwenImage — per-channel z-score.

    Matches thenoise's canonical latent format (the Qwen-Image VAE applies
    ``(z - mean) / std`` on encode and ``z / std + mean`` on decode).
    """
    if latents_mean is None:
        latents_mean = torch.tensor(_DEFAULT_MEAN)
    if latents_std is None:
        latents_std = torch.tensor(_DEFAULT_STD)
    return _ZScoreAdaptor(external_channels=16, mean=latents_mean, std=latents_std)


__all__ = ["LatentFormatAdaptor", "make_wan21"]

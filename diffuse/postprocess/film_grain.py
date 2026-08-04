"""Tensor post-processing filters for decoded pixel tensors.

Film Grain that adds luminance-only spatially-correlated noise, approximating
the look of analog film emulsion grain.

These functions operate on the fp32 GPU pixel tensor ``[C, H, W]`` produced by
``DiffusionModel.decode`` (values in ``[-1, 1]``) and return the same shape.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

# Gaussian 5x5 kernel (σ≈1.0), normalised so it sums to 1.
# Generated from the 1D kernel [1, 4, 6, 4, 1] / 16 outer-product.
_G = [
    [1, 4, 6, 4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1, 4, 6, 4, 1],
]
_G_SUM = sum(row for row in _G for row in row)  # 256.0


def film_grain(
    pixels: torch.Tensor,
    *,
    strength: float = 0.03,
    seed: int | None = None,
) -> torch.Tensor:
    """Add film-grain-like noise to the luminance channel of ``[C, H, W]`` pixels.

    The noise is generated as Gaussian random values, spatially correlated by a
    5×5 Gaussian blur, and added *only* to luminance (Rec. 601 weights). Adding
    a constant delta to luminance is equivalent to shifting all RGB channels by
    the same amount, so no full colour-space conversion is needed.

    Args:
        pixels:  ``[C, H, W]`` fp32 tensor in ``[-1, 1]``.
        strength: Noise amplitude. ``0.0`` means no grain; typical values are
                  ``0.01`` (barely visible) to ``0.08`` (pronounced).
        seed:     Optional RNG seed for reproducibility. If ``None`` a random
                  grain pattern is generated each call.

    Returns:
        A new ``[C, H, W]`` tensor with grain applied. Extra channels beyond
        RGB (e.g. alpha) pass through unchanged.
    """
    c, h, w = pixels.shape
    rgb = pixels[: min(3, c)]

    if strength == 0.0:
        return pixels

    # --- 1. Generate Gaussian noise [1, 1, H, W] ---------------------------
    noise = torch.empty(1, 1, h, w, dtype=pixels.dtype, device=pixels.device)
    if seed is not None:
        gen = torch.Generator(device=pixels.device)
        gen.manual_seed(seed)
        noise.normal_(generator=gen)
    else:
        noise.normal_()

    # --- 2. Spatially correlate via Gaussian blur (5x5) --------------------
    kernel = torch.tensor(_G, dtype=pixels.dtype, device=pixels.device) / _G_SUM
    kernel = kernel.view(1, 1, 5, 5)  # [1, 1, 5, 5]

    noise = F.pad(noise, (2, 2, 2, 2), mode="replicate")
    noise = F.conv2d(noise, kernel)  # [1, 1, H, W]

    # --- 3. Add scaled noise to luminance (shifts all RGB equally) ---------
    grain = noise * strength  # [1, 1, H, W]

    grained_rgb = rgb.unsqueeze(0) + grain  # [1, 3, H, W] (broadcasts over C)

    # --- 4. Reassemble with any extra channels ------------------------------
    if c > 3:
        return torch.cat([grained_rgb.squeeze(0), pixels[3:]], dim=0)
    return grained_rgb.squeeze(0)


__all__ = ["film_grain"]

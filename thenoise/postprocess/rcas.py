"""Tensor post-processing filters for decoded pixel tensors.

RCAS — Robust Contrast Adaptive Sharpening (AMD FidelityFX).

A single 5-tap cross filter that adapts its sharpening lobe to the local
contrast at each pixel.  Minimal halo artifacts, good general-purpose
sharpener with a single ``strength`` knob.

These functions operate on the fp32 GPU pixel tensor ``[C, H, W]`` produced by
``DiffusionModel.decode`` (values in ``[-1, 1]``) and return the same shape.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

# Maximum (negative) lobe magnitude — hard limit from the FSR reference.
_LOBE_MAX = -0.1875

# Small epsilon to avoid division by zero in the hit computation.
_EPS = 1e-6


def rcas(pixels: torch.Tensor, *, strength: float = 0.5) -> torch.Tensor:
    """Apply RCAS (Robust Contrast Adaptive Sharpening) to ``[C, H, W]`` pixels.

    Port of the AMD FidelityFX RCAS kernel.  For each pixel the five-tap
    cross (N, S, E, W, centre) is analysed: the local min / max across the
    cross *and* across colour channels determines a per-pixel sharpening
    ``lobe`` that is stronger in low-contrast regions and weaker near
    high-contrast edges (avoiding halos).

    Args:
        pixels:  ``[C, H, W]`` fp32 tensor in ``[-1, 1]``.
        strength: Sharpening multiplier. ``0.0`` is a no-op; typical values
                  are ``0.3`` (subtle) to ``0.8`` (aggressive). The internal
                  lobe is further clamped to ``[-0.1875, 0]``.

    Returns:
        A new ``[C, H, W]`` tensor with sharpening applied. Extra channels
        beyond RGB pass through unchanged.
    """
    c, h, w = pixels.shape
    rgb = pixels[:3 if c >= 3 else c]  # [3, H, W] when c>=3

    if strength == 0.0:
        return pixels

    # --- 1. Gather 5-tap cross with replicate padding ----------------------
    # Pad 1 pixel on each side; mode="replicate" ≈ "reflect" for this kernel.
    p = F.pad(rgb, (1, 1, 1, 1), mode="replicate")  # [3, H+2, W+2]

    n = p[:, 0:h, 1:w + 1]
    s = p[:, 2:h + 2, 1:w + 1]
    w_ = p[:, 1:h + 1, 0:w]
    e = p[:, 1:h + 1, 2:w + 2]
    center = rgb

    # --- 2. Local min / max across the 5 taps (per-channel) ----------------
    mn = _min(n, s, w_, e, center)
    mx = _max(n, s, w_, e, center)

    # --- 3. Compute the adaptive lobe --------------------------------------
    # hitMin = -mn / (4*mx),  hitMax = -(1-mx) / (4*(1-mn))
    # (reference uses [0,1]; formula is range-independent for linear data)
    hit_min = -mn / (mx * 4.0 + _EPS)
    hit_max = -(1.0 - mx) / (4.0 * (1.0 - mn) + _EPS)

    lobe = torch.max(hit_min, hit_max)  # [3, H, W]

    # Clamp lobe to the most conservative (smallest) value across channels.
    lobe = lobe.min(dim=0, keepdim=True).values  # [1, H, W]

    # --- 4. Apply strength and hard-clamp ----------------------------------
    lobe = lobe * strength
    lobe = lobe.clamp(_LOBE_MAX, 0.0)

    # --- 5. Blend: (center + lobe*(n+s+e+w)) / (1 + 4*lobe) ---------------
    norm = (lobe * 4.0 + 1.0).reciprocal()
    neighbors = (n + s + w_ + e) * lobe + center  # [3, H, W], lobe broadcasts
    sharpened = (neighbors * norm).clamp(-1.0, 1.0)

    # --- 6. Reassemble with any extra channels ------------------------------
    if c > 3:
        return torch.cat([sharpened, pixels[3:]], dim=0)
    return sharpened


def _min(*tensors: torch.Tensor) -> torch.Tensor:
    """Element-wise min across a variable number of tensors."""
    out = tensors[0]
    for t in tensors[1:]:
        out = torch.min(out, t)
    return out


def _max(*tensors: torch.Tensor) -> torch.Tensor:
    """Element-wise max across a variable number of tensors."""
    out = tensors[0]
    for t in tensors[1:]:
        out = torch.max(out, t)
    return out


__all__ = ["rcas"]

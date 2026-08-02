"""Vendored stub replacing sd-scripts' custom_offloading_utils.

Block swap (CPU offloading of DiT blocks) is intentionally NOT supported in this
engine. On Strix Halo (gfx1151, RDNA 3.5, 128GB unified RAM) the DiT, text encoder
and VAE fit in unified memory at once, so offloading DiT blocks to host RAM buys
nothing and only adds copy overhead.

anima_models.py references ``ModelOffloader`` in a runtime-evaluated attribute
annotation and constructs it only inside ``enable_block_swap`` (which is never
called here: ``blocks_to_swap`` stays 0). This stub exists so the name resolves;
actually using it raises.
"""
from __future__ import annotations

from typing import Any


class ModelOffloader:
    """Block swap offloader — intentionally unsupported in this engine."""

    def __init__(self, *args: Any, **kwargs: Any):  # pragma: no cover - never called
        raise NotImplementedError(
            "Block swap is not supported in the diffuse-rocm engine (128GB unified RAM makes "
            "it unnecessary). Vendor the real sd-scripts custom_offloading_utils.py if "
            "offloading is ever required."
        )


class BlockSwapConfig:
    """Placeholder for compatibility; block swap is disabled in this engine."""

    def __init__(self, *args: Any, **kwargs: Any):  # pragma: no cover - never called
        raise NotImplementedError("Block swap is not supported in the diffuse-rocm engine.")

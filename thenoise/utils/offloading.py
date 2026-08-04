"""Block-swap stubs.

Block swap (CPU offloading of DiT blocks) is intentionally unsupported: on Strix
Halo (gfx1151, 128GB unified RAM) the DiT, text encoder and VAE fit in unified
memory at once, so offloading DiT blocks to host RAM buys nothing and only adds
copy overhead.

Both model files import these names at module scope but only *use* them when
``blocks_to_swap > 0``, which this engine never sets. The stubs exist so the
imports resolve; actually using them raises.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class BlockSwapConfig:
    device: Optional[Any] = None
    supports_backward: bool = False
    use_pinned_memory: bool = False


class ModelOffloader:
    def __init__(self, *args: Any, **kwargs: Any):  # pragma: no cover - never called
        raise NotImplementedError(
            "Block swap is not supported in the thenoise engine (128GB unified RAM makes "
            "it unnecessary)."
        )


def create_offloader(*args, **kwargs):  # pragma: no cover - never called
    raise NotImplementedError(
        "Block swap is not supported in the thenoise engine (128GB unified RAM makes "
        "it unnecessary)."
    )

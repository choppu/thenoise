"""Vendored stub replacing musubi_tuner.modules.custom_offloading_utils.

Block swap (CPU offloading of DiT blocks) is intentionally NOT supported in this
engine. On Strix Halo (gfx1151, RDNA 3.5, 128GB unified RAM) the DiT, text encoder
and VAE all fit in unified memory at once, so offloading DiT blocks to host RAM buys
nothing and only adds copy overhead.

The real upstream file (~1100 lines) can be vendored back if offloading is ever
needed (e.g. to fit a larger model on a smaller card). krea2_mmdit.py imports
``BlockSwapConfig`` / ``create_offloader`` at module scope but only *uses* them when
``blocks_to_swap > 0``, which this engine never sets. These stubs exist only so the
import resolves; actually using them raises.
"""
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class BlockSwapConfig:
    """Placeholder for the upstream block-swap config.

    Not used by this engine (block swap is disabled). Kept import-compatible with
    krea2_mmdit.py's ``enable_block_swap`` signature.
    """

    device: Optional[Any] = None
    supports_backward: bool = False
    use_pinned_memory: bool = False


def create_offloader(*args, **kwargs):  # pragma: no cover - never called (block swap disabled)
    """Block swap is intentionally unsupported in this engine."""
    raise NotImplementedError(
        "Block swap is not supported in the diffuse-rocm engine (128GB unified RAM makes it "
        "unnecessary). Remove this raise and vendor the real custom_offloading_utils.py if "
        "offloading is ever required."
    )

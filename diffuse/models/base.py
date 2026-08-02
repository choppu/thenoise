"""Abstract interface for a diffusion model adapter.

A model class:
  * declares its ``name`` (the CLI/API-facing id),
  * owns a ``detect(f)`` routine that recognizes its DiT from an open
    safetensors handle,
  * implements ``generate()`` returning a single PIL image.

We detect **only the main model** (the DiT). Each model is paired with a specific
text encoder and VAE, so once the DiT is identified we assume the ``--text-encoder``
and ``--vae`` checkpoints are of the correct type — if they aren't, loading throws
and we fail anyway. There is no separate VAE/encoder detection.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from PIL import Image


class DiffusionModel(ABC):
    """Base class for model adapters. Subclasses must set ``name``."""

    name: str = ""

    # Model-owned defaults. Advanced sampler params (y1/y2/mu, flow_shift) live
    # here too but are NOT exposed to the API/CLI.
    DEFAULT_WIDTH = 1024
    DEFAULT_HEIGHT = 1024

    @staticmethod
    @abstractmethod
    def detect(f) -> bool:
        """Return True if the open safetensors handle ``f`` is this model's DiT.

        ``f`` is a ``safetensors.safe_open(..., framework="pt")`` handle, opened
        ONCE by the runtime and passed to each class in turn — the file is never
        re-opened per class.
        """

    @abstractmethod
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
    ) -> Image.Image:
        """Encode -> denoise -> decode. Returns a single PIL image."""

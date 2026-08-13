"""Common sampler API."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch

if TYPE_CHECKING:
    from thenoise.models.base import Conditioning, DiffusionModel


@dataclass
class Step:
    """One denoising step: current timestep ``t`` and Euler step size ``delta``.

    The shared integration updates ``latents -= delta * velocity``; ``t`` is handed
    to ``denoise_step`` to build the DiT's per-step conditioning.
    """

    t: torch.Tensor
    delta: torch.Tensor


class Sampler(ABC):
    """Base class for a denoising solver, bound to a model instance."""

    def __init__(self, model: "DiffusionModel"):
        self.model = model

    @abstractmethod
    def sample(
        self,
        x: torch.Tensor,
        schedule: List[Step],
        cond: "Conditioning",
        guidance_scale: float,
        seed: int,
        desc: str = "sampling",
    ) -> torch.Tensor:
        """Run one denoising pass over ``schedule``, returning the denoised latent.

        ``x`` is the model-internal latent produced by ``prepare_latent``; the return
        value is passed to ``finalize_latent``. Solvers may ignore ``seed`` (Euler).
        Integration runs in fp32 and is cast back to ``x.dtype`` as needed.
        """

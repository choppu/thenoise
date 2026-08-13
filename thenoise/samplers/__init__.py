"""Denoising solvers (samplers) for the diffusion pipeline.

A sampler owns one denoising pass over a schedule, calling the model's
``denoise_step`` exactly once per schedule step. Solver-specific state (e.g. ER-SDE's
higher-order terms, noise scaler, RNG) lives here rather than on the model, so the
model base class stays free of any particular integration scheme.

Each sampler is bound to a model instance (via ``Sampler(model)``) and duck-types the
methods it needs — ``denoise_step`` and (for ER-SDE) ``percent_to_sigma`` — so the
samplers never import the model classes.
"""
from __future__ import annotations

from typing import Dict, Type

from .base import Sampler, Step
from .euler import EulerSampler
from .er_sde import ErSdeSampler

#: name -> sampler class. New solvers register here.
SAMPLERS: Dict[str, Type[Sampler]] = {
    "euler": EulerSampler,
    "er_sde": ErSdeSampler,
}


def create_sampler(name: str, model) -> Sampler:
    """Instantiate the named sampler bound to ``model``.

    Raises ``ValueError`` for an unknown sampler name.
    """
    cls = SAMPLERS.get(name)
    if cls is None:
        raise ValueError(
            f"unknown sampler: {name!r} (choose {', '.join(sorted(SAMPLERS))})"
        )
    return cls(model)


__all__ = ["SAMPLERS", "Sampler", "Step", "create_sampler"]

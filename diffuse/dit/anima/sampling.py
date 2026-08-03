"""Anima flow-matching sampler helpers.

Only the sampling functions are used by the engine. The rotary-position-embedding
and adaptive-guidance helpers from the original (HunyuanImage-2.1-derived) module
are not used by the Anima inference path and are dropped.
"""
from __future__ import annotations

from typing import Tuple

import torch


def get_timesteps_sigmas(sampling_steps: int, shift: float, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate timesteps and sigmas for diffusion sampling.

    Args:
        sampling_steps: Number of sampling steps.
        shift: Sigma shift parameter for schedule modification.
        device: Target device for tensors.

    Returns:
        Tuple of (timesteps, sigmas) tensors.
    """
    sigmas = torch.linspace(1, 0, sampling_steps + 1)
    sigmas = (shift * sigmas) / (1 + (shift - 1) * sigmas)
    sigmas = sigmas.to(torch.float32)
    timesteps = (sigmas[:-1] * 1000).to(dtype=torch.float32, device=device)
    return timesteps, sigmas

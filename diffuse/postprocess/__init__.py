"""Tensor post-processing filters for decoded pixel tensors.

Filters operate on the fp32 GPU pixel tensor ``[C, H, W]`` produced by
``DiffusionModel.decode`` (values in ``[-1, 1]``) and return the same shape.
They are applied by the base model's ``postprocess`` hook.
"""
from __future__ import annotations

from .nyquist import nyquist_notch

__all__ = ["nyquist_notch"]

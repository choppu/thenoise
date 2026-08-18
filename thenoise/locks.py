"""Shared global inference lock.

Serializes all on-device model work across the process. Both the generate
pipeline (``PipelineController``) and the standalone pixel upscale controller
(``PixelUpscaleController``) acquire this same lock, so generation and upscaling
can never run concurrently with each other (or with themselves).

This lock must be process-global: the two controllers are separate objects but
they mutate the same on-device upscaler pool and model state, so a lock owned by
either one alone would not prevent cross-controller races.
"""
from __future__ import annotations

import threading

# One process-wide lock shared by every inference entry point.
inference_lock = threading.Lock()

__all__ = ["inference_lock"]

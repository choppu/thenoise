"""diffuse-rocm: focused diffusion inference engine for ROCm (Strix Halo / gfx1151).

This is deliberately a *focused server* (a few models, a small explicit API surface),
not a full framework like ComfyUI. The compute backend is PyTorch on ROCm; the model
code (``vendor/musubi_tuner``) is vendored so it can be optimized freely.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the vendored musubi_tuner tree importable as ``musubi_tuner.*``. It is NOT
# installed as a dependency -- these files are vendored under vendor/ so we can modify
# and optimize them without tracking an external package.
_VENDOR = Path(__file__).resolve().parent.parent / "vendor"
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

__version__ = "0.1.0"

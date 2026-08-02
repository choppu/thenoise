"""Runtime configuration: server knobs only.

Model checkpoints are NOT read from config — they come from the CLI
(``--dit`` / ``--vae`` / ``--text-encoder``). The config file (or env vars)
only hold serving/device settings.

Kept deliberately small -- the whole point of a focused engine is a limited,
explicit surface.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Settings:
    device: str = "cuda"      # ROCm torch aliases cuda -> hip
    dtype: str = "bfloat16"   # bf16 only (project decision)
    host: str = "127.0.0.1"
    port: int = 8000

    @property
    def torch_dtype(self):
        if self.dtype == "bfloat16":
            import torch
            return torch.bfloat16
        raise ValueError(f"unsupported dtype '{self.dtype}' (project is bf16-only)")


def load_settings(path: Optional[str] = None) -> Settings:
    s = Settings()
    if path:
        p = Path(path)
        if p.exists():
            data = json.loads(p.read_text())
            s.device = data.get("device", s.device)
            s.dtype = data.get("dtype", s.dtype)
            s.host = data.get("host", s.host)
            s.port = int(data.get("port", s.port))
        else:
            raise FileNotFoundError(f"config file not found: {path}")

    # Environment overrides (server knobs only; model paths are CLI-only).
    s.device = os.environ.get("DEVICE", s.device)
    s.host = os.environ.get("HOST", s.host)
    if os.environ.get("PORT"):
        s.port = int(os.environ["PORT"])
    return s

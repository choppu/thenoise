"""Runtime configuration: which models to load, where weights live, serving knobs.

Loaded from a JSON config file (see config.example.json) with environment-variable
overrides for the model paths. Kept deliberately small -- the whole point of a focused
engine is a limited, explicit surface.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Krea2Config:
    dit_path: str = ""                 # e.g. models/krea2/turbo.safetensors
    vae_path: str = ""
    text_encoder_path: str = ""
    lora_weights: list = field(default_factory=list)     # optional LoRA safetensors
    lora_multipliers: list = field(default_factory=list)

    @property
    def enabled(self) -> bool:
        return bool(self.dit_path and self.vae_path and self.text_encoder_path)


@dataclass
class AnimaConfig:
    dit_path: str = ""                 # e.g. models/anima/anima.safetensors
    vae_path: str = ""
    text_encoder_path: str = ""        # Qwen3-0.6B safetensors or dir
    lora_weights: list = field(default_factory=list)
    lora_multipliers: list = field(default_factory=list)

    @property
    def enabled(self) -> bool:
        return bool(self.dit_path and self.vae_path and self.text_encoder_path)


@dataclass
class Settings:
    device: str = "cuda"      # ROCm torch aliases cuda -> hip
    dtype: str = "bfloat16"   # bf16 only (project decision)
    host: str = "127.0.0.1"
    port: int = 8000
    krea2: Krea2Config = field(default_factory=Krea2Config)
    anima: AnimaConfig = field(default_factory=AnimaConfig)

    @property
    def torch_dtype(self):
        if self.dtype == "bfloat16":
            import torch
            return torch.bfloat16
        raise ValueError(f"unsupported dtype '{self.dtype}' (project is bf16-only)")


def _apply_krea2(cfg: Krea2Config, data: dict) -> None:
    k = data.get("krea2", {})
    for f in ("dit_path", "vae_path", "text_encoder_path"):
        if f in k:
            setattr(cfg, f, k[f])
    cfg.lora_weights = k.get("lora_weights", cfg.lora_weights)
    cfg.lora_multipliers = k.get("lora_multipliers", cfg.lora_multipliers)


def _apply_anima(cfg: AnimaConfig, data: dict) -> None:
    k = data.get("anima", {})
    for f in ("dit_path", "vae_path", "text_encoder_path"):
        if f in k:
            setattr(cfg, f, k[f])
    cfg.lora_weights = k.get("lora_weights", cfg.lora_weights)
    cfg.lora_multipliers = k.get("lora_multipliers", cfg.lora_multipliers)


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
            _apply_krea2(s.krea2, data)
            _apply_anima(s.anima, data)
        else:
            raise FileNotFoundError(f"config file not found: {path}")

    # Environment overrides.
    s.device = os.environ.get("DEVICE", s.device)
    if os.environ.get("KREA2_DIT"):
        s.krea2.dit_path = os.environ["KREA2_DIT"]
    if os.environ.get("KREA2_VAE"):
        s.krea2.vae_path = os.environ["KREA2_VAE"]
    if os.environ.get("KREA2_TEXT_ENCODER"):
        s.krea2.text_encoder_path = os.environ["KREA2_TEXT_ENCODER"]
    if os.environ.get("ANIMA_DIT"):
        s.anima.dit_path = os.environ["ANIMA_DIT"]
    if os.environ.get("ANIMA_VAE"):
        s.anima.vae_path = os.environ["ANIMA_VAE"]
    if os.environ.get("ANIMA_TEXT_ENCODER"):
        s.anima.text_encoder_path = os.environ["ANIMA_TEXT_ENCODER"]
    return s

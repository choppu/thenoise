"""Load-once model registry.

Keeps each loaded model resident in memory and exposes a uniform interface. Model
instances own their own inference lock (torch forward is not thread-safe), so the
registry itself is a simple name -> instance map.
"""
from __future__ import annotations

import logging
from typing import Any

from .config import Settings

logger = logging.getLogger(__name__)


class ModelRegistry:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._models: dict[str, Any] = {}
        self._loaded = False

    def load_all(self) -> None:
        if self._loaded:
            return
        s = self._settings

        if s.krea2.enabled:
            from .krea2 import Krea2Model
            self._models["krea2"] = Krea2Model(
                dit_path=s.krea2.dit_path,
                vae_path=s.krea2.vae_path,
                text_encoder_path=s.krea2.text_encoder_path,
                device=s.device,
                dtype=s.torch_dtype,
                lora_weights=s.krea2.lora_weights or None,
                lora_multipliers=s.krea2.lora_multipliers or None,
            )
            logger.info("Registered model 'krea2'")

        if s.anima.enabled:
            from .anima import AnimaModel
            self._models["anima"] = AnimaModel(
                dit_path=s.anima.dit_path,
                vae_path=s.anima.vae_path,
                text_encoder_path=s.anima.text_encoder_path,
                device=s.device,
                dtype=s.torch_dtype,
            )
            logger.info("Registered model 'anima'")

        self._loaded = True
        if not self._models:
            logger.warning("No models configured; the server will only answer /health")

    def get(self, name: str) -> Any:
        try:
            return self._models[name]
        except KeyError:
            raise KeyError(f"model '{name}' is not loaded (check config)") from None

    def available(self) -> list[str]:
        return list(self._models.keys())

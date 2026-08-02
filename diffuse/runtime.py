"""Single-model runtime: loads exactly one DiffusionModel at a time.

Replaces the multi-model registry. The runtime is deliberately thin and
model-agnostic: it knows how to construct a model class by name and holds the
single resident instance. Loading a new model swaps (unloads + GCs) the previous
one, so only one set of weights is ever resident.
"""
from __future__ import annotations

import gc
import importlib
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelPaths:
    """Checkpoint paths supplied by the CLI (never read from config)."""
    dit_path: str
    vae_path: str
    text_encoder_path: str
    lora_weights: list = field(default_factory=list)
    lora_multipliers: list = field(default_factory=list)


class NotLoadedError(RuntimeError):
    """Raised when no model is loaded."""


# name -> (class name, module) within the diffuse package.
# Phase 3 replaces this with a catalog of model classes that own their own detect().
_MODEL_CLASSES = {
    "krea2": ("Krea2Model", ".krea2"),
    "anima": ("AnimaModel", ".anima"),
}


class Runtime:
    def __init__(self, settings):
        self._settings = settings
        self._model: Any = None
        self._model_name: Optional[str] = None

    def load(self, model: str, paths: ModelPaths) -> None:
        name = model.strip().lower()
        if name not in _MODEL_CLASSES:
            raise ValueError(
                f"unknown model '{model}' (choose from {sorted(_MODEL_CLASSES)})"
            )
        cls_name, module = _MODEL_CLASSES[name]
        cls = getattr(importlib.import_module(module, package=__package__), cls_name)

        kwargs = dict(
            dit_path=paths.dit_path,
            vae_path=paths.vae_path,
            text_encoder_path=paths.text_encoder_path,
            device=self._settings.device,
            dtype=self._settings.torch_dtype,
        )
        if name == "krea2":
            kwargs["lora_weights"] = paths.lora_weights or None
            kwargs["lora_multipliers"] = paths.lora_multipliers or None

        self._unload()  # swap: only one model resident at a time
        logger.info("Loading model '%s'", name)
        self._model = cls(**kwargs)
        self._model_name = name

    def _unload(self) -> None:
        if self._model is None:
            return
        logger.info("Unloading model '%s'", self._model_name)
        del self._model
        self._model = None
        self._model_name = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:  # torch may be unavailable (config-only tests)
            pass

    @property
    def model(self) -> Any:
        if self._model is None:
            raise NotLoadedError("no model loaded")
        return self._model

    @property
    def model_name(self) -> str:
        if self._model is None:
            raise NotLoadedError("no model loaded")
        return self._model_name

    def available(self) -> list[str]:
        return [self._model_name] if self._model else []
